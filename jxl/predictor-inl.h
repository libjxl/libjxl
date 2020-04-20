// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(JXL_PREDICTOR_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_PREDICTOR_INL_H_
#undef JXL_PREDICTOR_INL_H_
#else
#define JXL_PREDICTOR_INL_H_
#endif

// DC coefficients serve as an image preview, so they are coded separately.
// Subtracting predicted values leads to a "residual" distribution with lower
// entropy and magnitudes than the original values. These can be coded more
// efficiently, even when context modeling is used.
//
// Our predictors use immediately adjacent causal pixels because more distant
// pixels are only weakly correlated in subsampled DC images.
//
// This module decreases final size of DC images by 2-4% vs. the standard
// MED/MAP predictor from JPEG-LS and processes 330 M coefficients per second.

#include <hwy/highway.h>
#include <stddef.h>
#include <stdint.h>

#include <limits>

#include "jxl/aux_out.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/common.h"
#include "jxl/entropy_coder.h"
#include "jxl/image.h"
#include "jxl/image_ops.h"
#include "jxl/predictor_shared.h"

namespace jxl {

// TODO(veluca): what happens if all valid predictors have an error of exactly
// 2**16-1? One option is to ensure that this does not happen.

// Sliding window of "causal" (already decoded) pixels, plus simple functions
// to predict the next pixel "c" from its neighbors: l n r
// The single-letter names shorten identifiers.      w c
//
// Predictions are more accurate when the preceding w pixel is available, but
// this interferes with SIMD because subsequent pixels depend on the decoding
// of their predecessor. The encoder can compute residuals in parallel because
// it knows all DC values up front, but its speed is less important. A diagonal
// 'wavefront' order would allow computing multiple predictions efficiently,
// but scattering those to the corresponding pixel positions would be slow.
// Interleaving pixels by the lane count (eight pixels with x mod 8 = 0, etc)
// would work if the two pixels before each prediction are already known, but
// scattering lanes to multiples of 10 would also be slow.
//
// We instead compute the various predictors using SIMD, especially because
// many of them are similar.
//
// The set of 8 predictors was chosen from a set of 16 as the combination that
// minimized a simple model of encoding cost. Their order matters because
// we choose the lowest i with lanes[i] == min.

#include <hwy/begin_target-inl.h>

struct PredictorPackSigned {
  static HWY_FUNC void Compute(const int32_t c,
                               const int32_t* JXL_RESTRICT pred,
                               uint32_t* JXL_RESTRICT error) {
    HWY_CAPPED(int32_t, kNumPredictors) di;
    HWY_CAPPED(uint32_t, kNumPredictors) du;
    auto cv = Set(di, c);
    for (size_t i = 0; i < kNumPredictors; i += di.N) {
      auto res = cv - Load(di, pred + i);
      auto res2 = ShiftLeft<1>(BitCast(du, Abs(res)));
      auto res_sign = BitCast(du, VecFromMask(res < Zero(di)));
      Store(res2 + res_sign, du, error + i);
    }
  }

  static JXL_INLINE uint32_t Residual(int32_t c, int32_t pred) {
    return PackSigned(c - pred);
  }

  static JXL_INLINE int32_t Original(uint32_t res, int32_t pred) {
    return pred + UnpackSigned(res);
  }
};

template <int16_t min, int16_t max>
struct PredictorPackSignedRange {
  static_assert(min <= max, "min <= max");
  static HWY_FUNC void Compute(const int32_t c,
                               const int32_t* JXL_RESTRICT pred,
                               uint32_t* JXL_RESTRICT error) {
    HWY_CAPPED(int32_t, kNumPredictors) di;
    HWY_CAPPED(uint32_t, kNumPredictors) du;
    auto cv = Set(di, c);
    auto minv = Set(di, min);
    auto maxv = Set(di, max);
    for (size_t i = 0; i < kNumPredictors; i += di.N) {
      auto predv = Load(di, pred + i);
      auto res = cv - predv;
      auto low = cv - minv;
      auto high = maxv - cv;
      auto res2 = ShiftLeft<1>(Abs(res));
      auto res_sign = VecFromMask(res < Zero(di));
      auto reshigh = IfThenElse(maxv == predv, high, res2 + res_sign);
      Store(BitCast(du, IfThenElse(minv == predv, low, reshigh)), du,
            error + i);
    }
  }

  static JXL_INLINE uint32_t Residual(int32_t c, int32_t pred) {
    if (pred == min) return c - min;
    if (pred == max) return max - c;
    return PackSigned(c - pred);
  }

  static JXL_INLINE int32_t Original(uint32_t res, int32_t pred) {
    if (pred == min) return min + res;
    if (pred == max) return max - res;
    return pred + UnpackSigned(res);
  }
};

HWY_FUNC void ChoosePredictor(const int32_t prediction[kNumPredictors],
                              const uint32_t expected_error[kNumPredictors],
                              int32_t* JXL_RESTRICT pred,
                              uint32_t* JXL_RESTRICT min_error,
                              uint32_t* JXL_RESTRICT num_correct) {
#if HWY_CAPS & HWY_CAP_GE256
  HWY_CAPPED(uint32_t, kNumPredictors) du;
  HWY_CAPPED(int32_t, kNumPredictors) di;
  static_assert(kNumPredictors == 8, "");
  const auto predv = Load(di, prediction);
  const auto errv = Load(du, expected_error);  // 76543210

  const auto errv1 = Shuffle2301(errv);  // 67452301
  const auto predv1 = Shuffle2301(predv);

  // errv1 >= errv
  auto max = Max(errv1, errv);
  const auto errv2 = IfThenElse(max == errv1, errv, errv1);  // 66442200
  const auto predv2 =
      IfThenElse(BitCast(di, max) == BitCast(di, errv1), predv, predv1);

  const auto errv3 = Shuffle1032(errv2);  // 44660022
  const auto predv3 = Shuffle1032(predv2);

  // errv3 >= errv2
  max = Max(errv3, errv2);
  const auto errv4 = IfThenElse(max == errv3, errv2, errv3);  // 44440000
  const auto predv4 =
      IfThenElse(BitCast(di, max) == BitCast(di, errv3), predv2, predv3);

  const auto plow = GetLane(LowerHalf(predv4));
  const auto phi = GetLane(UpperHalf(predv4));
  const auto elow = GetLane(LowerHalf(errv4));
  const auto ehi = GetLane(UpperHalf(errv4));

  if (ehi < elow) {
    *pred = phi;
    *min_error = ehi;
  } else {
    *pred = plow;
    *min_error = elow;
  }

  *num_correct = CountTrue(errv == Zero(du));
#else
  // TODO(veluca): 128 bit version.
  size_t idx_pred = 0;
  uint32_t min_cost = expected_error[0];
  for (size_t i = 1; i < kNumPredictors; ++i) {
    const uint32_t cost = expected_error[i];
    if (cost < min_cost) {
      min_cost = cost;
      idx_pred = i;
    }
  }
  *pred = prediction[idx_pred];
  *min_error = expected_error[idx_pred];

  HWY_CAPPED(uint32_t, kNumPredictors) du;
  size_t ret = 0;
  for (size_t i = 0; i < kNumPredictors; i += du.N) {
    auto l = Load(du, expected_error + i);
    ret += CountTrue(l == Zero(du));
  }
  *num_correct = ret;
#endif
}

template <typename ErrorMetric, typename Predictor, size_t channel,
          size_t kNumChannels, PixelType pixel_type>
struct ApplyPredictor {
  static_assert(channel < kNumChannels, "Invalid channel!");

  ApplyPredictor(RowType row_type, ColumnType column_type)
      : row_type(row_type), column_type(column_type) {}

  JXL_INLINE void Mask(uint32_t* JXL_RESTRICT mask) {
    if (pixel_type != PixelType::kInteriorPixel) {
      if (row_type == RowType::kFirstRow)
        error::Or::Apply(mask, Predictor::Row0Mask());
      if (column_type & kFirstColumn)
        error::Or::Apply(mask, Predictor::Col0Mask());
      if (column_type & kLastColumn)
        error::Or::Apply(mask, Predictor::LastColMask());
    }
  }

  template <typename Op>
  JXL_INLINE void ApplyMask(uint32_t* JXL_RESTRICT errors) {
    if (pixel_type != PixelType::kInteriorPixel) {
      uint32_t mask[kNumPredictors] = {};
      Mask(mask);
      Op::Apply(errors + channel * kNumPredictors, mask);
    }
  }

  HWY_FUNC void Prediction(size_t x, const int32_t* JXL_RESTRICT row_top,
                           const int32_t* JXL_RESTRICT row_cur,
                           int32_t* JXL_RESTRICT all_predictions) {
    const int32_t w =
        pixel_type != PixelType::kInteriorPixel && (column_type & kFirstColumn)
            ? 0
            : row_cur[(x - 1) * kNumChannels + channel];
    const int32_t l =
        pixel_type != PixelType::kInteriorPixel &&
                (row_type == RowType::kFirstRow || (column_type & kFirstColumn))
            ? 0
            : row_top[(x - 1) * kNumChannels + channel];
    // Clang mistakenly detect this as a potentially-undefined assignment, but
    // the conditions for that to happen cannot occur.
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
    const int32_t n = pixel_type != PixelType::kInteriorPixel &&
                              row_type == RowType::kFirstRow
                          ? 0
                          : row_top[x * kNumChannels + channel];
    const int32_t r =
        pixel_type != PixelType::kInteriorPixel &&
                (row_type == RowType::kFirstRow || (column_type & kLastColumn))
            ? 0
            : row_top[(x + 1) * kNumChannels + channel];
    Predictor::Predict(n, w, l, r, all_predictions + channel * kNumPredictors);
  }

  HWY_FUNC void ComputeError(const int32_t* JXL_RESTRICT pred,
                             const int32_t* JXL_RESTRICT actual,
                             uint32_t* JXL_RESTRICT error) {
    ErrorMetric::Compute(actual[channel], pred + channel * kNumPredictors,
                         error + channel * kNumPredictors);
    // Invalid predictions should become 0 because they should not affect error
    // that comes from other pixels for which they are valid. Advance will deal
    // with setting to max all predictions that have never been valid.
    ApplyMask<error::AndNot>(error);
  }
  RowType row_type;
  ColumnType column_type;
};

// Policy structs for 1/2/3 channels.
template <typename Predictor>
struct Predictors1 {
  enum { kNumChannels = 1 };
  template <typename ErrorMetric, PixelType pixel_type>
  struct Impl {
    using ApplyPredictor =
        ApplyPredictor<ErrorMetric, Predictor, 0, kNumChannels, pixel_type>;

    template <typename Op>
    HWY_FUNC static void ApplyMask(RowType row_type, ColumnType column_type,
                                   uint32_t* JXL_RESTRICT errors) {
      ApplyPredictor(row_type, column_type).template ApplyMask<Op>(errors);
    }

    HWY_FUNC static void Prediction(RowType row_type, ColumnType column_type,
                                    size_t x,
                                    const int32_t* JXL_RESTRICT row_top,
                                    const int32_t* JXL_RESTRICT row_cur,
                                    int32_t* JXL_RESTRICT all_predictions) {
      ApplyPredictor(row_type, column_type)
          .Prediction(x, row_top, row_cur, all_predictions);
    }

    HWY_FUNC static void ComputeError(RowType row_type, ColumnType column_type,
                                      const int32_t* JXL_RESTRICT pred,
                                      const int32_t* JXL_RESTRICT actual,
                                      uint32_t* JXL_RESTRICT error) {
      ApplyPredictor(row_type, column_type).ComputeError(pred, actual, error);
    }
  };
};

template <typename Predictor0, typename Predictor1>
struct Predictors2 {
  enum { kNumChannels = 2 };
  template <typename ErrorMetric, PixelType pixel_type>
  struct Impl {
    using ApplyPredictor0 =
        ApplyPredictor<ErrorMetric, Predictor0, 0, kNumChannels, pixel_type>;
    using ApplyPredictor1 =
        ApplyPredictor<ErrorMetric, Predictor1, 1, kNumChannels, pixel_type>;

    template <typename Op>
    JXL_INLINE static void ApplyMask(RowType row_type, ColumnType column_type,
                                     uint32_t* JXL_RESTRICT errors) {
      ApplyPredictor0(row_type, column_type).template ApplyMask<Op>(errors);
      ApplyPredictor1(row_type, column_type).template ApplyMask<Op>(errors);
    }

    HWY_FUNC static void Prediction(RowType row_type, ColumnType column_type,
                                    size_t x,
                                    const int32_t* JXL_RESTRICT row_top,
                                    const int32_t* JXL_RESTRICT row_cur,
                                    int32_t* JXL_RESTRICT all_predictions) {
      ApplyPredictor0(row_type, column_type)
          .Prediction(x, row_top, row_cur, all_predictions);
      ApplyPredictor1(row_type, column_type)
          .Prediction(x, row_top, row_cur, all_predictions);
    }

    HWY_FUNC static void ComputeError(RowType row_type, ColumnType column_type,
                                      const int32_t* JXL_RESTRICT pred,
                                      const int32_t* JXL_RESTRICT actual,
                                      uint32_t* JXL_RESTRICT error) {
      ApplyPredictor0(row_type, column_type).ComputeError(pred, actual, error);
      ApplyPredictor1(row_type, column_type).ComputeError(pred, actual, error);
    }
  };
};

template <typename Predictor0, typename Predictor1, typename Predictor2>
struct Predictors3 {
  enum { kNumChannels = 3 };
  template <typename ErrorMetric, PixelType pixel_type>
  struct Impl {
    using ApplyPredictor0 =
        ApplyPredictor<ErrorMetric, Predictor0, 0, kNumChannels, pixel_type>;
    using ApplyPredictor1 =
        ApplyPredictor<ErrorMetric, Predictor1, 1, kNumChannels, pixel_type>;
    using ApplyPredictor2 =
        ApplyPredictor<ErrorMetric, Predictor2, 2, kNumChannels, pixel_type>;

    template <typename Op>
    JXL_INLINE static void ApplyMask(RowType row_type, ColumnType column_type,
                                     uint32_t* JXL_RESTRICT errors) {
      ApplyPredictor0(row_type, column_type).template ApplyMask<Op>(errors);
      ApplyPredictor1(row_type, column_type).template ApplyMask<Op>(errors);
      ApplyPredictor2(row_type, column_type).template ApplyMask<Op>(errors);
    }

    HWY_FUNC static void Prediction(RowType row_type, ColumnType column_type,
                                    size_t x,
                                    const int32_t* JXL_RESTRICT row_top,
                                    const int32_t* JXL_RESTRICT row_cur,
                                    int32_t* JXL_RESTRICT all_predictions) {
      ApplyPredictor0(row_type, column_type)
          .Prediction(x, row_top, row_cur, all_predictions);
      ApplyPredictor1(row_type, column_type)
          .Prediction(x, row_top, row_cur, all_predictions);
      ApplyPredictor2(row_type, column_type)
          .Prediction(x, row_top, row_cur, all_predictions);
    }

    HWY_FUNC static void ComputeError(RowType row_type, ColumnType column_type,
                                      const int32_t* JXL_RESTRICT pred,
                                      const int32_t* JXL_RESTRICT actual,
                                      uint32_t* JXL_RESTRICT error) {
      ApplyPredictor0(row_type, column_type).ComputeError(pred, actual, error);
      ApplyPredictor1(row_type, column_type).ComputeError(pred, actual, error);
      ApplyPredictor2(row_type, column_type).ComputeError(pred, actual, error);
    }
  };
};

template <typename T, typename ErrorMetric, typename Predictors>
class ComputeResiduals {
  enum { kNumChannels = Predictors::kNumChannels };

  template <PixelType pixel_type>
  using ForEachChannel =
      typename Predictors::template Impl<ErrorMetric, pixel_type>;

 public:
  HWY_ATTR void Run(size_t xsize, size_t ysize, AuxOut* JXL_RESTRICT aux_out) {
    ProcessRow<RowType::kFirstRow>(xsize, 0, aux_out);
    if (ysize == 1) return;
    ProcessRow<RowType::kSecondRow>(xsize, 1, aux_out);
    for (size_t y = 2; y < ysize; y++) {
      ProcessRow<RowType::kRegularRow>(xsize, y, aux_out);
    }
  }

 private:
  // Computes the best prediction for each channel, calls Prediction and
  // updates error_w_ and error_l_.
  template <PixelType pixel_type>
  HWY_ATTR void Advance(RowType row_type, ColumnType column_type, size_t x,
                        size_t y, AuxOut* JXL_RESTRICT aux_out) {
    // Highway vectors may be larger, but the arrays used here have exactly
    // kNumChannels * kNumPredictors elements. To avoid overruns, we only
    // use this many lanes so vectors evenly divide the array size.
    using DU = HWY_CAPPED(uint32_t, kNumPredictors);
    auto max_it = [](uint32_t* JXL_RESTRICT dest,
                     uint32_t* JXL_RESTRICT mix) HWY_ATTR {
      for (size_t i = 0; i < kNumChannels * kNumPredictors; i += DU::N) {
        auto cur = Load(DU(), dest + i);
        auto m = Load(DU(), mix + i);
        Store(Max(cur, m), DU(), dest + i);
      }
#if defined(__arm__)
      // Compiler bug in clang-6 on arm can cause successive calls to max_it()
      // using the same "dest" variable, as it is used here, to  have a
      // read-after-write in the wrong order. The following statement prevents
      // that problem
      asm("" : : "m"(dest[0]));
#endif  // __arm__
    };
    HWY_ALIGN int32_t prediction_cur[kNumChannels * kNumPredictors];
    HWY_ALIGN uint32_t expected_error[kNumChannels * kNumPredictors];
    HWY_ALIGN uint32_t error_mask[kNumChannels * kNumPredictors];
    HWY_ALIGN uint32_t error_n[kNumChannels * kNumPredictors];
    const int32_t* JXL_RESTRICT row_top =
        pixel_type != PixelType::kInteriorPixel &&
                row_type != RowType::kRegularRow
            ? nullptr
            : rows_[(y - 2) & 3];
    const int32_t* JXL_RESTRICT row_mid =
        pixel_type != PixelType::kInteriorPixel &&
                row_type == RowType::kFirstRow
            ? nullptr
            : rows_[(y - 1) & 3];
    int32_t* JXL_RESTRICT row_cur = rows_[y & 3];

    // Compute prediction.
    ForEachChannel<pixel_type>::Prediction(row_type, column_type, x, row_mid,
                                           row_cur, prediction_cur);

    // Initialize expected error to 0.
    for (size_t c = 0; c < kNumChannels * kNumPredictors; c++) {
      expected_error[c] = 0;
    }

    // Compute previous row and column type. We don't care about correctly
    // detecting if the preceding row/column is the second, as that does not
    // affect predictor usability.
    const RowType previous_row = row_type == RowType::kRegularRow
                                     ? RowType::kRegularRow
                                     : RowType::kFirstRow;
    const ColumnType previous_column =
        column_type & kSecondColumn ? kFirstColumn : kRegularColumn;

    // Initialize error_mask to all ones. This is only needed in the first two
    // rows and columns: all predictors will be valid at least once in the 3
    // preceding pixels otherwise.
    if (pixel_type != PixelType::kInteriorPixel &&
        (row_type != RowType::kRegularRow ||
         column_type & (kFirstColumn | kSecondColumn))) {
      for (size_t c = 0; c < kNumChannels * kNumPredictors; c++) {
        error_mask[c] = kMaxError;
      }
    }

    // Do not take into account error on pixel n if in the first row.
    if (pixel_type == PixelType::kInteriorPixel ||
        row_type != RowType::kFirstRow) {
      HWY_ALIGN int32_t prediction_n[kNumChannels * kNumPredictors];
      ForEachChannel<pixel_type>::Prediction(previous_row, column_type, x,
                                             row_top, row_mid, prediction_n);
      ForEachChannel<pixel_type>::ComputeError(
          previous_row, column_type, prediction_n, row_mid + x * kNumChannels,
          error_n);
      max_it(expected_error, error_n);
      // Mark valid predictors as being usable.
      ForEachChannel<pixel_type>::template ApplyMask<error::And>(
          previous_row, column_type, error_mask);
    }

    // Do not take into account error on pixel l in either the first row or
    // first column.
    if (pixel_type == PixelType::kInteriorPixel ||
        (row_type != RowType::kFirstRow && (column_type & kFirstColumn) == 0)) {
      max_it(expected_error, error_l_);
      // Mark valid predictors as being usable.
      ForEachChannel<pixel_type>::template ApplyMask<error::And>(
          previous_row, previous_column, error_mask);
    }

    // Do not take into account error on pixel w if in the first column.
    if (pixel_type == PixelType::kInteriorPixel ||
        (column_type & kFirstColumn) == 0) {
      max_it(expected_error, error_w_);
      // Mark valid predictors as being usable.
      ForEachChannel<pixel_type>::template ApplyMask<error::And>(
          row_type, previous_column, error_mask);
    }

    // Set max error for predictors that did not ever receive error statistics
    // from neighboring pixels, or that are unavailable for the current pixel.
    // This only ever happens in the first two rows or columns or the last
    // column.
    // If no error information is available for any predictor, use the first
    // predictor that is available for this pixel.
    if (pixel_type != PixelType::kInteriorPixel &&
        (row_type != RowType::kRegularRow ||
         column_type & (kFirstColumn | kSecondColumn))) {
      ForEachChannel<pixel_type>::template ApplyMask<error::Or>(
          row_type, column_type, expected_error);

      for (size_t c = 0; c < kNumChannels; c++) {
        int32_t pred;
        uint32_t min_error;
        uint32_t num_correct;
        // Re-use ChoosePredictor to compute the minimum error in the error
        // mask.
        ChoosePredictor(prediction_cur + c * kNumPredictors,
                        error_mask + c * kNumPredictors, &pred, &min_error,
                        &num_correct);
        if (min_error != kMaxError) {
          error::Or::Apply(expected_error + c * kNumPredictors,
                           error_mask + c * kNumPredictors);
        } else {
          // Set all errors to kMaxError-1. This ensures that num_correct
          // will be zero, and that min_error will be high, not to ``poison''
          // the context model with low errors.
          for (size_t i = c * kNumPredictors; i < (c + 1) * kNumPredictors;
               i++) {
            expected_error[i] |= kMaxError - 1;
          }
        }
      }
    }
    if (pixel_type != PixelType::kInteriorPixel && column_type & kLastColumn) {
      ForEachChannel<pixel_type>::template ApplyMask<error::Or>(
          row_type, column_type, expected_error);
    }

    HWY_ALIGN int32_t prediction[kNumChannels < 4 ? 4 : kNumChannels];
    HWY_ALIGN uint32_t num_correct[kNumChannels];
    HWY_ALIGN uint32_t min_error[kNumChannels];

#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
    for (size_t i = kNumChannels; i < 4; i++) {
      prediction[i] = 0;
    }
#endif

    // Choose best predictor, compute stats.
    for (size_t channel = 0; channel < kNumChannels; channel++) {
      if (row_type != RowType::kFirstRow) {
        for (size_t i = 0; i < kNumPredictors; i++) {
          error_l_[channel * kNumPredictors + i] =
              error_n[channel * kNumPredictors + i];
        }
      }
      ChoosePredictor(prediction_cur + channel * kNumPredictors,
                      expected_error + channel * kNumPredictors,
                      prediction + channel, min_error + channel,
                      num_correct + channel);
    }

    int32_t* JXL_RESTRICT decoded = row_cur + x * kNumChannels;
    Prediction(x, y, prediction, num_correct, min_error, decoded);
    if (pixel_type == PixelType::kInteriorPixel ||
        (column_type & kLastColumn) == 0) {
      // Compute error for current pixel and save it in error_w_.
      ForEachChannel<pixel_type>::ComputeError(
          row_type, column_type, prediction_cur, decoded, error_w_);
    }
  }

  template <RowType row_type>
  HWY_FUNC void ProcessRow(size_t xsize, size_t y,
                           AuxOut* JXL_RESTRICT aux_out) {
    CallStartRow(y);
    // First column
    Advance<PixelType::kBorderPixel>(
        row_type, ColumnType((xsize == 1 ? kLastColumn : 0) | kFirstColumn), 0,
        y, aux_out);
    if (xsize == 1) return;
    // Second column
    Advance<PixelType::kBorderPixel>(
        row_type, ColumnType((xsize == 2 ? kLastColumn : 0) | kSecondColumn), 1,
        y, aux_out);
    if (xsize == 2) return;
    // All other columns
    for (size_t x = 2; x < xsize - 1; x++) {
      Advance<row_type == RowType::kRegularRow ? PixelType::kInteriorPixel
                                               : PixelType::kBorderPixel>(
          row_type, kRegularColumn, x, y, aux_out);
    }
    // Last column
    Advance<PixelType::kBorderPixel>(row_type, kLastColumn, xsize - 1, y,
                                     aux_out);
  }

  // This function is called with the prediction, number of predictors
  // expected to be correct and minimum prediction error for each channel, and
  // should write in `decoded` the decoder-side values.
  HWY_ATTR void Prediction(size_t x, size_t y,
                           const int32_t* JXL_RESTRICT predictions,
                           const uint32_t* JXL_RESTRICT num_correct,
                           const uint32_t* JXL_RESTRICT min_error,
                           int32_t* JXL_RESTRICT decoded) {
    static_cast<T*>(this)->Prediction(x, y, predictions, num_correct, min_error,
                                      decoded);
  }

  void StartRow(size_t y) {}

  HWY_ATTR void CallStartRow(size_t y) { static_cast<T*>(this)->StartRow(y); }

  // Predictors need the last two rows. We use a ringbuffer of size 4, as it
  // is faster to compute row numbers modulo 4 than modulo 3.
  int32_t rows_[4][kMaxLine * kNumChannels];

  // We re-use predictor error for pixel l from pixel n and error w
  // from the previous pixel.
  HWY_ALIGN uint32_t error_l_[kNumChannels * kNumPredictors];
  HWY_ALIGN uint32_t error_w_[kNumChannels * kNumPredictors];
};

// TODO(veluca): choose predictors.
template <typename T>
using DcPredictor =
    ComputeResiduals<T, PredictorPackSigned,
                     Predictors3<YPredictor, YPredictor, YPredictor>>;

#include <hwy/end_target-inl.h>

}  // namespace jxl

#endif  // include guard
