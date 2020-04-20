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

#include "jxl/entropy_coder.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "jxl/ac_context.h"
#include "jxl/ac_strategy.h"
#include "jxl/aux_out.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/coeff_order.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/common.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/epf.h"
#include "jxl/image.h"
#include "jxl/image_ops.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/entropy_coder.cc"
#include <hwy/foreach_target.h>

#include "jxl/predictor-inl.h"

namespace jxl {

#include <hwy/begin_target-inl.h>

// Returns number of non-zero coefficients (but skip LLF).
// We cannot rely on block[] being all-zero bits, so first truncate to integer.
// Also writes the per-8x8 block nzeros starting at nzeros_pos.
HWY_ATTR int32_t NumNonZeroExceptLLF(const size_t cx, const size_t cy,
                                     const AcStrategy acs,
                                     const size_t covered_blocks,
                                     const size_t log2_covered_blocks,
                                     const ac_qcoeff_t* JXL_RESTRICT block,
                                     const size_t nzeros_stride,
                                     int32_t* JXL_RESTRICT nzeros_pos) {
  const HWY_CAPPED(float, kBlockDim) df;
  const HWY_CAPPED(int32_t, kBlockDim) di;

  const auto zero = Zero(di);
  // Add FF..FF for every zero coefficient, negate to get #zeros.
  auto neg_sum_zero = zero;

  {
    // Mask sufficient for one row of coefficients.
    HWY_ALIGN const int32_t
        llf_mask_lanes[AcStrategy::kMaxCoeffBlocks * (1 + kBlockDim)] = {
            -1, -1, -1, -1};
    // First cx=1,2,4 elements are FF..FF, others 0.
    const int32_t* llf_mask_pos =
        llf_mask_lanes + AcStrategy::kMaxCoeffBlocks - cx;

    // Rows with LLF: mask out the LLF
    for (size_t y = 0; y < cy; y++) {
      for (size_t x = 0; x < cx * kBlockDim; x += df.N) {
        const auto llf_mask = BitCast(df, LoadU(di, llf_mask_pos + x));

        // LLF counts as zero so we don't include it in nzeros.
        const auto coef =
            AndNot(llf_mask, Load(df, &block[y * cx * kBlockDim + x]));

        neg_sum_zero += VecFromMask(ConvertTo(di, coef) == zero);
      }
    }
  }

  // Remaining rows: no mask
  for (size_t y = cy; y < cy * kBlockDim; y++) {
    for (size_t x = 0; x < cx * kBlockDim; x += df.N) {
      const auto coef = Load(df, &block[y * cx * kBlockDim + x]);
      neg_sum_zero += VecFromMask(ConvertTo(di, coef) == zero);
    }
  }

  // We want area - sum_zero, add because neg_sum_zero is already negated.
  const int32_t nzeros =
      int32_t(cx * cy * kDCTBlockSize) + GetLane(SumOfLanes(neg_sum_zero));

  const int32_t shifted_nzeros = static_cast<int32_t>(
      (nzeros + covered_blocks - 1) >> log2_covered_blocks);
  // Need non-canonicalized dimensions!
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
      nzeros_pos[x + y * nzeros_stride] = shifted_nzeros;
    }
  }

  return nzeros;
}

// Specialization for 8x8, where only top-left is LLF/DC.
// About 1% overall speedup vs. NumNonZeroExceptLLF.
HWY_ATTR int32_t NumNonZero8x8ExceptDC(const ac_qcoeff_t* JXL_RESTRICT block,
                                       int32_t* JXL_RESTRICT nzeros_pos) {
  const HWY_CAPPED(float, kBlockDim) df;
  const HWY_CAPPED(int32_t, kBlockDim) di;

  const auto zero = Zero(di);
  // Add FF..FF for every zero coefficient, negate to get #zeros.
  auto neg_sum_zero = zero;

  {
    // First row has DC, so mask
    const size_t y = 0;
    HWY_ALIGN const int32_t dc_mask_lanes[kBlockDim] = {-1};

    for (size_t x = 0; x < kBlockDim; x += df.N) {
      const auto dc_mask = BitCast(df, Load(di, dc_mask_lanes + x));

      // DC counts as zero so we don't include it in nzeros.
      const auto coef = AndNot(dc_mask, Load(df, &block[y * kBlockDim + x]));

      neg_sum_zero += VecFromMask(ConvertTo(di, coef) == zero);
    }
  }

  // Remaining rows: no mask
  for (size_t y = 1; y < kBlockDim; y++) {
    for (size_t x = 0; x < kBlockDim; x += df.N) {
      const auto coef = Load(df, &block[y * kBlockDim + x]);
      neg_sum_zero += VecFromMask(ConvertTo(di, coef) == zero);
    }
  }

  // We want 64 - sum_zero, add because neg_sum_zero is already negated.
  const int32_t nzeros =
      int32_t(kDCTBlockSize) + GetLane(SumOfLanes(neg_sum_zero));

  *nzeros_pos = nzeros;

  return nzeros;
}

// The number of nonzeros of each block is predicted from the top and the left
// blocks, with opportune scaling to take into account the number of blocks of
// each strategy.  The predicted number of nonzeros divided by two is used as a
// context; if this number is above 63, a specific context is used.  If the
// number of nonzeros of a strategy is above 63, it is written directly using a
// fixed number of bits (that depends on the size of the strategy).
// TODO(veluca): consider predicting #zeros with predictor-inl.h.
HWY_ATTR void TokenizeCoefficients(
    const coeff_order_t* JXL_RESTRICT orders, const Rect& rect,
    const ac_qcoeff_t* JXL_RESTRICT* JXL_RESTRICT ac_rows,
    const AcStrategyImage& ac_strategy, Image3I* JXL_RESTRICT tmp_num_nzeroes,
    std::vector<Token>* JXL_RESTRICT output) {
  const size_t xsize_blocks = rect.xsize();
  const size_t ysize_blocks = rect.ysize();

  // TODO(user): update the estimate: usually less coefficients are used.
  output->reserve(output->size() +
                  3 * xsize_blocks * ysize_blocks * kDCTBlockSize);

  size_t offset = 0;
  const size_t nzeros_stride = tmp_num_nzeroes->PixelsPerRow();
  for (size_t by = 0; by < ysize_blocks; ++by) {
    int32_t* JXL_RESTRICT row_nzeros[3] = {
        tmp_num_nzeroes->PlaneRow(0, by),
        tmp_num_nzeroes->PlaneRow(1, by),
        tmp_num_nzeroes->PlaneRow(2, by),
    };
    const int32_t* JXL_RESTRICT row_nzeros_top[3] = {
        by == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(0, by - 1),
        by == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(1, by - 1),
        by == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(2, by - 1),
    };
    AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      size_t cx = acs.covered_blocks_x();
      size_t cy = acs.covered_blocks_y();
      const size_t covered_blocks = cx * cy;  // = #LLF coefficients
      const size_t log2_covered_blocks =
          NumZeroBitsBelowLSBNonzero(covered_blocks);
      const size_t size = covered_blocks * kDCTBlockSize;

      CoefficientLayout(&cy, &cx);  // swap cx/cy to canonical order

      for (size_t c : {1, 0, 2}) {
        const uint32_t c_ctx_x5_tbl[3] = {5, 0, 5};
        const size_t c_ctx_x5 = c_ctx_x5_tbl[c];
        const ac_qcoeff_t* JXL_RESTRICT block = ac_rows[c] + offset;

        int32_t nzeros =
            (covered_blocks == 1)
                ? NumNonZero8x8ExceptDC(block, row_nzeros[c] + bx)
                : NumNonZeroExceptLLF(cx, cy, acs, covered_blocks,
                                      log2_covered_blocks, block, nzeros_stride,
                                      row_nzeros[c] + bx);

        size_t ord = kStrategyOrder[acs.RawStrategy()];
        const coeff_order_t* JXL_RESTRICT order =
            &orders[(ord * 3 + c) * AcStrategy::kMaxCoeffArea];
        ord = ord > 2 ? ord / 2 + 1 : ord;

        int32_t predicted_nzeros =
            PredictFromTopAndLeft(row_nzeros_top[c], row_nzeros[c], bx, 32);
        const int32_t nzero_ctx =
            NonZeroContext(predicted_nzeros, c_ctx_x5 + ord);
        TokenizeHybridUint(nzero_ctx, nzeros, output);
        const size_t histo_offset = ZeroDensityContextsOffset(c_ctx_x5 + ord);
        // Skip LLF.
        size_t prev = (nzeros > size / 16 ? 0 : 1);
        for (size_t k = covered_blocks; k < size && nzeros != 0; ++k) {
          int32_t coeff = static_cast<int32_t>(block[order[k]]);
          size_t ctx =
              histo_offset + ZeroDensityContext(nzeros, k, covered_blocks,
                                                log2_covered_blocks, prev);
          uint32_t u_coeff = PackSigned(coeff);
          TokenizeHybridUint(ctx, u_coeff, output);
          prev = coeff != 0;
          nzeros -= prev;
        }
        JXL_DASSERT(nzeros == 0);
      }
      offset += size;
    }
  }
}

constexpr size_t kNumResidualContexts = 8;
constexpr size_t kControlFieldContexts = 4 + kNumResidualContexts;

static_assert(kQuantFieldContexts == kControlFieldContexts,
              "Invalid QF context number");

// `correct` = 0..8 (number of predictors).
static int Context(size_t correct, size_t badness) {
  if (correct == 0) {
    JXL_ASSERT(badness != 0);
    badness = (badness + 1) >> 1;
    uint32_t badness_offset =
        std::min<size_t>(badness, kNumResidualContexts) - 1;
    return badness_offset;
  }
  return kNumResidualContexts + CeilLog2Nonzero(9 - correct);
}

static int PredictionContext(size_t pred, size_t correct) {
  return pred * kPerPredictionContexts + (correct > 5);
}

class AcStrategyCoder {
 public:
  using ErrorMetric = PredictorPackSigned;
  // TODO(veluca): change predictors?
  template <typename T>
  using Predictor = ComputeResiduals<T, ErrorMetric, Predictors1<XBPredictor>>;

  struct Decoder : public Predictor<Decoder> {
    Decoder(BitReader* JXL_RESTRICT br, ANSSymbolReader* decoder,
            const std::vector<uint8_t>* context_map,
            AcStrategyImage* JXL_RESTRICT ac_strategy, const Rect& rect,
            size_t base_context, AuxOut* JXL_RESTRICT aux_out)
        : br_(br),
          decoder_(decoder),
          context_map_(context_map),
          ac_strategy_(ac_strategy),
          rect_(rect),
          base_context_(base_context),
          aux_out_(aux_out) {}

    Status Decode() {
      Predictor<Decoder>::Run(rect_.xsize(), rect_.ysize(), aux_out_);
      if (invalid_) {
        return JXL_FAILURE("Invalid AC strategy");
      }
      return true;
    }

    JXL_INLINE void Prediction(size_t x, size_t y,
                               const int32_t* JXL_RESTRICT predictions,
                               const uint32_t* JXL_RESTRICT num_correct,
                               const uint32_t* JXL_RESTRICT min_error,
                               int32_t* JXL_RESTRICT decoded) {
      size_t ctx =
          base_context_ + PredictionContext(predictions[0], num_correct[0]);

      if (!ac_strategy_->IsValid(rect_.x0() + x, rect_.y0() + y)) {
        uint32_t raw_strategy = decoder_->ReadSymbol((*context_map_)[ctx], br_);
        if (!AcStrategy::IsRawStrategyValid(raw_strategy)) {
          invalid_ = true;
        }
        if (!invalid_) {
          // We can't create an AcStrategy from an invalid raw_strategy.
          AcStrategy acs = AcStrategy::FromRawStrategy(raw_strategy);
          if (y + acs.covered_blocks_y() > rect_.ysize()) {
            invalid_ = true;
          }
          if (x + acs.covered_blocks_x() > rect_.xsize()) {
            invalid_ = true;
          }
          if (!invalid_) {
            ac_strategy_->SetNoBoundsCheck(
                rect_.x0() + x, rect_.y0() + y,
                static_cast<AcStrategy::Type>(raw_strategy));
          }
        }
      }
      if (!invalid_) {
        decoded[0] = ac_strategy_->ConstRow(rect_, y)[x].RawStrategy();
      } else {
        decoded[0] = 0;
      }
    }

   private:
    BitReader* JXL_RESTRICT br_;
    ANSSymbolReader* decoder_;
    const std::vector<uint8_t>* context_map_;
    AcStrategyImage* JXL_RESTRICT ac_strategy_;
    Rect rect_;
    size_t base_context_;
    AuxOut* JXL_RESTRICT aux_out_;
    bool invalid_ = false;
  };

  struct Encoder : public Predictor<Encoder> {
    Encoder(std::vector<Token>* JXL_RESTRICT tokens, const AcStrategyRow row,
            size_t stride, size_t base_context, AuxOut* JXL_RESTRICT aux_out)
        : tokens_(tokens),
          row_(row),
          stride_(stride),
          base_context_(base_context),
          aux_out_(aux_out) {}

    void Encode(size_t xsize, size_t ysize) {
      Predictor<Encoder>::Run(xsize, ysize, aux_out_);
    }

    JXL_INLINE void Prediction(size_t x, size_t y,
                               const int32_t* JXL_RESTRICT predictions,
                               const uint32_t* JXL_RESTRICT num_correct,
                               const uint32_t* JXL_RESTRICT min_error,
                               int32_t* JXL_RESTRICT decoded) {
      size_t ctx =
          base_context_ + PredictionContext(predictions[0], num_correct[0]);
      decoded[0] = row_[y * stride_ + x].RawStrategy();
      JXL_ASSERT(AcStrategy::IsRawStrategyValid(decoded[0]));

      if (row_[y * stride_ + x].IsFirstBlock()) {
        tokens_->emplace_back(ctx, decoded[0], 0, 0);
      }
    }

   private:
    std::vector<Token>* JXL_RESTRICT tokens_;
    const AcStrategyRow row_;
    size_t stride_;
    size_t base_context_;
    AuxOut* JXL_RESTRICT aux_out_;
  };
};

class QuantFieldCoder {
 public:
  using ErrorMetric = PredictorPackSignedRange<0, 255>;
  // TODO(veluca): change predictors?
  template <typename T>
  using Predictor = ComputeResiduals<T, ErrorMetric, Predictors1<XBPredictor>>;

  struct Decoder : public Predictor<Decoder> {
    Decoder(BitReader* JXL_RESTRICT br, ANSSymbolReader* decoder,
            const std::vector<uint8_t>* context_map, int32_t* JXL_RESTRICT row,
            size_t stride, AcStrategyRow acs_row, size_t acs_stride,
            size_t base_context, AuxOut* JXL_RESTRICT aux_out)
        : br_(br),
          decoder_(decoder),
          context_map_(context_map),
          row_(row),
          stride_(stride),
          acs_row_(acs_row),
          acs_stride_(acs_stride),
          base_context_(base_context),
          aux_out_(aux_out) {}

    void Decode(size_t xsize, size_t ysize) {
      Predictor<Decoder>::Run(xsize, ysize, aux_out_);
    }

    JXL_INLINE void Prediction(size_t x, size_t y,
                               const int32_t* JXL_RESTRICT predictions,
                               const uint32_t* JXL_RESTRICT num_correct,
                               const uint32_t* JXL_RESTRICT min_error,
                               int32_t* JXL_RESTRICT decoded) {
      size_t ctx = base_context_ + Context(num_correct[0], min_error[0]);

      AcStrategy acs = acs_row_[y * acs_stride_ + x];
      if (acs.IsFirstBlock()) {
        uint32_t residual = decoder_->ReadHybridUint(ctx, br_, *context_map_);
        int qf = ErrorMetric::Original(residual, predictions[0]);
        if (qf < 0) qf = 0;
        if (qf > 255) qf = 255;
        for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
          for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
            row_[x + ix + (y + iy) * stride_] = qf + 1;
          }
        }
      }
      decoded[0] = row_[y * stride_ + x] - 1;
    }

   private:
    BitReader* JXL_RESTRICT br_;
    ANSSymbolReader* decoder_;
    const std::vector<uint8_t>* context_map_;
    int32_t* JXL_RESTRICT row_;
    size_t stride_;
    const AcStrategyRow acs_row_;
    size_t acs_stride_;
    size_t base_context_;
    AuxOut* JXL_RESTRICT aux_out_;
  };

  struct Encoder : public Predictor<Encoder> {
    Encoder(std::vector<Token>* JXL_RESTRICT tokens,
            const int32_t* JXL_RESTRICT row, size_t stride,
            AcStrategyRow acs_row, size_t acs_stride, size_t base_context,
            AuxOut* JXL_RESTRICT aux_out)
        : tokens_(tokens),
          row_(row),
          stride_(stride),
          acs_row_(acs_row),
          acs_stride_(acs_stride),
          base_context_(base_context),
          aux_out_(aux_out) {}

    void Encode(size_t xsize, size_t ysize) {
      Predictor<Encoder>::Run(xsize, ysize, aux_out_);
    }

    JXL_INLINE void Prediction(size_t x, size_t y,
                               const int32_t* JXL_RESTRICT predictions,
                               const uint32_t* JXL_RESTRICT num_correct,
                               const uint32_t* JXL_RESTRICT min_error,
                               int32_t* JXL_RESTRICT decoded) {
      size_t ctx = base_context_ + Context(num_correct[0], min_error[0]);

      decoded[0] = row_[y * stride_ + x] - 1;

      AcStrategy acs = acs_row_[y * acs_stride_ + x];
      if (acs.IsFirstBlock()) {
        uint32_t residual = ErrorMetric::Residual(decoded[0], predictions[0]);
        TokenizeHybridUint(ctx, residual, tokens_);
      }
    }

   private:
    std::vector<Token>* JXL_RESTRICT tokens_;
    const int32_t* JXL_RESTRICT row_;
    size_t stride_;
    const AcStrategyRow acs_row_;
    size_t acs_stride_;
    size_t base_context_;
    AuxOut* JXL_RESTRICT aux_out_;
  };
};

class ARParamsCoder {
 public:
  using ErrorMetric = PredictorPackSigned;
  // TODO(veluca): change predictors?
  template <typename T>
  using Predictor = ComputeResiduals<T, ErrorMetric, Predictors1<YPredictor>>;

  struct Decoder : public Predictor<Decoder> {
    Decoder(BitReader* JXL_RESTRICT br, ANSSymbolReader* decoder,
            const std::vector<uint8_t>* context_map, uint8_t* JXL_RESTRICT row,
            size_t stride, AcStrategyRow acs_row, size_t acs_stride,
            size_t base_context, AuxOut* JXL_RESTRICT aux_out)
        : br_(br),
          decoder_(decoder),
          context_map_(context_map),
          row_(row),
          stride_(stride),
          acs_row_(acs_row),
          acs_stride_(acs_stride),
          base_context_(base_context),
          aux_out_(aux_out) {}

    void Decode(size_t xsize, size_t ysize) {
      Predictor<Decoder>::Run(xsize, ysize, aux_out_);
    }

    JXL_INLINE void Prediction(size_t x, size_t y,
                               const int32_t* JXL_RESTRICT predictions,
                               const uint32_t* JXL_RESTRICT num_correct,
                               const uint32_t* JXL_RESTRICT min_error,
                               int32_t* JXL_RESTRICT decoded) {
      size_t ctx =
          base_context_ + PredictionContext(predictions[0], num_correct[0]);
      AcStrategy acs = acs_row_[y * acs_stride_ + x];
      if (acs.IsFirstBlock()) {
        uint32_t lut = decoder_->ReadSymbol((*context_map_)[ctx], br_);
        if (lut > kNumEpfSharpness - 1) lut = kNumEpfSharpness - 1;
        row_[y * stride_ + x] = lut;
        ctx = base_context_ + kPerPredictionContexts * kNumEpfSharpness + lut;
        for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
          for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
            if (ix == 0 && iy == 0) continue;
            uint32_t new_lut = decoder_->ReadSymbol((*context_map_)[ctx], br_);
            row_[(y + iy) * stride_ + x + ix] =
                new_lut > kNumEpfSharpness - 1 ? kNumEpfSharpness - 1 : new_lut;
          }
        }
      }

      decoded[0] = row_[y * stride_ + x];
    }

   private:
    BitReader* JXL_RESTRICT br_;
    ANSSymbolReader* decoder_;
    const std::vector<uint8_t>* context_map_;
    uint8_t* JXL_RESTRICT row_;
    size_t stride_;
    const AcStrategyRow acs_row_;
    size_t acs_stride_;
    size_t base_context_;
    AuxOut* JXL_RESTRICT aux_out_;
  };

  struct Encoder : public Predictor<Encoder> {
    Encoder(std::vector<Token>* JXL_RESTRICT tokens,
            const uint8_t* JXL_RESTRICT row, size_t stride,
            AcStrategyRow acs_row, size_t acs_stride, size_t base_context,
            AuxOut* JXL_RESTRICT aux_out)
        : tokens_(tokens),
          row_(row),
          stride_(stride),
          acs_row_(acs_row),
          acs_stride_(acs_stride),
          base_context_(base_context),
          aux_out_(aux_out) {}

    void Encode(size_t xsize, size_t ysize) {
      Predictor<Encoder>::Run(xsize, ysize, aux_out_);
    }

    JXL_INLINE void Prediction(size_t x, size_t y,
                               const int32_t* JXL_RESTRICT predictions,
                               const uint32_t* JXL_RESTRICT num_correct,
                               const uint32_t* JXL_RESTRICT min_error,
                               int32_t* JXL_RESTRICT decoded) {
      size_t ctx =
          base_context_ + PredictionContext(predictions[0], num_correct[0]);
      AcStrategy acs = acs_row_[y * acs_stride_ + x];
      if (acs.IsFirstBlock()) {
        uint8_t lut = row_[y * stride_ + x];
        tokens_->emplace_back(ctx, lut, 0, 0);
        ctx = base_context_ + kPerPredictionContexts * kNumEpfSharpness + lut;
        for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
          for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
            if (ix == 0 && iy == 0) continue;
            uint8_t new_lut = row_[(y + iy) * stride_ + x + ix];
            tokens_->emplace_back(ctx, new_lut, 0, 0);
          }
        }
      }

      decoded[0] = row_[y * stride_ + x];
    }

   private:
    std::vector<Token>* JXL_RESTRICT tokens_;
    const uint8_t* JXL_RESTRICT row_;
    size_t stride_;
    const AcStrategyRow acs_row_;
    size_t acs_stride_;
    size_t base_context_;
    AuxOut* JXL_RESTRICT aux_out_;
  };
};

void TokenizeQuantField(const Rect& rect, const ImageI& quant_field,
                        const AcStrategyImage& ac_strategy,
                        std::vector<Token>* JXL_RESTRICT output,
                        size_t base_context) {
  PROFILER_FUNC;
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  const size_t stride = quant_field.PixelsPerRow();
  const size_t acs_stride = ac_strategy.PixelsPerRow();

  AuxOut aux_out;
  QuantFieldCoder::Encoder enc(output, rect.ConstRow(quant_field, 0), stride,
                               ac_strategy.ConstRow(rect, 0), acs_stride,
                               base_context, &aux_out);
  enc.Encode(xsize, ysize);
}

// The `rect_qf` argument specifies, in block units, the location we should
// decode to inside the `quant_field` image, and the location we should read the
// AC strategy from inside `ac_strategy`. It does *not* apply to the `hint`
// argument.
Status DecodeQuantField(BitReader* JXL_RESTRICT br,
                        ANSSymbolReader* JXL_RESTRICT decoder,
                        const std::vector<uint8_t>& context_map,
                        const Rect& rect_qf,
                        const AcStrategyImage& JXL_RESTRICT ac_strategy,
                        ImageI* JXL_RESTRICT quant_field, size_t base_context) {
  PROFILER_FUNC;
  const size_t xsize = rect_qf.xsize();
  const size_t ysize = rect_qf.ysize();
  const size_t stride = quant_field->PixelsPerRow();
  const size_t acs_stride = ac_strategy.PixelsPerRow();

  AuxOut aux_out;
  QuantFieldCoder::Decoder dec(
      br, decoder, &context_map, rect_qf.Row(quant_field, 0), stride,
      ac_strategy.ConstRow(rect_qf, 0), acs_stride, base_context, &aux_out);
  dec.Decode(xsize, ysize);
  return true;
}

void TokenizeAcStrategy(const Rect& rect, const AcStrategyImage& ac_strategy,
                        std::vector<Token>* JXL_RESTRICT output,
                        size_t base_context) {
  PROFILER_FUNC;
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  const size_t stride = ac_strategy.PixelsPerRow();

  AuxOut aux_out;
  AcStrategyCoder::Encoder enc(output, ac_strategy.ConstRow(rect, 0), stride,
                               base_context, &aux_out);
  enc.Encode(xsize, ysize);
}

Status DecodeAcStrategy(BitReader* JXL_RESTRICT br,
                        ANSSymbolReader* JXL_RESTRICT decoder,
                        const std::vector<uint8_t>& context_map,
                        const Rect& rect,
                        AcStrategyImage* JXL_RESTRICT ac_strategy,
                        size_t base_context) {
  PROFILER_FUNC;
  AuxOut aux_out;
  AcStrategyCoder::Decoder dec(br, decoder, &context_map, ac_strategy, rect,
                               base_context, &aux_out);
  JXL_RETURN_IF_ERROR(dec.Decode());
  return true;
}

void TokenizeARParameters(const Rect& rect, const ImageB& epf_sharpness,
                          const AcStrategyImage& ac_strategy,
                          std::vector<Token>* JXL_RESTRICT output,
                          size_t base_context) {
  PROFILER_FUNC;
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  const size_t stride = epf_sharpness.PixelsPerRow();
  const size_t acs_stride = ac_strategy.PixelsPerRow();

  AuxOut aux_out;
  ARParamsCoder::Encoder enc(output, rect.ConstRow(epf_sharpness, 0), stride,
                             ac_strategy.ConstRow(rect, 0), acs_stride,
                             base_context, &aux_out);
  enc.Encode(xsize, ysize);
}

Status DecodeARParameters(BitReader* br, ANSSymbolReader* decoder,
                          const std::vector<uint8_t>& context_map,
                          const Rect& rect, const AcStrategyImage& ac_strategy,
                          ImageB* epf_sharpness, size_t base_context) {
  PROFILER_FUNC;
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  const size_t stride = epf_sharpness->PixelsPerRow();
  const size_t acs_stride = ac_strategy.PixelsPerRow();

  AuxOut aux_out;
  ARParamsCoder::Decoder dec(
      br, decoder, &context_map, rect.Row(epf_sharpness, 0), stride,
      ac_strategy.ConstRow(rect, 0), acs_stride, base_context, &aux_out);
  dec.Decode(xsize, ysize);
  return true;
}

#include <hwy/end_target-inl.h>

#if HWY_ONCE
HWY_EXPORT(TokenizeCoefficients)
HWY_EXPORT(TokenizeQuantField)
HWY_EXPORT(DecodeQuantField)
HWY_EXPORT(TokenizeAcStrategy)
HWY_EXPORT(DecodeAcStrategy)
HWY_EXPORT(TokenizeARParameters)
HWY_EXPORT(DecodeARParameters)

#endif  // HWY_ONCE

}  // namespace jxl
