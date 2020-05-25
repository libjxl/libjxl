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

#include "jxl/compressed_dc.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/compressed_dc.cc"
#include <hwy/foreach_target.h>
//

#include <algorithm>
#include <array>
#include <hwy/aligned_allocator.h>
#include <memory>
#include <utility>
#include <vector>

#include "jxl/ac_strategy.h"
#include "jxl/ans_params.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/common.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/dec_cache.h"
#include "jxl/enc_cache.h"
#include "jxl/entropy_coder.h"
#include "jxl/image.h"
#include "jxl/predictor-inl.h"

#ifndef JXL_NUM_DC_CONTEXTS
#define JXL_NUM_DC_CONTEXTS 36
#endif

#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

class DcCoderBase {
  // Must be 16-byte aligned for SIMD Load/Store; easier to guarantee that
  // for a separately allocated POD struct instead of the base class.
  struct Aligned {
    HWY_ALIGN float mul_dc[4];
    HWY_ALIGN float inv_mul_dc[4];
    HWY_ALIGN float cmap_factor[4];
    HWY_ALIGN uint32_t extra_levels[4];

    // One padding value every 3 to have aligned stores.
    HWY_ALIGN float dc_dec[4 * kDcGroupDimInBlocks];
    HWY_ALIGN float dc_quant_field[4 * kDcGroupDimInBlocks];
  };

 public:
  enum {
    kContextsPerChannel = JXL_NUM_DC_CONTEXTS / 3,
    kNumResidualContexts = kContextsPerChannel - 4,
  };

  uint32_t JXL_INLINE ResidualToEncoded(size_t c, size_t residual) {
    const uint32_t extra_levels = aligned_->extra_levels[c];
    if (residual > 7) {
      return ((residual + (1 << extra_levels) / 2) >> extra_levels) +
             extra_levels;
    }
    // Approximate mapping of actual residual to encoded residual, used for
    // computing context.
    constexpr size_t kResidualLut[4][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 3, 4, 4, 5},
        {0, 1, 2, 3, 3, 3, 4, 4},
        {0, 1, 2, 3, 3, 3, 4, 4},
    };
    return kResidualLut[extra_levels][residual];
  }

  size_t Context(size_t c, size_t correct, size_t badness) {
    if (correct == 0) {
      JXL_ASSERT(badness != 0);
      // Discard error sign. Assumes that badness comes from PackSigned.
      badness = (badness + 1) >> 1;
      badness = ResidualToEncoded(c, badness);
      size_t badness_offset =
          std::min<uint32_t>(badness, kNumResidualContexts) - 1;
      return kContextsPerChannel * c + badness_offset;
    }
    return kContextsPerChannel * c + kNumResidualContexts +
           CeilLog2Nonzero(9 - correct);
  }

  // Stores the temporary buffer row in the final image.
  void FlushRow(size_t y) {
    for (size_t c = 0; c < 3; c++) {
      float* JXL_RESTRICT dc_dec_row =
          dc_dec_rows_base_[c] + y * dc_dec_stride_;
      float* JXL_RESTRICT dc_quant_field_row =
          dc_quant_field_rows_base_[c] + y * dc_quant_field_stride_;
      for (size_t x = 0; x < xsize_; x++) {
        dc_quant_field_row[x] = aligned_->dc_quant_field[4 * x + c];
        dc_dec_row[x] = aligned_->dc_dec[4 * x + c];
      }
    }
  }

  void StartRow(size_t y) {
    if (y != 0) FlushRow(y - 1);
  }

  float GetDcValue(size_t c, size_t y, size_t x) {
    aligned_->dc_dec[4 * x + c] = 0;
    return dc_rows_[c][y * dc_stride_ + x] -
           aligned_->dc_dec[4 * x + 1] * aligned_->cmap_factor[c];
  }

  DcCoderBase(size_t xsize, const uint32_t* JXL_RESTRICT extra_levels,
              const float* JXL_RESTRICT mul_dc,
              const float* JXL_RESTRICT inv_mul_dc,
              const float* JXL_RESTRICT cmap_factor,
              const float* JXL_RESTRICT* JXL_RESTRICT dc_rows, size_t dc_stride,
              float** JXL_RESTRICT dc_dec_rows, size_t dc_dec_stride,
              float** JXL_RESTRICT dc_quant_field_rows,
              size_t dc_quant_field_stride)
      // TODO(janwas): pass through allocator funcs
      : aligned_(hwy::AllocateSingleAligned<Aligned>()),
        xsize_(xsize),
        dc_rows_(dc_rows),
        dc_stride_(dc_stride),
        dc_dec_rows_base_(dc_dec_rows),
        dc_dec_stride_(dc_dec_stride),
        dc_quant_field_rows_base_(dc_quant_field_rows),
        dc_quant_field_stride_(dc_quant_field_stride) {
    for (size_t i = 0; i < 4; ++i) {
      aligned_->mul_dc[i] = i == 3 ? 0 : mul_dc[i] / (1 << extra_levels[i]);
      aligned_->inv_mul_dc[i] = inv_mul_dc == nullptr ? 0.0f : inv_mul_dc[i];
      aligned_->cmap_factor[i] = cmap_factor[i];
      aligned_->extra_levels[i] = extra_levels[i];
    }
  }

  float JXL_INLINE Dequantize(size_t c, int val) {
    return val * aligned_->mul_dc[c];
  }

  int Quantize(size_t c, float val) {
    JXL_DASSERT(val >= 0);
    val *= aligned_->inv_mul_dc[c];
    const uint32_t extra_levels = aligned_->extra_levels[c];
    if (extra_levels == 0) {
      return static_cast<int>(val + 0.5f);
    }
    if (extra_levels == 1) {
      if (val <= 0.25f) {
        return 0;
      }
      if (val < 0.75f) {
        return 1;
      }
      return static_cast<int>(val + 0.5f) + 1;
    }
    if (extra_levels == 2) {
      if (val <= 0.125f) {
        return 0;
      }
      if (val < 0.375f) {
        return 1;
      }
      if (val < 0.75f) {
        return 2;
      }
      return static_cast<int>(val + 0.5f) + 2;
    }
    JXL_DASSERT(extra_levels == 3);
    if (val <= 0.0625f) {
      return 0;
    }
    if (val < 0.1875f) {
      return 1;
    }
    if (val < 0.375f) {
      return 2;
    }
    if (val < 0.75f) {
      return 3;
    }
    return static_cast<int>(val + 0.5f) + 3;
  }

  uint32_t ComputeResidual(size_t c, int32_t predicted, float actual) {
    float fpred = Dequantize(c, predicted);
    float fdelta = actual - fpred;
    bool negative = fdelta < 0;
    fdelta = std::abs(fdelta);

    int32_t delta = Quantize(c, fdelta);

    delta = negative ? -delta : delta;
    return PackSigned(delta);
  }

  int32_t JXL_INLINE ApplyResidual(size_t c, int32_t predicted,
                                   uint32_t residual,
                                   float* JXL_RESTRICT interval) {
    int32_t delta = UnpackSigned(residual);
    bool negative = delta < 0;
    delta = std::abs(delta);

    const uint32_t extra_levels = aligned_->extra_levels[c];
    if (static_cast<uint32_t>(delta) > extra_levels) {
      delta = (delta - extra_levels) << extra_levels;
      *interval *= 1 << extra_levels;
    } else if (delta != 0) {
      delta = 1 << (delta - 1);
      *interval *= delta;
    }

    delta = negative ? -delta : delta;
    return predicted + delta;
  }

  // Encoder only
  void ComputeDecoded(size_t c, size_t x, size_t y,
                      const int32_t* JXL_RESTRICT predictions,
                      const uint32_t* JXL_RESTRICT residuals,
                      int32_t* JXL_RESTRICT decoded) {
    int32_t prediction = predictions[c];

    aligned_->dc_quant_field[4 * x + c] = aligned_->mul_dc[c];
    aligned_->dc_dec[4 * x + c] = 0;

    int quant = ApplyResidual(c, prediction, residuals[c],
                              &aligned_->dc_quant_field[4 * x + c]);

    decoded[c] = quant;

    float dcd = Dequantize(c, quant);
    aligned_->dc_dec[4 * x + c] =
        dcd + aligned_->dc_dec[4 * x + 1] * aligned_->cmap_factor[c];
  }

  void ComputeDecoded3(size_t x, size_t y,
                       const int32_t* JXL_RESTRICT predictions,
                       const uint32_t* JXL_RESTRICT residuals,
                       int32_t* JXL_RESTRICT decoded) {
#if HWY_CAP_VARIABLE_SHIFT && HWY_TARGET != HWY_SCALAR
    using DU = HWY_CAPPED(uint32_t, 4);
    using DI = HWY_CAPPED(int32_t, 4);
    using DF = HWY_CAPPED(float, 4);
    const DI di;
    const DU du;
    const auto residual = Load(du, residuals);

    // UnpackSigned
    const auto one = Set(du, 1);
    const auto is_negative = TestBit(residual, one);  // sign bit = LSB
    // Workaround for >> rounding to neg infinity instead of zero:
    // If positive, LSB is 0 so the add disappears after the shift.
    // If negative, LSB is 1 so the add carries and we have incremented the
    // shifted result by one.
    const auto delta = ShiftRight<1>(residual + one);

    // Expand extra levels
    const auto extra_levels = Load(du, aligned_->extra_levels);
    // Build mask of "delta > extra_levels"
    const auto is_hi = MaskFromVec(
        BitCast(du, ShiftRight<31>(BitCast(di, extra_levels - delta))));
    const auto delta_hi = (delta - extra_levels) << extra_levels;
    const auto is_zero = delta == Zero(du);
    const auto deltam1 = delta - one;
    const auto delta_lo = one << deltam1;
    const auto final_delta =
        IfThenElse(is_hi, delta_hi, IfThenZeroElse(is_zero, delta_lo));

    // Apply prediction
    const auto prediction = BitCast(du, Load(di, predictions));
    const auto quantized =
        BitCast(di, IfThenElse(is_negative, prediction - final_delta,
                               prediction + final_delta));
    // Not aligned in predictor.h.
    StoreU(quantized, di, decoded);

    const auto mul = Load(DF(), aligned_->mul_dc);
    // Compute adjusted quant interval
    const auto mul_factor_hi = one << extra_levels;
    const auto mul_factor = BitCast(
        di,
        IfThenElse(is_hi, mul_factor_hi, IfThenElse(is_zero, one, delta_lo)));
    const auto adj_mul = mul * ConvertTo(DF(), mul_factor);
    Store(adj_mul, DF(), aligned_->dc_quant_field + 4 * x);

    const auto dequant = ConvertTo(DF(), quantized) * mul;
    const auto y_dequant = Broadcast<1>(dequant);
    // No MulAdd - causes different results in compressed_dc_test.
    const auto correlated =
        Load(DF(), aligned_->cmap_factor) * y_dequant + dequant;
    Store(correlated, DF(), aligned_->dc_dec + 4 * x);
#else
    for (size_t c : {1, 0, 2}) {
      ComputeDecoded(c, x, y, predictions, residuals, decoded);
    }
#endif
  }

 private:
  hwy::AlignedUniquePtr<Aligned> aligned_;

  const size_t xsize_;

  // Input (encoder-side) values.
  const float* JXL_RESTRICT* JXL_RESTRICT dc_rows_;
  const size_t dc_stride_;

  // Output (decoder-side) values.
  float** JXL_RESTRICT dc_dec_rows_base_;
  const size_t dc_dec_stride_;

  float** JXL_RESTRICT dc_quant_field_rows_base_;
  const size_t dc_quant_field_stride_;
};

class DCEncoder : public DcPredictor<DCEncoder> {
 public:
  DCEncoder(size_t xsize, const uint32_t* JXL_RESTRICT extra_levels,
            const float* JXL_RESTRICT mul_dc,
            const float* JXL_RESTRICT inv_mul_dc,
            const float* JXL_RESTRICT cmap_factor,
            const float* JXL_RESTRICT* JXL_RESTRICT dc_rows, size_t dc_stride,
            float** JXL_RESTRICT dc_dec_rows, size_t dc_dec_stride,
            float** JXL_RESTRICT dc_quant_field_rows,
            size_t dc_quant_field_stride)
      : predictor_(xsize, extra_levels, mul_dc, inv_mul_dc, cmap_factor,
                   dc_rows, dc_stride, dc_dec_rows, dc_dec_stride,
                   dc_quant_field_rows, dc_quant_field_stride) {}

  void Run(size_t xsize, size_t ysize, std::vector<Token>* JXL_RESTRICT tokens,
           AuxOut* JXL_RESTRICT aux_out) {
    tokens_ = tokens;
    DcPredictor<DCEncoder>::Run(xsize, ysize, aux_out);
    predictor_.FlushRow(ysize - 1);
  }

  JXL_INLINE void Prediction(size_t x, size_t y,
                             const int32_t* JXL_RESTRICT predictions,
                             const uint32_t* JXL_RESTRICT num_correct,
                             const uint32_t* JXL_RESTRICT min_error,
                             int32_t* JXL_RESTRICT decoded) {
    uint32_t residuals[3];
    // TODO(veluca): investigate possible SIMD-fication of this code.

    int32_t prediction = predictions[1];
    float dcv = predictor_.GetDcValue(1, y, x);
    residuals[1] = predictor_.ComputeResidual(1, prediction, dcv);
    predictor_.ComputeDecoded(1, x, y, predictions, residuals, decoded);

    prediction = predictions[0];
    dcv = predictor_.GetDcValue(0, y, x);
    residuals[0] = predictor_.ComputeResidual(0, prediction, dcv);
    predictor_.ComputeDecoded(0, x, y, predictions, residuals, decoded);

    prediction = predictions[2];
    dcv = predictor_.GetDcValue(2, y, x);
    residuals[2] = predictor_.ComputeResidual(2, prediction, dcv);
    predictor_.ComputeDecoded(2, x, y, predictions, residuals, decoded);

    int ctx = predictor_.Context(1, num_correct[1], min_error[1]);
    TokenizeHybridUint(ctx, residuals[1], tokens_);
    ctx = predictor_.Context(0, num_correct[0], min_error[0]);
    TokenizeHybridUint(ctx, residuals[0], tokens_);
    ctx = predictor_.Context(2, num_correct[2], min_error[2]);
    TokenizeHybridUint(ctx, residuals[2], tokens_);
  }

  void StartRow(size_t y) { predictor_.StartRow(y); }

 private:
  HWY_ALIGN DcCoderBase predictor_;
  std::vector<Token>* JXL_RESTRICT tokens_;
};

class DCDecoder : public DcPredictor<DCDecoder> {
 public:
  DCDecoder(size_t xsize, const uint32_t* JXL_RESTRICT extra_levels,
            const float* JXL_RESTRICT mul_dc,
            const float* JXL_RESTRICT cmap_factor,
            float** JXL_RESTRICT dc_dec_rows, size_t dc_dec_stride,
            float** JXL_RESTRICT dc_quant_field_rows,
            size_t dc_quant_field_stride)
      : predictor_(xsize, extra_levels, mul_dc, nullptr, cmap_factor, nullptr,
                   0, dc_dec_rows, dc_dec_stride, dc_quant_field_rows,
                   dc_quant_field_stride) {}

  void Run(size_t xsize, size_t ysize, BitReader* JXL_RESTRICT br,
           ANSSymbolReader* JXL_RESTRICT decoder,
           const std::vector<uint8_t>& JXL_RESTRICT context_map,
           AuxOut* JXL_RESTRICT aux_out) {
    br_ = br;
    decoder_ = decoder;
    context_map_ = &context_map;
    DcPredictor<DCDecoder>::Run(xsize, ysize, aux_out);
    predictor_.FlushRow(ysize - 1);
  }

  JXL_INLINE void Prediction(size_t x, size_t y,
                             const int32_t* JXL_RESTRICT predictions,
                             const uint32_t* JXL_RESTRICT num_correct,
                             const uint32_t* JXL_RESTRICT min_error,
                             int32_t* JXL_RESTRICT decoded) {
    HWY_ALIGN uint32_t residuals[4];
    residuals[3] = 0;

    for (size_t c : {1, 0, 2}) {
      int ctx = predictor_.Context(c, num_correct[c], min_error[c]);
      residuals[c] = decoder_->ReadHybridUint(ctx, br_, *context_map_);
    }
    predictor_.ComputeDecoded3(x, y, predictions, residuals, decoded);
  }
  void StartRow(size_t y) { predictor_.StartRow(y); }

 private:
  HWY_ALIGN DcCoderBase predictor_;
  BitReader* JXL_RESTRICT br_;
  ANSSymbolReader* JXL_RESTRICT decoder_;
  const std::vector<uint8_t>* JXL_RESTRICT context_map_;
};

HWY_ALIGN const uint32_t kExtraLevels[4][4] = {
    {0, 0, 0, /*unused*/ 0},
    {1, 1, 1, /*unused*/ 0},
    {1, 2, 1, /*unused*/ 0},
    {2, 3, 2, /*unused*/ 0},
};

void TokenizeDC(size_t group_index, const Image3F& dc,
                PassesEncoderState* JXL_RESTRICT enc_state, AuxOut* aux_out) {
  const Rect rect = enc_state->shared.DCGroupRect(group_index);
  const Quantizer& quantizer = enc_state->shared.quantizer;
  const ColorCorrelationMap& cmap = enc_state->shared.cmap;
  Image3F* dc_dec = &enc_state->shared.dc_storage;
  Image3F* dc_quant_field = &enc_state->shared.dc_quant_field;
  const float* JXL_RESTRICT dc_rows[3] = {rect.ConstPlaneRow(dc, 0, 0),
                                          rect.ConstPlaneRow(dc, 1, 0),
                                          rect.ConstPlaneRow(dc, 2, 0)};
  const size_t dc_stride = dc.PixelsPerRow();
  // Not restrict: the predictor will create aliasing pointers in these arrays,
  // and they are not used directly anyway.
  float* dc_dec_rows[3] = {rect.PlaneRow(dc_dec, 0, 0),
                           rect.PlaneRow(dc_dec, 1, 0),
                           rect.PlaneRow(dc_dec, 2, 0)};
  const size_t dc_dec_stride = dc_dec->PixelsPerRow();

  float* dc_quant_field_rows[3] = {rect.PlaneRow(dc_quant_field, 0, 0),
                                   rect.PlaneRow(dc_quant_field, 1, 0),
                                   rect.PlaneRow(dc_quant_field, 2, 0)};
  const size_t dc_quant_field_stride = dc_quant_field->PixelsPerRow();

  HWY_ALIGN DCEncoder coder(
      rect.xsize(), kExtraLevels[enc_state->extra_dc_levels[group_index]],
      quantizer.MulDC(), quantizer.InvMulDC(), cmap.DCFactors(), dc_rows,
      dc_stride, dc_dec_rows, dc_dec_stride, dc_quant_field_rows,
      dc_quant_field_stride);
  coder.Run(rect.xsize(), rect.ysize(), &enc_state->dc_tokens[group_index],
            aux_out);
}

void DecodeDC(BitReader* br, ANSSymbolReader* decoder,
              const std::vector<uint8_t>& context_map, const Rect& rect,
              const float* mul_dc, const float* cmap_factor, int extra_levels,
              Image3F* JXL_RESTRICT dc, Image3F* JXL_RESTRICT dc_quant_field,
              AuxOut* aux_out) {
  PROFILER_FUNC;
  // Not restrict: the predictor will create aliasing pointers in these arrays,
  // and they are not used directly anyway.
  float* dc_rows[3] = {rect.PlaneRow(dc, 0, 0), rect.PlaneRow(dc, 1, 0),
                       rect.PlaneRow(dc, 2, 0)};
  const size_t dc_stride = dc->PixelsPerRow();
  float* dc_quant_field_rows[3] = {rect.PlaneRow(dc_quant_field, 0, 0),
                                   rect.PlaneRow(dc_quant_field, 1, 0),
                                   rect.PlaneRow(dc_quant_field, 2, 0)};
  const size_t dc_quant_field_stride = dc_quant_field->PixelsPerRow();

  HWY_ALIGN DCDecoder coder(rect.xsize(), kExtraLevels[extra_levels], mul_dc,
                            cmap_factor, dc_rows, dc_stride,
                            dc_quant_field_rows, dc_quant_field_stride);
  coder.Run(rect.xsize(), rect.ysize(), br, decoder, context_map, aux_out);
}

using D = HWY_FULL(float);
using DScalar = HWY_CAPPED(float, 1);
using V = Vec<D>;

// TODO(veluca): optimize constants.
const float w1 = 0.20345139757231578f;
const float w2 = 0.0334829185968739f;
const float w0 = 1.0f - 4.0f * (w1 + w2);

template <class V>
V MaxWorkaround(V a, V b) {
#if (HWY_TARGET == HWY_AVX3) && HWY_COMPILER_CLANG <= 800
  // Prevents "Do not know how to split the result of this operator" error
  return IfThenElse(a > b, a, b);
#else
  return Max(a, b);
#endif
}

template <typename D>
JXL_INLINE void ComputePixelChannel(
    const D d, const float* JXL_RESTRICT dc_quant_row,
    const float* JXL_RESTRICT row_top, const float* JXL_RESTRICT row,
    const float* JXL_RESTRICT row_bottom, Vec<D>* JXL_RESTRICT mc,
    Vec<D>* JXL_RESTRICT sm, Vec<D>* JXL_RESTRICT gap, size_t x) {
  const auto tl = LoadU(d, row_top + x - 1);
  const auto tc = Load(d, row_top + x);
  const auto tr = LoadU(d, row_top + x + 1);

  const auto ml = LoadU(d, row + x - 1);
  *mc = Load(d, row + x);
  const auto mr = LoadU(d, row + x + 1);

  const auto bl = LoadU(d, row_bottom + x - 1);
  const auto bc = Load(d, row_bottom + x);
  const auto br = LoadU(d, row_bottom + x + 1);

  const auto w_center = Set(d, w0);
  const auto w_side = Set(d, w1);
  const auto w_corner = Set(d, w2);

  const auto corner = tl + tr + bl + br;
  const auto side = ml + mr + tc + bc;
  *sm = corner * w_corner + side * w_side + *mc * w_center;

  const auto dc_quant = Load(d, dc_quant_row + x);
  *gap = MaxWorkaround(*gap, Abs((*mc - *sm) / dc_quant));
}

template <typename D>
JXL_INLINE void ComputePixel(
    const float* JXL_RESTRICT* JXL_RESTRICT dc_quant_rows,
    const float* JXL_RESTRICT* JXL_RESTRICT rows_top,
    const float* JXL_RESTRICT* JXL_RESTRICT rows,
    const float* JXL_RESTRICT* JXL_RESTRICT rows_bottom,
    float* JXL_RESTRICT* JXL_RESTRICT out_rows, size_t x) {
  const D d;
  auto mc_x = Undefined(d);
  auto mc_y = Undefined(d);
  auto mc_b = Undefined(d);
  auto sm_x = Undefined(d);
  auto sm_y = Undefined(d);
  auto sm_b = Undefined(d);
  auto gap = Set(d, 0.5f);
  ComputePixelChannel(d, dc_quant_rows[0], rows_top[0], rows[0], rows_bottom[0],
                      &mc_x, &sm_x, &gap, x);
  ComputePixelChannel(d, dc_quant_rows[1], rows_top[1], rows[1], rows_bottom[1],
                      &mc_y, &sm_y, &gap, x);
  ComputePixelChannel(d, dc_quant_rows[2], rows_top[2], rows[2], rows_bottom[2],
                      &mc_b, &sm_b, &gap, x);
  auto factor = MulAdd(Set(d, -4.0f), gap, Set(d, 3.0f));
  factor = ZeroIfNegative(factor);

  auto out = MulAdd(sm_x - mc_x, factor, mc_x);
  Store(out, d, out_rows[0] + x);
  out = MulAdd(sm_y - mc_y, factor, mc_y);
  Store(out, d, out_rows[1] + x);
  out = MulAdd(sm_b - mc_b, factor, mc_b);
  Store(out, d, out_rows[2] + x);
}

void AdaptiveDCSmoothing(const Image3F& dc_quant_field, Image3F* dc,
                         ThreadPool* pool) {
  const size_t xsize = dc->xsize();
  const size_t ysize = dc->ysize();
  if (ysize <= 2 || xsize <= 2) return;

  // TODO(veluca): use tile-based processing?
  // TODO(veluca): decide if changes to the y channel should be propagated to
  // the x and b channels through color correlation.
  JXL_ASSERT(w1 + w2 < 0.25f);

  PROFILER_FUNC;

  Image3F smoothed(xsize, ysize);
  // Fill in borders that the loop below will not. First and last are unused.
  for (size_t c = 0; c < 3; c++) {
    for (size_t y : {size_t(0), ysize - 1}) {
      memcpy(smoothed.PlaneRow(c, y), dc->PlaneRow(c, y),
             xsize * sizeof(float));
    }
  }
  auto process_row = [&](int y, int /*thread*/) {
    const float* JXL_RESTRICT rows_top[3]{
        dc->ConstPlaneRow(0, y - 1),
        dc->ConstPlaneRow(1, y - 1),
        dc->ConstPlaneRow(2, y - 1),
    };
    const float* JXL_RESTRICT rows[3] = {
        dc->ConstPlaneRow(0, y),
        dc->ConstPlaneRow(1, y),
        dc->ConstPlaneRow(2, y),
    };
    const float* JXL_RESTRICT rows_bottom[3] = {
        dc->ConstPlaneRow(0, y + 1),
        dc->ConstPlaneRow(1, y + 1),
        dc->ConstPlaneRow(2, y + 1),
    };
    const float* JXL_RESTRICT rows_dc_quant[3] = {
        dc_quant_field.ConstPlaneRow(0, y),
        dc_quant_field.ConstPlaneRow(1, y),
        dc_quant_field.ConstPlaneRow(2, y),
    };
    float* JXL_RESTRICT rows_out[3] = {
        smoothed.PlaneRow(0, y),
        smoothed.PlaneRow(1, y),
        smoothed.PlaneRow(2, y),
    };
    for (size_t x : {size_t(0), xsize - 1}) {
      for (size_t c = 0; c < 3; c++) {
        rows_out[c][x] = rows[c][x];
      }
    }

    size_t x = 1;
    // First pixels
    const size_t N = Lanes(D());
    for (; x < std::min(N, xsize - 1); x++) {
      ComputePixel<DScalar>(rows_dc_quant, rows_top, rows, rows_bottom,
                            rows_out, x);
    }
    // Full vectors.
    for (; x + N <= xsize - 1; x += N) {
      ComputePixel<D>(rows_dc_quant, rows_top, rows, rows_bottom, rows_out, x);
    }
    // Last pixels.
    for (; x < xsize - 1; x++) {
      ComputePixel<DScalar>(rows_dc_quant, rows_top, rows, rows_bottom,
                            rows_out, x);
    }
  };
  RunOnPool(pool, 1, ysize - 1, ThreadPool::SkipInit(), process_row,
            "DCSmoothingRow");
  dc->Swap(smoothed);
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(TokenizeDC)
HWY_EXPORT(DecodeDC)
HWY_EXPORT(AdaptiveDCSmoothing)

constexpr size_t kCmapBaseContext = JXL_NUM_DC_CONTEXTS;
constexpr size_t kControlFieldBaseContext = kCmapBaseContext + kCmapContexts;
constexpr size_t kNumDCContexts =
    kControlFieldBaseContext + kNumControlFieldContexts;

Status EncodeDCGroup(const PassesEncoderState& enc_state, size_t group_idx,
                     BitWriter* writer, AuxOut* aux_out) {
  const Rect rect = enc_state.shared.DCGroupRect(group_idx);
  // Single set of tokens to avoid overhead of multiple ANS streams.
  std::vector<std::vector<Token>> tokens(1);

  if (!(enc_state.shared.frame_header.flags & FrameHeader::kUseDcFrame)) {
    JXL_ASSERT(enc_state.extra_dc_levels[group_idx] < 4);
    // Copy DC tokens from enc cache.
    tokens[0] = enc_state.dc_tokens[group_idx];

    BitWriter::Allotment allotment(writer, 1 + 2 * kBitsPerByte);
    // Write number of extra DC levels.
    writer->Write(2, enc_state.extra_dc_levels[group_idx]);
    ReclaimAndCharge(writer, &allotment, kLayerDC, aux_out);
  }

  JXL_ASSERT(rect.x0() % kColorTileDimInBlocks == 0);
  JXL_ASSERT(rect.y0() % kColorTileDimInBlocks == 0);
  Rect cmap_rect(rect.x0() / kColorTileDimInBlocks,
                 rect.y0() / kColorTileDimInBlocks,
                 DivCeil(rect.xsize(), kColorTileDimInBlocks),
                 DivCeil(rect.ysize(), kColorTileDimInBlocks));

  ChooseEncodeColorMap()(enc_state.shared.cmap, cmap_rect, &tokens[0],
                         kCmapBaseContext, aux_out);

  ChooseTokenizeAcStrategy()(rect, enc_state.shared.ac_strategy, &tokens[0],
                             kControlFieldBaseContext);

  ChooseTokenizeQuantField()(rect, enc_state.shared.raw_quant_field,
                             enc_state.shared.ac_strategy, &tokens[0],
                             kControlFieldBaseContext + kAcStrategyContexts);

  if (enc_state.shared.image_features.loop_filter.epf) {
    ChooseTokenizeARParameters()(
        rect, enc_state.shared.epf_sharpness, enc_state.shared.ac_strategy,
        &tokens[0],
        kControlFieldBaseContext + kAcStrategyContexts + kQuantFieldContexts);
  }

  std::vector<uint8_t> context_map;
  EntropyEncodingData codes;
  HistogramParams hist_params(enc_state.cparams.speed_tier, kNumDCContexts);
  BuildAndEncodeHistograms(hist_params, kNumDCContexts, tokens, &codes,
                           &context_map, writer, kLayerDC, aux_out);
  WriteTokens(tokens[0], codes, context_map, writer, kLayerDC, aux_out);

  return true;
}

// `rect`: block units.
Status DecodeDCGroup(BitReader* reader, size_t group_idx,
                     PassesDecoderState* dec_state, AuxOut* aux_out) {
  PROFILER_FUNC;
  int extra_dc_levels = 0;

  if (!(dec_state->shared->frame_header.flags & FrameHeader::kUseDcFrame)) {
    extra_dc_levels = reader->ReadFixedBits<2>();
  }

  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(DecodeHistograms(
      reader, kNumDCContexts, ANS_MAX_ALPHA_SIZE, &code, &context_map));
  ANSSymbolReader decoder(&code, reader);

  const Rect rect = dec_state->shared->DCGroupRect(group_idx);
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  Rect rect0(0, 0, xsize, ysize);
  if (!(dec_state->shared->frame_header.flags & FrameHeader::kUseDcFrame)) {
    ChooseDecodeDC()(reader, &decoder, context_map, rect,
                     dec_state->shared->quantizer.MulDC(),
                     dec_state->shared->cmap.DCFactors(), extra_dc_levels,
                     &dec_state->shared_storage.dc_storage,
                     &dec_state->shared_storage.dc_quant_field, aux_out);
  }

  JXL_ASSERT(rect.x0() % kColorTileDimInBlocks == 0);
  JXL_ASSERT(rect.y0() % kColorTileDimInBlocks == 0);
  Rect cmap_rect(rect.x0() / kColorTileDimInBlocks,
                 rect.y0() / kColorTileDimInBlocks,
                 DivCeil(rect.xsize(), kColorTileDimInBlocks),
                 DivCeil(rect.ysize(), kColorTileDimInBlocks));

  JXL_RETURN_IF_ERROR(ChooseDecodeColorMap()(
      reader, &decoder, context_map, &dec_state->shared_storage.cmap, cmap_rect,
      kCmapBaseContext, aux_out));

  if (!ChooseDecodeAcStrategy()(reader, &decoder, context_map, rect,
                                &dec_state->shared_storage.ac_strategy,
                                kControlFieldBaseContext)) {
    return JXL_FAILURE("Failed to decode AcStrategy.");
  }

  if (!ChooseDecodeQuantField()(
          reader, &decoder, context_map, rect,
          dec_state->shared_storage.ac_strategy,
          &dec_state->shared_storage.raw_quant_field,
          kControlFieldBaseContext + kAcStrategyContexts)) {
    return JXL_FAILURE("Failed to decode QuantField.");
  }

  if (dec_state->shared->image_features.loop_filter.epf &&
      !ChooseDecodeARParameters()(reader, &decoder, context_map, rect,
                                  dec_state->shared_storage.ac_strategy,
                                  &dec_state->shared_storage.epf_sharpness,
                                  kControlFieldBaseContext +
                                      kAcStrategyContexts +
                                      kQuantFieldContexts)) {
    return JXL_FAILURE("Failed to decode ARParameters.");
  }

  if (!decoder.CheckANSFinalState()) {
    return JXL_FAILURE("DC group: ANS checksum failure.");
  }

  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
