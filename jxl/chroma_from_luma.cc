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

#include "jxl/chroma_from_luma.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/chroma_from_luma.cc"
#include <hwy/foreach_target.h>

#include <float.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "jxl/aux_out.h"
#include "jxl/base/bits.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/enc_dct.h"
#include "jxl/entropy_coder.h"
#include "jxl/image_ops.h"
#include "jxl/modular/encoding/encoding.h"
#include "jxl/quantizer.h"

#include "jxl/dec_transforms-inl.h"
#include "jxl/enc_transforms-inl.h"
#include "jxl/predictor-inl.h"

#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

static HWY_FULL(float) df;

struct CFLFunction {
  static constexpr float kCoeff = 1.f / 3;
  static constexpr float kThres = 100.0f;
  static constexpr float kInvColorFactor = 1.0f / kDefaultColorFactor;
  CFLFunction(const float* values_m, const float* values_s, size_t num,
              float offset, float base, float distance_mul)
      : values_m(values_m),
        values_s(values_s),
        num(num),
        offset(offset),
        base(base),
        distance_mul(distance_mul) {}

  // Returns f'(x), where f is 1/3 * sum ((|color residual| + 1)^2-1) +
  // distance_mul * (x-offset)^2 * num.
  float Compute(float x, float eps, float* fpeps, float* fmeps) const {
    float first_derivative = 2 * distance_mul * num * (x - offset);
    float first_derivative_peps = 2 * distance_mul * num * (x + eps - offset);
    float first_derivative_meps = 2 * distance_mul * num * (x - eps - offset);

    const auto inv_color_factor = Set(df, kInvColorFactor);
    const auto thres = Set(df, kThres);
    const auto coeffx2 = Set(df, kCoeff * 2.0f);
    const auto one = Set(df, 1.0f);
    const auto zero = Set(df, 0.0f);
    const auto offset_v = Set(df, offset);
    const auto base_v = Set(df, base);
    const auto x_v = Set(df, x);
    const auto xpe_v = Set(df, x + eps);
    const auto xme_v = Set(df, x - eps);
    auto fd_v = Zero(df);
    auto fdpe_v = Zero(df);
    auto fdme_v = Zero(df);
    JXL_ASSERT(num % Lanes(df) == 0);

    for (size_t i = 0; i < num; i += Lanes(df)) {
      // color residual = ax + b
      const auto a = inv_color_factor * Load(df, values_m + i);
      const auto b = base_v * Load(df, values_m + i) - offset_v * a -
                     Load(df, values_s + i);
      const auto v = a * x_v + b;
      const auto vpe = a * xpe_v + b;
      const auto vme = a * xme_v + b;
      const auto av = Abs(v);
      const auto avpe = Abs(vpe);
      const auto avme = Abs(vme);
      auto d = coeffx2 * (av + one) * a;
      auto dpe = coeffx2 * (avpe + one) * a;
      auto dme = coeffx2 * (avme + one) * a;
      d = IfThenElse(v < zero, zero - d, d);
      dpe = IfThenElse(vpe < zero, zero - dpe, dpe);
      dme = IfThenElse(vme < zero, zero - dme, dme);
      fd_v += IfThenElse(av >= thres, zero, d);
      fdpe_v += IfThenElse(av >= thres, zero, dpe);
      fdme_v += IfThenElse(av >= thres, zero, dme);
    }

    *fpeps = first_derivative_peps + GetLane(SumOfLanes(fdpe_v));
    *fmeps = first_derivative_meps + GetLane(SumOfLanes(fdme_v));
    return first_derivative + GetLane(SumOfLanes(fd_v));
  }

  const float* JXL_RESTRICT values_m;
  const float* JXL_RESTRICT values_s;
  size_t num;
  float offset;
  float base;
  float distance_mul;
};

size_t FindBestMultiplier(const float* values_m, const float* values_s,
                          size_t num, float offset, float base,
                          float distance_mul) {
  constexpr float eps = 1;
  constexpr float kClamp = 20.0f;
  CFLFunction fn(values_m, values_s, num, offset, base, distance_mul);
  float x = offset;
  // Up to 20 Newton iterations, with approximate derivatives.
  // Derivatives are approximate due to the high amount of noise in the exact
  // derivatives.
  for (size_t i = 0; i < 20; i++) {
    float dfpeps, dfmeps;
    float df = fn.Compute(x, eps, &dfpeps, &dfmeps);
    float ddf = (dfpeps - dfmeps) / (2 * eps);
    float step = df / ddf;
    x -= std::min(kClamp, std::max(-kClamp, step));
    if (std::abs(step) < 3e-3) break;
  }
  return std::max(0.0f, std::min(255.0f, std::roundf(x)));
}

template <int MAIN_CHANNEL, int SIDE_CHANNEL, bool use_dct8, int SCALE,
          int OFFSET>
JXL_NOINLINE void FindBestCorrelation(
    const Image3F& opsin, ImageB* JXL_RESTRICT map, int* JXL_RESTRICT dc,
    float base, const DequantMatrices& dequant,
    const AcStrategyImage* ac_strategy, const ImageI* raw_quant_field,
    const Quantizer* quantizer, ThreadPool* pool) {
  size_t xsize_blocks = opsin.xsize() / kBlockDim;
  size_t ysize_blocks = opsin.ysize() / kBlockDim;
  // First row: main channel
  // Second row: side channel
  ImageF dc_values(RoundUpTo(xsize_blocks * ysize_blocks, Lanes(df)), 2);
  float* JXL_RESTRICT dc_values_m = dc_values.Row(0);
  float* JXL_RESTRICT dc_values_s = dc_values.Row(1);

  JXL_ASSERT(dc_values.xsize() != 0);
  // Zero-fill the last lanes
  for (size_t y = 0; y < 2; y++) {
    for (size_t x = dc_values.xsize() - Lanes(df); x < dc_values.xsize(); x++) {
      dc_values.Row(y)[x] = 0;
    }
  }

  constexpr float kDistanceMultiplierDC = 1e-5f;
  constexpr float kDistanceMultiplierAC = 1e-3f;

  auto process_row = [&](int ty, int thread) {
    HWY_ALIGN_MAX float block_m[AcStrategy::kMaxCoeffArea];
    HWY_ALIGN_MAX float block_s[AcStrategy::kMaxCoeffArea];
    HWY_ALIGN_MAX float coeffs_m[kColorTileDim * kColorTileDim];
    HWY_ALIGN_MAX float coeffs_s[kColorTileDim * kColorTileDim];
    HWY_ALIGN_MAX float
        dc_m[AcStrategy::kMaxCoeffBlocks * AcStrategy::kMaxCoeffBlocks] = {};
    HWY_ALIGN_MAX float
        dc_s[AcStrategy::kMaxCoeffBlocks * AcStrategy::kMaxCoeffBlocks] = {};
    uint8_t* JXL_RESTRICT row_out = map->Row(ty);
    for (size_t tx = 0; tx < map->xsize(); ++tx) {
      const size_t y0 = ty * kColorTileDimInBlocks;
      const size_t x0 = tx * kColorTileDimInBlocks;
      const size_t y1 = std::min<size_t>(y0 + kColorTileDimInBlocks,
                                         opsin.ysize() / kBlockDim);
      const size_t x1 = std::min<size_t>(x0 + kColorTileDimInBlocks,
                                         opsin.xsize() / kBlockDim);
      size_t num_ac = 0;

      for (size_t y = y0; y < y1; ++y) {
        const float* JXL_RESTRICT row_m =
            opsin.ConstPlaneRow(MAIN_CHANNEL, y * kBlockDim);
        const float* JXL_RESTRICT row_s =
            opsin.ConstPlaneRow(SIDE_CHANNEL, y * kBlockDim);
        size_t stride = opsin.PixelsPerRow();

        for (size_t x = x0; x < x1; x++) {
          AcStrategy acs =
              use_dct8 ? AcStrategy::FromRawStrategy(AcStrategy::Type::DCT)
                       : ac_strategy->ConstRow(y)[x];
          if (!acs.IsFirstBlock()) continue;
          size_t xs = acs.covered_blocks_x();
          TransformFromPixels(acs.Strategy(), row_m + x * kBlockDim, stride,
                              block_m);
          DCFromLowestFrequencies(acs.Strategy(), block_m, dc_m, xs);
          TransformFromPixels(acs.Strategy(), row_s + x * kBlockDim, stride,
                              block_s);
          DCFromLowestFrequencies(acs.Strategy(), block_s, dc_s, xs);
          const float* const JXL_RESTRICT qm =
              dequant.InvMatrix(acs.Strategy(), SIDE_CHANNEL);
          // Why does a constant seem to work better than
          // raw_quant_field->Row(y)[x] ?
          float q = use_dct8 ? 1 : quantizer->Scale() * 400.0f;
          float q_dc =
              use_dct8 ? 1 : 1.0f / quantizer->GetInvDcStep(SIDE_CHANNEL);

          // Copy DCs in dc_values.
          for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
            for (size_t ix = 0; ix < xs; ix++) {
              dc_values_m[(iy + y) * xsize_blocks + ix + x] =
                  dc_m[iy * xs + ix] * q_dc;
              dc_values_s[(iy + y) * xsize_blocks + ix + x] =
                  dc_s[iy * xs + ix] * q_dc;
            }
          }

          // Copy AC coefficients in the local block. The order in which
          // coefficients get stored does not matter.
          size_t cx = acs.covered_blocks_x();
          size_t cy = acs.covered_blocks_y();
          CoefficientLayout(&cx, &cy);
          for (size_t iy = 0; iy < cy * kBlockDim; iy++) {
            for (size_t ix = 0; ix < cx * kBlockDim; ix++) {
              if (iy < cy && ix < cx) {
                continue;
              }
              JXL_ASSERT(cx * kBlockDim * iy + ix <
                         acs.covered_blocks_y() * xs * 64);
              coeffs_m[num_ac] = block_m[cx * kBlockDim * iy + ix] * q *
                                 qm[cx * kBlockDim * iy + ix];
              coeffs_s[num_ac] = block_s[cx * kBlockDim * iy + ix] * q *
                                 qm[cx * kBlockDim * iy + ix];
              num_ac++;
            }
          }
        }
      }
      // Pad with zeros.
      while (num_ac % Lanes(df) != 0) {
        coeffs_m[num_ac] = 0;
        coeffs_s[num_ac] = 0;
        num_ac++;
      }
      row_out[tx] = FindBestMultiplier(coeffs_m, coeffs_s, num_ac, OFFSET, base,
                                       kDistanceMultiplierAC);
    }
  };

  RunOnPool(pool, 0, map->ysize(), ThreadPool::SkipInit(), process_row,
            "FindCorrelation");

  *dc = FindBestMultiplier(dc_values_m, dc_values_s, dc_values.xsize(), OFFSET,
                           base, kDistanceMultiplierDC);
}

void FindBestColorCorrelationMap(const Image3F& opsin,
                                 const DequantMatrices& dequant,
                                 const AcStrategyImage* ac_strategy,
                                 const ImageI* raw_quant_field,
                                 const Quantizer* quantizer, ThreadPool* pool,
                                 ColorCorrelationMap* cmap) {
  PROFILER_ZONE("enc YTo* correlation");

  int32_t ytob_dc = kColorOffset;
  int32_t ytox_dc = kColorOffset;

  if (ac_strategy == nullptr) {
    JXL_ASSERT(raw_quant_field == nullptr);
    JXL_ASSERT(quantizer == nullptr);
    FindBestCorrelation</* from Y */ 1, /* to B */ 2, /*use_dct8=*/true,
                        kDefaultColorFactor, kColorOffset>(
        opsin, &cmap->ytob_map, &ytob_dc, cmap->YtoBRatio(kColorOffset),
        dequant, ac_strategy, raw_quant_field, quantizer, pool);
    FindBestCorrelation</* from Y */ 1, /* to X */ 0, /*use_dct8=*/true,
                        kDefaultColorFactor, kColorOffset>(
        opsin, &cmap->ytox_map, &ytox_dc, cmap->YtoXRatio(kColorOffset),
        dequant, ac_strategy, raw_quant_field, quantizer, pool);
  } else {
    JXL_ASSERT(raw_quant_field != nullptr);
    JXL_ASSERT(quantizer != nullptr);
    FindBestCorrelation</* from Y */ 1, /* to B */ 2, /*use_dct8=*/false,
                        kDefaultColorFactor, kColorOffset>(
        opsin, &cmap->ytob_map, &ytob_dc, cmap->YtoBRatio(kColorOffset),
        dequant, ac_strategy, raw_quant_field, quantizer, pool);
    FindBestCorrelation</* from Y */ 1, /* to X */ 0, /*use_dct8=*/false,
                        kDefaultColorFactor, kColorOffset>(
        opsin, &cmap->ytox_map, &ytox_dc, cmap->YtoXRatio(kColorOffset),
        dequant, ac_strategy, raw_quant_field, quantizer, pool);
  }
  cmap->SetYToBDC(ytob_dc);
  cmap->SetYToXDC(ytox_dc);
}

class ColorCorrelationMapCoder {
 public:
  // TODO(veluca): change predictors?
  template <typename T>
  using Predictor = ComputeResiduals<T, PredictorPackSignedRange<0, 255>,
                                     Predictors2<YPredictor, YPredictor>>;

  enum {
    kNumResidualContexts = 8,
    kContextsPerChannel = kNumResidualContexts + 4,
    kNumContexts = 2 * kContextsPerChannel,
  };

  static int Context(size_t c, size_t correct, size_t badness) {
    if (correct == 0) {
      JXL_ASSERT(badness != 0);
      badness = (badness + 1) >> 1;
      size_t badness_offset =
          std::min<size_t>(badness, kNumResidualContexts) - 1;
      return kContextsPerChannel * c + badness_offset;
    }
    return kContextsPerChannel * c + kNumResidualContexts +
           CeilLog2Nonzero(9 - correct);
  }
  static_assert(kNumContexts == kCmapContexts,
                "Invalid number of cmap contexts");

  struct Decoder : public Predictor<Decoder> {
    BitReader* JXL_RESTRICT br;
    ANSSymbolReader* decoder;
    const std::vector<uint8_t>* context_map;
    uint8_t* JXL_RESTRICT rows[2];
    size_t stride;
    size_t base_context;
    AuxOut* JXL_RESTRICT aux_out;

    Decoder(BitReader* JXL_RESTRICT br, ANSSymbolReader* decoder,
            const std::vector<uint8_t>* context_map,
            uint8_t* JXL_RESTRICT rows[2], size_t stride, size_t base_context,
            AuxOut* JXL_RESTRICT aux_out)
        : br(br),
          decoder(decoder),
          context_map(context_map),
          rows{rows[0], rows[1]},
          stride(stride),
          base_context(base_context),
          aux_out(aux_out) {}

    void Decode(size_t xsize, size_t ysize) {
      Predictor<Decoder>::Run(xsize, ysize, aux_out);
    }

    JXL_INLINE void Prediction(size_t x, size_t y,
                               const int32_t* JXL_RESTRICT predictions,
                               const uint32_t* JXL_RESTRICT num_correct,
                               const uint32_t* JXL_RESTRICT min_error,
                               int32_t* JXL_RESTRICT decoded) {
      for (size_t c = 0; c < 2; c++) {
        size_t ctx = base_context + Context(c, num_correct[c], min_error[c]);

        uint32_t residual = decoder->ReadHybridUint(ctx, br, *context_map);

        if (x == 0 && y == 0) {
          decoded[c] = kColorOffset + UnpackSigned(residual);
        } else {
          decoded[c] = predictions[c] + UnpackSigned(residual);
        }

        rows[c][y * stride + x] = decoded[c];
      }
    }
  };

  struct Encoder : public Predictor<Encoder> {
    std::vector<Token>* JXL_RESTRICT tokens;
    const uint8_t* JXL_RESTRICT rows[2];
    size_t stride;
    size_t base_context;
    AuxOut* JXL_RESTRICT aux_out;

    Encoder(std::vector<Token>* JXL_RESTRICT tokens,
            const uint8_t* JXL_RESTRICT rows[2], size_t stride,
            size_t base_context, AuxOut* JXL_RESTRICT aux_out)
        : tokens(tokens),
          rows{rows[0], rows[1]},
          stride(stride),
          base_context(base_context),
          aux_out(aux_out) {}

    void Encode(size_t xsize, size_t ysize) {
      Predictor<Encoder>::Run(xsize, ysize, aux_out);
    }

    JXL_INLINE void Prediction(size_t x, size_t y,
                               const int32_t* JXL_RESTRICT predictions,
                               const uint32_t* JXL_RESTRICT num_correct,
                               const uint32_t* JXL_RESTRICT min_error,
                               int32_t* JXL_RESTRICT decoded) {
      for (size_t c = 0; c < 2; c++) {
        size_t ctx = base_context + Context(c, num_correct[c], min_error[c]);

        decoded[c] = rows[c][y * stride + x];

        uint32_t residual;
        if (x == 0 && y == 0) {
          residual = PackSigned(decoded[c] - kColorOffset);
        } else {
          residual = PackSigned(decoded[c] - predictions[c]);
        }

        TokenizeHybridUint(ctx, residual, tokens);
      }
    }
  };
};

Status DecodeColorMap(BitReader* JXL_RESTRICT br, ANSSymbolReader* decoder,
                      const std::vector<uint8_t>& context_map,
                      ColorCorrelationMap* cmap, const Rect& rect,
                      size_t base_context, AuxOut* JXL_RESTRICT aux_out) {
  uint8_t* JXL_RESTRICT rows[2] = {
      rect.Row(&cmap->ytox_map, 0),
      rect.Row(&cmap->ytob_map, 0),
  };
  const size_t stride = cmap->ytob_map.PixelsPerRow();

  ColorCorrelationMapCoder::Decoder(br, decoder, &context_map, rows, stride,
                                    base_context, aux_out)
      .Decode(rect.xsize(), rect.ysize());
  return true;
}

void EncodeColorMap(const ColorCorrelationMap& cmap, const Rect& rect,
                    std::vector<Token>* tokens, size_t base_context,
                    AuxOut* JXL_RESTRICT aux_out) {
  const uint8_t* JXL_RESTRICT rows[2] = {
      rect.ConstRow(cmap.ytox_map, 0),
      rect.ConstRow(cmap.ytob_map, 0),
  };
  const size_t stride = cmap.ytob_map.PixelsPerRow();
  ColorCorrelationMapCoder::Encoder(tokens, rows, stride, base_context, aux_out)
      .Encode(rect.xsize(), rect.ysize());
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(FindBestColorCorrelationMap)
void FindBestColorCorrelationMap(const Image3F& opsin,
                                 const DequantMatrices& dequant,
                                 const AcStrategyImage* ac_strategy,
                                 const ImageI* raw_quant_field,
                                 const Quantizer* quantizer, ThreadPool* pool,
                                 ColorCorrelationMap* cmap) {
  return HWY_DYNAMIC_DISPATCH(FindBestColorCorrelationMap)(
      opsin, dequant, ac_strategy, raw_quant_field, quantizer, pool, cmap);
}

HWY_EXPORT(EncodeColorMap)
void EncodeColorMap(const ColorCorrelationMap& cmap, const Rect& rect,
                    std::vector<Token>* tokens, size_t base_context,
                    AuxOut* JXL_RESTRICT aux_out) {
  return HWY_DYNAMIC_DISPATCH(EncodeColorMap)(cmap, rect, tokens, base_context,
                                              aux_out);
}

HWY_EXPORT(DecodeColorMap)
Status DecodeColorMap(BitReader* JXL_RESTRICT br, ANSSymbolReader* decoder,
                      const std::vector<uint8_t>& context_map,
                      ColorCorrelationMap* cmap, const Rect& rect,
                      size_t base_context, AuxOut* JXL_RESTRICT aux_out) {
  return HWY_DYNAMIC_DISPATCH(DecodeColorMap)(br, decoder, context_map, cmap,
                                              rect, base_context, aux_out);
}

ColorCorrelationMap::ColorCorrelationMap(size_t xsize, size_t ysize, bool XYB)
    : ytox_map(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim)),
      ytob_map(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim)) {
  FillImage(kColorOffset, &ytox_map);
  FillImage(kColorOffset, &ytob_map);
  if (!XYB) {
    base_correlation_b_ = 0;
  }
  RecomputeDCFactors();
}

}  // namespace jxl
#endif  // HWY_ONCE
