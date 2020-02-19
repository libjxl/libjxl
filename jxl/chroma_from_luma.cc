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

#include <float.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <hwy/static_targets.h>

#include "jxl/aux_out.h"
#include "jxl/base/bits.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/dct_util.h"
#include "jxl/entropy_coder.h"
#include "jxl/image_ops.h"
#include "jxl/modular/encoding/encoding.h"
#include "jxl/predictor.h"
#include "jxl/quantizer.h"

namespace jxl {

namespace {

template <int MAIN_CHANNEL, int SIDE_CHANNEL, int SCALE, int OFFSET>
HWY_ATTR JXL_NOINLINE void FindBestCorrelation(const Image3F& dct,
                                               ImageB* JXL_RESTRICT map,
                                               int* JXL_RESTRICT dc, float base,
                                               const DequantMatrices& dequant,
                                               ThreadPool* pool) {
  constexpr float kScale = SCALE;
  constexpr float kZeroThresh =
      kScale * kZeroBiasDefault[SIDE_CHANNEL] *
      0.9999f;  // just epsilon less for better rounding
  const float* const JXL_RESTRICT qm =
      dequant.InvMatrix(AcStrategy::Type::DCT, SIDE_CHANNEL);
  std::vector<int32_t> d_num_zeros_thread;
  auto process_row_init = [&d_num_zeros_thread](size_t num_threads) {
    d_num_zeros_thread.resize(256 * num_threads);
    return true;
  };
  auto process_row = [&](int ty, int thread) HWY_ATTR {
    uint8_t* JXL_RESTRICT row_out = map->Row(ty);
    for (size_t tx = 0; tx < map->xsize(); ++tx) {
      const size_t y0 = ty * kColorTileDimInBlocks;
      const size_t x0 = tx * kColorTileDimInBlocks * kDCTBlockSize;
      const size_t y1 =
          std::min<size_t>(y0 + kColorTileDimInBlocks, dct.ysize());
      const size_t x1 = std::min<size_t>(
          x0 + kColorTileDimInBlocks * kDCTBlockSize, dct.xsize());

      int32_t d_num_zeros[257] = {0};
      for (size_t y = y0; y < y1; ++y) {
        const float* JXL_RESTRICT row_m = dct.ConstPlaneRow(MAIN_CHANNEL, y);
        const float* JXL_RESTRICT row_s = dct.ConstPlaneRow(SIDE_CHANNEL, y);

        const HWY_FULL(float) df;
        const HWY_FULL(int32_t) di;
        const auto zero = Zero(df);
        const auto one = Set(df, 1.0f);
        const auto epsilon = Set(df, FLT_EPSILON);
        const auto abs_mask = BitCast(df, Set(di, 0x7FFFFFFF));

        for (size_t x = x0; x < x1; x += di.N) {
          const size_t k = x % kDCTBlockSize;  // index within block
          const auto quant = Load(df, qm + k);
          const auto scaled_m = Load(df, row_m + x) * quant;
          const auto scaled_s = Set(df, kScale) * Load(df, row_s + x) * quant +
                                Set(df, OFFSET - base * kScale) * scaled_m;
          // Increment num_zeros[idx] if
          //   std::abs(scaled_s - (idx - OFFSET) * scaled_m) < kZeroThresh
          const auto abs_scaled_m = scaled_m & abs_mask;
          const auto recip =
              IfThenElseZero(abs_scaled_m > epsilon, one / scaled_m);

          const auto m_sign = AndNot(abs_mask, scaled_m);
          const auto signed_thres = Set(df, kZeroThresh) | m_sign;
          const auto to =
              Min((scaled_s + signed_thres) * recip, Set(df, 255.0f));
          const auto from = Max(zero, (scaled_s - signed_thres) * recip);

          HWY_ALIGN int32_t top[di.N];
          HWY_ALIGN int32_t bot[di.N];
          HWY_ALIGN int32_t recip_lanes[df.N];
          Store(BitCast(di, recip), di, recip_lanes);
          Store(ConvertTo(di, Ceil(from)), di, top);
          Store(ConvertTo(di, Floor(to + one)), di, bot);

          for (size_t i = 0; i < di.N; ++i) {
            // Instead of clamping both values, just check that range is sane.
            // If top=bot, this is a no-op. Also avoid div by zero.
            if (recip_lanes[i] != 0 && top[i] < bot[i]) {
              d_num_zeros_thread[256 * thread + top[i]]++;
              if (bot[i] < 256) d_num_zeros_thread[256 * thread + bot[i]]--;
              if (k + i == 0) continue;  // skip DC
              d_num_zeros[top[i]]++;
              d_num_zeros[bot[i]]--;
            }
          }
        }
        int best = 0;
        int32_t best_sum = 0;
        FindIndexOfSumMaximum(d_num_zeros, 256, &best, &best_sum);
        row_out[tx] = best;
      }
    }
  };

  RunOnPool(pool, 0, map->ysize(), process_row_init, process_row,
            "FindCorrelation");

  size_t num_threads = d_num_zeros_thread.size() / 256;
  int32_t d_num_zeros_global[256] = {0};
  for (size_t t = 0; t < num_threads; t++) {
    for (size_t i = 0; i < 256; i++) {
      d_num_zeros_global[i] += d_num_zeros_thread[t * 256 + i];
    }
  }
  int global_best = 0;
  int32_t global_sum = 0;
  FindIndexOfSumMaximum(d_num_zeros_global, 256, &global_best, &global_sum);
  *dc = global_best;
}

}  // namespace

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

void FindBestColorCorrelationMap(const Image3F& opsin,
                                 const DequantMatrices& dequant,
                                 ThreadPool* pool, ColorCorrelationMap* cmap) {
  PROFILER_ZONE("enc YTo* correlation");

  const size_t xsize_blocks = opsin.xsize() / kBlockDim;
  const size_t ysize_blocks = opsin.ysize() / kBlockDim;
  Image3F dct(xsize_blocks * kDCTBlockSize, ysize_blocks);
  TransposedScaledDCT(opsin, &dct);

  int32_t ytob_dc = kColorOffset;
  int32_t ytox_dc = kColorOffset;

  FindBestCorrelation</* from Y */ 1, /* to B */ 2, kDefaultColorFactor,
                      kColorOffset>(dct, &cmap->ytob_map, &ytob_dc,
                                    cmap->YtoBRatio(kColorOffset), dequant,
                                    pool);
  FindBestCorrelation</* from Y */ 1, /* to X */ 0, kDefaultColorFactor,
                      kColorOffset>(dct, &cmap->ytox_map, &ytox_dc,
                                    cmap->YtoXRatio(kColorOffset), dequant,
                                    pool);
  cmap->SetYToBDC(ytob_dc);
  cmap->SetYToXDC(ytox_dc);
}

class ColorCorrelationMapCoder {
 public:
  // TODO(veluca): change predictors?
  template <typename T>
  using Predictor = predictor::ComputeResiduals<
      T, predictor::PackSignedRange<0, 255>,
      predictor::Predictors2<predictor::YPredictor, predictor::YPredictor>>;

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

    HWY_ATTR JXL_INLINE void Prediction(
        size_t x, size_t y, const int32_t* JXL_RESTRICT predictions,
        const uint32_t* JXL_RESTRICT num_correct,
        const uint32_t* JXL_RESTRICT min_error, int32_t* JXL_RESTRICT decoded) {
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

    HWY_ATTR JXL_INLINE void Prediction(
        size_t x, size_t y, const int32_t* JXL_RESTRICT predictions,
        const uint32_t* JXL_RESTRICT num_correct,
        const uint32_t* JXL_RESTRICT min_error, int32_t* JXL_RESTRICT decoded) {
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

bool DecodeFullColorMap(BitReader* JXL_RESTRICT br, ColorCorrelationMap* cmap,
                        bool use_new_cmap) {
  if (use_new_cmap) {
    if (!br->JumpToByteBoundary() || !br->AllReadsWithinBounds()) return false;
    const Span<const uint8_t> span = br->GetSpan();
    Rect rect0(cmap->ytob_map);
    size_t pos = 0;
    if (!modular_rect_decompress_2(span, &pos, &cmap->ytox_map, &cmap->ytob_map,
                                   rect0, kColorOffset, kColorOffset))
      return JXL_FAILURE("Failed to decode color map");
    br->SkipBits(8 * pos);
  } else {
    // no longer exists
  }
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

void EncodeFullColorMap(const ColorCorrelationMap& cmap, const Rect& rect,
                        BitWriter* writer, size_t layer, AuxOut* aux_out,
                        bool use_new_cmap) {
  if (use_new_cmap) {
    PaddedBytes enc_co;
    modular_options cfopts;
    set_default_modular_options(cfopts);
    cfopts.nb_repeats = 1;
    modular_rect_compress_2(cmap.ytox_map, cmap.ytob_map, rect, &enc_co,
                            &cfopts, kColorOffset, kColorOffset);

    if (aux_out != nullptr)
      aux_out->layers[layer].total_bits += enc_co.size() * 8;

    writer->ZeroPadToByte();
    *writer += enc_co;

  } else {
    // no longer exists
  }
}

}  // namespace jxl
