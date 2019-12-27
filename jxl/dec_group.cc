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

#include "jxl/dec_group.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <hwy/static_targets.h>
#include <utility>

#include "jxl/ac_context.h"
#include "jxl/ac_strategy.h"
#include "jxl/aux_out.h"
#include "jxl/base/bits.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/coeff_order.h"
#include "jxl/common.h"
#include "jxl/convolve.h"
#include "jxl/dec_cache.h"
#include "jxl/dec_reconstruct.h"
#include "jxl/dec_xyb.h"
#include "jxl/entropy_coder.h"
#include "jxl/epf.h"
#include "jxl/opsin_params.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"

namespace jxl {

namespace {
using D = HWY_FULL(float);
using DU = HWY_FULL(uint32_t);
constexpr D d;

template <class V>
HWY_ATTR void DequantLane(V scaled_dequant, V x_dm_multiplier,
                          const float* JXL_RESTRICT dequant_matrices,
                          size_t dq_ofs, size_t size, size_t k, V x_cc_mul,
                          V b_cc_mul, const float* JXL_RESTRICT biases,
                          float* JXL_RESTRICT block) {
  const auto x_mul =
      Load(d, dequant_matrices + dq_ofs + k) * scaled_dequant * x_dm_multiplier;
  const auto y_mul =
      Load(d, dequant_matrices + dq_ofs + size + k) * scaled_dequant;
  const auto b_mul =
      Load(d, dequant_matrices + dq_ofs + 2 * size + k) * scaled_dequant;

  const auto quantized_x = Load(d, block + k);
  const auto quantized_y = Load(d, block + size + k);
  const auto quantized_b = Load(d, block + 2 * size + k);

  const auto dequant_x_cc = AdjustQuantBias(d, 0, quantized_x, biases) * x_mul;
  const auto dequant_y = AdjustQuantBias(d, 1, quantized_y, biases) * y_mul;
  const auto dequant_b_cc = AdjustQuantBias(d, 2, quantized_b, biases) * b_mul;

  const auto dequant_x = MulAdd(x_cc_mul, dequant_y, dequant_x_cc);
  const auto dequant_b = MulAdd(b_cc_mul, dequant_y, dequant_b_cc);
  Store(dequant_x, d, block + k);
  Store(dequant_y, d, block + size + k);
  Store(dequant_b, d, block + 2 * size + k);
}

template <class V>
HWY_ATTR void DequantBlock(const AcStrategy& acs, float inv_global_scale,
                           int quant, float x_dm_multiplier, V x_cc_mul,
                           V b_cc_mul, size_t kind, size_t size,
                           const Quantizer& quantizer,
                           const float* JXL_RESTRICT dequant_matrices,
                           size_t covered_blocks, size_t bx,
                           const float* JXL_RESTRICT* JXL_RESTRICT dc_row,
                           size_t dc_stride, const float* JXL_RESTRICT biases,
                           float* JXL_RESTRICT block) {
  PROFILER_FUNC;

  const auto scaled_dequant = Set(d, inv_global_scale / quant);
  const auto x_dm_multiplier_v = Set(d, x_dm_multiplier);

  const size_t dq_ofs = quantizer.DequantMatrixOffset(kind, 0);

  for (size_t k = 0; k < covered_blocks * kDCTBlockSize; k += D::N) {
    DequantLane(scaled_dequant, x_dm_multiplier_v, dequant_matrices, dq_ofs,
                size, k, x_cc_mul, b_cc_mul, biases, block);
  }
  for (size_t c = 0; c < 3; c++) {
    acs.LowestFrequenciesFromDC(dc_row[c] + bx, dc_stride, block + c * size);
  }
}

template <class GetBlock>
HWY_ATTR Status DecodeGroupImpl(GetBlock* JXL_RESTRICT get_block,
                                PassesDecoderState* JXL_RESTRICT dec_state,
                                size_t thread, const Rect& block_rect,
                                bool save_decompressed,
                                bool apply_color_transform, AuxOut* aux_out,
                                Image3F* JXL_RESTRICT idct) {
  PROFILER_FUNC;

  const LoopFilter& lf = dec_state->shared->image_features.loop_filter;
  const AcStrategyImage& ac_strategy = dec_state->shared->ac_strategy;

  const size_t xsize_blocks = block_rect.xsize();
  const size_t ysize_blocks = block_rect.ysize();

  const size_t dc_stride = dec_state->shared->dc->PixelsPerRow();

  const float inv_global_scale = dec_state->shared->quantizer.InvGlobalScale();
  const float* JXL_RESTRICT dequant_matrices =
      dec_state->shared->quantizer.DequantMatrix(0, 0);

  const size_t sigma_stride = dec_state->sigma.PixelsPerRow();
  const size_t sharpness_stride =
      dec_state->shared->epf_sharpness.PixelsPerRow();

  bool save_to_decoded = !dec_state->keep_dct && (lf.epf || lf.gab);
  size_t padding = save_to_decoded ? 2 : 0;

  const size_t idct_stride =
      (!save_to_decoded ? *idct : dec_state->decoded).PixelsPerRow();

  const float quant_scale = dec_state->shared->quantizer.Scale();

  for (size_t by = 0; by < ysize_blocks; ++by) {
    get_block->StartRow(by);

    const int32_t* JXL_RESTRICT row_quant =
        block_rect.ConstRow(dec_state->shared->raw_quant_field, by);

    const float* JXL_RESTRICT dc_rows[3] = {
        block_rect.ConstPlaneRow(*dec_state->shared->dc, 0, by),
        block_rect.ConstPlaneRow(*dec_state->shared->dc, 1, by),
        block_rect.ConstPlaneRow(*dec_state->shared->dc, 2, by),
    };

    const size_t ty = (block_rect.y0() + by) / kColorTileDimInBlocks;
    AcStrategyRow acs_row = ac_strategy.ConstRow(block_rect, by);

    const uint8_t* JXL_RESTRICT row_cmap[3] = {
        dec_state->shared->cmap.ytox_map.ConstRow(ty),
        nullptr,
        dec_state->shared->cmap.ytob_map.ConstRow(ty),
    };

    float* JXL_RESTRICT sigma_row =
        lf.epf ? block_rect.Row(&dec_state->sigma, by) : nullptr;
    const uint8_t* JXL_RESTRICT sharpness_row =
        block_rect.ConstRow(dec_state->shared->epf_sharpness, by);
    float* JXL_RESTRICT idct_row[3];
    for (size_t c = 0; c < 3; c++) {
      idct_row[c] = (!save_to_decoded ? *idct : dec_state->decoded)
                        .PlaneRow(c, (block_rect.y0() + by) * kBlockDim) +
                    (block_rect.x0()) * kBlockDim;
    }

    for (size_t tx = 0; tx < DivCeil(xsize_blocks, kColorTileDimInBlocks);
         tx++) {
      size_t abs_tx = tx + block_rect.x0() / kColorTileDimInBlocks;
      auto x_cc_mul =
          Set(d, dec_state->shared->cmap.YtoXRatio(row_cmap[0][abs_tx]));
      auto b_cc_mul =
          Set(d, dec_state->shared->cmap.YtoBRatio(row_cmap[2][abs_tx]));
      // Increment bx by llf_x because those iterations would otherwise
      // immediately continue (!IsFirstBlock). Reduces mispredictions.
      for (size_t bx = tx * kColorTileDimInBlocks;
           bx < xsize_blocks && bx < (tx + 1) * kColorTileDimInBlocks;) {
        AcStrategy acs = acs_row[bx];
        const size_t llf_x = acs.covered_blocks_x();

        // Can only happen in the second or lower rows of a varblock.
        if (JXL_UNLIKELY(!acs.IsFirstBlock())) {
          bx += llf_x;
          continue;
        }
        PROFILER_ZONE("DecodeGroupImpl inner");
        const size_t log2_covered_blocks = acs.log2_covered_blocks();

        const size_t covered_blocks = 1 << log2_covered_blocks;
        const size_t size = covered_blocks * kDCTBlockSize;

        HWY_ALIGN float block[3 * AcStrategy::kMaxCoeffArea];
        JXL_RETURN_IF_ERROR(
            get_block->GetBlock(bx, by, acs, size, log2_covered_blocks, block));

        if (JXL_UNLIKELY(dec_state->keep_dct)) {
          if (acs.Strategy() != AcStrategy::Type::DCT)
            return JXL_FAILURE(
                "Can only decode to JPEG if only DCT-8 is used.");
          const std::vector<QuantEncoding>& qe =
              dec_state->shared->matrices.encodings();
          if (qe.size() == 0 ||
              qe[0].mode != QuantEncoding::Mode::kQuantModeRAW ||
              qe[0].qraw.qtable_den_shift != 0)
            return JXL_FAILURE(
                "Quantization table is not a JPEG quantization table.");

          for (size_t c : {1, 0, 2}) {
            float* JXL_RESTRICT idct_pos = idct_row[c] + bx * kBlockDim;
            idct_pos[0] = dc_rows[c][bx];
            if (c == 1) {
              for (int i = 1; i < 64; i++) {
                idct_pos[(i % 8) * idct_stride + (i / 8)] = block[c * size + i];
              }
            } else {
              float scale =
                  (c == 0
                       ? dec_state->shared->cmap.YtoXRatio(row_cmap[c][abs_tx])
                       : dec_state->shared->cmap.YtoBRatio(
                             row_cmap[c][abs_tx]));
              for (int i = 1; i < 64; i++) {
                size_t x = i % 8;
                size_t y = i / 8;
                // JPEG XL is transposed, JPEG is not.
                idct_pos[x * idct_stride + y] =
                    block[c * size + i] +
                    (int)(scale * block[size + i] *
                          (*qe[0].qraw.qtable)[64 + x * 8 + y] /
                          (*qe[0].qraw.qtable)[c * 64 + x * 8 + y]);
              }
            }
          }
          bx += llf_x;
          continue;
        }

        // Dequantize and add predictions.
        {
          DequantBlock(
              acs, inv_global_scale, row_quant[bx], dec_state->x_dm_multiplier,
              x_cc_mul, b_cc_mul, acs.RawStrategy(), size,
              dec_state->shared->quantizer, dequant_matrices,
              acs.covered_blocks_y() * acs.covered_blocks_x(), bx, dc_rows,
              dc_stride, dec_state->shared->opsin_params.quant_biases, block);
        }

        for (size_t c : {1, 0, 2}) {
          // IDCT
          float* JXL_RESTRICT idct_pos =
              idct_row[c] + (padding + bx + padding * idct_stride) * kBlockDim;
          acs.TransformToPixels(block + c * size, idct_pos, idct_stride);
        }

        if (dec_state->shared->image_features.loop_filter.epf) {
          size_t sbx = block_rect.x0() + bx;
          size_t sby = block_rect.y0() + by;
          size_t xbl = dec_state->shared->frame_dim.xsize_blocks;
          size_t ybl = dec_state->shared->frame_dim.ysize_blocks;
          float quant = 1.0f / (row_quant[bx] * quant_scale);
          float sigma_quant = quant * lf.epf_quant_mul;
          for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
            for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
              // Increase sigma near edges.
              float dc_range = 0;
              for (size_t c = 0; c < 3; c++) {
                const float* JXL_RESTRICT base_dc_ptr =
                    dc_rows[c] + bx + ix + iy * dc_stride;
                // UBSAN complains about overflowing unsigned addition here,
                // hence we use a slightly more convoluted syntax than simple
                // array access to ensure we only ever add or subtract positive
                // quantities.
                float dc_ref = *base_dc_ptr;
                float dc_top = *(base_dc_ptr - (sby + iy == 0 ? 0 : dc_stride));
                float dc_bottom =
                    base_dc_ptr[sby + iy + 1 == ybl ? 0 : dc_stride];
                float dc_left = *(base_dc_ptr - (sbx + ix == 0 ? 0 : 1));
                float dc_right = base_dc_ptr[sbx + ix + 1 == xbl ? 0 : 1];

                float dc_range_c = std::abs(dc_top - dc_ref);
                dc_range_c = std::max(dc_range_c, std::abs(dc_bottom - dc_ref));
                dc_range_c = std::max(dc_range_c, std::abs(dc_left - dc_ref));
                dc_range_c = std::max(dc_range_c, std::abs(dc_right - dc_ref));
                dc_range =
                    std::max(dc_range_c * lf.epf_channel_scale[c], dc_range);
              }
              float sigma =
                  sigma_quant *
                  (2.0f - 1.0f / (1.0f + lf.epf_dc_range_mul * dc_range));
              sigma *= lf.epf_sharp_lut[sharpness_row[bx + ix +
                                                      iy * sharpness_stride]];
              // Avoid infinities.
              sigma = std::max(1e-4f, sigma);
              sigma_row[bx + ix + 2 + (iy + 2) * sigma_stride] =
                  kInvSigmaNum / sigma;
            }
          }
          // Left padding with mirroring.
          if (bx + block_rect.x0() == 0) {
            for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
              sigma_row[1 + (iy + 2) * sigma_stride] =
                  sigma_row[2 + (iy + 2) * sigma_stride];
            }
          }
          // Right padding with mirroring.
          if (bx + block_rect.x0() + llf_x ==
              dec_state->shared->frame_dim.xsize_blocks) {
            for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
              sigma_row[(iy + 2) * sigma_stride + bx + llf_x + 2] =
                  sigma_row[(iy + 2) * sigma_stride + bx + llf_x + 1];
            }
          }
          // Offsets for row copying, in blocks.
          size_t offset_before = bx + block_rect.x0() == 0 ? 1 : bx + 2;
          size_t offset_after =
              bx + block_rect.x0() + llf_x ==
                      dec_state->shared->frame_dim.xsize_blocks
                  ? llf_x + bx + 3
                  : llf_x + bx + 2;
          size_t num = offset_after - offset_before;
          // Above
          if (by + block_rect.y0() == 0) {
            memcpy(sigma_row + offset_before + sigma_stride,
                   sigma_row + offset_before + 2 * sigma_stride,
                   num * sizeof(*sigma_row));
          }
          // Below
          if (by + block_rect.y0() + acs.covered_blocks_y() ==
              dec_state->shared->frame_dim.ysize_blocks) {
            memcpy(sigma_row + offset_before +
                       sigma_stride * (acs.covered_blocks_y() + 2),
                   sigma_row + offset_before +
                       sigma_stride * (acs.covered_blocks_y() + 1),
                   num * sizeof(*sigma_row));
          }
        }

        if (save_to_decoded) {
          for (size_t c : {1, 0, 2}) {
            // Left padding with mirroring.
            if (bx + block_rect.x0() == 0) {
              for (size_t iy = 0; iy < kBlockDim * acs.covered_blocks_y();
                   iy++) {
                size_t row_offset =
                    idct_stride * (2 * kBlockDim + iy) + kBlockDim;
                for (size_t ix = 0; ix < kBlockDim; ix++) {
                  idct_row[c][kBlockDim - ix - 1 + row_offset] =
                      idct_row[c][kBlockDim + ix + row_offset];
                }
              }
            }
            // Right padding with mirroring.
            if (bx + block_rect.x0() + llf_x ==
                dec_state->shared->frame_dim.xsize_blocks) {
              for (size_t iy = 0; iy < kBlockDim * acs.covered_blocks_y();
                   iy++) {
                size_t row_offset = idct_stride * (2 * kBlockDim + iy) +
                                    (bx + llf_x + 1) * kBlockDim;
                for (size_t ix = 0; ix < kBlockDim; ix++) {
                  idct_row[c][kBlockDim + ix + row_offset] =
                      idct_row[c][kBlockDim - ix - 1 + row_offset];
                }
              }
            }
            // Offsets for row copying, in blocks.
            size_t offset_before = bx + block_rect.x0() == 0 ? 1 : bx + 2;
            size_t offset_after =
                bx + block_rect.x0() + llf_x ==
                        dec_state->shared->frame_dim.xsize_blocks
                    ? llf_x + bx + 3
                    : llf_x + bx + 2;
            size_t num = offset_after - offset_before;
            // Above
            if (by + block_rect.y0() == 0) {
              for (size_t iy = 0; iy < kBlockDim; iy++) {
                float* JXL_RESTRICT row =
                    idct_row[c] + offset_before * kBlockDim;
                memcpy(row + (2 * kBlockDim - iy - 1) * idct_stride,
                       row + (2 * kBlockDim + iy) * idct_stride,
                       num * sizeof(**idct_row) * kBlockDim);
              }
            }
            // Below
            if (by + block_rect.y0() + acs.covered_blocks_y() ==
                dec_state->shared->frame_dim.ysize_blocks) {
              for (size_t iy = 0; iy < kBlockDim; iy++) {
                float* JXL_RESTRICT row =
                    idct_row[c] +
                    (acs.covered_blocks_y() + 1) * kBlockDim * idct_stride +
                    offset_before * kBlockDim;
                memcpy(row + (kBlockDim + iy) * idct_stride,
                       row + (kBlockDim - iy - 1) * idct_stride,
                       num * sizeof(**idct_row) * kBlockDim);
              }
            }
          }
        }
        bx += llf_x;
      }
    }
  }
  // No ApplyImageFeatures in keep_dct mode.
  if (JXL_UNLIKELY(dec_state->keep_dct)) return true;

  // Apply image features to
  // - the whole AC group, if no loop filtering is enabled, or
  // - only the interior part of the group, skipping 1 block of border
  // ... unless this is the first or the last group in a row, in which case we
  // process the corresponding border too.
  size_t xsize = xsize_blocks * kBlockDim;
  size_t ysize = ysize_blocks * kBlockDim;
  size_t xstart = block_rect.x0() != 0 && (lf.epf || lf.gab) ? kBlockDim : 0;
  size_t ystart = block_rect.y0() != 0 && (lf.epf || lf.gab) ? kBlockDim : 0;
  bool is_last_block_x = dec_state->shared->frame_dim.xsize_blocks ==
                         block_rect.x0() + block_rect.xsize();
  bool is_last_block_y = dec_state->shared->frame_dim.ysize_blocks ==
                         block_rect.y0() + block_rect.ysize();
  size_t xend =
      !is_last_block_x && (lf.epf || lf.gab) ? xsize - kBlockDim : xsize;
  size_t yend =
      !is_last_block_y && (lf.epf || lf.gab) ? ysize - kBlockDim : ysize;
  if (xstart >= xend) return true;
  if (ystart >= yend) return true;

  for (size_t ycur = ystart; ycur < yend; ycur += kApplyImageFeaturesTileDim) {
    size_t cur_size_y =
        std::min(yend, ycur + kApplyImageFeaturesTileDim) - ycur;
    for (size_t xcur = xstart; xcur < xend;
         xcur += kApplyImageFeaturesTileDim) {
      size_t cur_size_x =
          std::min(xend, xcur + kApplyImageFeaturesTileDim) - xcur;
      Rect rect(block_rect.x0() * kBlockDim + xcur,
                block_rect.y0() * kBlockDim + ycur, cur_size_x, cur_size_y);
      ApplyImageFeatures(idct, rect, dec_state, thread, aux_out,
                         save_decompressed, apply_color_transform);
    }
  }

  return true;
}

// Decode quantized AC coefficients of DCT blocks.
// LLF components in the output block will not be modified.
HWY_ATTR Status DecodeACVarBlock(
    size_t log2_covered_blocks, int32_t* JXL_RESTRICT row_nzeros,
    const int32_t* JXL_RESTRICT row_nzeros_top, size_t nzeros_stride, size_t c,
    size_t bx, size_t by, AcStrategy acs,
    const coeff_order_t* JXL_RESTRICT coeff_order, BitReader* JXL_RESTRICT br,
    ANSSymbolReader* JXL_RESTRICT decoder,
    const std::vector<uint8_t>& context_map, ac_qcoeff_t* JXL_RESTRICT block,
    size_t shift = 0) {
  PROFILER_FUNC;
  size_t c_ctx = c == 1 ? 0 : 1;
  // Equal to number of LLF coefficients.
  const size_t covered_blocks = 1 << log2_covered_blocks;
  const size_t size = covered_blocks * kDCTBlockSize;
  int32_t predicted_nzeros =
      PredictFromTopAndLeft(row_nzeros_top, row_nzeros, bx, 32);
  size_t nzeros = 0;

  size_t ord = kStrategyOrder[acs.RawStrategy()];
  const coeff_order_t* JXL_RESTRICT order =
      &coeff_order[(ord * 3 + c) * AcStrategy::kMaxCoeffArea];
  ord = ord > 2 ? ord / 2 + 1 : ord;
  // ord is in [0, 5), so we multiply c_ctx by 5.
  size_t block_ctx_id = c_ctx * 5 + ord;

  const size_t nzero_ctx = NonZeroContext(predicted_nzeros, block_ctx_id);
  nzeros = ReadHybridUint(nzero_ctx, br, decoder, context_map);
  if (nzeros > size) {
    return JXL_FAILURE("Invalid AC: nzeros too large");
  }
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
      row_nzeros[bx + x + y * nzeros_stride] =
          (nzeros + covered_blocks - 1) >> log2_covered_blocks;
    }
  }

  const size_t histo_offset = ZeroDensityContextsOffset(block_ctx_id);

  // Skip LLF
  {
    PROFILER_ZONE("AcDecSkipLLF, reader");
    size_t prev = (nzeros > size / 16 ? 0 : 1);
    for (size_t k = covered_blocks; k < size && nzeros != 0; ++k) {
      const size_t ctx =
          histo_offset + ZeroDensityContext(nzeros, k, covered_blocks,
                                            log2_covered_blocks, prev);
      const size_t u_coeff = ReadHybridUint(ctx, br, decoder, context_map);
      // Hand-rolled version of UnpackSigned, shifting before the conversion to
      // signed integer to avoid undefined behavior of shifting negative
      // numbers.
      const size_t magnitude = u_coeff >> 1;
      const size_t neg_sign = (~u_coeff) & 1;
      const intptr_t coeff =
          static_cast<intptr_t>((magnitude ^ (neg_sign - 1)) << shift);
      block[order[k]] += static_cast<ac_qcoeff_t>(coeff);
      prev = u_coeff != 0;
      nzeros -= prev;
    }
    if (JXL_UNLIKELY(nzeros != 0)) {
      return JXL_FAILURE("Invalid AC: nzeros not 0.");
    }
  }
  return true;
}

// Structs used by DecodeGroupImpl to get a quantized block.
// GetBlockFromBitstream uses ANS decoding (and thus keeps track of row
// pointers in row_nzeros), GetBlockFromEncoder simply reads the coefficient
// image provided by the encoder.

struct GetBlockFromBitstream {
  void StartRow(size_t by) {
    for (size_t i = 0; i < num_passes; i++) {
      for (size_t c = 0; c < 3; c++) {
        row_nzeros[i][c] = group_dec_cache->num_nzeroes[i].PlaneRow(c, by);
        row_nzeros_top[i][c] =
            by == 0 ? nullptr
                    : group_dec_cache->num_nzeroes[i].ConstPlaneRow(c, by - 1);
      }
    }
  }

  Status GetBlock(size_t bx, size_t by, const AcStrategy& acs, size_t size,
                  size_t log2_covered_blocks, float* JXL_RESTRICT block) {
    memset(block, 0, sizeof(float) * size * 3);
    for (size_t c : {1, 0, 2}) {
      float* JXL_RESTRICT block_c = block + c * size;

      for (size_t pass = 0; JXL_UNLIKELY(pass < num_passes); pass++) {
        JXL_RETURN_IF_ERROR(DecodeACVarBlock(
            log2_covered_blocks, row_nzeros[pass][c], row_nzeros_top[pass][c],
            nzeros_stride, c, bx, by, acs,
            &coeff_orders[idx[pass] * kCoeffOrderSize], readers[pass],
            &decoders[pass], context_map[idx[pass]], block_c,
            shift_for_pass[pass]));
      }
    }
    return true;
  }

  HWY_ATTR Status Init(BitReader* JXL_RESTRICT* JXL_RESTRICT readers,
                       size_t num_passes, size_t group_idx,
                       size_t histo_selector_bits,
                       GroupDecCache* JXL_RESTRICT group_dec_cache,
                       PassesDecoderState* dec_state) {
    this->coeff_orders = dec_state->shared->coeff_orders.data();
    this->context_map = dec_state->context_map.data();
    this->readers = readers;
    this->num_passes = num_passes;
    this->shift_for_pass = dec_state->shared->frame_header.passes.shift;
    this->group_dec_cache = group_dec_cache;
    for (size_t pass = 0; pass < num_passes; pass++) {
      // Select which histogram to use among those of the current pass.
      size_t cur_histogram = 0;
      if (histo_selector_bits != 0) {
        cur_histogram = readers[pass]->ReadBits(histo_selector_bits);
      }
      if (cur_histogram >= dec_state->shared->num_histograms) {
        return JXL_FAILURE("Invalid histogram selector");
      }
      // GetNumPasses() is *not* the same as num_passes!
      idx[pass] =
          cur_histogram * dec_state->shared->frame_header.passes.num_passes +
          pass;

      decoders[pass] =
          ANSSymbolReader(&dec_state->code[idx[pass]], readers[pass]);
    }
    nzeros_stride = group_dec_cache->num_nzeroes[0].PixelsPerRow();
    for (size_t i = 0; i < num_passes; i++) {
      JXL_ASSERT(
          nzeros_stride ==
          static_cast<size_t>(group_dec_cache->num_nzeroes[i].PixelsPerRow()));
    }
    return true;
  }

  const uint32_t* shift_for_pass = nullptr;  // not owned
  size_t idx[kMaxNumPasses];
  const coeff_order_t* JXL_RESTRICT coeff_orders;
  const std::vector<uint8_t>* JXL_RESTRICT context_map;
  ANSSymbolReader decoders[kMaxNumPasses];
  BitReader* JXL_RESTRICT* JXL_RESTRICT readers;
  size_t num_passes;
  size_t nzeros_stride;
  int32_t* JXL_RESTRICT row_nzeros[kMaxNumPasses][3];
  const int32_t* JXL_RESTRICT row_nzeros_top[kMaxNumPasses][3];
  GroupDecCache* JXL_RESTRICT group_dec_cache;
};

struct GetBlockFromEncoder {
  void StartRow(size_t by) {}

  Status GetBlock(size_t bx, size_t by, const AcStrategy& acs, size_t size,
                  size_t log2_covered_blocks, float* JXL_RESTRICT block) {
    memset(block, 0, size * 3 * sizeof(float));
    for (size_t c = 0; c < 3; c++) {
      // for each pass
      for (size_t i = 0; i < quantized_ac->size(); i++) {
        for (size_t k = 0; k < size; k++) {
          // TODO(veluca): SIMD.
          block[c * size + k] += rows[i][c][offset + k];
        }
      }
    }
    offset += size;
    return true;
  }

  GetBlockFromEncoder(const std::vector<ACImage3>& ac, size_t group_idx)
      : quantized_ac(&ac) {
    for (size_t i = 0; i < quantized_ac->size(); i++) {
      for (size_t c = 0; c < 3; c++) {
        rows[i][c] = (*quantized_ac)[i].ConstPlaneRow(c, group_idx);
      }
    }
  }

  const std::vector<ACImage3>* JXL_RESTRICT quantized_ac;
  size_t offset = 0;
  const float* JXL_RESTRICT rows[kMaxNumPasses][3];
};

}  // namespace

Status DecodeGroup(BitReader* JXL_RESTRICT* JXL_RESTRICT readers,
                   size_t num_passes, size_t group_idx,
                   PassesDecoderState* JXL_RESTRICT dec_state,
                   GroupDecCache* JXL_RESTRICT group_dec_cache, size_t thread,
                   Image3F* opsin, AuxOut* aux_out) {
  PROFILER_FUNC;
  size_t histo_selector_bits =
      dec_state->shared->num_histograms == 1
          ? 0
          : CeilLog2Nonzero(dec_state->shared->num_histograms - 1);

  const Rect block_group_rect = dec_state->shared->BlockGroupRect(group_idx);

  group_dec_cache->InitOnce(num_passes);

  GetBlockFromBitstream get_block;
  JXL_RETURN_IF_ERROR(get_block.Init(readers, num_passes, group_idx,
                                     histo_selector_bits, group_dec_cache,
                                     dec_state));

  JXL_RETURN_IF_ERROR(
      DecodeGroupImpl(&get_block, dec_state, thread, block_group_rect,
                      /*save_decompressed=*/true,
                      /*apply_color_transform=*/true, aux_out, opsin));

  for (size_t pass = 0; pass < num_passes; pass++) {
    if (!get_block.decoders[pass].CheckANSFinalState()) {
      return JXL_FAILURE("ANS checksum failure.");
    }
  }
  return true;
}

Status DecodeGroupForRoundtrip(const std::vector<ACImage3>& ac,
                               size_t group_idx,
                               PassesDecoderState* JXL_RESTRICT dec_state,
                               size_t thread, Image3F* JXL_RESTRICT opsin,
                               AuxOut* aux_out, bool save_decompressed,
                               bool apply_color_transform) {
  PROFILER_FUNC;

  const Rect block_group_rect = dec_state->shared->BlockGroupRect(group_idx);

  GetBlockFromEncoder get_block(ac, group_idx);

  return DecodeGroupImpl(&get_block, dec_state, thread, block_group_rect,
                         save_decompressed, apply_color_transform, aux_out,
                         opsin);
}
}  // namespace jxl
