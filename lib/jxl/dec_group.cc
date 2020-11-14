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

#include "lib/jxl/dec_group.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <utility>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/dec_group.cc"
#include <hwy/foreach_target.h>
// ^ must come before highway.h and any *-inl.h.

#include <hwy/highway.h>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/common.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/dct_scales.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_reconstruct.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/opsin_params.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer-inl.h"
#include "lib/jxl/quantizer.h"

#ifndef LIB_JXL_DEC_GRUOP_CC_
#define LIB_JXL_DEC_GRUOP_CC_

namespace {

// Mirror n floats starting at *p and store them before p.
JXL_INLINE void LeftMirror(float* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    *(p - 1 - i) = p[i];
  }
}

// Mirror n floats starting at *(p - n) and store them at *p.
JXL_INLINE void RightMirror(float* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    p[i] = *(p - 1 - i);
  }
}

}  // namespace

#endif  // LIB_JXL_DEC_GRUOP_CC_

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::ShiftRight;

using D = HWY_FULL(float);
using DU = HWY_FULL(uint32_t);
using DI = HWY_FULL(int32_t);
using DI16 = HWY_CAPPED(int16_t, MaxLanes(DI()));
constexpr D d;
constexpr DI di;
constexpr DI16 di16;

template <class V>
void DequantLane(V scaled_dequant, V x_dm_multiplier, V b_dm_multiplier,
                 const float* JXL_RESTRICT dequant_matrices, size_t dq_ofs,
                 size_t size, size_t k, V x_cc_mul, V b_cc_mul,
                 const float* JXL_RESTRICT biases, float* JXL_RESTRICT block) {
  const auto x_mul =
      Load(d, dequant_matrices + dq_ofs + k) * scaled_dequant * x_dm_multiplier;
  const auto y_mul =
      Load(d, dequant_matrices + dq_ofs + size + k) * scaled_dequant;
  const auto b_mul = Load(d, dequant_matrices + dq_ofs + 2 * size + k) *
                     scaled_dequant * b_dm_multiplier;

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
void DequantBlock(const AcStrategy& acs, float inv_global_scale, int quant,
                  float x_dm_multiplier, float b_dm_multiplier, V x_cc_mul,
                  V b_cc_mul, size_t kind, size_t size,
                  const Quantizer& quantizer,
                  const float* JXL_RESTRICT dequant_matrices,
                  size_t covered_blocks, const size_t* sbx,
                  const float* JXL_RESTRICT* JXL_RESTRICT dc_row,
                  size_t dc_stride, const float* JXL_RESTRICT biases,
                  float* JXL_RESTRICT block) {
  PROFILER_FUNC;

  const auto scaled_dequant = Set(d, inv_global_scale / quant);
  const auto x_dm_multiplier_v = Set(d, x_dm_multiplier);
  const auto b_dm_multiplier_v = Set(d, b_dm_multiplier);

  const size_t dq_ofs = quantizer.DequantMatrixOffset(kind, 0);

  for (size_t k = 0; k < covered_blocks * kDCTBlockSize; k += Lanes(d)) {
    DequantLane(scaled_dequant, x_dm_multiplier_v, b_dm_multiplier_v,
                dequant_matrices, dq_ofs, size, k, x_cc_mul, b_cc_mul, biases,
                block);
  }
  for (size_t c = 0; c < 3; c++) {
    LowestFrequenciesFromDC(acs.Strategy(), dc_row[c] + sbx[c], dc_stride,
                            block + c * size);
  }
}

template <class GetBlock>
Status DecodeGroupImpl(GetBlock* JXL_RESTRICT get_block,
                       GroupDecCache* JXL_RESTRICT group_dec_cache,
                       PassesDecoderState* JXL_RESTRICT dec_state,
                       size_t thread, const Rect& block_rect, AuxOut* aux_out,
                       Image3F* JXL_RESTRICT idct, const ImageBundle* decoded) {
  PROFILER_FUNC;

  const LoopFilter& lf = dec_state->shared->frame_header.loop_filter;
  const AcStrategyImage& ac_strategy = dec_state->shared->ac_strategy;

  const size_t xsize_blocks = block_rect.xsize();
  const size_t ysize_blocks = block_rect.ysize();

  const size_t dc_stride = dec_state->shared->dc->PixelsPerRow();

  const float inv_global_scale = dec_state->shared->quantizer.InvGlobalScale();
  const float* JXL_RESTRICT dequant_matrices =
      dec_state->shared->quantizer.DequantMatrix(0, 0);

  const size_t sigma_stride = dec_state->filter_weights.sigma.PixelsPerRow();
  const size_t sharpness_stride =
      dec_state->shared->epf_sharpness.PixelsPerRow();

  const YCbCrChromaSubsampling& cs =
      dec_state->shared->frame_header.chroma_subsampling;

  bool save_to_decoded =
      !decoded->IsJPEG() && (lf.epf_iters > 0 || lf.gab) && cs.Is444();
  size_t xpadding_blocks = save_to_decoded ? kMaxFilterPadding / kBlockDim : 0;

  const size_t idct_stride =
      (!save_to_decoded ? *idct : dec_state->decoded).PixelsPerRow();

  const float quant_scale = dec_state->shared->quantizer.Scale();

  HWY_ALIGN int32_t scaled_qtable[64 * 3];

  if (decoded->IsJPEG()) {
    if (!dec_state->shared->cmap.IsJPEGCompatible()) {
      return JXL_FAILURE("The CfL map is not JPEG-compatible");
    }
    const std::vector<QuantEncoding>& qe =
        dec_state->shared->matrices.encodings();
    if (qe.empty() || qe[0].mode != QuantEncoding::Mode::kQuantModeRAW ||
        qe[0].qraw.qtable_den_shift != 0) {
      return JXL_FAILURE(
          "Quantization table is not a JPEG quantization table.");
    }
    for (size_t c = 0; c < 3; c++) {
      for (size_t i = 0; i < 64; i++) {
        // Transpose the matrix, as it will be used on the transposed block.
        scaled_qtable[64 * c + (i % 8) * 8 + (i / 8)] =
            (1 << kCFLFixedPointPrecision) * (*qe[0].qraw.qtable)[64 + i] /
            (*qe[0].qraw.qtable)[64 * c + i];
      }
    }
  }

  // Apply image features to
  // - the whole AC group, if no loop filtering is enabled, or
  // - only the interior part of the group, skipping the border
  // ... unless this is the first or the last group in a row, in which case we
  // process the corresponding border too.
  size_t xsize = xsize_blocks * kBlockDim;
  size_t ysize = ysize_blocks * kBlockDim;
  size_t xstart = block_rect.x0() != 0 ? lf.PaddingCols() : 0;
  size_t ystart = block_rect.y0() != 0 ? lf.PaddingRows() : 0;
  bool is_last_group_x = dec_state->shared->frame_dim.xsize_blocks ==
                         block_rect.x0() + block_rect.xsize();
  bool is_last_group_y = dec_state->shared->frame_dim.ysize_blocks ==
                         block_rect.y0() + block_rect.ysize();
  size_t xend = is_last_group_x ? xsize : xsize - lf.PaddingCols();
  size_t yend = is_last_group_y ? ysize : ysize - lf.PaddingRows();

  // No ApplyImageFeatures in JPEG mode, or if using chroma subsampling. It will
  // be done after decoding the whole image (this allows it to work on the
  // chroma channels too), or if there is not at least one pass per AC group.
  bool run_apply_image_features = xstart < xend && ystart < yend &&
                                  !decoded->IsJPEG() && cs.Is444() &&
                                  !dec_state->has_partial_ac_groups;

  static_assert(kApplyImageFeaturesTileDim >= kGroupDim,
                "Groups are too large");
  const Rect aif_rect(block_rect.x0() * kBlockDim + xstart,
                      block_rect.y0() * kBlockDim + ystart, xend - xstart,
                      yend - ystart);

  int jpeg_c_map[3] = {1, 0, 2};
  if (decoded->IsJPEG() && decoded->jpeg_data->components.size() == 1) {
    jpeg_c_map[0] = jpeg_c_map[2] = 0;
  }

  if (!cs.Is444()) {
    size_t hshift[3] = {cs.HShift(0), cs.HShift(1), cs.HShift(2)};
    size_t vshift[3] = {cs.VShift(0), cs.VShift(1), cs.VShift(2)};
    Rect r[3];
    for (size_t i = 0; i < 3; i++) {
      r[i] = Rect(block_rect.x0() >> hshift[i], block_rect.y0() >> vshift[i],
                  block_rect.xsize() >> hshift[i],
                  block_rect.ysize() >> vshift[i]);
    }

    for (size_t by = 0; by < ysize_blocks; ++by) {
      get_block->StartRow(by);
      size_t sby[3] = {by >> vshift[0], by >> vshift[1], by >> vshift[2]};

      const int32_t* JXL_RESTRICT row_quant =
          block_rect.ConstRow(dec_state->shared->raw_quant_field, by);

      const float* JXL_RESTRICT dc_rows[3] = {
          r[0].ConstPlaneRow(*dec_state->shared->dc, 0, sby[0]),
          r[1].ConstPlaneRow(*dec_state->shared->dc, 1, sby[1]),
          r[2].ConstPlaneRow(*dec_state->shared->dc, 2, sby[2]),
      };

      AcStrategyRow acs_row = ac_strategy.ConstRow(block_rect, by);

      float* JXL_RESTRICT idct_row[3];
      int16_t* JXL_RESTRICT jpeg_row[3];
      for (size_t c = 0; c < 3; c++) {
        idct_row[c] = idct->PlaneRow(c, (r[c].y0() + sby[c]) * kBlockDim) +
                      r[c].x0() * kBlockDim;
        if (decoded->IsJPEG()) {
          auto& component = decoded->jpeg_data->components[jpeg_c_map[c]];
          jpeg_row[c] =
              component.coeffs.data() +
              (component.width_in_blocks * (r[c].y0() + sby[c]) + r[c].x0()) *
                  kDCTBlockSize;
        }
      }

      auto x_cc_mul = Set(d, 0.0f);
      auto b_cc_mul = Set(d, 0.0f);
      // Increment bx by llf_x because those iterations would otherwise
      // immediately continue (!IsFirstBlock). Reduces mispredictions.
      for (size_t bx = 0; bx < xsize_blocks;) {
        size_t sbx[3] = {bx >> hshift[0], bx >> hshift[1], bx >> hshift[2]};
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

        HWY_ALIGN float* const block = group_dec_cache->dec_group_block;
        JXL_RETURN_IF_ERROR(
            get_block->GetBlock(bx, by, acs, size, log2_covered_blocks, block));
        if (JXL_UNLIKELY(decoded->IsJPEG())) {
          HWY_ALIGN float* const local_block =
              group_dec_cache->dec_group_local_block;
          for (size_t c : {1, 0, 2}) {
            if (decoded->jpeg_data->components.size() == 1 && c != 1) {
              continue;
            }
            if ((sbx[c] << hshift[c] != bx) || (sby[c] << vshift[c] != by)) {
              continue;
            }
            Transpose<8, 8>::Run(DCTFrom(block + c * size, 8),
                                 DCTTo(local_block, 8));
            local_block[0] = dc_rows[c][sbx[c]];
            int16_t* JXL_RESTRICT jpeg_pos =
                jpeg_row[c] + sbx[c] * kDCTBlockSize;
            for (size_t i = 0; i < 64; i += Lanes(d)) {
              const auto inf = Load(d, local_block + i);
              const auto ini = ConvertTo(di, inf);
              const auto ini16 = DemoteTo(di16, ini);
              StoreU(ini16, di16, jpeg_pos + i);
            }
          }
          bx += llf_x;
          continue;
        }

        // Dequantize and add predictions.
        {
          DequantBlock(
              acs, inv_global_scale, row_quant[bx], dec_state->x_dm_multiplier,
              dec_state->b_dm_multiplier, x_cc_mul, b_cc_mul, acs.RawStrategy(),
              size, dec_state->shared->quantizer, dequant_matrices,
              acs.covered_blocks_y() * acs.covered_blocks_x(), sbx, dc_rows,
              dc_stride, dec_state->shared->opsin_params.quant_biases, block);
        }

        for (size_t c : {1, 0, 2}) {
          if ((sbx[c] << hshift[c] != bx) || (sby[c] << vshift[c] != by)) {
            continue;
          }
          float* JXL_RESTRICT idct_pos = idct_row[c] + sbx[c] * kBlockDim;
          TransformToPixels(acs.Strategy(), block + c * size, idct_pos,
                            idct_stride, group_dec_cache->scratch_space);
        }

        bx += llf_x;
      }
    }
  } else {
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

      const int8_t* JXL_RESTRICT row_cmap[3] = {
          dec_state->shared->cmap.ytox_map.ConstRow(ty),
          nullptr,
          dec_state->shared->cmap.ytob_map.ConstRow(ty),
      };

      float* JXL_RESTRICT sigma_row =
          lf.epf_iters > 0
              ? block_rect.Row(&dec_state->filter_weights.sigma, by)
              : nullptr;
      const uint8_t* JXL_RESTRICT sharpness_row =
          block_rect.ConstRow(dec_state->shared->epf_sharpness, by);
      float* JXL_RESTRICT idct_row[3];
      int16_t* JXL_RESTRICT jpeg_row[3];
      for (size_t c = 0; c < 3; c++) {
        idct_row[c] = (!save_to_decoded ? *idct : dec_state->decoded)
                          .PlaneRow(c, (block_rect.y0() + by) * kBlockDim) +
                      (block_rect.x0()) * kBlockDim;
        if (decoded->IsJPEG()) {
          auto& component = decoded->jpeg_data->components[jpeg_c_map[c]];
          jpeg_row[c] = component.coeffs.data() +
                        (component.width_in_blocks * (block_rect.y0() + by) +
                         block_rect.x0()) *
                            kDCTBlockSize;
        }
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
          size_t sbx[3] = {bx, bx, bx};
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

          HWY_ALIGN float* const block = group_dec_cache->dec_group_block;
          JXL_RETURN_IF_ERROR(get_block->GetBlock(bx, by, acs, size,
                                                  log2_covered_blocks, block));

          if (JXL_UNLIKELY(decoded->IsJPEG())) {
            if (acs.Strategy() != AcStrategy::Type::DCT) {
              return JXL_FAILURE(
                  "Can only decode to JPEG if only DCT-8 is used.");
            }

            HWY_ALIGN int transposed_dct_int[64];
            for (size_t c : {1, 0, 2}) {
              if (decoded->jpeg_data->components.size() == 1 && c != 1) {
                continue;
              }
              int16_t* JXL_RESTRICT jpeg_pos = jpeg_row[c] + bx * kDCTBlockSize;
              // JPEG XL is transposed, JPEG is not.
              HWY_ALIGN float transposed_dct[64];
              Transpose<8, 8>::Run(DCTFrom(block + c * size, 8),
                                   DCTTo(transposed_dct, 8));
              // No CfL - no need to store the block converted to integers.
              if (row_cmap[0][abs_tx] == 0 && row_cmap[2][abs_tx] == 0) {
                for (size_t i = 0; i < 64; i += Lanes(d)) {
                  const auto inf = Load(d, transposed_dct + i);
                  const auto ini = ConvertTo(di, inf);
                  const auto ini16 = DemoteTo(di16, ini);
                  StoreU(ini16, di16, jpeg_pos + i);
                }
              } else if (c == 1) {
                for (size_t i = 0; i < 64; i += Lanes(d)) {
                  const auto inf = Load(d, transposed_dct + i);
                  const auto ini = ConvertTo(di, inf);
                  Store(ini, di, transposed_dct_int + i);
                  const auto ini16 = DemoteTo(di16, ini);
                  StoreU(ini16, di16, jpeg_pos + i);
                }
              } else {
                // transposed_dct_int contains the y channel block, converted to
                // ints and transposed.
                const auto scale = Set(
                    di, dec_state->shared->cmap.RatioJPEG(row_cmap[c][abs_tx]));
                const auto round = Set(di, 1 << (kCFLFixedPointPrecision - 1));
                for (int i = 0; i < 64; i += Lanes(d)) {
                  auto in = ConvertTo(di, Load(d, transposed_dct + i));
                  auto in_y = Load(di, transposed_dct_int + i);
                  auto qt = Load(di, scaled_qtable + c * size + i);
                  auto coeff_scale =
                      ShiftRight<kCFLFixedPointPrecision>(qt * scale + round);
                  auto cfl_factor = ShiftRight<kCFLFixedPointPrecision>(
                      in_y * coeff_scale + round);
                  Store(DemoteTo(di16, in + cfl_factor), di16, jpeg_pos + i);
                }
              }
              jpeg_pos[0] = dc_rows[c][bx];
            }
            bx += llf_x;
            continue;
          }

          // Dequantize and add predictions.
          {
            DequantBlock(acs, inv_global_scale, row_quant[bx],
                         dec_state->x_dm_multiplier, dec_state->b_dm_multiplier,
                         x_cc_mul, b_cc_mul, acs.RawStrategy(), size,
                         dec_state->shared->quantizer, dequant_matrices,
                         acs.covered_blocks_y() * acs.covered_blocks_x(), sbx,
                         dc_rows, dc_stride,
                         dec_state->shared->opsin_params.quant_biases, block);
          }

          for (size_t c : {1, 0, 2}) {
            // IDCT
            float* JXL_RESTRICT idct_pos =
                idct_row[c] + (xpadding_blocks + bx) * kBlockDim;
            TransformToPixels(acs.Strategy(), block + c * size, idct_pos,
                              idct_stride, group_dec_cache->scratch_space);
          }

          if (lf.epf_iters > 0) {
            // quant_scale is smaller for low quality.
            // quant_scale is roughly 0.08 / butteraugli score.
            //
            // row_quant is smaller for low quality.
            // row_quant is a quantization multiplier of form 1.0 /
            // row_quant[bx]
            //
            // lf.epf_quant_mul is a parameter in the format
            // kInvSigmaNum is a constant
            float sigma_quant =
                lf.epf_quant_mul / (quant_scale * row_quant[bx] * kInvSigmaNum);
            for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
              for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
                float sigma =
                    sigma_quant *
                    lf.epf_sharp_lut[sharpness_row[bx + ix +
                                                   iy * sharpness_stride]];
                // Avoid infinities.
                sigma = std::min(-1e-4f, sigma);  // TODO(veluca): remove this.
                sigma_row[bx + ix + kSigmaPadding +
                          (iy + kSigmaPadding) * sigma_stride] = 1.0f / sigma;
              }
            }
            // Left padding with mirroring.
            if (bx + block_rect.x0() == 0) {
              for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
                LeftMirror(sigma_row + kSigmaPadding +
                               (iy + kSigmaPadding) * sigma_stride,
                           kSigmaBorder);
              }
            }
            // Right padding with mirroring.
            if (bx + block_rect.x0() + llf_x ==
                dec_state->shared->frame_dim.xsize_blocks) {
              for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
                RightMirror(sigma_row + kSigmaPadding + bx + llf_x +
                                (iy + kSigmaPadding) * sigma_stride,
                            kSigmaBorder);
              }
            }
            // Offsets for row copying, in blocks.
            size_t offset_before =
                bx + block_rect.x0() == 0 ? 1 : bx + kSigmaPadding;
            size_t offset_after =
                bx + block_rect.x0() + llf_x ==
                        dec_state->shared->frame_dim.xsize_blocks
                    ? kSigmaPadding + llf_x + bx + kSigmaBorder
                    : kSigmaPadding + llf_x + bx;
            size_t num = offset_after - offset_before;
            // Above
            if (by + block_rect.y0() == 0) {
              for (size_t iy = 0; iy < kSigmaBorder; iy++) {
                memcpy(sigma_row + offset_before +
                           (kSigmaPadding - 1 - iy) * sigma_stride,
                       sigma_row + offset_before +
                           (kSigmaPadding + iy) * sigma_stride,
                       num * sizeof(*sigma_row));
              }
            }
            // Below
            if (by + block_rect.y0() + acs.covered_blocks_y() ==
                dec_state->shared->frame_dim.ysize_blocks) {
              for (size_t iy = 0; iy < kSigmaBorder; iy++) {
                memcpy(sigma_row + offset_before +
                           sigma_stride *
                               (acs.covered_blocks_y() + kSigmaPadding + iy),
                       sigma_row + offset_before +
                           sigma_stride * (acs.covered_blocks_y() +
                                           kSigmaPadding - 1 - iy),
                       num * sizeof(*sigma_row));
              }
            }
          }

          if (save_to_decoded) {
            for (size_t c : {1, 0, 2}) {
              // Left padding with mirroring.
              if (bx + block_rect.x0() == 0) {
                for (size_t iy = 0; iy < kBlockDim * acs.covered_blocks_y();
                     iy++) {
                  // We need kMaxFilterBorder worth of mirrored pixels but the
                  // image is padded with kMaxFilterPadding pixels.
                  LeftMirror(idct_row[c] + kMaxFilterPadding + idct_stride * iy,
                             kMaxFilterBorder);
                }
              }
              // Right padding with mirroring.
              if (bx + block_rect.x0() + llf_x ==
                  dec_state->shared->frame_dim.xsize_blocks) {
                for (size_t iy = 0; iy < kBlockDim * acs.covered_blocks_y();
                     iy++) {
                  RightMirror(idct_row[c] + kMaxFilterPadding +
                                  (bx + llf_x) * kBlockDim + idct_stride * iy,
                              kMaxFilterBorder);
                }
              }
            }
          }
          bx += llf_x;
        }
      }
      // When a row of blocks is done, run ApplyImageFeaturesRow.
      // TODO(veluca): consider switching this to 4 rows of blocks.
      if (JXL_LIKELY(run_apply_image_features)) {
        ssize_t yb = by == 0 ? -kMaxFilterBorder : 0;
        ssize_t ye =
            kBlockDim + (by == ysize_blocks - 1 ? kMaxFilterBorder : 0);
        for (ssize_t y = yb; y < ye; y++) {
          ssize_t aif_y = by * kBlockDim + y - ystart;
          JXL_RETURN_IF_ERROR(ApplyImageFeaturesRow(idct, aif_rect, dec_state,
                                                    aif_y, thread, aux_out));
        }
      }
    }
  }
  return true;
}

// Decode quantized AC coefficients of DCT blocks.
// LLF components in the output block will not be modified.
Status DecodeACVarBlock(size_t ctx_offset, size_t log2_covered_blocks,
                        int32_t* JXL_RESTRICT row_nzeros,
                        const int32_t* JXL_RESTRICT row_nzeros_top,
                        size_t nzeros_stride, size_t c, size_t bx, size_t by,
                        size_t lbx, AcStrategy acs,
                        const coeff_order_t* JXL_RESTRICT coeff_order,
                        BitReader* JXL_RESTRICT br,
                        ANSSymbolReader* JXL_RESTRICT decoder,
                        const std::vector<uint8_t>& context_map,
                        const uint8_t* qdc_row, const int32_t* qf_row,
                        const BlockCtxMap& block_ctx_map,
                        ac_qcoeff_t* JXL_RESTRICT block, size_t shift = 0) {
  PROFILER_FUNC;
  // Equal to number of LLF coefficients.
  const size_t covered_blocks = 1 << log2_covered_blocks;
  const size_t size = covered_blocks * kDCTBlockSize;
  int32_t predicted_nzeros =
      PredictFromTopAndLeft(row_nzeros_top, row_nzeros, bx, 32);

  size_t ord = kStrategyOrder[acs.RawStrategy()];
  const coeff_order_t* JXL_RESTRICT order =
      &coeff_order[CoeffOrderOffset(ord, c)];

  size_t block_ctx = block_ctx_map.Context(qdc_row[lbx], qf_row[bx], ord, c);
  const int32_t nzero_ctx =
      block_ctx_map.NonZeroContext(predicted_nzeros, block_ctx) + ctx_offset;

  size_t nzeros = decoder->ReadHybridUint(nzero_ctx, br, context_map);
  if (nzeros + covered_blocks > size) {
    return JXL_FAILURE("Invalid AC: nzeros too large");
  }
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
      row_nzeros[bx + x + y * nzeros_stride] =
          (nzeros + covered_blocks - 1) >> log2_covered_blocks;
    }
  }

  const size_t histo_offset =
      ctx_offset + block_ctx_map.ZeroDensityContextsOffset(block_ctx);

  // Skip LLF
  {
    PROFILER_ZONE("AcDecSkipLLF, reader");
    size_t prev = (nzeros > size / 16 ? 0 : 1);
    for (size_t k = covered_blocks; k < size && nzeros != 0; ++k) {
      const size_t ctx =
          histo_offset + ZeroDensityContext(nzeros, k, covered_blocks,
                                            log2_covered_blocks, prev);
      const size_t u_coeff = decoder->ReadHybridUint(ctx, br, context_map);
      // Hand-rolled version of UnpackSigned, shifting before the conversion to
      // signed integer to avoid undefined behavior of shifting negative
      // numbers.
      const size_t magnitude = u_coeff >> 1;
      const size_t neg_sign = (~u_coeff) & 1;
      const intptr_t coeff =
          static_cast<intptr_t>((magnitude ^ (neg_sign - 1)) << shift);
      block[order[k]] += static_cast<ac_qcoeff_t>(coeff);
      prev = static_cast<size_t>(u_coeff != 0);
      nzeros -= prev;
    }
    if (JXL_UNLIKELY(nzeros != 0)) {
      return JXL_FAILURE(
          "Invalid AC: nzeros not 0. Block (%zu, %zu), channel %zu", bx, by, c);
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
    qf_row = rect.ConstRow(*qf, by);
    for (size_t c = 0; c < 3; c++) {
      size_t sby = by >> vshift[c];
      quant_dc_row = quant_dc->ConstRow(rect.y0() + by) + rect.x0();
      for (size_t i = 0; i < num_passes; i++) {
        row_nzeros[i][c] = group_dec_cache->num_nzeroes[i].PlaneRow(c, sby);
        row_nzeros_top[i][c] =
            sby == 0
                ? nullptr
                : group_dec_cache->num_nzeroes[i].ConstPlaneRow(c, sby - 1);
      }
    }
  }

  Status GetBlock(size_t bx, size_t by, const AcStrategy& acs, size_t size,
                  size_t log2_covered_blocks, float* JXL_RESTRICT block) {
    memset(block, 0, sizeof(float) * size * 3);
    for (size_t c : {1, 0, 2}) {
      float* JXL_RESTRICT block_c = block + c * size;
      size_t sbx = bx >> hshift[c];
      size_t sby = by >> vshift[c];
      if (JXL_UNLIKELY((sbx << hshift[c] != bx) || (sby << vshift[c] != by))) {
        continue;
      }

      for (size_t pass = 0; JXL_UNLIKELY(pass < num_passes); pass++) {
        JXL_RETURN_IF_ERROR(DecodeACVarBlock(
            ctx_offset[pass], log2_covered_blocks, row_nzeros[pass][c],
            row_nzeros_top[pass][c], nzeros_stride, c, sbx, sby, bx, acs,
            &coeff_orders[pass * kCoeffOrderSize], readers[pass],
            &decoders[pass], context_map[pass], quant_dc_row, qf_row,
            *block_ctx_map, block_c, shift_for_pass[pass]));
      }
    }
    return true;
  }

  Status Init(BitReader* JXL_RESTRICT* JXL_RESTRICT readers, size_t num_passes,
              size_t group_idx, size_t histo_selector_bits, const Rect& rect,
              GroupDecCache* JXL_RESTRICT group_dec_cache,
              PassesDecoderState* dec_state) {
    for (size_t i = 0; i < 3; i++) {
      hshift[i] = dec_state->shared->frame_header.chroma_subsampling.HShift(i);
      vshift[i] = dec_state->shared->frame_header.chroma_subsampling.VShift(i);
    }
    this->coeff_orders = dec_state->shared->coeff_orders.data();
    this->context_map = dec_state->context_map.data();
    this->readers = readers;
    this->num_passes = num_passes;
    this->shift_for_pass = dec_state->shared->frame_header.passes.shift;
    this->group_dec_cache = group_dec_cache;
    this->rect = rect;
    block_ctx_map = &dec_state->shared->block_ctx_map;
    qf = &dec_state->shared->raw_quant_field;
    quant_dc = &dec_state->shared->quant_dc;

    for (size_t pass = 0; pass < num_passes; pass++) {
      // Select which histogram set to use among those of the current pass.
      size_t cur_histogram = 0;
      if (histo_selector_bits != 0) {
        cur_histogram = readers[pass]->ReadBits(histo_selector_bits);
      }
      if (cur_histogram >= dec_state->shared->num_histograms) {
        return JXL_FAILURE("Invalid histogram selector");
      }
      ctx_offset[pass] = cur_histogram * block_ctx_map->NumACContexts();

      decoders[pass] = ANSSymbolReader(&dec_state->code[pass], readers[pass]);
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
  const coeff_order_t* JXL_RESTRICT coeff_orders;
  const std::vector<uint8_t>* JXL_RESTRICT context_map;
  ANSSymbolReader decoders[kMaxNumPasses];
  BitReader* JXL_RESTRICT* JXL_RESTRICT readers;
  size_t num_passes;
  size_t ctx_offset[kMaxNumPasses];
  size_t nzeros_stride;
  int32_t* JXL_RESTRICT row_nzeros[kMaxNumPasses][3];
  const int32_t* JXL_RESTRICT row_nzeros_top[kMaxNumPasses][3];
  GroupDecCache* JXL_RESTRICT group_dec_cache;
  const BlockCtxMap* block_ctx_map;
  const ImageI* qf;
  const ImageB* quant_dc;
  const int32_t* qf_row;
  const uint8_t* quant_dc_row;
  Rect rect;
  size_t hshift[3], vshift[3];
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
    // TODO(veluca): not supported with chroma subsampling.
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

Status DecodeGroup(BitReader* JXL_RESTRICT* JXL_RESTRICT readers,
                   size_t num_passes, size_t group_idx,
                   PassesDecoderState* JXL_RESTRICT dec_state,
                   GroupDecCache* JXL_RESTRICT group_dec_cache, size_t thread,
                   Image3F* opsin, ImageBundle* JXL_RESTRICT decoded,
                   AuxOut* aux_out) {
  PROFILER_FUNC;
  const Rect block_group_rect = dec_state->shared->BlockGroupRect(group_idx);

  group_dec_cache->InitOnce(num_passes);

  size_t histo_selector_bits =
      CeilLog2Nonzero(dec_state->shared->num_histograms);

  GetBlockFromBitstream get_block;
  JXL_RETURN_IF_ERROR(get_block.Init(readers, num_passes, group_idx,
                                     histo_selector_bits, block_group_rect,
                                     group_dec_cache, dec_state));

  JXL_RETURN_IF_ERROR(DecodeGroupImpl(&get_block, group_dec_cache, dec_state,
                                      thread, block_group_rect, aux_out, opsin,
                                      decoded));

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
                               GroupDecCache* JXL_RESTRICT group_dec_cache,
                               size_t thread, Image3F* JXL_RESTRICT opsin,
                               ImageBundle* JXL_RESTRICT decoded,
                               AuxOut* aux_out) {
  PROFILER_FUNC;

  const Rect block_group_rect = dec_state->shared->BlockGroupRect(group_idx);

  GetBlockFromEncoder get_block(ac, group_idx);

  return DecodeGroupImpl(&get_block, group_dec_cache, dec_state, thread,
                         block_group_rect, aux_out, opsin, decoded);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(DecodeGroup);
Status DecodeGroup(BitReader* JXL_RESTRICT* JXL_RESTRICT readers,
                   size_t num_passes, size_t group_idx,
                   PassesDecoderState* JXL_RESTRICT dec_state,
                   GroupDecCache* JXL_RESTRICT group_dec_cache, size_t thread,
                   Image3F* opsin, ImageBundle* JXL_RESTRICT decoded,
                   AuxOut* aux_out) {
  return HWY_DYNAMIC_DISPATCH(DecodeGroup)(readers, num_passes, group_idx,
                                           dec_state, group_dec_cache, thread,
                                           opsin, decoded, aux_out);
}

HWY_EXPORT(DecodeGroupForRoundtrip);
Status DecodeGroupForRoundtrip(const std::vector<ACImage3>& ac,
                               size_t group_idx,
                               PassesDecoderState* JXL_RESTRICT dec_state,
                               GroupDecCache* JXL_RESTRICT group_dec_cache,
                               size_t thread, Image3F* JXL_RESTRICT opsin,
                               ImageBundle* JXL_RESTRICT decoded,
                               AuxOut* aux_out) {
  return HWY_DYNAMIC_DISPATCH(DecodeGroupForRoundtrip)(ac, group_idx, dec_state,
                                                       group_dec_cache, thread,
                                                       opsin, decoded, aux_out);
}

}  // namespace jxl
#endif  // HWY_ONCE
