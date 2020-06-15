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

#include "tools/epf.h"

#include "jxl/ar_control_field.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/common.h"
#include "jxl/enc_adaptive_quantization.h"
#include "jxl/enc_xyb.h"
#include "jxl/epf.h"

namespace jpegxl {
namespace tools {

namespace {

// Requires `opsin` to be padded.
jxl::Image3F ComputeDC(const jxl::Image3F& opsin) {
  jxl::Image3F dc(opsin.xsize() / jxl::kBlockDim,
                  opsin.ysize() / jxl::kBlockDim);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < dc.ysize(); ++by) {
      float* const JXL_RESTRICT dc_row = dc.PlaneRow(c, by);
      for (size_t bx = 0; bx < dc.xsize(); ++bx) {
        dc_row[bx] = 0;
        for (size_t y = 0; y < jxl::kBlockDim; ++y) {
          const float* const JXL_RESTRICT row =
              opsin.ConstPlaneRow(c, by * jxl::kBlockDim + y) +
              bx * jxl::kBlockDim;
          for (size_t x = 0; x < jxl::kBlockDim; ++x) {
            dc_row[bx] += row[x];
          }
        }
        dc_row[bx] *= 1.f / (jxl::kBlockDim * jxl::kBlockDim);
      }
    }
  }
  return dc;
}

// Expects `state` to have `quantizer`, `raw_quant_field` and `dc` usable.
// TODO(sboukortt): extract this from dec_group.cc (DecodeGroupImpl) instead of
// copying it.
jxl::ImageF ComputeSigma(const jxl::Image3F& opsin,
                         const int sharpness_parameter,
                         const jxl::FrameDimensions& frame_dim,
                         jxl::PassesEncoderState* state,
                         const jxl::LoopFilter& lf) {
  jxl::ImageF sigma(frame_dim.xsize_blocks + 4, frame_dim.ysize_blocks + 4);
  const size_t sigma_stride = sigma.PixelsPerRow();
  const size_t dc_stride = state->shared.dc->PixelsPerRow();
  const float quant_scale = state->shared.quantizer.Scale();
  for (size_t by = 0; by < sigma.ysize(); ++by) {
    float* const JXL_RESTRICT sigma_row = sigma.Row(by);
    const int* const JXL_RESTRICT row_quant =
        state->shared.raw_quant_field.ConstRow(by);
    const float* const JXL_RESTRICT dc_rows[3] = {
        state->shared.dc->ConstPlaneRow(0, by),
        state->shared.dc->ConstPlaneRow(1, by),
        state->shared.dc->ConstPlaneRow(2, by),
    };
    for (size_t bx = 0; bx < sigma.xsize(); ++bx) {
      float quant = 1.0f / (row_quant[bx] * quant_scale);
      float sigma_quant = quant * lf.epf_quant_mul;
      // Increase sigma near edges.
      float dc_range = 0;
      for (size_t c = 0; c < 3; c++) {
        const float* JXL_RESTRICT base_dc_ptr = dc_rows[c] + bx;
        // UBSAN complains about overflowing unsigned addition here,
        // hence we use a slightly more convoluted syntax than simple
        // array access to ensure we only ever add or subtract positive
        // quantities.
        float dc_ref = *base_dc_ptr;
        float dc_top = *(base_dc_ptr - (by == 0 ? 0 : dc_stride));
        float dc_bottom =
            base_dc_ptr[by + 1 == frame_dim.ysize_blocks ? 0 : dc_stride];
        float dc_left = *(base_dc_ptr - (bx == 0 ? 0 : 1));
        float dc_right = base_dc_ptr[bx + 1 == frame_dim.xsize_blocks ? 0 : 1];

        float dc_range_c = std::abs(dc_top - dc_ref);
        dc_range_c = std::max(dc_range_c, std::abs(dc_bottom - dc_ref));
        dc_range_c = std::max(dc_range_c, std::abs(dc_left - dc_ref));
        dc_range_c = std::max(dc_range_c, std::abs(dc_right - dc_ref));
        dc_range = std::max(dc_range_c * lf.epf_channel_scale[c], dc_range);
      }
      float sigma =
          sigma_quant * (2.0f - 1.0f / (1.0f + lf.epf_dc_range_mul * dc_range));
      sigma *= lf.epf_sharp_lut[sharpness_parameter];
      // Avoid infinities.
      sigma = std::max(1e-4f, sigma);
      sigma_row[bx + 2 + 2 * sigma_stride] = jxl::kInvSigmaNum / sigma;
      // Left padding with mirroring.
      if (bx == 0) {
        sigma_row[1 + 2 * sigma_stride] = sigma_row[2 + 2 * sigma_stride];
      }
      // Right padding with mirroring.
      if (bx + 1 == frame_dim.xsize_blocks) {
        sigma_row[2 * sigma_stride + bx + 3] =
            sigma_row[2 * sigma_stride + bx + 2];
      }
      // Offsets for row copying, in blocks.
      size_t offset_before = bx == 0 ? 1 : bx + 2;
      size_t offset_after = bx + 1 == frame_dim.xsize_blocks ? bx + 4 : bx + 3;
      size_t num = offset_after - offset_before;
      // Above
      if (by == 0) {
        memcpy(sigma_row + offset_before + sigma_stride,
               sigma_row + offset_before + 2 * sigma_stride,
               num * sizeof(*sigma_row));
      }
      // Below
      if (by + 1 == frame_dim.ysize_blocks) {
        memcpy(sigma_row + offset_before + sigma_stride * 3,
               sigma_row + offset_before + sigma_stride * 2,
               num * sizeof(*sigma_row));
      }
    }
  }
  return sigma;
}

}  // namespace

jxl::Status RunEPF(const float distance, const int sharpness_parameter,
                   jxl::CodecInOut* const io, jxl::ThreadPool* const pool) {
  const jxl::ColorEncoding original_color_encoding =
      io->metadata.color_encoding;
  jxl::Image3F opsin(io->xsize(), io->ysize());
  jxl::ImageBundle unused_linear;
  (void)ToXYB(io->Main(), pool, &opsin, &unused_linear);

  const size_t original_xsize = opsin.xsize(), original_ysize = opsin.ysize();
  opsin = PadImageToMultiple(opsin, jxl::kBlockDim);

  jxl::FrameDimensions frame_dim;
  frame_dim.Set(original_xsize, original_ysize);

  static constexpr float kAcQuant = 0.84f;
  const float dc_quant = jxl::InitialQuantDC(distance);
  const float ac_quant = kAcQuant / distance;
  jxl::PassesEncoderState state;
  jxl::InitializePassesEncoder(opsin, pool, &state, /*aux_out=*/nullptr);
  state.shared.raw_quant_field =
      jxl::ImageI(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  state.shared.quantizer.SetQuant(dc_quant, ac_quant,
                                  &state.shared.raw_quant_field);
  state.shared.dc_storage = ComputeDC(opsin);

  jxl::LoopFilter lf;
  jxl::Image3F storage1(frame_dim.xsize_padded + 4 * jxl::kBlockDim,
                        jxl::kEpf1InputRows);
  jxl::Image3F storage2(frame_dim.xsize_padded + 2 * jxl::kBlockDim,
                        jxl::kEpf2InputRows);
  lf.gab = false;
  const jxl::ImageF sigma =
      ComputeSigma(opsin, sharpness_parameter, frame_dim, &state, lf);
  const jxl::Image3F padded = PadImageSymmetric(opsin, 2 * jxl::kBlockDim);

  jxl::Rect image_rect(0, 0, frame_dim.xsize_padded, frame_dim.ysize_padded);
  jxl::Rect sigma_rect(0, 0, frame_dim.xsize_blocks, frame_dim.ysize_blocks);

  EdgePreservingFilter(lf, image_rect, padded, sigma_rect, sigma, image_rect,
                       &opsin, &storage1, &storage2);

  opsin.ShrinkTo(original_xsize, original_ysize);
  jxl::OpsinParams opsin_params;
  opsin_params.Init();
  jxl::OpsinToLinearInplace(&opsin, pool, opsin_params);
  io->Main().SetFromImage(std::move(opsin),
                          jxl::ColorEncoding::LinearSRGB(io->Main().IsGray()));
  JXL_RETURN_IF_ERROR(io->TransformTo(original_color_encoding, pool));
  return true;
}

}  // namespace tools
}  // namespace jpegxl
