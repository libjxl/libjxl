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
#include "jxl/enc_modular.h"
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
// Computes the sigma values into the `sigma` variable.
// TODO(sboukortt): extract this from dec_group.cc (DecodeGroupImpl) instead of
// copying it.
void ComputeSigma(jxl::ImageF* sigma, const jxl::Image3F& opsin,
                  const int sharpness_parameter,
                  const jxl::FrameDimensions& frame_dim,
                  jxl::PassesEncoderState* state, const jxl::LoopFilter& lf) {
  JXL_DASSERT(sigma->xsize() == frame_dim.xsize_blocks + 4);
  JXL_DASSERT(sigma->ysize() == frame_dim.ysize_blocks + 4);
  const size_t sigma_stride = sigma->PixelsPerRow();
  const float quant_scale = state->shared.quantizer.Scale();
  for (size_t by = 0; by < sigma->ysize(); ++by) {
    float* const JXL_RESTRICT sigma_row = sigma->Row(by);
    const int* const JXL_RESTRICT row_quant =
        state->shared.raw_quant_field.ConstRow(by);
    for (size_t bx = 0; bx < sigma->xsize(); ++bx) {
      float quant = 1.0f / (row_quant[bx] * quant_scale);
      float sigma = quant * lf.epf_quant_mul;
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
}

}  // namespace

jxl::Status RunEPF(uint32_t epf_iters, const float distance,
                   const int sharpness_parameter, jxl::CodecInOut* const io,
                   jxl::ThreadPool* const pool) {
  const jxl::ColorEncoding original_color_encoding =
      io->metadata.color_encoding;
  jxl::Image3F opsin(io->xsize(), io->ysize());
  jxl::ImageBundle unused_linear;
  (void)ToXYB(io->Main(), pool, &opsin, &unused_linear);

  const size_t original_xsize = opsin.xsize(), original_ysize = opsin.ysize();
  opsin = PadImageToMultiple(opsin, jxl::kBlockDim);

  jxl::FrameDimensions frame_dim;
  frame_dim.Set(original_xsize, original_ysize, /*group_size_shift=*/1,
                /*max_hshift=*/0, /*max_vshift=*/0);

  static constexpr float kAcQuant = 0.84f;
  const float dc_quant = jxl::InitialQuantDC(distance);
  const float ac_quant = kAcQuant / distance;
  jxl::PassesEncoderState state;
  jxl::ModularFrameEncoder modular_frame_encoder(frame_dim, jxl::FrameHeader{},
                                                 jxl::CompressParams{});
  jxl::InitializePassesEncoder(opsin, pool, &state, &modular_frame_encoder,
                               /*aux_out=*/nullptr);
  state.shared.raw_quant_field =
      jxl::ImageI(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  state.shared.quantizer.SetQuant(dc_quant, ac_quant,
                                  &state.shared.raw_quant_field);
  state.shared.dc_storage = ComputeDC(opsin);

  jxl::LoopFilter lf;
  jxl::FilterStorage storage(frame_dim.xsize_padded);
  lf.gab = false;
  lf.epf_iters = epf_iters;

  jxl::FilterWeights filter_weights;
  filter_weights.Init(lf, frame_dim);
  ComputeSigma(&filter_weights.sigma, opsin, sharpness_parameter, frame_dim,
               &state, lf);

  const jxl::Image3F padded = PadImageSymmetric(opsin, 2 * jxl::kBlockDim);

  jxl::Rect image_rect(0, 0, frame_dim.xsize_padded, frame_dim.ysize_padded);
  jxl::Rect sigma_rect(0, 0, frame_dim.xsize_blocks, frame_dim.ysize_blocks);

  EdgePreservingFilter(lf, filter_weights, image_rect, padded, sigma_rect,
                       image_rect, &opsin, &storage);

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
