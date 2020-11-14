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

#include "jxl/headers.h"
#include "lib/jxl/ar_control_field.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/epf.h"

using jxl::kSigmaBorder;
using jxl::kSigmaPadding;

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
      sigma_row[bx + kSigmaPadding + kSigmaPadding * sigma_stride] =
          jxl::kInvSigmaNum / sigma;
      // Left padding with mirroring.
      if (bx == 0) {
        float* p = sigma_row + (kSigmaPadding * sigma_stride + kSigmaPadding);
        for (size_t ix = 0; ix < kSigmaBorder; ix++) {
          *(p - 1 - ix) = p[ix];
        }
      }
      // Right padding with mirroring.
      if (bx + 1 == frame_dim.xsize_blocks) {
        float* p = sigma_row + (kSigmaPadding * sigma_stride + kSigmaPadding +
                                frame_dim.xsize_blocks);
        for (size_t ix = 0; ix < kSigmaBorder; ix++) {
          p[ix] = *(p - 1 - ix);
        }
      }
      // Offsets for row copying, in blocks.
      size_t offset_before = kSigmaPadding - (bx == 0 ? kSigmaBorder : 0);
      size_t offset_after =
          kSigmaPadding + bx + 1 +
          (bx + 1 == frame_dim.xsize_blocks ? kSigmaBorder : 0);
      size_t num = offset_after - offset_before;
      // Above
      if (by == 0) {
        for (size_t iy = 0; iy < kSigmaBorder; iy++) {
          memcpy(
              sigma_row + offset_before +
                  (kSigmaPadding - 1 - iy) * sigma_stride,
              sigma_row + offset_before + (kSigmaPadding * iy) * sigma_stride,
              num * sizeof(*sigma_row));
        }
      }
      // Below
      if (by + 1 == frame_dim.ysize_blocks) {
        for (size_t iy = 0; iy < kSigmaBorder; iy++) {
          // sigma_row points to row "by" so we need to move one more row down
          // before mirroring, hence `kSigmaPadding + 1 + iy` is the destination
          // row.
          memcpy(
              sigma_row + offset_before +
                  (kSigmaPadding + 1 + iy) * sigma_stride,
              sigma_row + offset_before + (kSigmaPadding - iy) * sigma_stride,
              num * sizeof(*sigma_row));
        }
      }
    }
  }
}

}  // namespace

jxl::Status RunEPF(uint32_t epf_iters, const float distance,
                   const int sharpness_parameter, jxl::CodecInOut* const io,
                   jxl::ThreadPool* const pool) {
  const jxl::ColorEncoding original_color_encoding =
      io->metadata.m.color_encoding;
  jxl::Image3F opsin(io->xsize(), io->ysize());
  (void)ToXYB(io->Main(), pool, &opsin);

  JXL_CHECK(io->metadata.size.Set(opsin.xsize(), opsin.ysize()));

  opsin = PadImageToMultiple(opsin, jxl::kBlockDim);

  jxl::FrameHeader frame_header(&io->metadata);
  frame_header.loop_filter.gab = false;
  frame_header.loop_filter.epf_iters = epf_iters;
  jxl::FrameDimensions frame_dim = frame_header.ToFrameDimensions();
  const jxl::LoopFilter& lf = frame_header.loop_filter;
  frame_header.color_transform = jxl::ColorTransform::kXYB;

  static constexpr float kAcQuant = 0.84f;
  const float dc_quant = jxl::InitialQuantDC(distance);
  const float ac_quant = kAcQuant / distance;
  jxl::PassesEncoderState state;
  JXL_RETURN_IF_ERROR(
      jxl::InitializePassesSharedState(frame_header, &state.shared));
  // TODO(lode): must this be a separate one, or can the frame_header from
  // above be used for this?
  jxl::FrameHeader modular_frame_header(&io->metadata);
  jxl::ModularFrameEncoder modular_frame_encoder(modular_frame_header,
                                                 jxl::CompressParams{});
  state.shared.ac_strategy.FillDCT8();
  jxl::InitializePassesEncoder(opsin, pool, &state, &modular_frame_encoder,
                               /*aux_out=*/nullptr);
  state.shared.raw_quant_field =
      jxl::ImageI(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  state.shared.quantizer.SetQuant(dc_quant, ac_quant,
                                  &state.shared.raw_quant_field);
  state.shared.dc_storage = ComputeDC(opsin);

  jxl::FilterWeights filter_weights;
  filter_weights.Init(lf, frame_dim);
  ComputeSigma(&filter_weights.sigma, opsin, sharpness_parameter, frame_dim,
               &state, lf);

  const jxl::Image3F padded = PadImageMirror(opsin, jxl::kMaxFilterPadding, 0);

  jxl::Rect image_rect(0, 0, frame_dim.xsize_padded, frame_dim.ysize_padded);

  EdgePreservingFilter(lf, filter_weights, image_rect, padded, &opsin);

  opsin.ShrinkTo(frame_dim.xsize, frame_dim.xsize);
  jxl::OpsinParams opsin_params;
  opsin_params.Init(jxl::kDefaultIntensityTarget);
  jxl::OpsinToLinearInplace(&opsin, pool, opsin_params);
  io->Main().SetFromImage(std::move(opsin),
                          jxl::ColorEncoding::LinearSRGB(io->Main().IsGray()));
  JXL_RETURN_IF_ERROR(io->TransformTo(original_color_encoding, pool));
  return true;
}

}  // namespace tools
}  // namespace jpegxl
