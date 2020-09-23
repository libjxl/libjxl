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

#include "jxl/ar_control_field.h"

#include <stdint.h>
#include <stdlib.h>

#include <algorithm>

#include "jxl/ac_strategy.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/common.h"
#include "jxl/enc_adaptive_quantization.h"
#include "jxl/enc_params.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"

namespace jxl {

void FindBestArControlField(const Image3F& opsin, PassesEncoderState* enc_state,
                            ThreadPool* pool) {
  constexpr size_t N = kBlockDim;
  const size_t xsize_blocks = enc_state->shared.frame_dim.xsize_blocks;
  const size_t ysize_blocks = enc_state->shared.frame_dim.ysize_blocks;
  ImageB* JXL_RESTRICT epf_sharpness = &enc_state->shared.epf_sharpness;
  ImageF* JXL_RESTRICT quant = &enc_state->initial_quant_field;
  float default_quant = enc_state->cparams.butteraugli_distance;
  JXL_ASSERT(epf_sharpness->xsize() == xsize_blocks &&
             epf_sharpness->ysize() == ysize_blocks);

  if (enc_state->cparams.speed_tier > SpeedTier::kWombat ||
      !enc_state->shared.image_features.loop_filter.epf) {
    FillImage(static_cast<uint8_t>(4), epf_sharpness);
    return;
  }

  // Likely better to have a higher X weight, like:
  // const float kChannelWeights[3] = {47.0f, 4.35f, 0.287f};
  const float kChannelWeights[3] = {4.35f, 4.35f, 0.287f};
  const size_t sharpness_stride =
      static_cast<size_t>(epf_sharpness->PixelsPerRow());
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());

  const float* JXL_RESTRICT in_row[3] = {opsin.ConstPlaneRow(0, 0),
                                         opsin.ConstPlaneRow(1, 0),
                                         opsin.ConstPlaneRow(2, 0)};
  const auto process_row = [&](size_t by, int _) {
    AcStrategyRow acs_row = enc_state->shared.ac_strategy.ConstRow(by);
    uint8_t* JXL_RESTRICT out_row = epf_sharpness->Row(by);
    float* JXL_RESTRICT quant_row = nullptr;
    if (by < quant->ysize()) {
      quant_row = quant->Row(by);
    }
    for (size_t bx = 0; bx < xsize_blocks; bx++) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      // Calculate the L2 of the 3x3 Laplacian in an integral transform
      // (for example 32x32 dct). This relates to transforms ability
      // to propagate artefacts.
      //
      // Calculate the L2 of the 3x3 Laplacian in 4x4 blocks within the area
      // of the integral transform. Sample them within the integral transform
      // with two offsets (0,0) and (-2, -2) pixels (sqrsum_00 and sqrsum_22,
      //  respectively).
      //
      // If masking is high or amplitude of the artefacts is low, then no
      // smoothing is needed.
      float sqrsum_integral_transform = 0;
      float sqrsum_00[64] = {0};
      float sqrsum_22[81] = {0};
      int sqrsum_22_popcount[81] = {0};

      // The errors are going to be linear to the quantization value in this
      // locality. We only have access to the initial quant field here.
      float quant_val = default_quant;
      if (quant_row != nullptr && bx < quant->xsize()) {
        quant_val = 1.0f / quant_row[bx];
      }
      // Indexing iy and ix is a bit tricky as we include a 2 pixel border
      // around the block for evenness calculations. This is similar to what we
      // did in guetzli for the observability of artefacts, except there the
      // element is a sliding 5x5, not sparcely sampled 4x4 box like here.
      for (size_t iy = 0; iy < acs.covered_blocks_y() * N + 4; iy++) {
        size_t cy = by * N + iy;
        cy -= 2;
        if (cy >= opsin.ysize()) {
          continue;
        }
        const size_t prevY = cy >= 1 ? cy - 1 : cy;
        const size_t nextY = cy + 1 < opsin.ysize() ? cy + 1 : cy;
        for (size_t ix = 0; ix < acs.covered_blocks_x() * N + 4; ix++) {
          size_t cx = bx * N + ix;
          cx -= 2;
          if (cx >= opsin.xsize()) {
            continue;
          }
          const size_t prevX = cx >= 1 ? cx - 1 : cx;
          const size_t nextX = cx + 1 < opsin.xsize() ? cx + 1 : cx;
          float sumsqr = 0;
          for (size_t c = 0; c < 3; c++) {
            float laplacian =
                in_row[c][cy * opsin_stride + cx] -
                0.125f * (in_row[c][cy * opsin_stride + prevX] +
                          in_row[c][cy * opsin_stride + nextX] +
                          in_row[c][nextY * opsin_stride + prevX] +
                          in_row[c][nextY * opsin_stride + cx] +
                          in_row[c][nextY * opsin_stride + nextX] +
                          in_row[c][prevY * opsin_stride + prevX] +
                          in_row[c][prevY * opsin_stride + cx] +
                          in_row[c][prevY * opsin_stride + nextX]);
            laplacian *= kChannelWeights[c];
            sumsqr += laplacian * laplacian;
          }
          sqrsum_22[(iy / 4) * 9 + (ix / 4)] += sumsqr;
          sqrsum_22_popcount[(iy / 4) * 9 + (ix / 4)]++;
          if (iy >= 2 && ix >= 2 && iy < N * acs.covered_blocks_y() + 2 &&
              ix < N * acs.covered_blocks_x() + 2) {
            sqrsum_00[((iy - 2) / 4) * 8 + ((ix - 2) / 4)] += sumsqr;
            sqrsum_integral_transform += sumsqr;
          }
        }
      }
      for (int ii = 0; ii < 64; ++ii) {
        sqrsum_00[ii] *= 1.0f / 16;
        sqrsum_00[ii] = std::sqrt(sqrsum_00[ii]);
      }
      for (int ii = 0; ii < 81; ++ii) {
        if (sqrsum_22_popcount[ii] != 0) {
          sqrsum_22[ii] *= 1.0f / sqrsum_22_popcount[ii];
          sqrsum_22[ii] = std::sqrt(sqrsum_22[ii]);
        }
      }
      sqrsum_integral_transform /=
          N * N * acs.covered_blocks_x() * acs.covered_blocks_y();
      sqrsum_integral_transform = std::sqrt(sqrsum_integral_transform);
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          // Five 4x4 blocks for masking estimation, all within the
          // 8x8 area.
          float minval = sqrsum_00[(2 * iy) * 8 + 2 * ix];
          minval = std::min(minval, sqrsum_00[(2 * iy) * 8 + 2 * ix + 1]);
          minval = std::min(minval, sqrsum_00[(2 * iy + 1) * 8 + 2 * ix]);
          minval = std::min(minval, sqrsum_00[(2 * iy + 1) * 8 + 2 * ix + 1]);
          minval = std::min(minval, sqrsum_22[(2 * iy + 1) * 9 + 2 * ix + 1]);
          // Nine more 4x4 blocks for masking estimation, includes
          // the 2 pixel area around the 8x8 block being controlled.
          float minval2 = sqrsum_22[(2 * iy + 0) * 9 + 2 * ix + 0];
          minval2 = std::min(minval2, sqrsum_22[(2 * iy + 0) * 9 + 2 * ix + 1]);
          minval2 = std::min(minval2, sqrsum_22[(2 * iy + 0) * 9 + 2 * ix + 2]);
          minval2 = std::min(minval2, sqrsum_22[(2 * iy + 1) * 9 + 2 * ix + 0]);
          minval2 = std::min(minval2, sqrsum_22[(2 * iy + 1) * 9 + 2 * ix + 1]);
          minval2 = std::min(minval2, sqrsum_22[(2 * iy + 1) * 9 + 2 * ix + 2]);
          minval2 = std::min(minval2, sqrsum_22[(2 * iy + 2) * 9 + 2 * ix + 0]);
          minval2 = std::min(minval2, sqrsum_22[(2 * iy + 2) * 9 + 2 * ix + 1]);
          minval2 = std::min(minval2, sqrsum_22[(2 * iy + 2) * 9 + 2 * ix + 2]);
          float minval3 = std::min(minval, minval2);
          minval *= 0.125f;
          minval += 0.5f * minval3;
          minval += 0.25f * sqrsum_22[(2 * iy + 1) * 9 + 2 * (ix + 1)];
          minval += 0.125f * minval2;
          // Larger kBias, less smoothing for low intensity changes.
          float kBias = 0.015f * quant_val;
          float delta = (sqrsum_integral_transform + kBias) / (minval + kBias);

          int out = 4;
          if (delta > 1.75f) {
            out = 4;  // smooth
          } else {
            out = 0;
          }
          const float kSmoothLimit = 0.3f;
          if (minval < kSmoothLimit * kBias) {
            out = 4;
          }
          const float kSmoothLimit2 = 0.28f;
          if (minval2 < kSmoothLimit2 * kBias) {
            out = 4;
          }
          out_row[bx + sharpness_stride * iy + ix] = out;
        }
      }
    }
  };

  RunOnPool(pool, 0, ysize_blocks, ThreadPool::SkipInit(), process_row,
            "AR CF");
}

}  // namespace jxl
