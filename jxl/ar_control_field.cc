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
  JXL_ASSERT(epf_sharpness->xsize() == xsize_blocks &&
             epf_sharpness->ysize() == ysize_blocks);

  // TODO(veluca): choose a speed tier at which the heuristics here should be
  // enabled, once they are enabled at all.

  FillImage(uint8_t(4), epf_sharpness);
  return;
  // TODO(veluca) - the following code ends up returning only 7.

  constexpr float kChannelWeights[3] = {50.0f, 2.0f, 0.4f};
  const size_t sharpness_stride =
      static_cast<size_t>(epf_sharpness->PixelsPerRow());
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());

  const float* JXL_RESTRICT in_row[3] = {opsin.ConstPlaneRow(0, 0),
                                         opsin.ConstPlaneRow(1, 0),
                                         opsin.ConstPlaneRow(2, 0)};

  const auto process_row = [&](size_t by, int _) {
    AcStrategyRow acs_row = enc_state->shared.ac_strategy.ConstRow(by);
    uint8_t* JXL_RESTRICT out_row = epf_sharpness->Row(by);
    for (size_t bx = 0; bx < xsize_blocks; bx++) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      // Find the max Laplacian in a block (for example 32x32 dct).
      // Then find max Laplacian in two 4x4 pixel grids, offset by 2,2 pixels.
      // The 1st max indicates amplitude of the artefacts.
      // The 4x4 pixel grid Laplacian indicates masking of the artefacts.
      // If masking is high or amplitude is low, then no smoothing is needed.
      float maxval = 0;
      float maxarray[64] = {0};
      float maxarray2[81] = {0};
      float total_abs = 0;
      float total_2 = 0;
      for (size_t iy = 0; iy < acs.covered_blocks_y() * N; iy++) {
        const size_t cy = by * N + iy;
        const size_t prevY = cy >= 1 ? cy - 1 : cy;
        const size_t nextY = cy + 1 < opsin.ysize() ? cy + 1 : cy;
        for (size_t ix = 0; ix < acs.covered_blocks_x() * N; ix++) {
          const size_t cx = bx * N + ix;
          const size_t prevX = cx >= 1 ? cx - 1 : cx;
          const size_t nextX = cx + 1 < opsin.xsize() ? cx + 1 : cx;
          float sum = 0;
          for (size_t c = 0; c < 3; c++) {
            float v = in_row[c][cy * opsin_stride + cx] -
                      0.125f * (in_row[c][cy * opsin_stride + prevX] +
                                in_row[c][cy * opsin_stride + nextX] +
                                in_row[c][nextY * opsin_stride + prevX] +
                                in_row[c][nextY * opsin_stride + cx] +
                                in_row[c][nextY * opsin_stride + nextX] +
                                in_row[c][prevY * opsin_stride + prevX] +
                                in_row[c][prevY * opsin_stride + cx] +
                                in_row[c][prevY * opsin_stride + nextX]);
            v *= kChannelWeights[c];
            sum += std::abs(v);
          }
          maxarray[(iy / 4) * 8 + (ix / 4)] += sum * (1.0f / 16);
          maxarray2[((iy + 2) / 4) * 9 + ((ix + 2) / 4)] += sum * (1.0f / 16);
          total_2 += sum * sum *
                     (1.0f / (N * N * acs.covered_blocks_x() *
                              acs.covered_blocks_y()));
          total_abs += sum * (1.0f / (N * N * acs.covered_blocks_x() *
                                      acs.covered_blocks_y()));
        }
      }
      maxval = std::sqrt(total_2);
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          // Four 4x4 blocks for masking estimation.
          const float minval = (maxarray[(2 * iy) * 8 + 2 * ix] +
                                maxarray[(2 * iy) * 8 + 2 * ix + 1] +
                                maxarray[(2 * iy + 1) * 8 + 2 * ix] +
                                maxarray[(2 * iy + 1) * 8 + 2 * ix + 1] +
                                maxarray2[(2 * iy + 1) * 9 + 2 * (ix + 1)]) *
                               0.2f;
          // Larger kBias, less smoothing for low intensity changes.
          float kBias = 0.0005f;
          float delta = (maxval + kBias) / (minval + kBias);
          int sharpness1 = 7 + (delta - 0.5f) * 0.0;
          int sharpness2 = 7;  // maxval * 10;
          out_row[bx + sharpness_stride * iy + ix] =
              std::max(0, std::min(7, std::min(sharpness1, sharpness2)));
        }
      }
    }
  };

  RunOnPool(pool, 0, ysize_blocks, ThreadPool::SkipInit(), process_row,
            "AR CF");
}

}  // namespace jxl
