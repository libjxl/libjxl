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

#include "jxl/filters.h"

namespace jxl {

void FilterWeights::Init(const LoopFilter& lf,
                         const FrameDimensions& frame_dim) {
  if (lf.epf_iters > 0) {
    sigma = ImageF(frame_dim.xsize_blocks + 4, frame_dim.ysize_blocks + 4);
  }
  if (lf.gab) {
    GaborishWeights(lf);
  }
}

void FilterWeights::GaborishWeights(const LoopFilter& lf) {
  gab_weights[0] = 1;
  gab_weights[1] = lf.gab_x_weight1;
  gab_weights[2] = lf.gab_x_weight2;
  gab_weights[3] = 1;
  gab_weights[4] = lf.gab_y_weight1;
  gab_weights[5] = lf.gab_y_weight2;
  gab_weights[6] = 1;
  gab_weights[7] = lf.gab_b_weight1;
  gab_weights[8] = lf.gab_b_weight2;
  // Normalize
  for (size_t c = 0; c < 3; c++) {
    const float mul =
        1.0f / (gab_weights[3 * c] +
                4 * (gab_weights[3 * c + 1] + gab_weights[3 * c + 2]));
    gab_weights[3 * c] *= mul;
    gab_weights[3 * c + 1] *= mul;
    gab_weights[3 * c + 2] *= mul;
  }
}

}  // namespace jxl
