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

#ifndef JXL_FILTERS_H_
#define JXL_FILTERS_H_

#include <stddef.h>

#include "jxl/common.h"
#include "jxl/image.h"
#include "jxl/loop_filter.h"

namespace jxl {

struct FilterWeights {
  // Initialize the FilterWeights for the passed LoopFilter and FrameDimensions.
  void Init(const LoopFilter& lf, const FrameDimensions& frame_dim);

  // Normalized weights for gaborish, in XYB order, each weight for Manhattan
  // distance of 0, 1 and 2 respectively.
  float gab_weights[9];

  // Sigma values for EPF, if enabled.
  // Note that, for speed reasons, this is actually kInvSigmaNum / sigma.
  ImageF sigma;

 private:
  void GaborishWeights(const LoopFilter& lf);
};

// Line-based EPF only needs to keep in cache 13 lines of the image, so 256 is
// sufficient for everything to fit in the L2 cache.
constexpr size_t kApplyImageFeaturesTileDim = 256;

constexpr size_t kEpf1InputRows = 7;
constexpr size_t kEpf2InputRows = 3;

// Tile storage for ApplyImageFeatures steps. Storage1 has 2 blocks of padding
// per side, storage2 has 1.
struct FilterStorage {
  FilterStorage() : FilterStorage(kApplyImageFeaturesTileDim) {}
  // Since we use row-based processing and cyclic addressing, we only need 7
  // rows in storage1 and 3 in storage2.
  explicit FilterStorage(size_t xsize)
      : storage1{xsize + 4 * kBlockDim, kEpf1InputRows},
        storage2{xsize + 2 * kBlockDim, kEpf2InputRows} {}

  FilterStorage(const FilterStorage&) = delete;
  FilterStorage(FilterStorage&&) = default;

  Image3F storage1;
  Image3F storage2;
};

}  // namespace jxl

#endif  // JXL_FILTERS_H_
