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

#ifndef JXL_MODULAR_TRANSFORM_TRANSFORM_H_
#define JXL_MODULAR_TRANSFORM_TRANSFORM_H_

#include <cstdint>
#include <string>
#include <vector>

#include "jxl/base/data_parallel.h"
#include "jxl/modular/image/image.h"

namespace jxl {

enum class TransformId : uint32_t {
  // Lossless YCoCg color transformation
  kYCoCg = 0,

  // G, R-G, B-G and variants
  kRCT = 1,

  // Color palette. Parameters are: [begin_c] [end_c] [nb_colors]
  kPalette = 2,

  // JPEG-style (chroma) subsampling. Parameters are: [nb_subsampled_channels],
  // [channel], [sample_ratio_h], [sample_ratio_v], ... e.g. 2, 1, 2, 2, 2, 2, 2
  // corresponds to 4:2:0
  kChromaSubsample = 3,

  // Squeezing (Haar-style)
  kSqueeze = 4,

  // JPEG-style quantization. Parameters are quantization factors for each
  // channel
  // (encoded as part of channel metadata).
  kQuantize = 5,

  // this is lossy preprocessing, doesn't have an inverse transform and doesn't
  // exist from the decoder point of view
  kNearLossless = 6,

  // The total number of transforms. Update this if adding more transformations.
  kNumTransforms = 7,
};

typedef std::vector<int> TransformParams;

class Transform {
 public:
  const TransformId id;
  TransformParams parameters;

  explicit Transform(TransformId id) : id(id) {}

  bool IsValid() const {
    return static_cast<uint32_t>(id) <
           static_cast<uint32_t>(TransformId::kNumTransforms);
  }

  // Return the name of the transform (if valid).
  const char *Name() const;

  Status Forward(Image &input, ThreadPool *pool = nullptr);
  Status Inverse(Image &input, ThreadPool *pool = nullptr);
  Status MetaApply(Image &input);
};

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_TRANSFORM_H_
