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

#include <string>
#include <vector>

#include "jxl/base/data_parallel.h"
#include "jxl/modular/image/image.h"

namespace jxl {

// Lossless YCoCg color transformation
#define TRANSFORM_YCoCg 0

// G, R-G, B-G and variants
#define TRANSFORM_RCT 1

// Color palette. Parameters are: [begin_c] [end_c] [nb_colors]
#define TRANSFORM_PALETTE 2

// JPEG-style (chroma) subsampling. Parameters are: [nb_subsampled_channels],
// [channel], [sample_ratio_h], [sample_ratio_v], ... e.g. 2, 1, 2, 2, 2, 2, 2
// corresponds to 4:2:0
#define TRANSFORM_ChromaSubsample 3

// Squeezing (Haar-style)
#define TRANSFORM_SQUEEZE 4

// JPEG-style quantization. Parameters are quantization factors for each channel
// (encoded as part of channel metadata).
#define TRANSFORM_QUANTIZE 5

// this is lossy preprocessing, doesn't have an inverse transform and doesn't
// exist from the decoder point of view
#define TRANSFORM_NEAR_LOSSLESS 6

extern const std::vector<std::string> transform_name;

class Transform {
 public:
  const int ID;
  std::vector<int> parameters;

  explicit Transform(int id) : ID(id) {}
  bool apply(Image &input, bool inverse, jxl::ThreadPool *pool = nullptr);
  void meta_apply(Image &input);
  const char *name() const { return transform_name[ID].c_str(); }
};

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_TRANSFORM_H_
