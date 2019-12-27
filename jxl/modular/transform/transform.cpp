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

#include "jxl/modular/transform/transform.h"

#include "jxl/modular/image/image.h"

#include "jxl/modular/transform/near-lossless.h"
#include "jxl/modular/transform/palette.h"
#include "jxl/modular/transform/quantize.h"
#include "jxl/modular/transform/squeeze.h"
#include "jxl/modular/transform/subsample.h"
#include "jxl/modular/transform/subtractgreen.h"
#include "jxl/modular/transform/ycocg.h"

namespace jxl {

const std::vector<std::string> transform_name = {
    "YCoCg",   "RCT",          "Palette",      "Subsample",
    "Squeeze", "Quantization", "Near-Lossless"};

bool Transform::apply(Image &input, bool inverse, jxl::ThreadPool *pool) {
  switch (ID) {
    case TRANSFORM_ChromaSubsample:
      return subsample(input, inverse, parameters);
    case TRANSFORM_QUANTIZE:
      return quantize(input, inverse, parameters, pool);
    case TRANSFORM_YCoCg:
      return YCoCg(input, inverse, pool);
    case TRANSFORM_RCT:
      return subtract_green(input, inverse, parameters);
    case TRANSFORM_SQUEEZE:
      return squeeze(input, inverse, parameters, pool);
    case TRANSFORM_PALETTE:
      return palette(input, inverse, parameters, pool);
    case TRANSFORM_NEAR_LOSSLESS:
      return near_lossless(input, inverse, parameters);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%i)", ID);
  }
}

void Transform::meta_apply(Image &input) {
  switch (ID) {
    case TRANSFORM_YCoCg:
      return;
    case TRANSFORM_ChromaSubsample:
      meta_subsample(input, parameters);
      return;
    case TRANSFORM_QUANTIZE:
      meta_quantize(input);
      return;
    case TRANSFORM_RCT:
      return;
    case TRANSFORM_SQUEEZE:
      meta_squeeze(input, parameters);
      return;
    case TRANSFORM_PALETTE:
      meta_palette(input, parameters);
      return;
    default:
      JXL_FAILURE("Unknown transformation (ID=%i)", ID);
      return;
  }
}

}  // namespace jxl