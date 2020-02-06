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

namespace {
const char *transform_name[static_cast<uint32_t>(TransformId::kNumTransforms)] =
    {"YCoCg",   "RCT",          "Palette",      "Subsample",
     "Squeeze", "Quantization", "Near-Lossless"};
}  // namespace

const char *Transform::Name() const {
  if (IsValid()) return transform_name[static_cast<uint32_t>(id)];
  return "Unknown transformation";
}

Status Transform::Apply(Image &input, bool inverse, ThreadPool *pool) {
  switch (id) {
    case TransformId::kChromaSubsample:
      return subsample(input, inverse, parameters);
    case TransformId::kQuantize:
      return quantize(input, inverse, parameters, pool);
    case TransformId::kYCoCg:
      return YCoCg(input, inverse, pool);
    case TransformId::kRCT:
      return subtract_green(input, inverse, parameters);
    case TransformId::kSqueeze:
      return squeeze(input, inverse, parameters, pool);
    case TransformId::kPalette:
      return palette(input, inverse, parameters, pool);
    case TransformId::kNearLossless:
      return near_lossless(input, inverse, parameters);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)", id);
  }
}

Status Transform::MetaApply(Image &input) {
  switch (id) {
    case TransformId::kYCoCg:
      return true;
    case TransformId::kChromaSubsample:
      return meta_subsample(input, parameters);
    case TransformId::kQuantize:
      return meta_quantize(input);
    case TransformId::kRCT:
      return true;
    case TransformId::kSqueeze:
      return meta_squeeze(input, &parameters);
    case TransformId::kPalette:
      return meta_palette(input, parameters);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)", id);
  }
}

}  // namespace jxl
