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

#include "jxl/fields.h"
#include "jxl/modular/image/image.h"
#include "jxl/modular/transform/near-lossless.h"
#include "jxl/modular/transform/palette.h"
#include "jxl/modular/transform/quantize.h"
#include "jxl/modular/transform/squeeze.h"
#include "jxl/modular/transform/subtractgreen.h"

namespace jxl {

namespace {
const char *transform_name[static_cast<uint32_t>(TransformId::kNumTransforms)] =
    {"RCT", "Palette", "Squeeze", "Quantization", "Near-Lossless"};
}  // namespace

SqueezeParams::SqueezeParams() { Bundle::Init(this); }
Transform::Transform(TransformId id) {
  Bundle::Init(this);
  this->id = id;
}

const char *Transform::TransformName() const {
  return transform_name[static_cast<uint32_t>(id)];
}

Status Transform::Forward(Image &input, ThreadPool *pool) {
  switch (id) {
    case TransformId::kQuantize:
      return FwdQuantize(input, nonserialized_quant_factors);
    case TransformId::kRCT:
      return FwdSubtractGreen(input, begin_c, rct_type);
    case TransformId::kSqueeze:
      return FwdSqueeze(input, squeezes, pool);
    case TransformId::kPalette:
      return FwdPalette(input, begin_c, begin_c + num_c - 1, nb_colors,
                        ordered_palette);
    case TransformId::kNearLossless:
      return FwdNearLossless(input, begin_c, begin_c + num_c - 1,
                             max_delta_error);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)", id);
  }
}

Status Transform::Inverse(Image &input, ThreadPool *pool) {
  switch (id) {
    case TransformId::kQuantize:
      return InvQuantize(input, pool);
    case TransformId::kRCT:
      return InvSubtractGreen(input, begin_c, rct_type);
    case TransformId::kSqueeze:
      return InvSqueeze(input, squeezes, pool);
    case TransformId::kPalette:
      return InvPalette(input, begin_c, nb_colors, pool);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)", id);
  }
}

Status Transform::MetaApply(Image &input) {
  switch (id) {
    case TransformId::kQuantize:
      return MetaQuantize(input);
    case TransformId::kRCT:
      return true;
    case TransformId::kSqueeze:
      return MetaSqueeze(input, &squeezes);
    case TransformId::kPalette:
      return MetaPalette(input, begin_c, begin_c + num_c - 1, nb_colors);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)", id);
  }
}

}  // namespace jxl
