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
#include "jxl/fields.h"
#include "jxl/modular/image/image.h"
#include "jxl/modular/options.h"

namespace jxl {

enum class TransformId : uint32_t {
  // G, R-G, B-G and variants (including YCoCg).
  kRCT = 0,

  // Color palette. Parameters are: [begin_c] [end_c] [nb_colors]
  kPalette = 1,

  // Squeezing (Haar-style)
  kSqueeze = 2,

  // JPEG-style quantization. Parameters are quantization factors for each
  // channel
  // (encoded as part of channel metadata).
  kQuantize = 3,

  // this is lossy preprocessing, doesn't have an inverse transform and doesn't
  // exist from the decoder point of view
  kNearLossless = 4,

  // The total number of transforms. Update this if adding more transformations.
  kNumTransforms = 5,
};

struct SqueezeParams {
  static const char *Name() { return "SqueezeParams"; }
  bool horizontal;
  bool in_place;
  bool subsample_mode;
  uint32_t begin_c;
  uint32_t num_c;
  SqueezeParams();
  template <class Visitor>
  Status VisitFields(Visitor *JXL_RESTRICT visitor) {
    visitor->Bool(false, &horizontal);
    visitor->Bool(false, &in_place);
    visitor->Bool(false, &subsample_mode);
    visitor->U32(Bits(3), BitsOffset(6, 8), BitsOffset(10, 72),
                 BitsOffset(13, 1096), 0, &begin_c);
    visitor->U32(Val(1), Val(2), Val(3), BitsOffset(4, 4), 2, &num_c);
    return true;
  }
};

class Transform {
 public:
  TransformId id;
  // for Palette and RCT.
  uint32_t begin_c;
  // for RCT. 42 possible values starting from 0.
  uint32_t rct_type;
  // Only for Palette and NearLossless.
  uint32_t num_c;
  // Only for Palette.
  uint32_t nb_colors;
  // for Squeeze. Default squeeze if empty.
  std::vector<SqueezeParams> squeezes;
  // for Quantize, not serialized.
  std::vector<int> nonserialized_quant_factors;
  // for NearLossless, not serialized.
  int max_delta_error;
  Predictor predictor;
  // for Palette, not serialized.
  bool ordered_palette = true;

  explicit Transform(TransformId id);
  // default constructor for bundles.
  Transform() : Transform(TransformId::kNumTransforms) {}

  template <class Visitor>
  Status VisitFields(Visitor *JXL_RESTRICT visitor) {
    visitor->U32(
        Val((uint32_t)TransformId::kRCT), Val((uint32_t)TransformId::kPalette),
        Val((uint32_t)TransformId::kSqueeze),
        Val((uint32_t)TransformId::kQuantize), (uint32_t)TransformId::kRCT,
        reinterpret_cast<uint32_t *>(&id));
    if (visitor->Conditional(id == TransformId::kRCT ||
                             id == TransformId::kPalette)) {
      visitor->U32(Bits(3), BitsOffset(6, 8), BitsOffset(10, 72),
                   BitsOffset(13, 1096), 0, &begin_c);
    }
    if (visitor->Conditional(id == TransformId::kRCT)) {
      // 0-41, default YCoCg.
      visitor->U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10), 6,
                   &rct_type);
    }
    if (visitor->Conditional(id == TransformId::kPalette)) {
      visitor->U32(Val(1), Val(3), Val(4), BitsOffset(13, 1), 3, &num_c);
      visitor->U32(BitsOffset(8, 1), BitsOffset(10, 257), BitsOffset(12, 1281),
                   BitsOffset(16, 5377), 256, &nb_colors);
    }

    if (visitor->Conditional(id == TransformId::kSqueeze)) {
      uint32_t num_squeezes = squeezes.size();
      visitor->U32(Val(0), BitsOffset(4, 1), BitsOffset(6, 9),
                   BitsOffset(8, 41), 0, &num_squeezes);
      if (visitor->IsReading()) squeezes.resize(num_squeezes);
      for (size_t i = 0; i < num_squeezes; i++) {
        JXL_RETURN_IF_ERROR(visitor->VisitNested(&squeezes[i]));
      }
    }
    return true;
  }

  static const char *Name() { return "Transform"; }

  // Returns the name of the transform.
  const char *TransformName() const;

  Status Forward(Image &input, ThreadPool *pool = nullptr);
  Status Inverse(Image &input, ThreadPool *pool = nullptr);
  Status MetaApply(Image &input);
};

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_TRANSFORM_H_
