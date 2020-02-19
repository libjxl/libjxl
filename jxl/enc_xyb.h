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

#ifndef JXL_ENC_XYB_H_
#define JXL_ENC_XYB_H_

// Converts to XYB color space.

#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"

namespace jxl {

// Converts any color space to XYB. Copies internally to linear sRGB and
// multiplies that by `linear_multiplier`.
// Returns pointer to an ImageBundle in linear sRGB: either `in` if that is
// already in linear sRGB, in which case linear_storage is ignored; otherwise
// `*linear_storage` (default-constructed is sufficient). Note that even if
// the caller doesn't need the linear sRGB, it is still used internally.
// Runtime dispatch.
const ImageBundle* ToXYB(const ImageBundle& in, float linear_multiplier,
                         ThreadPool* pool, Image3F* JXL_RESTRICT xyb,
                         ImageBundle* JXL_RESTRICT linear_storage);

// DEPRECATED, used by opsin_image_wrapper.
Image3F OpsinDynamicsImage(const Image3B& srgb8);

// For opsin_image_test.
void TestCubeRoot();

}  // namespace jxl

#endif  // JXL_ENC_XYB_H_
