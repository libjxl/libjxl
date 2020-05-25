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

#ifndef JXL_ENC_TRANSFORMS_H_
#define JXL_ENC_TRANSFORMS_H_

// Facade for (non-inlined) integral transforms.

#include <stddef.h>
#include <stdint.h>

#include "jxl/ac_strategy.h"
#include "jxl/base/compiler_specific.h"

namespace jxl {

typedef void TransformFromPixelsFunc(const AcStrategy::Type strategy,
                                     const float* JXL_RESTRICT pixels,
                                     size_t pixels_stride,
                                     float* JXL_RESTRICT coefficients);
TransformFromPixelsFunc* ChooseTransformFromPixels();

// Equivalent of the above for DC image.
typedef void LowestFrequenciesFromDCFunc(const jxl::AcStrategy::Type strategy,
                                         const float* dc, size_t dc_stride,
                                         float* llf);
LowestFrequenciesFromDCFunc* ChooseLowestFrequenciesFromDC();

typedef void AFVDCT4x4Func(const float* JXL_RESTRICT pixels,
                           float* JXL_RESTRICT coeffs);
AFVDCT4x4Func* ChooseAFVDCT4x4();

}  // namespace jxl

#endif  // JXL_ENC_TRANSFORMS_H_
