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

#ifndef TOOLS_BUTTERAUGLI_PNORM_H_
#define TOOLS_BUTTERAUGLI_PNORM_H_

#include <stdint.h>
#include "jxl/image_bundle.h"

namespace jxl {

// Computes p-norm given the butteraugli distmap.
typedef double ComputeDistancePFunc(const ImageF& distmap, double p);
ComputeDistancePFunc* ChooseComputeDistanceP(uint32_t targets_bits);

typedef double ComputeDistance2Func(const ImageBundle& ib1,
                                    const ImageBundle& ib2);
ComputeDistance2Func* ChooseComputeDistance2(uint32_t targets_bits);

}  // namespace jxl

#endif  // TOOLS_BUTTERAUGLI_PNORM_H_
