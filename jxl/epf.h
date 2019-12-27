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

#ifndef JXL_EPF_H_
#define JXL_EPF_H_

// Fast SIMD edge preserving filter (adaptive, nonlinear).
// "In-loop" filter: edge-preserving filter + quantization constraint.

#include <stddef.h>

#include <hwy/runtime_targets.h>

#include "jxl/base/data_parallel.h"
#include "jxl/dec_cache.h"

namespace jxl {

static_assert(kNumEpfSharpness <= LoopFilter::kEpfSharpEntries, "Mismatch");

// 4 * (sqrt(0.5)-1), so that Weight(sigma) = 0.5.
static constexpr float kInvSigmaNum = -1.1715728752538099024f;

// Applies gaborish + EPF to the given `in_rect` part of the input image `in`,
// storing the result in the `out_rect` rect of the output image and reading
// sigma values from the `sigma_rect` portion of `sigma`. `in` and `sigma` are
// assumed to be padded with two blocks worth of data on each side (with only
// the innermost 3 pixels of the padding needing to be valid), and the
// corresponding rects should ignore this padding. `storage1` should be at least
// as wide as the output rect plus two blocks of padding on each side and should
// have at least 7 rows, while `storage2` should be at least as wide as the
// output rect plus one block of padding on each side and should have at least 3
// rows.
HWY_ATTR void EdgePreservingFilter(const LoopFilter& lf, const Rect& in_rect,
                                   const Image3F& in, const Rect& sigma_rect,
                                   const ImageF& sigma, const Rect& out_rect,
                                   Image3F* JXL_RESTRICT out,
                                   Image3F* JXL_RESTRICT storage1,
                                   Image3F* JXL_RESTRICT storage2);

}  // namespace jxl

#endif  // JXL_EPF_H_
