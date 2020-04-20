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

// Demo of functions that might be called from multiple SIMD modules (either
// other -inl.h files, or a .cc file between begin/end_target-inl). This is
// optional - all SIMD code can reside in .cc files. However, this allows
// splitting code into different files while still inlining instead of requiring
// calling through function pointers.

// Include guard (still compiled once per target)
#if defined(HWY_EXAMPLES_SKELETON_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWY_EXAMPLES_SKELETON_INL_H_
#undef HWY_EXAMPLES_SKELETON_INL_H_
#else
#define HWY_EXAMPLES_SKELETON_INL_H_
#endif

// It is fine to #include normal or *-inl headers.
#include <stddef.h>
#include "hwy/examples/skeleton_shared.h"
#include "hwy/highway.h"

namespace skeleton {

#include "hwy/begin_target-inl.h"

// Computes out[i] = in1[i] * kMultiplier + in2[i] for i < 256.
HWY_ATTR HWY_MAYBE_UNUSED void ExampleMulAdd(const float* HWY_RESTRICT in1,
                                             const float* HWY_RESTRICT in2,
                                             float* HWY_RESTRICT out) {
  // Descriptor(s) for all vector types used in this function.
  HWY_FULL(float) df;

  const auto mul = Set(df, kMultiplier);
  for (size_t i = 0; i < 256; i += df.N) {
    const auto result = MulAdd(mul, Load(df, in1 + i), Load(df, in2 + i));
    Store(result, df, out + i);
  }
}

// (This doesn't generate SIMD instructions, so HWY_ATTR is not required here)
HWY_ATTR HWY_MAYBE_UNUSED const char* ExampleGatherStrategy() {
  // Highway functions generate per-target implementations from the same source
  // code, but if needed, differing codepaths can be selected via #if.
#if HWY_CAPS & HWY_CAP_GATHER
  return "Has gather";
#else
  return "No gather, use scalar instead?";
#endif
}

#include "hwy/end_target-inl.h"

}  // namespace skeleton

#endif  // include guard
