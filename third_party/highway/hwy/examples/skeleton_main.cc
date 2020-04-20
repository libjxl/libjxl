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

// Example of runtime dispatch to the "skeleton" module.

#include <stdio.h>
#include "hwy/examples/skeleton.h"         // ChooseSkeleton
#include "hwy/examples/skeleton_shared.h"  // kMultiplier
#include "hwy/interface.h"                 // SupportedTargets
#undef HWY_USE_GTEST
#include "hwy/tests/test_util.h"  // ChooseAndCallForeachTarget

namespace skeleton {

void Main() {
  HWY_ALIGN_MAX float in1[256];
  HWY_ALIGN_MAX float in2[256];
  HWY_ALIGN_MAX float out[256];
  for (size_t i = 0; i < 256; ++i) {
    in1[i] = static_cast<float>(i);
    in2[i] = in1[i] + 300;
  }

  const uint32_t targets_bits = hwy::SupportedTargets();
  SkeletonFunc* best = ChooseSkeleton(targets_bits);
  (*best)(in1, in2, out);
  printf("Should be %.2f: %.2f\n", in1[255] * kMultiplier + in2[255], out[255]);

  // Tests would typically run for all targets to ensure all are OK.
  printf("\nNow running for all targets:\n\n");
  hwy::ChooseAndCallForeachTarget(&ChooseSkeleton, in1, in2, out);
}

}  // namespace skeleton

int main() {
  skeleton::Main();
  return 0;
}
