// Copyright 2020 Google LLC
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

#include "hwy/targets.h"

#include "hwy/tests/test_util-inl.h"

namespace hwy {

TEST(HwyTargetsTest, DisabledTargetsTest) {
  DisableTargets(~0u);
  // Check that the baseline can't be disabled.
  HWY_ASSERT(HWY_ENABLED_BASELINE == SupportedTargets());

  DisableTargets(0);  // Reset the mask.
  uint32_t current_targets = SupportedTargets();
  if ((current_targets & ~HWY_ENABLED_BASELINE) == 0) {
    // We can't test anything else if the only compiled target is the baseline.
    return;
  }
  // Get the lowest bit in the mask (the best target) and disable that one.
  uint32_t lowest_target = current_targets & (-current_targets);
  // The lowest target shouldn't be one in the baseline.
  HWY_ASSERT((lowest_target & ~HWY_ENABLED_BASELINE) != 0);
  DisableTargets(lowest_target);

  // Check that the other targets are still enabled.
  HWY_ASSERT((lowest_target ^ current_targets) == SupportedTargets());
  DisableTargets(0);  // Reset the mask.
}

}  // namespace hwy
