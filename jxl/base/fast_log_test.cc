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

#include "jxl/base/fast_log.h"

#include <random>

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/base/fast_log_test.cc"
#define HWY_USE_GTEST
#endif
#include <hwy/foreach_target.h>
#include <hwy/tests/test_util.h>

// After foreach_target
#include "jxl/fast_log-inl.h"

namespace jxl {

#include <hwy/tests/test_util-inl.h>

#include <hwy/begin_target-inl.h>

HWY_NOINLINE HWY_ATTR void TestFastLog12() {
  constexpr size_t kNumTrials = 1 << 23;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(1e-7f, 1e3f);
  float max_abs_err = 0;
  HWY_FULL(float) d;
  for (size_t i = 0; i < kNumTrials; i++) {
    const float f = dist(rng);
    const float actual = GetLane(FastLog2f_18bits(Set(d, f)));
    const float abs_err = std::abs(std::log2(f) - actual);
    EXPECT_LT(abs_err, 2.9E-6) << "f = " << f;
    max_abs_err = std::max(max_abs_err, abs_err);
  }
  printf("18: max abs err %e\n", static_cast<double>(max_abs_err));
}

#include <hwy/end_target-inl.h>

#if HWY_ONCE

HWY_EXPORT(TestFastLog12)

TEST(FastLogTest, TestFastLog) {
  constexpr size_t kNumTrials = 1 << 23;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(1e-7f, 1e3f);
  float max_abs_err = 0;
  for (size_t i = 0; i < kNumTrials; i++) {
    const float f = dist(rng);
    const float abs_err = std::abs(std::log2(f) - FastLog2f(f));
    EXPECT_LT(abs_err, 9.1E-3) << "f = " << f;
    max_abs_err = std::max(max_abs_err, abs_err);
  }
  printf("max abs err %e\n", static_cast<double>(max_abs_err));
}

TEST(FastLogTest, Run) { hwy::RunTest(&ChooseTestFastLog12); }

#endif  // HWY_ONCE

}  // namespace jxl
