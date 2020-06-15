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
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/base/fast_log_test.cc"
#include <hwy/foreach_target.h>

#include <random>

#include <stdio.h>
#include "jxl/fast_log-inl.h"

#include <hwy/tests/test_util-inl.h>

#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

HWY_NOINLINE void TestFastLog12() {
  constexpr size_t kNumTrials = 1 << 23;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(1e-7f, 1e3f);
  float max_abs_err = 0;
  HWY_FULL(float) d;
  for (size_t i = 0; i < kNumTrials; i++) {
    const float f = dist(rng);
    const F32xN actual_v = FastLog2f_18bits(d, Set(d, f));
    const float actual = GetLane(actual_v);
    const float abs_err = std::abs(std::log2(f) - actual);
    EXPECT_LT(abs_err, 2.9E-6) << "f = " << f;
    max_abs_err = std::max(max_abs_err, abs_err);
  }
  printf("18: max abs err %e\n", static_cast<double>(max_abs_err));
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {

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

class FastLogTargetTest : public hwy::TestWithParamTarget {};
HWY_TARGET_INSTANTIATE_TEST_SUITE_P(FastLogTargetTest);

HWY_EXPORT_AND_TEST_P(FastLogTargetTest, TestFastLog12)

}  // namespace jxl
#endif  // HWY_ONCE
