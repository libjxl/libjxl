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

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/base/fast_log_test.cc"

#define HWY_USE_GTEST
#include <hwy/tests/test_util.h>

struct FastLog12Test {
  HWY_DECLARE(void, ())
};
TEST(FastLog12Test, Run) { hwy::RunTests<FastLog12Test>(); }

#include <hwy/tests/test_target_util.h>
#include <random>

#include "jxl/base/fast_log.h"

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

}  // namespace jxl

#endif  // HWY_TARGET_INCLUDE

namespace jxl {
namespace HWY_NAMESPACE {
namespace {

#include <fast_log-inl.h>

HWY_NOINLINE HWY_ATTR void CheckFastLog12() {
  constexpr size_t kNumTrials = 1 << 23;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(1e-7f, 1e3f);
  float max_abs_err = 0;
  HWY_FULL(float) d;
  for (size_t i = 0; i < kNumTrials; i++) {
    const float f = dist(rng);
    const float actual = GetLane(FastLog2f_12bits(Set(d, f)));
    const float abs_err = std::abs(std::log2(f) - actual);
    EXPECT_LT(abs_err, 1.6E-4) << "f = " << f;
    max_abs_err = std::max(max_abs_err, abs_err);
  }
  printf("12: max abs err %e\n", static_cast<double>(max_abs_err));
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl

// Instantiate for the current target.
void FastLog12Test::HWY_FUNC() { jxl::HWY_NAMESPACE::CheckFastLog12(); }
