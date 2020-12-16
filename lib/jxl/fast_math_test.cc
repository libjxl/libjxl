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

#include <stdio.h>

#include <random>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/fast_math_test.cc"
#include <hwy/foreach_target.h>

#include "lib/jxl/fast_math-inl.h"

// Test utils
#include <hwy/highway.h>
#include <hwy/tests/test_util-inl.h>
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

HWY_NOINLINE void TestFastLog2() {
  constexpr size_t kNumTrials = 1 << 23;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(1e-7f, 1e3f);
  float max_abs_err = 0;
  HWY_FULL(float) d;
  for (size_t i = 0; i < kNumTrials; i++) {
    const float f = dist(rng);
    const auto actual_v = FastLog2f(d, Set(d, f));
    const float actual = GetLane(actual_v);
    const float abs_err = std::abs(std::log2(f) - actual);
    EXPECT_LT(abs_err, 2.9E-6) << "f = " << f;
    max_abs_err = std::max(max_abs_err, abs_err);
  }
  printf("max abs err %e\n", static_cast<double>(max_abs_err));
}

HWY_NOINLINE void TestFastCos() {
  constexpr size_t kNumTrials = 1 << 23;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1e3f, 1e3f);
  float max_abs_err = 0;
  HWY_FULL(float) d;
  for (size_t i = 0; i < kNumTrials; i++) {
    const float f = dist(rng);
    const auto actual_v = FastCosf(d, Set(d, f));
    const float actual = GetLane(actual_v);
    const float abs_err = std::abs(std::cos(f) - actual);
    EXPECT_LT(abs_err, 7E-5) << "f = " << f;
    max_abs_err = std::max(max_abs_err, abs_err);
  }
  printf("max abs err %e\n", static_cast<double>(max_abs_err));
}

HWY_NOINLINE void TestFastErf() {
  constexpr size_t kNumTrials = 1 << 23;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-5.f, 5.f);
  float max_abs_err = 0;
  HWY_FULL(float) d;
  for (size_t i = 0; i < kNumTrials; i++) {
    const float f = dist(rng);
    const auto actual_v = FastErff(d, Set(d, f));
    const float actual = GetLane(actual_v);
    const float abs_err = std::abs(std::erf(f) - actual);
    EXPECT_LT(abs_err, 7E-4) << "f = " << f;
    max_abs_err = std::max(max_abs_err, abs_err);
  }
  printf("max abs err %e\n", static_cast<double>(max_abs_err));
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

class FastMathTargetTest : public hwy::TestWithParamTarget {};
HWY_TARGET_INSTANTIATE_TEST_SUITE_P(FastMathTargetTest);

HWY_EXPORT_AND_TEST_P(FastMathTargetTest, TestFastLog2);
HWY_EXPORT_AND_TEST_P(FastMathTargetTest, TestFastCos);
HWY_EXPORT_AND_TEST_P(FastMathTargetTest, TestFastErf);

}  // namespace jxl
#endif  // HWY_ONCE
