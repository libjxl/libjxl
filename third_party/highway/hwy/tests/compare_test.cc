// Copyright 2019 Google LLC
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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/compare_test.cc"
#include "hwy/foreach_target.h"
#include "hwy/tests/test_util.h"

namespace hwy {

#include "hwy/tests/test_util-inl.h"

#include "hwy/begin_target-inl.h"

// All types.
struct TestEquality {
  template <typename T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, 2);
    const auto v2b = Iota(d, 2);
    const auto v3 = Iota(d, 3);

    HWY_ALIGN const T all_false[d.N] = {};
    HWY_ALIGN T all_true[d.N];
    memset(all_true, 0xFF, sizeof(all_true));

    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(v2 == v3));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 == v2));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 == v2b));
  }
};

// Integer and floating-point.
struct TestStrictT {
  template <typename T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, 2);
    const auto vn = Iota(d, -T(d.N));

    HWY_ALIGN const T all_false[d.N] = {};
    HWY_ALIGN T all_true[d.N];
    memset(all_true, 0xFF, sizeof(all_true));

    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 > vn));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(vn < v2));
    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(v2 < vn));
    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(vn > v2));
  }
};

HWY_NOINLINE HWY_ATTR void TestStrict() {
  const ForPartialVectors<TestStrictT> test;

  // HWY_CAP_INT64 is insufficient, so we cannot use ForSignedTypes.
  test(int8_t());
  test(int16_t());
  test(int32_t());
#if HWY_CAPS & HWY_CAP_CMP64
  test(int64_t());
#endif

  ForFloatTypes(test);
}

// Floating-point.
struct TestWeak {
  template <typename T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, 2);
    const auto vn = Iota(d, -T(d.N));

    HWY_ALIGN const T all_false[d.N] = {};
    HWY_ALIGN T all_true[d.N];
    memset(all_true, 0xFF, sizeof(all_true));

    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 >= v2));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(vn <= vn));

    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 >= vn));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(vn <= v2));

    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(v2 <= vn));
    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(vn >= v2));
  }
};

HWY_NOINLINE HWY_ATTR void TestCompare() {
  ForAllTypes(ForPartialVectors<TestEquality>());
  TestStrict();
  ForFloatTypes(ForPartialVectors<TestWeak>());
}

#include "hwy/end_target-inl.h"

#if HWY_ONCE
HWY_EXPORT(TestCompare)
#endif

}  // namespace hwy

#if HWY_ONCE
TEST(HwyCompareTest, Run) { hwy::RunTest(&hwy::ChooseTestCompare); }
#endif
