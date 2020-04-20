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
#define HWY_TARGET_INCLUDE "tests/hwy_test.cc"
#include "hwy/foreach_target.h"
#include "hwy/tests/test_util.h"

namespace hwy {

#include "hwy/tests/test_util-inl.h"

#include "hwy/begin_target-inl.h"

namespace examples {

template <class DF>
HWY_NOINLINE HWY_ATTR void FloorLog2(const DF df,
                                     const uint8_t* HWY_RESTRICT values,
                                     uint8_t* HWY_RESTRICT log2) {
  // Descriptors for all required data types:
  const Desc<int32_t, df.N> d32;
  const Desc<uint8_t, df.N> d8;

  const auto u8 = Load(d8, values);
  const auto bits = BitCast(d32, ConvertTo(df, PromoteTo(d32, u8)));
  const auto exponent = ShiftRight<23>(bits) - Set(d32, 127);
  Store(DemoteTo(d8, exponent), d8, log2);
}

struct TestFloorLog2 {
  template <class T, class DF>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, DF df) {
    const size_t kBytes = 32;
    static_assert(kBytes % df.N == 0, "Must be a multiple of df.N");

    HWY_ALIGN uint8_t in[kBytes];
    uint8_t expected[kBytes];
    RandomState rng{1234};
    for (size_t i = 0; i < kBytes; ++i) {
      expected[i] = Random32(&rng) & 7;
      in[i] = static_cast<uint8_t>(1u << expected[i]);
    }
    HWY_ALIGN uint8_t out[32];
    for (size_t i = 0; i < kBytes; i += df.N) {
      FloorLog2(df, in + i, out + i);
    }
    int sum = 0;
    for (size_t i = 0; i < kBytes; ++i) {
      HWY_ASSERT_EQ(expected[i], out[i]);
      sum += out[i];
    }
    PreventElision(sum);
  }
};

template <class D, typename T>
HWY_NOINLINE HWY_ATTR void MulAddLoop(const D d,
                                      const T* HWY_RESTRICT mul_array,
                                      const T* HWY_RESTRICT add_array,
                                      const size_t size,
                                      T* HWY_RESTRICT x_array) {
  // Type-agnostic (caller-specified lane type) and width-agnostic (uses
  // best available instruction set).
  for (size_t i = 0; i < size; i += d.N) {
    const auto mul = Load(d, mul_array + i);
    const auto add = Load(d, add_array + i);
    auto x = Load(d, x_array + i);
    x = MulAdd(mul, x, add);
    Store(x, d, x_array + i);
  }
}

struct TestSumMulAdd {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    RandomState rng{1234};
    const size_t kSize = 64;
    HWY_ALIGN T mul[kSize];
    HWY_ALIGN T x[kSize];
    HWY_ALIGN T add[kSize];
    for (size_t i = 0; i < kSize; ++i) {
      mul[i] = Random32(&rng) & 0xF;
      x[i] = Random32(&rng) & 0xFF;
      add[i] = Random32(&rng) & 0xFF;
    }
    MulAddLoop(d, mul, add, kSize, x);
    double sum = 0.0;
    for (auto xi : x) {
      sum += static_cast<double>(xi);
    }
    HWY_ASSERT_EQ(78944.0, sum);
  }
};

HWY_NOINLINE HWY_ATTR void TestExamples() {
  ForPartialVectors<TestFloorLog2>()(float());

  const ForPartialVectors<TestSumMulAdd> test_mul_add;
  test_mul_add(float());
#if HWY_CAPS & HWY_CAP_DOUBLE
  test_mul_add(double());
#endif
}

}  // namespace examples

namespace basic {

// util.h

HWY_NOINLINE HWY_ATTR void TestLimits() {
  HWY_ASSERT_EQ(uint8_t(0), LimitsMin<uint8_t>());
  HWY_ASSERT_EQ(uint16_t(0), LimitsMin<uint16_t>());
  HWY_ASSERT_EQ(uint32_t(0), LimitsMin<uint32_t>());
  HWY_ASSERT_EQ(uint64_t(0), LimitsMin<uint64_t>());

  HWY_ASSERT_EQ(int8_t(-128), LimitsMin<int8_t>());
  HWY_ASSERT_EQ(int16_t(-32768), LimitsMin<int16_t>());
  HWY_ASSERT_EQ(int32_t(0x80000000u), LimitsMin<int32_t>());
  HWY_ASSERT_EQ(int64_t(0x8000000000000000ull), LimitsMin<int64_t>());

  HWY_ASSERT_EQ(uint8_t(0xFF), LimitsMax<uint8_t>());
  HWY_ASSERT_EQ(uint16_t(0xFFFF), LimitsMax<uint16_t>());
  HWY_ASSERT_EQ(uint32_t(0xFFFFFFFFu), LimitsMax<uint32_t>());
  HWY_ASSERT_EQ(uint64_t(0xFFFFFFFFFFFFFFFFull), LimitsMax<uint64_t>());

  HWY_ASSERT_EQ(int8_t(0x7F), LimitsMax<int8_t>());
  HWY_ASSERT_EQ(int16_t(0x7FFF), LimitsMax<int16_t>());
  HWY_ASSERT_EQ(int32_t(0x7FFFFFFFu), LimitsMax<int32_t>());
  HWY_ASSERT_EQ(int64_t(0x7FFFFFFFFFFFFFFFull), LimitsMax<int64_t>());
}

// Test the ToString used to output test failures

HWY_NOINLINE HWY_ATTR void TestToString() {
  HWY_ASSERT_STRING_EQ("0", std::to_string(int64_t(0)).c_str());
  HWY_ASSERT_STRING_EQ("3", std::to_string(int64_t(3)).c_str());
  HWY_ASSERT_STRING_EQ("-1", std::to_string(int64_t(-1)).c_str());

  HWY_ASSERT_STRING_EQ("9223372036854775807",
                       std::to_string(0x7FFFFFFFFFFFFFFFLL).c_str());
  HWY_ASSERT_STRING_EQ("-9223372036854775808",
                       std::to_string(int64_t(0x8000000000000000ULL)).c_str());

  HWY_ASSERT_STRING_EQ("0.000000", std::to_string(0.0).c_str());
  HWY_ASSERT_STRING_EQ("4.000000", std::to_string(4.0).c_str());
  HWY_ASSERT_STRING_EQ("-1.000000", std::to_string(-1.0).c_str());
  HWY_ASSERT_STRING_EQ("-1.250000", std::to_string(-1.25).c_str());
  HWY_ASSERT_STRING_EQ("2.125000", std::to_string(2.125f).c_str());
}

struct TestIsUnsigned {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D /*unused*/) {
    static_assert(!IsFloat<T>(), "Expected !IsFloat");
    static_assert(!IsSigned<T>(), "Expected !IsSigned");
  }
};

struct TestIsSigned {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D /*unused*/) {
    static_assert(!IsFloat<T>(), "Expected !IsFloat");
    static_assert(IsSigned<T>(), "Expected IsSigned");
  }
};

struct TestIsFloat {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D /*unused*/) {
    static_assert(IsFloat<T>(), "Expected IsFloat");
    static_assert(IsSigned<T>(), "Floats are also considered signed");
  }
};

HWY_NOINLINE HWY_ATTR void TestType() {
  ForUnsignedTypes(ForPartialVectors<TestIsUnsigned>());
  ForSignedTypes(ForPartialVectors<TestIsSigned>());
  ForFloatTypes(ForPartialVectors<TestIsFloat>());
}

// Ensures wraparound (mod 2^bits)
struct TestOverflowT {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto v1 = Set(d, T(1));
    const auto vmax = Set(d, LimitsMax<T>());
    const auto vmin = Set(d, LimitsMin<T>());
    // Unsigned underflow / negative -> positive
    HWY_ASSERT_VEC_EQ(d, vmax, vmin - v1);
    // Unsigned overflow / positive -> negative
    HWY_ASSERT_VEC_EQ(d, vmin, vmax + v1);
  }
};

HWY_NOINLINE HWY_ATTR void TestOverflow() {
  ForIntegerTypes(ForPartialVectors<TestOverflowT>());
}

struct TestName {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    std::string expected = IsFloat<T>() ? "f" : (IsSigned<T>() ? "i" : "u");
    expected += std::to_string(sizeof(T) * 8);
    if (d.N != 1) {
      expected += 'x';
      expected += std::to_string(d.N);
    }
    if (expected != TypeName<T, d.N>()) {
      NotifyFailure(__FILE__, __LINE__, expected.c_str(), 0, expected.c_str(),
                    TypeName<T, d.N>());
    }
  }
};

struct TestSet {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    // Zero
    const auto v0 = Zero(d);
    HWY_ALIGN T expected[d.N] = {};  // zero-initialized.
    HWY_ASSERT_VEC_EQ(d, expected, v0);

    // Set
    const auto v2 = Set(d, T(2));
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = 2;
    }
    HWY_ASSERT_VEC_EQ(d, expected, v2);

    // iota
    const auto vi = Iota(d, T(5));
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = 5 + i;
    }
    HWY_ASSERT_VEC_EQ(d, expected, vi);

    // undefined
    const auto vu = Undefined(d);
    Store(vu, d, expected);
  }
};

struct TestCopyAndAssign {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    // copy V
    const auto v3 = Iota(d, 3);
    auto v3b(v3);
    HWY_ASSERT_VEC_EQ(d, v3, v3b);

    // assign V
    auto v3c = Undefined(d);
    v3c = v3;
    HWY_ASSERT_VEC_EQ(d, v3, v3c);
  }
};

struct TestLowerHalfT {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    constexpr size_t N2 = (d.N + 1) / 2;
    const Desc<T, N2> d2;

    HWY_ALIGN T lanes[d.N] = {0};
    const auto v = Iota(d, 1);
    Store(LowerHalf(v), d2, lanes);
    size_t i = 0;
    for (; i < N2; ++i) {
      HWY_ASSERT_EQ(T(1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < d.N; ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
  }
};

struct TestLowerQuarterT {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    constexpr size_t N4 = (d.N + 3) / 4;
    const HWY_CAPPED(T, N4) d4;

    HWY_ALIGN T lanes[d.N] = {0};
    const auto v = Iota(d, 1);
    const auto lo = LowerHalf(LowerHalf(v));
    Store(lo, d4, lanes);
    size_t i = 0;
    for (; i < N4; ++i) {
      HWY_ASSERT_EQ(T(i + 1), lanes[i]);
    }
    // Upper 3/4 remain unchanged
    for (; i < d.N; ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
  }
};

HWY_NOINLINE HWY_ATTR void TestLowerHalf() {
  ForAllTypes(
      ForPartialVectors<TestLowerHalfT, /*kDivLanes=*/1, /*kMinLanes=*/2>());
  ForAllTypes(
      ForPartialVectors<TestLowerQuarterT, /*kDivLanes=*/1, /*kMinLanes=*/4>());
}

struct TestUpperHalfT {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    // Scalar does not define UpperHalf.
#if HWY_TARGET != HWY_SCALAR
    size_t i;
    constexpr size_t N2 = (d.N + 1) / 2;
    const Desc<T, N2> d2;

    const auto v = Iota(d, 1);
    HWY_ALIGN T lanes[d.N] = {0};

    Store(UpperHalf(v), d2, lanes);
    i = 0;
    for (; i < N2; ++i) {
      HWY_ASSERT_EQ(T(N2 + 1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < d.N; ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE HWY_ATTR void TestUpperHalf() {
  ForAllTypes(ForGE128Vectors<TestUpperHalfT>());
}

HWY_NOINLINE HWY_ATTR void TestBasic() {
  TestLimits();
  TestToString();
  TestType();
  ForAllTypes(ForPartialVectors<TestName>());
  TestOverflow();
  ForAllTypes(ForPartialVectors<TestSet>());
  ForAllTypes(ForPartialVectors<TestCopyAndAssign>());
  TestLowerHalf();
  TestUpperHalf();
}

}  // namespace basic

HWY_NOINLINE HWY_ATTR void TestHwy() {
  examples::TestExamples();
  basic::TestBasic();
}

#include "hwy/end_target-inl.h"

#if HWY_ONCE
HWY_EXPORT(TestHwy)
#endif

}  // namespace hwy

#if HWY_ONCE
TEST(HwyHwyTest, Run) { hwy::RunTest(&hwy::ChooseTestHwy); }
#endif
