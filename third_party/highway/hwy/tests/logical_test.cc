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
#define HWY_TARGET_INCLUDE "tests/logical_test.cc"
#include "hwy/foreach_target.h"
#include "hwy/tests/test_util.h"

namespace hwy {

#include "hwy/tests/test_util-inl.h"

#include "hwy/begin_target-inl.h"

struct TestLogicalT {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);

    HWY_ASSERT_VEC_EQ(d, v0, v0 & vi);
    HWY_ASSERT_VEC_EQ(d, v0, vi & v0);
    HWY_ASSERT_VEC_EQ(d, vi, vi & vi);

    HWY_ASSERT_VEC_EQ(d, vi, v0 | vi);
    HWY_ASSERT_VEC_EQ(d, vi, vi | v0);
    HWY_ASSERT_VEC_EQ(d, vi, vi | vi);

    HWY_ASSERT_VEC_EQ(d, vi, v0 ^ vi);
    HWY_ASSERT_VEC_EQ(d, vi, vi ^ v0);
    HWY_ASSERT_VEC_EQ(d, v0, vi ^ vi);

    HWY_ASSERT_VEC_EQ(d, vi, AndNot(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, v0));
    HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, vi));

    auto v = vi;
    v &= vi;
    HWY_ASSERT_VEC_EQ(d, vi, v);
    v &= v0;
    HWY_ASSERT_VEC_EQ(d, v0, v);

    v |= vi;
    HWY_ASSERT_VEC_EQ(d, vi, v);
    v |= v0;
    HWY_ASSERT_VEC_EQ(d, vi, v);

    v ^= vi;
    HWY_ASSERT_VEC_EQ(d, v0, v);
    v ^= v0;
    HWY_ASSERT_VEC_EQ(d, v0, v);
  }
};

struct TestLogicalFloat {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);

    HWY_ASSERT_VEC_EQ(d, v0, And(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, And(vi, v0));
    HWY_ASSERT_VEC_EQ(d, vi, And(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, Or(v0, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Or(vi, v0));
    HWY_ASSERT_VEC_EQ(d, vi, Or(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, Xor(v0, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Xor(vi, v0));
    HWY_ASSERT_VEC_EQ(d, v0, Xor(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, AndNot(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, v0));
    HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, vi));

    auto v = vi;
    v = And(v, vi);
    HWY_ASSERT_VEC_EQ(d, vi, v);
    v = And(v, v0);
    HWY_ASSERT_VEC_EQ(d, v0, v);

    v = Or(v, vi);
    HWY_ASSERT_VEC_EQ(d, vi, v);
    v = Or(v, v0);
    HWY_ASSERT_VEC_EQ(d, vi, v);

    v = Xor(v, vi);
    HWY_ASSERT_VEC_EQ(d, v0, v);
    v = Xor(v, v0);
    HWY_ASSERT_VEC_EQ(d, v0, v);
  }
};

// Vec <-> Mask, IfThen*
struct TestIfThenElse {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    RandomState rng{1234};
    const T no(0);
    T yes;
    memset(&yes, 0xFF, sizeof(yes));

    HWY_ALIGN T in1[d.N] = {};         // Initialized for clang-analyzer.
    HWY_ALIGN T in2[d.N] = {};         // Initialized for clang-analyzer.
    HWY_ALIGN T mask_lanes[d.N] = {};  // Initialized for clang-analyzer.
    for (size_t i = 0; i < d.N; ++i) {
      in1[i] = static_cast<T>(Random32(&rng));
      in2[i] = static_cast<T>(Random32(&rng));
      mask_lanes[i] = (Random32(&rng) & 1024) ? no : yes;
    }

    const auto vec = Load(d, mask_lanes);
    const auto mask = MaskFromVec(vec);
    HWY_ASSERT_VEC_EQ(d, vec, VecFromMask(mask));

    HWY_ALIGN T out_lanes1[d.N];
    HWY_ALIGN T out_lanes2[d.N];
    HWY_ALIGN T out_lanes3[d.N];
    Store(IfThenElse(mask, Load(d, in1), Load(d, in2)), d, out_lanes1);
    Store(IfThenElseZero(mask, Load(d, in1)), d, out_lanes2);
    Store(IfThenZeroElse(mask, Load(d, in2)), d, out_lanes3);
    for (size_t i = 0; i < d.N; ++i) {
      // Cannot reliably compare against yes (NaN).
      HWY_ASSERT_EQ((mask_lanes[i] == no) ? in2[i] : in1[i], out_lanes1[i]);
      HWY_ASSERT_EQ((mask_lanes[i] == no) ? no : in1[i], out_lanes2[i]);
      HWY_ASSERT_EQ((mask_lanes[i] == no) ? in2[i] : no, out_lanes3[i]);
    }
  }
};

struct TestTestBit {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const size_t kNumBits = sizeof(T) * 8;
    for (size_t i = 0; i < kNumBits; ++i) {
      const auto bit1 = Set(d, 1ull << i);
      const auto bit2 = Set(d, 1ull << ((i + 1) % kNumBits));
      const auto bit3 = Set(d, 1ull << ((i + 2) % kNumBits));
      const auto bits12 = bit1 | bit2;
      const auto bits23 = bit2 | bit3;
      HWY_ASSERT(AllTrue(TestBit(bit1, bit1)));
      HWY_ASSERT(AllTrue(TestBit(bits12, bit1)));
      HWY_ASSERT(AllTrue(TestBit(bits12, bit2)));

      HWY_ASSERT(AllFalse(TestBit(bits12, bit3)));
      HWY_ASSERT(AllFalse(TestBit(bits23, bit1)));
      HWY_ASSERT(AllFalse(TestBit(bit1, bit2)));
      HWY_ASSERT(AllFalse(TestBit(bit2, bit1)));
      HWY_ASSERT(AllFalse(TestBit(bit1, bit3)));
      HWY_ASSERT(AllFalse(TestBit(bit3, bit1)));
      HWY_ASSERT(AllFalse(TestBit(bit2, bit3)));
      HWY_ASSERT(AllFalse(TestBit(bit3, bit2)));
    }
  }
};

struct TestAllTrueFalse {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto zero = Zero(d);
    const T max = LimitsMax<T>();
    const T min_nonzero = LimitsMin<T>() + 1;

    auto v = zero;
    HWY_ALIGN T lanes[d.N] = {};  // Initialized for clang-analyzer.
    Store(v, d, lanes);
    HWY_ASSERT(AllTrue(v == zero));
    HWY_ASSERT(!AllFalse(v == zero));

#if HWY_TARGET == HWY_SCALAR
    // Simple negation of the AllTrue result
    const bool expected_all_false = false;
#else
    // There are multiple lanes and one is nonzero
    const bool expected_all_false = true;
#endif

    // Set each lane to nonzero and back to zero
    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = max;
      v = Load(d, lanes);
      HWY_ASSERT(!AllTrue(v == zero));
      HWY_ASSERT(expected_all_false ^ AllFalse(v == zero));

      lanes[i] = min_nonzero;
      v = Load(d, lanes);
      HWY_ASSERT(!AllTrue(v == zero));
      HWY_ASSERT(expected_all_false ^ AllFalse(v == zero));

      // Reset to all zero
      lanes[i] = T(0);
      v = Load(d, lanes);
      HWY_ASSERT(AllTrue(v == zero));
      HWY_ASSERT(!AllFalse(v == zero));
    }
  }
};

class TestBitsFromMask {
 public:
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T t, D d) {
    // Fixed patterns: all off/on/odd/even.
    Bits(t, d, 0);
    Bits(t, d, ~0ull);
    Bits(t, d, 0x5555555555555555ull);
    Bits(t, d, 0xAAAAAAAAAAAAAAAAull);

    // Random mask patterns.
    std::mt19937_64 rng;
    for (size_t i = 0; i < 100; ++i) {
      Bits(t, d, rng());
    }
  }

 private:
  template <typename T, class D>
  HWY_NOINLINE HWY_ATTR void Bits(T /*unused*/, D d, uint64_t bits) {
    // Generate a mask matching the given bits
    HWY_ALIGN T mask_lanes[d.N];
    memset(mask_lanes, 0xFF, sizeof(mask_lanes));
    for (size_t i = 0; i < d.N; ++i) {
      if ((bits & (1ull << i)) == 0) mask_lanes[i] = 0;
    }
    const auto mask = MaskFromVec(Load(d, mask_lanes));

    const uint64_t actual_bits = BitsFromMask(mask);

    // Clear bits that cannot be returned for this D.
    constexpr size_t shift = 64 - d.N;  // 0..63 - avoids UB for d.N == 64.
    static_assert(shift < 64, "d.N out of range");  // Silences clang-tidy.
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    const uint64_t expected_bits = (bits << shift) >> shift;

    HWY_ASSERT_EQ(expected_bits, actual_bits);
  }
};

struct TestCountTrue {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    // For all combinations of zero/nonzero state of subset of lanes:
    const size_t max_lanes = std::min(d.N, size_t(10));

    HWY_ALIGN T lanes[d.N];
    std::fill(lanes, lanes + d.N, T(1));

    for (size_t code = 0; code < (1ull << max_lanes); ++code) {
      // Number of zeros written = number of mask lanes that are true.
      size_t expected = 0;
      for (size_t i = 0; i < max_lanes; ++i) {
        lanes[i] = T(1);
        if (code & (1ull << i)) {
          ++expected;
          lanes[i] = T(0);
        }
      }

      const auto mask = Load(d, lanes) == Zero(d);
      const size_t actual = CountTrue(mask);
      HWY_ASSERT_VEC_EQ(d, Set(d, expected), Set(d, actual));
    }
  }
};

HWY_NOINLINE HWY_ATTR void TestLogical() {
  ForIntegerTypes(ForPartialVectors<TestLogicalT>());
  ForFloatTypes(ForPartialVectors<TestLogicalFloat>());
  ForAllTypes(ForPartialVectors<TestIfThenElse>());

  // These only make sense for full vectors.
  ForIntegerTypes(ForFullVectors<TestTestBit>());
  ForAllTypes(ForFullVectors<TestAllTrueFalse>());
  ForAllTypes(ForFullVectors<TestBitsFromMask>());
  ForAllTypes(ForFullVectors<TestCountTrue>());
}

#include "hwy/end_target-inl.h"

#if HWY_ONCE
HWY_EXPORT(TestLogical)
#endif

}  // namespace hwy

#if HWY_ONCE
TEST(HwyLogicalTest, Run) { hwy::RunTest(&hwy::ChooseTestLogical); }
#endif
