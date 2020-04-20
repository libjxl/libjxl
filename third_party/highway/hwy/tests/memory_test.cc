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
#define HWY_TARGET_INCLUDE "tests/memory_test.cc"
#include "hwy/foreach_target.h"
#include "hwy/tests/test_util.h"

namespace hwy {

#include "hwy/tests/test_util-inl.h"

#include "hwy/begin_target-inl.h"

struct TestLoadStore {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto hi = Iota(d, 1 + d.N);
    const auto lo = Iota(d, 1);
    HWY_ALIGN T lanes[2 * d.N];
    Store(hi, d, lanes + d.N);
    Store(lo, d, lanes);

    // Aligned load
    const auto lo2 = Load(d, lanes);
    HWY_ASSERT_VEC_EQ(d, lo2, lo);

    // Aligned store
    HWY_ALIGN T lanes2[2 * d.N];
    Store(lo2, d, lanes2);
    Store(hi, d, lanes2 + d.N);
    for (size_t i = 0; i < 2 * d.N; ++i) {
      HWY_ASSERT_EQ(lanes[i], lanes2[i]);
    }

    // Unaligned load
    const auto vu = LoadU(d, lanes + 1);
    HWY_ALIGN T lanes3[d.N];
    Store(vu, d, lanes3);
    for (size_t i = 0; i < d.N; ++i) {
      HWY_ASSERT_EQ(T(i + 2), lanes3[i]);
    }

    // Unaligned store
    StoreU(lo2, d, lanes2 + d.N / 2);
    size_t i = 0;
    for (; i < d.N / 2; ++i) {
      HWY_ASSERT_EQ(lanes[i], lanes2[i]);
    }
    for (; i < 3 * d.N / 2; ++i) {
      HWY_ASSERT_EQ(T(i - d.N / 2 + 1), lanes2[i]);
    }
    // Subsequent values remain unchanged.
    for (; i < 2 * d.N; ++i) {
      HWY_ASSERT_EQ(T(i + 1), lanes2[i]);
    }
  }
};

struct TestLoadDup128 {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    // Scalar does not define LoadDup128.
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    constexpr size_t N128 = 16 / sizeof(T);
    alignas(16) T lanes[N128];
    for (size_t i = 0; i < N128; ++i) {
      lanes[i] = static_cast<T>(1 + i);
    }
    const auto v = LoadDup128(d, lanes);
    HWY_ALIGN T out[d.N];
    Store(v, d, out);
    for (size_t i = 0; i < d.N; ++i) {
      HWY_ASSERT_EQ(T(i % N128 + 1), out[i]);
    }
#else
    (void)d;
#endif
  }
};

struct TestStreamT {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, T(1));
    constexpr size_t kAffectedBytes =
        (d.N * sizeof(T) + HWY_STREAM_MULTIPLE - 1) &
        ~size_t(HWY_STREAM_MULTIPLE - 1);
    constexpr size_t kAffectedLanes = kAffectedBytes / sizeof(T);
    HWY_ALIGN T out[2 * kAffectedLanes] = {0};
    Stream(v, d, out);
    StoreFence();
    const auto actual = Load(d, out);
    HWY_ASSERT_VEC_EQ(d, v, actual);
    // Ensure Stream didn't modify more memory than expected
    for (size_t i = kAffectedLanes; i < 2 * kAffectedLanes; ++i) {
      HWY_ASSERT_EQ(T(0), out[i]);
    }
  }
};

#if (HWY_CAPS & HWY_CAP_GATHER) || HWY_IDE

// kShift must be log2(sizeof(T)).
struct TestGatherT {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    using Offset = MakeSigned<T>;

    // Base points to middle; |max_offset| + sizeof(T) <= kNumBytes / 2.
    constexpr size_t kNumBytes = kMaxVectorSize * 2;
    uint8_t bytes[kNumBytes];
    for (size_t i = 0; i < kNumBytes; ++i) {
      bytes[i] = static_cast<uint8_t>(i + 1);
    }
    const uint8_t* middle = bytes + kNumBytes / 2;

    // Offsets: combinations of aligned, repeated, negative.
    HWY_ALIGN Offset offset_lanes[HWY_MAX(d.N, 16)] = {
        2, 12, 4, 4, -16, -16, -21, -20, 8, 8, 8, -13, -13, -20, 20, 3};

    HWY_ALIGN T expected[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      HWY_ASSERT(std::abs(offset_lanes[i]) < Offset(kMaxVectorSize));
      CopyBytes<sizeof(T)>(middle + offset_lanes[i], &expected[i]);
    }

    const Desc<Offset, d.N> d_offset;
    const auto offsets = Load(d_offset, offset_lanes);
    auto actual = GatherOffset(d, reinterpret_cast<const T*>(middle), offsets);
    HWY_ASSERT_VEC_EQ(d, expected, actual);

    // Indices
    HWY_ALIGN const Offset index_lanes[HWY_MAX(d.N, 16)] = {
        1, -2, 0, 1, 3, -2, -1, 2, 4, -3, 5, -5, 0, 2, -4, 0};
    for (size_t i = 0; i < d.N; ++i) {
      CopyBytes<sizeof(T)>(
          middle + index_lanes[i] * static_cast<Offset>(sizeof(T)),
          &expected[i]);
    }
    const auto indices = Load(d_offset, index_lanes);
    actual = GatherIndex(d, reinterpret_cast<const T*>(middle), indices);
    HWY_ASSERT_VEC_EQ(d, expected, actual);
  }
};

template <int kShift>
struct TestFloatGatherT {
  template <class T, class D>
  HWY_NOINLINE HWY_ATTR void operator()(T /*unused*/, D d) {
    using Offset = MakeSigned<T>;
    static_assert(sizeof(T) == (1 << kShift), "Incorrect kShift");

    constexpr size_t kNumValues = 16;
    // Base points to middle; |max_index| < kNumValues / 2.
    HWY_ALIGN const T values[HWY_MAX(d.N, kNumValues)] = {
        T(100.0), T(110.0), T(111.0), T(128.0), T(1024.0), T(-1.0),
        T(-2.0),  T(-3.0),  T(0.25),  T(0.5),   T(0.75),   T(1.25),
        T(1.5),   T(1.75),  T(-0.25), T(-0.5)};
    const T* middle = values + kNumValues / 2;

    // Indices: combinations of aligned, repeated, negative.
    HWY_ALIGN const Offset index_lanes[HWY_MAX(d.N, 16)] = {1, -6, 0,  1,
                                                            3, -6, -1, 7};
    HWY_ALIGN T expected[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      CopyBytes<sizeof(T)>(middle + index_lanes[i], &expected[i]);
    }

    const Desc<Offset, d.N> d_offset;
    const auto indices = Load(d_offset, index_lanes);
    auto actual = GatherIndex(d, middle, indices);
    HWY_ASSERT_VEC_EQ(d, expected, actual);

    // Offsets: same as index * sizeof(T).
    const auto offsets = ShiftLeft<kShift>(indices);
    actual = GatherOffset(d, middle, offsets);
    HWY_ASSERT_VEC_EQ(d, expected, actual);
  }
};

#endif

HWY_NOINLINE HWY_ATTR void TestGather() {
#if (HWY_CAPS & HWY_CAP_GATHER) || HWY_IDE
  const ForPartialVectors<TestGatherT> test_int;
  // No u8,u16.
  test_int(uint32_t());
  test_int(uint64_t());
  // No i8,i16.
  test_int(int32_t());
  test_int(int64_t());

  ForPartialVectors<TestFloatGatherT<2>>()(float());
#if HWY_CAPS & HWY_CAP_DOUBLE
  ForPartialVectors<TestFloatGatherT<3>>()(double());
#endif
#endif
}

HWY_NOINLINE HWY_ATTR void TestStream() {
  const ForPartialVectors<TestStreamT> test;
  // No u8,u16.
  test(uint32_t());
  test(uint64_t());
  // No i8,i16.
  test(int32_t());
  test(int64_t());
  ForFloatTypes(ForPartialVectors<TestStreamT>());
}

HWY_NOINLINE HWY_ATTR void TestMemory() {
  ForAllTypes(ForPartialVectors<TestLoadStore>());
  ForAllTypes(ForGE128Vectors<TestLoadDup128>());
  TestGather();
  TestStream();

  // Test that these functions compile.
  LoadFence();
  StoreFence();
  int test = 0;
  Prefetch(&test);
  FlushCacheline(&test);
}

#include "hwy/end_target-inl.h"

#if HWY_ONCE
HWY_EXPORT(TestMemory)
#endif

}  // namespace hwy

#if HWY_ONCE
TEST(DISABLED_HwyMemoryTest, Run) { hwy::RunTest(&hwy::ChooseTestMemory); }
#endif
