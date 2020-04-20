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

// NOTE: this must be included inside the project's namespace, but outside of
// its begin/end_target-inl.

// This could have a HWY_TARGET_TOGGLE include guard, but the IDE cannot see
// through that and would believe these functions are undefined.

// No effect when compiling (already included), but this makes the IDE aware of
// the definitions to avoid error messages in this header.
#include "hwy/highway.h"
#include "hwy/tests/test_util.h"

#include "hwy/begin_target-inl.h"

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <class D, typename T2>
HWY_ATTR HWY_VEC(D) Iota(const D d, const T2 first) {
  HWY_ALIGN typename D::T lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = first + static_cast<T2>(i);
  }
  return Load(d, lanes);
}

HWY_NORETURN void NotifyFailure(const char* filename, const int line,
                                const char* type_name, const size_t lane,
                                const char* expected, const char* actual) {
  fprintf(stderr, "%s:%d: %s, %s lane %zu mismatch: expected '%s', got '%s'.\n",
          filename, line, hwy::TargetName(HWY_TARGET), type_name, lane,
          expected, actual);
  hwy::Trap();
}

// Compare non-vector, non-string T.
template <typename T>
void AssertEqual(const T expected, const T actual, const char* filename = "",
                 const int line = -1, const size_t lane = 0,
                 const char* name = nullptr) {
  if (name == nullptr) name = hwy::TypeName<T, 1>();
  // Rely on string comparison to ensure similar floats are "equal".
  const std::string expected_str = ToString(expected);
  const std::string actual_str = ToString(actual);
  if (expected_str != actual_str) {
    NotifyFailure(filename, line, name, lane, expected_str.c_str(),
                  actual_str.c_str());
  }
}

HWY_ATTR void AssertStringEqual(const char* expected, const char* actual,
                                const char* filename = "", const int line = -1,
                                const size_t lane = 0) {
  if (!hwy::StringsEqual(expected, actual)) {
    NotifyFailure(filename, line, "string", lane, expected, actual);
  }
}

// Compare expected vector to vector.
template <class D, class V>
HWY_ATTR void AssertVecEqual(D d, const V expected, const V actual,
                             const char* filename, const int line) {
  HWY_ALIGN typename D::T expected_lanes[d.N];
  HWY_ALIGN typename D::T actual_lanes[d.N];
  Store(expected, d, expected_lanes);
  Store(actual, d, actual_lanes);
  for (size_t i = 0; i < d.N; ++i) {
    AssertEqual(expected_lanes[i], actual_lanes[i], filename, line, i,
                hwy::TypeName<typename D::T, d.N>());
  }
}

// Compare expected lanes to vector.
template <class D, class V>
HWY_ATTR void AssertVecEqual(D d, const typename D::T (&expected)[D::N],
                             V actual, const char* filename, int line) {
  AssertVecEqual(d, LoadU(d, expected), actual, filename, line);
}

#ifndef HWY_ASSERT

#define HWY_ASSERT(condition)                                          \
  do {                                                                 \
    if (!(condition)) {                                                \
      NotifyFailure(__FILE__, __LINE__, "Assert", 0, "1", #condition); \
    }                                                                  \
  } while (0)

#define HWY_ASSERT_EQ(expected, actual) \
  AssertEqual(expected, actual, __FILE__, __LINE__)

#define HWY_ASSERT_STRING_EQ(expected, actual) \
  AssertStringEqual(expected, actual, __FILE__, __LINE__)

#define HWY_ASSERT_VEC_EQ(d, expected, actual) \
  AssertVecEqual(d, expected, actual, __FILE__, __LINE__)

#endif  // HWY_ASSERT

// Helpers for instantiating tests with combinations of lane types / counts.

// For all powers of two in [kMinLanes, N * kMinLanes] (so that recursion stops
// at N == 0)
template <typename T, size_t N, size_t kMinLanes, class Test>
struct ForeachSizeR {
  static HWY_ATTR void Do() {
    static_assert(N != 0, "End of recursion");
    Test()(T(), hwy::Desc<T, N * kMinLanes>());
    ForeachSizeR<T, N / 2, kMinLanes, Test>::Do();
  }
};

// Base case to stop the recursion.
template <typename T, size_t kMinLanes, class Test>
struct ForeachSizeR<T, 0, kMinLanes, Test> {
  static HWY_ATTR void Do() {}
};

// These adapters may be called directly, or via For*Types:

// Calls Test for all powers of two in [kMinLanes, HWY_FULL(T)::N / kDivLanes].
template <class Test, size_t kDivLanes = 1, size_t kMinLanes = 1>
struct ForPartialVectors {
  template <typename T>
  HWY_ATTR void operator()(T /*unused*/) const {
    ForeachSizeR<T, HWY_LANES(T) / kDivLanes / kMinLanes, kMinLanes,
                 Test>::Do();
  }
};

// Calls Test for all powers of two in [128 bits, max bits].
template <class Test>
struct ForGE128Vectors {
  template <typename T>
  HWY_ATTR void operator()(T /*unused*/) const {
    ForeachSizeR<T, HWY_LANES(T) / (16 / sizeof(T)), (16 / sizeof(T)),
                 Test>::Do();
  }
};

// Calls Test for full vectors only.
template <class Test>
struct ForFullVectors {
  template <typename T>
  HWY_ATTR void operator()(T t) const {
    Test()(t, HWY_FULL(T)());
  }
};

// Type lists to shorten call sites:

template <class Func>
void ForSignedTypes(const Func& func) {
  func(int8_t());
  func(int16_t());
  func(int32_t());
#if HWY_CAPS & HWY_CAP_INT64
  func(int64_t());
#endif
}

template <class Func>
void ForUnsignedTypes(const Func& func) {
  func(uint8_t());
  func(uint16_t());
  func(uint32_t());
#if HWY_CAPS & HWY_CAP_INT64
  func(uint64_t());
#endif
}

template <class Func>
void ForIntegerTypes(const Func& func) {
  ForSignedTypes(func);
  ForUnsignedTypes(func);
}

template <class Func>
void ForFloatTypes(const Func& func) {
  func(float());
#if HWY_CAPS & HWY_CAP_DOUBLE
  func(double());
#endif
}

template <class Func>
void ForAllTypes(const Func& func) {
  ForIntegerTypes(func);
  ForFloatTypes(func);
}

#include "hwy/end_target-inl.h"
