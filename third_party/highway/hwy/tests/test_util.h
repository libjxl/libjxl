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

// Need include guard because this is included first from the test, then from
// the subsequent test_util-inl.h so the IDE sees its dependencies.
#ifndef HWY_TESTS_TEST_UTIL_H_
#define HWY_TESTS_TEST_UTIL_H_

// SIMD-independent helper functions for use by *_test.cc.

#include <stdio.h>
#include <string.h>
#include <random>
#include <string>
#include <utility>  // std::forward

#include "hwy/highway.h"

// Caller wants to use gtest.
#ifdef HWY_USE_GTEST
#include "gtest/gtest.h"
#else
// The tests are gtest-compatible and we only need to define this. Note that
// cmake scans for a TEST() marker in the source code. This macro is followed by
// the body of the test, and we rely on C99/C++ guarantees that reaching the end
// of main is equivalent to return 0. Note that this means we can only have one
// test case per file, which also minimizes runtime-dispatch boilerplate.
#define TEST(test_case, test_name) int main()
#endif  // HWY_USE_GTEST

namespace hwy {

// The maximum vector size used in tests when defining test data. This is at
// least the kMaxVectorSize but it can be bigger. If you increased
// kMaxVectorSize, you also need to increase this constant and update all the
// tests that use it to define bigger arrays of test data.
constexpr size_t kTestMaxVectorSize = 64;
static_assert(kTestMaxVectorSize >= kMaxVectorSize,
              "All kTestMaxVectorSize test arrays need to be updated");

// Calls Functor for each individual target bit.
template <class Functor>
HWY_INLINE void ForeachTarget(const Functor& functor) {
  // Prevent infinite loop by only including targets that are enabled.
  uint32_t targets_bits = SupportedTargets() & HWY_TARGETS;

  while (targets_bits != 0) {
    // 2's complement negation flips all bits above the lowest 1-bit and
    // zeros lower bits, so ANDing with that clears all but the lowest 1-bit.
    const uint32_t isolated_bit = targets_bits & (~targets_bits + 1);
    functor(isolated_bit);
    targets_bits &= ~isolated_bit;
  }
}

// Calls function pointers returned by Choose for each individual target bit.
template <class Choose, typename... Args>
HWY_INLINE void ChooseAndCallForeachTarget(const Choose& choose,
                                           Args&&... args) {
  ForeachTarget([&](const uint32_t target_bit) {
    const auto func = choose(target_bit);
    (*func)(std::forward<Args>(args)...);
  });
}

// Calls test for each enabled and available target.
template <class Choose, typename... Args>
void RunTest(const Choose& choose, Args&&... args) {
  setvbuf(stdin, nullptr, _IONBF, 0);

  ChooseAndCallForeachTarget(choose, std::forward<Args>(args)...);

  std::string targets;
  ForeachTarget([&targets](uint32_t target_bit) {
    targets += TargetName(static_cast<int>(target_bit));
    targets += ", ";
  });
  targets.resize(targets.size() - 2);  // strip last comma
  printf("Successfully tested %s.\n", targets.c_str());

  if ((HWY_TARGETS & HWY_SCALAR) == 0) {
    fprintf(stderr, "WARNING: targets %x lack HWY_SCALAR, so it was skipped\n",
            HWY_TARGETS);
  }
}

// Random numbers
typedef std::mt19937 RandomState;
HWY_INLINE uint32_t Random32(RandomState* rng) {
  return static_cast<uint32_t>((*rng)());
}

// Prevents the compiler from eliding the computations that led to "output".
// Works by indicating to the compiler that "output" is being read and modified.
// The +r constraint avoids unnecessary writes to memory, but only works for
// built-in types.
template <class T>
inline void PreventElision(T&& output) {
#ifndef _MSC_VER
  asm volatile("" : "+r"(output) : : "memory");
#endif
}

// Returns a name for the vector/part/scalar. The type prefix is u/i/f for
// unsigned/signed/floating point, followed by the number of bits per lane;
// then 'x' followed by the number of lanes. Example: u8x16. This is useful for
// understanding which instantiation of a generic test failed.
template <typename T, size_t N>
const char* TypeName() {
  constexpr char prefix = IsFloat<T>() ? 'f' : (IsSigned<T>() ? 'i' : 'u');

  constexpr size_t bits = sizeof(T) * 8;
  constexpr char bits10 = '0' + (bits / 10);
  constexpr char bits1 = '0' + (bits % 10);

  // Scalars: omit the xN suffix.
  if (N == 1) {
    static constexpr char name1[8] = {prefix, bits1};
    static constexpr char name2[8] = {prefix, bits10, bits1};
    return sizeof(T) == 1 ? name1 : name2;
  }

  constexpr char N1 = (N < 10) ? '\0' : '0' + (N % 10);
  constexpr char N10 = (N < 10) ? '0' + (N % 10) : '0' + (N / 10);

  static constexpr char name1[8] = {prefix, bits1, 'x', N10, N1};
  static constexpr char name2[8] = {prefix, bits10, bits1, 'x', N10, N1};
  return sizeof(T) == 1 ? name1 : name2;
}

// Value to string

// We specialize for float/double below.
template <typename T>
inline std::string ToString(T value) {
  return std::to_string(value);
}

template <>
inline std::string ToString<float>(const float value) {
  // Ensure -0 and 0 are equivalent (required by some tests).
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7FFFFFFF) == 0) return "0";

  // to_string doesn't return enough digits and sstream is a
  // fairly large dependency (4KLOC).
  char buf[100];
  sprintf(buf, "%.8f", value);
  return buf;
}

template <>
inline std::string ToString<double>(const double value) {
  // Ensure -0 and 0 are equivalent (required by some tests).
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7FFFFFFFFFFFFFFFull) == 0) return "0";

  // to_string doesn't return enough digits and sstream is a
  // fairly large dependency (4KLOC).
  char buf[100];
  sprintf(buf, "%.16f", value);
  return buf;
}

// String comparison

template <typename T1, typename T2>
inline bool BytesEqual(const T1* p1, const T2* p2, const size_t size) {
  const uint8_t* bytes1 = reinterpret_cast<const uint8_t*>(p1);
  const uint8_t* bytes2 = reinterpret_cast<const uint8_t*>(p2);
  for (size_t i = 0; i < size; ++i) {
    if (bytes1[i] != bytes2[i]) return false;
  }
  return true;
}

inline bool StringsEqual(const char* s1, const char* s2) {
  while (*s1 == *s2++) {
    if (*s1++ == '\0') return true;
  }
  return false;
}

}  // namespace hwy

#endif  // HWY_TESTS_TEST_UTIL_H_
