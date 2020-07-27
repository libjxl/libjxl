// Copyright 2020 Google LLC
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

#ifndef HWY_HIGHWAY_H_
#define HWY_HIGHWAY_H_

// For SIMD module implementations; their callers only need base.h/targets.h.

#include <stddef.h>
#include <stdint.h>
#include <cmath>  // for scalar-inl.h

#include "hwy/targets.h"

// Clang 3.9 generates VINSERTF128 instead of the desired VBROADCASTF128,
// which would free up port5. However, inline assembly isn't supported on
// MSVC, results in incorrect output on GCC 8.3, and raises "invalid output size
// for constraint" errors on Clang (https://gcc.godbolt.org/z/-Jt_-F), hence we
// disable it.
#ifndef HWY_LOADDUP_ASM
#define HWY_LOADDUP_ASM 0
#endif

// Include platform-specific headers required by ops/*-inl.h. This must happen
// before the namespace hwy in this header, and cannot be done inside ops/*.h
// because those are potentially included inside the user's namespace.
#if HWY_TARGETS & (HWY_AVX2 | HWY_AVX3)
#include <immintrin.h>  // AVX2+
#elif HWY_TARGETS & HWY_SSE4
#include <smmintrin.h>  // SSE4
#elif HWY_TARGETS & HWY_WASM
#include <wasm_simd128.h>
#elif HWY_TARGETS & HWY_NEON
#include <arm_neon.h>
#endif

namespace hwy {

// Shorthand for implementations of Highway ops.
#define HWY_API HWY_INLINE HWY_FLATTEN HWY_MAYBE_UNUSED

// For functions in *-inl that use Highway (prevents IDE from showing as unused)
#define HWY_FUNC HWY_INLINE HWY_MAYBE_UNUSED

// Unfortunately the GCC/Clang intrinsics do not accept int64_t*.
using GatherIndex64 = long long int;  // NOLINT(google-runtime-int)
static_assert(sizeof(GatherIndex64) == 8, "Must be 64-bit type");

// The source/destination must not overlap/alias.
template <size_t kBytes, typename From, typename To>
HWY_INLINE void CopyBytes(const From* from, To* to) {
#if HWY_COMPILER_MSVC
  const uint8_t* HWY_RESTRICT from_bytes =
      reinterpret_cast<const uint8_t*>(from);
  uint8_t* HWY_RESTRICT to_bytes = reinterpret_cast<uint8_t*>(to);
  for (size_t i = 0; i < kBytes; ++i) {
    to_bytes[i] = from_bytes[i];
  }
#else
  // Avoids horrible codegen on Clang (series of PINSRB)
  __builtin_memcpy(to, from, kBytes);
#endif
}

static HWY_INLINE HWY_MAYBE_UNUSED size_t PopCount(const uint64_t x) {
#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
  return static_cast<size_t>(__builtin_popcountll(x));
#elif HWY_COMPILER_MSVC
  return _mm_popcnt_u64(x);
#else
#error "not supported"
#endif
}

//------------------------------------------------------------------------------
// Controlling overload resolution

// Insert into template/function arguments to enable this overload only for
// vectors of AT MOST this many bits.
//
// Note that enabling for exactly 128 bits is unnecessary because a function can
// simply be overloaded with Vec128<T> and Full128<T> descriptor. Enabling for
// other sizes (e.g. 64 bit) can be achieved with Simd<T, 8 / sizeof(T)>.
#define HWY_IF_LE128(T, N) hwy::EnableIf<N * sizeof(T) <= 16>* = nullptr
#define HWY_IF_LE64(T, N) hwy::EnableIf<N * sizeof(T) <= 8>* = nullptr
#define HWY_IF_LE32(T, N) hwy::EnableIf<N * sizeof(T) <= 4>* = nullptr

#define HWY_IF_FLOAT(T) hwy::EnableIf<hwy::IsFloat<T>()>* = nullptr

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

//------------------------------------------------------------------------------
// Conversion between types of the same size

// Unsigned/signed/floating-point types whose sizes are kSize bytes.
template <size_t kSize>
struct TypesOfSize;
template <>
struct TypesOfSize<1> {
  using Unsigned = uint8_t;
  using Signed = int8_t;
};
template <>
struct TypesOfSize<2> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
};
template <>
struct TypesOfSize<4> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
};
template <>
struct TypesOfSize<8> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
};

template <typename T>
using MakeUnsigned = typename TypesOfSize<sizeof(T)>::Unsigned;
template <typename T>
using MakeSigned = typename TypesOfSize<sizeof(T)>::Signed;
template <typename T>
using MakeFloat = typename TypesOfSize<sizeof(T)>::Float;

//------------------------------------------------------------------------------
// Descriptors

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Simd<T, N>. T is the lane type, N the requested number of
// lanes >= 1 (always a power of two). In the common case, users do not choose N
// directly, but instead use HWY_FULL (the largest available size). N may differ
// from the hardware vector size. If N is less, only that many lanes will be
// loaded/stored.
//
// Only HWY_FULL(T) and N <= 16 / sizeof(T) are guaranteed to be available - the
// latter are useful if >128 bit vectors are unnecessary or undesirable.
//
// Users should not use the N of a Simd<> but instead query the actual number of
// lanes via Lanes(). MaxLanes() is provided for template arguments and array
// dimensions, but this is discouraged because an upper bound might not exist.
template <typename Lane, size_t N>
struct Simd {
  constexpr Simd() = default;
  using T = Lane;
  static_assert((N & (N - 1)) == 0 && N != 0, "N must be a power of two");
};

#define HWY_FULL(T) hwy::Simd<T, HWY_LANES(T)>

// A vector of up to MAX_N lanes.
#define HWY_CAPPED(T, MAX_N) hwy::Simd<T, HWY_MIN(MAX_N, HWY_LANES(T))>

//------------------------------------------------------------------------------
// Export user functions for static/dynamic dispatch

// Evaluates to 0 inside a translation unit if it is generating anything but the
// static target (the last one if multiple targets are enabled). Prevents
// redefinitions of HWY_EXPORT and non-SIMD code outside begin/end_target.
#define HWY_ONCE ((HWY_TARGET == HWY_STATIC_TARGET) || HWY_IDE)

// HWY_STATIC_DISPATCH(FUNC_NAME) is the namespace-qualified FUNC_NAME for
// HWY_STATIC_TARGET (the only defined namespace unless HWY_TARGET_INCLUDE is
// defined), and can be used to deduce the return type of Choose*.
#if HWY_STATIC_TARGET == HWY_SCALAR
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_SCALAR::FUNC_NAME
#elif HWY_STATIC_TARGET == HWY_WASM
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_WASM::FUNC_NAME
#elif HWY_STATIC_TARGET == HWY_NEON
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_NEON::FUNC_NAME
#elif HWY_STATIC_TARGET == HWY_PPC8
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_PPC8::FUNC_NAME
#elif HWY_STATIC_TARGET == HWY_SSE4
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_SSE4::FUNC_NAME
#elif HWY_STATIC_TARGET == HWY_AVX2
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_AVX2::FUNC_NAME
#elif HWY_STATIC_TARGET == HWY_AVX3
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_AVX3::FUNC_NAME

#endif

// Dynamic dispatch declarations.

template <typename RetType, typename... Args>
struct FunctionCache {
 public:
  typedef RetType(FunctionType)(Args...);

  // A template function that when instantiated has the same signature as the
  // function being called. This function initializes the global cache of the
  // current supported targets mask used for dynamic dispatch and calls the
  // appropriate function. Since this mask used for dynamic dispatch is a
  // global cache, all the highway exported functions, even those exposed by
  // different modules, will be initialized after this function runs for any one
  // of those exported functions.
  template <FunctionType* const table[]>
  static RetType ChooseAndCall(Args... args) {
    // If we are running here it means we need to update the chosen target.
    chosen_target.Update();
    return (table[chosen_target.GetIndex()])(args...);
  }
};

// Factory function only used to infer the template parameters RetType and Args
// from a function passed to the factory.
template <typename RetType, typename... Args>
FunctionCache<RetType, Args...> FunctionCacheFactory(RetType (*)(Args...)) {
  return FunctionCache<RetType, Args...>();
}

// HWY_CHOOSE_*(FUNC_NAME) expands to the function pointer for that target or
// nullptr is that target was not compiled.
#if HWY_TARGETS & HWY_SCALAR
#define HWY_CHOOSE_SCALAR(FUNC_NAME) &N_SCALAR::FUNC_NAME
#else
// When scalar is not present and we try to use scalar because other targets
// were disabled at runtime we fall back to the baseline with
// HWY_STATIC_DISPATCH()
#define HWY_CHOOSE_SCALAR(FUNC_NAME) &HWY_STATIC_DISPATCH(FUNC_NAME)
#endif

#if HWY_TARGETS & HWY_WASM
#define HWY_CHOOSE_WASM(FUNC_NAME) &N_WASM::FUNC_NAME
#else
#define HWY_CHOOSE_WASM(FUNC_NAME) nullptr
#endif

#if HWY_TARGETS & HWY_NEON
#define HWY_CHOOSE_NEON(FUNC_NAME) &N_NEON::FUNC_NAME
#else
#define HWY_CHOOSE_NEON(FUNC_NAME) nullptr
#endif

#if HWY_TARGETS & HWY_PPC8
#define HWY_CHOOSE_PCC8(FUNC_NAME) &N_PPC8::FUNC_NAME
#else
#define HWY_CHOOSE_PPC8(FUNC_NAME) nullptr
#endif

#if HWY_TARGETS & HWY_SSE4
#define HWY_CHOOSE_SSE4(FUNC_NAME) &N_SSE4::FUNC_NAME
#else
#define HWY_CHOOSE_SSE4(FUNC_NAME) nullptr
#endif

#if HWY_TARGETS & HWY_AVX2
#define HWY_CHOOSE_AVX2(FUNC_NAME) &N_AVX2::FUNC_NAME
#else
#define HWY_CHOOSE_AVX2(FUNC_NAME) nullptr
#endif

#if HWY_TARGETS & HWY_AVX3
#define HWY_CHOOSE_AVX3(FUNC_NAME) &N_AVX3::FUNC_NAME
#else
#define HWY_CHOOSE_AVX3(FUNC_NAME) nullptr
#endif

#define HWY_DISPATCH_TABLE(FUNC_NAME) \
  HWY_CONCAT(FUNC_NAME, HighwayDispatchTable)

// HWY_EXPORT(FUNC_NAME) expands to a static array that is used by
// HWY_DYNAMIC_DISPATCH() to call the appropriate function at runtime. This
// static array must be defined at the same namespace level as the function
// it is exporting.
// After being exported, it can be called from other parts of the same source
// file using HWY_DYNAMIC_DISTPATCH(), in particular from a function wrapper
// like in the following example:
//
//   #include <hwy/before_namespace-inl.h>
//   namespace skeleton {
//   #include "hwy/begin_target-inl.h"
//
//   void MyFunction(int a, char b, const char* c) { ... }
//
//   #include "hwy/end_target-inl.h"
//   }  // namespace skeleton
//   #include <hwy/after_namespace-inl.h>
//
//   namespace skeleton {
//   HWY_EXPORT(MyFunction)  // Defines the dispatch table in this scope.
//
//   void MyFunction(int a, char b, const char* c) {
//     return HWY_DYNAMIC_DISPATCH(MyFunction)(a, b, c);
//   }
//   }  // namespace skeleton
//

#if HWY_IDE || ((HWY_TARGETS & (HWY_TARGETS - 1)) == 0)

// Simplified version for IDE or the dynamic dispatch case with only one target.
// This case still uses a table, although of a single element, to provide the
// same compile error conditions as with the dynamic dispatch case when multiple
// targets are being compiled.
#define HWY_EXPORT(FUNC_NAME)                                                \
  static decltype(&HWY_STATIC_DISPATCH(FUNC_NAME)) const HWY_DISPATCH_TABLE( \
      FUNC_NAME)[1] = {&HWY_STATIC_DISPATCH(FUNC_NAME)};
#define HWY_DYNAMIC_DISPATCH(FUNC_NAME) (*(HWY_DISPATCH_TABLE(FUNC_NAME)[0]))

#else

// Dynamic dispatch case with one entry per dynamic target plus the scalar
// mode and the initialization wrapper.
#define HWY_EXPORT(FUNC_NAME)                                              \
  static decltype(&HWY_STATIC_DISPATCH(FUNC_NAME))                         \
      const HWY_DISPATCH_TABLE(FUNC_NAME)[HWY_MAX_DYNAMIC_TARGETS + 2] = { \
          /* The first entry in the table initializes the global cache and \
           * calls the appropriate function. */                            \
          &decltype(hwy::FunctionCacheFactory(&HWY_STATIC_DISPATCH(        \
              FUNC_NAME)))::ChooseAndCall<HWY_DISPATCH_TABLE(FUNC_NAME)>,  \
          HWY_CHOOSE_TARGET_LIST(FUNC_NAME),                               \
          HWY_CHOOSE_SCALAR(FUNC_NAME),                                    \
  };
#define HWY_DYNAMIC_DISPATCH(FUNC_NAME) \
  (*(HWY_DISPATCH_TABLE(FUNC_NAME)[hwy::chosen_target.GetIndex()]))

#endif  // HWY_IDE || ((HWY_TARGETS & (HWY_TARGETS - 1)) == 0)

}  // namespace hwy

#endif  // HWY_HIGHWAY_H_
