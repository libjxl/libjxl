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

#include <stddef.h>
#include <stdint.h>
#include <atomic>
#include <cmath>  // for scalar-inl.h

#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
#include <stdio.h>
#endif

#include "hwy/interface.h"

//------------------------------------------------------------------------------
// Optional configuration

// See ../quick_reference.md for documentation of these macros.

// Uncomment to override the default baseline determined from predefined macros:
// #define HWY_BASELINE_TARGETS (HWY_SSE4 | HWY_SCALAR)

// Uncomment to override the default blacklist:
// #define HWY_BROKEN_TARGETS HWY_AVX3

// Uncomment to definitely avoid generating those target(s):
// #define HWY_DISABLED_TARGETS HWY_SSE4

// Uncomment to avoid emitting SSE2 cache-control instructions (useful for
// disabling all non-baseline instructions if HWY_DISABLE_* are also set)
// #define HWY_DISABLE_CACHE_CONTROL

// Uncomment to avoid emitting BMI/BMI2/FMA instructions (allows generating
// AVX2 target for VMs which support AVX2 but not the other instruction sets)
// #define HWY_DISABLE_BMI2_FMA

// Clang 3.9 generates VINSERTF128 instead of the desired VBROADCASTF128,
// which would free up port5. However, inline assembly isn't supported on
// MSVC, results in incorrect output on GCC 8.3, and raises "invalid output size
// for constraint" errors on Clang (https://gcc.godbolt.org/z/-Jt_-F), hence we
// disable it.
#ifndef HWY_LOADDUP_ASM
#define HWY_LOADDUP_ASM 0
#endif

//------------------------------------------------------------------------------
// Detect compiler using predefined macros

#ifdef _MSC_VER
#define HWY_COMPILER_MSVC _MSC_VER
#else
#define HWY_COMPILER_MSVC 0
#endif

#ifdef __GNUC__
#define HWY_COMPILER_GCC (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#define HWY_COMPILER_GCC 0
#endif

// Clang can masquerade as MSVC/GCC, in which case both are set.
#ifdef __clang__
#define HWY_COMPILER_CLANG (__clang_major__ * 100 + __clang_minor__)
#else
#define HWY_COMPILER_CLANG 0
#endif

// More than one may be nonzero, but we want at least one.
#if !HWY_COMPILER_MSVC && !HWY_COMPILER_GCC && !HWY_COMPILER_CLANG
#error "Unsupported compiler"
#endif

//------------------------------------------------------------------------------
// Compiler-specific definitions

#if HWY_COMPILER_MSVC

#define HWY_INLINE __forceinline
#define HWY_NOINLINE __declspec(noinline)
#define HWY_FLATTEN
#define HWY_NORETURN __declspec(noreturn)
#define HWY_LIKELY(expr) expr
#define HWY_TARGET_ATTR(feature_str)
#define HWY_DIAGNOSTICS(tokens) __pragma(warning(tokens))
#define HWY_DIAGNOSTICS_OFF(msc, gcc) HWY_DIAGNOSTICS(msc)
#define HWY_MAYBE_UNUSED

#else

#define HWY_INLINE inline __attribute__((always_inline))
#define HWY_NOINLINE __attribute__((noinline))
#define HWY_FLATTEN __attribute__((flatten))
#define HWY_NORETURN __attribute__((noreturn))
#define HWY_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define HWY_TARGET_ATTR(feature_str) __attribute__((target(feature_str)))
#define HWY_PRAGMA(tokens) _Pragma(#tokens)
#define HWY_DIAGNOSTICS(tokens) HWY_PRAGMA(GCC diagnostic tokens)
#define HWY_DIAGNOSTICS_OFF(msc, gcc) HWY_DIAGNOSTICS(gcc)
// Encountered "attribute list cannot appear here" when using the C++17
// [[maybe_unused]], so only use the old style attribute for now.
#define HWY_MAYBE_UNUSED __attribute__((unused))

#endif  // !HWY_COMPILER_MSVC

// Add to #if conditions to prevent IDE from graying out code.
#if (defined __CDT_PARSER__) || (defined __INTELLISENSE__) || \
    (defined Q_CREATOR_RUN)
#define HWY_IDE 1
#else
#define HWY_IDE 0
#endif

//------------------------------------------------------------------------------
// Target capabilities

// HWY_CAPS from begin_target-inl.h is zero or more of these arbitrary bits:

// 64-bit floating-point lanes
#define HWY_CAP_DOUBLE 1

// 64-bit signed/unsigned lanes
#define HWY_CAP_INT64 2

// Comparing 64-bit signed
#define HWY_CAP_CMP64 4

// Bit-shifting each lane by non-constant amounts
#define HWY_CAP_VARIABLE_SHIFT 8

// Per-lane loads with index or offset (GatherIndex/GatherOffset)
#define HWY_CAP_GATHER 16

// Vectors of at least 256 bits are available
#define HWY_CAP_GE256 32

// Vectors of at least 512 bits are available
#define HWY_CAP_GE512 64

//------------------------------------------------------------------------------
// Targets

// Unique bit value for each target. A lower value is "better" (e.g. more lanes)
// than a higher value within the same group/platform - see HWY_STATIC_TARGET.
//
// All values are unconditionally defined so we can test HWY_TARGETS without
// first checking the HWY_ARCH_*.
//
// The C99 preprocessor evaluates #if expressions using intmax_t types, so we
// can use 32-bit literals.

// 1,2,4: reserved
#define HWY_AVX3 8
#define HWY_AVX2 16
// 32: reserved for AVX
#define HWY_SSE4 64
// 0x80, 0x100, 0x200: reserved for SSSE3, SSE3, SSE2

// 0x400, 0x800, 0x1000 reserved for SVE, SVE2, Helium
#define HWY_NEON 0x2000

// 0x4000, 0x8000 reserved
#define HWY_PPC8 0x10000  // v2.07 or 3
// 0x20000, 0x40000 reserved for prior VSX/AltiVec

// 0x80000 reserved
#define HWY_WASM 0x100000

// 0x200000, 0x400000, 0x800000, 0x1000000, 0x2000000, 0x4000000, 0x8000000,
// 0x10000000 reserved

#define HWY_SCALAR 0x20000000
// Cannot use higher values, otherwise HWY_TARGETS computation might overflow.

//------------------------------------------------------------------------------
// Detect architecture using predefined macros

#if defined(__i386__) || defined(_M_IX86)
#define HWY_ARCH_X86_32 1
#else
#define HWY_ARCH_X86_32 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define HWY_ARCH_X86_64 1
#else
#define HWY_ARCH_X86_64 0
#endif

#if HWY_ARCH_X86_32 || HWY_ARCH_X86_64
#define HWY_ARCH_X86 1
#else
#define HWY_ARCH_X86 0
#endif

#if defined(__powerpc64__) || defined(_M_PPC)
#define HWY_ARCH_PPC 1
#else
#define HWY_ARCH_PPC 0
#endif

#if defined(__arm__) || defined(_M_ARM) || defined(__aarch64__)
#define HWY_ARCH_ARM 1
#else
#define HWY_ARCH_ARM 0
#endif

// There isn't yet a standard __wasm or __wasm__.
#ifdef __EMSCRIPTEN__
#define HWY_ARCH_WASM 1
#else
#define HWY_ARCH_WASM 0
#endif

#if HWY_ARCH_X86 + HWY_ARCH_PPC + HWY_ARCH_ARM + HWY_ARCH_WASM != 1
#error "Must detect exactly one platform"
#endif

//------------------------------------------------------------------------------
// Set default blacklist

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS 0
#endif

#ifndef HWY_BROKEN_TARGETS

// MSVC, early Clang, or 32-bit fail to compile AVX2/3.
#if (HWY_COMPILER_CLANG != 0 && HWY_COMPILER_CLANG < 700) || \
    HWY_COMPILER_MSVC != 0 || HWY_ARCH_X86_32
#define HWY_BROKEN_TARGETS (HWY_AVX2 | HWY_AVX3)
#else
#define HWY_BROKEN_TARGETS 0
#endif

#endif  // HWY_BROKEN_TARGETS

//------------------------------------------------------------------------------
// Detect baseline targets using predefined macros

// These are interpreted as the targets for which the compiler is allowed to
// generate instructions, implying the target CPU would have to support them.

#ifndef HWY_BASELINE_TARGETS

#ifdef __wasm_simd128__
#define HWY_BASELINE_WASM HWY_WASM
#else
#define HWY_BASELINE_WASM 0
#endif

#ifdef __VSX__
#define HWY_BASELINE_PPC8 HWY_PPC8
#else
#define HWY_BASELINE_PPC8 0
#endif

// GCC 4.5.4 only defines the former; 5.4 defines both.
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define HWY_BASELINE_NEON HWY_NEON
#else
#define HWY_BASELINE_NEON 0
#endif

#ifndef __SSE2__
#undef HWY_DISABLE_CACHE_CONTROL
#define HWY_DISABLE_CACHE_CONTROL
#endif

#ifdef __SSE4_1__
#define HWY_BASELINE_SSE4 HWY_SSE4
#else
#define HWY_BASELINE_SSE4 0
#endif

#ifdef __AVX2__
#define HWY_BASELINE_AVX2 HWY_AVX2
#else
#define HWY_BASELINE_AVX2 0
#endif

#ifdef __AVX512F__
#define HWY_BASELINE_AVX3 HWY_AVX3
#else
#define HWY_BASELINE_AVX3 0
#endif

#define HWY_BASELINE_TARGETS                                                \
  (HWY_SCALAR | HWY_BASELINE_WASM | HWY_BASELINE_PPC8 | HWY_BASELINE_NEON | \
   HWY_BASELINE_SSE4 | HWY_BASELINE_AVX2 | HWY_BASELINE_AVX3)

#endif  // HWY_BASELINE_TARGETS

// Apply blacklist (-mavx2 does not imply we can successfully compile AVX2 -
// compiler bugs may prevent it). Use this instead of HWY_BASELINE_TARGETS.
// These determine HWY_STATIC_TARGET, and we also exclude any remaining ones
// from HWY_TARGETS.
#define HWY_ENABLED_BASELINE \
  ((HWY_BASELINE_TARGETS) & ~((HWY_DISABLED_TARGETS) | (HWY_BROKEN_TARGETS)))

#if HWY_ENABLED_BASELINE == 0
#error "At least one baseline target must be defined and enabled"
#endif

//------------------------------------------------------------------------------
// Choose target for static and targets for dynamic

// Least-significant 1-bit within HWY_ENABLED_BASELINE. Because lower target
// values imply "better", this is the best baseline, i.e. the one that will be
// used for static dispatch.
#define HWY_STATIC_TARGET (HWY_ENABLED_BASELINE & -HWY_ENABLED_BASELINE)

// Start by assuming static dispatch. If we later use dynamic dispatch, this
// will be defined to other targets during the multiple-inclusion, and finally
// return to the initial value. Defining this outside begin/end_target ensures
// inl headers successfully compile by themselves (required by Bazel).
#define HWY_TARGET HWY_STATIC_TARGET

#ifdef HWY_BASELINE_TARGET_ONLY

// Include best baseline only. Useful for converting existing dynamic-dispatch
// code to static without having to remove all foreach_target/HWY_EXPORT etc.
#define HWY_TARGETS HWY_STATIC_TARGET

#else

// HWY_ENABLED_TARGETS is a mask that prevents setting unavailable bits in
// HWY_TARGET, which would lead to invalid HWY_TARGET.
#if HWY_ARCH_X86
// No compiler support needed (beyond not being broken/disabled).
#define HWY_ENABLED_TARGETS                        \
  ((HWY_SCALAR | HWY_SSE4 | HWY_AVX2 | HWY_AVX3) & \
   ~((HWY_BROKEN_TARGETS) | (HWY_DISABLED_TARGETS)))
#else
// Cannot use unless enabled in the compiler.
#define HWY_ENABLED_TARGETS HWY_ENABLED_BASELINE
#endif

// Best baseline and any better (i.e. lower) _enabled_ target.
#define HWY_TARGETS ((2 * HWY_STATIC_TARGET - 1) & HWY_ENABLED_TARGETS)

#endif  // HWY_BASELINE_TARGET_ONLY

#if HWY_TARGETS == 0
#error "At least one target must be defined"
#endif

// HWY_ONCE and the multiple-inclusion mechanism rely on HWY_STATIC_TARGET being
// one of the dynamic targets, so verify that.
#if (HWY_TARGETS & HWY_STATIC_TARGET) == 0
#error "Logic error: best baseline should be included in dynamic targets"
#endif

//------------------------------------------------------------------------------
// Include platform-specific headers (before namespace)

#if HWY_TARGETS & (HWY_AVX2 | HWY_AVX3)
#include <immintrin.h>  // AVX2+
#elif HWY_TARGETS & HWY_SSE4
#include <smmintrin.h>  // SSE4
#elif HWY_ARCH_X86 && !defined(HWY_DISABLE_CACHE_CONTROL)
#include <emmintrin.h>  // SSE2 for cache control
#endif

#if HWY_COMPILER_MSVC
#include <intrin.h>
#endif

namespace hwy {

//------------------------------------------------------------------------------
// Subset of type_traits / numeric_limits for faster compilation.

template <bool Condition, class T>
struct EnableIfT {};
template <class T>
struct EnableIfT<true, T> {
  using type = T;
};

template <bool Condition, class T = void>
using EnableIf = typename EnableIfT<Condition, T>::type;

template <typename T>
constexpr bool IsFloat() {
  return T(1.25) != T(1);
}

template <typename T>
constexpr bool IsSigned() {
  return T(0) > T(-1);
}

// Insert into template/function arguments to enable this overload only for
// vectors of AT MOST this many bits.
//
// Note that enabling for exactly 128 bits is unnecessary because a function can
// simply be overloaded with Vec128<T> and Full128<T> descriptor. Enabling for
// other sizes (e.g. 64 bit) can be achieved with Desc<T, 8 / sizeof(T)>.
#define HWY_IF_LE128(T, N) hwy::EnableIf<N * sizeof(T) <= 16>* = nullptr
#define HWY_IF_LE64(T, N) hwy::EnableIf<N * sizeof(T) <= 8>* = nullptr
#define HWY_IF_LE32(T, N) hwy::EnableIf<N * sizeof(T) <= 4>* = nullptr

#define HWY_IF_FLOAT(T) hwy::EnableIf<hwy::IsFloat<T>()>* = nullptr

// Largest/smallest representable integer values.
template <typename T>
constexpr T LimitsMax() {
  return IsSigned<T>() ? T((1ULL << (sizeof(T) * 8 - 1)) - 1)
                       : static_cast<T>(~0ull);
}
template <typename T>
constexpr T LimitsMin() {
  return IsSigned<T>() ? T(-1) - LimitsMax<T>() : T(0);
}

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

// The unsigned integer type whose size is kSize bytes.
template <size_t kSize>
struct MakeUnsignedT;
template <>
struct MakeUnsignedT<1> {
  using type = uint8_t;
};
template <>
struct MakeUnsignedT<2> {
  using type = uint16_t;
};
template <>
struct MakeUnsignedT<4> {
  using type = uint32_t;
};
template <>
struct MakeUnsignedT<8> {
  using type = uint64_t;
};

template <typename T>
using MakeUnsigned = typename MakeUnsignedT<sizeof(T)>::type;

// The signed integer type whose size is kSize bytes.
template <size_t kSize>
struct MakeSignedT;
template <>
struct MakeSignedT<1> {
  using type = int8_t;
};
template <>
struct MakeSignedT<2> {
  using type = int16_t;
};
template <>
struct MakeSignedT<4> {
  using type = int32_t;
};
template <>
struct MakeSignedT<8> {
  using type = int64_t;
};

template <typename T>
using MakeSigned = typename MakeSignedT<sizeof(T)>::type;

// The floating-point type whose size is kSize bytes.
template <size_t kSize>
struct MakeFloatT;
template <>
struct MakeFloatT<4> {
  using type = float;
};
template <>
struct MakeFloatT<8> {
  using type = double;
};

template <typename T>
using MakeFloat = typename MakeFloatT<sizeof(T)>::type;

//------------------------------------------------------------------------------
// Definitions shared between target-specific headers and possibly also users.

// Shorthand for implementations of Highway ops. Note that x86_128-inl uses the
// _current_ HWY_ATTR instead of HWY_ATTR_SSE4 to enable use of VL.
#define HWY_API HWY_ATTR HWY_INLINE HWY_FLATTEN

// For functions in *-inl that use Highway (prevents IDE from showing as unused)
#define HWY_FUNC HWY_ATTR HWY_INLINE HWY_MAYBE_UNUSED

#define HWY_STR_IMPL(macro) #macro
#define HWY_STR(macro) HWY_STR_IMPL(macro)

#define HWY_CONCAT_IMPL(a, b) a##b
#define HWY_CONCAT(a, b) HWY_CONCAT_IMPL(a, b)

#define HWY_MIN(a, b) ((a) < (b) ? (a) : (b))
#define HWY_MAX(a, b) ((a) < (b) ? (b) : (a))

// Alternative for asm volatile("" : : : "memory"), which has no effect.
#define HWY_FENCE std::atomic_thread_fence(std::memory_order_acq_rel)

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Desc<T, N>. T is the lane type, N the number of lanes >= 1.
template <typename Lane, size_t kNumLanes>
struct Desc {
  constexpr Desc() = default;
  using T = Lane;
  static constexpr size_t N = kNumLanes;
  static_assert((N & (N - 1)) == 0 && N != 0, "N must be a power of two");
};

// Avoids linker errors in pre-C++17 debug builds.
template <typename Lane, size_t kNumLanes>
constexpr size_t Desc<Lane, kNumLanes>::N;

#define HWY_FULL(T) hwy::Desc<T, HWY_LANES(T)>

// A vector of up to MAX_N lanes.
#define HWY_CAPPED(T, MAX_N) hwy::Desc<T, HWY_MIN(MAX_N, HWY_LANES(T))>

// Alias for the actual vector data, e.g. Vec0<float> for Desc<float, 0>,
// To avoid inadvertent conversions between vectors of different lengths, they
// have distinct types (Vec128<T, N>) on x86. Must use a macro to defer
// expansion until after Zero<> has been defined.
#define HWY_VEC(D) decltype(Zero(D()))

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
#if HWY_COMPILER_MSVC
  return _mm_popcnt_u64(x);
#else
  return static_cast<size_t>(__builtin_popcountll(x));
#endif
}

static HWY_INLINE HWY_MAYBE_UNUSED HWY_NORETURN void Trap() {
#if HWY_COMPILER_MSVC
  __debugbreak();
#else
  __builtin_trap();
#endif
}

static inline HWY_MAYBE_UNUSED const char* TargetName(int32_t target) {
  switch (target) {
#if HWY_ARCH_X86
    case HWY_SSE4:
      return "SSE4";
    case HWY_AVX2:
      return "AVX2";
    case HWY_AVX3:
      return "AVX3";
#endif

#if HWY_ARCH_ARM
    case HWY_NEON:
      return "Neon";
#endif

#if HWY_ARCH_PPC
    case HWY_PPC8:
      return "Power8";
#endif

#if HWY_ARCH_WASM
    case HWY_WASM:
      return "Wasm";
#endif

    case HWY_SCALAR:
      return "Scalar";

    default:
      return "?";
  }
}

//------------------------------------------------------------------------------
// Cache control

// Even if N*sizeof(T) is smaller, Stream may write a multiple of this size.
#define HWY_STREAM_MULTIPLE 16

// The following functions may also require an attribute.
#if HWY_ARCH_X86 && !defined(HWY_DISABLE_CACHE_CONTROL)
#define HWY_ATTR_CACHE HWY_TARGET_ATTR("sse2")
#else
#define HWY_ATTR_CACHE
#endif

// Delays subsequent loads until prior loads are visible. On Intel CPUs, also
// serves as a full fence (waits for all prior instructions to complete).
// No effect on non-x86.
HWY_INLINE HWY_ATTR_CACHE void LoadFence() {
#if HWY_ARCH_X86 && !defined(HWY_DISABLE_CACHE_CONTROL)
  _mm_lfence();
#endif
}

// Ensures previous weakly-ordered stores are visible. No effect on non-x86.
HWY_INLINE HWY_ATTR_CACHE void StoreFence() {
#if HWY_ARCH_X86 && !defined(HWY_DISABLE_CACHE_CONTROL)
  _mm_sfence();
#endif
}

// Begins loading the cache line containing "p".
template <typename T>
HWY_INLINE HWY_ATTR_CACHE void Prefetch(const T* p) {
#if HWY_ARCH_X86 && !defined(HWY_DISABLE_CACHE_CONTROL)
  _mm_prefetch(const_cast<T*>(p), _MM_HINT_T0);
#elif HWY_COMPILER_GCC || HWY_COMPILER_CLANG
  // Hint=0 (NTA) behavior differs, but skipping outer caches is probably not
  // desirable, so use the default 3 (keep in caches).
  __builtin_prefetch(p, /*write=*/0, /*hint=*/3);
#else
  (void)p;
#endif
}

// Invalidates and flushes the cache line containing "p". No effect on non-x86.
HWY_INLINE HWY_ATTR_CACHE void FlushCacheline(const void* p) {
#if HWY_ARCH_X86 && !defined(HWY_DISABLE_CACHE_CONTROL)
  _mm_clflush(p);
#else
  (void)p;
#endif
}

//------------------------------------------------------------------------------
// Export user functions for static/dynamic dispatch

// Evaluates to 0 inside a translation unit that includes foreach_target,
// until the final target is being generated. Used to prevent multiple
// definitions of HWY_EXPORT and non-SIMD code outside begin/end_target.
#define HWY_ONCE ((HWY_TARGET == HWY_STATIC_TARGET) || HWY_IDE)

// HWY_EXPORT(FUNC_NAME) defines a ChooseFUNC_NAME function consisting of one
// HWY_CHOOSE_$ conditional return statement per enabled target.
//
// HWY_STATIC_DISPATCH(FUNC_NAME) is the namespace-qualified FUNC_NAME for
// HWY_STATIC_TARGET (the only defined namespace unless foreach_targets.h is
// included), and can be used to deduce the return type of Choose*.
#if HWY_TARGETS & HWY_SCALAR
#define HWY_CHOOSE_SCALAR(FUNC_NAME) \
  if (targets_bits & HWY_SCALAR) return &N_SCALAR::FUNC_NAME;
#if HWY_STATIC_TARGET == HWY_SCALAR
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_SCALAR::FUNC_NAME
#endif

#else
#define HWY_CHOOSE_SCALAR(FUNC_NAME)
#endif

#if HWY_TARGETS & HWY_WASM
#define HWY_ATTR_WASM HWY_TARGET_ATTR("simd128")
#define HWY_CHOOSE_WASM(FUNC_NAME) \
  if (targets_bits & HWY_WASM) return &N_WASM::FUNC_NAME;
#if HWY_STATIC_TARGET == HWY_WASM
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_WASM::FUNC_NAME
#endif

#else
#define HWY_CHOOSE_WASM(FUNC_NAME)
#endif

#if HWY_TARGETS & HWY_NEON
#define HWY_ATTR_NEON HWY_TARGET_ATTR("crypto")
#define HWY_CHOOSE_NEON(FUNC_NAME) \
  if (targets_bits & HWY_NEON) return &N_NEON::FUNC_NAME;
#if HWY_STATIC_TARGET == HWY_NEON
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_NEON::FUNC_NAME
#endif

#else
#define HWY_CHOOSE_NEON(FUNC_NAME)
#endif

#if HWY_TARGETS & HWY_PPC8
#define HWY_CHOOSE_PCC8(FUNC_NAME) \
  if (targets_bits & HWY_PPC8) return &N_PPC8::FUNC_NAME;
#if HWY_STATIC_TARGET == HWY_PPC8
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_PPC8::FUNC_NAME
#endif

#else
#define HWY_CHOOSE_PPC8(FUNC_NAME)
#endif

#if HWY_TARGETS & HWY_SSE4
#define HWY_ATTR_SSE4 HWY_TARGET_ATTR("sse2,ssse3,sse4.1")
#define HWY_CHOOSE_SSE4(FUNC_NAME) \
  if (targets_bits & HWY_SSE4) return &N_SSE4::FUNC_NAME;
#if HWY_STATIC_TARGET == HWY_SSE4
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_SSE4::FUNC_NAME
#endif

#else
#define HWY_CHOOSE_SSE4(FUNC_NAME)
#endif

#if HWY_TARGETS & HWY_AVX2
#ifdef HWY_DISABLE_BMI2_FMA  // See supported_targets.cc
#define HWY_ATTR_AVX2 HWY_TARGET_ATTR("avx,avx2")
#else
#define HWY_ATTR_AVX2 HWY_TARGET_ATTR("avx,avx2,bmi,bmi2,fma")
#endif

#define HWY_CHOOSE_AVX2(FUNC_NAME) \
  if (targets_bits & HWY_AVX2) return &N_AVX2::FUNC_NAME;
#if HWY_STATIC_TARGET == HWY_AVX2
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_AVX2::FUNC_NAME
#endif

#else
#define HWY_CHOOSE_AVX2(FUNC_NAME)
#endif

#if HWY_TARGETS & HWY_AVX3
// Must include contents of HWY_ATTR_AVX2 because an AVX-512 test may call
// AVX2 functions (e.g. when converting to half-vectors).
// HWY_DISABLE_BMI2_FMA is not relevant because if we have AVX-512, we should
// also have BMI2/FMA.
#define HWY_ATTR_AVX3 \
  HWY_TARGET_ATTR("avx,avx2,bmi,bmi2,fma,avx512f,avx512vl,avx512dq,avx512bw")

#define HWY_CHOOSE_AVX3(FUNC_NAME) \
  if (targets_bits & HWY_AVX3) return &N_AVX3::FUNC_NAME;
#if HWY_STATIC_TARGET == HWY_AVX3
#define HWY_STATIC_DISPATCH(FUNC_NAME) N_AVX3::FUNC_NAME
#endif

#else
#define HWY_CHOOSE_AVX3(FUNC_NAME)
#endif

// Simplified version for IDE: prevents unused-function warning for the exported
// function while also avoiding warnings about undefined namespaces.
#if HWY_IDE
#define HWY_EXPORT(FUNC_NAME)                               \
  decltype(&HWY_STATIC_DISPATCH(FUNC_NAME)) HWY_CONCAT(     \
      Choose, FUNC_NAME)(const uint32_t /*targets_bits*/) { \
    return &HWY_STATIC_DISPATCH(FUNC_NAME);                 \
  }

#else

// Expands to a Choose* function that returns a pointer to a function inside
// one of the target-specific namespaces (the best one whose bit in
// `targets_bits` is set).
#define HWY_EXPORT(FUNC_NAME)                                     \
  decltype(&HWY_STATIC_DISPATCH(FUNC_NAME)) HWY_CONCAT(           \
      Choose, FUNC_NAME)(const uint32_t targets_bits) {           \
    /* In priority order because these may return immediately. */ \
    HWY_CHOOSE_WASM(FUNC_NAME)                                    \
    HWY_CHOOSE_NEON(FUNC_NAME)                                    \
    HWY_CHOOSE_PPC8(FUNC_NAME)                                    \
    HWY_CHOOSE_AVX3(FUNC_NAME)                                    \
    HWY_CHOOSE_AVX2(FUNC_NAME)                                    \
    HWY_CHOOSE_SSE4(FUNC_NAME)                                    \
    HWY_CHOOSE_SCALAR(FUNC_NAME)                                  \
    /* No overlap between HWY_TARGETS and targets_bits. */        \
    /* This is a logic error. */                                  \
    hwy::Trap();                                                  \
  }

#endif  // HWY_IDE

}  // namespace hwy

#endif  // HWY_HIGHWAY_H_
