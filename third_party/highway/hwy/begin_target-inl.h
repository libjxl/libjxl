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

// Include guard (required when re-including the TU) - will be undefined in
// end_target-inl.h so this header is active again for the next target.
#ifndef HWY_BEGIN_TARGET_INL_H_
#define HWY_BEGIN_TARGET_INL_H_

// IDE is parsing only this header, or user forgot to include header:
#if !defined(HWY_TARGET)
#include "hwy/highway.h"  // only for IDE - avoids warnings.
#define HWY_END_NAMESPACE_FOR_IDE
#if !HWY_IDE
// Users must have included before_namespace-inl.h - we can't do it here because
// this may be included inside the project's namespace.)
#error "Must include before_namespace-inl.h before begin_target-inl.h"
#endif  // !HWY_IDE
#endif  // !HWY_TARGET

// Prepare for the implementation section of the file that included us:
// 1) Define HWY_ALIGN etc. based on HWY_TARGET.
// 2) Open a target-specific namespace, later closed by end_target-inl.h.
// 3) Define ops functions inside that namespace.

// NOTE: the HWY_OPS_* are external include guards. These protect the ops/*-inl
// headers from re-inclusion if there are multiple begin/end per translation
// unit (e.g. if there are any *-inl.h headers). We cannot use the toggle-guard
// mechanism used by other *-inl.h headers because they only work if their
// contents are included for EVERY target, whereas only some ops headers
// include others as dependencies (e.g. x86_256 -> x86_128).

//-----------------------------------------------------------------------------
// SSE4
#if HWY_TARGET == HWY_SSE4

#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#define HWY_CAP_GATHER 0
#define HWY_CAP_VARIABLE_SHIFT 0
#define HWY_CAP_INT64 1
#define HWY_CAP_CMP64 0
#define HWY_CAP_DOUBLE 1
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

namespace N_SSE4 {
namespace {

#if !defined(HWY_OPS_X86_128) && !defined(HWY_NESTED_BEGIN)
#define HWY_OPS_X86_128
#include "hwy/ops/x86_128-inl.h"
#endif  // HWY_OPS_X86_128

//-----------------------------------------------------------------------------
// AVX2
#elif HWY_TARGET == HWY_AVX2

#define HWY_ALIGN alignas(32)
#define HWY_LANES(T) (32 / sizeof(T))

#define HWY_CAP_GATHER 1
#define HWY_CAP_VARIABLE_SHIFT 1
#define HWY_CAP_INT64 1
#define HWY_CAP_CMP64 1
#define HWY_CAP_DOUBLE 1
#define HWY_CAP_GE256 1
#define HWY_CAP_GE512 0

namespace N_AVX2 {
namespace {

#if !defined(HWY_OPS_X86_256) && !defined(HWY_NESTED_BEGIN)
#define HWY_OPS_X86_256
#include "hwy/ops/x86_256-inl.h"
#endif  // HWY_OPS_X86_256

//-----------------------------------------------------------------------------
// AVX3
#elif HWY_TARGET == HWY_AVX3

#define HWY_ALIGN alignas(64)
#define HWY_LANES(T) (64 / sizeof(T))

#define HWY_CAP_GATHER 1
#define HWY_CAP_VARIABLE_SHIFT 1
#define HWY_CAP_INT64 1
#define HWY_CAP_CMP64 1
#define HWY_CAP_DOUBLE 1
#define HWY_CAP_GE256 1
#define HWY_CAP_GE512 1

namespace N_AVX3 {
namespace {

#if !defined(HWY_OPS_X86_512) && !defined(HWY_NESTED_BEGIN)
#define HWY_OPS_X86_512
#include "hwy/ops/x86_512-inl.h"
#endif  // HWY_OPS_X86_512

//-----------------------------------------------------------------------------
// PPC8
#elif HWY_TARGET == HWY_PPC8

#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#define HWY_CAP_GATHER 0
#define HWY_CAP_VARIABLE_SHIFT 1
#define HWY_CAP_INT64 1
#define HWY_CAP_CMP64 1
#define HWY_CAP_DOUBLE 1
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

namespace N_PPC8 {
namespace {

//-----------------------------------------------------------------------------
// NEON
#elif HWY_TARGET == HWY_NEON

#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#define HWY_CAP_VARIABLE_SHIFT 1
#define HWY_CAP_GATHER 0
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

#ifdef __arm__
#define HWY_CAP_INT64 0
#define HWY_CAP_CMP64 0
#define HWY_CAP_DOUBLE 0
#else
#define HWY_CAP_INT64 1
#define HWY_CAP_CMP64 1
#define HWY_CAP_DOUBLE 1
#endif

namespace N_NEON {
namespace {

#if !defined(HWY_OPS_ARM_NEON) && !defined(HWY_NESTED_BEGIN)
#define HWY_OPS_ARM_NEON
#include "hwy/ops/arm_neon-inl.h"
#endif  // HWY_OPS_ARM_NEON

//-----------------------------------------------------------------------------
// WASM
#elif HWY_TARGET == HWY_WASM

#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#define HWY_CAP_GATHER 0
#define HWY_CAP_VARIABLE_SHIFT 0
#define HWY_CAP_INT64 0
#define HWY_CAP_CMP64 0
#define HWY_CAP_DOUBLE 0
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

namespace N_WASM {
namespace {

#if !defined(HWY_OPS_WASM_128) && !defined(HWY_NESTED_BEGIN)
#define HWY_OPS_WASM_128
#include "hwy/ops/wasm_128-inl.h"
#endif  // HWY_OPS_WASM_128

//-----------------------------------------------------------------------------
// SCALAR
#elif HWY_TARGET == HWY_SCALAR

#define HWY_ALIGN
#define HWY_LANES(T) 1

#define HWY_CAP_GATHER 1
#define HWY_CAP_VARIABLE_SHIFT 1
#define HWY_CAP_INT64 1
#define HWY_CAP_CMP64 1
#define HWY_CAP_DOUBLE 1
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

namespace N_SCALAR {
namespace {

// No pragma attribute needed.

#if !defined(HWY_OPS_SCALAR) && !defined(HWY_NESTED_BEGIN)
#define HWY_OPS_SCALAR
#include "hwy/ops/scalar-inl.h"
#endif  // HWY_OPS_SCALAR

#else
#pragma message("HWY_TARGET does not match any known target")
#endif  // HWY_TARGET

//-----------------------------------------------------------------------------
// Shared definitions common to all targets

#if defined(HWY_BEGIN_TARGET_INL_SHARED) == defined(HWY_TARGET_TOGGLE)
#ifdef HWY_BEGIN_TARGET_INL_SHARED
#undef HWY_BEGIN_TARGET_INL_SHARED
#else
#define HWY_BEGIN_TARGET_INL_SHARED
#endif

// Returns the closest value to v within [lo, hi].
template <class V>
HWY_API V Clamp(const V v, const V lo, const V hi) {
  return Min(Max(lo, v), hi);
}

// Corresponding vector type, e.g. Vec128<float> for Simd<float, 4>,
template <class D>
using Vec = decltype(Zero(D()));

using U8xN = Vec<HWY_FULL(uint8_t)>;
using U16xN = Vec<HWY_FULL(uint16_t)>;
using U32xN = Vec<HWY_FULL(uint32_t)>;
using U64xN = Vec<HWY_FULL(uint64_t)>;

using I8xN = Vec<HWY_FULL(int8_t)>;
using I16xN = Vec<HWY_FULL(int16_t)>;
using I32xN = Vec<HWY_FULL(int32_t)>;
using I64xN = Vec<HWY_FULL(int64_t)>;

using F32xN = Vec<HWY_FULL(float)>;
using F64xN = Vec<HWY_FULL(double)>;

// Compile-time-constant upper bound (even for variable-length vectors), useful
// for array dimensions.
template <typename T, size_t N>
HWY_INLINE HWY_MAYBE_UNUSED constexpr size_t MaxLanes(hwy::Simd<T, N>) {
  return N;
}

// (Potentially) non-constant actual size of the vector at runtime, subject to
// the limit imposed by the Simd. Useful for advancing loop counters.
template <typename T, size_t N>
HWY_INLINE HWY_MAYBE_UNUSED size_t Lanes(hwy::Simd<T, N>) {
  return N;
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <class D, typename T2>
Vec<D> Iota(const D d, const T2 first) {
  using T = typename D::T;
  HWY_ALIGN T lanes[MaxLanes(d)];
  for (size_t i = 0; i < Lanes(d); ++i) {
    lanes[i] = first + static_cast<T2>(i);
  }
  return Load(d, lanes);
}
#endif  // HWY_BEGIN_TARGET_INL_SHARED

// (Followed by user's SIMD code, then #include end_target-inl.h)

// For IDE only: avoid unmatched-brace warning.
#if defined(HWY_END_NAMESPACE_FOR_IDE) || defined(HWY_NESTED_BEGIN)
#undef HWY_END_NAMESPACE_FOR_IDE
}  // namespace
}  // namespace N_$TARGET
#endif  // HWY_END_NAMESPACE_FOR_IDE

#endif  // include guard
