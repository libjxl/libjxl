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

// NOTE: atypical include guard.

// For nested inclusion via ops/*-inl.h, do nothing. This allows those headers
// to include their own begin/end_target-inl.h to avoid IDE warnings, without
// affecting the compilation of other headers/TU that also include
// begin/end_target-inl.h.
#ifdef HWY_BEGIN_TARGET_INL_H_

// Allowing arbitrary begin/end nesting would require a counter, but we only
// provide a single guard macro. That is sufficient because users include their
// -inl.h outside the begin/end block.
#ifdef HWY_BEGIN_TARGET_NESTED
#error "More than one recursive begin_target-inl.h inclusion is not allowed"
#endif
#define HWY_BEGIN_TARGET_NESTED

#else  // not nested:
// Include guard. This only prevents recursion and will be undefined in
// end_target-inl.h so this header is active again for the next target.
#define HWY_BEGIN_TARGET_INL_H_

// IDE is parsing only this header, or user forgot to include header:
#if !defined(HWY_TARGET)
#include "hwy/highway.h"  // only for IDE - avoids warnings.
#define HWY_END_NAMESPACE_FOR_IDE
#if !HWY_IDE
// Users must include foreach_target.h - we can't do it here when compiling
// because begin_target-inl.h may be included inside the project's namespace,
// whereas foreach_target.h must be in the global namespace.)
#error "Must include foreach_target.h before begin_target-inl.h"
#endif  // !HWY_IDE
#endif  // !HWY_*_TARGETS

// Prepare for the implementation section of the file that included us:
// 1) Define HWY_ATTR etc. based on HWY_TARGET.
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

#define HWY_ATTR HWY_ATTR_SSE4
#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))
#define HWY_CAPS (HWY_CAP_DOUBLE | HWY_CAP_INT64)

namespace N_SSE4 {
namespace {

#ifndef HWY_OPS_X86_128
#define HWY_OPS_X86_128
#include "hwy/ops/x86_128-inl.h"
#endif  // HWY_OPS_X86_128

//-----------------------------------------------------------------------------
// AVX2
#elif HWY_TARGET == HWY_AVX2

#define HWY_ATTR HWY_ATTR_AVX2
#define HWY_ALIGN alignas(32)
#define HWY_LANES(T) (32 / sizeof(T))
#define HWY_CAPS                                                             \
  (HWY_CAP_DOUBLE | HWY_CAP_INT64 | HWY_CAP_CMP64 | HWY_CAP_VARIABLE_SHIFT | \
   HWY_CAP_GATHER | HWY_CAP_GE256)

namespace N_AVX2 {
namespace {

#ifndef HWY_OPS_X86_256
#define HWY_OPS_X86_256
#include "hwy/ops/x86_256-inl.h"
#endif  // HWY_OPS_X86_256

//-----------------------------------------------------------------------------
// AVX3
#elif HWY_TARGET == HWY_AVX3

#define HWY_ATTR HWY_ATTR_AVX3
#define HWY_ALIGN alignas(64)
#define HWY_LANES(T) (64 / sizeof(T))
#define HWY_CAPS                                                             \
  (HWY_CAP_DOUBLE | HWY_CAP_INT64 | HWY_CAP_CMP64 | HWY_CAP_VARIABLE_SHIFT | \
   HWY_CAP_GATHER | HWY_CAP_GE256 | HWY_CAP_GE512)

namespace N_AVX3 {
namespace {

#ifndef HWY_OPS_X86_512
#define HWY_OPS_X86_512
#include "hwy/ops/x86_512-inl.h"
#endif  // HWY_OPS_X86_512

//-----------------------------------------------------------------------------
// PPC8
#elif HWY_TARGET == HWY_PPC8

#define HWY_ATTR
#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))
#define HWY_CAPS \
  (HWY_CAP_DOUBLE | HWY_CAP_INT64 | HWY_CAP_CMP64 | HWY_CAP_VARIABLE_SHIFT)

namespace N_PPC8 {
namespace {

//-----------------------------------------------------------------------------
// NEON
#elif HWY_TARGET == HWY_NEON

#define HWY_ATTR HWY_ATTR_NEON
#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#ifdef __arm__
#define HWY_CAPS HWY_CAP_VARIABLE_SHIFT
#else
#define HWY_CAPS \
  (HWY_CAP_DOUBLE | HWY_CAP_INT64 | HWY_CAP_CMP64 | HWY_CAP_VARIABLE_SHIFT)
#endif

namespace N_NEON {
namespace {

#ifndef HWY_OPS_ARM_NEON
#define HWY_OPS_ARM_NEON
#include "hwy/ops/arm_neon-inl.h"
#endif  // HWY_OPS_ARM_NEON

//-----------------------------------------------------------------------------
// WASM
#elif HWY_TARGET == HWY_WASM

#define HWY_ATTR HWY_ATTR_WASM
#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))
#define HWY_CAPS 0 /* none */

namespace N_WASM {
namespace {

#ifndef HWY_OPS_WASM_128
#define HWY_OPS_WASM_128
#include "hwy/ops/wasm_128-inl.h"
#endif  // HWY_OPS_WASM_128

//-----------------------------------------------------------------------------
// SCALAR
#elif HWY_TARGET == HWY_SCALAR

#define HWY_ATTR
#define HWY_ALIGN
#define HWY_LANES(T) 1
#define HWY_CAPS                                                             \
  (HWY_CAP_DOUBLE | HWY_CAP_INT64 | HWY_CAP_CMP64 | HWY_CAP_VARIABLE_SHIFT | \
   HWY_CAP_GATHER)

namespace N_SCALAR {
namespace {

#ifndef HWY_OPS_SCALAR
#define HWY_OPS_SCALAR
#include "hwy/ops/scalar-inl.h"
#endif  // HWY_OPS_SCALAR

#else
#pragma message("HWY_TARGET does not match any known target")
#endif  // HWY_TARGET

// (Followed by the implementation section)

// For IDE only: avoid unmatched-brace warning.
#ifdef HWY_END_NAMESPACE_FOR_IDE
#undef HWY_END_NAMESPACE_FOR_IDE
}  // namespace
}  // namespace N_$TARGET
#endif  // HWY_END_NAMESPACE_FOR_IDE

#endif  // include guard
