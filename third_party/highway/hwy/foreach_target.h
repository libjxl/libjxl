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

// Included in the global namespace, hence we cannot include ops/* from here.
// The include guard prevents infinite recursion.
#ifndef HWY_FOREACH_TARGET_H_
#define HWY_FOREACH_TARGET_H_

#include "hwy/highway.h"

// IDE is parsing only this header, or user forgot to define this:
#ifndef HWY_TARGET_INCLUDE
#if HWY_IDE
#define HWY_TARGET_INCLUDE <stddef.h>  // any header - avoids warnings.
#else
#error "Must define HWY_TARGET_INCLUDE before including foreach_target.h"
#endif
#endif

// Re-includes the translation unit zero or more times to compile for any
// targets except HWY_STATIC_TARGET. HWY_TARGET determines the macros/namespace
// in begin_target-inl.h.

// *_inl.h may include other headers, which requires include guards to prevent
// repeated inclusion. The guards must be reset after compiling each target, so
// the header is again visible. This is done by flipping HWY_TARGET_TOGGLE,
// defining it if undefined and vice versa. This macro is initially undefined
// so that IDEs don't gray out the contents of each header.
#ifdef HWY_TARGET_TOGGLE
#error "This macro must not be defined before foreach_target.h"
#endif

#if (HWY_TARGETS & HWY_SCALAR) && (HWY_STATIC_TARGET != HWY_SCALAR)
#undef HWY_TARGET
#define HWY_TARGET HWY_SCALAR
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_NEON) && (HWY_STATIC_TARGET != HWY_NEON)
#undef HWY_TARGET
#define HWY_TARGET HWY_NEON
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_SSE4) && (HWY_STATIC_TARGET != HWY_SSE4)
#undef HWY_TARGET
#define HWY_TARGET HWY_SSE4
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_AVX2) && (HWY_STATIC_TARGET != HWY_AVX2)
#undef HWY_TARGET
#define HWY_TARGET HWY_AVX2
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_AVX3) && (HWY_STATIC_TARGET != HWY_AVX3)
#undef HWY_TARGET
#define HWY_TARGET HWY_AVX3
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_WASM) && (HWY_STATIC_TARGET != HWY_WASM)
#undef HWY_TARGET
#define HWY_TARGET HWY_WASM
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_PPC8) && (HWY_STATIC_TARGET != HWY_PPC8)
#undef HWY_TARGET
#define HWY_TARGET HWY_PPC8
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

// If we re-include once per enabled target, the translation unit's
// implementation would have to be skipped via #if to avoid redefining symbols.
// We instead skip the re-include for HWY_STATIC_TARGET, and generate its
// implementation when resuming compilation of the translation unit. Reverting
// to the initial value of HWY_TARGET also causes HWY_ONCE to expand to 1.
#undef HWY_TARGET
#define HWY_TARGET HWY_STATIC_TARGET

#endif  // HWY_FOREACH_TARGET_H_
