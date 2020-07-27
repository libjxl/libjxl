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

// Applies target-specific attributes to all functions before the corresponding
// #include "hwy/after_namespace-inl.h". Clang and GCC require such attributes
// in order to use SIMD intrinsics.
//
// Uses a single pragma instead of user-specified annotations to each function,
// which are error-prone (omitting them may cause failures on some compilers).
// HWY_ATTR is unfortunately still required for lambdas because Clang does not
// provide an apply_to for lambdas.
//
// Must be included at file scope - clang prior to v9 raises errors if these
// pragma occur within a namespace.
//
// No include guard so the pragma is issued for each re-inclusion.

#include "hwy/highway.h"

// NOTE: clang and GCC do not expand macros inside #pragma, so we need one
// pragma attribute per target. We can at least factor this out:
#if HWY_TARGET != HWY_SCALAR && (HWY_COMPILER_GCC && !HWY_COMPILER_CLANG)
#pragma GCC push_options
#endif

#undef HWY_ATTR

//-----------------------------------------------------------------------------
// SSE4
#if HWY_TARGET == HWY_SSE4

#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
#define HWY_ATTR __attribute__((target("sse2,ssse3,sse4.1")))
#endif

#if HWY_COMPILER_CLANG
#pragma clang attribute push(__attribute__((target("sse2,ssse3,sse4.1"))), \
                             apply_to = function)
#elif HWY_COMPILER_GCC
#pragma GCC target("sse2,ssse3,sse4.1")
#endif  // HWY_COMPILER_*

//-----------------------------------------------------------------------------
// AVX2 \ FMA
#elif HWY_TARGET == HWY_AVX2 && defined(HWY_DISABLE_BMI2_FMA)

#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
#define HWY_ATTR __attribute__((target("avx,avx2")))
#endif

#if HWY_COMPILER_CLANG
#pragma clang attribute push(__attribute__((target("avx,avx2"))), \
                             apply_to = function)
#elif HWY_COMPILER_GCC
#pragma GCC target("avx,avx2")
#endif  // HWY_COMPILER_*

//-----------------------------------------------------------------------------
// AVX2 + FMA
#elif HWY_TARGET == HWY_AVX2 && !defined(HWY_DISABLE_BMI2_FMA)

#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
#define HWY_ATTR __attribute__((target("avx,avx2,bmi,bmi2,fma")))
#endif

#if HWY_COMPILER_CLANG
#pragma clang attribute push(__attribute__((target("avx,avx2,bmi,bmi2,fma"))), \
                             apply_to = function)
#elif HWY_COMPILER_GCC
#pragma GCC target("avx,avx2,bmi,bmi2,fma")
#endif  // HWY_COMPILER_*

//-----------------------------------------------------------------------------
// AVX3
#elif HWY_TARGET == HWY_AVX3

#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
#define HWY_ATTR \
  __attribute__( \
      (target("avx,avx2,bmi,bmi2,fma,avx512f,avx512vl,avx512dq,avx512bw")))
#endif

// Must include AVX2 because an AVX3 test may call AVX2 functions (e.g. when
// converting to half-vectors). HWY_DISABLE_BMI2_FMA is not relevant because if
// we have AVX3, we should also have BMI2/FMA.
#if HWY_COMPILER_CLANG
#pragma clang attribute push(                                                  \
    __attribute__(                                                             \
        (target("avx,avx2,bmi,bmi2,fma,avx512f,avx512vl,avx512dq,avx512bw"))), \
    apply_to = function)
#elif HWY_COMPILER_GCC
#pragma GCC target("avx,avx2,bmi,bmi2,fma,avx512f,avx512vl,avx512dq,avx512bw")
#endif  // HWY_COMPILER_*

//-----------------------------------------------------------------------------
// PPC8
#elif HWY_TARGET == HWY_PPC8

#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
#define HWY_ATTR __attribute__((target("altivec,vsx")))
#endif

#if HWY_COMPILER_CLANG
#pragma clang attribute push(__attribute__((target("altivec,vsx"))), \
                             apply_to = function)
#elif HWY_COMPILER_GCC
#pragma GCC target("altivec,vsx")
#endif  // HWY_COMPILER_*

//-----------------------------------------------------------------------------
// NEON
#elif HWY_TARGET == HWY_NEON

#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
#define HWY_ATTR __attribute__((target("crypto")))
#endif

#if HWY_COMPILER_CLANG
#pragma clang attribute push(__attribute__((target("crypto"))), \
                             apply_to = function)
#elif HWY_COMPILER_GCC
#pragma GCC target("crypto")
#endif  // HWY_COMPILER_*

//-----------------------------------------------------------------------------
// WASM
#elif HWY_TARGET == HWY_WASM

#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
#define HWY_ATTR __attribute__((target("simd128")))
#endif

#if HWY_COMPILER_CLANG
#pragma clang attribute push(__attribute__((target("simd128"))), \
                             apply_to = function)
#elif HWY_COMPILER_GCC
#pragma GCC target("simd128")
#endif  // HWY_COMPILER_*

//-----------------------------------------------------------------------------
// SCALAR
#elif HWY_TARGET == HWY_SCALAR

// No pragma attribute needed.
#define HWY_ATTR

#else
#pragma message("HWY_TARGET does not match any known target")
#endif  // HWY_TARGET
