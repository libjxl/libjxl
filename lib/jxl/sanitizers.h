// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_SANITIZERS_H_
#define LIB_JXL_SANITIZERS_H_

#include <stddef.h>

#include "lib/jxl/base/compiler_specific.h"

#ifdef MEMORY_SANITIZER
#define JXL_MEMORY_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define JXL_MEMORY_SANITIZER 1
#else
#define JXL_MEMORY_SANITIZER 0
#endif
#else
#define JXL_MEMORY_SANITIZER 0
#endif

#if JXL_MEMORY_SANITIZER
#include "sanitizer/msan_interface.h"
#endif

namespace jxl {

#if JXL_MEMORY_SANITIZER

// Chosen so that kSanitizerSentinel is four copies of kSanitizerSentinelByte.
constexpr uint8_t kSanitizerSentinelByte = 0x48;
constexpr float kSanitizerSentinel = 205089.125f;

static JXL_INLINE JXL_MAYBE_UNUSED void PoisonMemory(const volatile void *m,
                                                     size_t size) {
  __msan_poison(m, size);
}

static JXL_INLINE JXL_MAYBE_UNUSED void UnpoisonMemory(const volatile void *m,
                                                       size_t size) {
  __msan_unpoison(m, size);
}

#else  // JXL_MEMORY_SANITIZER

static JXL_INLINE JXL_MAYBE_UNUSED void PoisonMemory(const volatile void *,
                                                     size_t) {}
static JXL_INLINE JXL_MAYBE_UNUSED void UnpoisonMemory(const volatile void *,
                                                       size_t) {}

#endif

}  // namespace jxl

#endif  // LIB_JXL_SANITIZERS_H_
