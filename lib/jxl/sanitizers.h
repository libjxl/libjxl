// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_SANITIZERS_H_
#define LIB_JXL_SANITIZERS_H_

#include <stddef.h>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/image.h"

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
#include "lib/jxl/base/status.h"
#include "sanitizer/msan_interface.h"
#endif

namespace jxl {
namespace msan {

#if JXL_MEMORY_SANITIZER

// Chosen so that kSanitizerSentinel is four copies of kSanitizerSentinelByte.
constexpr uint8_t kSanitizerSentinelByte = 0x48;
constexpr float kSanitizerSentinel = 205089.125f;

static JXL_INLINE JXL_MAYBE_UNUSED void PoisonMemory(const volatile void* m,
                                                     size_t size) {
  __msan_poison(m, size);
}

static JXL_INLINE JXL_MAYBE_UNUSED void UnpoisonMemory(const volatile void* m,
                                                       size_t size) {
  __msan_unpoison(m, size);
}

// Mark all the bytes of an image (including padding) as poisoned bytes.
static JXL_INLINE JXL_MAYBE_UNUSED void PoisonImage(const PlaneBase& im) {
  PoisonMemory(im.bytes(), im.bytes_per_row() * im.ysize());
}

template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED void PoisonImage(const Image3<T>& im) {
  PoisonImage(im.Plane(0));
  PoisonImage(im.Plane(1));
  PoisonImage(im.Plane(2));
}

// Check that all the pixels in the provided rect of the image are initialized
// (not poisoned). If any of the values is poisoned it will abort.
template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED void CheckImageInitialized(
    const Plane<T>& im, const Rect& r, const char* message) {
  JXL_ASSERT(r.x0() <= im.xsize());
  JXL_ASSERT(r.x0() + r.xsize() <= im.xsize());
  JXL_ASSERT(r.y0() <= im.ysize());
  JXL_ASSERT(r.y0() + r.ysize() <= im.ysize());
  for (size_t y = r.y0(); y < r.y0() + r.ysize(); y++) {
    const auto* row = im.Row(y);
    intptr_t ret = __msan_test_shadow(row + r.x0(), sizeof(*row) * r.xsize());
    if (ret != -1) {
      JXL_DEBUG(1,
                "Checking an image of %zu x %zu, rect x0=%zu, y0=%zu, "
                "xsize=%zu, ysize=%zu",
                im.xsize(), im.ysize(), r.x0(), r.y0(), r.xsize(), r.ysize());
      size_t x = ret / sizeof(*row);
      JXL_DEBUG(1, "CheckImageInitialized failed at x=%zu, y=%zu: %s", x, y,
                message ? message : "");
    }
    // This will report an error if memory is not initialized.
    __msan_check_mem_is_initialized(row + r.x0(), sizeof(*row) * r.xsize());
  }
}

template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED void CheckImageInitialized(
    const Image3<T>& im, const Rect& r, const char* message) {
  for (size_t c = 0; c < 3; c++) {
    std::string str_message(message);
    str_message += " c=" + std::to_string(c);
    CheckImageInitialized(im.Plane(c), r, str_message.c_str());
  }
}

#define JXL_CHECK_IMAGE_INITIALIZED(im, r) \
  ::jxl::msan::CheckImageInitialized(im, r, "im=" #im ", r=" #r);

#else  // JXL_MEMORY_SANITIZER

// In non-msan mode these functions don't use volatile since it is not needed
// for the empty functions.

static JXL_INLINE JXL_MAYBE_UNUSED void PoisonMemory(const void*, size_t) {}
static JXL_INLINE JXL_MAYBE_UNUSED void UnpoisonMemory(const void*, size_t) {}

static JXL_INLINE JXL_MAYBE_UNUSED void PoisonImage(const PlaneBase& im) {}
template <typename T>
static JXL_INLINE JXL_MAYBE_UNUSED void PoisonImage(const Plane<T>& im) {}

#define JXL_CHECK_IMAGE_INITIALIZED(im, r)

#endif

}  // namespace msan
}  // namespace jxl

#endif  // LIB_JXL_SANITIZERS_H_
