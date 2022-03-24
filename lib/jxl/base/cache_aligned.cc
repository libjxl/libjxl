// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/base/cache_aligned.h"

#include <stdio.h>
#include <stdlib.h>

// Disabled: slower than malloc + alignment.
#define JXL_USE_MMAP 0

#if JXL_USE_MMAP
#include <sys/mman.h>
#endif

#include <algorithm>  // std::max
#include <atomic>
#include <hwy/base.h>  // kMaxVectorSize
#include <limits>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/sanitizers.h"

namespace jxl {

// Avoids linker errors in pre-C++17 builds.
constexpr size_t CacheAligned::kPointerSize;
constexpr size_t CacheAligned::kCacheLineSize;
constexpr size_t CacheAligned::kAlignment;
constexpr size_t CacheAligned::kAlias;

void CacheAligned::PrintStats() {}

size_t CacheAligned::NextOffset() { return 0; }

void* CacheAligned::Allocate(const size_t payload_size, size_t offset) {
  return JXL_ASSUME_ALIGNED(
      aligned_alloc(kAlignment, RoundUpTo(payload_size, kAlignment)), 64);
}

void CacheAligned::Free(const void* aligned_pointer) {
  free(const_cast<void*>(aligned_pointer));
}

}  // namespace jxl
