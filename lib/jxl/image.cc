// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/image.h"

#include <algorithm>  // fill, swap
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"
#include "lib/jxl/cache_aligned.h"
#include "lib/jxl/simd_util.h"

#if defined(MEMORY_SANITIZER) || HWY_IDE
#include "lib/jxl/base/common.h"
#include "lib/jxl/sanitizers.h"
#endif

namespace jxl {

PlaneBase::PlaneBase(const size_t xsize, const size_t ysize,
                     const size_t sizeof_t)
    : xsize_(static_cast<uint32_t>(xsize)),
      ysize_(static_cast<uint32_t>(ysize)),
      orig_xsize_(static_cast<uint32_t>(xsize)),
      orig_ysize_(static_cast<uint32_t>(ysize)) {
  JXL_CHECK(xsize == xsize_);
  JXL_CHECK(ysize == ysize_);

  JXL_ASSERT(sizeof_t == 1 || sizeof_t == 2 || sizeof_t == 4 || sizeof_t == 8);

  bytes_per_row_ = 0;
  // Dimensions can be zero, e.g. for lazily-allocated images. Only allocate
  // if nonzero, because "zero" bytes still have padding/bookkeeping overhead.
  if (xsize != 0 && ysize != 0) {
    bytes_per_row_ = BytesPerRow(xsize, sizeof_t);
    bytes_ = AllocateArray(bytes_per_row_ * ysize);
    JXL_CHECK(bytes_.get());
    InitializePadding(sizeof_t);
  }
}

void PlaneBase::InitializePadding(const size_t sizeof_t) {
#if defined(MEMORY_SANITIZER) || HWY_IDE
  if (xsize_ == 0 || ysize_ == 0) return;

  const size_t vec_size = MaxVectorSize();
  if (vec_size == 0) return;  // Scalar mode: no padding needed

  const size_t valid_size = xsize_ * sizeof_t;
  const size_t initialize_size = RoundUpTo(valid_size, vec_size);
  if (valid_size == initialize_size) return;

  for (size_t y = 0; y < ysize_; ++y) {
    uint8_t* JXL_RESTRICT row = static_cast<uint8_t*>(VoidRow(y));
#if defined(__clang__) &&                                           \
    ((!defined(__apple_build_version__) && __clang_major__ <= 6) || \
     (defined(__apple_build_version__) &&                           \
      __apple_build_version__ <= 10001145))
    // There's a bug in msan in clang-6 when handling AVX2 operations. This
    // workaround allows tests to pass on msan, although it is slower and
    // prevents msan warnings from uninitialized images.
    std::fill(row, msan::kSanitizerSentinelByte, initialize_size);
#else
    memset(row + valid_size, msan::kSanitizerSentinelByte,
           initialize_size - valid_size);
#endif  // clang6
  }
#endif  // MEMORY_SANITIZER
}

void PlaneBase::Swap(PlaneBase& other) {
  std::swap(xsize_, other.xsize_);
  std::swap(ysize_, other.ysize_);
  std::swap(orig_xsize_, other.orig_xsize_);
  std::swap(orig_ysize_, other.orig_ysize_);
  std::swap(bytes_per_row_, other.bytes_per_row_);
  std::swap(bytes_, other.bytes_);
}

}  // namespace jxl
