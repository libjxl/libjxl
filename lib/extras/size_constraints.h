// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_SIZE_CONSTRAINTS_H_
#define LIB_JXL_SIZE_CONSTRAINTS_H_

#include <cstdint>
#include <type_traits>

#include "lib/jxl/base/status.h"

namespace jxl {

struct SizeConstraints {
  // Upper limit on pixel dimensions/area, enforced by VerifyDimensions
  // (called from decoders). Fuzzers set smaller values to limit memory use.
  // Default values correspond to JXL level 10.
  uint32_t dec_max_xsize = 1u << 30;
  uint32_t dec_max_ysize = 1u << 30;
  uint64_t dec_max_pixels = static_cast<uint64_t>(1u) << 40;
  // Upper limit on the size of a single pixel buffer that a decoder may
  // allocate based on image dimensions, enforced by VerifyBufferSize.
  // Compressed inputs (PNG, EXR, ...) can describe images whose decompressed
  // buffers exceed any feasible allocation by orders of magnitude, so
  // decoders must reject such buffers up front instead of attempting the
  // allocation.
  uint64_t dec_max_buffer_size = uint64_t{1} << 36;  // 64 GiB
};

template <typename T,
          class = typename std::enable_if<std::is_unsigned<T>::value>::type>
Status VerifyDimensions(const SizeConstraints* constraints, T xs, T ys) {
  SizeConstraints limit = {};
  if (constraints) limit = *constraints;

  if (xs == 0 || ys == 0) return JXL_FAILURE("Empty image.");
  if (xs > limit.dec_max_xsize) return JXL_FAILURE("Image too wide.");
  if (ys > limit.dec_max_ysize) return JXL_FAILURE("Image too tall.");

  const uint64_t num_pixels = static_cast<uint64_t>(xs) * ys;
  if (num_pixels > limit.dec_max_pixels) {
    return JXL_FAILURE("Image too big.");
  }

  return true;
}

// Verifies that a pixel buffer of `size` bytes, derived from image
// dimensions, does not exceed the buffer-size limit. Only guards against
// infeasible allocation requests; the actual allocation may still fail
// depending on the available memory.
inline Status VerifyBufferSize(const SizeConstraints* constraints,
                               uint64_t size) {
  SizeConstraints limit = {};
  if (constraints) limit = *constraints;

  if (size > limit.dec_max_buffer_size) {
    return JXL_FAILURE("Pixel buffer too big.");
  }

  return true;
}

}  // namespace jxl

#endif  // LIB_JXL_SIZE_CONSTRAINTS_H_
