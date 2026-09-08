// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_MA_COMMON_H_
#define LIB_JXL_MODULAR_ENCODING_MA_COMMON_H_

#include <stddef.h>

namespace jxl {

enum MATreeContext : size_t {
  kSplitValContext = 0,
  kPropertyContext = 1,
  kPredictorContext = 2,
  kOffsetContext = 3,
  kMultiplierLogContext = 4,
  kMultiplierBitsContext = 5,

  kNumTreeContexts = 6,
};

static constexpr size_t kMaxTreeSize = 1 << 22;

inline constexpr size_t MaxGlobalTreeSize(size_t xsize, size_t ysize,
                                          size_t num_channels) {
  constexpr size_t kBaseTreeSize = 1024;
  constexpr size_t kScaledSizeLimit = (kMaxTreeSize - kBaseTreeSize) * 16;
  if (xsize != 0 && ysize > kScaledSizeLimit / xsize) {
    return kMaxTreeSize;
  }
  const size_t num_pixels = xsize * ysize;
  if (num_channels != 0 && num_pixels > kScaledSizeLimit / num_channels) {
    return kMaxTreeSize;
  }
  return kBaseTreeSize + num_pixels * num_channels / 16;
}

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_MA_COMMON_H_
