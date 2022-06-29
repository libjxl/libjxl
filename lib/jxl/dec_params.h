// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_PARAMS_H_
#define LIB_JXL_DEC_PARAMS_H_

// Parameters and flags that govern JXL decompression.

#include <stddef.h>
#include <stdint.h>

#include <limits>

#include "lib/jxl/base/override.h"

namespace jxl {

struct DecompressParams {
  // If true, skip dequant and iDCT and decode to JPEG (only if possible)
  bool decode_to_jpeg = false;

  // How many passes to decode at most. By default, decode everything.
  uint32_t max_passes = std::numeric_limits<uint32_t>::max();
  // Alternatively, one can specify the maximum tolerable downscaling factor
  // with respect to the full size of the image. By default, nothing less than
  // the full size is requested.
  size_t max_downsampling = 1;

  // Internal test-only setting: whether or not to use the slow rendering
  // pipeline.
  bool use_slow_render_pipeline = false;
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_PARAMS_H_
