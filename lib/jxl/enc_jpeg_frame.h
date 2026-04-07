// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_JPEG_FRAME_H_
#define LIB_JXL_ENC_JPEG_FRAME_H_

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {

// Optimize the thresholds and context map for JPEG coefficients.
// This is an optimization that reduces the encoded size of the
// JPEG AC coefficients by finding optimal DC thresholds
// and clustering similar contexts together.
Status OptimizeJPEGContextMap(const jpeg::JPEGData& jpeg_data,
                              SpeedTier speed_tier,
                              BlockCtxMap& ctx_map, ThreadPool* pool);

}  // namespace jxl

#endif  // LIB_JXL_ENC_JPEG_FRAME_H_
