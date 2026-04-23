// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_JPEG_FRAME_H_
#define LIB_JXL_ENC_JPEG_FRAME_H_

#include <array>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/image.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {

// JPEG/CfL context passed from the encoder to the JPEG transcode optimizer.
// It always carries the JPEG/JXL component mapping; when `enabled` is true,
// CfL-transformed target AC coefficients are modeled as residuals before
// building optimization data, matching what the entropy coder sees.
struct JpegCflContext {
  // JXL plane -> JPEG component mapping used by the JPEG transcode path.
  const std::array<int, 3>& plane_to_jpeg;
  // Whether optimizer CfL residual modeling is active for this image.
  bool enabled;
  // CfL maps for the two transformed targets:
  // [0] corresponds to JXL plane 0 / JPEG component `plane_to_jpeg[0]`
  // and [1] corresponds to JXL plane 2 / JPEG component `plane_to_jpeg[2]`.
  // The predictor source is always JXL plane 1 / JPEG component
  // `plane_to_jpeg[1]`.
  // Indexed by color tile:
  //   map->Row(by / kColorTileDimInBlocks)[bx / kColorTileDimInBlocks]
  const ImageSB* cfl_map[2];
  // Optimizer-only quant-table ratios for the transformed targets:
  //   scaled_qtable[i][coeffpos] = qt_source[coeffpos] / qt_target_i[coeffpos]
  // with the same plane order as `cfl_map`.
  const int32_t* scaled_qtable[2];
};

// Optimize the thresholds and context map for JPEG coefficients.
// This is an optimization that reduces the encoded size of the
// JPEG AC coefficients by finding optimal DC thresholds
// and clustering similar contexts together.
// `cfl_ctx` carries the JPEG/JXL component mapping for the optimizer.
// When `cfl_ctx.enabled` is true, it also models CfL-residual target AC
// coefficients instead of raw coefficients.
Status OptimizeJPEGContextMap(const jpeg::JPEGData& jpeg_data,
                              SpeedTier speed_tier,
                              const JpegCflContext& cfl_ctx,
                              BlockCtxMap& ctx_map, ThreadPool* pool);

}  // namespace jxl

#endif  // LIB_JXL_ENC_JPEG_FRAME_H_
