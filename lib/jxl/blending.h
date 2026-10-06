// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BLENDING_H_
#define LIB_JXL_BLENDING_H_

#include <jxl/memory_manager.h>

#include <cstddef>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {

bool NeedsBlending(const FrameHeader& frame_header);

// Reads `bg` and writes `out` at offset `x0`; reads `fg` at offset `fg_x0`.
// Splitting bg/out and fg offsets lets callers pre-shift `fg` to a non-negative
// origin instead of relying on unsigned-wrap pointer arithmetic to align it.
Status PerformBlending(JxlMemoryManager* memory_manager, const float* const* bg,
                       const float* const* fg, float* const* out, size_t x0,
                       size_t fg_x0, size_t xsize,
                       const PatchBlending& color_blending,
                       const PatchBlending* ec_blending,
                       const std::vector<ExtraChannelInfo>& extra_channel_info);

}  // namespace jxl

#endif  // LIB_JXL_BLENDING_H_
