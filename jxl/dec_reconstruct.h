// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef JXL_DEC_RECONSTRUCT_H_
#define JXL_DEC_RECONSTRUCT_H_

#include <stddef.h>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/dec_cache.h"
#include "jxl/dot_dictionary.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/loop_filter.h"
#include "jxl/multiframe.h"
#include "jxl/noise.h"
#include "jxl/patch_dictionary.h"
#include "jxl/quantizer.h"
#include "jxl/splines.h"

namespace jxl {

// Finalizes the decoding of a pass by running per-pass post processing:
// smoothing and adaptive reconstruction. Writes linear sRGB to `idct` and
// shrinks it to `x/ysize` to undo prior padding.
Status FinalizeFrameDecoding(Image3F* JXL_RESTRICT idct,
                             PassesDecoderState* dec_state, ThreadPool* pool,
                             AuxOut* aux_out, bool save_decompressed,
                             bool apply_color_transform);

// Applies image features on the given `idct_rect` of `idct`, interpreted as the
// `image_rect` region of the full image.
HWY_ATTR void ApplyImageFeatures(Image3F* JXL_RESTRICT idct, const Rect& rect,
                                 PassesDecoderState* dec_state, size_t thread,
                                 AuxOut* aux_out, bool save_decompressed,
                                 bool apply_color_transform);

// Same as ApplyImageFeatures, but only processes row `y` of
// dec_state->decoded. `y` should be relative to `rect`.
// The first row in `rect` corresponds to a value of `y` of `2*kBlockDim`.
// This function should be called for `rect.ysize() + 2 * lf.PaddingRows()`
// values of `y`, in increasing order, starting from
// `y=2*kBlockDim-lf.PaddingRows()`.
HWY_ATTR void ApplyImageFeaturesRow(Image3F* JXL_RESTRICT idct,
                                    const Rect& rect,
                                    PassesDecoderState* dec_state, size_t y,
                                    size_t thread, AuxOut* aux_out,
                                    bool save_decompressed,
                                    bool apply_color_transform);

}  // namespace jxl

#endif  // JXL_DEC_RECONSTRUCT_H_
