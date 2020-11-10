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

#ifndef LIB_JXL_DEC_RECONSTRUCT_H_
#define LIB_JXL_DEC_RECONSTRUCT_H_

#include <stddef.h>

#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dot_dictionary.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/patch_dictionary.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/splines.h"

namespace jxl {

// Finalizes the decoding of a frame by applying image features if necessary,
// doing color transforms (unless the frame header specifies
// `SaveBeforeColorTransform()`) and applying upsampling.
//
// Writes pixels in the appropriate colorspace to `idct`, shrinking it if
// necessary.
Status FinalizeFrameDecoding(Image3F* JXL_RESTRICT idct,
                             PassesDecoderState* dec_state, ThreadPool* pool,
                             AuxOut* aux_out);

// Applies image features on the given `idct_rect` of `idct`, interpreted as the
// `image_rect` region of the full image.
Status ApplyImageFeatures(Image3F* JXL_RESTRICT idct, const Rect& rect,
                          PassesDecoderState* dec_state, size_t thread,
                          AuxOut* aux_out);

// Same as ApplyImageFeatures, but only processes row `y` of
// dec_state->decoded. `y` should be relative to `rect`.
// The first row in `rect` corresponds to a value of `y` of `2*kBlockDim`.
// This function should be called for `rect.ysize() + 2 * lf.PaddingRows()`
// values of `y`, in increasing order, starting from `y=-lf.PaddingRows()`.
Status ApplyImageFeaturesRow(Image3F* JXL_RESTRICT idct, const Rect& in_rect,
                             PassesDecoderState* dec_state, ssize_t y,
                             size_t thread, AuxOut* aux_out);

}  // namespace jxl

#endif  // LIB_JXL_DEC_RECONSTRUCT_H_
