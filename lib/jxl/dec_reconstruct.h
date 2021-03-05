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
#include "lib/jxl/quantizer.h"
#include "lib/jxl/splines.h"

namespace jxl {

// Finalizes the decoding of a frame by applying image features if necessary,
// doing color transforms (unless the frame header specifies
// `SaveBeforeColorTransform()`) and applying upsampling.
//
// Writes pixels in the appropriate colorspace to `idct`, shrinking it if
// necessary.
// `skip_blending` is necessary because the encoder butteraugli loop does not
// (yet) handle blending.
Status FinalizeFrameDecoding(ImageBundle* JXL_RESTRICT decoded,
                             PassesDecoderState* dec_state, ThreadPool* pool,
                             bool rerender, bool skip_blending);

// Renders the `output_rect` portion of the final image to `output_image`
// (unless the frame is upsampled - in which case, `output_rect` is scaled
// accordingly). `input_rect` should have the same shape. Color data is taken
// from `image_data:input_rect`; some X *and* Y padding is needed, so if
// `input_rect` is on a border of `input_image` padding will be added by
// mirroring. `input_rect` always refers to the non-padded pixels.
// `output_rect.x0()` is guaranteed to be a multiple of kBlockDim.
// `output_rect.xsize()` is either a multiple of kBlockDim, or is such that
// `output_rect.x0() + output_rect.xsize() == frame_dim.xsize`.
Status FinalizeImageRect(const Image3F& input_image, const Rect& input_rect,
                         PassesDecoderState* dec_state, size_t thread,
                         ImageBundle* JXL_RESTRICT output_image,
                         const Rect& output_rect);

// Ensures that there is a border of `xpadding x ypadding` valid pixels
// accessible around `src:src_rect`, and of `xborder` not-necessarily-valid
// pixels along the x axis by copying the area to `storage` if necessary and
// setting `output` and `output_rect` appropriately.
void EnsurePadding(const Image3F& src, const Rect& src_rect, Image3F* storage,
                   const Image3F** output, Rect* output_rect, size_t xpadding,
                   size_t ypadding, size_t xborder);

}  // namespace jxl

#endif  // LIB_JXL_DEC_RECONSTRUCT_H_
