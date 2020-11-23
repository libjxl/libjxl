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

#ifndef LIB_JXL_EPF_H_
#define LIB_JXL_EPF_H_

// Fast SIMD "in-loop" edge preserving filter (adaptive, nonlinear).

#include <stddef.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/filters.h"
#include "lib/jxl/passes_state.h"

namespace jxl {

// 4 * (sqrt(0.5)-1), so that Weight(sigma) = 0.5.
static constexpr float kInvSigmaNum = -1.1715728752538099024f;

// Mirror n floats starting at *p and store them before p.
JXL_INLINE void LeftMirror(float* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    *(p - 1 - i) = p[i];
  }
}

// Mirror n floats starting at *(p - n) and store them at *p.
JXL_INLINE void RightMirror(float* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    p[i] = *(p - 1 - i);
  }
}

// Fills the `state->filter_weights.sigma` image with the precomputed sigma
// values in the area inside `block_rect`. Accesses the AC strategy, quant field
// and epf_sharpness fields in the corresponding positions.
void ComputeSigma(const Rect& block_rect, PassesDecoderState* state);

// Applies gaborish + EPF to the given `rect` part of the input image `in`,
// storing the result in the same `rect` rect of the output image and reading
// sigma values from the corresponding `rect` portion (downsized by kBlockDim)
// of `filter_weights.sigma`. `in` must be padded with kMaxFilterPadding worth
// of mirrored data to the left and right, but no extra rows below or above.
// `filter_weights.sigma` must be padded with kMaxFilterPadding/kBlockDim pixels
// worth of data on each side. The `rect` should ignore this padding.
void EdgePreservingFilter(const LoopFilter& lf,
                          const FilterWeights& filter_weights, const Rect& rect,
                          const Image3F& in, Image3F* JXL_RESTRICT out);

// Same as EdgePreservingFilter, but only processes row `y` of
// dec_state->decoded. If an output row was produced, it is returned in
// `output_row`. `y` should be relative to `rect` (`output_row` will be too).
// The first row in `rect` corresponds to a value of `y` of 0.
// This function should be called for `rect.ysize() + 2 * lf.PaddingRows()`
// values of `y`, in increasing order, starting from `y = -lf.PaddingRows()`.
Status ApplyLoopFiltersRow(PassesDecoderState* dec_state, const Rect& rect,
                           ssize_t y, size_t thread, Image3F* JXL_RESTRICT out,
                           size_t* JXL_RESTRICT output_row);

}  // namespace jxl

#endif  // LIB_JXL_EPF_H_
