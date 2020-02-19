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

#ifndef JXL_DEC_XYB_H_
#define JXL_DEC_XYB_H_

// XYB -> linear sRGB.

#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/image.h"
#include "jxl/opsin_params.h"

namespace jxl {

// Parameters for XYB->sRGB conversion.
struct OpsinParams {
  HWY_ALIGN float inverse_opsin_matrix[9 * 4];
  HWY_ALIGN float opsin_biases[4];
  HWY_ALIGN float opsin_biases_cbrt[4];
  float quant_biases[4];
  void Init();
};

// Converts `inout` (not padded) from opsin to linear sRGB in-place. Called from
// per-pass postprocessing, hence parallelized.
HWY_ATTR void OpsinToLinear(Image3F* JXL_RESTRICT inout, ThreadPool* pool,
                            const OpsinParams& opsin_params);

// Converts `opsin:rect` (opsin may be padded, rect.x0 must be vector-aligned)
// to linear sRGB. Called from whole-frame encoder, hence parallelized.
HWY_ATTR void OpsinToLinear(const Image3F& opsin, const Rect& rect,
                            ThreadPool* pool, Image3F* JXL_RESTRICT linear,
                            const OpsinParams& opsin_params);

HWY_ATTR void YcbcrToRgb(const ImageF& y_plane, const ImageF& cb_plane,
                         const ImageF& cr_plane, ImageF* r_plane,
                         ImageF* g_plane, ImageF* b_plane, ThreadPool* pool);

HWY_ATTR ImageF UpsampleV2(const ImageF& src, ThreadPool* pool);
HWY_ATTR ImageF UpsampleH2(const ImageF& src, ThreadPool* pool);

}  // namespace jxl

#endif  // JXL_DEC_XYB_H_
