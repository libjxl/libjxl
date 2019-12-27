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

#include <hwy/static_targets.h>

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

// Inverts the pixel-wise RGB->XYB conversion in OpsinDynamicsImage() (including
// the gamma mixing and simple gamma). Avoids clamping to [0, 255] - out of
// (sRGB) gamut values may be in-gamut after transforming to a wider space.
// "inverse_matrix" points to 9 broadcasted vectors, which are the 3x3 entries
// of the (row-major) opsin absorbance matrix inverse. Pre-multiplying its
// entries by c is equivalent to multiplying linear_* by c afterwards.
template <class D, class V>
HWY_ATTR JXL_INLINE void XybToRgb(D d, const V opsin_x, const V opsin_y,
                                  const V opsin_b,
                                  const OpsinParams& opsin_params,
                                  V* const JXL_RESTRICT linear_r,
                                  V* const JXL_RESTRICT linear_g,
                                  V* const JXL_RESTRICT linear_b) {
#if HWY_BITS == 0
  const auto inv_scale_x = Set(d, kInvScaleR);
  const auto inv_scale_y = Set(d, kInvScaleG);
  const auto neg_bias_r = Set(d, opsin_params.opsin_biases[0]);
  const auto neg_bias_g = Set(d, opsin_params.opsin_biases[1]);
  const auto neg_bias_b = Set(d, opsin_params.opsin_biases[2]);
#else
  const auto neg_bias_rgb = LoadDup128(d, opsin_params.opsin_biases);
  HWY_ALIGN const float inv_scale_lanes[4] = {kInvScaleR, kInvScaleG};
  const auto inv_scale = LoadDup128(d, inv_scale_lanes);
  const auto inv_scale_x = hwy::Broadcast<0>(inv_scale);
  const auto inv_scale_y = hwy::Broadcast<1>(inv_scale);
  const auto neg_bias_r = hwy::Broadcast<0>(neg_bias_rgb);
  const auto neg_bias_g = hwy::Broadcast<1>(neg_bias_rgb);
  const auto neg_bias_b = hwy::Broadcast<2>(neg_bias_rgb);
#endif

  // Color space: XYB -> RGB
  auto gamma_r = inv_scale_x * (opsin_y + opsin_x);
  auto gamma_g = inv_scale_y * (opsin_y - opsin_x);
  auto gamma_b = opsin_b;

  gamma_r -= Set(d, opsin_params.opsin_biases_cbrt[0]);
  gamma_g -= Set(d, opsin_params.opsin_biases_cbrt[1]);
  gamma_b -= Set(d, opsin_params.opsin_biases_cbrt[2]);

  // Undo gamma compression: linear = gamma^3 for efficiency.
  const auto gamma_r2 = gamma_r * gamma_r;
  const auto gamma_g2 = gamma_g * gamma_g;
  const auto gamma_b2 = gamma_b * gamma_b;
  const auto mixed_r = MulAdd(gamma_r2, gamma_r, neg_bias_r);
  const auto mixed_g = MulAdd(gamma_g2, gamma_g, neg_bias_g);
  const auto mixed_b = MulAdd(gamma_b2, gamma_b, neg_bias_b);

  const float* JXL_RESTRICT inverse_matrix = opsin_params.inverse_opsin_matrix;

  // Unmix (multiply by 3x3 inverse_matrix)
  *linear_r = LoadDup128(d, &inverse_matrix[0 * 4]) * mixed_r;
  *linear_g = LoadDup128(d, &inverse_matrix[3 * 4]) * mixed_r;
  *linear_b = LoadDup128(d, &inverse_matrix[6 * 4]) * mixed_r;
  *linear_r = MulAdd(LoadDup128(d, &inverse_matrix[1 * 4]), mixed_g, *linear_r);
  *linear_g = MulAdd(LoadDup128(d, &inverse_matrix[4 * 4]), mixed_g, *linear_g);
  *linear_b = MulAdd(LoadDup128(d, &inverse_matrix[7 * 4]), mixed_g, *linear_b);
  *linear_r = MulAdd(LoadDup128(d, &inverse_matrix[2 * 4]), mixed_b, *linear_r);
  *linear_g = MulAdd(LoadDup128(d, &inverse_matrix[5 * 4]), mixed_b, *linear_g);
  *linear_b = MulAdd(LoadDup128(d, &inverse_matrix[8 * 4]), mixed_b, *linear_b);
}

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
