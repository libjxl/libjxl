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

// XYB -> linear sRGB helper function.

#if defined(LIB_JXL_DEC_XYB_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_DEC_XYB_INL_H_
#undef LIB_JXL_DEC_XYB_INL_H_
#else
#define LIB_JXL_DEC_XYB_INL_H_
#endif

#include <hwy/highway.h>

#include "lib/jxl/dec_xyb.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Broadcast;

// Inverts the pixel-wise RGB->XYB conversion in OpsinDynamicsImage() (including
// the gamma mixing and simple gamma). Avoids clamping to [0, 1] - out of (sRGB)
// gamut values may be in-gamut after transforming to a wider space.
// "inverse_matrix" points to 9 broadcasted vectors, which are the 3x3 entries
// of the (row-major) opsin absorbance matrix inverse. Pre-multiplying its
// entries by c is equivalent to multiplying linear_* by c afterwards.
template <class D, class V>
HWY_INLINE HWY_MAYBE_UNUSED void XybToRgb(D d, const V opsin_x, const V opsin_y,
                                          const V opsin_b,
                                          const OpsinParams& opsin_params,
                                          V* const HWY_RESTRICT linear_r,
                                          V* const HWY_RESTRICT linear_g,
                                          V* const HWY_RESTRICT linear_b) {
#if HWY_TARGET == HWY_SCALAR
  const auto inv_scale_x = Set(d, kInvScaleR);
  const auto inv_scale_y = Set(d, kInvScaleG);
  const auto neg_bias_r = Set(d, opsin_params.opsin_biases[0]);
  const auto neg_bias_g = Set(d, opsin_params.opsin_biases[1]);
  const auto neg_bias_b = Set(d, opsin_params.opsin_biases[2]);
#else
  const auto neg_bias_rgb = LoadDup128(d, opsin_params.opsin_biases);
  HWY_ALIGN const float inv_scale_lanes[4] = {kInvScaleR, kInvScaleG};
  const auto inv_scale = LoadDup128(d, inv_scale_lanes);
  const auto inv_scale_x = Broadcast<0>(inv_scale);
  const auto inv_scale_y = Broadcast<1>(inv_scale);
  const auto neg_bias_r = Broadcast<0>(neg_bias_rgb);
  const auto neg_bias_g = Broadcast<1>(neg_bias_rgb);
  const auto neg_bias_b = Broadcast<2>(neg_bias_rgb);
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

  const float* HWY_RESTRICT inverse_matrix = opsin_params.inverse_opsin_matrix;

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

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_DEC_XYB_INL_H_
