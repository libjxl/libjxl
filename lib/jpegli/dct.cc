// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/dct.h"

#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jpegli/dct.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/enc_transforms.h"

HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {

constexpr float kZeroBiasMulXYB[] = {0.5f, 0.5f, 0.5f};
constexpr float kZeroBiasMulYCbCr[] = {0.7f, 1.0f, 0.8f};

void QuantizeBlock(const float* dct, const float* qmc, const float zero_bias,
                   coeff_t* block) {
  for (size_t iy = 0, i = 0; iy < 8; iy++) {
    for (size_t ix = 0; ix < 8; ix++, i++) {
      float coeff = 2040 * dct[ix * 8 + iy] * qmc[i];
      int cc = std::abs(coeff) < zero_bias ? 0 : std::round(coeff);
      block[i] = cc;
    }
  }
  // Center DC values around zero.
  block[0] = std::round((2040 * dct[0] - 1024) * qmc[0]);
}

void QuantizeBlockNoAQ(const float* dct, const float* qmc, coeff_t* block) {
  for (size_t iy = 0, i = 0; iy < 8; iy++) {
    for (size_t ix = 0; ix < 8; ix++, i++) {
      block[i] = std::round(2040 * dct[ix * 8 + iy] * qmc[i]);
    }
  }
  // Center DC values around zero.
  block[0] = std::round((2040 * dct[0] - 1024) * qmc[0]);
}

void ComputeDCTCoefficients(
    j_compress_ptr cinfo,
    std::vector<std::vector<jpegli::coeff_t> >* all_coeffs) {
  jpeg_comp_master* m = cinfo->master;
  std::vector<float> zero_bias_mul(cinfo->num_components, 0.5f);
  const bool xyb = m->xyb_mode && cinfo->jpeg_color_space == JCS_RGB;
  if (m->distance <= 1.0f) {
    for (int c = 0; c < 3 && c < cinfo->num_components; ++c) {
      zero_bias_mul[c] = xyb ? kZeroBiasMulXYB[c] : kZeroBiasMulYCbCr[c];
    }
  }
  HWY_ALIGN float scratch_space[2 * kDCTBlockSize];
  jxl::ImageF tmp;
  for (int c = 0; c < cinfo->num_components; c++) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    const size_t xsize_blocks = comp->width_in_blocks;
    const size_t ysize_blocks = comp->height_in_blocks;
    JXL_DASSERT(cinfo->max_h_samp_factor % comp->h_samp_factor == 0);
    JXL_DASSERT(cinfo->max_v_samp_factor % comp->v_samp_factor == 0);
    const int h_factor = cinfo->max_h_samp_factor / comp->h_samp_factor;
    const int v_factor = cinfo->max_v_samp_factor / comp->v_samp_factor;
    std::vector<coeff_t> coeffs(xsize_blocks * ysize_blocks * kDCTBlockSize);
    JQUANT_TBL* quant_table = cinfo->quant_tbl_ptrs[comp->quant_tbl_no];
    std::vector<float> qmc(kDCTBlockSize);
    for (size_t k = 0; k < kDCTBlockSize; k++) {
      qmc[k] = 1.0f / quant_table->quantval[k];
    }
    RowBuffer<float>* plane = &m->input_buffer[c];
    for (size_t by = 0, bix = 0; by < ysize_blocks; by++) {
      const float* row = plane->Row(8 * by);
      for (size_t bx = 0; bx < xsize_blocks; bx++, bix++) {
        coeff_t* block = &coeffs[bix * kDCTBlockSize];
        HWY_ALIGN float dct[kDCTBlockSize];
        TransformFromPixels(jxl::AcStrategy::Type::DCT, row + 8 * bx,
                            plane->stride(), dct, scratch_space);
        if (m->use_adaptive_quantization) {
          // Create more zeros in areas where jpeg xl would have used a lower
          // quantization multiplier.
          float relq = m->quant_field.Row(by * v_factor)[bx * h_factor];
          float zero_bias = 0.5f + zero_bias_mul[c] * relq;
          zero_bias = std::min(1.5f, zero_bias);
          QuantizeBlock(dct, &qmc[0], zero_bias, block);
        } else {
          QuantizeBlockNoAQ(dct, &qmc[0], block);
        }
      }
    }
    all_coeffs->emplace_back(std::move(coeffs));
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jpegli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jpegli {

HWY_EXPORT(ComputeDCTCoefficients);

void ComputeDCTCoefficients(
    j_compress_ptr cinfo, std::vector<std::vector<jpegli::coeff_t> >* coeffs) {
  HWY_DYNAMIC_DISPATCH(ComputeDCTCoefficients)(cinfo, coeffs);
}

}  // namespace jpegli
#endif  // HWY_ONCE
