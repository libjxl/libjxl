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

#include "lib/jpegli/dct-inl.h"
#include "lib/jpegli/memory_manager.h"

HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {
namespace {

void ComputeCoefficientBlock(const float* JXL_RESTRICT pixels, size_t stride,
                             const float* JXL_RESTRICT qmc, float aq_strength,
                             float zero_bias_mul, float* JXL_RESTRICT tmp,
                             JCOEF* block) {
  float* JXL_RESTRICT dct = tmp;
  float* JXL_RESTRICT scratch_space = tmp + DCTSIZE2;
  TransformFromPixels(pixels, stride, dct, scratch_space);
  if (aq_strength > 0.0f) {
    // Create more zeros in areas where jpeg xl would have used a lower
    // quantization multiplier.
    float zero_bias = 0.5f + zero_bias_mul * aq_strength;
    zero_bias = std::min(1.5f, zero_bias);
    QuantizeBlock(dct, qmc, zero_bias, block);
  } else {
    QuantizeBlockNoAQ(dct, qmc, block);
  }
  // Center DC values around zero.
  block[0] = std::round((dct[0] - kDCBias) * qmc[0]);
}

void ComputeDCTCoefficients(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  float* tmp = m->dct_buffer;
  for (int c = 0; c < cinfo->num_components; c++) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    int by0 = m->next_iMCU_row * comp->v_samp_factor;
    int block_rows_left = comp->height_in_blocks - by0;
    int max_block_rows = std::min(comp->v_samp_factor, block_rows_left);
    JBLOCKARRAY ba = (*cinfo->mem->access_virt_barray)(
        reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[c], by0,
        max_block_rows, true);
    float* qmc = m->quant_mul[c];
    RowBuffer<float>* plane = m->raw_data[c];
    const int h_factor = m->h_factor[c];
    const int v_factor = m->v_factor[c];
    const float zero_bias_mul = m->zero_bias_mul[c];
    float aq_strength = 0.0f;
    for (int iy = 0; iy < comp->v_samp_factor; iy++) {
      size_t by = by0 + iy;
      if (by >= comp->height_in_blocks) continue;
      JBLOCKROW brow = ba[iy];
      const float* row = plane->Row(8 * by);
      for (size_t bx = 0; bx < comp->width_in_blocks; bx++) {
        JCOEF* block = &brow[bx][0];
        if (m->use_adaptive_quantization) {
          aq_strength = m->quant_field.Row(by * v_factor)[bx * h_factor];
        }
        ComputeCoefficientBlock(row + 8 * bx, plane->stride(), qmc, aq_strength,
                                zero_bias_mul, tmp, block);
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace jpegli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jpegli {

HWY_EXPORT(ComputeCoefficientBlock);
HWY_EXPORT(ComputeDCTCoefficients);

void ComputeCoefficientBlock(const float* JXL_RESTRICT pixels, size_t stride,
                             const float* JXL_RESTRICT qmc, float aq_strength,
                             float zero_bias_mul, float* JXL_RESTRICT tmp,
                             JCOEF* block) {
  HWY_DYNAMIC_DISPATCH(ComputeCoefficientBlock)
  (pixels, stride, qmc, aq_strength, zero_bias_mul, tmp, block);
}

void ComputeDCTCoefficients(j_compress_ptr cinfo) {
  HWY_DYNAMIC_DISPATCH(ComputeDCTCoefficients)(cinfo);
}

}  // namespace jpegli
#endif  // HWY_ONCE
