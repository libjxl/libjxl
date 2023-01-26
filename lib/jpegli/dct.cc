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
#include "lib/jxl/image.h"
HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {

constexpr float kZeroBiasMulXYB[] = {0.5f, 0.5f, 0.5f};
constexpr float kZeroBiasMulYCbCr[] = {0.7f, 1.0f, 0.8f};

void DownsampleImage(const jxl::ImageF& input, size_t factor_x, size_t factor_y,
                     jxl::ImageF* output) {
  output->ShrinkTo(DivCeil(input.xsize(), factor_x),
                   DivCeil(input.ysize(), factor_y));
  size_t in_stride = input.PixelsPerRow();
  for (size_t y = 0; y < output->ysize(); y++) {
    float* row_out = output->Row(y);
    const float* row_in = input.Row(factor_y * y);
    for (size_t x = 0; x < output->xsize(); x++) {
      size_t cnt = 0;
      float sum = 0;
      for (size_t iy = 0; iy < factor_y && iy + factor_y * y < input.ysize();
           iy++) {
        for (size_t ix = 0; ix < factor_x && ix + factor_x * x < input.xsize();
             ix++) {
          sum += row_in[iy * in_stride + x * factor_x + ix];
          cnt++;
        }
      }
      row_out[x] = sum / cnt;
    }
  }
}

void DownsampleImage(jxl::ImageF* image, size_t factor_x, size_t factor_y) {
  // Allocate extra space to avoid a reallocation when padding.
  jxl::ImageF downsampled(DivCeil(image->xsize(), factor_x) + DCTSIZE,
                          DivCeil(image->ysize(), factor_y) + DCTSIZE);
  DownsampleImage(*image, factor_x, factor_y, &downsampled);
  *image = std::move(downsampled);
}

void ComputeDCTCoefficients(
    j_compress_ptr cinfo, const jxl::Image3F& opsin, float distance,
    const bool xyb, const jxl::ImageF& qf, const float* qm,
    std::vector<std::vector<jpegli::coeff_t> >* all_coeffs) {
  float qfmin, qfmax;
  ImageMinMax(qf, &qfmin, &qfmax);
  float zero_bias_mul[3] = {0.5f, 0.5f, 0.5f};
  if (distance <= 1.0f) {
    memcpy(zero_bias_mul, xyb ? kZeroBiasMulXYB : kZeroBiasMulYCbCr,
           sizeof(zero_bias_mul));
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
    const jxl::ImageF* plane = &opsin.Plane(c);
    if (h_factor > 1 || v_factor > 1) {
      tmp = CopyImage(*plane);
      DownsampleImage(&tmp, h_factor, v_factor);
      plane = &tmp;
    }
    std::vector<coeff_t> coeffs(xsize_blocks * ysize_blocks * kDCTBlockSize);
    const float* qmc = &qm[c * kDCTBlockSize];
    for (size_t by = 0, bix = 0; by < ysize_blocks; by++) {
      for (size_t bx = 0; bx < xsize_blocks; bx++, bix++) {
        coeff_t* block = &coeffs[bix * kDCTBlockSize];
        HWY_ALIGN float dct[kDCTBlockSize];
        TransformFromPixels(jxl::AcStrategy::Type::DCT,
                            plane->Row(8 * by) + 8 * bx, plane->PixelsPerRow(),
                            dct, scratch_space);
        // Create more zeros in areas where jpeg xl would have used a lower
        // quantization multiplier.
        float relq = qfmax / qf.Row(by * v_factor)[bx * h_factor];
        float zero_bias = 0.5f + zero_bias_mul[c] * (relq - 1.0f);
        zero_bias = std::min(1.5f, zero_bias);
        for (size_t iy = 0, i = 0; iy < 8; iy++) {
          for (size_t ix = 0; ix < 8; ix++, i++) {
            float coeff = 2040 * dct[ix * 8 + iy] * qmc[i];
            int cc = std::abs(coeff) < zero_bias ? 0 : std::round(coeff);
            block[i] = cc;
          }
        }
        if (xyb || c == 0) {
          // Only the U, V channels of YUV input are zero-centered, for the
          // rest apply an offset to the DC values instead.
          block[0] = std::round((2040 * dct[0] - 1024) * qmc[0]);
        } else {
          block[0] = std::round(2040 * dct[0] * qmc[0]);
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
    j_compress_ptr cinfo, const jxl::Image3F& opsin, float distance,
    const bool xyb, const jxl::ImageF& qf, const float* qm,
    std::vector<std::vector<jpegli::coeff_t> >* coeffs) {
  HWY_DYNAMIC_DISPATCH(ComputeDCTCoefficients)
  (cinfo, opsin, distance, xyb, qf, qm, coeffs);
}

}  // namespace jpegli
#endif  // HWY_ONCE
