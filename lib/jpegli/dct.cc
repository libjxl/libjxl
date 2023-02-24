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

#include "lib/jpegli/memory_manager.h"
#include "lib/jxl/enc_transforms.h"

HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::Round;

constexpr float kZeroBiasMulXYB[] = {0.5f, 0.5f, 0.5f};
constexpr float kZeroBiasMulYCbCr[] = {0.7f, 1.0f, 0.8f};

using D8 = HWY_CAPPED(float, 8);
using DI8 = HWY_CAPPED(int32_t, 8);
constexpr D8 d8;
constexpr DI8 di8;

#if HWY_CAP_GE256
JXL_INLINE void Transpose8x8Block(const float* JXL_RESTRICT from,
                                  float* JXL_RESTRICT to) {
  const D8 d;
  auto i0 = Load(d, from);
  auto i1 = Load(d, from + 1 * 8);
  auto i2 = Load(d, from + 2 * 8);
  auto i3 = Load(d, from + 3 * 8);
  auto i4 = Load(d, from + 4 * 8);
  auto i5 = Load(d, from + 5 * 8);
  auto i6 = Load(d, from + 6 * 8);
  auto i7 = Load(d, from + 7 * 8);

  const auto q0 = InterleaveLower(d, i0, i2);
  const auto q1 = InterleaveLower(d, i1, i3);
  const auto q2 = InterleaveUpper(d, i0, i2);
  const auto q3 = InterleaveUpper(d, i1, i3);
  const auto q4 = InterleaveLower(d, i4, i6);
  const auto q5 = InterleaveLower(d, i5, i7);
  const auto q6 = InterleaveUpper(d, i4, i6);
  const auto q7 = InterleaveUpper(d, i5, i7);

  const auto r0 = InterleaveLower(d, q0, q1);
  const auto r1 = InterleaveUpper(d, q0, q1);
  const auto r2 = InterleaveLower(d, q2, q3);
  const auto r3 = InterleaveUpper(d, q2, q3);
  const auto r4 = InterleaveLower(d, q4, q5);
  const auto r5 = InterleaveUpper(d, q4, q5);
  const auto r6 = InterleaveLower(d, q6, q7);
  const auto r7 = InterleaveUpper(d, q6, q7);

  i0 = ConcatLowerLower(d, r4, r0);
  i1 = ConcatLowerLower(d, r5, r1);
  i2 = ConcatLowerLower(d, r6, r2);
  i3 = ConcatLowerLower(d, r7, r3);
  i4 = ConcatUpperUpper(d, r4, r0);
  i5 = ConcatUpperUpper(d, r5, r1);
  i6 = ConcatUpperUpper(d, r6, r2);
  i7 = ConcatUpperUpper(d, r7, r3);

  Store(i0, d, to);
  Store(i1, d, to + 1 * 8);
  Store(i2, d, to + 2 * 8);
  Store(i3, d, to + 3 * 8);
  Store(i4, d, to + 4 * 8);
  Store(i5, d, to + 5 * 8);
  Store(i6, d, to + 6 * 8);
  Store(i7, d, to + 7 * 8);
}
#elif HWY_TARGET != HWY_SCALAR
JXL_INLINE void Transpose8x8Block(const float* JXL_RESTRICT from,
                                  float* JXL_RESTRICT to) {
  const HWY_CAPPED(float, 4) d;
  for (size_t n = 0; n < 8; n += 4) {
    for (size_t m = 0; m < 8; m += 4) {
      auto p0 = Load(d, from + n * 8 + m);
      auto p1 = Load(d, from + (n + 1) * 8 + m);
      auto p2 = Load(d, from + (n + 2) * 8 + m);
      auto p3 = Load(d, from + (n + 3) * 8 + m);
      const auto q0 = InterleaveLower(d, p0, p2);
      const auto q1 = InterleaveLower(d, p1, p3);
      const auto q2 = InterleaveUpper(d, p0, p2);
      const auto q3 = InterleaveUpper(d, p1, p3);

      const auto r0 = InterleaveLower(d, q0, q1);
      const auto r1 = InterleaveUpper(d, q0, q1);
      const auto r2 = InterleaveLower(d, q2, q3);
      const auto r3 = InterleaveUpper(d, q2, q3);
      Store(r0, d, to + m * 8 + n);
      Store(r1, d, to + (1 + m) * 8 + n);
      Store(r2, d, to + (2 + m) * 8 + n);
      Store(r3, d, to + (3 + m) * 8 + n);
    }
  }
}
#else
JXL_INLINE void Transpose8x8Block(const float* JXL_RESTRICT from,
                                  float* JXL_RESTRICT to) {
  for (size_t n = 0; n < 8; ++n) {
    for (size_t m = 0; m < 8; ++m) {
      to[8 * n + m] = from[8 * m + n];
    }
  }
}
#endif

void QuantizeBlock(const float* dct, const float* qmc, const float zero_bias,
                   coeff_t* block) {
  for (size_t k = 0; k < kDCTBlockSize; ++k) {
    float coeff = dct[k] * qmc[k];
    int cc = std::abs(coeff) < zero_bias ? 0 : std::round(coeff);
    block[k] = cc;
  }
}

void QuantizeBlockNoAQ(const float* dct, const float* qmc, int32_t* block) {
  for (size_t k = 0; k < kDCTBlockSize; k += Lanes(d8)) {
    const auto val = Load(d8, dct + k);
    const auto q = Load(d8, qmc + k);
    const auto ival = ConvertTo(di8, Round(Mul(val, q)));
    Store(ival, di8, block + k);
  }
}

static constexpr float kDCBias = 128.0f / 255.0f;

void ComputeDCTCoefficients(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  std::vector<float> zero_bias_mul(cinfo->num_components, 0.5f);
  const bool xyb = m->xyb_mode && cinfo->jpeg_color_space == JCS_RGB;
  if (m->distance <= 1.0f) {
    for (int c = 0; c < 3 && c < cinfo->num_components; ++c) {
      zero_bias_mul[c] = xyb ? kZeroBiasMulXYB[c] : kZeroBiasMulYCbCr[c];
    }
  }
  HWY_ALIGN float dct0[kDCTBlockSize];
  HWY_ALIGN float dct1[kDCTBlockSize];
  HWY_ALIGN int32_t blocki[kDCTBlockSize];
  for (int c = 0; c < cinfo->num_components; c++) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    const size_t xsize_blocks = comp->width_in_blocks;
    const size_t ysize_blocks = comp->height_in_blocks;
    JXL_DASSERT(cinfo->max_h_samp_factor % comp->h_samp_factor == 0);
    JXL_DASSERT(cinfo->max_v_samp_factor % comp->v_samp_factor == 0);
    const int h_factor = cinfo->max_h_samp_factor / comp->h_samp_factor;
    const int v_factor = cinfo->max_v_samp_factor / comp->v_samp_factor;
    size_t num_coeffs = xsize_blocks * ysize_blocks * kDCTBlockSize;
    coeff_t* coeffs = Allocate<coeff_t>(cinfo, num_coeffs, JPOOL_IMAGE_ALIGNED);
    m->coefficients[c] = coeffs;
    JQUANT_TBL* quant_table = cinfo->quant_tbl_ptrs[comp->quant_tbl_no];
    std::vector<float> qmc(kDCTBlockSize);
    for (size_t k = 0; k < kDCTBlockSize; k++) {
      qmc[k] = 2040.0f / quant_table->quantval[k];
    }
    RowBuffer<float>* plane = &m->input_buffer[c];
    for (size_t by = 0, bix = 0; by < ysize_blocks; by++) {
      const float* row = plane->Row(8 * by);
      for (size_t bx = 0; bx < xsize_blocks; bx++, bix++) {
        coeff_t* block = &coeffs[bix * kDCTBlockSize];
        TransformFromPixels(jxl::AcStrategy::Type::DCT, row + 8 * bx,
                            plane->stride(), dct0, dct1);
        Transpose8x8Block(dct0, dct1);
        if (m->use_adaptive_quantization) {
          // Create more zeros in areas where jpeg xl would have used a lower
          // quantization multiplier.
          float relq = m->quant_field.Row(by * v_factor)[bx * h_factor];
          float zero_bias = 0.5f + zero_bias_mul[c] * relq;
          zero_bias = std::min(1.5f, zero_bias);
          QuantizeBlock(dct1, &qmc[0], zero_bias, block);
        } else {
          QuantizeBlockNoAQ(dct1, &qmc[0], blocki);
          for (size_t k = 0; k < kDCTBlockSize; ++k) {
            block[k] = blocki[k];
          }
        }
        // Center DC values around zero.
        block[0] = std::round((dct1[0] - kDCBias) * qmc[0]);
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jpegli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jpegli {

HWY_EXPORT(ComputeDCTCoefficients);

void ComputeDCTCoefficients(j_compress_ptr cinfo) {
  HWY_DYNAMIC_DISPATCH(ComputeDCTCoefficients)(cinfo);
}

}  // namespace jpegli
#endif  // HWY_ONCE
