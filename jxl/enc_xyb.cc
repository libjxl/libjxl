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

#include "jxl/enc_xyb.h"

#include <algorithm>
#include <cstdlib>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/fields.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/opsin_params.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/enc_xyb.cc"
#include <hwy/foreach_target.h>

namespace jxl {

#include <hwy/begin_target-inl.h>

// Returns cbrt(x) + add with 6 ulp max error.
// Modified from vectormath_exp.h, Apache 2 license.
// https://www.agner.org/optimize/vectorclass.zip
template <class V>
HWY_ATTR V CubeRootAndAdd(const V x, const V add) {
  const HWY_FULL(float) df;
  const HWY_FULL(int32_t) di;

  const auto kExpBias = Set(di, 0x54800000);  // cast(1.) + cast(1.) / 3
  const auto kExpMul = Set(di, 0x002AAAAA);   // shifted 1/3
  const auto k1_3 = Set(df, 1.0f / 3);
  const auto k4_3 = Set(df, 4.0f / 3);

  const auto xa = x;  // assume inputs never negative
  const auto xa_3 = k1_3 * xa;

  // Multiply exponent by -1/3
  const auto m1 = BitCast(di, xa);
  // Special case for 0. 0 is represented with an exponent of 0, so the
  // "kExpBias - 1/3 * exp" below gives the wrong result. The IfThenZeroElse()
  // sets those values as 0, which prevents having NaNs in the computations
  // below.
  const auto m2 =
      IfThenZeroElse(m1 == Zero(di), kExpBias - (ShiftRight<23>(m1)) * kExpMul);
  auto r = BitCast(df, m2);

  // Newton-Raphson iterations
  for (int i = 0; i < 3; i++) {
    const auto r2 = r * r;
    r = NegMulAdd(xa_3, r2 * r2, k4_3 * r);
  }
  // Final iteration
  auto r2 = r * r;
  r = MulAdd(k1_3, NegMulAdd(xa, r2 * r2, r), r);
  r2 = r * r;
  r = MulAdd(r2, x, add);

  return r;
}

template <class V>
HWY_ATTR void LinearXybTransform(const V r, V g, const V b,
                                 float* JXL_RESTRICT valx,
                                 float* JXL_RESTRICT valy,
                                 float* JXL_RESTRICT valz) {
  const HWY_FULL(float) d;
  const V half = Set(d, 0.5f);
  Store(half * (r - g), d, valx);
  Store(half * (r + g), d, valy);
  Store(b, d, valz);
}

// 4x3 matrix * 3x1 SIMD vectors
template <class V>
HWY_ATTR JXL_INLINE void OpsinAbsorbance(
    const V r, const V g, const V b, const float* JXL_RESTRICT premul_absorb,
    V* JXL_RESTRICT mixed0, V* JXL_RESTRICT mixed1, V* JXL_RESTRICT mixed2) {
  const float* bias = &kOpsinAbsorbanceBias[0];
  const HWY_FULL(float) d;
  const auto m0 = Load(d, premul_absorb + 0 * d.N);
  const auto m1 = Load(d, premul_absorb + 1 * d.N);
  const auto m2 = Load(d, premul_absorb + 2 * d.N);
  const auto m3 = Load(d, premul_absorb + 3 * d.N);
  const auto m4 = Load(d, premul_absorb + 4 * d.N);
  const auto m5 = Load(d, premul_absorb + 5 * d.N);
  const auto m6 = Load(d, premul_absorb + 6 * d.N);
  const auto m7 = Load(d, premul_absorb + 7 * d.N);
  const auto m8 = Load(d, premul_absorb + 8 * d.N);
  *mixed0 = MulAdd(m0, r, MulAdd(m1, g, MulAdd(m2, b, Set(d, bias[0]))));
  *mixed1 = MulAdd(m3, r, MulAdd(m4, g, MulAdd(m5, b, Set(d, bias[1]))));
  *mixed2 = MulAdd(m6, r, MulAdd(m7, g, MulAdd(m8, b, Set(d, bias[2]))));
}

// Converts one RGB vector to XYB.
template <class V>
HWY_ATTR void LinearToXyb(const V r, const V g, const V b,
                          const float* JXL_RESTRICT premul_absorb,
                          float* JXL_RESTRICT valx, float* JXL_RESTRICT valy,
                          float* JXL_RESTRICT valz) {
  V mixed0, mixed1, mixed2;
  OpsinAbsorbance(r, g, b, premul_absorb, &mixed0, &mixed1, &mixed2);

  // mixed* should be non-negative even for wide-gamut, so clamp to zero.
  mixed0 = ZeroIfNegative(mixed0);
  mixed1 = ZeroIfNegative(mixed1);
  mixed2 = ZeroIfNegative(mixed2);

  const HWY_FULL(float) d;
  mixed0 = CubeRootAndAdd(mixed0, Load(d, premul_absorb + 9 * d.N));
  mixed1 = CubeRootAndAdd(mixed1, Load(d, premul_absorb + 10 * d.N));
  mixed2 = CubeRootAndAdd(mixed2, Load(d, premul_absorb + 11 * d.N));
  LinearXybTransform(mixed0, mixed1, mixed2, valx, valy, valz);

  // For wide-gamut inputs, r/g/b and valx (but not y/z) are often negative.
}

// Ensures infinity norm is bounded.
HWY_ATTR void TestCubeRoot() {
  const HWY_FULL(float) d;
  float max_err = 0.0f;
  for (uint64_t x5 = 0; x5 < 2000000; x5++) {
    const float x = x5 * 1E-5f;
    const float expected = std::cbrt(x);
    HWY_ALIGN float approx[d.N];
    Store(CubeRootAndAdd(Set(d, x), Zero(d)), d, approx);

    // All lanes are same
    for (size_t i = 1; i < d.N; ++i) {
      JXL_ASSERT(std::abs(approx[0] - approx[i]) <= 1.2E-7f);
    }

    const float err = std::abs(approx[0] - expected);
    max_err = std::max(max_err, err);
  }
  // printf("max err %e\n", max_err);
  JXL_ASSERT(max_err < 8E-7f);
}

// This is different from butteraugli::OpsinDynamicsImage() in the sense that
// it does not contain a sensitivity multiplier based on the blurred image.
HWY_ATTR const ImageBundle* ToXYB(const ImageBundle& in, ThreadPool* pool,
                                  Image3F* JXL_RESTRICT xyb,
                                  ImageBundle* JXL_RESTRICT linear_storage) {
  PROFILER_FUNC;

  const size_t xsize = in.xsize();
  const size_t ysize = in.ysize();
  JXL_ASSERT(SameSize(in, *xyb));

  const ColorEncoding& c = ColorEncoding::LinearSRGB(in.IsGray());
  // Whether we can skip both TransformIfNeeded and SRGBToLinear.
  const bool already_linear = c.SameColorEncoding(in.c_current());
  if (!already_linear) {
    // No: will need storage. OK to reuse metadata, will not be changed.
    JXL_ASSERT(linear_storage != nullptr);
    *linear_storage = ImageBundle(const_cast<ImageMetadata*>(in.metadata()));
  }

  // Will point to linear sRGB (with or without actually transforming)
  const ImageBundle* linear_ptr = already_linear ? &in : linear_storage;

  const bool already_srgb = in.IsSRGB();  // Whether to call SRGBToLinear

  if (already_srgb) {
    JXL_ASSERT(!already_linear);
    linear_storage->SetFromImage(Image3F(xsize, ysize), c);
  } else {
    JXL_CHECK(TransformIfNeeded(in, c, pool, linear_storage, &linear_ptr));
  }

  const HWY_FULL(float) d;
  HWY_ALIGN float premul_absorb[d.N * 12];
  for (size_t i = 0; i < 9; ++i) {
    const auto absorb = Set(d, kOpsinAbsorbanceMatrix[i]);
    Store(absorb, d, premul_absorb + i * d.N);
  }
  for (size_t i = 0; i < 3; ++i) {
    const auto neg_bias_cbrt = Set(d, -std::cbrt(kOpsinAbsorbanceBias[i]));
    Store(neg_bias_cbrt, d, premul_absorb + (9 + i) * d.N);
  }

  const Image3F& in3 = in.color();
  Image3F* JXL_RESTRICT linear3 = linear_ptr->MutableColor();

  // TODO(janwas): move into -inl.h if indirect call is too expensive.
  auto srgb_to_linear = ChooseSRGBToLinear(hwy::SupportedTargets());

  if (already_srgb) {
    RunOnPool(
        pool, 0, static_cast<int>(ysize), ThreadPool::SkipInit(),
        [&](const int task, const int /*thread*/) HWY_ATTR {
          const size_t y = static_cast<size_t>(task);
          const float* JXL_RESTRICT row_srgb0 = in3.ConstPlaneRow(0, y);
          const float* JXL_RESTRICT row_srgb1 = in3.ConstPlaneRow(1, y);
          const float* JXL_RESTRICT row_srgb2 = in3.ConstPlaneRow(2, y);

          float* JXL_RESTRICT row_in0 = linear3->PlaneRow(0, y);
          float* JXL_RESTRICT row_in1 = linear3->PlaneRow(1, y);
          float* JXL_RESTRICT row_in2 = linear3->PlaneRow(2, y);

          srgb_to_linear(in.xsize(), row_srgb0, row_in0);
          srgb_to_linear(in.xsize(), row_srgb1, row_in1);
          srgb_to_linear(in.xsize(), row_srgb2, row_in2);

          float* JXL_RESTRICT row_xyb0 = xyb->PlaneRow(0, y);
          float* JXL_RESTRICT row_xyb1 = xyb->PlaneRow(1, y);
          float* JXL_RESTRICT row_xyb2 = xyb->PlaneRow(2, y);
          for (size_t x = 0; x < xsize; x += d.N) {
            const auto in_x = Load(d, row_in0 + x);
            const auto in_y = Load(d, row_in1 + x);
            const auto in_b = Load(d, row_in2 + x);
            LinearToXyb(in_x, in_y, in_b, premul_absorb, row_xyb0 + x,
                        row_xyb1 + x, row_xyb2 + x);
          }
        },
        "SRGBToXYB");
  } else {
    RunOnPool(
        pool, 0, static_cast<int>(ysize), ThreadPool::SkipInit(),
        [&](const int task, const int /*thread*/) HWY_ATTR {
          const size_t y = static_cast<size_t>(task);
          const float* JXL_RESTRICT row_in0 = linear3->ConstPlaneRow(0, y);
          const float* JXL_RESTRICT row_in1 = linear3->ConstPlaneRow(1, y);
          const float* JXL_RESTRICT row_in2 = linear3->ConstPlaneRow(2, y);
          float* JXL_RESTRICT row_xyb0 = xyb->PlaneRow(0, y);
          float* JXL_RESTRICT row_xyb1 = xyb->PlaneRow(1, y);
          float* JXL_RESTRICT row_xyb2 = xyb->PlaneRow(2, y);

          for (size_t x = 0; x < xsize; x += d.N) {
            const auto in_x = Load(d, row_in0 + x);
            const auto in_y = Load(d, row_in1 + x);
            const auto in_b = Load(d, row_in2 + x);
            LinearToXyb(in_x, in_y, in_b, premul_absorb, row_xyb0 + x,
                        row_xyb1 + x, row_xyb2 + x);
          }
        },
        "LinearToXYB");
  }

  return linear_ptr;
}

#include <hwy/end_target-inl.h>

#if HWY_ONCE
HWY_EXPORT(TestCubeRoot)
HWY_EXPORT(ToXYB)

// DEPRECATED
Image3F OpsinDynamicsImage(const Image3B& srgb8) {
  ImageMetadata metadata;
  metadata.bits_per_sample = 8;
  metadata.floating_point_sample = false;
  metadata.color_encoding = ColorEncoding::SRGB();
  ImageBundle ib(&metadata);
  ib.SetFromImage(StaticCastImage3<float>(srgb8), metadata.color_encoding);
  JXL_CHECK(ib.TransformTo(ColorEncoding::LinearSRGB(ib.IsGray())));
  ThreadPool* null_pool = nullptr;
  Image3F xyb(srgb8.xsize(), srgb8.ysize());

  ImageBundle linear_storage(&metadata);
  (void)ChooseToXYB(hwy::SupportedTargets())(ib, null_pool, &xyb,
                                             &linear_storage);
  return xyb;
}

#endif  // HWY_ONCE

}  // namespace jxl
