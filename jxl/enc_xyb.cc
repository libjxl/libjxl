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

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/enc_xyb.cc"

#include <algorithm>
#include <cstdlib>
#include <hwy/runtime_dispatch.h>

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

namespace jxl {

struct TestCubeRoot_T {
  HWY_DECLARE(void, ())
};

// See enc_xyb.h.
struct ToXYB_T {
  HWY_DECLARE(void, (const ImageBundle& in, float linear_multiplier,
                     ThreadPool* pool, Image3F* JXL_RESTRICT xyb,
                     ImageBundle* JXL_RESTRICT linear = nullptr))
};

void TestCubeRoot() { hwy::TargetBitfield().Foreach(TestCubeRoot_T()); }

void ToXYB(const ImageBundle& in, const float linear_multiplier,
           ThreadPool* pool, Image3F* JXL_RESTRICT xyb,
           ImageBundle* JXL_RESTRICT linear) {
  Dispatch(hwy::TargetBitfield().Best(), ToXYB_T(), in, linear_multiplier, pool,
           xyb, linear);
}

// DEPRECATED
Image3F OpsinDynamicsImage(const Image3B& srgb8) {
  ImageMetadata metadata;
  metadata.bits_per_sample = 8;
  metadata.color_encoding = ColorEncoding::SRGB();
  ImageBundle ib(&metadata);
  ib.SetFromImage(StaticCastImage3<float>(srgb8), metadata.color_encoding);
  JXL_CHECK(ib.TransformTo(ColorEncoding::LinearSRGB(ib.IsGray())));
  ThreadPool* null_pool = nullptr;
  Image3F xyb(srgb8.xsize(), srgb8.ysize());
  ToXYB(ib, 1.0f, null_pool, &xyb);
  return xyb;
}

}  // namespace jxl

#endif  // end of non-SIMD code

#include <hwy/foreach_target.h>

namespace jxl {
namespace HWY_NAMESPACE {
namespace {

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
  const auto m2 = IfThenZeroElse(
      m1 == Zero(di), kExpBias - (hwy::ShiftRight<23>(m1)) * kExpMul);
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

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE

// Ensures infinity norm is bounded.
HWY_ATTR void TestCubeRoot_T::HWY_FUNC() {
  const HWY_FULL(float) d;
  float max_err = 0.0f;
  for (uint64_t x5 = 0; x5 < 2000000; x5++) {
    const float x = x5 * 1E-5f;
    const float expected = std::cbrt(x);
    HWY_ALIGN float approx[d.N];
    Store(HWY_NAMESPACE::CubeRootAndAdd(Set(d, x), Zero(d)), d, approx);

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
HWY_ATTR void ToXYB_T::HWY_FUNC(const ImageBundle& in, float linear_multiplier,
                                ThreadPool* pool, Image3F* JXL_RESTRICT xyb,
                                ImageBundle* JXL_RESTRICT linear) {
  PROFILER_FUNC;

  const size_t xsize = in.xsize();
  const size_t ysize = in.ysize();
  JXL_ASSERT(xyb->xsize() == xsize);
  JXL_ASSERT(xyb->ysize() == ysize);

  // Convert to linear sRGB (unless already in that space)
  const ImageBundle* linear_srgb = &in;
  const ColorEncoding& c = ColorEncoding::LinearSRGB(in.IsGray());
  ImageMetadata metadata;
  metadata.color_encoding = c;
  ImageBundle copy(&metadata);
  JXL_CHECK(TransformIfNeeded(in, c, pool, linear == nullptr ? &copy : linear,
                              &linear_srgb));
  if (linear != nullptr) {
    // Copy output to `linear` if TransformIfNeeded did not already do this.
    if (linear != linear_srgb) {
      *linear = linear_srgb->Copy();
    }
    if (linear_multiplier != 1.f) {
      ScaleImage(linear_multiplier, linear->MutableColor());
      if (linear == linear_srgb) {
        // Since linear_srgb has been multiplied (via linear), it should not be
        // multiplied further below.
        linear_multiplier = 1.f;
      }
    }
  }

  const HWY_FULL(float) d;
  HWY_ALIGN float premul_absorb[d.N * 12];
  for (size_t i = 0; i < 9; ++i) {
    const auto absorb = Set(d, kOpsinAbsorbanceMatrix[i]);
    Store(absorb * Set(d, linear_multiplier), d, premul_absorb + i * d.N);
  }
  for (size_t i = 0; i < 3; ++i) {
    const auto neg_bias_cbrt = Set(d, -std::cbrt(kOpsinAbsorbanceBias[i]));
    Store(neg_bias_cbrt, d, premul_absorb + (9 + i) * d.N);
  }

  RunOnPool(
      pool, 0, static_cast<int>(ysize), ThreadPool::SkipInit(),
      [&](const int task, const int /*thread*/) HWY_ATTR {
        const size_t y = static_cast<size_t>(task);
        const float* JXL_RESTRICT row_in0 =
            linear_srgb->color().ConstPlaneRow(0, y);
        const float* JXL_RESTRICT row_in1 =
            linear_srgb->color().ConstPlaneRow(1, y);
        const float* JXL_RESTRICT row_in2 =
            linear_srgb->color().ConstPlaneRow(2, y);
        float* JXL_RESTRICT row_xyb0 = xyb->PlaneRow(0, y);
        float* JXL_RESTRICT row_xyb1 = xyb->PlaneRow(1, y);
        float* JXL_RESTRICT row_xyb2 = xyb->PlaneRow(2, y);
        for (size_t x = 0; x < xsize; x += d.N) {
          const auto in_x = Load(d, row_in0 + x);
          const auto in_y = Load(d, row_in1 + x);
          const auto in_b = Load(d, row_in2 + x);
          HWY_NAMESPACE::LinearToXyb(in_x, in_y, in_b, premul_absorb,
                                     row_xyb0 + x, row_xyb1 + x, row_xyb2 + x);
        }
      },
      "ToXYB");
}

}  // namespace jxl
