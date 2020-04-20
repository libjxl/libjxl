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

#include "tools/butteraugli_pnorm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <atomic>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/profiler.h"
#include "jxl/color_encoding.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tools/butteraugli_pnorm.cc"
#include <hwy/foreach_target.h>

namespace jxl {

#include <hwy/begin_target-inl.h>

HWY_ATTR double ComputeDistanceP(const ImageF& distmap, double p) {
  PROFILER_FUNC;
  const double onePerPixels = 1.0 / (distmap.ysize() * distmap.xsize());
  if (std::abs(p - 3.0) < 1E-6) {
#if HWY_CAPS & HWY_CAP_DOUBLE
    const HWY_FULL(double) dd;
    const HWY_CAPPED(float, dd.N) df;
    auto sums0 = Zero(dd);
    auto sums1 = Zero(dd);
    auto sums2 = Zero(dd);
#else
    const HWY_FULL(float) df;
    auto sums0 = Zero(df);
    auto sums1 = Zero(df);
    auto sums2 = Zero(df);
#endif
    double sum1[3] = {0.0};
    for (size_t y = 0; y < distmap.ysize(); ++y) {
      const float* JXL_RESTRICT row = distmap.ConstRow(y);
      size_t x = 0;
      for (; x + df.N <= distmap.xsize(); x += df.N) {
#if HWY_CAPS & HWY_CAP_DOUBLE
        const auto d1 = PromoteTo(dd, Load(df, row + x));
#else
        const auto d1 = Load(df, row + x);
#endif
        const auto d2 = d1 * d1 * d1;
        sums0 += d2;
        const auto d3 = d2 * d2;
        sums1 += d3;
        const auto d4 = d3 * d3;
        sums2 += d4;
      }
      for (; x < distmap.xsize(); ++x) {
        const double d1 = row[x];
        double d2 = d1 * d1 * d1;
        sum1[0] += d2;
        d2 *= d2;
        sum1[1] += d2;
        d2 *= d2;
        sum1[2] += d2;
      }
    }
    double v = 0;
    v += pow(onePerPixels * (sum1[0] + GetLane(SumOfLanes(sums0))),
             1.0 / (p * 1.0));
    v += pow(onePerPixels * (sum1[1] + GetLane(SumOfLanes(sums1))),
             1.0 / (p * 2.0));
    v += pow(onePerPixels * (sum1[2] + GetLane(SumOfLanes(sums2))),
             1.0 / (p * 4.0));
    v /= 3.0;
    return v;
  } else {
    static std::atomic<int> once{0};
    if (once.fetch_add(1, std::memory_order_relaxed) == 0) {
      fprintf(stderr, "WARNING: using slow ComputeDistanceP\n");
    }
    double sum1[3] = {0.0};
    for (size_t y = 0; y < distmap.ysize(); ++y) {
      const float* JXL_RESTRICT row = distmap.ConstRow(y);
      for (size_t x = 0; x < distmap.xsize(); ++x) {
        double d2 = std::pow(row[x], p);
        sum1[0] += d2;
        d2 *= d2;
        sum1[1] += d2;
        d2 *= d2;
        sum1[2] += d2;
      }
    }
    double v = 0;
    for (int i = 0; i < 3; ++i) {
      v += pow(onePerPixels * (sum1[i]), 1.0 / (p * (1 << i)));
    }
    v /= 3.0;
    return v;
  }
}

// TODO(lode): take alpha into account when needed
HWY_ATTR double ComputeDistance2(const ImageBundle& ib1,
                                 const ImageBundle& ib2) {
  PROFILER_FUNC;
  // Convert to sRGB - closer to perception than linear.
  const Image3F* srgb1 = &ib1.color();
  Image3F copy1;
  if (!ib1.IsSRGB()) {
    JXL_CHECK(ib1.CopyTo(Rect(ib1), ColorEncoding::SRGB(ib1.IsGray()), &copy1));
    srgb1 = &copy1;
  }
  const Image3F* srgb2 = &ib2.color();
  Image3F copy2;
  if (!ib2.IsSRGB()) {
    JXL_CHECK(ib2.CopyTo(Rect(ib2), ColorEncoding::SRGB(ib2.IsGray()), &copy2));
    srgb2 = &copy2;
  }

  JXL_CHECK(SameSize(*srgb1, *srgb2));

  HWY_FULL(float) d;

  auto sums = Zero(d);
  double result = 0;
  // Weighted PSNR as in JPEG-XL: chroma counts 1/8 (they compute on YCbCr).
  // Avoid squaring the weight - 1/64 is too extreme.
  const float weights[3] = {1.0f / 8, 6.0f / 8, 1.0f / 8};
  for (size_t c = 0; c < 3; ++c) {
    const auto weight = Set(d, weights[c]);
    for (size_t y = 0; y < srgb1->ysize(); ++y) {
      const float* JXL_RESTRICT row1 = srgb1->ConstPlaneRow(c, y);
      const float* JXL_RESTRICT row2 = srgb2->ConstPlaneRow(c, y);
      size_t x = 0;
      for (; x + d.N <= srgb1->xsize(); x += d.N) {
        const auto diff = Load(d, row1 + x) - Load(d, row2 + x);
        sums += diff * diff * weight;
      }
      for (; x < srgb1->xsize(); ++x) {
        const float diff = row1[x] - row2[x];
        result += diff * diff * weights[c];
      }
    }
  }
  const float sum = GetLane(SumOfLanes(sums));
  return sum + result;
}

#include <hwy/end_target-inl.h>

#if HWY_ONCE
HWY_EXPORT(ComputeDistanceP)
HWY_EXPORT(ComputeDistance2)
#endif

}  // namespace jxl
