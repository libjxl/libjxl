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

#include "lib/jxl/enc_adaptive_quantization.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_adaptive_quantization.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/dct_scales.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_group.h"
#include "lib/jxl/dec_reconstruct.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/fast_math-inl.h"
#include "lib/jxl/gauss_blur.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/opsin_params.h"
#include "lib/jxl/quant_weights.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Rebind;

// The following functions modulate an exponent (out_val) and return the updated
// value. Their descriptor is limited to 8 lanes for 8x8 blocks.

template <class D, class V>
V ComputeMask(const D d, const V out_val) {
  const auto kBase = Set(d, -0.75074198942923331f);
  const auto kMul4 = Set(d, 3.5679454110355913f);
  const auto kMul2 = Set(d, 12.922411049392476f);
  const auto kOffset2 = Set(d, 304.79134236526278f);
  const auto kMul3 = Set(d, 4.9317013412351525f);
  const auto kOffset3 = Set(d, 1.9251828251393233f);
  const auto kOffset4 = Set(d, 0.25f) * kOffset3;
  const auto kMul0 = Set(d, 0.75558830247369591f);
  const auto k1 = Set(d, 1.0f);

  // Avoid division by zero.
  const auto v1 = Max(out_val * kMul0, Set(d, 1e-3f));
  const auto v2 = k1 / (v1 + kOffset2);
  const auto v3 = k1 / MulAdd(v1, v1, kOffset3);
  const auto v4 = k1 / MulAdd(v1, v1, kOffset4);
  // TODO(jyrki):
  // A log or two here could make sense. In butteraugli we have effectively
  // log(log(x + C)) for this kind of use, as a single log is used in
  // saturating visual masking and here the modulation values are exponential,
  // another log would counter that.
  return kBase + MulAdd(kMul4, v4, MulAdd(kMul2, v2, kMul3 * v3));
}

const float* Quant64() {
  static constexpr double kQuant64[64] = {
      0.00, 4.10, 3.30, 3.30, 1.10, 1.15, 0.70, 0.70, 4.10, 3.30, 3.30,
      1.10, 1.15, 1.30, 0.70, 0.50, 3.00, 3.30, 2.90, 2.10, 1.30, 0.70,
      0.50, 0.50, 0.87, 2.90, 2.10, 1.40, 0.70, 0.50, 0.50, 0.50, 0.87,
      1.40, 1.40, 1.60, 0.50, 0.50, 0.50, 0.50, 1.40, 0.90, 1.60, 0.50,
      0.50, 0.50, 0.50, 0.50, 0.90, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
      0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
  };
  constexpr double kPow = 2.0757549489102436;
  HWY_ALIGN static float quant[64];
  for (size_t i = 0; i < 64; i++) {
    quant[i] = static_cast<float>(std::pow(kQuant64[i], kPow));
  }
  return quant;
}

// For converting full vectors to a subset. Assumes `vfull` lanes are identical.
template <class D, class VFull>
Vec<D> CapTo(const D d, VFull vfull) {
  using T = typename D::T;
  const HWY_FULL(T) dfull;
  HWY_ALIGN T lanes[MaxLanes(dfull)];
  Store(vfull, dfull, lanes);
  return Load(d, lanes);
}

// Increase precision in 8x8 blocks that are complicated in DCT space.
template <class D, class V>
V DctModulation(const D d, const size_t x, const size_t y, const ImageF& xyb,
                const V out_val) {
  HWY_ALIGN float dct[kDCTBlockSize];
  HWY_ALIGN float scratch_space[kDCTBlockSize * 2];
  ComputeTransposedScaledDCT<8>()(
      DCTFrom(xyb.ConstRow(y) + x, xyb.PixelsPerRow()), DCTTo(dct, 8),
      scratch_space);

  // Dealing with all 64 coefficients, don't need to limit to 8 here.
  const HWY_FULL(float) dfull;
  auto entropyQL2v = Zero(dfull);
  auto entropyQL4v = Zero(dfull);
  auto entropyQL8v = Zero(dfull);
  static const float* quant = Quant64();
  for (size_t i = 0; i < kDCTBlockSize; i += Lanes(dfull)) {
    auto v = Load(dfull, dct + i);
    v *= v;
    const auto q = Load(dfull, quant + i);
    entropyQL2v = MulAdd(q, v, entropyQL2v);
    v *= v;
    entropyQL4v = MulAdd(q, v, entropyQL4v);
    v *= v;
    entropyQL8v = MulAdd(q, v, entropyQL8v);
  }
  // Narrow to `d` so we can return the result.
  auto entropyQL2 = CapTo(d, SumOfLanes(entropyQL2v));
  auto entropyQL4 = CapTo(d, SumOfLanes(entropyQL4v));
  auto entropyQL8 = CapTo(d, SumOfLanes(entropyQL8v));
  entropyQL2 = Sqrt(entropyQL2);
  entropyQL4 = Sqrt(Sqrt(entropyQL4));
  entropyQL8 = Sqrt(Sqrt(Sqrt(entropyQL8)));
  const auto mulQL2 = Set(d, 0.045655288457332764f);
  const auto mulQL4 = Set(d, -0.46444076773581316f);
  const auto mulQL8 = Set(d, 0.36332928423385757f);
  const auto v = MulAdd(mulQL2, entropyQL2,
                        MulAdd(mulQL4, entropyQL4, mulQL8 * entropyQL8));
  const auto kMul = Set(d, 0.74718448352037614f * 8.0f);
  return MulAdd(kMul, v, out_val);
}

// mul and mul2 represent a scaling difference between jxl and butteraugli.
constexpr float kSGmul = 200.18996185254082f;
constexpr float kSGmul2 = 1.0f / 74.246832380856091f;
constexpr float kLog2 = 0.693147181f;
// Includes correction factor for std::log -> log2.
constexpr float kSGRetMul = kSGmul2 * 18.6580932135f * kLog2;
constexpr float kSGRetAdd = kSGmul2 * -20.2789020414f;
constexpr float kSGVOffset = 7.14672470003f;

template <typename D, typename V>
V SimpleGamma(const D d, V v) {
  // A simple HDR compatible gamma function.
  const auto mul = Set(d, kSGmul);
  const auto kRetMul = Set(d, kSGRetMul);
  const auto kRetAdd = Set(d, kSGRetAdd);
  const auto kVOffset = Set(d, kSGVOffset);

  v *= mul;

  // This should happen rarely, but may lead to a NaN, which is rather
  // undesirable. Since negative photons don't exist we solve the NaNs by
  // clamping here.
  // TODO(veluca): with FastLog2f, this no longer leads to NaNs.
  v = ZeroIfNegative(v);
  return kRetMul * FastLog2f(d, v + kVOffset) + kRetAdd;
}

template <bool invert, typename D, typename V>
V RatioOfDerivativesOfCubicRootToSimpleGamma(const D d, V v) {
  // The opsin space in jxl is the cubic root of photons, i.e., v * v * v
  // is related to the number of photons.
  //
  // SimpleGamma(v * v * v) is the psychovisual space in butteraugli.
  // This ratio allows quantization to move from jxl's opsin space to
  // butteraugli's log-gamma space.
  v = ZeroIfNegative(v);
  const auto kNumMul = Set(d, kSGRetMul * 3 * kSGmul);
  const auto kVOffset = Set(d, kSGVOffset * kLog2);
  const auto kDenMul = Set(d, kLog2 * kSGmul);

  const auto v2 = v * v;

  const auto num = kNumMul * v2;
  const auto den = MulAdd(kDenMul * v, v2, kVOffset);
  return invert ? num / den : den / num;
}

template <bool invert = false>
static float RatioOfDerivativesOfCubicRootToSimpleGamma(float v) {
  using DScalar = HWY_CAPPED(float, 1);
  auto vscalar = Load(DScalar(), &v);
  return GetLane(
      RatioOfDerivativesOfCubicRootToSimpleGamma<invert>(DScalar(), vscalar));
}

// TODO(veluca): this function computes an approximation of the derivative of
// SimpleGamma with (f(x+eps)-f(x))/eps. Consider two-sided approximation or
// exact derivatives.
template <class D, class V>
V GammaModulation(const D d, const size_t x, const size_t y,
                  const ImageF& xyb_x, const ImageF& xyb_y, const V out_val) {
  const float kBias = 0.16f;
  JXL_DASSERT(kBias > kOpsinAbsorbanceBias[0]);
  JXL_DASSERT(kBias > kOpsinAbsorbanceBias[1]);
  JXL_DASSERT(kBias > kOpsinAbsorbanceBias[2]);
  auto overall_ratio = Zero(d);
  auto bias = Set(d, kBias);
  auto half = Set(d, 0.5f);
  for (size_t dy = 0; dy < 8; ++dy) {
    const float* const JXL_RESTRICT row_in_x = xyb_x.Row(y + dy);
    const float* const JXL_RESTRICT row_in_y = xyb_y.Row(y + dy);
    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
      const auto iny = Load(d, row_in_y + x + dx) + bias;
      const auto inx = Load(d, row_in_x + x + dx);
      const auto r = iny - inx;
      const auto g = iny + inx;
      const auto ratio_r =
          RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/true>(d, r);
      const auto ratio_g =
          RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/true>(d, g);
      const auto avg_ratio = half * (ratio_r + ratio_g);

      overall_ratio += avg_ratio;
    }
  }
  overall_ratio = SumOfLanes(overall_ratio);
  overall_ratio *= Set(d, 1.0f / 64);
  // ideally -1.0, but likely optimal correction adds some entropy, so slightly
  // less than that.
  // ln(2) constant folded in because we want std::log but have FastLog2f.
  const auto kGam = Set(d, -0.15526878023684174f * 0.693147180559945f);
  return MulAdd(kGam, FastLog2f(d, overall_ratio), out_val);
}

// Increase precision in 8x8 blocks that have high dynamic range.
template <class D, class V>
V RangeModulation(const D d, const size_t x, const size_t y,
                  const ImageF& xyb_x, const ImageF& xyb_y, const V out_val) {
  auto minval_x = Set(d, 1e30f);
  auto minval_y = Set(d, 1e30f);
  auto maxval_x = Set(d, -1e30f);
  auto maxval_y = Set(d, -1e30f);
  for (size_t dy = 0; dy < 8; ++dy) {
    const float* const JXL_RESTRICT row_in_x = xyb_x.Row(y + dy);
    const float* const JXL_RESTRICT row_in_y = xyb_y.Row(y + dy);
    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
      const auto vx = Load(d, row_in_x + x + dx);
      const auto vy = Load(d, row_in_y + x + dx);
      minval_x = Min(minval_x, vx);
      maxval_x = Max(maxval_x, vx);
      minval_y = Min(minval_y, vy);
      maxval_y = Max(maxval_y, vy);
    }
  }

  const auto min_x = MinOfLanes(minval_x);
  const auto max_x = MaxOfLanes(maxval_x);
  const auto min_y = MinOfLanes(minval_y);
  const auto max_y = MaxOfLanes(maxval_y);

  // TODO(jyrki): should be about 3.0 ?!
  const auto xmul = Set(d, 2.4844670385447296f);
  const auto range_x = xmul * (max_x - min_x);
  const auto range_y = max_y - min_y;
  // This is not really a sound approach but it seems to yield better results
  // than the previous approach of just using range_y.
  const auto range0 = Sqrt(range_x * range_y);
  const auto mul0 = Set(d, -0.62955951344971228f);
  const auto range1 = Sqrt(range_x * range_x + range_y * range_y);
  const auto mul1 = Set(d, 0.50271244247851776f);
  const auto range2 = Max(range_x, range_y);
  const auto mul2 = Set(d, -1.1307891017142655f);
  const auto range3 = Min(range_x, range_y);
  const auto mul3 = Set(d, 0.42443064585099255f);
  const auto range4 = range_x;
  const auto mul4 = Set(d, 4.5935248516155864f);
  // Clamp to [-7, 7] for precaution. Values very far from 0 appear to occur in
  // some pathological cases and cause problems downstream.
  return out_val +
         Clamp(MulAdd(mul0, range0,
                      MulAdd(mul1, range1,
                             MulAdd(mul2, range2,
                                    MulAdd(mul3, range3, mul4 * range4)))),
               Set(d, -7.f), Set(d, 7.f));
}

// Change precision in 8x8 blocks that have high frequency content.
template <class D, class V>
V HfModulation(const D d, const size_t x, const size_t y, const ImageF& xyb,
               const V out_val) {
  // Zero out the invalid differences for the rightmost value per row.
  const Rebind<uint32_t, D> du;
  HWY_ALIGN constexpr uint32_t kMaskRight[kBlockDim] = {~0u, ~0u, ~0u, ~0u,
                                                        ~0u, ~0u, ~0u, 0};

  auto sum = Zero(d);  // sum of absolute differences with right and below

  for (size_t dy = 0; dy < 8; ++dy) {
    const float* JXL_RESTRICT row_in = xyb.Row(y + dy) + x;
    const float* JXL_RESTRICT row_in_next =
        dy == 7 ? row_in : xyb.Row(y + dy + 1) + x;

    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
      const auto p = Load(d, row_in + dx);
      const auto pr = LoadU(d, row_in + dx + 1);
      const auto mask = BitCast(d, Load(du, kMaskRight + dx));
      sum += And(mask, AbsDiff(p, pr));

      const auto pd = Load(d, row_in_next + dx);
      sum += AbsDiff(p, pd);
    }
  }

  sum = SumOfLanes(sum);
  return MulAdd(sum, Set(d, -2.0052193233688884f / 112), out_val);
}

void PerBlockModulations(const float butteraugli_target, const ImageF& xyb_x,
                         const ImageF& xyb_y, const float scale,
                         ThreadPool* pool, ImageF* out) {
  JXL_ASSERT(SameSize(xyb_x, xyb_y));
  JXL_ASSERT(DivCeil(xyb_x.xsize(), kBlockDim) == out->xsize());
  JXL_ASSERT(DivCeil(xyb_x.ysize(), kBlockDim) == out->ysize());

  float base_level = 0.5f * scale;
  float kDampenRampStart = 7.0f;
  float kDampenRampEnd = 14.0f;
  float dampen = 1.0f;
  if (butteraugli_target >= kDampenRampStart) {
    dampen = 1.0f - ((butteraugli_target - kDampenRampStart) /
                     (kDampenRampEnd - kDampenRampStart));
    if (dampen < 0) {
      dampen = 0;
    }
  }
  const float mul = scale * dampen;
  const float add = (1.0f - dampen) * base_level;
  RunOnPool(
      pool, 0, static_cast<uint32_t>(DivCeil(xyb_x.ysize(), kBlockDim)),
      ThreadPool::SkipInit(),
      [&](const int task, const int /*thread*/) {
        const size_t iy = static_cast<size_t>(task);
        const size_t y = iy * 8;
        float* const JXL_RESTRICT row_out = out->Row(iy);

        const HWY_CAPPED(float, kBlockDim) df;

        for (size_t x = 0; x < xyb_x.xsize(); x += 8) {
          auto out_val = Set(df, row_out[x / 8]);
          out_val = ComputeMask(df, out_val);
          out_val = DctModulation(df, x, y, xyb_y, out_val);
          out_val = RangeModulation(df, x, y, xyb_x, xyb_y, out_val);
          out_val = HfModulation(df, x, y, xyb_y, out_val);
          out_val = GammaModulation(df, x, y, xyb_x, xyb_y, out_val);

          // We want multiplicative quantization field, so everything
          // until this point has been modulating the exponent.
          row_out[x / 8] = std::exp(GetLane(out_val)) * mul + add;
        }
      },
      "AQ PerBlockModulations");
}

template <typename D, typename V>
V MaskingLog(const D d, V v) {
  constexpr float kLogOffset = 13.362478857394517f;
  constexpr float kMul = 3.1101290961753842f;
  const auto mul_v = Set(d, kMul * 10000);
  const auto offset_v = Set(d, kLogOffset);
  return FastLog2f(d, MulAdd(v, mul_v, offset_v));
}

float MaskingLog(const float v) {
  using DScalar = HWY_CAPPED(float, 1);
  auto vscalar = Load(DScalar(), &v);
  return GetLane(MaskingLog(DScalar(), vscalar));
}

// Returns image (padded to multiple of 8x8) of local pixel differences.
ImageF DiffPrecompute(const Image3F& xyb, const FrameDimensions& frame_dim,
                      ThreadPool* pool) {
  PROFILER_ZONE("aq DiffPrecompute");
  const size_t xsize = frame_dim.xsize;
  const size_t ysize = frame_dim.ysize;
  const size_t padded_xsize = RoundUpToBlockDim(xsize);
  const size_t padded_ysize = RoundUpToBlockDim(ysize);
  ImageF padded_diff(padded_xsize, padded_ysize);
  const float mul0 = 2.3462298721446873f;

  // The XYB gamma is 3.0 to be able to decode faster with two muls.
  // Butteraugli's gamma is matching the gamma of human eye, around 2.6.
  // We approximate the gamma difference by adding one cubic root into
  // the adaptive quantization. This gives us a total gamma of 2.6666
  // for quantization uses.
  const float match_gamma_offset = 0.16f;

  const HWY_FULL(float) df;

  RunOnPool(
      pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
      [&](const int task, int /*thread*/) {
        const size_t y = static_cast<size_t>(task);
        size_t y2;
        if (y + 1 < ysize) {
          y2 = y + 1;
        } else if (y > 0) {
          y2 = y - 1;
        } else {
          y2 = y;
        }
        size_t y1;
        if (y == 0 && ysize >= 2) {
          y1 = y + 1;
        } else if (y > 0) {
          y1 = y - 1;
        } else {
          y1 = y;
        }
        const float* row_in = xyb.PlaneRow(1, y);
        const float* row_in1 = xyb.PlaneRow(1, y1);
        const float* row_in2 = xyb.PlaneRow(1, y2);
        float* JXL_RESTRICT row_out = padded_diff.Row(y);

        size_t x = 0;
        // First pixel of the row.
        {
          const float base = 0.5f * (row_in2[x] + row_in1[x]);
          const float mul_gammac =
              mul0 * RatioOfDerivativesOfCubicRootToSimpleGamma(
                         row_in[x] + match_gamma_offset);
          float diff = mul_gammac * (row_in[x] - base);
          diff *= diff;
          row_out[x] = MaskingLog(diff);
          ++x;
        }
        // SIMD
        const auto mul0v = Set(df, mul0);
        const auto match_gamma_offset_v = Set(df, match_gamma_offset);
        const auto half = Set(df, 0.5f);
        for (; x + 1 + Lanes(df) < xsize; x += Lanes(df)) {
          const auto in = LoadU(df, row_in + x);
          const auto in_r = LoadU(df, row_in + x + 1);
          const auto in_l = LoadU(df, row_in + x - 1);
          const auto in_t = LoadU(df, row_in2 + x);
          const auto in_b = LoadU(df, row_in1 + x);
          auto base0 = half * (in_r + in_l);
          auto base1 = half * (in_t + in_b);
          auto mul0v_gammac =
              mul0v *
              RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/false>(
                  df, in + match_gamma_offset_v);
          auto diff0 = mul0v_gammac * (in - base0);
          auto diff1 = mul0v_gammac * (in - base1);
          auto diff = diff0 * diff0 + diff1 * diff1;
          diff = MaskingLog(df, diff);
          StoreU(diff, df, row_out + x);
        }
        // Scalar
        for (; x + 1 < xsize; ++x) {
          const size_t x2 = x + 1;
          const size_t x1 = x - 1;
          const float base0 = 0.5f * (row_in2[x] + row_in1[x]);
          const float base1 = 0.5f * (row_in[x1] + row_in[x2]);
          const float mul_gammac =
              mul0 * RatioOfDerivativesOfCubicRootToSimpleGamma(
                         row_in[x] + match_gamma_offset);
          float diff0 = mul_gammac * (row_in[x] - base0);
          float diff1 = mul_gammac * (row_in[x] - base1);
          float diff = diff0 * diff0 + diff1 * diff1;
          row_out[x] = MaskingLog(diff);
        }
        // Last pixel of the row.
        {
          const float base = (1.0f / 2.0f) * (row_in2[x] + row_in1[x]);
          const float mul_gammac =
              mul0 * RatioOfDerivativesOfCubicRootToSimpleGamma(
                         row_in[x] + match_gamma_offset);
          float diff = mul_gammac * (row_in[x] - base);
          diff *= diff;
          row_out[x] = MaskingLog(diff);
          ++x;
        }

        // Extend to multiple of 8 columns
        float lastval = row_out[xsize - 1];
        if (xsize >= 3) {
          lastval += row_out[xsize - 3];
          lastval += row_out[xsize - 2];
          lastval *= 1.0f / 3;
        } else if (xsize >= 2) {
          lastval += row_out[xsize - 2];
          lastval *= 0.5f;
        }
        for (; x < padded_diff.xsize(); ++x) {
          row_out[x] = lastval;
        }
      },
      "AQ DiffPrecompute");

  // Last row.
  {
    const size_t y = ysize - 1;
    const float* const JXL_RESTRICT row_in = xyb.PlaneRow(1, y);
    float* const JXL_RESTRICT row_out = padded_diff.Row(y);
    for (size_t x = 0; x + 1 < xsize; ++x) {
      const size_t x2 = x + 1;
      const size_t x1 = (x == 0) ? x2 : x - 1;
      const float base = 0.5f * (row_in[x1] + row_in[x2]);
      const float mul_gammac =
          mul0 * RatioOfDerivativesOfCubicRootToSimpleGamma(row_in[x] +
                                                            match_gamma_offset);
      float diff = mul_gammac * (row_in[x] - base);
      diff *= diff;
      row_out[x] = MaskingLog(diff);
    }
    // Last pixel of the last row.
    {
      const size_t x = xsize - 1;
      if (x > 0) {
        row_out[x] = row_out[x - 1];
      }
    }
  }
  // Extend to multiple of 8 rows
  if (ysize != padded_diff.ysize()) {
    const float* JXL_RESTRICT last_row = padded_diff.Row(ysize - 1);
    for (size_t x = 0; x < padded_diff.xsize(); ++x) {
      float lastval = last_row[x];
      if (ysize >= 3) {
        lastval += padded_diff.Row(ysize - 2)[x];
        lastval += padded_diff.Row(ysize - 3)[x];
        lastval *= 1.0f / 3;
      } else if (ysize >= 2) {
        lastval += padded_diff.Row(ysize - 2)[x];
        lastval *= 0.5f;
      }
      for (size_t y = ysize; y < padded_diff.ysize(); ++y) {
        padded_diff.Row(y)[x] = lastval;
      }
    }
  }

  return padded_diff;
}

}  // namespace

ImageF AdaptiveQuantizationMap(const float butteraugli_target,
                               const Image3F& opsin,
                               const ImageF& intensity_ac_x,
                               const ImageF& intensity_ac_y,
                               const FrameDimensions& frame_dim, float scale,
                               ThreadPool* pool) {
  PROFILER_ZONE("aq AdaptiveQuantMap");
  const float limited_butteraugli_target = std::min(16.0f, butteraugli_target);
  constexpr float kSigmaBase = 7;
  constexpr float kSigmaMul = 0.3f;

  const float kSigma = kSigmaBase + kSigmaMul * limited_butteraugli_target;
  const int kRadius = static_cast<int>(2 * kSigma + 0.5f);
  std::vector<float> kernel = GaussianKernel(kRadius, kSigma);

  ImageF out = DiffPrecompute(opsin, frame_dim, pool);
  JXL_ASSERT(out.xsize() % kBlockDim == 0 && out.ysize() % kBlockDim == 0);

  // (Faster than RecursiveGaussian due to the subsampling)
  out = ConvolveAndSample(out, kernel, kBlockDim);
  PerBlockModulations(butteraugli_target, intensity_ac_x, intensity_ac_y, scale,
                      pool, &out);
  return out;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(AdaptiveQuantizationMap);

namespace {
bool FLAGS_log_search_state = false;
// If true, prints the quantization maps at each iteration.
bool FLAGS_dump_quant_state = false;

bool AdjustQuantVal(float* const JXL_RESTRICT q, const float d,
                    const float factor, const float quant_max) {
  if (*q >= 0.999f * quant_max) return false;
  const float inv_q = 1.0f / *q;
  const float adj_inv_q = inv_q - factor / (d + 1.0f);
  *q = 1.0f / std::max(1.0f / quant_max, adj_inv_q);
  return true;
}

void DumpHeatmap(const AuxOut* aux_out, const std::string& label,
                 const ImageF& image, float good_threshold,
                 float bad_threshold) {
  Image3B heatmap = CreateHeatMapImage(image, good_threshold, bad_threshold);
  char filename[200];
  snprintf(filename, sizeof(filename), "%s%05d", label.c_str(),
           aux_out->num_butteraugli_iters);
  aux_out->DumpImage(filename, heatmap);
}

void DumpHeatmaps(const AuxOut* aux_out, float ba_target,
                  const ImageF& quant_field, const ImageF& tile_heatmap) {
  if (!WantDebugOutput(aux_out)) return;
  ImageF inv_qmap(quant_field.xsize(), quant_field.ysize());
  for (size_t y = 0; y < quant_field.ysize(); ++y) {
    const float* JXL_RESTRICT row_q = quant_field.ConstRow(y);
    float* JXL_RESTRICT row_inv_q = inv_qmap.Row(y);
    for (size_t x = 0; x < quant_field.xsize(); ++x) {
      row_inv_q[x] = 1.0f / row_q[x];  // never zero
    }
  }
  DumpHeatmap(aux_out, "quant_heatmap", inv_qmap, 4.0f * ba_target,
              6.0f * ba_target);
  DumpHeatmap(aux_out, "tile_heatmap", tile_heatmap, ba_target,
              1.5f * ba_target);
}

ImageF TileDistMap(const ImageF& distmap, int tile_size, int margin,
                   const AcStrategyImage& ac_strategy) {
  PROFILER_FUNC;
  const int tile_xsize = (distmap.xsize() + tile_size - 1) / tile_size;
  const int tile_ysize = (distmap.ysize() + tile_size - 1) / tile_size;
  ImageF tile_distmap(tile_xsize, tile_ysize);
  size_t distmap_stride = tile_distmap.PixelsPerRow();
  for (int tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(tile_y);
    float* JXL_RESTRICT dist_row = tile_distmap.Row(tile_y);
    for (int tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      AcStrategy acs = ac_strategy_row[tile_x];
      if (!acs.IsFirstBlock()) continue;
      int this_tile_xsize = acs.covered_blocks_x() * tile_size;
      int this_tile_ysize = acs.covered_blocks_y() * tile_size;
      int y_begin = std::max<int>(0, tile_size * tile_y - margin);
      int y_end = std::min<int>(distmap.ysize(),
                                tile_size * tile_y + this_tile_ysize + margin);
      int x_begin = std::max<int>(0, tile_size * tile_x - margin);
      int x_end = std::min<int>(distmap.xsize(),
                                tile_size * tile_x + this_tile_xsize + margin);
      float dist_norm = 0.0;
      double pixels = 0;
      for (int y = y_begin; y < y_end; ++y) {
        float ymul = 1.0;
        constexpr float kBorderMul = 0.98f;
        constexpr float kCornerMul = 0.7f;
        if (margin != 0 && (y == y_begin || y == y_end - 1)) {
          ymul = kBorderMul;
        }
        const float* const JXL_RESTRICT row = distmap.Row(y);
        for (int x = x_begin; x < x_end; ++x) {
          float xmul = ymul;
          if (margin != 0 && (x == x_begin || x == x_end - 1)) {
            if (xmul == 1.0) {
              xmul = kBorderMul;
            } else {
              xmul = kCornerMul;
            }
          }
          float v = row[x];
          v *= v;
          v *= v;
          v *= v;
          v *= v;
          dist_norm += xmul * v;
          pixels += xmul;
        }
      }
      if (pixels == 0) pixels = 1;
      // 16th norm is less than the max norm, we reduce the difference
      // with this normalization factor.
      constexpr float kTileNorm = 1.2f;
      const float tile_dist =
          kTileNorm * std::pow(dist_norm / pixels, 1.0f / 16.0f);
      dist_row[tile_x] = tile_dist;
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          dist_row[tile_x + distmap_stride * iy + ix] = tile_dist;
        }
      }
    }
  }
  return tile_distmap;
}

ImageF DistToPeakMap(const ImageF& field, float peak_min, int local_radius,
                     float peak_weight) {
  ImageF result(field.xsize(), field.ysize());
  FillImage(-1.0f, &result);
  for (size_t y0 = 0; y0 < field.ysize(); ++y0) {
    for (size_t x0 = 0; x0 < field.xsize(); ++x0) {
      int x_min = std::max<int>(0, static_cast<int>(x0) - local_radius);
      int y_min = std::max<int>(0, static_cast<int>(y0) - local_radius);
      int x_max = std::min<size_t>(field.xsize(), x0 + 1 + local_radius);
      int y_max = std::min<size_t>(field.ysize(), y0 + 1 + local_radius);
      float local_max = peak_min;
      for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
          local_max = std::max(local_max, field.Row(y)[x]);
        }
      }
      if (field.Row(y0)[x0] >
          (1.0f - peak_weight) * peak_min + peak_weight * local_max) {
        for (int y = y_min; y < y_max; ++y) {
          for (int x = x_min; x < x_max; ++x) {
            float dist = std::max(std::abs(y - static_cast<int>(y0)),
                                  std::abs(x - static_cast<int>(x0)));
            float cur_dist = result.Row(y)[x];
            if (cur_dist < 0.0 || cur_dist > dist) {
              result.Row(y)[x] = dist;
            }
          }
        }
      }
    }
  }
  return result;
}

void AdjustQuantField(const AcStrategyImage& ac_strategy, ImageF* quant_field) {
  // Replace the whole quant_field in non-8x8 blocks with the maximum of each
  // 8x8 block.
  size_t stride = quant_field->PixelsPerRow();
  for (size_t y = 0; y < quant_field->ysize(); ++y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(y);
    float* JXL_RESTRICT quant_row = quant_field->Row(y);
    for (size_t x = 0; x < quant_field->xsize(); ++x) {
      AcStrategy acs = ac_strategy_row[x];
      if (!acs.IsFirstBlock()) continue;
      JXL_ASSERT(x + acs.covered_blocks_x() <= quant_field->xsize());
      JXL_ASSERT(y + acs.covered_blocks_y() <= quant_field->ysize());
      float max = quant_row[x];
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          max = std::max(quant_row[x + ix + iy * stride], max);
        }
      }
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          quant_row[x + ix + iy * stride] = max;
        }
      }
    }
  }
}

constexpr float kDcQuantPow = 0.55f;
constexpr float kDcQuant = 1.18f;
constexpr float kAcQuant = 0.84f;

void FindBestQuantization(const ImageBundle& linear, const Image3F& opsin,
                          PassesEncoderState* enc_state, ThreadPool* pool,
                          AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;
  ImageF& quant_field = enc_state->initial_quant_field;

  const float butteraugli_target = cparams.butteraugli_distance;
  ButteraugliParams params = cparams.ba_params;
  params.intensity_target = linear.metadata()->IntensityTarget();
  JxlButteraugliComparator comparator(params);
  ImageMetadata metadata;
  JXL_CHECK(comparator.SetReferenceImage(linear));
  bool lower_is_better =
      (comparator.GoodQualityScore() < comparator.BadQualityScore());
  const float initial_quant_dc = InitialQuantDC(butteraugli_target);
  AdjustQuantField(enc_state->shared.ac_strategy, &quant_field);
  ImageF tile_distmap;
  ImageF tile_distmap_localopt;
  ImageF initial_quant_field = CopyImage(quant_field);
  ImageF last_quant_field = CopyImage(initial_quant_field);
  ImageF last_tile_distmap_localopt;

  float initial_qf_min, initial_qf_max;
  ImageMinMax(initial_quant_field, &initial_qf_min, &initial_qf_max);
  float initial_qf_ratio = initial_qf_max / initial_qf_min;
  float qf_max_deviation_low = std::sqrt(250 / initial_qf_ratio);
  float asymmetry = 2;
  if (qf_max_deviation_low < asymmetry) asymmetry = qf_max_deviation_low;
  float qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
  float qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);

  JXL_ASSERT(qf_higher / qf_lower < 253);

  constexpr int kOriginalComparisonRound = 1;
  constexpr float kMaximumDistanceIncreaseFactor = 1.015;

  for (int i = 0; i < cparams.max_butteraugli_iters + 1; ++i) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }

    quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field);
    ImageMetadata metadata;
    metadata.SetFloat32Samples();
    metadata.color_encoding = ColorEncoding::LinearSRGB();
    metadata.SetIntensityTarget(linear.metadata()->IntensityTarget());
    ImageBundle linear(&metadata);
    linear.SetFromImage(RoundtripImage(opsin, enc_state, pool),
                        metadata.color_encoding);
    PROFILER_ZONE("enc Butteraugli");
    float score;
    ImageF diffmap;
    JXL_CHECK(comparator.CompareWith(linear, &diffmap, &score));
    if (!lower_is_better) {
      score = -score;
      diffmap = ScaleImage(-1.0f, diffmap);
    }
    static constexpr int kMargins[100] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    tile_distmap =
        TileDistMap(diffmap, 8, kMargins[i], enc_state->shared.ac_strategy);
    tile_distmap_localopt =
        TileDistMap(diffmap, 8, 2, enc_state->shared.ac_strategy);
    if (WantDebugOutput(aux_out)) {
      DumpHeatmaps(aux_out, butteraugli_target, quant_field, tile_distmap);
    }
    if (aux_out != nullptr) ++aux_out->num_butteraugli_iters;
    if (FLAGS_log_search_state) {
      float minval, maxval;
      ImageMinMax(quant_field, &minval, &maxval);
      printf("\nButteraugli iter: %d/%d\n", i, cparams.max_butteraugli_iters);
      printf("Butteraugli distance: %f\n", score);
      printf("quant range: %f ... %f  DC quant: %f\n", minval, maxval,
             initial_quant_dc);
      if (FLAGS_dump_quant_state) {
        quantizer.DumpQuantizationMap(raw_quant_field);
      }
    }

    if (i > kOriginalComparisonRound) {
      // Undo last round if it made things worse (i.e. increased the quant value
      // AND the distance in nearby pixels by at least some percentage).
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        const float* const JXL_RESTRICT row_dist = tile_distmap_localopt.Row(y);
        const float* const JXL_RESTRICT row_last_dist =
            last_tile_distmap_localopt.Row(y);
        const float* const JXL_RESTRICT row_last_q = last_quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          if (row_q[x] > row_last_q[x] &&
              row_dist[x] > kMaximumDistanceIncreaseFactor * row_last_dist[x]) {
            row_q[x] = row_last_q[x];
          }
        }
      }
    }
    last_quant_field = CopyImage(quant_field);
    last_tile_distmap_localopt = CopyImage(tile_distmap_localopt);
    if (i == cparams.max_butteraugli_iters) break;

    double kPow[8] = {
        0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    double kPowMod[8] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    if (i == kOriginalComparisonRound) {
      // Don't allow optimization to make the quant field a lot worse than
      // what the initial guess was. This allows the AC field to have enough
      // precision to reduce the oscillations due to the dc reconstruction.
      double kInitMul = 0.6;
      const double kOneMinusInitMul = 1.0 - kInitMul;
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        const float* const JXL_RESTRICT row_init = initial_quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          double clamp = kOneMinusInitMul * row_q[x] + kInitMul * row_init[x];
          if (row_q[x] < clamp) {
            row_q[x] = clamp;
            if (row_q[x] > qf_higher) row_q[x] = qf_higher;
            if (row_q[x] < qf_lower) row_q[x] = qf_lower;
          }
        }
      }
    }

    double cur_pow = 0.0;
    if (i < 7) {
      cur_pow = kPow[i] + (butteraugli_target - 1.0) * kPowMod[i];
      if (cur_pow < 0) {
        cur_pow = 0;
      }
    }
    // pow(x, 0) == 1, so skip pow.
    if (cur_pow == 0.0) {
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        const float* const JXL_RESTRICT row_dist = tile_distmap.Row(y);
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / butteraugli_target;
          if (diff > 1.0f) {
            float old = row_q[x];
            row_q[x] *= diff;
            int qf_old = old * quantizer.InvGlobalScale() + 0.5;
            int qf_new = row_q[x] * quantizer.InvGlobalScale() + 0.5;
            if (qf_old == qf_new) {
              row_q[x] = old + quantizer.Scale();
            }
          }
          if (row_q[x] > qf_higher) row_q[x] = qf_higher;
          if (row_q[x] < qf_lower) row_q[x] = qf_lower;
        }
      }
    } else {
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        const float* const JXL_RESTRICT row_dist = tile_distmap.Row(y);
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / butteraugli_target;
          if (diff <= 1.0f) {
            row_q[x] *= std::pow(diff, cur_pow);
          } else {
            float old = row_q[x];
            row_q[x] *= diff;
            int qf_old = old * quantizer.InvGlobalScale() + 0.5;
            int qf_new = row_q[x] * quantizer.InvGlobalScale() + 0.5;
            if (qf_old == qf_new) {
              row_q[x] = old + quantizer.Scale();
            }
          }
          if (row_q[x] > qf_higher) row_q[x] = qf_higher;
          if (row_q[x] < qf_lower) row_q[x] = qf_lower;
        }
      }
    }
  }
  quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field);
}

void FindBestQuantizationMaxError(const Image3F& opsin,
                                  PassesEncoderState* enc_state,
                                  ThreadPool* pool, AuxOut* aux_out) {
  // TODO(veluca): this only works if opsin is in XYB. The current encoder does
  // not have code paths that produce non-XYB opsin here.
  JXL_CHECK(enc_state->shared.frame_header.color_transform ==
            ColorTransform::kXYB);
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;
  ImageF& quant_field = enc_state->initial_quant_field;

  // TODO(veluca): better choice of this value.
  const float initial_quant_dc =
      16 * std::sqrt(0.1f / cparams.butteraugli_distance);
  AdjustQuantField(enc_state->shared.ac_strategy, &quant_field);

  const float inv_max_err[3] = {1.0f / enc_state->cparams.max_error[0],
                                1.0f / enc_state->cparams.max_error[1],
                                1.0f / enc_state->cparams.max_error[2]};

  for (int i = 0; i < cparams.max_butteraugli_iters + 1; ++i) {
    quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field);
    if (aux_out)
      aux_out->DumpXybImage(("ops" + std::to_string(i)).c_str(), opsin);
    Image3F decoded = RoundtripImage(opsin, enc_state, pool);
    if (aux_out)
      aux_out->DumpXybImage(("dec" + std::to_string(i)).c_str(), decoded);

    for (size_t by = 0; by < enc_state->shared.frame_dim.ysize_blocks; by++) {
      AcStrategyRow ac_strategy_row =
          enc_state->shared.ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < enc_state->shared.frame_dim.xsize_blocks; bx++) {
        AcStrategy acs = ac_strategy_row[bx];
        if (!acs.IsFirstBlock()) continue;
        float max_error = 0;
        for (size_t c = 0; c < 3; c++) {
          for (size_t y = by * kBlockDim;
               y < (by + acs.covered_blocks_y()) * kBlockDim; y++) {
            if (y >= decoded.ysize()) continue;
            const float* JXL_RESTRICT in_row = opsin.ConstPlaneRow(c, y);
            const float* JXL_RESTRICT dec_row = decoded.ConstPlaneRow(c, y);
            for (size_t x = bx * kBlockDim;
                 x < (bx + acs.covered_blocks_x()) * kBlockDim; x++) {
              if (x >= decoded.xsize()) continue;
              max_error = std::max(
                  std::abs(in_row[x] - dec_row[x]) * inv_max_err[c], max_error);
            }
          }
        }
        // Target an error between max_error/2 and max_error.
        // If the error in the varblock is above the target, increase the qf to
        // compensate. If the error is below the target, decrease the qf.
        // However, to avoid an excessive increase of the qf, only do so if the
        // error is less than half the maximum allowed error.
        const float qf_mul = (max_error < 0.5f)   ? max_error * 2.0f
                             : (max_error > 1.0f) ? max_error
                                                  : 1.0f;
        for (size_t qy = by; qy < by + acs.covered_blocks_y(); qy++) {
          float* JXL_RESTRICT quant_field_row = quant_field.Row(qy);
          for (size_t qx = bx; qx < bx + acs.covered_blocks_x(); qx++) {
            quant_field_row[qx] *= qf_mul;
          }
        }
      }
    }
  }
  quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field);
}

void FindBestQuantizationHQ(const ImageBundle& linear, const Image3F& opsin,
                            PassesEncoderState* enc_state, ThreadPool* pool,
                            AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;
  ImageF& quant_field = enc_state->initial_quant_field;
  const AcStrategyImage& ac_strategy = enc_state->shared.ac_strategy;

  ButteraugliParams params = cparams.ba_params;
  params.intensity_target = linear.metadata()->IntensityTarget();
  JxlButteraugliComparator comparator(params);
  ImageMetadata metadata;
  JXL_CHECK(comparator.SetReferenceImage(linear));
  AdjustQuantField(ac_strategy, &quant_field);
  ImageF best_quant_field = CopyImage(quant_field);
  bool lower_is_better =
      (comparator.GoodQualityScore() < comparator.BadQualityScore());
  float best_score = 1000000.0f;
  ImageF tile_distmap;
  constexpr int kMaxOuterIters = 2;
  int outer_iter = 0;
  int butteraugli_iter = 0;
  int search_radius = 0;
  float quant_ceil = 5.0f;
  float quant_dc = 1.2f;
  float best_quant_dc = quant_dc;
  int num_stalling_iters = 0;
  int max_iters = cparams.max_butteraugli_iters_guetzli_mode;
  const float butteraugli_target = cparams.butteraugli_distance;

  for (;;) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }
    float qmin, qmax;
    ImageMinMax(quant_field, &qmin, &qmax);
    ++butteraugli_iter;
    float score = 0.0;
    ImageF diffmap;
    quantizer.SetQuantField(quant_dc, quant_field, &raw_quant_field);
    ImageMetadata metadata;
    metadata.SetFloat32Samples();
    metadata.color_encoding = ColorEncoding::LinearSRGB();
    metadata.SetIntensityTarget(linear.metadata()->IntensityTarget());
    ImageBundle linear(&metadata);
    linear.SetFromImage(RoundtripImage(opsin, enc_state, pool),
                        metadata.color_encoding);
    JXL_CHECK(comparator.CompareWith(linear, &diffmap, &score));

    if (!lower_is_better) {
      score = -score;
      ScaleImage(-1.0f, &diffmap);
    }
    bool best_quant_updated = false;
    if (score <= best_score) {
      best_quant_field = CopyImage(quant_field);
      best_score = std::max<float>(score, butteraugli_target);
      best_quant_updated = true;
      best_quant_dc = quant_dc;
      num_stalling_iters = 0;
    } else if (outer_iter == 0) {
      ++num_stalling_iters;
    }
    tile_distmap = TileDistMap(diffmap, 8, 0, ac_strategy);
    if (WantDebugOutput(aux_out)) {
      DumpHeatmaps(aux_out, butteraugli_target, quant_field, tile_distmap);
    }
    if (aux_out) {
      ++aux_out->num_butteraugli_iters;
    }
    if (FLAGS_log_search_state) {
      float minval, maxval;
      ImageMinMax(quant_field, &minval, &maxval);
      printf("\nButteraugli iter: %d/%d%s\n", butteraugli_iter, max_iters,
             best_quant_updated ? " (*)" : "");
      printf("Butteraugli distance: %f\n", score);
      printf(
          "quant range: %f ... %f  DC quant: "
          "%f\n",
          minval, maxval, quant_dc);
      printf("search radius: %d\n", search_radius);
      if (FLAGS_dump_quant_state) {
        quantizer.DumpQuantizationMap(raw_quant_field);
      }
    }
    if (butteraugli_iter >= max_iters) {
      break;
    }
    bool changed = false;
    while (!changed && score > butteraugli_target) {
      for (int radius = 0; radius <= search_radius && !changed; ++radius) {
        ImageF dist_to_peak_map =
            DistToPeakMap(tile_distmap, butteraugli_target, radius, 0.0);
        for (size_t y = 0; y < quant_field.ysize(); ++y) {
          float* const JXL_RESTRICT row_q = quant_field.Row(y);
          const float* const JXL_RESTRICT row_dist = dist_to_peak_map.Row(y);
          for (size_t x = 0; x < quant_field.xsize(); ++x) {
            if (row_dist[x] >= 0.0f) {
              static constexpr float kAdjSpeed[kMaxOuterIters] = {0.1f, 0.04f};
              const float factor =
                  kAdjSpeed[outer_iter] * tile_distmap.Row(y)[x];
              if (AdjustQuantVal(&row_q[x], row_dist[x], factor, quant_ceil)) {
                changed = true;
              }
            }
          }
        }
      }
      if (!changed || num_stalling_iters >= 3) {
        // Try to extend the search parameters.
        if ((search_radius < 4) &&
            (qmax < 0.99f * quant_ceil || quant_ceil >= 3.0f + search_radius)) {
          ++search_radius;
          continue;
        }
        if (quant_dc < 0.4f * quant_ceil - 0.8f) {
          quant_dc += 0.2f;
          changed = true;
          continue;
        }
        if (quant_ceil < 8.0f) {
          quant_ceil += 0.5f;
          continue;
        }
        break;
      }
    }
    if (!changed) {
      if (++outer_iter == kMaxOuterIters) break;
      constexpr float kQuantScale = 0.75f;
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          quant_field.Row(y)[x] *= kQuantScale;
        }
      }
      num_stalling_iters = 0;
    }
  }
  quantizer.SetQuantField(best_quant_dc, best_quant_field, &raw_quant_field);
}

}  // namespace

float InitialQuantDC(float butteraugli_target) {
  const float kDcMul = 2.9;  // Butteraugli target where non-linearity kicks in.
  const float butteraugli_target_dc = std::max<float>(
      0.85f * butteraugli_target,
      std::min<float>(butteraugli_target,
                      kDcMul * std::pow((1.0f / kDcMul) * butteraugli_target,
                                        kDcQuantPow)));
  // We want the maximum DC value to be at most 2**15 * kInvDCQuant / quant_dc.
  // The maximum DC value might not be in the kXybRange because of inverse
  // gaborish, so we add some slack to the maximum theoretical quant obtained
  // this way (64).
  return std::min(kDcQuant / butteraugli_target_dc, 50.f);
}

ImageF InitialQuantField(const float butteraugli_target, const Image3F& opsin,
                         const FrameDimensions& frame_dim, ThreadPool* pool,
                         float rescale) {
  PROFILER_FUNC;
  const float quant_ac = kAcQuant / butteraugli_target;
  return HWY_DYNAMIC_DISPATCH(AdaptiveQuantizationMap)(
      butteraugli_target, opsin, opsin.Plane(0), opsin.Plane(1), frame_dim,
      quant_ac * rescale, pool);
}

void FindBestQuantizer(const ImageBundle* linear, const Image3F& opsin,
                       PassesEncoderState* enc_state, ThreadPool* pool,
                       AuxOut* aux_out, double rescale) {
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;
  if (cparams.max_error_mode) {
    PROFILER_ZONE("enc find best maxerr");
    FindBestQuantizationMaxError(opsin, enc_state, pool, aux_out);
  } else if (cparams.speed_tier == SpeedTier::kFalcon) {
    const float quant_dc = InitialQuantDC(cparams.butteraugli_distance);
    // TODO(veluca): tune constant.
    const float quant_ac = kAcQuant / cparams.butteraugli_distance;
    quantizer.SetQuant(quant_dc, quant_ac, &raw_quant_field);
  } else if (cparams.uniform_quant > 0.0) {
    quantizer.SetQuant(cparams.uniform_quant * rescale,
                       cparams.uniform_quant * rescale, &raw_quant_field);
  } else if (cparams.speed_tier > SpeedTier::kKitten) {
    PROFILER_ZONE("enc fast quant");
    const float quant_dc = InitialQuantDC(cparams.butteraugli_distance);
    AdjustQuantField(enc_state->shared.ac_strategy,
                     &enc_state->initial_quant_field);
    quantizer.SetQuantField(quant_dc, enc_state->initial_quant_field,
                            &raw_quant_field);
  } else {
    // Normal encoding to a butteraugli score.
    PROFILER_ZONE("enc find best2");
    if (cparams.speed_tier == SpeedTier::kTortoise) {
      FindBestQuantizationHQ(*linear, opsin, enc_state, pool, aux_out);
    } else {
      FindBestQuantization(*linear, opsin, enc_state, pool, aux_out);
    }
  }
}

Image3F RoundtripImage(const Image3F& opsin, PassesEncoderState* enc_state,
                       ThreadPool* pool) {
  PROFILER_ZONE("enc roundtrip");
  PassesDecoderState dec_state;
  dec_state.shared = &enc_state->shared;
  JXL_ASSERT(opsin.ysize() % kBlockDim == 0);

  const size_t xsize_groups = DivCeil(opsin.xsize(), kGroupDim);
  const size_t ysize_groups = DivCeil(opsin.ysize(), kGroupDim);
  const size_t num_groups = xsize_groups * ysize_groups;

  // Dummy metadata with grayscale = off.
  ImageMetadata metadata = enc_state->shared.metadata->m;
  metadata.color_encoding = ColorEncoding::SRGB();

  ModularFrameEncoder modular_frame_encoder(enc_state->shared.frame_header,
                                            enc_state->cparams);
  InitializePassesEncoder(opsin, pool, enc_state, &modular_frame_encoder,
                          nullptr);
  dec_state.Init(pool);

  Image3F idct(opsin.xsize(), opsin.ysize());
  ImageBundle decoded(&metadata);

  hwy::AlignedUniquePtr<GroupDecCache[]> group_dec_caches;
  const auto allocate_storage = [&](size_t num_threads) {
    dec_state.EnsureStorage(num_threads);
    group_dec_caches = hwy::MakeUniqueAlignedArray<GroupDecCache>(num_threads);
    return true;
  };
  const auto process_group = [&](const int group_index, const int thread) {
    if (dec_state.shared->frame_header.loop_filter.epf_iters > 0) {
      ComputeSigma(dec_state.shared->BlockGroupRect(group_index), &dec_state);
    }
    JXL_CHECK(DecodeGroupForRoundtrip(enc_state->coeffs, group_index,
                                      &dec_state, &group_dec_caches[thread],
                                      thread, &idct, &decoded, nullptr));
  };
  RunOnPool(pool, 0, num_groups, allocate_storage, process_group, "AQ loop");

  // Fine to do a JXL_ASSERT instead of error handling, since this only happens
  // on the encoder side where we can't be fed with invalid data.
  JXL_CHECK(FinalizeFrameDecoding(&idct, &dec_state, pool, nullptr));
  return idct;
}

}  // namespace jxl
#endif  // HWY_ONCE
