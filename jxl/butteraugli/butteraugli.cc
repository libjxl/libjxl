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
//
// Author: Jyrki Alakuijala (jyrki.alakuijala@gmail.com)
//
// The physical architecture of butteraugli is based on the following naming
// convention:
//   * Opsin - dynamics of the photosensitive chemicals in the retina
//             with their immediate electrical processing
//   * Xyb - hybrid opponent/trichromatic color space
//     x is roughly red-subtract-green.
//     y is yellow.
//     b is blue.
//     Xyb values are computed from Opsin mixing, not directly from rgb.
//   * Mask - for visual masking
//   * Hf - color modeling for spatially high-frequency features
//   * Lf - color modeling for spatially low-frequency features
//   * Diffmap - to cluster and build an image of error between the images
//   * Blur - to hold the smoothing code

#include "jxl/butteraugli/butteraugli.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <new>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/butteraugli/butteraugli.cc"
#include <hwy/foreach_target.h>
//

#include "jxl/base/os_specific.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/convolve.h"
#include "jxl/fast_log-inl.h"
#include "jxl/image_ops.h"

#ifndef JXL_BUTTERAUGLI_ONCE
#define JXL_BUTTERAUGLI_ONCE

namespace jxl {

std::vector<float> ComputeKernel(float sigma) {
  const float m = 2.25;  // Accuracy increases when m is increased.
  const float scaler = -1.0 / (2 * sigma * sigma);
  const int diff = std::max<int>(1, m * std::fabs(sigma));
  std::vector<float> kernel(2 * diff + 1);
  for (int i = -diff; i <= diff; ++i) {
    kernel[i + diff] = std::exp(scaler * i * i);
  }
  return kernel;
}

void ConvolveBorderColumn(const ImageF& in, const std::vector<float>& kernel,
                          const float weight_no_border,
                          const float border_ratio, const size_t x,
                          float* BUTTERAUGLI_RESTRICT row_out) {
  const size_t offset = kernel.size() / 2;
  int minx = x < offset ? 0 : x - offset;
  int maxx = std::min<int>(in.xsize() - 1, x + offset);
  float weight = 0.0f;
  for (int j = minx; j <= maxx; ++j) {
    weight += kernel[j - x + offset];
  }
  // Interpolate linearly between the no-border scaling and border scaling.
  weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
  float scale = 1.0f / weight;
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y);
    float sum = 0.0f;
    for (int j = minx; j <= maxx; ++j) {
      sum += row_in[j] * kernel[j - x + offset];
    }
    row_out[y] = sum * scale;
  }
}

// Computes a horizontal convolution and transposes the result.
void ConvolutionWithTranspose(const ImageF& in,
                              const std::vector<float>& kernel,
                              const float border_ratio,
                              ImageF* BUTTERAUGLI_RESTRICT out) {
  PROFILER_FUNC;
  JXL_CHECK(out->xsize() == in.ysize());
  JXL_CHECK(out->ysize() == in.xsize());
  const size_t len = kernel.size();
  const size_t offset = len / 2;
  float weight_no_border = 0.0f;
  for (size_t j = 0; j < len; ++j) {
    weight_no_border += kernel[j];
  }
  const float scale_no_border = 1.0f / weight_no_border;
  const size_t border1 = std::min(in.xsize(), offset);
  const size_t border2 = in.xsize() > offset ? in.xsize() - offset : 0;
  std::vector<float> scaled_kernel(len / 2 + 1);
  for (size_t i = 0; i <= len / 2; ++i) {
    scaled_kernel[i] = kernel[i] * scale_no_border;
  }

  // middle
  switch (len) {
#if 1  // speed-optimized version
    case 7: {
      PROFILER_ZONE("conv7");
      const float sk0 = scaled_kernel[0];
      const float sk1 = scaled_kernel[1];
      const float sk2 = scaled_kernel[2];
      const float sk3 = scaled_kernel[3];
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          const float sum0 = (row_in[0] + row_in[6]) * sk0;
          const float sum1 = (row_in[1] + row_in[5]) * sk1;
          const float sum2 = (row_in[2] + row_in[4]) * sk2;
          const float sum = (row_in[3]) * sk3 + sum0 + sum1 + sum2;
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum;
        }
      }
    } break;
    case 15: {
      PROFILER_ZONE("conv15");
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum0 = (row_in[0] + row_in[14]) * scaled_kernel[0];
          float sum1 = (row_in[1] + row_in[13]) * scaled_kernel[1];
          float sum2 = (row_in[2] + row_in[12]) * scaled_kernel[2];
          float sum3 = (row_in[3] + row_in[11]) * scaled_kernel[3];
          sum0 += (row_in[4] + row_in[10]) * scaled_kernel[4];
          sum1 += (row_in[5] + row_in[9]) * scaled_kernel[5];
          sum2 += (row_in[6] + row_in[8]) * scaled_kernel[6];
          const float sum = (row_in[7]) * scaled_kernel[7];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum + sum0 + sum1 + sum2 + sum3;
        }
      }
      break;
    }
    case 25: {
      PROFILER_ZONE("conv25");
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum0 = (row_in[0] + row_in[24]) * scaled_kernel[0];
          float sum1 = (row_in[1] + row_in[23]) * scaled_kernel[1];
          float sum2 = (row_in[2] + row_in[22]) * scaled_kernel[2];
          float sum3 = (row_in[3] + row_in[21]) * scaled_kernel[3];
          sum0 += (row_in[4] + row_in[20]) * scaled_kernel[4];
          sum1 += (row_in[5] + row_in[19]) * scaled_kernel[5];
          sum2 += (row_in[6] + row_in[18]) * scaled_kernel[6];
          sum3 += (row_in[7] + row_in[17]) * scaled_kernel[7];
          sum0 += (row_in[8] + row_in[16]) * scaled_kernel[8];
          sum1 += (row_in[9] + row_in[15]) * scaled_kernel[9];
          sum2 += (row_in[10] + row_in[14]) * scaled_kernel[10];
          sum3 += (row_in[11] + row_in[13]) * scaled_kernel[11];
          const float sum = (row_in[12]) * scaled_kernel[12];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum + sum0 + sum1 + sum2 + sum3;
        }
      }
      break;
    }
    case 33: {
      PROFILER_ZONE("conv33");
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum0 = (row_in[0] + row_in[32]) * scaled_kernel[0];
          float sum1 = (row_in[1] + row_in[31]) * scaled_kernel[1];
          float sum2 = (row_in[2] + row_in[30]) * scaled_kernel[2];
          float sum3 = (row_in[3] + row_in[29]) * scaled_kernel[3];
          sum0 += (row_in[4] + row_in[28]) * scaled_kernel[4];
          sum1 += (row_in[5] + row_in[27]) * scaled_kernel[5];
          sum2 += (row_in[6] + row_in[26]) * scaled_kernel[6];
          sum3 += (row_in[7] + row_in[25]) * scaled_kernel[7];
          sum0 += (row_in[8] + row_in[24]) * scaled_kernel[8];
          sum1 += (row_in[9] + row_in[23]) * scaled_kernel[9];
          sum2 += (row_in[10] + row_in[22]) * scaled_kernel[10];
          sum3 += (row_in[11] + row_in[21]) * scaled_kernel[11];
          sum0 += (row_in[12] + row_in[20]) * scaled_kernel[12];
          sum1 += (row_in[13] + row_in[19]) * scaled_kernel[13];
          sum2 += (row_in[14] + row_in[18]) * scaled_kernel[14];
          sum3 += (row_in[15] + row_in[17]) * scaled_kernel[15];
          const float sum = (row_in[16]) * scaled_kernel[16];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum + sum0 + sum1 + sum2 + sum3;
        }
      }
      break;
    }
    case 37: {
      PROFILER_ZONE("conv37");
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum0 = (row_in[0] + row_in[36]) * scaled_kernel[0];
          float sum1 = (row_in[1] + row_in[35]) * scaled_kernel[1];
          float sum2 = (row_in[2] + row_in[34]) * scaled_kernel[2];
          float sum3 = (row_in[3] + row_in[33]) * scaled_kernel[3];
          sum0 += (row_in[4] + row_in[32]) * scaled_kernel[4];
          sum0 += (row_in[5] + row_in[31]) * scaled_kernel[5];
          sum0 += (row_in[6] + row_in[30]) * scaled_kernel[6];
          sum0 += (row_in[7] + row_in[29]) * scaled_kernel[7];
          sum0 += (row_in[8] + row_in[28]) * scaled_kernel[8];
          sum1 += (row_in[9] + row_in[27]) * scaled_kernel[9];
          sum2 += (row_in[10] + row_in[26]) * scaled_kernel[10];
          sum3 += (row_in[11] + row_in[25]) * scaled_kernel[11];
          sum0 += (row_in[12] + row_in[24]) * scaled_kernel[12];
          sum1 += (row_in[13] + row_in[23]) * scaled_kernel[13];
          sum2 += (row_in[14] + row_in[22]) * scaled_kernel[14];
          sum3 += (row_in[15] + row_in[21]) * scaled_kernel[15];
          sum0 += (row_in[16] + row_in[20]) * scaled_kernel[16];
          sum1 += (row_in[17] + row_in[19]) * scaled_kernel[17];
          const float sum = (row_in[18]) * scaled_kernel[18];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum + sum0 + sum1 + sum2 + sum3;
        }
      }
      break;
    }
    default:
      printf("Warning: Unexpected kernel size! %zu\n", len);
#else
    default:
#endif
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y);
        for (size_t x = border1; x < border2; ++x) {
          const int d = x - offset;
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          float sum = 0.0f;
          size_t j;
          for (j = 0; j <= len / 2; ++j) {
            sum += row_in[d + j] * scaled_kernel[j];
          }
          for (; j < len; ++j) {
            sum += row_in[d + j] * scaled_kernel[len - 1 - j];
          }
          row_out[y] = sum;
        }
      }
  }
  // left border
  for (size_t x = 0; x < border1; ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out->Row(x));
  }

  // right border
  for (size_t x = border2; x < in.xsize(); ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out->Row(x));
  }
}

// A blur somewhat similar to a 2D Gaussian blur.
// See: https://en.wikipedia.org/wiki/Gaussian_blur
void Blur(const ImageF& in, float sigma, float border_ratio,
          ImageF* HWY_RESTRICT temp, ImageF* HWY_RESTRICT out) {
  std::vector<float> kernel = ComputeKernel(sigma);
  if (kernel.size() == 5) {
    float sum_weights = 0.0f;
    for (const float w : kernel) {
      sum_weights += w;
    }
    const float scale = 1.0f / sum_weights;
    const float w0 = kernel[2] * scale;
    const float w1 = kernel[1] * scale;
    const float w2 = kernel[0] * scale;
    const WeightsSeparable5 weights = {
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2)},
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2)},
    };
    auto conv = ChooseSeparable5();
    conv(in, Rect(in), weights, /*pool=*/nullptr, out);
    return;
  }

  ConvolutionWithTranspose(in, kernel, border_ratio, temp);
  ConvolutionWithTranspose(*temp, kernel, border_ratio, out);
}

// Allows PaddedMaltaUnit to call either function via overloading.
struct MaltaTagLF {};
struct MaltaTag {};

// Purpose of kInternalGoodQualityThreshold:
// Normalize 'ok' image degradation to 1.0 across different versions of
// butteraugli.
static const double kInternalGoodQualityThreshold = 42.5;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

// For Mask()
static std::array<double, 512> MakeMask(double extmul, double extoff,
                                        double mul, double offset,
                                        double scaler) {
  std::array<double, 512> lut;
  for (size_t i = 0; i < lut.size(); ++i) {
    const double c = mul / ((0.01 * scaler * i) + offset);
    lut[i] = kGlobalScale * (1.0 + extmul * (c + extoff));
    if (lut[i] < 1e-5) {
      lut[i] = 1e-5;
    }
    JXL_DASSERT(lut[i] >= 0.0);
    lut[i] *= lut[i];
  }
  return lut;
}

// Clamping linear interpolator.
inline double InterpolateClampNegative(const double* array, int size,
                                       double ix) {
  if (ix < 0) {
    ix = 0;
  }
  int baseix = static_cast<int>(ix);
  double res;
  if (baseix >= size - 1) {
    res = array[size - 1];
  } else if (baseix < 0) {
    res = array[0];
  } else {
    double mix = ix - baseix;
    int nextix = baseix + 1;
    res = array[baseix] + mix * (array[nextix] - array[baseix]);
  }
  return res;
}

}  // namespace jxl

#endif  // JXL_BUTTERAUGLI_ONCE

// SIMD code
#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

template <class D, class V>
HWY_INLINE V MaximumClamp(D d, V v, double kMaxVal) {
  static const double kMul = 0.934914340314;
  const V mul = Set(d, kMul);
  const V maxval = Set(d, kMaxVal);
  // If greater than maxval or less than -maxval, replace with if_*.
  const V if_pos = MulAdd(v - maxval, mul, maxval);
  const V if_neg = MulSub(v + maxval, mul, maxval);
  const V pos_or_v = IfThenElse(v >= maxval, if_pos, v);
  return IfThenElse(v < Neg(maxval), if_neg, pos_or_v);
}

// Make area around zero less important (remove it).
template <class D, class V>
HWY_INLINE V RemoveRangeAroundZero(const D d, const double kw, const V x) {
  const auto w = Set(d, kw);
  return IfThenElse(x > w, x - w, IfThenElseZero(x < Neg(w), x + w));
}

// Make area around zero more important (2x it until the limit).
template <class D, class V>
HWY_INLINE V AmplifyRangeAroundZero(const D d, const double kw, const V x) {
  const auto w = Set(d, kw);
  return IfThenElse(x > w, x + w, IfThenElse(x < Neg(w), x - w, x + x));
}

// XybLowFreqToVals converts from low-frequency XYB space to the 'vals' space.
// Vals space can be converted to L2-norm space (Euclidean and normalized)
// through visual masking.
template <class D, class V>
HWY_INLINE void XybLowFreqToVals(const D d, const V& x, const V& y,
                                 const V& b_arg, V* HWY_RESTRICT valx,
                                 V* HWY_RESTRICT valy, V* HWY_RESTRICT valb) {
  static const double xmuli = 16.728334267161084;
  static const double ymuli = 28.850249498135561;
  static const double bmuli = 44.392179956162011;
  static const double y_to_b_muli = -0.43716072660089195;
  const V xmul = Set(d, xmuli);
  const V ymul = Set(d, ymuli);
  const V bmul = Set(d, bmuli);
  const V y_to_b_mul = Set(d, y_to_b_muli);
  const V b = MulAdd(y_to_b_mul, y, b_arg);
  *valb = b * bmul;
  *valx = x * xmul;
  *valy = y * ymul;
}

void SuppressXByY(const ImageF& in_x, const ImageF& in_y, const double yw,
                  ImageF* HWY_RESTRICT out) {
  JXL_DASSERT(SameSize(in_x, in_y) && SameSize(in_x, *out));
  const size_t xsize = in_x.xsize();
  const size_t ysize = in_x.ysize();

  const HWY_FULL(float) d;
  static const double s = 0.941388349694;
  const auto sv = Set(d, s);
  const auto one_minus_s = Set(d, 1.0 - s);
  const auto ywv = Set(d, yw);

  for (size_t y = 0; y < ysize; ++y) {
    const float* HWY_RESTRICT row_x = in_x.ConstRow(y);
    const float* HWY_RESTRICT row_y = in_y.ConstRow(y);
    float* HWY_RESTRICT row_out = out->Row(y);

    for (size_t x = 0; x < xsize; x += Lanes(d)) {
      const auto vx = Load(d, row_x + x);
      const auto vy = Load(d, row_y + x);
      const auto scaler = MulAdd(ywv / MulAdd(vy, vy, ywv), one_minus_s, sv);
      Store(scaler * vx, d, row_out + x);
    }
  }
}

static void SeparateFrequencies(size_t xsize, size_t ysize, const Image3F& xyb,
                                PsychoImage& ps) {
  PROFILER_FUNC;
  const HWY_FULL(float) d;

  ImageF blur_temp(ysize, xsize);  // transposed

  // Extract lf ...
  static const double kSigmaLf = 7.15593339443;
  static const double kSigmaHf = 3.22489901262;
  static const double kSigmaUhf = 1.56416327805;
  // Border handling is complicated.
  static const double border_lf = 0.0;
  static const double border_mf = 0.0;
  static const double border_hf = 0.0;
  ps.mf = Image3F(xsize, ysize);
  ps.hf[0] = ImageF(xsize, ysize);
  ps.hf[1] = ImageF(xsize, ysize);
  ps.lf = Image3F(xyb.xsize(), xyb.ysize());
  ps.mf = Image3F(xyb.xsize(), xyb.ysize());
  for (int i = 0; i < 3; ++i) {
    Blur(xyb.Plane(i), kSigmaLf, border_lf, &blur_temp,
         const_cast<ImageF*>(&ps.lf.Plane(i)));

    // ... and keep everything else in mf.
    for (size_t y = 0; y < ysize; ++y) {
      const float* BUTTERAUGLI_RESTRICT row_xyb = xyb.PlaneRow(i, y);
      const float* BUTTERAUGLI_RESTRICT row_lf = ps.lf.ConstPlaneRow(i, y);
      float* BUTTERAUGLI_RESTRICT row_mf = ps.mf.PlaneRow(i, y);
      for (size_t x = 0; x < xsize; x += Lanes(d)) {
        const auto mf = Load(d, row_xyb + x) - Load(d, row_lf + x);
        Store(mf, d, row_mf + x);
      }
    }
    if (i == 2) {
      Blur(ps.mf.Plane(i), kSigmaHf, border_mf, &blur_temp,
           const_cast<ImageF*>(&ps.mf.Plane(i)));
      break;
    }
    // Divide mf into mf and hf.
    for (size_t y = 0; y < ysize; ++y) {
      float* BUTTERAUGLI_RESTRICT row_mf = ps.mf.PlaneRow(i, y);
      float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[i].Row(y);
      for (size_t x = 0; x < xsize; x += Lanes(d)) {
        Store(Load(d, row_mf + x), d, row_hf + x);
      }
    }
    Blur(ps.mf.Plane(i), kSigmaHf, border_mf, &blur_temp,
         const_cast<ImageF*>(&ps.mf.Plane(i)));
    static const double kRemoveMfRange = 0.3;
    static const double kAddMfRange = 0.1;
    if (i == 0) {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT row_mf = ps.mf.PlaneRow(0, y);
        float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[0].Row(y);
        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          auto mf = Load(d, row_mf + x);
          auto hf = Load(d, row_hf + x) - mf;
          mf = RemoveRangeAroundZero(d, kRemoveMfRange, mf);
          Store(mf, d, row_mf + x);
          Store(hf, d, row_hf + x);
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT row_mf = ps.mf.PlaneRow(1, y);
        float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[1].Row(y);
        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          auto mf = Load(d, row_mf + x);
          auto hf = Load(d, row_hf + x) - mf;

          mf = AmplifyRangeAroundZero(d, kAddMfRange, mf);
          Store(mf, d, row_mf + x);
          Store(hf, d, row_hf + x);
        }
      }
    }
  }

  // Temporarily used as output of SuppressXByY
  ps.uhf[0] = ImageF(xsize, ysize);
  ps.uhf[1] = ImageF(xsize, ysize);

  // Suppress red-green by intensity change in the high freq channels.
  static const double suppress = 286.09942757;
  SuppressXByY(ps.hf[0], ps.hf[1], suppress, &ps.uhf[0]);
  // hf is the SuppressXByY output, uhf will be written below.
  ps.hf[0].Swap(ps.uhf[0]);

  for (int i = 0; i < 2; ++i) {
    // Divide hf into hf and uhf.
    for (size_t y = 0; y < ysize; ++y) {
      float* BUTTERAUGLI_RESTRICT row_uhf = ps.uhf[i].Row(y);
      float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[i].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_uhf[x] = row_hf[x];
      }
    }
    Blur(ps.hf[i], kSigmaUhf, border_hf, &blur_temp, &ps.hf[i]);
    static const double kRemoveHfRange = 0.12;
    static const double kAddHfRange = 0.03;
    static const double kRemoveUhfRange = 0.08;
    static const double kAddUhfRange = 0.02;
    static const double kMaxclampHf = 78.7416747972;
    static const double kMaxclampUhf = 4.62878535439;
    static double kMulYHf = 1.16155986803;
    static double kMulYUhf = 2.32552960949;
    if (i == 0) {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT row_uhf = ps.uhf[0].Row(y);
        float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[0].Row(y);
        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          auto hf = Load(d, row_hf + x);
          auto uhf = Load(d, row_uhf + x) - hf;
          hf = RemoveRangeAroundZero(d, kRemoveHfRange, hf);
          uhf = RemoveRangeAroundZero(d, kRemoveUhfRange, uhf);
          Store(hf, d, row_hf + x);
          Store(uhf, d, row_uhf + x);
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT row_uhf = ps.uhf[1].Row(y);
        float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[1].Row(y);
        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          auto hf = Load(d, row_hf + x);
          hf = MaximumClamp(d, hf, kMaxclampHf);

          auto uhf = Load(d, row_uhf + x) - hf;
          uhf = MaximumClamp(d, uhf, kMaxclampUhf);
          uhf *= Set(d, kMulYUhf);
          uhf = AmplifyRangeAroundZero(d, kAddUhfRange, uhf);
          Store(uhf, d, row_uhf + x);

          hf *= Set(d, kMulYHf);
          hf = AmplifyRangeAroundZero(d, kAddHfRange, hf);
          Store(hf, d, row_hf + x);
        }
      }
    }
  }
  // Modify range around zero code only concerns the high frequency
  // planes and only the X and Y channels.
  // Convert low freq xyb to vals space so that we can do a simple squared sum
  // diff on the low frequencies later.
  for (size_t y = 0; y < ysize; ++y) {
    float* BUTTERAUGLI_RESTRICT row_x = ps.lf.PlaneRow(0, y);
    float* BUTTERAUGLI_RESTRICT row_y = ps.lf.PlaneRow(1, y);
    float* BUTTERAUGLI_RESTRICT row_b = ps.lf.PlaneRow(2, y);
    for (size_t x = 0; x < xsize; x += Lanes(d)) {
      auto valx = Undefined(d);
      auto valy = Undefined(d);
      auto valb = Undefined(d);
      XybLowFreqToVals(d, Load(d, row_x + x), Load(d, row_y + x),
                       Load(d, row_b + x), &valx, &valy, &valb);
      Store(valx, d, row_x + x);
      Store(valy, d, row_y + x);
      Store(valb, d, row_b + x);
    }
  }
}

template <class D>
Vec<D> MaltaUnit(MaltaTagLF /*tag*/, const D df,
                 const float* BUTTERAUGLI_RESTRICT d, const intptr_t xs) {
  const intptr_t xs3 = 3 * xs;

  const auto center = LoadU(df, d);

  // x grows, y constant
  const auto sum_yconst = LoadU(df, d - 4) + LoadU(df, d - 2) + center +
                          LoadU(df, d + 2) + LoadU(df, d + 4);
  // Will return this, sum of all line kernels
  auto retval = sum_yconst * sum_yconst;
  {
    // y grows, x constant
    auto sum = LoadU(df, d - xs3 - xs) + LoadU(df, d - xs - xs) + center +
               LoadU(df, d + xs + xs) + LoadU(df, d + xs3 + xs);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // both grow
    auto sum = LoadU(df, d - xs3 - 3) + LoadU(df, d - xs - xs - 2) + center +
               LoadU(df, d + xs + xs + 2) + LoadU(df, d + xs3 + 3);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // y grows, x shrinks
    auto sum = LoadU(df, d - xs3 + 3) + LoadU(df, d - xs - xs + 2) + center +
               LoadU(df, d + xs + xs - 2) + LoadU(df, d + xs3 - 3);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    auto sum = LoadU(df, d - xs3 - xs + 1) + LoadU(df, d - xs - xs + 1) +
               center + LoadU(df, d + xs + xs - 1) +
               LoadU(df, d + xs3 + xs - 1);
    retval = MulAdd(sum, sum, retval);
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    auto sum = LoadU(df, d - xs3 - xs - 1) + LoadU(df, d - xs - xs - 1) +
               center + LoadU(df, d + xs + xs + 1) +
               LoadU(df, d + xs3 + xs + 1);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    auto sum = LoadU(df, d - 4 - xs) + LoadU(df, d - 2 - xs) + center +
               LoadU(df, d + 2 + xs) + LoadU(df, d + 4 + xs);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    auto sum = LoadU(df, d - 4 + xs) + LoadU(df, d - 2 + xs) + center +
               LoadU(df, d + 2 - xs) + LoadU(df, d + 4 - xs);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7______*__
       8_________ */
    auto sum = LoadU(df, d - xs3 - 2) + LoadU(df, d - xs - xs - 1) + center +
               LoadU(df, d + xs + xs + 1) + LoadU(df, d + xs3 + 2);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7__*______
       8_________ */
    auto sum = LoadU(df, d - xs3 + 2) + LoadU(df, d - xs - xs + 1) + center +
               LoadU(df, d + xs + xs - 1) + LoadU(df, d + xs3 - 2);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__*______
       4____0____
       5______*__
       6_______*_
       7_________
       8_________ */
    auto sum = LoadU(df, d - xs - xs - 3) + LoadU(df, d - xs - 2) + center +
               LoadU(df, d + xs + 2) + LoadU(df, d + xs + xs + 3);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_______*_
       3______*__
       4____0____
       5__*______
       6_*_______
       7_________
       8_________ */
    auto sum = LoadU(df, d - xs - xs + 3) + LoadU(df, d - xs + 2) + center +
               LoadU(df, d + xs - 2) + LoadU(df, d + xs + xs - 3);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2________*
       3______*__
       4____0____
       5__*______
       6*________
       7_________
       8_________ */

    auto sum = LoadU(df, d + xs + xs - 4) + LoadU(df, d + xs - 2) + center +
               LoadU(df, d - xs + 2) + LoadU(df, d - xs - xs + 4);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2*________
       3__*______
       4____0____
       5______*__
       6________*
       7_________
       8_________ */
    auto sum = LoadU(df, d - xs - xs - 4) + LoadU(df, d - xs - 2) + center +
               LoadU(df, d + xs + 2) + LoadU(df, d + xs + xs + 4);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0__*______
       1_________
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7_________
       8______*__ */
    auto sum = LoadU(df, d - xs3 - xs - 2) + LoadU(df, d - xs - xs - 1) +
               center + LoadU(df, d + xs + xs + 1) +
               LoadU(df, d + xs3 + xs + 2);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0______*__
       1_________
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7_________
       8__*______ */
    auto sum = LoadU(df, d - xs3 - xs + 2) + LoadU(df, d - xs - xs + 1) +
               center + LoadU(df, d + xs + xs - 1) +
               LoadU(df, d + xs3 + xs - 2);
    retval = MulAdd(sum, sum, retval);
  }
  return retval;
}

template <class D>
Vec<D> MaltaUnit(MaltaTag /*tag*/, const D df,
                 const float* BUTTERAUGLI_RESTRICT d, const intptr_t xs) {
  const intptr_t xs3 = 3 * xs;

  const auto center = LoadU(df, d);

  // x grows, y constant
  const auto sum_yconst = LoadU(df, d - 4) + LoadU(df, d - 3) +
                          LoadU(df, d - 2) + LoadU(df, d - 1) + center +
                          LoadU(df, d + 1) + LoadU(df, d + 2) +
                          LoadU(df, d + 3) + LoadU(df, d + 4);
  // Will return this, sum of all line kernels
  auto retval = sum_yconst * sum_yconst;

  {
    // y grows, x constant
    auto sum = LoadU(df, d - xs3 - xs) + LoadU(df, d - xs3) +
               LoadU(df, d - xs - xs) + LoadU(df, d - xs) + center +
               LoadU(df, d + xs) + LoadU(df, d + xs + xs) + LoadU(df, d + xs3) +
               LoadU(df, d + xs3 + xs);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // both grow
    auto sum = LoadU(df, d - xs3 - 3) + LoadU(df, d - xs - xs - 2) +
               LoadU(df, d - xs - 1) + center + LoadU(df, d + xs + 1) +
               LoadU(df, d + xs + xs + 2) + LoadU(df, d + xs3 + 3);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // y grows, x shrinks
    auto sum = LoadU(df, d - xs3 + 3) + LoadU(df, d - xs - xs + 2) +
               LoadU(df, d - xs + 1) + center + LoadU(df, d + xs - 1) +
               LoadU(df, d + xs + xs - 2) + LoadU(df, d + xs3 - 3);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    auto sum = LoadU(df, d - xs3 - xs + 1) + LoadU(df, d - xs3 + 1) +
               LoadU(df, d - xs - xs + 1) + LoadU(df, d - xs) + center +
               LoadU(df, d + xs) + LoadU(df, d + xs + xs - 1) +
               LoadU(df, d + xs3 - 1) + LoadU(df, d + xs3 + xs - 1);
    retval = MulAdd(sum, sum, retval);
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    auto sum = LoadU(df, d - xs3 - xs - 1) + LoadU(df, d - xs3 - 1) +
               LoadU(df, d - xs - xs - 1) + LoadU(df, d - xs) + center +
               LoadU(df, d + xs) + LoadU(df, d + xs + xs + 1) +
               LoadU(df, d + xs3 + 1) + LoadU(df, d + xs3 + xs + 1);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    auto sum = LoadU(df, d - 4 - xs) + LoadU(df, d - 3 - xs) +
               LoadU(df, d - 2 - xs) + LoadU(df, d - 1) + center +
               LoadU(df, d + 1) + LoadU(df, d + 2 + xs) +
               LoadU(df, d + 3 + xs) + LoadU(df, d + 4 + xs);
    retval = MulAdd(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    auto sum = LoadU(df, d - 4 + xs) + LoadU(df, d - 3 + xs) +
               LoadU(df, d - 2 + xs) + LoadU(df, d - 1) + center +
               LoadU(df, d + 1) + LoadU(df, d + 2 - xs) +
               LoadU(df, d + 3 - xs) + LoadU(df, d + 4 - xs);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3___*_____
       4____0____
       5_____*___
       6_____*___
       7______*__
       8_________ */
    auto sum = LoadU(df, d - xs3 - 2) + LoadU(df, d - xs - xs - 1) +
               LoadU(df, d - xs - 1) + center + LoadU(df, d + xs + 1) +
               LoadU(df, d + xs + xs + 1) + LoadU(df, d + xs3 + 2);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_____*___
       4____0____
       5___*_____
       6___*_____
       7__*______
       8_________ */
    auto sum = LoadU(df, d - xs3 + 2) + LoadU(df, d - xs - xs + 1) +
               LoadU(df, d - xs + 1) + center + LoadU(df, d + xs - 1) +
               LoadU(df, d + xs + xs - 1) + LoadU(df, d + xs3 - 2);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__**_____
       4____0____
       5_____**__
       6_______*_
       7_________
       8_________ */
    auto sum = LoadU(df, d - xs - xs - 3) + LoadU(df, d - xs - 2) +
               LoadU(df, d - xs - 1) + center + LoadU(df, d + xs + 1) +
               LoadU(df, d + xs + 2) + LoadU(df, d + xs + xs + 3);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_______*_
       3_____**__
       4____0____
       5__**_____
       6_*_______
       7_________
       8_________ */
    auto sum = LoadU(df, d - xs - xs + 3) + LoadU(df, d - xs + 2) +
               LoadU(df, d - xs + 1) + center + LoadU(df, d + xs - 1) +
               LoadU(df, d + xs - 2) + LoadU(df, d + xs + xs - 3);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_________
       3______***
       4___*0*___
       5***______
       6_________
       7_________
       8_________ */

    auto sum = LoadU(df, d + xs - 4) + LoadU(df, d + xs - 3) +
               LoadU(df, d + xs - 2) + LoadU(df, d - 1) + center +
               LoadU(df, d + 1) + LoadU(df, d - xs + 2) +
               LoadU(df, d - xs + 3) + LoadU(df, d - xs + 4);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_________
       3***______
       4___*0*___
       5______***
       6_________
       7_________
       8_________ */
    auto sum = LoadU(df, d - xs - 4) + LoadU(df, d - xs - 3) +
               LoadU(df, d - xs - 2) + LoadU(df, d - 1) + center +
               LoadU(df, d + 1) + LoadU(df, d + xs + 2) +
               LoadU(df, d + xs + 3) + LoadU(df, d + xs + 4);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0___*_____
       1___*_____
       2___*_____
       3____*____
       4____0____
       5____*____
       6_____*___
       7_____*___
       8_____*___ */
    auto sum = LoadU(df, d - xs3 - xs - 1) + LoadU(df, d - xs3 - 1) +
               LoadU(df, d - xs - xs - 1) + LoadU(df, d - xs) + center +
               LoadU(df, d + xs) + LoadU(df, d + xs + xs + 1) +
               LoadU(df, d + xs3 + 1) + LoadU(df, d + xs3 + xs + 1);
    retval = MulAdd(sum, sum, retval);
  }
  {
    /* 0_____*___
       1_____*___
       2____ *___
       3____*____
       4____0____
       5____*____
       6___*_____
       7___*_____
       8___*_____ */
    auto sum = LoadU(df, d - xs3 - xs + 1) + LoadU(df, d - xs3 + 1) +
               LoadU(df, d - xs - xs + 1) + LoadU(df, d - xs) + center +
               LoadU(df, d + xs) + LoadU(df, d + xs + xs - 1) +
               LoadU(df, d + xs3 - 1) + LoadU(df, d + xs3 + xs - 1);
    retval = MulAdd(sum, sum, retval);
  }
  return retval;
}

// Returns MaltaUnit. Avoids bounds-checks when x0 and y0 are known
// to be far enough from the image borders. "diffs" is a packed image.
template <class Tag>
static BUTTERAUGLI_INLINE float PaddedMaltaUnit(const ImageF& diffs,
                                                const size_t x0,
                                                const size_t y0) {
  const float* BUTTERAUGLI_RESTRICT d = diffs.ConstRow(y0) + x0;
  const HWY_CAPPED(float, 1) df;
  if ((x0 >= 4 && y0 >= 4 && x0 < (diffs.xsize() - 4) &&
       y0 < (diffs.ysize() - 4))) {
    return GetLane(MaltaUnit(Tag(), df, d, diffs.PixelsPerRow()));
  }

  float borderimage[12 * 9];  // round up to 4
  for (int dy = 0; dy < 9; ++dy) {
    int y = y0 + dy - 4;
    if (y < 0 || static_cast<size_t>(y) >= diffs.ysize()) {
      for (int dx = 0; dx < 12; ++dx) {
        borderimage[dy * 12 + dx] = 0.0f;
      }
      continue;
    }

    const float* row_diffs = diffs.ConstRow(y);
    for (int dx = 0; dx < 9; ++dx) {
      int x = x0 + dx - 4;
      if (x < 0 || static_cast<size_t>(x) >= diffs.xsize()) {
        borderimage[dy * 12 + dx] = 0.0f;
      } else {
        borderimage[dy * 12 + dx] = row_diffs[x];
      }
    }
    std::fill(borderimage + dy * 12 + 9, borderimage + dy * 12 + 12, 0.0f);
  }
  return GetLane(MaltaUnit(Tag(), df, &borderimage[4 * 12 + 4], 12));
}

template <class Tag>
static void MaltaDiffMapT(const Tag tag, const ImageF& lum0, const ImageF& lum1,
                          const double w_0gt1, const double w_0lt1,
                          const double norm1, const double len,
                          const double mulli, ImageF* HWY_RESTRICT diffs,
                          Image3F* HWY_RESTRICT block_diff_ac, size_t c) {
  JXL_DASSERT(SameSize(lum0, lum1) && SameSize(lum0, *diffs));
  const size_t xsize_ = lum0.xsize();
  const size_t ysize_ = lum0.ysize();

  const float kWeight0 = 0.5;
  const float kWeight1 = 0.33;

  const double w_pre0gt1 = mulli * std::sqrt(kWeight0 * w_0gt1) / (len * 2 + 1);
  const double w_pre0lt1 = mulli * std::sqrt(kWeight1 * w_0lt1) / (len * 2 + 1);
  const float norm2_0gt1 = w_pre0gt1 * norm1;
  const float norm2_0lt1 = w_pre0lt1 * norm1;

  for (size_t y = 0; y < ysize_; ++y) {
    const float* HWY_RESTRICT row0 = lum0.ConstRow(y);
    const float* HWY_RESTRICT row1 = lum1.ConstRow(y);
    float* HWY_RESTRICT row_diffs = diffs->Row(y);
    for (size_t x = 0; x < xsize_; ++x) {
      const float absval = 0.5f * (std::abs(row0[x]) + std::abs(row1[x]));
      const float diff = row0[x] - row1[x];
      const float scaler = norm2_0gt1 / (static_cast<float>(norm1) + absval);

      // Primary symmetric quadratic objective.
      row_diffs[x] = scaler * diff;

      const float scaler2 = norm2_0lt1 / (static_cast<float>(norm1) + absval);
      const double fabs0 = std::fabs(row0[x]);

      // Secondary half-open quadratic objectives.
      const double too_small = 0.55 * fabs0;
      const double too_big = 1.05 * fabs0;

      if (row0[x] < 0) {
        if (row1[x] > -too_small) {
          double impact = scaler2 * (row1[x] + too_small);
          if (diff < 0) {
            row_diffs[x] -= impact;
          } else {
            row_diffs[x] += impact;
          }
        } else if (row1[x] < -too_big) {
          double impact = scaler2 * (-row1[x] - too_big);
          if (diff < 0) {
            row_diffs[x] -= impact;
          } else {
            row_diffs[x] += impact;
          }
        }
      } else {
        if (row1[x] < too_small) {
          double impact = scaler2 * (too_small - row1[x]);
          if (diff < 0) {
            row_diffs[x] -= impact;
          } else {
            row_diffs[x] += impact;
          }
        } else if (row1[x] > too_big) {
          double impact = scaler2 * (row1[x] - too_big);
          if (diff < 0) {
            row_diffs[x] -= impact;
          } else {
            row_diffs[x] += impact;
          }
        }
      }
    }
  }

  size_t y0 = 0;
  // Top
  for (; y0 < 4; ++y0) {
    float* BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->PlaneRow(c, y0);
    for (size_t x0 = 0; x0 < xsize_; ++x0) {
      row_diff[x0] += PaddedMaltaUnit<Tag>(*diffs, x0, y0);
    }
  }

  const HWY_FULL(float) df;
  const size_t aligned_x = std::max(size_t(4), Lanes(df));
  const intptr_t stride = diffs->PixelsPerRow();

  // Middle
  for (; y0 < ysize_ - 4; ++y0) {
    const float* BUTTERAUGLI_RESTRICT row_in = diffs->ConstRow(y0);
    float* BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->PlaneRow(c, y0);
    size_t x0 = 0;
    for (; x0 < aligned_x; ++x0) {
      row_diff[x0] += PaddedMaltaUnit<Tag>(*diffs, x0, y0);
    }
    for (; x0 + Lanes(df) + 4 <= xsize_; x0 += Lanes(df)) {
      auto diff = Load(df, row_diff + x0);
      diff += MaltaUnit(Tag(), df, row_in + x0, stride);
      Store(diff, df, row_diff + x0);
    }

    for (; x0 < xsize_; ++x0) {
      row_diff[x0] += PaddedMaltaUnit<Tag>(*diffs, x0, y0);
    }
  }

  // Bottom
  for (; y0 < ysize_; ++y0) {
    float* BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->PlaneRow(c, y0);
    for (size_t x0 = 0; x0 < xsize_; ++x0) {
      row_diff[x0] += PaddedMaltaUnit<Tag>(*diffs, x0, y0);
    }
  }
}

// Need non-template wrapper functions for HWY_EXPORT.
void MaltaDiffMap(const ImageF& lum0, const ImageF& lum1, const double w_0gt1,
                  const double w_0lt1, const double norm1, const double len,
                  const double mulli, ImageF* HWY_RESTRICT diffs,
                  Image3F* HWY_RESTRICT block_diff_ac, size_t c) {
  MaltaDiffMapT(MaltaTag(), lum0, lum1, w_0gt1, w_0lt1, norm1, len, mulli,
                diffs, block_diff_ac, c);
}

void MaltaDiffMapLF(const ImageF& lum0, const ImageF& lum1, const double w_0gt1,
                    const double w_0lt1, const double norm1, const double len,
                    const double mulli, ImageF* HWY_RESTRICT diffs,
                    Image3F* HWY_RESTRICT block_diff_ac, size_t c) {
  MaltaDiffMapT(MaltaTagLF(), lum0, lum1, w_0gt1, w_0lt1, norm1, len, mulli,
                diffs, block_diff_ac, c);
}

double MaskX(double delta) {
  static const double extmul = 1.9741728897212369;
  static const double extoff = 2.0124797155052101;
  static const double offset = 0.26118960206891428;
  static const double scaler = 386.58249095811573;
  static const double mul = 4.0096602151572194;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskY(double delta) {
  static const double extmul = 3.2760071347758579;
  static const double extoff = -3.3377693215128232;
  static const double offset = 1.9592295279427681;
  static const double scaler = 1.4422616544932898;
  static const double mul = 9.6463717073130066;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcX(double delta) {
  static const double extmul = 7.9213692914149458;
  static const double extoff = 2.5312740226978478;
  static const double offset = 0.1060403567938254;
  static const double scaler = 43.546844862805052;
  static const double mul = 0.32110246816137022;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcY(double delta) {
  static const double extmul = 0.019875068816088557;
  static const double extoff = 0.1;
  static const double offset = 0.1;
  static const double scaler = 7.0;
  static const double mul = 9.8744858184185791;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

// x0 is aligned, x1 and x2 possibly not.
template <class D>
void DiffPrecomputeX_T(const D d, const float* BUTTERAUGLI_RESTRICT row0_in0,
                       const float* BUTTERAUGLI_RESTRICT row1_in0,
                       const float* BUTTERAUGLI_RESTRICT row0_in1,
                       const float* BUTTERAUGLI_RESTRICT row1_in1,
                       const float* BUTTERAUGLI_RESTRICT row0_in2,
                       const float* BUTTERAUGLI_RESTRICT row1_in2,
                       const size_t x0, const size_t x1, const size_t x2,
                       const float mul, const float cutoff,
                       float* BUTTERAUGLI_RESTRICT row_out) {
  const auto r0_i0_x0 = Load(d, row0_in0 + x0);
  const auto r0_i0_x1 = LoadU(d, row0_in0 + x1);
  const auto r0_i0_x2 = LoadU(d, row0_in0 + x2);

  const auto r0_i1_x0 = Load(d, row0_in1 + x0);
  const auto r0_i2_x0 = Load(d, row0_in2 + x0);

  const auto r1_i0_x0 = Load(d, row1_in0 + x0);
  const auto r1_i0_x1 = LoadU(d, row1_in0 + x1);
  const auto r1_i0_x2 = LoadU(d, row1_in0 + x2);

  const auto r1_i1_x0 = Load(d, row1_in1 + x0);
  const auto r1_i2_x0 = Load(d, row1_in2 + x0);

  const auto k3 = Set(d, 3.0f);
  const auto half = Set(d, 0.5f);
  const auto vmul = Set(d, mul);
  const auto vcutoff = Set(d, cutoff);

  const auto sup0 =
      AbsDiff(r0_i0_x0, r0_i0_x2) + AbsDiff(r0_i0_x0, r0_i2_x0) +
      AbsDiff(r0_i0_x0, r0_i0_x1) + AbsDiff(r0_i0_x0, r0_i1_x0) +
      k3 * (AbsDiff(r0_i2_x0, r0_i1_x0) + AbsDiff(r0_i0_x1, r0_i0_x2));
  const auto sup1 =
      AbsDiff(r1_i0_x0, r1_i0_x2) + AbsDiff(r1_i0_x0, r1_i2_x0) +
      AbsDiff(r1_i0_x0, r1_i0_x1) + AbsDiff(r1_i0_x0, r1_i1_x0) +
      k3 * (AbsDiff(r1_i2_x0, r1_i1_x0) + AbsDiff(r1_i0_x1, r1_i0_x2));

  auto out = vmul * Min(sup0, sup1);
  out = Min(out, vcutoff);

  const auto limit2 = vcutoff * half;
  out = IfThenElse(out >= limit2, (out + limit2) * half, out);

  const auto limit4 = limit2 * half;
  out = IfThenElse(out >= limit4, (out + limit4) * half, out);

  Store(out, d, row_out + x0);
}

void DiffPrecomputeX(const ImageF& xyb0, const ImageF& xyb1, float mul,
                     float cutoff, ImageF* BUTTERAUGLI_RESTRICT result) {
  PROFILER_FUNC;

  JXL_CHECK(SameSize(xyb0, xyb1) && SameSize(xyb0, *result));
  const size_t xsize = xyb0.xsize();
  const size_t ysize = xyb0.ysize();

  for (size_t y = 0; y < ysize; ++y) {
    const size_t y1 = y == 0 ? (ysize >= 2 ? 1 : 0) : y - 1;
    const size_t y2 = y + 1 < ysize ? y + 1 : y == 0 ? 0 : y - 1;
    const float* BUTTERAUGLI_RESTRICT row0_in0 = xyb0.Row(y);
    const float* BUTTERAUGLI_RESTRICT row1_in0 = xyb1.Row(y);
    const float* BUTTERAUGLI_RESTRICT row0_in1 = xyb0.Row(y1);
    const float* BUTTERAUGLI_RESTRICT row1_in1 = xyb1.Row(y1);
    const float* BUTTERAUGLI_RESTRICT row0_in2 = xyb0.Row(y2);
    const float* BUTTERAUGLI_RESTRICT row1_in2 = xyb1.Row(y2);
    float* BUTTERAUGLI_RESTRICT row_out = result->Row(y);

    const HWY_CAPPED(float, 1) d1;
    const HWY_FULL(float) d;
    size_t x = 0;
    for (; x < std::min(xsize - 1, Lanes(d)); ++x) {
      const size_t x1 = x == 0 ? (xsize >= 2 ? 1 : 0) : x - 1;
      const size_t x2 = x + 1;
      DiffPrecomputeX_T(d1, row0_in0, row1_in0, row0_in1, row1_in1, row0_in2,
                        row1_in2, x, x1, x2, mul, cutoff, row_out);
    }

    for (; x + Lanes(d) + 1 <= xsize; x += Lanes(d)) {
      const size_t x1 = x - 1;
      const size_t x2 = x + 1;
      DiffPrecomputeX_T(d, row0_in0, row1_in0, row0_in1, row1_in1, row0_in2,
                        row1_in2, x, x1, x2, mul, cutoff, row_out);
    }

    for (; x < xsize; ++x) {
      const size_t x1 = x - 1;
      const size_t x2 = x + 1 < xsize ? x + 1 : x - 1;
      DiffPrecomputeX_T(d1, row0_in0, row1_in0, row0_in1, row1_in1, row0_in2,
                        row1_in2, x, x1, x2, mul, cutoff, row_out);
    }
  }
}

// x0 is aligned, x1 and x2 possibly not.
template <class D>
void DiffPrecomputeY_T(const D d, const float* BUTTERAUGLI_RESTRICT row0_in0,
                       const float* BUTTERAUGLI_RESTRICT row1_in0,
                       const float* BUTTERAUGLI_RESTRICT row0_in1,
                       const float* BUTTERAUGLI_RESTRICT row1_in1,
                       const float* BUTTERAUGLI_RESTRICT row0_in2,
                       const float* BUTTERAUGLI_RESTRICT row1_in2,
                       const size_t x0, const size_t x1, const size_t x2,
                       float mul, float mul2,
                       float* BUTTERAUGLI_RESTRICT row_out0,
                       float* BUTTERAUGLI_RESTRICT row_out1) {
  const auto r0_i0_x0 = Load(d, row0_in0 + x0);
  const auto r0_i0_x1 = LoadU(d, row0_in0 + x1);
  const auto r0_i0_x2 = LoadU(d, row0_in0 + x2);

  const auto r0_i1_x0 = Load(d, row0_in1 + x0);
  const auto r0_i2_x0 = Load(d, row0_in2 + x0);

  const auto r1_i0_x0 = Load(d, row1_in0 + x0);
  const auto r1_i0_x1 = LoadU(d, row1_in0 + x1);
  const auto r1_i0_x2 = LoadU(d, row1_in0 + x2);

  const auto r1_i1_x0 = Load(d, row1_in1 + x0);
  const auto r1_i2_x0 = Load(d, row1_in2 + x0);

  const auto k3 = Set(d, 3.0f);
  const auto kBase = Set(d, 0.69314718056f);  // 1 / log2(e)
  const auto vmul = Set(d, mul);
  const auto vmul2 = Set(d, mul2);

  // kBias makes log behave more linearly.
  const auto kBias = Set(d, 7);
  const auto log_bias = Set(d, 1.94591014906f);  // = std::log(kBias)

  const auto sup0 =
      (AbsDiff(r0_i0_x0, r0_i0_x2) + AbsDiff(r0_i0_x0, r0_i2_x0) +
       AbsDiff(r0_i0_x0, r0_i0_x1) + AbsDiff(r0_i0_x0, r0_i1_x0) +
       k3 * (AbsDiff(r0_i2_x0, r0_i1_x0) + AbsDiff(r0_i0_x1, r0_i0_x2)));
  const auto sup1 =
      (AbsDiff(r1_i0_x0, r1_i0_x2) + AbsDiff(r1_i0_x0, r1_i2_x0) +
       AbsDiff(r1_i0_x0, r1_i0_x1) + AbsDiff(r1_i0_x0, r1_i1_x0) +
       k3 * (AbsDiff(r1_i2_x0, r1_i1_x0) + AbsDiff(r1_i0_x1, r1_i0_x2)));
  const auto biased0 = MulAdd(sup0 * sup0, vmul2, kBias);
  const auto biased1 = MulAdd(sup1 * sup1, vmul2, kBias);
  const auto out0 =
      vmul * MulSub(FastLog2f_18bits(d, biased0), kBase, log_bias);
  const auto out1 =
      vmul * MulSub(FastLog2f_18bits(d, biased1), kBase, log_bias);
  Store(out0, d, row_out0 + x0);
  Store(out1, d, row_out1 + x0);
}

// Precalculates masking for y channel, giving masks for
// both images back so that they can be used for similarity comparisons
// too.
void DiffPrecomputeY(const ImageF& xyb0, const ImageF& xyb1, float mul,
                     float mul2, ImageF* out0, ImageF* out1) {
  PROFILER_FUNC;

  const size_t xsize = xyb0.xsize();
  const size_t ysize = xyb0.ysize();

  for (size_t y = 0; y < ysize; ++y) {
    const size_t y1 = y == 0 ? (ysize >= 2 ? 1 : 0) : y - 1;
    const size_t y2 = y + 1 < ysize ? y + 1 : y == 0 ? 0 : y - 1;
    const float* BUTTERAUGLI_RESTRICT row0_in0 = xyb0.Row(y);
    const float* BUTTERAUGLI_RESTRICT row1_in0 = xyb1.Row(y);
    const float* BUTTERAUGLI_RESTRICT row0_in1 = xyb0.Row(y1);
    const float* BUTTERAUGLI_RESTRICT row1_in1 = xyb1.Row(y1);
    const float* BUTTERAUGLI_RESTRICT row0_in2 = xyb0.Row(y2);
    const float* BUTTERAUGLI_RESTRICT row1_in2 = xyb1.Row(y2);
    float* BUTTERAUGLI_RESTRICT row_out0 = out0->Row(y);
    float* BUTTERAUGLI_RESTRICT row_out1 = out1->Row(y);

    const HWY_CAPPED(float, 1) d1;
    const HWY_FULL(float) d;
    size_t x = 0;
    for (; x < std::min(xsize - 1, Lanes(d)); ++x) {
      const size_t x1 = x == 0 ? (xsize >= 2 ? 1 : 0) : x - 1;
      const size_t x2 = x + 1;
      DiffPrecomputeY_T(d1, row0_in0, row1_in0, row0_in1, row1_in1, row0_in2,
                        row1_in2, x, x1, x2, mul, mul2, row_out0, row_out1);
    }

    for (; x + 1 + Lanes(d) <= xsize; x += Lanes(d)) {
      const size_t x1 = x - 1;
      const size_t x2 = x + 1;
      DiffPrecomputeY_T(d, row0_in0, row1_in0, row0_in1, row1_in1, row0_in2,
                        row1_in2, x, x1, x2, mul, mul2, row_out0, row_out1);
    }

    for (; x < xsize; ++x) {
      const size_t x1 = x - 1;
      const size_t x2 = x + 1 < xsize ? x + 1 : x - 1;
      DiffPrecomputeY_T(d1, row0_in0, row1_in0, row0_in1, row1_in1, row0_in2,
                        row1_in2, x, x1, x2, mul, mul2, row_out0, row_out1);
    }
  }
}

// Compute values of local frequency and dc masking based on the activity
// in the two images. img_diff_ac may be null.
void Mask(const Image3F& xyb0, const Image3F& xyb1,
          MaskImage* BUTTERAUGLI_RESTRICT mask,
          MaskImage* BUTTERAUGLI_RESTRICT mask_dc,
          ImageF* BUTTERAUGLI_RESTRICT img_diff_ac) {
  PROFILER_FUNC;
  const size_t xsize = xyb0.xsize();
  const size_t ysize = xyb0.ysize();
  *mask = MaskImage(xsize, ysize);
  *mask_dc = MaskImage(xsize, ysize);

  const HWY_FULL(float) d;
  const double kMul0 = 0.580411307192999992;
  const double kMul1 = 0.236069675367;
  const double normalizer = 1.0 / (kMul0 + kMul1);
  const auto norm_mul0 = Set(d, normalizer * kMul0);
  const auto norm_mul1 = Set(d, normalizer * kMul1);
  const auto kMaskToErrorMul = Set(d, 0.2);
  const auto masked_mul0 = kMaskToErrorMul * norm_mul0;
  const auto masked_mul1 = kMaskToErrorMul * norm_mul1;

  static const double r0 = 1.63479141169;
  static const double r1 = 5.5;
  static const double r2 = 8.0;
  static const double border_ratio = 0;

  ImageF diff0(xsize, ysize);
  ImageF blur_temp(ysize, xsize);  // transposed

  {
    // X component
    static const double mul = 0.533043878407;
    static const double cutoff = 0.5;
    DiffPrecomputeX(xyb0.Plane(0), xyb1.Plane(0), mul, cutoff, &diff0);
    Blur(diff0, r2, border_ratio, &blur_temp, &mask->mask_x);
  }
  {
    // Y component
    static const double mul = 0.559;
    static const double mul2 = 1.0;
    ImageF diff1(xsize, ysize);
    ImageF blurred0_a(xsize, ysize);
    ImageF blurred0_b(xsize, ysize);
    ImageF blurred1_a(xsize, ysize);
    ImageF blurred1_b(xsize, ysize);
    DiffPrecomputeY(xyb0.Plane(1), xyb1.Plane(1), mul, mul2, &diff0, &diff1);
    Blur(diff0, r0, border_ratio, &blur_temp, &blurred0_a);
    Blur(diff0, r1, border_ratio, &blur_temp, &blurred0_b);
    Blur(diff1, r0, border_ratio, &blur_temp, &blurred1_a);
    Blur(diff1, r1, border_ratio, &blur_temp, &blurred1_b);

    for (size_t y = 0; y < ysize; ++y) {
      const float* JXL_RESTRICT row_blurred0_a = blurred0_a.Row(y);
      const float* JXL_RESTRICT row_blurred0_b = blurred0_b.Row(y);
      const float* JXL_RESTRICT row_blurred1_a = blurred1_a.ConstRow(y);
      const float* JXL_RESTRICT row_blurred1_b = blurred1_b.ConstRow(y);
      float* JXL_RESTRICT row_mask_yb = mask->mask_yb.Row(y);

      if (img_diff_ac == nullptr) {
        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          const auto blurred1_a = Load(d, row_blurred1_a + x);
          const auto blurred1_b = Load(d, row_blurred1_b + x);
          const auto mask_yb = norm_mul0 * blurred1_a + norm_mul1 * blurred1_b;
          Store(mask_yb, d, row_mask_yb + x);
        }
      } else {
        float* JXL_RESTRICT row_diff_ac = img_diff_ac->Row(y);

        for (size_t x = 0; x < xsize; x += Lanes(d)) {
          const auto blurred1_a = Load(d, row_blurred1_a + x);
          const auto blurred1_b = Load(d, row_blurred1_b + x);
          const auto mask_yb = norm_mul0 * blurred1_a + norm_mul1 * blurred1_b;
          Store(mask_yb, d, row_mask_yb + x);
          const auto va = Load(d, row_blurred0_a + x) - blurred1_a;
          const auto vb = Load(d, row_blurred0_b + x) - blurred1_b;
          const auto wa = masked_mul0 * va;
          const auto wb = masked_mul1 * vb;
          const auto diff_ac = Load(d, row_diff_ac + x);
          Store(diff_ac + wa * wa + wb * wb, d, row_diff_ac + x);
        }
      }
    }
  }
  // B component
  static const double w00 = 564.23707674490015;
  static const double w11 = 11.480266246995546;
  static const double p1_to_p0 = 0.022922908053854666;

  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      const double s0 = mask->mask_x.Row(y)[x];
      const double s1 = mask->mask_yb.Row(y)[x];
      const double p1 = w11 * s1;
      const double p0 = w00 * s0 + p1_to_p0 * p1;

      mask->mask_x.Row(y)[x] = MaskX(p0);
      mask->mask_yb.Row(y)[x] = MaskY(p1);
      mask_dc->mask_x.Row(y)[x] = MaskDcX(p0);
      mask_dc->mask_yb.Row(y)[x] = MaskDcY(p1);
    }
  }
}

void LinearCombination(const ImageF& in1, const ImageF& in2, double mul1,
                       double mul2, ImageF* BUTTERAUGLI_RESTRICT out) {
  JXL_DASSERT(SameSize(in1, in2) && SameSize(in1, *out));
  const size_t xsize = in1.xsize();
  const size_t ysize = in1.ysize();

  const HWY_FULL(float) d;
  const auto w1 = Set(d, mul1);
  const auto w2 = Set(d, mul2);

  for (size_t y = 0; y < ysize; ++y) {
    const float* BUTTERAUGLI_RESTRICT row_in1 = in1.Row(y);
    const float* BUTTERAUGLI_RESTRICT row_in2 = in2.Row(y);
    float* BUTTERAUGLI_RESTRICT row_out = out->Row(y);

    for (size_t x = 0; x < xsize; x += Lanes(d)) {
      const auto v1 = Load(d, row_in1 + x);
      const auto v2 = Load(d, row_in2 + x);
      Store(w1 * v1 + w2 * v2, d, row_out + x);
    }
  }
}

// `diff_ac` may be null.
void MaskPsychoImage(const PsychoImage& pi0, const PsychoImage& pi1,
                     const size_t xsize, const size_t ysize,
                     MaskImage* BUTTERAUGLI_RESTRICT mask,
                     MaskImage* BUTTERAUGLI_RESTRICT mask_dc,
                     ImageF* BUTTERAUGLI_RESTRICT diff_ac) {
  Image3F mask_xyb0(xsize, ysize);
  Image3F mask_xyb1(xsize, ysize);
  static const double muls[4] = {
      0.0,
      0.0632641915861,
      0.308212951541,
      1.16513324377,
  };
  for (int i = 0; i < 2; ++i) {
    LinearCombination(pi0.uhf[i], pi0.hf[i], muls[2 * i], muls[2 * i + 1],
                      const_cast<ImageF*>(&mask_xyb0.Plane(i)));
    LinearCombination(pi1.uhf[i], pi1.hf[i], muls[2 * i], muls[2 * i + 1],
                      const_cast<ImageF*>(&mask_xyb1.Plane(i)));
  }
  Mask(mask_xyb0, mask_xyb1, mask, mask_dc, diff_ac);
}

// Diffmap := sqrt of sum{diff images by multplied by X and Y/B masks}
void CombineChannelsToDiffmap(const MaskImage& mask_xyb,
                              const MaskImage& mask_xyb_dc,
                              const Image3F& block_diff_dc,
                              const Image3F& block_diff_ac, ImageF* diffmap) {
  PROFILER_FUNC;
  JXL_CHECK(SameSize(mask_xyb.mask_x, *diffmap));

  const HWY_FULL(float) d;

  for (size_t y = 0; y < diffmap->ysize(); ++y) {
    const float* JXL_RESTRICT row_mask_ac_x = mask_xyb.mask_x.ConstRow(y);
    const float* JXL_RESTRICT row_mask_ac_yb = mask_xyb.mask_yb.ConstRow(y);
    const float* JXL_RESTRICT row_mask_dc_x = mask_xyb_dc.mask_x.ConstRow(y);
    const float* JXL_RESTRICT row_mask_dc_yb = mask_xyb_dc.mask_yb.ConstRow(y);

    const float* JXL_RESTRICT row_diff_dc_x = block_diff_dc.PlaneRow(0, y);
    const float* JXL_RESTRICT row_diff_dc_y = block_diff_dc.PlaneRow(1, y);
    const float* JXL_RESTRICT row_diff_dc_b = block_diff_dc.PlaneRow(2, y);

    const float* JXL_RESTRICT row_diff_ac_x = block_diff_ac.PlaneRow(0, y);
    const float* JXL_RESTRICT row_diff_ac_y = block_diff_ac.PlaneRow(1, y);
    const float* JXL_RESTRICT row_diff_ac_b = block_diff_ac.PlaneRow(2, y);

    float* BUTTERAUGLI_RESTRICT row_out = diffmap->Row(y);
    for (size_t x = 0; x < diffmap->xsize(); x += Lanes(d)) {
      const auto diff_dc_x = Load(d, row_diff_dc_x + x);
      const auto diff_dc_y = Load(d, row_diff_dc_y + x);
      const auto diff_dc_b = Load(d, row_diff_dc_b + x);
      const auto mask_dc_yb = Load(d, row_mask_dc_yb + x);
      auto sum_x = diff_dc_x * Load(d, row_mask_dc_x + x);
      auto sum_y = diff_dc_y * mask_dc_yb;
      auto sum_b = diff_dc_b * mask_dc_yb;

      const auto diff_ac_x = Load(d, row_diff_ac_x + x);
      const auto diff_ac_y = Load(d, row_diff_ac_y + x);
      const auto diff_ac_b = Load(d, row_diff_ac_b + x);
      const auto mask_ac_yb = Load(d, row_mask_ac_yb + x);

      sum_x = MulAdd(diff_ac_x, Load(d, row_mask_ac_x + x), sum_x);
      sum_y = MulAdd(diff_ac_y, mask_ac_yb, sum_y);
      sum_b = MulAdd(diff_ac_b, mask_ac_yb, sum_b);

      const auto combined = sum_x + sum_y + sum_b;
      Store(Sqrt(combined), d, row_out + x);
    }
  }
}

static void L2Diff(const ImageF& i0, const ImageF& i1, const float w,
                   Image3F* BUTTERAUGLI_RESTRICT diffmap, size_t c) {
  if (w == 0) return;

  const HWY_FULL(float) d;
  const auto weight = Set(d, w);

  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row0 = i0.ConstRow(y);
    const float* BUTTERAUGLI_RESTRICT row1 = i1.ConstRow(y);
    float* BUTTERAUGLI_RESTRICT row_diff = diffmap->PlaneRow(c, y);

    for (size_t x = 0; x < i0.xsize(); x += Lanes(d)) {
      const auto diff = Load(d, row0 + x) - Load(d, row1 + x);
      const auto diff2 = diff * diff;
      const auto prev = Load(d, row_diff + x);
      Store(MulAdd(diff2, weight, prev), d, row_diff + x);
    }
  }
}

// i0 is the original image.
// i1 is the deformed copy.
static void L2DiffAsymmetric(const ImageF& i0, const ImageF& i1, double w_0gt1,
                             double w_0lt1,
                             Image3F* BUTTERAUGLI_RESTRICT diffmap, size_t c) {
  if (w_0gt1 == 0 && w_0lt1 == 0) {
    return;
  }

  const HWY_FULL(float) d;
  const auto vw_0gt1 = Set(d, w_0gt1 * 0.8);
  const auto vw_0lt1 = Set(d, w_0lt1 * 0.8);

  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row0 = i0.Row(y);
    const float* BUTTERAUGLI_RESTRICT row1 = i1.Row(y);
    float* BUTTERAUGLI_RESTRICT row_diff = diffmap->PlaneRow(c, y);

    for (size_t x = 0; x < i0.xsize(); x += Lanes(d)) {
      const auto val0 = Load(d, row0 + x);
      const auto val1 = Load(d, row1 + x);

      // Primary symmetric quadratic objective.
      const auto diff = val0 - val1;
      auto total = MulAdd(diff * diff, vw_0gt1, Load(d, row_diff + x));

      // Secondary half-open quadratic objectives.
      const auto fabs0 = Abs(val0);
      const auto too_small = Set(d, 0.4) * fabs0;
      const auto too_big = fabs0;

      const auto if_neg =
          IfThenElse(val1 > Neg(too_small), val1 + too_small,
                     IfThenElseZero(val1 < Neg(too_big), Neg(val1) - too_big));
      const auto if_pos =
          IfThenElse(val1 < too_small, too_small - val1,
                     IfThenElseZero(val1 > too_big, val1 - too_big));
      const auto v = IfThenElse(val0 < Zero(d), if_neg, if_pos);
      total += vw_0lt1 * v * v;
      Store(total, d, row_diff + x);
    }
  }
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(SeparateFrequencies)
HWY_EXPORT(MaskPsychoImage)
HWY_EXPORT(L2DiffAsymmetric)
HWY_EXPORT(L2Diff)
HWY_EXPORT(CombineChannelsToDiffmap)
HWY_EXPORT(MaltaDiffMap)
HWY_EXPORT(MaltaDiffMapLF)
HWY_EXPORT(DiffPrecomputeX)
HWY_EXPORT(DiffPrecomputeY)

static inline bool IsNan(const float x) {
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  const uint32_t bitmask_exp = 0x7F800000;
  return (bits & bitmask_exp) == bitmask_exp && (bits & 0x7FFFFF);
}

static inline bool IsNan(const double x) {
  uint64_t bits;
  memcpy(&bits, &x, sizeof(bits));
  return (0x7ff0000000000001ULL <= bits && bits <= 0x7fffffffffffffffULL) ||
         (0xfff0000000000001ULL <= bits && bits <= 0xffffffffffffffffULL);
}

static inline void CheckImage(const ImageF& image, const char* name) {
  PROFILER_FUNC;
  for (size_t y = 0; y < image.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      if (IsNan(row[x])) {
        printf("NAN: Image %s @ %zu,%zu (of %zu,%zu)\n", name, x, y,
               image.xsize(), image.ysize());
        exit(1);
      }
    }
  }
}

#if BUTTERAUGLI_ENABLE_CHECKS

#define CHECK_NAN(x, str)                \
  do {                                   \
    if (IsNan(x)) {                      \
      printf("%d: %s\n", __LINE__, str); \
      abort();                           \
    }                                    \
  } while (0)

#define CHECK_IMAGE(image, name) CheckImage(image, name)

#else

#define CHECK_NAN(x, str)
#define CHECK_IMAGE(image, name)

#endif

// Calculate a 2x2 subsampled image for purposes of recursive butteraugli at
// multiresolution.
static Image3F SubSample2x(const Image3F& in) {
  size_t xs = (in.xsize() + 1) / 2;
  size_t ys = (in.ysize() + 1) / 2;
  Image3F retval(xs, ys);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ys; ++y) {
      for (size_t x = 0; x < xs; ++x) {
        retval.PlaneRow(c, y)[x] = 0;
      }
    }
  }
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < in.ysize(); ++y) {
      for (size_t x = 0; x < in.xsize(); ++x) {
        retval.PlaneRow(c, y / 2)[x / 2] += 0.25 * in.PlaneRow(c, y)[x];
      }
    }
    if ((in.xsize() & 1) != 0) {
      for (size_t y = 0; y < retval.ysize(); ++y) {
        size_t last_column = retval.xsize() - 1;
        retval.PlaneRow(c, y)[last_column] *= 2.0;
      }
    }
    if ((in.ysize() & 1) != 0) {
      for (size_t x = 0; x < retval.xsize(); ++x) {
        size_t last_row = retval.ysize() - 1;
        retval.PlaneRow(c, last_row)[x] *= 2.0;
      }
    }
  }
  return retval;
}

// Supersample src by 2x and add it to dest.
static void AddSupersampled2x(const ImageF& src, float w, ImageF& dest) {
  for (size_t y = 0; y < dest.ysize(); ++y) {
    for (size_t x = 0; x < dest.xsize(); ++x) {
      // There will be less errors from the more averaged images.
      // We take it into account to some extent using a scaler.
      static const double kHeuristicMixingValue = 0.3;
      dest.Row(y)[x] *= 1.0 - kHeuristicMixingValue * w;
      dest.Row(y)[x] += w * src.Row(y / 2)[x / 2];
    }
  }
}

double SimpleGamma(double v) {
  // A simple HDR compatible gamma function.
  static const double kRetMul = 19.245013259874995;
  static const double kRetAdd = -23.16046239805755;
  static const double kVOffset = 9.9710635769299145;
  if (v < 0) {
    // This should happen rarely, but may lead to a NaN in log, which is
    // undesirable. Since negative photons don't exist we solve the NaNs by
    // clamping here.
    v = 0;
  }
  return kRetMul * std::log(v + kVOffset) + kRetAdd;
}

static inline double Gamma(double v) {
  // SimpleGamma must be used when using an intensity_target with butteraugli
  // to get values above 255.0.
  // GammaPolynomial is faster but may only be used if the maximum input value
  // is 255.0.
  // TODO(lode): allow jxl to specify which to use depending on the intensity
  //             target.
  return SimpleGamma(v);
  // return GammaPolynomial(v);
}

template <class V>
BUTTERAUGLI_INLINE void OpsinAbsorbance(const V& in0, const V& in1,
                                        const V& in2, V* JXL_RESTRICT out0,
                                        V* JXL_RESTRICT out1,
                                        V* JXL_RESTRICT out2) {
  // https://en.wikipedia.org/wiki/Photopsin absorbance modeling.
  static const double mixi0 = 0.29956550340058319;
  static const double mixi1 = 0.63373087833825936;
  static const double mixi2 = 0.077705617820981968;
  static const double mixi3 = 1.7557483643287353;
  static const double mixi4 = 0.22158691104574774;
  static const double mixi5 = 0.69391388044116142;
  static const double mixi6 = 0.0987313588422;
  static const double mixi7 = 1.7557483643287353;
  static const double mixi8 = 0.02;
  static const double mixi9 = 0.02;
  static const double mixi10 = 0.20480129041026129;
  static const double mixi11 = 12.226454707163354;

  const V mix0(mixi0);
  const V mix1(mixi1);
  const V mix2(mixi2);
  const V mix3(mixi3);
  const V mix4(mixi4);
  const V mix5(mixi5);
  const V mix6(mixi6);
  const V mix7(mixi7);
  const V mix8(mixi8);
  const V mix9(mixi9);
  const V mix10(mixi10);
  const V mix11(mixi11);

  *out0 = mix0 * in0 + mix1 * in1 + mix2 * in2 + mix3;
  *out1 = mix4 * in0 + mix5 * in1 + mix6 * in2 + mix7;
  *out2 = mix8 * in0 + mix9 * in1 + mix10 * in2 + mix11;
}

template <class V>
BUTTERAUGLI_INLINE void RgbToXyb(const V& r, const V& g, const V& b,
                                 V* JXL_RESTRICT valx, V* JXL_RESTRICT valy,
                                 V* JXL_RESTRICT valb) {
  *valx = r - g;
  *valy = r + g;
  *valb = b;
}

Image3F OpsinDynamicsImage(const Image3F& rgb) {
  PROFILER_FUNC;
  Image3F xyb(rgb.xsize(), rgb.ysize());
  const double kSigma = 1.2;
  ImageF blur_temp(rgb.ysize(), rgb.xsize());  // transposed
  Image3F blurred(rgb.xsize(), rgb.ysize());
  Blur(rgb.Plane(0), kSigma, 0.0, &blur_temp,
       const_cast<ImageF*>(&blurred.Plane(0)));
  Blur(rgb.Plane(1), kSigma, 0.0, &blur_temp,
       const_cast<ImageF*>(&blurred.Plane(1)));
  Blur(rgb.Plane(2), kSigma, 0.0, &blur_temp,
       const_cast<ImageF*>(&blurred.Plane(2)));
  for (size_t y = 0; y < rgb.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row_r = rgb.ConstPlaneRow(0, y);
    const float* BUTTERAUGLI_RESTRICT row_g = rgb.ConstPlaneRow(1, y);
    const float* BUTTERAUGLI_RESTRICT row_b = rgb.ConstPlaneRow(2, y);
    const float* BUTTERAUGLI_RESTRICT row_blurred_r =
        blurred.ConstPlaneRow(0, y);
    const float* BUTTERAUGLI_RESTRICT row_blurred_g =
        blurred.ConstPlaneRow(1, y);
    const float* BUTTERAUGLI_RESTRICT row_blurred_b =
        blurred.ConstPlaneRow(2, y);
    float* BUTTERAUGLI_RESTRICT row_out_x = xyb.PlaneRow(0, y);
    float* BUTTERAUGLI_RESTRICT row_out_y = xyb.PlaneRow(1, y);
    float* BUTTERAUGLI_RESTRICT row_out_b = xyb.PlaneRow(2, y);
    for (size_t x = 0; x < rgb.xsize(); ++x) {
      float sensitivity[3];
      {
        // Calculate sensitivity based on the smoothed image gamma derivative.
        float pre_mixed0, pre_mixed1, pre_mixed2;
        OpsinAbsorbance(row_blurred_r[x], row_blurred_g[x], row_blurred_b[x],
                        &pre_mixed0, &pre_mixed1, &pre_mixed2);
        if (pre_mixed0 < 1e-4f) {
          pre_mixed0 = 1e-4f;
        }
        if (pre_mixed1 < 1e-4f) {
          pre_mixed1 = 1e-4f;
        }
        if (pre_mixed2 < 1e-4f) {
          pre_mixed2 = 1e-4f;
        }
        sensitivity[0] = Gamma(pre_mixed0) / pre_mixed0;
        sensitivity[1] = Gamma(pre_mixed1) / pre_mixed1;
        sensitivity[2] = Gamma(pre_mixed2) / pre_mixed2;
        if (sensitivity[0] < 1e-4f) {
          sensitivity[0] = 1e-4f;
        }
        if (sensitivity[1] < 1e-4f) {
          sensitivity[1] = 1e-4f;
        }
        if (sensitivity[2] < 1e-4f) {
          sensitivity[2] = 1e-4f;
        }
      }
      float cur_mixed0, cur_mixed1, cur_mixed2;
      OpsinAbsorbance(row_r[x], row_g[x], row_b[x], &cur_mixed0, &cur_mixed1,
                      &cur_mixed2);
      cur_mixed0 *= sensitivity[0];
      cur_mixed1 *= sensitivity[1];
      cur_mixed2 *= sensitivity[2];
      RgbToXyb(cur_mixed0, cur_mixed1, cur_mixed2, &row_out_x[x], &row_out_y[x],
               &row_out_b[x]);
    }
  }
  return xyb;
}

ButteraugliComparator::ButteraugliComparator(const Image3F& rgb0,
                                             double hf_asymmetry)
    : xsize_(rgb0.xsize()),
      ysize_(rgb0.ysize()),
      hf_asymmetry_(hf_asymmetry),
      sub_(nullptr) {
  if (xsize_ < 8 || ysize_ < 8) {
    return;
  }
  Image3F xyb0 = OpsinDynamicsImage(rgb0);
  ChooseSeparateFrequencies()(xsize_, ysize_, xyb0, pi0_);

  // Awful recursive construction of samples of different resolution.
  // This is an after-thought and possibly somewhat parallel in
  // functionality with the PsychoImage multi-resolution approach.
  sub_ = new ButteraugliComparator(SubSample2x(rgb0), hf_asymmetry);
}

ButteraugliComparator::~ButteraugliComparator() { delete sub_; }

void ButteraugliComparator::Mask(MaskImage* BUTTERAUGLI_RESTRICT mask,
                                 MaskImage* BUTTERAUGLI_RESTRICT
                                     mask_dc) const {
  ChooseMaskPsychoImage()(pi0_, pi0_, xsize_, ysize_, mask, mask_dc, nullptr);
}

void ButteraugliComparator::Diffmap(const Image3F& rgb1, ImageF& result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) {
    ZeroFillImage(&result);
    return;
  }
  DiffmapOpsinDynamicsImage(OpsinDynamicsImage(rgb1), result);
  if (sub_) {
    if (sub_->xsize_ < 8 || sub_->ysize_ < 8) {
      return;
    }
    ImageF subresult;
    sub_->DiffmapOpsinDynamicsImage(OpsinDynamicsImage(SubSample2x(rgb1)),
                                    subresult);
    AddSupersampled2x(subresult, 0.5, result);
  }
}

void ButteraugliComparator::DiffmapOpsinDynamicsImage(const Image3F& xyb1,
                                                      ImageF& result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) {
    ZeroFillImage(&result);
    return;
  }
  PsychoImage pi1;
  ChooseSeparateFrequencies()(xsize_, ysize_, xyb1, pi1);
  result = ImageF(xsize_, ysize_);
  DiffmapPsychoImage(pi1, result);
}

namespace {

void MaltaDiffMap(const ImageF& lum0, const ImageF& lum1, const double w_0gt1,
                  const double w_0lt1, const double norm1,
                  ImageF* HWY_RESTRICT diffs,
                  Image3F* HWY_RESTRICT block_diff_ac, size_t c) {
  PROFILER_FUNC;
  const double len = 3.75;
  static const double mulli = 0.359826387683;
  ChooseMaltaDiffMap()(lum0, lum1, w_0gt1, w_0lt1, norm1, len, mulli, diffs,
                       block_diff_ac, c);
}

void MaltaDiffMapLF(const ImageF& lum0, const ImageF& lum1, const double w_0gt1,
                    const double w_0lt1, const double norm1,
                    ImageF* HWY_RESTRICT diffs,
                    Image3F* HWY_RESTRICT block_diff_ac, size_t c) {
  PROFILER_FUNC;
  const double len = 3.75;
  static const double mulli = 0.737143715861;
  ChooseMaltaDiffMapLF()(lum0, lum1, w_0gt1, w_0lt1, norm1, len, mulli, diffs,
                         block_diff_ac, c);
}

}  // namespace

void ButteraugliComparator::DiffmapPsychoImage(const PsychoImage& pi1,
                                               ImageF& diffmap) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) {
    ZeroFillImage(&diffmap);
    return;
  }
  ImageF diffs(xsize_, ysize_);  // temp, used in each call
  Image3F block_diff_dc(xsize_, ysize_);
  ZeroFillImage(&block_diff_dc);
  Image3F block_diff_ac(xsize_, ysize_);
  ZeroFillImage(&block_diff_ac);
  static const double wUhfMalta = 8.3230030810258917;
  static const double norm1Uhf = 81.97966510743784;
  MaltaDiffMap(pi0_.uhf[1], pi1.uhf[1], wUhfMalta * hf_asymmetry_,
               wUhfMalta / hf_asymmetry_, norm1Uhf, &diffs, &block_diff_ac, 1);

  static const double wUhfMaltaX = 51.738409933971198;
  static const double norm1UhfX = 19.390881354476463;
  MaltaDiffMap(pi0_.uhf[0], pi1.uhf[0], wUhfMaltaX * hf_asymmetry_,
               wUhfMaltaX / hf_asymmetry_, norm1UhfX, &diffs, &block_diff_ac,
               0);

  static const double wHfMalta = 240.0908262374156;
  static const double norm1Hf = 213.151889155;
  MaltaDiffMapLF(pi0_.hf[1], pi1.hf[1], wHfMalta * std::sqrt(hf_asymmetry_),
                 wHfMalta / std::sqrt(hf_asymmetry_), norm1Hf, &diffs,
                 &block_diff_ac, 1);

  static const double wHfMaltaX = 160.0;
  static const double norm1HfX = 80.0;
  MaltaDiffMapLF(pi0_.hf[0], pi1.hf[0], wHfMaltaX * std::sqrt(hf_asymmetry_),
                 wHfMaltaX / std::sqrt(hf_asymmetry_), norm1HfX, &diffs,
                 &block_diff_ac, 0);

  static const double wMfMalta = 163.56963612738494;
  static const double norm1Mf = 0.1635008533899469;
  MaltaDiffMapLF(pi0_.mf.Plane(1), pi1.mf.Plane(1), wMfMalta, wMfMalta, norm1Mf,
                 &diffs, &block_diff_ac, 1);

  static const double wMfMaltaX = 6164.558625327204;
  static const double norm1MfX = 1002.5;
  MaltaDiffMapLF(pi0_.mf.Plane(0), pi1.mf.Plane(0), wMfMaltaX, wMfMaltaX,
                 norm1MfX, &diffs, &block_diff_ac, 0);

  static const double wmul[9] = {
      32,
      5.0,
      0,
      32,
      5,
      237.33703833286302,
      0.8170086922843028,
      1.0323708525451885,
      5.5346699491372346,
  };
  for (size_t c = 0; c < 3; ++c) {
    if (c < 2) {  // No blue channel error accumulated at HF.
      ChooseL2DiffAsymmetric()(pi0_.hf[c], pi1.hf[c], wmul[c] * hf_asymmetry_,
                               wmul[c] / hf_asymmetry_, &block_diff_ac, c);
    }
    ChooseL2Diff()(pi0_.mf.Plane(c), pi1.mf.Plane(c), wmul[3 + c],
                   &block_diff_ac, c);
    ChooseL2Diff()(pi0_.lf.Plane(c), pi1.lf.Plane(c), wmul[6 + c],
                   &block_diff_dc, c);
  }

  MaskImage mask_xyb_ac;
  MaskImage mask_xyb_dc;
  ChooseMaskPsychoImage()(pi0_, pi1, xsize_, ysize_, &mask_xyb_ac, &mask_xyb_dc,
                          const_cast<ImageF*>(&block_diff_ac.Plane(1)));

  ChooseCombineChannelsToDiffmap()(mask_xyb_ac, mask_xyb_dc, block_diff_dc,
                                   block_diff_ac, &diffmap);
}

double ButteraugliScoreFromDiffmap(const ImageF& diffmap) {
  PROFILER_FUNC;
  float retval = 0.0f;
  for (size_t y = 0; y < diffmap.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row = diffmap.ConstRow(y);
    for (size_t x = 0; x < diffmap.xsize(); ++x) {
      retval = std::max(retval, row[x]);
    }
  }
  return retval;
}

bool ButteraugliDiffmap(const Image3F& rgb0, const Image3F& rgb1,
                        double hf_asymmetry, ImageF& diffmap) {
  PROFILER_FUNC;
  const size_t xsize = rgb0.xsize();
  const size_t ysize = rgb0.ysize();
  if (xsize < 1 || ysize < 1) {
    return JXL_FAILURE("Zero-sized image");
  }
  if (!SameSize(rgb0, rgb1)) {
    return JXL_FAILURE("Size mismatch");
  }
  static const int kMax = 8;
  if (xsize < kMax || ysize < kMax) {
    // Butteraugli values for small (where xsize or ysize is smaller
    // than 8 pixels) images are non-sensical, but most likely it is
    // less disruptive to try to compute something than just give up.
    // Temporarily extend the borders of the image to fit 8 x 8 size.
    size_t xborder = xsize < kMax ? (kMax - xsize) / 2 : 0;
    size_t yborder = ysize < kMax ? (kMax - ysize) / 2 : 0;
    size_t xscaled = std::max<size_t>(kMax, xsize);
    size_t yscaled = std::max<size_t>(kMax, ysize);
    Image3F scaled0(xscaled, yscaled);
    Image3F scaled1(xscaled, yscaled);
    for (int i = 0; i < 3; ++i) {
      for (size_t y = 0; y < yscaled; ++y) {
        for (size_t x = 0; x < xscaled; ++x) {
          size_t x2 =
              std::min<size_t>(xsize - 1, std::max<size_t>(0, x - xborder));
          size_t y2 =
              std::min<size_t>(ysize - 1, std::max<size_t>(0, y - yborder));
          scaled0.PlaneRow(i, y)[x] = rgb0.PlaneRow(i, y2)[x2];
          scaled1.PlaneRow(i, y)[x] = rgb1.PlaneRow(i, y2)[x2];
        }
      }
    }
    ImageF diffmap_scaled;
    const bool ok =
        ButteraugliDiffmap(scaled0, scaled1, hf_asymmetry, diffmap_scaled);
    diffmap = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        diffmap.Row(y)[x] = diffmap_scaled.Row(y + yborder)[x + xborder];
      }
    }
    return ok;
  }
  ButteraugliComparator butteraugli(rgb0, hf_asymmetry);
  butteraugli.Diffmap(rgb1, diffmap);
  return true;
}

bool ButteraugliInterface(const Image3F& rgb0, const Image3F& rgb1,
                          float hf_asymmetry, ImageF& diffmap,
                          double& diffvalue) {
#if PROFILER_ENABLED
  double t0 = Now();
#endif
  if (!ButteraugliDiffmap(rgb0, rgb1, hf_asymmetry, diffmap)) {
    return false;
  }
#if PROFILER_ENABLED
  double t1 = Now();
  const size_t mp = rgb0.xsize() * rgb0.ysize();
  printf("diff MP/s %f\n", mp / (t1 - t0) * 1E-6);
#endif
  diffvalue = ButteraugliScoreFromDiffmap(diffmap);
  return true;
}

double ButteraugliFuzzyClass(double score) {
  static const double fuzzy_width_up = 4.8;
  static const double fuzzy_width_down = 4.8;
  static const double m0 = 2.0;
  static const double scaler = 0.7777;
  double val;
  if (score < 1.0) {
    // val in [scaler .. 2.0]
    val = m0 / (1.0 + exp((score - 1.0) * fuzzy_width_down));
    val -= 1.0;           // from [1 .. 2] to [0 .. 1]
    val *= 2.0 - scaler;  // from [0 .. 1] to [0 .. 2.0 - scaler]
    val += scaler;        // from [0 .. 2.0 - scaler] to [scaler .. 2.0]
  } else {
    // val in [0 .. scaler]
    val = m0 / (1.0 + exp((score - 1.0) * fuzzy_width_up));
    val *= scaler;
  }
  return val;
}

// #define PRINT_OUT_NORMALIZATION

double ButteraugliFuzzyInverse(double seek) {
  double pos = 0;
  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double range = 1.0; range >= 1e-10; range *= 0.5) {
    double cur = ButteraugliFuzzyClass(pos);
    if (cur < seek) {
      pos -= range;
    } else {
      pos += range;
    }
  }
#ifdef PRINT_OUT_NORMALIZATION
  if (seek == 1.0) {
    fprintf(stderr, "Fuzzy inverse %g\n", pos);
  }
#endif
  return pos;
}

#ifdef PRINT_OUT_NORMALIZATION
static double print_out_normalization = ButteraugliFuzzyInverse(1.0);
#endif

namespace {

void ScoreToRgb(double score, double good_threshold, double bad_threshold,
                uint8_t rgb[3]) {
  double heatmap[12][3] = {
      {0, 0, 0},       {0, 0, 1},
      {0, 1, 1},       {0, 1, 0},  // Good level
      {1, 1, 0},       {1, 0, 0},  // Bad level
      {1, 0, 1},       {0.5, 0.5, 1.0},
      {1.0, 0.5, 0.5},  // Pastel colors for the very bad quality range.
      {1.0, 1.0, 0.5}, {1, 1, 1},
      {1, 1, 1},  // Last color repeated to have a solid range of white.
  };
  if (score < good_threshold) {
    score = (score / good_threshold) * 0.3;
  } else if (score < bad_threshold) {
    score = 0.3 +
            (score - good_threshold) / (bad_threshold - good_threshold) * 0.15;
  } else {
    score = 0.45 + (score - bad_threshold) / (bad_threshold * 12) * 0.5;
  }
  static const int kTableSize = sizeof(heatmap) / sizeof(heatmap[0]);
  score = std::min<double>(std::max<double>(score * (kTableSize - 1), 0.0),
                           kTableSize - 2);
  int ix = static_cast<int>(score);
  ix = std::min(std::max(0, ix), kTableSize - 2);  // Handle NaN
  double mix = score - ix;
  for (int i = 0; i < 3; ++i) {
    double v = mix * heatmap[ix + 1][i] + (1 - mix) * heatmap[ix][i];
    rgb[i] = static_cast<uint8_t>(255 * pow(v, 0.5) + 0.5);
  }
}

}  // namespace

Image3B CreateHeatMapImage(const ImageF& distmap, double good_threshold,
                           double bad_threshold) {
  Image3B heatmap(distmap.xsize(), distmap.ysize());
  for (size_t y = 0; y < distmap.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row_distmap = distmap.ConstRow(y);
    uint8_t* BUTTERAUGLI_RESTRICT row_h0 = heatmap.PlaneRow(0, y);
    uint8_t* BUTTERAUGLI_RESTRICT row_h1 = heatmap.PlaneRow(1, y);
    uint8_t* BUTTERAUGLI_RESTRICT row_h2 = heatmap.PlaneRow(2, y);
    for (size_t x = 0; x < distmap.xsize(); ++x) {
      const float d = row_distmap[x];
      uint8_t rgb[3];
      ScoreToRgb(d, good_threshold, bad_threshold, rgb);
      row_h0[x] = rgb[0];
      row_h1[x] = rgb[1];
      row_h2[x] = rgb[2];
    }
  }
  return heatmap;
}

}  // namespace jxl
#endif  // HWY_ONCE
