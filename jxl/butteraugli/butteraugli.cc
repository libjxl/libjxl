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

#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/image_ops.h"

#ifndef PROFILER_ENABLED
#define PROFILER_ENABLED 0
#endif
#if PROFILER_ENABLED
#else
#define PROFILER_FUNC
#define PROFILER_ZONE(name)
#endif

namespace jxl {
namespace butteraugli {

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
static Image3F SubSample2x(const Image3F &in) {
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
static void AddSupersampled2x(const ImageF &src, float w, ImageF &dest) {
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


// Purpose of kInternalGoodQualityThreshold:
// Normalize 'ok' image degradation to 1.0 across different versions of
// butteraugli.
static const double kInternalGoodQualityThreshold = 42.5;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

inline float MaskColor(const float color[3], const float mask[2]) {
  return color[0] * mask[0] + color[1] * mask[1] + color[2] * mask[1];
}

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
  float* BUTTERAUGLI_RESTRICT scaled_kernel =
      static_cast<float*>(malloc((len / 2 + 1) * sizeof(float)));
  for (size_t i = 0; i <= len / 2; ++i) {
    scaled_kernel[i] = kernel[i] * scale_no_border;
  }
  // left border
  for (size_t x = 0; x < border1; ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out->Row(x));
  }
  // middle
  switch (len) {
#if 1  // speed-optimized version
    case 5: {
      const float sk0 = scaled_kernel[0];
      const float sk1 = scaled_kernel[1];
      const float sk2 = scaled_kernel[2];
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum = (row_in[0] + row_in[4]) * sk0;
          sum += (row_in[1] + row_in[3]) * sk1;
          sum += (row_in[2]) * sk2;
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum;
        }
      }
    } break;
    case 9: {
      const float sk0 = scaled_kernel[0];
      const float sk1 = scaled_kernel[1];
      const float sk2 = scaled_kernel[2];
      const float sk3 = scaled_kernel[3];
      const float sk4 = scaled_kernel[4];
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum = (row_in[0] + row_in[8]) * sk0;
          sum += (row_in[1] + row_in[7]) * sk1;
          sum += (row_in[2] + row_in[6]) * sk2;
          sum += (row_in[3] + row_in[5]) * sk3;
          sum += (row_in[4]) * sk4;
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum;
        }
      }
    } break;
    case 17:
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum = (row_in[0] + row_in[16]) * scaled_kernel[0];
          sum += (row_in[1] + row_in[15]) * scaled_kernel[1];
          sum += (row_in[2] + row_in[14]) * scaled_kernel[2];
          sum += (row_in[3] + row_in[13]) * scaled_kernel[3];
          sum += (row_in[4] + row_in[12]) * scaled_kernel[4];
          sum += (row_in[5] + row_in[11]) * scaled_kernel[5];
          sum += (row_in[6] + row_in[10]) * scaled_kernel[6];
          sum += (row_in[7] + row_in[9]) * scaled_kernel[7];
          sum += (row_in[8]) * scaled_kernel[8];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum;
        }
      }
      break;
    case 33:
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum = (row_in[0] + row_in[32]) * scaled_kernel[0];
          sum += (row_in[1] + row_in[31]) * scaled_kernel[1];
          sum += (row_in[2] + row_in[30]) * scaled_kernel[2];
          sum += (row_in[3] + row_in[29]) * scaled_kernel[3];
          sum += (row_in[4] + row_in[28]) * scaled_kernel[4];
          sum += (row_in[5] + row_in[27]) * scaled_kernel[5];
          sum += (row_in[6] + row_in[26]) * scaled_kernel[6];
          sum += (row_in[7] + row_in[25]) * scaled_kernel[7];
          sum += (row_in[8] + row_in[24]) * scaled_kernel[8];
          sum += (row_in[9] + row_in[23]) * scaled_kernel[9];
          sum += (row_in[10] + row_in[22]) * scaled_kernel[10];
          sum += (row_in[11] + row_in[21]) * scaled_kernel[11];
          sum += (row_in[12] + row_in[20]) * scaled_kernel[12];
          sum += (row_in[13] + row_in[19]) * scaled_kernel[13];
          sum += (row_in[14] + row_in[18]) * scaled_kernel[14];
          sum += (row_in[15] + row_in[17]) * scaled_kernel[15];
          sum += (row_in[16]) * scaled_kernel[16];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum;
        }
      }
      break;
    case 11:
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum = (row_in[0] + row_in[10]) * scaled_kernel[0];
          sum += (row_in[1] + row_in[9]) * scaled_kernel[1];
          sum += (row_in[2] + row_in[8]) * scaled_kernel[2];
          sum += (row_in[3] + row_in[7]) * scaled_kernel[3];
          sum += (row_in[4] + row_in[6]) * scaled_kernel[4];
          sum += (row_in[5]) * scaled_kernel[5];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum;
        }
      }
      break;
    case 41:
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum = (row_in[0] + row_in[40]) * scaled_kernel[0];
          sum += (row_in[1] + row_in[39]) * scaled_kernel[1];
          sum += (row_in[2] + row_in[38]) * scaled_kernel[2];
          sum += (row_in[3] + row_in[37]) * scaled_kernel[3];
          sum += (row_in[4] + row_in[36]) * scaled_kernel[4];
          sum += (row_in[5] + row_in[35]) * scaled_kernel[5];
          sum += (row_in[6] + row_in[34]) * scaled_kernel[6];
          sum += (row_in[7] + row_in[33]) * scaled_kernel[7];
          sum += (row_in[8] + row_in[32]) * scaled_kernel[8];
          sum += (row_in[9] + row_in[31]) * scaled_kernel[9];
          sum += (row_in[10] + row_in[30]) * scaled_kernel[10];
          sum += (row_in[11] + row_in[29]) * scaled_kernel[11];
          sum += (row_in[12] + row_in[28]) * scaled_kernel[12];
          sum += (row_in[13] + row_in[27]) * scaled_kernel[13];
          sum += (row_in[14] + row_in[26]) * scaled_kernel[14];
          sum += (row_in[15] + row_in[25]) * scaled_kernel[15];
          sum += (row_in[16] + row_in[24]) * scaled_kernel[16];
          sum += (row_in[17] + row_in[23]) * scaled_kernel[17];
          sum += (row_in[18] + row_in[22]) * scaled_kernel[18];
          sum += (row_in[19] + row_in[21]) * scaled_kernel[19];
          sum += (row_in[20]) * scaled_kernel[20];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum;
        }
      }
      break;
    case 47:
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y) + border1 - offset;
        for (size_t x = border1; x < border2; ++x, ++row_in) {
          float sum = (row_in[0] + row_in[46]) * scaled_kernel[0];
          sum += (row_in[1] + row_in[45]) * scaled_kernel[1];
          sum += (row_in[2] + row_in[44]) * scaled_kernel[2];
          sum += (row_in[3] + row_in[43]) * scaled_kernel[3];
          sum += (row_in[4] + row_in[42]) * scaled_kernel[4];
          sum += (row_in[5] + row_in[41]) * scaled_kernel[5];
          sum += (row_in[6] + row_in[40]) * scaled_kernel[6];
          sum += (row_in[7] + row_in[39]) * scaled_kernel[7];
          sum += (row_in[8] + row_in[38]) * scaled_kernel[8];
          sum += (row_in[9] + row_in[37]) * scaled_kernel[9];
          sum += (row_in[10] + row_in[36]) * scaled_kernel[10];
          sum += (row_in[11] + row_in[35]) * scaled_kernel[11];
          sum += (row_in[12] + row_in[34]) * scaled_kernel[12];
          sum += (row_in[13] + row_in[33]) * scaled_kernel[13];
          sum += (row_in[14] + row_in[32]) * scaled_kernel[14];
          sum += (row_in[15] + row_in[31]) * scaled_kernel[15];
          sum += (row_in[16] + row_in[30]) * scaled_kernel[16];
          sum += (row_in[17] + row_in[29]) * scaled_kernel[17];
          sum += (row_in[18] + row_in[28]) * scaled_kernel[18];
          sum += (row_in[19] + row_in[27]) * scaled_kernel[19];
          sum += (row_in[20] + row_in[26]) * scaled_kernel[20];
          sum += (row_in[21] + row_in[25]) * scaled_kernel[21];
          sum += (row_in[22] + row_in[24]) * scaled_kernel[22];
          sum += (row_in[23]) * scaled_kernel[23];
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          row_out[y] = sum;
        }
      }
      break;
    default:
      //      printf("Warning: Unexpected kernel size! %d\n", len);
#else
    default:
#endif
      for (size_t y = 0; y < in.ysize(); ++y) {
        const float* BUTTERAUGLI_RESTRICT row_in = in.Row(y);
        for (size_t j, x = border1; x < border2; ++x) {
          const int d = x - offset;
          float* BUTTERAUGLI_RESTRICT row_out = out->Row(x);
          float sum = 0.0f;
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
  // right border
  for (size_t x = border2; x < in.xsize(); ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out->Row(x));
  }
  free(scaled_kernel);
}

// A blur somewhat similar to a 2D Gaussian blur.
// See: https://en.wikipedia.org/wiki/Gaussian_blur
void Blur(const ImageF& in, float sigma, float border_ratio,
          ImageF* BUTTERAUGLI_RESTRICT out) {
  std::vector<float> kernel = ComputeKernel(sigma);
  ImageF tmp(in.ysize(), in.xsize());
  ConvolutionWithTranspose(in, kernel, border_ratio, &tmp);
  ConvolutionWithTranspose(tmp, kernel, border_ratio, out);
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

double GammaMinArg() {
  double out0, out1, out2;
  OpsinAbsorbance(0.0, 0.0, 0.0, &out0, &out1, &out2);
  return std::min(out0, std::min(out1, out2));
}

double GammaMaxArg() {
  double out0, out1, out2;
  OpsinAbsorbance(255.0, 255.0, 255.0, &out0, &out1, &out2);
  return std::max(out0, std::max(out1, out2));
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
  return kRetMul * log(v + kVOffset) + kRetAdd;
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

Image3F OpsinDynamicsImage(const Image3F& rgb) {
  PROFILER_FUNC;
  Image3F xyb(rgb.xsize(), rgb.ysize());
  const double kSigma = 1.2;
  Image3F blurred(rgb.xsize(), rgb.ysize());
  Blur(rgb.Plane(0), kSigma, 0.0, const_cast<ImageF*>(&blurred.Plane(0)));
  Blur(rgb.Plane(1), kSigma, 0.0, const_cast<ImageF*>(&blurred.Plane(1)));
  Blur(rgb.Plane(2), kSigma, 0.0, const_cast<ImageF*>(&blurred.Plane(2)));
  for (size_t y = 0; y < rgb.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row_r = rgb.ConstPlaneRow(0, y);
    const float* BUTTERAUGLI_RESTRICT row_g = rgb.ConstPlaneRow(1, y);
    const float* BUTTERAUGLI_RESTRICT row_b = rgb.ConstPlaneRow(2, y);
    const float* BUTTERAUGLI_RESTRICT row_blurred_r = blurred.ConstPlaneRow(0, y);
    const float* BUTTERAUGLI_RESTRICT row_blurred_g = blurred.ConstPlaneRow(1, y);
    const float* BUTTERAUGLI_RESTRICT row_blurred_b = blurred.ConstPlaneRow(2, y);
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
        if (pre_mixed0 < 1e-4) {
          pre_mixed0 = 1e-4;
        }
        if (pre_mixed1 < 1e-4) {
          pre_mixed1 = 1e-4;
        }
        if (pre_mixed2 < 1e-4) {
          pre_mixed2 = 1e-4;
        }
        sensitivity[0] = Gamma(pre_mixed0) / pre_mixed0;
        sensitivity[1] = Gamma(pre_mixed1) / pre_mixed1;
        sensitivity[2] = Gamma(pre_mixed2) / pre_mixed2;
        if (sensitivity[0] < 1e-4) {
          sensitivity[0] = 1e-4;
        }
        if (sensitivity[1] < 1e-4) {
          sensitivity[1] = 1e-4;
        }
        if (sensitivity[2] < 1e-4) {
          sensitivity[2] = 1e-4;
        }
      }
      float cur_mixed0, cur_mixed1, cur_mixed2;
      OpsinAbsorbance(row_r[x], row_g[x], row_b[x], &cur_mixed0, &cur_mixed1,
                      &cur_mixed2);
      cur_mixed0 *= sensitivity[0];
      cur_mixed1 *= sensitivity[1];
      cur_mixed2 *= sensitivity[2];
      RgbToXyb(cur_mixed0, cur_mixed1, cur_mixed2,
               &row_out_x[x], &row_out_y[x], &row_out_b[x]);
    }
  }
  return xyb;
}

// Make area around zero less important (remove it).
static BUTTERAUGLI_INLINE float RemoveRangeAroundZero(float w, float x) {
  return x > w ? x - w : x < -w ? x + w : 0.0f;
}

// Make area around zero more important (2x it until the limit).
static BUTTERAUGLI_INLINE float AmplifyRangeAroundZero(float w, float x) {
  return x > w ? x + w : x < -w ? x - w : 2.0f * x;
}

// XybLowFreqToVals converts from low-frequency XYB space to the 'vals' space.
// Vals space can be converted to L2-norm space (Euclidean and normalized)
// through visual masking.
template <class V>
BUTTERAUGLI_INLINE void XybLowFreqToVals(const V& x, const V& y, const V& b_arg,
                                         V* BUTTERAUGLI_RESTRICT valx,
                                         V* BUTTERAUGLI_RESTRICT valy,
                                         V* BUTTERAUGLI_RESTRICT valb) {
  static const double xmuli = 16.728334267161084;
  static const double ymuli = 28.850249498135561;
  static const double bmuli = 44.392179956162011;
  static const double y_to_b_muli = -0.43716072660089195;
  const V xmul(xmuli);
  const V ymul(ymuli);
  const V bmul(bmuli);
  const V y_to_b_mul(y_to_b_muli);
  const V b = b_arg + y_to_b_mul * y;
  *valb = b * bmul;
  *valx = x * xmul;
  *valy = y * ymul;
}

static ImageF SuppressInBrightAreas(size_t xsize, size_t ysize, double mul,
                                    double mul2, double reg, const ImageF& hf,
                                    const ImageF& brightness) {
  PROFILER_FUNC;
  ImageF inew(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* rowhf = hf.Row(y);
    const float* rowbr = brightness.Row(y);
    float* rownew = inew.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      float v = rowhf[x];
      float scaler = mul * reg / (reg + rowbr[x]);
      rownew[x] = scaler * v;
    }
  }
  return inew;
}

static float MaximumClamp(float v, float maxval) {
  static const double kMul = 0.934914340314;
  if (v >= maxval) {
    v -= maxval;
    v *= kMul;
    v += maxval;
  } else if (v < -maxval) {
    v += maxval;
    v *= kMul;
    v -= maxval;
  }
  return v;
}

static ImageF MaximumClamping(size_t xsize, size_t ysize, const ImageF& ix,
                              double yw) {
  static const double kMul = 0.70036978414;
  ImageF inew(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* rowx = ix.Row(y);
    float* rownew = inew.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      double v = rowx[x];
      if (v >= yw) {
        v -= yw;
        v *= kMul;
        v += yw;
      } else if (v < -yw) {
        v += yw;
        v *= kMul;
        v -= yw;
      }
      rownew[x] = v;
    }
  }
  return inew;
}

static ImageF SuppressXByY(size_t xsize, size_t ysize, const ImageF& ix,
                           const ImageF& iy, const double yw) {
  static const double s = 0.941388349694;
  ImageF inew(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* rowx = ix.Row(y);
    const float* rowy = iy.Row(y);
    float* rownew = inew.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      const double xval = rowx[x];
      const double yval = rowy[x];
      const double scaler = s + (yw * (1.0 - s)) / (yw + yval * yval);
      rownew[x] = scaler * xval;
    }
  }
  return inew;
}

static void SeparateFrequencies(size_t xsize, size_t ysize,
                                const Image3F& xyb,
                                PsychoImage& ps) {
  PROFILER_FUNC;
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
    Blur(xyb.Plane(i), kSigmaLf, border_lf,
         const_cast<ImageF*>(&ps.lf.Plane(i)));

    // ... and keep everything else in mf.
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        ps.mf.PlaneRow(i, y)[x] =
            xyb.PlaneRow(i, y)[x] - ps.lf.ConstPlaneRow(i, y)[x];
      }
    }
    if (i == 2) {
      Blur(ps.mf.Plane(i), kSigmaHf, border_mf,
           const_cast<ImageF*>(&ps.mf.Plane(i)));
      break;
    }
    // Divide mf into mf and hf.
    for (size_t y = 0; y < ysize; ++y) {
      float* BUTTERAUGLI_RESTRICT row_mf = ps.mf.PlaneRow(i, y);
      float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[i].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_hf[x] = row_mf[x];
      }
    }
    Blur(ps.mf.Plane(i), kSigmaHf, border_mf,
         const_cast<ImageF*>(&ps.mf.Plane(i)));
    static const double kRemoveMfRange = 0.3;
    static const double kAddMfRange = 0.1;
    if (i == 0) {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT row_mf = ps.mf.PlaneRow(0, y);
        float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[0].Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_hf[x] -= row_mf[x];
          row_mf[x] = RemoveRangeAroundZero(kRemoveMfRange, row_mf[x]);
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT row_mf = ps.mf.PlaneRow(1, y);
        float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[1].Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_hf[x] -= row_mf[x];
          row_mf[x] = AmplifyRangeAroundZero(kAddMfRange, row_mf[x]);
        }
      }
    }
  }
  // Suppress red-green by intensity change in the high freq channels.
  static const double suppress = 286.09942757;
  ps.hf[0] = SuppressXByY(xsize, ysize, ps.hf[0], ps.hf[1], suppress);

  ps.uhf[0] = ImageF(xsize, ysize);
  ps.uhf[1] = ImageF(xsize, ysize);
  for (int i = 0; i < 2; ++i) {
    // Divide hf into hf and uhf.
    for (size_t y = 0; y < ysize; ++y) {
      float* BUTTERAUGLI_RESTRICT row_uhf = ps.uhf[i].Row(y);
      float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[i].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_uhf[x] = row_hf[x];
      }
    }
    Blur(ps.hf[i], kSigmaUhf, border_hf, &ps.hf[i]);
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
        for (size_t x = 0; x < xsize; ++x) {
          row_uhf[x] -= row_hf[x];
          row_hf[x] = RemoveRangeAroundZero(kRemoveHfRange, row_hf[x]);
          row_uhf[x] = RemoveRangeAroundZero(kRemoveUhfRange, row_uhf[x]);
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        float* BUTTERAUGLI_RESTRICT row_uhf = ps.uhf[1].Row(y);
        float* BUTTERAUGLI_RESTRICT row_hf = ps.hf[1].Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_uhf[x] -= row_hf[x];
          row_hf[x] = MaximumClamp(row_hf[x], kMaxclampHf);
          row_uhf[x] = MaximumClamp(row_uhf[x], kMaxclampUhf);
          row_uhf[x] *= kMulYUhf;
          row_hf[x] *= kMulYHf;
          row_hf[x] = AmplifyRangeAroundZero(kAddHfRange, row_hf[x]);
          row_uhf[x] = AmplifyRangeAroundZero(kAddUhfRange, row_uhf[x]);
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
    for (size_t x = 0; x < xsize; ++x) {
      float valx, valy, valb;
      XybLowFreqToVals(row_x[x], row_y[x], row_b[x], &valx, &valy, &valb);
      row_x[x] = valx;
      row_y[x] = valy;
      row_b[x] = valb;
    }
  }
}

static void L2Diff(const ImageF& i0, const ImageF& i1, const float w,
                   Image3F* BUTTERAUGLI_RESTRICT diffmap, size_t c) {
  if (w == 0) {
    return;
  }
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row0 = i0.ConstRow(y);
    const float* BUTTERAUGLI_RESTRICT row1 = i1.ConstRow(y);
    float* BUTTERAUGLI_RESTRICT row_diff = diffmap->PlaneRow(c, y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      const float diff = row0[x] - row1[x];
      row_diff[x] += w * diff * diff;
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
  w_0gt1 *= 0.8;
  w_0lt1 *= 0.8;
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row0 = i0.Row(y);
    const float* BUTTERAUGLI_RESTRICT row1 = i1.Row(y);
    float* BUTTERAUGLI_RESTRICT row_diff = diffmap->PlaneRow(c, y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      // Primary symmetric quadratic objective.
      double diff = row0[x] - row1[x];
      row_diff[x] += w_0gt1 * diff * diff;

      // Secondary half-open quadratic objectives.
      const double fabs0 = std::fabs(row0[x]);
      const double too_small = 0.4 * fabs0;
      const double too_big = 1.0 * fabs0;

      if (row0[x] < 0) {
        if (row1[x] > -too_small) {
          double v = row1[x] + too_small;
          row_diff[x] += w_0lt1 * v * v;
        } else if (row1[x] < -too_big) {
          double v = -row1[x] - too_big;
          row_diff[x] += w_0lt1 * v * v;
        }
      } else {
        if (row1[x] < too_small) {
          double v = too_small - row1[x];
          row_diff[x] += w_0lt1 * v * v;
        } else if (row1[x] > too_big) {
          double v = row1[x] - too_big;
          row_diff[x] += w_0lt1 * v * v;
        }
      }
    }
  }
}

ImageF CalculateDiffmap(const ImageF& diffmap_in) {
  PROFILER_FUNC;
  // Take square root.
  ImageF diffmap(diffmap_in.xsize(), diffmap_in.ysize());
  static const float kInitialSlope = 100.0f;
  for (size_t y = 0; y < diffmap.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT row_in = diffmap_in.Row(y);
    float* BUTTERAUGLI_RESTRICT row_out = diffmap.Row(y);
    for (size_t x = 0; x < diffmap.xsize(); ++x) {
      const float orig_val = row_in[x];
      // TODO(b/29974893): Until that is fixed do not call sqrt on very small
      // numbers.
      row_out[x] = (orig_val < (1.0f / (kInitialSlope * kInitialSlope))
                        ? kInitialSlope * orig_val
                        : std::sqrt(orig_val));
    }
  }
  return diffmap;
}

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
    double a = muls[2 * i];
    double b = muls[2 * i + 1];
    for (size_t y = 0; y < ysize; ++y) {
      const float* BUTTERAUGLI_RESTRICT row_hf0 = pi0.hf[i].Row(y);
      const float* BUTTERAUGLI_RESTRICT row_hf1 = pi1.hf[i].Row(y);
      const float* BUTTERAUGLI_RESTRICT row_uhf0 = pi0.uhf[i].Row(y);
      const float* BUTTERAUGLI_RESTRICT row_uhf1 = pi1.uhf[i].Row(y);
      float* BUTTERAUGLI_RESTRICT row0 = mask_xyb0.PlaneRow(i, y);
      float* BUTTERAUGLI_RESTRICT row1 = mask_xyb1.PlaneRow(i, y);
      for (size_t x = 0; x < xsize; ++x) {
        row0[x] = a * row_uhf0[x] + b * row_hf0[x];
        row1[x] = a * row_uhf1[x] + b * row_hf1[x];
      }
    }
  }
  Mask(mask_xyb0, mask_xyb1, mask, mask_dc, diff_ac);
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
  SeparateFrequencies(xsize_, ysize_, xyb0, pi0_);

  // Awful recursive construction of samples of different resolution.
  // This is an after-thought and possibly somewhat parallel in
  // functionality with the PsychoImage multi-resolution approach.
  sub_ = new ButteraugliComparator(SubSample2x(rgb0), hf_asymmetry);
}

ButteraugliComparator::~ButteraugliComparator() {
  delete sub_;
}


void ButteraugliComparator::Mask(
    MaskImage* BUTTERAUGLI_RESTRICT mask,
    MaskImage* BUTTERAUGLI_RESTRICT mask_dc) const {
  MaskPsychoImage(pi0_, pi0_, xsize_, ysize_, mask, mask_dc, nullptr);
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
    sub_->DiffmapOpsinDynamicsImage(
        OpsinDynamicsImage(SubSample2x(rgb1)), subresult);
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
  SeparateFrequencies(xsize_, ysize_, xyb1, pi1);
  result = ImageF(xsize_, ysize_);
  DiffmapPsychoImage(pi1, result);
}

void ButteraugliComparator::DiffmapPsychoImage(const PsychoImage& pi1,
                                               ImageF& result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) {
    ZeroFillImage(&result);
    return;
  }
  Image3F block_diff_dc(xsize_, ysize_);
  ZeroFillImage(&block_diff_dc);
  Image3F block_diff_ac(xsize_, ysize_);
  ZeroFillImage(&block_diff_ac);
  static const double wUhfMalta = 8.3230030810258917;
  static const double norm1Uhf = 81.97966510743784;
  MaltaDiffMap(pi0_.uhf[1], pi1.uhf[1], wUhfMalta * hf_asymmetry_,
               wUhfMalta / hf_asymmetry_, norm1Uhf, &block_diff_ac, 1);

  static const double wUhfMaltaX = 51.738409933971198;
  static const double norm1UhfX = 19.390881354476463;
  MaltaDiffMap(pi0_.uhf[0], pi1.uhf[0], wUhfMaltaX * hf_asymmetry_,
               wUhfMaltaX / hf_asymmetry_, norm1UhfX, &block_diff_ac, 0);

  static const double wHfMalta = 240.0908262374156;
  static const double norm1Hf = 213.151889155;
  MaltaDiffMapLF(pi0_.hf[1], pi1.hf[1], wHfMalta * std::sqrt(hf_asymmetry_),
                 wHfMalta / std::sqrt(hf_asymmetry_), norm1Hf, &block_diff_ac,
                 1);

  static const double wHfMaltaX = 160.0;
  static const double norm1HfX = 80.0;
  MaltaDiffMapLF(pi0_.hf[0], pi1.hf[0], wHfMaltaX * std::sqrt(hf_asymmetry_),
                 wHfMaltaX / std::sqrt(hf_asymmetry_), norm1HfX, &block_diff_ac,
                 0);

  static const double wMfMalta = 163.56963612738494;
  static const double norm1Mf = 0.1635008533899469;
  MaltaDiffMapLF(pi0_.mf.Plane(1), pi1.mf.Plane(1), wMfMalta, wMfMalta, norm1Mf,
                 &block_diff_ac, 1);

  static const double wMfMaltaX = 6164.558625327204;
  static const double norm1MfX = 1002.5;
  MaltaDiffMapLF(pi0_.mf.Plane(0), pi1.mf.Plane(0), wMfMaltaX, wMfMaltaX,
                 norm1MfX, &block_diff_ac, 0);

  static const double wmul[9] = {
    32, 5.0, 0, 32, 5, 237.33703833286302, 0.8170086922843028,
    1.0323708525451885, 5.5346699491372346,
  };
  for (size_t c = 0; c < 3; ++c) {
    if (c < 2) {  // No blue channel error accumulated at HF.
      L2DiffAsymmetric(pi0_.hf[c], pi1.hf[c], wmul[c] * hf_asymmetry_,
                       wmul[c] / hf_asymmetry_, &block_diff_ac, c);
    }
    L2Diff(pi0_.mf.Plane(c), pi1.mf.Plane(c), wmul[3 + c], &block_diff_ac, c);
    L2Diff(pi0_.lf.Plane(c), pi1.lf.Plane(c), wmul[6 + c], &block_diff_dc, c);
  }

  MaskImage mask_xyb_ac;
  MaskImage mask_xyb_dc;
  MaskPsychoImage(pi0_, pi1, xsize_, ysize_, &mask_xyb_ac, &mask_xyb_dc,
                  const_cast<ImageF*>(&block_diff_ac.Plane(1)));

  result = CalculateDiffmap(
      CombineChannels(mask_xyb_ac, mask_xyb_dc, block_diff_dc, block_diff_ac));
}

// Allows PaddedMaltaUnit to call either function via overloading.
struct MaltaTagLF {};
struct MaltaTag {};

static float MaltaUnit(MaltaTagLF /*tag*/, const float* BUTTERAUGLI_RESTRICT d,
                       const int xs) {
  const int xs3 = 3 * xs;
  float retval = 0;
  {
    // x grows, y constant
    float sum = d[-4] + d[-2] + d[0] + d[2] + d[4];
    retval += sum * sum;
  }
  {
    // y grows, x constant
    float sum = d[-xs3 - xs] + d[-xs - xs] + d[0] + d[xs + xs] + d[xs3 + xs];
    retval += sum * sum;
  }
  {
    // both grow
    float sum =
        d[-xs3 - 3] + d[-xs - xs - 2] + d[0] + d[xs + xs + 2] + d[xs3 + 3];
    retval += sum * sum;
  }
  {
    // y grows, x shrinks
    float sum =
        d[-xs3 + 3] + d[-xs - xs + 2] + d[0] + d[xs + xs - 2] + d[xs3 - 3];
    retval += sum * sum;
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    float sum = d[-xs3 - xs + 1] + d[-xs - xs + 1] + d[0] + d[xs + xs - 1] +
                d[xs3 + xs - 1];
    retval += sum * sum;
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    float sum = d[-xs3 - xs - 1] + d[-xs - xs - 1] + d[0] + d[xs + xs + 1] +
                d[xs3 + xs + 1];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    float sum = d[-4 - xs] + d[-2 - xs] + d[0] + d[2 + xs] + d[4 + xs];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    float sum = d[-4 + xs] + d[-2 + xs] + d[0] + d[2 - xs] + d[4 - xs];
    retval += sum * sum;
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
    float sum =
        d[-xs3 - 2] + d[-xs - xs - 1] + d[0] + d[xs + xs + 1] + d[xs3 + 2];
    retval += sum * sum;
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
    float sum =
        d[-xs3 + 2] + d[-xs - xs + 1] + d[0] + d[xs + xs - 1] + d[xs3 - 2];
    retval += sum * sum;
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
    float sum =
        d[-xs - xs - 3] + d[-xs - 2] + d[0] + d[xs + 2] + d[xs + xs + 3];
    retval += sum * sum;
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
    float sum =
        d[-xs - xs + 3] + d[-xs + 2] + d[0] + d[xs - 2] + d[xs + xs - 3];
    retval += sum * sum;
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

    float sum =
        d[xs + xs - 4] + d[xs - 2] + d[0] + d[-xs + 2] + d[-xs - xs + 4];
    retval += sum * sum;
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
    float sum =
        d[-xs - xs - 4] + d[-xs - 2] + d[0] + d[xs + 2] + d[xs + xs + 4];
    retval += sum * sum;
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
    float sum = d[-xs3 - xs - 2] + d[-xs - xs - 1] + d[0] + d[xs + xs + 1] +
                d[xs3 + xs + 2];
    retval += sum * sum;
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
    float sum = d[-xs3 - xs + 2] + d[-xs - xs + 1] + d[0] + d[xs + xs - 1] +
                d[xs3 + xs - 2];
    retval += sum * sum;
  }
  return retval;
}

static float MaltaUnit(MaltaTag /*tag*/, const float* BUTTERAUGLI_RESTRICT d,
                       const int xs) {
  const int xs3 = 3 * xs;
  float retval = 0;
  {
    // x grows, y constant
    float sum =
        d[-4] + d[-3] + d[-2] + d[-1] + d[0] + d[1] + d[2] + d[3] + d[4];
    retval += sum * sum;
  }
  {
    // y grows, x constant
    float sum = d[-xs3 - xs] + d[-xs3] + d[-xs - xs] + d[-xs] + d[0] + d[xs] +
                d[xs + xs] + d[xs3] + d[xs3 + xs];
    retval += sum * sum;
  }
  {
    // both grow
    float sum = d[-xs3 - 3] + d[-xs - xs - 2] + d[-xs - 1] + d[0] + d[xs + 1] +
                d[xs + xs + 2] + d[xs3 + 3];
    retval += sum * sum;
  }
  {
    // y grows, x shrinks
    float sum = d[-xs3 + 3] + d[-xs - xs + 2] + d[-xs + 1] + d[0] + d[xs - 1] +
                d[xs + xs - 2] + d[xs3 - 3];
    retval += sum * sum;
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    float sum = d[-xs3 - xs + 1] + d[-xs3 + 1] + d[-xs - xs + 1] + d[-xs] +
                d[0] + d[xs] + d[xs + xs - 1] + d[xs3 - 1] + d[xs3 + xs - 1];
    retval += sum * sum;
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    float sum = d[-xs3 - xs - 1] + d[-xs3 - 1] + d[-xs - xs - 1] + d[-xs] +
                d[0] + d[xs] + d[xs + xs + 1] + d[xs3 + 1] + d[xs3 + xs + 1];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    float sum = d[-4 - xs] + d[-3 - xs] + d[-2 - xs] + d[-1] + d[0] + d[1] +
                d[2 + xs] + d[3 + xs] + d[4 + xs];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    float sum = d[-4 + xs] + d[-3 + xs] + d[-2 + xs] + d[-1] + d[0] + d[1] +
                d[2 - xs] + d[3 - xs] + d[4 - xs];
    retval += sum * sum;
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
    float sum = d[-xs3 - 2] + d[-xs - xs - 1] + d[-xs - 1] + d[0] + d[xs + 1] +
                d[xs + xs + 1] + d[xs3 + 2];
    retval += sum * sum;
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
    float sum = d[-xs3 + 2] + d[-xs - xs + 1] + d[-xs + 1] + d[0] + d[xs - 1] +
                d[xs + xs - 1] + d[xs3 - 2];
    retval += sum * sum;
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
    float sum = d[-xs - xs - 3] + d[-xs - 2] + d[-xs - 1] + d[0] + d[xs + 1] +
                d[xs + 2] + d[xs + xs + 3];
    retval += sum * sum;
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
    float sum = d[-xs - xs + 3] + d[-xs + 2] + d[-xs + 1] + d[0] + d[xs - 1] +
                d[xs - 2] + d[xs + xs - 3];
    retval += sum * sum;
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

    float sum = d[xs - 4] + d[xs - 3] + d[xs - 2] + d[-1] +
                d[0] + d[1] + d[-xs + 2] + d[-xs + 3] + d[-xs + 4];
    retval += sum * sum;
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
    float sum = d[-xs - 4] + d[-xs - 3] + d[-xs - 2] + d[-1] +
                d[0] + d[1] + d[xs + 2] + d[xs + 3] + d[xs + 4];
    retval += sum * sum;
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
    float sum = d[-xs3 - xs - 1] + d[-xs3 - 1] + d[-xs - xs - 1] + d[-xs] +
                d[0] + d[xs] + d[xs + xs + 1] + d[xs3 + 1] + d[xs3 + xs + 1];
    retval += sum * sum;
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
    float sum = d[-xs3 - xs + 1] + d[-xs3 + 1] + d[-xs - xs + 1] + d[-xs] +
                d[0] + d[xs] + d[xs + xs - 1] + d[xs3 - 1] + d[xs3 + xs - 1];
    retval += sum * sum;
  }
  return retval;
}

// Returns MaltaUnit. "fastMode" avoids bounds-checks when x0 and y0 are known
// to be far enough from the image borders. "diffs" is a packed image.
template <bool fastMode, class Tag>
static BUTTERAUGLI_INLINE float PaddedMaltaUnit(
    const float* BUTTERAUGLI_RESTRICT diffs, const size_t x0, const size_t y0,
    const size_t xsize_, const size_t ysize_) {
  int ix0 = y0 * xsize_ + x0;
  const float* BUTTERAUGLI_RESTRICT d = &diffs[ix0];
  if (fastMode ||
      (x0 >= 4 && y0 >= 4 && x0 < (xsize_ - 4) && y0 < (ysize_ - 4))) {
    return MaltaUnit(Tag(), d, xsize_);
  }

  float borderimage[9 * 9];
  for (int dy = 0; dy < 9; ++dy) {
    int y = y0 + dy - 4;
    if (y < 0 || static_cast<size_t>(y) >= ysize_) {
      for (int dx = 0; dx < 9; ++dx) {
        borderimage[dy * 9 + dx] = 0.0f;
      }
    } else {
      for (int dx = 0; dx < 9; ++dx) {
        int x = x0 + dx - 4;
        if (x < 0 || static_cast<size_t>(x) >= xsize_) {
          borderimage[dy * 9 + dx] = 0.0f;
        } else {
          borderimage[dy * 9 + dx] = diffs[y * xsize_ + x];
        }
      }
    }
  }
  return MaltaUnit(Tag(), &borderimage[4 * 9 + 4], 9);
}

template <class Tag>
static void MaltaDiffMapImpl(const ImageF& lum0, const ImageF& lum1,
                             const size_t xsize_, const size_t ysize_,
                             const double w_0gt1, const double w_0lt1,
                             const double norm1, const double len,
                             const double mulli,
                             Image3F* BUTTERAUGLI_RESTRICT block_diff_ac,
                             size_t c) {
  const float kWeight0 = 0.5;
  const float kWeight1 = 0.33;

  const double w_pre0gt1 = mulli * std::sqrt(kWeight0 * w_0gt1) / (len * 2 + 1);
  const double w_pre0lt1 = mulli * std::sqrt(kWeight1 * w_0lt1) / (len * 2 + 1);
  const float norm2_0gt1 = w_pre0gt1 * norm1;
  const float norm2_0lt1 = w_pre0lt1 * norm1;

  std::vector<float> diffs(ysize_ * xsize_);
  for (size_t y = 0, ix = 0; y < ysize_; ++y) {
    const float* BUTTERAUGLI_RESTRICT row0 = lum0.Row(y);
    const float* BUTTERAUGLI_RESTRICT row1 = lum1.Row(y);
    for (size_t x = 0; x < xsize_; ++x, ++ix) {
      const float absval = 0.5f * (std::abs(row0[x]) + std::abs(row1[x]));
      const float diff = row0[x] - row1[x];
      const float scaler = norm2_0gt1 / (static_cast<float>(norm1) + absval);

      // Primary symmetric quadratic objective.
      diffs[ix] = scaler * diff;

      const float scaler2 = norm2_0lt1 / (static_cast<float>(norm1) + absval);
      const double fabs0 = std::fabs(row0[x]);

      // Secondary half-open quadratic objectives.
      const double too_small = 0.55 * fabs0;
      const double too_big = 1.05 * fabs0;

      if (row0[x] < 0) {
        if (row1[x] > -too_small) {
          double impact = scaler2 * (row1[x] + too_small);
          if (diff < 0) {
            diffs[ix] -= impact;
          } else {
            diffs[ix] += impact;
          }
        } else if (row1[x] < -too_big) {
          double impact = scaler2 * (-row1[x] - too_big);
          if (diff < 0) {
            diffs[ix] -= impact;
          } else {
            diffs[ix] += impact;
          }
        }
      } else {
        if (row1[x] < too_small) {
          double impact = scaler2 * (too_small - row1[x]);
          if (diff < 0) {
            diffs[ix] -= impact;
          } else {
            diffs[ix] += impact;
          }
        } else if (row1[x] > too_big) {
          double impact = scaler2 * (row1[x] - too_big);
          if (diff < 0) {
            diffs[ix] -= impact;
          } else {
            diffs[ix] += impact;
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
      row_diff[x0] +=
          PaddedMaltaUnit<false, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }
  }

  // Middle
  for (; y0 < ysize_ - 4; ++y0) {
    float* BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->PlaneRow(c, y0);
    size_t x0 = 0;
    for (; x0 < 4; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<false, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }
    for (; x0 < xsize_ - 4; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<true, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }

    for (; x0 < xsize_; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<false, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }
  }

  // Bottom
  for (; y0 < ysize_; ++y0) {
    float* BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->PlaneRow(c, y0);
    for (size_t x0 = 0; x0 < xsize_; ++x0) {
      row_diff[x0] +=
          PaddedMaltaUnit<false, Tag>(&diffs[0], x0, y0, xsize_, ysize_);
    }
  }
}

void ButteraugliComparator::MaltaDiffMap(
    const ImageF& lum0, const ImageF& lum1, const double w_0gt1,
    const double w_0lt1, const double norm1,
    Image3F* BUTTERAUGLI_RESTRICT block_diff_ac, size_t c) const {
  PROFILER_FUNC;
  const double len = 3.75;
  static const double mulli = 0.359826387683;
  MaltaDiffMapImpl<MaltaTag>(lum0, lum1, xsize_, ysize_, w_0gt1, w_0lt1, norm1,
                             len, mulli, block_diff_ac, c);
}

void ButteraugliComparator::MaltaDiffMapLF(
    const ImageF& lum0, const ImageF& lum1, const double w_0gt1,
    const double w_0lt1, const double norm1,
    Image3F* BUTTERAUGLI_RESTRICT block_diff_ac, size_t c) const {
  PROFILER_FUNC;
  const double len = 3.75;
  static const double mulli = 0.737143715861;
  MaltaDiffMapImpl<MaltaTagLF>(lum0, lum1, xsize_, ysize_, w_0gt1, w_0lt1,
                               norm1, len, mulli, block_diff_ac, c);
}

ImageF ButteraugliComparator::CombineChannels(
    const MaskImage& mask_xyb, const MaskImage& mask_xyb_dc,
    const Image3F& block_diff_dc, const Image3F& block_diff_ac) const {
  PROFILER_FUNC;
  ImageF result(xsize_, ysize_);
  for (size_t y = 0; y < ysize_; ++y) {
    float* BUTTERAUGLI_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x < xsize_; ++x) {
      float mask[2] = {
        mask_xyb.mask_x.Row(y)[x],
        mask_xyb.mask_yb.Row(y)[x],
      };
      float dc_mask[2] = {
        mask_xyb_dc.mask_x.Row(y)[x],
        mask_xyb_dc.mask_yb.Row(y)[x],
      };
      float diff_dc[3];
      float diff_ac[3];
      for (int i = 0; i < 3; ++i) {
        diff_dc[i] = block_diff_dc.PlaneRow(i, y)[x];
        diff_ac[i] = block_diff_ac.PlaneRow(i, y)[x];
      }
      row_out[x] = MaskColor(diff_dc, dc_mask) + MaskColor(diff_ac, mask);
    }
  }
  return result;
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

// ===== Functions used by Mask only =====
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

ImageF DiffPrecomputeX(const ImageF& xyb0, const ImageF& xyb1,
                       float mul, float cutoff) {
  PROFILER_FUNC;
  const size_t xsize = xyb0.xsize();
  const size_t ysize = xyb0.ysize();
  ImageF result(xsize, ysize);
  size_t x1, y1;
  size_t x2, y2;
  for (size_t y = 0; y < ysize; ++y) {
    if (y + 1 < ysize) {
      y2 = y + 1;
    } else if (y > 0) {
      y2 = y - 1;
    } else {
      y2 = y;
    }
    if (y == 0 && ysize >= 2) {
      y1 = y + 1;
    } else if (y > 0) {
      y1 = y - 1;
    } else {
      y1 = y;
    }
    const float* BUTTERAUGLI_RESTRICT row0_in = xyb0.Row(y);
    const float* BUTTERAUGLI_RESTRICT row1_in = xyb1.Row(y);
    const float* BUTTERAUGLI_RESTRICT row0_in1 = xyb0.Row(y1);
    const float* BUTTERAUGLI_RESTRICT row1_in1 = xyb1.Row(y1);
    const float* BUTTERAUGLI_RESTRICT row0_in2 = xyb0.Row(y2);
    const float* BUTTERAUGLI_RESTRICT row1_in2 = xyb1.Row(y2);
    float* BUTTERAUGLI_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      if (x + 1 < xsize) {
        x2 = x + 1;
      } else if (x > 0) {
        x2 = x - 1;
      } else {
        x2 = x;
      }
      if (x == 0 && xsize >= 2) {
        x1 = x + 1;
      } else if (x > 0) {
        x1 = x - 1;
      } else {
        x1 = x;
      }
      double sup0 = (std::fabs(row0_in[x] - row0_in[x2]) +
                     std::fabs(row0_in[x] - row0_in2[x]) +
                     std::fabs(row0_in[x] - row0_in[x1]) +
                     std::fabs(row0_in[x] - row0_in1[x]) +
                     3 * (std::fabs(row0_in2[x] - row0_in1[x]) +
                          std::fabs(row0_in[x1] - row0_in[x2])));
      double sup1 = (std::fabs(row1_in[x] - row1_in[x2]) +
                     std::fabs(row1_in[x] - row1_in2[x]) +
                     std::fabs(row1_in[x] - row1_in[x1]) +
                     std::fabs(row1_in[x] - row1_in1[x]) +
                     3 * (std::fabs(row1_in2[x] - row1_in1[x]) +
                          std::fabs(row1_in[x1] - row1_in[x2])));

      row_out[x] = mul * std::min(sup0, sup1);
      if (row_out[x] >= cutoff) {
        row_out[x] = cutoff;
      }
      {
        static const double limit = 0.5 * cutoff;
        if (row_out[x] >= limit) {
          row_out[x] += limit;
          row_out[x] *= 0.5;
        }
      }
      {
        static const double limit = 0.25 * cutoff;
        if (row_out[x] >= limit) {
          row_out[x] += limit;
          row_out[x] *= 0.5;
        }
      }
    }
  }
  return result;
}

// Precalculates masking for y channel, giving masks for
// both images back so that they can be used for similarity comparisons
// too.
void DiffPrecomputeY(const ImageF& xyb0, const ImageF& xyb1,
                     float mul, float mul2,
                     ImageF *out0, ImageF *out1) {
  PROFILER_FUNC;
  const size_t xsize = xyb0.xsize();
  const size_t ysize = xyb0.ysize();
  size_t x1, y1;
  size_t x2, y2;
  for (size_t y = 0; y < ysize; ++y) {
    if (y + 1 < ysize) {
      y2 = y + 1;
    } else if (y > 0) {
      y2 = y - 1;
    } else {
      y2 = y;
    }
    if (y == 0 && ysize >= 2) {
      y1 = y + 1;
    } else if (y > 0) {
      y1 = y - 1;
    } else {
      y1 = y;
    }
    const float* BUTTERAUGLI_RESTRICT row0_in = xyb0.Row(y);
    const float* BUTTERAUGLI_RESTRICT row1_in = xyb1.Row(y);
    const float* BUTTERAUGLI_RESTRICT row0_in1 = xyb0.Row(y1);
    const float* BUTTERAUGLI_RESTRICT row1_in1 = xyb1.Row(y1);
    const float* BUTTERAUGLI_RESTRICT row0_in2 = xyb0.Row(y2);
    const float* BUTTERAUGLI_RESTRICT row1_in2 = xyb1.Row(y2);
    float* BUTTERAUGLI_RESTRICT row_out0 = out0->Row(y);
    float* BUTTERAUGLI_RESTRICT row_out1 = out1->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      if (x + 1 < xsize) {
        x2 = x + 1;
      } else if (x > 0) {
        x2 = x - 1;
      } else {
        x2 = x;
      }
      if (x == 0 && xsize >= 2) {
        x1 = x + 1;
      } else if (x > 0) {
        x1 = x - 1;
      } else {
        x1 = x;
      }
      double sup0 = (std::fabs(row0_in[x] - row0_in[x2]) +
                     std::fabs(row0_in[x] - row0_in2[x]) +
                     std::fabs(row0_in[x] - row0_in[x1]) +
                     std::fabs(row0_in[x] - row0_in1[x]) +
                     3 * (std::fabs(row0_in2[x] - row0_in1[x]) +
                          std::fabs(row0_in[x1] - row0_in[x2])));
      double sup1 = (std::fabs(row1_in[x] - row1_in[x2]) +
                     std::fabs(row1_in[x] - row1_in2[x]) +
                     std::fabs(row1_in[x] - row1_in[x1]) +
                     std::fabs(row1_in[x] - row1_in1[x]) +
                     3 * (std::fabs(row1_in2[x] - row1_in1[x]) +
                          std::fabs(row1_in[x1] - row1_in[x2])));
      // kBias makes log behave more linearly.
      static const double kBias = 7;
      row_out0[x] = mul * (log(sup0 * sup0 * mul2 + kBias) - log(kBias));
      row_out1[x] = mul * (log(sup1 * sup1 * mul2 + kBias) - log(kBias));
    }
  }
}

void Mask(const Image3F& xyb0, const Image3F& xyb1,
          MaskImage* BUTTERAUGLI_RESTRICT mask,
          MaskImage* BUTTERAUGLI_RESTRICT mask_dc,
          ImageF* BUTTERAUGLI_RESTRICT diff_ac) {
  PROFILER_FUNC;
  const size_t xsize = xyb0.xsize();
  const size_t ysize = xyb0.ysize();
  *mask = MaskImage(xsize, ysize);
  *mask_dc = MaskImage(xsize, ysize);
  static const double muls[2] = {
    0.580411307192999992,
    0.236069675367,
  };
  double normalizer = {
    1.0 / (muls[0] + muls[1]),
  };
  static const double r0 = 1.63479141169;
  static const double r1 = 5.5;
  static const double r2 = 8.0;
  static const double border_ratio = 0;

  {
    // X component
    static const double mul = 0.533043878407;
    static const double cutoff = 0.5;
    ImageF diff = DiffPrecomputeX(xyb0.Plane(0), xyb1.Plane(0), mul, cutoff);
    ImageF blurred(diff.xsize(), diff.ysize());
    Blur(diff, r2, border_ratio, &blurred);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        mask->mask_x.Row(y)[x] = blurred.Row(y)[x];
      }
    }
  }
  {
    // Y component
    static const double mul = 0.559;
    static const double mul2 = 1.0;
    ImageF diff0(xsize, ysize);
    ImageF diff1(xsize, ysize);
    ImageF blurred0_a(xsize, ysize);
    ImageF blurred0_b(xsize, ysize);
    ImageF blurred1_a(xsize, ysize);
    ImageF blurred1_b(xsize, ysize);
    DiffPrecomputeY(xyb0.Plane(1), xyb1.Plane(1), mul, mul2, &diff0, &diff1);
    Blur(diff0, r0, border_ratio, &blurred0_a);
    Blur(diff0, r1, border_ratio, &blurred0_b);
    Blur(diff1, r0, border_ratio, &blurred1_a);
    Blur(diff1, r1, border_ratio, &blurred1_b);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        const double val = normalizer * (
            muls[0] * blurred1_a.Row(y)[x] +
            muls[1] * blurred1_b.Row(y)[x]);
        mask->mask_yb.Row(y)[x] = val;
        if (diff_ac != nullptr) {
          static const double kMaskToErrorMul = 0.2;
          double va = blurred0_a.Row(y)[x] - blurred1_a.Row(y)[x];
          double wa = kMaskToErrorMul * normalizer * muls[0] * va;
          double vb = blurred0_b.Row(y)[x] - blurred1_b.Row(y)[x];
          double wb = kMaskToErrorMul * normalizer * muls[1] * vb;
          diff_ac->Row(y)[x] += wa * wa + wb * wb;
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

bool ButteraugliDiffmap(const Image3F& rgb0, const Image3F& rgb1,
                        double hf_asymmetry, ImageF& result_image) {
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
    result_image = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        result_image.Row(y)[x] = diffmap_scaled.Row(y + yborder)[x + xborder];
      }
    }
    return ok;
  }
  ButteraugliComparator butteraugli(rgb0, hf_asymmetry);
  butteraugli.Diffmap(rgb1, result_image);
  return true;
}

bool ButteraugliInterface(const Image3F& rgb0, const Image3F& rgb1,
                          float hf_asymmetry, ImageF& diffmap,
                          double& diffvalue) {
  if (!ButteraugliDiffmap(rgb0, rgb1, hf_asymmetry, diffmap)) {
    return false;
  }
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
    val -= 1.0;  // from [1 .. 2] to [0 .. 1]
    val *= 2.0 - scaler;  // from [0 .. 1] to [0 .. 2.0 - scaler]
    val += scaler;  // from [0 .. 2.0 - scaler] to [scaler .. 2.0]
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
      {0, 0, 0},
      {0, 0, 1},
      {0, 1, 1},
      {0, 1, 0},  // Good level
      {1, 1, 0},
      {1, 0, 0},  // Bad level
      {1, 0, 1},
      {0.5, 0.5, 1.0},
      {1.0, 0.5, 0.5},  // Pastel colors for the very bad quality range.
      {1.0, 1.0, 0.5},
      {1, 1, 1},
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

}  // namespace butteraugli
}  // namespace jxl
