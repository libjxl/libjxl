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

#ifndef JXL_BUTTERAUGLI_BUTTERAUGLI_H_
#define JXL_BUTTERAUGLI_BUTTERAUGLI_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <memory>
#include <vector>

#include "jxl/base/compiler_specific.h"
#include "jxl/image.h"
#include "jxl/image_ops.h"

#define BUTTERAUGLI_ENABLE_CHECKS 0
#define BUTTERAUGLI_RESTRICT JXL_RESTRICT

// This is the main interface to butteraugli image similarity
// analysis function.

namespace jxl {
namespace butteraugli {

// ButteraugliInterface defines the public interface for butteraugli.
//
// It calculates the difference between rgb0 and rgb1.
//
// rgb0 and rgb1 contain the images. rgb0[c][px] and rgb1[c][px] contains
// the red image for c == 0, green for c == 1, blue for c == 2. Location index
// px is calculated as y * xsize + x.
//
// Value of pixels of images rgb0 and rgb1 need to be represented as raw
// intensity. Most image formats store gamma corrected intensity in pixel
// values. This gamma correction has to be removed, by applying the following
// function:
// butteraugli_val = 255.0 * pow(png_val / 255.0, gamma);
// A typical value of gamma is 2.2. It is usually stored in the image header.
// Take care not to confuse that value with its inverse. The gamma value should
// be always greater than one.
// Butteraugli does not work as intended if the caller does not perform
// gamma correction.
//
// hf_asymmetry is a multiplier for penalizing new HF artifacts more than
// blurring away features (1.0 -> neutral).
//
// diffmap will contain an image of the size xsize * ysize, containing
// localized differences for values px (indexed with the px the same as rgb0
// and rgb1). diffvalue will give a global score of similarity.
//
// A diffvalue smaller than kButteraugliGood indicates that images can be
// observed as the same image.
// diffvalue larger than kButteraugliBad indicates that a difference between
// the images can be observed.
// A diffvalue between kButteraugliGood and kButteraugliBad indicates that
// a subtle difference can be observed between the images.
//
// Returns true on success.

bool ButteraugliInterface(const Image3F &rgb0,
                          const Image3F &rgb1,
                          float hf_asymmetry,
                          ImageF &diffmap,
                          double &diffvalue);

// Converts the butteraugli score into fuzzy class values that are continuous
// at the class boundary. The class boundary location is based on human
// raters, but the slope is arbitrary. Particularly, it does not reflect
// the expectation value of probabilities of the human raters. It is just
// expected that a smoother class boundary will allow for higher-level
// optimization algorithms to work faster.
//
// Returns 2.0 for a perfect match, and 1.0 for 'ok', 0.0 for bad. Because the
// scoring is fuzzy, a butteraugli score of 0.96 would return a class of
// around 1.9.
double ButteraugliFuzzyClass(double score);

// Input values should be in range 0 (bad) to 2 (good). Use
// kButteraugliNormalization as normalization.
double ButteraugliFuzzyInverse(double seek);

// Implementation details, don't use anything below or your code will
// break in the future.

#ifdef _MSC_VER
#define BUTTERAUGLI_INLINE __forceinline
#else
#define BUTTERAUGLI_INLINE inline
#endif

#ifdef __clang__
// Early versions of Clang did not support __builtin_assume_aligned.
#define BUTTERAUGLI_HAS_ASSUME_ALIGNED __has_builtin(__builtin_assume_aligned)
#elif defined(__GNUC__)
#define BUTTERAUGLI_HAS_ASSUME_ALIGNED 1
#else
#define BUTTERAUGLI_HAS_ASSUME_ALIGNED 0
#endif

// Returns a void* pointer which the compiler then assumes is N-byte aligned.
// Example: float* JXL_RESTRICT aligned = (float*)JXL_ASSUME_ALIGNED(in, 32);
//
// The assignment semantics are required by GCC/Clang. ICC provides an in-place
// __assume_aligned, whereas MSVC's __assume appears unsuitable.
#if BUTTERAUGLI_HAS_ASSUME_ALIGNED
#define BUTTERAUGLI_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#else
#define BUTTERAUGLI_ASSUME_ALIGNED(ptr, align) (ptr)
#endif  // BUTTERAUGLI_HAS_ASSUME_ALIGNED

struct MaskImage {
  MaskImage() {}
  MaskImage(int xs, int ys) :
      mask_x(xs, ys),
      mask_yb(xs, ys) {}
  ImageF mask_x;
  ImageF mask_yb;
};

struct PsychoImage {
  ImageF uhf[2];  // XY
  ImageF hf[2];   // XY
  Image3F mf;     // XYB
  Image3F lf;     // XYB
};

class ButteraugliComparator {
 public:
  ButteraugliComparator(const Image3F &rgb0, double hf_asymmetry);
  virtual ~ButteraugliComparator();

  // Computes the butteraugli map between the original image given in the
  // constructor and the distorted image give here.
  void Diffmap(const Image3F &rgb1, ImageF &result) const;

  // Same as above, but OpsinDynamicsImage() was already applied.
  void DiffmapOpsinDynamicsImage(const Image3F &xyb1, ImageF &result) const;

  // Same as above, but the frequency decomposition was already applied.
  void DiffmapPsychoImage(const PsychoImage& ps1, ImageF &result) const;

void Mask(MaskImage *BUTTERAUGLI_RESTRICT mask, MaskImage *BUTTERAUGLI_RESTRICT mask_dc) const;

 private:
  void MaltaDiffMapLF(const ImageF &y0, const ImageF &y1, double w_0gt1,
                      double w_0lt1, double normalization,
                      Image3F *BUTTERAUGLI_RESTRICT block_diff_ac,
                      size_t c) const;

  void MaltaDiffMap(const ImageF &y0, const ImageF &y1, double w_0gt1,
                    double w_0lt1, double normalization,
                    Image3F *BUTTERAUGLI_RESTRICT block_diff_ac,
                    size_t c) const;

  ImageF CombineChannels(const MaskImage &mask_ac,
                         const MaskImage &mask_dc,
                         const Image3F &block_diff_dc,
                         const Image3F &block_diff_ac) const;

  const size_t xsize_;
  const size_t ysize_;
  float hf_asymmetry_;
  PsychoImage pi0_;
  ButteraugliComparator *sub_;
};

bool ButteraugliDiffmap(const Image3F &rgb0, const Image3F &rgb1,
                        double hf_asymmetry, ImageF &diffmap);

double ButteraugliScoreFromDiffmap(const ImageF& distmap);

// Generate rgb-representation of the distance between two images.
Image3B CreateHeatMapImage(const ImageF &distmap, double good_threshold,
                           double bad_threshold);

// Compute values of local frequency and dc masking based on the activity
// in the two images.
void Mask(const Image3F &xyb0, const Image3F &xyb1,
          MaskImage *BUTTERAUGLI_RESTRICT mask,
          MaskImage *BUTTERAUGLI_RESTRICT mask_dc, ImageF *diff_ac = nullptr);

template <class V>
BUTTERAUGLI_INLINE void RgbToXyb(const V &r, const V &g, const V &b,
                                 V *JXL_RESTRICT valx, V *JXL_RESTRICT valy,
                                 V *JXL_RESTRICT valb) {
  *valx = r - g;
  *valy = r + g;
  *valb = b;
}

template <class V>
BUTTERAUGLI_INLINE void OpsinAbsorbance(const V &in0, const V &in1,
                                        const V &in2, V *JXL_RESTRICT out0,
                                        V *JXL_RESTRICT out1,
                                        V *JXL_RESTRICT out2) {
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

Image3F OpsinDynamicsImage(const Image3F &rgb);

ImageF Blur(const ImageF& in, float sigma, float border_ratio);

double SimpleGamma(double v);

double GammaMinArg();
double GammaMaxArg();

// Polynomial evaluation via Clenshaw's scheme (similar to Horner's).
// Template enables compile-time unrolling of the recursion, but must reside
// outside of a class due to the specialization.
template <int INDEX>
static inline void ClenshawRecursion(const double x, const double *coefficients,
                                     double *b1, double *b2) {
  const double x_b1 = x * (*b1);
  const double t = (x_b1 + x_b1) - (*b2) + coefficients[INDEX];
  *b2 = *b1;
  *b1 = t;

  ClenshawRecursion<INDEX - 1>(x, coefficients, b1, b2);
}

// Base case
template <>
inline void ClenshawRecursion<0>(
    const double x, const double *coefficients, double *b1,
    double *b2) {  // NOLINT(readability-non-const-parameter)
  const double x_b1 = x * (*b1);
  // The final iteration differs - no 2 * x_b1 here.
  *b1 = x_b1 - (*b2) + coefficients[0];
}

// Rational polynomial := dividing two polynomial evaluations. These are easier
// to find than minimax polynomials.
struct RationalPolynomial {
  template <int N>
  static double EvaluatePolynomial(const double x,
                                   const double (&coefficients)[N]) {
    double b1 = 0.0;
    double b2 = 0.0;
    ClenshawRecursion<N - 1>(x, coefficients, &b1, &b2);
    return b1;
  }

  // Evaluates the polynomial at x (in [min_value, max_value]).
  inline double operator()(const double x) const {
    // First normalize to [0, 1].
    const double x01 = (x - min_value) / (max_value - min_value);
    // And then to [-1, 1] domain of Chebyshev polynomials.
    const double xc = 2.0 * x01 - 1.0;

    const double yp = EvaluatePolynomial(xc, p);
    const double yq = EvaluatePolynomial(xc, q);
    if (yq == 0.0) return 0.0;
    return static_cast<float>(yp / yq);
  }

  // Domain of the polynomials; they are undefined elsewhere.
  double min_value;
  double max_value;

  // Coefficients of T_n (Chebyshev polynomials of the first kind).
  // Degree 5/5 is a compromise between accuracy (0.1%) and numerical stability.
  double p[5 + 1];
  double q[5 + 1];
};

static inline double GammaPolynomial(double value) {
  static const RationalPolynomial r = {
    0.971783, 590.188894,
    {
      98.7821300963361, 164.273222212631, 92.948112871376,
      33.8165311212688, 6.91626704983562, 0.556380877028234
    },
    {
      1, 1.64339473427892, 0.89392405219969, 0.298947051776379,
      0.0507146002577288, 0.00226495093949756
    }};
  return r(value);
}

}  // namespace butteraugli
}  // namespace jxl

#endif  // JXL_BUTTERAUGLI_BUTTERAUGLI_H_
