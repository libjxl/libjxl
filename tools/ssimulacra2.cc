// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
SSIMULACRA 2 - Structural SIMilarity Unveiling Local And Compression Related
Artifacts

Perceptual metric developed by Jon Sneyers (Cloudinary) in July 2022.
Design:
- XYB color space (X+0.5, Y, Y-B+1.0)
- SSIM map
- 'blockiness/ringing' map (distorted has edges where original is smooth)
- 'smoothing' map (distorted is smooth where original has edges)
- error maps are computed at 6 scales (1:1 to 1:32) for each component (X,Y,B)
- for all 6*3*3=54 maps, two norms are computed: 1-norm (mean) and 4-norm
- a weighted sum of these 54*2=108 norms leads to the final score
- weights were tuned based on a large set of subjective scores for images
  compressed with JPEG, JPEG 2000, JPEG XL, WebP, AVIF, and HEIC.
*/

#include "tools/ssimulacra2.h"

#include <stdio.h>

#include <cmath>

#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/gauss_blur.h"
#include "lib/jxl/image_ops.h"

namespace {

using jxl::Image3F;
using jxl::ImageF;

static const float kC1 = 0.0001f;
static const float kC2 = 0.0003f;
static const int kNumScales = 6;

Image3F Downsample(const Image3F& in, size_t fx, size_t fy) {
  const size_t out_xsize = (in.xsize() + fx - 1) / fx;
  const size_t out_ysize = (in.ysize() + fy - 1) / fy;
  Image3F out(out_xsize, out_ysize);
  const float normalize = 1.0f / (fx * fy);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t oy = 0; oy < out_ysize; ++oy) {
      float* JXL_RESTRICT row_out = out.PlaneRow(c, oy);
      for (size_t ox = 0; ox < out_xsize; ++ox) {
        float sum = 0.0f;
        for (size_t iy = 0; iy < fy; ++iy) {
          for (size_t ix = 0; ix < fx; ++ix) {
            const size_t x = std::min(ox * fx + ix, in.xsize() - 1);
            const size_t y = std::min(oy * fy + iy, in.ysize() - 1);
            sum += in.PlaneRow(c, y)[x];
          }
        }
        row_out[ox] = sum * normalize;
      }
    }
  }
  return out;
}

void Multiply(const Image3F& a, const Image3F& b, Image3F* mul) {
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < a.ysize(); ++y) {
      const float* JXL_RESTRICT in1 = a.PlaneRow(c, y);
      const float* JXL_RESTRICT in2 = b.PlaneRow(c, y);
      float* JXL_RESTRICT out = mul->PlaneRow(c, y);
      for (size_t x = 0; x < a.xsize(); ++x) {
        out[x] = in1[x] * in2[x];
      }
    }
  }
}

// Temporary storage for Gaussian blur, reused for multiple images.
class Blur {
 public:
  Blur(const size_t xsize, const size_t ysize)
      : rg_(jxl::CreateRecursiveGaussian(1.5)), temp_(xsize, ysize) {}

  void operator()(const ImageF& in, ImageF* JXL_RESTRICT out) {
    jxl::ThreadPool* null_pool = nullptr;
    FastGaussian(rg_, in, null_pool, &temp_, out);
  }

  Image3F operator()(const Image3F& in) {
    Image3F out(in.xsize(), in.ysize());
    operator()(in.Plane(0), &out.Plane(0));
    operator()(in.Plane(1), &out.Plane(1));
    operator()(in.Plane(2), &out.Plane(2));
    return out;
  }

  // Allows reusing across scales.
  void ShrinkTo(const size_t xsize, const size_t ysize) {
    temp_.ShrinkTo(xsize, ysize);
  }

 private:
  hwy::AlignedUniquePtr<jxl::RecursiveGaussian> rg_;
  ImageF temp_;
};

double tothe4th(double x) {
  x *= x;
  x *= x;
  return x;
}
void SSIMMap(const Image3F& m1, const Image3F& m2, const Image3F& s11,
             const Image3F& s22, const Image3F& s12, double* plane_averages) {
  const double onePerPixels = 1.0 / (m1.ysize() * m1.xsize());
  for (size_t c = 0; c < 3; ++c) {
    double sum1[2] = {0.0};
    for (size_t y = 0; y < m1.ysize(); ++y) {
      const float* JXL_RESTRICT row_m1 = m1.PlaneRow(c, y);
      const float* JXL_RESTRICT row_m2 = m2.PlaneRow(c, y);
      const float* JXL_RESTRICT row_s11 = s11.PlaneRow(c, y);
      const float* JXL_RESTRICT row_s22 = s22.PlaneRow(c, y);
      const float* JXL_RESTRICT row_s12 = s12.PlaneRow(c, y);
      for (size_t x = 0; x < m1.xsize(); ++x) {
        float mu1 = row_m1[x];
        float mu2 = row_m2[x];
        float mu11 = mu1 * mu1;
        float mu22 = mu2 * mu2;
        float mu12 = mu1 * mu2;
        float num_m = 2 * mu12 + kC1;
        float num_s = 2 * (row_s12[x] - mu12) + kC2;
        float denom_m = mu11 + mu22 + kC1;
        float denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + kC2;
        double d = 1.0 - ((num_m * num_s) / (denom_m * denom_s));
        d = std::max(d, 0.0);
        sum1[0] += d;
        sum1[1] += tothe4th(d);
      }
    }
    plane_averages[c * 2] = onePerPixels * sum1[0];
    plane_averages[c * 2 + 1] = sqrt(sqrt(onePerPixels * sum1[1]));
  }
}

void EdgeDiffMap(const Image3F& img1, const Image3F& mu1, const Image3F& img2,
                 const Image3F& mu2, double* plane_averages) {
  const double onePerPixels = 1.0 / (img1.ysize() * img1.xsize());
  for (size_t c = 0; c < 3; ++c) {
    double sum1[4] = {0.0};
    for (size_t y = 0; y < img1.ysize(); ++y) {
      const float* JXL_RESTRICT row1 = img1.PlaneRow(c, y);
      const float* JXL_RESTRICT row2 = img2.PlaneRow(c, y);
      const float* JXL_RESTRICT rowm1 = mu1.PlaneRow(c, y);
      const float* JXL_RESTRICT rowm2 = mu2.PlaneRow(c, y);
      for (size_t x = 0; x < img1.xsize(); ++x) {
        double d1 = (1.0 + std::abs(row2[x] - rowm2[x])) /
                        (1.0 + std::abs(row1[x] - rowm1[x])) -
                    1.0;
        // d1 > 0: distorted has an edge where original is smooth
        //         (indicating ringing, color banding, blockiness, etc)
        // d1 < 0: original has an edge where distorted is smooth
        //         (indicating smoothing, blurring, smearing, etc)
        double artifact = std::max(d1, 0.0);
        sum1[0] += artifact;
        sum1[1] += tothe4th(artifact);
        double detail_lost = std::max(-d1, 0.0);
        sum1[2] += detail_lost;
        sum1[3] += tothe4th(detail_lost);
      }
    }
    plane_averages[c * 4] = onePerPixels * sum1[0];
    plane_averages[c * 4 + 1] = sqrt(sqrt(onePerPixels * sum1[1]));
    plane_averages[c * 4 + 2] = onePerPixels * sum1[2];
    plane_averages[c * 4 + 3] = sqrt(sqrt(onePerPixels * sum1[3]));
  }
}

// Add 0.5 to X and turn B into 1 + B-Y
// (SSIM expects non-negative ranges)
void MakePositiveXYB(jxl::Image3F& img) {
  for (size_t y = 0; y < img.ysize(); ++y) {
    const float* JXL_RESTRICT rowY = img.PlaneRow(1, y);
    float* JXL_RESTRICT rowB = img.PlaneRow(2, y);
    float* JXL_RESTRICT rowX = img.PlaneRow(0, y);
    for (size_t x = 0; x < img.xsize(); ++x) {
      rowB[x] += 1.0f - rowY[x];
      rowX[x] += 0.5f;
    }
  }
}

void AlphaBlend(jxl::ImageBundle& img, float bg) {
  for (size_t y = 0; y < img.ysize(); ++y) {
    float* JXL_RESTRICT r = img.color()->PlaneRow(0, y);
    float* JXL_RESTRICT g = img.color()->PlaneRow(1, y);
    float* JXL_RESTRICT b = img.color()->PlaneRow(2, y);
    const float* JXL_RESTRICT a = img.alpha()->Row(y);
    for (size_t x = 0; x < img.xsize(); ++x) {
      r[x] = a[x] * r[x] + (1.f - a[x]) * bg;
      g[x] = a[x] * g[x] + (1.f - a[x]) * bg;
      b[x] = a[x] * b[x] + (1.f - a[x]) * bg;
    }
  }
}

}  // namespace

/*
The final score is based on a weighted sum of 108 sub-scores:
- for 6 scales (1:1 to 1:32)
- for 3 components (X + 0.5, Y, B - Y + 1.0)
- using 2 norms (the 1-norm and the 4-norm)
- over 3 error maps:
    - SSIM
    - "ringing" (distorted edges where there are no orig edges)
    - "blurring" (orig edges where there are no distorted edges)

The weights were obtained by running Nelder-Mead simplex search,
optimizing to minimize MSE and maximize Kendall and Pearson correlation
for training data consisting of 17611 subjective quality scores,
validated on separate validation data consisting of 4292 scores.
*/
double Msssim::Score() const {
  double ssim = 0.0;

  constexpr double weight[108] = {-0.4887721343447775,
                                  0.023424809409418046,
                                  -2.030114616889109,
                                  0.09295054976135864,
                                  -0.6104037165108389,
                                  -1.5043796645843681,
                                  0.7157171902116809,
                                  0.021100488642748494,
                                  0.020230927541050825,
                                  0.2133671136286971,
                                  2.907924945735136,
                                  0.022228357764969564,
                                  -0.5071214811492599,
                                  0.02509144129036578,
                                  0.022718837619484678,
                                  0.06120691522596422,
                                  0.30589062196252637,
                                  1.0329845093332668,
                                  0.014860334913185502,
                                  0.01765439482987108,
                                  0.015223496420421201,
                                  0.3352969449271075,
                                  0.022351073746802674,
                                  4.920455953428159,
                                  0.024538210211043743,
                                  0.019983582316880577,
                                  0.03292085152661295,
                                  0.030445280321387935,
                                  10.0,
                                  0.018398131276209817,
                                  0.01980139702737216,
                                  0.01896441623922851,
                                  0.022574971070788208,
                                  0.01961469275632899,
                                  0.017925070068654958,
                                  0.022549199991725777,
                                  0.042694824314348456,
                                  -1.1040485020844724,
                                  -0.8328856967713545,
                                  0.010486287554451912,
                                  -0.15815014109963288,
                                  0.5276802305511856,
                                  -0.16586473934630663,
                                  -1.6713862869185236,
                                  -0.5685528568072895,
                                  0.015496263249374809,
                                  0.3313776030981479,
                                  0.5841917745193543,
                                  0.9532541813860999,
                                  1.6364826627594853,
                                  -1.4297400383069743,
                                  0.054491944876914444,
                                  0.17913906220820508,
                                  0.44577334447914807,
                                  0.5681719372906395,
                                  3.32951901196007,
                                  0.020829537529045927,
                                  -0.5816250842771562,
                                  -0.45135245825393433,
                                  -0.7025952973955096,
                                  0.4838975315421883,
                                  8.711509549878194,
                                  0.791270929375639,
                                  0.581356024160242,
                                  0.3922969596921355,
                                  -0.7387562349132657,
                                  -1.2614352370812663,
                                  -0.8858225265098523,
                                  0.019758617519097244,
                                  0.14685779225447715,
                                  -1.8664586782764085,
                                  0.17359675572073996,
                                  0.03043962244431553,
                                  1.68973994737285,
                                  0.14241547168885893,
                                  -0.006681229849861969,
                                  0.035286585384673774,
                                  -0.1867844252373887,
                                  -0.05325949297845933,
                                  -0.6188421202762768,
                                  0.019256498690954693,
                                  0.011261488314097567,
                                  -0.8734726170866911,
                                  0.2611408152583321,
                                  0.018228313151663178,
                                  0.9920747746816749,
                                  -1.9939450259663953,
                                  -0.010909816645386039,
                                  0.8744010916528506,
                                  0.3484037761057752,
                                  0.01700115030331162,
                                  0.0200509230497925,
                                  0.9049230270226147,
                                  -0.03502086486907441,
                                  -0.13860433328031307,
                                  -0.5008190311548453,
                                  0.13113827477880657,
                                  3.814948743048878,
                                  3.369522386056538,
                                  0.002328237048341175,
                                  0.9372362537216947,
                                  0.9359420764223362,
                                  -1.0601842054652018,
                                  0.01621181769404767,
                                  -0.4876602083027073,
                                  0.1933464422710256,
                                  -2.10812470816676,
                                  0.018569908820821213};
  size_t i = 0;
  for (size_t c = 0; c < 3; ++c) {
    for (size_t scale = 0; scale < scales.size(); ++scale) {
      for (size_t n = 0; n < 2; n++) {
#ifdef SSIMULACRA2_OUTPUT_RAW_SCORES_FOR_WEIGHT_TUNING
        printf("%.12f,%.12f,%.12f,", scales[scale].avg_ssim[c * 2 + n],
               scales[scale].avg_edgediff[c * 4 + n],
               scales[scale].avg_edgediff[c * 4 + 2 + n]);
#endif
        ssim += weight[i++] * std::abs(scales[scale].avg_ssim[c * 2 + n]);
        ssim += weight[i++] * std::abs(scales[scale].avg_edgediff[c * 4 + n]);
        ssim +=
            weight[i++] * std::abs(scales[scale].avg_edgediff[c * 4 + n + 2]);
      }
    }
  }

  ssim *= 272.58355078216107;
  ssim += -0.12953872587232876;

  if (ssim > 0) {
    ssim = 100.0 - 10.0 * pow(ssim, 0.6940375837127916);
  } else {
    ssim = 100.0;
  }
  return ssim;
}

Msssim ComputeSSIMULACRA2(const jxl::ImageBundle& orig,
                          const jxl::ImageBundle& dist, float bg) {
  Msssim msssim;

  jxl::Image3F img1(orig.xsize(), orig.ysize());
  jxl::Image3F img2(img1.xsize(), img1.ysize());

  if (orig.HasAlpha()) {
    jxl::ImageBundle orig2 = orig.Copy();
    AlphaBlend(orig2, bg);
    jxl::ToXYB(orig2, nullptr, &img1, jxl::GetJxlCms(), nullptr);
  } else {
    jxl::ToXYB(orig, nullptr, &img1, jxl::GetJxlCms(), nullptr);
  }
  if (dist.HasAlpha()) {
    jxl::ImageBundle dist2 = dist.Copy();
    AlphaBlend(dist2, bg);
    jxl::ToXYB(dist2, nullptr, &img2, jxl::GetJxlCms(), nullptr);
  } else {
    jxl::ToXYB(dist, nullptr, &img2, jxl::GetJxlCms(), nullptr);
  }
  MakePositiveXYB(img1);
  MakePositiveXYB(img2);

  Image3F mul(img1.xsize(), img1.ysize());
  Blur blur(img1.xsize(), img1.ysize());

  for (int scale = 0; scale < kNumScales; scale++) {
    if (img1.xsize() < 8 || img1.ysize() < 8) {
      break;
    }
    if (scale) {
      img1 = Downsample(img1, 2, 2);
      img2 = Downsample(img2, 2, 2);
    }
    mul.ShrinkTo(img1.xsize(), img2.ysize());
    blur.ShrinkTo(img1.xsize(), img2.ysize());

    Multiply(img1, img1, &mul);
    Image3F sigma1_sq = blur(mul);

    Multiply(img2, img2, &mul);
    Image3F sigma2_sq = blur(mul);

    Multiply(img1, img2, &mul);
    Image3F sigma12 = blur(mul);

    Image3F mu1 = blur(img1);
    Image3F mu2 = blur(img2);

    MsssimScale sscale;
    SSIMMap(mu1, mu2, sigma1_sq, sigma2_sq, sigma12, sscale.avg_ssim);
    EdgeDiffMap(img1, mu1, img2, mu2, sscale.avg_edgediff);
    msssim.scales.push_back(sscale);
  }
  return msssim;
}

Msssim ComputeSSIMULACRA2(const jxl::ImageBundle& orig,
                          const jxl::ImageBundle& distorted) {
  return ComputeSSIMULACRA2(orig, distorted, 0.5f);
}
