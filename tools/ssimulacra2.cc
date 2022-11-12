// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
SSIMULACRA 2
Structural SIMilarity Unveiling Local And Compression Related Artifacts

Perceptual metric developed by Jon Sneyers (Cloudinary) in July 2022.
Design:
- XYB color space (X+0.5, Y, Y-B+1.0)
- SSIM map (with correction: no double gamma correction)
- 'blockiness/ringing' map (distorted has edges where original is smooth)
- 'smoothing' map (distorted is smooth where original has edges)
- error maps are computed at 6 scales (1:1 to 1:32) for each component (X,Y,B)
- downscaling is done in linear RGB
- for all 6*3*3=54 maps, two norms are computed: 1-norm (mean) and 4-norm
- a weighted sum of these 54*2=108 norms leads to the final score
- weights were tuned based on a large set of subjective scores for images
  compressed with JPEG, JPEG 2000, JPEG XL, WebP, AVIF, and HEIC.
*/

#include "tools/ssimulacra2.h"

#include <stdio.h>

#include <cmath>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/gauss_blur.h"
#include "lib/jxl/image_ops.h"

namespace {

using jxl::Image3F;
using jxl::ImageF;

static const float kC2 = 0.0009f;
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
        float num_m = 1.0 - (mu1 - mu2) * (mu1 - mu2);
        float num_s = 2 * (row_s12[x] - mu12) + kC2;
        float denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + kC2;
        double d = 1.0 - ((num_m * num_s) / (denom_s));
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
    float* JXL_RESTRICT rowY = img.PlaneRow(1, y);
    float* JXL_RESTRICT rowB = img.PlaneRow(2, y);
    float* JXL_RESTRICT rowX = img.PlaneRow(0, y);
    for (size_t x = 0; x < img.xsize(); ++x) {
      rowB[x] += 1.1f - rowY[x];
      rowX[x] += 0.5f;
      rowY[x] += 0.05f;
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
  constexpr double weight[108] = {0.0,
                                  0.0,
                                  0.0,
                                  1.0035479352512353,
                                  0.00011322061110474735,
                                  0.00040442991823685936,
                                  0.0018953834105783773,
                                  0.0,
                                  0.0,
                                  8.982542997575905,
                                  0.9899785796045556,
                                  0.0,
                                  0.9748315131207942,
                                  0.9581575169937973,
                                  0.0,
                                  0.5133611777952946,
                                  1.0423189317331243,
                                  0.000308010928520841,
                                  12.149584966240063,
                                  0.9565577248115467,
                                  0.0,
                                  1.0406668123136824,
                                  81.51139046057362,
                                  0.30593391895330946,
                                  1.0752214433626779,
                                  1.1039042369464611,
                                  0.0,
                                  1.021911638819618,
                                  1.1141823296855722,
                                  0.9730845751441705,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.9833918426095505,
                                  0.7920385137059867,
                                  0.9710740411514053,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.5387077903152638,
                                  0.0,
                                  3.4036945601155804,
                                  0.0,
                                  0.0,
                                  0.0,
                                  2.337569295661117,
                                  0.0,
                                  5.707946510901609,
                                  37.83086423878157,
                                  0.0,
                                  0.0,
                                  3.8258200594305185,
                                  0.0,
                                  0.0,
                                  24.073659674271497,
                                  0.0,
                                  0.0,
                                  13.181871265286068,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  10.00750121262895,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  52.51428385603891,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.9946464267894417,
                                  0.0,
                                  0.0,
                                  0.0006040447715934816,
                                  0.0,
                                  0.0,
                                  0.9945171491374072,
                                  0.0,
                                  2.8260043809454376,
                                  1.0052642766534516,
                                  8.201441997546244e-05,
                                  12.154041855876695,
                                  32.292928706201266,
                                  0.992837130387521,
                                  0.0,
                                  30.71925517844603,
                                  0.00012309907022278743,
                                  0.0,
                                  0.9826260237051734,
                                  0.0,
                                  0.0,
                                  0.9980928367837651,
                                  0.012142430067163312};

  size_t i = 0;
  char ch[] = "XYB";
  const bool verbose = false;
  for (size_t c = 0; c < 3; ++c) {
    for (size_t scale = 0; scale < scales.size(); ++scale) {
      for (size_t n = 0; n < 2; n++) {
#ifdef SSIMULACRA2_OUTPUT_RAW_SCORES_FOR_WEIGHT_TUNING
        printf("%.12f,%.12f,%.12f,", scales[scale].avg_ssim[c * 2 + n],
               scales[scale].avg_edgediff[c * 4 + n],
               scales[scale].avg_edgediff[c * 4 + 2 + n]);
#endif
        if (verbose) {
          printf("%f from channel %c, scale 1:%i, %" PRIuS
                 "-norm (weight %f)\n",
                 weight[i] * std::abs(scales[scale].avg_ssim[c * 2 + n]), ch[c],
                 1 << scale, n * 3 + 1, weight[i]);
        }
        ssim += weight[i++] * std::abs(scales[scale].avg_ssim[c * 2 + n]);
        if (verbose) {
          printf("%f from channel %c ringing, scale 1:%i, %" PRIuS
                 "-norm (weight %f)\n",
                 weight[i] * std::abs(scales[scale].avg_edgediff[c * 4 + n]),
                 ch[c], 1 << scale, n * 3 + 1, weight[i]);
        }
        ssim += weight[i++] * std::abs(scales[scale].avg_edgediff[c * 4 + n]);
        if (verbose) {
          printf(
              "%f from channel %c blur, scale 1:%i, %" PRIuS
              "-norm (weight %f)\n",
              weight[i] * std::abs(scales[scale].avg_edgediff[c * 4 + n + 2]),
              ch[c], 1 << scale, n * 3 + 1, weight[i]);
        }
        ssim +=
            weight[i++] * std::abs(scales[scale].avg_edgediff[c * 4 + n + 2]);
      }
    }
  }

  ssim = ssim * 17.829717797575952 - 1.634169143917183;

  if (ssim > 0) {
    ssim = 100.0 - 10.0 * pow(ssim, 0.5453261009510213);
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

  jxl::ImageBundle orig2 = orig.Copy();
  jxl::ImageBundle dist2 = dist.Copy();

  if (orig.HasAlpha()) AlphaBlend(orig2, bg);
  if (dist.HasAlpha()) AlphaBlend(dist2, bg);
  orig2.ClearExtraChannels();
  dist2.ClearExtraChannels();

  JXL_CHECK(orig2.TransformTo(jxl::ColorEncoding::LinearSRGB(orig2.IsGray()),
                              jxl::GetJxlCms()));
  JXL_CHECK(dist2.TransformTo(jxl::ColorEncoding::LinearSRGB(dist2.IsGray()),
                              jxl::GetJxlCms()));

  jxl::ToXYB(orig2, nullptr, &img1, jxl::GetJxlCms(), nullptr);
  jxl::ToXYB(dist2, nullptr, &img2, jxl::GetJxlCms(), nullptr);
  MakePositiveXYB(img1);
  MakePositiveXYB(img2);

  Image3F mul(img1.xsize(), img1.ysize());
  Blur blur(img1.xsize(), img1.ysize());

  for (int scale = 0; scale < kNumScales; scale++) {
    if (img1.xsize() < 8 || img1.ysize() < 8) {
      break;
    }
    if (scale) {
      orig2.SetFromImage(Downsample(*orig2.color(), 2, 2),
                         jxl::ColorEncoding::LinearSRGB(orig2.IsGray()));
      dist2.SetFromImage(Downsample(*dist2.color(), 2, 2),
                         jxl::ColorEncoding::LinearSRGB(dist2.IsGray()));
      img1.ShrinkTo(orig2.xsize(), orig2.ysize());
      img2.ShrinkTo(orig2.xsize(), orig2.ysize());
      jxl::ToXYB(orig2, nullptr, &img1, jxl::GetJxlCms(), nullptr);
      jxl::ToXYB(dist2, nullptr, &img2, jxl::GetJxlCms(), nullptr);
      MakePositiveXYB(img1);
      MakePositiveXYB(img2);
    }
    mul.ShrinkTo(img1.xsize(), img1.ysize());
    blur.ShrinkTo(img1.xsize(), img1.ysize());

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
