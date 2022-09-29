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
  constexpr double weight[108] = {
      4.219667647997749e-05,  0.012686211358327482,   3.107147477665606e-05,
      0.0005435962381676873,  0.09395129733837515,    0.00023116489501884274,
      3.1161782753752476e-05, 0.03927085987454604,    3.112320351661424e-05,
      15.207946778270552,     0.11685373060645432,    0.10825883042600981,
      3.116785767387498e-05,  3.131457976301988e-05,  3.114664519432431e-05,
      3.111881734196853e-05,  0.1505260790864622,     1.181932253296347,
      0.023779401135092804,   3.118721767259025e-05,  3.107147477665606e-05,
      3.107147477665606e-05,  0.29263071159729126,    100.0,
      0.07835191903023642,    0.308749239640701,      3.110101392123088e-05,
      0.03313472929067718,    1.2615585738398967,     1.2865041534163861,
      0.0005007158018729418,  3.114135552706454e-05,  3.107147477665606e-05,
      0.0996219886985672,     0.07444482577438882,    0.11372427084611647,
      5.518066533005683e-05,  3.135558661193638e-05,  3.1165492501039616e-05,
      0.34750942964683273,    3.4565270945252635e-05, 4.0885439725990835,
      3.401042790207587e-05,  3.107147477665606e-05,  3.1316775810030784e-05,
      0.00353728778695106,    0.00028891881745896075, 13.56776514419144,
      28.427922207790395,     4.698319951601526e-05,  3.1247764029185277e-05,
      0.1304924308955202,     2.8128347927967736,     7.902846378027295e-05,
      1.3106634271023248,     0.00021573043084699428, 0.00013016160297185664,
      3.4061442495967658,     4.460412915533889,      3.107147477665606e-05,
      3.2773610579184265e-05, 0.10369457277204852,    3.629363118118345e-05,
      0.0008483509905105047,  1.1933830424964742,     3.342669917216767e-05,
      3.1129364631232725e-05, 3.111597216765016e-05,  0.002772786993656906,
      5.50680530699843e-05,   3.107147477665606e-05,  3.113120547104664e-05,
      3.109181778038206e-05,  3.107147477665606e-05,  3.111874829531125e-05,
      3.271770143775665e-05,  0.0001592648376030903,  7.958992275525212e-05,
      3.2765921379684926e-05, 3.11977840244948e-05,   3.11737542622037e-05,
      3.2698540317954716e-05, 0.0002066952296724267,  8.39634553865265e-05,
      3.4445126357751654e-05, 4.973593015122901e-05,  3.108593217115985e-05,
      7.448916645891313e-05,  0.0006505495770876557,  4.3423265674080724e-05,
      7.247563231427279e-05,  0.00021223544764059632, 3.11772963338397e-05,
      0.4067536289734678,     0.13898049837088255,    4.54117813611484,
      0.06853491105140475,    0.15581252655659317,    0.09982664921024764,
      3.440168932652795,      0.12829653103408623,    56.59930986733967,
      5.773410728426853e-05,  0.1067440463539433,     3.108444898647367e-05,
      3.374827724533791e-05,  0.020250432987237055,   0.1334684230723412};

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

  ssim = ssim * 11.480665013024748 - 1.0204610491040174;

  if (ssim > 0) {
    ssim = 100.0 - 10.0 * pow(ssim, 0.6402032009298979);
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
