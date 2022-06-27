// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Multi-Scale SSIM in XYB color space

#include "tools/xybmsssim.h"

#include <stdio.h>

#include <cmath>

#include "lib/extras/codec.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/gauss_blur.h"
#include "lib/jxl/image_ops.h"

namespace xybmsssim {
namespace {

using jxl::Image3F;
using jxl::ImageF;

static const float kC1 = 0.0001f;
static const float kC2 = 0.0004f;
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

void SSIMMap(const Image3F& m1, const Image3F& m2, const Image3F& s11,
             const Image3F& s22, const Image3F& s12, Image3F* out,
             double* plane_averages) {
  const double onePerPixels = 1.0 / (out->ysize() * out->xsize());
  for (size_t c = 0; c < 3; ++c) {
    double sum1[4] = {0.0};
    for (size_t y = 0; y < out->ysize(); ++y) {
      const float* JXL_RESTRICT row_m1 = m1.PlaneRow(c, y);
      const float* JXL_RESTRICT row_m2 = m2.PlaneRow(c, y);
      const float* JXL_RESTRICT row_s11 = s11.PlaneRow(c, y);
      const float* JXL_RESTRICT row_s22 = s22.PlaneRow(c, y);
      const float* JXL_RESTRICT row_s12 = s12.PlaneRow(c, y);
      float* JXL_RESTRICT row_out = out->PlaneRow(c, y);
      for (size_t x = 0; x < out->xsize(); ++x) {
        float mu1 = row_m1[x];
        float mu2 = row_m2[x];
        float mu11 = mu1 * mu1;
        float mu22 = mu2 * mu2;
        float mu12 = mu1 * mu2;
        float nom_m = 2 * mu12 + kC1;
        float nom_s = 2 * (row_s12[x] - mu12) + kC2;
        float denom_m = mu11 + mu22 + kC1;
        float denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + kC2;
        row_out[x] = 1.f - ((nom_m * nom_s) / (denom_m * denom_s));
        double d2 = row_out[x];
        sum1[0] += d2;
        d2 *= d2;
        sum1[1] += d2;
        d2 *= d2;
        sum1[2] += d2;
        d2 *= d2;
        sum1[3] += d2;
      }
    }
    for (int i = 0; i < 4; ++i) {
      double e = pow(onePerPixels * sum1[i], 1.0 / (1.0 * (1 << i)));
      plane_averages[c * 4 + i] = e;
    }
  }
}

}  // namespace

double Msssim::Score() const {
  double ssim = 0.0;
  const double weight[72 * 2 + 2] = {
      0.021973116402930787,  -11.523616782747345,    -147.37059822708056,
      -1.6143866083529559,   -39.003437698173244,    -31.77851289789551,
      -49.9672868154597,     62.78586191436433,      -11.053513167403734,
      -25.25525534752599,    -11.415621837152546,    -23.38918350888089,
      718.4310147320011,     -1023.9555732634483,    13.575607877577777,
      141.50928115232477,    -16.788153923948656,    -11.14813741602568,
      -16.743372236131428,   -11.11197562810807,     -19.316636306695123,
      -20.138778296356875,   -19.477270281874418,    7.614588849564161,
      0.07063913261015303,   227.040974959351,       313.52827186664183,
      54.37789467869065,     322.4617307612118,      -11.04667377598248,
      -20.540074595777423,   -29.548527190845817,    -0.9328007883120797,
      -28.272946227881985,   16.672112311471277,     -11.526153821254828,
      0.0009536336122358943, -0.0024173620930892925, 0.00276134245313746,
      -19.675262442378354,   -0.33481507050708403,   -28.164158014477607,
      -22.653668830863715,   -18.153649722461513,    -16.415731256266568,
      35.61593605404771,     -32.39618248697417,     -241.4545530623837,
      -17.340979714640063,   46381.62732487409,      -17.629683220501953,
      13080.751179399924,    -0.18398984292625342,   -20.732735069582514,
      -21.019782851043495,   -10.563109909082382,    -0.030149429486623774,
      -26.798577154623352,   -36.8686653964214,      -15.645312144949866,
      -30.865582183753954,   -0.004469813254280001,  -0.016411462368678867,
      222311.65551417434,    0.01898423014962192,    -30.503711598127694,
      -0.14054091135601224,  23.851160727703203,     0.03056930762289608,
      -16.818967502698584,   -27.16862823681399,     -19.621844143993187,
      16.5611193910485,      -1.116156823452251,     -0.677777124861732,
      -1.2492563529750185,   9.677625317807392,      7.1531846932376375,
      27.54129929386308,     -0.2633102031643941,    -1.3272962342215187,
      51.000093303695444,    -1.1102844310813411,    11.913941162288666,
      17.494045423192453,    -0.7914956525753191,    24.446457083956567,
      -0.44272175599737584,  0.16384944378708885,    -1.1211940846283721,
      8.132191144243372,     -1.6563024399769932,    -0.9166303361655113,
      -0.8103546757732437,   -0.808317399866697,     -0.9795764111692445,
      12.173967812437901,    25.044499271495702,     13.101551006910066,
      -1.1463247029072159,   -0.4222594330259958,    -1.132327957487064,
      9.363842191435552,     38.83335743612862,      -1.0998677284469365,
      9.561955730070371,     -0.8577540499265555,    10.164679872122045,
      5.174164800803601,     6.980151957954371,      4.564248527942422,
      -0.6468187230353699,   10.614172148130983,     9.031116407927799,
      18.089446230343704,    -0.7325871691753574,    18.634836028802297,
      -1.1128013178866873,   26.168040233780744,     7.181927080200893,
      -0.7596683767842739,   6.249875147821439,      -1.15666833927224,
      8.594112201346285,     6.411467686728347,      -1.5457033203228143,
      -1.1859673880665245,   -1.1153019963530784,    8.267647945954227,
      18.25883548652105,     7.201468614200209,      10.006791677877976,
      -1.0073087931737255,   6.048253005305727,      5.27839167059301,
      9.086796676752908,     13.262383047279133,     -1.1512776806463452,
      8.558057521289651,     -1.0737872485390612,    8.022981622712607,
      16.08638451276799,     13.65961727400643,      15.093366785654075,
      -0.06491732157830722,  0.3337168175410301};

  for (size_t c = 0; c < 3; ++c) {
    for (size_t scale = 0; scale < scales.size(); ++scale) {
      for (size_t n = 0; n < 4; n++) {
        size_t idx = scale * 12 + c * 4 + n;
        ssim += std::max(-10.0, 1.0 + weight[idx]) *
                pow(std::abs(scales[scale].avg_ssim[c * 4 + n]),
                    std::max(0.125, std::min(weight[72 + idx] + 1.0, 8.0)));
      }
    }
  }
  ssim = 100.0 - (1.0 + weight[145]) * pow(std::abs(ssim), 1.0 + weight[144]);

  //  if (ssim < 0) ssim = 0;
  //  if (ssim > 100) ssim = 100;
  return ssim;
}

Msssim ComputeMSSSIM(jxl::Image3F& img1, jxl::Image3F& img2) {
  Msssim msssim;

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
    // Reuse mul as "ssim_map".
    MsssimScale sscale;
    SSIMMap(mu1, mu2, sigma1_sq, sigma2_sq, sigma12, &mul, sscale.avg_ssim);
    msssim.scales.push_back(sscale);
  }
  return msssim;
}

namespace {

int PrintUsage(char** argv) {
  fprintf(stderr, "Usage: %s orig.png distorted.png\n", argv[0]);
  return 1;
}

int Run(int argc, char** argv) {
  if (argc != 3) return PrintUsage(argv);

  jxl::CodecInOut io1;
  jxl::CodecInOut io2;
  JXL_CHECK(SetFromFile(argv[1], jxl::extras::ColorHints(), &io1));
  JXL_CHECK(SetFromFile(argv[2], jxl::extras::ColorHints(), &io2));
  JXL_CHECK(io1.TransformTo(jxl::ColorEncoding::LinearSRGB(io1.Main().IsGray()),
                            jxl::GetJxlCms()));
  JXL_CHECK(io2.TransformTo(jxl::ColorEncoding::LinearSRGB(io2.Main().IsGray()),
                            jxl::GetJxlCms()));

  if (io1.xsize() != io2.xsize() || io1.ysize() != io2.ysize()) {
    fprintf(stderr, "Image size mismatch\n");
    return 1;
  }
  if (io1.xsize() < 8 || io1.ysize() < 8) {
    fprintf(stderr, "Minimum image size is 8x8 pixels\n");
    return 1;
  }
  jxl::Image3F orig(io1.xsize(), io1.ysize());
  jxl::ToXYB(io1.Main(), nullptr, &orig, jxl::GetJxlCms(), nullptr);
  jxl::Image3F dist(io2.xsize(), io2.ysize());
  jxl::ToXYB(io2.Main(), nullptr, &dist, jxl::GetJxlCms(), nullptr);

  Msssim msssim = ComputeMSSSIM(orig, dist);
  printf("%.8f\n", msssim.Score());
  return 0;
}

}  // namespace
}  // namespace xybmsssim

int main(int argc, char** argv) { return xybmsssim::Run(argc, argv); }
