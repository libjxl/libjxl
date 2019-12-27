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

// Re-implementation of //third_party/ssimulacra/ssimulacra.cpp using jxl's
// ImageF library instead of opencv.

#include "tools/ssimulacra.h"

#include "jxl/image_ops.h"

namespace ssimulacra {
namespace {

using jxl::Image3F;
using jxl::ImageF;

static const double kC1 = 0.0001;
static const double kC2 = 0.0004;
static const int kNumScales = 6;
static const double kScaleWeights[kNumScales][3] = {
    {0.04480, 0.00300, 0.00300}, {0.28560, 0.00896, 0.00896},
    {0.30010, 0.05712, 0.05712}, {0.23630, 0.06002, 0.06002},
    {0.13330, 0.06726, 0.06726}, {0.10000, 0.05000, 0.05000},
};
const double kMinScaleWeights[kNumScales][3] = {
    {0.02000, 0.00005, 0.00005}, {0.03000, 0.00025, 0.00025},
    {0.02500, 0.00100, 0.00100}, {0.02000, 0.00150, 0.00150},
    {0.01200, 0.00175, 0.00175}, {0.00500, 0.00175, 0.00175},
};
const double kEdgeWeight[3] = {1.5, 0.1, 0.1};
const double kGridWeight[3] = {1.0, 0.1, 0.1};

inline void Rgb2Lab(float r, float g, float b, float* L, float* A, float* B) {
  const float epsilon = 0.00885645167903563081f;
  const float s = 0.13793103448275862068f;
  const float k = 7.78703703703703703703f;
  const float scale = 1.0f / 255.0f;
  r *= scale;
  g *= scale;
  b *= scale;
  float fx = (r * 0.43393624408206207259f + g * 0.37619779063650710152f +
              b * 0.18983429773803261441f);
  float fy = (r * 0.2126729f + g * 0.7151522f + b * 0.0721750f);
  float fz = (r * 0.01775381083562901744f + g * 0.10945087235996326905f +
              b * 0.87263921028466483011f);
  const float gamma = 1.0f / 3.0f;
  float X = (fx > epsilon) ? powf(fx, gamma) - s : k * fx;
  float Y = (fy > epsilon) ? powf(fy, gamma) - s : k * fy;
  float Z = (fz > epsilon) ? powf(fz, gamma) - s : k * fz;
  *L = Y * 1.16f;
  *A = (0.39181818181818181818f + 2.27272727272727272727f * (X - Y));
  *B = (0.49045454545454545454f + 0.90909090909090909090f * (Y - Z));
}

Image3F Rgb2Lab(const Image3F& in) {
  Image3F out(in.xsize(), in.ysize());
  for (int y = 0; y < in.ysize(); ++y) {
    const float* row_in[3];
    float* row_out[3];
    for (int c = 0; c < 3; c++) {
      row_in[c] = in.PlaneRow(c, y);
      row_out[c] = out.PlaneRow(c, y);
    }
    for (int x = 0; x < in.xsize(); ++x) {
      Rgb2Lab(row_in[0][x], row_in[1][x], row_in[2][x], &row_out[0][x],
              &row_out[1][x], &row_out[2][x]);
    }
  }
  return out;
}

Image3F Downsample(const Image3F& in, int fx, int fy) {
  int out_xsize = (in.xsize() + fx - 1) / fx;
  int out_ysize = (in.ysize() + fy - 1) / fy;
  Image3F out(out_xsize, out_ysize);
  for (int oy = 0; oy < out_ysize; ++oy) {
    for (int ox = 0; ox < out_xsize; ++ox) {
      for (int c = 0; c < 3; ++c) {
        float sum = 0.0;
        for (int iy = 0; iy < fy; ++iy) {
          for (int ix = 0; ix < fx; ++ix) {
            int x = std::min<int>(ox * fx + ix, in.xsize() - 1);
            int y = std::min<int>(oy * fy + iy, in.ysize() - 1);
            sum += in.PlaneRow(c, y)[x];
          }
        }
        sum /= (fx * fy);
        out.PlaneRow(c, oy)[ox] = sum;
      }
    }
  }
  return out;
}

Image3F Multiply(const Image3F& a, const Image3F& b) {
  Image3F out(a.xsize(), a.ysize());
  for (int y = 0; y < a.ysize(); ++y) {
    for (int x = 0; x < a.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        out.PlaneRow(c, y)[x] = a.PlaneRow(c, y)[x] * b.PlaneRow(c, y)[x];
      }
    }
  }
  return out;
}

void RowColAvgP2(const ImageF& in, double* rp2, double* cp2) {
  std::vector<double> ravg(in.ysize());
  std::vector<double> cavg(in.xsize());
  for (int y = 0; y < in.ysize(); ++y) {
    auto row = in.Row(y);
    for (int x = 0; x < in.xsize(); ++x) {
      const float val = row[x];
      ravg[y] += val;
      cavg[x] += val;
    }
  }
  std::sort(ravg.begin(), ravg.end());
  std::sort(cavg.begin(), cavg.end());
  *rp2 = ravg[ravg.size() / 50] / in.xsize();
  *cp2 = cavg[cavg.size() / 50] / in.ysize();
}

Image3F EdgeDiffMap(const Image3F& img1, const Image3F& mu1,
                    const Image3F& img2, const Image3F& mu2) {
  Image3F out(img1.xsize(), img1.ysize());
  for (int y = 0; y < img1.ysize(); ++y) {
    for (int c = 0; c < 3; ++c) {
      auto row1 = img1.PlaneRow(c, y);
      auto row2 = img2.PlaneRow(c, y);
      auto rowm1 = mu1.PlaneRow(c, y);
      auto rowm2 = mu2.PlaneRow(c, y);
      for (int x = 0; x < img1.xsize(); ++x) {
        float edgediff = std::max(
            std::abs(row2[x] - rowm2[x]) - std::abs(row1[x] - rowm1[x]), 0.0f);
        out.PlaneRow(c, y)[x] = 1.0f - edgediff;
      }
    }
  }
  return out;
}

std::vector<float> GaussianKernel(int radius, float sigma) {
  const float scaler = -1.0 / (2 * sigma * sigma);
  std::vector<float> kernel(2 * radius + 1);
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(scaler * i * i);
  }
  return kernel;
}

inline void ExtrapolateBorders(const float* const JXL_RESTRICT row_in,
                               float* const JXL_RESTRICT row_out,
                               const int xsize, const int radius) {
  const int lastcol = xsize - 1;
  for (int x = 1; x <= radius; ++x) {
    row_out[-x] = row_in[std::min(x, xsize - 1)];
  }
  memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
  for (int x = 1; x <= radius; ++x) {
    row_out[lastcol + x] = row_in[std::max(0, lastcol - x)];
  }
}

ImageF MirrorConvolveXTranspose(const ImageF& in,
                                const std::vector<float>& kernel) {
  JXL_CHECK(kernel.size() % 2 == 1);
  ImageF out(in.ysize(), in.xsize());
  float weight = 0.0f;
  for (int i = 0; i < kernel.size(); ++i) {
    weight += kernel[i];
  }
  float scale = 1.0f / weight;
  const int r = kernel.size() / 2;
  std::vector<float> row_tmp(in.xsize() + 2 * r);
  float* const JXL_RESTRICT rowp = &row_tmp[r];
  const float* const kernelp = &kernel[r];
  for (int y = 0; y < in.ysize(); ++y) {
    ExtrapolateBorders(in.Row(y), rowp, in.xsize(), r);
    for (int x = 0; x < in.xsize(); ++x) {
      float sum = 0.0f;
      for (int i = -r; i <= r; ++i) {
        sum += rowp[x + i] * kernelp[i];
      }
      out.Row(x)[y] = sum * scale;
    }
  }
  return out;
}

ImageF GaussianBlur(const ImageF& in, int radius, float sigma) {
  std::vector<float> kernel = GaussianKernel(radius, sigma);
  return MirrorConvolveXTranspose(MirrorConvolveXTranspose(in, kernel), kernel);
}

Image3F GaussianBlur(const Image3F& in, int radius, float sigma) {
  return Image3F(GaussianBlur(in.Plane(0), radius, sigma),
                 GaussianBlur(in.Plane(1), radius, sigma),
                 GaussianBlur(in.Plane(2), radius, sigma));
}

Image3F SSIMMap(const Image3F& m1, const Image3F& m2, const Image3F& s11,
                const Image3F& s22, const Image3F& s12) {
  Image3F out(m1.xsize(), m1.ysize());
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < out.ysize(); ++y) {
      auto row_m1 = m1.PlaneRow(c, y);
      auto row_m2 = m2.PlaneRow(c, y);
      auto row_s11 = s11.PlaneRow(c, y);
      auto row_s22 = s22.PlaneRow(c, y);
      auto row_s12 = s12.PlaneRow(c, y);
      auto row_out = out.PlaneRow(c, y);
      for (int x = 0; x < out.xsize(); ++x) {
        float mu1 = row_m1[x];
        float mu2 = row_m2[x];
        float mu11 = mu1 * mu1;
        float mu22 = mu2 * mu2;
        float mu12 = mu1 * mu2;
        float nom_m = 2 * mu12 + kC1;
        float nom_s = 2 * (row_s12[x] - mu12) + kC2;
        float denom_m = mu11 + mu22 + kC1;
        float denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + kC2;
        row_out[x] = (nom_m * nom_s) / (denom_m * denom_s);
      }
    }
  }
  return out;
}

}  // namespace

double Ssimulacra::Score() const {
  double ssim = 0.0;
  double ssim_max = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int scale = 0; scale < scales.size(); ++scale) {
      ssim += kScaleWeights[scale][c] * scales[scale].avg_ssim[c];
      ssim_max += kScaleWeights[scale][c];
      ssim += kMinScaleWeights[scale][c] * scales[scale].min_ssim[c];
      ssim_max += kMinScaleWeights[scale][c];
    }
    ssim += kEdgeWeight[c] * avg_edgediff[c];
    ssim_max += kEdgeWeight[c];
    ssim += kGridWeight[c] *
            (row_p2[0][c] + row_p2[1][c] + col_p2[0][c] + col_p2[1][c]);
    ssim_max += 4.0 * kGridWeight[c];
  }
  double dssim = ssim_max / ssim - 1.0;
  return std::min(1.0, std::max(0.0, dssim));
}

inline void PrintItem(const char* name, int scale, const double* vals,
                      const double* w) {
  printf("scale %d %s = [%.10f %.10f %.10f]  w = [%.5f %.5f %.5f]\n", scale,
         name, vals[0], vals[1], vals[2], w[0], w[1], w[2]);
}

void Ssimulacra::PrintDetails() const {
  for (int s = 0; s < scales.size(); ++s) {
    PrintItem("avg ssim", s, scales[s].avg_ssim, &kScaleWeights[s][0]);
    PrintItem("min ssim", s, scales[s].min_ssim, &kMinScaleWeights[s][0]);
    if (s == 0) {
      PrintItem("avg edif", s, avg_edgediff, kEdgeWeight);
      PrintItem("rp2 ssim", s, &row_p2[0][0], kGridWeight);
      PrintItem("cp2 ssim", s, &col_p2[0][0], kGridWeight);
      PrintItem("rp2 edif", s, &row_p2[1][0], kGridWeight);
      PrintItem("cp2 edif", s, &col_p2[1][0], kGridWeight);
    }
  }
}

Ssimulacra ComputeDiff(const Image3F& img1_arg, const Image3F& img2_arg) {
  Ssimulacra ssimulacra;
  Image3F img1 = Rgb2Lab(img1_arg);
  Image3F img2 = Rgb2Lab(img2_arg);
  for (int scale = 0; scale < kNumScales; scale++) {
    if (img1.xsize() < 8 || img1.ysize() < 8) {
      break;
    }
    if (scale) {
      img1 = Downsample(img1, 2, 2);
      img2 = Downsample(img2, 2, 2);
    }
    SsimulacraScale sscale;
    Image3F img1_sq = Multiply(img1, img1);
    Image3F img2_sq = Multiply(img2, img2);
    Image3F img1_img2 = Multiply(img1, img2);
    Image3F mu1 = GaussianBlur(img1, 5, 1.5f);
    Image3F mu2 = GaussianBlur(img2, 5, 1.5f);
    Image3F sigma1_sq = GaussianBlur(img1_sq, 5, 1.5f);
    Image3F sigma2_sq = GaussianBlur(img2_sq, 5, 1.5f);
    Image3F sigma12 = GaussianBlur(img1_img2, 5, 1.5f);
    Image3F ssim_map = SSIMMap(mu1, mu2, sigma1_sq, sigma2_sq, sigma12);

    for (unsigned int i = 0; i < 3; i++) {
      sscale.avg_ssim[i] = jxl::ImageAverage(ssim_map.Plane(i));
    }
    ssim_map = Downsample(ssim_map, 4, 4);
    for (unsigned int c = 0; c < 3; c++) {
      float minval, maxval;
      ImageMinMax(ssim_map.Plane(c), &minval, &maxval);
      sscale.min_ssim[c] = minval;
    }
    ssimulacra.scales.push_back(sscale);
    if (scale == 0) {
      Image3F edgediff = EdgeDiffMap(img1, mu1, img2, mu2);
      for (unsigned int c = 0; c < 3; c++) {
        ssimulacra.avg_edgediff[c] = jxl::ImageAverage(edgediff.Plane(c));
        RowColAvgP2(ssim_map.Plane(c), &ssimulacra.row_p2[0][c],
                    &ssimulacra.col_p2[0][c]);
        RowColAvgP2(edgediff.Plane(c), &ssimulacra.row_p2[1][c],
                    &ssimulacra.col_p2[1][c]);
      }
    }
  }
  return ssimulacra;
}

}  // namespace ssimulacra
