// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Re-implementation of //tools/ssimulacra.tct using jxl's
// ImageF library instead of opencv.

#include "tools/ssimulacra.h"

#include <jxl/memory_manager.h>

#include <cmath>

#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "tools/gauss_blur.h"
#include "tools/no_memory_manager.h"

namespace ssimulacra {
namespace {

using ::jxl::Image3F;
using ::jxl::ImageF;
using ::jxl::Status;
using ::jxl::StatusOr;

const float kC1 = 0.0001f;
const float kC2 = 0.0004f;
const int kNumScales = 6;
// Premultiplied by chroma weight 0.2
const double kScaleWeights[kNumScales][3] = {
    {0.04480, 0.00300, 0.00300}, {0.28560, 0.00896, 0.00896},
    {0.30010, 0.05712, 0.05712}, {0.23630, 0.06002, 0.06002},
    {0.13330, 0.06726, 0.06726}, {0.10000, 0.05000, 0.05000},
};
// Premultiplied by min weights 0.1, 0.005, 0.005
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

StatusOr<Image3F> Rgb2Lab(const Image3F& in) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  JXL_ASSIGN_OR_RETURN(Image3F out,
                       Image3F::Create(memory_manager, in.xsize(), in.ysize()));
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* JXL_RESTRICT row_in0 = in.PlaneRow(0, y);
    const float* JXL_RESTRICT row_in1 = in.PlaneRow(1, y);
    const float* JXL_RESTRICT row_in2 = in.PlaneRow(2, y);
    float* JXL_RESTRICT row_out0 = out.PlaneRow(0, y);
    float* JXL_RESTRICT row_out1 = out.PlaneRow(1, y);
    float* JXL_RESTRICT row_out2 = out.PlaneRow(2, y);

    for (size_t x = 0; x < in.xsize(); ++x) {
      Rgb2Lab(row_in0[x], row_in1[x], row_in2[x], &row_out0[x], &row_out1[x],
              &row_out2[x]);
    }
  }
  return out;
}

StatusOr<Image3F> Downsample(const Image3F& in, size_t fx, size_t fy) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  const size_t out_xsize = (in.xsize() + fx - 1) / fx;
  const size_t out_ysize = (in.ysize() + fy - 1) / fy;
  JXL_ASSIGN_OR_RETURN(Image3F out,
                       Image3F::Create(memory_manager, out_xsize, out_ysize));
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

void RowColAvgP2(const ImageF& in, double* rp2, double* cp2) {
  std::vector<double> ravg(in.ysize());
  std::vector<double> cavg(in.xsize());
  for (size_t y = 0; y < in.ysize(); ++y) {
    const auto* row = in.Row(y);
    for (size_t x = 0; x < in.xsize(); ++x) {
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

class StreamingAverage {
 public:
  void Add(const float v) {
    // Numerically stable method.
    double delta = v - result_;
    n_ += 1;
    result_ += delta / n_;
  }

  double Get() const { return result_; }

 private:
  double result_ = 0.0;
  size_t n_ = 0;
};

void EdgeDiffMap(const Image3F& img1, const Image3F& mu1, const Image3F& img2,
                 const Image3F& mu2, Image3F* out, double* plane_avg) {
  for (size_t c = 0; c < 3; ++c) {
    StreamingAverage avg;
    for (size_t y = 0; y < img1.ysize(); ++y) {
      const float* JXL_RESTRICT row1 = img1.PlaneRow(c, y);
      const float* JXL_RESTRICT row2 = img2.PlaneRow(c, y);
      const float* JXL_RESTRICT rowm1 = mu1.PlaneRow(c, y);
      const float* JXL_RESTRICT rowm2 = mu2.PlaneRow(c, y);
      float* JXL_RESTRICT row_out = out->PlaneRow(c, y);
      for (size_t x = 0; x < img1.xsize(); ++x) {
        float edgediff = std::max(
            std::abs(row2[x] - rowm2[x]) - std::abs(row1[x] - rowm1[x]), 0.0f);
        row_out[x] = 1.0f - edgediff;
        avg.Add(row_out[x]);
      }
    }
    plane_avg[c] = avg.Get();
  }
}

// Temporary storage for Gaussian blur, reused for multiple images.
class Blur {
 public:
  static StatusOr<Blur> Create(const size_t xsize, const size_t ysize) {
    JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
    Blur result;
    JXL_ASSIGN_OR_RETURN(result.temp_,
                         ImageF::Create(memory_manager, xsize, ysize));
    return result;
  }

  Status BlurPlane(const ImageF& in, ImageF* JXL_RESTRICT out) {
    JXL_RETURN_IF_ERROR(FastGaussian(
        rg_, in.xsize(), in.ysize(), [&](size_t y) { return in.ConstRow(y); },
        [&](size_t y) { return temp_.Row(y); },
        [&](size_t y) { return out->Row(y); }));
    return true;
  }

  StatusOr<Image3F> operator()(const Image3F& in) {
    JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
    JXL_ASSIGN_OR_RETURN(
        Image3F out, Image3F::Create(memory_manager, in.xsize(), in.ysize()));
    JXL_RETURN_IF_ERROR(BlurPlane(in.Plane(0), &out.Plane(0)));
    JXL_RETURN_IF_ERROR(BlurPlane(in.Plane(1), &out.Plane(1)));
    JXL_RETURN_IF_ERROR(BlurPlane(in.Plane(2), &out.Plane(2)));
    return out;
  }

  // Allows reusing across scales.
  Status ShrinkTo(const size_t xsize, const size_t ysize) {
    return temp_.ShrinkTo(xsize, ysize);
  }

 private:
  Blur() : rg_(jxl::CreateRecursiveGaussian(1.5)) {}
  jxl::RecursiveGaussian rg_;
  ImageF temp_;
};

void SSIMMap(const Image3F& m1, const Image3F& m2, const Image3F& s11,
             const Image3F& s22, const Image3F& s12, Image3F* out,
             double* plane_averages) {
  for (size_t c = 0; c < 3; ++c) {
    StreamingAverage avg;
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
        row_out[x] = (nom_m * nom_s) / (denom_m * denom_s);
        avg.Add(row_out[x]);
      }
    }
    plane_averages[c] = avg.Get();
  }
}

}  // namespace

double Ssimulacra::Score() const {
  double ssim = 0.0;
  double ssim_max = 0.0;
  for (size_t c = 0; c < 3; ++c) {
    for (size_t scale = 0; scale < scales.size(); ++scale) {
      ssim += kScaleWeights[scale][c] * scales[scale].avg_ssim[c];
      ssim_max += kScaleWeights[scale][c];
      ssim += kMinScaleWeights[scale][c] * scales[scale].min_ssim[c];
      ssim_max += kMinScaleWeights[scale][c];
    }
    if (!simple) {
      ssim += kEdgeWeight[c] * avg_edgediff[c];
      ssim_max += kEdgeWeight[c];
      ssim += kGridWeight[c] *
              (row_p2[0][c] + row_p2[1][c] + col_p2[0][c] + col_p2[1][c]);
      ssim_max += 4.0 * kGridWeight[c];
    }
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
  for (size_t s = 0; s < scales.size(); ++s) {
    if (s < kNumScales) {
      PrintItem("avg ssim", s, scales[s].avg_ssim, kScaleWeights[s]);
      PrintItem("min ssim", s, scales[s].min_ssim, kMinScaleWeights[s]);
    }
    if (s == 0 && !simple) {
      PrintItem("avg edif", s, avg_edgediff, kEdgeWeight);
      PrintItem("rp2 ssim", s, &row_p2[0][0], kGridWeight);
      PrintItem("cp2 ssim", s, &col_p2[0][0], kGridWeight);
      PrintItem("rp2 edif", s, &row_p2[1][0], kGridWeight);
      PrintItem("cp2 edif", s, &col_p2[1][0], kGridWeight);
    }
  }
}

StatusOr<Ssimulacra> ComputeDiff(const Image3F& orig, const Image3F& distorted,
                                 bool simple) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  Ssimulacra ssimulacra;

  ssimulacra.simple = simple;
  JXL_ASSIGN_OR_RETURN(Image3F img1, Rgb2Lab(orig));
  JXL_ASSIGN_OR_RETURN(Image3F img2, Rgb2Lab(distorted));

  JXL_ASSIGN_OR_RETURN(
      Image3F mul, Image3F::Create(memory_manager, orig.xsize(), orig.ysize()));
  JXL_ASSIGN_OR_RETURN(Blur blur, Blur::Create(img1.xsize(), img1.ysize()));

  for (int scale = 0; scale < kNumScales; scale++) {
    if (img1.xsize() < 8 || img1.ysize() < 8) {
      break;
    }
    if (scale) {
      JXL_ASSIGN_OR_RETURN(img1, Downsample(img1, 2, 2));
      JXL_ASSIGN_OR_RETURN(img2, Downsample(img2, 2, 2));
    }
    JXL_RETURN_IF_ERROR(mul.ShrinkTo(img1.xsize(), img2.ysize()));
    JXL_RETURN_IF_ERROR(blur.ShrinkTo(img1.xsize(), img2.ysize()));

    Multiply(img1, img1, &mul);
    JXL_ASSIGN_OR_RETURN(Image3F sigma1_sq, blur(mul));

    Multiply(img2, img2, &mul);
    JXL_ASSIGN_OR_RETURN(Image3F sigma2_sq, blur(mul));

    Multiply(img1, img2, &mul);
    JXL_ASSIGN_OR_RETURN(Image3F sigma12, blur(mul));

    JXL_ASSIGN_OR_RETURN(Image3F mu1, blur(img1));
    JXL_ASSIGN_OR_RETURN(Image3F mu2, blur(img2));
    // Reuse mul as "ssim_map".
    SsimulacraScale sscale;
    SSIMMap(mu1, mu2, sigma1_sq, sigma2_sq, sigma12, &mul, sscale.avg_ssim);

    JXL_ASSIGN_OR_RETURN(const Image3F ssim_map, Downsample(mul, 4, 4));
    for (size_t c = 0; c < 3; c++) {
      float minval;
      float maxval;
      ImageMinMax(ssim_map.Plane(c), &minval, &maxval);
      sscale.min_ssim[c] = static_cast<double>(minval);
    }
    ssimulacra.scales.push_back(sscale);

    if (scale == 0 && !simple) {
      Image3F* edgediff = &sigma1_sq;  // reuse
      EdgeDiffMap(img1, mu1, img2, mu2, edgediff, ssimulacra.avg_edgediff);
      for (size_t c = 0; c < 3; c++) {
        RowColAvgP2(ssim_map.Plane(c), &ssimulacra.row_p2[0][c],
                    &ssimulacra.col_p2[0][c]);
        RowColAvgP2(edgediff->Plane(c), &ssimulacra.row_p2[1][c],
                    &ssimulacra.col_p2[1][c]);
      }
    }
  }
  return ssimulacra;
}

}  // namespace ssimulacra
