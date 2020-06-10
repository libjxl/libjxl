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

#include "jxl/gauss_blur.h"

#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "jxl/base/os_specific.h"
#include "jxl/base/robust_statistics.h"
#include "jxl/convolve.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"

namespace jxl {

bool NearEdge(const int64_t width, const int64_t peak) {
  // When around 3*sigma from the edge, there is negligible truncation.
  return peak < 10 || peak > width - 10;
}

// Follow the curve downwards by scanning right from `peak` and verifying
// identical values at the same offset to the left.
void VerifySymmetric(const int64_t width, const int64_t peak,
                     const float* out) {
  const double tolerance = NearEdge(width, peak) ? 0.015 : 6E-7;
  for (int64_t i = 1;; ++i) {
    // Stop if we passed either end of the array
    if (peak - i < 0 || peak + i >= width) break;
    EXPECT_GT(out[peak + i - 1] + tolerance, out[peak + i]);  // descending
    EXPECT_NEAR(out[peak - i], out[peak + i], tolerance);     // symmetric
  }
}

void TestImpulseResponse(size_t width, size_t peak) {
  const auto rg3 = CreateRecursiveGaussian(3.0);
  const auto rg4 = CreateRecursiveGaussian(4.0);
  const auto rg5 = CreateRecursiveGaussian(5.0);

  // Extra padding for 4x unrolling
  std::vector<float> in(width + 3);
  in[peak] = 1.0f;

  std::vector<float> out3(width + 3);
  std::vector<float> out4(width + 3);
  std::vector<float> out5(width + 3);
  ChooseFastGaussian1D()(rg3, in.data(), width, out3.data());
  ChooseFastGaussian1D()(rg4, out3.data(), width, out4.data());
  ChooseFastGaussian1D()(rg5, in.data(), width, out5.data());

  VerifySymmetric(width, peak, out3.data());
  VerifySymmetric(width, peak, out4.data());
  VerifySymmetric(width, peak, out5.data());

  // Wider kernel has flatter peak
  EXPECT_LT(out5[peak] + 0.05, out3[peak]);

  // Gauss3 o Gauss4 ~= Gauss5
  const double tolerance = NearEdge(width, peak) ? 0.04 : 0.01;
  for (size_t i = 0; i < width; ++i) {
    EXPECT_NEAR(out4[i], out5[i], tolerance);
  }
}

void TestImpulseResponseForWidth(size_t width) {
  for (size_t i = 0; i < width; ++i) {
    TestImpulseResponse(width, i);
  }
}

TEST(GaussBlurTest, ImpulseResponse) {
  TestImpulseResponseForWidth(10);  // tiny even
  TestImpulseResponseForWidth(15);  // small odd
  TestImpulseResponseForWidth(32);  // power of two
  TestImpulseResponseForWidth(31);  // power of two - 1
  TestImpulseResponseForWidth(33);  // power of two + 1
}

void TestDirac2D(size_t xsize, size_t ysize, double sigma) {
  ImageF in(xsize, ysize);
  ZeroFillImage(&in);
  // We anyway ignore the border below, so might as well choose the middle.
  in.Row(ysize / 2)[xsize / 2] = 1.0f;

  ImageF temp(xsize, ysize);
  ImageF out(xsize, ysize);
  const auto rg = CreateRecursiveGaussian(sigma);
  ThreadPool* null_pool = nullptr;
  FastGaussian(rg, in, null_pool, &temp, &out);

  const std::vector<float> kernel =
      GaussianKernel(static_cast<int>(4 * sigma), static_cast<float>(sigma));
  const ImageF expected = Convolve(in, kernel);

  const double max_l1 = sigma < 1.5 ? 5E-3 : 6E-4;
  const size_t border = 2 * sigma;
  VerifyRelativeError(expected, out, max_l1, 1E-8, border);
}

TEST(GaussBlurTest, Test2D) {
  const std::vector<int> dimensions{6, 15, 17, 64, 50, 49};
  for (int xsize : dimensions) {
    for (int ysize : dimensions) {
      for (double sigma : {1.0, 2.5, 3.6}) {
        TestDirac2D(static_cast<size_t>(xsize), static_cast<size_t>(ysize),
                    sigma);
      }
    }
  }
}

// Returns megapixels/sec. "div" is a divisor for the number of repetitions,
// used to reduce benchmark duration. Func returns elapsed time.
template <class Func>
double Measure(const size_t xsize, const size_t ysize, int div,
               const Func& func) {
  const int reps = 100 / div;
  std::vector<double> elapsed;
  for (int i = 0; i < reps; ++i) {
    elapsed.push_back(func(xsize, ysize));
  }

  double mean_elapsed;
  // Potential loss of precision, and also enough samples for mode.
  if (reps > 50) {
    std::sort(elapsed.begin(), elapsed.end());
    mean_elapsed = jxl::HalfSampleMode()(elapsed.data(), elapsed.size());
  } else {
    // Skip first(noisier)
    mean_elapsed = Geomean(elapsed.data() + 1, elapsed.size() - 1);
  }
  return (xsize * ysize * 1E-6) / mean_elapsed;
}

void Benchmark(size_t xsize, size_t ysize, double sigma) {
  const double mps_rg =
      Measure(xsize, ysize, 1, [sigma](size_t xsize, size_t ysize) {
        ImageF in(xsize, ysize);
        const float expected = xsize + ysize;
        FillImage(expected, &in);

        ImageF temp(xsize, ysize);
        ImageF out(xsize, ysize);
        const auto rg = CreateRecursiveGaussian(sigma);
        ThreadPool* null_pool = nullptr;
        const double t0 = Now();
        FastGaussian(rg, in, null_pool, &temp, &out);
        const double t1 = Now();
        // Prevent optimizing out
        const float actual = out.ConstRow(ysize / 2)[xsize / 2];
        const float rel_err = std::abs(actual - expected) / expected;
        EXPECT_LT(rel_err, 1.5E-5);
        return t1 - t0;
      });

  const double mps_old =
      Measure(xsize, ysize, 10, [sigma](size_t xsize, size_t ysize) {
        ImageF in(xsize, ysize);
        const float expected = xsize + ysize;
        FillImage(expected, &in);
        const std::vector<float> kernel = GaussianKernel(
            static_cast<int>(4 * sigma), static_cast<float>(sigma));
        const double t0 = Now();
        const ImageF out = Convolve(in, kernel);
        const double t1 = Now();

        // Prevent optimizing out
        const float actual = out.ConstRow(ysize / 2)[xsize / 2];
        const float rel_err = std::abs(actual - expected) / expected;
        EXPECT_LT(rel_err, 5E-6);
        return t1 - t0;
      });

  const double mps_sep7 =
      Measure(xsize, ysize, 1, [](size_t xsize, size_t ysize) {
        ImageF in(xsize, ysize);
        const float expected = xsize + ysize;
        FillImage(expected, &in);
        ImageF out(xsize, ysize);
        // Gaussian with sigma 1
        const WeightsSeparable7 weights = {
            {HWY_REP4(0.383103f), HWY_REP4(0.241843f), HWY_REP4(0.060626f),
             HWY_REP4(0.00598f)},
            {HWY_REP4(0.383103f), HWY_REP4(0.241843f), HWY_REP4(0.060626f),
             HWY_REP4(0.00598f)}};
        ThreadPool* null_pool = nullptr;
        const auto sep7 = ChooseSeparable7();
        const double t0 = Now();
        sep7(in, Rect(in), weights, null_pool, &out);
        const double t1 = Now();

        // Prevent optimizing out
        const float actual = out.ConstRow(ysize / 2)[xsize / 2];
        const float rel_err = std::abs(actual - expected) / expected;
        EXPECT_LT(rel_err, 5E-6);
        return t1 - t0;
      });

  printf("%4zu x %4zu @%.1f: old %.1f, sep7 %.1f, rg %.1f\n", xsize, ysize,
         sigma, mps_old, mps_sep7, mps_rg);
}

TEST(GaussBlurTest, Benchmark) {
  Benchmark(128, 128, 2);
  Benchmark(128, 128, 4);

  Benchmark(300, 300, 2);
  Benchmark(300, 300, 4);

  Benchmark(501, 501, 4);
  PROFILER_PRINT_RESULTS();
}

}  // namespace jxl
