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

#include "jxl/convolve.h"

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/thread_pool_internal.h"
#define HWY_USE_GTEST  // prevent redefining TEST
#include <hwy/tests/test_util.h>

#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"

#ifndef JXL_DEBUG_CONVOLVE
#define JXL_DEBUG_CONVOLVE 0
#endif

namespace jxl {
namespace {

HWY_ATTR void TestNeighborsImpl() {
  const Neighbors::D d;
  const Neighbors::V v = Iota(d, 0);
  HWY_ALIGN float actual[hwy::kTestMaxVectorSize / sizeof(float)] = {0};

  HWY_ALIGN float first_l1[hwy::kTestMaxVectorSize / sizeof(float)] = {
      0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  Store(Neighbors::FirstL1(v), d, actual);
  EXPECT_EQ(std::vector<float>(first_l1, first_l1 + d.N),
            std::vector<float>(actual, actual + d.N));

#if HWY_BITS != 0
  HWY_ALIGN float first_l2[hwy::kTestMaxVectorSize / sizeof(float)] = {
      1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  Store(Neighbors::FirstL2(v), d, actual);
  EXPECT_EQ(std::vector<float>(first_l2, first_l2 + d.N),
            std::vector<float>(actual, actual + d.N));

  HWY_ALIGN float first_l3[hwy::kTestMaxVectorSize / sizeof(float)] = {
      2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Store(Neighbors::FirstL3(v), d, actual);
  EXPECT_EQ(std::vector<float>(first_l3, first_l3 + d.N),
            std::vector<float>(actual, actual + d.N));
#endif
}

TEST(ConvolveTest, TestNeighbors) { TestNeighborsImpl(); }

// Weights i=0..2 are for Manhattan distance i from center.
template <int64_t kRadius, class Wrap>
struct Symmetric3x3Convolution {
  static_assert(kRadius == 1, "Wrong kRadius");

  template <class Kernel>
  static void Run(const ImageF& in, const Rect& rect, const Kernel& kernel,
                  ImageF* out) {
    PROFILER_ZONE("slow::Symmetric3::Run");
    JXL_CHECK(SameSize(rect, *out));
    const Weights3x3& weights = kernel.Weights();

    const size_t xsize = rect.xsize();
    const size_t ysize = rect.ysize();
    for (int64_t y = 0; y < ysize; ++y) {
      const float* const JXL_RESTRICT row_t = in.ConstRow(Wrap()(y - 1, ysize));
      const float* const JXL_RESTRICT row_m = in.ConstRow(y);
      const float* const JXL_RESTRICT row_b = in.ConstRow(Wrap()(y + 1, ysize));
      float* const JXL_RESTRICT row_out = out->Row(y);

      for (int64_t x = 0; x < xsize; ++x) {
        float mul = row_m[x] * weights.mc[0];
        const int64_t xm1 = Wrap()(x - 1, xsize);
        const int64_t xp1 = Wrap()(x + 1, xsize);
        const float tl = row_t[xm1];
        const float ml = row_m[xm1];
        const float bl = row_b[xm1];
        const float tr = row_t[xp1];
        const float mr = row_m[xp1];
        const float br = row_b[xp1];
        mul += (row_t[x] + row_b[x] + ml + mr) * weights.tc[0];
        mul += (tl + tr + bl + br) * weights.tl[0];
        row_out[x] = mul;
      }
    }
  }
};

template <class Slow, class ImageOrView, class Kernel>
ImageF SlowConvolve(const ImageOrView& in, const Rect& rect,
                    const Kernel& kernel) {
  ImageF out(rect.xsize(), rect.ysize());
  Slow::Run(in, rect, kernel, &out);
  return out;
}

// Compares ConvolveT<> against SlowConvolve.
template <template <int64_t, class> class SlowT, class Strategy, class Kernel,
          class Random>
HWY_ATTR void Verify(const size_t xsize, const size_t ysize,
                     const Kernel& kernel, ThreadPool* pool, Random* rng) {
  constexpr size_t kRadius = Strategy::kRadius;
  static_assert(kRadius <= kConvolveMaxRadius, "Update max radius");

  ImageF in(xsize, ysize);
  GenerateImage(GeneratorRandom<float, Random>(rng, 1.0f), &in);

  using Slow = SlowT<kRadius, WrapMirror>;
  const Rect rect(0, 0, xsize, ysize);
  const ImageF out_expected = SlowConvolve<Slow>(in, rect, kernel);

  ImageF out_actual(xsize, ysize);
  ConvolveT<Strategy>::Run(in, rect, kernel, pool, &out_actual);
  VerifyRelativeError(out_expected, out_actual, 1E-5f, 1E-5f);
}

// Call Verify for null and non-null pool.
template <template <int64_t, class> class SlowT, class Strategy, class Kernel,
          class Random>
HWY_ATTR void VerifyAll(const size_t xsize, const size_t ysize,
                        const Kernel& kernel, Random* rng, ThreadPool* pool) {
  JXL_DEBUG(JXL_DEBUG_CONVOLVE, "pool0");
  Verify<SlowT, Strategy>(xsize, ysize, kernel, /*pool=*/nullptr, rng);

  JXL_DEBUG(JXL_DEBUG_CONVOLVE, "Used pool");
  Verify<SlowT, Strategy>(xsize, ysize, kernel, pool, rng);
}

// Compares slow::Symmetric3x3Convolution against slow::SymmetricConvolution.
template <class Random>
HWY_ATTR void VerifyConvolveSymmetric3x3(const size_t xsize, const size_t ysize,
                                         Random* rng) {
  const size_t kRadius = 1;
  JXL_CHECK(xsize > kRadius);
  JXL_CHECK(ysize > kRadius);
  const Rect rect(0, 0, xsize, ysize);

  ImageF in(xsize, ysize);
  GenerateImage(GeneratorRandom<float, Random>(rng, 1.0f), &in);

  ImageF out_expected(xsize, ysize);
  ImageF out_actual(xsize, ysize);

  Symmetric3x3Convolution<1, WrapMirror>::Run(in, rect, kernel::Lowpass3(),
                                              &out_expected);

  // Expanded form of kernel::Lowpass3: lower-right quadrant.
  const float weights_symmetric[4] = {0.36208932f, 0.12820096f,  //
                                      0.12820096f, 0.03127668f};
  ThreadPool* null_pool = nullptr;
  slow::SymmetricConvolution<kRadius, WrapClamp>::Run(
      in, rect, weights_symmetric, null_pool, &out_actual);

  VerifyRelativeError(out_expected, out_actual, 1E-5f, 1E-5f);
}

// Compares ConvolveT<> against slow::ConvolveSymmetric.
template <class Random>
HWY_ATTR void VerifyConvolveSymmetric5x5(const size_t xsize, const size_t ysize,
                                         Random* rng) {
  const size_t kRadius = 2;
  JXL_CHECK(xsize > kRadius);
  JXL_CHECK(ysize > kRadius);
  const Rect rect(0, 0, xsize, ysize);

  ImageF in(xsize, ysize);
  GenerateImage(GeneratorRandom<float, Random>(rng, 1.0f), &in);

  ImageF out_expected(xsize, ysize);
  ImageF out_actual(xsize, ysize);

  ConvolveT<strategy::Separable5>::Run(in, Rect(in), kernel::Lowpass5(),
                                       /*pool=*/nullptr, &out_expected);

  // Expanded form of kernel::Lowpass5: lower-right quadrant.
  const float weights_symmetric[9] = {0.1740135f, 0.1065369f, 0.0150310f,  //
                                      0.1065369f, 0.0652254f, 0.0092025f,  //
                                      0.0150310f, 0.0092025f, 0.0012984f};
  ThreadPool* null_pool = nullptr;
  slow::SymmetricConvolution<kRadius, WrapMirror>::Run(
      in, rect, weights_symmetric, null_pool, &out_actual);

  VerifyRelativeError(out_expected, out_actual, 1E-5f, 1E-5f);
}

// For all xsize/ysize and kernels:
HWY_ATTR void TestSameResultsImpl() {
  ThreadPoolInternal pool(8);
  constexpr size_t kMinWidth = kConvolveLanes + kConvolveMaxRadius;
  pool.Run(kMinWidth, 40, ThreadPool::SkipInit(),
           [](const int task, int /*thread*/) {
             const size_t xsize = task;
             std::mt19937_64 rng(129 + 13 * xsize);

             ThreadPoolInternal pool3(3);
             for (size_t ysize = kConvolveMaxRadius; ysize < 16; ++ysize) {
               JXL_DEBUG(JXL_DEBUG_CONVOLVE,
                         "%zu x %zu=====================================",
                         xsize, ysize);

               JXL_DEBUG(JXL_DEBUG_CONVOLVE, "Sym3x3------------------");
               VerifyAll<Symmetric3x3Convolution, strategy::Symmetric3>(
                   xsize, ysize, kernel::Lowpass3(), &rng, &pool3);

               JXL_DEBUG(JXL_DEBUG_CONVOLVE, "Sep5x5------------------");
               VerifyAll<slow::SeparableConvolution, strategy::Separable5>(
                   xsize, ysize, kernel::Lowpass5(), &rng, &pool3);

               JXL_DEBUG(JXL_DEBUG_CONVOLVE, "Sep7x7------------------");
               VerifyAll<slow::SeparableConvolution, strategy::Separable7>(
                   xsize, ysize, kernel::Gaussian7Sigma8(), &rng, &pool3);

               VerifyConvolveSymmetric3x3(xsize, ysize, &rng);
               VerifyConvolveSymmetric5x5(xsize, ysize, &rng);
             }
           });
}

TEST(ConvolveTest, TestSameResults) { TestSameResultsImpl(); }

}  // namespace
}  // namespace jxl
