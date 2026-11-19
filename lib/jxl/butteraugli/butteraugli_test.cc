// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/butteraugli/butteraugli.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/extras/metrics.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/test_image.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {

// Forward declarations of file-scope helpers in butteraugli.cc — used by the
// blur-equivalence test below to verify that the new direct H+V path produces
// bit-identical output to the legacy transpose-based path.
std::vector<float> ComputeKernel(float sigma);
Status ConvolutionWithTranspose(const ImageF& in,
                                const std::vector<float>& kernel,
                                ImageF* out);
Status ConvolutionHorizontal(const ImageF& in,
                             const std::vector<float>& kernel,
                             ImageF* out);
Status ConvolutionVertical(const ImageF& in,
                           const std::vector<float>& kernel,
                           ImageF* out);

namespace {

using ::jxl::test::GetColorImage;
using ::jxl::test::TestImage;

Image3F SinglePixelImage(float red, float green, float blue) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(Image3F img, Image3F::Create(memory_manager, 1, 1));
  img.PlaneRow(0, 0)[0] = red;
  img.PlaneRow(1, 0)[0] = green;
  img.PlaneRow(2, 0)[0] = blue;
  return img;
}

void AddUniformNoise(Image3F* img, float d, uint64_t seed) {
  Rng generator(seed);
  for (size_t y = 0; y < img->ysize(); ++y) {
    for (int c = 0; c < 3; ++c) {
      for (size_t x = 0; x < img->xsize(); ++x) {
        img->PlaneRow(c, y)[x] += generator.UniformF(-d, d);
      }
    }
  }
}

void AddEdge(Image3F* img, float d, size_t x0, size_t y0) {
  const size_t h = std::min<size_t>(img->ysize() - y0, 100);
  const size_t w = std::min<size_t>(img->xsize() - x0, 5);
  for (size_t dy = 0; dy < h; ++dy) {
    for (size_t dx = 0; dx < w; ++dx) {
      img->PlaneRow(1, y0 + dy)[x0 + dx] += d;
    }
  }
}

// Verifies that the direct H+V blur path (ConvolutionHorizontal followed by
// ConvolutionVertical) produces output identical to the legacy
// transpose-based path (ConvolutionWithTranspose ∘ ConvolutionWithTranspose
// through a transposed scratch). Tested at every kernel size the manual
// case-N unrolls cover (7, 13, 15, 33) and at sizes that exercise both
// interior and border code (small enough that border > 2*offset overlaps).
TEST(ButteraugliBlurEquivalence, HorizontalThenVerticalMatchesTranspose) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();

  // Sigmas chosen to land exactly on each case-N kernel size that
  // ConvolutionWithTranspose / ConvolutionHorizontal / ConvolutionVertical
  // hand-unroll. ComputeKernel uses len = 2 * max(1, 2.25 * sigma) + 1.
  const struct {
    float sigma;
    size_t expected_size;
  } cases[] = {
      {1.50f, 7},   // diff = (int)(2.25*1.50) = 3
      {2.70f, 13},  // diff = (int)(2.25*2.70) = 6
      {3.20f, 15},  // diff = (int)(2.25*3.20) = 7
      {7.20f, 33},  // diff = (int)(2.25*7.20) = 16
  };

  // Sizes hit all border configurations: tiny (mostly border for kernel 33),
  // square interior, and a non-square taller-than-wide image.
  const struct {
    size_t xsize;
    size_t ysize;
  } sizes[] = {
      {8, 8},
      {17, 19},
      {64, 32},
      {128, 96},
  };

  for (const auto& s : sizes) {
    JXL_TEST_ASSIGN_OR_DIE(ImageF in,
                           ImageF::Create(memory_manager, s.xsize, s.ysize));
    Rng rng(0xB17EB17Eu ^ (s.xsize * 31 + s.ysize));
    for (size_t y = 0; y < s.ysize; ++y) {
      float* row = in.Row(y);
      for (size_t x = 0; x < s.xsize; ++x) {
        // Mix of negative and positive, no NaN/Inf — same value regime as
        // the XYB intermediates butteraugli sees in production.
        row[x] = rng.UniformF(-1.0f, 1.0f);
      }
    }

    for (const auto& c : cases) {
      const std::vector<float> kernel = ComputeKernel(c.sigma);
      ASSERT_EQ(kernel.size(), c.expected_size)
          << "ComputeKernel(" << c.sigma << ") did not produce kernel size "
          << c.expected_size;

      // Legacy: transpose buffer ∘ ConvolutionWithTranspose twice.
      JXL_TEST_ASSIGN_OR_DIE(
          ImageF temp_t, ImageF::Create(memory_manager, s.ysize, s.xsize));
      JXL_TEST_ASSIGN_OR_DIE(
          ImageF out_old,
          ImageF::Create(memory_manager, s.xsize, s.ysize));
      ASSERT_TRUE(ConvolutionWithTranspose(in, kernel, &temp_t));
      ASSERT_TRUE(ConvolutionWithTranspose(temp_t, kernel, &out_old));

      // New: direct H then V, no transpose.
      JXL_TEST_ASSIGN_OR_DIE(
          ImageF h_out,
          ImageF::Create(memory_manager, s.xsize, s.ysize));
      JXL_TEST_ASSIGN_OR_DIE(
          ImageF out_new,
          ImageF::Create(memory_manager, s.xsize, s.ysize));
      ASSERT_TRUE(ConvolutionHorizontal(in, kernel, &h_out));
      ASSERT_TRUE(ConvolutionVertical(h_out, kernel, &out_new));

      // Bit-identical: same kernel weights, same accumulation order
      // (sum0/sum1/sum2/sum3 with the same pairings) — the only thing
      // that changed is the memory layout of intermediates.
      for (size_t y = 0; y < s.ysize; ++y) {
        const float* row_old = out_old.ConstRow(y);
        const float* row_new = out_new.ConstRow(y);
        for (size_t x = 0; x < s.xsize; ++x) {
          EXPECT_FLOAT_EQ(row_new[x], row_old[x])
              << "size " << s.xsize << "x" << s.ysize << " kernel "
              << c.expected_size << " at (" << x << "," << y << ")";
        }
      }
    }
  }
}

TEST(ButteraugliInPlaceTest, SinglePixel) {
  Image3F rgb0 = SinglePixelImage(0.5f, 0.5f, 0.5f);
  Image3F rgb1 = SinglePixelImage(0.5f, 0.49f, 0.5f);
  ButteraugliParams butteraugli_params;
  ImageF diffmap;
  double diffval;
  EXPECT_TRUE(
      ButteraugliInterface(rgb0, rgb1, butteraugli_params, diffmap, diffval));
  EXPECT_NEAR(diffval, 2.5, 0.5);
  ImageF diffmap2;
  double diffval2;
  EXPECT_TRUE(ButteraugliInterfaceInPlace(std::move(rgb0), std::move(rgb1),
                                          butteraugli_params, diffmap2,
                                          diffval2));
  EXPECT_NEAR(diffval, diffval2, 1e-10);
}

TEST(ButteraugliInPlaceTest, LargeImage) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const size_t xsize = 1024;
  const size_t ysize = 1024;
  TestImage img;
  ASSERT_TRUE(img.SetDimensions(xsize, ysize));
  JXL_TEST_ASSIGN_OR_DIE(auto frame, img.AddFrame());
  frame.RandomFill(777);
  JXL_TEST_ASSIGN_OR_DIE(Image3F rgb0, GetColorImage(img.ppf()));
  JXL_TEST_ASSIGN_OR_DIE(Image3F rgb1,
                         Image3F::Create(memory_manager, xsize, ysize));
  ASSERT_TRUE(CopyImageTo(rgb0, &rgb1));
  AddUniformNoise(&rgb1, 0.02f, 7777);
  AddEdge(&rgb1, 0.1f, xsize / 2, xsize / 2);
  ButteraugliParams butteraugli_params;
  ImageF diffmap;
  double diffval;
  EXPECT_TRUE(
      ButteraugliInterface(rgb0, rgb1, butteraugli_params, diffmap, diffval));
  JXL_TEST_ASSIGN_OR_DIE(double distp,
                         ComputeDistanceP(diffmap, butteraugli_params, 3.0));
  EXPECT_NEAR(diffval, 4.0, 0.5);
  EXPECT_NEAR(distp, 1.5, 0.5);
  ImageF diffmap2;
  double diffval2;
  EXPECT_TRUE(ButteraugliInterfaceInPlace(std::move(rgb0), std::move(rgb1),
                                          butteraugli_params, diffmap2,
                                          diffval2));
  JXL_TEST_ASSIGN_OR_DIE(double distp2,
                         ComputeDistanceP(diffmap2, butteraugli_params, 3.0));
  EXPECT_NEAR(diffval, diffval2, 5e-7);
  EXPECT_NEAR(distp, distp2, 1e-7);
}

}  // namespace
}  // namespace jxl
