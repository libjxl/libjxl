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

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "jxl/ac_strategy.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/common.h"
#include "jxl/epf.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"
#include "jxl/loop_filter.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"

namespace jxl {
namespace {

constexpr bool kPrint = false;

const size_t xsize = 8;
const size_t ysize = 8;

void GenerateFlat(const float background, const float foreground,
                  std::vector<Image3F>* images) {
  for (size_t c = 0; c < Image3F::kNumPlanes; ++c) {
    Image3F in(xsize, ysize);
    // Plane c = foreground, all others = background.
    for (size_t y = 0; y < ysize; ++y) {
      float* rows[3] = {in.PlaneRow(0, y), in.PlaneRow(1, y),
                        in.PlaneRow(2, y)};
      for (size_t x = 0; x < xsize; ++x) {
        rows[0][x] = rows[1][x] = rows[2][x] = background;
        rows[c][x] = foreground;
      }
    }
    images->push_back(std::move(in));
  }
}

// Single foreground point at any position in any channel
void GeneratePoints(const float background, const float foreground,
                    std::vector<Image3F>* images) {
  for (size_t c = 0; c < Image3F::kNumPlanes; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        Image3F in(xsize, ysize);
        FillImage(background, &in);
        in.PlaneRow(c, y)[x] = foreground;
        images->push_back(std::move(in));
      }
    }
  }
}

void GenerateHorzEdges(const float background, const float foreground,
                       std::vector<Image3F>* images) {
  for (size_t c = 0; c < Image3F::kNumPlanes; ++c) {
    // Begin of foreground rows
    for (size_t y = 1; y < ysize; ++y) {
      Image3F in(xsize, ysize);
      FillImage(background, &in);
      for (size_t iy = y; iy < ysize; ++iy) {
        std::fill(in.PlaneRow(c, iy), in.PlaneRow(c, iy) + xsize, foreground);
      }
      images->push_back(std::move(in));
    }
  }
}

void GenerateVertEdges(const float background, const float foreground,
                       std::vector<Image3F>* images) {
  for (size_t c = 0; c < Image3F::kNumPlanes; ++c) {
    // Begin of foreground columns
    for (size_t x = 1; x < xsize; ++x) {
      Image3F in(xsize, ysize);
      FillImage(background, &in);
      for (size_t iy = 0; iy < ysize; ++iy) {
        float* JXL_RESTRICT row = in.PlaneRow(c, iy);
        for (size_t ix = x; ix < xsize; ++ix) {
          row[ix] = foreground;
        }
      }
      images->push_back(std::move(in));
    }
  }
}

// Ensures input remains unchanged by filter - verifies the edge-preserving
// nature of the filter because inputs are piecewise constant.
void EnsureUnchanged(const float background, const float foreground) {
  std::vector<Image3F> images;
  GenerateFlat(background, foreground, &images);
  GeneratePoints(background, foreground, &images);
  GenerateHorzEdges(background, foreground, &images);
  GenerateVertEdges(background, foreground, &images);

  DequantMatrices dequant;
  size_t xsize_blocks = DivCeil(xsize, kBlockDim);
  size_t ysize_blocks = DivCeil(ysize, kBlockDim);
  LoopFilter lf;
  ImageF sigma(xsize_blocks + 4, ysize_blocks + 4);
  FillImage(-0.5f, &sigma);

  Image3F storage1(xsize + 4 * kBlockDim, kEpf1InputRows);
  Image3F storage2(xsize + 2 * kBlockDim, kEpf2InputRows);
  lf.gab = false;

  for (size_t idx_image = 0; idx_image < images.size(); ++idx_image) {
    const Image3F& in = images[idx_image];

    if (kPrint) {
      printf("--In\n");
      for (size_t y = 0; y < ysize; ++y) {
        const float* row_x = in.ConstPlaneRow(0, y);
        const float* row_y = in.ConstPlaneRow(1, y);
        const float* row_b = in.ConstPlaneRow(2, y);
        for (size_t x = 0; x < ysize; ++x) {
          printf("%6.2f|%6.2f|%6.2f ", row_x[x], row_y[x], row_b[x]);
        }
        printf("\n");
      }
    }

    Image3F out = CopyImage(in);  // = in_out
    Image3F padded = PadImageSymmetric(in, 2 * kBlockDim);
    EdgePreservingFilter(lf, Rect(0, 0, xsize, ysize), padded,
                         Rect(0, 0, xsize_blocks, ysize_blocks), sigma,
                         Rect(0, 0, xsize, ysize), &out, &storage1, &storage2);

    if (kPrint) {
      printf("--Out\n");
      for (size_t y = 0; y < ysize; ++y) {
        const float* row_x = out.ConstPlaneRow(0, y);
        const float* row_y = out.ConstPlaneRow(1, y);
        const float* row_b = out.ConstPlaneRow(2, y);
        for (size_t x = 0; x < ysize; ++x) {
          printf("%6.2f|%6.2f|%6.2f ", row_x[x], row_y[x], row_b[x]);
        }
        printf("\n");
      }
    }

    VerifyRelativeError(in, out, 1E-3, 1E-4);
  }
}

TEST(AdaptiveReconstructionTest, TestBright) { EnsureUnchanged(1.0f, 128.0f); }
TEST(AdaptiveReconstructionTest, TestDark) { EnsureUnchanged(128.0f, 1.0f); }

}  // namespace
}  // namespace jxl
