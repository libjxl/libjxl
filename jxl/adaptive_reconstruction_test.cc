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

const size_t xsize = 16;
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

void DumpTestImage(const char* name, const Image3F& img) {
  fprintf(stderr, "Image %s:\n", name);
  for (size_t y = 0; y < img.ysize(); ++y) {
    const float* row_x = img.ConstPlaneRow(0, y);
    const float* row_y = img.ConstPlaneRow(1, y);
    const float* row_b = img.ConstPlaneRow(2, y);
    for (size_t x = 0; x < img.xsize(); ++x) {
      fprintf(stderr, "%5.1f|%5.1f|%5.1f ", row_x[x], row_y[x], row_b[x]);
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

// Ensures input remains unchanged by filter - verifies the edge-preserving
// nature of the filter because inputs are piecewise constant.
void EnsureUnchanged(const float background, const float foreground,
                     uint32_t epf_iters) {
  std::vector<Image3F> images;
  GenerateFlat(background, foreground, &images);
  GeneratePoints(background, foreground, &images);
  GenerateHorzEdges(background, foreground, &images);
  GenerateVertEdges(background, foreground, &images);

  DequantMatrices dequant;
  LoopFilter lf;

  FilterStorage storage(xsize);
  FillImage(-77.f, &storage.storage1);  // Initialized with garbage.
  FillImage(-33.f, &storage.storage2);  // Initialized with garbage.
  lf.gab = false;
  lf.epf_iters = epf_iters;

  FrameDimensions frame_dim;
  frame_dim.Set(xsize, ysize, /*group_size_shift=*/1,
                /*max_hshift=*/0, /*max_vshift=*/0);
  FilterWeights filter_weights;
  filter_weights.Init(lf, frame_dim);
  FillImage(-0.5f, &filter_weights.sigma);

  for (size_t idx_image = 0; idx_image < images.size(); ++idx_image) {
    const Image3F& in = images[idx_image];

    Image3F out = CopyImage(in);  // = in_out
    Image3F padded = PadImageSymmetric(in, 2 * kBlockDim);
    EdgePreservingFilter(
        lf, filter_weights, Rect(0, 0, xsize, ysize), padded,
        Rect(0, 0, frame_dim.xsize_blocks, frame_dim.ysize_blocks),
        Rect(0, 0, xsize, ysize), &out, &storage);

    VerifyRelativeError(in, out, 1E-3, 1E-4);
    if (testing::Test::HasFatalFailure()) {
      DumpTestImage("in", in);
      DumpTestImage("out", out);
    }
  }
}

}  // namespace

class AdaptiveReconstructionTest : public testing::TestWithParam<uint32_t> {};

INSTANTIATE_TEST_SUITE_P(EPFItersGroup, AdaptiveReconstructionTest,
                         testing::Values(1, 2, 3));

TEST_P(AdaptiveReconstructionTest, TestBright) {
  EnsureUnchanged(1.0f, 128.0f, GetParam());
}
TEST_P(AdaptiveReconstructionTest, TestDark) {
  EnsureUnchanged(128.0f, 1.0f, GetParam());
}

}  // namespace jxl
