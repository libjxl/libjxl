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

#include "jxl/dct_util.h"

#include <stdint.h>
#include <string>

#include "gtest/gtest.h"
#include "jxl/base/bits.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/codec_in_out.h"
#include "jxl/dct_scales.h"
#include "jxl/dec_dct.h"
#include "jxl/enc_dct.h"
#include "jxl/enc_xyb.h"
#include "jxl/extras/codec.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"
#include "jxl/testdata.h"

namespace jxl {
namespace {

// Zeroes out the top-left 2x2 corner of each DCT block.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
static void ZeroOut2x2(Image3F* coeffs) {
  JXL_ASSERT(coeffs->xsize() % kDCTBlockSize == 0);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < coeffs->ysize(); ++y) {
      float* JXL_RESTRICT row = coeffs->PlaneRow(c, y);
      for (size_t x = 0; x < coeffs->xsize(); x += kDCTBlockSize) {
        row[x] = row[x + 1] = row[x + kBlockDim] = row[x + kBlockDim + 1] =
            0.0f;
      }
    }
  }
}

Image3F KeepOnly2x2Corners(const Image3F& coeffs) {
  Image3F copy = CopyImage(coeffs);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < coeffs.ysize(); ++y) {
      float* JXL_RESTRICT row = copy.PlaneRow(c, y);
      // TODO(user): might be better to copy 4 values, zero-out, place-back.
      for (size_t x = 0; x < coeffs.xsize(); x += kDCTBlockSize) {
        for (size_t k = 0; k < kDCTBlockSize; ++k) {
          if ((k >= (2 * kBlockDim)) || ((k % kBlockDim) >= 2)) {
            row[x + k] = 0.0f;
          }
        }
      }
    }
  }
  return copy;
}

// Returns a 2*N x 2*M image which is defined by the following 3 transforms:
//  1) zero out every coefficient that is outside the top 2x2 corner
//  2) apply ComputeTransposedScaledIDCT() to every block
//  3) subsample the result 4x4 by taking simple averages
// REQUIRES: coeffs.xsize() == kBlockSize*N, coeffs.ysize() == M
static Image3F GetPixelSpaceImageFrom0HVD_64(const Image3F& coeffs) {
  constexpr size_t N = kBlockDim;
  JXL_ASSERT(coeffs.xsize() % kDCTBlockSize == 0);
  const size_t block_xsize = coeffs.xsize() / kDCTBlockSize;
  const size_t block_ysize = coeffs.ysize();
  Image3F out(block_xsize * 2, block_ysize * 2);
  const float kScale01 = N * DCTResampleScales<8, 2>::kScales[1] *
                         DCTScales<N>()[0] * DCTScales<N>()[1];
  const float kScale11 = N * DCTResampleScales<8, 2>::kScales[1] *
                         DCTResampleScales<8, 2>::kScales[1] *
                         DCTScales<N>()[1] * DCTScales<N>()[1];
  for (size_t c = 0; c < 3; ++c) {
    for (size_t by = 0; by < block_ysize; ++by) {
      const float* JXL_RESTRICT row_coeffs = coeffs.PlaneRow(c, by);
      float* JXL_RESTRICT row_out0 = out.PlaneRow(c, 2 * by + 0);
      float* JXL_RESTRICT row_out1 = out.PlaneRow(c, 2 * by + 1);
      for (size_t bx = 0; bx < block_xsize; ++bx) {
        const float* block = row_coeffs + bx * kDCTBlockSize;
        const float a00 = block[0];
        const float a01 = block[N] * kScale01;
        const float a10 = block[1] * kScale01;
        const float a11 = block[N + 1] * kScale11;
        row_out0[2 * bx + 0] = a00 + a01 + a10 + a11;
        row_out0[2 * bx + 1] = a00 - a01 + a10 - a11;
        row_out1[2 * bx + 0] = a00 + a01 - a10 - a11;
        row_out1[2 * bx + 1] = a00 - a01 - a10 + a11;
      }
    }
  }
  return out;
}

// Puts back the top 2x2 corner of each 8x8 block of *coeffs from the
// transformed pixel space image img.
// REQUIRES: coeffs->xsize() == 64*N, coeffs->ysize() == M
static void Add2x2CornersFromPixelSpaceImage(const Image3F& img,
                                             Image3F* coeffs) {
  constexpr size_t N = kBlockDim;
  JXL_ASSERT(coeffs->xsize() % kDCTBlockSize == 0);
  const size_t block_xsize = coeffs->xsize() / kDCTBlockSize;
  const size_t block_ysize = coeffs->ysize();
  JXL_ASSERT(block_xsize * 2 <= img.xsize());
  JXL_ASSERT(block_ysize * 2 <= img.ysize());
  const float kScale01 = N * DCTResampleScales<8, 2>::kScales[1] *
                         DCTScales<N>()[0] * DCTScales<N>()[1];
  const float kScale11 = N * DCTResampleScales<8, 2>::kScales[1] *
                         DCTResampleScales<8, 2>::kScales[1] *
                         DCTScales<N>()[1] * DCTScales<N>()[1];
  for (size_t c = 0; c < 3; ++c) {
    for (size_t by = 0; by < block_ysize; ++by) {
      const float* JXL_RESTRICT row0 = img.PlaneRow(c, 2 * by + 0);
      const float* JXL_RESTRICT row1 = img.PlaneRow(c, 2 * by + 1);
      float* row_out = coeffs->PlaneRow(c, by);
      for (size_t bx = 0; bx < block_xsize; ++bx) {
        const float b00 = row0[2 * bx + 0];
        const float b01 = row0[2 * bx + 1];
        const float b10 = row1[2 * bx + 0];
        const float b11 = row1[2 * bx + 1];
        const float a00 = 0.25f * (b00 + b01 + b10 + b11);
        const float a01 = 0.25f * (b00 - b01 + b10 - b11);
        const float a10 = 0.25f * (b00 + b01 - b10 - b11);
        const float a11 = 0.25f * (b00 - b01 - b10 + b11);
        float* JXL_RESTRICT block = &row_out[bx * kDCTBlockSize];
        block[0] = a00;
        block[1] = a10 / kScale01;
        block[N] = a01 / kScale01;
        block[N + 1] = a11 / kScale11;
      }
    }
  }
}

// Returns an N x M image where each pixel is the average of the corresponding
// f x f block in the original.
// REQUIRES: image.xsize() == f*N, image.ysize() == f *M
static Image3F Subsample(const Image3F& image, int f) {
  JXL_CHECK(image.xsize() % f == 0);
  JXL_CHECK(image.ysize() % f == 0);
  const size_t shift = CeilLog2Nonzero(static_cast<uint32_t>(f));
  JXL_CHECK(f == (1 << shift));
  const size_t nxs = image.xsize() >> shift;
  const size_t nys = image.ysize() >> shift;
  Image3F retval(nxs, nys);
  ZeroFillImage(&retval);
  const float mul = 1.0f / (f * f);
  for (size_t y = 0; y < image.ysize(); ++y) {
    const float* row_in[3] = {image.PlaneRow(0, y), image.PlaneRow(1, y),
                              image.PlaneRow(2, y)};
    const size_t ny = y >> shift;
    float* row_out[3] = {retval.PlaneRow(0, ny), retval.PlaneRow(1, ny),
                         retval.PlaneRow(2, ny)};
    for (size_t c = 0; c < 3; ++c) {
      for (size_t x = 0; x < image.xsize(); ++x) {
        size_t nx = x >> shift;
        row_out[c][nx] += mul * row_in[c][x];
      }
    }
  }
  return retval;
}

static Image3F OpsinTestImage() {
  const PaddedBytes orig =
      ReadTestData("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  JXL_CHECK(SetFromBytes(Span<const uint8_t>(orig), &io, /*pool=*/nullptr));
  ThreadPool* null_pool = nullptr;
  Image3F opsin(io.xsize(), io.ysize());
  ImageBundle unused_linear;
  (void)(*ChooseToXYB)()(io.Main(), null_pool, &opsin, &unused_linear);
  opsin.ShrinkTo(opsin.ysize() & ~7, opsin.xsize() & ~7);
  return opsin;
}

TEST(DctUtilTest, DCTRoundtrip) {
  Image3F opsin = OpsinTestImage();
  const size_t xsize_blocks = opsin.xsize() / kBlockDim;
  const size_t ysize_blocks = opsin.ysize() / kBlockDim;

  Image3F coeffs(xsize_blocks * kDCTBlockSize, ysize_blocks);
  Image3F recon(xsize_blocks * kBlockDim, ysize_blocks * kBlockDim);

  ChooseTransposedScaledDCT()(opsin, &coeffs);
  ChooseTransposedScaledIDCT()(coeffs, &recon);
  VerifyRelativeError(opsin, recon, 1e-6, 1e-6);
}

TEST(DctUtilTest, Transform2x2Corners) {
  Image3F opsin = OpsinTestImage();
  const size_t xsize_blocks = opsin.xsize() / kBlockDim;
  const size_t ysize_blocks = opsin.ysize() / kBlockDim;

  Image3F coeffs(xsize_blocks * kDCTBlockSize, ysize_blocks);
  Image3F recon(xsize_blocks * kBlockDim, ysize_blocks * kBlockDim);
  ChooseTransposedScaledDCT()(opsin, &coeffs);
  Image3F t1 = GetPixelSpaceImageFrom0HVD_64(coeffs);
  ChooseTransposedScaledIDCT()(KeepOnly2x2Corners(coeffs), &recon);
  Image3F t2 = Subsample(recon, 4);
  VerifyRelativeError(t1, t2, 1e-6, 1e-6);
}

TEST(DctUtilTest, Roundtrip2x2Corners) {
  Image3F opsin = OpsinTestImage();
  const size_t xsize_blocks = opsin.xsize() / kBlockDim;
  const size_t ysize_blocks = opsin.ysize() / kBlockDim;

  Image3F coeffs(xsize_blocks * kDCTBlockSize, ysize_blocks);
  ChooseTransposedScaledDCT()(opsin, &coeffs);
  Image3F tmp = GetPixelSpaceImageFrom0HVD_64(coeffs);
  Image3F coeffs_out = CopyImage(coeffs);
  ZeroOut2x2(&coeffs_out);
  Add2x2CornersFromPixelSpaceImage(tmp, &coeffs_out);
  VerifyRelativeError(coeffs, coeffs_out, 1e-6, 1e-6);
}

}  // namespace
}  // namespace jxl
