// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/image_ops.h"

#include <cstddef>
#include <cstring>
#include <utility>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"

namespace jxl {

void PadImageToBlockMultipleInPlace(Image3F* JXL_RESTRICT in,
                                    size_t block_dim) {
  const size_t xsize_orig = in->xsize();
  const size_t ysize_orig = in->ysize();
  const size_t xsize = RoundUpTo(xsize_orig, block_dim);
  const size_t ysize = RoundUpTo(ysize_orig, block_dim);
  // Expands image size to the originally-allocated size.
  in->ShrinkTo(xsize, ysize);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize_orig; y++) {
      float* JXL_RESTRICT row = in->PlaneRow(c, y);
      for (size_t x = xsize_orig; x < xsize; x++) {
        row[x] = row[xsize_orig - 1];
      }
    }
    const float* JXL_RESTRICT row_src = in->ConstPlaneRow(c, ysize_orig - 1);
    for (size_t y = ysize_orig; y < ysize; y++) {
      memcpy(in->PlaneRow(c, y), row_src, xsize * sizeof(float));
    }
  }
}

static void DownsampleImage(const ImageF& input, size_t factor,
                            ImageF* output) {
  JXL_ASSERT(factor != 1);
  output->ShrinkTo(DivCeil(input.xsize(), factor),
                   DivCeil(input.ysize(), factor));
  size_t in_stride = input.PixelsPerRow();
  for (size_t y = 0; y < output->ysize(); y++) {
    float* row_out = output->Row(y);
    const float* row_in = input.Row(factor * y);
    for (size_t x = 0; x < output->xsize(); x++) {
      size_t cnt = 0;
      float sum = 0;
      for (size_t iy = 0; iy < factor && iy + factor * y < input.ysize();
           iy++) {
        for (size_t ix = 0; ix < factor && ix + factor * x < input.xsize();
             ix++) {
          sum += row_in[iy * in_stride + x * factor + ix];
          cnt++;
        }
      }
      row_out[x] = sum / cnt;
    }
  }
}

void DownsampleImage(ImageF* image, size_t factor) {
  // Allocate extra space to avoid a reallocation when padding.
  ImageF downsampled(DivCeil(image->xsize(), factor) + kBlockDim,
                     DivCeil(image->ysize(), factor) + kBlockDim);
  DownsampleImage(*image, factor, &downsampled);
  *image = std::move(downsampled);
}

void DownsampleImage(Image3F* opsin, size_t factor) {
  JXL_ASSERT(factor != 1);
  // Allocate extra space to avoid a reallocation when padding.
  Image3F downsampled(DivCeil(opsin->xsize(), factor) + kBlockDim,
                      DivCeil(opsin->ysize(), factor) + kBlockDim);
  downsampled.ShrinkTo(downsampled.xsize() - kBlockDim,
                       downsampled.ysize() - kBlockDim);
  for (size_t c = 0; c < 3; c++) {
    DownsampleImage(opsin->Plane(c), factor, &downsampled.Plane(c));
  }
  *opsin = std::move(downsampled);
}

}  // namespace jxl
