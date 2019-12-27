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

#include "jxl/image.h"

#include <stdint.h>

#include <utility>

#include "jxl/base/profiler.h"
#include "jxl/common.h"

namespace jxl {

CacheAlignedUniquePtr AllocateImageBytes(size_t size, size_t xsize,
                                         size_t ysize) {
  // (Can't profile CacheAligned itself because it is used by profiler.h)
  PROFILER_FUNC;

  // Note: size may be zero.
  CacheAlignedUniquePtr bytes = AllocateArray(size);
  JXL_ASSERT(reinterpret_cast<uintptr_t>(bytes.get()) % kImageAlign == 0);
  return bytes;
}

ImageB ImageFromPacked(const uint8_t* packed, const size_t xsize,
                       const size_t ysize, const size_t bytes_per_row) {
  JXL_ASSERT(bytes_per_row >= xsize);
  ImageB image(xsize, ysize);
  PROFILER_FUNC;
  for (size_t y = 0; y < ysize; ++y) {
    uint8_t* const JXL_RESTRICT row = image.Row(y);
    const uint8_t* const JXL_RESTRICT packed_row = packed + y * bytes_per_row;
    memcpy(row, packed_row, xsize);
  }
  return image;
}

// Note that using mirroring here gives slightly worse results.
ImageF PadImage(const ImageF& in, const size_t xsize, const size_t ysize) {
  JXL_ASSERT(xsize >= in.xsize());
  JXL_ASSERT(ysize >= in.ysize());
  ImageF out(xsize, ysize);
  size_t y = 0;
  for (; y < in.ysize(); ++y) {
    const float* JXL_RESTRICT row_in = in.ConstRow(y);
    float* JXL_RESTRICT row_out = out.Row(y);
    memcpy(row_out, row_in, in.xsize() * sizeof(row_in[0]));
    const int lastcol = in.xsize() - 1;
    const float lastval = row_out[lastcol];
    for (size_t x = in.xsize(); x < xsize; ++x) {
      row_out[x] = lastval;
    }
  }

  // TODO(janwas): no need to copy if we can 'extend' image: if rows are
  // pointers to any memory? Or allocate larger image before IO?
  const int lastrow = in.ysize() - 1;
  for (; y < ysize; ++y) {
    const float* JXL_RESTRICT row_in = out.ConstRow(lastrow);
    float* JXL_RESTRICT row_out = out.Row(y);
    memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
  }
  return out;
}

Image3F PadImageSymmetric(const Image3F& in, const size_t border) {
  size_t xsize = in.xsize();
  size_t ysize = in.ysize();
  Image3F out(xsize + 2 * border, ysize + 2 * border);
  CopyImageTo(in, Rect(border, border, xsize, ysize), &out);
  for (size_t c = 0; c < 3; c++) {
    // Horizontal pad.
    for (size_t y = 0; y < ysize; y++) {
      for (size_t x = 0; x < border; x++) {
        out.PlaneRow(c, y + border)[x] = in.ConstPlaneRow(c, y)[0];
        out.PlaneRow(c, y + border)[x + xsize + border] =
            in.ConstPlaneRow(c, y)[xsize - 1];
      }
    }
    // Vertical pad.
    for (size_t y = 0; y < border; y++) {
      memcpy(out.PlaneRow(c, y), out.ConstPlaneRow(c, border),
             out.xsize() * sizeof(float));
      memcpy(out.PlaneRow(c, y + ysize + border),
             out.ConstPlaneRow(c, ysize + border - 1),
             out.xsize() * sizeof(float));
    }
  }
  return out;
}

Image3F PadImageToMultiple(const Image3F& in, const size_t N) {
  PROFILER_FUNC;
  const size_t xsize_blocks = DivCeil(in.xsize(), N);
  const size_t ysize_blocks = DivCeil(in.ysize(), N);
  const size_t xsize = N * xsize_blocks;
  const size_t ysize = N * ysize_blocks;
  ImageF out[3];
  for (size_t c = 0; c < 3; ++c) {
    out[c] = PadImage(in.Plane(c), xsize, ysize);
  }
  return Image3F(std::move(out[0]), std::move(out[1]), std::move(out[2]));
}

void PadImageToBlockMultipleInPlace(Image3F* JXL_RESTRICT in) {
  PROFILER_FUNC;
  const size_t xsize_orig = in->xsize();
  const size_t ysize_orig = in->ysize();
  const size_t xsize = RoundUpToBlockDim(xsize_orig);
  const size_t ysize = RoundUpToBlockDim(ysize_orig);
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

float DotProduct(const ImageF& a, const ImageF& b) {
  double sum = 0.0;
  for (size_t y = 0; y < a.ysize(); ++y) {
    const float* const JXL_RESTRICT row_a = a.ConstRow(y);
    const float* const JXL_RESTRICT row_b = b.ConstRow(y);
    for (size_t x = 0; x < a.xsize(); ++x) {
      sum += row_a[x] * row_b[x];
    }
  }
  return sum;
}

}  // namespace jxl
