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

#ifndef JXL_DCT_UTIL_H_
#define JXL_DCT_UTIL_H_

#include <stddef.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/image.h"

namespace jxl {
using ac_qcoeff_t = float;
using ACImage = Plane<ac_qcoeff_t>;
using ACImage3 = Image3<ac_qcoeff_t>;

// Returns an N x M image by taking the DC coefficient from each 64x1 block.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
template <typename T>
Plane<T> DCImage(const Plane<T>& coeffs) {
  JXL_ASSERT(coeffs.xsize() % kDCTBlockSize == 0);
  Plane<T> out(coeffs.xsize() / kDCTBlockSize, coeffs.ysize());
  for (size_t y = 0; y < out.ysize(); ++y) {
    const T* JXL_RESTRICT row_in = coeffs.ConstRow(y);
    T* JXL_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < out.xsize(); ++x) {
      row_out[x] = row_in[x * kDCTBlockSize];
    }
  }
  return out;
}

template <typename T>
Image3<T> DCImage(const Image3<T>& coeffs) {
  return Image3<T>(DCImage(coeffs.Plane(0)), DCImage(coeffs.Plane(1)),
                   DCImage(coeffs.Plane(2)));
}

// Scatters dc into "coeffs" at offset 0 within 1x64 blocks.
template <typename T>
void FillDC(const Plane<T>& dc, T* JXL_RESTRICT dst, size_t dst_stride) {
  for (size_t y = 0; y < dc.ysize(); y++) {
    const T* JXL_RESTRICT row_dc = dc.ConstRow(y);
    T* JXL_RESTRICT row_out = dst + dst_stride * y;
    for (size_t x = 0; x < dc.xsize(); ++x) {
      row_out[kDCTBlockSize * x] = row_dc[x];
    }
  }
}

template <typename T>
void FillDC(const Image3<T>& dc, Image3<T>* JXL_RESTRICT coeffs) {
  for (size_t c = 0; c < 3; ++c) {
    FillDC(dc.Plane(c), coeffs->PlaneRow(c, 0), coeffs->PixelsPerRow());
  }
}

}  // namespace jxl

#endif  // JXL_DCT_UTIL_H_
