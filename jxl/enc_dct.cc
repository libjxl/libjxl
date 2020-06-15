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

#include "jxl/enc_dct.h"

#include "jxl/base/profiler.h"
#include "jxl/common.h"
#include "jxl/dct_scales.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/enc_dct.cc"
#include <hwy/foreach_target.h>

#include "jxl/enc_dct-inl.h"

// SIMD code
#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

void TransposedScaledDCT8(float* block) {
  ComputeTransposedScaledDCT<8>()(FromBlock(8, 8, block), ToBlock(8, 8, block));
}

ImageF Dct8(const ImageF& image) {
  constexpr size_t N = kBlockDim;
  static_assert(N == 8, "JPEG block dim must be 8");
  static_assert(kDCTBlockSize == N * N, "JPEG block size must be 64");

  JXL_ASSERT(image.xsize() % N == 0);
  JXL_ASSERT(image.ysize() % N == 0);
  const size_t xsize_blocks = image.xsize() / N;
  const size_t ysize_blocks = image.ysize() / N;
  ImageF dct(xsize_blocks * kDCTBlockSize, ysize_blocks);

  for (size_t by = 0; by < ysize_blocks; ++by) {
    const float* JXL_RESTRICT row_in = image.ConstRow(by * kBlockDim);
    float* JXL_RESTRICT row_dct = dct.Row(by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      ComputeTransposedScaledDCT<kBlockDim>()(
          FromLines(row_in + bx * kBlockDim, image.PixelsPerRow()),
          ScaleToBlock(kBlockDim, kBlockDim, row_dct + bx * kDCTBlockSize));
    }
  }
  return dct;
}

void TransposedScaledDCT(const Image3F& image, Image3F* JXL_RESTRICT dct) {
  PROFILER_ZONE("TransposedScaledDCT facade");
  JXL_ASSERT(image.xsize() % kBlockDim == 0);
  JXL_ASSERT(image.ysize() % kBlockDim == 0);
  const size_t xsize_blocks = image.xsize() / kBlockDim;
  const size_t ysize_blocks = image.ysize() / kBlockDim;
  JXL_ASSERT(dct->xsize() == xsize_blocks * kDCTBlockSize);
  JXL_ASSERT(dct->ysize() == ysize_blocks);

  for (size_t c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* JXL_RESTRICT row_in = image.ConstPlaneRow(c, by * kBlockDim);
      float* JXL_RESTRICT row_dct = dct->PlaneRow(c, by);

      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        ComputeTransposedScaledDCT<kBlockDim>()(
            FromLines(row_in + bx * kBlockDim, image.PixelsPerRow()),
            ScaleToBlock(kBlockDim, kBlockDim, row_dct + bx * kDCTBlockSize));
      }
    }
  }
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(TransposedScaledDCT8)
void TransposedScaledDCT8(float* block) {
  return HWY_DYNAMIC_DISPATCH(TransposedScaledDCT8)(block);
}

HWY_EXPORT(Dct8)
ImageF Dct8(const ImageF& image) { return HWY_DYNAMIC_DISPATCH(Dct8)(image); }

HWY_EXPORT(TransposedScaledDCT)
void TransposedScaledDCT(const Image3F& image, Image3F* JXL_RESTRICT dct) {
  return HWY_DYNAMIC_DISPATCH(TransposedScaledDCT)(image, dct);
}

}  // namespace jxl
#endif  // HWY_ONCE
