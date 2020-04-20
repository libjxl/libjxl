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

#include "jxl/dec_dct.h"

#include "jxl/base/profiler.h"
#include "jxl/common.h"
#include "jxl/dct_scales.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/dec_dct.cc"
#include <hwy/foreach_target.h>

#include "jxl/dec_dct-inl.h"

namespace jxl {

#include <hwy/begin_target-inl.h>

HWY_ATTR void IDct8(const size_t xsize_blocks, const size_t ysize_blocks,
                    const ImageF& dequantized, ThreadPool* pool,
                    ImageF* JXL_RESTRICT pixels) {
  constexpr size_t N = kBlockDim;
  const size_t xsize_groups = DivCeil(xsize_blocks, kGroupDimInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kGroupDimInBlocks);
  const size_t pixels_stride = pixels->PixelsPerRow();

  const auto idct = [&](int idx, int /* thread */) HWY_ATTR {
    const size_t gx = idx % xsize_groups;
    const size_t gy = idx / xsize_groups;
    const Rect group_rect_blocks(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                                 kGroupDimInBlocks, kGroupDimInBlocks,
                                 xsize_blocks, ysize_blocks);
    const size_t bx0 = group_rect_blocks.x0();
    const size_t bx1 = bx0 + group_rect_blocks.xsize();
    const size_t by0 = group_rect_blocks.y0();
    const size_t by1 = by0 + group_rect_blocks.ysize();
    for (size_t by = by0; by < by1; ++by) {
      const float* JXL_RESTRICT dequantized_row = dequantized.Row(by);
      float* JXL_RESTRICT pixels_row = pixels->Row(by * N);
      for (size_t bx = bx0; bx < bx1; ++bx) {
        ComputeTransposedScaledIDCT<N>()(
            FromBlock<N>(dequantized_row + bx * kDCTBlockSize),
            ToLines<N>(pixels_row + bx * N, pixels_stride));
        GenericTransposeBlockInplace<N>(
            FromLines<N>(pixels_row + bx * N, pixels_stride),
            ToLines<N>(pixels_row + bx * N, pixels_stride));
      }
    }
  };
  RunOnPool(pool, 0, static_cast<int>(xsize_groups * ysize_groups),
            ThreadPool::SkipInit(), idct, "Brunsli:IDCT");
}

HWY_ATTR void TransposedScaledIDCT(const Image3F& dct,
                                   Image3F* JXL_RESTRICT idct) {
  PROFILER_ZONE("IDCT facade");
  JXL_ASSERT(dct.xsize() % kDCTBlockSize == 0);
  const size_t xsize_blocks = dct.xsize() / kDCTBlockSize;
  const size_t ysize_blocks = dct.ysize();
  JXL_ASSERT(idct->xsize() == xsize_blocks * kBlockDim);
  JXL_ASSERT(idct->ysize() == ysize_blocks * kBlockDim);

  for (size_t c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* JXL_RESTRICT row_dct = dct.ConstPlaneRow(c, by);
      float* JXL_RESTRICT row_idct = idct->PlaneRow(c, by * kBlockDim);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        ComputeTransposedScaledIDCT<kBlockDim>()(
            FromBlock<kBlockDim>(row_dct + bx * kDCTBlockSize),
            ToLines<kBlockDim>(row_idct + bx * kBlockDim,
                               idct->PixelsPerRow()));
      }
    }
  }
}

#include <hwy/end_target-inl.h>

#if HWY_ONCE
HWY_EXPORT(IDct8)
HWY_EXPORT(TransposedScaledIDCT)
#endif  // HWY_ONCE

}  // namespace jxl
