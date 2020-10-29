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

#include "lib/jxl/dec_dct.h"

#include "lib/jxl/base/profiler.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_scales.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/dec_dct.cc"
#include <hwy/foreach_target.h>
// ^ must come before highway.h and any *-inl.h.

#include <hwy/highway.h>

#include "lib/jxl/dct-inl.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

void IDct8(const size_t xsize_blocks, const size_t ysize_blocks,
           const ImageF& dequantized, ThreadPool* pool,
           ImageF* JXL_RESTRICT pixels) {
  HWY_ALIGN float scratch_space[64 * 2];
  constexpr size_t N = kBlockDim;
  const size_t xsize_groups = DivCeil(xsize_blocks, kGroupDimInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kGroupDimInBlocks);
  const size_t pixels_stride = pixels->PixelsPerRow();

  const auto idct = [&](int idx, int /* thread */) {
    HWY_ALIGN float block[N * N];
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
            DCTFrom(dequantized_row + bx * kDCTBlockSize, N), DCTTo(block, N),
            scratch_space);
        Transpose<N, N>::Run(DCTFrom(block, N),
                             DCTTo(pixels_row + bx * N, pixels_stride));
      }
    }
  };
  RunOnPool(pool, 0, static_cast<int>(xsize_groups * ysize_groups),
            ThreadPool::SkipInit(), idct, "Brunsli:IDCT");
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(IDct8);
void IDct8(const size_t xsize_blocks, const size_t ysize_blocks,
           const ImageF& dequantized, ThreadPool* pool,
           ImageF* JXL_RESTRICT pixels) {
  return HWY_DYNAMIC_DISPATCH(IDct8)(xsize_blocks, ysize_blocks, dequantized,
                                     pool, pixels);
}

}  // namespace jxl
#endif  // HWY_ONCE
