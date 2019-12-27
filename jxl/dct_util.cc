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

#include <hwy/static_targets.h>

#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/block.h"
#include "jxl/common.h"
#include "jxl/dct.h"

namespace jxl {

HWY_ATTR void TransposedScaledDCT(const Image3F& image,
                                  Image3F* JXL_RESTRICT dct) {
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
            FromLines<kBlockDim>(row_in + bx * kBlockDim, image.PixelsPerRow()),
            ScaleToBlock<kBlockDim>(row_dct + bx * kDCTBlockSize));
      }
    }
  }
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

}  // namespace jxl
