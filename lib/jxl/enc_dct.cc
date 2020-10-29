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

#include "lib/jxl/enc_dct.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_dct.cc"
#include <hwy/foreach_target.h>
// ^ must come before highway.h and any *-inl.h.

#include <hwy/highway.h>

#include "lib/jxl/base/profiler.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct-inl.h"
#include "lib/jxl/dct_scales.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

ImageF Dct8(const ImageF& image) {
  constexpr size_t N = kBlockDim;
  static_assert(N == 8, "JPEG block dim must be 8");
  static_assert(kDCTBlockSize == N * N, "JPEG block size must be 64");

  JXL_ASSERT(image.xsize() % N == 0);
  JXL_ASSERT(image.ysize() % N == 0);
  const size_t xsize_blocks = image.xsize() / N;
  const size_t ysize_blocks = image.ysize() / N;
  ImageF dct(xsize_blocks * kDCTBlockSize, ysize_blocks);

  HWY_ALIGN float scratch_space[64 * 2];
  for (size_t by = 0; by < ysize_blocks; ++by) {
    const float* JXL_RESTRICT row_in = image.ConstRow(by * kBlockDim);
    float* JXL_RESTRICT row_dct = dct.Row(by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      ComputeTransposedScaledDCT<kBlockDim>()(
          DCTFrom(row_in + bx * kBlockDim, image.PixelsPerRow()),
          DCTTo(row_dct + bx * kDCTBlockSize, kBlockDim), scratch_space);
    }
  }
  return dct;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(Dct8);
ImageF Dct8(const ImageF& image) { return HWY_DYNAMIC_DISPATCH(Dct8)(image); }

}  // namespace jxl
#endif  // HWY_ONCE
