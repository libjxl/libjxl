// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/progressive_split.h"

#include <string.h>

#include <algorithm>
#include <memory>

#include "lib/jxl/common.h"
#include "lib/jxl/image.h"

namespace jxl {

template <typename T>
void ProgressiveSplitter::SplitACCoefficients(
    const T* JXL_RESTRICT block, size_t size, const AcStrategy& acs, size_t bx,
    size_t by, size_t offset, T* JXL_RESTRICT output[kMaxNumPasses][3]) {
  auto shift_right_round0 = [&](T v, int shift) {
    T one_if_negative = static_cast<uint32_t>(v) >> 31;
    T add = (one_if_negative << shift) - one_if_negative;
    return (v + add) >> shift;
  };
  // Early quit for the simple case of only one pass.
  if (mode_.num_passes == 1) {
    for (size_t c = 0; c < 3; c++) {
      memcpy(output[0][c] + offset, block + c * size, sizeof(T) * size);
    }
    return;
  }
  size_t ncoeffs_all_done_from_earlier_passes = 1;

  int previous_pass_shift = 0;
  for (size_t num_pass = 0; num_pass < mode_.num_passes; num_pass++) {  // pass
    // Zero out output block.
    for (size_t c = 0; c < 3; c++) {
      memset(output[num_pass][c] + offset, 0, size * sizeof(T));
    }
    const int pass_shift = mode_.passes[num_pass].shift;
    size_t frame_ncoeffs = mode_.passes[num_pass].num_coefficients;
    for (size_t c = 0; c < 3; c++) {  // color-channel
      size_t xsize = acs.covered_blocks_x();
      size_t ysize = acs.covered_blocks_y();
      CoefficientLayout(&ysize, &xsize);
      for (size_t y = 0; y < ysize * frame_ncoeffs; y++) {    // superblk-y
        for (size_t x = 0; x < xsize * frame_ncoeffs; x++) {  // superblk-x
          size_t pos = y * xsize * kBlockDim + x;
          if (x < xsize * ncoeffs_all_done_from_earlier_passes &&
              y < ysize * ncoeffs_all_done_from_earlier_passes) {
            // This coefficient was already included in an earlier pass,
            // which included a genuinely smaller set of coefficients.
            continue;
          }
          T v = block[c * size + pos];
          // Previous pass discarded some bits: do not encode them again.
          if (previous_pass_shift != 0) {
            T previous_v = shift_right_round0(v, previous_pass_shift) *
                           (1 << previous_pass_shift);
            v -= previous_v;
          }
          output[num_pass][c][offset + pos] = shift_right_round0(v, pass_shift);
        }  // superblk-x
      }    // superblk-y
    }      // color-channel
    // We just finished a pass.
    // Hence, we are now guaranteed to have included all coeffs up to
    // frame_ncoeffs in every block, unless the current pass is shifted.
    if (mode_.passes[num_pass].shift == 0) {
      ncoeffs_all_done_from_earlier_passes = frame_ncoeffs;
    }
    previous_pass_shift = mode_.passes[num_pass].shift;
  }  // num_pass
}

template void ProgressiveSplitter::SplitACCoefficients<int32_t>(
    const int32_t* JXL_RESTRICT, size_t, const AcStrategy&, size_t, size_t,
    size_t, int32_t* JXL_RESTRICT[kMaxNumPasses][3]);

template void ProgressiveSplitter::SplitACCoefficients<int16_t>(
    const int16_t* JXL_RESTRICT, size_t, const AcStrategy&, size_t, size_t,
    size_t, int16_t* JXL_RESTRICT[kMaxNumPasses][3]);

}  // namespace jxl
