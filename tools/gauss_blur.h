// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_GAUSS_BLUR_H_
#define LIB_JXL_GAUSS_BLUR_H_

#include <cstddef>
#include <functional>
#include <hwy/base.h>  // HWY_ALIGN_MAX

#include "lib/jxl/base/data_parallel.h"

namespace jxl {

// Only for use by CreateRecursiveGaussian and FastGaussian*.
#pragma pack(push, 1)
struct HWY_ALIGN_MAX RecursiveGaussian {
  // For k={1,3,5} in that order, each broadcasted 4x for LoadDup128. Used only
  // for vertical passes.
  float n2[3 * 4];
  float d1[3 * 4];

  // We unroll horizontal passes 4x - one output per lane. These are each lane's
  // multiplier for the previous output (relative to the first of the four
  // outputs). Indexing: 4 * 0..2 (for {1,3,5}) + 0..3 for the lane index.
  float mul_prev[3 * 4];
  // Ditto for the second to last output.
  float mul_prev2[3 * 4];

  // We multiply a vector of inputs 0..3 by a vector shifted from this array.
  // in=0 uses all 4 (nonzero) terms; for in=3, the lower three lanes are 0.
  float mul_in[3 * 4];

  size_t radius;
};
#pragma pack(pop)

// Precomputation for FastGaussian*; users may use the same pointer/storage in
// subsequent calls to FastGaussian* with the same sigma.
RecursiveGaussian CreateRecursiveGaussian(double sigma);

// 1D Gaussian with zero-pad boundary handling and runtime independent of sigma.
void FastGaussian1D(const RecursiveGaussian& rg, size_t xsize,
                    const float* JXL_RESTRICT in, float* JXL_RESTRICT out);

typedef std::function<const float*(size_t /*y*/)> GetConstRow;
typedef std::function<float*(size_t /*y*/)> GetRow;

// 2D Gaussian with zero-pad boundary handling and runtime independent of sigma.
Status FastGaussian(const RecursiveGaussian& rg, size_t xsize, size_t ysize,
                    const GetConstRow& in, const GetRow& temp,
                    const GetRow& out, ThreadPool* pool = nullptr);

}  // namespace jxl

#endif  // LIB_JXL_GAUSS_BLUR_H_
