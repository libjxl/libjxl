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

#ifndef JXL_ENC_DCT_H_
#define JXL_ENC_DCT_H_

// DCT interface.

#include "jxl/base/data_parallel.h"
#include "jxl/image.h"

namespace jxl {

void TransposedScaledDCT8(float* block);

ImageF Dct8(const ImageF& image);

// Fills a preallocated (N*N)*W x H `dct` with (N*N)x1 blocks produced by
// ComputeTransposedScaledDCT() from the corresponding NxN block of
// `image`. Note that `dct` coefficients are scaled by 1 / (N*N), so that
// ComputeTransposedScaledIDCT applied to each block of TransposedScaledIDCT
// will return the original input.
// REQUIRES: image.xsize() == N*W, image.ysize() == N*H
void TransposedScaledDCT(const Image3F& image, Image3F* JXL_RESTRICT dct);

}  // namespace jxl

#endif  // JXL_ENC_DCT_H_
