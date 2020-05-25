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

#ifndef JXL_DEC_DCT_H_
#define JXL_DEC_DCT_H_

// IDCT interface.

#include "jxl/base/data_parallel.h"
#include "jxl/image.h"

namespace jxl {

typedef void IDct8Func(const size_t xsize_blocks, const size_t ysize_blocks,
                       const ImageF& dequantized, ThreadPool* pool,
                       ImageF* JXL_RESTRICT pixels);
IDct8Func* ChooseIDct8();

// Fills a preallocated N*W x N*H `idct` with NxN blocks produced by
// ComputeTransposedScaledIDCT() from the (N*N)x1 blocks of `dct`.
// REQUIRES: dct.xsize() == N*N*W, dct.ysize() == H
typedef void TransposedScaledIDCTFunc(const Image3F& dct,
                                      Image3F* JXL_RESTRICT idct);
TransposedScaledIDCTFunc* ChooseTransposedScaledIDCT();

}  // namespace jxl

#endif  // JXL_DEC_DCT_H_
