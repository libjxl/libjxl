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

#ifndef JXL_GABORISH_H_
#define JXL_GABORISH_H_

// Linear smoothing (3x3 convolution) for deblocking without too much blur.

#include <stdint.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/image.h"

namespace jxl {

// Used in encoder to reduce the impact of the decoder's smoothing.
// This is approximate and slow (unoptimized 5x5 convolution).
Image3F GaborishInverse(const Image3F& opsin, double mul, ThreadPool* pool);

// Caller must ensure LoopFilter.gab == true. Otherwise, it is faster and
// simpler to skip the convolution. out must be preallocated.
void ConvolveGaborish(const ImageF& in, float weight1, float weight2,
                      ThreadPool* pool, ImageF* JXL_RESTRICT out);

}  // namespace jxl

#endif  // JXL_GABORISH_H_
