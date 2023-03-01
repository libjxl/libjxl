// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_COLOR_TRANSFORM_H_
#define LIB_JPEGLI_COLOR_TRANSFORM_H_

/* clang-format off */
#include <stdio.h>
#include <jpeglib.h>
/* clang-format on */

#include "lib/jxl/base/compiler_specific.h"

namespace jpegli {

void YCCKToCMYK(float* JXL_RESTRICT row0, float* JXL_RESTRICT row1,
                float* JXL_RESTRICT row2, float* JXL_RESTRICT row3,
                size_t xsize);

void YCbCrToRGB(float* JXL_RESTRICT row0, float* JXL_RESTRICT row1,
                float* JXL_RESTRICT row2, size_t xsize);

void ChooseColorTransform(j_compress_ptr cinfo);

}  // namespace jpegli

#endif  // LIB_JPEGLI_COLOR_TRANSFORM_H_
