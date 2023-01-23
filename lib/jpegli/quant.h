// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_QUANT_H_
#define LIB_JPEGLI_QUANT_H_

/* clang-format off */
#include <stdio.h>
#include <jpeglib.h>
/* clang-format on */

namespace jpegli {

enum QuantMode {
  QUANT_XYB,
  QUANT_YUV,
  QUANT_STD,
  NUM_QUANT_MODES,
};

void AddJpegQuantMatrices(j_compress_ptr cinfo, QuantMode mode, float dc_scale,
                          float ac_scale, float* qm);

}  // namespace jpegli

#endif  // LIB_JPEGLI_QUANT_H_
