// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_ADAPTIVE_QUANTIZATION_H_
#define LIB_JPEGLI_ADAPTIVE_QUANTIZATION_H_

#include <stddef.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/image.h"

namespace jpegli {

// Returns an image subsampled by kBlockDim in each direction. If the value
// at pixel (x,y) in the returned image is greater than 1.0, it means that
// more fine-grained quantization should be used in the corresponding block
// of the input image, while a value less than 1.0 indicates that less
// fine-grained quantization should be enough. Returns a mask, too, which
// can later be used to make better decisions about ac strategy.
jxl::ImageF InitialQuantField(float butteraugli_target,
                              const jxl::ImageF& opsin_y, jxl::ThreadPool* pool,
                              float rescale);

float InitialQuantDC(float butteraugli_target);

}  // namespace jpegli

#endif  // LIB_JPEGLI_ADAPTIVE_QUANTIZATION_H_
