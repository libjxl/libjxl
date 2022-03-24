// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_EXTRAS_DEC_NPY_H_
#define LIB_EXTRAS_DEC_NPY_H_

// Decodes a subset of NPY arrays into memory.

#include <stddef.h>
#include <stdint.h>

// TODO(janwas): workaround for incorrect Win64 codegen (cause unknown)
#include <hwy/highway.h>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"

namespace jxl {
namespace extras {

// Decodes `bytes` into `ppf`. color_hints may specify "color_space", which
// defaults to sRGB.
Status DecodeImageNPY(Span<const uint8_t> bytes, const ColorHints& color_hints,
                      const SizeConstraints& constraints, PackedPixelFile* ppf);

}  // namespace extras
}  // namespace jxl

#endif  // LIB_EXTRAS_DEC_NPY_H_
