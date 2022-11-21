// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_DECODE_MARKER_H_
#define LIB_JPEGLI_DECODE_MARKER_H_

/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <jpeglib.h>
/* clang-format on */

namespace jpegli {

// Returns true if [data, data + len) contains a valid marker segment (it does
// not need to be at the start of data), and sets *pos to the offset of the end
// of the marker segment and fills in the relevant parts of cinfo.
bool ProcessMarker(j_decompress_ptr cinfo, const uint8_t* data, size_t len,
                   size_t* pos);

}  // namespace jpegli

#endif  // LIB_JPEGLI_DECODE_MARKER_H_
