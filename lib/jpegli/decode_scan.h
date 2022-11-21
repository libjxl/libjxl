// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_DECODE_SCAN_H_
#define LIB_JPEGLI_DECODE_SCAN_H_

/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <jpeglib.h>
/* clang-format on */

namespace jpegli {

// Returns true if [data, data + len) contains a valid entropy coded scan, and
// sets *pos to the offset of the end of the scan data.
bool ProcessScan(j_decompress_ptr cinfo, const uint8_t* data, size_t len,
                 size_t* pos);

}  // namespace jpegli

#endif  // LIB_JPEGLI_DECODE_SCAN_H_
