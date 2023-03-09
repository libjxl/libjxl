// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_BITSTREAM_H_
#define LIB_JPEGLI_BITSTREAM_H_

#include <initializer_list>
#include <vector>

#include "lib/jpegli/encode_internal.h"

namespace jpegli {

void EncodeAPP0(j_compress_ptr cinfo);
void EncodeAPP14(j_compress_ptr cinfo);
void EncodeSOF(j_compress_ptr cinfo);
void EncodeSOS(j_compress_ptr cinfo, int scan_index);
void EncodeDHT(j_compress_ptr cinfo, const JPEGHuffmanCode* huffman_codes,
               size_t num_huffman_codes, bool pre_shifted = false);
void EncodeDQT(j_compress_ptr cinfo);
bool EncodeDRI(j_compress_ptr cinfo);

bool EncodeScan(j_compress_ptr cinfo, int scan_index);

void EncodeSingleScan(j_compress_ptr cinfo);

void WriteiMCURow(j_compress_ptr cinfo);

}  // namespace jpegli

#endif  // LIB_JPEGLI_BITSTREAM_H_
