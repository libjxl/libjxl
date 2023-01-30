// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_ENCODE_INTERNAL_H_
#define LIB_JPEGLI_ENCODE_INTERNAL_H_

/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <jpeglib.h>
/* clang-format on */

#include <array>
#include <vector>

#include "lib/jpegli/common_internal.h"
#include "lib/jpegli/encode.h"
#include "lib/jxl/image.h"

namespace jpegli {

struct JPEGHuffmanCode {
  // Bit length histogram.
  std::array<uint32_t, kJpegHuffmanMaxBitLength + 1> counts = {};
  // Symbol values sorted by increasing bit lengths.
  std::array<uint32_t, kJpegHuffmanAlphabetSize + 1> values = {};
  // The index of the Huffman code in the current set of Huffman codes. For AC
  // component Huffman codes, 0x10 is added to the index.
  int slot_id = 0;
  // Set to true if this Huffman code is the last one within its marker segment
  bool is_last = true;
};

// DCTCodingState: maximum number of correction bits to buffer
const int kJPEGMaxCorrectionBits = 1u << 16;

struct HuffmanCodeTable {
  int depth[256];
  int code[256];
};

struct ScanCodingInfo {
  uint32_t dc_tbl_idx[MAX_COMPS_IN_SCAN];
  uint32_t ac_tbl_idx[MAX_COMPS_IN_SCAN];
  // Number of Huffman codes defined in the DHT segment preceding this scan.
  size_t num_huffman_codes = 0;
};

typedef int16_t coeff_t;

}  // namespace jpegli

struct jpeg_comp_master {
  jxl::Image3F input;
  float distance = 1.0;
  bool xyb_mode = false;
  bool use_std_tables = false;
  bool use_adaptive_quantization = true;
  int progressive_level = 2;
  size_t xsize_blocks = 0;
  size_t ysize_blocks = 0;
  std::vector<jpegli::ScanCodingInfo> scan_coding_info;
  std::vector<std::vector<uint8_t>> special_markers;
  std::vector<uint8_t>* cur_marker_data = nullptr;
  JpegliDataType data_type = JPEGLI_TYPE_UINT8;
  JpegliEndianness endianness = JPEGLI_NATIVE_ENDIAN;
  std::array<jpegli::HuffmanCodeTable, jpegli::kMaxHuffmanTables> dc_huff_table;
  std::array<jpegli::HuffmanCodeTable, jpegli::kMaxHuffmanTables> ac_huff_table;
};

#endif  // LIB_JPEGLI_ENCODE_INTERNAL_H_
