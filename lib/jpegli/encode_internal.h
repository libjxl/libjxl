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

#include "lib/jpegli/bit_writer.h"
#include "lib/jpegli/common_internal.h"
#include "lib/jpegli/encode.h"

namespace jpegli {

constexpr unsigned char kICCSignature[12] = {
    0x49, 0x43, 0x43, 0x5F, 0x50, 0x52, 0x4F, 0x46, 0x49, 0x4C, 0x45, 0x00};
constexpr int kICCMarker = JPEG_APP0 + 2;

struct JPEGHuffmanCode {
  // Bit length histogram.
  std::array<uint32_t, kJpegHuffmanMaxBitLength + 1> counts = {};
  // Symbol values sorted by increasing bit lengths.
  std::array<uint32_t, kJpegHuffmanAlphabetSize + 1> values = {};
  // The index of the Huffman code in the current set of Huffman codes. For AC
  // component Huffman codes, 0x10 is added to the index.
  int slot_id = 0;
  boolean sent_table = FALSE;
};

// DCTCodingState: maximum number of correction bits to buffer
const int kJPEGMaxCorrectionBits = 1u << 16;

static constexpr float kDCBias = 128.0f;

constexpr int kDefaultProgressiveLevel = 2;

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
  jpegli::RowBuffer<float> input_buffer[jpegli::kMaxComponents];
  jpegli::RowBuffer<float> downsampler_output[jpegli::kMaxComponents];
  jpegli::RowBuffer<float>* raw_data[jpegli::kMaxComponents];
  float distance = 1.0;
  bool force_baseline = true;
  bool xyb_mode = false;
  bool use_std_tables = false;
  bool use_adaptive_quantization = true;
  int progressive_level = jpegli::kDefaultProgressiveLevel;
  size_t xsize_blocks = 0;
  size_t ysize_blocks = 0;
  size_t blocks_per_iMCU_row = 0;
  std::vector<jpegli::ScanCodingInfo> scan_coding_info;
  std::vector<std::vector<uint8_t>> special_markers;
  uint8_t* next_marker_byte = nullptr;
  JpegliDataType data_type = JPEGLI_TYPE_UINT8;
  JpegliEndianness endianness = JPEGLI_NATIVE_ENDIAN;
  void (*input_method)(const uint8_t* row_in, size_t len,
                       float* row_out[jpegli::kMaxComponents]);
  void (*color_transform)(float* row[jpegli::kMaxComponents], size_t len);
  void (*downsample_method[jpegli::kMaxComponents])(
      float* rows_in[MAX_SAMP_FACTOR], size_t len, float* row_out);
  float* quant_mul[jpegli::kMaxComponents];
  float zero_bias_mul[jpegli::kMaxComponents];
  int h_factor[jpegli::kMaxComponents];
  int v_factor[jpegli::kMaxComponents];
  std::vector<jpegli::JPEGHuffmanCode> huffman_codes;
  jpegli::HuffmanCodeTable huff_tables[8];
  std::array<jpegli::HuffmanCodeTable, jpegli::kMaxHuffmanTables> dc_huff_table;
  std::array<jpegli::HuffmanCodeTable, jpegli::kMaxHuffmanTables> ac_huff_table;
  float* diff_buffer;
  jpegli::RowBuffer<float> fuzzy_erosion_tmp;
  jpegli::RowBuffer<float> pre_erosion;
  jpegli::RowBuffer<float> quant_field;
  jvirt_barray_ptr* coeff_buffers = nullptr;
  size_t next_input_row;
  size_t next_iMCU_row;
  size_t last_dht_index;
  size_t last_restart_interval;
  JCOEF last_dc_coeff[MAX_COMPS_IN_SCAN];
  jpegli::JpegBitWriter bw;
  float* dct_buffer;
  int32_t* block_tmp;
};

#endif  // LIB_JPEGLI_ENCODE_INTERNAL_H_
