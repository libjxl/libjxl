// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/bitstream.h"

#include "lib/jpegli/entropy_coding.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jxl/base/bits.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jpegli/bitstream.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::AndNot;
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::GetLane;

int NumNonZero8x8ExceptDC(const coeff_t* block) {
  const HWY_CAPPED(coeff_t, 8) di;

  const auto zero = Zero(di);
  // Add FFFF for every zero coefficient, negate to get #zeros.
  auto neg_sum_zero = zero;

  {
    // First row has DC, so mask
    const size_t y = 0;
    HWY_ALIGN const coeff_t dc_mask_lanes[8] = {-1};

    for (size_t x = 0; x < 8; x += Lanes(di)) {
      const auto dc_mask = Load(di, dc_mask_lanes + x);

      // DC counts as zero so we don't include it in nzeros.
      const auto coef = AndNot(dc_mask, Load(di, &block[y * 8 + x]));

      neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
    }
  }

  // Remaining rows: no mask
  for (size_t y = 1; y < 8; y++) {
    for (size_t x = 0; x < 8; x += Lanes(di)) {
      const auto coef = Load(di, &block[y * 8 + x]);
      neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
    }
  }

  // We want 64 - sum_zero, add because neg_sum_zero is already negated.
  return kDCTBlockSize + GetLane(SumOfLanes(di, neg_sum_zero));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jpegli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jpegli {
namespace {
HWY_EXPORT(NumNonZero8x8ExceptDC);

// JpegBitWriter: buffer size
const size_t kJpegBitWriterChunkSize = 16384;

// Handles the packing of bits into output bytes.
struct JpegBitWriter {
  j_compress_ptr cinfo;
  std::vector<uint8_t> buffer;
  uint8_t* data;
  size_t pos;
  uint64_t put_buffer;
  int put_bits;
  bool healthy;
};

// Holds data that is buffered between 8x8 blocks in progressive mode.
struct DCTCodingState {
  // The run length of end-of-band symbols in a progressive scan.
  int eob_run_;
  // The huffman table to be used when flushing the state.
  HuffmanCodeTable* cur_ac_huff_;
  // The sequence of currently buffered refinement bits for a successive
  // approximation scan (one where Ah > 0).
  std::vector<int> refinement_bits_;
};

// Returns non-zero if and only if x has a zero byte, i.e. one of
// x & 0xff, x & 0xff00, ..., x & 0xff00000000000000 is zero.
static JXL_INLINE uint64_t HasZeroByte(uint64_t x) {
  return (x - 0x0101010101010101ULL) & ~x & 0x8080808080808080ULL;
}

void JpegBitWriterInit(JpegBitWriter* bw, j_compress_ptr cinfo) {
  bw->cinfo = cinfo;
  bw->buffer.resize(kJpegBitWriterChunkSize);
  bw->data = bw->buffer.data();
  bw->pos = 0;
  bw->put_buffer = 0;
  bw->put_bits = 64;
  bw->healthy = true;
}

static JXL_INLINE void Reserve(JpegBitWriter* bw, size_t n_bytes) {
  if (JXL_UNLIKELY((bw->pos + n_bytes) > kJpegBitWriterChunkSize)) {
    WriteOutput(bw->cinfo, bw->data, bw->pos);
    bw->data = bw->buffer.data();
    bw->pos = 0;
  }
}

/**
 * Writes the given byte to the output, writes an extra zero if byte is 0xFF.
 *
 * This method is "careless" - caller must make sure that there is enough
 * space in the output buffer. Emits up to 2 bytes to buffer.
 */
static JXL_INLINE void EmitByte(JpegBitWriter* bw, int byte) {
  bw->data[bw->pos++] = byte;
  if (byte == 0xFF) bw->data[bw->pos++] = 0;
}

static JXL_INLINE void DischargeBitBuffer(JpegBitWriter* bw) {
  // At this point we are ready to emit the most significant 6 bytes of
  // put_buffer_ to the output.
  // The JPEG format requires that after every 0xff byte in the entropy
  // coded section, there is a zero byte, therefore we first check if any of
  // the 6 most significant bytes of put_buffer_ is 0xFF.
  Reserve(bw, 12);
  if (HasZeroByte(~bw->put_buffer | 0xFFFF)) {
    // We have a 0xFF byte somewhere, examine each byte and append a zero
    // byte if necessary.
    EmitByte(bw, (bw->put_buffer >> 56) & 0xFF);
    EmitByte(bw, (bw->put_buffer >> 48) & 0xFF);
    EmitByte(bw, (bw->put_buffer >> 40) & 0xFF);
    EmitByte(bw, (bw->put_buffer >> 32) & 0xFF);
    EmitByte(bw, (bw->put_buffer >> 24) & 0xFF);
    EmitByte(bw, (bw->put_buffer >> 16) & 0xFF);
  } else {
    // We don't have any 0xFF bytes, output all 6 bytes without checking.
    bw->data[bw->pos] = (bw->put_buffer >> 56) & 0xFF;
    bw->data[bw->pos + 1] = (bw->put_buffer >> 48) & 0xFF;
    bw->data[bw->pos + 2] = (bw->put_buffer >> 40) & 0xFF;
    bw->data[bw->pos + 3] = (bw->put_buffer >> 32) & 0xFF;
    bw->data[bw->pos + 4] = (bw->put_buffer >> 24) & 0xFF;
    bw->data[bw->pos + 5] = (bw->put_buffer >> 16) & 0xFF;
    bw->pos += 6;
  }
  bw->put_buffer <<= 48;
  bw->put_bits += 48;
}

static JXL_INLINE void WriteBits(JpegBitWriter* bw, int nbits, uint64_t bits) {
  // This is an optimization; if everything goes well,
  // then |nbits| is positive; if non-existing Huffman symbol is going to be
  // encoded, its length should be zero; later encoder could check the
  // "health" of JpegBitWriter.
  if (nbits == 0) {
    bw->healthy = false;
    return;
  }
  bw->put_bits -= nbits;
  bw->put_buffer |= (bits << bw->put_bits);
  if (bw->put_bits <= 16) DischargeBitBuffer(bw);
}

void EmitMarker(JpegBitWriter* bw, int marker) {
  Reserve(bw, 2);
  JXL_DASSERT(marker != 0xFF);
  bw->data[bw->pos++] = 0xFF;
  bw->data[bw->pos++] = marker;
}

void JumpToByteBoundary(JpegBitWriter* bw) {
  size_t n_bits = bw->put_bits & 7u;
  uint8_t pad_pattern = (1u << n_bits) - 1;

  Reserve(bw, 16);

  while (bw->put_bits <= 56) {
    int c = (bw->put_buffer >> 56) & 0xFF;
    EmitByte(bw, c);
    bw->put_buffer <<= 8;
    bw->put_bits += 8;
  }
  if (bw->put_bits < 64) {
    int pad_mask = 0xFFu >> (64 - bw->put_bits);
    int c = ((bw->put_buffer >> 56) & ~pad_mask) | pad_pattern;
    EmitByte(bw, c);
  }
  bw->put_buffer = 0;
  bw->put_bits = 64;
}

void JpegBitWriterFinish(JpegBitWriter* bw) {
  if (bw->pos == 0) return;
  WriteOutput(bw->cinfo, bw->data, bw->pos);
  bw->data = nullptr;
  bw->pos = 0;
}

void DCTCodingStateInit(DCTCodingState* s) {
  s->eob_run_ = 0;
  s->cur_ac_huff_ = nullptr;
  s->refinement_bits_.clear();
  s->refinement_bits_.reserve(kJPEGMaxCorrectionBits);
}

static JXL_INLINE void WriteSymbol(int symbol, const HuffmanCodeTable* table,
                                   JpegBitWriter* bw) {
  WriteBits(bw, table->depth[symbol], table->code[symbol]);
}

// Emit all buffered data to the bit stream using the given Huffman code and
// bit writer.
static JXL_INLINE void Flush(DCTCodingState* s, JpegBitWriter* bw) {
  if (s->eob_run_ > 0) {
    int nbits = jxl::FloorLog2Nonzero<uint32_t>(s->eob_run_);
    int symbol = nbits << 4u;
    WriteSymbol(symbol, s->cur_ac_huff_, bw);
    if (nbits > 0) {
      WriteBits(bw, nbits, s->eob_run_ & ((1 << nbits) - 1));
    }
    s->eob_run_ = 0;
  }
  for (size_t i = 0; i < s->refinement_bits_.size(); ++i) {
    WriteBits(bw, 1, s->refinement_bits_[i]);
  }
  s->refinement_bits_.clear();
}

// Buffer some more data at the end-of-band (the last non-zero or newly
// non-zero coefficient within the [Ss, Se] spectral band).
static JXL_INLINE void BufferEndOfBand(DCTCodingState* s,
                                       HuffmanCodeTable* ac_huff,
                                       const std::vector<int>* new_bits,
                                       JpegBitWriter* bw) {
  if (s->eob_run_ == 0) {
    s->cur_ac_huff_ = ac_huff;
  }
  ++s->eob_run_;
  if (new_bits) {
    s->refinement_bits_.insert(s->refinement_bits_.end(), new_bits->begin(),
                               new_bits->end());
  }
  if (s->eob_run_ == 0x7FFF ||
      s->refinement_bits_.size() > kJPEGMaxCorrectionBits - kDCTBlockSize + 1) {
    Flush(s, bw);
  }
}

bool BuildHuffmanCodeTable(const JPEGHuffmanCode& huff,
                           HuffmanCodeTable* table) {
  int huff_code[kJpegHuffmanAlphabetSize];
  // +1 for a sentinel element.
  uint32_t huff_size[kJpegHuffmanAlphabetSize + 1];
  int p = 0;
  for (size_t l = 1; l <= kJpegHuffmanMaxBitLength; ++l) {
    int i = huff.counts[l];
    if (p + i > kJpegHuffmanAlphabetSize + 1) {
      return false;
    }
    while (i--) huff_size[p++] = l;
  }

  if (p == 0) {
    return true;
  }

  // Reuse sentinel element.
  int last_p = p - 1;
  huff_size[last_p] = 0;

  int code = 0;
  uint32_t si = huff_size[0];
  p = 0;
  while (huff_size[p]) {
    while ((huff_size[p]) == si) {
      huff_code[p++] = code;
      code++;
    }
    code <<= 1;
    si++;
  }
  for (p = 0; p < last_p; p++) {
    int i = huff.values[p];
    table->depth[i] = huff_size[p];
    table->code[i] = huff_code[p];
  }
  return true;
}

bool EncodeDCTBlockSequential(const coeff_t* coeffs, HuffmanCodeTable* dc_huff,
                              HuffmanCodeTable* ac_huff, int num_zero_runs,
                              coeff_t* last_dc_coeff, JpegBitWriter* bw) {
  coeff_t temp2;
  coeff_t temp;
  temp2 = coeffs[0];
  temp = temp2 - *last_dc_coeff;
  *last_dc_coeff = temp2;
  temp2 = temp;
  if (temp < 0) {
    temp = -temp;
    if (temp < 0) return false;
    temp2--;
  }
  int dc_nbits = (temp == 0) ? 0 : (jxl::FloorLog2Nonzero<uint32_t>(temp) + 1);
  WriteSymbol(dc_nbits, dc_huff, bw);
  if (dc_nbits >= 12) return false;
  if (dc_nbits > 0) {
    WriteBits(bw, dc_nbits, temp2 & ((1u << dc_nbits) - 1));
  }
  int r = 0;
  for (int k = 1; k < 64; ++k) {
    if ((temp = coeffs[kJPEGNaturalOrder[k]]) == 0) {
      r++;
      continue;
    }
    if (temp < 0) {
      temp = -temp;
      if (temp < 0) return false;
      temp2 = ~temp;
    } else {
      temp2 = temp;
    }
    while (r > 15) {
      WriteSymbol(0xf0, ac_huff, bw);
      r -= 16;
    }
    int ac_nbits = jxl::FloorLog2Nonzero<uint32_t>(temp) + 1;
    if (ac_nbits >= 16) return false;
    int symbol = (r << 4u) + ac_nbits;
    WriteSymbol(symbol, ac_huff, bw);
    WriteBits(bw, ac_nbits, temp2 & ((1 << ac_nbits) - 1));
    r = 0;
  }
  for (int i = 0; i < num_zero_runs; ++i) {
    WriteSymbol(0xf0, ac_huff, bw);
    r -= 16;
  }
  if (r > 0) {
    WriteSymbol(0, ac_huff, bw);
  }
  return true;
}

bool EncodeDCTBlockProgressive(const coeff_t* coeffs, HuffmanCodeTable* dc_huff,
                               HuffmanCodeTable* ac_huff, int Ss, int Se,
                               int Al, int num_zero_runs,
                               DCTCodingState* coding_state,
                               coeff_t* last_dc_coeff, JpegBitWriter* bw) {
  bool eob_run_allowed = Ss > 0;
  coeff_t temp2;
  coeff_t temp;
  if (Ss == 0) {
    temp2 = coeffs[0] >> Al;
    temp = temp2 - *last_dc_coeff;
    *last_dc_coeff = temp2;
    temp2 = temp;
    if (temp < 0) {
      temp = -temp;
      if (temp < 0) return false;
      temp2--;
    }
    int nbits = (temp == 0) ? 0 : (jxl::FloorLog2Nonzero<uint32_t>(temp) + 1);
    WriteSymbol(nbits, dc_huff, bw);
    if (nbits > 0) {
      WriteBits(bw, nbits, temp2 & ((1 << nbits) - 1));
    }
    ++Ss;
  }
  if (Ss > Se) {
    return true;
  }
  int r = 0;
  for (int k = Ss; k <= Se; ++k) {
    if ((temp = coeffs[kJPEGNaturalOrder[k]]) == 0) {
      r++;
      continue;
    }
    if (temp < 0) {
      temp = -temp;
      if (temp < 0) return false;
      temp >>= Al;
      temp2 = ~temp;
    } else {
      temp >>= Al;
      temp2 = temp;
    }
    if (temp == 0) {
      r++;
      continue;
    }
    Flush(coding_state, bw);
    while (r > 15) {
      WriteSymbol(0xf0, ac_huff, bw);
      r -= 16;
    }
    int nbits = jxl::FloorLog2Nonzero<uint32_t>(temp) + 1;
    int symbol = (r << 4u) + nbits;
    WriteSymbol(symbol, ac_huff, bw);
    WriteBits(bw, nbits, temp2 & ((1 << nbits) - 1));
    r = 0;
  }
  if (num_zero_runs > 0) {
    Flush(coding_state, bw);
    for (int i = 0; i < num_zero_runs; ++i) {
      WriteSymbol(0xf0, ac_huff, bw);
      r -= 16;
    }
  }
  if (r > 0) {
    BufferEndOfBand(coding_state, ac_huff, nullptr, bw);
    if (!eob_run_allowed) {
      Flush(coding_state, bw);
    }
  }
  return true;
}

bool EncodeRefinementBits(const coeff_t* coeffs, HuffmanCodeTable* ac_huff,
                          int Ss, int Se, int Al, DCTCodingState* coding_state,
                          JpegBitWriter* bw) {
  bool eob_run_allowed = Ss > 0;
  if (Ss == 0) {
    // Emit next bit of DC component.
    WriteBits(bw, 1, (coeffs[0] >> Al) & 1);
    ++Ss;
  }
  if (Ss > Se) {
    return true;
  }
  int abs_values[kDCTBlockSize];
  int eob = 0;
  for (int k = Ss; k <= Se; k++) {
    const coeff_t abs_val = std::abs(coeffs[kJPEGNaturalOrder[k]]);
    abs_values[k] = abs_val >> Al;
    if (abs_values[k] == 1) {
      eob = k;
    }
  }
  int r = 0;
  std::vector<int> refinement_bits;
  refinement_bits.reserve(kDCTBlockSize);
  for (int k = Ss; k <= Se; k++) {
    if (abs_values[k] == 0) {
      r++;
      continue;
    }
    while (r > 15 && k <= eob) {
      Flush(coding_state, bw);
      WriteSymbol(0xf0, ac_huff, bw);
      r -= 16;
      for (int bit : refinement_bits) {
        WriteBits(bw, 1, bit);
      }
      refinement_bits.clear();
    }
    if (abs_values[k] > 1) {
      refinement_bits.push_back(abs_values[k] & 1u);
      continue;
    }
    Flush(coding_state, bw);
    int symbol = (r << 4u) + 1;
    int new_non_zero_bit = (coeffs[kJPEGNaturalOrder[k]] < 0) ? 0 : 1;
    WriteSymbol(symbol, ac_huff, bw);
    WriteBits(bw, 1, new_non_zero_bit);
    for (int bit : refinement_bits) {
      WriteBits(bw, 1, bit);
    }
    refinement_bits.clear();
    r = 0;
  }
  if (r > 0 || !refinement_bits.empty()) {
    BufferEndOfBand(coding_state, ac_huff, &refinement_bits, bw);
    if (!eob_run_allowed) {
      Flush(coding_state, bw);
    }
  }
  return true;
}

}  // namespace

void WriteOutput(j_compress_ptr cinfo, const uint8_t* buf, size_t bufsize) {
  size_t pos = 0;
  while (pos < bufsize) {
    if (cinfo->dest->free_in_buffer == 0 &&
        !(*cinfo->dest->empty_output_buffer)(cinfo)) {
      JPEGLI_ERROR("Destination suspension is not supported.");
    }
    size_t len = std::min<size_t>(cinfo->dest->free_in_buffer, bufsize - pos);
    memcpy(cinfo->dest->next_output_byte, buf + pos, len);
    pos += len;
    cinfo->dest->free_in_buffer -= len;
    cinfo->dest->next_output_byte += len;
  }
}

void WriteOutput(j_compress_ptr cinfo, const std::vector<uint8_t>& bytes) {
  WriteOutput(cinfo, bytes.data(), bytes.size());
}

void WriteOutput(j_compress_ptr cinfo, std::initializer_list<uint8_t> bytes) {
  WriteOutput(cinfo, bytes.begin(), bytes.size());
}

void EncodeAPP0(j_compress_ptr cinfo) {
  WriteOutput(cinfo,
              {0xff, 0xe0, 0, 16, 'J', 'F', 'I', 'F', '\0',
               cinfo->JFIF_major_version, cinfo->JFIF_minor_version,
               cinfo->density_unit, static_cast<uint8_t>(cinfo->X_density >> 8),
               static_cast<uint8_t>(cinfo->X_density & 0xff),
               static_cast<uint8_t>(cinfo->Y_density >> 8),
               static_cast<uint8_t>(cinfo->Y_density & 0xff), 0, 0});
}

void EncodeAPP14(j_compress_ptr cinfo) {
  uint8_t color_transform = cinfo->jpeg_color_space == JCS_YCbCr  ? 1
                            : cinfo->jpeg_color_space == JCS_YCCK ? 2
                                                                  : 0;
  WriteOutput(cinfo, {0xff, 0xee, 0, 14, 'A', 'd', 'o', 'b', 'e', 0, 100, 0, 0,
                      0, 0, color_transform});
}

void EncodeSOF(j_compress_ptr cinfo) {
  const uint8_t marker = cinfo->progressive_mode ? 0xc2 : 0xc1;
  const size_t n_comps = cinfo->num_components;
  const size_t marker_len = 8 + 3 * n_comps;
  std::vector<uint8_t> data(marker_len + 2);
  size_t pos = 0;
  data[pos++] = 0xFF;
  data[pos++] = marker;
  data[pos++] = marker_len >> 8u;
  data[pos++] = marker_len & 0xFFu;
  data[pos++] = kJpegPrecision;
  data[pos++] = cinfo->image_height >> 8u;
  data[pos++] = cinfo->image_height & 0xFFu;
  data[pos++] = cinfo->image_width >> 8u;
  data[pos++] = cinfo->image_width & 0xFFu;
  data[pos++] = n_comps;
  for (size_t i = 0; i < n_comps; ++i) {
    jpeg_component_info* comp = &cinfo->comp_info[i];
    data[pos++] = comp->component_id;
    data[pos++] = ((comp->h_samp_factor << 4u) | (comp->v_samp_factor));
    const uint32_t quant_idx = comp->quant_tbl_no;
    if (cinfo->quant_tbl_ptrs[quant_idx] == nullptr) {
      JPEGLI_ERROR("Invalid component quant table index %u.", quant_idx);
    }
    data[pos++] = quant_idx;
  }
  WriteOutput(cinfo, data);
}

void EncodeSOS(j_compress_ptr cinfo, int scan_index) {
  const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
  const ScanCodingInfo& sci = cinfo->master->scan_coding_info[scan_index];
  const size_t marker_len = 6 + 2 * scan_info->comps_in_scan;
  std::vector<uint8_t> data(marker_len + 2);
  size_t pos = 0;
  data[pos++] = 0xFF;
  data[pos++] = 0xDA;
  data[pos++] = marker_len >> 8u;
  data[pos++] = marker_len & 0xFFu;
  data[pos++] = scan_info->comps_in_scan;
  for (int i = 0; i < scan_info->comps_in_scan; ++i) {
    int comp_idx = scan_info->component_index[i];
    data[pos++] = cinfo->comp_info[comp_idx].component_id;
    data[pos++] = (sci.dc_tbl_idx[i] << 4u) + (sci.ac_tbl_idx[i] - 4);
  }
  data[pos++] = scan_info->Ss;
  data[pos++] = scan_info->Se;
  data[pos++] = ((scan_info->Ah << 4u) | (scan_info->Al));
  WriteOutput(cinfo, data);
}

void EncodeDHT(j_compress_ptr cinfo, const JPEGHuffmanCode* huffman_codes,
               size_t num_huffman_codes) {
  if (num_huffman_codes == 0) {
    return;
  }

  size_t marker_len = 2;
  for (size_t i = 0; i < num_huffman_codes; ++i) {
    const JPEGHuffmanCode& huff = huffman_codes[i];
    if (huff.sent_table) continue;
    marker_len += kJpegHuffmanMaxBitLength;
    for (size_t j = 0; j < huff.counts.size(); ++j) {
      marker_len += huff.counts[j];
    }
  }
  if (marker_len == 2) {
    return;
  }
  std::vector<uint8_t> data(marker_len + 2);
  size_t pos = 0;
  data[pos++] = 0xFF;
  data[pos++] = 0xC4;
  data[pos++] = marker_len >> 8u;
  data[pos++] = marker_len & 0xFFu;
  for (size_t i = 0; i < num_huffman_codes; ++i) {
    const JPEGHuffmanCode& huff = huffman_codes[i];
    size_t index = huff.slot_id;
    HuffmanCodeTable* huff_table;
    if (index & 0x10) {
      huff_table = &cinfo->master->huff_tables[index - 12];
    } else {
      huff_table = &cinfo->master->huff_tables[index];
    }
    // TODO(eustas): cache
    // TODO(eustas): set up non-existing symbols
    if (!BuildHuffmanCodeTable(huff, huff_table)) {
      JPEGLI_ERROR("Failed to build Huffman code table.");
    }
    if (huff.sent_table) continue;
    size_t total_count = 0;
    size_t max_length = 0;
    for (size_t i = 0; i < huff.counts.size(); ++i) {
      if (huff.counts[i] != 0) {
        max_length = i;
      }
      total_count += huff.counts[i];
    }
    --total_count;
    data[pos++] = huff.slot_id;
    for (size_t i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
      data[pos++] = (i == max_length ? huff.counts[i] - 1 : huff.counts[i]);
    }
    for (size_t i = 0; i < total_count; ++i) {
      data[pos++] = huff.values[i];
    }
  }
  WriteOutput(cinfo, data);
}

void EncodeDQT(j_compress_ptr cinfo) {
  uint8_t data[4 + NUM_QUANT_TBLS * (1 + 2 * DCTSIZE2)];  // 520 bytes
  size_t pos = 0;
  data[pos++] = 0xFF;
  data[pos++] = 0xDB;
  pos += 2;  // Length will be filled in later.
  for (int i = 0; i < NUM_QUANT_TBLS; ++i) {
    JQUANT_TBL* quant_table = cinfo->quant_tbl_ptrs[i];
    if (quant_table == nullptr || quant_table->sent_table) continue;
    int precision = 0;
    for (size_t k = 0; k < DCTSIZE2; ++k) {
      if (quant_table->quantval[k] > 255) precision = 1;
    }
    data[pos++] = (precision << 4) + i;
    for (size_t j = 0; j < DCTSIZE2; ++j) {
      int val_idx = kJPEGNaturalOrder[j];
      int val = quant_table->quantval[val_idx];
      if (val == 0) {
        JPEGLI_ERROR("Invalid quantval 0.");
      }
      if (precision) {
        data[pos++] = val >> 8;
      }
      data[pos++] = val & 0xFFu;
    }
    quant_table->sent_table = TRUE;
  }
  if (pos > 4) {
    data[2] = (pos - 2) >> 8u;
    data[3] = (pos - 2) & 0xFFu;
    WriteOutput(cinfo, data, pos);
  }
}

bool EncodeDRI(j_compress_ptr cinfo) {
  WriteOutput(cinfo, {0xFF, 0xDD, 0, 4,
                      static_cast<uint8_t>(cinfo->restart_interval >> 8),
                      static_cast<uint8_t>(cinfo->restart_interval & 0xFF)});
  return true;
}

bool EncodeScan(j_compress_ptr cinfo, int scan_index) {
  jpeg_comp_master* m = cinfo->master;
  const int restart_interval = cinfo->restart_interval;
  int restarts_to_go = restart_interval;
  int next_restart_marker = 0;

  JpegBitWriter bw;
  JpegBitWriterInit(&bw, cinfo);

  coeff_t last_dc_coeff[MAX_COMPS_IN_SCAN] = {0};
  DCTCodingState coding_state;
  DCTCodingStateInit(&coding_state);

  const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
  const ScanCodingInfo& sci = m->scan_coding_info[scan_index];
  // "Non-interleaved" means color data comes in separate scans, in other words
  // each scan can contain only one color component.
  const bool is_interleaved = (scan_info->comps_in_scan > 1);
  jpeg_component_info* base_comp =
      &cinfo->comp_info[scan_info->component_index[0]];
  // h_group / v_group act as numerators for converting number of blocks to
  // number of MCU. In interleaved mode it is 1, so MCU is represented with
  // max_*_samp_factor blocks. In non-interleaved mode we choose numerator to
  // be the samping factor, consequently MCU is always represented with single
  // block.
  const int h_group = is_interleaved ? 1 : base_comp->h_samp_factor;
  const int v_group = is_interleaved ? 1 : base_comp->v_samp_factor;
  int MCUs_per_row =
      DivCeil(cinfo->image_width * h_group, 8 * cinfo->max_h_samp_factor);
  int MCU_rows =
      DivCeil(cinfo->image_height * v_group, 8 * cinfo->max_v_samp_factor);
  const bool is_progressive = cinfo->progressive_mode;
  const int Al = scan_info->Al;
  const int Ah = scan_info->Ah;
  const int Ss = scan_info->Ss;
  const int Se = scan_info->Se;
  constexpr coeff_t kDummyBlock[DCTSIZE2] = {0};

  JBLOCKARRAY ba[MAX_COMPS_IN_SCAN];
  for (int mcu_y = 0; mcu_y < MCU_rows; ++mcu_y) {
    for (int i = 0; i < scan_info->comps_in_scan; ++i) {
      int comp_idx = scan_info->component_index[i];
      jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
      int n_blocks_y = is_interleaved ? comp->v_samp_factor : 1;
      int by0 = mcu_y * n_blocks_y;
      int block_rows_left = comp->height_in_blocks - by0;
      int max_block_rows = std::min(n_blocks_y, block_rows_left);
      ba[i] = (*cinfo->mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[comp_idx],
          by0, max_block_rows, false);
    }
    for (int mcu_x = 0; mcu_x < MCUs_per_row; ++mcu_x) {
      // Possibly emit a restart marker.
      if (restart_interval > 0 && restarts_to_go == 0) {
        Flush(&coding_state, &bw);
        JumpToByteBoundary(&bw);
        EmitMarker(&bw, 0xD0 + next_restart_marker);
        next_restart_marker += 1;
        next_restart_marker &= 0x7;
        restarts_to_go = restart_interval;
        memset(last_dc_coeff, 0, sizeof(last_dc_coeff));
      }
      // Encode one MCU
      for (int i = 0; i < scan_info->comps_in_scan; ++i) {
        int comp_idx = scan_info->component_index[i];
        jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
        HuffmanCodeTable* dc_huff = &m->huff_tables[sci.dc_tbl_idx[i]];
        HuffmanCodeTable* ac_huff = &m->huff_tables[sci.ac_tbl_idx[i]];
        int n_blocks_y = is_interleaved ? comp->v_samp_factor : 1;
        int n_blocks_x = is_interleaved ? comp->h_samp_factor : 1;
        for (int iy = 0; iy < n_blocks_y; ++iy) {
          for (int ix = 0; ix < n_blocks_x; ++ix) {
            size_t block_y = mcu_y * n_blocks_y + iy;
            size_t block_x = mcu_x * n_blocks_x + ix;
            size_t num_zero_runs = 0;
            const coeff_t* block;
            if (block_x >= comp->width_in_blocks ||
                block_y >= comp->height_in_blocks) {
              block = kDummyBlock;
            } else {
              block = &ba[i][iy][block_x][0];
            }
            bool ok;
            if (!is_progressive) {
              ok = EncodeDCTBlockSequential(block, dc_huff, ac_huff,
                                            num_zero_runs, last_dc_coeff + i,
                                            &bw);
            } else if (Ah == 0) {
              ok = EncodeDCTBlockProgressive(block, dc_huff, ac_huff, Ss, Se,
                                             Al, num_zero_runs, &coding_state,
                                             last_dc_coeff + i, &bw);
            } else {
              ok = EncodeRefinementBits(block, ac_huff, Ss, Se, Al,
                                        &coding_state, &bw);
            }
            if (!ok) return false;
          }
        }
      }
      --restarts_to_go;
    }
  }
  Flush(&coding_state, &bw);
  JumpToByteBoundary(&bw);
  JpegBitWriterFinish(&bw);
  if (!bw.healthy) return false;

  return true;
}

struct Token {
  uint8_t histo_idx;
  uint8_t symbol;
  uint16_t bits;
  Token(int i, int s, int b) : histo_idx(i), symbol(s), bits(b) {}
};

void ComputeTokensForBlock(const coeff_t* block, int histo_dc, int histo_ac,
                           coeff_t* last_dc_coeff, Token** tokens_ptr) {
  Token* next_token = *tokens_ptr;
  coeff_t temp2;
  coeff_t temp;
  temp2 = block[0];
  temp = temp2 - *last_dc_coeff;
  if (temp == 0) {
    *next_token++ = Token(histo_dc, 0, 0);
  } else {
    *last_dc_coeff = temp2;
    temp2 = temp;
    if (temp < 0) {
      temp = -temp;
      temp2--;
    }
    int dc_nbits = jxl::FloorLog2Nonzero<uint32_t>(temp) + 1;
    int dc_mask = (1 << dc_nbits) - 1;
    *next_token++ = Token(histo_dc, dc_nbits, temp2 & dc_mask);
  }
  int num_nonzeros = HWY_DYNAMIC_DISPATCH(NumNonZero8x8ExceptDC)(block);
  for (int k = 1; k < 64; ++k) {
    if (num_nonzeros == 0) {
      *next_token++ = Token(histo_ac, 0, 0);
      break;
    }
    int r = 0;
    while ((temp = block[kJPEGNaturalOrder[k]]) == 0) {
      r++;
      k++;
    }
    --num_nonzeros;
    if (temp < 0) {
      temp = -temp;
      temp2 = ~temp;
    } else {
      temp2 = temp;
    }
    while (r > 15) {
      *next_token++ = Token(histo_ac, 0xf0, 0);
      r -= 16;
    }
    int ac_nbits = jxl::FloorLog2Nonzero<uint32_t>(temp) + 1;
    int ac_mask = (1 << ac_nbits) - 1;
    int symbol = (r << 4u) + ac_nbits;
    *next_token++ = Token(histo_ac, symbol, temp2 & ac_mask);
  }
  *tokens_ptr = next_token;
}

struct TokenArray {
  Token* tokens = nullptr;
  size_t num_tokens = 0;
};

size_t MaxNumTokensPerMCURow(j_compress_ptr cinfo) {
  int MCUs_per_row = DivCeil(cinfo->image_width, 8 * cinfo->max_h_samp_factor);
  size_t blocks_per_mcu = 0;
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    blocks_per_mcu += comp->h_samp_factor * comp->v_samp_factor;
  }
  return kDCTBlockSize * blocks_per_mcu * MCUs_per_row;
}

size_t EstimateNumTokens(j_compress_ptr cinfo, size_t mcu_y, size_t ysize_mcus,
                         size_t num_tokens, size_t max_per_row) {
  size_t estimate;
  if (mcu_y == 0) {
    estimate = 16 * max_per_row;
  } else {
    estimate = (4 * ysize_mcus * num_tokens) / (3 * mcu_y);
  }
  size_t mcus_left = ysize_mcus - mcu_y;
  return std::min(mcus_left * max_per_row,
                  std::max(max_per_row, estimate - num_tokens));
}

void ComputeTokens(j_compress_ptr cinfo,
                   std::vector<TokenArray>* token_arrays) {
  jpeg_comp_master* m = cinfo->master;
  TokenArray ta;
  Token* next_token = ta.tokens;
  size_t num_tokens = 0;
  size_t total_num_tokens = 0;
  size_t max_tokens_per_mcu_row = MaxNumTokensPerMCURow(cinfo);
  int xsize_mcus = DivCeil(cinfo->image_width, 8 * cinfo->max_h_samp_factor);
  int ysize_mcus = DivCeil(cinfo->image_height, 8 * cinfo->max_v_samp_factor);
  coeff_t last_dc_coeff[MAX_COMPS_IN_SCAN] = {0};
  JBLOCKARRAY ba[MAX_COMPS_IN_SCAN];
  for (int mcu_y = 0; mcu_y < ysize_mcus; ++mcu_y) {
    ta.num_tokens = next_token - ta.tokens;
    if (ta.num_tokens + max_tokens_per_mcu_row > num_tokens) {
      if (ta.tokens) {
        token_arrays->push_back(ta);
        total_num_tokens += ta.num_tokens;
      }
      num_tokens = EstimateNumTokens(cinfo, mcu_y, ysize_mcus, total_num_tokens,
                                     max_tokens_per_mcu_row);
      ta.tokens = Allocate<Token>(cinfo, num_tokens, JPOOL_IMAGE);
      next_token = ta.tokens;
    }
    for (int c = 0; c < cinfo->num_components; ++c) {
      jpeg_component_info* comp = &cinfo->comp_info[c];
      int by0 = mcu_y * comp->v_samp_factor;
      int block_rows_left = comp->height_in_blocks - by0;
      int max_block_rows = std::min(comp->v_samp_factor, block_rows_left);
      ba[c] = (*cinfo->mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[c], by0,
          max_block_rows, false);
    }
    if (cinfo->max_h_samp_factor == 1 && cinfo->max_v_samp_factor == 1) {
      for (int mcu_x = 0; mcu_x < xsize_mcus; ++mcu_x) {
        for (int c = 0; c < cinfo->num_components; ++c) {
          ComputeTokensForBlock(&ba[c][0][mcu_x][0], c, c + 4,
                                &last_dc_coeff[c], &next_token);
        }
      }
      continue;
    }
    for (int mcu_x = 0; mcu_x < xsize_mcus; ++mcu_x) {
      for (int c = 0; c < cinfo->num_components; ++c) {
        jpeg_component_info* comp = &cinfo->comp_info[c];
        for (int iy = 0; iy < comp->v_samp_factor; ++iy) {
          for (int ix = 0; ix < comp->h_samp_factor; ++ix) {
            size_t block_y = mcu_y * comp->v_samp_factor + iy;
            size_t block_x = mcu_x * comp->h_samp_factor + ix;
            if (block_x >= comp->width_in_blocks ||
                block_y >= comp->height_in_blocks) {
              *next_token++ = Token(c, 0, 0);
              *next_token++ = Token(c + 4, 0, 0);
              continue;
            }
            ComputeTokensForBlock(&ba[c][iy][block_x][0], c, c + 4,
                                  &last_dc_coeff[c], &next_token);
          }
        }
      }
    }
  }
  ta.num_tokens = next_token - ta.tokens;
  token_arrays->push_back(ta);
}

void WriteTokens(const Token* tokens, size_t num_tokens,
                 const HuffmanCodeTable* huff_tables, const int* context_map,
                 JpegBitWriter* bw) {
  for (size_t i = 0; i < num_tokens; ++i) {
    Token t = tokens[i];
    int nbits = t.symbol & 0xf;
    WriteSymbol(t.symbol, &huff_tables[context_map[t.histo_idx]], bw);
    if (nbits > 0) {
      WriteBits(bw, nbits, t.bits);
    }
  }
}

void BuildHistograms(const Token* tokens, size_t num_tokens,
                     Histogram* histograms) {
  for (size_t j = 0; j < num_tokens; ++j) {
    Token t = tokens[j];
    ++histograms[t.histo_idx].count[t.symbol];
  }
}

void EncodeSingleScan(j_compress_ptr cinfo) {
  std::vector<TokenArray> token_arrays;
  ComputeTokens(cinfo, &token_arrays);
  Histogram histograms[8] = {};
  for (size_t i = 0; i < token_arrays.size(); ++i) {
    Token* tokens = token_arrays[i].tokens;
    size_t num_tokens = token_arrays[i].num_tokens;
    BuildHistograms(tokens, num_tokens, histograms);
  }
  JpegClusteredHistograms dc_clusters;
  ClusterJpegHistograms(histograms, 4, &dc_clusters);
  JpegClusteredHistograms ac_clusters;
  ClusterJpegHistograms(histograms + 4, 4, &ac_clusters);

  std::vector<JPEGHuffmanCode> huffman_codes;
  for (size_t i = 0; i < dc_clusters.histograms.size(); ++i) {
    AddJpegHuffmanCode(dc_clusters.histograms[i], i, &huffman_codes);
  }
  for (size_t i = 0; i < ac_clusters.histograms.size(); ++i) {
    AddJpegHuffmanCode(ac_clusters.histograms[i], 0x10 + i, &huffman_codes);
  }

  int context_map[8];
  ScanCodingInfo sci;
  for (int c = 0; c < cinfo->num_components; ++c) {
    sci.dc_tbl_idx[c] = dc_clusters.histogram_indexes[c];
    sci.ac_tbl_idx[c] = ac_clusters.histogram_indexes[c] + 4;
    context_map[c] = sci.dc_tbl_idx[c];
    context_map[c + 4] = sci.ac_tbl_idx[c];
  }
  sci.num_huffman_codes = huffman_codes.size();
  cinfo->master->scan_coding_info.emplace_back(std::move(sci));
  EncodeDHT(cinfo, huffman_codes.data(), huffman_codes.size());
  EncodeSOS(cinfo, 0);

  JpegBitWriter bw;
  JpegBitWriterInit(&bw, cinfo);
  HuffmanCodeTable* huff_tables = cinfo->master->huff_tables;
  for (size_t i = 0; i < token_arrays.size(); ++i) {
    Token* tokens = token_arrays[i].tokens;
    size_t num_tokens = token_arrays[i].num_tokens;
    WriteTokens(tokens, num_tokens, huff_tables, context_map, &bw);
  }
  JumpToByteBoundary(&bw);
  JpegBitWriterFinish(&bw);
  if (!bw.healthy) {
    JPEGLI_ERROR("Failed to encode scan.");
  }
}

}  // namespace jpegli
#endif  // HWY_ONCE
