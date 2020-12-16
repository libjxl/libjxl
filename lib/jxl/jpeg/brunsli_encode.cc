// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lib/jxl/jpeg/brunsli_encode.h"

#include <brotli/encode.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/jpeg/enc_jpeg_data_reader.h"
#include "lib/jxl/jpeg/enc_jpeg_state.h"
#include "lib/jxl/jpeg/jpeg_constants.h"
#include "lib/jxl/jpeg/jpeg_lehmer_code.h"
#include "lib/jxl/jpeg/jpeg_quant_matrix.h"

namespace jxl {
namespace jpeg {

namespace {

void JXL_INLINE WriteBits(size_t n_bits, uint64_t bits, BitWriter* writer) {
  writer->Write(n_bits, bits);
}

static JXL_INLINE void Append(std::vector<uint8_t>* dst, const uint8_t* begin,
                              const uint8_t* end) {
  dst->insert(dst->end(), begin, end);
}
static JXL_INLINE void Append(std::vector<uint8_t>* dst,
                              const std::vector<uint8_t>& src) {
  Append(dst, src.data(), src.data() + src.size());
}

}  // namespace

constexpr int kBrotliQuality = 6;
constexpr int kBrotliWindowBits = 18;

// Returns an upper bound on the encoded size of the jpeg internals section.
size_t EstimateAuxDataSize(const JPEGData& jpg) {
  size_t size = (jpg.marker_order.size() + 272 * jpg.huffman_code.size() +
                 7 * jpg.scan_info.size() + 16);
  for (size_t i = 0; i < jpg.scan_info.size(); ++i) {
    size += 7 * jpg.scan_info[i].reset_points.size();
    size += 7 * jpg.scan_info[i].extra_zero_runs.size();
  }
  size_t nsize = jpg.has_zero_padding_bit ? jpg.padding_bits.size() : 0;
  // We have maximum 4 * 9 bits for describing nsize, plus nsize padding bits,
  // plus maximum 7 bits for going to the next byte boundary.
  size += (nsize + 43) >> 3;
  for (size_t i = 0; i < jpg.inter_marker_data.size(); ++i) {
    size += 5 + jpg.inter_marker_data[i].size();
  }
  return size;
}

size_t GetMaximumBrunsliEncodedSize(const JPEGData& jpg) {
  // Rough estimate is 1.2 * uncompressed size plus some more for the header.
  size_t hdr_size = 1 << 20;
  hdr_size += EstimateAuxDataSize(jpg);
  for (const auto& data : jpg.app_data) {
    hdr_size += data.size();
  }
  for (const auto& data : jpg.com_data) {
    hdr_size += data.size();
  }
  hdr_size += jpg.tail_data.size();
  return 1.2 * jpg.width * jpg.height * jpg.components.size() + hdr_size;
}

size_t Base128Size(size_t val) {
  size_t size = 1;
  for (; val >= 128; val >>= 7) ++size;
  return size;
}

size_t EncodeBase128(size_t val, uint8_t* data) {
  size_t len = 0;
  do {
    data[len++] = (val & 0x7f) | (val >= 128 ? 0x80 : 0);
    val >>= 7;
  } while (val > 0);
  return len;
}

void EncodeBase128Fix(size_t val, size_t len, uint8_t* data) {
  for (size_t i = 0; i < len; ++i) {
    *(data++) = (val & 0x7f) | (i + 1 < len ? 0x80 : 0);
    val >>= 7;
  }
}

bool TransformApp0Marker(const std::vector<uint8_t>& s,
                         std::vector<uint8_t>* out) {
  if (s.size() != 17) return false;
  if (memcmp(s.data(), AppData_0xe0, 9) != 0) return false;
  if ((s[9] == 1 || s[9] == 2) &&  // version / 1.1 or 1.2
      s[10] < 4 &&                 // density units
      s[15] == 0 && s[16] == 0) {  // thumbnail size / no thumbnail
    const uint8_t x_dens_hi = s[11];
    const uint8_t x_dens_lo = s[12];
    int x_dens = (x_dens_hi << 8) + x_dens_lo;
    const uint8_t y_dens_hi = s[13];
    const uint8_t y_dens_lo = s[14];
    int y_dens = (y_dens_hi << 8) + y_dens_lo;
    int density_ix = -1;
    for (size_t k = 0; k < kMaxApp0Densities; ++k) {
      if (x_dens == kApp0Densities[k] && y_dens == x_dens) {
        density_ix = k;
      }
    }
    if (density_ix >= 0) {
      uint8_t app0_status = (s[9] - 1) | s[10] << 1 | density_ix << 3;
      *out = std::vector<uint8_t>(1);
      out->at(0) = app0_status;
      return true;
    }
  }
  return false;
}

bool TransformApp2Marker(const std::vector<uint8_t>& s,
                         std::vector<uint8_t>* out) {
  if (s.size() == 3161 && !memcmp(s.data(), AppData_0xe2, 84) &&
      !memcmp(s.data() + 85, AppData_0xe2 + 85, 3161 - 85)) {
    std::vector<uint8_t> code(2);
    code[0] = 0x80;
    code[1] = s[84];
    *out = code;
    return true;
  }
  return false;
}

bool TransformApp12Marker(const std::vector<uint8_t>& s,
                          std::vector<uint8_t>* out) {
  if (s.size() == 18 && !memcmp(s.data(), AppData_0xec, 15) &&
      !memcmp(s.data() + 16, AppData_0xec + 16, 18 - 16)) {
    std::vector<uint8_t> code(2);
    code[0] = 0x81;
    code[1] = s[15];
    *out = code;
    return true;
  }
  return false;
}

bool TransformApp14Marker(const std::vector<uint8_t>& s,
                          std::vector<uint8_t>* out) {
  if (s.size() == 15 && !memcmp(&s[0], AppData_0xee, 10) &&
      !memcmp(&s[11], AppData_0xee + 11, 15 - 11)) {
    std::vector<uint8_t> code(2);
    code[0] = 0x82;
    code[1] = s[10];
    *out = code;
    return true;
  }
  return false;
}

std::vector<uint8_t> TransformAppMarker(const std::vector<uint8_t>& s,
                                        size_t* transformed_marker_count) {
  std::vector<uint8_t> out;
  if (TransformApp0Marker(s, &out)) {
    (*transformed_marker_count)++;
    return out;
  }
  if (TransformApp2Marker(s, &out)) {
    (*transformed_marker_count)++;
    return out;
  }
  if (TransformApp12Marker(s, &out)) {
    (*transformed_marker_count)++;
    return out;
  }
  if (TransformApp14Marker(s, &out)) {
    (*transformed_marker_count)++;
    return out;
  }
  return s;
}

int GetQuantTableId(const JPEGQuantTable& q, bool is_chroma,
                    uint8_t dst[kDCTBlockSize]) {
  for (int j = 0; j < kNumStockQuantTables; ++j) {
    bool match_found = true;
    for (size_t k = 0; match_found && k < kDCTBlockSize; ++k) {
      if (q.values[k] != kStockQuantizationTables[is_chroma][j][k]) {
        match_found = false;
      }
    }
    if (match_found) {
      return j;
    }
  }
  return kNumStockQuantTables + FindBestMatrix(&q.values[0], is_chroma, dst);
}

void EncodeVarint(int n, int max_bits, BitWriter* writer) {
  int b;
  JXL_DASSERT(n < (1 << max_bits));
  for (b = 0; n != 0 && b < max_bits; ++b) {
    if (b + 1 != max_bits) {
      WriteBits(1, 1, writer);
    }
    WriteBits(1, n & 1, writer);
    n >>= 1;
  }
  if (b < max_bits) {
    WriteBits(1, 0, writer);
  }
}

// encodes an integer with packets of 'nbits' bits, limited to 'max_symbols'
// emitted symbols.
void EncodeLimitedVarint(size_t bits, int nbits, int max_symbols,
                         BitWriter* writer) {
  const size_t mask = (static_cast<size_t>(1) << nbits) - 1;
  for (int b = 0; b < max_symbols; ++b) {
    WriteBits(1, bits != 0, writer);
    if (bits == 0) break;
    WriteBits(nbits, bits & mask, writer);
    bits >>= nbits;
  }
}

bool EncodeQuantTables(const JPEGData& jpg, BitWriter* writer) {
  if (jpg.quant.empty() || jpg.quant.size() > 4) {
    // If ReadJpeg() succeeded with JPEG_READ_ALL mode, this should not happen.
    return false;
  }
  WriteBits(2, jpg.quant.size() - 1, writer);
  for (size_t i = 0; i < jpg.quant.size(); ++i) {
    const JPEGQuantTable& q = jpg.quant[i];
    for (size_t k = 0; k < kDCTBlockSize; ++k) {
      const int j = kJPEGNaturalOrder[k];
      if (q.values[j] == 0) {
        // Note: ReadJpeg() checks this case and discards such JPEG files.
        return false;
      }
    }

    uint8_t quant_approx[kDCTBlockSize];
    const int code = GetQuantTableId(q, i > 0, quant_approx);
    WriteBits(1, (code >= kNumStockQuantTables), writer);
    if (code < kNumStockQuantTables) {
      WriteBits(3, code, writer);
    } else {
      size_t q_factor = code - kNumStockQuantTables;
      JXL_DASSERT(q_factor < kQFactorLimit);
      WriteBits(kQFactorBits, q_factor, writer);
      int last_diff = 0;  // difference predictor
      for (size_t k = 0; k < kDCTBlockSize; ++k) {
        const int j = kJPEGNaturalOrder[k];
        const int new_diff = q.values[j] - quant_approx[j];
        int diff = new_diff - last_diff;
        last_diff = new_diff;
        WriteBits(1, diff != 0, writer);
        if (diff) {
          WriteBits(1, diff < 0, writer);
          if (diff < 0) diff = -diff;
          diff -= 1;
          // This only happens on 16-bit precision with crazy values,
          // e.g. [..., 65535, 1, 65535,...]
          if (diff > 65535) return false;
          EncodeVarint(diff, 16, writer);
        }
      }
    }
  }
  for (size_t i = 0; i < jpg.components.size(); ++i) {
    WriteBits(2, jpg.components[i].quant_idx, writer);
  }
  return true;
}

bool EncodeHuffmanCode(const JPEGHuffmanCode& huff, bool is_known_last,
                       BitWriter* writer) {
  WriteBits(2, huff.slot_id & 0xf, writer);
  WriteBits(1, huff.slot_id >> 4, writer);
  if (!is_known_last) {
    WriteBits(1, huff.is_last, writer);
  } else if (!huff.is_last) {
    return false;
  }
  int is_dc_table = (huff.slot_id >> 4) == 0;
  int total_count = 0;
  int space = 1 << kJpegHuffmanMaxBitLength;
  int max_len = kJpegHuffmanMaxBitLength;
  int max_count = is_dc_table ? kJpegDCAlphabetSize : kJpegHuffmanAlphabetSize;
  int found_match = 0;
  int stock_table_idx = 0;
  if (is_dc_table) {
    for (int i = 0; i < kNumStockDCHuffmanCodes && !found_match; ++i) {
      if (memcmp(&huff.counts[1], kStockDCHuffmanCodeCounts[i],
                 sizeof(kStockDCHuffmanCodeCounts[i])) == 0 &&
          memcmp(&huff.values[0], kStockDCHuffmanCodeValues[i],
                 sizeof(kStockDCHuffmanCodeValues[i])) == 0) {
        found_match = 1;
        stock_table_idx = i;
      }
    }
  } else {
    for (int i = 0; i < kNumStockACHuffmanCodes && !found_match; ++i) {
      if (memcmp(&huff.counts[1], kStockACHuffmanCodeCounts[i],
                 sizeof(kStockACHuffmanCodeCounts[i])) == 0 &&
          memcmp(&huff.values[0], kStockACHuffmanCodeValues[i],
                 sizeof(kStockACHuffmanCodeValues[i])) == 0) {
        found_match = 1;
        stock_table_idx = i;
      }
    }
  }
  WriteBits(1, found_match, writer);
  if (found_match) {
    WriteBits(1, stock_table_idx, writer);
    return true;
  }
  while (max_len > 0 && huff.counts[max_len] == 0) --max_len;
  if (huff.counts[0] != 0 || max_len == 0) {
    return false;
  }
  WriteBits(4, max_len - 1, writer);
  space -= (1 << (kJpegHuffmanMaxBitLength - max_len));
  for (int i = 1; i <= max_len; ++i) {
    int count = huff.counts[i] - (i == max_len ? 1 : 0);
    int count_limit = std::min(max_count - total_count,
                               space >> (kJpegHuffmanMaxBitLength - i));
    if (count > count_limit) {
      JXL_WARNING("len=%d count=%d limit=%d space=%d, total=%d", i, count,
                  count_limit, space, total_count);
      return false;
    }
    if (count_limit > 0) {
      int nbits = FloorLog2Nonzero<uint32_t>(count_limit) + 1;
      WriteBits(nbits, count, writer);
      total_count += count;
      space -= count * (1 << (kJpegHuffmanMaxBitLength - i));
    }
  }
  if (huff.values[total_count] != kJpegHuffmanAlphabetSize) {
    return false;
  }

  PermutationCoder p;
  p.Init(
      is_dc_table
          ? std::vector<uint8_t>(kDefaultDCValues, std::end(kDefaultDCValues))
          : std::vector<uint8_t>(kDefaultACValues, std::end(kDefaultACValues)));
  for (int i = 0; i < total_count; ++i) {
    const int val = huff.values[i];
    int code, nbits;
    if (!p.RemoveValue(val, &code, &nbits)) {
      return false;
    }
    EncodeLimitedVarint(code, 2, (nbits + 1) >> 1, writer);
  }
  return true;
}

bool EncodeScanInfo(const JPEGScanInfo& si, BitWriter* writer) {
  WriteBits(6, si.Ss, writer);
  WriteBits(6, si.Se, writer);
  WriteBits(4, si.Ah, writer);
  WriteBits(4, si.Al, writer);
  WriteBits(2, si.num_components - 1, writer);
  for (size_t i = 0; i < si.num_components; ++i) {
    const JPEGComponentScanInfo& csi = si.components[i];
    WriteBits(2, csi.comp_idx, writer);
    WriteBits(2, csi.dc_tbl_idx, writer);
    WriteBits(2, csi.ac_tbl_idx, writer);
  }
  int last_block_idx = -1;
  for (const auto& block_idx : si.reset_points) {
    WriteBits(1, 1, writer);
    JXL_DASSERT(block_idx >= last_block_idx + 1);
    EncodeVarint(block_idx - last_block_idx - 1, 28, writer);
    last_block_idx = block_idx;
  }
  WriteBits(1, 0, writer);

  last_block_idx = 0;
  for (size_t i = 0; i < si.extra_zero_runs.size(); ++i) {
    int block_idx = si.extra_zero_runs[i].block_idx;
    int num = si.extra_zero_runs[i].num_extra_zero_runs;
    JXL_DASSERT(block_idx >= last_block_idx);
    for (int j = 0; j < num; ++j) {
      WriteBits(1, 1, writer);
      EncodeVarint(block_idx - last_block_idx, 28, writer);
      last_block_idx = block_idx;
    }
  }
  WriteBits(1, 0, writer);

  return true;
}

int MatchComponentIds(const std::vector<JPEGComponent>& comps) {
  if (comps.size() == 1 && comps[0].id == 1) {
    return kComponentIdsGray;
  }
  if (comps.size() == 3) {
    if (comps[0].id == 1 && comps[1].id == 2 && comps[2].id == 3) {
      return kComponentIds123;
    } else if (comps[0].id == 'R' && comps[1].id == 'G' && comps[2].id == 'B') {
      return kComponentIdsRGB;
    }
  }
  return kComponentIdsCustom;
}

bool EncodeAuxData(const JPEGData& jpg, BitWriter* writer) {
  if (jpg.marker_order.empty() || jpg.marker_order.back() != 0xd9) {
    return false;
  }
  bool have_dri = false;
  size_t num_scans = 0;
  for (size_t i = 0; i < jpg.marker_order.size(); ++i) {
    uint8_t marker = jpg.marker_order[i];
    if (marker < 0xc0) {
      return false;
    }
    WriteBits(6, marker - 0xc0, writer);
    if (marker == 0xdd) have_dri = true;
    if (marker == 0xda) ++num_scans;
  }
  if (have_dri) {
    WriteBits(16, jpg.restart_interval, writer);
  }

  JXL_DASSERT(jpg.huffman_code.size() < kMaxDHTMarkers);
  for (size_t i = 0; i < jpg.huffman_code.size(); ++i) {
    const bool is_known_last = ((i + 1) == jpg.huffman_code.size());
    WriteBits(1, is_known_last, writer);
    if (!EncodeHuffmanCode(jpg.huffman_code[i], is_known_last, writer)) {
      return false;
    }
  }

  if (num_scans != jpg.scan_info.size()) {
    return false;
  }
  for (size_t i = 0; i < jpg.scan_info.size(); ++i) {
    if (!EncodeScanInfo(jpg.scan_info[i], writer)) {
      return false;
    }
  }
  WriteBits(2, jpg.quant.size() - 1, writer);
  for (size_t i = 0; i < jpg.quant.size(); ++i) {
    WriteBits(2, jpg.quant[i].index, writer);
    if (i != jpg.quant.size() - 1) {
      WriteBits(1, jpg.quant[i].is_last, writer);
    } else if (!jpg.quant[i].is_last) {
      return false;
    }
    WriteBits(4, jpg.quant[i].precision, writer);
  }
  int comp_ids = MatchComponentIds(jpg.components);
  WriteBits(2, comp_ids, writer);
  if (comp_ids == kComponentIdsCustom) {
    for (size_t i = 0; i < jpg.components.size(); ++i) {
      WriteBits(8, jpg.components[i].id, writer);
    }
  }
  size_t nsize = jpg.has_zero_padding_bit ? jpg.padding_bits.size() : 0;
  if (nsize > PaddingBitsLimit(jpg)) return false;
  // we limit to 32b for nsize
  EncodeLimitedVarint(nsize, 8, 4, writer);
  if (nsize > 0) {
    for (size_t i = 0; i < nsize; ++i) {
      WriteBits(1, jpg.padding_bits[i], writer);
    }
  }
  writer->ZeroPadToByte();
  for (size_t i = 0; i < jpg.inter_marker_data.size(); ++i) {
    const auto& s = jpg.inter_marker_data[i];
    uint8_t buffer[(sizeof(size_t) * 8 + 6) / 7];
    size_t len = EncodeBase128(s.size(), buffer);
    writer->AppendByteAligned(Span<const uint8_t>(buffer, len));
    writer->AppendByteAligned(Span<const uint8_t>(s.data(), s.size()));
  }
  return true;
}

uint32_t FrameTypeCode(const JPEGData& jpg) {
  uint32_t code = 0;
  int shift = 0;
  for (size_t i = 0; i < jpg.components.size() && i < 4; ++i) {
    uint32_t h_samp = jpg.components[i].h_samp_factor - 1;
    uint32_t v_samp = jpg.components[i].v_samp_factor - 1;
    code |= (h_samp << (shift + 4)) | (v_samp << shift);
    shift += 8;
  }
  return code;
}

bool EncodeSignature(size_t len, uint8_t* data, size_t* pos) {
  if (len < kBrunsliSignatureSize || *pos > len - kBrunsliSignatureSize) {
    return false;
  }
  memcpy(&data[*pos], kBrunsliSignature, kBrunsliSignatureSize);
  *pos += kBrunsliSignatureSize;
  return true;
}

static void EncodeValue(uint8_t tag, size_t value, uint8_t* data, size_t* pos) {
  data[(*pos)++] = ValueMarker(tag);
  *pos += EncodeBase128(value, data + *pos);
}

bool EncodeHeader(const JPEGData& jpg, uint8_t* data, size_t* len) {
  size_t version = jpg.version;
  bool is_fallback = (version & 1);
  // Fallback can not be combined with anything else.
  if (is_fallback && (version != 1)) return false;
  // Non-fallback image can not be empty.
  if ((!is_fallback && (jpg.width == 0 || jpg.height == 0)) ||
      jpg.components.empty() || jpg.components.size() > kMaxComponents) {
    return false;
  }
  // Only 3 bits are defined.
  if (version & ~7u) return false;

  size_t version_comp = (jpg.components.size() - 1) | (version << 2);
  size_t subsampling = FrameTypeCode(jpg);

  size_t pos = 0;
  EncodeValue(kBrunsliHeaderWidthTag, jpg.width, data, &pos);
  EncodeValue(kBrunsliHeaderHeightTag, jpg.height, data, &pos);
  EncodeValue(kBrunsliHeaderVersionCompTag, version_comp, data, &pos);
  EncodeValue(kBrunsliHeaderSubsamplingTag, subsampling, data, &pos);

  *len = pos;
  return true;
}

bool EncodeMetaData(const JPEGData& jpg, uint8_t* data, size_t* len) {
  // Concatenate all the (possibly transformed) metadata pieces into one string.
  std::vector<uint8_t> metadata;
  size_t transformed_marker_count = 0;
  for (size_t i = 0; i < jpg.app_data.size(); ++i) {
    const auto& s = jpg.app_data[i];
    Append(&metadata, TransformAppMarker(s, &transformed_marker_count));
  }
  if (transformed_marker_count > kBrunsliShortMarkerLimit) {
    JXL_WARNING("Too many short markers: %zu", transformed_marker_count);
    return false;
  }
  for (const auto& s : jpg.com_data) {
    Append(&metadata, s);
  }
  if (!jpg.tail_data.empty()) {
    metadata.push_back(0xD9);
    Append(&metadata, jpg.tail_data);
  }
  if (metadata.empty()) {
    *len = 0;
    return true;
  } else if (metadata.size() == 1) {
    *len = 1;
    data[0] = metadata[0];
    return true;
  }

  // Write base-128 encoding of the original metadata size.
  size_t pos = EncodeBase128(metadata.size(), data);

  // Write the compressed metadata directly to the output.
  size_t compressed_size = *len - pos;
  if (!BrotliEncoderCompress(kBrotliQuality, kBrotliWindowBits,
                             BROTLI_DEFAULT_MODE, metadata.size(),
                             metadata.data(), &compressed_size, &data[pos])) {
    JXL_WARNING("Brotli compression failed: input size=%zu, pos=%zu, len=%zu",
                metadata.size(), pos, *len);
    return false;
  }
  pos += compressed_size;
  *len = pos;
  return true;
}

bool EncodeJPEGInternals(const JPEGData& jpg, uint8_t* data, size_t* len) {
  BitWriter writer;
  BitWriter::Allotment allotment(&writer, *len * kBitsPerByte);

  if (!EncodeAuxData(jpg, &writer)) {
    return false;
  }

  writer.ZeroPadToByte();
  // TODO(deymo): This should use ReclaimAndCharge() if we had an AuxOut.
  size_t used_bits, unused_bits;
  allotment.PrivateReclaim(&writer, &used_bits, &unused_bits);

  auto span = writer.GetSpan();
  *len = span.size();
  memcpy(data, span.data(), span.size());
  return true;
}

bool EncodeQuantData(const JPEGData& jpg, uint8_t* data, size_t* len) {
  BitWriter writer;
  BitWriter::Allotment allotment(&writer, *len * kBitsPerByte);

  if (!EncodeQuantTables(jpg, &writer)) {
    return false;
  }

  writer.ZeroPadToByte();
  // TODO(deymo): This should use ReclaimAndCharge() if we had an AuxOut.
  size_t used_bits, unused_bits;
  allotment.PrivateReclaim(&writer, &used_bits, &unused_bits);

  auto span = writer.GetSpan();
  *len = span.size();
  memcpy(data, span.data(), span.size());
  return true;
}

typedef bool (*EncodeSectionDataFn)(const JPEGData& jpg, uint8_t* data,
                                    size_t* len);

bool EncodeSection(const JPEGData& jpg, uint8_t tag,
                   EncodeSectionDataFn write_section, size_t section_size_bytes,
                   size_t len, uint8_t* data, size_t* pos) {
  // Write the marker byte for the section.
  const size_t pos_start = *pos;
  const uint8_t marker = SectionMarker(tag);
  data[(*pos)++] = marker;

  // Skip enough bytes for a valid (though not necessarily optimal) base-128
  // encoding of the size of the section.
  *pos += section_size_bytes;

  size_t section_size = len - *pos;
  if (!write_section(jpg, &data[*pos], &section_size)) {
    return false;
  }
  *pos += section_size;

  if ((section_size >> (7 * section_size_bytes)) > 0) {
    JXL_WARNING(
        "Section 0x%.2x size %zu too large for %zu bytes base128 number",
        marker, section_size, section_size_bytes);
    return false;
  }

  // Write the final size of the section after the marker byte.
  EncodeBase128Fix(section_size, section_size_bytes, &data[pos_start + 1]);
  return true;
}

bool BrunsliSerialize(const JPEGData& jpg, uint32_t skip_sections,
                      uint8_t* data, size_t* len) {
  size_t pos = 0;

  // TODO(eustas): refactor to remove repetitive params.
  bool ok = true;

  const auto encode_section = [&](uint8_t tag, EncodeSectionDataFn fn,
                                  size_t size) {
    return EncodeSection(jpg, tag, fn, size, *len, data, &pos);
  };

  if (!(skip_sections & (1u << kBrunsliSignatureTag))) {
    ok = EncodeSignature(*len, data, &pos);
    if (!ok) return false;
  }

  if (!(skip_sections & (1u << kBrunsliHeaderTag))) {
    ok = encode_section(kBrunsliHeaderTag, EncodeHeader, 1);
    if (!ok) return false;
  }

  if (!(skip_sections & (1u << kBrunsliJPEGInternalsTag))) {
    ok = encode_section(kBrunsliJPEGInternalsTag, EncodeJPEGInternals,
                        Base128Size(EstimateAuxDataSize(jpg)));
    if (!ok) return false;
  }

  if (!(skip_sections & (1u << kBrunsliMetaDataTag))) {
    ok = encode_section(kBrunsliMetaDataTag, EncodeMetaData,
                        Base128Size(*len - pos));
    if (!ok) return false;
  }

  if (!(skip_sections & (1u << kBrunsliQuantDataTag))) {
    ok = encode_section(kBrunsliQuantDataTag, EncodeQuantData, 2);
    if (!ok) return false;
  }

  *len = pos;
  return true;
}

}  // namespace jpeg
}  // namespace jxl
