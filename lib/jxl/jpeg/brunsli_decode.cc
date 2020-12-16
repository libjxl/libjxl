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

// This module implements the internal functions declared in dec_jpeg_status.h

#include <brotli/decode.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/jpeg/brunsli_status.h"
#include "lib/jxl/jpeg/dec_jpeg_data_writer.h"
#include "lib/jxl/jpeg/dec_jpeg_state.h"
#include "lib/jxl/jpeg/dec_jpeg_state_internal.h"
#include "lib/jxl/jpeg/jpeg_constants.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/jpeg/jpeg_lehmer_code.h"
#include "lib/jxl/jpeg/jpeg_quant_matrix.h"

namespace jxl {
namespace jpeg {

namespace {
static JXL_INLINE void Append(std::vector<uint8_t>* dst, const uint8_t* begin,
                              const uint8_t* end) {
  dst->insert(dst->end(), begin, end);
}
static JXL_INLINE void Append(std::vector<uint8_t>* dst, const uint8_t* begin,
                              size_t length) {
  Append(dst, begin, begin + length);
}
}  // namespace

static const uint32_t kKnownSectionTags =
    (1u << kBrunsliSignatureTag) | (1u << kBrunsliHeaderTag) |
    (1u << kBrunsliMetaDataTag) | (1u << kBrunsliJPEGInternalsTag) |
    (1u << kBrunsliQuantDataTag);

static const uint32_t kKnownHeaderVarintTags =
    (1u << kBrunsliHeaderWidthTag) | (1u << kBrunsliHeaderHeightTag) |
    (1u << kBrunsliHeaderVersionCompTag) | (1u << kBrunsliHeaderSubsamplingTag);

bool DecodeVarint(VarintState* s, BitReader* br, size_t max_bits) {
  if (s->stage == VarintState::INIT) {
    s->value = 0;
    s->i = 0;
    s->stage = VarintState::READ_CONTINUATION;
  }

  while (true) {
    switch (s->stage) {
      case VarintState::READ_CONTINUATION: {
        if (s->i >= max_bits) {
          s->stage = VarintState::INIT;
          return true;
        }
        if (s->i + 1 != max_bits) {
          if (!br->CanReadWithinBounds(1)) return false;
          if (!br->ReadFixedBits<1>()) {
            s->stage = VarintState::INIT;
            return true;
          }
        }
        s->stage = VarintState::READ_DATA;
        continue;
      }
      case VarintState::READ_DATA: {
        if (!br->CanReadWithinBounds(1)) return false;
        size_t next_bit = br->ReadFixedBits<1>();
        s->value |= next_bit << s->i;
        ++s->i;
        s->stage = VarintState::READ_CONTINUATION;
        continue;
      }
      default: {
        JXL_CHECK(false);
        return false;
      }
    }
  }
}

template <size_t kChunkSize>
bool DecodeLimitedVarint(VarintState* s, BitReader* br, size_t max_symbols) {
  if (s->stage == VarintState::INIT) {
    s->value = 0;
    s->i = 0;
    s->stage = VarintState::READ_CONTINUATION;
  }
  while (true) {
    switch (s->stage) {
      case VarintState::READ_CONTINUATION: {
        if (s->i < max_symbols) {
          if (!br->CanReadWithinBounds(1)) return false;
          if (br->ReadFixedBits<1>()) {
            s->stage = VarintState::READ_DATA;
            continue;
          }
        }
        s->stage = VarintState::INIT;
        return true;
      }
      case VarintState::READ_DATA: {
        if (!br->CanReadWithinBounds(kChunkSize)) return false;
        size_t next_bits = br->ReadFixedBits<kChunkSize>();
        s->value |= next_bits << (s->i * kChunkSize);
        ++s->i;
        s->stage = VarintState::READ_CONTINUATION;
        continue;
      }
      default: {
        JXL_CHECK(false);
        return false;
      }
    }
  }
}

std::vector<uint8_t> GenerateApp0Marker(uint8_t app0_status) {
  std::vector<uint8_t> app0_marker(AppData_0xe0, AppData_0xe0 + 17);
  app0_marker[9] = app0_status & 1u ? 2 : 1;
  app0_status >>= 1u;
  app0_marker[10] = app0_status & 0x3u;
  app0_status >>= 2u;
  uint16_t x_dens = kApp0Densities[app0_status];
  app0_marker[11] = app0_marker[13] = x_dens >> 8u;
  app0_marker[12] = app0_marker[14] = x_dens & 0xFFu;
  return app0_marker;
}

std::vector<uint8_t> GenerateAppMarker(uint8_t marker, uint8_t code) {
  std::vector<uint8_t> s;
  if (marker == 0x80) {
    s = std::vector<uint8_t>(AppData_0xe2, AppData_0xe2 + 3161);
    s[84] = code;
  } else if (marker == 0x81) {
    s = std::vector<uint8_t>(AppData_0xec, AppData_0xec + 18);
    s[15] = code;
  } else {
    JXL_DASSERT(marker == 0x82);
    s = std::vector<uint8_t>(AppData_0xee, AppData_0xee + 15);
    s[10] = code;
  }
  return s;
}

bool ProcessMetaData(const uint8_t* data, size_t len, MetadataState* state,
                     JPEGData* jpg) {
  size_t pos = 0;
  while (pos < len) {
    switch (state->stage) {
      case MetadataState::READ_MARKER: {
        state->marker = static_cast<uint8_t>(data[pos++]);
        if (state->marker == 0xD9) {
          jpg->tail_data = std::vector<uint8_t>();
          state->stage = MetadataState::READ_TAIL;
          continue;
        } else if (state->marker < 0x40) {
          state->short_marker_count++;
          if (state->short_marker_count > kBrunsliShortMarkerLimit) {
            return false;
          }
          jpg->app_data.push_back(GenerateApp0Marker(state->marker));
          continue;
        } else if (state->marker >= 0x80 && state->marker <= 0x82) {
          state->short_marker_count++;
          if (state->short_marker_count > kBrunsliShortMarkerLimit) {
            return false;
          }
          state->stage = MetadataState::READ_CODE;
          continue;
        }
        // Otherwise - mutlibyte sequence.
        if ((state->marker != 0xFE) && ((state->marker >> 4u) != 0x0E)) {
          return false;
        }
        state->stage = MetadataState::READ_LENGTH_HI;
        continue;
      }

      case MetadataState::READ_TAIL: {
        Append(&jpg->tail_data, data + pos, data + len);
        pos = len;
        continue;
      }

      case MetadataState::READ_CODE: {
        const uint8_t code = data[pos++];
        jpg->app_data.push_back(GenerateAppMarker(state->marker, code));
        state->stage = MetadataState::READ_MARKER;
        continue;
      }

      case MetadataState::READ_LENGTH_HI: {
        state->length_hi = data[pos++];
        state->stage = MetadataState::READ_LENGTH_LO;
        continue;
      }

      case MetadataState::READ_LENGTH_LO: {
        const uint8_t lo = data[pos++];
        size_t marker_len = (state->length_hi << 8u) + lo;
        if (marker_len < 2) return false;
        state->remaining_multibyte_length = marker_len - 2;
        uint8_t head[3] = {state->marker, state->length_hi, lo};
        auto* dest = (state->marker == 0xFE) ? &jpg->com_data : &jpg->app_data;
        dest->emplace_back(head, head + 3);
        state->multibyte_sink = &dest->back();
        // Turn state machine to default state in case there is no payload in
        // multibyte sequence. This is important when such a sequence concludes
        // the input.
        state->stage = (state->remaining_multibyte_length > 0)
                           ? MetadataState::READ_MULTIBYTE
                           : MetadataState::READ_MARKER;
        continue;
      }

      case MetadataState::READ_MULTIBYTE: {
        size_t chunk_size =
            std::min(state->remaining_multibyte_length, len - pos);
        Append(state->multibyte_sink, data + pos, chunk_size);
        state->remaining_multibyte_length -= chunk_size;
        pos += chunk_size;
        if (state->remaining_multibyte_length == 0) {
          state->stage = MetadataState::READ_MARKER;
        }
        continue;
      }

      default:
        return false;
    }
  }
  return true;
}

static BrunsliStatus DecodeHuffmanCode(DecState* state, JPEGData* jpg) {
  InternalState& s = *state->internal;
  JpegInternalsState& js = s.internals;
  BitReader* br = &js.br;

  while (true) {
    switch (js.stage) {
      case JpegInternalsState::READ_HUFFMAN_LAST: {
        if (!br->CanReadWithinBounds(1)) return BRUNSLI_NOT_ENOUGH_DATA;
        js.is_known_last_huffman_code = br->ReadFixedBits<1>();
        jpg->huffman_code.emplace_back();
        js.stage = JpegInternalsState::READ_HUFFMAN_SIMPLE;
        continue;
      }
      case JpegInternalsState::READ_HUFFMAN_SIMPLE: {
        if (!br->CanReadWithinBounds(5 + !js.is_known_last_huffman_code)) {
          return BRUNSLI_NOT_ENOUGH_DATA;
        }
        JPEGHuffmanCode* huff = &jpg->huffman_code.back();

        huff->slot_id = br->ReadFixedBits<2>();
        js.is_dc_table = (br->ReadFixedBits<1>() == 0);
        huff->slot_id += js.is_dc_table ? 0 : 0x10;
        huff->is_last = js.is_known_last_huffman_code || br->ReadFixedBits<1>();
        huff->counts[0] = 0;
        int found_match = br->ReadFixedBits<1>();
        if (found_match) {
          if (js.is_dc_table) {
            int huff_table_idx = br->ReadFixedBits<1>();
            memcpy(&huff->counts[1], kStockDCHuffmanCodeCounts[huff_table_idx],
                   sizeof(kStockDCHuffmanCodeCounts[0]));
            memcpy(&huff->values[0], kStockDCHuffmanCodeValues[huff_table_idx],
                   sizeof(kStockDCHuffmanCodeValues[0]));
          } else {
            int huff_table_idx = br->ReadFixedBits<1>();
            memcpy(&huff->counts[1], kStockACHuffmanCodeCounts[huff_table_idx],
                   sizeof(kStockACHuffmanCodeCounts[0]));
            memcpy(&huff->values[0], kStockACHuffmanCodeValues[huff_table_idx],
                   sizeof(kStockACHuffmanCodeValues[0]));
          }
          js.stage = JpegInternalsState::HUFFMAN_UPDATE;
        } else {
          // One less bit is used than requested, but it is guaranteed to be
          // consumed in complex Huffman code case.
          js.p.Init(js.is_dc_table
                        ? std::vector<uint8_t>(kDefaultDCValues,
                                               std::end(kDefaultDCValues))
                        : std::vector<uint8_t>(kDefaultACValues,
                                               std::end(kDefaultACValues)));
          js.stage = JpegInternalsState::READ_HUFFMAN_MAX_LEN;
        }
        continue;
      }
      case JpegInternalsState::READ_HUFFMAN_MAX_LEN: {
        if (!br->CanReadWithinBounds(4)) return BRUNSLI_NOT_ENOUGH_DATA;
        js.max_len = br->ReadFixedBits<4>() + 1;
        js.total_count = 0;
        js.max_count =
            js.is_dc_table ? kJpegDCAlphabetSize : kJpegHuffmanAlphabetSize;
        js.space = (1u << kJpegHuffmanMaxBitLength) -
                   (1u << (kJpegHuffmanMaxBitLength - js.max_len));
        js.i = 1;
        js.stage = JpegInternalsState::READ_HUFFMAN_COUNT;
        continue;
      }
      case JpegInternalsState::READ_HUFFMAN_COUNT: {
        JPEGHuffmanCode* huff = &jpg->huffman_code.back();
        if (js.i <= js.max_len) {
          size_t shift = kJpegHuffmanMaxBitLength - js.i;
          int count_limit =
              std::min(js.max_count - js.total_count, js.space >> shift);
          if (count_limit > 0) {
            int nbits = FloorLog2Nonzero<uint32_t>(count_limit) + 1;
            if (!br->CanReadWithinBounds(nbits)) {
              return BRUNSLI_NOT_ENOUGH_DATA;
            }
            int count = br->ReadBits(nbits);
            if (count > count_limit) {
              return BRUNSLI_INVALID_BRN;
            }
            huff->counts[js.i] = count;
            js.total_count += count;
            js.space -= count * (1u << shift);
          }
          ++js.i;
          continue;
        }
        ++huff->counts[js.max_len];
        js.i = 0;
        js.stage = JpegInternalsState::READ_HUFFMAN_PERMUTATION;
        continue;
      }
      case JpegInternalsState::READ_HUFFMAN_PERMUTATION: {
        JPEGHuffmanCode* huff = &jpg->huffman_code.back();
        if (js.i < js.total_count) {
          const int nbits = js.p.num_bits();
          if (!DecodeLimitedVarint<2>(&js.varint, br, (nbits + 1) >> 1u)) {
            return BRUNSLI_NOT_ENOUGH_DATA;
          }
          uint8_t value;
          if (!js.p.Remove(js.varint.value, &value)) {
            return BRUNSLI_INVALID_BRN;
          }
          huff->values[js.i] = value;
          ++js.i;
          continue;
        }
        huff->values[js.total_count] = kJpegHuffmanAlphabetSize;
        js.stage = JpegInternalsState::HUFFMAN_UPDATE;
        continue;
      }
      case JpegInternalsState::HUFFMAN_UPDATE: {
        // This stage does not perform reading -> transient.
        if (jpg->huffman_code.back().is_last) {
          js.terminal_huffman_code_count++;
        }
        if (js.is_known_last_huffman_code) {
          js.p.Clear();
          return BRUNSLI_OK;
        }
        if (jpg->huffman_code.size() >= kMaxDHTMarkers) {
          // Too many Huffman codes for a valid bit-stream. Normally, a jpeg
          // file can have any arbitrary number of DHT, DQT, etc. But i prefer
          // we force a reasonable lower bound instead of open door to likely
          // forged BRN input.
          return BRUNSLI_INVALID_BRN;
        }
        js.stage = JpegInternalsState::READ_HUFFMAN_LAST;
        continue;
      }
      default:
        return BRUNSLI_DECOMPRESSION_ERROR;
    }
  }
  return BRUNSLI_OK;
}

BrunsliStatus DecodeScanInfo(DecState* state, JPEGData* jpg) {
  InternalState& s = *state->internal;
  JpegInternalsState& js = s.internals;
  BitReader* br = &js.br;

  const auto maybe_add_zero_run = [&js, jpg]() {
    if (js.last_num > 0) {
      JPEGScanInfo::ExtraZeroRunInfo info;
      info.block_idx = js.last_block_idx;
      info.num_extra_zero_runs = js.last_num;
      jpg->scan_info[js.i].extra_zero_runs.push_back(info);
      js.last_num = 0;
    }
  };

  while (true) {
    switch (js.stage) {
      case JpegInternalsState::READ_SCAN_COMMON: {
        JPEGScanInfo* si = &jpg->scan_info[js.i];
        if (!br->CanReadWithinBounds(22)) return BRUNSLI_NOT_ENOUGH_DATA;
        si->Ss = br->ReadFixedBits<6>();
        si->Se = br->ReadFixedBits<6>();
        si->Ah = br->ReadFixedBits<4>();
        si->Al = br->ReadFixedBits<4>();
        si->num_components = br->ReadFixedBits<2>() + 1;
        js.j = 0;
        js.stage = JpegInternalsState::READ_SCAN_COMPONENT;
        continue;
      }
      case JpegInternalsState::READ_SCAN_COMPONENT: {
        JPEGScanInfo* si = &jpg->scan_info[js.i];
        if (js.j < si->num_components) {
          if (!br->CanReadWithinBounds(6)) return BRUNSLI_NOT_ENOUGH_DATA;
          si->components[js.j].comp_idx = br->ReadFixedBits<2>();
          si->components[js.j].dc_tbl_idx = br->ReadFixedBits<2>();
          si->components[js.j].ac_tbl_idx = br->ReadFixedBits<2>();
          js.j++;
        } else {
          js.last_block_idx = -1;
          js.stage = JpegInternalsState::READ_SCAN_RESET_POINT_CONTINUATION;
        }
        continue;
      }
      case JpegInternalsState::READ_SCAN_RESET_POINT_CONTINUATION: {
        if (!br->CanReadWithinBounds(1)) return BRUNSLI_NOT_ENOUGH_DATA;
        if (br->ReadFixedBits<1>()) {
          js.stage = JpegInternalsState::READ_SCAN_RESET_POINT_DATA;
        } else {
          js.last_block_idx = 0;
          js.last_num = 0;
          js.stage = JpegInternalsState::READ_SCAN_ZERO_RUN_CONTINUATION;
        }
        continue;
      }
      case JpegInternalsState::READ_SCAN_RESET_POINT_DATA: {
        JPEGScanInfo* si = &jpg->scan_info[js.i];
        if (!DecodeVarint(&js.varint, br, 28)) return BRUNSLI_NOT_ENOUGH_DATA;
        int block_idx = js.last_block_idx + js.varint.value + 1;
        si->reset_points.emplace_back(block_idx);
        js.last_block_idx = block_idx;
        // TODO(eustas): limit to exact number of blocks.
        if (js.last_block_idx > (1 << 30)) {
          // At most 8K x 8K x num_channels blocks are expected. That is,
          // typically, 1.5 * 2^27. 2^30 should be sufficient for any sane
          // image.
          return BRUNSLI_INVALID_BRN;
        }
        js.stage = JpegInternalsState::READ_SCAN_RESET_POINT_CONTINUATION;
        continue;
      }
      case JpegInternalsState::READ_SCAN_ZERO_RUN_CONTINUATION: {
        if (!br->CanReadWithinBounds(1)) return BRUNSLI_NOT_ENOUGH_DATA;
        if (br->ReadFixedBits<1>()) {
          js.stage = JpegInternalsState::READ_SCAN_ZERO_RUN_DATA;
        } else {
          maybe_add_zero_run();
          ++js.i;
          if (js.i < js.num_scans) {
            js.stage = JpegInternalsState::READ_SCAN_COMMON;
            continue;
          }
          return BRUNSLI_OK;
        }
        continue;
      }
      case JpegInternalsState::READ_SCAN_ZERO_RUN_DATA: {
        if (!DecodeVarint(&js.varint, br, 28)) return BRUNSLI_NOT_ENOUGH_DATA;
        int block_idx = js.last_block_idx + js.varint.value;
        if (block_idx > js.last_block_idx) maybe_add_zero_run();
        ++js.last_num;
        js.last_block_idx = block_idx;
        // TODO(eustas): limit to exact number of blocks.
        if (js.last_block_idx > (1 << 30)) {
          // At most 8K x 8K x num_channels blocks are expected. That is,
          // typically, 1.5 * 2^27. 2^30 should be sufficient for any sane
          // image.
          return BRUNSLI_INVALID_BRN;
        }
        js.stage = JpegInternalsState::READ_SCAN_ZERO_RUN_CONTINUATION;
        continue;
      }
      default:
        return BRUNSLI_DECOMPRESSION_ERROR;
    }
  }
}

static bool CheckCanRead(DecState* state, size_t required) {
  // TODO(eustas): dcheck len > pos
  size_t available = state->len - state->pos;
  return required <= available;
}

static bool CheckCanReadByte(DecState* state) {
  // TODO(eustas): dcheck len > pos
  return state->pos != state->len;
}

static uint8_t ReadByte(DecState* state) {
  // TODO(eustas): dcheck len > pos
  return state->data[state->pos++];
}

static uint8_t PeekByte(DecState* state, size_t offset) {
  // TODO(eustas): dcheck overflow.
  return state->data[state->pos + offset];
}

static void SkipBytes(DecState* state, size_t len) {
  // TODO(eustas): dcheck overflow.
  state->pos += len;
}

static size_t GetBytesAvailable(DecState* state) {
  // TODO(eustas): dcheck len > pos
  return state->len - state->pos;
}

static size_t SkipAvailableBytes(DecState* state, size_t len) {
  size_t available = GetBytesAvailable(state);
  size_t skip_bytes = std::min(available, len);
  state->pos += skip_bytes;
  return skip_bytes;
}

static BrunsliStatus DecodeBase128(DecState* state, size_t* val) {
  *val = 0;
  uint64_t b = 0x80;
  size_t i = 0;
  while ((i < 9) && (b & 0x80u)) {
    if (!CheckCanRead(state, i + 1)) return BRUNSLI_NOT_ENOUGH_DATA;
    b = PeekByte(state, i);
    *val |= (b & 0x7Fu) << (i * 7);
    ++i;
  }
  SkipBytes(state, i);
  return ((b & 0x80u) == 0) ? BRUNSLI_OK : BRUNSLI_INVALID_BRN;
}

static Stage Fail(DecState* state, BrunsliStatus result) {
  InternalState& s = *state->internal;
  s.result = result;
  // Preserve current stage for continuation / error reporting.
  s.last_stage = state->stage;
  return Stage::ERROR;
}

static BrunsliStatus ReadTag(DecState* state, SectionState* section) {
  if (!CheckCanReadByte(state)) return BRUNSLI_NOT_ENOUGH_DATA;
  const uint8_t marker = ReadByte(state);

  const size_t tag = marker >> 3u;
  if (tag == 0 || tag > 15) return BRUNSLI_INVALID_BRN;
  section->tag = tag;

  const size_t wiring_type = marker & 0x7u;
  if (wiring_type != kBrunsliWiringTypeVarint &&
      wiring_type != kBrunsliWiringTypeLengthDelimited) {
    return BRUNSLI_INVALID_BRN;
  }
  section->is_section = (wiring_type == kBrunsliWiringTypeLengthDelimited);

  const uint32_t tag_bit = 1u << tag;
  if (section->tags_met & tag_bit) {
    JXL_WARNING("Duplicate marker 0x%.2x", static_cast<int>(marker));
    return BRUNSLI_INVALID_BRN;
  }
  section->tags_met |= tag_bit;

  return BRUNSLI_OK;
}

static BrunsliStatus EnterSection(DecState* state, SectionState* section) {
  size_t section_size;
  BrunsliStatus status = DecodeBase128(state, &section_size);
  if (status != BRUNSLI_OK) return status;
  section->is_active = true;
  section->remaining = section_size;
  section->milestone = state->pos;
  section->projected_end = state->pos + section->remaining;
  return BRUNSLI_OK;
}

static void LeaveSection(SectionState* section) {
  section->is_active = false;
}

static bool IsOutOfSectionBounds(DecState* state) {
  return state->pos > state->internal->section.projected_end;
}

static size_t RemainingSectionLength(DecState* state) {
  // TODO(eustas): remove this check?
  if (IsOutOfSectionBounds(state)) return 0;
  return state->internal->section.projected_end - state->pos;
}

static bool IsAtSectionBoundary(DecState* state) {
  return state->pos == state->internal->section.projected_end;
}

Stage VerifySignature(DecState* state) {
  InternalState& s = *state->internal;

  if (!CheckCanRead(state, kBrunsliSignatureSize)) {
    return Fail(state, BRUNSLI_NOT_ENOUGH_DATA);
  }
  const bool is_signature_ok =
      (memcmp(state->data + state->pos, kBrunsliSignature,
              kBrunsliSignatureSize) != 0);
  state->pos += kBrunsliSignatureSize;
  s.section.tags_met |= 1u << kBrunsliSignatureTag;
  if (is_signature_ok) return Fail(state, BRUNSLI_INVALID_BRN);
  return Stage::HEADER;
}

// Parses the brunsli header starting at data[*pos] and fills in *jpg.
// Sets *pos to the position after the header.
// Returns BRUNSLI_OK, unless the data is not valid brunsli byte stream
// or is truncated.
Stage DecodeHeader(DecState* state, JPEGData* jpg) {
  InternalState& s = *state->internal;
  HeaderState& hs = s.header;

  while (hs.stage != HeaderState::DONE) {
    switch (hs.stage) {
      case HeaderState::READ_TAG: {
        BrunsliStatus status = ReadTag(state, &s.section);
        if (status != BRUNSLI_OK) return Fail(state, status);
        if (s.section.tag != kBrunsliHeaderTag || !s.section.is_section) {
          return Fail(state, BRUNSLI_INVALID_BRN);
        }
        hs.stage = HeaderState::ENTER_SECTION;
        break;
      }

      case HeaderState::ENTER_SECTION: {
        BrunsliStatus status = EnterSection(state, &s.section);
        if (status != BRUNSLI_OK) return Fail(state, status);
        hs.stage = HeaderState::ITEM_READ_TAG;
        break;
      }

      case HeaderState::ITEM_READ_TAG: {
        if (IsAtSectionBoundary(state)) {
          hs.stage = HeaderState::FINALE;
          break;
        }
        BrunsliStatus status = ReadTag(state, &hs.section);
        if (status != BRUNSLI_OK) return Fail(state, status);
        const uint32_t tag_bit = 1u << hs.section.tag;
        if (hs.section.is_section) {
          if (kKnownHeaderVarintTags & tag_bit) {
            Fail(state, BRUNSLI_INVALID_BRN);
          }
          hs.stage = HeaderState::ITEM_ENTER_SECTION;
          break;
        }
        hs.stage = HeaderState::ITEM_READ_VALUE;
        break;
      }

      case HeaderState::ITEM_ENTER_SECTION: {
        BrunsliStatus status = DecodeBase128(state, &hs.remaining_skip_length);
        if (status != BRUNSLI_OK) return Fail(state, status);
        hs.stage = HeaderState::ITEM_SKIP_CONTENTS;
        break;
      }

      case HeaderState::ITEM_SKIP_CONTENTS: {
        size_t bytes_skipped =
            SkipAvailableBytes(state, hs.remaining_skip_length);
        hs.remaining_skip_length -= bytes_skipped;
        if (hs.remaining_skip_length > 0) {
          return Fail(state, BRUNSLI_NOT_ENOUGH_DATA);
        }
        hs.stage = HeaderState::ITEM_READ_TAG;
        break;
      }

      case HeaderState::ITEM_READ_VALUE: {
        size_t value;
        BrunsliStatus status = DecodeBase128(state, &value);
        if (status != BRUNSLI_OK) return Fail(state, status);
        hs.varint_values[hs.section.tag] = value;
        hs.stage = HeaderState::ITEM_READ_TAG;
        break;
      }

      case HeaderState::FINALE: {
        const bool has_version =
            hs.section.tags_met & (1u << kBrunsliHeaderVersionCompTag);
        if (!has_version) return Fail(state, BRUNSLI_INVALID_BRN);
        const size_t version_and_comp_count =
            hs.varint_values[kBrunsliHeaderVersionCompTag];

        const int version = version_and_comp_count >> 2u;
        jpg->version = version;

        if (version == 1) {  // fallback mode
          // TODO(eustas): do we need this?
          jpg->width = 0;
          jpg->height = 0;
          hs.stage = HeaderState::DONE;
          break;
        }

        // Wrong mode = fallback + something.
        if ((version & 1u) != 0) {
          return Fail(state, BRUNSLI_INVALID_BRN);
        }
        // Unknown mode - only 3 bits are defined.
        if ((version & ~0x7u) != 0) {
          return Fail(state, BRUNSLI_INVALID_BRN);
        }

        // Otherwise regular brunsli.
        state->use_legacy_context_model = !(version & 2);

        // Do not allow "original_jpg" for regular Brunsli files.
        s.section.tags_met |= 1u << kBrunsliOriginalJpgTag;

        const bool has_width =
            hs.section.tags_met & (1u << kBrunsliHeaderWidthTag);
        if (!has_width) return Fail(state, BRUNSLI_INVALID_BRN);
        const size_t width = hs.varint_values[kBrunsliHeaderWidthTag];
        const bool has_height =
            hs.section.tags_met & (1u << kBrunsliHeaderHeightTag);
        if (!has_height) return Fail(state, BRUNSLI_INVALID_BRN);
        const size_t height = hs.varint_values[kBrunsliHeaderHeightTag];

        if (width == 0 || height == 0) return Fail(state, BRUNSLI_INVALID_BRN);
        if (width > kMaxDimPixels || height > kMaxDimPixels) {
          return Fail(state, BRUNSLI_INVALID_BRN);
        }
        jpg->width = width;
        jpg->height = height;

        const size_t num_components = (version_and_comp_count & 3u) + 1u;
        jpg->components.resize(num_components);

        const bool has_subsampling =
            hs.section.tags_met & (1u << kBrunsliHeaderSubsamplingTag);
        if (!has_subsampling) return Fail(state, BRUNSLI_INVALID_BRN);
        size_t subsampling_code =
            hs.varint_values[kBrunsliHeaderSubsamplingTag];

        for (size_t i = 0; i < jpg->components.size(); ++i) {
          JPEGComponent* c = &jpg->components[i];
          c->v_samp_factor = (subsampling_code & 0xFu) + 1;
          subsampling_code >>= 4u;
          c->h_samp_factor = (subsampling_code & 0xFu) + 1;
          subsampling_code >>= 4u;
          if (c->v_samp_factor > kBrunsliMaxSampling) {
            return Fail(state, BRUNSLI_INVALID_BRN);
          }
          if (c->h_samp_factor > kBrunsliMaxSampling) {
            return Fail(state, BRUNSLI_INVALID_BRN);
          }
        }
        if (!UpdateSubsamplingDerivatives(jpg)) {
          return Fail(state, BRUNSLI_INVALID_BRN);
        }

        PrepareMeta(jpg, state);

        hs.stage = HeaderState::DONE;
        break;
      }

      default:
        return Fail(state, BRUNSLI_DECOMPRESSION_ERROR);
    }
  }

  LeaveSection(&s.section);
  return (jpg->version == 1) ? Stage::FALLBACK : Stage::SECTION;
}

static BrunsliStatus DecodeMetaDataSection(DecState* state, JPEGData* jpg) {
  InternalState& s = *state->internal;
  MetadataState& ms = s.metadata;

  if (ms.decompression_stage == MetadataDecompressionStage::DONE) {
    return BRUNSLI_INVALID_BRN;
  }

  if (ms.decompression_stage == MetadataDecompressionStage::INITIAL) {
    if (IsAtSectionBoundary(state)) {
      ms.decompression_stage = MetadataDecompressionStage::DONE;
      return BRUNSLI_OK;
    }
    if (RemainingSectionLength(state) == 1) {
      if (!CheckCanReadByte(state)) {
        return BRUNSLI_NOT_ENOUGH_DATA;
      }
      uint8_t data[1];
      data[0] = ReadByte(state);
      bool ok = ProcessMetaData(data, 1, &ms, jpg) && ms.CanFinish();
      ms.decompression_stage = MetadataDecompressionStage::DONE;
      return ok ? BRUNSLI_OK : BRUNSLI_INVALID_BRN;
    }
    ms.decompression_stage = MetadataDecompressionStage::READ_LENGTH;
  }

  if (ms.decompression_stage == MetadataDecompressionStage::READ_LENGTH) {
    BrunsliStatus status = DecodeBase128(state, &ms.metadata_size);
    if (status != BRUNSLI_OK) return status;
    // TODO(eustas): ms.metadata_size should be limited to avoid "zip-bombs".
    if (IsOutOfSectionBounds(state)) return BRUNSLI_INVALID_BRN;
    if (RemainingSectionLength(state) == 0) return BRUNSLI_INVALID_BRN;
    ms.brotli = BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
    if (ms.brotli == nullptr) return BRUNSLI_DECOMPRESSION_ERROR;
    ms.decompression_stage = MetadataDecompressionStage::DECOMPRESSING;
  }

  if (ms.decompression_stage == MetadataDecompressionStage::DECOMPRESSING) {
    // Free Brotli decoder and return result
    const auto finish_decompression = [&ms](BrunsliStatus result) {
      JXL_DASSERT(ms.brotli != nullptr);
      BrotliDecoderDestroyInstance(ms.brotli);
      ms.brotli = nullptr;
      ms.decompression_stage = MetadataDecompressionStage::DONE;
      return result;
    };

    while (true) {
      size_t available_bytes =
          std::min(GetBytesAvailable(state), RemainingSectionLength(state));
      size_t available_in = available_bytes;
      const uint8_t* next_in = state->data + state->pos;
      size_t available_out = 0;
      BrotliDecoderResult result = BrotliDecoderDecompressStream(
          ms.brotli, &available_in, &next_in, &available_out, nullptr, nullptr);
      if (result == BROTLI_DECODER_RESULT_ERROR) {
        return finish_decompression(BRUNSLI_INVALID_BRN);
      }
      size_t chunk_size = 0;
      const uint8_t* chunk_data =
          BrotliDecoderTakeOutput(ms.brotli, &chunk_size);
      ms.decompressed_size += chunk_size;
      if (ms.decompressed_size > ms.metadata_size) {
        return finish_decompression(BRUNSLI_INVALID_BRN);
      }
      size_t consumed_bytes = available_bytes - available_in;
      SkipBytes(state, consumed_bytes);
      bool chunk_ok = ProcessMetaData(chunk_data, chunk_size, &ms, jpg);
      if (!chunk_ok) return finish_decompression(BRUNSLI_INVALID_BRN);
      if (result == BROTLI_DECODER_RESULT_SUCCESS) {
        if (RemainingSectionLength(state) != 0) {
          return finish_decompression(BRUNSLI_INVALID_BRN);
        }
        if (ms.decompressed_size != ms.metadata_size) {
          return finish_decompression(BRUNSLI_INVALID_BRN);
        }
        if (!ms.CanFinish()) return finish_decompression(BRUNSLI_INVALID_BRN);
        return finish_decompression(BRUNSLI_OK);
      }
      if (result == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) continue;
      JXL_DASSERT(result == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT);
      if (RemainingSectionLength(state) == 0) {
        return finish_decompression(BRUNSLI_INVALID_BRN);
      }
      return BRUNSLI_NOT_ENOUGH_DATA;
    }
  }

  // Unreachable.
  JXL_DASSERT(false);
  return BRUNSLI_DECOMPRESSION_ERROR;
}

/**
 * Wraps result, depending on the state of input.
 *
 * If parser needs more data, but section data is depleted,
 * then input is corrupted.
 */
static BrunsliStatus CheckBoundary(DecState* state, BrunsliStatus result) {
  if (result == BRUNSLI_NOT_ENOUGH_DATA) {
    bool last = (RemainingSectionLength(state) <= GetBytesAvailable(state));
    return last ? BRUNSLI_INVALID_BRN : BRUNSLI_NOT_ENOUGH_DATA;
  } else {
    return result;
  }
}

static void PrepareBitReader(BitReader* br, DecState* state) {
  size_t chunk_len =
      std::min(GetBytesAvailable(state), RemainingSectionLength(state));
  br->Resume(jxl::Span<const uint8_t>(state->data + state->pos, chunk_len));
  JXL_DASSERT(br->AllReadsWithinBounds());
}

// This should suspend the bit-reader and mark all the input data as consumed
// already, bu we don't su
static BrunsliStatus SuspendBitReader(BitReader* br, DecState* state,
                                      BrunsliStatus result) {
  size_t chunk_len =
      std::min(GetBytesAvailable(state), RemainingSectionLength(state));
  size_t unused_bytes = br->Suspend();
  size_t consumed_bytes = chunk_len - unused_bytes;
  SkipBytes(state, consumed_bytes);
  result = CheckBoundary(state, result);
  // Once BitReader becomes unhealthy, further decoding should be impossible.
  JXL_DASSERT(
      br->AllReadsWithinBounds() ||
      ((result != BRUNSLI_OK) && (result != BRUNSLI_NOT_ENOUGH_DATA)));
  return result;
}

static BrunsliStatus DecodeJPEGInternalsSection(DecState* state, JPEGData* jpg) {
  InternalState& s = *state->internal;
  JpegInternalsState& js = s.internals;
  BitReader* br = &js.br;

  if (js.stage == JpegInternalsState::INIT) {
    *br = BitReader();
    js.stage = JpegInternalsState::READ_MARKERS;
  }
  PrepareBitReader(br, state);

  const auto suspend_bit_reader = [&](BrunsliStatus result) -> BrunsliStatus {
    return SuspendBitReader(br, state, result);
  };

  if (js.stage == JpegInternalsState::READ_MARKERS) {
    while (true) {
      if (!br->CanReadWithinBounds(6)) {
        return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
      }
      uint8_t marker = 0xc0 + br->ReadFixedBits<6>();
      jpg->marker_order.push_back(marker);
      if (marker == 0xc4) ++js.dht_count;
      if (marker == 0xdd) js.have_dri = true;
      if (marker == 0xda) ++js.num_scans;
      if (marker == 0xd9) break;
    }
    js.stage = JpegInternalsState::READ_DRI;
  }

  if (js.stage == JpegInternalsState::READ_DRI) {
    if (js.have_dri) {
      if (!br->CanReadWithinBounds(16)) {
        return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
      }
      jpg->restart_interval = br->ReadFixedBits<16>();
    }
    js.stage = JpegInternalsState::READ_HUFFMAN_LAST;
  }

  if (js.stage & JpegInternalsState::DECODE_HUFFMAN_MASK) {
    BrunsliStatus status = DecodeHuffmanCode(state, jpg);
    if (status != BRUNSLI_OK) return suspend_bit_reader(status);
    js.stage = JpegInternalsState::PREPARE_READ_SCANS;
  }

  if (js.stage == JpegInternalsState::PREPARE_READ_SCANS) {
    if (js.dht_count != js.terminal_huffman_code_count) {
      JXL_WARNING("Invalid number of DHT markers");
      return suspend_bit_reader(BRUNSLI_INVALID_BRN);
    }
    if (js.num_scans > 0) {
      jpg->scan_info.resize(js.num_scans);
      js.i = 0;
      js.stage = JpegInternalsState::READ_SCAN_COMMON;
    } else {
      js.stage = JpegInternalsState::READ_NUM_QUANT;
    }
  }

  if (js.stage & JpegInternalsState::DECODE_SCAN_MASK) {
    BrunsliStatus status = DecodeScanInfo(state, jpg);
    if (status != BRUNSLI_OK) return suspend_bit_reader(status);
    js.stage = JpegInternalsState::READ_NUM_QUANT;
  }

  if (js.stage == JpegInternalsState::READ_NUM_QUANT) {
    if (!br->CanReadWithinBounds(2)) {
      return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
    }
    int num_quant_tables = br->ReadFixedBits<2>() + 1;
    jpg->quant.resize(num_quant_tables);
    js.i = 0;
    js.stage = JpegInternalsState::READ_QUANT;
  }

  while (js.stage == JpegInternalsState::READ_QUANT) {
    if (js.i >= jpg->quant.size()) {
      js.stage = JpegInternalsState::READ_COMP_ID_SCHEME;
      break;
    }
    // 6 or 7 bits are used, but we know that at least one more bit is
    // guaranteed to be used by varint out of the loop.
    if (!br->CanReadWithinBounds(7)) {
      return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
    }
    JPEGQuantTable* q = &jpg->quant[js.i];
    q->index = br->ReadFixedBits<2>();
    q->is_last = (js.i == jpg->quant.size() - 1) || br->ReadFixedBits<1>();
    q->precision = br->ReadFixedBits<4>();
    if (q->precision > 1) {
      JXL_WARNING("Invalid quantization table precision: %d", q->precision);
      return suspend_bit_reader(BRUNSLI_INVALID_BRN);
    }
    // note that q->values[] are initialized to invalid 0 values.
    ++js.i;
  }

  if (js.stage == JpegInternalsState::READ_COMP_ID_SCHEME) {
    if (!br->CanReadWithinBounds(2)) {
      return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
    }
    int comp_ids = br->ReadFixedBits<2>();
    static const size_t kMinRequiredComponents[4] = {
        3 /* Ids123*/, 1 /* IdsGray */, 3 /* IdsRGB */, 0 /* IdsCustom */
    };
    if (jpg->components.size() < kMinRequiredComponents[comp_ids]) {
      JXL_WARNING("Insufficient number of components for ColorId #%d",
                  comp_ids);
      return suspend_bit_reader(BRUNSLI_INVALID_BRN);
    }
    js.stage = JpegInternalsState::READ_NUM_PADDING_BITS;
    if (comp_ids == kComponentIds123) {
      jpg->components[0].id = 1;
      jpg->components[1].id = 2;
      jpg->components[2].id = 3;
    } else if (comp_ids == kComponentIdsGray) {
      jpg->components[0].id = 1;
    } else if (comp_ids == kComponentIdsRGB) {
      jpg->components[0].id = 'R';
      jpg->components[1].id = 'G';
      jpg->components[2].id = 'B';
    } else {
      JXL_DASSERT(comp_ids == kComponentIdsCustom);
      js.i = 0;
      js.stage = JpegInternalsState::READ_COMP_ID;
    }
  }

  if (js.stage == JpegInternalsState::READ_COMP_ID) {
    while (js.i < jpg->components.size()) {
      if (!br->CanReadWithinBounds(8)) {
        return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
      }
      jpg->components[js.i].id = br->ReadFixedBits<8>();
      ++js.i;
    }
    js.stage = JpegInternalsState::READ_NUM_PADDING_BITS;
  }

  if (js.stage == JpegInternalsState::READ_NUM_PADDING_BITS) {
    // TODO(eustas): sanitize: should not be bigger than
    //               7 x (num_scans + num_blocks / dri)
    // security: limit is 32b for n_size
    if (!DecodeLimitedVarint<8>(&js.varint, br, 4)) {
      return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
    }
    js.num_padding_bits = js.varint.value;
    jpg->has_zero_padding_bit = (js.num_padding_bits > 0);
    if (js.num_padding_bits > PaddingBitsLimit(*jpg)) {
      JXL_WARNING("Suspicious number of padding bits %zu", js.num_padding_bits);
      return suspend_bit_reader(BRUNSLI_INVALID_BRN);
    }
    js.i = 0;
    js.stage = JpegInternalsState::READ_PADDING_BITS;
  }

  if (js.stage == JpegInternalsState::READ_PADDING_BITS) {
    while (js.i < js.num_padding_bits) {
      if (!br->CanReadWithinBounds(1)) {
        return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
      }
      jpg->padding_bits.emplace_back(br->ReadFixedBits<1>());
      ++js.i;
    }
    suspend_bit_reader(BRUNSLI_OK);
    Status all_padding_is_zero = br->JumpToByteBoundary();
    if (!br->Close() || !all_padding_is_zero) return BRUNSLI_INVALID_BRN;
    js.i = 0;
    js.stage = JpegInternalsState::ITERATE_MARKERS;
  } else {
    // no-op
    suspend_bit_reader(BRUNSLI_OK);
  }

  while (true) {
    switch (js.stage) {
      case JpegInternalsState::ITERATE_MARKERS: {
        if (js.i >= jpg->marker_order.size()) {
          js.stage = JpegInternalsState::DONE;
        } else if (jpg->marker_order[js.i] == 0xFF) {
          js.stage = JpegInternalsState::READ_INTERMARKER_LENGTH;
        } else {
          ++js.i;
        }
        continue;
      }

      case JpegInternalsState::READ_INTERMARKER_LENGTH: {
        BrunsliStatus status = DecodeBase128(state, &js.intermarker_length);
        if (status != BRUNSLI_OK) return CheckBoundary(state, status);
        if (js.intermarker_length > RemainingSectionLength(state)) {
          return BRUNSLI_INVALID_BRN;
        }
        jpg->inter_marker_data.emplace_back();
        js.stage = JpegInternalsState::READ_INTERMARKER_DATA;
        continue;
      }

      case JpegInternalsState::READ_INTERMARKER_DATA: {
        auto& dest = jpg->inter_marker_data.back();
        size_t piece_limit = js.intermarker_length - dest.size();
        size_t piece_size = std::min(piece_limit, GetBytesAvailable(state));
        Append(&dest, state->data + state->pos, piece_size);
        SkipBytes(state, piece_size);
        if (dest.size() < js.intermarker_length) {
          JXL_DASSERT(GetBytesAvailable(state) == 0);
          JXL_DASSERT(RemainingSectionLength(state) > 0);
          return BRUNSLI_NOT_ENOUGH_DATA;
        }
        ++js.i;
        js.stage = JpegInternalsState::ITERATE_MARKERS;
        continue;
      }

      default: { /* no-op */ }
    }
    break;  // no matching stage has been found; exit the loop.
  }

  if (!IsAtSectionBoundary(state)) return BRUNSLI_INVALID_BRN;

  return BRUNSLI_OK;
}

static BrunsliStatus DecodeQuantDataSection(DecState* state, JPEGData* jpg) {
  InternalState& s = *state->internal;
  QuantDataState& qs = s.quant;
  BitReader* br = &qs.br;

  if (qs.stage == QuantDataState::INIT) {
    *br = BitReader();
    qs.stage = QuantDataState::READ_NUM_QUANT;
  }
  PrepareBitReader(br, state);

  const auto suspend_bit_reader = [&](BrunsliStatus result) -> BrunsliStatus {
    return SuspendBitReader(br, state, result);
  };

  if (qs.stage == QuantDataState::READ_NUM_QUANT) {
    if (!br->CanReadWithinBounds(2)) {
      return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
    }
    size_t num_quant_tables = br->ReadFixedBits<2>() + 1;
    if (jpg->quant.size() != num_quant_tables) {
      return suspend_bit_reader(BRUNSLI_INVALID_BRN);
    }
    qs.predictor.resize(kDCTBlockSize);
    qs.i = 0;
    qs.stage = QuantDataState::READ_STOCK;
  }

  while (true) {
    switch (qs.stage) {
      case QuantDataState::READ_STOCK: {
        if (qs.i >= jpg->quant.size()) {
          std::vector<uint8_t>().swap(qs.predictor);
          qs.i = 0;
          qs.stage = QuantDataState::READ_QUANT_IDX;
          continue;
        }
        // Depending on еру 1-st bit, it is guaranteed that we will need to read
        // at least 3 or 6 more bits.
        if (!br->CanReadWithinBounds(4)) {
          return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
        }
        qs.data_precision = 0;
        bool is_short = !br->ReadFixedBits<1>();
        if (is_short) {
          const size_t short_code = br->ReadFixedBits<3>();
          int32_t* table = jpg->quant[qs.i].values.data();
          size_t selector = (qs.i > 0) ? 1 : 0;
          for (size_t k = 0; k < kDCTBlockSize; ++k) {
            table[k] = kStockQuantizationTables[selector][short_code][k];
          }
          qs.stage = QuantDataState::UPDATE;
        } else {
          qs.stage = QuantDataState::READ_Q_FACTOR;
        }
        continue;
      }

      case QuantDataState::READ_Q_FACTOR: {
        if (!br->CanReadWithinBounds(6)) {
          return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
        }
        const uint32_t q_factor = br->ReadFixedBits<6>();
        FillQuantMatrix(qs.i > 0, q_factor, qs.predictor.data());
        qs.j = 0;
        qs.delta = 0;
        qs.stage = QuantDataState::READ_DIFF_IS_ZERO;
        continue;
      }

      case QuantDataState::READ_DIFF_IS_ZERO: {
        if (qs.j >= kDCTBlockSize) {
          qs.stage = QuantDataState::UPDATE;
          continue;
        }
        if (!br->CanReadWithinBounds(1)) {
          return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
        }
        if (br->ReadFixedBits<1>()) {
          qs.stage = QuantDataState::READ_DIFF_SIGN;
        } else {
          qs.stage = QuantDataState::APPLY_DIFF;
        }
        continue;
      }

      case QuantDataState::READ_DIFF_SIGN: {
        if (!br->CanReadWithinBounds(1)) {
          return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
        }
        qs.sign = br->ReadFixedBits<1>() ? -1 : 1;
        qs.stage = QuantDataState::READ_DIFF;
        continue;
      }

      case QuantDataState::READ_DIFF: {
        if (!DecodeVarint(&qs.vs, br, 16)) {
          return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
        }
        const int diff = qs.vs.value + 1;
        qs.delta += qs.sign * diff;
        qs.stage = QuantDataState::APPLY_DIFF;
        continue;
      }

      case QuantDataState::APPLY_DIFF: {
        const int k = kJPEGNaturalOrder[qs.j];
        const int quant_value = qs.predictor[k] + qs.delta;
        jpg->quant[qs.i].values[k] = quant_value;
        if (quant_value <= 0) {
          return suspend_bit_reader(BRUNSLI_INVALID_BRN);
        }
        if (quant_value >= 256) {
          qs.data_precision = 1;
        }
        if (quant_value >= 65536) {
          return suspend_bit_reader(BRUNSLI_INVALID_BRN);
        }
        ++qs.j;
        qs.stage = QuantDataState::READ_DIFF_IS_ZERO;
        continue;
      }

      case QuantDataState::UPDATE: {
        if (jpg->quant[qs.i].precision != qs.data_precision) {
          return suspend_bit_reader(BRUNSLI_INVALID_BRN);
        }
        ++qs.i;
        qs.stage = QuantDataState::READ_STOCK;
        continue;
      }

      default: { /* no-op */ }
    }
    break;  // no matching stage has been found; exit the loop.
  }

  while (qs.stage == QuantDataState::READ_QUANT_IDX) {
    if (qs.i >= jpg->components.size()) {
      qs.stage = QuantDataState::FINISH;
      continue;
    }
    JPEGComponent* c = &jpg->components[qs.i];
    if (!br->CanReadWithinBounds(2)) {
      return suspend_bit_reader(BRUNSLI_NOT_ENOUGH_DATA);
    }
    c->quant_idx = br->ReadFixedBits<2>();
    if (c->quant_idx >= jpg->quant.size()) {
      return suspend_bit_reader(BRUNSLI_INVALID_BRN);
    }
    ++qs.i;
  }

  JXL_DASSERT(qs.stage == QuantDataState::FINISH);
  suspend_bit_reader(BRUNSLI_OK);
  Status all_padding_is_zero = br->JumpToByteBoundary();
  if (!br->Close() || !all_padding_is_zero) return BRUNSLI_INVALID_BRN;
  if (!IsAtSectionBoundary(state)) return BRUNSLI_INVALID_BRN;
  return BRUNSLI_OK;
}

static Stage ParseSection(DecState* state) {
  InternalState& s = *state->internal;
  SectionHeaderState& sh = s.section_header;

  Stage result = Stage::ERROR;

  while (sh.stage != SectionHeaderState::DONE) {
    switch (sh.stage) {
      case SectionHeaderState::READ_TAG: {
        BrunsliStatus status = ReadTag(state, &s.section);
        if (status == BRUNSLI_NOT_ENOUGH_DATA) {
          if (HasSection(state, kBrunsliACDataTag)) return Stage::DONE;
        }
        if (status != BRUNSLI_OK) return Fail(state, status);
        if (s.section.is_section) {
          sh.stage = SectionHeaderState::ENTER_SECTION;
          continue;
        }
        const uint32_t tag_bit = 1u << s.section.tag;
        const bool is_known_section_tag = kKnownSectionTags & tag_bit;
        if (is_known_section_tag) return Fail(state, BRUNSLI_INVALID_BRN);
        sh.stage = SectionHeaderState::READ_VALUE;
        continue;
      }

      case SectionHeaderState::READ_VALUE: {
        // No known varint tags on top level.
        size_t dummy;
        BrunsliStatus status = DecodeBase128(state, &dummy);
        if (status != BRUNSLI_OK) return Fail(state, status);
        result = Stage::SECTION;
        sh.stage = SectionHeaderState::DONE;
        continue;
      }

      case SectionHeaderState::ENTER_SECTION: {
        BrunsliStatus status = EnterSection(state, &s.section);
        if (status != BRUNSLI_OK) return Fail(state, status);
        result = Stage::SECTION_BODY;
        sh.stage = SectionHeaderState::DONE;
        continue;
      }

      default:
        return Fail(state, BRUNSLI_DECOMPRESSION_ERROR);
    }
  }

  sh.stage = SectionHeaderState::READ_TAG;
  JXL_DASSERT(result != Stage::ERROR);
  return result;
}

static Stage ProcessSection(DecState* state, JPEGData* jpg) {
  InternalState& s = *state->internal;

  const int32_t tag_bit = 1u << s.section.tag;
  const bool is_known_section_tag = kKnownSectionTags & tag_bit;

  const bool skip_section =
      !is_known_section_tag || (state->skip_tags & tag_bit);

  if (skip_section) {
    // Skip section content.
    size_t to_skip =
        std::min(GetBytesAvailable(state), RemainingSectionLength(state));
    state->pos += to_skip;
    if (RemainingSectionLength(state) != 0) {
      JXL_DASSERT(GetBytesAvailable(state) == 0);
      return Fail(state, BRUNSLI_NOT_ENOUGH_DATA);
    }
    return Stage::SECTION;
  }

  switch (s.section.tag) {
    case kBrunsliMetaDataTag: {
      BrunsliStatus status = DecodeMetaDataSection(state, jpg);
      if (status != BRUNSLI_OK) return Fail(state, status);
      break;
    }

    case kBrunsliJPEGInternalsTag: {
      BrunsliStatus status = DecodeJPEGInternalsSection(state, jpg);
      if (status != BRUNSLI_OK) return Fail(state, status);
      break;
    }

    case kBrunsliQuantDataTag: {
      if (!HasSection(state, kBrunsliJPEGInternalsTag)) {
        return Fail(state, BRUNSLI_INVALID_BRN);
      }
      BrunsliStatus status = DecodeQuantDataSection(state, jpg);
      if (status != BRUNSLI_OK) return Fail(state, status);
      break;
    }

    default:
      /* Unreachable */
      return Fail(state, BRUNSLI_INVALID_BRN);
  }

  if (!IsAtSectionBoundary(state)) {
    return Fail(state, BRUNSLI_INVALID_BRN);
  }

  // Nothing is expected after the AC data.
  if (s.section.tag == kBrunsliACDataTag) {
    return Stage::DONE;
  }

  return Stage::SECTION;
}

bool UpdateSubsamplingDerivatives(JPEGData* jpg) {
  for (size_t i = 0; i < jpg->components.size(); ++i) {
    JPEGComponent* c = &jpg->components[i];
    jpg->max_h_samp_factor = std::max(jpg->max_h_samp_factor, c->h_samp_factor);
    jpg->max_v_samp_factor = std::max(jpg->max_v_samp_factor, c->v_samp_factor);
  }
  jpg->MCU_rows = DivCeil(jpg->height, jpg->max_v_samp_factor * 8);
  jpg->MCU_cols = DivCeil(jpg->width, jpg->max_h_samp_factor * 8);
  for (size_t i = 0; i < jpg->components.size(); ++i) {
    JPEGComponent* c = &jpg->components[i];
    c->width_in_blocks = jpg->MCU_cols * c->h_samp_factor;
    c->height_in_blocks = jpg->MCU_rows * c->v_samp_factor;
    // 8205 == max[ceil((65535 / (i * 8)) * i) for i in range(1, 16 + 1)]
    JXL_DASSERT(c->width_in_blocks <= 8205);
    JXL_DASSERT(c->height_in_blocks <= 8205);
    uint32_t num_blocks = c->width_in_blocks * c->height_in_blocks;
    if (num_blocks > kBrunsliMaxNumBlocks) {
      return false;
    }
    c->num_blocks = num_blocks;
  }
  return true;
}

void PrepareMeta(const JPEGData* jpg, DecState* state) {
  InternalState& s = *state->internal;

  size_t num_components = jpg->components.size();
  s.block_state_.resize(num_components);
  std::vector<DecComponentMeta>& meta = state->meta;
  meta.resize(num_components);
  for (size_t i = 0; i < num_components; ++i) {
    const JPEGComponent& c = jpg->components[i];
    DecComponentMeta& m = meta[i];
    m.h_samp = c.h_samp_factor;
    m.v_samp = c.v_samp_factor;
    m.width_in_blocks = jpg->MCU_cols * m.h_samp;
    m.height_in_blocks = jpg->MCU_rows * m.v_samp;
  }
}

BrunsliStatus DoProcessJpeg(DecState* state, JPEGData* jpg) {
  while (true) {
    switch (state->stage) {
      case Stage::SIGNATURE:
        state->stage = VerifySignature(state);
        break;

      case Stage::HEADER:
        state->stage = DecodeHeader(state, jpg);
        break;

      case Stage::SECTION:
        state->stage = ParseSection(state);
        break;

      case Stage::SECTION_BODY:
        state->stage = ProcessSection(state, jpg);
        break;

      case Stage::DONE:
        // It is expected that there is no garbage after the valid brunsli
        // stream.
        if (state->pos != state->len) {
          state->stage = Fail(state, BRUNSLI_INVALID_BRN);
          break;
        }
        return BRUNSLI_OK;

      case Stage::ERROR:
        return state->internal->result;

      default:
        /* Unreachable */
        state->stage = Fail(state, BRUNSLI_DECOMPRESSION_ERROR);
        break;
    }
  }
}

/** Adds new input to buffer. */
void ChargeBuffer(DecState* state) {
  InternalState& s = *state->internal;
  Buffer& b = s.buffer;

  b.borrowed_len = 0;
  b.external_data = state->data;
  b.external_pos = state->pos;
  b.external_len = state->len;
}

constexpr size_t kBufferMaxReadAhead = 600;

/** Sets input source either to buffered, or to external data. */
void LoadInput(DecState* state) {
  InternalState& s = *state->internal;
  Buffer& b = s.buffer;

  // No data buffered. Just pass external data as is.
  if (b.data_len == 0) {
    state->data = b.external_data;
    state->pos = b.external_pos;
    state->len = b.external_len;
    return;
  }

  JXL_DASSERT(b.data_len <= kBufferMaxReadAhead);

  // Otherwise use buffered data.
  size_t available = b.external_len - b.external_pos;
  // Always try to borrow as much as parser could require. This way, when
  // buffer is unable to provide enough input, we could switch to unbuffered
  // input.
  b.borrowed_len = std::min(kBufferMaxReadAhead, available);
  memcpy(b.data.data() + b.data_len, b.external_data + b.external_pos,
         b.borrowed_len);
  state->data = b.data.data();
  state->pos = 0;
  state->len = b.data_len + b.borrowed_len;
}

/**
 * Cancel borrowed bytes, if any.
 *
 * Returns false, if it is impossible to continue parsing.
 */
bool UnloadInput(DecState* state, BrunsliStatus result) {
  InternalState& s = *state->internal;
  Buffer& b = s.buffer;

  // Non-buffered input; put tail to buffer.
  if (state->data == b.external_data) {
    b.external_pos = state->pos;
    JXL_DASSERT(b.external_pos <= b.external_len);
    if (result != BRUNSLI_NOT_ENOUGH_DATA) return true;
    JXL_DASSERT(b.data_len == 0);
    size_t available = b.external_len - b.external_pos;
    JXL_DASSERT(available < kBufferMaxReadAhead);
    if (b.data.empty()) b.data.resize(2 * kBufferMaxReadAhead);
    b.data_len = available;
    memcpy(b.data.data(), b.external_data + b.external_pos, b.data_len);
    b.external_pos += available;
    return false;
  }

  // Buffer depleted; switch to non-buffered input.
  if (state->pos >= b.data_len) {
    size_t used_borrowed_bytes = state->pos - b.data_len;
    b.data_len = 0;
    b.external_pos += used_borrowed_bytes;
    return true;
  }

  // Buffer not depleted; either problem discovered was already buffered data,
  // or extra input was too-short.
  b.data_len -= state->pos;
  if (result == BRUNSLI_NOT_ENOUGH_DATA) {
    // We couldn't have taken more bytes.
    JXL_DASSERT(b.external_pos + b.borrowed_len == b.external_len);
    // Remaining piece is not too large.
    JXL_DASSERT(b.data_len + b.borrowed_len < kBufferMaxReadAhead);
    b.data_len += b.borrowed_len;
    b.external_pos += b.borrowed_len;
  }
  JXL_DASSERT(!b.data.empty());
  if (state->pos > 0 && b.data_len > 0) {
    memmove(b.data.data(), b.data.data() + state->pos, b.data_len);
  }
  JXL_DASSERT(b.data_len <= kBufferMaxReadAhead);

  return (result != BRUNSLI_NOT_ENOUGH_DATA);
}

/** Sets back user-provided input. */
void UnchargeBuffer(DecState* state) {
  InternalState& s = *state->internal;
  Buffer& b = s.buffer;

  state->data = b.external_data;
  state->pos = b.external_pos;
  state->len = b.external_len;
}

BrunsliStatus ProcessJpeg(DecState* state, JPEGData* jpg) {
  InternalState& s = *state->internal;

  if (state->pos > state->len) return BRUNSLI_INVALID_PARAM;
  ChargeBuffer(state);

  BrunsliStatus result = BRUNSLI_NOT_ENOUGH_DATA;
  while (result == BRUNSLI_NOT_ENOUGH_DATA) {
    if (state->stage == Stage::ERROR) {
      // General error -> no recovery.
      if (s.result != BRUNSLI_NOT_ENOUGH_DATA) return s.result;
      // Continue parsing.
      s.result = BRUNSLI_OK;
      state->stage = s.last_stage;
      s.last_stage = Stage::ERROR;
    }

    LoadInput(state);
    if (s.section.is_active) {
      s.section.milestone = state->pos;
      s.section.projected_end = s.section.milestone + s.section.remaining;
    }

    s.section.tags_met |= state->tags_met;
    result = DoProcessJpeg(state, jpg);

    if (s.section.is_active) {
      // TODO(eustas): dcheck state->pos > s.section.milestone
      size_t processed_len = state->pos - s.section.milestone;
      // TODO(eustas): dcheck processed_len < s.section.remaining
      s.section.remaining -= processed_len;
    }

    if (!UnloadInput(state, result)) break;
  }
  UnchargeBuffer(state);
  return result;
}

}  // namespace jpeg
}  // namespace jxl
