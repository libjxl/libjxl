// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/decode_marker.h"

#include <string.h>

#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/huffman.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jpegli/source_manager.h"
#include "lib/jxl/base/printf_macros.h"

typedef jpeg_decomp_master::State State;

namespace jpegli {
namespace {

constexpr int kMaxSampling = 2;
constexpr int kMaxHuffmanTables = 4;
constexpr int kMaxQuantTables = 4;
constexpr int kMaxDimPixels = 65535;
constexpr uint8_t kIccProfileTag[12] = "ICC_PROFILE";

// Macros for commonly used error conditions.

#define JPEG_VERIFY_LEN(n)                                     \
  if (pos + (n) > len) {                                       \
    return JPEGLI_ERROR("Unexpected end of input: pos=%" PRIuS \
                        " need=%d len=%" PRIuS,                \
                        pos, static_cast<int>(n), len);        \
  }

#define JPEG_VERIFY_INPUT(var, low, high)                               \
  if ((var) < (low) || (var) > (high)) {                                \
    return JPEGLI_ERROR("Invalid " #var ": %d", static_cast<int>(var)); \
  }

#define JPEG_VERIFY_MARKER_END()                                  \
  if (pos != len) {                                               \
    return JPEGLI_ERROR("Invalid marker length: declared=%" PRIuS \
                        " actual=%" PRIuS,                        \
                        len, pos);                                \
  }

inline int ReadUint8(const uint8_t* data, size_t* pos) {
  return data[(*pos)++];
}

inline int ReadUint16(const uint8_t* data, size_t* pos) {
  int v = (data[*pos] << 8) + data[*pos + 1];
  *pos += 2;
  return v;
}

void ProcessSOF(j_decompress_ptr cinfo, const uint8_t* data, size_t len) {
  jpeg_decomp_master* m = cinfo->master;
  if (!m->found_soi_) {
    JPEGLI_ERROR("Unexpected SOF marker.");
  }
  if (m->found_sof_) {
    JPEGLI_ERROR("Duplicate SOF marker.");
  }
  m->found_sof_ = true;
  m->is_progressive_ = (data[1] == 0xc2);
  size_t pos = 4;
  JPEG_VERIFY_LEN(6);
  int precision = ReadUint8(data, &pos);
  cinfo->image_height = ReadUint16(data, &pos);
  cinfo->image_width = ReadUint16(data, &pos);
  cinfo->arith_code = 0;
  cinfo->num_components = ReadUint8(data, &pos);
  JPEG_VERIFY_INPUT(precision, 8, 8);
  JPEG_VERIFY_INPUT(cinfo->image_height, 1, kMaxDimPixels);
  JPEG_VERIFY_INPUT(cinfo->image_width, 1, kMaxDimPixels);
  JPEG_VERIFY_INPUT(cinfo->num_components, 1, kMaxComponents);
  JPEG_VERIFY_LEN(3 * cinfo->num_components);
  m->components_.resize(cinfo->num_components);

  // Read sampling factors and quant table index for each component.
  std::vector<bool> ids_seen(256, false);
  m->max_h_samp_ = 1;
  m->max_v_samp_ = 1;
  for (size_t i = 0; i < m->components_.size(); ++i) {
    JPEGComponent* c = &m->components_[i];
    const int id = ReadUint8(data, &pos);
    if (ids_seen[id]) {  // (cf. section B.2.2, syntax of Ci)
      JPEGLI_ERROR("Duplicate ID %d in SOF.", id);
    }
    ids_seen[id] = true;
    c->id = id;
    int factor = ReadUint8(data, &pos);
    int h_samp_factor = factor >> 4;
    int v_samp_factor = factor & 0xf;
    JPEG_VERIFY_INPUT(h_samp_factor, 1, kMaxSampling);
    JPEG_VERIFY_INPUT(v_samp_factor, 1, kMaxSampling);
    c->h_samp_factor = h_samp_factor;
    c->v_samp_factor = v_samp_factor;
    m->max_h_samp_ = std::max(m->max_h_samp_, h_samp_factor);
    m->max_v_samp_ = std::max(m->max_v_samp_, v_samp_factor);
    uint8_t quant_tbl_idx = ReadUint8(data, &pos);
    bool found_quant_tbl = false;
    for (size_t j = 0; j < m->quant_.size(); ++j) {
      if (m->quant_[j].index == quant_tbl_idx) {
        c->quant_idx = j;
        found_quant_tbl = true;
        break;
      }
    }
    if (!found_quant_tbl) {
      JPEGLI_ERROR("Quantization table with index %u not found", quant_tbl_idx);
    }
  }
  JPEG_VERIFY_MARKER_END();

  if (cinfo->num_components == 1) {
    m->is_ycbcr_ = true;
  }
  if (!m->found_app0_ && cinfo->num_components == 3 &&
      m->components_[0].id == 'R' && m->components_[1].id == 'G' &&
      m->components_[2].id == 'B') {
    m->is_ycbcr_ = false;
  }

  // We have checked above that none of the sampling factors are 0, so the max
  // sampling factors can not be 0.
  m->iMCU_height_ = m->max_v_samp_ * DCTSIZE;
  m->iMCU_width_ = m->max_h_samp_ * DCTSIZE;
  m->iMCU_rows_ = DivCeil(cinfo->image_height, m->iMCU_height_);
  m->iMCU_cols_ = DivCeil(cinfo->image_width, m->iMCU_width_);
  // Compute the block dimensions for each component.
  for (size_t i = 0; i < m->components_.size(); ++i) {
    JPEGComponent* c = &m->components_[i];
    if (m->max_h_samp_ % c->h_samp_factor != 0 ||
        m->max_v_samp_ % c->v_samp_factor != 0) {
      JPEGLI_ERROR("Non-integral subsampling ratios.");
    }
    c->width_in_blocks = m->iMCU_cols_ * c->h_samp_factor;
    c->height_in_blocks = m->iMCU_rows_ * c->v_samp_factor;
    const uint64_t num_blocks =
        static_cast<uint64_t>(c->width_in_blocks) * c->height_in_blocks;
    c->coeffs = hwy::AllocateAligned<coeff_t>(num_blocks * DCTSIZE2);
    memset(c->coeffs.get(), 0, num_blocks * DCTSIZE2 * sizeof(coeff_t));
  }
  memset(m->scan_progression_, 0, sizeof(m->scan_progression_));
}

void ProcessSOS(j_decompress_ptr cinfo, const uint8_t* data, size_t len) {
  jpeg_decomp_master* m = cinfo->master;
  if (!m->found_sof_) {
    JPEGLI_ERROR("Unexpected SOS marker.");
  }
  m->found_sos_ = true;
  size_t pos = 4;
  JPEG_VERIFY_LEN(1);
  size_t comps_in_scan = ReadUint8(data, &pos);
  JPEG_VERIFY_INPUT(comps_in_scan, 1, m->components_.size());

  m->scan_info_.num_components = comps_in_scan;
  JPEG_VERIFY_LEN(2 * comps_in_scan);
  bool is_interleaved = (m->scan_info_.num_components > 1);
  std::vector<bool> ids_seen(256, false);
  for (size_t i = 0; i < m->scan_info_.num_components; ++i) {
    JPEGComponentScanInfo* si = &m->scan_info_.components[i];
    uint32_t id = ReadUint8(data, &pos);
    if (ids_seen[id]) {  // (cf. section B.2.3, regarding CSj)
      return JPEGLI_ERROR("Duplicate ID %d in SOS.", id);
    }
    ids_seen[id] = true;
    JPEGComponent* comp = nullptr;
    for (size_t j = 0; j < m->components_.size(); ++j) {
      if (m->components_[j].id == id) {
        si->comp_idx = j;
        comp = &m->components_[j];
      }
    }
    if (!comp) {
      return JPEGLI_ERROR("SOS marker: Could not find component with id %d",
                          id);
    }
    int c = ReadUint8(data, &pos);
    si->dc_tbl_idx = c >> 4;
    si->ac_tbl_idx = c & 0xf;
    JPEG_VERIFY_INPUT(static_cast<int>(si->dc_tbl_idx), 0, 3);
    JPEG_VERIFY_INPUT(static_cast<int>(si->ac_tbl_idx), 0, 3);
    si->mcu_xsize_blocks = is_interleaved ? comp->h_samp_factor : 1;
    si->mcu_ysize_blocks = is_interleaved ? comp->v_samp_factor : 1;
  }
  JPEG_VERIFY_LEN(3);
  m->scan_info_.Ss = ReadUint8(data, &pos);
  m->scan_info_.Se = ReadUint8(data, &pos);
  JPEG_VERIFY_INPUT(static_cast<int>(m->scan_info_.Ss), 0, 63);
  JPEG_VERIFY_INPUT(m->scan_info_.Se, m->scan_info_.Ss, 63);
  int c = ReadUint8(data, &pos);
  m->scan_info_.Ah = c >> 4;
  m->scan_info_.Al = c & 0xf;
  JPEG_VERIFY_MARKER_END();

  if (m->scan_info_.Ah != 0 && m->scan_info_.Al != m->scan_info_.Ah - 1) {
    // section G.1.1.1.2 : Successive approximation control only improves
    // by one bit at a time.
    JPEGLI_ERROR("Invalid progressive parameters: Al=%d Ah=%d",
                 m->scan_info_.Al, m->scan_info_.Ah);
  }
  if (!m->is_progressive_) {
    m->scan_info_.Ss = 0;
    m->scan_info_.Se = 63;
    m->scan_info_.Ah = 0;
    m->scan_info_.Al = 0;
  }
  const uint16_t scan_bitmask = m->scan_info_.Ah == 0
                                    ? (0xffff << m->scan_info_.Al)
                                    : (1u << m->scan_info_.Al);
  const uint16_t refinement_bitmask = (1 << m->scan_info_.Al) - 1;
  for (size_t i = 0; i < m->scan_info_.num_components; ++i) {
    int comp_idx = m->scan_info_.components[i].comp_idx;
    for (uint32_t k = m->scan_info_.Ss; k <= m->scan_info_.Se; ++k) {
      if (m->scan_progression_[comp_idx][k] & scan_bitmask) {
        return JPEGLI_ERROR(
            "Overlapping scans: component=%d k=%d prev_mask: %u cur_mask %u",
            comp_idx, k, m->scan_progression_[i][k], scan_bitmask);
      }
      if (m->scan_progression_[comp_idx][k] & refinement_bitmask) {
        return JPEGLI_ERROR(
            "Invalid scan order, a more refined scan was already done: "
            "component=%d k=%d prev_mask=%u cur_mask=%u",
            comp_idx, k, m->scan_progression_[i][k], scan_bitmask);
      }
      m->scan_progression_[comp_idx][k] |= scan_bitmask;
    }
  }
  if (m->scan_info_.Al > 10) {
    return JPEGLI_ERROR("Scan parameter Al=%d is not supported.",
                        m->scan_info_.Al);
  }
  // Check that all the Huffman tables needed for this scan are defined.
  for (size_t i = 0; i < comps_in_scan; ++i) {
    if (m->scan_info_.Ss == 0 &&
        !m->huff_slot_defined_[m->scan_info_.components[i].dc_tbl_idx]) {
      return JPEGLI_ERROR(
          "SOS marker: Could not find DC Huffman table with index %d",
          m->scan_info_.components[i].dc_tbl_idx);
    }
    if (m->scan_info_.Se > 0 &&
        !m->huff_slot_defined_[m->scan_info_.components[i].ac_tbl_idx + 16]) {
      return JPEGLI_ERROR(
          "SOS marker: Could not find AC Huffman table with index %d",
          m->scan_info_.components[i].ac_tbl_idx);
    }
  }
  m->scan_info_.MCU_rows = m->iMCU_rows_;
  m->scan_info_.MCU_cols = m->iMCU_cols_;
  if (!is_interleaved) {
    const JPEGComponent& c =
        m->components_[m->scan_info_.components[0].comp_idx];
    m->scan_info_.MCU_cols =
        DivCeil(cinfo->image_width * c.h_samp_factor, m->iMCU_width_);
    m->scan_info_.MCU_rows =
        DivCeil(cinfo->image_height * c.v_samp_factor, m->iMCU_height_);
  }
  memset(m->last_dc_coeff_, 0, sizeof(m->last_dc_coeff_));
  m->restarts_to_go_ = m->restart_interval_;
  m->next_restart_marker_ = 0;
  m->eobrun_ = -1;
  m->scan_mcu_row_ = 0;
  m->scan_mcu_col_ = 0;
  m->codestream_bits_ahead_ = 0;
  size_t mcu_size = 0;
  for (size_t i = 0; i < m->scan_info_.num_components; ++i) {
    JPEGComponentScanInfo* si = &m->scan_info_.components[i];
    mcu_size += si->mcu_ysize_blocks * si->mcu_xsize_blocks;
  }
  m->mcu_.coeffs.resize(mcu_size * DCTSIZE2);
  m->state_ = State::kScan;
}

// Reads the Define Huffman Table (DHT) marker segment and builds the Huffman
// decoding table in either dc_huff_lut_ or ac_huff_lut_, depending on the type
// and solt_id of Huffman code being read.
void ProcessDHT(j_decompress_ptr cinfo, const uint8_t* data, size_t len) {
  jpeg_decomp_master* m = cinfo->master;
  constexpr int kLutSize = kMaxHuffmanTables * kJpegHuffmanLutSize;
  m->dc_huff_lut_.resize(kLutSize);
  m->ac_huff_lut_.resize(kLutSize);
  size_t pos = 4;
  if (pos == len) {
    return JPEGLI_ERROR("DHT marker: no Huffman table found");
  }
  while (pos < len) {
    JPEG_VERIFY_LEN(1 + kJpegHuffmanMaxBitLength);
    // The index of the Huffman code in the current set of Huffman codes. For AC
    // component Huffman codes, 0x10 is added to the index.
    int slot_id = ReadUint8(data, &pos);
    m->huff_slot_defined_[slot_id] = 1;
    int huffman_index = slot_id;
    int is_ac_table = (slot_id & 0x10) != 0;
    HuffmanTableEntry* huff_lut;
    if (is_ac_table) {
      huffman_index -= 0x10;
      JPEG_VERIFY_INPUT(huffman_index, 0, 3);
      huff_lut = &m->ac_huff_lut_[huffman_index * kJpegHuffmanLutSize];
    } else {
      JPEG_VERIFY_INPUT(huffman_index, 0, 3);
      huff_lut = &m->dc_huff_lut_[huffman_index * kJpegHuffmanLutSize];
    }
    // Bit length histogram->
    std::array<uint32_t, kJpegHuffmanMaxBitLength + 1> counts = {};
    counts[0] = 0;
    int total_count = 0;
    int space = 1 << kJpegHuffmanMaxBitLength;
    int max_depth = 1;
    for (size_t i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
      int count = ReadUint8(data, &pos);
      if (count != 0) {
        max_depth = i;
      }
      counts[i] = count;
      total_count += count;
      space -= count * (1 << (kJpegHuffmanMaxBitLength - i));
    }
    if (is_ac_table) {
      JPEG_VERIFY_INPUT(total_count, 0, kJpegHuffmanAlphabetSize);
    } else {
      JPEG_VERIFY_INPUT(total_count, 0, kJpegDCAlphabetSize);
    }
    JPEG_VERIFY_LEN(total_count);
    // Symbol values sorted by increasing bit lengths.
    std::array<uint32_t, kJpegHuffmanAlphabetSize + 1> values = {};
    std::vector<bool> values_seen(256, false);
    for (int i = 0; i < total_count; ++i) {
      int value = ReadUint8(data, &pos);
      if (!is_ac_table) {
        JPEG_VERIFY_INPUT(value, 0, kJpegDCAlphabetSize - 1);
      }
      if (values_seen[value]) {
        return JPEGLI_ERROR("Duplicate Huffman code value %d", value);
      }
      values_seen[value] = true;
      values[i] = value;
    }
    // Add an invalid symbol that will have the all 1 code.
    ++counts[max_depth];
    values[total_count] = kJpegHuffmanAlphabetSize;
    space -= (1 << (kJpegHuffmanMaxBitLength - max_depth));
    if (space < 0) {
      return JPEGLI_ERROR("Invalid Huffman code lengths.");
    } else if (space > 0 && huff_lut[0].value != 0xffff) {
      // Re-initialize the values to an invalid symbol so that we can recognize
      // it when reading the bit stream using a Huffman code with space > 0.
      for (int i = 0; i < kJpegHuffmanLutSize; ++i) {
        huff_lut[i].bits = 0;
        huff_lut[i].value = 0xffff;
      }
    }
    BuildJpegHuffmanTable(&counts[0], &values[0], huff_lut);
  }
  JPEG_VERIFY_MARKER_END();
}

void ProcessDQT(j_decompress_ptr cinfo, const uint8_t* data, size_t len) {
  jpeg_decomp_master* m = cinfo->master;
  size_t pos = 4;
  if (pos == len) {
    return JPEGLI_ERROR("DQT marker: no quantization table found");
  }
  while (pos < len && m->quant_.size() < kMaxQuantTables) {
    JPEG_VERIFY_LEN(1);
    int quant_table_index = ReadUint8(data, &pos);
    int precision = quant_table_index >> 4;
    JPEG_VERIFY_INPUT(precision, 0, 1);
    quant_table_index &= 0xf;
    JPEG_VERIFY_INPUT(quant_table_index, 0, 3);
    JPEG_VERIFY_LEN((precision + 1) * DCTSIZE2);
    JPEGQuantTable table;
    table.index = quant_table_index;
    for (size_t i = 0; i < DCTSIZE2; ++i) {
      int quant_val =
          precision ? ReadUint16(data, &pos) : ReadUint8(data, &pos);
      JPEG_VERIFY_INPUT(quant_val, 1, 65535);
      table.values[kJPEGNaturalOrder[i]] = quant_val;
    }
    m->quant_.push_back(table);
  }
  JPEG_VERIFY_MARKER_END();
}

void ProcessDRI(j_decompress_ptr cinfo, const uint8_t* data, size_t len) {
  jpeg_decomp_master* m = cinfo->master;
  if (m->found_dri_) {
    return JPEGLI_ERROR("Duplicate DRI marker.");
  }
  m->found_dri_ = true;
  size_t pos = 4;
  JPEG_VERIFY_LEN(2);
  m->restart_interval_ = ReadUint16(data, &pos);
  JPEG_VERIFY_MARKER_END();
}

void ProcessAPP(j_decompress_ptr cinfo, const uint8_t* data, size_t len) {
  jpeg_decomp_master* m = cinfo->master;
  const uint8_t marker = data[1];
  const uint8_t* payload = data + 4;
  size_t payload_size = len - 4;
  if (marker == 0xE0) {
    m->found_app0_ = true;
    m->is_ycbcr_ = true;
  } else if (!m->found_app0_ && marker == 0xEE && payload_size == 12 &&
             memcmp(payload, "Adobe", 5) == 0 && payload[11] == 0) {
    m->is_ycbcr_ = false;
  }
  if (marker == 0xE2) {
    if (payload_size >= sizeof(kIccProfileTag) &&
        memcmp(payload, kIccProfileTag, sizeof(kIccProfileTag)) == 0) {
      payload += sizeof(kIccProfileTag);
      payload_size -= sizeof(kIccProfileTag);
      if (payload_size < 2) {
        return JPEGLI_ERROR("ICC chunk is too small.");
      }
      uint8_t index = payload[0];
      uint8_t total = payload[1];
      ++m->icc_index_;
      if (m->icc_index_ != index) {
        return JPEGLI_ERROR("Invalid ICC chunk order.");
      }
      if (total == 0) {
        return JPEGLI_ERROR("Invalid ICC chunk total.");
      }
      if (m->icc_total_ == 0) {
        m->icc_total_ = total;
      } else if (m->icc_total_ != total) {
        return JPEGLI_ERROR("Invalid ICC chunk total.");
      }
      if (m->icc_index_ > m->icc_total_) {
        return JPEGLI_ERROR("Invalid ICC chunk index.");
      }
      m->icc_profile_.insert(m->icc_profile_.end(), payload + 2,
                             payload + payload_size);
    }
  }
}

void ProcessCOM(j_decompress_ptr cinfo, const uint8_t* data, size_t len) {
  // Nothing to do.
}

void SaveMarker(j_decompress_ptr cinfo, const uint8_t* data, size_t len) {
  const uint8_t marker = data[1];
  const uint8_t* payload = data + 4;
  size_t payload_size = len - 4;

  // Insert new saved marker to the head of the list.
  jpeg_saved_marker_ptr next = cinfo->marker_list;
  cinfo->marker_list = (jpeg_marker_struct*)malloc(sizeof(jpeg_marker_struct));
  cinfo->marker_list->next = next;
  cinfo->marker_list->marker = marker;
  cinfo->marker_list->original_length = payload_size;
  cinfo->marker_list->data_length = payload_size;
  cinfo->marker_list->data = (uint8_t*)malloc(payload_size);
  memcpy(cinfo->marker_list->data, payload, payload_size);

  // Remember to free the newly allocated pointers.
  auto mem = reinterpret_cast<jpegli::MemoryManager*>(cinfo->mem);
  mem->owned_ptrs.push_back(cinfo->marker_list);
  mem->owned_ptrs.push_back(cinfo->marker_list->data);
}

}  // namespace

bool ProcessMarker(j_decompress_ptr cinfo, const uint8_t* data, size_t len,
                   size_t* pos) {
  jpeg_decomp_master* m = cinfo->master;
  // kIsValidMarker[i] == 1 means (0xc0 + i) is a valid marker.
  static const uint8_t kIsValidMarker[] = {
      1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
      1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
  };
  // Skip bytes between markers.
  size_t num_skipped = 0;
  while (*pos + 1 < len && (data[*pos] != 0xff || data[*pos + 1] < 0xc0 ||
                            !kIsValidMarker[data[*pos + 1] - 0xc0])) {
    ++(*pos);
    ++num_skipped;
  }
  if (*pos + 2 > len) {
    return false;
  }
  if (num_skipped > 0) {
    AdvanceInput(cinfo, num_skipped);
  }
  uint8_t marker = data[*pos + 1];
  if (marker == 0xd9) {
    m->found_eoi_ = true;
    m->state_ = m->is_progressive_ ? State::kRender : State::kEnd;
    *pos += 2;
    AdvanceInput(cinfo, 2);
    return true;
  }
  if (*pos + 4 > len) {
    return false;
  }
  const uint8_t* marker_data = &data[*pos];
  size_t marker_len = (data[*pos + 2] << 8) + data[*pos + 3] + 2;
  if (marker_len < 4) {
    JPEGLI_ERROR("Invalid marker length");
  }
  if (*pos + marker_len > len) {
    return false;
  }
  if (m->markers_to_save_.find(marker) != m->markers_to_save_.end()) {
    SaveMarker(cinfo, marker_data, marker_len);
  }
  if (marker == 0xc0 || marker == 0xc1 || marker == 0xc2) {
    ProcessSOF(cinfo, marker_data, marker_len);
  } else if (marker == 0xc4) {
    ProcessDHT(cinfo, marker_data, marker_len);
  } else if (marker == 0xda) {
    ProcessSOS(cinfo, marker_data, marker_len);
  } else if (marker == 0xdb) {
    ProcessDQT(cinfo, marker_data, marker_len);
  } else if (marker == 0xdd) {
    ProcessDRI(cinfo, marker_data, marker_len);
  } else if (marker >= 0xe0 && marker <= 0xef) {
    ProcessAPP(cinfo, marker_data, marker_len);
  } else if (marker == 0xfe) {
    ProcessCOM(cinfo, marker_data, marker_len);
  } else {
    JPEGLI_ERROR("Unexpected marker 0x%x", marker);
  }
  *pos += marker_len;
  AdvanceInput(cinfo, marker_len);
  return true;
}

}  // namespace jpegli
