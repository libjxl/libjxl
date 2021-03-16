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

#include "lib/jxl/jpeg/jpeg_data.h"

#include "lib/jxl/base/status.h"

namespace jxl {
namespace jpeg {

namespace {
enum JPEGComponentType : uint32_t {
  kGray = 0,
  kYCbCr = 1,
  kRGB = 2,
  kCustom = 3,
};

struct JPEGInfo {
  size_t num_app_markers = 0;
  size_t num_com_markers = 0;
  size_t num_scans = 0;
  size_t num_intermarker = 0;
  bool has_dri = false;
};

Status VisitMarker(uint8_t* marker, Visitor* visitor, JPEGInfo* info) {
  uint32_t marker32 = *marker - 0xc0;
  JXL_RETURN_IF_ERROR(visitor->Bits(6, 0x00, &marker32));
  *marker = marker32 + 0xc0;
  if ((*marker & 0xf0) == 0xe0) {
    info->num_app_markers++;
  }
  if (*marker == 0xfe) {
    info->num_com_markers++;
  }
  if (*marker == 0xda) {
    info->num_scans++;
  }
  // We use a fake 0xff marker to signal intermarker data.
  if (*marker == 0xff) {
    info->num_intermarker++;
  }
  if (*marker == 0xdd) {
    info->has_dri = true;
  }
  return true;
}

}  // namespace

Status JPEGData::VisitFields(Visitor* visitor) {
  bool is_gray = components.size() == 1;
  JXL_RETURN_IF_ERROR(visitor->Bool(false, &is_gray));
  if (visitor->IsReading()) {
    components.resize(is_gray ? 1 : 3);
  }
  JPEGInfo info;
  if (visitor->IsReading()) {
    uint8_t marker = 0xc0;
    do {
      JXL_RETURN_IF_ERROR(VisitMarker(&marker, visitor, &info));
      marker_order.push_back(marker);
      if (marker_order.size() > 16384) {
        return JXL_FAILURE("Too many markers: %zu\n", marker_order.size());
      }
    } while (marker != 0xd9);
  } else {
    if (marker_order.size() > 16384) {
      return JXL_FAILURE("Too many markers: %zu\n", marker_order.size());
    }
    for (size_t i = 0; i < marker_order.size(); i++) {
      JXL_RETURN_IF_ERROR(VisitMarker(&marker_order[i], visitor, &info));
    }
    if (!marker_order.empty()) {
      // Last marker should always be EOI marker.
      JXL_CHECK(marker_order.back() == 0xd9);
    }
  }

  // Size of the APP and COM markers.
  if (visitor->IsReading()) {
    app_data.resize(info.num_app_markers);
    app_marker_type.resize(info.num_app_markers);
    com_data.resize(info.num_com_markers);
    scan_info.resize(info.num_scans);
    inter_marker_data.resize(info.num_intermarker);
  }
  JXL_ASSERT(app_data.size() == info.num_app_markers);
  JXL_ASSERT(app_marker_type.size() == info.num_app_markers);
  JXL_ASSERT(com_data.size() == info.num_com_markers);
  JXL_ASSERT(scan_info.size() == info.num_scans);
  JXL_ASSERT(inter_marker_data.size() == info.num_intermarker);
  for (size_t i = 0; i < app_data.size(); i++) {
    auto& app = app_data[i];
    // Encodes up to 8 different values.
    JXL_RETURN_IF_ERROR(
        visitor->U32(Val(0), Val(1), BitsOffset(1, 2), BitsOffset(2, 4), 0,
                     reinterpret_cast<uint32_t*>(&app_marker_type[i])));
    if (app_marker_type[i] != AppMarkerType::kUnknown &&
        app_marker_type[i] != AppMarkerType::kICC &&
        app_marker_type[i] != AppMarkerType::kExif &&
        app_marker_type[i] != AppMarkerType::kXMP) {
      return JXL_FAILURE("Unknown app marker type %u",
                         static_cast<uint32_t>(app_marker_type[i]));
    }
    uint32_t len = app.size() - 1;
    JXL_RETURN_IF_ERROR(visitor->Bits(16, 0, &len));
    if (visitor->IsReading()) app.resize(len + 1);
    if (app.size() < 3) {
      return JXL_FAILURE("Invalid marker size: %zu\n", app.size());
    }
  }
  for (auto& com : com_data) {
    uint32_t len = com.size() - 1;
    JXL_RETURN_IF_ERROR(visitor->Bits(16, 0, &len));
    if (visitor->IsReading()) com.resize(len + 1);
    if (com.size() < 3) {
      return JXL_FAILURE("Invalid marker size: %zu\n", com.size());
    }
  }

  uint32_t num_quant_tables = quant.size();
  JXL_RETURN_IF_ERROR(
      visitor->U32(Val(1), Val(2), Val(3), Val(4), 2, &num_quant_tables));
  if (num_quant_tables == 4) {
    return JXL_FAILURE("Invalid number of quant tables");
  }
  if (visitor->IsReading()) {
    quant.resize(num_quant_tables);
  }
  for (size_t i = 0; i < num_quant_tables; i++) {
    if (quant[i].precision > 1) {
      return JXL_FAILURE(
          "Quant tables with more than 16 bits are not supported");
    }
    JXL_RETURN_IF_ERROR(visitor->Bits(1, 0, &quant[i].precision));
    JXL_RETURN_IF_ERROR(visitor->Bits(2, i, &quant[i].index));
    JXL_RETURN_IF_ERROR(visitor->Bool(true, &quant[i].is_last));
  }

  JPEGComponentType component_type =
      components.size() == 1 && components[0].id == 1
          ? JPEGComponentType::kGray
          : components.size() == 3 && components[0].id == 1 &&
                  components[1].id == 2 && components[2].id == 3
              ? JPEGComponentType::kYCbCr
              : components.size() == 3 && components[0].id == 'R' &&
                      components[1].id == 'G' && components[2].id == 'B'
                    ? JPEGComponentType::kRGB
                    : JPEGComponentType::kCustom;
  JXL_RETURN_IF_ERROR(
      visitor->Bits(2, JPEGComponentType::kYCbCr,
                    reinterpret_cast<uint32_t*>(&component_type)));
  uint32_t num_components;
  if (component_type == JPEGComponentType::kGray) {
    num_components = 1;
  } else if (component_type != JPEGComponentType::kCustom) {
    num_components = 3;
  } else {
    num_components = components.size();
    JXL_RETURN_IF_ERROR(
        visitor->U32(Val(1), Val(2), Val(3), Val(4), 3, &num_components));
    if (num_components != 1 && num_components != 3) {
      return JXL_FAILURE("Invalid number of components: %u", num_components);
    }
  }
  if (visitor->IsReading()) {
    components.resize(num_components);
  }
  if (component_type == JPEGComponentType::kCustom) {
    for (size_t i = 0; i < components.size(); i++) {
      JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &components[i].id));
    }
  } else if (component_type == JPEGComponentType::kGray) {
    components[0].id = 1;
  } else if (component_type == JPEGComponentType::kRGB) {
    components[0].id = 'R';
    components[1].id = 'G';
    components[2].id = 'B';
  } else {
    components[0].id = 1;
    components[1].id = 2;
    components[2].id = 3;
  }
  size_t used_tables = 0;
  for (size_t i = 0; i < components.size(); i++) {
    JXL_RETURN_IF_ERROR(visitor->Bits(2, 0, &components[i].quant_idx));
    if (components[i].quant_idx >= quant.size()) {
      return JXL_FAILURE("Invalid quant table for component %zu: %u\n", i,
                         components[i].quant_idx);
    }
    used_tables |= 1U << components[i].quant_idx;
  }
  if (used_tables + 1 != 1U << quant.size()) {
    return JXL_FAILURE(
        "Not all quant tables are used (%zu tables, %zx used table mask)",
        quant.size(), used_tables);
  }

  uint32_t num_huff = huffman_code.size();
  JXL_RETURN_IF_ERROR(visitor->U32(Val(4), BitsOffset(3, 2), BitsOffset(4, 10),
                                   BitsOffset(6, 26), 4, &num_huff));
  if (visitor->IsReading()) {
    huffman_code.resize(num_huff);
  }
  for (JPEGHuffmanCode& hc : huffman_code) {
    bool is_ac = hc.slot_id >> 4;
    uint32_t id = hc.slot_id & 0xF;
    JXL_RETURN_IF_ERROR(visitor->Bool(false, &is_ac));
    JXL_RETURN_IF_ERROR(visitor->Bits(2, 0, &id));
    hc.slot_id = (static_cast<uint32_t>(is_ac) << 4) | id;
    JXL_RETURN_IF_ERROR(visitor->Bool(true, &hc.is_last));
    size_t num_symbols = 0;
    for (size_t i = 0; i <= 16; i++) {
      JXL_RETURN_IF_ERROR(visitor->U32(Val(0), Val(1), BitsOffset(3, 2),
                                       Bits(8), 0, &hc.counts[i]));
      num_symbols += hc.counts[i];
    }
    if (num_symbols > hc.values.size()) {
      return JXL_FAILURE("Huffman code too large (%zu)", num_symbols);
    }
    for (size_t i = 0; i < num_symbols; i++) {
      // Goes up to 256, included. Might have the same symbol appear twice...
      JXL_RETURN_IF_ERROR(visitor->U32(Bits(2), BitsOffset(2, 4),
                                       BitsOffset(4, 8), BitsOffset(8, 1), 0,
                                       &hc.values[i]));
    }
  }

  for (auto& scan : scan_info) {
    JXL_RETURN_IF_ERROR(
        visitor->U32(Val(1), Val(2), Val(3), Val(4), 1, &scan.num_components));
    if (scan.num_components >= 4) {
      return JXL_FAILURE("Invalid number of components in SOS marker");
    }
    JXL_RETURN_IF_ERROR(visitor->Bits(6, 0, &scan.Ss));
    JXL_RETURN_IF_ERROR(visitor->Bits(6, 63, &scan.Se));
    JXL_RETURN_IF_ERROR(visitor->Bits(4, 0, &scan.Al));
    JXL_RETURN_IF_ERROR(visitor->Bits(4, 0, &scan.Ah));
    for (size_t i = 0; i < scan.num_components; i++) {
      JXL_RETURN_IF_ERROR(visitor->Bits(2, 0, &scan.components[i].comp_idx));
      if (scan.components[i].comp_idx >= components.size()) {
        return JXL_FAILURE("Invalid component idx in SOS marker");
      }
      JXL_RETURN_IF_ERROR(visitor->Bits(2, 0, &scan.components[i].ac_tbl_idx));
      JXL_RETURN_IF_ERROR(visitor->Bits(2, 0, &scan.components[i].dc_tbl_idx));
    }
    // TODO(veluca): actually set and use this value.
    JXL_RETURN_IF_ERROR(visitor->U32(Val(0), Val(1), Val(2), BitsOffset(3, 3),
                                     kMaxNumPasses - 1,
                                     &scan.last_needed_pass));
  }

  // From here on, this is data that is not strictly necessary to get a valid
  // JPEG, but necessary for bit-exact JPEG reconstruction.
  if (info.has_dri) {
    JXL_RETURN_IF_ERROR(visitor->Bits(16, 0, &restart_interval));
  }
  for (auto& scan : scan_info) {
    uint32_t num_reset_points = scan.reset_points.size();
    JXL_RETURN_IF_ERROR(visitor->U32(Val(0), BitsOffset(2, 1), BitsOffset(4, 4),
                                     BitsOffset(16, 20), 0, &num_reset_points));
    if (visitor->IsReading()) {
      scan.reset_points.resize(num_reset_points);
    }
    int last_block_idx = -1;
    for (auto& block_idx : scan.reset_points) {
      block_idx -= last_block_idx + 1;
      JXL_RETURN_IF_ERROR(visitor->U32(Val(0), BitsOffset(3, 1),
                                       BitsOffset(5, 9), BitsOffset(28, 41), 0,
                                       &block_idx));
      block_idx += last_block_idx + 1;
      if (static_cast<int>(block_idx) < last_block_idx + 1) {
        return JXL_FAILURE("Invalid block ID: %u, last block was %d", block_idx,
                           last_block_idx);
      }
      if (block_idx > (1u << 30)) {
        // At most 8K x 8K x num_channels blocks are expected. That is,
        // typically, 1.5 * 2^27. 2^30 should be sufficient for any sane
        // image.
        return JXL_FAILURE("Invalid block ID: %u", block_idx);
      }
      last_block_idx = block_idx;
    }

    uint32_t num_extra_zero_runs = scan.extra_zero_runs.size();
    JXL_RETURN_IF_ERROR(visitor->U32(Val(0), BitsOffset(2, 1), BitsOffset(4, 4),
                                     BitsOffset(16, 20), 0,
                                     &num_extra_zero_runs));
    if (visitor->IsReading()) {
      scan.extra_zero_runs.resize(num_extra_zero_runs);
    }
    last_block_idx = -1;
    for (size_t i = 0; i < scan.extra_zero_runs.size(); ++i) {
      uint32_t& block_idx = scan.extra_zero_runs[i].block_idx;
      JXL_RETURN_IF_ERROR(visitor->U32(
          Val(1), BitsOffset(2, 2), BitsOffset(4, 5), BitsOffset(8, 20), 1,
          &scan.extra_zero_runs[i].num_extra_zero_runs));
      block_idx -= last_block_idx + 1;
      JXL_RETURN_IF_ERROR(visitor->U32(Val(0), BitsOffset(3, 1),
                                       BitsOffset(5, 9), BitsOffset(28, 41), 0,
                                       &block_idx));
      block_idx += last_block_idx + 1;
      if (static_cast<int>(block_idx) < last_block_idx + 1) {
        return JXL_FAILURE("Invalid block ID: %u, last block was %d", block_idx,
                           last_block_idx);
      }
      if (block_idx > (1u << 30)) {
        // At most 8K x 8K x num_channels blocks are expected. That is,
        // typically, 1.5 * 2^27. 2^30 should be sufficient for any sane
        // image.
        return JXL_FAILURE("Invalid block ID: %u", block_idx);
      }
      last_block_idx = block_idx;
    }
  }
  for (auto& inter_marker : inter_marker_data) {
    uint32_t len = inter_marker.size();
    JXL_RETURN_IF_ERROR(visitor->Bits(16, 0, &len));
    if (visitor->IsReading()) inter_marker.resize(len);
  }
  uint32_t tail_data_len = tail_data.size();
  JXL_RETURN_IF_ERROR(visitor->U32(Val(0), BitsOffset(8, 1),
                                   BitsOffset(16, 257), BitsOffset(22, 65793),
                                   0, &tail_data_len));
  if (visitor->IsReading()) {
    tail_data.resize(tail_data_len);
  }

  JXL_RETURN_IF_ERROR(visitor->Bool(false, &has_zero_padding_bit));
  if (has_zero_padding_bit) {
    uint32_t nbit = padding_bits.size();
    JXL_RETURN_IF_ERROR(visitor->Bits(24, 0, &nbit));
    if (visitor->IsReading()) {
      padding_bits.resize(nbit);
    }
    for (uint8_t& bit : padding_bits) {
      bool bbit = bit;
      JXL_RETURN_IF_ERROR(visitor->Bool(false, &bbit));
      bit = bbit;
    }
  }

  return true;
}

Status SetJPEGDataFromICC(const PaddedBytes& icc, jpeg::JPEGData* jpeg_data) {
  size_t icc_pos = 0;
  for (size_t i = 0; i < jpeg_data->app_data.size(); i++) {
    if (jpeg_data->app_marker_type[i] != jpeg::AppMarkerType::kICC) {
      continue;
    }
    size_t len = jpeg_data->app_data[i].size() - 17;
    if (icc_pos + len > icc.size()) {
      return JXL_FAILURE(
          "ICC length is less than APP markers: requested %zu more bytes, "
          "%zu available",
          len, icc.size() - icc_pos);
    }
    memcpy(&jpeg_data->app_data[i][17], icc.data() + icc_pos, len);
    icc_pos += len;
  }
  if (icc_pos != icc.size() && icc_pos != 0) {
    return JXL_FAILURE("ICC length is more than APP markers");
  }
  return true;
}

}  // namespace jpeg
}  // namespace jxl
