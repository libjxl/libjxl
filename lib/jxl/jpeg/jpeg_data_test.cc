// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/jpeg/jpeg_data.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/jpeg/dec_jpeg_data.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace jpeg {
namespace {

// Serialized JPEG reconstruction bundle declaring `n_app` APP markers of
// `app_len` payload bytes each plus `n_inter` inter-marker payloads of
// `inter_len` bytes, followed by `brotli_suffix`.
//
// The APP markers are typed kICC, so their payload is synthesized by the
// decoder and never read from the brotli stream; only the inter-marker
// payloads need actual decompressed bytes. This lets the test declare huge
// marker budgets without shipping huge fixtures.
std::vector<uint8_t> MakeBundle(size_t n_app, size_t app_len, size_t n_inter,
                                size_t inter_len,
                                const std::vector<uint8_t>& brotli_suffix) {
  JPEGData jpeg_data;
  jpeg_data.marker_order.push_back(0xc4);  // DHT
  for (size_t i = 0; i < n_app; i++) {
    jpeg_data.marker_order.push_back(0xe0);  // APPn
    jpeg_data.app_marker_type.push_back(AppMarkerType::kICC);
    jpeg_data.app_data.emplace_back(app_len);
  }
  for (size_t i = 0; i < n_inter; i++) {
    jpeg_data.marker_order.push_back(0xff);  // fake marker: inter-marker data
    jpeg_data.inter_marker_data.emplace_back(inter_len);
  }
  jpeg_data.marker_order.push_back(0xda);  // SOS
  jpeg_data.marker_order.push_back(0xd9);  // EOI

  // One quant table, used by the single gray component.
  JPEGQuantTable quant;
  quant.precision = 0;
  quant.index = 0;
  quant.is_last = true;
  jpeg_data.quant.push_back(quant);
  JPEGComponent component;
  component.id = 1;
  component.quant_idx = 0;
  jpeg_data.components.push_back(component);

  // One DHT marker holding an empty DC table 0 and an empty AC table 0, so
  // that the scan below finds its tables defined.
  JPEGHuffmanCode dc;
  dc.slot_id = 0x00;
  dc.is_last = false;
  jpeg_data.huffman_code.push_back(dc);
  JPEGHuffmanCode ac;
  ac.slot_id = 0x10;
  ac.is_last = true;
  jpeg_data.huffman_code.push_back(ac);

  JPEGScanInfo scan;
  scan.Ss = 0;
  scan.Se = 63;
  scan.Ah = 0;
  scan.Al = 0;
  scan.num_components = 1;
  scan.components[0].comp_idx = 0;
  scan.components[0].dc_tbl_idx = 0;
  scan.components[0].ac_tbl_idx = 0;
  scan.last_needed_pass = 0;
  jpeg_data.scan_info.push_back(scan);

  BitWriter writer{jxl::test::MemoryManager()};
  AuxOut aux_out;
  if (!Bundle::Write(jpeg_data, &writer, LayerType::Header, &aux_out)) {
    return {};
  }
  writer.ZeroPadToByte();
  std::vector<uint8_t> bytes(writer.GetSpan().begin(), writer.GetSpan().end());
  bytes.insert(bytes.end(), brotli_suffix.begin(), brotli_suffix.end());
  return bytes;
}

// An empty but finished brotli stream: all payloads are empty.
const std::vector<uint8_t> kEmptyBrotli = {0x06};
// Brotli stream of exactly 3 decompressed zero bytes
// (BrotliEncoderCompress q=5, mode generic).
const std::vector<uint8_t> kThreeZeroBytes = {0x0b, 0x01, 0x80, 0x00,
                                              0x00, 0x00, 0x03};

// Declared marker payloads must stay within the decoder budget; without the
// budget a ~3 KiB bundle declares 1025 * 64 KiB of APP payload and the
// decoder allocates all of it before reading a single payload byte.
TEST(JPEGDataTest, MarkerBudgetRejectsOversizedAppMarkers) {
  std::vector<uint8_t> bytes = MakeBundle(/*n_app=*/1025, /*app_len=*/65536,
                                          /*n_inter=*/0, /*inter_len=*/0,
                                          kEmptyBrotli);
  ASSERT_FALSE(bytes.empty());
  JPEGData jpeg_data;
  EXPECT_FALSE(DecodeJPEGData(Span<const uint8_t>(bytes), &jpeg_data));
}

// Inter-marker payloads share the same budget (they are allocated from the
// same attacker-controlled 16-bit lengths).
TEST(JPEGDataTest, MarkerBudgetRejectsOversizedInterMarkerData) {
  // 1024 * 65536 == 64 MiB of APP payload (exactly at the budget) plus a
  // 3-byte inter-marker payload pushes the total over the budget.
  std::vector<uint8_t> bytes = MakeBundle(/*n_app=*/1024, /*app_len=*/65536,
                                          /*n_inter=*/1, /*inter_len=*/3,
                                          kThreeZeroBytes);
  ASSERT_FALSE(bytes.empty());
  JPEGData jpeg_data;
  EXPECT_FALSE(DecodeJPEGData(Span<const uint8_t>(bytes), &jpeg_data));
}

// Control: a realistic marker budget decodes fine.
TEST(JPEGDataTest, MarkerBudgetAllowsRealisticBundle) {
  std::vector<uint8_t> bytes = MakeBundle(/*n_app=*/4, /*app_len=*/64,
                                          /*n_inter=*/0, /*inter_len=*/0,
                                          kEmptyBrotli);
  ASSERT_FALSE(bytes.empty());
  JPEGData jpeg_data;
  EXPECT_TRUE(DecodeJPEGData(Span<const uint8_t>(bytes), &jpeg_data));
  EXPECT_EQ(jpeg_data.app_data.size(), 4u);
  EXPECT_EQ(jpeg_data.marker_order.size(), 7u);
}

}  // namespace
}  // namespace jpeg
}  // namespace jxl
