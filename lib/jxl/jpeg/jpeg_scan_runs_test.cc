// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/jpeg/jpeg_data.h"

#include <cstdint>
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

// Serialized JPEG reconstruction bundle with `n_scans` scans, each declaring
// `n_reset` reset points and `n_zero` extra zero runs with valid (strictly
// increasing) block indices, followed by `brotli_suffix`. The entry counts are
// what drive the allocation in JPEGData::VisitFields; the entries themselves
// are the minimum-size encodings (delta 0).
std::vector<uint8_t> MakeBundle(size_t n_scans, size_t n_reset, size_t n_zero,
                                const std::vector<uint8_t>& brotli_suffix) {
  JPEGData jpeg_data;
  jpeg_data.marker_order.push_back(0xc4);  // DHT
  for (size_t s = 0; s < n_scans; s++) {
    jpeg_data.marker_order.push_back(0xda);  // SOS
  }
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
  // that the scans below find their tables defined.
  JPEGHuffmanCode dc;
  dc.slot_id = 0x00;
  dc.is_last = false;
  jpeg_data.huffman_code.push_back(dc);
  JPEGHuffmanCode ac;
  ac.slot_id = 0x10;
  ac.is_last = true;
  jpeg_data.huffman_code.push_back(ac);

  for (size_t s = 0; s < n_scans; s++) {
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
    // Strictly increasing block indices; delta 0 is the cheapest encoding.
    scan.reset_points.resize(n_reset);
    for (size_t i = 0; i < n_reset; i++) {
      scan.reset_points[i] = static_cast<uint32_t>(i);
    }
    scan.extra_zero_runs.resize(n_zero);
    for (size_t i = 0; i < n_zero; i++) {
      scan.extra_zero_runs[i].block_idx = static_cast<uint32_t>(i);
      scan.extra_zero_runs[i].num_extra_zero_runs = 1;
    }
    jpeg_data.scan_info.push_back(scan);
  }

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

// Declared per-scan entry counts must stay within the decoder budget; without
// the budget 86 scans x 65555 reset points + 65555 extra zero runs declare
// ~67.6 MB of entry storage and the decoder allocates all of it before
// reading a single entry. The whole bundle is only ~4.3 MB.
TEST(JPEGScanRunsTest, ScanRunBudgetRejectsOversizedEntryCounts) {
  std::vector<uint8_t> bytes =
      MakeBundle(/*n_scans=*/86, /*n_reset=*/65555, /*n_zero=*/65555,
                 kEmptyBrotli);
  ASSERT_FALSE(bytes.empty());
  JPEGData jpeg_data;
  EXPECT_FALSE(DecodeJPEGData(Span<const uint8_t>(bytes), &jpeg_data));
}

// The extra-zero-run family alone is budgeted the same way (8 bytes per
// entry): 128 scans x 65555 entries = ~67.1 MB, just over the budget.
TEST(JPEGScanRunsTest, ScanRunBudgetRejectsOversizedExtraZeroRuns) {
  std::vector<uint8_t> bytes =
      MakeBundle(/*n_scans=*/128, /*n_reset=*/0, /*n_zero=*/65555,
                 kEmptyBrotli);
  ASSERT_FALSE(bytes.empty());
  JPEGData jpeg_data;
  EXPECT_FALSE(DecodeJPEGData(Span<const uint8_t>(bytes), &jpeg_data));
}

// Control: bundles whose declared entry storage stays within the budget decode
// fine, both before and after the change.
TEST(JPEGScanRunsTest, ScanRunBudgetAllowsRealisticBundle) {
  std::vector<uint8_t> bytes =
      MakeBundle(/*n_scans=*/3, /*n_reset=*/8, /*n_zero=*/4, kEmptyBrotli);
  ASSERT_FALSE(bytes.empty());
  JPEGData jpeg_data;
  EXPECT_TRUE(DecodeJPEGData(Span<const uint8_t>(bytes), &jpeg_data));
  EXPECT_EQ(jpeg_data.scan_info.size(), 3u);
  EXPECT_EQ(jpeg_data.scan_info[0].reset_points.size(), 8u);
  EXPECT_EQ(jpeg_data.scan_info[0].extra_zero_runs.size(), 4u);
}

}  // namespace
}  // namespace jpeg
}  // namespace jxl
