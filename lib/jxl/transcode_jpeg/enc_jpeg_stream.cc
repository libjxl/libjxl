// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of the packed AC-stream format for JPEG lossless
// recompression. See `enc_jpeg_stream.h` for the public interface.

#include "lib/jxl/transcode_jpeg/enc_jpeg_stream.h"

#include <algorithm>
#include <array>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

namespace {

// Prefix-summed per-bin sizes plus the list of bins that actually occur.
struct ActiveBinLayout {
  std::vector<uint32_t> bin_end_lo;
  std::vector<uint32_t> bin_end_hi;
  std::vector<uint32_t> active_bins;
  size_t raw_size = 0;
};

// Flattened `(dc0, dc1, dc2)` keys split into the `dc0` low/high halves.
struct SortedBinHalves {
  std::vector<uint32_t> flat_lo;
  std::vector<uint32_t> flat_hi;
};

// One emitted raw bin together with the histogram/context grouping keys used
// for reset-frame flags and stream order.
struct EmitBin {
  uint32_t raw_bin;
  uint32_t hist_key;
};

std::vector<EmitBin> BuildEmitBins(const ActiveBinLayout& layout,
                                   const JPEGOptData& d) {
  std::vector<EmitBin> emit_bins;
  emit_bins.reserve(layout.active_bins.size());
  for (uint32_t raw_bin : layout.active_bins) {
    emit_bins.push_back({raw_bin, d.ACHistogramKey(raw_bin)});
  }
  if (d.ac_hist_model == JPEGTranscodeACModel::kToken420) {
    std::sort(emit_bins.begin(), emit_bins.end(),
              [](const EmitBin& a, const EmitBin& b) {
                if (a.hist_key != b.hist_key) return a.hist_key < b.hist_key;
                return a.raw_bin < b.raw_bin;
              });
  }
  return emit_bins;
}

// Counts the number of entries in each sparse AC bin and builds prefix sums
// for the later scatter pass.
ActiveBinLayout CountActiveBins(const JPEGOptData& d) {
  const uint32_t BIN_N = d.channels * kMaxACSymbolCount;
  ActiveBinLayout layout;
  // Split on `dc0` bit 10 (the MSB of the 11-bit index):
  // `lo = dc0 < 1024`, `hi = dc0 >= 1024`.
  // Each half packs `(dc0_low10 << 22 | dc1 << 11 | dc2)` into
  // a single `uint32_t` (10+11+11 = 32 bits), halving `flat[]` memory
  // vs the full `uint64_t` encoding.
  layout.bin_end_lo.assign(BIN_N + 1, 0);
  layout.bin_end_hi.assign(BIN_N + 1, 0);

  for (uint32_t c = 0; c < d.channels; ++c) {
    uint32_t w = d.block_grid_w[c];
    uint32_t h = d.block_grid_h[c];
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t b = y * w + x;
        uint32_t b0 = MapTopLeftBlockIndex(d, c, y, x, 0);
        uint32_t dc0 = d.block_DC_idx[0][b0];
        for (uint32_t pi = d.block_offsets[c][b];
             pi < d.block_offsets[c][b + 1]; ++pi) {
          uint32_t bin = d.block_bins[c][pi];
          if (dc0 & 0x400u) {
            ++layout.bin_end_hi[bin + 1];
          } else {
            ++layout.bin_end_lo[bin + 1];
          }
        }
      }
    }
  }

  // Prefix sums and active bins.
  layout.active_bins.reserve(BIN_N);
  for (uint32_t b = 0; b < BIN_N; ++b) {
    uint32_t c_lo = layout.bin_end_lo[b + 1];
    uint32_t c_hi = layout.bin_end_hi[b + 1];
    layout.bin_end_lo[b + 1] = layout.bin_end_lo[b] + c_lo;
    layout.bin_end_hi[b + 1] = layout.bin_end_hi[b] + c_hi;
    if (c_lo != 0 || c_hi != 0) layout.active_bins.push_back(b);
  }
  layout.active_bins.shrink_to_fit();
  layout.raw_size = layout.bin_end_lo[BIN_N] + layout.bin_end_hi[BIN_N];
  return layout;
}

// Scatters per-block DC keys into dense per-bin arrays and sorts each active
// bin half so emission can run-length-encode identical triples.
StatusOr<SortedBinHalves> ScatterAndSortBins(const JPEGOptData& d,
                                             const ActiveBinLayout& layout,
                                             ThreadPool* pool) {
  SortedBinHalves halves;
  halves.flat_lo.resize(layout.bin_end_lo.back());
  halves.flat_hi.resize(layout.bin_end_hi.back());
  std::vector<uint32_t> write_lo = layout.bin_end_lo;
  std::vector<uint32_t> write_hi = layout.bin_end_hi;

  // Scatter pass: `dc0 & 0x3FF` drops bit 10 (already encoded by lo/hi half);
  // `dc0_base` (0 or 0x400) re-adds it during decode in `emit_half` later.
  if (d.channels == 1) {
    for (uint32_t b = 0; b < d.num_blocks[0]; ++b) {
      uint32_t dc0 = d.block_DC_idx[0][b];
      for (uint32_t pi = d.block_offsets[0][b]; pi < d.block_offsets[0][b + 1];
           ++pi) {
        uint32_t bin = d.block_bins[0][pi];
        uint32_t dc_key = (dc0 & 0x3FFu) << 22;
        if (dc0 & 0x400u) {
          halves.flat_hi[write_hi[bin]++] = dc_key;
        } else {
          halves.flat_lo[write_lo[bin]++] = dc_key;
        }
      }
    }
  } else {
    for (uint32_t c = 0; c < kNumCh; ++c) {
      uint32_t w = d.block_grid_w[c];
      uint32_t h = d.block_grid_h[c];
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
          uint32_t b = y * w + x;
          uint32_t b0 = MapTopLeftBlockIndex(d, c, y, x, 0);
          uint32_t b1 = MapTopLeftBlockIndex(d, c, y, x, 1);
          uint32_t b2 = MapTopLeftBlockIndex(d, c, y, x, 2);

          uint32_t dc0 = d.block_DC_idx[0][b0];
          uint32_t dc1 = d.block_DC_idx[1][b1];
          uint32_t dc2 = d.block_DC_idx[2][b2];
          for (uint32_t pi = d.block_offsets[c][b];
               pi < d.block_offsets[c][b + 1]; ++pi) {
            uint32_t bin = d.block_bins[c][pi];
            uint32_t dc_key = ((dc0 & 0x3FFu) << 22) | (dc1 << 11) | dc2;
            if (dc0 & 0x400u) {
              halves.flat_hi[write_hi[bin]++] = dc_key;
            } else {
              halves.flat_lo[write_lo[bin]++] = dc_key;
            }
          }
        }
      }
    }
  }

  // Sort each bin's lo/hi halves in parallel.
  const uint32_t kChunk = 1024;
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0,
      static_cast<uint32_t>((layout.active_bins.size() + kChunk - 1) / kChunk),
      ThreadPool::NoInit,
      [&](uint32_t chunk, size_t) -> Status {
        uint32_t idx_start = chunk * kChunk;
        uint32_t idx_end =
            std::min(idx_start + kChunk,
                     static_cast<uint32_t>(layout.active_bins.size()));
        for (uint32_t idx = idx_start; idx < idx_end; ++idx) {
          uint32_t b = layout.active_bins[idx];
          uint32_t s_lo = layout.bin_end_lo[b];
          uint32_t e_lo = layout.bin_end_lo[b + 1];
          uint32_t s_hi = layout.bin_end_hi[b];
          uint32_t e_hi = layout.bin_end_hi[b + 1];
          if (s_lo < e_lo)
            std::sort(halves.flat_lo.data() + s_lo,
                      halves.flat_lo.data() + e_lo);
          if (s_hi < e_hi)
            std::sort(halves.flat_hi.data() + s_hi,
                      halves.flat_hi.data() + e_hi);
        }
        return true;
      },
      "SortBins"));
  return halves;
}

// Emits the final packed stream and the associated sparse-to-dense AC-symbol
// tables from the sorted per-bin DC-key arrays.
ACStreamData EmitACStream(const ActiveBinLayout& layout,
                          const SortedBinHalves& halves, const JPEGOptData& d) {
  // `AC_stream` layout:
  //   Regular frame (bit 31 = 0):
  //     30..27  `Δdc0`   (4b, 0..15)
  //     26..16  `dc1`    (11b, absolute)
  //     15..5   `dc2`    (11b, absolute)
  //     4..0    `run-1`  (5b; 0..30 = run 1..31; 31 = long-run escape)
  //   Long-run frame (follows regular frame with `run-1 = 31`):
  //     31..0   `run`    (32b, 2^26 >= actual run >= 32)
  //   Reset frame (bit 31 = 1),
  //     emitted for the first raw-bin entry, histogram-bin change, or
  //     `Δdc0 > 15`:
  //     30      `ctx_changed` (1b)
  //     29      `bin_changed` (1b)
  //     28..7   `bin`      (22b,
  //                         `c * kMaxACSymbolCount + zdc * kDCTRange + ai`)
  //     6..0    `dc0 >> 4` (7b, coarse; fine bits recovered from next `Δdc0`)
  //
  // `dc0/dc1/dc2` are 11-bit indices into the active DC-coefficient tables.
  // The layout gives max 5% stream overhead in the worst case tested,
  // dominated by the reset frames for `Δdc0 > 15`.
  ACStreamData out;
  out.stream.reserve(layout.raw_size + layout.raw_size / 16);
  uint32_t ctx_len = 0;
  uint32_t prev_bin = UINT32_MAX;
  uint32_t prev_hist_key = UINT32_MAX;
  uint32_t prev_ctx_key = UINT32_MAX;
  std::array<uint32_t, kZeroDensityContextCount> zdc_len = {};

  auto& hist = out.ac_histogram;
  const uint32_t hist_symbol_count =
      d.ac_hist_model == JPEGTranscodeACModel::kRawAI
          ? kMaxACSymbolCount
          : kZeroDensityContextCount * kACTokenCount;
  hist.compact_map_h.assign(hist_symbol_count, kInvalidCompactH);
  hist.dense_to_zdcvalue.reserve(layout.active_bins.size());

  for (uint32_t bin : layout.active_bins) {
    hist.AddUniqueSymbol(d.ACHistogramSymbol(bin));
  }

  std::vector<EmitBin> emit_bins = BuildEmitBins(layout, d);
  for (const EmitBin& emit_bin : emit_bins) {
    uint32_t bin = emit_bin.raw_bin;
    uint32_t ctx_key = d.ACBinCZDC(bin);
    uint32_t start_lo = layout.bin_end_lo[bin];
    uint32_t start_hi = layout.bin_end_hi[bin];
    uint32_t end_lo = layout.bin_end_lo[bin + 1];
    uint32_t end_hi = layout.bin_end_hi[bin + 1];

    // Stats: track per-ctx entry counts.
    bool new_ctx = prev_ctx_key == UINT32_MAX || ctx_key != prev_ctx_key;
    if (new_ctx) {
      if (prev_bin != UINT32_MAX) {
        zdc_len[d.ACBinZDC(prev_bin)] += ctx_len;
      }
      ctx_len = 0;
    }
    ctx_len += (end_lo - start_lo) + (end_hi - start_hi);

    // `bin_change` / `ctx_change` mark reset-frame flags (bits 29/30).
    // `bin_change` tracks the selected histogram model, while the payload
    // keeps carrying the raw `bin`.
    bool bin_change =
        prev_hist_key != UINT32_MAX && emit_bin.hist_key != prev_hist_key;
    bool ctx_change = prev_ctx_key != UINT32_MAX && ctx_key != prev_ctx_key;

    // Emit one sorted half (lo or hi) into stream.
    // `dc0_base` is added back to recover the full 11-bit `dc0`.
    uint32_t cur_dc0 = 0;
    bool first_in_bin = true;
    auto emit_half = [&](const std::vector<uint32_t>& data, uint32_t start,
                         uint32_t end, uint32_t dc0_base) {
      uint32_t i = start;
      while (i < end) {
        uint32_t j = i + 1;
        while (j < end && data[j] == data[i]) ++j;
        uint32_t run = j - i;
        uint32_t dc0 = (data[i] >> 22) | dc0_base;
        uint32_t dc1 = (data[i] >> 11) & 0x7FFu;
        uint32_t dc2 = data[i] & 0x7FFu;

        // Reset frame: emitted on the first entry of each raw-bin run, on
        // histogram/context change, or when `Δdc0 > 15`. `cur_dc0` is snapped
        // to the coarsest 16-aligned value ≤ dc0 so the 4-bit `Δdc0` in the
        // next normal frame is always in [0,15].
        if (first_in_bin || (dc0 - cur_dc0 > 15u)) {
          out.stream.push_back(
              (1u << 31) |
              (static_cast<uint32_t>(ctx_change && first_in_bin) << 30) |
              (static_cast<uint32_t>(bin_change && first_in_bin) << 29) |
              (bin << 7) | (dc0 >> 4));
          cur_dc0 = (dc0 >> 4) << 4;
        }

        // Normal frame: `Δdc0` in bits 30..27, `dc1` in bits 26..16,
        // `dc2` in bits 15..5, `run-1` in bits 4..0 (max run = 32 per frame).
        // Long runs are split into consecutive normal frames with the same
        // dc0/dc1/dc2; the reset-frame guard above fires only on the first.
        uint32_t delta_dc0 = dc0 - cur_dc0;
        uint32_t header = (delta_dc0 << 27) | (dc1 << 16) | (dc2 << 5);
        // Continuation header: Δdc0=0, same dc1/dc2 (both are absolute).
        const uint32_t cont_header = (dc1 << 16) | (dc2 << 5);
        while (run > 32) {
          out.stream.push_back(header | 31u);  // run = 32, run-1 = 31
          run -= 32;
          header = cont_header;
        }
        out.stream.push_back(header | (run - 1));
        cur_dc0 = dc0;
        first_in_bin = false;
        i = j;
      }
    };

    emit_half(halves.flat_lo, start_lo, end_lo, 0u);
    emit_half(halves.flat_hi, start_hi, end_hi, 0x400u);
    prev_bin = bin;
    prev_hist_key = emit_bin.hist_key;
    prev_ctx_key = ctx_key;
  }

  if (prev_bin != UINT32_MAX) zdc_len[d.ACBinZDC(prev_bin)] += ctx_len;
  out.stream.shrink_to_fit();
  JXL_DEBUG_V(
      2, "JPEG transcode compact_map_h (%s) uses %u / %i keys\n",
      d.ac_hist_model == JPEGTranscodeACModel::kRawAI ? "raw-ai" : "token420",
      hist.num_zdcvalue, static_cast<int>(hist.compact_map_h.size()));
  // Clustering may merge contexts from different channels, so
  // `hist_N[merged][zdc]` can accumulate counts from all channels for the
  // same `zdc`. Use the actual per-`zdc` cross-channel total as the bound
  // for `ftab` size.
  out.max_zdc_total = *std::max_element(zdc_len.begin(), zdc_len.end());

  return out;
}

}  // namespace

// Builds the packed AC stream from precomputed per-block AC data in `d`.
// Runs the three logical phases in order: count bins, scatter+sort DC keys,
// then emit the final packed stream and derived AC-symbol tables.
StatusOr<ACStreamData> BuildACStream(const JPEGOptData& d, ThreadPool* pool) {
  ActiveBinLayout layout = CountActiveBins(d);
  JXL_ASSIGN_OR_RETURN(SortedBinHalves halves,
                       ScatterAndSortBins(d, layout, pool));
  return EmitACStream(layout, halves, d);
}

}  // namespace jxl
