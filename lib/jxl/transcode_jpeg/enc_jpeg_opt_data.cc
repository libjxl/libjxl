// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Construction of the shared optimization data model for JPEG transcoding.
//
// This file implements the preprocessing passes that turn `jpeg::JPEGData`
// into `JPEGOptData`: fixed-point entropy tables, DC statistics, compacted AC
// symbol ids, per-block side data, and spatial DC orderings. Packed AC stream
// construction itself lives in `enc_jpeg_stream.cc`.

#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"

#include <cmath>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_Knuth_partition.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_stream.h"

namespace jxl {

namespace {

uint32_t EncodeJpegTranscodeACToken(int16_t coeff) {
  static const HybridUintConfig cfg(4, 2, 0);
  uint32_t token;
  uint32_t nbits;
  uint32_t bits;
  cfg.Encode(PackSigned(coeff), &token, &nbits, &bits);
  JXL_DASSERT(token < kACTokenCount);
  return token;
}

}  // namespace

// Precomputed values of `n*log2(n)*kFScale` for `n` up to `max_n`.
void JPEGOptData::InitFTab(size_t max_n) {
  const size_t old_size = ftab.size();
  if (max_n < old_size) return;
  ftab.resize(max_n + 1, 0);
  for (size_t i = std::max<size_t>(1, old_size); i <= max_n; ++i) {
    double n = static_cast<double>(i);
    ftab[i] = static_cast<int64_t>(std::llround(n * std::log2(n) * kFScale));
  }
}
// `ftab` may not cover counts as large as `num_blocks`; fall back to
// direct calculation for large values.
int64_t JPEGOptData::NZFTab(uint32_t n) const {
  if (n < ftab.size()) return ftab[n];
  const double nd = static_cast<double>(n);
  return static_cast<int64_t>(std::llround(nd * std::log2(nd) * kFScale));
}

// Map sparse `zdc_token` to dense index `[0, num_zdctok)`.
uint32_t JPEGOptData::CompactHBin(uint32_t zdc_token) const {
  JXL_DASSERT(zdc_token < compact_map_h.size());
  if (zdc_token >= compact_map_h.size()) return kInvalidCompactH;
  uint32_t dense = compact_map_h[zdc_token];
  JXL_DASSERT(dense != kInvalidCompactH);
  return dense;
}

// Threshold initialization for one axis via the same contiguous-partition DP
// used in later refinement, but with a cheaper 1D surrogate cost:
// `cost[l..n] = f(sum_{i=l}^n DC_cnt[i])`.
// The interval-cost matrix is written in the same diff form consumed by the
// shared Knuth solver, so this stage reuses the exact same partition backend
// as the AC-driven optimization path.
Thresholds InitThresh(const JPEGOptData& d, uint32_t axis,
                      uint32_t target_intervals) {
  if (target_intervals <= 1) return {};
  const Thresholds& dc_vals = d.DC_vals[axis];
  const uint32_t M = static_cast<uint32_t>(dc_vals.size());
  if (M == 0) return {};

  // Fast exit: fewer distinct values than requested intervals.
  if (M <= target_intervals) {
    // Thresholds denote the first value of the next interval, so if every
    // distinct DC value gets its own bucket we exclude the first one.
    return Thresholds(dc_vals.begin() + 1, dc_vals.end());
  }

  std::vector<uint32_t> prefix(M + 1, 0);
  for (uint32_t i = 0; i < M; ++i) {
    prefix[i + 1] = prefix[i] + d.DC_cnt[axis][dc_vals[i] + kDCTOff];
  }

  KnuthPartitionSolver solver(M);
  solver.ResetCosts(static_cast<size_t>(M) * M);
  for (uint32_t n = 0; n < M; ++n) {
    int64_t prev_base = 0;
    for (uint32_t l = 0; l <= n; ++l) {
      uint32_t total = prefix[n + 1] - prefix[l];
      int64_t base = d.NZFTab(total);
      if (n > 0 && l < n) {
        base -= d.NZFTab(prefix[n] - prefix[l]);
      }
      solver.costs[n * M + l] = base - prev_base;
      prev_base = base;
    }
  }

  std::vector<uint32_t> split_points = solver.Solve(target_intervals, M);
  Thresholds thresholds;
  for (uint32_t split_point : split_points) {
    thresholds.push_back(dc_vals[split_point]);
  }
  return thresholds;
}

// Enumerates all maximal factorizations `(a, b, c)` of DC threshold counts.
// A factorization is maximal when no single factor can be increased without
// violating `kMaxCells` (`a*b*c <= 64`), `kMaxIntervals` (each factor <= 16),
// or exceeding the number of distinct DC values on that axis. Lower
// factorizations are dominated: coarser partitioning can only increase entropy.
//
// Grayscale note: in a grayscale image `d.channels == 1`, so Cb and Cb are
// absent. The JPEG XL context map format still requires three sets of contexts
// (one per component). The two absent components each contribute one empty
// context, consuming one of the 16 available cluster slots. This leaves at most
// 15 clusters for the Y-channel DC partitioning, so the Y-axis factor `a` is
// capped at 15. Only the single factorization `(min(a,15), 1, 1)` is returned:
// the general enumeration always produces exactly `(a, 1, 1)` for grayscale
// since `cap1 == cap2 == 1`.
Factorizations MaximalFactorizations(const JPEGOptData& d) {
  uint32_t cap0 = std::max(1u, static_cast<uint32_t>(d.DC_vals[0].size()));
  uint32_t cap1 = std::max(1u, static_cast<uint32_t>(d.DC_vals[1].size()));
  uint32_t cap2 = std::max(1u, static_cast<uint32_t>(d.DC_vals[2].size()));
  Factorizations result;
  for (uint32_t a = 1; a <= cap0 && a <= kMaxCells && a <= kMaxIntervals; ++a) {
    for (uint32_t b = 1; b <= cap1 && a * b <= kMaxCells && b <= kMaxIntervals;
         ++b) {
      for (uint32_t c = 1;
           c <= cap2 && a * b * c <= kMaxCells && c <= kMaxIntervals; ++c) {
        bool can_inc_a =
            (a < cap0) && (a < kMaxIntervals) && ((a + 1) * b * c <= kMaxCells);
        bool can_inc_b =
            (b < cap1) && (b < kMaxIntervals) && (a * (b + 1) * c <= kMaxCells);
        bool can_inc_c =
            (c < cap2) && (c < kMaxIntervals) && (a * b * (c + 1) <= kMaxCells);
        if (!can_inc_a && !can_inc_b && !can_inc_c)
          result.push_back(Factorization{{a, b, c}});
      }
    }
  }
  if (d.channels == 1) {
    // Cap Y-axis intervals at 15: one cluster slot is consumed by the shared
    // Cb+Cr context required by the JPEG XL context map format.
    // Applies to both true grayscale and formally 3-channel images whose
    // chrominance was collapsed to `channels == 1` in `BuildFromJPEG`.
    result[0][0] = std::min(result[0][0], 15u);
  }
  return result;
}

// -------------------------------------------------------//
// JPEG processing: each pass sweeps all image data once. //
// -------------------------------------------------------//

using ACCounts = JPEGOptData::ACCounts;
// ----------------------------------------------------//
// Pass: Count DC + AC values (parallel by component). //
// ----------------------------------------------------//
StatusOr<std::unique_ptr<ACCounts>> JPEGOptData::CountDCAC(
    const jpeg::JPEGData& jpeg_data, ThreadPool* pool) {
  auto ac_cnt = jxl::make_unique<ACCounts>();
  memset(DC_cnt, 0, sizeof(DC_cnt));
  memset(ac_cnt->data(), 0, sizeof(*ac_cnt));
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, channels, ThreadPool::NoInit,
      [&](uint32_t c, size_t) -> Status {
        const auto& comp = jpeg_data.components[c];
        uint32_t wb = comp.width_in_blocks;
        uint32_t hb = comp.height_in_blocks;
        for (uint32_t by = 0; by < hb; ++by) {
          for (uint32_t bx = 0; bx < wb; ++bx) {
            const int16_t* q = comp.coeffs.data() + (by * wb + bx) * 64;
            ++DC_cnt[c][q[0] + kDCTOff];
            for (uint8_t p = 0; p < kNumPos; ++p) {
              ++(*ac_cnt)[c][p][q[jpeg::kJPEGNaturalOrder[p + 1]] + kDCTOff];
            }
          }
        }
        return true;
      },
      "CountDCAC"));

  // Build DC value lists and LUTs.
  for (auto& v : DC_vals) {
    v.clear();
    v.reserve(kDCTRange);
  }
  for (uint32_t c = 0; c < channels; ++c) {
    auto& v = DC_vals[c];
    for (int di = 0; di < kDCTRange; ++di) {
      if (DC_cnt[c][di]) {
        DC_idx_LUT[c][di] = static_cast<uint32_t>(v.size());
        v.push_back(static_cast<int16_t>(di - kDCTOff));
      }
    }
  }
  return ac_cnt;
}

// JPEG XL high effort mode scans AC coefficients in descending nonzero
// abundance (ties keep zigzag order).
using ScanOrder = std::array<std::array<uint8_t, kNumPos>, kNumCh>;

ScanOrder BuildACScanOrder(const ACCounts& ac_cnt) {
  ScanOrder scan_order;
  for (uint32_t c = 0; c < kNumCh; ++c) {
    std::array<uint8_t, kNumPos> pos_order;
    for (uint8_t i = 0; i < kNumPos; ++i) pos_order[i] = i;
    std::stable_sort(pos_order.begin(), pos_order.end(),
                     [&](uint8_t a, uint8_t b) {
                       return ac_cnt[c][a][kDCTOff] < ac_cnt[c][b][kDCTOff];
                     });
    for (uint8_t s = 0; s < kNumPos; ++s) {
      scan_order[c][s] = jpeg::kJPEGNaturalOrder[pos_order[s] + 1];
    }
  }
  return scan_order;
}

// -------------------------------------------------------------------------//
// Pass: Collect AC bins, DC indices and `nz` per block (parallel by comp). //
// -------------------------------------------------------------------------//
Status JPEGOptData::BuildBlockOptData(const jpeg::JPEGData& jpeg_data,
                                      ThreadPool* pool,
                                      const ACCounts& ac_cnt) {
  ScanOrder scan_order = BuildACScanOrder(ac_cnt);

  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, channels, ThreadPool::NoInit,
      [&](uint32_t c, size_t) -> Status {
        const auto& comp = jpeg_data.components[c];
        uint32_t wb = comp.width_in_blocks;
        uint32_t hb = comp.height_in_blocks;

        block_DC_idx[c].assign(num_blocks[c], 0);
        block_offsets[c].resize(num_blocks[c] + 1);
        block_nonzeros[c].assign(num_blocks[c], 0);
        block_nz_pred_bucket[c].resize(num_blocks[c]);
        block_bins[c].clear();
        std::vector<uint32_t> row_top(wb, 32u);
        std::vector<uint32_t> row_cur(wb, 32u);
        bool has_top = false;
        uint32_t bc = 0;
        for (uint32_t by = 0; by < hb; ++by) {
          for (uint32_t bx = 0; bx < wb; ++bx, ++bc) {
            block_offsets[c][bc] = static_cast<uint32_t>(block_bins[c].size());

            const int16_t* q = comp.coeffs.data() + (by * wb + bx) * 64;
            uint16_t DC_idx =
                static_cast<uint16_t>(DC_idx_LUT[c][q[0] + kDCTOff]);
            block_DC_idx[c][bc] = DC_idx;

            uint32_t nonzeros_left = 0;
            for (uint32_t s = 1; s <= kNumPos; ++s)
              if (q[s] != 0) ++nonzeros_left;
            block_nonzeros[c][bc] = static_cast<uint8_t>(nonzeros_left);

            uint32_t predicted_nz;
            if (bx == 0) {
              predicted_nz = has_top ? row_top[bx] : 32u;
            } else if (!has_top) {
              predicted_nz = row_cur[bx - 1];
            } else {
              predicted_nz = (row_top[bx] + row_cur[bx - 1] + 1u) / 2u;
            }
            block_nz_pred_bucket[c][bc] = static_cast<uint8_t>(
                (predicted_nz < 8) ? predicted_nz : (4 + predicted_nz / 2));
            row_cur[bx] = nonzeros_left;

            for (uint32_t s = 0; s < kNumPos; ++s) {
              if (nonzeros_left == 0) break;
              int16_t coeff = q[scan_order[c][s]];
              bool nz_prev = (s > 0 && q[scan_order[c][s - 1]] != 0) ||
                             (s == 0 && nonzeros_left > 4);
              uint32_t zdc = static_cast<uint32_t>(
                  ZeroDensityContext(nonzeros_left, s + 1, 1, 0, nz_prev));
              uint32_t token = EncodeJpegTranscodeACToken(coeff);
              block_bins[c].push_back(MakeJpegTranscodeACBin(c, zdc, token));
              nonzeros_left -= (coeff != 0);
            }
          }
          row_top.swap(row_cur);
          has_top = true;
        }
        block_offsets[c][num_blocks[c]] =
            static_cast<uint32_t>(block_bins[c].size());
        return true;
      },
      "CollectBlockOptData"));
  return true;
}

// ----------------------------------------------------------------//
// Pass: Prune DC values - remove DC values with no active bins.   //
//       Build per-axis sorted block positions (parallel by comp). //
// ----------------------------------------------------------------//
Status JPEGOptData::FinalizeSpatialIndexing(ThreadPool* pool) {
  memset(DC_cnt, 0, sizeof(DC_cnt));

  // Loop over image blocks
  if (channels == 1) {
    for (uint32_t b = 0; b < num_blocks[0]; ++b) {
      uint16_t dc0 = block_DC_idx[0][b];

      ++DC_cnt[0][DC_vals[0][dc0] + kDCTOff];
    }
  } else {
    for (uint32_t c = 0; c < kNumCh; ++c) {
      for (uint32_t y = 0; y < block_grid_h[c]; ++y) {
        for (uint32_t x = 0; x < block_grid_w[c]; ++x) {
          uint16_t dc0 =
              block_DC_idx[0][MapTopLeftBlockIndex(*this, c, y, x, 0)];
          uint16_t dc1 =
              block_DC_idx[1][MapTopLeftBlockIndex(*this, c, y, x, 1)];
          uint16_t dc2 =
              block_DC_idx[2][MapTopLeftBlockIndex(*this, c, y, x, 2)];

          ++DC_cnt[0][DC_vals[0][dc0] + kDCTOff];
          ++DC_cnt[1][DC_vals[1][dc1] + kDCTOff];
          ++DC_cnt[2][DC_vals[2][dc2] + kDCTOff];
        }
      }
    }
  }

  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, channels, ThreadPool::NoInit,
      [&](uint32_t c, size_t) -> Status {
        Thresholds pruned;
        uint32_t idx = 0;
        for (int di = 0; di < kDCTRange; ++di) {
          if (DC_cnt[c][di]) {
            pruned.push_back(static_cast<int16_t>(di - kDCTOff));
            DC_idx_LUT[c][di] = idx++;
          }
        }
        uint32_t M = static_cast<uint32_t>(pruned.size());
        DC_block_offsets[c].assign(M + 1, 0);
        for (uint16_t& dc_idx : block_DC_idx[c]) {
          dc_idx = DC_idx_LUT[c][DC_vals[c][dc_idx] + kDCTOff];
          ++DC_block_offsets[c][dc_idx + 1];
        }
        for (uint32_t v = 0; v < M; ++v)
          DC_block_offsets[c][v + 1] += DC_block_offsets[c][v];
        DC_vals[c].swap(pruned);
        return true;
      },
      "PruneDC"));

  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, channels, ThreadPool::NoInit,
      [&](uint32_t c, size_t) -> Status {
        DC_sorted_blocks[c].resize(num_blocks[c]);
        std::vector<uint32_t> write_pos = DC_block_offsets[c];
        for (uint32_t y = 0; y < block_grid_h[c]; ++y) {
          for (uint32_t x = 0; x < block_grid_w[c]; ++x) {
            uint32_t b = y * block_grid_w[c] + x;
            uint32_t v = block_DC_idx[c][b];
            DC_sorted_blocks[c][write_pos[v]++] = (y << 16) | x;
          }
        }
        return true;
      },
      "BuildSortedBlocks"));
  return true;
}

// Orchestrates the JPEG data extraction pipeline.
Status JPEGOptData::BuildFromJPEG(const jpeg::JPEGData& jpeg_data,
                                  ThreadPool* pool) {
  channels = static_cast<uint32_t>(jpeg_data.components.size());
  w_max = 0;
  h_max = 0;
  for (uint32_t c = 0; c < channels; ++c) {
    block_grid_w[c] = jpeg_data.components[c].width_in_blocks;
    block_grid_h[c] = jpeg_data.components[c].height_in_blocks;
    num_blocks[c] = block_grid_w[c] * block_grid_h[c];
    w_max = std::max(w_max, block_grid_w[c]);
    h_max = std::max(h_max, block_grid_h[c]);
  }
  for (uint32_t c = 0; c < channels; ++c) {
    const uint32_t sy = h_max / block_grid_h[c];
    const uint32_t sx = w_max / block_grid_w[c];
    if ((sx != 1 && sx != 2) || (sy != 1 && sy != 2)) {
      return JXL_FAILURE(
          "Unsupported JPEG subsampling factor for lossless transcoding");
    }
    vshift[c] = static_cast<uint8_t>(sy == 2);
    hshift[c] = static_cast<uint8_t>(sx == 2);
  }
  for (uint32_t c = channels; c < kNumCh; ++c) {
    block_grid_w[c] = 0;
    block_grid_h[c] = 0;
    num_blocks[c] = 0;
    vshift[c] = 0;
    hshift[c] = 0;
  }

  {
    JXL_ASSIGN_OR_RETURN(auto ac_cnt, CountDCAC(jpeg_data, pool));

    // Formally 3-channel JPEGs with constant Cb/Cr and no nonzero AC
    // in chrominance channels are treated as effectively monochrome:
    // reducing `channels` to 1 causes all downstream passes to take the
    // grayscale fast path, keeping Cb/Cr zero-event blocks out of Y's
    // histograms and capping Y at 15 cluster slots (one is reserved for
    // the shared Cb/Cr "all-zeros" context).
    auto ChromaIsEmpty = [&](uint32_t c) {
      if (DC_vals[c].size() > 1) return false;
      for (uint32_t p = 0; p < kNumPos; ++p) {
        for (int di = 0; di < kDCTRange; ++di) {
          if (di != kDCTOff && (*ac_cnt)[c][p][di]) return false;
        }
      }
      return true;
    };
    if (channels > 1 && ChromaIsEmpty(1) && ChromaIsEmpty(2)) channels = 1;

    JXL_RETURN_IF_ERROR(BuildBlockOptData(jpeg_data, pool, *ac_cnt));
  }

  JXL_RETURN_IF_ERROR(FinalizeSpatialIndexing(pool));

  JXL_ASSIGN_OR_RETURN(ACStreamData stream_data, BuildACStream(*this, pool));
  AC_stream = std::move(stream_data.stream);
  compact_map_h = std::move(stream_data.compact_map_h);
  dense_to_zdctok = std::move(stream_data.dense_to_zdctok);
  num_zdctok = stream_data.num_zdctok;
  InitFTab(stream_data.max_zdc_total + 1);

  return true;
}

}  // namespace jxl
