// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_jpeg_frame.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace {

constexpr int64_t kFScale = 1LL << 25;
constexpr int kDCTOff = 1024;
constexpr int kDCTRange = 2048;
// Each threshold count is written with 4 bits (max 15), so each dimension
// of intervals is capped at 16.
constexpr uint32_t kMaxIntervals = 16;
constexpr int kMaxCells = 64;
constexpr int kMaxClusters = 16;
constexpr int kNumPos = 63;
constexpr int kNumCh = 3;
constexpr uint32_t kMTarget = 256;
constexpr uint32_t kBinCount = kMaxCells / 2 * kDCTRange >> 6;
constexpr uint32_t kGroupCount = kMaxCells / 2 * kDCTRange >> 12;
constexpr uint32_t kJPEGNZBucketCount = 36;
constexpr uint32_t kJPEGNZCountLimit = 64;
constexpr uint32_t kJPEGNZHistogramSize =
    kJPEGNZBucketCount * kJPEGNZCountLimit;

using Thresholds = std::vector<int16_t>;
using ACEntry = uint32_t;
using ContextMap = std::vector<uint8_t>;
using Factorizations = std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>;

JXL_INLINE int CountrZero64(uint64_t x) {
#if JXL_COMPILER_MSVC
  unsigned long idx;
  return _BitScanForward64(&idx, x) ? static_cast<int>(idx) : 64;
#else
  return x ? __builtin_ctzll(x) : 64;
#endif
}

struct ThresholdSet {
  std::array<Thresholds, kNumCh> T;
  Thresholds& TY() { return T[0]; }
  const Thresholds& TY() const { return T[0]; }
  Thresholds& TCb() { return T[1]; }
  const Thresholds& TCb() const { return T[1]; }
  Thresholds& TCr() { return T[2]; }
  const Thresholds& TCr() const { return T[2]; }
};

constexpr uint32_t kInvalidCompactH = std::numeric_limits<uint32_t>::max();

struct CompactHistogram {
  CompactHistogram() = default;
  explicit CompactHistogram(size_t size)
      : counts(size, 0), pos_in_used(size, kInvalidCompactH) {}

  CompactHistogram(const CompactHistogram& other)
      : counts(other.counts.size(), 0),
        used_ids(other.used_ids),
        pos_in_used(other.pos_in_used.size(), kInvalidCompactH) {
    for (uint32_t i = 0; i < used_ids.size(); ++i) {
      uint32_t id = used_ids[i];
      counts[id] = other.counts[id];
      pos_in_used[id] = i;
    }
  }

  CompactHistogram& operator=(const CompactHistogram& other) {
    if (this == &other) return *this;
    if (counts.size() != other.counts.size()) {
      counts.assign(other.counts.size(), 0);
      pos_in_used.assign(other.pos_in_used.size(), kInvalidCompactH);
      used_ids.clear();
    } else {
      Clear();
    }
    used_ids = other.used_ids;
    for (uint32_t i = 0; i < used_ids.size(); ++i) {
      uint32_t id = used_ids[i];
      counts[id] = other.counts[id];
      pos_in_used[id] = i;
    }
    return *this;
  }

  CompactHistogram(CompactHistogram&&) = default;
  CompactHistogram& operator=(CompactHistogram&&) = default;

  bool empty() const { return used_ids.empty(); }

  uint32_t Get(uint32_t id) const { return counts[id]; }

  void Add(uint32_t id, uint32_t value = 1) {
    uint32_t& freq = counts[id];
    if (freq == 0) {
      pos_in_used[id] = static_cast<uint32_t>(used_ids.size());
      used_ids.push_back(id);
    }
    freq += value;
  }

  void Subtract(uint32_t id, uint32_t value = 1) {
    uint32_t& freq = counts[id];
    JXL_DASSERT(freq >= value);
    if (freq < value) return;
    freq -= value;
    if (freq != 0) return;
    uint32_t pos = pos_in_used[id];
    JXL_DASSERT(pos != kInvalidCompactH);
    if (pos == kInvalidCompactH) return;
    uint32_t last = used_ids.back();
    used_ids[pos] = last;
    pos_in_used[last] = pos;
    used_ids.pop_back();
    pos_in_used[id] = kInvalidCompactH;
  }

  void AddFrom(const CompactHistogram& other) {
    for (uint32_t id : other.used_ids) Add(id, other.counts[id]);
  }

  void Clear() {
    for (uint32_t id : used_ids) {
      counts[id] = 0;
      pos_in_used[id] = kInvalidCompactH;
    }
    used_ids.clear();
  }

  void swap(CompactHistogram& other) {
    counts.swap(other.counts);
    used_ids.swap(other.used_ids);
    pos_in_used.swap(other.pos_in_used);
  }

  std::vector<uint32_t> counts;
  std::vector<uint32_t> used_ids;
  std::vector<uint32_t> pos_in_used;
};

struct JPEGCtxEffortParams {
  uint32_t keep_top_k;
  uint32_t rank_m_target;
  uint32_t rank_iters;
  uint32_t final_m_target;
  uint32_t final_iters;
  bool overhead_aware_tail;
  uint32_t refine_iters;
  ptrdiff_t refine_radius;

  static JPEGCtxEffortParams FromSpeedTier(SpeedTier speed_tier) {
    switch (speed_tier) {
      case SpeedTier::kSquirrel:
        return {/*keep_top_k=*/4,
                /*rank_m_target=*/0,
                /*rank_iters=*/0,
                /*final_m_target=*/64,
                /*final_iters=*/2,
                /*overhead_aware_tail=*/false,
                /*refine_iters=*/0,
                /*refine_radius=*/0};
      case SpeedTier::kKitten:
        return {/*keep_top_k=*/6,
                /*rank_m_target=*/64,
                /*rank_iters=*/1,
                /*final_m_target=*/128,
                /*final_iters=*/4,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/1,
                /*refine_radius=*/4};
      case SpeedTier::kTortoise:
        return {/*keep_top_k=*/12,
                /*rank_m_target=*/128,
                /*rank_iters=*/2,
                /*final_m_target=*/256,
                /*final_iters=*/8,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/2,
                /*refine_radius=*/8};
      case SpeedTier::kTectonicPlate:
      case SpeedTier::kGlacier:
      default:
        return {/*keep_top_k=*/0,
                /*rank_m_target=*/0,
                /*rank_iters=*/0,
                /*final_m_target=*/kMTarget,
                /*final_iters=*/20,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/5,
                /*refine_radius=*/16};
    }
  }
};

struct JPEGOptData {
  using ACCounts =
      std::array<std::array<std::array<uint32_t, kDCTRange>, kNumPos>, kNumCh>;

  std::vector<int64_t> ftab;

  uint32_t channels;
  uint32_t num_blocks[kNumCh];
  uint32_t block_grid_w[kNumCh];
  uint32_t block_grid_h[kNumCh];
  uint32_t w_max;
  uint32_t h_max;
  uint32_t ss_y[kNumCh];
  uint32_t ss_x[kNumCh];
  Thresholds dc_vals[kNumCh];
  uint32_t DC_cnt[kNumCh][kDCTRange];
  uint16_t DC_idx_LUT[kNumCh][kDCTRange];
  uint32_t num_zdcai;
  std::vector<uint32_t> compact_map_h;
  std::vector<uint32_t> dense_to_zdcai;

  std::vector<ACEntry> AC_stream;
  // AC events of consequitive blocks per component
  std::array<std::vector<uint32_t>, kNumCh> block_bins;
  // Indices into `block_bins`, separating consequent blocks data,
  // size `y_comp * x_comp`
  std::array<std::vector<uint32_t>, kNumCh> block_offsets;
  // Block nonzero number and nonzero prediction context,
  // size `y_comp * x_comp`
  std::array<std::vector<uint8_t>, kNumCh> block_nonzeros;
  std::array<std::vector<uint8_t>, kNumCh> nz_pred_bucket;
  // std::vector<uint32_t> compact_map_h;
  // DC indices of blocks of components (DC of the block component),
  // size `y_comp * x_comp`
  std::array<std::vector<uint16_t>, kNumCh> block_dc_idx;
  // Indices into `block_dc_idx`, sorted by DC index,
  // size `y_comp * x_comp`
  std::array<std::vector<uint32_t>, kNumCh> dc_sorted_blocks;
  // Indices into `dc_sorted_blocks, separating different DC indices,
  // size `M_comp`
  std::array<std::vector<uint32_t>, kNumCh> dc_block_offsets;

  void InitFTab(size_t max_n) {
    const size_t old_size = ftab.size();
    if (max_n < old_size) return;
    ftab.resize(max_n + 1, 0);
    for (size_t i = std::max<size_t>(1, old_size); i <= max_n; ++i) {
      double n = static_cast<double>(i);
      ftab[i] = static_cast<int64_t>(std::llround(n * std::log2(n) * kFScale));
    }
  }
  // `ftab` may not cover counts as large as `N_blocks`; fall back to
  // floating-point for large values.
  int64_t nz_entropy(uint32_t n) const {
    if (n < ftab.size()) return ftab[n];
    const double nd = static_cast<double>(n);
    return static_cast<int64_t>(std::llround(nd * std::log2(nd) * kFScale));
  }

  uint32_t CompactHBin(uint32_t zdc_ai) const {
    JXL_DASSERT(zdc_ai < compact_map_h.size());
    if (zdc_ai >= compact_map_h.size()) return kInvalidCompactH;
    uint32_t dense = compact_map_h[zdc_ai];
    JXL_DASSERT(dense != kInvalidCompactH);
    return dense;
  }

  Thresholds InitThresh(int axis, uint32_t n_intervals) const {
    uint64_t total = 0;
    for (int i = 0; i < kDCTRange; ++i) total += DC_cnt[axis][i];
    if (total == 0) return {};
    Thresholds result;
    result.reserve(kMaxCells);
    uint64_t cum = 0;
    for (int i = 0; i < kDCTRange; i++) {
      cum += DC_cnt[axis][i];
      for (uint32_t k = static_cast<uint32_t>(result.size() + 1);
           k < n_intervals; k++) {
        if (cum * n_intervals >= k * total) {
          int16_t v = static_cast<int16_t>(i - kDCTOff);
          if (result.empty() || result.back() != v) result.push_back(v);
          break;
        }
      }
      if (result.size() == n_intervals - 1) break;
    }
    return result;
  }

  Factorizations MaximalFactorizations() const {
    uint32_t cap0 = std::max(1u, static_cast<uint32_t>(dc_vals[0].size()));
    uint32_t cap1 = std::max(1u, static_cast<uint32_t>(dc_vals[1].size()));
    uint32_t cap2 = std::max(1u, static_cast<uint32_t>(dc_vals[2].size()));
    Factorizations result;
    for (uint32_t a = 1; a <= cap0 && a <= kMaxCells && a <= kMaxIntervals;
         ++a) {
      for (uint32_t b = 1;
           b <= cap1 && a * b <= kMaxCells && b <= kMaxIntervals; ++b) {
        for (uint32_t c = 1;
             c <= cap2 && a * b * c <= kMaxCells && c <= kMaxIntervals; ++c) {
          bool can_inc_a = (a < cap0) && (a < kMaxIntervals) &&
                           ((a + 1) * b * c <= kMaxCells);
          bool can_inc_b = (b < cap1) && (b < kMaxIntervals) &&
                           (a * (b + 1) * c <= kMaxCells);
          bool can_inc_c = (c < cap2) && (c < kMaxIntervals) &&
                           (a * b * (c + 1) <= kMaxCells);
          if (!can_inc_a && !can_inc_b && !can_inc_c)
            result.emplace_back(a, b, c);
        }
      }
    }
    // On greyscale images we need to join the rest two components into
    // an additional empty context, so we have 1 less context for optimization,
    // max 15 contexts
    if (channels == 1) {
      uint32_t ctxs = std::min(std::get<0>(result[0]), 15u);
      std::get<0>(result[0]) = ctxs;
      while (--ctxs > 0) {
        result.emplace_back(ctxs, 1, 1);
      }
    }
    return result;
  }

  Status CountDCAC(const jpeg::JPEGData& jpeg_data, ThreadPool* pool,
                   ACCounts& ac_cnt) {
    memset(DC_cnt, 0, sizeof(DC_cnt));
    memset(ac_cnt.data(), 0, sizeof(ac_cnt));
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
                ++ac_cnt[c][p][q[jpeg::kJPEGNaturalOrder[p + 1]] + kDCTOff];
              }
            }
          }
          return true;
        },
        "CountDCAC"));

    for (auto& v : dc_vals) v.clear();
    for (uint32_t c = 0; c < channels; ++c) {
      auto& v = dc_vals[c];
      for (int di = 0; di < kDCTRange; ++di) {
        if (DC_cnt[c][di]) {
          DC_idx_LUT[c][di] = static_cast<uint32_t>(v.size());
          v.push_back(static_cast<int16_t>(di - kDCTOff));
        }
      }
    }
    return true;
  }

  static void BuildACScanOrder(
      const JPEGOptData::ACCounts& ac_cnt,
      std::array<std::array<uint8_t, kNumPos>, kNumCh>& scan_order,
      std::array<std::array<bool, kNumPos>, kNumCh>& active_scan) {
    uint8_t pos_to_scan[kNumCh][kNumPos] = {};
    for (uint32_t c = 0; c < kNumCh; ++c) {
      std::array<uint8_t, kNumPos> pos_order;
      for (uint8_t i = 0; i < kNumPos; i++) pos_order[i] = i;
      std::stable_sort(pos_order.begin(), pos_order.end(),
                       [&](uint8_t a, uint8_t b) {
                         return ac_cnt[c][a][kDCTOff] < ac_cnt[c][b][kDCTOff];
                       });
      for (uint8_t s = 0; s < kNumPos; ++s) {
        scan_order[c][s] = jpeg::kJPEGNaturalOrder[pos_order[s] + 1];
        pos_to_scan[c][pos_order[s]] = s;
      }
    }
    for (uint32_t c = 0; c < kNumCh; ++c) {
      for (uint8_t p = 0; p < kNumPos; ++p) {
        uint8_t s = pos_to_scan[c][p];
        bool first_bin = true;
        for (int ai = 0; ai < kDCTRange; ++ai) {
          if (ac_cnt[c][p][ai] != 0) {
            if (first_bin) {
              first_bin = false;
            } else {
              active_scan[c][s] = true;
              break;
            }
          }
        }
      }
    }
  }

  Status CollectACBins(const jpeg::JPEGData& jpeg_data, ThreadPool* pool,
                       const ACCounts& ac_cnt) {
    std::array<std::array<uint8_t, kNumPos>, kNumCh> scan_order = {};
    std::array<std::array<bool, kNumPos>, kNumCh> active_scan = {};
    BuildACScanOrder(ac_cnt, scan_order, active_scan);

    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, channels, ThreadPool::NoInit,
        [&](uint32_t c, size_t) -> Status {
          const auto& comp = jpeg_data.components[c];
          uint32_t wb = comp.width_in_blocks;
          uint32_t hb = comp.height_in_blocks;

          block_dc_idx[c].assign(num_blocks[c], 0);
          block_offsets[c].resize(num_blocks[c] + 1);
          block_nonzeros[c].assign(num_blocks[c], 0);
          block_bins[c].clear();
          uint32_t bc = 0;
          for (uint32_t by = 0; by < hb; ++by) {
            for (uint32_t bx = 0; bx < wb; ++bx, ++bc) {
              block_offsets[c][bc] =
                  static_cast<uint32_t>(block_bins[c].size());
              const int16_t* q = comp.coeffs.data() + (by * wb + bx) * 64;
              uint16_t dc_idx =
                  static_cast<uint16_t>(DC_idx_LUT[c][q[0] + kDCTOff]);
              block_dc_idx[c][bc] = dc_idx;
              uint32_t nonzeros_left = 0;
              for (uint32_t s = 0; s < kNumPos; ++s)
                if (q[scan_order[c][s]] != 0) ++nonzeros_left;
              block_nonzeros[c][bc] = static_cast<uint8_t>(nonzeros_left);
              for (uint32_t s = 0; s < kNumPos; ++s) {
                if (nonzeros_left == 0) break;
                int16_t coeff = q[scan_order[c][s]];
                if (active_scan[c][s]) {
                  bool nz_prev = (s > 0 && q[scan_order[c][s - 1]] != 0) ||
                                 (s == 0 && nonzeros_left > 4);
                  uint32_t zdc = static_cast<uint32_t>(
                      ZeroDensityContext(nonzeros_left, s + 1, 1, 0, nz_prev));
                  uint32_t ai = static_cast<uint32_t>(coeff + kDCTOff);
                  block_bins[c].push_back((c << 20) | (zdc << 11) | ai);
                }
                nonzeros_left -= (coeff != 0);
              }
            }
          }
          block_offsets[c][num_blocks[c]] =
              static_cast<uint32_t>(block_bins[c].size());
          return true;
        },
        "CollectBins"));
    // Precompute predict bucket per (channel, block). `predicted_nz` comes
    // from actual nonzero counts of top/left neighbors and is independent of
    // threshold assignments, so it can be precomputed once here.
    std::vector<uint32_t> row_top;
    std::vector<uint32_t> row_cur;
    for (uint32_t c = 0; c < kNumCh; ++c) {
      const uint32_t w = block_grid_w[c];
      const uint32_t h = block_grid_h[c];
      nz_pred_bucket[c].resize(num_blocks[c]);
      row_top.assign(w, 32u);
      row_cur.assign(w, 32u);
      bool has_top = false;
      for (uint32_t by = 0; by < h; ++by) {
        for (uint32_t bx = 0; bx < w; ++bx) {
          const uint32_t b = by * w + bx;
          uint32_t predicted_nz;
          if (bx == 0) {
            predicted_nz = has_top ? row_top[bx] : 32u;
          } else if (!has_top) {
            predicted_nz = row_cur[bx - 1];
          } else {
            predicted_nz = (row_top[bx] + row_cur[bx - 1] + 1u) / 2u;
          }
          nz_pred_bucket[c][b] = static_cast<uint8_t>(
              (predicted_nz < 8) ? predicted_nz : (4 + predicted_nz / 2));
          row_cur[bx] = block_nonzeros[c][b];
        }
        row_top.swap(row_cur);
        has_top = true;
      }
    }
    return true;
  }

  Status FinalizeSpatialIndexing(ThreadPool* pool) {
    memset(DC_cnt, 0, sizeof(DC_cnt));

    // Loop over image blocks
    if (channels == 1) {
      for (uint32_t b = 0; b < num_blocks[0]; ++b) {
        uint16_t dc0 = block_dc_idx[0][b];

        ++DC_cnt[0][dc_vals[0][dc0] + kDCTOff];
      }
    } else {
      for (uint32_t c = 0; c < kNumCh; ++c) {
        for (uint32_t y = 0; y < block_grid_h[c]; ++y) {
          for (uint32_t x = 0; x < block_grid_w[c]; ++x) {
            uint32_t by = y * ss_y[c];
            uint32_t bx = x * ss_x[c];
            uint16_t dc0 = block_dc_idx[0][(by / ss_y[0]) * block_grid_w[0] +
                                           bx / ss_x[0]];
            uint16_t dc1 = block_dc_idx[1][(by / ss_y[1]) * block_grid_w[1] +
                                           bx / ss_x[1]];
            uint16_t dc2 = block_dc_idx[2][(by / ss_y[2]) * block_grid_w[2] +
                                           bx / ss_x[2]];

            ++DC_cnt[0][dc_vals[0][dc0] + kDCTOff];
            ++DC_cnt[1][dc_vals[1][dc1] + kDCTOff];
            ++DC_cnt[2][dc_vals[2][dc2] + kDCTOff];
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
          dc_block_offsets[c].assign(M + 1, 0);
          for (uint16_t& dc_idx : block_dc_idx[c]) {
            dc_idx = DC_idx_LUT[c][dc_vals[c][dc_idx] + kDCTOff];
            ++dc_block_offsets[c][dc_idx + 1];
          }
          for (uint32_t v = 0; v < M; ++v)
            dc_block_offsets[c][v + 1] += dc_block_offsets[c][v];
          dc_vals[c].swap(pruned);
          return true;
        },
        "PruneDC"));

    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, channels, ThreadPool::NoInit,
        [&](uint32_t c, size_t) -> Status {
          dc_sorted_blocks[c].resize(num_blocks[c]);
          std::vector<uint32_t> write_pos = dc_block_offsets[c];
          for (uint32_t b = 0; b < num_blocks[c]; ++b) {
            uint32_t v = block_dc_idx[c][b];
            dc_sorted_blocks[c][write_pos[v]++] =
                (b / block_grid_w[c]) << 16 | (b % block_grid_w[c]);
          }
          return true;
        },
        "BuildSortedBlocks"));
    return true;
  }

  Status GenerateRLEStream(ThreadPool* pool) {
    const uint32_t BIN_N = channels << 20;
    std::vector<uint32_t> bin_start_lo(BIN_N + 1, 0);
    std::vector<uint32_t> bin_start_hi(BIN_N + 1, 0);

    for (uint32_t c = 0; c < channels; ++c) {
      uint32_t w = block_grid_w[c];
      uint32_t h = block_grid_h[c];
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
          uint32_t b = y * w + x;
          uint32_t b0 = (y * ss_y[c] / ss_y[0]) * block_grid_w[0] +
                        (x * ss_x[c] / ss_x[0]);
          uint32_t dc0 = block_dc_idx[0][b0];
          for (uint32_t pi = block_offsets[c][b]; pi < block_offsets[c][b + 1];
               ++pi) {
            uint32_t bin = block_bins[c][pi];
            if (dc0 & 0x400u) {
              ++bin_start_hi[bin + 1];
            } else {
              ++bin_start_lo[bin + 1];
            }
          }
        }
      }
    }

    std::vector<uint32_t> active_bins;
    active_bins.reserve(BIN_N);
    for (uint32_t b = 0; b < BIN_N; ++b) {
      uint32_t c_lo = bin_start_lo[b + 1];
      uint32_t c_hi = bin_start_hi[b + 1];
      bin_start_lo[b + 1] = bin_start_lo[b] + c_lo;
      bin_start_hi[b + 1] = bin_start_hi[b] + c_hi;
      if (c_lo != 0 || c_hi != 0) active_bins.push_back(b);
    }
    active_bins.shrink_to_fit();

    size_t lo_size = bin_start_lo[BIN_N];
    size_t hi_size = bin_start_hi[BIN_N];
    size_t raw_size = lo_size + hi_size;
    std::vector<uint32_t> flat_lo(lo_size);
    std::vector<uint32_t> flat_hi(hi_size);

    if (channels == 1) {
      for (uint32_t b = 0; b < num_blocks[0]; ++b) {
        uint32_t dc0 = block_dc_idx[0][b];
        for (uint32_t pi = block_offsets[0][b]; pi < block_offsets[0][b + 1];
             ++pi) {
          uint32_t bin = block_bins[0][pi];
          uint32_t dc_key = (dc0 & 0x3FFu) << 22;
          if (dc0 & 0x400u) {
            flat_hi[bin_start_hi[bin]++] = dc_key;
          } else {
            flat_lo[bin_start_lo[bin]++] = dc_key;
          }
        }
      }
    } else {
      for (uint32_t c = 0; c < kNumCh; ++c) {
        uint32_t w = block_grid_w[c];
        uint32_t h = block_grid_h[c];
        for (uint32_t y = 0; y < h; ++y) {
          for (uint32_t x = 0; x < w; ++x) {
            uint32_t b = y * w + x;
            uint32_t b0 = (y * ss_y[c] / ss_y[0]) * block_grid_w[0] +
                          (x * ss_x[c] / ss_x[0]);
            uint32_t b1 = (y * ss_y[c] / ss_y[1]) * block_grid_w[1] +
                          (x * ss_x[c] / ss_x[1]);
            uint32_t b2 = (y * ss_y[c] / ss_y[2]) * block_grid_w[2] +
                          (x * ss_x[c] / ss_x[2]);

            uint32_t dc0 = block_dc_idx[0][b0];
            uint32_t dc1 = block_dc_idx[1][b1];
            uint32_t dc2 = block_dc_idx[2][b2];
            for (uint32_t pi = block_offsets[c][b];
                 pi < block_offsets[c][b + 1]; ++pi) {
              uint32_t bin = block_bins[c][pi];
              uint32_t dc_key = ((dc0 & 0x3FFu) << 22) | (dc1 << 11) | dc2;
              if (dc0 & 0x400u) {
                flat_hi[bin_start_hi[bin]++] = dc_key;
              } else {
                flat_lo[bin_start_lo[bin]++] = dc_key;
              }
            }
          }
        }
      }
    }

    uint32_t kChunk = 1024;
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0,
        static_cast<uint32_t>((active_bins.size() + kChunk - 1) / kChunk),
        ThreadPool::NoInit,
        [&](uint32_t chunk, size_t) -> Status {
          uint32_t idx_start = chunk * kChunk;
          uint32_t idx_end = std::min(
              idx_start + kChunk, static_cast<uint32_t>(active_bins.size()));
          for (uint32_t idx = idx_start; idx < idx_end; ++idx) {
            uint32_t b = active_bins[idx];
            uint32_t s_lo = (b == 0) ? 0 : bin_start_lo[b - 1];
            uint32_t e_lo = bin_start_lo[b];
            uint32_t s_hi = (b == 0) ? 0 : bin_start_hi[b - 1];
            uint32_t e_hi = bin_start_hi[b];
            if (s_lo < e_lo)
              std::sort(flat_lo.data() + s_lo, flat_lo.data() + e_lo);
            if (s_hi < e_hi)
              std::sort(flat_hi.data() + s_hi, flat_hi.data() + e_hi);
          }
          return true;
        },
        "SortBins"));

    std::vector<ACEntry> stream;
    stream.reserve(raw_size + raw_size / 16);
    uint32_t ctx_len = 0;
    uint32_t prev_bin = UINT32_MAX;
    std::array<uint32_t, kZeroDensityContextCount> zdc_total = {};

    const uint32_t max_key_h_sparse =
        static_cast<uint32_t>(kZeroDensityContextCount) * 2048;
    compact_map_h.assign(max_key_h_sparse, kInvalidCompactH);
    dense_to_zdcai.clear();
    dense_to_zdcai.reserve(active_bins.size());
    num_zdcai = 0;

    uint32_t lo_pos = 0;
    uint32_t hi_pos = 0;
    for (unsigned int bin : active_bins) {
      uint32_t zdc_ai = bin & 0xFFFFFu;
      if (compact_map_h[zdc_ai] == kInvalidCompactH) {
        compact_map_h[zdc_ai] = num_zdcai++;
        dense_to_zdcai.push_back(zdc_ai);
      }

      uint32_t start_lo = lo_pos;
      uint32_t start_hi = hi_pos;
      uint32_t end_lo = bin_start_lo[bin];
      uint32_t end_hi = bin_start_hi[bin];
      lo_pos = end_lo;
      hi_pos = end_hi;

      bool new_ctx =
          (prev_bin == UINT32_MAX) || ((bin >> 11) != (prev_bin >> 11));
      if (new_ctx) {
        if (prev_bin != UINT32_MAX) {
          zdc_total[(prev_bin >> 11) & 0x1FFu] += ctx_len;
        }
        ctx_len = 0;
      }
      ctx_len += (end_lo - start_lo) + (end_hi - start_hi);

      bool bin_change = (prev_bin != UINT32_MAX);
      bool ctx_change = bin_change && ((bin >> 11) != (prev_bin >> 11));

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
          if (first_in_bin || (dc0 - cur_dc0 > 15u)) {
            stream.push_back(
                (1u << 31) |
                (static_cast<uint32_t>(ctx_change && first_in_bin) << 30) |
                (static_cast<uint32_t>(bin_change && first_in_bin) << 29) |
                (bin << 7) | (dc0 >> 4));
            cur_dc0 = (dc0 >> 4) << 4;
          }
          uint32_t delta_dc0 = dc0 - cur_dc0;
          uint32_t header = (delta_dc0 << 27) | (dc1 << 16) | (dc2 << 5);
          if (run <= 31) {
            stream.push_back(header | (run - 1));
          } else {
            stream.push_back(header | 0x1Fu);
            stream.push_back(run);
          }
          cur_dc0 = dc0;
          first_in_bin = false;
          i = j;
        }
      };

      emit_half(flat_lo, start_lo, end_lo, 0u);
      emit_half(flat_hi, start_hi, end_hi, 0x400u);
      prev_bin = bin;
    }

    if (prev_bin != UINT32_MAX) {
      zdc_total[(prev_bin >> 11) & 0x1FFu] += ctx_len;
    }
    stream.shrink_to_fit();
    AC_stream = std::move(stream);

    uint32_t max_zdc_total =
        *std::max_element(zdc_total.begin(), zdc_total.end());
    InitFTab(max_zdc_total + 1);

    return true;
  }

  Status BuildFromJPEG(const jpeg::JPEGData& jpeg_data, ThreadPool* pool);
};

struct PartitioningCtx {
  std::shared_ptr<const JPEGOptData> data_;

  std::vector<uint32_t> h_cnt;
  std::vector<uint32_t> N_cnt;
  std::vector<int64_t> costs;
  std::vector<int64_t> DP_prev;
  std::vector<int64_t> DP_curr;
  std::vector<uint16_t> split_table;
  std::vector<uint16_t> row_done;
  std::vector<int64_t> row_cum;

  using Bin = std::pair<uint16_t, uint32_t>;
  using CellHistory = std::array<std::vector<Bin>, kMaxCells / 2>;
  CellHistory h_history;
  CellHistory N_history;

  std::vector<uint64_t> touched_h;
  std::vector<uint64_t> touched_N;
  std::vector<uint64_t> group_touched_h;
  std::vector<uint64_t> group_touched_N;

  std::vector<uint16_t> ax0_to_k;
  std::vector<uint16_t> k_to_dc0;
  std::vector<uint16_t> ax1_row;
  std::vector<uint16_t> ax2_col;

  std::vector<CompactHistogram> scratch_hist_h_;
  std::vector<std::array<uint32_t, kZeroDensityContextCount>> scratch_hist_N_;
  std::vector<std::array<uint32_t, kJPEGNZBucketCount>> scratch_hist_nz_N_;
  std::vector<std::array<uint32_t, kJPEGNZHistogramSize>> scratch_hist_nz_h_;
  std::array<std::vector<uint8_t>, kNumCh> local_cluster_map_;

  explicit PartitioningCtx(std::shared_ptr<const JPEGOptData> d)
      : data_(std::move(d)),
        DP_prev(kDCTRange, 0),
        DP_curr(kDCTRange, 0),
        row_done(kDCTRange, 0),
        row_cum(kDCTRange, 0),
        h_history(),
        N_history(),
        touched_h(kBinCount, 0),
        touched_N(kBinCount, 0),
        group_touched_h(kGroupCount, 0),
        group_touched_N(kGroupCount, 0),
        ax0_to_k(kDCTRange, 0),
        k_to_dc0(kDCTRange, 0),
        ax1_row(kDCTRange, 0),
        ax2_col(kDCTRange, 0) {}

  const JPEGOptData& data() const { return *data_; }
  const std::vector<ACEntry>& ac_stream() const { return data().AC_stream; }

  static ptrdiff_t Bkt(int dc, const Thresholds& T) {
    for (size_t i = 0; i < T.size(); i++)
      if (T[i] > dc) return i;
    return T.size();
    // return std::upper_bound(T.begin(), T.end(), dc) - T.begin();
  }

  void UpdateMaps(uint32_t axis, const Thresholds& T0, const Thresholds& T1,
                  const Thresholds& T2, bool ax0_identity = false) {
    const JPEGOptData& d = data();
    uint32_t ax0 = axis;
    uint32_t ax1 = (axis + 1) % 3;
    uint32_t ax2 = (axis + 2) % 3;
    uint32_t M0 = static_cast<uint32_t>(d.dc_vals[ax0].size());
    uint32_t M1 = static_cast<uint32_t>(d.dc_vals[ax1].size());
    uint32_t M2 = static_cast<uint32_t>(d.dc_vals[ax2].size());
    for (uint32_t i = 0; i < M0; i++) {
      ax0_to_k[i] = ax0_identity
                        ? static_cast<uint16_t>(i)
                        : static_cast<uint16_t>(Bkt(d.dc_vals[ax0][i], T0));
      if (ax0_identity) k_to_dc0[i] = static_cast<uint16_t>(i);
    }
    uint32_t n2 = static_cast<uint32_t>(T2.size() + 1);
    for (uint32_t i = 0; i < M1; i++)
      ax1_row[i] = static_cast<uint16_t>(Bkt(d.dc_vals[ax1][i], T1) * n2);
    for (uint32_t i = 0; i < M2; i++)
      ax2_col[i] = static_cast<uint16_t>(Bkt(d.dc_vals[ax2][i], T2));
  }

  void UpdateMaps(const ThresholdSet& T) {
    UpdateMaps(0, T.TY(), T.TCb(), T.TCr());
  }

  uint32_t PrepareBucketing(uint32_t axis, uint32_t num_intervals,
                            const Thresholds& T1, const Thresholds& T2,
                            uint32_t M_target) {
    const JPEGOptData& d = data();
    UpdateMaps(axis, {}, T1, T2);
    uint32_t M = static_cast<uint32_t>(d.dc_vals[axis].size());
    uint32_t M_eff = std::min(M, M_target);
    if (M_eff < M) {
      Thresholds bkt_thresh = d.InitThresh(static_cast<int>(axis), M_eff);
      uint32_t cur_k = 0;
      k_to_dc0[0] = 0;
      for (uint16_t i = 0; i < M; i++) {
        uint16_t k = static_cast<uint16_t>(Bkt(d.dc_vals[axis][i], bkt_thresh));
        ax0_to_k[i] = k;
        while (cur_k < k) k_to_dc0[++cur_k] = i;
      }
      M_eff = static_cast<uint32_t>(ax0_to_k[M - 1]) + 1;
      if (M_eff < num_intervals) {
        UpdateMaps(axis, {}, T1, T2, true);
        M_eff = M;
      }
    } else {
      UpdateMaps(axis, {}, T1, T2, true);
    }
    return M_eff;
  }

  template <class FlushH, class FlushN, class OnRun>
  void StreamSweep(FlushH&& flush_h, FlushN&& flush_N, OnRun&& on_run) const {
    const std::vector<uint32_t>& stream = ac_stream();
    uint32_t dc0_idx = 0;
    uint32_t bin_state = 0;
    for (size_t si = 0; si < stream.size(); ++si) {
      const uint32_t frame = stream[si];
      if (frame >> 31) {
        dc0_idx = (frame & 0x7Fu) << 4;
        bin_state = (frame >> 7) & 0x3FFFFFu;
        if ((frame >> 29) & 1) {
          flush_h();
          if ((frame >> 30) & 1) flush_N();
        }
        continue;
      }
      dc0_idx += (frame >> 27) & 0xFu;
      uint32_t dc1_idx = (frame >> 16) & 0x7FFu;
      uint32_t dc2_idx = (frame >> 5) & 0x7FFu;
      uint32_t run_sym = frame & 0x1Fu;
      uint32_t run = (run_sym == 0x1Fu) ? stream[++si] : run_sym + 1;
      on_run(dc0_idx, dc1_idx, dc2_idx, run, bin_state);
    }
  }

  void EnsureCost(uint32_t M, uint32_t l, uint32_t n) {
    l = std::min(l, n);
    if (row_done[n] != UINT16_MAX && row_done[n] >= l) return;
    // Row `nn` needs columns up to `min(l, nn)` because row `nn+1` accesses
    // `costs[nn*M+v]` for `v` in `[0, min(l, nn)]`. Processing rows from 0
    // to `n` guarantees that `costs[(nn-1)*M+v]` is already available when
    // computing row `nn`.
    for (uint32_t nn = 0; nn <= n; ++nn) {
      uint32_t needed = std::min(l, nn);
      if (row_done[nn] != UINT16_MAX && row_done[nn] >= needed) continue;
      uint32_t start = (row_done[nn] == UINT16_MAX) ? 0u : (row_done[nn] + 1);
      int64_t cum = (start == 0) ? 0 : row_cum[nn];
      for (uint32_t v = start; v <= needed; ++v) {
        cum += costs[nn * M + v];
        if (nn > 0 && v < nn) {
          costs[nn * M + v] = costs[(nn - 1) * M + v] + cum;
        } else {
          costs[nn * M + v] = cum;
        }
      }
      row_cum[nn] = cum;
      row_done[nn] = static_cast<uint16_t>(needed);
    }
  }

  int64_t GetCost(uint32_t M, uint32_t l, uint32_t n) {
    EnsureCost(M, l, n);
    return costs[n * M + l];
  }

  Thresholds KnuthSolver(uint32_t axis, uint32_t K, uint32_t M,
                         uint32_t M_eff = 0) {
    if (!M_eff) M_eff = M;
    const JPEGOptData& d = data();
    if (M_eff <= 1 || K <= 1) return {};

    split_table.assign(K * M_eff, 0);
    std::fill(row_done.begin(), row_done.begin() + M_eff, UINT16_MAX);
    std::fill(row_cum.begin(), row_cum.begin() + M_eff, 0);

    for (uint32_t n = 0; n < M_eff; n++) DP_prev[n] = GetCost(M_eff, 0, n);

    constexpr int64_t INF = std::numeric_limits<int64_t>::max();
    for (uint32_t k = 1; k < K; k++) {
      std::fill(DP_curr.begin(), DP_curr.begin() + M_eff, INF);
      uint16_t* split_curr = &split_table[k * M_eff];
      const uint16_t* split_prev = split_curr - M_eff;
      uint32_t s_max = M_eff - 1;
      for (uint32_t n_plus_1 = M_eff; n_plus_1 > 0; --n_plus_1) {
        uint32_t n = n_plus_1 - 1;
        uint32_t s_min = std::max<uint32_t>(split_prev[n], k);
        if (s_min > s_max) continue;
        uint32_t best_s = s_min;
        for (uint32_t s = s_min; s <= s_max; s++) {
          int64_t val = DP_prev[s - 1] + GetCost(M_eff, s, n);
          if (val < DP_curr[n]) {
            DP_curr[n] = val;
            best_s = s;
          }
        }
        split_curr[n] = static_cast<uint16_t>(best_s);
        s_max = std::min(best_s, n - 1);
      }
      std::swap(DP_prev, DP_curr);
    }

    Thresholds T;
    T.reserve(K - 1);
    uint32_t v = M_eff - 1;
    for (uint32_t k = K - 1; k > 0; k--) {
      uint32_t s = split_table[k * M_eff + v];
      T.push_back(d.dc_vals[axis][k_to_dc0[s]]);
      v = s - 1;
    }
    std::reverse(T.begin(), T.end());
    return T;
  }

  Thresholds OptimizeAxisSingleSplit(uint32_t axis, uint32_t ncells,
                                     uint32_t M_eff) {
    const JPEGOptData& d = data();
    if (costs.size() < M_eff + 1) {
      costs.assign(M_eff + 1, 0);
    } else {
      std::fill(costs.data(), costs.data() + M_eff + 1, 0);
    }

    uint64_t bin_mask = 0;
    uint64_t ctx_mask = 0;

    if (axis == 0) {
      auto flush_hist = [&](CellHistory& ch, uint64_t& mask, int sign) {
        while (mask) {
          uint32_t ci = static_cast<uint32_t>(CountrZero64(mask));
          mask &= mask - 1;
          auto& hist = ch[ci];
          uint32_t total = 0;
          for (const auto& hi : hist) total += hi.second;
          uint32_t j_before_l = 0;
          uint32_t l = 1;
          int64_t prev_term = 0;
          for (const auto& h : hist) {
            int64_t term =
                sign * (d.ftab[j_before_l] + d.ftab[total - j_before_l]);
            costs[l] += term - prev_term;
            j_before_l += h.second;
            l = static_cast<uint32_t>(h.first) + 1;
            prev_term = term;
          }
          costs[l] += sign * static_cast<int64_t>(d.ftab[total]) - prev_term;
          hist.clear();
        }
      };

      StreamSweep(
          [&]() { flush_hist(h_history, bin_mask, -1); },
          [&]() { flush_hist(N_history, ctx_mask, +1); },
          [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx,
              uint32_t run, uint32_t) {
            uint32_t dc_k_bkt = ax0_to_k[dc0_idx];
            uint32_t ci =
                static_cast<uint32_t>(ax1_row[dc1_idx] + ax2_col[dc2_idx]);
            bin_mask |= (1ULL << ci);
            ctx_mask |= (1ULL << ci);
            if (!h_history[ci].empty() &&
                h_history[ci].back().first == dc_k_bkt) {
              h_history[ci].back().second += run;
            } else {
              h_history[ci].emplace_back(static_cast<uint16_t>(dc_k_bkt), run);
            }
            if (!N_history[ci].empty() &&
                N_history[ci].back().first == dc_k_bkt) {
              N_history[ci].back().second += run;
            } else {
              N_history[ci].emplace_back(static_cast<uint16_t>(dc_k_bkt), run);
            }
          });
      flush_hist(h_history, bin_mask, -1);
      flush_hist(N_history, ctx_mask, +1);
    } else {
      uint32_t cnt_size = M_eff * ncells;
      if (h_cnt.size() < cnt_size) h_cnt.assign(cnt_size, 0);
      if (N_cnt.size() < cnt_size) N_cnt.assign(cnt_size, 0);

      auto flush_dense = [&](std::vector<uint32_t>& cnt, uint64_t& mask,
                             int sign) {
        while (mask) {
          uint32_t ci = static_cast<uint32_t>(CountrZero64(mask));
          mask &= mask - 1;
          uint32_t total = 0;
          for (uint32_t n = 0; n < M_eff; n++) total += cnt[ci * M_eff + n];
          uint32_t j_before_l = 0;
          uint32_t l = 1;
          int64_t prev_term = 0;
          for (uint32_t n = 0; n < M_eff; n++) {
            uint32_t f = cnt[ci * M_eff + n];
            cnt[ci * M_eff + n] = 0;
            if (f == 0) continue;
            int64_t term = sign * (static_cast<int64_t>(d.ftab[j_before_l]) +
                                   d.ftab[total - j_before_l]);
            costs[l] += term - prev_term;
            prev_term = term;
            j_before_l += f;
            l = n + 1;
          }
          costs[l] += sign * static_cast<int64_t>(d.ftab[total]) - prev_term;
        }
      };

      StreamSweep([&]() { flush_dense(h_cnt, bin_mask, -1); },
                  [&]() { flush_dense(N_cnt, ctx_mask, +1); },
                  [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx,
                      uint32_t run, uint32_t) {
                    uint32_t dc_arr[3] = {dc0_idx, dc1_idx, dc2_idx};
                    uint32_t dc_k_bkt = ax0_to_k[dc_arr[axis]];
                    uint32_t ax1 = (axis + 1) % 3;
                    uint32_t ax2 = (axis + 2) % 3;
                    uint32_t ci = static_cast<uint32_t>(ax1_row[dc_arr[ax1]] +
                                                        ax2_col[dc_arr[ax2]]);
                    uint32_t idx = ci * M_eff + dc_k_bkt;
                    h_cnt[idx] += run;
                    bin_mask |= (1ULL << ci);
                    N_cnt[idx] += run;
                    ctx_mask |= (1ULL << ci);
                  });
      flush_dense(h_cnt, bin_mask, -1);
      flush_dense(N_cnt, ctx_mask, +1);
    }

    int64_t cur = 0;
    int64_t best = std::numeric_limits<int64_t>::max();
    uint32_t best_s = 1;
    for (uint32_t s = 1; s < M_eff; s++) {
      cur += costs[s];
      if (cur < best) {
        best = cur;
        best_s = s;
      }
    }
    return {d.dc_vals[axis][k_to_dc0[best_s]]};
  }

  template <int sign>
  void FlushTerm(std::vector<uint32_t>& cnt, uint64_t* group_touched,
                 uint64_t* touched, uint32_t M_eff) {
    const JPEGOptData& d = data();
    auto& history = h_history[0];
    history.clear();
    uint32_t cur_ci = UINT32_MAX;
    for (uint32_t hi_idx = 0; hi_idx < kGroupCount; hi_idx++) {
      uint64_t group_mask = group_touched[hi_idx];
      while (group_mask) {
        uint32_t lo_idx = static_cast<uint32_t>(CountrZero64(group_mask));
        uint32_t group_idx = (hi_idx << 6) | lo_idx;
        uint64_t cell_mask = touched[group_idx];
        while (cell_mask) {
          uint32_t t = static_cast<uint32_t>(CountrZero64(cell_mask));
          uint32_t bit_idx = (group_idx << 6) | t;
          uint32_t ci = bit_idx / M_eff;
          uint32_t n = bit_idx % M_eff;
          if (ci != cur_ci) {
            history.clear();
            cur_ci = ci;
          }
          int64_t* cost_row = &costs[n * M_eff];
          uint32_t freq = cnt[bit_idx];
          uint32_t j_n = freq;
          if (!history.empty()) {
            j_n += history.back().second;
            uint32_t l = 0;
            uint32_t j_before_l = 0;
            int64_t prev_term = 0;
            for (const Bin& h : history) {
              uint32_t j_ln = j_n - j_before_l;
              int64_t term = sign * (d.ftab[j_ln - freq] - d.ftab[j_ln]);
              cost_row[l] += term - prev_term;
              l = h.first + 1;
              j_before_l = h.second;
              prev_term = term;
            }
            cost_row[l] -= sign * d.ftab[freq] + prev_term;
          }
          history.emplace_back(static_cast<uint16_t>(n), j_n);
          cnt[bit_idx] = 0;
          cell_mask &= cell_mask - 1;
        }
        touched[group_idx] = 0;
        group_mask &= group_mask - 1;
      }
      group_touched[hi_idx] = 0;
    }
  }

  Thresholds OptimizeAxisSingleSweep(uint32_t axis, uint32_t num_intervals,
                                     const Thresholds& T1, const Thresholds& T2,
                                     uint32_t M_target = kMTarget) {
    const JPEGOptData& d = data();
    if (num_intervals == 1) return {};
    uint32_t M = static_cast<uint32_t>(d.dc_vals[axis].size());
    if (M <= num_intervals)
      return Thresholds(d.dc_vals[axis].begin() + 1, d.dc_vals[axis].end());

    uint32_t ax1 = (axis + 1) % 3;
    uint32_t ax2 = (axis + 2) % 3;
    uint32_t n1 = static_cast<uint32_t>(T1.size() + 1);
    uint32_t n2 = static_cast<uint32_t>(T2.size() + 1);
    uint32_t ncells = n1 * n2;
    uint32_t M_eff = PrepareBucketing(axis, num_intervals, T1, T2, M_target);

    if (num_intervals == 2) return OptimizeAxisSingleSplit(axis, ncells, M_eff);

    size_t cnt_size = static_cast<size_t>(ncells) * M_eff;
    if (h_cnt.size() < cnt_size) h_cnt.assign(cnt_size, 0);
    if (N_cnt.size() < cnt_size) N_cnt.assign(cnt_size, 0);
    size_t cost_size = static_cast<size_t>(M_eff) * M_eff;
    if (costs.size() < cost_size) {
      costs.assign(cost_size, 0);
    } else {
      std::fill(costs.data(), costs.data() + cost_size, 0);
    }

    StreamSweep(
        [&]() {
          FlushTerm<+1>(h_cnt, group_touched_h.data(), touched_h.data(), M_eff);
        },
        [&]() {
          FlushTerm<-1>(N_cnt, group_touched_N.data(), touched_N.data(), M_eff);
        },
        [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
            uint32_t) {
          uint32_t dc_arr[3] = {dc0_idx, dc1_idx, dc2_idx};
          uint32_t dc_k_bkt = ax0_to_k[dc_arr[axis]];
          uint32_t ci = static_cast<uint32_t>(ax1_row[dc_arr[ax1]] +
                                              ax2_col[dc_arr[ax2]]);
          uint32_t idx = ci * M_eff + dc_k_bkt;
          if (h_cnt[idx] == 0) {
            size_t gi = idx >> 6;
            group_touched_h[gi >> 6] |= (1ULL << (gi & 63));
            touched_h[gi] |= (1ULL << (idx & 63));
          }
          if (N_cnt[idx] == 0) {
            size_t gi = idx >> 6;
            group_touched_N[gi >> 6] |= (1ULL << (gi & 63));
            touched_N[gi] |= (1ULL << (idx & 63));
          }
          h_cnt[idx] += run;
          N_cnt[idx] += run;
        });
    FlushTerm<+1>(h_cnt, group_touched_h.data(), touched_h.data(), M_eff);
    FlushTerm<-1>(N_cnt, group_touched_N.data(), touched_N.data(), M_eff);

    return KnuthSolver(axis, num_intervals, M, M_eff);
  }

  std::pair<int64_t, ThresholdSet> OptimizeThresholds(
      ThresholdSet T, uint32_t M_target = UINT32_MAX, uint32_t max_iters = 20) {
    uint32_t a = static_cast<uint32_t>(T.TY().size() + 1);
    uint32_t b = static_cast<uint32_t>(T.TCb().size() + 1);
    uint32_t c = static_cast<uint32_t>(T.TCr().size() + 1);
    ThresholdSet newT;
    for (size_t i = 0; i < kNumCh; i++) newT.T[i].reserve(kMaxCells);

    bool TY_changed = (a != 1);
    bool TCb_changed = (b != 1);
    bool TCr_changed = (c != 1);
    for (uint32_t iter = 0; iter < max_iters; iter++) {
      if ((a != 1) && (iter == 0 || TCb_changed || TCr_changed)) {
        newT.TY() = OptimizeAxisSingleSweep(0, a, T.TCb(), T.TCr(), M_target);
        TY_changed = (newT.TY() != T.TY());
        std::swap(T.TY(), newT.TY());
      } else {
        TY_changed = false;
      }
      if ((b != 1) && (iter == 0 || TY_changed || TCr_changed)) {
        newT.TCb() = OptimizeAxisSingleSweep(1, b, T.TCr(), T.TY(), M_target);
        TCb_changed = (newT.TCb() != T.TCb());
        std::swap(T.TCb(), newT.TCb());
      } else {
        TCb_changed = false;
      }
      if ((c != 1) && (iter == 0 || TY_changed || TCb_changed)) {
        newT.TCr() = OptimizeAxisSingleSweep(2, c, T.TY(), T.TCb(), M_target);
        TCr_changed = (newT.TCr() != T.TCr());
        std::swap(T.TCr(), newT.TCr());
      } else {
        TCr_changed = false;
      }
      if (!TY_changed && !TCb_changed && !TCr_changed) break;
    }
    return {TotalCost(T), T};
  }

  int64_t TotalCost(const ThresholdSet& T) {
    const JPEGOptData& d = data();
    uint32_t na = static_cast<uint32_t>(T.TY().size() + 1);
    uint32_t num_cells = na * static_cast<uint32_t>(T.TCb().size() + 1) *
                         static_cast<uint32_t>(T.TCr().size() + 1);
    uint32_t state_size = num_cells;
    UpdateMaps(T);
    if (h_cnt.size() < state_size) h_cnt.assign(state_size, 0);
    if (N_cnt.size() < state_size) N_cnt.assign(state_size, 0);
    int64_t cost = 0;

    auto flush_h = [&]() {
      for (size_t gi = 0; gi < kGroupCount; ++gi) {
        uint64_t gm = group_touched_h[gi];
        while (gm) {
          size_t gi2 = (gi << 6) | static_cast<uint32_t>(CountrZero64(gm));
          uint64_t cm = touched_h[gi2];
          while (cm) {
            size_t idx = (gi2 << 6) | static_cast<uint32_t>(CountrZero64(cm));
            cost -= d.ftab[h_cnt[idx]];
            h_cnt[idx] = 0;
            cm &= cm - 1;
          }
          touched_h[gi2] = 0;
          gm &= gm - 1;
        }
        group_touched_h[gi] = 0;
      }
    };
    auto flush_N = [&]() {
      for (size_t gi = 0; gi < kGroupCount; ++gi) {
        uint64_t gm = group_touched_N[gi];
        while (gm) {
          size_t gi2 = (gi << 6) | static_cast<uint32_t>(CountrZero64(gm));
          uint64_t cm = touched_N[gi2];
          while (cm) {
            size_t idx = (gi2 << 6) | static_cast<uint32_t>(CountrZero64(cm));
            cost += d.ftab[N_cnt[idx]];
            N_cnt[idx] = 0;
            cm &= cm - 1;
          }
          touched_N[gi2] = 0;
          gm &= gm - 1;
        }
        group_touched_N[gi] = 0;
      }
    };

    StreamSweep(flush_h, flush_N,
                [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx,
                    uint32_t run, uint32_t) {
                  uint32_t ci = ax1_row[dc1_idx] + ax2_col[dc2_idx];
                  uint32_t idx = ci * na + ax0_to_k[dc0_idx];
                  if (h_cnt[idx] == 0) {
                    size_t gi = idx >> 6;
                    group_touched_h[gi >> 6] |= 1ULL << (gi & 63);
                    touched_h[gi] |= 1ULL << (idx & 63);
                  }
                  if (N_cnt[idx] == 0) {
                    size_t gi = idx >> 6;
                    group_touched_N[gi >> 6] |= 1ULL << (gi & 63);
                    touched_N[gi] |= 1ULL << (idx & 63);
                  }
                  h_cnt[idx] += run;
                  N_cnt[idx] += run;
                });
    flush_h();
    flush_N();
    return cost;
  }

  using DenseNHistogram = std::array<uint32_t, kZeroDensityContextCount>;
  using DenseNHistogramSet = std::vector<DenseNHistogram>;
  using DenseNZPredHistogram = std::array<uint32_t, kJPEGNZBucketCount>;
  using DenseNZPredHistogramSet = std::vector<DenseNZPredHistogram>;
  using DenseNZHistogram = std::array<uint32_t, kJPEGNZHistogramSize>;
  using DenseNZHistogramSet = std::vector<DenseNZHistogram>;
  using CompactHistogramSet = std::vector<CompactHistogram>;

  template <size_t Size>
  static bool DenseHistogramEmpty(const std::array<uint32_t, Size>& hist) {
    for (uint32_t freq : hist) {
      if (freq != 0) return false;
    }
    return true;
  }

  template <size_t Size>
  static void AddDenseHistogram(std::array<uint32_t, Size>& dst,
                                const std::array<uint32_t, Size>& src) {
    for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
  }

  static uint32_t NZHistogramIndex(uint32_t pb, uint32_t nz_count) {
    JXL_DASSERT(pb < kJPEGNZBucketCount);
    JXL_DASSERT(nz_count < kJPEGNZCountLimit);
    return pb * kJPEGNZCountLimit + nz_count;
  }

  struct Clustering {
    struct RollbackScratch {
      DenseNHistogram hist_N_a = {};
      DenseNHistogram hist_N_b = {};
      DenseNZPredHistogram hist_nz_N_a = {};
      DenseNZPredHistogram hist_nz_N_b = {};
      DenseNZHistogram hist_nz_h_a = {};
      DenseNZHistogram hist_nz_h_b = {};
      CompactHistogram hist_h_a;
      CompactHistogram hist_h_b;
      std::vector<uint32_t> active;
    };

    int64_t clustered_cost;
    uint32_t ctx_num;
    ContextMap ctx_map;

    CompactHistogramSet hist_h;
    DenseNHistogramSet hist_N;
    DenseNZHistogramSet hist_nz_h;
    DenseNZPredHistogramSet hist_nz_N;

    Clustering() : clustered_cost(0), ctx_num(0) {}
    Clustering(Clustering&&) = default;
    Clustering& operator=(Clustering&&) = default;
    Clustering(const Clustering&) = delete;
    Clustering& operator=(const Clustering&) = delete;

    Status AgglomerativeClusterCore(const JPEGOptData& d, uint32_t total_ctxs,
                                    uint32_t num_clusters,
                                    bool overhead_aware_tail,
                                    ThreadPool* pool) {
      auto rollback_scratch = make_unique<RollbackScratch>();
      RollbackScratch& rollback = *rollback_scratch;
      num_clusters = std::max(num_clusters, uint32_t{1});
      ctx_map.assign(total_ctxs, 0);

      std::vector<int64_t> E(total_ctxs, 0);
      JXL_RETURN_IF_ERROR(RunOnPool(
          pool, 0, total_ctxs, ThreadPool::NoInit,
          [&](uint32_t i, size_t) -> Status {
            int64_t local_E = 0;
            for (uint32_t freq : hist_N[i]) {
              if (freq != 0) local_E += d.ftab[freq];
            }
            for (uint32_t id : hist_h[i].used_ids) {
              local_E -= d.ftab[hist_h[i].Get(id)];
            }
            for (uint32_t freq : hist_nz_N[i]) {
              if (freq != 0) local_E += d.nz_entropy(freq);
            }
            for (uint32_t freq : hist_nz_h[i]) {
              if (freq != 0) local_E -= d.nz_entropy(freq);
            }
            E[i] = local_E;
            return true;
          },
          "InitEntropy"));

      int64_t initial_cost = 0;
      for (int64_t e : E) initial_cost += e;

      // List of active clusters - starting from all nonempty contexts
      std::vector<uint32_t> active;
      active.reserve(total_ctxs);
      for (uint32_t i = 0; i < total_ctxs; i++) {
        if (!DenseHistogramEmpty(hist_N[i]) ||
            !DenseHistogramEmpty(hist_nz_N[i])) {
          active.push_back(i);
        }
      }
      std::vector<uint32_t> initial_active = active;

      uint32_t active_clusters = static_cast<uint32_t>(active.size());
      std::vector<uint32_t> parent(total_ctxs);
      std::iota(parent.begin(), parent.end(), 0);

      std::vector<int64_t> deltas;
      auto get_delta = [&](uint32_t cl_a, uint32_t cl_b) -> int64_t& {
        return deltas[std::min(cl_a, cl_b) * total_ctxs + std::max(cl_a, cl_b)];
      };
      auto merge_delta = [&](uint32_t cl_a, uint32_t cl_b) -> int64_t {
        int64_t delta = 0;

        for (size_t bin = 0; bin < hist_N[cl_a].size(); ++bin) {
          uint32_t freq_a = hist_N[cl_a][bin];
          uint32_t freq_b = hist_N[cl_b][bin];
          if (freq_a != 0 && freq_b != 0) {
            delta += d.ftab[freq_a + freq_b] - d.ftab[freq_a] - d.ftab[freq_b];
          }
        }
        const CompactHistogram& hist_h_a = hist_h[cl_a];
        const CompactHistogram& hist_h_b = hist_h[cl_b];
        const CompactHistogram* iter_h = &hist_h_a;
        if (hist_h_a.used_ids.size() > hist_h_b.used_ids.size()) {
          iter_h = &hist_h_b;
        }
        for (uint32_t id : iter_h->used_ids) {
          uint32_t freq_a = hist_h_a.Get(id);
          uint32_t freq_b = hist_h_b.Get(id);
          if (freq_a != 0 && freq_b != 0) {
            delta -= d.ftab[freq_a + freq_b] - d.ftab[freq_a] - d.ftab[freq_b];
          }
        }
        for (size_t bin = 0; bin < hist_nz_N[cl_a].size(); ++bin) {
          uint32_t freq_a = hist_nz_N[cl_a][bin];
          uint32_t freq_b = hist_nz_N[cl_b][bin];
          if (freq_a != 0 && freq_b != 0) {
            delta += d.nz_entropy(freq_a + freq_b) - d.nz_entropy(freq_a) -
                     d.nz_entropy(freq_b);
          }
        }
        for (size_t bin = 0; bin < hist_nz_h[cl_a].size(); ++bin) {
          uint32_t freq_a = hist_nz_h[cl_a][bin];
          uint32_t freq_b = hist_nz_h[cl_b][bin];
          if (freq_a != 0 && freq_b != 0) {
            delta -= d.nz_entropy(freq_a + freq_b) - d.nz_entropy(freq_a) -
                     d.nz_entropy(freq_b);
          }
        }
        return delta;
      };

      if (active_clusters > 1) {
        deltas.assign(total_ctxs * total_ctxs, 0);
        JXL_RETURN_IF_ERROR(RunOnPool(
            pool, 0, active_clusters - 1, ThreadPool::NoInit,
            [&](uint32_t i, size_t) -> Status {
              uint32_t id_i = active[i];
              for (uint32_t j = i + 1; j < active_clusters; j++) {
                get_delta(id_i, active[j]) = merge_delta(id_i, active[j]);
              }
              return true;
            },
            "MergeDelta"));
      }

      auto find_best_merge = [&](size_t& best_i, size_t& best_j,
                                 int64_t& best_delta) -> Status {
        best_delta = std::numeric_limits<int64_t>::max();
        best_i = 0;
        best_j = 1;
        std::mutex best_mtx;

        JXL_RETURN_IF_ERROR(RunOnPool(
            pool, 0, active_clusters - 1, ThreadPool::NoInit,
            [&](uint32_t i, size_t) -> Status {
              uint32_t id_i = active[i];
              size_t local_best_j = i + 1;
              int64_t local_best_diff = get_delta(id_i, active[local_best_j]);
              for (size_t j = i + 2; j < active_clusters; j++) {
                int64_t diff = get_delta(id_i, active[j]);
                if (diff < local_best_diff) {
                  local_best_diff = diff;
                  local_best_j = j;
                }
              }
              std::lock_guard<std::mutex> lock(best_mtx);
              if (local_best_diff < best_delta) {
                best_delta = local_best_diff;
                best_i = i;
                best_j = local_best_j;
              }
              return true;
            },
            "FindBestMerge"));
        return true;
      };

      auto apply_merge =
          [&](size_t best_i, size_t best_j,
              int64_t best_delta) -> std::pair<uint32_t, uint32_t> {
        uint32_t a_id = active[best_i];
        uint32_t b_id = active[best_j];
        E[a_id] += E[b_id] + best_delta;
        AddDenseHistogram(hist_N[a_id], hist_N[b_id]);
        hist_h[a_id].AddFrom(hist_h[b_id]);
        AddDenseHistogram(hist_nz_N[a_id], hist_nz_N[b_id]);
        AddDenseHistogram(hist_nz_h[a_id], hist_nz_h[b_id]);
        hist_N[b_id].fill(0);
        hist_h[b_id].Clear();
        hist_nz_N[b_id].fill(0);
        hist_nz_h[b_id].fill(0);
        parent[b_id] = a_id;
        std::swap(active[best_j], active.back());
        active.pop_back();
        --active_clusters;
        return {a_id, b_id};
      };

      auto update_distances = [&](uint32_t a_id) -> Status {
        if (active_clusters <= 1) return true;
        JXL_RETURN_IF_ERROR(RunOnPool(
            pool, 0, active_clusters, ThreadPool::NoInit,
            [&](uint32_t k, size_t) -> Status {
              if (active[k] != a_id)
                get_delta(a_id, active[k]) = merge_delta(a_id, active[k]);
              return true;
            },
            "UpdateDist"));
        return true;
      };

      int64_t current_entropy_cost = initial_cost;
      while (active_clusters > num_clusters) {
        size_t best_i = 0;
        size_t best_j = 1;
        int64_t best_delta = 0;
        JXL_RETURN_IF_ERROR(find_best_merge(best_i, best_j, best_delta));
        std::pair<uint32_t, uint32_t> merged =
            apply_merge(best_i, best_j, best_delta);
        uint32_t a_id = merged.first;
        current_entropy_cost += best_delta;
        JXL_RETURN_IF_ERROR(update_distances(a_id));
      }

      if (overhead_aware_tail) {
        // On small images max number of clusters can be an overkkill and
        // gains will be nullified by the signalling cost. To avoid this problem
        // we continue greedy agglomeration from current state while
        // entropy + signalling overhead drops.
        int64_t best_total_cost = std::numeric_limits<int64_t>::max();
        JXL_ASSIGN_OR_RETURN(int64_t initial_overhead,
                             ComputeSignallingOverhead(d));
        best_total_cost = current_entropy_cost + initial_overhead;

        while (active_clusters > 1) {
          size_t best_i = 0;
          size_t best_j = 1;
          int64_t best_delta = 0;
          JXL_RETURN_IF_ERROR(find_best_merge(best_i, best_j, best_delta));

          uint32_t a_id = active[best_i];
          uint32_t b_id = active[best_j];
          int64_t old_Ea = E[a_id];
          uint32_t old_parent_b = parent[b_id];
          rollback.active = active;
          uint32_t old_active_clusters = active_clusters;
          rollback.hist_N_a = hist_N[a_id];
          rollback.hist_h_a = hist_h[a_id];
          rollback.hist_nz_N_a = hist_nz_N[a_id];
          rollback.hist_nz_h_a = hist_nz_h[a_id];
          rollback.hist_N_b = hist_N[b_id];
          rollback.hist_nz_N_b = hist_nz_N[b_id];
          rollback.hist_nz_h_b = hist_nz_h[b_id];
          hist_N[b_id].fill(0);
          hist_nz_N[b_id].fill(0);
          hist_nz_h[b_id].fill(0);
          rollback.hist_h_b = CompactHistogram();
          rollback.hist_h_b.swap(hist_h[b_id]);

          E[a_id] += E[b_id] + best_delta;
          AddDenseHistogram(hist_N[a_id], rollback.hist_N_b);
          hist_h[a_id].AddFrom(rollback.hist_h_b);
          AddDenseHistogram(hist_nz_N[a_id], rollback.hist_nz_N_b);
          AddDenseHistogram(hist_nz_h[a_id], rollback.hist_nz_h_b);
          parent[b_id] = a_id;
          std::swap(active[best_j], active.back());
          active.pop_back();
          --active_clusters;
          current_entropy_cost += best_delta;
          JXL_RETURN_IF_ERROR(update_distances(a_id));

          JXL_ASSIGN_OR_RETURN(int64_t overhead, ComputeSignallingOverhead(d));
          int64_t merged_total_cost = current_entropy_cost + overhead;
          if (merged_total_cost < best_total_cost) {
            best_total_cost = merged_total_cost;
            continue;
          }

          // Roll back the rejected merge and stop.
          active = rollback.active;
          active_clusters = old_active_clusters;
          E[a_id] = old_Ea;
          parent[b_id] = old_parent_b;
          hist_N[a_id] = rollback.hist_N_a;
          hist_h[a_id] = rollback.hist_h_a;
          hist_nz_N[a_id] = rollback.hist_nz_N_a;
          hist_nz_h[a_id] = rollback.hist_nz_h_a;
          hist_N[b_id] = rollback.hist_N_b;
          hist_nz_N[b_id] = rollback.hist_nz_N_b;
          hist_nz_h[b_id] = rollback.hist_nz_h_b;
          hist_h[b_id].swap(rollback.hist_h_b);
          current_entropy_cost -= best_delta;
          break;
        }
      }

      clustered_cost = 0;
      for (unsigned int k : active) clustered_cost += E[k];

      ctx_num = active_clusters;
      std::function<uint32_t(uint32_t)> find_cluster =
          [&parent, &find_cluster](uint32_t ctx) -> uint32_t {
        return parent[ctx] == ctx ? ctx
                                  : parent[ctx] = find_cluster(parent[ctx]);
      };
      for (uint32_t i : initial_active) {
        ctx_map[i] = std::find(active.begin(), active.end(), find_cluster(i)) -
                     active.begin();
      }
      // Save cluster histograms
      CompactHistogramSet tmp(active_clusters, CompactHistogram(d.num_zdcai));
      uint32_t ind = 0;
      for (uint32_t i : active) tmp[ind++].swap(hist_h[i]);
      hist_h.swap(tmp);
      DenseNHistogramSet tmp_dense(active_clusters);
      ind = 0;
      for (uint32_t i : active) tmp_dense[ind++].swap(hist_N[i]);
      hist_N.swap(tmp_dense);
      DenseNZHistogramSet tmp_nz_h(active_clusters);
      ind = 0;
      for (uint32_t i : active) tmp_nz_h[ind++].swap(hist_nz_h[i]);
      hist_nz_h.swap(tmp_nz_h);
      DenseNZPredHistogramSet tmp_nz_N(active_clusters);
      ind = 0;
      for (uint32_t i : active) tmp_nz_N[ind++].swap(hist_nz_N[i]);
      hist_nz_N.swap(tmp_nz_N);

      return true;
    }

    // Compute the signalling overhead (header cost) for the clustered
    // histograms. This estimates the bit cost of encoding the histogram
    // headers in the bitstream, which is not included in the entropy cost
    // computed by `TotalCost()`.
    //
    // Each ANS histogram header costs approximately:
    // - 2 bits for markers (not-small-tree, not-flat)
    // - 1-7 bits for shift parameter
    // - 1-21 bits for alphabet size (varint8)
    // - 3-7 bits per symbol for bit widths (static Huffman)
    // - 0-3 bits per symbol for precision bits (if shift > 0)
    //
    // For typical histograms with alphabet size 20-60, this is ~100-300 bits.
    // We use `ANSPopulationCost() - ShannonEntropy` to estimate the overhead.
    StatusOr<int64_t> ComputeSignallingOverhead(const JPEGOptData& d) const {
      int64_t overhead = 0;

      // Default HybridUintConfig for AC coefficients: (split_exponent=4,
      // msb_in_token=2, lsb_in_token=0)
      const HybridUintConfig hybrid_uint_config(4, 2, 0);

      // Process `hist_h`: split by `zdc` and compute overhead per histogram
      for (const auto& cluster : hist_h) {
        if (cluster.empty()) continue;

        // Group symbols by `zdc` value, applying `HybridUintConfig`
        std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>
            by_zdc;
        for (uint32_t id : cluster.used_ids) {
          uint32_t symbol = d.dense_to_zdcai[id];
          uint32_t zdc = symbol >> 11;
          uint32_t ai = symbol & 0x7FFu;
          uint32_t token, nbits, bits;
          hybrid_uint_config.Encode(ai, &token, &nbits, &bits);
          by_zdc[zdc][token] += cluster.Get(id);
        }

        // Compute overhead for each `zdc` histogram
        for (const auto& zdc_pair : by_zdc) {
          const auto& token_hist = zdc_pair.second;
          if (token_hist.empty()) continue;

          uint32_t max_token = 0;
          size_t total = 0;
          for (const auto& kv : token_hist) {
            max_token = std::max(max_token, kv.first);
            total += kv.second;
          }
          if (total == 0) continue;

          JXL_ENSURE(max_token < ANS_MAX_ALPHABET_SIZE);
          size_t alphabet_size = max_token + 1;
          if (alphabet_size == 0) continue;

          Histogram h(alphabet_size);
          for (const auto& kv : token_hist) {
            h.counts[kv.first] = static_cast<ANSHistBin>(kv.second);
          }
          h.total_count = total;

          // `ANSPopulationCost()` includes header + data cost
          JXL_ASSIGN_OR_RETURN(float ans_cost, h.ANSPopulationCost());
          // Shannon entropy is the ideal data cost
          float shannon = h.ShannonEntropy();
          // Overhead = total cost - data cost
          float header_cost = ans_cost - shannon;
          if (header_cost > 0) {
            overhead += static_cast<int64_t>(header_cost * kFScale);
          }
        }
      }

      // Process `hist_nz_h`: split by predicted bucket `pb`
      for (const auto& cluster : hist_nz_h) {
        if (DenseHistogramEmpty(cluster)) continue;
        for (uint32_t pb = 0; pb < kJPEGNZBucketCount; ++pb) {
          uint32_t max_nz = 0;
          size_t total = 0;
          const uint32_t base = pb * kJPEGNZCountLimit;
          for (uint32_t nz_count = 0; nz_count < kJPEGNZCountLimit;
               ++nz_count) {
            uint32_t freq = cluster[base + nz_count];
            if (freq == 0) continue;
            max_nz = nz_count;
            total += freq;
          }
          if (total == 0) continue;

          size_t alphabet_size = max_nz + 1;
          Histogram h(alphabet_size);
          for (uint32_t nz_count = 0; nz_count <= max_nz; ++nz_count) {
            h.counts[nz_count] =
                static_cast<ANSHistBin>(cluster[base + nz_count]);
          }
          h.total_count = total;

          JXL_ASSIGN_OR_RETURN(float ans_cost, h.ANSPopulationCost());
          float shannon = h.ShannonEntropy();
          float header_cost = ans_cost - shannon;
          if (header_cost > 0) {
            overhead += static_cast<int64_t>(header_cost * kFScale);
          }
        }
      }

      return overhead;
    }

    // Removes thresholds that are structurally inert: i.e. the `ctx_map`
    // assigns the same cluster on both sides. Rebuilds `ctx_map` to match the
    // pruned grid.
    ThresholdSet PruneDeadThresholds(const ThresholdSet& thresholds) {
      ThresholdSet T = thresholds;
      // Greyscale: the grid is Y axis only, `ctx_map` is flat per channel.
      if (T.TCb().empty() && T.TCr().empty()) {
        Thresholds new_thr;
        new_thr.reserve(T.TY().size());
        std::vector<uint32_t> old_from_new = {0};
        for (uint32_t t = 0; t < T.TY().size(); ++t) {
          if (ctx_map[t] != ctx_map[t + 1]) {
            old_from_new.push_back(t + 1);
            new_thr.push_back(T.TY()[t]);
          }
        }
        T.TY().swap(new_thr);
        const uint32_t new_n = static_cast<uint32_t>(T.TY().size() + 1);
        ContextMap new_ctx_map(kNumCh * new_n, 0);
        for (uint32_t x = 0; x < new_n; ++x) {
          new_ctx_map[x] = ctx_map[old_from_new[x]];
        }
        ctx_map.swap(new_ctx_map);
        return T;
      }

      const uint32_t sizes[3] = {static_cast<uint32_t>(T.TY().size() + 1),
                                 static_cast<uint32_t>(T.TCb().size() + 1),
                                 static_cast<uint32_t>(T.TCr().size() + 1)};
      const uint32_t num_cells_init = sizes[0] * sizes[1] * sizes[2];
      // Stride when stepping from cell `t` to `t+1` along each axis.
      // Cell layout: `index = (b[1]*sizes[2] + b[2])*sizes[0] + b[0]`
      // where b[0]=Y, b[1]=Cb, b[2]=Cr (Y is innermost/stride-1, matching
      // the `dc_idx` formula in `compressed_dc.cc` and the `UpdateMaps`
      // bucket maps).
      const uint32_t axis_stride[3] = {1, sizes[0] * sizes[2], sizes[0]};

      std::array<std::vector<uint32_t>, kNumCh> old_from_new = {
          {{0}, {0}, {0}}};
      for (uint32_t axis = 0; axis < kNumCh; ++axis) {
        Thresholds& thr = T.T[axis];
        const uint32_t ax1 = (axis + 1) % 3;
        const uint32_t ax2 = (axis + 2) % 3;
        Thresholds new_thr;
        new_thr.reserve(thr.size());
        std::vector<uint32_t>& ofn = old_from_new[axis];
        auto add_active = [&](uint32_t t) {
          uint32_t b[3] = {};
          b[axis] = t;
          for (uint32_t c = 0; c < kNumCh; ++c) {
            const uint32_t c_base = c * num_cells_init;
            for (uint32_t k1 = 0; k1 < sizes[ax1]; ++k1) {
              b[ax1] = k1;
              for (uint32_t k2 = 0; k2 < sizes[ax2]; ++k2) {
                b[ax2] = k2;
                const uint32_t gl = (b[1] * sizes[2] + b[2]) * sizes[0] + b[0];
                if (ctx_map[c_base + gl] !=
                    ctx_map[c_base + gl + axis_stride[axis]]) {
                  ofn.push_back(t + 1);
                  new_thr.push_back(thr[t]);
                  return;
                }
              }
            }
          }
        };

        for (uint32_t t = 0; t < thr.size(); ++t) {
          add_active(t);
        }
        thr.swap(new_thr);
      }

      const uint32_t new_sizes[3] = {static_cast<uint32_t>(T.TY().size() + 1),
                                     static_cast<uint32_t>(T.TCb().size() + 1),
                                     static_cast<uint32_t>(T.TCr().size() + 1)};
      const uint32_t new_num_cells = new_sizes[0] * new_sizes[1] * new_sizes[2];
      ContextMap new_ctx_map(kNumCh * new_num_cells, 0);
      for (uint32_t c = 0; c < kNumCh; ++c) {
        const uint32_t old_base = c * num_cells_init;
        const uint32_t new_base = c * new_num_cells;
        for (uint32_t Cb = 0; Cb < new_sizes[1]; ++Cb) {
          for (uint32_t Cr = 0; Cr < new_sizes[2]; ++Cr) {
            for (uint32_t Y = 0; Y < new_sizes[0]; ++Y) {
              const uint32_t g_old =
                  (old_from_new[1][Cb] * sizes[2] + old_from_new[2][Cr]) *
                      sizes[0] +
                  old_from_new[0][Y];
              const uint32_t g_new =
                  (Cb * new_sizes[2] + Cr) * new_sizes[0] + Y;
              new_ctx_map[new_base + g_new] = ctx_map[old_base + g_old];
            }
          }
        }
      }
      ctx_map.swap(new_ctx_map);
      return T;
    }
  };

  void PopulateClusterHistograms(const JPEGOptData& d, uint32_t n0,
                                 uint32_t num_cells, Clustering* cl) {
    StreamSweep(
        []() {}, []() {},
        [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
            uint32_t bin_state) {
          uint32_t c = (bin_state >> 20) & 0x3u;
          uint32_t cell = static_cast<uint32_t>(
              (ax1_row[dc1_idx] + ax2_col[dc2_idx]) * n0 + ax0_to_k[dc0_idx]);
          uint32_t ctx_id = c * num_cells + cell;
          cl->hist_h[ctx_id].Add(d.CompactHBin(bin_state & 0xFFFFFu), run);
          uint32_t zdc = (bin_state >> 11) & 0x1FFu;
          cl->hist_N[ctx_id][zdc] += run;
        });

    if (d.channels == 1) {
      for (uint32_t b = 0; b < d.num_blocks[0]; ++b) {
        uint32_t ctx_id = ax0_to_k[d.block_dc_idx[0][b]];
        uint32_t nz_count = d.block_nonzeros[0][b];
        uint32_t pb = d.nz_pred_bucket[0][b];
        ++cl->hist_nz_h[ctx_id][NZHistogramIndex(pb, nz_count)];
        ++cl->hist_nz_N[ctx_id][pb];
      }
      return;
    }

    for (uint32_t c = 0; c < kNumCh; ++c) {
      for (uint32_t by = 0; by < d.block_grid_h[c]; ++by) {
        for (uint32_t bx = 0; bx < d.block_grid_w[c]; ++bx) {
          uint32_t x = bx * d.ss_x[c];
          uint32_t y = by * d.ss_y[c];
          uint32_t x0 = x / d.ss_x[0];
          uint32_t y0 = y / d.ss_y[0];
          uint32_t b0 = y0 * d.block_grid_w[0] + x0;
          uint32_t x1 = x / d.ss_x[1];
          uint32_t y1 = y / d.ss_y[1];
          uint32_t b1 = y1 * d.block_grid_w[1] + x1;
          uint32_t x2 = x / d.ss_x[2];
          uint32_t y2 = y / d.ss_y[2];
          uint32_t b2 = y2 * d.block_grid_w[2] + x2;
          uint32_t cell = (ax1_row[d.block_dc_idx[1][b1]] +
                           ax2_col[d.block_dc_idx[2][b2]]) *
                              n0 +
                          ax0_to_k[d.block_dc_idx[0][b0]];
          uint32_t ctx_id = c * num_cells + cell;

          uint32_t b = by * d.block_grid_w[c] + bx;
          uint32_t nz_count = d.block_nonzeros[c][b];
          uint32_t pb = d.nz_pred_bucket[c][b];
          ++cl->hist_nz_h[ctx_id][NZHistogramIndex(pb, nz_count)];
          ++cl->hist_nz_N[ctx_id][pb];
        }
      }
    }
  }

  StatusOr<Clustering> ClusterContexts(const ThresholdSet& thresholds,
                                       uint32_t num_clusters = kMaxClusters,
                                       bool overhead_aware_tail = true,
                                       ThreadPool* pool = nullptr) {
    const JPEGOptData& d = data();
    uint32_t n0 = static_cast<uint32_t>(thresholds.TY().size() + 1);
    uint32_t n1 = static_cast<uint32_t>(thresholds.TCb().size() + 1);
    uint32_t n2 = static_cast<uint32_t>(thresholds.TCr().size() + 1);
    uint32_t num_cells = n0 * n1 * n2;
    uint32_t total_ctxs = kNumCh * num_cells;
    UpdateMaps(thresholds);

    Clustering cl;
    cl.hist_h.assign(total_ctxs, CompactHistogram(d.num_zdcai));
    cl.hist_N.resize(total_ctxs);
    cl.hist_nz_h.resize(total_ctxs);
    cl.hist_nz_N.resize(total_ctxs);
    PopulateClusterHistograms(d, n0, num_cells, &cl);

    JXL_RETURN_IF_ERROR(cl.AgglomerativeClusterCore(d, total_ctxs, num_clusters,
                                                    overhead_aware_tail, pool));
    return cl;
  }

  struct RefineResult {
    ThresholdSet thresholds;
    int64_t cost;     // total cost (AC + nz)
    int64_t nz_cost;  // nz cost portion only
  };

  RefineResult RefineClustered(const ThresholdSet& thresholds,
                               Clustering& clustering, uint32_t max_iters = 5,
                               ptrdiff_t search_radius = 2048) {
    const JPEGOptData& d = data();
    CompactHistogramSet& scratch_hist_h = scratch_hist_h_;
    DenseNHistogramSet& scratch_hist_N = scratch_hist_N_;
    DenseNZPredHistogramSet& scratch_hist_nz_N = scratch_hist_nz_N_;
    DenseNZHistogramSet& scratch_hist_nz_h = scratch_hist_nz_h_;

    ThresholdSet cur_T = clustering.PruneDeadThresholds(thresholds);

    std::array<std::vector<uint8_t>, kNumCh>& local_cluster_map =
        local_cluster_map_;
    uint32_t size_Y = static_cast<uint32_t>(cur_T.TY().size() + 1);
    uint32_t size_Cb = static_cast<uint32_t>(cur_T.TCb().size() + 1);
    uint32_t size_Cr = static_cast<uint32_t>(cur_T.TCr().size() + 1);
    uint32_t num_cells = size_Y * size_Cb * size_Cr;
    for (uint32_t axis = 0; axis < d.channels; ++axis) {
      uint32_t ax1 = (axis + 1) % 3;
      uint32_t ax2 = (axis + 2) % 3;
      uint32_t na = static_cast<uint32_t>(cur_T.T[axis].size() + 1);
      uint32_t n1 = static_cast<uint32_t>(cur_T.T[ax1].size() + 1);
      uint32_t n2 = static_cast<uint32_t>(cur_T.T[ax2].size() + 1);
      local_cluster_map[axis].assign(kNumCh * num_cells, 0);
      for (uint32_t c = 0; c < kNumCh; ++c) {
        for (uint32_t k0 = 0; k0 < na; ++k0) {
          for (uint32_t k1 = 0; k1 < n1; ++k1) {
            for (uint32_t k2 = 0; k2 < n2; ++k2) {
              uint32_t bkt[3] = {};
              bkt[axis] = k0;
              bkt[ax1] = k1;
              bkt[ax2] = k2;
              uint32_t global_cell =
                  (bkt[1] * size_Cr + bkt[2]) * size_Y + bkt[0];
              uint32_t local_cell = (k1 * n2 + k2) * na + k0;
              local_cluster_map[axis][c * num_cells + local_cell] =
                  clustering.ctx_map[c * num_cells + global_cell];
            }
          }
        }
      }
    }

    // Coordinate descent loop repeatedly jittering all 3 color axes
    int64_t base_cost = clustering.clustered_cost;
    bool changed = true;
    for (uint32_t iter = 0; iter < max_iters && changed; iter++) {
      changed = false;
      auto optimize_axis = [&](uint32_t axis) {
        Thresholds& thr = cur_T.T[axis];
        if (thr.empty()) return;
        const auto& DC_axis = d.dc_vals[axis];
        uint32_t ax1 = (axis + 1) % 3;
        uint32_t ax2 = (axis + 2) % 3;
        UpdateMaps(axis, thr, cur_T.T[ax1], cur_T.T[ax2]);
        uint32_t na = static_cast<uint32_t>(thr.size() + 1);
        const auto& sorted_blocks = d.dc_sorted_blocks[axis];
        const auto& block_off = d.dc_block_offsets[axis];

        for (uint32_t thr_ind = 0; thr_ind < thr.size(); thr_ind++) {
          ptrdiff_t cur_idx = Bkt(thr[thr_ind] - 1, DC_axis);
          ptrdiff_t lo_edge =
              std::max({cur_idx - search_radius,
                        (thr_ind == 0) ? static_cast<ptrdiff_t>(0)
                                       : Bkt(thr[thr_ind - 1] - 1, DC_axis)});
          ptrdiff_t hi_edge =
              std::min({cur_idx + search_radius,
                        (thr_ind == thr.size() - 1)
                            ? static_cast<ptrdiff_t>(DC_axis.size())
                            : Bkt(thr[thr_ind + 1] - 1, DC_axis)});

          ptrdiff_t best_idx = cur_idx;
          int64_t best_cost = base_cost;

          // Incremental cost update for moving one item from
          // `old_cluster` to `new_cluster`.
          auto apply_delta_dense_nz =
              [&](uint32_t old_cl, uint32_t new_cl, DenseNZHistogramSet& hist_h,
                  DenseNZPredHistogramSet& hist_N, uint32_t h_bin,
                  uint32_t N_bin) -> int64_t {
            int64_t delta = 0;

            auto dec_bin_h = [&](DenseNZHistogramSet& hist, uint32_t cl,
                                 uint32_t bin, int64_t sign) {
              uint32_t& freq = hist[cl][bin];
              JXL_DASSERT(freq > 0);
              if (freq == 0) return;
              uint32_t old_freq = freq;
              uint32_t new_freq = old_freq - 1;
              delta += sign * (d.nz_entropy(new_freq) - d.nz_entropy(old_freq));
              freq = new_freq;
            };
            auto inc_bin_h = [&](DenseNZHistogramSet& hist, uint32_t cl,
                                 uint32_t bin, int64_t sign) {
              uint32_t& freq = hist[cl][bin];
              uint32_t old_freq = freq;
              uint32_t new_freq = old_freq + 1;
              delta += sign * (d.nz_entropy(new_freq) - d.nz_entropy(old_freq));
              freq = new_freq;
            };
            auto dec_bin_N = [&](DenseNZPredHistogramSet& hist, uint32_t cl,
                                 uint32_t bin, int64_t sign) {
              uint32_t& freq = hist[cl][bin];
              JXL_DASSERT(freq > 0);
              if (freq == 0) return;
              uint32_t old_freq = freq;
              uint32_t new_freq = old_freq - 1;
              delta += sign * (d.nz_entropy(new_freq) - d.nz_entropy(old_freq));
              freq = new_freq;
            };
            auto inc_bin_N = [&](DenseNZPredHistogramSet& hist, uint32_t cl,
                                 uint32_t bin, int64_t sign) {
              uint32_t& freq = hist[cl][bin];
              uint32_t old_freq = freq;
              uint32_t new_freq = old_freq + 1;
              delta += sign * (d.nz_entropy(new_freq) - d.nz_entropy(old_freq));
              freq = new_freq;
            };
            // h-term: negative contribution to cost.
            dec_bin_h(hist_h, old_cl, h_bin, -1);
            inc_bin_h(hist_h, new_cl, h_bin, -1);

            // N-term: positive contribution to cost.
            dec_bin_N(hist_N, old_cl, N_bin, +1);
            inc_bin_N(hist_N, new_cl, N_bin, +1);

            return delta;
          };
          auto apply_delta_dense_h =
              [&](uint32_t old_cl, uint32_t new_cl, CompactHistogramSet& hist_h,
                  DenseNHistogramSet& hist_N, uint32_t h_bin,
                  uint32_t N_bin) -> int64_t {
            int64_t delta = 0;

            auto dec_bin = [&](DenseNHistogramSet& hist, uint32_t cl,
                               uint32_t bin, int64_t sign) {
              uint32_t& freq = hist[cl][bin];
              JXL_DASSERT(freq > 0);
              if (freq == 0) return;
              uint32_t old_freq = freq;
              uint32_t new_freq = old_freq - 1;
              delta += sign * (d.nz_entropy(new_freq) - d.nz_entropy(old_freq));
              freq = new_freq;
            };
            auto inc_bin = [&](DenseNHistogramSet& hist, uint32_t cl,
                               uint32_t bin, int64_t sign) {
              uint32_t& freq = hist[cl][bin];
              uint32_t old_freq = freq;
              uint32_t new_freq = old_freq + 1;
              delta += sign * (d.nz_entropy(new_freq) - d.nz_entropy(old_freq));
              freq = new_freq;
            };
            auto dec_bin_h = [&](CompactHistogramSet& hist, uint32_t cl,
                                 uint32_t bin, int64_t sign) {
              CompactHistogram& m = hist[cl];
              uint32_t old_freq = m.Get(bin);
              JXL_DASSERT(old_freq > 0);
              if (old_freq == 0) return;
              uint32_t new_freq = old_freq - 1;
              delta += sign * (d.nz_entropy(new_freq) - d.nz_entropy(old_freq));
              m.Subtract(bin, 1);
            };
            auto inc_bin_h = [&](CompactHistogramSet& hist, uint32_t cl,
                                 uint32_t bin, int64_t sign) {
              CompactHistogram& m = hist[cl];
              uint32_t old_freq = m.Get(bin);
              uint32_t new_freq = old_freq + 1;
              delta += sign * (d.nz_entropy(new_freq) - d.nz_entropy(old_freq));
              m.Add(bin, 1);
            };

            // h-term: negative contribution to cost.
            dec_bin_h(hist_h, old_cl, h_bin, -1);
            inc_bin_h(hist_h, new_cl, h_bin, -1);

            // N-term: positive contribution to cost.
            dec_bin(hist_N, old_cl, N_bin, +1);
            inc_bin(hist_N, new_cl, N_bin, +1);

            return delta;
          };

          auto apply_slice = [&](ptrdiff_t slice, bool upward) -> int64_t {
            int64_t cost_change = 0;
            auto div_ceil_u32 = [](uint32_t num, uint32_t den) -> uint32_t {
              return (num + den - 1) / den;
            };
            // Blocks with `DC[axis] = slice`
            uint32_t blk_lo = block_off[static_cast<size_t>(slice)];
            uint32_t blk_hi = block_off[static_cast<size_t>(slice + 1)];
            for (uint32_t bi = blk_lo; bi < blk_hi; ++bi) {
              uint32_t b0_xy = sorted_blocks[bi];
              uint32_t b0_y = b0_xy >> 16;
              uint32_t b0_x = b0_xy & 0xFFFF;
              uint32_t b0 = b0_y * d.block_grid_w[axis] + b0_x;

              if (d.channels == kNumCh) {
                // Top-left mapped reference blocks for the moved axis block.
                uint32_t b1_ref_y = b0_y * d.ss_y[axis] / d.ss_y[ax1];
                uint32_t b1_ref_x = b0_x * d.ss_x[axis] / d.ss_x[ax1];
                uint32_t b2_ref_y = b0_y * d.ss_y[axis] / d.ss_y[ax2];
                uint32_t b2_ref_x = b0_x * d.ss_x[axis] / d.ss_x[ax2];

                // Invert floor mapping used by context construction:
                //   `b_axis = floor(b_comp * ss_comp / ss_axis)`.
                // The set of component blocks affected by moved axis block `b0`
                // is `[ceil(b0 * ss_axis / ss_comp),
                //     ceil((b0 + 1) * ss_axis / ss_comp))`.
                uint32_t b1_y = div_ceil_u32(b0_y * d.ss_y[axis], d.ss_y[ax1]);
                uint32_t b1_x = div_ceil_u32(b0_x * d.ss_x[axis], d.ss_x[ax1]);
                uint32_t b1e_y =
                    div_ceil_u32((b0_y + 1) * d.ss_y[axis], d.ss_y[ax1]);
                uint32_t b1e_x =
                    div_ceil_u32((b0_x + 1) * d.ss_x[axis], d.ss_x[ax1]);

                uint32_t b2_y = div_ceil_u32(b0_y * d.ss_y[axis], d.ss_y[ax2]);
                uint32_t b2_x = div_ceil_u32(b0_x * d.ss_x[axis], d.ss_x[ax2]);
                uint32_t b2e_y =
                    div_ceil_u32((b0_y + 1) * d.ss_y[axis], d.ss_y[ax2]);
                uint32_t b2e_x =
                    div_ceil_u32((b0_x + 1) * d.ss_x[axis], d.ss_x[ax2]);

                // `axis` component block
                uint32_t b1 = b1_ref_y * d.block_grid_w[ax1] + b1_ref_x;
                uint32_t b2 = b2_ref_y * d.block_grid_w[ax2] + b2_ref_x;
                uint32_t dc_ax1_ind = d.block_dc_idx[ax1][b1];
                uint32_t dc_ax2_ind = d.block_dc_idx[ax2][b2];

                uint32_t ci = ax1_row[dc_ax1_ind] + ax2_col[dc_ax2_ind];
                uint32_t ci_base = axis * num_cells + ci * na;
                uint32_t old_cl =
                    upward ? local_cluster_map[axis][ci_base + thr_ind + 1]
                           : local_cluster_map[axis][ci_base + thr_ind];
                uint32_t new_cl =
                    upward ? local_cluster_map[axis][ci_base + thr_ind]
                           : local_cluster_map[axis][ci_base + thr_ind + 1];
                if (old_cl != new_cl) {
                  uint32_t N_bin = d.nz_pred_bucket[axis][b0];
                  uint32_t h_bin =
                      NZHistogramIndex(N_bin, d.block_nonzeros[axis][b0]);
                  cost_change +=
                      apply_delta_dense_nz(old_cl, new_cl, scratch_hist_nz_h,
                                           scratch_hist_nz_N, h_bin, N_bin);
                  for (uint32_t pi = d.block_offsets[axis][b0];
                       pi < d.block_offsets[axis][b0 + 1]; ++pi) {
                    uint32_t bin = d.block_bins[axis][pi] & 0xFFFFF;
                    uint32_t zdc = bin >> 11;
                    cost_change += apply_delta_dense_h(
                        old_cl, new_cl, scratch_hist_h, scratch_hist_N,
                        d.CompactHBin(bin), zdc);
                  }
                }

                // `ax1` component blocks
                for (uint32_t y1 = b1_y; y1 < b1e_y; ++y1) {
                  for (uint32_t x1 = b1_x; x1 < b1e_x; ++x1) {
                    b1 = y1 * d.block_grid_w[ax1] + x1;
                    dc_ax1_ind = d.block_dc_idx[ax1][b1];
                    uint32_t y2 = y1 * d.ss_y[ax1] / d.ss_y[ax2];
                    uint32_t x2 = x1 * d.ss_x[ax1] / d.ss_x[ax2];
                    b2 = y2 * d.block_grid_w[ax2] + x2;
                    dc_ax2_ind = d.block_dc_idx[ax2][b2];

                    ci = ax1_row[dc_ax1_ind] + ax2_col[dc_ax2_ind];
                    ci_base = ax1 * num_cells + ci * na;
                    old_cl =
                        upward ? local_cluster_map[axis][ci_base + thr_ind + 1]
                               : local_cluster_map[axis][ci_base + thr_ind];
                    new_cl =
                        upward ? local_cluster_map[axis][ci_base + thr_ind]
                               : local_cluster_map[axis][ci_base + thr_ind + 1];
                    if (old_cl != new_cl) {
                      uint32_t N_bin = d.nz_pred_bucket[ax1][b1];
                      uint32_t h_bin =
                          NZHistogramIndex(N_bin, d.block_nonzeros[ax1][b1]);
                      cost_change += apply_delta_dense_nz(
                          old_cl, new_cl, scratch_hist_nz_h, scratch_hist_nz_N,
                          h_bin, N_bin);
                      for (uint32_t pi = d.block_offsets[ax1][b1];
                           pi < d.block_offsets[ax1][b1 + 1]; ++pi) {
                        uint32_t bin = d.block_bins[ax1][pi] & 0xFFFFF;
                        uint32_t zdc = bin >> 11;
                        cost_change += apply_delta_dense_h(
                            old_cl, new_cl, scratch_hist_h, scratch_hist_N,
                            d.CompactHBin(bin), zdc);
                      }
                    }
                  }
                }
                // `ax2` component blocks
                for (uint32_t y2 = b2_y; y2 < b2e_y; ++y2) {
                  for (uint32_t x2 = b2_x; x2 < b2e_x; ++x2) {
                    b2 = y2 * d.block_grid_w[ax2] + x2;
                    dc_ax2_ind = d.block_dc_idx[ax2][b2];
                    uint32_t y1 = y2 * d.ss_y[ax2] / d.ss_y[ax1];
                    uint32_t x1 = x2 * d.ss_x[ax2] / d.ss_x[ax1];
                    b1 = y1 * d.block_grid_w[ax1] + x1;
                    dc_ax1_ind = d.block_dc_idx[ax1][b1];

                    ci = ax1_row[dc_ax1_ind] + ax2_col[dc_ax2_ind];
                    ci_base = ax2 * num_cells + ci * na;
                    old_cl =
                        upward ? local_cluster_map[axis][ci_base + thr_ind + 1]
                               : local_cluster_map[axis][ci_base + thr_ind];
                    new_cl =
                        upward ? local_cluster_map[axis][ci_base + thr_ind]
                               : local_cluster_map[axis][ci_base + thr_ind + 1];
                    if (old_cl != new_cl) {
                      uint32_t N_bin = d.nz_pred_bucket[ax2][b2];
                      uint32_t h_bin =
                          NZHistogramIndex(N_bin, d.block_nonzeros[ax2][b2]);
                      cost_change += apply_delta_dense_nz(
                          old_cl, new_cl, scratch_hist_nz_h, scratch_hist_nz_N,
                          h_bin, N_bin);
                      for (uint32_t pi = d.block_offsets[ax2][b2];
                           pi < d.block_offsets[ax2][b2 + 1]; ++pi) {
                        uint32_t bin = d.block_bins[ax2][pi] & 0xFFFFF;
                        uint32_t zdc = bin >> 11;
                        cost_change += apply_delta_dense_h(
                            old_cl, new_cl, scratch_hist_h, scratch_hist_N,
                            d.CompactHBin(bin), zdc);
                      }
                    }
                  }
                }
              } else {
                uint32_t old_cl = upward ? local_cluster_map[0][thr_ind + 1]
                                         : local_cluster_map[0][thr_ind];
                uint32_t new_cl = upward ? local_cluster_map[0][thr_ind]
                                         : local_cluster_map[0][thr_ind + 1];
                if (old_cl != new_cl) {
                  uint32_t N_bin = d.nz_pred_bucket[0][b0];
                  uint32_t h_bin =
                      NZHistogramIndex(N_bin, d.block_nonzeros[0][b0]);
                  cost_change +=
                      apply_delta_dense_nz(old_cl, new_cl, scratch_hist_nz_h,
                                           scratch_hist_nz_N, h_bin, N_bin);
                  for (uint32_t pi = d.block_offsets[0][b0];
                       pi < d.block_offsets[0][b0 + 1]; ++pi) {
                    uint32_t bin = d.block_bins[0][pi] & 0xFFFFF;
                    uint32_t zdc = bin >> 11;
                    cost_change += apply_delta_dense_h(
                        old_cl, new_cl, scratch_hist_h, scratch_hist_N,
                        d.CompactHBin(bin), zdc);
                  }
                }
              }
            }
            return cost_change;
          };

          // Scan down
          scratch_hist_h = clustering.hist_h;
          scratch_hist_N = clustering.hist_N;
          scratch_hist_nz_N = clustering.hist_nz_N;
          scratch_hist_nz_h = clustering.hist_nz_h;
          int64_t current_cost = base_cost;
          for (ptrdiff_t idx = cur_idx - 1; idx > lo_edge; --idx) {
            current_cost += apply_slice(idx, false);
            if (current_cost < best_cost) {
              best_cost = current_cost;
              best_idx = idx;
            }
          }

          // Scan up
          scratch_hist_h = clustering.hist_h;
          scratch_hist_N = clustering.hist_N;
          scratch_hist_nz_N = clustering.hist_nz_N;
          scratch_hist_nz_h = clustering.hist_nz_h;
          current_cost = base_cost;
          for (ptrdiff_t idx = cur_idx + 1; idx < hi_edge; ++idx) {
            current_cost += apply_slice(idx - 1, true);
            if (current_cost < best_cost) {
              best_cost = current_cost;
              best_idx = idx;
            }
          }

          // Permanently apply the winning shift
          if (best_idx != cur_idx) {
            scratch_hist_h = clustering.hist_h;
            scratch_hist_N = clustering.hist_N;
            scratch_hist_nz_N = clustering.hist_nz_N;
            scratch_hist_nz_h = clustering.hist_nz_h;
            if (best_idx < cur_idx) {
              for (ptrdiff_t idx = cur_idx - 1; idx >= best_idx; --idx)
                apply_slice(idx, false);
            } else {
              for (ptrdiff_t idx = cur_idx + 1; idx <= best_idx; ++idx)
                apply_slice(idx - 1, true);
            }
            thr[thr_ind] = DC_axis[static_cast<size_t>(best_idx)];
            base_cost = best_cost;
            std::swap(clustering.hist_h, scratch_hist_h);
            std::swap(clustering.hist_N, scratch_hist_N);
            std::swap(clustering.hist_nz_N, scratch_hist_nz_N);
            std::swap(clustering.hist_nz_h, scratch_hist_nz_h);
            changed = true;
          }
        }
      };

      optimize_axis(0);
      optimize_axis(1);
      optimize_axis(2);
    }

    // Recompute `nz_cost` for return.
    int64_t final_nz_cost = 0;
    for (const auto& cl_N : clustering.hist_nz_N)
      for (uint32_t freq : cl_N)
        if (freq != 0) final_nz_cost += d.nz_entropy(freq);
    for (const auto& cl_h : clustering.hist_nz_h)
      for (uint32_t freq : cl_h)
        if (freq != 0) final_nz_cost -= d.nz_entropy(freq);

    return {cur_T, base_cost, final_nz_cost};
  }
};

Status JPEGOptData::BuildFromJPEG(const jpeg::JPEGData& jpeg_data,
                                  ThreadPool* pool) {
  // Keep optimization axes in JPEG component order (Y, Cb, Cr).

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
    ss_y[c] = h_max / block_grid_h[c];
    ss_x[c] = w_max / block_grid_w[c];
  }
  for (uint32_t c = channels; c < kNumCh; ++c) {
    block_grid_w[c] = 0;
    block_grid_h[c] = 0;
    num_blocks[c] = 0;
    ss_y[c] = 0;
    ss_x[c] = 0;
  }

  {
    auto ac_cnt = jxl::make_unique<ACCounts>();
    JXL_RETURN_IF_ERROR(CountDCAC(jpeg_data, pool, *ac_cnt));

    JXL_RETURN_IF_ERROR(CollectACBins(jpeg_data, pool, *ac_cnt));
  }

  JXL_RETURN_IF_ERROR(FinalizeSpatialIndexing(pool));

  JXL_RETURN_IF_ERROR(GenerateRLEStream(pool));

  return true;
}

double bit_cost(int64_t cost) { return static_cast<double>(cost) / kFScale; }

}  // namespace

Status OptimizeJPEGContextMap(const jpeg::JPEGData& jpeg_data,
                              SpeedTier speed_tier, BlockCtxMap& ctx_map,
                              ThreadPool* pool) {
  auto opt_data = std::make_shared<JPEGOptData>();
  JXL_RETURN_IF_ERROR(opt_data->BuildFromJPEG(jpeg_data, pool));
  const JPEGCtxEffortParams effort =
      JPEGCtxEffortParams::FromSpeedTier(speed_tier);

  auto factorizations = opt_data->MaximalFactorizations();
  if (factorizations.empty()) return true;
  JXL_DEBUG_V(2, "Searching %i maximal factorizations\n",
              static_cast<int>(factorizations.size()));

  struct FactorizationCandidate {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    ThresholdSet init;
    int64_t rank_cost = std::numeric_limits<int64_t>::max();
  };

  std::vector<FactorizationCandidate> candidates;
  candidates.reserve(factorizations.size());
  for (const auto& factorization : factorizations) {
    FactorizationCandidate candidate;
    candidate.a = std::get<0>(factorization);
    candidate.b = std::get<1>(factorization);
    candidate.c = std::get<2>(factorization);
    candidate.init.T[0] = opt_data->InitThresh(0, candidate.a);
    candidate.init.T[1] = opt_data->InitThresh(1, candidate.b);
    candidate.init.T[2] = opt_data->InitThresh(2, candidate.c);
    candidates.push_back(std::move(candidate));
  }

  if (effort.keep_top_k != 0 && candidates.size() > effort.keep_top_k) {
    std::vector<PartitioningCtx> rank_ctx_pool;
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, static_cast<uint32_t>(candidates.size()),
        [&](size_t num_threads) -> Status {
          rank_ctx_pool.reserve(num_threads);
          for (size_t i = 0; i < num_threads; ++i) {
            rank_ctx_pool.emplace_back(opt_data);
          }
          return true;
        },
        [&](uint32_t idx, size_t thread_id) -> Status {
          FactorizationCandidate& candidate = candidates[idx];
          PartitioningCtx& ctx = rank_ctx_pool[thread_id];
          candidate.rank_cost =
              (effort.rank_iters == 0)
                  ? ctx.TotalCost(candidate.init)
                  : ctx.OptimizeThresholds(candidate.init, effort.rank_m_target,
                                           effort.rank_iters)
                        .first;
          return true;
        },
        "JpegCtxRank"));

    auto better_candidate = [](const FactorizationCandidate& lhs,
                               const FactorizationCandidate& rhs) {
      if (lhs.rank_cost != rhs.rank_cost) return lhs.rank_cost < rhs.rank_cost;
      const uint32_t lhs_cells = lhs.a * lhs.b * lhs.c;
      const uint32_t rhs_cells = rhs.a * rhs.b * rhs.c;
      if (lhs_cells != rhs_cells) return lhs_cells > rhs_cells;
      if (lhs.a != rhs.a) return lhs.a > rhs.a;
      return std::tie(lhs.a, lhs.b, lhs.c) < std::tie(rhs.a, rhs.b, rhs.c);
    };
    std::stable_sort(candidates.begin(), candidates.end(), better_candidate);
    candidates.resize(effort.keep_top_k);
  }

  JXL_DEBUG_V(2,
              "JPEG ctx effort at speed tier %i: %i candidates, rank_iters=%u "
              "final_iters=%u refine_iters=%u\n",
              static_cast<int>(speed_tier), static_cast<int>(candidates.size()),
              effort.rank_iters, effort.final_iters, effort.refine_iters);

  int64_t best_cost = std::numeric_limits<int64_t>::max();
  ThresholdSet best_thr;
  ContextMap best_ctx;
  std::mutex mu;

  auto compute_nz_cost = [&](const PartitioningCtx::Clustering& clustering) {
    int64_t nz_cost = 0;
    for (const auto& cl_N : clustering.hist_nz_N) {
      for (uint32_t freq : cl_N)
        if (freq != 0) nz_cost += opt_data->nz_entropy(freq);
    }
    for (const auto& cl_h : clustering.hist_nz_h) {
      for (uint32_t freq : cl_h)
        if (freq != 0) nz_cost -= opt_data->nz_entropy(freq);
    }
    return nz_cost;
  };

  std::vector<PartitioningCtx> ctx_pool;
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, static_cast<uint32_t>(candidates.size()),
      [&](size_t num_threads) -> Status {
        ctx_pool.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
          ctx_pool.emplace_back(opt_data);
        }
        return true;
      },
      [&](uint32_t idx, size_t thread_id) -> Status {
        const FactorizationCandidate& candidate = candidates[idx];
        uint32_t a = candidate.a;
        uint32_t b = candidate.b;
        uint32_t c = candidate.c;
        PartitioningCtx& ctx = ctx_pool[thread_id];
        auto opt_result = ctx.OptimizeThresholds(
            candidate.init, effort.final_m_target, effort.final_iters);

        PartitioningCtx::Clustering cl_result;
        JXL_ASSIGN_OR_RETURN(
            cl_result, ctx.ClusterContexts(
                           opt_result.second,
                           kMaxClusters - (jpeg_data.components.size() == 1),
                           effort.overhead_aware_tail, nullptr));
        ContextMap& cluster_map = cl_result.ctx_map;

        ThresholdSet refined_thr;
        int64_t entropy_cost = 0;
        int64_t nz_cost = 0;
        if (effort.refine_iters == 0) {
          refined_thr = cl_result.PruneDeadThresholds(opt_result.second);
          entropy_cost = cl_result.clustered_cost;
          nz_cost = compute_nz_cost(cl_result);
        } else {
          auto refine_result =
              ctx.RefineClustered(opt_result.second, cl_result,
                                  effort.refine_iters, effort.refine_radius);
          refined_thr = refine_result.thresholds;
          entropy_cost = refine_result.cost;
          nz_cost = refine_result.nz_cost;
        }
        int64_t total_cost = entropy_cost;

        // Add signalling overhead for histogram headers
        int64_t overhead = 0;
        auto overhead_or = cl_result.ComputeSignallingOverhead(*opt_data);
        if (overhead_or.ok()) {
          overhead = std::move(overhead_or).value_();
          total_cost += overhead;
        }

        std::lock_guard<std::mutex> lock(mu);
        // JXL_DEBUG_V(
        //     2,
        printf(
            "(%u,%u,%u) cost: unclustered=%.2f clustered=%.2f "
            "refined=%.2f nz=%.2f overhead=%.2f total=%.2f\n",
            a, b, c, bit_cost(opt_result.first),
            bit_cost(cl_result.clustered_cost), bit_cost(entropy_cost),
            bit_cost(nz_cost), bit_cost(overhead), bit_cost(total_cost));
        if (total_cost < best_cost) {
          best_cost = total_cost;
          best_thr = refined_thr;
          best_ctx = cluster_map;
        }
        return true;
      },
      "JpegCtxOpt"));

  size_t na_Y = best_thr.TY().size() + 1;
  size_t na_Cb = best_thr.TCb().size() + 1;
  size_t na_Cr = best_thr.TCr().size() + 1;
  size_t num_dc_ctxs = na_Y * na_Cb * na_Cr;

  JXL_ENSURE(num_dc_ctxs <= kMaxCells);
  JXL_ENSURE(na_Y <= kMaxIntervals && na_Cb <= kMaxIntervals &&
             na_Cr <= kMaxIntervals);

  ctx_map.num_dc_ctxs = num_dc_ctxs;

  ctx_map.dc_thresholds[1].clear();
  ctx_map.dc_thresholds[0].clear();
  ctx_map.dc_thresholds[2].clear();

  for (int16_t t : best_thr.TY()) ctx_map.dc_thresholds[1].push_back(t - 1);
  for (int16_t t : best_thr.TCb()) ctx_map.dc_thresholds[0].push_back(t - 1);
  for (int16_t t : best_thr.TCr()) ctx_map.dc_thresholds[2].push_back(t - 1);

  // Note: `best_ctx` and `ctx_map` are in JPEG order (Y, Cb, Cr)
  ctx_map.ctx_map.assign(3 * kNumOrders * num_dc_ctxs, 0);
  for (size_t c = 0; c < opt_data->channels; c++) {
    for (size_t cell = 0; cell < num_dc_ctxs; ++cell) {
      ctx_map.ctx_map[c * kNumOrders * num_dc_ctxs + cell] =
          best_ctx[c * num_dc_ctxs + cell] + (opt_data->channels == 1);
    }
  }
  size_t num_ctxs =
      *std::max_element(ctx_map.ctx_map.begin(), ctx_map.ctx_map.end()) + 1;
  JXL_ENSURE(num_ctxs <= kMaxClusters);
  ctx_map.num_ctxs = num_ctxs;

  return true;
}

}  // namespace jxl
