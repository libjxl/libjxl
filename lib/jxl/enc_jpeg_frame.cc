// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_jpeg_frame.h"

#include <algorithm>
#include <array>
#include <chrono>
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
#include "lib/jxl/frame_header.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace {

constexpr int64_t kFScale = 1LL << 25;
constexpr int kDCTOff = 1024;
constexpr int kDCTRange = 2048;
constexpr int kMaxCells = 64;
constexpr int kNumPos = 63;
constexpr int kNumCh = 3;
constexpr uint32_t kMTarget = 256;

typedef std::vector<int16_t> Thresholds;
typedef uint32_t ACEntry;
typedef std::vector<uint8_t> ContextMap;
typedef std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> Factorizations;

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

struct JPEGOptData {
  using ACCounts =
      std::array<std::array<std::array<uint32_t, kDCTRange>, kNumPos>, kNumCh>;

  std::vector<int64_t> ftab;

  uint32_t num_blocks[kNumCh];
  uint32_t block_grid_w[kNumCh];
  uint32_t block_grid_h[kNumCh];
  Thresholds dc_vals[kNumCh];
  uint32_t DC_cnt[kNumCh][kDCTRange];
  uint32_t DC_idx_LUT[kNumCh][kDCTRange];
  uint32_t num_zdcai;

  std::vector<ACEntry> AC_stream;
  std::array<std::vector<uint32_t>, kNumCh> block_bins;
  std::array<std::vector<uint32_t>, kNumCh> block_offsets;
  std::array<std::vector<uint8_t>, kNumCh> block_nonzeros;
  std::vector<uint32_t> compact_map_h;
  std::array<std::vector<uint16_t>, kNumCh> block_dc;
  std::array<std::vector<uint32_t>, kNumCh> dc_sorted_blocks;
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

  uint32_t ToChBlock(uint32_t b, uint32_t c) const {
    if (c == 0) return b;
    uint32_t bx = b % block_grid_w[0];
    uint32_t by = b / block_grid_w[0];
    return (by * block_grid_h[c] / block_grid_h[0]) * block_grid_w[c] +
           bx * block_grid_w[c] / block_grid_w[0];
  }

  Factorizations MaximalFactorizations() const {
    uint32_t cap0 = std::max(1u, static_cast<uint32_t>(dc_vals[0].size()));
    uint32_t cap1 = std::max(1u, static_cast<uint32_t>(dc_vals[1].size()));
    uint32_t cap2 = std::max(1u, static_cast<uint32_t>(dc_vals[2].size()));
    // Each threshold count is written with 4 bits (max 15), so each dimension
    // of intervals is capped at 16.
    constexpr uint32_t kMaxIntervals = 16;
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
    return result;
  }

  Status CountDCAC(const jpeg::JPEGData& jpeg_data,
                   const std::array<int, 3>& jpeg_c_map, ThreadPool* pool,
                   ACCounts& ac_cnt) {
    memset(DC_cnt, 0, sizeof(DC_cnt));
    memset(ac_cnt.data(), 0, sizeof(ac_cnt));
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, kNumCh, ThreadPool::NoInit,
        [&](uint32_t c, size_t) -> Status {
          uint32_t jpeg_c = static_cast<uint32_t>(jpeg_c_map[c]);
          const auto& comp = jpeg_data.components[jpeg_c];
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

    for (uint32_t c = 0; c < kNumCh; ++c) {
      auto& v = dc_vals[c];
      v.clear();
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

  Status CollectACBins(const jpeg::JPEGData& jpeg_data,
                       const std::array<int, 3>& jpeg_c_map, ThreadPool* pool,
                       const ACCounts& ac_cnt) {
    std::array<std::array<uint8_t, kNumPos>, kNumCh> scan_order = {};
    std::array<std::array<bool, kNumPos>, kNumCh> active_scan = {};
    BuildACScanOrder(ac_cnt, scan_order, active_scan);

    uint32_t wb0 = block_grid_w[0];
    uint32_t hb0 = block_grid_h[0];
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, kNumCh, ThreadPool::NoInit,
        [&](uint32_t c, size_t) -> Status {
          uint32_t jpeg_c = static_cast<uint32_t>(jpeg_c_map[c]);
          const auto& comp = jpeg_data.components[jpeg_c];
          uint32_t wb = comp.width_in_blocks;
          uint32_t hb = comp.height_in_blocks;
          uint32_t f_h = wb0 / std::max(1u, wb);
          uint32_t f_v = hb0 / std::max(1u, hb);

          block_dc[c].assign(num_blocks[0], 0);
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
              for (uint32_t dy = 0; dy < f_v; ++dy) {
                uint32_t by0 = by * f_v + dy;
                if (by0 >= hb0) break;
                for (uint32_t dx = 0; dx < f_h; ++dx) {
                  uint32_t bx0 = bx * f_h + dx;
                  if (bx0 >= wb0) break;
                  block_dc[c][by0 * wb0 + bx0] = dc_idx;
                }
              }
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
                  block_bins[c].push_back((zdc << 11) | ai);
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
    return true;
  }

  Status FinalizeSpatialIndexing(ThreadPool* pool) {
    memset(DC_cnt, 0, sizeof(DC_cnt));
    uint32_t wb0 = block_grid_w[0];
    uint32_t hb0 = block_grid_h[0];
    for (uint32_t y = 0; y < hb0; ++y) {
      for (uint32_t x = 0; x < wb0; ++x) {
        uint32_t b = y * wb0 + x;
        uint32_t dc0 = block_dc[0][b];
        uint32_t dc1 = block_dc[1][b];
        uint32_t dc2 = block_dc[2][b];
        for (uint32_t c = 0; c < kNumCh; ++c) {
          uint32_t bc = ToChBlock(b, c);
          for (uint32_t pi = block_offsets[c][bc];
               pi < block_offsets[c][bc + 1]; ++pi) {
            ++DC_cnt[0][dc_vals[0][dc0] + kDCTOff];
            ++DC_cnt[1][dc_vals[1][dc1] + kDCTOff];
            ++DC_cnt[2][dc_vals[2][dc2] + kDCTOff];
          }
        }
      }
    }

    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, kNumCh, ThreadPool::NoInit,
        [&](uint32_t c, size_t) -> Status {
          Thresholds pruned;
          uint32_t idx = 0;
          for (int di = 0; di < kDCTRange; ++di) {
            if (DC_cnt[c][di]) {
              pruned.push_back(static_cast<int16_t>(di - kDCTOff));
              DC_idx_LUT[c][di] = idx++;
            }
          }
          uint32_t N_blocks = num_blocks[0];
          uint32_t M = static_cast<uint32_t>(pruned.size());
          dc_block_offsets[c].assign(M + 1, 0);
          for (uint32_t b = 0; b < N_blocks; ++b) {
            uint32_t raw_val =
                static_cast<uint32_t>(dc_vals[c][block_dc[c][b]]) + kDCTOff;
            uint16_t new_dc = static_cast<uint16_t>(DC_idx_LUT[c][raw_val]);
            block_dc[c][b] = new_dc;
            ++dc_block_offsets[c][new_dc + 1];
          }
          for (uint32_t v = 0; v < M; ++v)
            dc_block_offsets[c][v + 1] += dc_block_offsets[c][v];
          dc_vals[c].swap(pruned);
          return true;
        },
        "PruneDC"));

    uint32_t N_blocks = num_blocks[0];
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, kNumCh, ThreadPool::NoInit,
        [&](uint32_t axis, size_t) -> Status {
          dc_sorted_blocks[axis].resize(N_blocks);
          std::vector<uint32_t> write_pos = dc_block_offsets[axis];
          for (uint32_t b = 0; b < N_blocks; ++b) {
            uint32_t v = block_dc[axis][b];
            dc_sorted_blocks[axis][write_pos[v]++] = b;
          }
          return true;
        },
        "BuildSortedBlocks"));
    return true;
  }

  Status GenerateRLEStream(ThreadPool* pool) {
    constexpr uint32_t BIN_N = 3u << 20;
    std::vector<uint32_t> bin_start_lo(BIN_N + 1, 0);
    std::vector<uint32_t> bin_start_hi(BIN_N + 1, 0);

    uint32_t w = block_grid_w[0];
    uint32_t h = block_grid_h[0];
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t b = y * w + x;
        uint32_t dc0 = block_dc[0][b];
        for (uint32_t c = 0; c < kNumCh; ++c) {
          uint32_t bc = ToChBlock(b, c);
          uint32_t ch_prefix = c << 20;
          for (uint32_t pi = block_offsets[c][bc];
               pi < block_offsets[c][bc + 1]; ++pi) {
            uint32_t bin = ch_prefix | block_bins[c][pi];
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

    size_t lo_size = bin_start_lo[BIN_N];
    size_t hi_size = bin_start_hi[BIN_N];
    size_t raw_size = lo_size + hi_size;
    std::vector<uint32_t> flat_lo(lo_size);
    std::vector<uint32_t> flat_hi(hi_size);

    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t b = y * w + x;
        uint32_t dc0 = block_dc[0][b];
        uint32_t dc1 = block_dc[1][b];
        uint32_t dc2 = block_dc[2][b];
        for (uint32_t c = 0; c < kNumCh; ++c) {
          uint32_t bc = ToChBlock(b, c);
          uint32_t ch_prefix = c << 20;
          for (uint32_t pi = block_offsets[c][bc];
               pi < block_offsets[c][bc + 1]; ++pi) {
            uint32_t bin = ch_prefix | block_bins[c][pi];
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
    uint32_t prev_b = UINT32_MAX;
    std::array<uint32_t, kZeroDensityContextCount> zdc_total = {};

    uint32_t MAX_KEY_H_SPARSE =
        static_cast<uint32_t>(kZeroDensityContextCount) * 2048;
    compact_map_h.assign(MAX_KEY_H_SPARSE, 0xFFFFFFFF);
    num_zdcai = 0;

    uint32_t lo_pos = 0;
    uint32_t hi_pos = 0;
    for (size_t bi = 0; bi < active_bins.size(); ++bi) {
      uint32_t b = active_bins[bi];
      uint32_t zdc_ai = b & 0xFFFFF;
      if (compact_map_h[zdc_ai] == 0xFFFFFFFF)
        compact_map_h[zdc_ai] = num_zdcai++;

      uint32_t start_lo = lo_pos;
      uint32_t start_hi = hi_pos;
      uint32_t end_lo = bin_start_lo[b];
      uint32_t end_hi = bin_start_hi[b];
      lo_pos = end_lo;
      hi_pos = end_hi;

      bool new_ctx = (prev_b == UINT32_MAX) || ((b >> 11) != (prev_b >> 11));
      if (new_ctx) {
        if (prev_b != UINT32_MAX) {
          zdc_total[(prev_b >> 11) & 0x1FFu] += ctx_len;
        }
        ctx_len = 0;
      }
      ctx_len += (end_lo - start_lo) + (end_hi - start_hi);

      bool bin_ch = (prev_b != UINT32_MAX);
      bool ctx_ch = bin_ch && ((b >> 11) != (prev_b >> 11));

      uint32_t cur_dc0 = 0;
      bool first_in_bin = true;
      auto emit_half = [&](const uint32_t* data_ptr, uint32_t start,
                           uint32_t end, uint32_t dc0_base) {
        uint32_t i = start;
        while (i < end) {
          uint32_t j = i + 1;
          while (j < end && data_ptr[j] == data_ptr[i]) ++j;
          uint32_t run = j - i;
          uint32_t dc0 = (data_ptr[i] >> 22) | dc0_base;
          uint32_t dc1 = (data_ptr[i] >> 11) & 0x7FFu;
          uint32_t dc2 = data_ptr[i] & 0x7FFu;
          if (first_in_bin || (dc0 - cur_dc0 > 15u)) {
            stream.push_back(
                (1u << 31) |
                (static_cast<uint32_t>(ctx_ch && first_in_bin) << 30) |
                (static_cast<uint32_t>(bin_ch && first_in_bin) << 29) |
                (b << 7) | (dc0 >> 4));
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

      emit_half(flat_lo.data(), start_lo, end_lo, 0u);
      emit_half(flat_hi.data(), start_hi, end_hi, 0x400u);
      prev_b = b;
    }

    if (prev_b != UINT32_MAX) {
      zdc_total[(prev_b >> 11) & 0x1FFu] += ctx_len;
    }
    stream.shrink_to_fit();
    AC_stream = std::move(stream);

    uint32_t max_zdc_total =
        *std::max_element(zdc_total.begin(), zdc_total.end());
    InitFTab(max_zdc_total + 1);

    return true;
  }

  Status BuildFromJPEG(const jpeg::JPEGData& jpeg_data,
                       const std::array<int, 3>& jpeg_c_map, ThreadPool* pool);
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

  typedef std::pair<uint16_t, uint32_t> Bin;
  typedef std::array<std::vector<Bin>, kMaxCells / 2> CellHistory;
  CellHistory h_hist;
  CellHistory N_hist;

  std::vector<uint64_t> touched_h;
  std::vector<uint64_t> touched_N;
  std::vector<uint64_t> group_touched_h;
  std::vector<uint64_t> group_touched_N;

  std::vector<uint32_t> cluster_hist_h;
  std::vector<uint32_t> cluster_hist_N;
  std::vector<uint32_t> cluster_touched_h;
  std::vector<uint32_t> cluster_touched_N;

  std::vector<uint16_t> ax0_to_k;
  std::vector<uint16_t> k_to_dc0;
  std::vector<uint16_t> ax1_row;
  std::vector<uint16_t> ax2_col;

  explicit PartitioningCtx(std::shared_ptr<const JPEGOptData> d)
      : data_(std::move(d)),
        DP_prev(kDCTRange, 0),
        DP_curr(kDCTRange, 0),
        row_done(kDCTRange, 0),
        row_cum(kDCTRange, 0),
        h_hist(),
        N_hist(),
        touched_h(kMaxCells / 2 * kDCTRange >> 6, 0),
        touched_N(kMaxCells / 2 * kDCTRange >> 6, 0),
        group_touched_h(kMaxCells / 2 * kDCTRange >> 12, 0),
        group_touched_N(kMaxCells / 2 * kDCTRange >> 12, 0),
        ax0_to_k(kDCTRange, 0),
        k_to_dc0(kDCTRange, 0),
        ax1_row(kDCTRange, 0),
        ax2_col(kDCTRange, 0) {}

  const JPEGOptData& data() const { return *data_; }
  const std::vector<ACEntry>& ac_stream() const { return data().AC_stream; }

  static ptrdiff_t Bkt(int dc, const Thresholds& T) {
    return std::upper_bound(T.begin(), T.end(), dc) - T.begin();
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
  void StreamSweep(const std::vector<uint32_t>& stream, FlushH&& flush_h,
                   FlushN&& flush_N, OnRun&& on_run) const {
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
    uint32_t start = (row_done[n] == UINT16_MAX) ? 0u : (row_done[n] + 1);
    int64_t cum = (start == 0) ? 0 : row_cum[n];
    for (uint32_t v = start; v <= l; ++v) {
      cum += costs[n * M + v];
      if (n > 0 && v < n) {
        EnsureCost(M, v, n - 1);
        costs[n * M + v] = costs[(n - 1) * M + v] + cum;
      } else {
        costs[n * M + v] = cum;
      }
    }
    row_cum[n] = cum;
    row_done[n] = static_cast<uint16_t>(l);
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
                                     const std::vector<uint32_t>& stream,
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
          for (size_t hi = 0; hi < hist.size(); hi++) total += hist[hi].second;
          uint32_t j_before_l = 0;
          uint32_t l = 1;
          int64_t prev_term = 0;
          for (size_t hi = 0; hi < hist.size(); hi++) {
            const Bin& h = hist[hi];
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
          stream, [&]() { flush_hist(h_hist, bin_mask, -1); },
          [&]() { flush_hist(N_hist, ctx_mask, +1); },
          [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx,
              uint32_t run, uint32_t) {
            uint32_t dc_k_bkt = ax0_to_k[dc0_idx];
            uint32_t ci =
                static_cast<uint32_t>(ax1_row[dc1_idx] + ax2_col[dc2_idx]);
            bin_mask |= (1ULL << ci);
            ctx_mask |= (1ULL << ci);
            if (!h_hist[ci].empty() && h_hist[ci].back().first == dc_k_bkt) {
              h_hist[ci].back().second += run;
            } else {
              h_hist[ci].push_back({static_cast<uint16_t>(dc_k_bkt), run});
            }
            if (!N_hist[ci].empty() && N_hist[ci].back().first == dc_k_bkt) {
              N_hist[ci].back().second += run;
            } else {
              N_hist[ci].push_back({static_cast<uint16_t>(dc_k_bkt), run});
            }
          });
      flush_hist(h_hist, bin_mask, -1);
      flush_hist(N_hist, ctx_mask, +1);
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

      StreamSweep(
          stream, [&]() { flush_dense(h_cnt, bin_mask, -1); },
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
    auto& hist = h_hist[0];
    hist.clear();
    uint32_t cur_ci = UINT32_MAX;
    for (uint32_t hi_idx = 0; hi_idx < 16; hi_idx++) {
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
            hist.clear();
            cur_ci = ci;
          }
          int64_t* cost_row = &costs[n * M_eff];
          uint32_t freq = cnt[bit_idx];
          uint32_t j_n = freq;
          if (!hist.empty()) {
            j_n += hist.back().second;
            uint32_t l = 0;
            uint32_t j_before_l = 0;
            int64_t prev_term = 0;
            for (size_t hi = 0; hi < hist.size(); hi++) {
              const Bin& h = hist[hi];
              uint32_t j_ln = j_n - j_before_l;
              int64_t term = sign * (d.ftab[j_ln - freq] - d.ftab[j_ln]);
              cost_row[l] += term - prev_term;
              l = h.first + 1;
              j_before_l = h.second;
              prev_term = term;
            }
            cost_row[l] -= sign * d.ftab[freq] + prev_term;
          }
          hist.push_back({static_cast<uint16_t>(n), j_n});
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
                                     const std::vector<uint32_t>& stream,
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

    if (num_intervals == 2)
      return OptimizeAxisSingleSplit(axis, ncells, stream, M_eff);

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
        stream,
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
      ThresholdSet T, const std::vector<uint32_t>& stream,
      uint32_t M_target = UINT32_MAX, uint32_t max_iters = 20) {
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
        newT.TY() =
            OptimizeAxisSingleSweep(0, a, T.TCb(), T.TCr(), stream, M_target);
        TY_changed = (newT.TY() != T.TY());
        std::swap(T.TY(), newT.TY());
      } else {
        TY_changed = false;
      }
      if ((b != 1) && (iter == 0 || TY_changed || TCr_changed)) {
        newT.TCb() =
            OptimizeAxisSingleSweep(1, b, T.TCr(), T.TY(), stream, M_target);
        TCb_changed = (newT.TCb() != T.TCb());
        std::swap(T.TCb(), newT.TCb());
      } else {
        TCb_changed = false;
      }
      if ((c != 1) && (iter == 0 || TY_changed || TCb_changed)) {
        newT.TCr() =
            OptimizeAxisSingleSweep(2, c, T.TY(), T.TCb(), stream, M_target);
        TCr_changed = (newT.TCr() != T.TCr());
        std::swap(T.TCr(), newT.TCr());
      } else {
        TCr_changed = false;
      }
      if (!TY_changed && !TCb_changed && !TCr_changed) break;
    }
    return {TotalCost(T, stream), T};
  }

  int64_t TotalCost(const ThresholdSet& T,
                    const std::vector<uint32_t>& stream) {
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
      for (size_t gi = 0; gi < 16; ++gi) {
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
      for (size_t gi = 0; gi < 16; ++gi) {
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

    StreamSweep(stream, flush_h, flush_N,
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

  typedef std::vector<std::unordered_map<uint32_t, uint32_t>> SparseHistogram;
  struct Clustering {
    int64_t clustered_cost;
    ContextMap ctx_map;
  };

  StatusOr<Clustering> AgglomerativeClusterCore(
      uint32_t total_ctxs, uint32_t num_clusters, SparseHistogram& hist_h,
      SparseHistogram& hist_N, uint32_t P, ThreadPool* pool) const {
    const JPEGOptData& d = data();

    std::vector<int64_t> E(total_ctxs, 0);
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, total_ctxs, ThreadPool::NoInit,
        [&](uint32_t i, size_t) -> Status {
          int64_t local_E = 0;
          for (uint32_t p = 0; p < P; ++p) {
            uint32_t idx = i * P + p;
            for (auto kv = hist_N[idx].begin(); kv != hist_N[idx].end(); ++kv)
              local_E += d.ftab[kv->second];
            for (auto kv = hist_h[idx].begin(); kv != hist_h[idx].end(); ++kv)
              local_E -= d.ftab[kv->second];
          }
          E[i] = local_E;
          return true;
        },
        "InitEntropy"));

    int64_t initial_cost = 0;
    for (uint32_t i = 0; i < total_ctxs; i++) initial_cost += E[i];

    if (total_ctxs <= num_clusters) {
      ContextMap ctx_table(total_ctxs);
      std::iota(ctx_table.begin(), ctx_table.end(), 0);
      return Clustering{initial_cost, ctx_table};
    }

    auto is_ctx_active = [&](uint32_t ctx) -> bool {
      for (uint32_t p = 0; p < P; ++p) {
        if (!hist_N[ctx * P + p].empty()) return true;
      }
      return false;
    };

    std::vector<uint32_t> active;
    active.reserve(total_ctxs);
    for (uint32_t i = 0; i < total_ctxs; i++) {
      if (is_ctx_active(i)) active.push_back(i);
    }

    std::vector<uint32_t> parent(total_ctxs);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<uint32_t(uint32_t)> find = [&parent,
                                              &find](uint32_t x) -> uint32_t {
      return parent[x] == x ? x : parent[x] = find(parent[x]);
    };

    std::vector<int64_t> deltas(total_ctxs * total_ctxs, 0);
    auto get_delta = [&](uint32_t a, uint32_t b) -> int64_t& {
      return deltas[std::min(a, b) * total_ctxs + std::max(a, b)];
    };
    auto merge_delta = [&](uint32_t a, uint32_t b) -> int64_t {
      int64_t delta = 0;
      for (uint32_t p = 0; p < P; ++p) {
        uint32_t ia = a * P + p;
        uint32_t ib = b * P + p;
        for (auto kv = hist_N[ia].begin(); kv != hist_N[ia].end(); ++kv) {
          auto it = hist_N[ib].find(kv->first);
          if (it != hist_N[ib].end()) {
            delta += d.ftab[kv->second + it->second] - d.ftab[kv->second] -
                     d.ftab[it->second];
          }
        }
        for (auto kv = hist_h[ia].begin(); kv != hist_h[ia].end(); ++kv) {
          auto it = hist_h[ib].find(kv->first);
          if (it != hist_h[ib].end()) {
            delta -= d.ftab[kv->second + it->second] - d.ftab[kv->second] -
                     d.ftab[it->second];
          }
        }
      }
      return delta;
    };

    uint32_t num_active = static_cast<uint32_t>(active.size());
    if (num_active >= 2) {
      JXL_RETURN_IF_ERROR(RunOnPool(
          pool, 0, num_active - 1, ThreadPool::NoInit,
          [&](uint32_t i, size_t) -> Status {
            uint32_t id_i = active[i];
            for (uint32_t j = i + 1; j < num_active; j++)
              get_delta(id_i, active[j]) = merge_delta(id_i, active[j]);
            return true;
          },
          "MergeDelta"));
    }

    while (active.size() > num_clusters && active.size() > 1) {
      int64_t best_delta = std::numeric_limits<int64_t>::max();
      size_t best_i = 0;
      size_t best_j = 1;
      std::mutex best_mtx;

      uint32_t cur_active = static_cast<uint32_t>(active.size());
      JXL_RETURN_IF_ERROR(RunOnPool(
          pool, 0, cur_active - 1, ThreadPool::NoInit,
          [&](uint32_t i, size_t) -> Status {
            uint32_t id_i = active[i];
            size_t local_best_j = i + 1;
            int64_t local_best_diff = get_delta(id_i, active[local_best_j]);
            size_t K = active.size();
            for (size_t j = i + 2; j < K; j++) {
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

      uint32_t a_id = active[best_i];
      uint32_t b_id = active[best_j];
      E[a_id] += E[b_id] + best_delta;
      for (uint32_t p = 0; p < P; ++p) {
        uint32_t ia = a_id * P + p;
        uint32_t ib = b_id * P + p;
        for (auto kv = hist_N[ib].begin(); kv != hist_N[ib].end(); ++kv)
          hist_N[ia][kv->first] += kv->second;
        for (auto kv = hist_h[ib].begin(); kv != hist_h[ib].end(); ++kv)
          hist_h[ia][kv->first] += kv->second;
      }
      parent[b_id] = a_id;
      active.erase(active.begin() + static_cast<ptrdiff_t>(best_j));
      num_active = static_cast<uint32_t>(active.size());
      if (num_active > 0) {
        JXL_RETURN_IF_ERROR(RunOnPool(
            pool, 0, num_active, ThreadPool::NoInit,
            [&](uint32_t k, size_t) -> Status {
              if (active[k] != a_id)
                get_delta(a_id, active[k]) = merge_delta(a_id, active[k]);
              return true;
            },
            "UpdateDist"));
      }
    }

    std::unordered_map<uint32_t, uint32_t> rep_to_cluster;
    int64_t clustered_cost = 0;
    for (size_t k = 0; k < active.size(); k++) {
      rep_to_cluster[active[k]] = static_cast<uint32_t>(k);
      clustered_cost += E[active[k]];
    }

    ContextMap result(total_ctxs, 0);
    for (uint32_t i = 0; i < total_ctxs; i++) {
      auto it = rep_to_cluster.find(find(i));
      if (it != rep_to_cluster.end()) {
        result[i] = static_cast<uint8_t>(it->second);
      } else if (!is_ctx_active(i)) {
        result[i] = 0;
      }
    }
    return Clustering{clustered_cost, result};
  }

  StatusOr<Clustering> ClusterContexts(const ThresholdSet& thresholds,
                                       uint32_t num_clusters = 16,
                                       ThreadPool* pool = nullptr) {
    const auto& stream = ac_stream();
    uint32_t na = static_cast<uint32_t>(thresholds.TY().size() + 1);
    uint32_t n1 = static_cast<uint32_t>(thresholds.TCb().size() + 1);
    uint32_t n2 = static_cast<uint32_t>(thresholds.TCr().size() + 1);
    uint32_t num_cells = na * n1 * n2;
    uint32_t total_ctxs = kNumCh * num_cells;
    UpdateMaps(thresholds);

    uint32_t P = 1;
    SparseHistogram hist_h(total_ctxs * P);
    SparseHistogram hist_N(total_ctxs * P);

    StreamSweep(
        stream, []() {}, []() {},
        [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
            uint32_t bin_state) {
          uint32_t cur_c = (bin_state >> 20) & 0x3u;
          uint32_t cell = static_cast<uint32_t>(
              (ax1_row[dc1_idx] + ax2_col[dc2_idx]) * na + ax0_to_k[dc0_idx]);
          uint32_t ctx_id = cur_c * num_cells + cell;
          uint32_t idx = ctx_id * P;
          hist_h[idx][bin_state & 0xFFFFFu] += run;
          uint32_t cur_zdc = (bin_state >> 11) & 0x1FFu;
          hist_N[idx][cur_zdc] += run;
        });

    return AgglomerativeClusterCore(total_ctxs, num_clusters, hist_h, hist_N, P,
                                    pool);
  }

  int64_t BuildBaselineHistograms(const ThresholdSet& thresholds,
                                  const ContextMap& cluster_map,
                                  uint32_t num_clusters = 16) {
    const JPEGOptData& d = data();
    const auto& stream = ac_stream();
    uint32_t na = static_cast<uint32_t>(thresholds.TY().size() + 1);
    uint32_t n1 = static_cast<uint32_t>(thresholds.TCb().size() + 1);
    uint32_t n2 = static_cast<uint32_t>(thresholds.TCr().size() + 1);
    uint32_t num_cells = na * n1 * n2;
    UpdateMaps(thresholds);

    uint32_t MAX_KEY_H = d.num_zdcai;
    uint32_t MAX_KEY_N = static_cast<uint32_t>(kZeroDensityContextCount);

    if (cluster_hist_h.size() < num_clusters * MAX_KEY_H) {
      cluster_hist_h.assign(num_clusters * MAX_KEY_H, 0);
      cluster_hist_N.assign(num_clusters * MAX_KEY_N, 0);
    } else {
      for (uint32_t idx : cluster_touched_N) cluster_hist_N[idx] = 0;
      for (uint32_t idx : cluster_touched_h) cluster_hist_h[idx] = 0;
    }
    cluster_touched_h.clear();
    cluster_touched_N.clear();

    StreamSweep(
        stream, []() {}, []() {},
        [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
            uint32_t bin_state) {
          uint32_t cur_c = (bin_state >> 20) & 0x3u;
          uint32_t cell = static_cast<uint32_t>(
              (ax1_row[dc1_idx] + ax2_col[dc2_idx]) * na + ax0_to_k[dc0_idx]);
          uint32_t ctx_id = cur_c * num_cells + cell;
          uint8_t cluster_id = cluster_map[ctx_id];
          uint32_t h_idx =
              cluster_id * MAX_KEY_H + d.compact_map_h[bin_state & 0xFFFFFu];
          uint32_t N_idx =
              cluster_id * MAX_KEY_N + ((bin_state >> 11) & 0x1FFu);
          if (cluster_hist_h[h_idx] == 0) cluster_touched_h.push_back(h_idx);
          cluster_hist_h[h_idx] += run;
          if (cluster_hist_N[N_idx] == 0) cluster_touched_N.push_back(N_idx);
          cluster_hist_N[N_idx] += run;
        });

    int64_t E = 0;
    for (uint32_t idx : cluster_touched_N) E += d.ftab[cluster_hist_N[idx]];
    for (uint32_t idx : cluster_touched_h) E -= d.ftab[cluster_hist_h[idx]];
    return E;
  }

  int64_t NZClusteredCost(const ThresholdSet& thresholds,
                          const ContextMap& cluster_map) {
    const JPEGOptData& d = data();
    uint32_t na = static_cast<uint32_t>(thresholds.TY().size() + 1);
    uint32_t n1 = static_cast<uint32_t>(thresholds.TCb().size() + 1);
    uint32_t n2 = static_cast<uint32_t>(thresholds.TCr().size() + 1);
    uint32_t num_cells = na * n1 * n2;
    uint32_t total_ctxs = kNumCh * num_cells;
    if (cluster_map.size() != total_ctxs) return 0;
    UpdateMaps(thresholds);

    uint32_t num_clusters = 0;
    for (size_t i = 0; i < cluster_map.size(); i++) {
      num_clusters =
          std::max(num_clusters, static_cast<uint32_t>(cluster_map[i]) + 1);
    }

    uint32_t N_keys = 36 * num_clusters;
    uint32_t H_keys = N_keys << 6;
    if (h_cnt.size() < N_keys) {
      h_cnt.assign(N_keys, 0);
    } else {
      std::fill(h_cnt.begin(), h_cnt.begin() + N_keys, 0);
    }
    if (N_cnt.size() < H_keys) {
      N_cnt.assign(H_keys, 0);
    } else {
      std::fill(N_cnt.begin(), N_cnt.begin() + H_keys, 0);
    }

    uint32_t w0 = d.block_grid_w[0];
    uint32_t h0 = d.block_grid_h[0];
    std::vector<uint32_t> row_top(w0);
    std::vector<uint32_t> row_cur(w0);

    for (uint32_t c = 0; c < kNumCh; ++c) {
      std::fill(row_top.begin(), row_top.end(), 32u);
      std::fill(row_cur.begin(), row_cur.end(), 32u);
      bool has_top = false;
      for (uint32_t by = 0; by < h0; ++by) {
        for (uint32_t bx = 0; bx < w0; ++bx) {
          uint32_t b = by * w0 + bx;
          uint32_t bc = d.ToChBlock(b, c);
          uint32_t predicted_nz;
          if (bx == 0) {
            predicted_nz = has_top ? row_top[bx] : 32u;
          } else if (!has_top) {
            predicted_nz = row_cur[bx - 1];
          } else {
            predicted_nz = (row_top[bx] + row_cur[bx - 1] + 1u) / 2u;
          }

          uint32_t cell = static_cast<uint32_t>(
              (ax1_row[d.block_dc[1][b]] + ax2_col[d.block_dc[2][b]]) * na +
              ax0_to_k[d.block_dc[0][b]]);
          uint32_t block_ctx = cluster_map[c * num_cells + cell];
          uint32_t nzero_ctx =
              ((predicted_nz < 8) ? predicted_nz : (4 + predicted_nz / 2)) *
                  num_clusters +
              block_ctx;
          uint32_t nz = d.block_nonzeros[c][bc];
          h_cnt[nzero_ctx] += 1;
          N_cnt[(nzero_ctx << 6) | nz] += 1;
          row_cur[bx] = nz;
        }
        row_top.swap(row_cur);
        has_top = true;
      }
    }

    // `h_cnt`/`N_cnt` here store block counts per context,
    // and can grow up to about `num_blocks`, but `ftab` is sized
    // only to cover `max_zdc_total`.
    auto entropy_term = [&d](uint32_t n) -> int64_t {
      if (n < d.ftab.size()) return d.ftab[n];
      const double nd = static_cast<double>(n);
      return static_cast<int64_t>(std::llround(nd * std::log2(nd) * kFScale));
    };
    int64_t cost = 0;
    for (uint32_t i = 0; i < N_keys; i++) cost += entropy_term(h_cnt[i]);
    for (uint32_t i = 0; i < H_keys; i++) cost -= entropy_term(N_cnt[i]);
    return cost;
  }

  std::pair<ThresholdSet, int64_t> RefineClustered(
      const ThresholdSet& thresholds, const ContextMap& cluster_map,
      uint32_t max_iters = 5, ptrdiff_t search_radius = 2048) {
    const JPEGOptData& d = data();
    ThresholdSet cur_T = thresholds;
    std::vector<uint32_t> scratch_hist_h;
    std::vector<uint32_t> scratch_hist_N;
    int64_t base_cost = BuildBaselineHistograms(cur_T, cluster_map, 16);

    uint32_t size_Y = static_cast<uint32_t>(cur_T.TY().size() + 1);
    uint32_t size_Cb = static_cast<uint32_t>(cur_T.TCb().size() + 1);
    uint32_t size_Cr = static_cast<uint32_t>(cur_T.TCr().size() + 1);
    uint32_t num_cells = size_Y * size_Cb * size_Cr;

    std::vector<uint8_t> local_cluster_map[3];
    local_cluster_map[0] = cluster_map;
    for (uint32_t axis : {1u, 2u}) {
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
                  cluster_map[c * num_cells + global_cell];
            }
          }
        }
      }
    }

    bool changed = true;
    for (uint32_t iter = 0; iter < max_iters && changed; iter++) {
      changed = false;
      auto optimize_axis = [&](Thresholds& thr, uint32_t axis) {
        if (thr.empty()) return;
        const auto& DC_axis = d.dc_vals[axis];
        uint32_t ax1 = (axis + 1) % 3;
        uint32_t ax2 = (axis + 2) % 3;
        uint32_t na = static_cast<uint32_t>(cur_T.T[axis].size() + 1);
        UpdateMaps(axis, cur_T.T[axis], cur_T.T[ax1], cur_T.T[ax2]);
        const auto& sorted_blocks = d.dc_sorted_blocks[axis];
        const auto& block_off = d.dc_block_offsets[axis];
        uint32_t MAX_KEY_H = d.num_zdcai;
        uint32_t MAX_KEY_N = static_cast<uint32_t>(kZeroDensityContextCount);

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

          auto apply_delta = [&](uint32_t old_cl, uint32_t new_cl,
                                 uint32_t h_bin, uint32_t zdc_bin,
                                 uint32_t freq) -> int64_t {
            int64_t delta = 0;
            uint32_t h_old = old_cl * MAX_KEY_H + h_bin;
            uint32_t h_new = new_cl * MAX_KEY_H + h_bin;
            uint32_t old_freq = scratch_hist_h[h_old];
            uint32_t new_freq = old_freq - freq;
            delta -= d.ftab[new_freq] - d.ftab[old_freq];
            scratch_hist_h[h_old] = new_freq;
            old_freq = scratch_hist_h[h_new];
            new_freq = old_freq + freq;
            delta -= d.ftab[new_freq] - d.ftab[old_freq];
            scratch_hist_h[h_new] = new_freq;
            if (old_freq == 0) cluster_touched_h.push_back(h_new);

            uint32_t N_old = old_cl * MAX_KEY_N + zdc_bin;
            uint32_t N_new = new_cl * MAX_KEY_N + zdc_bin;
            old_freq = scratch_hist_N[N_old];
            new_freq = old_freq - freq;
            delta += d.ftab[new_freq] - d.ftab[old_freq];
            scratch_hist_N[N_old] = new_freq;
            old_freq = scratch_hist_N[N_new];
            new_freq = old_freq + freq;
            delta += d.ftab[new_freq] - d.ftab[old_freq];
            scratch_hist_N[N_new] = new_freq;
            if (old_freq == 0) cluster_touched_N.push_back(N_new);
            return delta;
          };

          auto apply_slice = [&](ptrdiff_t slice, bool upward) -> int64_t {
            int64_t cost_change = 0;
            uint32_t blk_lo = block_off[static_cast<size_t>(slice)];
            uint32_t blk_hi = block_off[static_cast<size_t>(slice + 1)];
            for (uint32_t bi = blk_lo; bi < blk_hi; ++bi) {
              uint32_t b = sorted_blocks[bi];
              uint32_t dc_ax1_val = d.block_dc[ax1][b];
              uint32_t dc_ax2_val = d.block_dc[ax2][b];
              uint32_t ci = static_cast<uint32_t>(ax1_row[dc_ax1_val] +
                                                  ax2_col[dc_ax2_val]);
              for (uint32_t c = 0; c < kNumCh; ++c) {
                uint32_t bc = d.ToChBlock(b, c);
                uint32_t ci_base = c * num_cells + ci * na;
                uint32_t old_cl =
                    upward ? local_cluster_map[axis][ci_base + thr_ind + 1]
                           : local_cluster_map[axis][ci_base + thr_ind];
                uint32_t new_cl =
                    upward ? local_cluster_map[axis][ci_base + thr_ind]
                           : local_cluster_map[axis][ci_base + thr_ind + 1];
                if (old_cl == new_cl) continue;
                for (uint32_t pi = d.block_offsets[c][bc];
                     pi < d.block_offsets[c][bc + 1]; ++pi) {
                  uint32_t bin = d.block_bins[c][pi];
                  uint32_t zdc = bin >> 11;
                  uint32_t h_bin = d.compact_map_h[bin];
                  cost_change += apply_delta(old_cl, new_cl, h_bin, zdc, 1);
                }
              }
            }
            return cost_change;
          };

          scratch_hist_h = cluster_hist_h;
          scratch_hist_N = cluster_hist_N;
          int64_t current_cost = base_cost;
          for (ptrdiff_t idx = cur_idx - 1; idx > lo_edge; --idx) {
            current_cost += apply_slice(idx, false);
            if (current_cost < best_cost) {
              best_cost = current_cost;
              best_idx = idx;
            }
          }

          scratch_hist_h = cluster_hist_h;
          scratch_hist_N = cluster_hist_N;
          current_cost = base_cost;
          for (ptrdiff_t idx = cur_idx + 1; idx < hi_edge; ++idx) {
            current_cost += apply_slice(idx - 1, true);
            if (current_cost < best_cost) {
              best_cost = current_cost;
              best_idx = idx;
            }
          }

          if (best_idx != cur_idx) {
            scratch_hist_h = cluster_hist_h;
            scratch_hist_N = cluster_hist_N;
            if (best_idx < cur_idx) {
              for (ptrdiff_t idx = cur_idx - 1; idx >= best_idx; --idx)
                apply_slice(idx, false);
            } else {
              for (ptrdiff_t idx = cur_idx + 1; idx <= best_idx; ++idx)
                apply_slice(idx - 1, true);
            }
            thr[thr_ind] = DC_axis[static_cast<size_t>(best_idx)];
            base_cost = best_cost;
            std::swap(cluster_hist_h, scratch_hist_h);
            std::swap(cluster_hist_N, scratch_hist_N);
            changed = true;
          }
        }
      };

      optimize_axis(cur_T.TY(), 0);
      optimize_axis(cur_T.TCb(), 1);
      optimize_axis(cur_T.TCr(), 2);
    }
    return {cur_T, base_cost};
  }
};

Status JPEGOptData::BuildFromJPEG(const jpeg::JPEGData& jpeg_data,
                                  const std::array<int, 3>& jpeg_c_map,
                                  ThreadPool* pool) {
  for (uint32_t c = 0; c < kNumCh; ++c) {
    uint32_t jpeg_c = static_cast<uint32_t>(jpeg_c_map[c]);
    block_grid_w[c] = jpeg_data.components[jpeg_c].width_in_blocks;
    block_grid_h[c] = jpeg_data.components[jpeg_c].height_in_blocks;
    num_blocks[c] = block_grid_w[c] * block_grid_h[c];
  }

  {
    auto ac_cnt = make_unique<ACCounts>();
    JXL_RETURN_IF_ERROR(CountDCAC(jpeg_data, jpeg_c_map, pool, *ac_cnt));

    JXL_RETURN_IF_ERROR(CollectACBins(jpeg_data, jpeg_c_map, pool, *ac_cnt));
  }

  JXL_RETURN_IF_ERROR(FinalizeSpatialIndexing(pool));

  JXL_RETURN_IF_ERROR(GenerateRLEStream(pool));

  return true;
}

}  // namespace

Status OptimizeJPEGContextMap(const jpeg::JPEGData& jpeg_data,
                              const FrameHeader& frame_header,
                              PassesEncoderState* enc_state, ThreadPool* pool,
                              bool verbose) {
  // Keep optimization axes in JPEG component order (Y, Cb, Cr).
  const auto jpeg_c_map = (jpeg_data.components.size() == 1)
                              ? std::array<int, 3>{{0, 0, 0}}
                              : std::array<int, 3>{{0, 1, 2}};

  auto opt_data = std::make_shared<JPEGOptData>();
  JXL_RETURN_IF_ERROR(opt_data->BuildFromJPEG(jpeg_data, jpeg_c_map, pool));

  auto facts = opt_data->MaximalFactorizations();
  if (facts.empty()) return true;
  if (verbose) {
    printf("Searching %zu maximal factorizations\n", facts.size());
  }

  int64_t best_cost = std::numeric_limits<int64_t>::max();
  ThresholdSet best_thr;
  ContextMap best_ctx;
  std::mutex mu;

  // TODO: use array of PartitioningCtx per thread instead of unique_ptr
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, static_cast<uint32_t>(facts.size()), ThreadPool::NoInit,
      [&](uint32_t idx, size_t) -> Status {
        const auto t0 = std::chrono::steady_clock::now();
        uint32_t a = std::get<0>(facts[idx]);
        uint32_t b = std::get<1>(facts[idx]);
        uint32_t c = std::get<2>(facts[idx]);
        auto ctx = make_unique<PartitioningCtx>(opt_data);
        ThresholdSet init;
        init.T[0] = opt_data->InitThresh(0, a);
        init.T[1] = opt_data->InitThresh(1, b);
        init.T[2] = opt_data->InitThresh(2, c);

        auto opt_result =
            ctx->OptimizeThresholds(init, ctx->ac_stream(), kMTarget);

        PartitioningCtx::Clustering cl_result;
        JXL_ASSIGN_OR_RETURN(
            cl_result, ctx->ClusterContexts(opt_result.second, 16, nullptr));
        ContextMap& cluster_map = cl_result.ctx_map;

        auto refine_result =
            ctx->RefineClustered(opt_result.second, cluster_map);
        int64_t refined_cost = refine_result.second;
        ThresholdSet refined_thr = refine_result.first;

        int64_t nz_cost = ctx->NZClusteredCost(refined_thr, cluster_map);
        int64_t total_cost = refined_cost + nz_cost;

        std::lock_guard<std::mutex> lock(mu);
        if (verbose) {
          const double dt = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
          printf(
              "(%u,%u,%u) unclustered=%.2f clustered=%.2f refined=%.2f "
              "nz=%.2f total=%.2f [%.2fs]\n",
              a, b, c, static_cast<double>(opt_result.first) / kFScale,
              static_cast<double>(cl_result.clustered_cost) / kFScale,
              static_cast<double>(refined_cost) / kFScale,
              static_cast<double>(nz_cost) / kFScale,
              static_cast<double>(total_cost) / kFScale, dt);
        }
        if (total_cost < best_cost) {
          best_cost = total_cost;
          best_thr = refined_thr;
          best_ctx = cluster_map;
        }
        return true;
      },
      "JpegCtxOpt"));

  auto& bcm = enc_state->shared.block_ctx_map;
  bcm.dc_thresholds[1].clear();
  bcm.dc_thresholds[0].clear();
  bcm.dc_thresholds[2].clear();

  for (int16_t t : best_thr.TY()) bcm.dc_thresholds[1].push_back(t - 1);
  for (int16_t t : best_thr.TCb()) bcm.dc_thresholds[0].push_back(t - 1);
  for (int16_t t : best_thr.TCr()) bcm.dc_thresholds[2].push_back(t - 1);

  size_t na_Y = best_thr.TY().size() + 1;
  size_t na_Cb = best_thr.TCb().size() + 1;
  size_t na_Cr = best_thr.TCr().size() + 1;
  bcm.num_dc_ctxs = na_Y * na_Cb * na_Cr;

  JXL_ENSURE(bcm.num_dc_ctxs <= 64);
  JXL_ENSURE(na_Y <= 16 && na_Cb <= 16 && na_Cr <= 16);

  bcm.ctx_map.assign(3 * kNumOrders * bcm.num_dc_ctxs, 0);
  for (size_t cell = 0; cell < bcm.num_dc_ctxs; cell++) {
    for (size_t ord = 0; ord < kNumOrders; ord++) {
      for (size_t c = 0; c < 3; c++) {
        bcm.ctx_map[c * kNumOrders * bcm.num_dc_ctxs + ord * bcm.num_dc_ctxs +
                    cell] = best_ctx[c * bcm.num_dc_ctxs + cell];
      }
    }
  }
  bcm.num_ctxs = *std::max_element(bcm.ctx_map.begin(), bcm.ctx_map.end()) + 1;
  JXL_ENSURE(bcm.num_ctxs <= 16);

  if (verbose) {
    printf("(%zu, %zu, %zu)\n", na_Y, na_Cb, na_Cr);
    // JPEG XL thresholds are `<=` value
    printf("TY: { ");
    for (int t : best_thr.TY()) printf("%d, ", t - 1);
    printf(" }\n");
    printf("TCb: { ");
    for (int t : best_thr.TCb()) printf("%d, ", t - 1);
    printf(" }\n");
    printf("TCr: { ");
    for (int t : best_thr.TCr()) printf("%d, ", t - 1);
    printf(" }\n");
  }

  return true;
}

}  // namespace jxl
