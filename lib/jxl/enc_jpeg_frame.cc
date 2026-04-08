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

// This file implements context-map optimization for JPEG-source images being
// re-encoded into JPEG XL. The single public entry point is
// `OptimizeJPEGContextMap()`, which optimizes per-channel DC thresholds
// and AC context clustering to minimise the entropy-coded size
// of the JPEG AC DCT coefficients.
//
// The pipeline is:
//   1. `JPEGOptData::BuildFromJPEG` extracts per-block DC/AC statistics from
//      the source JPEG and builds a bin-indexed AC stream.
//   2. Enumerate "maximal factorisations" of the (Y, Cb, Cr) DC-threshold
//      space; each factorisation defines how many DC intervals each channel
//      is split into.
//   3. For every candidate factorisation (optionally pre-filtered by a cheap
//      ranking pass), `PartitioningCtx::OptimizeThresholds` refines the DC
//      thresholds via iterative greedy descent, then
//      `PartitioningCtx::ClusterContexts` merges similar contexts using
//      entropy-cost-guided agglomerative clustering.
//   4. The candidate with the lowest total cost (entropy + histogram signalling
//      overhead) is selected and written into the BlockCtxMap consumed by the
//      rest of the encoder.
//
// All calculations (besides formation of `f(n) = n*log2(n)` lookup) are done in
// integer arithmetic to avoid floating point inaccuracies. Counters are
// generally `uint32_t`, that is enough for JPEG.
//
// General notation:
// `N` - number of 8x8 blocks of an image component
// `M` - number of distinct DC values along a component axis
// `M_eff` - number of distinct DC buckets along a component axis,
//           use of full `M` for large values is too slow by cache misses
// `ci` - cell index in 2D array of cells perpendicular to current axis,
//        `ci < 32` as axis search is meaningful with at least 2 cells per axis
// `K` - number of intervals (cells) along an axis
// `l` - left bound of interval (inclusive)
// `n` - right bound of interval (inclusive)
// `c` - component index
// `zdc` - `jxl::ZeroDensityContext` of an AC coefficient in a block
// `czdc` - `(c,zdc)` = `(c<<9|zdc)`, [0, 3*512)
// `ai` - AC index = value of AC coefficient + `kDCTOff`
//
// Key internal data structures:
//   - `CompactHistogram`    — sparse histogram for efficient incremental
//   updates.
//   - `ThresholdSet`        — per-channel (Y, Cb, Cr) DC threshold arrays.
//   - `JPEGCtxEffortParams` — effort-level knobs derived from the encoder speed
//                             tier (number of candidates, refinement
//                             iterations, etc.).

namespace {

///////////////
// Constants //
///////////////

// ---------- `f(n) = n*log2(n)` lookup, fixed-point ----------
// Scale is choosen to keep `f(freq) * kFScale` in `int64_t`.
// Overflow bound: max number of blocks `N = 2^26` for max JPEG image,
// max 4 AC positions in bin (see `kCoeffFreqContext`), then max bin `freq` is
// 2^28 and max `f(freq) = 2^28 * 28`, fixed point `f(freq) * kFScale = 2^53 *
// 28` fits in `int64_t` (and there is still room for clustering that can
// increase `n` three-fold).
constexpr int64_t kFScale = 1LL << 25;

// Partitioning limits are defined by the JPEG XL standard I.2.2.
// Max number of DC cells in 3D DC space.
constexpr int kMaxCells = 64;
// Each DC threshold count is written with 4 bits.
constexpr uint32_t kMaxIntervals = 16;
// Max number of clusters in context map.
constexpr int kMaxClusters = 16;

// JPEG DCT coefficients range: [-1024, 1024)
constexpr int kDCTRange = 2048;
constexpr int kDCTOff = 1024;  // shift into [0, 2048)
// Number of AC positions in a block.
constexpr int kNumPos = 63;
// Number of channels.
constexpr int kNumCh = 3;

// Target number of rarefied DC thresholds to reduce size of cost matrix
// for Knuth DP optimization of Dc axis partitioning.
constexpr uint32_t kMTarget = 256;

// `nz` is number of nonzero AC coefficients in a block.
// Number of contexts for prediction of `nz`, one less than `kNonZeroBuckets`.
constexpr uint32_t kJPEGNonZeroBuckets = 36;
// Number of possible values for `nz` [0, 64).
constexpr uint32_t kJPEGNonZeroRange = 64;
// Number of bins in all histograms for `nz` per context.
constexpr uint32_t kNZHistogramsSize = kJPEGNonZeroBuckets * kJPEGNonZeroRange;

// Vector of DC thresholds for a channel.
using Thresholds = std::vector<int16_t>;
// AC coefficient entry in the event stream.
using ACEntry = uint32_t;
// Context map.
using ContextMap = std::vector<uint8_t>;
// Factorizations of DC thresholds into number of intervals per channel.
using Factorizations = std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>;

JXL_INLINE int CountrZero64(uint64_t x) {
#if JXL_COMPILER_MSVC
  unsigned long idx;
  return _BitScanForward64(&idx, x) ? static_cast<int>(idx) : 64;
#else
  return x ? __builtin_ctzll(x) : 64;
#endif
}

double bit_cost(int64_t cost) { return static_cast<double>(cost) / kFScale; }

// Set of DC thresholds for each channel.
struct ThresholdSet {
  std::array<Thresholds, kNumCh> T;
  Thresholds& TY() { return T[0]; }
  const Thresholds& TY() const { return T[0]; }
  Thresholds& TCb() { return T[1]; }
  const Thresholds& TCb() const { return T[1]; }
  Thresholds& TCr() { return T[2]; }
  const Thresholds& TCr() const { return T[2]; }
};

// Knuth-optimized solver for contiguous 1D partitioning on a diff-form cost
// matrix. Callers fill `costs` in diff form and then call `Solve`.
struct KnuthPartitionSolver {
  // Lazily materialized `M * M` interval cost matrix.
  // In diff form, `costs[n * M + l]` is the incremental contribution needed to
  // recover the total cost of putting ranks `[l..n]` into one cell.
  std::vector<int64_t> costs;

  // DP tables for row-wise optimal partitioning.
  std::vector<int64_t> DP_prev;
  std::vector<int64_t> DP_curr;
  std::vector<uint16_t> split_table;
  std::vector<uint16_t> row_done;
  std::vector<int64_t> row_cum;

  explicit KnuthPartitionSolver(size_t max_m = 0)
      : DP_prev(max_m, 0),
        DP_curr(max_m, 0),
        row_done(max_m, 0),
        row_cum(max_m, 0) {}

  void ResetCosts(size_t cost_size) {
    if (costs.size() < cost_size) {
      costs.assign(cost_size, 0);
    } else {
      std::fill(costs.begin(), costs.begin() + cost_size, 0);
    }
  }

  // Lazily materialise `costs[n * M + l]` = entropy of merging ranks
  // `[l..n]` into one cell. The matrix is stored in diff form after the cost
  // build pass: `costs[n * M + v]` holds the diff contributed by rank `v` to
  // row `n`.
  //
  // Recurrence: `cost[l..n] = cost[l..n-1] + Σ_{v=l}^{n} diff(n, v)`.
  // That is, merging rank `n` into an interval already covering `[l..n-1]`
  // adds the total diff accumulated at row `n` from column `l` onward.
  // `row_cum[n]` caches the running prefix sum of diffs already scanned in
  // row `n`; `row_done[n]` is the last column fully converted (UINT16_MAX
  // means none yet, used as a sentinel to distinguish "not started" from
  // "column 0 done").
  void EnsureDiffCost(uint32_t M, uint32_t l, uint32_t n) {
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

  // Accessor for materialized interval cost.
  int64_t GetDiffCost(uint32_t M, uint32_t l, uint32_t n) {
    EnsureDiffCost(M, l, n);
    return costs[n * M + l];
  }

  // Implements Knuth’s dynamic programming optimization to solve
  // the 1D optimal partitioning problem.
  // `O(K × M)` using both monotonicity bounds via right-to-left `n`
  // traversal. Works on diff-form `costs` and lazily materialises
  // `cost[l..n]` entries on demand while DP requests them.
  Thresholds Solve(const Thresholds& vals,
                   const std::vector<uint16_t>& k_to_dc0, uint32_t K,
                   uint32_t M_eff) {
    if (M_eff <= 1 || K <= 1) return {};

    split_table.assign(K * M_eff, 0);
    if (row_done.size() < M_eff) row_done.resize(M_eff);
    if (row_cum.size() < M_eff) row_cum.resize(M_eff);
    std::fill(row_done.begin(), row_done.begin() + M_eff, UINT16_MAX);
    std::fill(row_cum.begin(), row_cum.begin() + M_eff, 0);
    if (DP_prev.size() < M_eff) DP_prev.resize(M_eff);
    if (DP_curr.size() < M_eff) DP_curr.resize(M_eff);

    for (uint32_t n = 0; n < M_eff; ++n) {
      DP_prev[n] = GetDiffCost(M_eff, 0, n);
    }

    constexpr int64_t INF = std::numeric_limits<int64_t>::max();
    for (uint32_t k = 1; k < K; ++k) {
      std::fill(DP_curr.begin(), DP_curr.begin() + M_eff, INF);
      uint16_t* split_curr = split_table.data() + k * M_eff;
      const uint16_t* split_prev = split_curr - M_eff;
      uint32_t s_max = M_eff - 1;
      for (uint32_t n_plus_1 = M_eff; n_plus_1 > 0; --n_plus_1) {
        uint32_t n = n_plus_1 - 1;
        uint32_t s_min = std::max<uint32_t>(split_prev[n], k);
        if (s_min > s_max) continue;
        uint32_t best_s = s_min;
        for (uint32_t s = s_min; s <= s_max; ++s) {
          int64_t val = DP_prev[s - 1] + GetDiffCost(M_eff, s, n);
          if (val < DP_curr[n]) {
            DP_curr[n] = val;
            best_s = s;
          }
        }
        split_curr[n] = static_cast<uint16_t>(best_s);
        s_max = std::min(best_s, n - 1);
      }
      DP_prev.swap(DP_curr);
    }

    Thresholds T;
    T.reserve(K - 1);
    uint32_t v = M_eff - 1;
    for (uint32_t k = K - 1; k > 0; --k) {
      uint32_t s = split_table[k * M_eff + v];
      T.push_back(vals[k_to_dc0[s]]);
      v = s - 1;
    }
    std::reverse(T.begin(), T.end());
    return T;
  }
};

// Invalid position in `pos_in_used`.
constexpr uint32_t kInvalidCompactH = std::numeric_limits<uint32_t>::max();

// Compact histogram for efficient incremental updates.
// Used for AC coefficient histograms as they are large and sparse.
struct CompactHistogram {
  std::vector<uint32_t> counts;
  std::vector<uint32_t> used_ids;
  std::vector<uint32_t> pos_in_used;

  CompactHistogram() = default;
  explicit CompactHistogram(size_t size)
      : counts(size, 0), used_ids(), pos_in_used(size, kInvalidCompactH) {}

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

  uint32_t at(uint32_t id) const { return counts[id]; }
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
};

// Dense histogram with a `CompactHistogram`-like interface for small fixed
// alphabets. Unlike `CompactHistogram`, `empty()` and `Clear()` scan or touch
// the full storage, which is acceptable for these bounded-size tables.
template <size_t Size>
struct DenseHistogram {
  using Array = std::array<uint32_t, Size>;
  using iterator = typename Array::iterator;
  using const_iterator = typename Array::const_iterator;

  DenseHistogram() = default;
  DenseHistogram(const DenseHistogram&) = default;
  DenseHistogram& operator=(const DenseHistogram&) = default;
  DenseHistogram(DenseHistogram&&) = default;
  DenseHistogram& operator=(DenseHistogram&&) = default;

  std::array<uint32_t, Size> counts = {};

  bool empty() const {
    for (uint32_t freq : counts) {
      if (freq != 0) return false;
    }
    return true;
  }

  constexpr size_t size() const { return counts.size(); }

  uint32_t Get(uint32_t id) const { return counts[id]; }

  void Add(uint32_t id, uint32_t value = 1) { counts[id] += value; }

  void Subtract(uint32_t id, uint32_t value = 1) {
    uint32_t& freq = counts[id];
    JXL_DASSERT(freq >= value);
    if (freq < value) return;
    freq -= value;
  }

  void AddFrom(const DenseHistogram& other) {
    for (size_t i = 0; i < Size; ++i) counts[i] += other.counts[i];
  }

  void Clear() { counts.fill(0); }

  void fill(uint32_t value) { counts.fill(value); }

  uint32_t* data() { return counts.data(); }
  const uint32_t* data() const { return counts.data(); }

  uint32_t& operator[](size_t idx) { return counts[idx]; }
  const uint32_t& operator[](size_t idx) const { return counts[idx]; }

  iterator begin() { return counts.begin(); }
  iterator end() { return counts.end(); }
  const_iterator begin() const { return counts.begin(); }
  const_iterator end() const { return counts.end(); }

  void swap(DenseHistogram& other) { counts.swap(other.counts); }
};

using DenseNHistogram = DenseHistogram<kZeroDensityContextCount>;
using DenseNHistogramSet = std::vector<DenseNHistogram>;
using DenseNZPredHistogram = DenseHistogram<kJPEGNonZeroBuckets>;
using DenseNZPredHistogramSet = std::vector<DenseNZPredHistogram>;
using DenseNZHistogram = DenseHistogram<kNZHistogramsSize>;
using DenseNZHistogramSet = std::vector<DenseNZHistogram>;
using CompactHistogramSet = std::vector<CompactHistogram>;

JXL_INLINE uint32_t NZHistogramIndex(uint32_t pb, uint32_t nz_count) {
  JXL_DASSERT(pb < kJPEGNonZeroBuckets);
  JXL_DASSERT(nz_count < kJPEGNonZeroRange);
  return pb * kJPEGNonZeroRange + nz_count;
}

// Immutable data for JPEG recompression optimization.
struct JPEGOptData {
  // Histogram of AC coefficients per channel, position and value.
  using ACCounts =
      std::array<std::array<std::array<uint32_t, kDCTRange>, kNumPos>, kNumCh>;

  // Fixed-point entropy table: `f(n) = n*log2(n)*kFScale`.
  std::vector<int64_t> ftab;

  uint32_t channels;
  // Number of 8x8 blocks in image for each component.
  uint32_t num_blocks[kNumCh];
  // Block grid dimensions per component.
  uint32_t block_grid_w[kNumCh];
  uint32_t block_grid_h[kNumCh];
  // Max width and height of the block grid.
  uint32_t w_max;
  uint32_t h_max;
  // Vertical and horizontal subsampling factors.
  uint32_t ss_y[kNumCh];
  uint32_t ss_x[kNumCh];
  // DC values active in image for each component.
  Thresholds DC_vals[kNumCh];
  // Counts of each DC value.
  uint32_t DC_cnt[kNumCh][kDCTRange];
  // Index of each DC value in `DC_vals`.
  uint16_t DC_idx_LUT[kNumCh][kDCTRange];

  // Number of `(zdc,ai)` bins active in image - `CompactHistogram` size.
  uint32_t num_zdcai;
  // Map from `(zdc,ai)` to compact index.
  std::vector<uint32_t> compact_map_h;
  // Map from compact index to `(zdc,ai)`.
  std::vector<uint32_t> dense_to_zdcai;

  // Run-length-encoded AC data, sorted by `(bin, dc0, dc1, dc2)`.
  // 32-bit packed format; see `GenerateRLEStream` for layout details.
  std::vector<ACEntry> AC_stream;

  // AC events of consequitive blocks per component.
  std::array<std::vector<uint32_t>, kNumCh> block_bins;
  // Indices into `block_bins`, separating consequent blocks data,
  // size `block_grid_h[c] * block_grid_w[c]`.
  std::array<std::vector<uint32_t>, kNumCh> block_offsets;
  // Block nonzero number and nonzero prediction context,
  // size `block_grid_h[c] * block_grid_w[c]`.
  std::array<std::vector<uint8_t>, kNumCh> block_nonzeros;
  std::array<std::vector<uint8_t>, kNumCh> block_nz_pred_bucket;
  // DC indices of blocks of components (DC of the block component),
  // size `block_grid_h[c] * block_grid_w[c]`.
  std::array<std::vector<uint16_t>, kNumCh> block_DC_idx;
  // Coordinates of component blocks, sorted by DC index,
  // size `block_grid_h[c] * block_grid_w[c]`.
  // `y` in 16 MSB, `x` in 16 LSB.
  std::array<std::vector<uint32_t>, kNumCh> DC_sorted_blocks;
  // Indices into `dc_sorted_blocks, separating different DC indices,
  // size - number of active DC values `M_comp`.
  std::array<std::vector<uint32_t>, kNumCh> DC_block_offsets;

  // Precomputed values of `n*log2(n)*kFScale` for `n` up to `max_n`.
  void InitFTab(size_t max_n) {
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
  int64_t NZFTab(uint32_t n) const {
    if (n < ftab.size()) return ftab[n];
    const double nd = static_cast<double>(n);
    return static_cast<int64_t>(std::llround(nd * std::log2(nd) * kFScale));
  }

  // Map sparse `zdc_ai` to dense index `[0, num_zdcai)`.
  uint32_t CompactHBin(uint32_t zdc_ai) const {
    JXL_DASSERT(zdc_ai < compact_map_h.size());
    if (zdc_ai >= compact_map_h.size()) return kInvalidCompactH;
    uint32_t dense = compact_map_h[zdc_ai];
    JXL_DASSERT(dense != kInvalidCompactH);
    return dense;
  }

  // Threshold initialization for one axis via the same contiguous-partition DP
  // used in later refinement, but with a cheaper 1D surrogate cost:
  // `cost[l..n] = f(sum_{i=l}^n DC_cnt[i])`.
  // The interval-cost matrix is written in the same diff form consumed by the
  // shared Knuth solver, so this stage reuses the exact same partition backend
  // as the AC-driven optimization path.
  Thresholds InitThresh(int axis, uint32_t target_intervals) const {
    if (target_intervals <= 1) return {};
    const Thresholds& dc_vals = DC_vals[axis];
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
      prefix[i + 1] =
          prefix[i] + DC_cnt[axis][static_cast<uint32_t>(dc_vals[i] + kDCTOff)];
    }

    KnuthPartitionSolver solver(M);
    solver.ResetCosts(static_cast<size_t>(M) * M);
    for (uint32_t n = 0; n < M; ++n) {
      int64_t prev_base = 0;
      for (uint32_t l = 0; l <= n; ++l) {
        uint32_t total = prefix[n + 1] - prefix[l];
        int64_t base = NZFTab(total);
        if (n > 0 && l < n) {
          base -= NZFTab(prefix[n] - prefix[l]);
        }
        solver.costs[n * M + l] = base - prev_base;
        prev_base = base;
      }
    }

    std::vector<uint16_t> k_to_dc0(M);
    std::iota(k_to_dc0.begin(), k_to_dc0.end(), uint16_t{0});
    return solver.Solve(dc_vals, k_to_dc0, target_intervals, M);
  }

  // Find all factorizations `(a,b,c)` for DC thresholds.
  // Return only maximal factorizations (cannot increase any factor):
  // lower are not winning against higher factorizations for entropy.
  Factorizations MaximalFactorizations() const {
    uint32_t cap0 = std::max(1u, static_cast<uint32_t>(DC_vals[0].size()));
    uint32_t cap1 = std::max(1u, static_cast<uint32_t>(DC_vals[1].size()));
    uint32_t cap2 = std::max(1u, static_cast<uint32_t>(DC_vals[2].size()));
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

  // -------------------------------------------------------//
  // JPEG processing: each pass sweeps all image data once. //
  // -------------------------------------------------------//

  // ------------------------------------------------------//
  // Pass 1: Count DC + AC values (parallel by component). //
  // ------------------------------------------------------//
  StatusOr<std::unique_ptr<ACCounts>> CountDCAC(const jpeg::JPEGData& jpeg_data,
                                                ThreadPool* pool) {
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

  using ScanOrder = std::array<std::array<uint8_t, kNumPos>, kNumCh>;
  // JPEG XL high effort mode scans AC coefficients in descending nonzero
  // abundance (ties keep zigzag order).
  static ScanOrder BuildACScanOrder(const ACCounts& ac_cnt) {
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

  // -------------------------------------------------------//
  // Pass 2: Collect AC bins, DC indices and `nz` per block //
  //         (parallel by comp).                            //
  // -------------------------------------------------------//
  Status BuildBlockOptData(const jpeg::JPEGData& jpeg_data, ThreadPool* pool,
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
              block_offsets[c][bc] =
                  static_cast<uint32_t>(block_bins[c].size());

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
                uint32_t ai = static_cast<uint32_t>(coeff + kDCTOff);
                block_bins[c].push_back((c << 20) | (zdc << 11) | ai);
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

  // ------------------------------------------------------------------//
  // Pass 3: Prune DC values - remove DC values with no active bins.   //
  //         Build per-axis sorted block positions (parallel by comp). //
  // ------------------------------------------------------------------//
  Status FinalizeSpatialIndexing(ThreadPool* pool) {
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
            uint32_t by = y * ss_y[c];
            uint32_t bx = x * ss_x[c];
            uint16_t dc0 = block_DC_idx[0][(by / ss_y[0]) * block_grid_w[0] +
                                           bx / ss_x[0]];
            uint16_t dc1 = block_DC_idx[1][(by / ss_y[1]) * block_grid_w[1] +
                                           bx / ss_x[1]];
            uint16_t dc2 = block_DC_idx[2][(by / ss_y[2]) * block_grid_w[2] +
                                           bx / ss_x[2]];

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

  // ---------------------------------------------------------------------//
  // Prepare AC stream via two-pass counting sort on the 22-bit bin key   //
  // `(c:2b, zdc:9b, ai:11b)` and emit on the third pass.                 //
  //                                                                      //
  // Pass 4: Find bin positions in the stream with pruned DC indices.     //
  // Pass 5: Scatter into `flat_lo` and `flat_hi` with pruned DC indices. //
  // Pass 6: Emit AC stream with run-length encoding.                     //
  // ---------------------------------------------------------------------//
  Status GenerateRLEStream(ThreadPool* pool) {
    const uint32_t BIN_N = channels << 20;
    // Split on `dc0` bit 10 (the MSB of the 11-bit index):
    // `lo = dc0 < 1024`, `hi = dc0 >= 1024`.
    // Each half packs `(dc0_low10 << 22 | dc1 << 11 | dc2)` into
    // a single `uint32_t` (10+11+11 = 32 bits), halving `flat[]` memory
    // vs the full `uint64_t` encoding.
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
          uint32_t dc0 = block_DC_idx[0][b0];
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

    // Prefix sums and active bins
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

    // `AC_stream` layout:
    //   Regular frame (bit 31 = 0):
    //     30..27  `Δdc0`   (4b, 0..15)
    //     26..16  `dc1`    (11b, absolute)
    //     15..5   `dc2`    (11b, absolute)
    //     4..0    `run-1`  (5b; 0..30 = run 1..31; 31 = long-run escape)
    //   Long-run frame (follows regular frame with `run-1 = 31`):
    //     31..0   `run`    (32b, 2^26 ≥ actual run ≥ 32)
    //   Reset frame (bit 31 = 1),
    //     emitted for the first entry, bin change, `Δdc0 > 15`:
    //     30      `ctx_changed` (1b)
    //     29      `bin_changed` (1b)
    //     28..7   `bin`      (22b = `c:2 + zdc:9 + ai:11`, used in clustering)
    //     6..0    `dc0 >> 4` (7b, coarse; fine bits recovered from next `Δdc0`)
    //
    // The layout gives max 5% stream overhead in the worst case tested,
    // dominated by the reset frames for `Δdc0 > 15`

    // --------------------------------------------------------------------
    // Pass 5: Scatter into `flat_lo` and `flat_hi` with pruned DC indices.
    // --------------------------------------------------------------------
    size_t lo_size = bin_start_lo[BIN_N];
    size_t hi_size = bin_start_hi[BIN_N];
    size_t raw_size = lo_size + hi_size;
    std::vector<uint32_t> flat_lo(lo_size);
    std::vector<uint32_t> flat_hi(hi_size);

    // Scatter pass: `dc0 & 0x3FF` drops bit 10 (already encoded by lo/hi half);
    // `dc0_base` (0 or 0x400) re-adds it during decode in `emit_half` later.
    if (channels == 1) {
      for (uint32_t b = 0; b < num_blocks[0]; ++b) {
        uint32_t dc0 = block_DC_idx[0][b];
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

            uint32_t dc0 = block_DC_idx[0][b0];
            uint32_t dc1 = block_DC_idx[1][b1];
            uint32_t dc2 = block_DC_idx[2][b2];
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

    // Sort each bin's lo/hi halves in parallel.
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

    // ------------------------------------------------
    // Pass 6: Emit AC stream with run-length encoding.
    // ------------------------------------------------
    std::vector<ACEntry> stream;
    stream.reserve(raw_size + raw_size / 16);
    uint32_t ctx_len = 0;
    uint32_t prev_bin = UINT32_MAX;
    std::array<uint32_t, kZeroDensityContextCount> zdc_total = {};

    // `compact_map_h` maps sparse `(zdc<<11|ai)` keys (20-bit, max ~1M) to a
    // dense id `[0, num_unique_h)`. Only keys that appear in `active_bins`
    // get assigned; the rest remain 0xFFFFFFFF (unused sentinel).
    // Actual number of active bins seen is ~35000.
    const uint32_t max_key_h_sparse =
        static_cast<uint32_t>(kZeroDensityContextCount * kDCTRange);
    compact_map_h.assign(max_key_h_sparse, kInvalidCompactH);
    dense_to_zdcai.clear();
    dense_to_zdcai.reserve(active_bins.size());
    num_zdcai = 0;

    uint32_t lo_pos = 0;
    uint32_t hi_pos = 0;
    for (unsigned int bin : active_bins) {
      uint32_t zdc_ai =
          bin & 0xFFFFFu;  // strip channel bits (top 2 bits of `bin`)
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

      // Stats: track per-ctx entry counts.
      // `bin >> 11` extracts `(c<<9|zdc)`, which identifies the context.
      bool new_ctx =
          (prev_bin == UINT32_MAX) || ((bin >> 11) != (prev_bin >> 11));
      if (new_ctx) {
        if (prev_bin != UINT32_MAX) {
          zdc_total[(prev_bin >> 11) & 0x1FFu] += ctx_len;
        }
        ctx_len = 0;
      }
      ctx_len += (end_lo - start_lo) + (end_hi - start_hi);

      // `bin_change` / `ctx_change` mark reset-frame flags (bits 29/30).
      // `bin_change` is false only for the very first bin in the entire stream.
      bool bin_change = (prev_bin != UINT32_MAX);
      bool ctx_change = bin_change && ((bin >> 11) != (prev_bin >> 11));

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

          // Reset frame: emitted on first entry, bin change, or `Δdc0 > 15`.
          // `cur_dc0` is snapped to the coarsest 16-aligned value ≤ dc0
          // so the 4-bit `Δdc0` in the next normal frame is always in [0,15].
          if (first_in_bin || (dc0 - cur_dc0 > 15u)) {
            stream.push_back(
                (1u << 31) |
                (static_cast<uint32_t>(ctx_change && first_in_bin) << 30) |
                (static_cast<uint32_t>(bin_change && first_in_bin) << 29) |
                (bin << 7) | (dc0 >> 4));
            cur_dc0 = (dc0 >> 4) << 4;
          }

          // Normal frame: `Δdc0` in bits 30..27, `dc1` in bits 26..16,
          // `dc2` in bits 15..5.
          uint32_t delta_dc0 = dc0 - cur_dc0;
          uint32_t header = (delta_dc0 << 27) | (dc1 << 16) | (dc2 << 5);
          if (run <= 31) {
            stream.push_back(header | (run - 1));  // `run-1` fits in 5 bits
          } else {
            stream.push_back(header | 0x1Fu);  // escape: `run-1 == 31`
            stream.push_back(run);  // long-run frame carries actual run
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

    // Clustering may merge contexts from different channels, so
    // `hist_N[merged][zdc]` can accumulate counts from all channels for the
    // same `zdc`. Use the actual per-`zdc` cross-channel total as the bound
    // for `ftab` size.
    uint32_t max_zdc_total =
        *std::max_element(zdc_total.begin(), zdc_total.end());
    InitFTab(max_zdc_total + 1);

    return true;
  }

  // Orchestrates the JPEG data extraction pipeline.
  Status BuildFromJPEG(const jpeg::JPEGData& jpeg_data, ThreadPool* pool);
};

struct ClusterBoundary {
  uint8_t lo;
  uint8_t hi;
};

struct AxisMaps;

// Clustering of DC cells into `ctx_map`.
struct Clustering {
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

  // Greedily merges contexts into at most `num_clusters` clusters to minimise
  // total entropy cost, then optionally continues merging while the sum of
  // entropy and signalling overhead keeps decreasing.
  //
  // **Entropy cost model**
  // For each cluster `i`:
  //   `E[i] = sum_zdc ftab[N[zdc]] - sum_id ftab[h[zdc][ai]]`
  //           + NZ analogues (`hist_nz_N / hist_nz_h`)
  // where `N[zdc]` is the count of all AC values in `zdc` context,
  // `h[zdc][ai]` is the count of AC coefficient value `ai` in that context.
  // Since entropy is convex, merging two clusters always increases `E` by
  // a non-negative amount.
  //
  // **Merge delta**
  // `merge_delta(a, b)` computes:
  //   `Δ = E(merged) − E(a) − E(b)`
  //     `= Σ_zdc [ftab[N_a+N_b] − ftab[N_a] − ftab[N_b]]`  (N-term, ≥0)
  //       `− Σ_zdc Σ_ai [ftab[h_a+h_b] − ftab[h_a] − ftab[h_b]]`  (h-term, ≥0)
  //       `+ NZ analogues`
  //
  // **Main greedy loop (phase 1)**
  // Keeps a symmetric `deltas[total_ctxs × total_ctxs]` cache. Each
  // iteration:
  //   1. Scan all `active^2/2` pairs in parallel to find the minimum-Δ merge.
  //   2. Apply the merge: combine histograms into the surviving cluster (lower
  //      id wins), set `parent[b] = a`, remove `b` from the active list.
  //   3. Recompute distances from the new merged cluster `a` to all survivors.
  // Repeats until `active_clusters == num_clusters`.
  //
  // **Overhead-aware tail (phase 2, if `overhead_aware_tail`)**
  // On small images the signalling overhead (histogram headers) can outweigh
  // entropy savings, so `num_clusters` may be too large. Phase 2 continues
  // greedy merging past `num_clusters`, each time comparing
  //   `(entropy + signalling_overhead)`
  // before and after the tentative merge. If the merge improves the total,
  // it is committed; otherwise it is rolled back via `RollbackScratch` and
  // the loop stops.
  //
  // **Finalisation**
  // `parent[]` forms a forest; path-compressed `find_cluster()` maps every
  // original context to its surviving root. `ctx_map[i]` is then the index
  // of that root within `active`, giving each context its cluster id.
  // The histogram arrays are compacted in-place to contain only the
  // `active_clusters` surviving cluster histograms, that are then used for
  // refinement.
  template <class StreamSweepFn>
  static StatusOr<Clustering> Build(const JPEGOptData& d, AxisMaps& axis_maps,
                                    const ThresholdSet& thresholds,
                                    uint32_t num_clusters,
                                    bool overhead_aware_tail, ThreadPool* pool,
                                    StreamSweepFn& stream_sweep);

  Status AgglomerativeClustering(const JPEGOptData& d, uint32_t num_clusters,
                                 bool overhead_aware_tail, ThreadPool* pool);

  // Compute the signalling overhead (header cost) for the clustered histograms.
  // This estimates the bit cost of encoding the histogram headers in the
  // bitstream, which is not included in the entropy cost computed by
  // `TotalCost()`. We use `ANSPopulationCost() - ShannonEntropy` to estimate
  // the overhead.
  // When `cutoff` is finite, returns early once the accumulated positive
  // overhead reaches that value, which is sufficient for the greedy tail merge
  // rejection test.
  StatusOr<int64_t> ComputeSignallingOverhead(
      const JPEGOptData& d,
      int64_t cutoff = std::numeric_limits<int64_t>::max()) const {
    int64_t overhead = 0;

    // Default `HybridUintConfig` for AC coefficients: (split_exponent=4,
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
        by_zdc[zdc][token] += cluster.at(id);
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
          if (overhead >= cutoff) return overhead;
        }
      }
    }

    // Process `hist_nz_h`: split by predicted bucket `pb`
    for (const auto& cluster : hist_nz_h) {
      if (cluster.empty()) continue;
      for (uint32_t pb = 0; pb < kJPEGNonZeroBuckets; ++pb) {
        uint32_t max_nz = 0;
        size_t total = 0;
        const uint32_t base = pb * kJPEGNonZeroRange;
        for (uint32_t nz_count = 0; nz_count < kJPEGNonZeroRange; ++nz_count) {
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
          if (overhead >= cutoff) return overhead;
        }
      }
    }

    return overhead;
  }

  int64_t ComputeNZCost(const JPEGOptData& d) const {
    int64_t nz_cost = 0;
    for (const auto& cl_N : hist_nz_N) {
      for (uint32_t freq : cl_N) {
        if (freq != 0) nz_cost += d.NZFTab(freq);
      }
    }
    for (const auto& cl_h : hist_nz_h) {
      for (uint32_t freq : cl_h) {
        if (freq != 0) nz_cost -= d.NZFTab(freq);
      }
    }
    return nz_cost;
  }

  // Build threshold-major boundary views for axis-local cluster lookups:
  // `(channel, thr_ind, ci) -> {cluster(k0 = thr_ind), cluster(k0 =
  // thr_ind+1)}`. `ci` enumerates the other two bucket axes in local `(k1, k2)`
  // order, so `ApplySlice` can scan all rows touched by one threshold
  // contiguously.
  std::array<std::vector<ClusterBoundary>, kNumCh> BuildLocalClusterBoundaries(
      const ThresholdSet& thresholds, uint32_t channels) const {
    std::array<std::vector<ClusterBoundary>, kNumCh> local_cluster_boundary;
    const uint32_t size_Y = static_cast<uint32_t>(thresholds.TY().size() + 1);
    const uint32_t size_Cb = static_cast<uint32_t>(thresholds.TCb().size() + 1);
    const uint32_t size_Cr = static_cast<uint32_t>(thresholds.TCr().size() + 1);
    const uint32_t num_cells = size_Y * size_Cb * size_Cr;

    for (uint32_t axis = 0; axis < channels; ++axis) {
      const uint32_t ax1 = (axis + 1) % 3;
      const uint32_t ax2 = (axis + 2) % 3;
      const uint32_t na = static_cast<uint32_t>(thresholds.T[axis].size() + 1);
      const uint32_t n1 = static_cast<uint32_t>(thresholds.T[ax1].size() + 1);
      const uint32_t n2 = static_cast<uint32_t>(thresholds.T[ax2].size() + 1);
      const uint32_t num_rows = n1 * n2;

      local_cluster_boundary[axis].assign(
          static_cast<size_t>(channels) * (na - 1) * num_rows, {});

      for (uint32_t c = 0; c < kNumCh; ++c) {
        for (uint32_t thr_ind = 0; thr_ind + 1 < na; ++thr_ind) {
          ClusterBoundary* dst = local_cluster_boundary[axis].data() +
                                 (c * (na - 1) + thr_ind) * num_rows;
          for (uint32_t k1 = 0; k1 < n1; ++k1) {
            for (uint32_t k2 = 0; k2 < n2; ++k2) {
              uint32_t bkt[3] = {};
              bkt[axis] = thr_ind;
              bkt[ax1] = k1;
              bkt[ax2] = k2;
              const uint32_t lo_global_cell =
                  (bkt[1] * size_Cr + bkt[2]) * size_Y + bkt[0];
              bkt[axis] = thr_ind + 1;
              const uint32_t hi_global_cell =
                  (bkt[1] * size_Cr + bkt[2]) * size_Y + bkt[0];
              const uint32_t ci = k1 * n2 + k2;
              dst[ci] = {ctx_map[c * num_cells + lo_global_cell],
                         ctx_map[c * num_cells + hi_global_cell]};
            }
          }
        }
      }
    }
    return local_cluster_boundary;
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

    std::array<std::vector<uint32_t>, kNumCh> old_from_new = {{{0}, {0}, {0}}};
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
            const uint32_t g_new = (Cb * new_sizes[2] + Cr) * new_sizes[0] + Y;
            new_ctx_map[new_base + g_new] = ctx_map[old_base + g_old];
          }
        }
      }
    }
    ctx_map.swap(new_ctx_map);
    return T;
  }
};

struct AgglomerativeCtx {
  struct RollbackScratch {
    DenseNHistogram hist_N_a;
    DenseNHistogram hist_N_b;
    DenseNZPredHistogram hist_nz_N_a;
    DenseNZPredHistogram hist_nz_N_b;
    DenseNZHistogram hist_nz_h_a;
    DenseNZHistogram hist_nz_h_b;
    CompactHistogram hist_h_a;
    CompactHistogram hist_h_b;
    std::vector<uint32_t> active;
  };

  Clustering& clustering;
  const JPEGOptData& d;
  ThreadPool* pool;
  CompactHistogramSet& hist_h;
  DenseNHistogramSet& hist_N;
  DenseNZHistogramSet& hist_nz_h;
  DenseNZPredHistogramSet& hist_nz_N;
  ContextMap& ctx_map;
  const uint32_t total_ctxs;
  std::vector<int64_t> E;
  std::vector<uint32_t> active;
  std::vector<uint32_t> initial_active;
  uint32_t active_clusters;
  std::vector<uint32_t> parent;
  std::vector<int64_t> deltas;
  int64_t current_entropy_cost;

  AgglomerativeCtx(Clustering& clustering, const JPEGOptData& d,
                   ThreadPool* pool)
      : clustering(clustering),
        d(d),
        pool(pool),
        hist_h(clustering.hist_h),
        hist_N(clustering.hist_N),
        hist_nz_h(clustering.hist_nz_h),
        hist_nz_N(clustering.hist_nz_N),
        ctx_map(clustering.ctx_map),
        total_ctxs(static_cast<uint32_t>(clustering.hist_N.size())),
        E(total_ctxs, 0),
        active_clusters(0),
        parent(total_ctxs),
        current_entropy_cost(0) {}

  int64_t& Delta(uint32_t cl_a, uint32_t cl_b) {
    return deltas[std::min(cl_a, cl_b) * total_ctxs + std::max(cl_a, cl_b)];
  }

  const int64_t& Delta(uint32_t cl_a, uint32_t cl_b) const {
    return deltas[std::min(cl_a, cl_b) * total_ctxs + std::max(cl_a, cl_b)];
  }

  Status InitEntropy() {
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, total_ctxs, ThreadPool::NoInit,
        [&](uint32_t i, size_t) -> Status {
          int64_t local_E = 0;
          for (uint32_t id : hist_h[i].used_ids)
            local_E -= d.ftab[hist_h[i].at(id)];
          for (uint32_t freq : hist_N[i]) local_E += d.ftab[freq];
          for (uint32_t freq : hist_nz_N[i]) local_E += d.NZFTab(freq);
          for (uint32_t freq : hist_nz_h[i]) local_E -= d.NZFTab(freq);
          E[i] = local_E;
          return true;
        },
        "InitEntropy"));
    return true;
  }

  void InitActiveClusters() {
    active.clear();
    active.reserve(total_ctxs);
    for (uint32_t i = 0; i < total_ctxs; ++i) {
      if (!hist_N[i].empty() || !hist_nz_N[i].empty()) {
        active.push_back(i);
      }
    }
    initial_active = active;
    active_clusters = static_cast<uint32_t>(active.size());
    std::iota(parent.begin(), parent.end(), 0);
  }

  int64_t MergeDelta(uint32_t cl_a, uint32_t cl_b) const {
    int64_t delta = 0;

    const CompactHistogram& hist_h_a = hist_h[cl_a];
    const CompactHistogram& hist_h_b = hist_h[cl_b];
    const CompactHistogram* iter_h = &hist_h_a;
    if (hist_h_a.used_ids.size() > hist_h_b.used_ids.size()) {
      iter_h = &hist_h_b;
    }
    for (uint32_t id : iter_h->used_ids) {
      uint32_t freq_a = hist_h_a.at(id);
      uint32_t freq_b = hist_h_b.at(id);
      if (freq_a != 0 && freq_b != 0) {
        delta -= d.ftab[freq_a + freq_b] - d.ftab[freq_a] - d.ftab[freq_b];
      }
    }
    for (size_t bin = 0; bin < hist_N[cl_a].size(); ++bin) {
      uint32_t freq_a = hist_N[cl_a][bin];
      uint32_t freq_b = hist_N[cl_b][bin];
      if (freq_a != 0 && freq_b != 0) {
        delta += d.ftab[freq_a + freq_b] - d.ftab[freq_a] - d.ftab[freq_b];
      }
    }
    for (size_t bin = 0; bin < hist_nz_h[cl_a].size(); ++bin) {
      uint32_t freq_a = hist_nz_h[cl_a][bin];
      uint32_t freq_b = hist_nz_h[cl_b][bin];
      if (freq_a != 0 && freq_b != 0) {
        delta -=
            d.NZFTab(freq_a + freq_b) - d.NZFTab(freq_a) - d.NZFTab(freq_b);
      }
    }
    for (size_t bin = 0; bin < hist_nz_N[cl_a].size(); ++bin) {
      uint32_t freq_a = hist_nz_N[cl_a][bin];
      uint32_t freq_b = hist_nz_N[cl_b][bin];
      if (freq_a != 0 && freq_b != 0) {
        delta +=
            d.NZFTab(freq_a + freq_b) - d.NZFTab(freq_a) - d.NZFTab(freq_b);
      }
    }
    return delta;
  }

  Status InitDeltas() {
    if (active_clusters <= 1) return true;
    deltas.assign(total_ctxs * total_ctxs, 0);
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, active_clusters - 1, ThreadPool::NoInit,
        [&](uint32_t i, size_t) -> Status {
          uint32_t id_i = active[i];
          for (uint32_t j = i + 1; j < active_clusters; ++j) {
            Delta(id_i, active[j]) = MergeDelta(id_i, active[j]);
          }
          return true;
        },
        "MergeDelta"));
    return true;
  }

  Status FindBestMerge(size_t* best_i, size_t* best_j, int64_t* best_delta) {
    *best_delta = std::numeric_limits<int64_t>::max();
    *best_i = 0;
    *best_j = 1;
    std::mutex best_mtx;

    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, active_clusters - 1, ThreadPool::NoInit,
        [&](uint32_t i, size_t) -> Status {
          uint32_t id_i = active[i];
          size_t local_best_j = i + 1;
          int64_t local_best_diff = Delta(id_i, active[local_best_j]);
          for (size_t j = i + 2; j < active_clusters; ++j) {
            int64_t diff = Delta(id_i, active[j]);
            if (diff < local_best_diff) {
              local_best_diff = diff;
              local_best_j = j;
            }
          }
          std::lock_guard<std::mutex> lock(best_mtx);
          if (local_best_diff < *best_delta) {
            *best_delta = local_best_diff;
            *best_i = i;
            *best_j = local_best_j;
          }
          return true;
        },
        "FindBestMerge"));
    return true;
  }

  std::pair<uint32_t, uint32_t> ApplyMerge(size_t best_i, size_t best_j,
                                           int64_t best_delta) {
    uint32_t a_id = active[best_i];
    uint32_t b_id = active[best_j];
    E[a_id] += E[b_id] + best_delta;
    hist_N[a_id].AddFrom(hist_N[b_id]);
    hist_h[a_id].AddFrom(hist_h[b_id]);
    hist_nz_N[a_id].AddFrom(hist_nz_N[b_id]);
    hist_nz_h[a_id].AddFrom(hist_nz_h[b_id]);
    hist_N[b_id].Clear();
    hist_h[b_id].Clear();
    hist_nz_N[b_id].Clear();
    hist_nz_h[b_id].Clear();
    parent[b_id] = a_id;
    std::swap(active[best_j], active.back());
    active.pop_back();
    --active_clusters;
    return {a_id, b_id};
  }

  Status UpdateDistances(uint32_t a_id) {
    if (active_clusters <= 1) return true;
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, active_clusters, ThreadPool::NoInit,
        [&](uint32_t k, size_t) -> Status {
          if (active[k] != a_id)
            Delta(a_id, active[k]) = MergeDelta(a_id, active[k]);
          return true;
        },
        "UpdateDist"));
    return true;
  }

  void SaveRollback(RollbackScratch* rollback, uint32_t a_id, uint32_t b_id) {
    rollback->active = active;
    rollback->hist_N_a = hist_N[a_id];
    rollback->hist_h_a = hist_h[a_id];
    rollback->hist_nz_N_a = hist_nz_N[a_id];
    rollback->hist_nz_h_a = hist_nz_h[a_id];
    rollback->hist_N_b.Clear();
    rollback->hist_nz_N_b.Clear();
    rollback->hist_nz_h_b.Clear();
    rollback->hist_N_b.swap(hist_N[b_id]);
    rollback->hist_nz_N_b.swap(hist_nz_N[b_id]);
    rollback->hist_nz_h_b.swap(hist_nz_h[b_id]);
    rollback->hist_h_b = CompactHistogram();
    rollback->hist_h_b.swap(hist_h[b_id]);
  }

  void RestoreRollback(const RollbackScratch& rollback, uint32_t a_id,
                       uint32_t b_id, int64_t old_Ea, uint32_t old_parent_b,
                       int64_t best_delta) {
    active = rollback.active;
    ++active_clusters;
    E[a_id] = old_Ea;
    parent[b_id] = old_parent_b;
    hist_N[a_id] = rollback.hist_N_a;
    hist_h[a_id] = rollback.hist_h_a;
    hist_nz_N[a_id] = rollback.hist_nz_N_a;
    hist_nz_h[a_id] = rollback.hist_nz_h_a;
    hist_N[b_id] = rollback.hist_N_b;
    hist_h[b_id] = rollback.hist_h_b;
    hist_nz_N[b_id] = rollback.hist_nz_N_b;
    hist_nz_h[b_id] = rollback.hist_nz_h_b;
    current_entropy_cost -= best_delta;
  }

  Status RunGreedyMerges(uint32_t num_clusters) {
    while (active_clusters > num_clusters) {
      size_t best_i = 0;
      size_t best_j = 1;
      int64_t best_delta = 0;
      JXL_RETURN_IF_ERROR(FindBestMerge(&best_i, &best_j, &best_delta));
      std::pair<uint32_t, uint32_t> merged =
          ApplyMerge(best_i, best_j, best_delta);
      current_entropy_cost += best_delta;
      JXL_RETURN_IF_ERROR(UpdateDistances(merged.first));
    }
    return true;
  }

  Status RunOverheadAwareTail() {
    auto rollback = jxl::make_unique<RollbackScratch>();
    JXL_ASSIGN_OR_RETURN(int64_t initial_overhead,
                         clustering.ComputeSignallingOverhead(d));
    int64_t best_total_cost = current_entropy_cost + initial_overhead;

    while (active_clusters > 1) {
      size_t best_i = 0;
      size_t best_j = 1;
      int64_t best_delta = 0;
      JXL_RETURN_IF_ERROR(FindBestMerge(&best_i, &best_j, &best_delta));

      uint32_t a_id = active[best_i];
      uint32_t b_id = active[best_j];
      int64_t old_Ea = E[a_id];
      uint32_t old_parent_b = parent[b_id];
      SaveRollback(rollback.get(), a_id, b_id);

      E[a_id] += E[b_id] + best_delta;
      hist_N[a_id].AddFrom(rollback->hist_N_b);
      hist_h[a_id].AddFrom(rollback->hist_h_b);
      hist_nz_N[a_id].AddFrom(rollback->hist_nz_N_b);
      hist_nz_h[a_id].AddFrom(rollback->hist_nz_h_b);
      parent[b_id] = a_id;
      std::swap(active[best_j], active.back());
      active.pop_back();
      --active_clusters;
      current_entropy_cost += best_delta;
      JXL_RETURN_IF_ERROR(UpdateDistances(a_id));

      const int64_t overhead_cutoff = best_total_cost - current_entropy_cost;
      if (overhead_cutoff > 0) {
        JXL_ASSIGN_OR_RETURN(
            int64_t overhead,
            clustering.ComputeSignallingOverhead(d, overhead_cutoff));
        if (overhead < overhead_cutoff) {
          best_total_cost = current_entropy_cost + overhead;
          continue;
        }
      }

      // Roll back the rejected merge and stop.
      RestoreRollback(*rollback, a_id, b_id, old_Ea, old_parent_b, best_delta);
      break;
    }
    return true;
  }

  void Finalize() {
    clustering.clustered_cost = 0;
    for (uint32_t k : active) clustering.clustered_cost += E[k];

    clustering.ctx_num = active_clusters;
    std::function<uint32_t(uint32_t)> find_cluster =
        [this, &find_cluster](uint32_t ctx) -> uint32_t {
      return parent[ctx] == ctx ? ctx : parent[ctx] = find_cluster(parent[ctx]);
    };
    for (uint32_t i : initial_active) {
      ctx_map[i] = std::find(active.begin(), active.end(), find_cluster(i)) -
                   active.begin();
    }

    CompactHistogramSet compact_h(active_clusters);
    uint32_t ind = 0;
    for (uint32_t i : active) compact_h[ind++].swap(hist_h[i]);
    hist_h.swap(compact_h);

    DenseNHistogramSet compact_N(active_clusters);
    ind = 0;
    for (uint32_t i : active) compact_N[ind++].swap(hist_N[i]);
    hist_N.swap(compact_N);

    DenseNZHistogramSet compact_nz_h(active_clusters);
    ind = 0;
    for (uint32_t i : active) compact_nz_h[ind++].swap(hist_nz_h[i]);
    hist_nz_h.swap(compact_nz_h);

    DenseNZPredHistogramSet compact_nz_N(active_clusters);
    ind = 0;
    for (uint32_t i : active) compact_nz_N[ind++].swap(hist_nz_N[i]);
    hist_nz_N.swap(compact_nz_N);
  }

  Status Run(uint32_t num_clusters, bool overhead_aware_tail) {
    num_clusters = std::max(num_clusters, uint32_t{1});
    JXL_DASSERT(total_ctxs == hist_h.size());
    JXL_DASSERT(total_ctxs == hist_nz_N.size());
    JXL_DASSERT(total_ctxs == hist_nz_h.size());
    ctx_map.assign(total_ctxs, 0);

    JXL_RETURN_IF_ERROR(InitEntropy());
    current_entropy_cost = 0;
    for (int64_t e : E) current_entropy_cost += e;

    InitActiveClusters();
    JXL_RETURN_IF_ERROR(InitDeltas());
    JXL_RETURN_IF_ERROR(RunGreedyMerges(num_clusters));
    if (overhead_aware_tail) {
      JXL_RETURN_IF_ERROR(RunOverheadAwareTail());
    }
    Finalize();
    return true;
  }
};

Status Clustering::AgglomerativeClustering(const JPEGOptData& d,
                                           uint32_t num_clusters,
                                           bool overhead_aware_tail,
                                           ThreadPool* pool) {
  AgglomerativeCtx ctx(*this, d, pool);
  return ctx.Run(num_clusters, overhead_aware_tail);
}

struct AxisMaps {
  const JPEGOptData& image_;
  // Fine-grained bucketing (up to `M_eff` intervals) for the swept axis.
  // `ax0_to_k` provides the high-resolution grid that groups stream statistics,
  // which the Knuth DP solver evaluates to find new optimal coarse thresholds.
  // `k_to_dc0` is the inverse map used during DP backtracing to translate
  // split points (in bucket space `k`) back to actual `DC` bounds, maps `k` ->
  // first DC rank in that bucket.
  std::vector<uint16_t> ax0_to_k;
  std::vector<uint16_t> k_to_dc0;
  // Maps a DC value index on `ax0/ax1/ax2` to its partition-cell offset.
  // `ax0_cell[i] = bkt(dc_vals[ax0][i], T0) * n1 * n2`,
  // `ax1_row[i] = bkt(dc_vals[ax1][i], T1) * n2`,
  // `ax2_col[i] = bkt(dc_vals[ax2][i], T2)`.
  // Cell index in the 2D grid:
  //   `ci = ax1_row[dc_ax1_idx] + ax2_col[dc_ax2_idx]`.
  // Cell index in the 3D grid:
  //   `cell = ax0_cell[dc_ax0_idx] + ci`.
  std::vector<uint16_t> ax1_row;
  std::vector<uint16_t> ax2_col;

  explicit AxisMaps(const JPEGOptData& image)
      : image_(image),
        ax0_to_k(kDCTRange, 0),
        k_to_dc0(kDCTRange, 0),
        ax1_row(kDCTRange, 0),
        ax2_col(kDCTRange, 0) {}

  // Bucket assignment by thresholds. Linear search as max 15 thresholds.
  template <typename Points>
  static uint16_t Bkt(int DC, const Points& T) {
    for (uint16_t i = 0; i < T.size(); ++i)
      if (T[i] > DC) return i;
    return static_cast<uint16_t>(T.size());
  }

  // Update the maps that map DC values to partition-cell indices.
  void Update(uint32_t axis, const Thresholds& T0, const Thresholds& T1,
              const Thresholds& T2, bool ax0_identity = false) {
    uint32_t ax0 = axis;
    uint32_t ax1 = (axis + 1) % 3;
    uint32_t ax2 = (axis + 2) % 3;
    uint16_t M0 = static_cast<uint16_t>(image_.DC_vals[ax0].size());
    size_t M1 = image_.DC_vals[ax1].size();
    size_t M2 = image_.DC_vals[ax2].size();
    for (uint16_t i = 0; i < M0; ++i) {
      ax0_to_k[i] = ax0_identity ? i : Bkt(image_.DC_vals[ax0][i], T0);
      if (ax0_identity) k_to_dc0[i] = i;
    }
    size_t n2 = T2.size() + 1;
    for (size_t i = 0; i < M1; ++i)
      ax1_row[i] = static_cast<uint16_t>(Bkt(image_.DC_vals[ax1][i], T1) * n2);
    for (size_t i = 0; i < M2; ++i)
      ax2_col[i] = Bkt(image_.DC_vals[ax2][i], T2);
  }

  void Update(const ThresholdSet& thresholds) {
    Update(0, thresholds.TY(), thresholds.TCb(), thresholds.TCr());
  }

  // Prepare cell maps `(ax1_row, ax2_col)` and axis bucket maps
  // `(dc0_to_k, k_to_dc0)`. Returns the actual number of axis buckets `M_eff`.
  uint32_t PrepareBucketing(uint32_t axis, uint32_t M_target,
                            const Thresholds& T1, const Thresholds& T2) {
    uint32_t M = static_cast<uint32_t>(image_.DC_vals[axis].size());
    uint32_t M_eff = std::min(M, M_target);
    if (M_eff == M) {
      Update(axis, {}, T1, T2, true);
      return M;
    }

    // TODO: do not repeat this each time, save axis buckets somewhere
    Thresholds bkt_thresh = image_.InitThresh(static_cast<int>(axis), M_eff);
    Update(axis, bkt_thresh, T1, T2);

    uint32_t cur_k = 0;
    k_to_dc0[0] = 0;
    for (uint16_t i = 0; i < M; ++i) {
      uint16_t k = ax0_to_k[i];
      while (cur_k < k) k_to_dc0[++cur_k] = i;
    }
    JXL_DASSERT(cur_k + 1 == M_eff);
    JXL_DASSERT(ax0_to_k[M - 1] + 1 == M_eff);
    return M_eff;
  }
};

template <class StreamSweepFn>
struct ClusteringBuildCtx {
  const JPEGOptData& d;
  AxisMaps& axis_maps;
  StreamSweepFn& stream_sweep;
  uint32_t n0;
  uint32_t num_cells;
  uint32_t total_ctxs;

  ClusteringBuildCtx(const JPEGOptData& d, AxisMaps& axis_maps,
                     const ThresholdSet& thresholds,
                     StreamSweepFn& stream_sweep)
      : d(d),
        axis_maps(axis_maps),
        stream_sweep(stream_sweep),
        n0(static_cast<uint32_t>(thresholds.TY().size() + 1)),
        num_cells(n0 * static_cast<uint32_t>(thresholds.TCb().size() + 1) *
                  static_cast<uint32_t>(thresholds.TCr().size() + 1)),
        total_ctxs(kNumCh * num_cells) {
    axis_maps.Update(thresholds);
  }

  // Fills all four histogram arrays in `cl` with counts derived from the image
  // data, ready for `AgglomerativeClustering`. Each array is indexed by:
  //   `ctx_id = c * num_cells + cell`
  // where `c` ∈ [0, kNumCh) is the channel and
  //   cell = (ax1_row[dc1] + ax2_col[dc2]) * n0 + ax0_to_k[dc0]
  // is the DC-threshold cell that the block belongs to.
  //
  // **Pass 1 — AC coefficients (via `StreamSweep`)**
  // Each reset stream frame carries a packed `bin_state` word:
  //   bits 21-20 : channel `c`
  //   bits 19-11 : `zdc`  (`ZeroDensityContext`, the AC coding context)
  //   bits  10-0 : `ai`   (AC coefficient value index)
  //
  // - `hist_h[ctx_id]` accumulates counts of `(zdc, ai)` bins (compacted via
  //   `CompactHBin`); this is the AC-symbol histogram used in entropy coding.
  // - `hist_N[ctx_id]` accumulates counts in `zdc` contexts; this is the
  //   "context frequency" histogram `N` used in the entropy cost model.
  //
  // **Pass 2 — nonzero-count histograms (block iteration)**
  // For each block, its nonzero AC count and predictor bucket `pb` are added:
  // - `hist_nz_h[ctx_id]` counts `(pb, nz_count)` bins (via
  //   `NZHistogramIndex`); used as the histogram for nz-count coding.
  // - `hist_nz_N[ctx_id]` counts events in predictor bucket `pb`; the
  //   corresponding N.
  // Grayscale (`d.channels == 1`) takes a fast path: there is only one channel
  // and the cell index collapses to `ax0_to_k[dc0_bucket]`, so the multichannel
  // subsampling coordinate mapping is skipped entirely.
  void PopulateHistograms(Clustering* cl) {
    stream_sweep(
        []() {}, []() {},
        [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
            uint32_t bin_state) {
          uint32_t c = (bin_state >> 20) & 0x3u;
          uint32_t cell = static_cast<uint32_t>(
              (axis_maps.ax1_row[dc1_idx] + axis_maps.ax2_col[dc2_idx]) * n0 +
              axis_maps.ax0_to_k[dc0_idx]);
          uint32_t ctx_id = c * num_cells + cell;
          cl->hist_h[ctx_id].Add(d.CompactHBin(bin_state & 0xFFFFFu), run);
          uint32_t zdc = (bin_state >> 11) & 0x1FFu;
          cl->hist_N[ctx_id].Add(zdc, run);
        });

    if (d.channels == 1) {
      for (uint32_t b = 0; b < d.num_blocks[0]; ++b) {
        uint32_t ctx_id = axis_maps.ax0_to_k[d.block_DC_idx[0][b]];
        uint32_t nz_count = d.block_nonzeros[0][b];
        uint32_t pb = d.block_nz_pred_bucket[0][b];
        cl->hist_nz_h[ctx_id].Add(NZHistogramIndex(pb, nz_count));
        cl->hist_nz_N[ctx_id].Add(pb);
      }
      return;
    }

    for (uint32_t c = 0; c < kNumCh; ++c) {
      for (uint32_t by = 0; by < d.block_grid_h[c]; ++by) {
        for (uint32_t bx = 0; bx < d.block_grid_w[c]; ++bx) {
          // Convert block `(bx, by)` in channel `c` to a common block
          // coordinate `(x, y)` in the common (not subsampled) plane, then
          // find the corresponding block index in each channel. Channels have
          // different block grids due to subsampling, so all three DC values
          // needed to determine `cell` must be looked up in their own grids.
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
          uint32_t cell = (axis_maps.ax1_row[d.block_DC_idx[1][b1]] +
                           axis_maps.ax2_col[d.block_DC_idx[2][b2]]) *
                              n0 +
                          axis_maps.ax0_to_k[d.block_DC_idx[0][b0]];
          uint32_t ctx_id = c * num_cells + cell;

          uint32_t b = by * d.block_grid_w[c] + bx;
          uint32_t nz_count = d.block_nonzeros[c][b];
          uint32_t pb = d.block_nz_pred_bucket[c][b];
          cl->hist_nz_h[ctx_id].Add(NZHistogramIndex(pb, nz_count));
          cl->hist_nz_N[ctx_id].Add(pb);
        }
      }
    }
  }

  StatusOr<Clustering> Build(uint32_t num_clusters, bool overhead_aware_tail,
                             ThreadPool* pool) {
    Clustering cl;
    cl.hist_h.assign(total_ctxs, CompactHistogram(d.num_zdcai));
    cl.hist_N.resize(total_ctxs);
    cl.hist_nz_h.resize(total_ctxs);
    cl.hist_nz_N.resize(total_ctxs);
    PopulateHistograms(&cl);
    JXL_RETURN_IF_ERROR(
        cl.AgglomerativeClustering(d, num_clusters, overhead_aware_tail, pool));
    return cl;
  }
};

template <class SweepOwner>
struct StreamSweepAdapter {
  const SweepOwner& owner;

  template <class FlushH, class FlushN, class OnRun>
  void operator()(FlushH&& flush_h, FlushN&& flush_N, OnRun&& on_run) const {
    owner.StreamSweep(std::forward<FlushH>(flush_h),
                      std::forward<FlushN>(flush_N),
                      std::forward<OnRun>(on_run));
  }
};

template <class StreamSweepFn>
StatusOr<Clustering> Clustering::Build(
    const JPEGOptData& d, AxisMaps& axis_maps, const ThresholdSet& thresholds,
    uint32_t num_clusters, bool overhead_aware_tail, ThreadPool* pool,
    StreamSweepFn& stream_sweep) {
  ClusteringBuildCtx<StreamSweepFn> build_ctx(d, axis_maps, thresholds,
                                              stream_sweep);
  return build_ctx.Build(num_clusters, overhead_aware_tail, pool);
}

struct RefineScratch {
  CompactHistogramSet hist_h;
  DenseNHistogramSet hist_N;
  DenseNZPredHistogramSet hist_nz_N;
  DenseNZHistogramSet hist_nz_h;
  std::array<std::vector<ClusterBoundary>, kNumCh> local_cluster_boundary;

  void Reset(const Clustering& clustering) {
    hist_h = clustering.hist_h;
    hist_N = clustering.hist_N;
    hist_nz_N = clustering.hist_nz_N;
    hist_nz_h = clustering.hist_nz_h;
  }
};

// Short-lived per-axis refinement context. Owns the axis-local view of the
// immutable image data and the mutable scratch used while moving thresholds.
struct RefineCtx {
  struct ProjectedBlocks {
    uint32_t ref_block;
    uint32_t y_begin;
    uint32_t x_begin;
    uint32_t y_end;
    uint32_t x_end;
  };

  const JPEGOptData& d;
  AxisMaps& axis_maps;
  uint32_t axis;
  uint32_t ax1;
  uint32_t ax2;
  const std::vector<int16_t>& dc_axis;
  ptrdiff_t search_radius;
  uint32_t na;
  uint32_t num_rows;
  RefineScratch& scratch;

  RefineCtx(const JPEGOptData& d, AxisMaps& axis_maps, uint32_t axis,
            const ThresholdSet& thresholds, RefineScratch& scratch,
            ptrdiff_t search_radius)
      : d(d),
        axis_maps(axis_maps),
        axis(axis),
        ax1((axis + 1) % 3),
        ax2((axis + 2) % 3),
        dc_axis(d.DC_vals[axis]),
        search_radius(search_radius),
        na(static_cast<uint32_t>(thresholds.T[axis].size() + 1)),
        num_rows(static_cast<uint32_t>(
            ((thresholds.TY().size() + 1) * (thresholds.TCb().size() + 1) *
             (thresholds.TCr().size() + 1)) /
            (thresholds.T[axis].size() + 1))),
        scratch(scratch) {
    axis_maps.Update(axis, thresholds.T[axis], thresholds.T[ax1],
                     thresholds.T[ax2]);
  }

  template <typename HistogramSet>
  int64_t MoveHistogramBin(HistogramSet* hist, uint32_t old_cl, uint32_t new_cl,
                           uint32_t bin, int64_t sign) const {
    auto& from = (*hist)[old_cl];
    uint32_t old_freq = from.Get(bin);
    JXL_DASSERT(old_freq > 0);
    if (old_freq == 0) return 0;

    int64_t delta = sign * (d.NZFTab(old_freq - 1) - d.NZFTab(old_freq));
    from.Subtract(bin, 1);

    auto& to = (*hist)[new_cl];
    old_freq = to.Get(bin);
    delta += sign * (d.NZFTab(old_freq + 1) - d.NZFTab(old_freq));
    to.Add(bin, 1);
    return delta;
  }

  template <typename HistogramHSet, typename HistogramNSet>
  int64_t MoveHistogramEvent(uint32_t old_cl, uint32_t new_cl,
                             HistogramHSet* hist_h, HistogramNSet* hist_N,
                             uint32_t h_bin, uint32_t N_bin) const {
    return MoveHistogramBin(hist_h, old_cl, new_cl, h_bin, -1) +
           MoveHistogramBin(hist_N, old_cl, new_cl, N_bin, +1);
  }

  std::pair<uint32_t, uint32_t> BoundaryClusters(uint32_t thr_ind,
                                                 uint32_t channel, uint32_t ci,
                                                 bool upward) const {
    const auto& boundary_map = scratch.local_cluster_boundary[axis];
    const ClusterBoundary& boundary =
        boundary_map[(channel * (na - 1) + thr_ind) * num_rows + ci];
    return upward ? std::pair<uint32_t, uint32_t>{boundary.hi, boundary.lo}
                  : std::pair<uint32_t, uint32_t>{boundary.lo, boundary.hi};
  }

  uint32_t CellIndex(uint32_t dc_ax1_ind, uint32_t dc_ax2_ind) const {
    return axis_maps.ax1_row[dc_ax1_ind] + axis_maps.ax2_col[dc_ax2_ind];
  }

  static uint32_t DivCeilU32(uint32_t num, uint32_t den) {
    return (num + den - 1) / den;
  }

  uint32_t BlockIndex(uint32_t channel, uint32_t y, uint32_t x) const {
    return y * d.block_grid_w[channel] + x;
  }

  // The scheme used here is different with respect to other parts of the
  // library, where usually iterations are over common plane and just skip
  // subsampled blocks. This is done to avoid memory bloat, since we are
  // already using 2*32-bits per single AC coefficient in `block_bins` and
  // `AC_stream`.
  //
  // Match `PopulateClusterHistograms`: cross-component block references are
  // chosen from the block's top-left coordinate in the common plane.
  // `ref_block` is that direct top-left-mapped block in `dst_axis`.
  // `[y_begin, y_end) x [x_begin, x_end)` enumerates `dst_axis` blocks
  // whose own top-left anchors fall inside the moved `src_axis` block, so
  // these ranges may legitimately be empty under subsampling.
  ProjectedBlocks ProjectBlock(uint32_t src_axis, uint32_t dst_axis, uint32_t y,
                               uint32_t x) const {
    uint32_t scaled_y = y * d.ss_y[src_axis];
    uint32_t scaled_x = x * d.ss_x[src_axis];
    return {BlockIndex(dst_axis, scaled_y / d.ss_y[dst_axis],
                       scaled_x / d.ss_x[dst_axis]),
            DivCeilU32(scaled_y, d.ss_y[dst_axis]),
            DivCeilU32(scaled_x, d.ss_x[dst_axis]),
            DivCeilU32((y + 1) * d.ss_y[src_axis], d.ss_y[dst_axis]),
            DivCeilU32((x + 1) * d.ss_x[src_axis], d.ss_x[dst_axis])};
  }

  uint32_t CellIndexFromBlocks(uint32_t b1, uint32_t b2) const {
    return CellIndex(d.block_DC_idx[ax1][b1], d.block_DC_idx[ax2][b2]);
  }

  int64_t ApplyBlockMove(uint32_t channel, uint32_t block, uint32_t thr_ind,
                         uint32_t ci, bool upward) {
    std::pair<uint32_t, uint32_t> boundary =
        BoundaryClusters(thr_ind, channel, ci, upward);
    uint32_t old_cl = boundary.first;
    uint32_t new_cl = boundary.second;
    if (old_cl == new_cl) return 0;

    uint32_t N_bin = d.block_nz_pred_bucket[channel][block];
    uint32_t h_bin = NZHistogramIndex(N_bin, d.block_nonzeros[channel][block]);
    int64_t delta = MoveHistogramEvent(old_cl, new_cl, &scratch.hist_nz_h,
                                       &scratch.hist_nz_N, h_bin, N_bin);
    for (uint32_t pi = d.block_offsets[channel][block];
         pi < d.block_offsets[channel][block + 1]; ++pi) {
      uint32_t bin = d.block_bins[channel][pi] & 0xFFFFF;
      uint32_t zdc = bin >> 11;
      delta += MoveHistogramEvent(old_cl, new_cl, &scratch.hist_h,
                                  &scratch.hist_N, d.CompactHBin(bin), zdc);
    }
    return delta;
  }

  int64_t ApplySlice(uint32_t thr_ind, ptrdiff_t slice, bool upward) {
    const auto& sorted_blocks = d.DC_sorted_blocks[axis];
    const auto& block_off = d.DC_block_offsets[axis];

    int64_t cost_change = 0;
    uint32_t blk_lo = block_off[static_cast<size_t>(slice)];
    uint32_t blk_hi = block_off[static_cast<size_t>(slice + 1)];
    for (uint32_t bi = blk_lo; bi < blk_hi; ++bi) {
      uint32_t b0_xy = sorted_blocks[bi];
      uint32_t b0_y = b0_xy >> 16;
      uint32_t b0_x = b0_xy & 0xFFFF;
      uint32_t b0 = BlockIndex(axis, b0_y, b0_x);

      if (d.channels == 1) {
        cost_change += ApplyBlockMove(0, b0, thr_ind, 0, upward);
        continue;
      }

      ProjectedBlocks b1_area = ProjectBlock(axis, ax1, b0_y, b0_x);
      ProjectedBlocks b2_area = ProjectBlock(axis, ax2, b0_y, b0_x);

      cost_change += ApplyBlockMove(
          axis, b0, thr_ind,
          CellIndexFromBlocks(b1_area.ref_block, b2_area.ref_block), upward);

      for (uint32_t y1 = b1_area.y_begin; y1 < b1_area.y_end; ++y1) {
        for (uint32_t x1 = b1_area.x_begin; x1 < b1_area.x_end; ++x1) {
          uint32_t b1 = BlockIndex(ax1, y1, x1);
          uint32_t y2 = y1 * d.ss_y[ax1] / d.ss_y[ax2];
          uint32_t x2 = x1 * d.ss_x[ax1] / d.ss_x[ax2];
          uint32_t b2 = BlockIndex(ax2, y2, x2);
          cost_change += ApplyBlockMove(ax1, b1, thr_ind,
                                        CellIndexFromBlocks(b1, b2), upward);
        }
      }

      for (uint32_t y2 = b2_area.y_begin; y2 < b2_area.y_end; ++y2) {
        for (uint32_t x2 = b2_area.x_begin; x2 < b2_area.x_end; ++x2) {
          uint32_t b2 = BlockIndex(ax2, y2, x2);
          uint32_t y1 = y2 * d.ss_y[ax2] / d.ss_y[ax1];
          uint32_t x1 = x2 * d.ss_x[ax2] / d.ss_x[ax1];
          uint32_t b1 = BlockIndex(ax1, y1, x1);
          cost_change += ApplyBlockMove(ax2, b2, thr_ind,
                                        CellIndexFromBlocks(b1, b2), upward);
        }
      }
    }
    return cost_change;
  }

  int16_t OptimizeThreshold(uint32_t thr_ind, const Thresholds& thresholds,
                            Clustering& clustering, int64_t* base_cost) {
    ptrdiff_t cur_idx = AxisMaps::Bkt(thresholds[thr_ind] - 1, dc_axis);
    ptrdiff_t lo_edge = std::max(
        {cur_idx - search_radius,
         (thr_ind == 0) ? static_cast<ptrdiff_t>(0)
                        : AxisMaps::Bkt(thresholds[thr_ind - 1] - 1, dc_axis)});
    ptrdiff_t hi_edge =
        std::min({cur_idx + search_radius,
                  (thr_ind == thresholds.size() - 1)
                      ? static_cast<ptrdiff_t>(dc_axis.size())
                      : AxisMaps::Bkt(thresholds[thr_ind + 1] - 1, dc_axis)});

    ptrdiff_t best_idx = cur_idx;
    int64_t best_cost = *base_cost;

    scratch.Reset(clustering);
    int64_t current_cost = *base_cost;
    for (ptrdiff_t idx = cur_idx - 1; idx > lo_edge; --idx) {
      current_cost += ApplySlice(thr_ind, idx, false);
      if (current_cost < best_cost) {
        best_cost = current_cost;
        best_idx = idx;
      }
    }

    scratch.Reset(clustering);
    current_cost = *base_cost;
    for (ptrdiff_t idx = cur_idx + 1; idx < hi_edge; ++idx) {
      current_cost += ApplySlice(thr_ind, idx - 1, true);
      if (current_cost < best_cost) {
        best_cost = current_cost;
        best_idx = idx;
      }
    }

    if (best_idx == cur_idx) return thresholds[thr_ind];

    scratch.Reset(clustering);
    if (best_idx < cur_idx) {
      for (ptrdiff_t idx = cur_idx - 1; idx >= best_idx; --idx) {
        ApplySlice(thr_ind, idx, false);
      }
    } else {
      for (ptrdiff_t idx = cur_idx + 1; idx <= best_idx; ++idx) {
        ApplySlice(thr_ind, idx - 1, true);
      }
    }
    *base_cost = best_cost;
    std::swap(clustering.hist_h, scratch.hist_h);
    std::swap(clustering.hist_N, scratch.hist_N);
    std::swap(clustering.hist_nz_N, scratch.hist_nz_N);
    std::swap(clustering.hist_nz_h, scratch.hist_nz_h);
    return dc_axis[static_cast<size_t>(best_idx)];
  }
};

// Context for parallel computation of optimal block partitioning.
struct PartitioningCtx {
  // Immutable reference to precomputed image data.
  std::shared_ptr<const JPEGOptData> image;

  AxisMaps axis_maps;

  // Scratch buffers for optimization sweeps and `total_cost`.
  // `h_cnt` - counts of AC coefficients for each cell and each DC rank.
  // `N_cnt` - total counts of AC coefficients for each cell.
  // Normal state is zeroed out: each nonzero entry is cleared on flush.
  std::vector<uint32_t> h_cnt;
  std::vector<uint32_t> N_cnt;

  // Reusable solver state for the swept-axis partition DP.
  // Owns the lazily materialized diff-form cost matrix and the DP scratch
  // buffers that are reused across optimization sweeps.
  KnuthPartitionSolver Knuth_solver;

  // Sparse lists of `(rank, cumulative_count)` seen so far in each 2D
  // partition cell. Used for incremental cost matrix updates.
  using Bin = std::pair<uint16_t, uint32_t>;
  using CellHistory = std::array<std::vector<Bin>, kMaxCells / 2>;
  CellHistory h_history;
  CellHistory N_history;

  // Two-level bitmask for sparse dirty tracking over flat index space
  // `idx = ci * M_eff + n` (up to 32768 slots).
  // `touched[group]`       — one bit per idx within each 64-slot group.
  // `group_touched[tier]`  — one bit per group within each 64-group tier.
  // On write to `idx`: set bit in `touched[idx>>6]` and bit in
  // `group_touched[idx>>12]`. Flush iterates only non-zero words, so only
  // actually written slots are visited — no full-array scan needed.
  std::vector<uint64_t> touched_h;
  std::vector<uint64_t> touched_N;
  std::vector<uint64_t> group_touched_h;
  std::vector<uint64_t> group_touched_N;

  // Number of bins in the running histogram for AC coefficients.
  static constexpr uint32_t kBinCount = kMaxCells / 2 * kDCTRange >> 6;
  // Number of groups of bins in the running histogram for AC coefficients.
  static constexpr uint32_t kGroupCount = kMaxCells / 2 * kDCTRange >> 12;

  explicit PartitioningCtx(std::shared_ptr<const JPEGOptData> d)
      : image(std::move(d)),
        axis_maps(*image),
        Knuth_solver(kDCTRange),
        h_history(),
        N_history(),
        touched_h(kBinCount, 0),
        touched_N(kBinCount, 0),
        group_touched_h(kGroupCount, 0),
        group_touched_N(kGroupCount, 0) {}

  const JPEGOptData& data() const { return *image; }
  const std::vector<ACEntry>& ac_stream() const { return data().AC_stream; }

  // Process the AC coefficient stream and compute costs.
  // Stream is sorted by bin index. Structure of elements is:
  // - Reset frame:
  //   `(1<<31) | (ctx_change<<30) | (bin_change<<29) | (bin<<7) | (dc0>>4)`.
  // - Normal frame: `(delta_dc0<<27) | (dc1<<16) | (dc2<<5) | (run-1)`,
  //   `delta_dc0 <= 15`, so that bit 31 is 0.
  // - Long-run frame: `(delta_dc0<<27) | (dc1<<16) | (dc2<<5) | 0x1F` followed
  //   by `run`.
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

  // K=2 fast path for a single threshold `s`: evaluates `E(s)` for all
  // `s` in `[1, M-1]` simultaneously in one stream pass, without building
  // the full `M×M` cost matrix.
  //
  // For a split at rank `s`, cost `E(s)` decomposes as:
  //  `+ sum_{ctx,ci} (ftab[N_left(ctx,ci,s)] + ftab[N_right(ctx,ci,s)])`
  //  `- sum_{bin,ci} (ftab[h_left(bin,ci,s)] + ftab[h_right(bin,ci,s)])`
  // where left = DC ranks `[0..s-1]`, right = DC ranks `[s..M-1]`.
  //
  // For each `(bin/ctx, ci)` group the per-split term is a step function of
  // `s` that changes only at the ranks present in that group. We encode it
  // into `score_diff` with the one-write diff trick: at each rank boundary
  // write only `(term - prev_term)`, so the close of one segment and the
  // open of the next collapse into a single addition. A prefix-sum over
  // `s=1..M-1` then recovers the full cost curve in `O(M)`.
  //
  // `axis==0`: sparse-hist path (bins already sorted by DC0 rank, no sort
  //            needed; uses `h_history`/`N_history`).
  // `axis!=0`: dense-cnt path (uses `h_cnt`/`N_cnt`, sweeps ranks linearly).
  Thresholds OptimizeAxisSingleSplit(uint32_t axis, uint32_t ncells,
                                     uint32_t M_eff) {
    const JPEGOptData& d = data();
    // Reuse `costs` as a 1D diff buffer (indices `1..M_eff-1`).
    // `score_diff[s]` holds the delta `E(s) - E(s-1)` in bucket space.
    // After the stream pass, prefix-summing gives `E(s) - E(0)` up to a
    // constant — argmin is unchanged by the constant.
    auto& score_diff = Knuth_solver.costs;
    Knuth_solver.ResetCosts(M_eff + 1);

    // `bin_mask` and `ctx_mask` are used to track which cells have been visited
    // during bins and contexts collection. The number of cells is less or equal
    // to 32, so we can use `uint32_t` to store the mask.
    uint32_t bin_mask = 0;
    uint32_t ctx_mask = 0;

    if (axis == 0) {
      // Converts the sparse `(rank, count)` list accumulated in `ch[ci]` for
      // each touched cell into additive contributions to `score_diff`.
      //
      // `ch[ci]` contains `(dc_k_idx, run)` bins in strictly increasing rank
      // order (the accumulation loop merges same-rank entries on the fly).
      //
      // For split rank `s`, the entropy term for this group is:
      //   `term(s) = sign * (ftab[j_before_l(s)] + ftab[total -
      //   j_before_l(s)])`
      // where `j_before_l(s)` = count of entries with rank < `s`. This is a
      // step function of `s` that changes only at the ranks present in the
      // group. We encode it with the one-write diff trick: for each boundary
      // at rank `r_i`, instead of writing `+term` and `-term` at both ends of
      // the constant segment, we write only `(term - prev_term)` at `l =
      // r_{i-1}+1`. The segment close is deferred into the next open, halving
      // the writes. The last segment needs no explicit close since indices
      // beyond `M-1` are never read.
      //
      // `sign=-1` for `h_hist` (h-term subtracts from entropy),
      // `sign=+1` for `N_hist` (N-term adds to entropy).
      auto flush_histogram = [&](CellHistory& ch, uint32_t& mask, int sign) {
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
            score_diff[l] += term - prev_term;
            j_before_l += h.second;
            l = static_cast<uint32_t>(h.first) + 1;
            prev_term = term;
          }
          score_diff[l] +=
              sign * static_cast<int64_t>(d.ftab[total]) - prev_term;
          hist.clear();
        }
      };

      // Single pass over `AC_stream` (32-bit frames). Decode and accumulate
      // `(dc_k_idx, run)` into `h_hist[ci]` and `N_hist[ci]`.
      // K=2 fast path is `axis==0`: `dc_k=dc0, ax1=dc1, ax2=dc2`.
      StreamSweep(
          [&]() { flush_histogram(h_history, bin_mask, -1); },
          [&]() { flush_histogram(N_history, ctx_mask, +1); },
          [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx,
              uint32_t run, uint32_t) {
            uint32_t dc_k_bkt = axis_maps.ax0_to_k[dc0_idx];
            uint32_t ci =
                axis_maps.ax1_row[dc1_idx] + axis_maps.ax2_col[dc2_idx];
            bin_mask |= (1U << ci);
            ctx_mask |= (1U << ci);
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
      flush_histogram(h_history, bin_mask, -1);
      flush_histogram(N_history, ctx_mask, +1);
    } else {
      // K=2 fast path for `axis != 0` using dense `h_cnt` sweep.
      // Accumulates `h_cnt[ci * M_eff + dc_k_bkt]` in one stream pass, then
      // at each bin/ctx flush sweeps `n=0..M_eff-1` per touched cell
      // (O(M_eff)) and applies the one-write diff to `score_diff`.
      // Another approach is to use history like above but with sorting —
      // it is slower.
      uint32_t cnt_size = M_eff * ncells;
      if (h_cnt.size() < cnt_size) h_cnt.assign(cnt_size, 0);
      if (N_cnt.size() < cnt_size) N_cnt.assign(cnt_size, 0);

      // Sweep dense `h_cnt` for each touched cell, update `score_diff`.
      // `sign=-1` for h-term, `sign=+1` for N-term.
      auto flush_dense = [&](std::vector<uint32_t>& cnt, uint32_t& mask,
                             int sign) {
        while (mask) {
          uint32_t ci = static_cast<uint32_t>(CountrZero64(mask));
          mask &= mask - 1;
          // Pass 1: compute total count per cell.
          uint32_t total = 0;
          for (uint32_t n = 0; n < M_eff; ++n) total += cnt[ci * M_eff + n];
          // Pass 2: one-write diff over ranks `0..M_eff-1`, reset entries.
          uint32_t j_before_l = 0;
          uint32_t l = 1;
          int64_t prev_term = 0;
          for (uint32_t n = 0; n < M_eff; ++n) {
            uint32_t f = cnt[ci * M_eff + n];
            cnt[ci * M_eff + n] = 0;
            if (f == 0) continue;
            int64_t term = sign * (static_cast<int64_t>(d.ftab[j_before_l]) +
                                   d.ftab[total - j_before_l]);
            score_diff[l] += term - prev_term;
            prev_term = term;
            j_before_l += f;
            l = n + 1;
          }
          score_diff[l] +=
              sign * static_cast<int64_t>(d.ftab[total]) - prev_term;
        }
      };

      uint32_t ax1 = (axis + 1) % 3;
      uint32_t ax2 = (axis + 2) % 3;
      StreamSweep([&]() { flush_dense(h_cnt, bin_mask, -1); },
                  [&]() { flush_dense(N_cnt, ctx_mask, +1); },
                  [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx,
                      uint32_t run, uint32_t) {
                    uint32_t dc_arr[3] = {dc0_idx, dc1_idx, dc2_idx};
                    uint32_t dc_k_bkt = axis_maps.ax0_to_k[dc_arr[axis]];
                    uint32_t ci = axis_maps.ax1_row[dc_arr[ax1]] +
                                  axis_maps.ax2_col[dc_arr[ax2]];
                    uint32_t idx = ci * M_eff + dc_k_bkt;
                    h_cnt[idx] += run;
                    bin_mask |= (1U << ci);
                    N_cnt[idx] += run;
                    ctx_mask |= (1U << ci);
                  });
      flush_dense(h_cnt, bin_mask, -1);
      flush_dense(N_cnt, ctx_mask, +1);
    }

    // Prefix-sum `score_diff[1..M_eff-1]` recovers `E(s) - E(0)` up to a
    // global constant; we only need the argmin, so the constant cancels.
    int64_t cur = 0;
    int64_t best = std::numeric_limits<int64_t>::max();
    uint32_t best_s = 1;
    for (uint32_t s = 1; s < M_eff; ++s) {
      cur += score_diff[s];
      if (cur < best) {
        best = cur;
        best_s = s;
      }
    }
    return {d.DC_vals[axis][axis_maps.k_to_dc0[best_s]]};
  }

  // Drains collected `cnt` entries into the `Knuth_solver.costs` `M_eff×M_eff`
  // matrix (used by `OptimizeAxisSingleSweep` for `K≥3` intervals).
  //
  // The Knuth solver expects `costs[n * M_eff + l]` to hold the total entropy
  // contribution for the interval of DC ranks `[l, n]` across all (bin/ctx,
  // cell) groups.  Each flushed entry `(ci, n)` with count `freq` adds:
  //
  //   `sign * (ftab[j_ln - freq] - ftab[j_ln])`   for every `l ≤ n`
  //
  // where `j_ln = j_n - j_before_l` is the cumulative count in `[l, n]` and
  // `j_n` is the cumulative count in `[0, n]` for this cell.
  // `sign = +1` for h-terms (histogram, subtracts entropy),
  // `sign = -1` for N-terms (context, adds entropy).
  //
  // Writing to every `cost_row[l]` for `l = 0..n-1` would be `O(n²)` total; we
  // instead use the one-write diff trick: the term is a step function of `l`
  // that changes only at ranks that appear in `history` (the previously seen
  // nonzero ranks for this cell, kept in sorted order). At each boundary we
  // write only `(term - prev_term)` into `cost_row[l]`. The Knuth solver
  // later prefix-sums each column lazily to recover the actual per-interval
  // costs.
  //
  // The two-level bitmask pair (`group_touched`, `touched`) acts as a sparse
  // index over the flat `cnt` array, so only dirty `(ci, n)` entries are
  // visited instead of scanning all `ncells * M_eff` slots. Entries are visited
  // in increasing `bit_idx = ci * M_eff + n` order, which guarantees that
  // within each cell ranks are encountered in ascending order — a prerequisite
  // for the history accumulation. `cnt`, `touched`, and `group_touched` are all
  // reset to zero here so they are ready for the next stream segment.
  template <int sign>
  void FlushTerm(std::vector<uint32_t>& cnt,
                 std::vector<uint64_t>& group_touched,
                 std::vector<uint64_t>& touched, uint32_t M_eff) {
    const JPEGOptData& d = data();
    std::vector<int64_t>& costs = Knuth_solver.costs;
    auto& history = h_history[0];
    history.clear();
    uint32_t cur_ci = UINT32_MAX;
    for (uint32_t hi_idx = 0; hi_idx < kGroupCount; ++hi_idx) {
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
            // Last segment: `j_before_l = j_n - freq`, so `j_ln = freq`,
            // `j_ln - freq = 0`, `ftab[0] = 0` → `term = -sign * ftab[freq]`.
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

  // Generic stream sweep used by coordinate-descent optimization.
  //
  // `M_target`: bucketing resolution (default `kMTarget`).
  // When `M > M_target`, `M_eff = min(M, M_target)` best-equal-population
  // buckets are formed via `PrepareBucketing`. This keeps each cell's
  // distinct-rank count `D ≈ M_eff/ncells`. Pass `UINT32_MAX` or any value
  // `>= M` for a full-resolution (unbucketed) sweep.
  Thresholds OptimizeAxisSingleSweep(uint32_t axis, uint32_t num_intervals,
                                     const Thresholds& T1, const Thresholds& T2,
                                     uint32_t M_target = kMTarget) {
    const JPEGOptData& d = data();
    if (num_intervals == 1) return {};
    uint32_t M = static_cast<uint32_t>(d.DC_vals[axis].size());
    if (M <= num_intervals)  // exclude first DC value from thresholds
      return Thresholds(d.DC_vals[axis].begin() + 1, d.DC_vals[axis].end());

    uint32_t ax1 = (axis + 1) % 3;
    uint32_t ax2 = (axis + 2) % 3;
    uint32_t n1 = static_cast<uint32_t>(T1.size() + 1);
    uint32_t n2 = static_cast<uint32_t>(T2.size() + 1);
    uint32_t ncells = n1 * n2;
    uint32_t M_eff = axis_maps.PrepareBucketing(axis, M_target, T1, T2);

    // Fast path with `O(M_eff)` memory complexity
    if (num_intervals == 2) return OptimizeAxisSingleSplit(axis, ncells, M_eff);
    // Extension of fast path above for `K=3` has proven disastrous
    // for performance (it has the same `O(M_eff^2)` complexity as the general
    // path) and is not implemented.
    // Divide and Conquer approach was also tested with no avail.

    // General path with `O(M_eff^2)` memory complexity
    // Total number of cells probed is `n1 * n2 * M_eff`
    size_t cnt_size = ncells * M_eff;
    if (h_cnt.size() < cnt_size) h_cnt.assign(cnt_size, 0);
    if (N_cnt.size() < cnt_size) N_cnt.assign(cnt_size, 0);

    Knuth_solver.ResetCosts(M_eff * M_eff);

    StreamSweep(
        [&]() { FlushTerm<+1>(h_cnt, group_touched_h, touched_h, M_eff); },
        [&]() { FlushTerm<-1>(N_cnt, group_touched_N, touched_N, M_eff); },
        [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
            uint32_t) {
          uint32_t dc_arr[3] = {dc0_idx, dc1_idx, dc2_idx};
          uint32_t dc_k_bkt = axis_maps.ax0_to_k[dc_arr[axis]];
          uint32_t ci = static_cast<uint32_t>(axis_maps.ax1_row[dc_arr[ax1]] +
                                              axis_maps.ax2_col[dc_arr[ax2]]);
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
    FlushTerm<+1>(h_cnt, group_touched_h, touched_h, M_eff);
    FlushTerm<-1>(N_cnt, group_touched_N, touched_N, M_eff);

    return Knuth_solver.Solve(d.DC_vals[axis], axis_maps.k_to_dc0,
                              num_intervals, M_eff);
  }

  // Performs iterative coordinate descent to find optimal threshold vectors
  // `(TY, TCb, TCr)` for a given target factorization `(a, b, c)`.
  //
  // In each step, it optimizes one axis at a time by fixing the thresholds of
  // the other two axes. It uses `OptimizeAxisSingleSweep` to find the best
  // split points for the current axis.
  //
  // The process continues until convergence (no changes in thresholds) or
  // until `max_iters` is reached. In practice no more than 5 iterations were
  // seen.
  std::pair<int64_t, ThresholdSet> OptimizeThresholds(
      ThresholdSet T, uint32_t M_target = UINT32_MAX, uint32_t max_iters = 20) {
    uint32_t a = static_cast<uint32_t>(T.TY().size() + 1);
    uint32_t b = static_cast<uint32_t>(T.TCb().size() + 1);
    uint32_t c = static_cast<uint32_t>(T.TCr().size() + 1);
    ThresholdSet newT;
    for (size_t i = 0; i < kNumCh; ++i) newT.T[i].reserve(kMaxCells);

    bool TY_changed = (a != 1);
    bool TCb_changed = (b != 1);
    bool TCr_changed = (c != 1);
    for (uint32_t iter = 0; iter < max_iters; ++iter) {
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

  // Compute total entropy cost for fixed thresholds over a stream.
  // Returns cost in fixed-point units, divide by `kFScale` for bits.
  int64_t TotalCost(const ThresholdSet& T) {
    const JPEGOptData& d = data();
    uint32_t na = static_cast<uint32_t>(T.TY().size() + 1);
    uint32_t num_cells = na * static_cast<uint32_t>(T.TCb().size() + 1) *
                         static_cast<uint32_t>(T.TCr().size() + 1);
    axis_maps.Update(T);
    if (h_cnt.size() < num_cells) h_cnt.assign(num_cells, 0);
    if (N_cnt.size() < num_cells) N_cnt.assign(num_cells, 0);
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
                  uint32_t ci =
                      axis_maps.ax1_row[dc1_idx] + axis_maps.ax2_col[dc2_idx];
                  uint32_t idx = ci * na + axis_maps.ax0_to_k[dc0_idx];
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

  // Cluster `3*(a*b*c)` channel-cell contexts into `num_clusters` or less
  // groups using greedy agglomerative clustering that minimises entropy.
  // Returns `cluster_id[c * num_cells + cell]` for each (channel, 3D-DC cell)
  // and the total entropy of clustered distribution.
  StatusOr<Clustering> ClusterContexts(const ThresholdSet& thresholds,
                                       uint32_t num_clusters = kMaxClusters,
                                       bool overhead_aware_tail = true,
                                       ThreadPool* pool = nullptr) {
    StreamSweepAdapter<PartitioningCtx> stream_sweep = {*this};
    return Clustering::Build(data(), axis_maps, thresholds, num_clusters,
                             overhead_aware_tail, pool, stream_sweep);
  }

  // Return value of `RefineClustered`: the threshold set and entropy costs
  // after coordinate-descent refinement. `cost` is the combined AC + nz entropy
  // cost used by the outer optimizer to compare configurations. `nz_cost` (the
  // nonzero-count entropy portion) is broken out separately for logging.
  struct RefineResult {
    ThresholdSet thresholds;
    int64_t cost;
    int64_t nz_cost;
  };

  bool OptimizeRefineAxis(uint32_t axis, ThresholdSet* cur_T,
                          Clustering& clustering, RefineScratch& scratch,
                          int64_t* base_cost, ptrdiff_t search_radius) {
    Thresholds& thr = cur_T->T[axis];
    if (thr.empty()) return false;

    RefineCtx refine_ctx(data(), axis_maps, axis, *cur_T, scratch,
                         search_radius);

    bool changed = false;
    for (uint32_t thr_ind = 0; thr_ind < thr.size(); ++thr_ind) {
      int16_t optimized =
          refine_ctx.OptimizeThreshold(thr_ind, thr, clustering, base_cost);
      changed = (optimized != thr[thr_ind]) || changed;
      thr[thr_ind] = optimized;
    }
    return changed;
  }

  RefineResult RefineClustered(const ThresholdSet& thresholds,
                               Clustering& clustering, uint32_t max_iters = 5,
                               ptrdiff_t search_radius = 2048) {
    const JPEGOptData& d = data();
    RefineScratch scratch;
    ThresholdSet cur_T = clustering.PruneDeadThresholds(thresholds);
    scratch.local_cluster_boundary =
        clustering.BuildLocalClusterBoundaries(cur_T, d.channels);

    int64_t base_cost = clustering.clustered_cost;
    bool changed = true;
    for (uint32_t iter = 0; iter < max_iters && changed; ++iter) {
      changed = false;
      for (uint32_t axis = 0; axis < kNumCh; ++axis) {
        changed = OptimizeRefineAxis(axis, &cur_T, clustering, scratch,
                                     &base_cost, search_radius) ||
                  changed;
      }
    }

    return {cur_T, base_cost, clustering.ComputeNZCost(d)};
  }
};

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
    JXL_ASSIGN_OR_RETURN(auto ac_cnt, CountDCAC(jpeg_data, pool));

    JXL_RETURN_IF_ERROR(BuildBlockOptData(jpeg_data, pool, *ac_cnt));
  }

  JXL_RETURN_IF_ERROR(FinalizeSpatialIndexing(pool));

  JXL_RETURN_IF_ERROR(GenerateRLEStream(pool));

  return true;
}

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
                /*overhead_aware_tail=*/true,
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

struct FactorizationCandidate {
  uint32_t a;
  uint32_t b;
  uint32_t c;
  ThresholdSet init;
  int64_t rank_cost = std::numeric_limits<int64_t>::max();

  bool operator<(const FactorizationCandidate& rhs) const {
    if (rank_cost != rhs.rank_cost) return rank_cost < rhs.rank_cost;
    const uint32_t lhs_cells = a * b * c;
    const uint32_t rhs_cells = rhs.a * rhs.b * rhs.c;
    if (lhs_cells != rhs_cells) return lhs_cells > rhs_cells;
    if (a != rhs.a) return a > rhs.a;
    return std::tie(b, c) < std::tie(rhs.b, rhs.c);
  }
};

StatusOr<std::vector<FactorizationCandidate>>
RankAndTrimFactorizationCandidates(std::shared_ptr<const JPEGOptData> opt_data,
                                   const JPEGCtxEffortParams& effort,
                                   ThreadPool* pool) {
  const auto factorizations = opt_data->MaximalFactorizations();
  if (!factorizations.empty()) {
    JXL_DEBUG_V(2, "Searching %i maximal factorizations\n",
                static_cast<int>(factorizations.size()));
  }

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
  if (effort.keep_top_k == 0 || candidates.size() <= effort.keep_top_k) {
    return candidates;
  }

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

  std::stable_sort(candidates.begin(), candidates.end());
  candidates.resize(effort.keep_top_k);
  return candidates;
}

}  // namespace

Status OptimizeJPEGContextMap(const jpeg::JPEGData& jpeg_data,
                              SpeedTier speed_tier, BlockCtxMap& ctx_map,
                              ThreadPool* pool) {
  auto opt_data = std::make_shared<JPEGOptData>();
  JXL_RETURN_IF_ERROR(opt_data->BuildFromJPEG(jpeg_data, pool));
  const JPEGCtxEffortParams effort =
      JPEGCtxEffortParams::FromSpeedTier(speed_tier);

  JXL_ASSIGN_OR_RETURN(
      std::vector<FactorizationCandidate> candidates,
      RankAndTrimFactorizationCandidates(opt_data, effort, pool));
  if (candidates.empty()) return true;

  JXL_DEBUG_V(2,
              "JPEG ctx effort at speed tier %i: %i candidates, rank_iters=%u "
              "final_iters=%u refine_iters=%u\n",
              static_cast<int>(speed_tier), static_cast<int>(candidates.size()),
              effort.rank_iters, effort.final_iters, effort.refine_iters);

  int64_t best_cost = std::numeric_limits<int64_t>::max();
  ThresholdSet best_thr;
  ContextMap best_ctx;
  std::mutex mu;

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

        Clustering cl_result;
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
          nz_cost = cl_result.ComputeNZCost(*opt_data);
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
  for (size_t c = 0; c < opt_data->channels; ++c) {
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
