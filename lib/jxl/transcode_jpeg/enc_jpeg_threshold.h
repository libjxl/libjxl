// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DC threshold optimization for JPEG lossless recompression.
//
// During JPEG recompression, AC coefficient runs are grouped into coding
// contexts determined by the associated DC values of the current component
// block and projected blocks from other components. The DC value ranges are
// partitioned into intervals by threshold vectors `(TY, TCb, TCr)`, one per
// channel. Each combination of intervals forms a 3D "cell"; all blocks whose
// associated DC values fall into the same cell share a coding context.
//
// Finer partitioning yields better-adapted contexts and lower entropy at the
// cost of negligibly more signalling overhead: each threshold value is encoded
// via `kDCThresholdDist` (6 bits for |packed| ≤ 15, 10 bits for |packed| ≤ 271,
// 18 bits for |packed| ≤ 65807; typical JPEG DC thresholds cost 10–18 bits),
// plus some bits for encoding extended context map - worst case of no
// compression at all gives
// (4 bits per added context id) * (3 components) * (32 added cells) = 384 bits
// (ANS histogram header overhead is near zero as it depends primarily on the
// number of clusters and not on the number of cells).
//
// This file provides `PartitioningCtx`, which owns the mutable scratch state
// and sweep logic for threshold optimization. The packed AC stream decoder
// lives in `enc_jpeg_stream.h`, and the DC-axis lookup tables live in
// `enc_jpeg_axis_maps.h`.
//
//   `PartitioningCtx`
//     Owns all mutable scratch state and implements the optimization sweeps:
//
//     - `OptimizeAxisSingleSplit` (K=2 fast path): finds the optimal single
//       threshold for one axis in `O(M_eff)` time using a one-write diff trick
//       over a score-diff array followed by one prefix sum.
//
//     - `FlushTerm` / `OptimizeAxisSingleSweep` (K≥3 general path): builds the
//       `M_eff×M_eff` cost matrix for the Knuth-optimized DP solver
//       incrementally via the same diff trick, using two-level bitmasks for
//       sparse dirty tracking.
//
//     - `OptimizeThresholds`: coordinate descent over all three channel axes,
//       calling `OptimizeAxisSingleSweep` per axis until convergence.
//
//     - `TotalCost`: evaluates the entropy cost `E = Σ ftab[N] − Σ ftab[h]`
//       for a fixed threshold set, used to compare configurations.
//
// All costs are in fixed-point units (divide by `kFScale` to get bits).

#ifndef LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_THRESHOLD_H_
#define LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_THRESHOLD_H_

#include <memory>

#include "lib/jxl/enc_Knuth_partition.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_axis_maps.h"

namespace jxl {

JXL_INLINE uint32_t CountrZero64(uint64_t x) {
#if JXL_COMPILER_MSVC
  unsigned long idx;
  return _BitScanForward64(&idx, x) ? static_cast<int>(idx) : 64;
#else
  return x ? __builtin_ctzll(x) : 64;
#endif
}

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

  // Number of `uint64_t` words in `touched_h`/`touched_N`: one bit per flat
  // index slot `ci * M_eff + n`, packed 64 per word.
  static constexpr uint32_t kBinCount = kMaxCells / 2 * kDCTRange >> 6;
  // Number of `uint64_t` words in `group_touched_h`/`group_touched_N`: one bit
  // per 64-word group of `touched`, packed 64 per word.
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
  Thresholds OptimizeAxisSingleSplit(uint32_t axis, uint32_t ncells,
                                     uint32_t M_eff);
  // `axis==0`: sparse-hist path (bins already sorted by DC0 rank, no sort
  //            needed; uses `h_history`/`N_history`).
  void AccumulateSingleSplitAxis0(std::vector<int64_t>& score_diff,
                                  uint32_t& bin_mask, uint32_t& ctx_mask);
  // `axis!=0`: dense-cnt path (uses `h_cnt`/`N_cnt`, sweeps ranks linearly).
  void AccumulateSingleSplitOther(uint32_t axis, uint32_t ncells,
                                  uint32_t M_eff,
                                  std::vector<int64_t>& score_diff,
                                  uint32_t& bin_mask, uint32_t& ctx_mask);

  template <uint32_t Axis>
  void AccumulateSingleSplitOtherAxis(uint32_t ncells, uint32_t M_eff,
                                      std::vector<int64_t>& score_diff,
                                      uint32_t& bin_mask, uint32_t& ctx_mask);

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
    // With `idx = ci*M_eff + n` (`ci`-major), all ranks for a given `ci`
    // are contiguous in scan order, so a single history suffices.
    auto& history = h_history[0];
    history.clear();

    // Track the current cell boundary incrementally.
    uint32_t n_ci_start = 0;

    for (uint32_t hi_idx = 0; hi_idx < kGroupCount; ++hi_idx) {
      uint64_t group_mask = group_touched[hi_idx];

      while (group_mask) {
        uint32_t lo_idx = CountrZero64(group_mask);
        uint32_t group_idx = (hi_idx << 6) | lo_idx;
        uint64_t cell_mask = touched[group_idx];
        uint32_t base_bit_idx = group_idx << 6;

        // Fast-forward `n_ci_start` to the cell containing `base_bit_idx`.
        // Amortised O(1): total advances are bounded by `kMaxCells / 2`.
        while (base_bit_idx >= n_ci_start + M_eff) {
          n_ci_start += M_eff;
          history.clear();
        }

        while (cell_mask) {
          uint32_t t = CountrZero64(cell_mask);
          uint32_t bit_idx = base_bit_idx | t;

          // Advance if this bit crossed into the next cell.
          // A `while` handles the rare case where a single 64-bit word spans
          // multiple cells (only possible when `M_eff < 64`).
          while (bit_idx >= n_ci_start + M_eff) {
            n_ci_start += M_eff;
            history.clear();
          }

          uint32_t n = bit_idx - n_ci_start;

          // `costs[n * M_eff + l]` = diff contribution for interval `[l..n]`.
          int64_t* cost_row = &costs[n * M_eff];

          uint32_t freq = cnt[bit_idx];
          // `j_n` = cumulative count for ranks `[0..n]` in this `ci` group.
          uint32_t j_n = freq;  // starts with just the current rank `n`
          if (!history.empty()) {
            // Carry forward the running cumulative from earlier ranks.
            j_n += history.back().second;

            // One-write diff trick for the cost matrix:
            // adding `freq` at rank `n` changes `cost[l..n]` for all l ≤ n.
            // The change is piecewise-constant between spotted ranks, so we
            // only write at the boundaries (where the term changes).
            uint32_t l = 0;
            uint32_t j_before_l = 0;
            int64_t prev_term = 0;
            for (const Bin& h : history) {
              // `j_ln` = counts in `[l..n]` before adding `freq`.
              uint32_t j_ln = j_n - j_before_l;
              // Entropy change for interval `[l..n]` from adding `freq`.
              int64_t term = sign * (d.ftab[j_ln - freq] - d.ftab[j_ln]);
              cost_row[l] += term - prev_term;
              // Move to the next interval boundary.
              l = h.first + 1;
              j_before_l = h.second;
              prev_term = term;
            }
            // Last segment: `j_before_l = j_n - freq`, so `j_ln = freq`,
            // `j_ln - freq = 0`, `ftab[0] = 0` → `term = -sign * ftab[freq]`.
            cost_row[l] -= sign * d.ftab[freq] + prev_term;
          }
          // Record this rank for future `n`.
          history.emplace_back(static_cast<uint16_t>(n), j_n);
          cnt[bit_idx] = 0;            // reset count to clean the histogram
          cell_mask &= cell_mask - 1;  // clear processed mask bit
        }
        touched[group_idx] = 0;        // reset processed mask
        group_mask &= group_mask - 1;  // clear processed mask bit
      }
      group_touched[hi_idx] = 0;  // reset processed mask
    }
  }

  // Generic stream sweep used by coordinate-descent optimization.
  //
  // Optimizes `T->T[axis]` while keeping the other two axes fixed as specified
  // by `*T`. The newly computed thresholds are written into `*scratch`, and
  // swapped into `*T` only if they differ from the current axis thresholds.
  // Returns whether the axis changed.
  //
  // `bucket_thresholds` provides the per-axis bucketing prepared by
  // `InitThresh`. `PrepareBuckets` maps the swept axis onto `M_eff =
  // bucket_thresholds.size() + 1` closest-to-equal-population buckets,
  // with an internal identity fast path when `M_eff == M`.
  bool OptimizeAxisSingleSweep(uint32_t axis, ThresholdSet* T,
                               Thresholds* scratch,
                               const Thresholds& bucket_thresholds);

  template <uint32_t Axis>
  void SweepGeneralAxis(uint32_t M_eff);

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
  ThresholdSet OptimizeThresholds(ThresholdSet T,
                                  uint32_t M_target = UINT32_MAX,
                                  uint32_t max_iters = 20);

  // Compute total entropy cost for fixed thresholds over a stream.
  // Returns cost in fixed-point units, divide by `kFScale` for bits.
  int64_t TotalCost(const ThresholdSet& T);
};

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_THRESHOLD_H_
