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
  // `idx = ci * M_eff + n` (up to 64ki slots for full `M_eff = kDCTRange`).
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
  // index over the flat `cnt` array, so only dirty `(n, ci)` entries are
  // visited instead of scanning all `M_eff * ncells` slots. The flat index is
  // `bit_idx = n * ncells + ci` (`n`-major / rank-major), so scanning in
  // increasing `bit_idx` order visits all `ci` values for rank 0, then all
  // `ci` values for rank 1, etc. Ranks for a given `ci` are therefore seen in
  // ascending order across the scan — a prerequisite for per-cell history
  // accumulation. `cnt`, `touched`, and `group_touched` are all reset to zero
  // here so they are ready for the next stream segment.
  template <int sign>
  void FlushTerm(std::vector<uint32_t>& cnt,
                 std::vector<uint64_t>& group_touched,
                 std::vector<uint64_t>& touched, uint32_t M_eff,
                 uint32_t ncells);

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
  void SweepGeneralAxis(uint32_t M_eff, uint32_t ncells);

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
