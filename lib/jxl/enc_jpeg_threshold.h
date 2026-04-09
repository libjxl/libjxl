// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_JPEG_THRESHOLD_H_
#define LIB_JXL_ENC_JPEG_THRESHOLD_H_

#include "lib/jxl/enc_Knuth_partition.h"
#include "lib/jxl/enc_jpeg_opt_data.h"

namespace jxl {

struct AxisMaps {
  const JPEGOptData& image;
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
      : image(image),
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
    uint16_t M0 = static_cast<uint16_t>(image.DC_vals[ax0].size());
    size_t M1 = image.DC_vals[ax1].size();
    size_t M2 = image.DC_vals[ax2].size();
    for (uint16_t i = 0; i < M0; ++i) {
      ax0_to_k[i] = ax0_identity ? i : Bkt(image.DC_vals[ax0][i], T0);
      if (ax0_identity) k_to_dc0[i] = i;
    }
    size_t n2 = T2.size() + 1;
    for (size_t i = 0; i < M1; ++i)
      ax1_row[i] = static_cast<uint16_t>(Bkt(image.DC_vals[ax1][i], T1) * n2);
    for (size_t i = 0; i < M2; ++i) ax2_col[i] = Bkt(image.DC_vals[ax2][i], T2);
  }

  void Update(const ThresholdSet& thresholds) {
    Update(0, thresholds.TY(), thresholds.TCb(), thresholds.TCr());
  }

  // Prepare cell maps `(ax1_row, ax2_col)` and axis bucket maps
  // `(dc0_to_k, k_to_dc0)`. Returns the actual number of axis buckets `M_eff`.
  uint32_t PrepareBucketing(uint32_t axis, uint32_t M_target,
                            const Thresholds& T1, const Thresholds& T2) {
    uint32_t M = static_cast<uint32_t>(image.DC_vals[axis].size());
    uint32_t M_eff = std::min(M, M_target);
    if (M_eff == M) {
      Update(axis, {}, T1, T2, true);
      return M;
    }

    // TODO: do not repeat this each time, save axis buckets somewhere
    Thresholds bkt_thresh = image.InitThresh(static_cast<int>(axis), M_eff);
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

    std::vector<uint32_t> solution = Knuth_solver.Solve(num_intervals, M_eff);
    Thresholds thresholds;
    for (auto t : solution)
      thresholds.push_back(d.DC_vals[axis][axis_maps.k_to_dc0[t]]);
    return thresholds;
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
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_JPEG_THRESHOLD_H_
