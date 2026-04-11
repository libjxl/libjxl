// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/transcode_jpeg/enc_jpeg_threshold.h"

#include <limits>

#include "lib/jxl/transcode_jpeg/enc_jpeg_stream.h"

namespace jxl {

// Accumulates the K=2 score-diff curve for the sparse `axis == 0` path.
void PartitioningCtx::AccumulateSingleSplitAxis0(
    std::vector<int64_t>& score_diff, uint32_t& bin_mask, uint32_t& ctx_mask) {
  const JPEGOptData& d = data();
  // Converts the sparse `(rank, count)` list accumulated in `ch[ci]` for
  // each touched cell into additive contributions to `score_diff`.
  //
  // `ch[ci]` contains `(dc_k_idx, run)` bins in strictly increasing rank
  // order (the accumulation loop merges same-rank entries on the fly).
  //
  // For split rank `s`, the entropy term for this group is:
  //   `term(s) = sign * (ftab[j_before_l(s)] + ftab[total - j_before_l(s)])`
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
      uint32_t ci = CountrZero64(mask);
      mask &= mask - 1;
      auto& hist = ch[ci];
      uint32_t total = 0;
      for (const auto& hi : hist) total += hi.second;
      uint32_t j_before_l = 0;
      uint32_t l = 1;
      int64_t prev_term = 0;
      for (const auto& h : hist) {
        int64_t term = sign * (d.ftab[j_before_l] + d.ftab[total - j_before_l]);
        score_diff[l] += term - prev_term;
        j_before_l += h.second;
        l = static_cast<uint32_t>(h.first) + 1;
        prev_term = term;
      }
      score_diff[l] += sign * d.ftab[total] - prev_term;
      hist.clear();
    }
  };

  // Single pass over `AC_stream`. Decode and accumulate
  // `(dc_k_idx, run)` into `h_hist[ci]` and `N_hist[ci]`.
  SweepACStream(
      d.AC_stream, [&]() { flush_histogram(h_history, bin_mask, -1); },
      [&]() { flush_histogram(N_history, ctx_mask, +1); },
      [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
          uint32_t) {
        uint32_t dc_k_bkt = axis_maps.ax0_to_k[dc0_idx];
        uint32_t ci = axis_maps.ax1_row[dc1_idx] + axis_maps.ax2_col[dc2_idx];
        bin_mask |= (1U << ci);
        ctx_mask |= (1U << ci);
        if (!h_history[ci].empty() && h_history[ci].back().first == dc_k_bkt) {
          h_history[ci].back().second += run;
        } else {
          h_history[ci].emplace_back(static_cast<uint16_t>(dc_k_bkt), run);
        }
        if (!N_history[ci].empty() && N_history[ci].back().first == dc_k_bkt) {
          N_history[ci].back().second += run;
        } else {
          N_history[ci].emplace_back(static_cast<uint16_t>(dc_k_bkt), run);
        }
      });
  flush_histogram(h_history, bin_mask, -1);
  flush_histogram(N_history, ctx_mask, +1);
}

// K=2 fast path for `axis != 0` using dense `h_cnt` sweep.
// Accumulates `h_cnt[ci * M_eff + dc_k_bkt]` in one stream pass, then at each
// bin/ctx flush sweeps `n=0..M_eff-1` per touched cell (O(M_eff)) and applies
// the one-write diff to `score_diff`. Another approach is to use history like
// in `AccumulateSingleSplitAxis0` but with sorting — it is slower.
void PartitioningCtx::AccumulateSingleSplitOther(
    uint32_t axis, uint32_t ncells, uint32_t M_eff,
    std::vector<int64_t>& score_diff, uint32_t& bin_mask, uint32_t& ctx_mask) {
  const JPEGOptData& d = data();
  uint32_t cnt_size = M_eff * ncells;
  if (h_cnt.size() < cnt_size) h_cnt.assign(cnt_size, 0);
  if (N_cnt.size() < cnt_size) N_cnt.assign(cnt_size, 0);

  // Sweep dense `h_cnt` for each touched cell, update `score_diff`.
  // `sign=-1` for h-term, `sign=+1` for N-term.
  auto flush_dense = [&](std::vector<uint32_t>& cnt, uint32_t& mask, int sign) {
    while (mask) {
      uint32_t ci = CountrZero64(mask);
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
        int64_t term = sign * (d.ftab[j_before_l] + d.ftab[total - j_before_l]);
        score_diff[l] += term - prev_term;
        prev_term = term;
        j_before_l += f;
        l = n + 1;
      }
      score_diff[l] += sign * d.ftab[total] - prev_term;
    }
  };

  uint32_t ax1 = (axis + 1) % 3;
  uint32_t ax2 = (axis + 2) % 3;
  SweepACStream(
      d.AC_stream, [&]() { flush_dense(h_cnt, bin_mask, -1); },
      [&]() { flush_dense(N_cnt, ctx_mask, +1); },
      [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
          uint32_t) {
        uint32_t dc_arr[3] = {dc0_idx, dc1_idx, dc2_idx};
        uint32_t dc_k_bkt = axis_maps.ax0_to_k[dc_arr[axis]];
        uint32_t ci =
            axis_maps.ax1_row[dc_arr[ax1]] + axis_maps.ax2_col[dc_arr[ax2]];
        uint32_t idx = ci * M_eff + dc_k_bkt;
        h_cnt[idx] += run;
        bin_mask |= (1U << ci);
        N_cnt[idx] += run;
        ctx_mask |= (1U << ci);
      });
  flush_dense(h_cnt, bin_mask, -1);
  flush_dense(N_cnt, ctx_mask, +1);
}

// Evaluates the K=2 case for one axis in a single stream pass. It builds a
// diff-form score curve over split positions and returns the best cut.
Thresholds PartitioningCtx::OptimizeAxisSingleSplit(uint32_t axis,
                                                    uint32_t ncells,
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
    AccumulateSingleSplitAxis0(score_diff, bin_mask, ctx_mask);
  } else {
    AccumulateSingleSplitOther(axis, ncells, M_eff, score_diff, bin_mask,
                               ctx_mask);
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

// Optimizes one axis while keeping the other two threshold vectors fixed.
// Uses the K=2 fast path for a single split and the Knuth-DP path otherwise.
bool PartitioningCtx::OptimizeAxisSingleSweep(
    uint32_t axis, ThresholdSet* T, Thresholds* scratch,
    const Thresholds& bucket_thresholds) {
  JXL_DASSERT(T != nullptr);
  JXL_DASSERT(scratch != nullptr);
  const JPEGOptData& d = data();
  Thresholds& T0 = T->T[axis];
  const Thresholds& T1 = T->T[(axis + 1) % 3];
  const Thresholds& T2 = T->T[(axis + 2) % 3];
  uint32_t num_intervals = static_cast<uint32_t>(T0.size() + 1);
  if (num_intervals == 1) return false;
  uint32_t M = static_cast<uint32_t>(d.DC_vals[axis].size());
  if (M <= num_intervals) {  // exclude first DC value from thresholds
    scratch->assign(d.DC_vals[axis].begin() + 1, d.DC_vals[axis].end());
  } else {
    uint32_t ax1 = (axis + 1) % 3;
    uint32_t ax2 = (axis + 2) % 3;
    uint32_t n1 = static_cast<uint32_t>(T1.size() + 1);
    uint32_t n2 = static_cast<uint32_t>(T2.size() + 1);
    uint32_t ncells = n1 * n2;
    uint32_t M_eff = axis_maps.PrepareBuckets(axis, bucket_thresholds, T1, T2);

    if (num_intervals == 2) {
      // Fast path with `O(M_eff)` memory complexity
      *scratch = OptimizeAxisSingleSplit(axis, ncells, M_eff);
      // Extension of fast path above for `K=3` has proven disastrous
      // for performance (it has the same `O(M_eff^2)` complexity as the general
      // path) and is not implemented.
      // Divide and Conquer approach was also tested with no avail.
    } else {
      // General path with `O(M_eff^2)` memory complexity
      // Total number of cells probed is `n1 * n2 * M_eff`
      size_t cnt_size = ncells * M_eff;
      if (h_cnt.size() < cnt_size) h_cnt.assign(cnt_size, 0);
      if (N_cnt.size() < cnt_size) N_cnt.assign(cnt_size, 0);

      Knuth_solver.ResetCosts(M_eff * M_eff);

      SweepACStream(
          d.AC_stream,
          [&]() { FlushTerm<+1>(h_cnt, group_touched_h, touched_h, M_eff); },
          [&]() { FlushTerm<-1>(N_cnt, group_touched_N, touched_N, M_eff); },
          [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx,
              uint32_t run, uint32_t) {
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
      scratch->clear();
      for (auto t : solution) {
        scratch->push_back(d.DC_vals[axis][axis_maps.k_to_dc0[t]]);
      }
    }
  }

  bool changed = (*scratch != T0);
  if (changed) std::swap(T0, *scratch);
  return changed;
}

// Runs coordinate descent over Y/Cb/Cr threshold vectors for one factorization.
// Re-optimizes each axis until the thresholds stop changing or the iteration
// cap is hit.
ThresholdSet PartitioningCtx::OptimizeThresholds(ThresholdSet T,
                                                 uint32_t M_target,
                                                 uint32_t max_iters) {
  const JPEGOptData& d = data();
  uint32_t a = static_cast<uint32_t>(T.TY().size() + 1);
  uint32_t b = static_cast<uint32_t>(T.TCb().size() + 1);
  uint32_t c = static_cast<uint32_t>(T.TCr().size() + 1);
  Thresholds scratch;
  scratch.reserve(kMaxCells);
  // Build per-axis bucketing for the coordinate-descent sweeps.
  std::array<Thresholds, 3> bucket_thresholds;
  if (a != 1) bucket_thresholds[0] = InitThresh(d, 0, M_target);
  if (b != 1) bucket_thresholds[1] = InitThresh(d, 1, M_target);
  if (c != 1) bucket_thresholds[2] = InitThresh(d, 2, M_target);

  bool TY_changed = (a != 1);
  bool TCb_changed = (b != 1);
  bool TCr_changed = (c != 1);
  for (uint32_t iter = 0; iter < max_iters; ++iter) {
    if ((a != 1) && (iter == 0 || TCb_changed || TCr_changed)) {
      TY_changed =
          OptimizeAxisSingleSweep(0, &T, &scratch, bucket_thresholds[0]);
    } else {
      TY_changed = false;
    }
    if ((b != 1) && (iter == 0 || TY_changed || TCr_changed)) {
      TCb_changed =
          OptimizeAxisSingleSweep(1, &T, &scratch, bucket_thresholds[1]);
    } else {
      TCb_changed = false;
    }
    if ((c != 1) && (iter == 0 || TY_changed || TCb_changed)) {
      TCr_changed =
          OptimizeAxisSingleSweep(2, &T, &scratch, bucket_thresholds[2]);
    } else {
      TCr_changed = false;
    }
    if (!TY_changed && !TCb_changed && !TCr_changed) break;
  }
  return T;
}

// Computes the exact entropy objective for a fixed threshold set.
// It streams AC events once, accumulates per-cell histograms, and flushes them
// sparsely.
int64_t PartitioningCtx::TotalCost(const ThresholdSet& T) {
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
        size_t gi2 = (gi << 6) | CountrZero64(gm);
        uint64_t cm = touched_h[gi2];
        while (cm) {
          size_t idx = (gi2 << 6) | CountrZero64(cm);
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
        size_t gi2 = (gi << 6) | CountrZero64(gm);
        uint64_t cm = touched_N[gi2];
        while (cm) {
          size_t idx = (gi2 << 6) | CountrZero64(cm);
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

  SweepACStream(d.AC_stream, flush_h, flush_N,
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

}  // namespace jxl
