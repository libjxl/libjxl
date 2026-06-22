// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_KNUTH_PARTITION_H_
#define LIB_JXL_ENC_KNUTH_PARTITION_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace jxl {

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
  std::vector<uint32_t> Solve(uint32_t K, uint32_t M_eff,
                              int64_t* best_cost = nullptr) {
    if (M_eff == 0) {
      if (best_cost != nullptr) *best_cost = 0;
      return {};
    }
    if (M_eff <= 1 || K <= 1) {
      if (best_cost != nullptr) *best_cost = GetDiffCost(M_eff, 0, M_eff - 1);
      return {};
    }

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

    std::vector<uint32_t> thresholds;
    thresholds.reserve(K - 1);
    uint32_t v = M_eff - 1;
    for (uint32_t k = K - 1; k > 0; --k) {
      uint32_t s = split_table[k * M_eff + v];
      thresholds.push_back(s);
      v = s - 1;
    }
    std::reverse(thresholds.begin(), thresholds.end());
    if (best_cost != nullptr) *best_cost = DP_prev[M_eff - 1];
    return thresholds;
  }
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_KNUTH_PARTITION_H_
