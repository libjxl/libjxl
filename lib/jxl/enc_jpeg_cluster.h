// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_JPEG_CLUSTER_H_
#define LIB_JXL_ENC_JPEG_CLUSTER_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <numeric>
#include <unordered_map>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_jpeg_histogram.h"
#include "lib/jxl/enc_jpeg_opt_data.h"
#include "lib/jxl/enc_jpeg_threshold.h"

namespace jxl {

struct ClusterBoundary {
  uint8_t lo;
  uint8_t hi;
};

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
  static StatusOr<Clustering> Build(const JPEGOptData& d,
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

template <class StreamSweepFn>
struct ClusteringBuildCtx {
  const JPEGOptData& d;
  AxisMaps axis_maps;
  StreamSweepFn& stream_sweep;
  uint32_t n0;
  uint32_t num_cells;
  uint32_t total_ctxs;

  ClusteringBuildCtx(const JPEGOptData& d, const ThresholdSet& thresholds,
                     StreamSweepFn& stream_sweep)
      : d(d),
        axis_maps(d),
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

// Small C++11 adapter that turns `owner.StreamSweep(...)` into a first-class
// callable object we can pass into `Clustering::Build(...)`. This keeps the
// build path generic over "something that can sweep the AC stream" without
// using generic lambdas, which are unavailable in C++11 mode.
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
    const JPEGOptData& d, const ThresholdSet& thresholds, uint32_t num_clusters,
    bool overhead_aware_tail, ThreadPool* pool, StreamSweepFn& stream_sweep) {
  ClusteringBuildCtx<StreamSweepFn> build_ctx(d, thresholds, stream_sweep);
  return build_ctx.Build(num_clusters, overhead_aware_tail, pool);
}

// Cluster `3*(a*b*c)` channel-DC cell contexts into `num_clusters` or less
// groups using greedy agglomerative clustering that minimises entropy.
// Returns `cluster_id[c * num_cells + cell]` for each (channel, 3D-DC cell)
// and the total entropy of clustered distribution.
StatusOr<Clustering> ClusterContexts(const PartitioningCtx& partitioning_ctx,
                                     const ThresholdSet& thresholds,
                                     uint32_t num_clusters = kMaxClusters,
                                     bool overhead_aware_tail = true,
                                     ThreadPool* pool = nullptr) {
  StreamSweepAdapter<PartitioningCtx> stream_sweep = {partitioning_ctx};
  const JPEGOptData& d = partitioning_ctx.data();
  return Clustering::Build(d, thresholds, num_clusters, overhead_aware_tail,
                           pool, stream_sweep);
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_JPEG_CLUSTER_H_
