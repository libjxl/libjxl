// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Merge phase of context clustering for JPEG lossless recompression.
// See `enc_jpeg_cluster.h` for the public interface and algorithm overview.
//
// This file contains one internal type kept out of the header:
//
//   `AgglomerativeCtx`
//     Implements the two-phase greedy merge loop described in
//     `Clustering::AgglomerativeClustering`. Holds all working state (entropy
//     vector `E`, the active-cluster list, the symmetric `deltas` cache, and
//     the union-find `parent` array) and exposes one public entry point `Run`.

#include "lib/jxl/transcode_jpeg/enc_jpeg_cluster.h"

#include <algorithm>
#include <mutex>
#include <numeric>

namespace jxl {

// Entropy increase of merging two histogram bins with counts `a` and `b`.
// Uses the bounded `ftab` table; only valid when `a + b < ftab.size()`.
JXL_INLINE int64_t MergeCost(const JPEGOptData& d, uint32_t a, uint32_t b) {
  return d.ftab[a + b] - d.ftab[a] - d.ftab[b];
}

// NZ-histogram variant: falls back to direct computation for large counts.
JXL_INLINE int64_t NZMergeCost(const JPEGOptData& d, uint32_t a, uint32_t b) {
  return d.NZFTab(a + b) - d.NZFTab(a) - d.NZFTab(b);
}

namespace {

// Stateful worker that implements the agglomerative algorithm on top of a
// `Clustering`. Decomposed into small focused methods so the two phases can
// share infrastructure:
//   - Phase 1 (`RunGreedyMerges`): iterate `FindBestMerge → ApplyMerge
//     → UpdateDistances` until `active_clusters == num_clusters`.
//     Uses a symmetric `deltas[total_ctxs²]` cache; recomputes only
//     the row of the surviving cluster after each merge.
//   - Phase 2 (`RunOverheadAwareTail`): continue merging past
//     `num_clusters` as long as `entropy + signalling_overhead` decreases.
//     Each candidate merge is applied tentatively; `SaveRollback`/
//     `RestoreRollback` undo it if rejected. An early-exit `cutoff` passed
//     to `ComputeSignallingOverhead` avoids full overhead recomputation when
//     it cannot improve.
struct AgglomerativeCtx {
  // Snapshot of the two cluster histograms involved in a candidate phase-2
  // merge, plus the active list, sufficient to fully undo the merge.
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

  // References into the `Clustering` being built.
  Clustering& clustering;
  const JPEGOptData& d;
  ThreadPool* pool;
  CompactHistogramSet& hist_h;
  DenseNHistogramSet& hist_N;
  DenseNZHistogramSet& hist_nz_h;
  DenseNZPredHistogramSet& hist_nz_N;
  ContextMap& ctx_map;
  // Total number of initial contexts (`kNumCh * num_cells`).
  const uint32_t total_ctxs;
  // Per-context entropy `E[i] = Σ ftab[N] − Σ ftab[h]`.
  std::vector<int64_t> E;
  // Indices of currently live (not yet merged) cluster roots.
  std::vector<uint32_t> active;
  // Snapshot of `active` before any merges; used in `Finalize` to walk all
  // original contexts through the union-find.
  std::vector<uint32_t> initial_active;
  uint32_t active_clusters = 0;
  // Union-find parent array; `parent[i] == i` for roots.
  std::vector<uint32_t> parent;
  // Symmetric pairwise merge-cost cache: `deltas[min(a,b)*total_ctxs+max(a,b)]`
  // holds `MergeDelta(a, b)`. Indexed via `Delta(a, b)`.
  std::vector<int64_t> deltas;
  int64_t current_entropy_cost = 0;

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
        parent(total_ctxs) {}

  // Accessor for the symmetric delta cache.
  int64_t& Delta(uint32_t cl_a, uint32_t cl_b) {
    return deltas[std::min(cl_a, cl_b) * total_ctxs + std::max(cl_a, cl_b)];
  }

  // Computes `E[i]` for every context in parallel. `current_entropy_cost` is
  // initialised separately by summing `E` after this call.
  Status InitEntropy() {
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, total_ctxs, ThreadPool::NoInit,
        [&](uint32_t i, size_t) -> Status {
          int64_t local_E = 0;
          for (uint32_t id : hist_h[i].used_ids)
            local_E -= d.ftab[hist_h[i].Get(id)];
          for (uint32_t freq : hist_N[i]) local_E += d.ftab[freq];
          for (uint32_t freq : hist_nz_N[i]) local_E += d.NZFTab(freq);
          for (uint32_t freq : hist_nz_h[i]) local_E -= d.NZFTab(freq);
          E[i] = local_E;
          return true;
        },
        "InitEntropy"));
    return true;
  }

  // Populates `active` with all non-empty context indices, copies it to
  // `initial_active`, and resets `parent` to the identity mapping.
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

  // Returns `E(merged) − E(a) − E(b)`: the entropy increase of merging `cl_a`
  // and `cl_b`. Iterates over the smaller `used_ids` set for the `hist_h` term.
  int64_t MergeDelta(uint32_t cl_a, uint32_t cl_b) const {
    int64_t delta = 0;

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
        delta -= MergeCost(d, freq_a, freq_b);
      }
    }
    for (size_t bin = 0; bin < hist_N[cl_a].size(); ++bin) {
      uint32_t freq_a = hist_N[cl_a][bin];
      uint32_t freq_b = hist_N[cl_b][bin];
      if (freq_a != 0 && freq_b != 0) {
        delta += MergeCost(d, freq_a, freq_b);
      }
    }
    for (size_t bin = 0; bin < hist_nz_h[cl_a].size(); ++bin) {
      uint32_t freq_a = hist_nz_h[cl_a][bin];
      uint32_t freq_b = hist_nz_h[cl_b][bin];
      if (freq_a != 0 && freq_b != 0) {
        delta -= NZMergeCost(d, freq_a, freq_b);
      }
    }
    for (size_t bin = 0; bin < hist_nz_N[cl_a].size(); ++bin) {
      uint32_t freq_a = hist_nz_N[cl_a][bin];
      uint32_t freq_b = hist_nz_N[cl_b][bin];
      if (freq_a != 0 && freq_b != 0) {
        delta += NZMergeCost(d, freq_a, freq_b);
      }
    }
    return delta;
  }

  // Fills the full `deltas` cache for all active pairs in parallel.
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

  // Scans the delta cache in parallel to find the pair `(best_i, best_j)`
  // with the smallest merge cost, reducing per-thread bests under a mutex.
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

  // Commits the merge: adds `b`'s histograms into `a`, clears `b`, sets
  // `parent[b] = a`, and removes `b` from `active`. Returns `{a_id, b_id}`.
  std::pair<uint32_t, uint32_t> ApplyMerge(size_t best_i, size_t best_j,
                                           int64_t best_delta) {
    uint32_t a_id = active[best_i];
    uint32_t b_id = active[best_j];
    E[a_id] += E[b_id] + best_delta;
    hist_N[a_id].AddHistogram(hist_N[b_id]);
    hist_h[a_id].AddHistogram(hist_h[b_id]);
    hist_nz_N[a_id].AddHistogram(hist_nz_N[b_id]);
    hist_nz_h[a_id].AddHistogram(hist_nz_h[b_id]);
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

  // Recomputes `Delta(a_id, x)` for every surviving cluster `x ≠ a_id`.
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

  // Snapshots `a_id`/`b_id` histograms and the active list into `rollback`.
  // `b_id`'s histograms are moved (not copied) to avoid a redundant allocation.
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

  // Reverts the merge of `a_id` ← `b_id` using the saved snapshot.
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

  // Phase 1: greedy merge loop. Repeatedly finds and applies the minimum-cost
  // merge until `active_clusters == num_clusters`.
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

  // Phase 2: continue merging past `num_clusters` while (entropy +
  // signalling overhead) keeps decreasing. Each candidate merge is applied
  // tentatively and rolled back if it does not improve the total cost.
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
      hist_N[a_id].AddHistogram(rollback->hist_N_b);
      hist_h[a_id].AddHistogram(rollback->hist_h_b);
      hist_nz_N[a_id].AddHistogram(rollback->hist_nz_N_b);
      hist_nz_h[a_id].AddHistogram(rollback->hist_nz_h_b);
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

  // Finalises the clustering: path-compresses `parent[]` to assign `ctx_map`,
  // then compacts the histogram arrays to contain only surviving clusters.
  void Finalize() {
    clustering.clustered_cost = 0;
    for (uint32_t k : active) clustering.clustered_cost += E[k];

    clustering.ctx_num = active_clusters;
    // Two-pass iterative path compression: walk to root, then compress.
    auto find_cluster = [&](uint32_t ctx) -> uint32_t {
      uint32_t root = ctx;
      while (parent[root] != root) root = parent[root];
      while (parent[ctx] != root) {
        uint32_t next = parent[ctx];
        parent[ctx] = root;
        ctx = next;
      }
      return root;
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

  // Top-level entry point: runs both phases and finalises the clustering.
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

}  // namespace

// Runs agglomerative clustering on the histograms already stored in `this`.
Status Clustering::AgglomerativeClustering(const JPEGOptData& d,
                                           uint32_t num_clusters,
                                           bool overhead_aware_tail,
                                           ThreadPool* pool) {
  AgglomerativeCtx ctx(*this, d, pool);
  return ctx.Run(num_clusters, overhead_aware_tail);
}

}  // namespace jxl
