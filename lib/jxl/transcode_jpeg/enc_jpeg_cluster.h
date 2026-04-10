// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Context clustering for JPEG lossless recompression.
//
// After DC thresholds partition the DC value space into cells (see
// `enc_jpeg_threshold.h`), each `(channel, cell)` pair is an initial coding
// context. As number of contexts is limited by 16 by JPEG XL standard, we need
// to cluster initial contexts into that or less number of clusters. This file
// reduces the number of distinct contexts by merging cells whose AC-coefficient
// distributions are similar enough that sharing a coding context produces
// minimal penalty - or even saves more entropy than it costs due to the
// decrease of histogram signalling cost.
//
// **Data structures**
//
//   `Clustering`
//     The result of one clustering run. Holds the per-cluster AC and
//     nonzero-count histograms (`hist_h, hist_N, hist_nz_h, hist_nz_N`),
//     the flat `ctx_map` that maps each `(channel, cell)` to a cluster index,
//     and the total entropy cost after clustering.  Key methods:
//       - `Build`: static factory; populates histograms from the image data
//         and runs `AgglomerativeClustering` to produce the initial clustering.
//       - `AgglomerativeClustering`: re-runs the greedy merge on the current
//         histograms (used after histogram re-population for a new threshold).
//       - `ComputeSignallingOverhead`: estimates ANS histogram header cost
//         (`ANSPopulationCost − ShannonEntropy`) summed over all clusters.
//       - `ComputeNZCost`: nonzero-count entropy portion of the total cost.
//       - `BuildLocalClusterBoundaries`: precomputes per-axis threshold
//         boundary views for incremental threshold refinement.
//       - `PruneDeadThresholds`: removes thresholds where every cell pair
//         across the boundary maps to the same cluster, then rebuilds
//         `ctx_map` on the pruned grid.
//
//   `ClusterBoundary`
//     Tiny helper holding the cluster IDs on the low and high sides of one
//     threshold boundary, used by the threshold refinement sweep.
#ifndef LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_CLUSTER_H_
#define LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_CLUSTER_H_

#include <cstdint>
#include <limits>
#include <memory>

#include "lib/jxl/base/status.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_histogram.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"

namespace jxl {

// Cluster IDs on both sides of one DC threshold boundary along a swept axis.
// `lo` = cluster of the interval below the threshold, `hi` = above.
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

  // Maximum token index produced by `HybridUintConfig(4, 2, 0)` for `ai ≤ 2047`:
  //   `token = split_token + (n - split_exp) * 4 + top_2_bits`
  //         `= 16 + (10 - 4) * 4 + 3 = 43   (n = FloorLog2(2047) = 10)`
  // Array size = 43 + 1 = 44.
  static constexpr uint32_t kSignallingMaxToken = 44;
  using SignallingTokenHist =
      std::array<std::array<uint32_t, kSignallingMaxToken>,
                 kZeroDensityContextCount>;

  // Scratch buffer for `ComputeSignallingOverhead`: maps `(zdc, token) → count`.
  // Heap-backed to keep `Clustering` small enough for the stack budget enforced
  // by `libjxl`. Declared `mutable` because the method is logically const; the
  // buffer holds no observable state between calls and is allocated lazily.
  mutable std::unique_ptr<SignallingTokenHist> signalling_token_hist_;

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
  static StatusOr<Clustering> Build(const JPEGOptData& d,
                                    const ThresholdSet& thresholds,
                                    uint32_t num_clusters,
                                    bool overhead_aware_tail, ThreadPool* pool);

  // Runs agglomerative clustering on the histograms already stored in `this`
  // (populated externally, e.g. after a threshold change). Updates `ctx_map`,
  // `ctx_num`, and `clustered_cost` in place.
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
      int64_t cutoff = std::numeric_limits<int64_t>::max()) const;

  // Computes the signalling overhead contribution of a single cluster.
  // Used by the overhead-aware merge tail to update the running overhead
  // incrementally after tentative merges.
  StatusOr<int64_t> ComputeClusterSignallingOverhead(
      const JPEGOptData& d, uint32_t cluster_id,
      int64_t cutoff = std::numeric_limits<int64_t>::max()) const;

  // Returns the nonzero-count entropy portion of the total cost:
  // `Σ NZFTab[N_nz] − Σ NZFTab[h_nz]` summed over all clusters and histograms.
  int64_t ComputeNZCost(const JPEGOptData& d) const;

  // Build threshold-major boundary views for axis-local cluster lookups:
  // `(channel, thr_ind, ci) -> {cluster_left, cluster_right}`. `ci` enumerates
  // the other two bucket axes in local `(k1, k2)` order, so `ApplySlice` can
  // scan all rows touched by one threshold contiguously.
  std::array<std::vector<ClusterBoundary>, kNumCh> BuildLocalClusterBoundaries(
      const ThresholdSet& thresholds, uint32_t channels) const;

  // Removes thresholds that are structurally inert: i.e. the `ctx_map`
  // assigns the same cluster on both sides. Rebuilds `ctx_map` to match the
  // pruned grid.
  ThresholdSet PruneDeadThresholds(const ThresholdSet& thresholds);
};

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_CLUSTER_H_
