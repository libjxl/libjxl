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
  FixedPointCost clustered_cost;
  uint32_t ctx_num;
  ContextMap ctx_map;

  CompactHistogramSet hist_h;
  DenseNHistogramSet hist_N;
  DenseNZHistogramSet hist_nz_h;
  DenseNZPredHistogramSet hist_nz_N;

  struct SignallingTokenHist {
    std::array<std::array<uint32_t, kACTokenCount>, kZeroDensityContextCount>
        hist;

    StatusOr<FixedPointCost> ClusterSignallingOverhead(
        const JPEGOptData& d, const CompactHistogram& cluster,
        FixedPointCost cutoff);
  };

  // Scratch buffer for `ComputeSignallingOverhead`: maps `(zdc, token) ->
  // count`. Heap-backed to keep `Clustering` small enough for the stack budget
  // enforced by `libjxl`. Declared `mutable` because the method is logically
  // const; the buffer holds no observable state between calls and is allocated
  // lazily.
  mutable std::unique_ptr<SignallingTokenHist> signalling_token_hist_;

  Clustering() : clustered_cost(0), ctx_num(0) {}
  Clustering(Clustering&&) = default;
  Clustering& operator=(Clustering&&) = default;
  Clustering(const Clustering&) = delete;
  Clustering& operator=(const Clustering&) = delete;

  // Builds a fresh clustering for one threshold set: populates the per-cell
  // AC and nonzero-count histograms from `JPEGOptData`, then runs
  // `AgglomerativeClustering` to merge them down to the requested cluster
  // budget.
  static StatusOr<Clustering> Build(const JPEGOptData& d,
                                    const ThresholdSet& thresholds,
                                    uint32_t num_clusters,
                                    bool overhead_aware_tail, ThreadPool* pool);

  // Runs agglomerative clustering on the histograms already stored in `this`
  // (populated externally).
  // Updates `ctx_map`, `ctx_num`, and `clustered_cost` in place.
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
  StatusOr<FixedPointCost> ComputeSignallingOverhead(
      const JPEGOptData& d,
      FixedPointCost cutoff = std::numeric_limits<FixedPointCost>::max()) const;

  // Computes the signalling overhead contribution of a single cluster.
  // Used by the overhead-aware merge tail to update the running overhead
  // incrementally after tentative merges.
  StatusOr<FixedPointCost> ComputeClusterSignallingOverhead(
      const JPEGOptData& d, uint32_t cluster_id,
      FixedPointCost cutoff = std::numeric_limits<FixedPointCost>::max()) const;

  // Returns the nonzero-count entropy portion of the total cost:
  // `Σ NZFTab[N_nz] − Σ NZFTab[h_nz]` summed over all clusters and histograms.
  FixedPointCost ComputeNZCost(const JPEGOptData& d) const;

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
