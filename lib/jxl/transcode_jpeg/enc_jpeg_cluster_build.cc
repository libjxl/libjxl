// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Construction of clustered JPEG contexts from image data.
// See `enc_jpeg_cluster.h` for the public interface and role of `Clustering`.
//
// This file contains one internal type kept out of the header:
//
//   `ClusteringBuildCtx`
//     Populates per-cell histograms from `JPEGOptData` for one threshold set,
//     then invokes `Clustering::AgglomerativeClustering` to build a fresh
//     clustered solution.

#include "lib/jxl/transcode_jpeg/enc_jpeg_axis_maps.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_cluster.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_stream.h"

namespace jxl {

namespace {

// Drives histogram population for a given threshold set, then hands off to
// `AgglomerativeClustering`. Used exclusively by `Clustering::Build`.
struct ClusteringBuildCtx {
  const JPEGOptData& d;
  AxisMaps axis_maps;
  // Number of Y intervals (`thresholds.TY().size() + 1`).
  uint32_t n0;
  uint32_t num_cells;
  uint32_t total_ctxs;

  // Precomputes axis maps and cell-count fields from the threshold set.
  ClusteringBuildCtx(const JPEGOptData& d, const ThresholdSet& thresholds)
      : d(d),
        axis_maps(d),
        n0(static_cast<uint32_t>(thresholds.TY().size() + 1)),
        num_cells(n0 * static_cast<uint32_t>(thresholds.TCb().size() + 1) *
                  static_cast<uint32_t>(thresholds.TCr().size() + 1)),
        total_ctxs(kNumCh * num_cells) {
    axis_maps.Update(thresholds);
    // This build context only ever needs full 3D cell ids, so fold the final
    // `* n0` into the two cross-axis lookup tables once up front. That turns
    // the hot `cell` computation below into three table lookups and two adds.
    for (uint16_t& row : axis_maps.ax1_row) row = row * n0;
    for (uint16_t& col : axis_maps.ax2_col) col = col * n0;
  }

  // Fills all four histogram arrays in `cl` with counts derived from the image
  // data, ready for `AgglomerativeClustering`. Each array is indexed by:
  //   `ctx_id = c * num_cells + cell`
  // where `c` ∈ [0, kNumCh) is the channel and
  //   `cell = ax1_row[dc1] + ax2_col[dc2] + ax0_to_k[dc0]`
  // is the DC-threshold cell that the block belongs to; `ax1_row`/`ax2_col`
  // are already premultiplied by `n0` in the constructor.
  //
  // **Pass 1 — AC coefficients (via `SweepACStream`)**
  // Each reset stream frame carries a packed `bin_state` word:
  //   `bin_state = channel * kACSymbolCount + zdc * kACTokenCount + token`
  //
  // - `hist_h[ctx_id]` accumulates counts of `(zdc, token)` bins (compacted via
  //   `CompactHBin`); this is the AC-symbol histograms used in entropy coding.
  // - `hist_N[ctx_id]` accumulates counts in `zdc` contexts; this is the
  //   "context frequency" histogram `N` used in the entropy cost model.
  //
  // **Pass 2 — nonzero-count histograms (block iteration)**
  // For each block, its nonzero AC count and predictor bucket `pb` are added:
  // - `hist_nz_h[ctx_id]` counts `(pb, nz_count)` bins (via
  //   `NZHistogramIndex`); used as the histogram for nz-count coding.
  // - `hist_nz_N[ctx_id]` counts events in predictor bucket `pb`; the
  //   corresponding N.
  // Grayscale (`d.channels == 1`) takes a fast path: there is only one
  // channel and the cell index collapses to `ax0_to_k[dc0_bucket]`, so the
  // multichannel subsampling coordinate mapping is skipped entirely.
  void PopulateHistograms(Clustering* cl) {
    SweepACStream(
        d.AC_stream, []() {}, []() {},
        [&](uint32_t dc0_idx, uint32_t dc1_idx, uint32_t dc2_idx, uint32_t run,
            uint32_t bin_state) {
          const uint32_t c = JpegTranscodeACBinChannel(bin_state);
          const uint32_t cell = axis_maps.ax1_row[dc1_idx] +
                                axis_maps.ax2_col[dc2_idx] +
                                axis_maps.ax0_to_k[dc0_idx];
          const uint32_t ctx_id = c * num_cells + cell;
          cl->hist_h[ctx_id].Add(
              d.CompactHBin(JpegTranscodeACBinSymbol(bin_state)), run);
          const uint32_t zdc = JpegTranscodeACBinZDC(bin_state);
          cl->hist_N[ctx_id].Add(zdc, run);
        });

    if (d.channels == 1) {
      for (uint32_t b = 0; b < d.num_blocks[0]; ++b) {
        const uint32_t ctx_id = axis_maps.ax0_to_k[d.block_DC_idx[0][b]];
        const uint32_t nz_count = d.block_nonzeros[0][b];
        const uint32_t pb = d.block_nz_pred_bucket[0][b];
        cl->hist_nz_h[ctx_id].Add(NZHistogramIndex(pb, nz_count));
        cl->hist_nz_N[ctx_id].Add(pb);
      }
      return;
    }

    for (uint32_t c = 0; c < kNumCh; ++c) {
      for (uint32_t by = 0; by < d.block_grid_h[c]; ++by) {
        for (uint32_t bx = 0; bx < d.block_grid_w[c]; ++bx) {
          // Cross-component DC references use the block reached by projecting
          // the current block's top-left anchor into each component grid.
          const uint32_t b0 = MapTopLeftBlockIndex(d, c, by, bx, 0);
          const uint32_t b1 = MapTopLeftBlockIndex(d, c, by, bx, 1);
          const uint32_t b2 = MapTopLeftBlockIndex(d, c, by, bx, 2);
          const uint32_t cell = axis_maps.ax1_row[d.block_DC_idx[1][b1]] +
                                axis_maps.ax2_col[d.block_DC_idx[2][b2]] +
                                axis_maps.ax0_to_k[d.block_DC_idx[0][b0]];
          const uint32_t ctx_id = c * num_cells + cell;

          const uint32_t b = by * d.block_grid_w[c] + bx;
          const uint32_t nz_count = d.block_nonzeros[c][b];
          const uint32_t pb = d.block_nz_pred_bucket[c][b];
          cl->hist_nz_h[ctx_id].Add(NZHistogramIndex(pb, nz_count));
          cl->hist_nz_N[ctx_id].Add(pb);
        }
      }
    }
  }

  // Allocates histograms, populates them, then runs agglomerative clustering.
  StatusOr<Clustering> Build(uint32_t num_clusters, bool overhead_aware_tail,
                             ThreadPool* pool) {
    Clustering cl;
    cl.hist_h.assign(total_ctxs, CompactHistogram(d.num_zdctok));
    cl.hist_N.resize(total_ctxs);
    cl.hist_nz_h.resize(total_ctxs);
    cl.hist_nz_N.resize(total_ctxs);
    PopulateHistograms(&cl);
    JXL_RETURN_IF_ERROR(
        cl.AgglomerativeClustering(d, num_clusters, overhead_aware_tail, pool));
    return cl;
  }
};

}  // namespace

// Populates per-cell histograms from `d` for the given `thresholds` and
// runs agglomerative clustering. The main entry point for building a fresh
// `Clustering` from image data.
StatusOr<Clustering> Clustering::Build(const JPEGOptData& d,
                                       const ThresholdSet& thresholds,
                                       uint32_t num_clusters,
                                       bool overhead_aware_tail,
                                       ThreadPool* pool) {
  ClusteringBuildCtx build_ctx(d, thresholds);
  return build_ctx.Build(num_clusters, overhead_aware_tail, pool);
}

}  // namespace jxl
