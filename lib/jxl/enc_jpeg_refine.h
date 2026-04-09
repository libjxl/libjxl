// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_JPEG_REFINE_H_
#define LIB_JXL_ENC_JPEG_REFINE_H_

#include "lib/jxl/enc_jpeg_cluster.h"
#include "lib/jxl/enc_jpeg_histogram.h"
#include "lib/jxl/enc_jpeg_opt_data.h"
#include "lib/jxl/enc_jpeg_threshold.h"

namespace jxl {

struct RefineScratch {
  CompactHistogramSet hist_h;
  DenseNHistogramSet hist_N;
  DenseNZPredHistogramSet hist_nz_N;
  DenseNZHistogramSet hist_nz_h;
  std::array<std::vector<ClusterBoundary>, kNumCh> local_cluster_boundary;

  void Reset(const Clustering& clustering) {
    hist_h = clustering.hist_h;
    hist_N = clustering.hist_N;
    hist_nz_N = clustering.hist_nz_N;
    hist_nz_h = clustering.hist_nz_h;
  }
};

// Short-lived per-axis refinement context. Owns the axis-local view of the
// immutable image data and the mutable scratch used while moving thresholds.
struct RefineCtx {
  struct ProjectedBlocks {
    uint32_t ref_block;
    uint32_t y_begin;
    uint32_t x_begin;
    uint32_t y_end;
    uint32_t x_end;
  };

  AxisMaps axis_maps;
  uint32_t axis;
  uint32_t ax1;
  uint32_t ax2;
  const std::vector<int16_t>& dc_axis;
  ptrdiff_t search_radius;
  uint32_t na;
  uint32_t num_rows;
  RefineScratch& scratch;

  RefineCtx(const JPEGOptData& d, uint32_t axis, const ThresholdSet& thresholds,
            RefineScratch& scratch, ptrdiff_t search_radius)
      : axis_maps(d),
        axis(axis),
        ax1((axis + 1) % 3),
        ax2((axis + 2) % 3),
        dc_axis(d.DC_vals[axis]),
        search_radius(search_radius),
        na(static_cast<uint32_t>(thresholds.T[axis].size() + 1)),
        num_rows(static_cast<uint32_t>(
            ((thresholds.TY().size() + 1) * (thresholds.TCb().size() + 1) *
             (thresholds.TCr().size() + 1)) /
            (thresholds.T[axis].size() + 1))),
        scratch(scratch) {
    axis_maps.Update(axis, thresholds.T[axis], thresholds.T[ax1],
                     thresholds.T[ax2]);
  }

  template <typename HistogramSet>
  int64_t MoveHistogramBin(HistogramSet* hist, uint32_t old_cl, uint32_t new_cl,
                           uint32_t bin, int64_t sign) const {
    const JPEGOptData& d = axis_maps.image;
    auto& from = (*hist)[old_cl];
    uint32_t old_freq = from.Get(bin);
    JXL_DASSERT(old_freq > 0);
    if (old_freq == 0) return 0;

    int64_t delta = sign * (d.NZFTab(old_freq - 1) - d.NZFTab(old_freq));
    from.Subtract(bin, 1);

    auto& to = (*hist)[new_cl];
    old_freq = to.Get(bin);
    delta += sign * (d.NZFTab(old_freq + 1) - d.NZFTab(old_freq));
    to.Add(bin, 1);
    return delta;
  }

  template <typename HistogramHSet, typename HistogramNSet>
  int64_t MoveHistogramEvent(uint32_t old_cl, uint32_t new_cl,
                             HistogramHSet* hist_h, HistogramNSet* hist_N,
                             uint32_t h_bin, uint32_t N_bin) const {
    return MoveHistogramBin(hist_h, old_cl, new_cl, h_bin, -1) +
           MoveHistogramBin(hist_N, old_cl, new_cl, N_bin, +1);
  }

  std::pair<uint32_t, uint32_t> BoundaryClusters(uint32_t thr_ind,
                                                 uint32_t channel, uint32_t ci,
                                                 bool upward) const {
    const auto& boundary_map = scratch.local_cluster_boundary[axis];
    const ClusterBoundary& boundary =
        boundary_map[(channel * (na - 1) + thr_ind) * num_rows + ci];
    return upward ? std::pair<uint32_t, uint32_t>{boundary.hi, boundary.lo}
                  : std::pair<uint32_t, uint32_t>{boundary.lo, boundary.hi};
  }

  uint32_t CellIndex(uint32_t dc_ax1_ind, uint32_t dc_ax2_ind) const {
    return axis_maps.ax1_row[dc_ax1_ind] + axis_maps.ax2_col[dc_ax2_ind];
  }

  static uint32_t DivCeilU32(uint32_t num, uint32_t den) {
    return (num + den - 1) / den;
  }

  uint32_t BlockIndex(uint32_t channel, uint32_t y, uint32_t x) const {
    return y * axis_maps.image.block_grid_w[channel] + x;
  }

  // The scheme used here is different with respect to other parts of the
  // library, where usually iterations are over common plane and just skip
  // subsampled blocks. This is done to avoid memory bloat, since we are
  // already using 2*32-bits per single AC coefficient in `block_bins` and
  // `AC_stream`.
  //
  // Match `PopulateClusterHistograms`: cross-component block references are
  // chosen from the block's top-left coordinate in the common plane.
  // `ref_block` is that direct top-left-mapped block in `dst_axis`.
  // `[y_begin, y_end) x [x_begin, x_end)` enumerates `dst_axis` blocks
  // whose own top-left anchors fall inside the moved `src_axis` block, so
  // these ranges may legitimately be empty under subsampling.
  ProjectedBlocks ProjectBlock(uint32_t src_axis, uint32_t dst_axis, uint32_t y,
                               uint32_t x) const {
    const JPEGOptData& d = axis_maps.image;

    uint32_t scaled_y = y * d.ss_y[src_axis];
    uint32_t scaled_x = x * d.ss_x[src_axis];
    return {BlockIndex(dst_axis, scaled_y / d.ss_y[dst_axis],
                       scaled_x / d.ss_x[dst_axis]),
            DivCeilU32(scaled_y, d.ss_y[dst_axis]),
            DivCeilU32(scaled_x, d.ss_x[dst_axis]),
            DivCeilU32((y + 1) * d.ss_y[src_axis], d.ss_y[dst_axis]),
            DivCeilU32((x + 1) * d.ss_x[src_axis], d.ss_x[dst_axis])};
  }

  uint32_t CellIndexFromBlocks(uint32_t b1, uint32_t b2) const {
    const JPEGOptData& d = axis_maps.image;
    return CellIndex(d.block_DC_idx[ax1][b1], d.block_DC_idx[ax2][b2]);
  }

  int64_t ApplyBlockMove(uint32_t channel, uint32_t block, uint32_t thr_ind,
                         uint32_t ci, bool upward) {
    const JPEGOptData& d = axis_maps.image;

    std::pair<uint32_t, uint32_t> boundary =
        BoundaryClusters(thr_ind, channel, ci, upward);
    uint32_t old_cl = boundary.first;
    uint32_t new_cl = boundary.second;
    if (old_cl == new_cl) return 0;

    uint32_t N_bin = d.block_nz_pred_bucket[channel][block];
    uint32_t h_bin = NZHistogramIndex(N_bin, d.block_nonzeros[channel][block]);
    int64_t delta = MoveHistogramEvent(old_cl, new_cl, &scratch.hist_nz_h,
                                       &scratch.hist_nz_N, h_bin, N_bin);
    for (uint32_t pi = d.block_offsets[channel][block];
         pi < d.block_offsets[channel][block + 1]; ++pi) {
      uint32_t bin = d.block_bins[channel][pi] & 0xFFFFF;
      uint32_t zdc = bin >> 11;
      delta += MoveHistogramEvent(old_cl, new_cl, &scratch.hist_h,
                                  &scratch.hist_N, d.CompactHBin(bin), zdc);
    }
    return delta;
  }

  int64_t ApplySlice(uint32_t thr_ind, ptrdiff_t slice, bool upward) {
    const JPEGOptData& d = axis_maps.image;

    const auto& sorted_blocks = d.DC_sorted_blocks[axis];
    const auto& block_off = d.DC_block_offsets[axis];

    int64_t cost_change = 0;
    uint32_t blk_lo = block_off[static_cast<size_t>(slice)];
    uint32_t blk_hi = block_off[static_cast<size_t>(slice + 1)];
    for (uint32_t bi = blk_lo; bi < blk_hi; ++bi) {
      uint32_t b0_xy = sorted_blocks[bi];
      uint32_t b0_y = b0_xy >> 16;
      uint32_t b0_x = b0_xy & 0xFFFF;
      uint32_t b0 = BlockIndex(axis, b0_y, b0_x);

      if (d.channels == 1) {
        cost_change += ApplyBlockMove(0, b0, thr_ind, 0, upward);
        continue;
      }

      ProjectedBlocks b1_area = ProjectBlock(axis, ax1, b0_y, b0_x);
      ProjectedBlocks b2_area = ProjectBlock(axis, ax2, b0_y, b0_x);

      cost_change += ApplyBlockMove(
          axis, b0, thr_ind,
          CellIndexFromBlocks(b1_area.ref_block, b2_area.ref_block), upward);

      for (uint32_t y1 = b1_area.y_begin; y1 < b1_area.y_end; ++y1) {
        for (uint32_t x1 = b1_area.x_begin; x1 < b1_area.x_end; ++x1) {
          uint32_t b1 = BlockIndex(ax1, y1, x1);
          uint32_t y2 = y1 * d.ss_y[ax1] / d.ss_y[ax2];
          uint32_t x2 = x1 * d.ss_x[ax1] / d.ss_x[ax2];
          uint32_t b2 = BlockIndex(ax2, y2, x2);
          cost_change += ApplyBlockMove(ax1, b1, thr_ind,
                                        CellIndexFromBlocks(b1, b2), upward);
        }
      }

      for (uint32_t y2 = b2_area.y_begin; y2 < b2_area.y_end; ++y2) {
        for (uint32_t x2 = b2_area.x_begin; x2 < b2_area.x_end; ++x2) {
          uint32_t b2 = BlockIndex(ax2, y2, x2);
          uint32_t y1 = y2 * d.ss_y[ax2] / d.ss_y[ax1];
          uint32_t x1 = x2 * d.ss_x[ax2] / d.ss_x[ax1];
          uint32_t b1 = BlockIndex(ax1, y1, x1);
          cost_change += ApplyBlockMove(ax2, b2, thr_ind,
                                        CellIndexFromBlocks(b1, b2), upward);
        }
      }
    }
    return cost_change;
  }

  int16_t RefineThreshold(uint32_t thr_ind, const Thresholds& thresholds,
                          Clustering& clustering, int64_t* base_cost) {
    ptrdiff_t cur_idx = AxisMaps::Bkt(thresholds[thr_ind] - 1, dc_axis);
    ptrdiff_t lo_edge = std::max(
        {cur_idx - search_radius,
         (thr_ind == 0) ? static_cast<ptrdiff_t>(0)
                        : AxisMaps::Bkt(thresholds[thr_ind - 1] - 1, dc_axis)});
    ptrdiff_t hi_edge =
        std::min({cur_idx + search_radius,
                  (thr_ind == thresholds.size() - 1)
                      ? static_cast<ptrdiff_t>(dc_axis.size())
                      : AxisMaps::Bkt(thresholds[thr_ind + 1] - 1, dc_axis)});

    ptrdiff_t best_idx = cur_idx;
    int64_t best_cost = *base_cost;

    scratch.Reset(clustering);
    int64_t current_cost = *base_cost;
    for (ptrdiff_t idx = cur_idx - 1; idx > lo_edge; --idx) {
      current_cost += ApplySlice(thr_ind, idx, false);
      if (current_cost < best_cost) {
        best_cost = current_cost;
        best_idx = idx;
      }
    }

    scratch.Reset(clustering);
    current_cost = *base_cost;
    for (ptrdiff_t idx = cur_idx + 1; idx < hi_edge; ++idx) {
      current_cost += ApplySlice(thr_ind, idx - 1, true);
      if (current_cost < best_cost) {
        best_cost = current_cost;
        best_idx = idx;
      }
    }

    if (best_idx == cur_idx) return thresholds[thr_ind];

    scratch.Reset(clustering);
    if (best_idx < cur_idx) {
      for (ptrdiff_t idx = cur_idx - 1; idx >= best_idx; --idx) {
        ApplySlice(thr_ind, idx, false);
      }
    } else {
      for (ptrdiff_t idx = cur_idx + 1; idx <= best_idx; ++idx) {
        ApplySlice(thr_ind, idx - 1, true);
      }
    }
    *base_cost = best_cost;
    std::swap(clustering.hist_h, scratch.hist_h);
    std::swap(clustering.hist_N, scratch.hist_N);
    std::swap(clustering.hist_nz_N, scratch.hist_nz_N);
    std::swap(clustering.hist_nz_h, scratch.hist_nz_h);
    return dc_axis[static_cast<size_t>(best_idx)];
  }
};

// Return value of `RefineClustered`: the threshold set and entropy costs
// after coordinate-descent refinement. `cost` is the combined AC + nz entropy
// cost used by the outer optimizer to compare configurations. `nz_cost` (the
// nonzero-count entropy portion) is broken out separately for logging.
struct RefineResult {
  ThresholdSet thresholds;
  int64_t cost;
  int64_t nz_cost;
};

RefineResult RefineClustered(const JPEGOptData& d,
                             const ThresholdSet& thresholds,
                             Clustering& clustering, uint32_t max_iters = 5,
                             ptrdiff_t search_radius = 2048) {
  RefineScratch scratch;
  ThresholdSet cur_T = clustering.PruneDeadThresholds(thresholds);
  scratch.local_cluster_boundary =
      clustering.BuildLocalClusterBoundaries(cur_T, d.channels);

  int64_t base_cost = clustering.clustered_cost;
  bool changed = true;
  for (uint32_t iter = 0; iter < max_iters && changed; ++iter) {
    changed = false;
    for (uint32_t axis = 0; axis < kNumCh; ++axis) {
      Thresholds& thr = cur_T.T[axis];
      if (thr.empty()) continue;

      RefineCtx refine_ctx(d, axis, cur_T, scratch, search_radius);

      for (uint32_t thr_ind = 0; thr_ind < thr.size(); ++thr_ind) {
        int16_t optimized =
            refine_ctx.RefineThreshold(thr_ind, thr, clustering, &base_cost);
        changed = (optimized != thr[thr_ind]) || changed;
        thr[thr_ind] = optimized;
      }
    }
  }

  return {cur_T, base_cost, clustering.ComputeNZCost(d)};
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_JPEG_REFINE_H_
