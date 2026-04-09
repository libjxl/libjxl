// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DC-axis lookup tables for JPEG lossless recompression.
//
// `AxisMaps` translates per-component DC value indices into partition-cell
// indices and axis-bucket indices under a given threshold set. It is shared
// between the threshold optimizer (`enc_jpeg_threshold.h`) and the histogram
// populator in `enc_jpeg_cluster.cc`, so it lives in its own header to avoid
// pulling the full `PartitioningCtx` machinery into the clustering code.

#ifndef LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_AXIS_MAPS_H_
#define LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_AXIS_MAPS_H_

#include <cstdint>
#include <vector>

#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"

namespace jxl {

// Lookup tables that map DC value indices to partition-cell offsets and
// axis bucket indices under a fixed threshold set.
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

  // Identity bucketing: each distinct DC value on `axis` is its own bucket.
  // Updates T1/T2 cell maps and sets `ax0_to_k`/`k_to_dc0` to identity.
  // Returns M (= M_eff in the identity case).
  uint32_t PrepareIdentityBuckets(uint32_t axis, const Thresholds& T1,
                                  const Thresholds& T2) {
    Update(axis, {}, T1, T2, /*ax0_identity=*/true);
    return static_cast<uint32_t>(image.DC_vals[axis].size());
  }

  // Equal-population bucketing: sets `ax0_to_k`/`k_to_dc0` from the
  // pre-computed `bkt_thresh_axis` (computed by `InitThresh` for this axis)
  // and updates T1/T2 cell maps. The caller owns the threshold cache.
  // Returns M_eff = `bkt_thresh_axis.size() + 1`.
  uint32_t PrepareBuckets(uint32_t axis, const Thresholds& bkt_thresh_axis,
                          const Thresholds& T1, const Thresholds& T2) {
    uint32_t M = static_cast<uint32_t>(image.DC_vals[axis].size());
    uint32_t M_eff = static_cast<uint32_t>(bkt_thresh_axis.size()) + 1;
    Update(axis, bkt_thresh_axis, T1, T2);
    uint32_t cur_k = 0;
    k_to_dc0[0] = 0;
    for (uint32_t i = 0; i < M; ++i) {
      uint16_t k = ax0_to_k[i];
      while (cur_k < k) k_to_dc0[++cur_k] = static_cast<uint16_t>(i);
    }
    JXL_DASSERT(cur_k + 1 == M_eff);
    JXL_DASSERT(ax0_to_k[M - 1] + 1 == M_eff);
    return M_eff;
  }
};

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_AXIS_MAPS_H_
