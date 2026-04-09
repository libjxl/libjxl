// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// State-side operations on clustered JPEG contexts.
// See `enc_jpeg_cluster.h` for the public interface and role of `Clustering`.
//
// This file implements methods that operate on an already-built clustering:
//
//   `ComputeSignallingOverhead`
//     Estimates histogram header cost for the current clustered state.
//
//   `ComputeNZCost`
//     Computes the nonzero-count entropy portion of the clustered cost.
//
//   `BuildLocalClusterBoundaries`
//     Builds threshold-major `{lo, hi}` boundary views for refinement.
//
//   `PruneDeadThresholds`
//     Removes structurally inert thresholds and rebuilds `ctx_map`.

#include "lib/jxl/transcode_jpeg/enc_jpeg_cluster.h"

#include <algorithm>
#include <unordered_map>

#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans_params.h"

namespace jxl {

StatusOr<int64_t> Clustering::ComputeSignallingOverhead(const JPEGOptData& d,
                                                        int64_t cutoff) const {
  int64_t overhead = 0;

  // Default `HybridUintConfig` for AC coefficients: (split_exponent=4,
  // msb_in_token=2, lsb_in_token=0)
  const HybridUintConfig hybrid_uint_config(4, 2, 0);

  // Process `hist_h`: split by `zdc` and compute overhead per histogram.
  for (const auto& cluster : hist_h) {
    if (cluster.empty()) continue;

    // Group symbols by `zdc` value, applying `HybridUintConfig`.
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> by_zdc;
    for (uint32_t id : cluster.used_ids) {
      uint32_t symbol = d.dense_to_zdcai[id];
      uint32_t zdc = symbol >> 11;
      uint32_t ai = symbol & 0x7FFu;
      uint32_t token, nbits, bits;
      hybrid_uint_config.Encode(ai, &token, &nbits, &bits);
      by_zdc[zdc][token] += cluster.Get(id);
    }

    // Compute overhead for each `zdc` histogram.
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

      // `ANSPopulationCost()` includes header + data cost.
      JXL_ASSIGN_OR_RETURN(float ans_cost, h.ANSPopulationCost());
      // Shannon entropy is the ideal data cost.
      float shannon = h.ShannonEntropy();
      // Overhead = total cost - data cost.
      float header_cost = ans_cost - shannon;
      if (header_cost > 0) {
        overhead += static_cast<int64_t>(header_cost * kFScale);
        if (overhead >= cutoff) return overhead;
      }
    }
  }

  // Process `hist_nz_h`: split by predicted bucket `pb`.
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
        h.counts[nz_count] = static_cast<ANSHistBin>(cluster[base + nz_count]);
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

int64_t Clustering::ComputeNZCost(const JPEGOptData& d) const {
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

std::array<std::vector<ClusterBoundary>, kNumCh>
Clustering::BuildLocalClusterBoundaries(const ThresholdSet& thresholds,
                                        uint32_t channels) const {
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

    for (uint32_t c = 0; c < channels; ++c) {
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

ThresholdSet Clustering::PruneDeadThresholds(const ThresholdSet& thresholds) {
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

}  // namespace jxl
