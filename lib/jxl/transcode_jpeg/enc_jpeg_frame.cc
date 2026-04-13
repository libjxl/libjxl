// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_jpeg_frame.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_cluster.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_refine.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_search.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_threshold.h"

namespace jxl {

// This file implements context-map optimization for JPEG-source images being
// re-encoded into JPEG XL. The single public entry point is
// `OptimizeJPEGContextMap()`, which optimizes per-channel DC thresholds
// and AC context clustering to minimise the entropy-coded size
// of the JPEG AC DCT coefficients.
//
// The pipeline is:
//   1. `JPEGOptData::BuildFromJPEG` extracts per-block DC/AC statistics from
//      the source JPEG and builds a bin-indexed AC stream.
//   2. Enumerate "maximal factorisations" of the (Y, Cb, Cr) DC-threshold
//      space; each factorisation defines how many DC intervals each channel
//      is split into.
//   3. For every candidate factorisation (optionally pre-filtered by a cheap
//      ranking pass), `PartitioningCtx::OptimizeThresholds` refines the DC
//      thresholds via iterative greedy descent, `Clustering::Build` merges
//      similar contexts using entropy-cost-guided agglomerative clustering,
//      and `RefineClustered` performs a final local threshold search on the
//      clustered solution.
//   4. The candidate with the lowest total cost (entropy + histogram signalling
//      overhead) is selected and written into the `BlockCtxMap` consumed by
//      the rest of the encoder.
//
// All calculations (besides formation of `f(n) = n*log2(n)` lookup) are done in
// integer arithmetic to avoid floating point inaccuracies. Counters are
// generally `uint32_t`, that is enough for JPEG.
//
// General notation:
// `N` - number of 8x8 blocks of an image component
// `M` - number of distinct DC values along a component axis
// `M_eff` - number of distinct DC buckets along a component axis,
//           use of full `M` for large values is too slow by cache misses
// `ci` - cell index in 2D array of cells perpendicular to current axis,
//        `ci < 32` as axis search is meaningful with at least 2 cells per axis
// `K` - number of intervals (cells) along an axis
// `l` - left bound of interval (inclusive)
// `n` - right bound of interval (inclusive)
// `c` - component (axis) index, [0,3)
// `zdc` - `jxl::ZeroDensityContext` of an AC coefficient in a block
// `czdc` - `(c,zdc)` = `channel * kZeroDensityContextCount + zdc`, [0,3*458)
// `ai` - AC index = value of AC coefficient + `kDCTOff`

Status OptimizeJPEGContextMap(const jpeg::JPEGData& jpeg_data,
                              SpeedTier speed_tier,
                              const JpegCflContext& cfl_ctx,
                              BlockCtxMap& ctx_map, ThreadPool* pool) {
  const JPEGCtxEffortParams effort =
      JPEGCtxEffortParams::FromSpeedTier(speed_tier);
  auto opt_data = std::make_shared<JPEGOptData>();
  JXL_RETURN_IF_ERROR(
      opt_data->BuildFromJPEG(jpeg_data, effort.ac_hist_model, cfl_ctx, pool));

  JXL_ASSIGN_OR_RETURN(std::vector<FactorizationCandidate> candidates,
                       RankAndTrimFactorizations(opt_data, effort, pool));
  if (candidates.empty()) return true;

  JXL_DEBUG_V(2,
              "JPEG ctx effort at speed tier %i: %i candidates, rank_iters=%u "
              "main_iters=%u refine_iters=%u\n",
              static_cast<int>(speed_tier), static_cast<int>(candidates.size()),
              effort.rank_iters, effort.main_iters, effort.refine_iters);

  int64_t best_cost = std::numeric_limits<int64_t>::max();
  ThresholdSet best_thr;
  ContextMap best_ctx;
  std::mutex mu;

  std::vector<PartitioningCtx> ctx_pool;
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, static_cast<uint32_t>(candidates.size()),
      [&](size_t num_threads) -> Status {
        ctx_pool.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
          ctx_pool.emplace_back(opt_data);
        }
        return true;
      },
      [&](uint32_t idx, size_t thread_id) -> Status {
        const FactorizationCandidate& candidate = candidates[idx];
        PartitioningCtx& ctx = ctx_pool[thread_id];

        int64_t opt_cost = 0;
        ThresholdSet opt_thr = ctx.OptimizeThresholds(
            candidate.init, effort.main_m_target, effort.main_iters, &opt_cost);

        JXL_ASSIGN_OR_RETURN(
            Clustering cl_result,
            Clustering::Build(*opt_data, opt_thr,
                              kMaxClusters - (opt_data->channels == 1),
                              effort.overhead_aware_tail, nullptr));
        ContextMap& cluster_map = cl_result.ctx_map;

        auto refine_result =
            RefineClustered(*opt_data, opt_thr, cl_result, effort.refine_iters,
                            effort.refine_radius);
        ThresholdSet refined_thr = refine_result.thresholds;
        int64_t entropy_cost = refine_result.cost;
        int64_t nz_cost = refine_result.nz_cost;
        (void)nz_cost;
        int64_t total_cost = entropy_cost;

        // Add signalling overhead for histogram headers
        JXL_ASSIGN_OR_RETURN(int64_t overhead,
                             cl_result.ComputeSignallingOverhead(*opt_data));
        total_cost += overhead;

        std::lock_guard<std::mutex> lock(mu);
        JXL_DEBUG_V(2,
                    "(%u,%u,%u) cost: unclustered=%.2f clustered=%.2f "
                    "refined=%.2f nz=%.2f overhead=%.2f total=%.2f\n",
                    candidate.a, candidate.b, candidate.c, bit_cost(opt_cost),
                    bit_cost(cl_result.clustered_cost), bit_cost(entropy_cost),
                    bit_cost(nz_cost), bit_cost(overhead),
                    bit_cost(total_cost));
        if (total_cost < best_cost) {
          best_cost = total_cost;
          best_thr = refined_thr;
          best_ctx = cluster_map;
        }
        return true;
      },
      "JpegCtxOpt"));

  size_t na_Y = best_thr.TY().size() + 1;
  size_t na_Cb = best_thr.TCb().size() + 1;
  size_t na_Cr = best_thr.TCr().size() + 1;
  size_t num_dc_ctxs = na_Y * na_Cb * na_Cr;

  JXL_ENSURE(num_dc_ctxs <= kMaxCells);
  JXL_ENSURE(na_Y <= kMaxIntervals && na_Cb <= kMaxIntervals &&
             na_Cr <= kMaxIntervals);

  ctx_map.num_dc_ctxs = num_dc_ctxs;

  ctx_map.dc_thresholds[1].clear();
  ctx_map.dc_thresholds[0].clear();
  ctx_map.dc_thresholds[2].clear();

  // If the image is effectively grayscale, we use only one channel.
  uint32_t effective_channels = opt_data->channels;
  ctx_map.ctx_map.assign(3 * kNumOrders * num_dc_ctxs, 0);
  if (effective_channels == 1) {
    JXL_DASSERT(best_thr.TCb().empty());
    JXL_DASSERT(best_thr.TCr().empty());
    const size_t active_plane = opt_data->jpeg_to_plane[0];
    for (int16_t t : best_thr.TY()) {
      ctx_map.dc_thresholds[active_plane].push_back(t - 1);
    }
    const size_t slot = active_plane < 2 ? active_plane ^ 1 : 2;
    for (size_t cell = 0; cell < num_dc_ctxs; ++cell) {
      ctx_map.ctx_map[slot * kNumOrders * num_dc_ctxs + cell] =
          best_ctx[cell] + 1;
    }
  } else {
    // `dc_thresholds` is indexed by JXL plane (0, 1, 2). `ctx_map` uses the
    // same planes but with the historical 0/1 swap applied by
    // `BlockCtxMap::Context`.
    for (size_t plane = 0; plane < 3; ++plane) {
      const uint32_t jpeg_c =
          static_cast<uint32_t>(cfl_ctx.plane_to_jpeg[plane]);
      for (int16_t t : best_thr.T[jpeg_c]) {
        ctx_map.dc_thresholds[plane].push_back(t - 1);
      }
      const size_t slot = plane < 2 ? plane ^ 1 : 2;
      for (size_t cell = 0; cell < num_dc_ctxs; ++cell) {
        ctx_map.ctx_map[slot * kNumOrders * num_dc_ctxs + cell] =
            best_ctx[jpeg_c * num_dc_ctxs + cell];
      }
    }
  }
  size_t num_ctxs =
      *std::max_element(ctx_map.ctx_map.begin(), ctx_map.ctx_map.end()) + 1;
  JXL_ENSURE(num_ctxs <= kMaxClusters);
  ctx_map.num_ctxs = num_ctxs;

  return true;
}

}  // namespace jxl
