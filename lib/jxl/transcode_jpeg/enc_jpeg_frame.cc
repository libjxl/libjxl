// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_jpeg_frame.h"
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
//      thresholds via iterative greedy descent, then
//      `ClusterContexts` merges similar contexts using
//      entropy-cost-guided agglomerative clustering.
//   4. The candidate with the lowest total cost (entropy + histogram signalling
//      overhead) is selected and written into the BlockCtxMap consumed by the
//      rest of the encoder.
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
// `czdc` - `(c,zdc)` = `(c<<9|zdc)`, [0,3*512)
// `ai` - AC index = value of AC coefficient + `kDCTOff`

namespace {}  // namespace

Status OptimizeJPEGContextMap(const jpeg::JPEGData& jpeg_data,
                              SpeedTier speed_tier, BlockCtxMap& ctx_map,
                              ThreadPool* pool) {
  auto opt_data = std::make_shared<JPEGOptData>();
  JXL_RETURN_IF_ERROR(opt_data->BuildFromJPEG(jpeg_data, pool));
  const JPEGCtxEffortParams effort =
      JPEGCtxEffortParams::FromSpeedTier(speed_tier);

  JXL_ASSIGN_OR_RETURN(std::vector<FactorizationCandidate> candidates,
                       RankAndTrimFactorizations(opt_data, effort, pool));
  if (candidates.empty()) return true;

  JXL_DEBUG_V(2,
              "JPEG ctx effort at speed tier %i: %i candidates, rank_iters=%u "
              "final_iters=%u refine_iters=%u\n",
              static_cast<int>(speed_tier), static_cast<int>(candidates.size()),
              effort.rank_iters, effort.final_iters, effort.refine_iters);

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

        auto opt_result = ctx.OptimizeThresholds(
            candidate.init, effort.main_m_target, effort.main_iters);
        ThresholdSet opt_thr = opt_result.second;

        JXL_ASSIGN_OR_RETURN(
            Clustering cl_result,
            Clustering::Build(*opt_data, opt_thr,
                              kMaxClusters - (jpeg_data.components.size() == 1),
                              effort.overhead_aware_tail, nullptr));
        ContextMap& cluster_map = cl_result.ctx_map;

        ThresholdSet refined_thr;
        int64_t entropy_cost = 0;
        int64_t nz_cost = 0;
        if (effort.refine_iters == 0) {
          refined_thr = cl_result.PruneDeadThresholds(opt_thr);
          entropy_cost = cl_result.clustered_cost;
          nz_cost = cl_result.ComputeNZCost(*opt_data);
        } else {
          auto refine_result =
              RefineClustered(*opt_data, opt_thr, cl_result,
                              effort.refine_iters, effort.refine_radius);
          refined_thr = refine_result.thresholds;
          entropy_cost = refine_result.cost;
          nz_cost = refine_result.nz_cost;
        }
        int64_t total_cost = entropy_cost;

        // Add signalling overhead for histogram headers
        int64_t overhead = 0;
        auto overhead_or = cl_result.ComputeSignallingOverhead(*opt_data);
        if (overhead_or.ok()) {
          overhead = std::move(overhead_or).value_();
          total_cost += overhead;
        }

        std::lock_guard<std::mutex> lock(mu);
        // JXL_DEBUG_V(
        //     2,
        printf(
            "(%u,%u,%u) cost: unclustered=%.2f clustered=%.2f "
            "refined=%.2f nz=%.2f overhead=%.2f total=%.2f\n",
            candidate.a, candidate.b, candidate.c, bit_cost(opt_result.first),
            bit_cost(cl_result.clustered_cost), bit_cost(entropy_cost),
            bit_cost(nz_cost), bit_cost(overhead), bit_cost(total_cost));
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

  for (int16_t t : best_thr.TY()) ctx_map.dc_thresholds[1].push_back(t - 1);
  for (int16_t t : best_thr.TCb()) ctx_map.dc_thresholds[0].push_back(t - 1);
  for (int16_t t : best_thr.TCr()) ctx_map.dc_thresholds[2].push_back(t - 1);

  // `dc_thresholds` is indexed in JXL XYB order (X=0, Y=1, B=2), which maps
  // JPEG components as Cb→[0], Y→[1], Cr→[2]. `best_ctx` and `ctx_map` remain
  // in JPEG component order (Y=0, Cb=1, Cr=2) as produced by the optimizer.
  ctx_map.ctx_map.assign(3 * kNumOrders * num_dc_ctxs, 0);
  for (size_t c = 0; c < opt_data->channels; ++c) {
    for (size_t cell = 0; cell < num_dc_ctxs; ++cell) {
      ctx_map.ctx_map[c * kNumOrders * num_dc_ctxs + cell] =
          best_ctx[c * num_dc_ctxs + cell] + (opt_data->channels == 1);
    }
  }
  size_t num_ctxs =
      *std::max_element(ctx_map.ctx_map.begin(), ctx_map.ctx_map.end()) + 1;
  JXL_ENSURE(num_ctxs <= kMaxClusters);
  ctx_map.num_ctxs = num_ctxs;

  return true;
}

}  // namespace jxl
