// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_JPEG_SEARCH_H_
#define LIB_JXL_ENC_JPEG_SEARCH_H_

#include <cstddef>
#include <cstdint>

#include "lib/jxl/common.h"
#include "lib/jxl/enc_jpeg_opt_data.h"
#include "lib/jxl/enc_jpeg_threshold.h"

namespace jxl {

// Effort-level knobs derived from the encoder speed tier (number of candidates,
// refinement iterations, etc.).
struct JPEGCtxEffortParams {
  uint32_t keep_top_k;
  uint32_t rank_m_target;
  uint32_t rank_iters;
  uint32_t final_m_target;
  uint32_t final_iters;
  bool overhead_aware_tail;
  uint32_t refine_iters;
  ptrdiff_t refine_radius;

  static JPEGCtxEffortParams FromSpeedTier(SpeedTier speed_tier) {
    switch (speed_tier) {
      case SpeedTier::kSquirrel:
        return {/*keep_top_k=*/4,
                /*rank_m_target=*/0,
                /*rank_iters=*/0,
                /*final_m_target=*/64,
                /*final_iters=*/2,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/0,
                /*refine_radius=*/0};
      case SpeedTier::kKitten:
        return {/*keep_top_k=*/6,
                /*rank_m_target=*/64,
                /*rank_iters=*/1,
                /*final_m_target=*/128,
                /*final_iters=*/4,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/1,
                /*refine_radius=*/4};
      case SpeedTier::kTortoise:
        return {/*keep_top_k=*/12,
                /*rank_m_target=*/128,
                /*rank_iters=*/2,
                /*final_m_target=*/256,
                /*final_iters=*/8,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/2,
                /*refine_radius=*/8};
      case SpeedTier::kTectonicPlate:
      case SpeedTier::kGlacier:
      default:
        return {/*keep_top_k=*/0,
                /*rank_m_target=*/0,
                /*rank_iters=*/0,
                /*final_m_target=*/kDCTRange,
                /*final_iters=*/20,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/5,
                /*refine_radius=*/16};
    }
  }
};

struct FactorizationCandidate {
  uint32_t a;
  uint32_t b;
  uint32_t c;
  ThresholdSet init;
  int64_t rank_cost = std::numeric_limits<int64_t>::max();

  bool operator<(const FactorizationCandidate& rhs) const {
    if (rank_cost != rhs.rank_cost) return rank_cost < rhs.rank_cost;
    const uint32_t lhs_cells = a * b * c;
    const uint32_t rhs_cells = rhs.a * rhs.b * rhs.c;
    if (lhs_cells != rhs_cells) return lhs_cells > rhs_cells;
    if (a != rhs.a) return a > rhs.a;
    return std::tie(b, c) < std::tie(rhs.b, rhs.c);
  }
};

StatusOr<std::vector<FactorizationCandidate>> RankAndTrimFactorizations(
    std::shared_ptr<const JPEGOptData> opt_data,
    const JPEGCtxEffortParams& effort, ThreadPool* pool) {
  const auto factorizations = opt_data->MaximalFactorizations();
  if (!factorizations.empty()) {
    JXL_DEBUG_V(2, "Searching %i maximal factorizations\n",
                static_cast<int>(factorizations.size()));
  }

  std::vector<FactorizationCandidate> candidates;
  candidates.reserve(factorizations.size());
  for (const auto& factorization : factorizations) {
    FactorizationCandidate candidate;
    candidate.a = std::get<0>(factorization);
    candidate.b = std::get<1>(factorization);
    candidate.c = std::get<2>(factorization);
    candidate.init.T[0] = opt_data->InitThresh(0, candidate.a);
    candidate.init.T[1] = opt_data->InitThresh(1, candidate.b);
    candidate.init.T[2] = opt_data->InitThresh(2, candidate.c);
    candidates.push_back(std::move(candidate));
  }
  if (effort.keep_top_k == 0 || candidates.size() <= effort.keep_top_k) {
    return candidates;
  }

  std::vector<PartitioningCtx> rank_ctx_pool;
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, static_cast<uint32_t>(candidates.size()),
      [&](size_t num_threads) -> Status {
        rank_ctx_pool.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
          rank_ctx_pool.emplace_back(opt_data);
        }
        return true;
      },
      [&](uint32_t idx, size_t thread_id) -> Status {
        FactorizationCandidate& candidate = candidates[idx];
        PartitioningCtx& ctx = rank_ctx_pool[thread_id];
        candidate.rank_cost =
            (effort.rank_iters == 0)
                ? ctx.TotalCost(candidate.init)
                : ctx.OptimizeThresholds(candidate.init, effort.rank_m_target,
                                         effort.rank_iters)
                      .first;
        return true;
      },
      "JpegCtxRank"));

  std::stable_sort(candidates.begin(), candidates.end());
  candidates.resize(effort.keep_top_k);
  return candidates;
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_JPEG_SEARCH_H_
