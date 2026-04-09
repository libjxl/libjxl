// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Search configuration and candidate ranking for JPEG lossless recompression.
//
// After `JPEGOptData` has been built, the outer optimizer enumerates candidate
// DC-threshold factorizations, optionally ranks them with a cheaper threshold
// optimization pass, and keeps only the most promising ones for the full
// threshold / clustering / refinement pipeline. This file exposes the public
// types used by that search step.
//
// `JPEGCtxEffortParams`
//   Speed-tier-dependent knobs controlling how many factorizations are kept
//   and how much work is spent on ranking, final optimization, and refinement.
//
// `FactorizationCandidate`
//   One candidate `(a, b, c)` factorization together with its initialized
//   threshold set and optional ranking cost.
//
// `RankAndTrimFactorizations`
//   Builds the maximal-factorization candidate list and, when enabled, ranks
//   and trims it to the best `keep_top_k` entries.

#ifndef LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_SEARCH_H_
#define LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_SEARCH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>

#include "lib/jxl/common.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"

namespace jxl {

// Effort-level knobs derived from the encoder speed tier (number of candidates,
// refinement iterations, etc.).
struct JPEGCtxEffortParams {
  uint32_t keep_top_k;
  uint32_t rank_m_target;
  uint32_t rank_iters;
  uint32_t main_m_target;
  uint32_t main_iters;
  bool overhead_aware_tail;
  uint32_t refine_iters;
  ptrdiff_t refine_radius;

  static JPEGCtxEffortParams FromSpeedTier(SpeedTier speed_tier) {
    switch (speed_tier) {
      case SpeedTier::kSquirrel:
        return {/*keep_top_k=*/8,
                /*rank_m_target=*/32,
                /*rank_iters=*/1,
                /*main_m_target=*/64,
                /*main_iters=*/2,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/0,
                /*refine_radius=*/0};
      case SpeedTier::kKitten:
        return {/*keep_top_k=*/16,
                /*rank_m_target=*/64,
                /*rank_iters=*/1,
                /*main_m_target=*/128,
                /*main_iters=*/4,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/1,
                /*refine_radius=*/4};
      case SpeedTier::kTortoise:
        return {/*keep_top_k=*/24,
                /*rank_m_target=*/128,
                /*rank_iters=*/2,
                /*main_m_target=*/256,
                /*main_iters=*/8,
                /*overhead_aware_tail=*/true,
                /*refine_iters=*/2,
                /*refine_radius=*/8};
      case SpeedTier::kTectonicPlate:
      case SpeedTier::kGlacier:
      default:
        return {/*keep_top_k=*/0,
                /*rank_m_target=*/0,
                /*rank_iters=*/0,
                /*main_m_target=*/kMTarget,
                /*main_iters=*/20,
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
    const JPEGCtxEffortParams& effort, ThreadPool* pool);

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_SEARCH_H_
