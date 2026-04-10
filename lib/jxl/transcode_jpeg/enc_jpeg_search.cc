// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of factorization search for JPEG lossless recompression.
// See `enc_jpeg_search.h` for the public interface and search overview.
//
// This file currently contains one public entry point:
//
//   `RankAndTrimFactorizations`
//     Builds the maximal-factorization candidate list, initializes thresholds
//     for each candidate, and optionally ranks them in parallel before keeping
//     only the best entries for the full optimization pipeline.

#include "lib/jxl/transcode_jpeg/enc_jpeg_search.h"

#include "lib/jxl/transcode_jpeg/enc_jpeg_threshold.h"

namespace jxl {

StatusOr<std::vector<FactorizationCandidate>> RankAndTrimFactorizations(
    std::shared_ptr<const JPEGOptData> opt_data,
    const JPEGCtxEffortParams& effort, ThreadPool* pool) {
  const auto factorizations = MaximalFactorizations(*opt_data);
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
    candidate.init.T[0] = InitThresh(*opt_data, 0, candidate.a);
    candidate.init.T[1] = InitThresh(*opt_data, 1, candidate.b);
    candidate.init.T[2] = InitThresh(*opt_data, 2, candidate.c);
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
        if (effort.rank_iters == 0) {
          candidate.rank_cost = ctx.TotalCost(candidate.init);
        } else {
          candidate.rank_cost =
              ctx.TotalCost(ctx.OptimizeThresholds(
                  candidate.init, effort.rank_m_target, effort.rank_iters));
        }
        return true;
      },
      "JpegCtxRank"));

  std::stable_sort(candidates.begin(), candidates.end());
  candidates.resize(effort.keep_top_k);
  return candidates;
}

}  // namespace jxl
