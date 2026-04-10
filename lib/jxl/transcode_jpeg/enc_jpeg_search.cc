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

#include <array>
#include <unordered_map>

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

  // Compute only the initial thresholds that are actually used by the
  // factorization list.
  std::array<std::unordered_map<uint32_t, Thresholds>, kNumCh>
      init_thresh_cache;
  for (const auto& f : factorizations) {
    for (uint32_t axis = 0; axis < kNumCh; ++axis) {
      auto insert_result =
              init_thresh_cache[axis].emplace(f[axis], Thresholds());
      if (insert_result.second) { // Only compute if not already computed.
        insert_result.first->second =
            InitThresh(*opt_data, axis, f[axis]);
      }
    }
  }

  std::vector<FactorizationCandidate> candidates;
  candidates.reserve(factorizations.size());
  for (const auto& factorization : factorizations) {
    FactorizationCandidate candidate;
    candidate.a = factorization[0];
    candidate.b = factorization[1];
    candidate.c = factorization[2];
    candidate.init.T[0] = init_thresh_cache[0].at(candidate.a);
    candidate.init.T[1] = init_thresh_cache[1].at(candidate.b);
    candidate.init.T[2] = init_thresh_cache[2].at(candidate.c);
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
