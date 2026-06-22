// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Local threshold refinement for JPEG lossless recompression.
//
// After threshold optimization and context clustering, the transcoder runs a
// final local search that nudges thresholds by a limited radius while updating
// clustered histogram costs incrementally. This file exposes the small public
// surface of that pass.
//
// `RefineResult`
//   Final threshold set plus the tracked entropy costs returned to the outer
//   search loop.
//
// `RefineClustered`
//   Runs the coordinate-descent refinement pass on a clustered solution.
//
// The temporary scratch and per-axis search machinery live privately in
// `enc_jpeg_refine.cc`.

#ifndef LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_REFINE_H_
#define LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_REFINE_H_

#include <cstddef>

#include "lib/jxl/transcode_jpeg/enc_jpeg_cluster.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"

namespace jxl {

// Return value of `RefineClustered`: the threshold set and entropy costs
// after coordinate-descent refinement. `cost` is the combined AC + nz entropy
// cost used by the outer optimizer to compare configurations. `nz_cost` (the
// nonzero-count entropy portion) is broken out separately for logging.
struct RefineResult {
  ThresholdSet thresholds;
  FixedPointCost cost;
  FixedPointCost nz_cost;
};

// Performs the local refinement pass on an already clustered threshold set.
// It jitters thresholds axis by axis, accepting moves that reduce the tracked
// entropy cost (including AC and nz costs).
RefineResult RefineClustered(const JPEGOptData& d,
                             const ThresholdSet& thresholds,
                             Clustering& clustering, uint32_t max_iters = 5,
                             ptrdiff_t search_radius = 2048);

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_REFINE_H_
