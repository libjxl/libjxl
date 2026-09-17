// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_H_
#define LIB_JXL_ENC_LZ77_H_

#include <cstddef>
#include <vector>

#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

constexpr uint32_t kLZ77SkipMarker = static_cast<uint32_t>(~0u);

struct TrialLZ77MatchResult {
  bool has_matches = false;
  size_t total_pixels = 0;
  size_t matched_pixels = 0;
  size_t num_matches = 0;
  float bit_decrease = 0.0f;
  Predictor predictor = Predictor::Zero;
  // Per pixel in raster/stream order:
  // 0 = literal
  // kLZ77SkipMarker = inside match (skip)
  // > 0 and != kLZ77SkipMarker: match start of length match_info[i]
  std::vector<uint32_t> match_info;
};

// Runs a fast LZ77 trial pass on the image channels using the specified predictor.
TrialLZ77MatchResult RunFastTrialLZ77(const Image& image, Predictor pred,
                                     size_t min_len = 3,
                                     size_t max_chan_size = 0xFFFFFF);

// Returns a vector of token streams with the LZ77 compression applied
// in accordance with parameters sent. If compression is not beneficial,
// returns an empty vector.
std::vector<std::vector<Token>> ApplyLZ77(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77);

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_H_
