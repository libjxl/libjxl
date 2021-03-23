// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LIB_JXL_ANS_PARAMS_H_
#define LIB_JXL_ANS_PARAMS_H_

// Common parameters that are needed for both the ANS entropy encoding and
// decoding methods.

#include <stdint.h>
#include <stdlib.h>

#include "lib/jxl/enc_params.h"

namespace jxl {

// TODO(veluca): decide if 12 is the best constant here (valid range is up to
// 16). This requires recomputing the Huffman tables in {enc,dec}_ans.cc
// 14 gives a 0.2% improvement at d1 and makes d8 slightly worse. This is
// likely not worth the increase in encoder complexity.
#define ANS_LOG_TAB_SIZE 12u
#define ANS_TAB_SIZE (1 << ANS_LOG_TAB_SIZE)
#define ANS_TAB_MASK (ANS_TAB_SIZE - 1)

// Largest possible symbol to be encoded by either ANS or prefix coding.
#define PREFIX_MAX_ALPHABET_SIZE 4096
#define ANS_MAX_ALPHABET_SIZE 256

// Max number of bits for prefix coding.
#define PREFIX_MAX_BITS 15

#define ANS_SIGNATURE 0x13  // Initial state, used as CRC.

struct HistogramParams {
  enum class ClusteringType {
    kFastest,  // Only 4 clusters.
    kFast,
    kBest,
  };

  enum class HybridUintMethod {
    kNone,        // just use kHybridUint420Config.
    kFast,        // just try a couple of options.
    kContextMap,  // fast choice for ctx map.
    kBest,
  };

  enum class LZ77Method {
    kNone,     // do not try lz77.
    kRLE,      // only try doing RLE.
    kLZ77,     // try lz77 with backward references.
    kOptimal,  // optimal-matching LZ77 parsing.
  };

  enum class ANSHistogramStrategy {
    kFast,         // Only try some methods, early exit.
    kApproximate,  // Only try some methods.
    kPrecise,      // Try all methods.
  };

  HistogramParams() = default;

  HistogramParams(SpeedTier tier, size_t num_ctx) {
    if (tier == SpeedTier::kFalcon) {
      clustering = ClusteringType::kFastest;
      lz77_method = LZ77Method::kNone;
    } else if (tier > SpeedTier::kTortoise) {
      clustering = ClusteringType::kFast;
    } else {
      clustering = ClusteringType::kBest;
    }
    if (tier > SpeedTier::kTortoise) {
      uint_method = HybridUintMethod::kNone;
    }
    if (tier >= SpeedTier::kSquirrel) {
      ans_histogram_strategy = ANSHistogramStrategy::kApproximate;
    }
  }

  ClusteringType clustering = ClusteringType::kBest;
  HybridUintMethod uint_method = HybridUintMethod::kBest;
  LZ77Method lz77_method = LZ77Method::kRLE;
  ANSHistogramStrategy ans_histogram_strategy = ANSHistogramStrategy::kPrecise;
  std::vector<size_t> image_widths;
  size_t max_histograms = ~0;
  bool force_huffman = false;
};

}  // namespace jxl

#endif  // LIB_JXL_ANS_PARAMS_H_
