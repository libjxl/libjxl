// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_OPTIONS_H_
#define LIB_JXL_MODULAR_OPTIONS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "lib/jxl/enc_ans_params.h"

namespace jxl {

using PropertyVal = int32_t;
using Properties = std::vector<PropertyVal>;

enum class Predictor : uint32_t {
  Zero = 0,
  Left = 1,
  Top = 2,
  Average0 = 3,
  Select = 4,
  Gradient = 5,
  Weighted = 6,
  TopRight = 7,
  TopLeft = 8,
  LeftLeft = 9,
  Average1 = 10,
  Average2 = 11,
  Average3 = 12,
  Average4 = 13,
  // The following predictors are encoder-only.
  Best = 14,  // Best of Gradient and Weighted
  Variable =
      15,  // Find the best decision tree for predictors/predictor per row
};

constexpr Predictor kUndefinedPredictor = static_cast<Predictor>(~0u);

constexpr size_t kNumModularPredictors =
    static_cast<size_t>(Predictor::Average4) + 1;
constexpr size_t kNumModularEncoderPredictors =
    static_cast<size_t>(Predictor::Variable) + 1;

static constexpr ptrdiff_t kNumStaticProperties = 2;  // channel, group_id.

using StaticPropRange =
    std::array<std::array<uint32_t, 2>, kNumStaticProperties>;

struct ModularMultiplierInfo {
  StaticPropRange range;
  uint32_t multiplier;
};

// A sequence of residuals: `prefix`, then `period` repeated forever.
struct ResidualPattern {
  std::vector<int32_t> prefix;
  std::vector<int32_t> period;
  // The pattern starts at the first sample of the stream, meta channels
  // (palette entries) included, instead of after them (jxl_from_tree's
  // MetaResiduals).
  bool include_meta = false;
};

struct ModularOptions {
  /// Used in both encode and decode:

  // Stop encoding/decoding when reaching a (non-meta) channel that has a
  // dimension bigger than max_chan_size.
  size_t max_chan_size = 0xFFFFFF;

  // Used during decoding for validation of transforms (sqeeezing) scheme.
  size_t group_dim = 0x1FFFFFFF;

  /// Encode options:
  // Fraction of pixels to look at to learn a MA tree
  // Number of iterations to do to learn a MA tree
  // (if zero there is no MA context model)
  float nb_repeats = .5f;

  // Maximum number of (previous channel) properties to use in the MA trees
  int max_properties = 0;  // no previous channels

  // Alternative heuristic tweaks.
  // Properties default to channel, group, weighted, gradient residual, W-NW,
  // NW-N, N-NE, N-NN
  std::vector<uint32_t> splitting_heuristics_properties;
  float splitting_heuristics_node_threshold = 96;
  size_t max_property_values = 32;

  // Predictor to use for each channel.
  Predictor predictor = kUndefinedPredictor;

  int wp_mode = 0;
  // A custom weighted predictor header (jxl_from_tree's WPParams): p1C, p2C,
  // p3Ca..p3Ce (0..31) and w0..w3 (0..15). Used instead of wp_mode's if set.
  bool has_wp_params = false;
  std::array<uint32_t, 11> wp_params = {};

  float fast_decode_multiplier = 1.01f;

  // Forces the encoder to produce a tree that is compatible with the WP-only
  // decode path (or with the no-wp path, or the gradient-only path).
  enum class TreeMode { kGradientOnly, kWPOnly, kNoWP, kDefault };
  TreeMode wp_tree_mode = TreeMode::kDefault;

  // Skip fast paths in the encoder.
  bool skip_encoder_fast_path = false;

  // Kind of tree to use.
  // TODO(veluca): add tree kinds for JPEG recompression with CfL enabled,
  // general AC metadata, different DC qualities, and others.
  enum class TreeKind {
    kTrivialTreeNoPredictor,
    kLearn,
    kJpegTranscodeACMeta,
    kFalconACMeta,
    kACMeta,
    kWPFixedDC,
    kGradientFixedDC,
  };
  TreeKind tree_kind = TreeKind::kLearn;

  HistogramParams histogram_params;

  // Ignore the image and just pretend all tokens are zeroes
  bool zero_tokens = false;
  // With zero_tokens: the meta channels (e.g. palette entries) are still coded
  // from the image (real residuals); only the other channels are all zeroes.
  bool code_meta_channels = false;
  // With zero_tokens: if not null, the residuals of the streams are these
  // patterns (by stream ID, -1 for the streams not listed; all zero for the
  // others), coded without materializing them: the prefix and one period as
  // symbols (in the contexts the tree gives), the rest as one LZ77 copy.
  // Every stream is then coded this way, and the tokens come with LZ77
  // already applied (HistogramParams::tokens_have_lz77).
  const std::map<int, ResidualPattern>* residual_patterns = nullptr;
  // With residual_patterns: LZ77 also inside the prefix and the first period
  // (where it pays off, by an estimate).
  bool residual_inner_lz77 = false;
  // Literal costs per context in the inner LZ77 match estimate.
  bool residual_lz77_context_costs = false;

  ModularOptions() {
    // GCC has complaints about inline vector initialization; do it manually.
    static const std::vector<uint32_t> kDefaultSplittingHeuristicsProperties = {
        0, 1, 15, 9, 10, 11, 12, 13};
    splitting_heuristics_properties = kDefaultSplittingHeuristicsProperties;
  }
};

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_OPTIONS_H_
