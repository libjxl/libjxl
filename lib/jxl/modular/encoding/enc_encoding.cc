// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

namespace {
// Plot tree (if enabled) and predictor usage map.
constexpr bool kWantDebug = true;
// constexpr bool kPrintTree = false;

inline std::array<uint8_t, 3> PredictorColor(Predictor p) {
  switch (p) {
    case Predictor::Zero:
      return {{0, 0, 0}};
    case Predictor::Left:
      return {{255, 0, 0}};
    case Predictor::Top:
      return {{0, 255, 0}};
    case Predictor::Average0:
      return {{0, 0, 255}};
    case Predictor::Average4:
      return {{192, 128, 128}};
    case Predictor::Select:
      return {{255, 255, 0}};
    case Predictor::Gradient:
      return {{255, 0, 255}};
    case Predictor::Weighted:
      return {{0, 255, 255}};
      // TODO(jon)
    default:
      return {{255, 255, 255}};
  };
}

// `cutoffs` must be sorted.
Tree MakeFixedTree(int property, const std::vector<int32_t> &cutoffs,
                   Predictor pred, size_t num_pixels, int bitdepth) {
  size_t log_px = CeilLog2Nonzero(num_pixels);
  size_t min_gap = 0;
  // Reduce fixed tree height when encoding small images.
  if (log_px < 14) {
    min_gap = 8 * (14 - log_px);
  }
  const int shift = bitdepth > 11 ? std::min(4, bitdepth - 11) : 0;
  const int mul = 1 << shift;
  Tree tree;
  struct NodeInfo {
    size_t begin, end, pos;
  };
  std::queue<NodeInfo> q;
  // Leaf IDs will be set by roundtrip decoding the tree.
  tree.push_back(PropertyDecisionNode::Leaf(pred));
  q.push(NodeInfo{0, cutoffs.size(), 0});
  while (!q.empty()) {
    NodeInfo info = q.front();
    q.pop();
    if (info.begin + min_gap >= info.end) continue;
    uint32_t split = (info.begin + info.end) / 2;
    int32_t cutoff = cutoffs[split] * mul;
    tree[info.pos] = PropertyDecisionNode::Split(property, cutoff, tree.size());
    q.push(NodeInfo{split + 1, info.end, tree.size()});
    tree.push_back(PropertyDecisionNode::Leaf(pred));
    q.push(NodeInfo{info.begin, split, tree.size()});
    tree.push_back(PropertyDecisionNode::Leaf(pred));
  }
  return tree;
}

Status GatherTreeData(const Image &image, pixel_type chan, size_t group_id,
                      const weighted::Header &wp_header,
                      const ModularOptions &options, TreeSamples &tree_samples,
                      size_t *total_pixels) {
  const Channel &channel = image.channel[chan];
  JxlMemoryManager *memory_manager = channel.memory_manager();

  JXL_DEBUG_V(7, "Learning %" PRIuS "x%" PRIuS " channel %d", channel.w,
              channel.h, chan);

  std::array<pixel_type, kNumStaticProperties> static_props = {
      {chan, static_cast<int>(group_id)}};
  Properties properties(kNumNonrefProperties +
                        kExtraPropsPerChannel * options.max_properties);
  double pixel_fraction = std::min(1.0f, options.nb_repeats);
  // a fraction of 0 is used to disable learning entirely.
  if (pixel_fraction > 0) {
    pixel_fraction = std::max(pixel_fraction,
                              std::min(1.0, 1024.0 / (channel.w * channel.h)));
  }
  uint64_t threshold =
      (std::numeric_limits<uint64_t>::max() >> 32) * pixel_fraction;
  uint64_t s[2] = {static_cast<uint64_t>(0x94D049BB133111EBull),
                   static_cast<uint64_t>(0xBF58476D1CE4E5B9ull)};
  // Xorshift128+ adapted from xorshift128+-inl.h
  auto use_sample = [&]() {
    auto s1 = s[0];
    const auto s0 = s[1];
    const auto bits = s1 + s0;  // b, c
    s[0] = s0;
    s1 ^= s1 << 23;
    s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
    s[1] = s1;
    return (bits >> 32) <= threshold;
  };

  const ptrdiff_t onerow = channel.plane.PixelsPerRow();
  JXL_ASSIGN_OR_RETURN(
      Channel references,
      Channel::Create(memory_manager, properties.size() - kNumNonrefProperties,
                      channel.w));
  weighted::State wp_state(wp_header, channel.w, channel.h);
  tree_samples.PrepareForSamples(pixel_fraction * channel.h * channel.w + 64);
  const bool multiple_predictors = tree_samples.NumPredictors() != 1;
  auto compute_sample = [&](const pixel_type *p, size_t x, size_t y) {
    pixel_type_w pred[kNumModularPredictors];
    if (multiple_predictors) {
      PredictLearnAll(&properties, channel.w, p + x, onerow, x, y, references,
                      &wp_state, pred);
    } else {
      pred[static_cast<int>(tree_samples.PredictorFromIndex(0))] =
          PredictLearn(&properties, channel.w, p + x, onerow, x, y,
                       tree_samples.PredictorFromIndex(0), references,
                       &wp_state)
              .guess;
    }
    (*total_pixels)++;
    if (use_sample()) {
      tree_samples.AddSample(p[x], properties, pred);
    }
    wp_state.UpdateErrors(p[x], x, y, channel.w);
  };

  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    PrecomputeReferences(channel, y, image, chan, &references);
    InitPropsRow(&properties, static_props, y);

    // TODO(veluca): avoid computing WP if we don't use its property or
    // predictions.
    if (y > 1 && channel.w > 8 && references.w == 0) {
      for (size_t x = 0; x < 2; x++) {
        compute_sample(p, x, y);
      }
      for (size_t x = 2; x < channel.w - 2; x++) {
        pixel_type_w pred[kNumModularPredictors];
        if (multiple_predictors) {
          PredictLearnAllNEC(&properties, channel.w, p + x, onerow, x, y,
                             references, &wp_state, pred);
        } else {
          pred[static_cast<int>(tree_samples.PredictorFromIndex(0))] =
              PredictLearnNEC(&properties, channel.w, p + x, onerow, x, y,
                              tree_samples.PredictorFromIndex(0), references,
                              &wp_state)
                  .guess;
        }
        (*total_pixels)++;
        if (use_sample()) {
          tree_samples.AddSample(p[x], properties, pred);
        }
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
      for (size_t x = channel.w - 2; x < channel.w; x++) {
        compute_sample(p, x, y);
      }
    } else {
      for (size_t x = 0; x < channel.w; x++) {
        compute_sample(p, x, y);
      }
    }
  }
  return true;
}

StatusOr<Tree> LearnTree(
    TreeSamples &&tree_samples, size_t total_pixels,
    const ModularOptions &options,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {},
    StaticPropRange static_prop_range = {}) {
  Tree tree;
  for (size_t i = 0; i < kNumStaticProperties; i++) {
    if (static_prop_range[i][1] == 0) {
      static_prop_range[i][1] = std::numeric_limits<uint32_t>::max();
    }
  }
  if (!tree_samples.HasSamples()) {
    tree.emplace_back();
    tree.back().predictor = tree_samples.PredictorFromIndex(0);
    tree.back().property = -1;
    tree.back().predictor_offset = 0;
    tree.back().multiplier = 1;
    return tree;
  }
  float pixel_fraction = tree_samples.NumSamples() * 1.0f / total_pixels;
  float required_cost = pixel_fraction * 0.9 + 0.1;
  tree_samples.AllSamplesDone();
  JXL_RETURN_IF_ERROR(ComputeBestTree(
      tree_samples, options.splitting_heuristics_node_threshold * required_cost,
      multiplier_info, static_prop_range, options.fast_decode_multiplier,
      &tree));
  return tree;
}

Status EncodeModularChannelMAANS(const Image &image, pixel_type chan,
                                 const weighted::Header &wp_header,
                                 const Tree &global_tree, Token **tokenpp,
                                 size_t group_id, bool skip_encoder_fast_path) {
  const Channel &channel = image.channel[chan];
  JxlMemoryManager *memory_manager = channel.memory_manager();
  Token *tokenp = *tokenpp;
  JXL_ENSURE(channel.w != 0 && channel.h != 0);

  Image3F predictor_img;
  if (kWantDebug) {
    JXL_ASSIGN_OR_RETURN(predictor_img,
                         Image3F::Create(memory_manager, channel.w, channel.h));
  }

  JXL_DEBUG_V(6,
              "Encoding %" PRIuS "x%" PRIuS
              " channel %d, "
              "(shift=%i,%i)",
              channel.w, channel.h, chan, channel.hshift, channel.vshift);

  std::array<pixel_type, kNumStaticProperties> static_props = {
      {chan, static_cast<int>(group_id)}};
  bool use_wp;
  bool is_wp_only;
  bool is_gradient_only;
  size_t num_props;
  FlatTree tree = FilterTree(global_tree, static_props, &num_props, &use_wp,
                             &is_wp_only, &is_gradient_only);
  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Encoding using a MA tree with %" PRIuS " nodes", tree.size());

  // Check if this tree is a WP-only tree with a small enough property value
  // range.
  // Initialized to avoid clang-tidy complaining.
  auto tree_lut = jxl::make_unique<TreeLut<uint16_t, false, false>>();
  if (is_wp_only) {
    is_wp_only = TreeToLookupTable(tree, *tree_lut);
  }
  if (is_gradient_only) {
    is_gradient_only = TreeToLookupTable(tree, *tree_lut);
  }

  if (is_wp_only && !skip_encoder_fast_path) {
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(Predictor::Weighted)[c]),
                &predictor_img.Plane(c));
    }
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    weighted::State wp_state(wp_header, channel.w, channel.h);
    Properties properties(1);
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        size_t offset = 0;
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        pixel_type_w topright =
            (x + 1 < channel.w && y ? *(r + x + 1 - onerow) : top);
        pixel_type_w toptop = (y > 1 ? *(r + x - onerow - onerow) : top);
        int32_t guess = wp_state.Predict</*compute_properties=*/true>(
            x, y, channel.w, top, left, topright, topleft, toptop, &properties,
            offset);
        uint32_t pos =
            kPropRangeFast +
            jxl::Clamp1(properties[0], -kPropRangeFast, kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut->context_lookup[pos];
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(ctx_id, PackSigned(residual));
        wp_state.UpdateErrors(r[x], x, y, channel.w);
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (tree.size() == 1 && tree[0].predictor == Predictor::Gradient &&
             tree[0].multiplier == 1 && tree[0].predictor_offset == 0 &&
             !skip_encoder_fast_path) {
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(Predictor::Gradient)[c]),
                &predictor_img.Plane(c));
    }
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(tree[0].childID, PackSigned(residual));
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (is_gradient_only && !skip_encoder_fast_path) {
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(Predictor::Gradient)[c]),
                &predictor_img.Plane(c));
    }
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        uint32_t pos =
            kPropRangeFast +
            std::min<pixel_type_w>(
                std::max<pixel_type_w>(-kPropRangeFast, top + left - topleft),
                kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut->context_lookup[pos];
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(ctx_id, PackSigned(residual));
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (tree.size() == 1 && tree[0].predictor == Predictor::Zero &&
             tree[0].multiplier == 1 && tree[0].predictor_offset == 0 &&
             !skip_encoder_fast_path) {
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(Predictor::Zero)[c]),
                &predictor_img.Plane(c));
    }
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        *tokenp++ = Token(tree[0].childID, PackSigned(p[x]));
      }
    }
  } else if (tree.size() == 1 && tree[0].predictor != Predictor::Weighted &&
             (tree[0].multiplier & (tree[0].multiplier - 1)) == 0 &&
             tree[0].predictor_offset == 0 && !skip_encoder_fast_path) {
    // multiplier is a power of 2.
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(tree[0].predictor)[c]),
                &predictor_img.Plane(c));
    }
    uint32_t mul_shift =
        FloorLog2Nonzero(static_cast<uint32_t>(tree[0].multiplier));
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult pred = PredictNoTreeNoWP(channel.w, r + x, onerow, x,
                                                  y, tree[0].predictor);
        pixel_type_w residual = r[x] - pred.guess;
        JXL_DASSERT((residual >> mul_shift) * tree[0].multiplier == residual);
        *tokenp++ = Token(tree[0].childID, PackSigned(residual >> mul_shift));
      }
    }

  } else if (!use_wp && !skip_encoder_fast_path) {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    Properties properties(num_props);
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, image, chan, &references);
      float *pred_img_row[3];
      if (kWantDebug) {
        for (size_t c = 0; c < 3; c++) {
          pred_img_row[c] = predictor_img.PlaneRow(c, y);
        }
      }
      InitPropsRow(&properties, static_props, y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeNoWP(&properties, channel.w, p + x, onerow, x, y,
                            tree_lookup, references);
        if (kWantDebug) {
          for (size_t i = 0; i < 3; i++) {
            pred_img_row[i][x] = PredictorColor(res.predictor)[i];
          }
        }
        pixel_type_w residual = p[x] - res.guess;
        JXL_DASSERT(residual % res.multiplier == 0);
        *tokenp++ = Token(res.context, PackSigned(residual / res.multiplier));
      }
    }
  } else {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    Properties properties(num_props);
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    weighted::State wp_state(wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, image, chan, &references);
      float *pred_img_row[3];
      if (kWantDebug) {
        for (size_t c = 0; c < 3; c++) {
          pred_img_row[c] = predictor_img.PlaneRow(c, y);
        }
      }
      InitPropsRow(&properties, static_props, y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeWP(&properties, channel.w, p + x, onerow, x, y,
                          tree_lookup, references, &wp_state);
        if (kWantDebug) {
          for (size_t i = 0; i < 3; i++) {
            pred_img_row[i][x] = PredictorColor(res.predictor)[i];
          }
        }
        pixel_type_w residual = p[x] - res.guess;
        JXL_DASSERT(residual % res.multiplier == 0);
        *tokenp++ = Token(res.context, PackSigned(residual / res.multiplier));
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
    }
  }
  /* TODO(szabadka): Add cparams to the call stack here.
  if (kWantDebug && WantDebugOutput(cparams)) {
    DumpImage(
        cparams,
        ("pred_" + ToString(group_id) + "_" + ToString(chan)).c_str(),
        predictor_img);
  }
  */
  *tokenpp = tokenp;
  return true;
}

}  // namespace

Tree PredefinedTree(ModularOptions::TreeKind tree_kind, size_t total_pixels,
                    int bitdepth, int prevprop) {
  switch (tree_kind) {
    case ModularOptions::TreeKind::kJpegTranscodeACMeta:
      // All the data is 0, so no need for a fancy tree.
      return {PropertyDecisionNode::Leaf(Predictor::Zero)};
    case ModularOptions::TreeKind::kTrivialTreeNoPredictor:
      // All the data is 0, so no need for a fancy tree.
      return {PropertyDecisionNode::Leaf(Predictor::Zero)};
    case ModularOptions::TreeKind::kFalconACMeta:
      // All the data is 0 except the quant field. TODO(veluca): make that 0
      // too.
      return {PropertyDecisionNode::Leaf(Predictor::Left)};
    case ModularOptions::TreeKind::kACMeta: {
      // Small image.
      if (total_pixels < 1024) {
        return {PropertyDecisionNode::Leaf(Predictor::Left)};
      }
      Tree tree;
      // 0: c > 1
      tree.push_back(PropertyDecisionNode::Split(0, 1, 1));
      // 1: c > 2
      tree.push_back(PropertyDecisionNode::Split(0, 2, 3));
      // 2: c > 0
      tree.push_back(PropertyDecisionNode::Split(0, 0, 5));
      // 3: EPF control field (all 0 or 4), top > 3
      tree.push_back(PropertyDecisionNode::Split(6, 3, 21));
      // 4: ACS+QF, y > 0
      tree.push_back(PropertyDecisionNode::Split(2, 0, 7));
      // 5: CfL x
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Gradient));
      // 6: CfL b
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Gradient));
      // 7: QF: split according to the left quant value.
      tree.push_back(PropertyDecisionNode::Split(7, 5, 9));
      // 8: ACS: split in 4 segments (8x8 from 0 to 3, large square 4-5, large
      // rectangular 6-11, 8x8 12+), according to previous ACS value.
      tree.push_back(PropertyDecisionNode::Split(7, 5, 15));
      // QF
      tree.push_back(PropertyDecisionNode::Split(7, 11, 11));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 13));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      // ACS
      tree.push_back(PropertyDecisionNode::Split(7, 11, 17));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 19));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      // EPF, left > 3
      tree.push_back(PropertyDecisionNode::Split(7, 3, 23));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 25));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      return tree;
    }
    case ModularOptions::TreeKind::kWPFixedDC: {
      std::vector<int32_t> cutoffs = {
          -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15,
          -11,  -7,   -4,   -3,   -1,   0,   1,   3,   5,   7,   11,
          15,   23,   31,   47,   63,   95,  127, 191, 255, 392, 500};
      return MakeFixedTree(kWPProp, cutoffs, Predictor::Weighted, total_pixels,
                           bitdepth);
    }
    case ModularOptions::TreeKind::kGradientFixedDC: {
      std::vector<int32_t> cutoffs = {
          -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15,
          -11,  -7,   -4,   -3,   -1,   0,   1,   3,   5,   7,   11,
          15,   23,   31,   47,   63,   95,  127, 191, 255, 392, 500};
      return MakeFixedTree(
          prevprop > 0 ? kNumNonrefProperties + 2 : kGradientProp, cutoffs,
          Predictor::Gradient, total_pixels, bitdepth);
    }
    case ModularOptions::TreeKind::kLearn: {
      JXL_DEBUG_ABORT("internal: kLearn is not predefined tree");
      return {};
    }
  }
  JXL_DEBUG_ABORT("internal: unexpected TreeKind: %d",
                  static_cast<int>(tree_kind));
  return {};
}

void SetWPHeader(const ModularOptions& options, weighted::Header* header) {
  Bundle::Init(header);
  weighted::PredictorMode(options.wp_mode, header);
  if (options.has_wp_params) {
    const auto& p = options.wp_params;
    header->p1C = p[0];
    header->p2C = p[1];
    header->p3Ca = p[2];
    header->p3Cb = p[3];
    header->p3Cc = p[4];
    header->p3Cd = p[5];
    header->p3Ce = p[6];
    for (size_t k = 0; k < 4; k++) header->w[k] = p[7 + k];
  }
}

StatusOr<Tree> LearnTree(
    const Image *images, const ModularOptions *options, const uint32_t start,
    const uint32_t stop,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {}) {
  TreeSamples tree_samples;
  JXL_RETURN_IF_ERROR(tree_samples.SetPredictor(options[start].predictor,
                                                options[start].wp_tree_mode));
  JXL_RETURN_IF_ERROR(
      tree_samples.SetProperties(options[start].splitting_heuristics_properties,
                                 options[start].wp_tree_mode));
  uint32_t max_c = 0;
  std::vector<pixel_type> pixel_samples;
  std::vector<pixel_type> diff_samples;
  std::vector<uint32_t> group_pixel_count;
  std::vector<uint32_t> channel_pixel_count;
  for (uint32_t i = start; i < stop; i++) {
    max_c = std::max<uint32_t>(images[i].channel.size(), max_c);
    CollectPixelSamples(images[i], options[i], i, group_pixel_count,
                        channel_pixel_count, pixel_samples, diff_samples);
  }
  StaticPropRange range;
  range[0] = {{0, max_c}};
  range[1] = {{start, stop}};

  tree_samples.PreQuantizeProperties(
      range, multiplier_info, group_pixel_count, channel_pixel_count,
      pixel_samples, diff_samples, options[start].max_property_values);

  size_t total_pixels = 0;
  for (size_t i = 0; i < images[start].channel.size(); i++) {
    if (i >= images[start].nb_meta_channels &&
        (images[start].channel[i].w > options[start].max_chan_size ||
         images[start].channel[i].h > options[start].max_chan_size)) {
      break;
    }
    total_pixels += images[start].channel[i].w * images[start].channel[i].h;
  }
  total_pixels = std::max<size_t>(total_pixels, 1);

  weighted::Header wp_header;

  for (size_t i = start; i < stop; i++) {
    size_t nb_channels = images[i].channel.size();

    if (images[i].w == 0 || images[i].h == 0 || nb_channels < 1)
      continue;  // is there any use for a zero-channel image?
    if (images[i].error) return JXL_FAILURE("Invalid image");
    JXL_ENSURE(options[i].tree_kind == ModularOptions::TreeKind::kLearn);

    JXL_DEBUG_V(
        2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
        nb_channels, images[i].bitdepth, images[i].w, images[i].h);

    // encode transforms
    Bundle::Init(&wp_header);
    if (PredictorHasWeighted(options[i].predictor)) {
      SetWPHeader(options[i], &wp_header);
    }

    // Gather tree data
    for (size_t c = 0; c < nb_channels; c++) {
      if (c >= images[i].nb_meta_channels &&
          (images[i].channel[c].w > options[i].max_chan_size ||
           images[i].channel[c].h > options[i].max_chan_size)) {
        break;
      }
      if (!images[i].channel[c].w || !images[i].channel[c].h) {
        continue;  // skip empty channels
      }
      JXL_RETURN_IF_ERROR(GatherTreeData(images[i], c, i, wp_header, options[i],
                                         tree_samples, &total_pixels));
    }
  }

  // TODO(veluca): parallelize more.
  JXL_ASSIGN_OR_RETURN(Tree tree,
                       LearnTree(std::move(tree_samples), total_pixels,
                                 options[start], multiplier_info, range));
  return tree;
}

Status EvaluateTreeWithZeroResiduals(const Tree& tree, size_t group_id,
                                     Image* image,
                                     const weighted::Header& wp_header) {
  JxlMemoryManager* memory_manager = image->memory_manager();
  for (size_t chan = 0; chan < image->channel.size(); chan++) {
    Channel& channel = image->channel[chan];
    if (channel.w == 0 || channel.h == 0) continue;
    std::array<pixel_type, kNumStaticProperties> static_props = {
        {static_cast<pixel_type>(chan), static_cast<int>(group_id)}};
    bool has_wp;
    bool is_wp_only;
    bool is_gradient_only;
    size_t num_props;
    FlatTree flat_tree = FilterTree(tree, static_props, &num_props, &has_wp,
                                    &is_wp_only, &is_gradient_only);
    MATreeLookup tree_lookup(flat_tree);
    Properties properties(num_props);
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    weighted::State wp_state(wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type* JXL_RESTRICT p = channel.Row(y);
      InitPropsRow(&properties, static_props, y);
      PrecomputeReferences(channel, y, *image, chan, &references);
      for (size_t x = 0; x < channel.w; x++) {
        // A zero residual: the sample is the prediction (a decoder computes
        // residual * multiplier + guess).
        PredictionResult res =
            PredictTreeWP(&properties, channel.w, p + x, onerow, x, y,
                          tree_lookup, references, &wp_state);
        p[x] = res.guess;
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
    }
  }
  return true;
}

// The tokens of a stream whose residuals are `pattern` (all zero if null), for
// the channels [0, num_coded) of `image` (only their sizes are used), with LZ77
// applied: the residuals of the first prefix + period samples as symbols, in
// the contexts `tree` gives there (their samples are computed: prediction plus
// residual), then one LZ77 copy with the period as distance for the rest.
// `distance_multiplier` is the stream's (the widest channel), `num_contexts`
// the tree's; the distance symbol goes to context num_contexts.
Status TokenizeResidualPattern(const Tree& tree, size_t group_id,
                               const Image& image, size_t num_coded,
                               const ResidualPattern* pattern,
                               size_t distance_multiplier, size_t num_contexts,
                               bool inner_lz77, bool context_costs,
                               const weighted::Header& wp_header,
                               std::vector<Token>* tokens) {
  static const ResidualPattern kZero = {{}, {0}};
  const ResidualPattern& p = pattern ? *pattern : kZero;
  JXL_ENSURE(!p.period.empty());
  size_t total = 0;
  // The meta channels (e.g. palette entries) keep zero residuals: the
  // pattern starts at the first other channel.
  size_t meta = 0;
  for (size_t i = 0; i < num_coded; i++) {
    total += image.channel[i].w * image.channel[i].h;
    if (i < image.nb_meta_channels && !p.include_meta) {
      meta += image.channel[i].w * image.channel[i].h;
    }
  }
  if (total == 0) return true;
  const size_t period = p.period.size();
  const size_t prefix = meta + p.prefix.size();
  const auto residual = [&](size_t k) -> int32_t {
    if (k < meta) return 0;
    return k < prefix ? p.prefix[k - meta] : p.period[(k - prefix) % period];
  };
  // Symbols up to here; the LZ77 copy needs at least min_length (3) values.
  size_t literals = std::min(total, prefix + period);
  if (total - literals < 3) literals = total;
  // Samples of the first literals (+ 1 for the copy's context).
  const size_t needed = std::min(total, literals + 1);
  JxlMemoryManager* memory_manager = image.memory_manager();
  Image work(memory_manager);
  work.w = image.w;
  work.h = image.h;
  work.bitdepth = image.bitdepth;
  work.nb_meta_channels = image.nb_meta_channels;
  // Residual symbols and contexts of the first `needed` samples.
  std::vector<uint32_t> values;
  std::vector<int> contexts;
  values.reserve(needed);
  contexts.reserve(needed);
  size_t k = 0;
  for (size_t chan = 0; chan < num_coded && k < needed; chan++) {
    const Channel& from = image.channel[chan];
    JXL_ASSIGN_OR_RETURN(
        Channel ch, Channel::Create(memory_manager, from.w, from.h, from.hshift,
                                    from.vshift));
    work.channel.emplace_back(std::move(ch));
    Channel& channel = work.channel.back();
    if (channel.w == 0 || channel.h == 0) continue;
    std::array<pixel_type, kNumStaticProperties> static_props = {
        {static_cast<pixel_type>(chan), static_cast<int>(group_id)}};
    bool has_wp;
    bool is_wp_only;
    bool is_gradient_only;
    size_t num_props;
    FlatTree flat_tree = FilterTree(tree, static_props, &num_props, &has_wp,
                                    &is_wp_only, &is_gradient_only);
    MATreeLookup tree_lookup(flat_tree);
    Properties properties(num_props);
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    weighted::State wp_state(wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h && k < needed; y++) {
      pixel_type* JXL_RESTRICT row = channel.Row(y);
      InitPropsRow(&properties, static_props, y);
      PrecomputeReferences(channel, y, work, chan, &references);
      for (size_t x = 0; x < channel.w && k < needed; x++, k++) {
        PredictionResult res =
            PredictTreeWP(&properties, channel.w, row + x, onerow, x, y,
                          tree_lookup, references, &wp_state);
        const int32_t r = residual(k);
        row[x] = static_cast<pixel_type>(res.guess + static_cast<int64_t>(r) *
                                                         res.multiplier);
        wp_state.UpdateErrors(row[x], x, y, channel.w);
        values.push_back(PackSigned(r));
        contexts.push_back(res.context);
      }
    }
  }
  const auto distance_symbol = [&](size_t distance) {
    return static_cast<uint32_t>(distance_multiplier != 0
                                     ? distance + kNumSpecialDistances - 1
                                     : distance - 1);
  };
  // The literal part, with LZ77 where it repeats itself (greedy: runs and
  // repeated blocks of at least kMinMatch values, found with a hash of the
  // next 4 values and the most recent candidates).
  constexpr size_t kMinMatch = 8;
  constexpr size_t kMaxCandidates = 32;
  // Estimated cost of the literals, so that a match is only taken where it
  // replaces more bits than it costs: from the frequencies of the values
  // (with a floor), or with context_costs from their frequencies in their own
  // context. Neither is always better (contexts get clustered), so the tool
  // tries both.
  std::vector<double> lit_cost(literals + 1, 0.0);
  {
    std::unordered_map<uint64_t, size_t> freq;
    std::unordered_map<int, size_t> ctx_total;
    const auto key = [&](size_t i) -> uint64_t {
      if (!context_costs) return values[i];
      return (static_cast<uint64_t>(static_cast<uint32_t>(contexts[i])) << 32) |
             values[i];
    };
    for (size_t i = 0; i < literals; i++) {
      freq[key(i)]++;
      ctx_total[contexts[i]]++;
    }
    for (size_t i = 0; i < literals; i++) {
      const double f = static_cast<double>(freq[key(i)]);
      lit_cost[i + 1] =
          lit_cost[i] +
          (context_costs
               ? std::log2(static_cast<double>(ctx_total[contexts[i]]) / f)
               : std::max(0.05, std::log2(static_cast<double>(literals) / f)));
    }
  }
  const auto match_cost = [](size_t len, size_t dist) {
    return 6.0 + 2.0 * std::log2(static_cast<double>(len)) +
           std::log2(static_cast<double>(dist) + 1.0);
  };
  std::unordered_map<uint64_t, std::vector<uint32_t>> recent;
  const auto hash_at = [&](size_t i) {
    uint64_t h = 0;
    for (size_t j = 0; j < 4; j++)
      h = h * 0x9E3779B97F4A7C15ull + values[i + j];
    return h;
  };
  for (size_t i = 0; i < literals;) {
    size_t best_len = 0;
    if (!inner_lz77) {
      tokens->emplace_back(contexts[i], values[i]);
      i++;
      continue;
    }
    size_t best_dist = 0;
    size_t run_len = 0;
    const auto try_candidate = [&](size_t j) {
      size_t len = 0;
      while (i + len < literals && values[j + len] == values[i + len]) len++;
      if (j + 1 == i) run_len = len;
      if (len >= kMinMatch &&
          lit_cost[i + len] - lit_cost[i] > match_cost(len, i - j) &&
          len > best_len) {
        best_len = len;
        best_dist = i - j;
      }
    };
    if (i > 0) try_candidate(i - 1);
    if (best_len < kMinMatch && run_len >= kMinMatch) {
      // A run that is not worth a match (its literals are cheap): no match
      // starting inside it gains more, so it is all literals (and it is not
      // scanned again from every position: quadratic on long runs).
      for (size_t e = i; e < i + run_len; e++) {
        tokens->emplace_back(contexts[e], values[e]);
      }
      for (size_t e = i; e < i + run_len && e + 4 <= literals; e++) {
        auto& list = recent[hash_at(e)];
        if (list.size() == kMaxCandidates) list.erase(list.begin());
        list.push_back(static_cast<uint32_t>(e));
      }
      i += run_len;
      continue;
    }
    if (i + 4 <= literals) {
      auto it = recent.find(hash_at(i));
      if (it != recent.end()) {
        for (size_t c = it->second.size(); c-- > 0;)
          try_candidate(it->second[c]);
      }
    }
    const size_t step = best_len >= kMinMatch ? best_len : 1;
    if (best_len >= kMinMatch) {
      tokens->emplace_back(contexts[i], static_cast<uint32_t>(best_len - 3));
      tokens->back().is_lz77_length = true;
      tokens->emplace_back(static_cast<uint32_t>(num_contexts),
                           distance_symbol(best_dist));
    } else {
      tokens->emplace_back(contexts[i], values[i]);
    }
    for (size_t e = i; e < i + step; e++) {
      if (e + 4 > literals) break;
      auto& list = recent[hash_at(e)];
      if (list.size() == kMaxCandidates) list.erase(list.begin());
      list.push_back(static_cast<uint32_t>(e));
    }
    i += step;
  }
  if (literals < total) {
    // The rest: a copy of the values `period` back (the pattern repeats with
    // that period from the end of the prefix on).
    tokens->emplace_back(contexts[literals],
                         static_cast<uint32_t>(total - literals - 3));
    tokens->back().is_lz77_length = true;
    tokens->emplace_back(static_cast<uint32_t>(num_contexts),
                         distance_symbol(period));
  }
  return true;
}

Status EvaluateTreeWithResiduals(const Tree& tree, size_t group_id,
                                 const ResidualPattern* pattern,
                                 size_t first_channel, size_t num_channels,
                                 Image* image, int64_t* min_value,
                                 int64_t* max_value,
                                 const weighted::Header& wp_header) {
  JxlMemoryManager* memory_manager = image->memory_manager();
  size_t meta = 0;
  const bool include_meta = pattern != nullptr && pattern->include_meta;
  for (size_t i = 0;
       !include_meta && i < image->nb_meta_channels && i < num_channels; i++) {
    meta += image->channel[i].w * image->channel[i].h;
  }
  const size_t prefix = pattern ? meta + pattern->prefix.size() : 0;
  const auto residual = [&](size_t k) -> int64_t {
    if (!pattern || k < meta) return 0;
    if (k < prefix) return pattern->prefix[k - meta];
    return pattern->period[(k - prefix) % pattern->period.size()];
  };
  size_t k = 0;
  for (size_t chan = 0; chan < num_channels; chan++) {
    Channel& channel = image->channel[chan];
    if (channel.w == 0 || channel.h == 0) continue;
    if (chan < first_channel) {
      k += channel.w * channel.h;
      for (size_t y = 0; y < channel.h; y++) {
        for (size_t x = 0; x < channel.w; x++) {
          *min_value = std::min<int64_t>(*min_value, channel.Row(y)[x]);
          *max_value = std::max<int64_t>(*max_value, channel.Row(y)[x]);
        }
      }
      continue;
    }
    std::array<pixel_type, kNumStaticProperties> static_props = {
        {static_cast<pixel_type>(chan), static_cast<int>(group_id)}};
    bool has_wp;
    bool is_wp_only;
    bool is_gradient_only;
    size_t num_props;
    FlatTree flat_tree = FilterTree(tree, static_props, &num_props, &has_wp,
                                    &is_wp_only, &is_gradient_only);
    MATreeLookup tree_lookup(flat_tree);
    Properties properties(num_props);
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    weighted::State wp_state(wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type* JXL_RESTRICT row = channel.Row(y);
      InitPropsRow(&properties, static_props, y);
      PrecomputeReferences(channel, y, *image, chan, &references);
      for (size_t x = 0; x < channel.w; x++, k++) {
        PredictionResult res =
            PredictTreeWP(&properties, channel.w, row + x, onerow, x, y,
                          tree_lookup, references, &wp_state);
        const int64_t v = res.guess + residual(k) * res.multiplier;
        *min_value = std::min(*min_value, v);
        *max_value = std::max(*max_value, v);
        // Saturate so that later predictions stay well defined.
        row[x] = static_cast<pixel_type>(std::min<int64_t>(
            std::max<int64_t>(v, std::numeric_limits<pixel_type>::min()),
            std::numeric_limits<pixel_type>::max()));
        wp_state.UpdateErrors(row[x], x, y, channel.w);
      }
    }
  }
  return true;
}

Status ModularCompress(const Image &image, const ModularOptions &options,
                       size_t group_id, const Tree &tree, GroupHeader &header,
                       std::vector<Token> &tokens, size_t *width) {
  size_t nb_channels = image.channel.size();

  if (image.w == 0 || image.h == 0 || nb_channels < 1)
    return true;  // is there any use for a zero-channel image?
  if (image.error) return JXL_FAILURE("Invalid image");

  JXL_DEBUG_V(
      2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
      nb_channels, image.bitdepth, image.w, image.h);

  // encode transforms
  Bundle::Init(&header);
  if (PredictorHasWeighted(options.predictor) || options.has_wp_params) {
    SetWPHeader(options, &header.wp_header);
  }
  header.transforms = image.transform;
  header.use_global_tree = true;

  size_t image_width = 0;
  size_t total_tokens = 0;
  for (size_t i = 0; i < nb_channels; i++) {
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    if (image.channel[i].w > image_width) image_width = image.channel[i].w;
    total_tokens += image.channel[i].w * image.channel[i].h;
  }
  if (options.zero_tokens && options.residual_patterns != nullptr) {
    size_t num_coded = 0;
    while (num_coded < nb_channels &&
           !(num_coded >= image.nb_meta_channels &&
             (image.channel[num_coded].w > options.max_chan_size ||
              image.channel[num_coded].h > options.max_chan_size))) {
      num_coded++;
    }
    const ResidualPattern* pattern = nullptr;
    auto it = options.residual_patterns->find(static_cast<int>(group_id));
    if (it == options.residual_patterns->end()) {
      it = options.residual_patterns->find(-1);
    }
    if (it != options.residual_patterns->end()) pattern = &it->second;
    JXL_RETURN_IF_ERROR(TokenizeResidualPattern(
        tree, group_id, image, num_coded, pattern, image_width,
        (tree.size() + 1) / 2, options.residual_inner_lz77,
        options.residual_lz77_context_costs, header.wp_header, &tokens));
  } else if (options.zero_tokens && options.code_meta_channels &&
             image.nb_meta_channels > 0) {
    // The meta channels are coded from the image; the other channels are
    // what the tree gives with zero residuals, tokenized with their real
    // contexts (so that the meta channels' contexts only hold their tokens).
    size_t num_coded = 0;
    while (num_coded < nb_channels &&
           !(num_coded >= image.nb_meta_channels &&
             (image.channel[num_coded].w > options.max_chan_size ||
              image.channel[num_coded].h > options.max_chan_size))) {
      num_coded++;
    }
    // Only the channels of this stream (the others may be placeholders).
    Image coded(image.memory_manager());
    coded.w = image.w;
    coded.h = image.h;
    coded.bitdepth = image.bitdepth;
    coded.nb_meta_channels = image.nb_meta_channels;
    for (size_t i = 0; i < num_coded; i++) {
      const Channel& from = image.channel[i];
      JXL_ASSIGN_OR_RETURN(
          Channel ch, Channel::Create(image.memory_manager(), from.w, from.h,
                                      from.hshift, from.vshift));
      coded.channel.emplace_back(std::move(ch));
    }
    JXL_RETURN_IF_ERROR(EvaluateTreeWithZeroResiduals(tree, group_id, &coded,
                                                      header.wp_header));
    for (size_t i = 0; i < image.nb_meta_channels && i < num_coded; i++) {
      for (size_t y = 0; y < image.channel[i].h; y++) {
        memcpy(coded.channel[i].Row(y), image.channel[i].Row(y),
               image.channel[i].w * sizeof(pixel_type));
      }
    }
    size_t pos = tokens.size();
    tokens.resize(pos + total_tokens);
    Token* tokenp = tokens.data() + pos;
    for (size_t i = 0; i < num_coded; i++) {
      if (!coded.channel[i].w || !coded.channel[i].h) continue;
      JXL_RETURN_IF_ERROR(
          EncodeModularChannelMAANS(coded, i, header.wp_header, tree, &tokenp,
                                    group_id, options.skip_encoder_fast_path));
    }
    JXL_ENSURE(tokenp == tokens.data() + tokens.size());
  } else if (options.zero_tokens) {
    tokens.resize(tokens.size() + total_tokens, {0, 0});
  } else {
    // Do one big allocation for all the tokens we'll need,
    // to avoid reallocs that might require copying.
    size_t pos = tokens.size();
    tokens.resize(pos + total_tokens);
    Token *tokenp = tokens.data() + pos;
    for (size_t i = 0; i < nb_channels; i++) {
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      JXL_RETURN_IF_ERROR(
          EncodeModularChannelMAANS(image, i, header.wp_header, tree, &tokenp,
                                    group_id, options.skip_encoder_fast_path));
    }
    // Make sure we actually wrote all tokens
    JXL_ENSURE(tokenp == tokens.data() + tokens.size());
  }

  *width = image_width;

  return true;
}

Status ModularGenericCompress(const Image &image, const ModularOptions &opts,
                              BitWriter &writer, AuxOut *aux_out,
                              LayerType layer, size_t group_id) {
  size_t nb_channels = image.channel.size();

  if (image.w == 0 || image.h == 0 || nb_channels < 1)
    return true;  // is there any use for a zero-channel image?
  if (image.error) return JXL_FAILURE("Invalid image");

  ModularOptions options = opts;  // Make a copy to modify it.
  if (options.predictor == kUndefinedPredictor) {
    options.predictor = Predictor::Gradient;
  }

  size_t bits = writer.BitsWritten();

  JxlMemoryManager *memory_manager = image.memory_manager();
  JXL_DEBUG_V(
      2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
      nb_channels, image.bitdepth, image.w, image.h);

  // encode transforms
  GroupHeader header;
  Bundle::Init(&header);
  if (PredictorHasWeighted(options.predictor) || options.has_wp_params) {
    SetWPHeader(options, &header.wp_header);
  }
  header.transforms = image.transform;

  JXL_RETURN_IF_ERROR(Bundle::Write(header, &writer, layer, aux_out));

  // Compute tree.
  Tree tree;
  if (options.tree_kind == ModularOptions::TreeKind::kLearn) {
    JXL_ASSIGN_OR_RETURN(tree, LearnTree(&image, &options, 0, 1));
  } else {
    size_t total_pixels = 0;
    for (size_t i = 0; i < nb_channels; i++) {
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      total_pixels += image.channel[i].w * image.channel[i].h;
    }
    total_pixels = std::max<size_t>(total_pixels, 1);

    tree = PredefinedTree(options.tree_kind, total_pixels, image.bitdepth,
                          options.max_properties);
  }

  Tree decoded_tree;
  std::vector<std::vector<Token>> tree_tokens(1);
  JXL_RETURN_IF_ERROR(TokenizeTree(tree, tree_tokens.data(), &decoded_tree));
  JXL_ENSURE(tree.size() == decoded_tree.size());
  tree = std::move(decoded_tree);

  /* TODO(szabadka) Add text output callback
  if (kWantDebug && kPrintTree && WantDebugOutput(aux_out)) {
    PrintTree(*tree, aux_out->debug_prefix + "/tree_" + ToString(group_id));
  } */

  // Write tree
  EntropyEncodingData code;
  JXL_ASSIGN_OR_RETURN(
      size_t cost,
      BuildAndEncodeHistograms(memory_manager, options.histogram_params,
                               kNumTreeContexts, tree_tokens, &code, &writer,
                               LayerType::ModularTree, aux_out));
  JXL_RETURN_IF_ERROR(WriteTokens(tree_tokens[0], code, 0, &writer,
                                  LayerType::ModularTree, aux_out));

  size_t image_width = 0;
  std::vector<std::vector<Token>> tokens(1);
  // it puts `use_global_tree = true` in the header, but this is not used
  // further
  JXL_RETURN_IF_ERROR(ModularCompress(image, options, group_id, tree, header,
                                      tokens[0], &image_width));

  // Write data
  code = {};
  HistogramParams histo_params = options.histogram_params;
  histo_params.image_widths.push_back(image_width);
  JXL_ASSIGN_OR_RETURN(
      cost, BuildAndEncodeHistograms(memory_manager, histo_params,
                                     (tree.size() + 1) / 2, tokens, &code,
                                     &writer, layer, aux_out));
  (void)cost;
  JXL_RETURN_IF_ERROR(WriteTokens(tokens[0], code, 0, &writer, layer, aux_out));

  bits = writer.BitsWritten() - bits;
  JXL_DEBUG_V(4,
              "Modular-encoded a %" PRIuS "x%" PRIuS
              " bitdepth=%i nbchans=%" PRIuS " image in %" PRIuS " bytes",
              image.w, image.h, image.bitdepth, image.channel.size(), bits / 8);
  (void)bits;

  return true;
}

}  // namespace jxl
