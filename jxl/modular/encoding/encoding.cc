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

#include "jxl/modular/encoding/encoding.h"

#include <stdint.h>
#include <stdlib.h>

#include <cinttypes>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "jxl/base/fast_log.h"
#include "jxl/base/status.h"
#include "jxl/brotli.h"
#include "jxl/common.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/entropy_coder.h"
#include "jxl/fields.h"
#include "jxl/modular/encoding/context_predict.h"
#include "jxl/modular/encoding/ma.h"
#include "jxl/modular/options.h"
#include "jxl/modular/transform/transform.h"
#include "jxl/toc.h"

namespace jxl {

// Tries all the predictors, excluding Weighted, per row.
Predictor FindBest(const Image &image, pixel_type chan,
                   const pixel_type *JXL_RESTRICT p, intptr_t onerow, size_t y,
                   Predictor prev_predictor) {
  const Channel &channel = image.channel[chan];
  // TODO(veluca): use entropy/lz77 complexity?
  uint64_t sum_of_abs_residuals[kNumModularPredictors] = {};
  pixel_type_w predictions[kNumModularPredictors] = {};
  for (size_t x = 0; x < channel.w; x++) {
    PredictAllNoWP(channel, p + x, onerow, x, y, predictions);
    for (size_t i = 0; i < kNumModularPredictors; i++) {
      sum_of_abs_residuals[i] += abs(p[x] - predictions[i]);
    }
  }
  uint64_t best = sum_of_abs_residuals[0];
  uint64_t best_predictor = 0;
  for (size_t i = 1; i < kNumModularPredictors; i++) {
    if (i == (int)Predictor::Weighted) continue;
    if (best > sum_of_abs_residuals[i]) {
      best = sum_of_abs_residuals[i];
      best_predictor = i;
    }
  }
  uint64_t prev = sum_of_abs_residuals[(int)prev_predictor];
  // only change predictor if residuals are 10% smaller
  if (prev < best * 1.1) return prev_predictor;
  return (Predictor)best_predictor;
}

namespace {

// Removes all nodes that use a static property (i.e. channel or group ID) from
// the tree and collapses each node on even levels with its two children to
// produce a flatter tree. Also computes whether the resulting tree requires
// using the weighted predictor.
FlatTree FilterTree(const Tree &global_tree,
                    std::array<pixel_type, kNumStaticProperties> &static_props,
                    bool *use_wp) {
  *use_wp = false;
  size_t used_properties = 0;
  FlatTree output;
  std::queue<size_t> nodes;
  nodes.push(0);
  // Produces a trimmed and flattened tree by doing a BFS visit of the original
  // tree, ignoring branches that are known to be false and proceeding two
  // levels at a time to collapse nodes in a flatter tree; if an inner parent
  // node has a leaf as a child, the leaf is duplicated and an implicit fake
  // node is added. This allows to reduce the number of branches when traversing
  // the resulting flat tree.
  while (!nodes.empty()) {
    size_t cur = nodes.front();
    nodes.pop();
    // Skip nodes that we can decide now, by jumping directly to their children.
    while (global_tree[cur].property < kNumStaticProperties &&
           global_tree[cur].property != -1) {
      if (static_props[global_tree[cur].property] > global_tree[cur].splitval) {
        cur = global_tree[cur].childID;
      } else {
        cur = global_tree[cur].childID + 1;
      }
    }
    FlatDecisionNode flat;
    if (global_tree[cur].property == -1) {
      flat.property0 = -1;
      flat.childID = global_tree[cur].childID;
      flat.predictor = global_tree[cur].predictor;
      flat.predictor_offset = global_tree[cur].predictor_offset;
      if (flat.predictor == Predictor::Weighted) *use_wp = true;
      output.push_back(flat);
      continue;
    }
    flat.childID = output.size() + nodes.size() + 1;

    flat.property0 = global_tree[cur].property;
    flat.splitval0 = global_tree[cur].splitval;

    size_t child_id = global_tree[cur].childID;
    for (size_t i = 0; i < 2; i++) {
      size_t cur_child = child_id + i;
      // Skip nodes that we can decide now.
      while (global_tree[cur_child].property < kNumStaticProperties &&
             global_tree[cur_child].property != -1) {
        if (static_props[global_tree[cur_child].property] >
            global_tree[cur_child].splitval) {
          cur_child = global_tree[cur_child].childID;
        } else {
          cur_child = global_tree[cur_child].childID + 1;
        }
      }
      // We ended up in a leaf, add a dummy decision and two copies of the leaf.
      if (global_tree[cur_child].property == -1) {
        flat.properties[i] = 0;
        flat.splitvals[i] = std::numeric_limits<int32_t>::max();
        nodes.push(cur_child);
        nodes.push(cur_child);
      } else {
        flat.properties[i] = global_tree[cur_child].property;
        flat.splitvals[i] = global_tree[cur_child].splitval;
        nodes.push(global_tree[cur_child].childID);
        nodes.push(global_tree[cur_child].childID + 1);
      }
    }

    for (size_t j = 0; j < 2; j++) {
      if (flat.properties[j] >= kNumStaticProperties) {
        used_properties |= 1 << flat.properties[j];
      }
    }
    if (flat.property0 >= kNumStaticProperties) {
      used_properties |= 1 << flat.property0;
    }
    output.push_back(flat);
  }
  if (used_properties &
      (1 << (kNumNonrefProperties - weighted::kNumProperties))) {
    *use_wp = true;
  }

  return output;
}

}  // namespace

Status EncodeModularChannelMAANS(const Image &image, pixel_type chan,
                                 const weighted::Header &wp_header,
                                 const Tree &global_tree,
                                 const ModularOptions &options,
                                 std::vector<Token> *tokens, AuxOut *aux_out,
                                 size_t group_id, bool want_debug) {
  const Channel &channel = image.channel[chan];

  JXL_ASSERT(channel.w != 0 && channel.h != 0);

  Image3F predictor_img(channel.w, channel.h);

  JXL_DEBUG_V(6,
              "Encoding %zux%zu channel %d, "
              "(shift=%i,%i, cshift=%i,%i)",
              channel.w, channel.h, chan, channel.hshift, channel.vshift,
              channel.hcshift, channel.vcshift);

  Properties properties(NumProperties(image, options));
  std::array<pixel_type, kNumStaticProperties> static_props = {chan,
                                                               (int)group_id};
  bool use_wp;
  FlatTree tree = FilterTree(global_tree, static_props, &use_wp);
  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Encoding using a MA tree with %zu nodes", tree.size());

  const intptr_t onerow = channel.plane.PixelsPerRow();
  Channel references(properties.size() - kNumNonrefProperties, channel.w);
  weighted::State wp_state(wp_header, channel.w, channel.h);
  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    PrecomputeReferences(channel, y, image, chan, options, &references);
    float *pred_img_row[3] = {predictor_img.PlaneRow(0, y),
                              predictor_img.PlaneRow(1, y),
                              predictor_img.PlaneRow(2, y)};
    for (size_t x = 0; x < channel.w; x++) {
      PredictionResult res =
          PredictTreeWP(&properties, channel, static_props, p + x, onerow, x, y,
                        tree_lookup, references, &wp_state);
      for (size_t i = 0; i < 3; i++) {
        pred_img_row[i][x] = PredictorColor(res.predictor)[i];
      }
      tokens->emplace_back(res.context, PackSigned(p[x] - res.guess));
      wp_state.UpdateErrors(p[x], x, y, channel.w);
    }
  }
  if (want_debug && WantDebugOutput(aux_out)) {
    aux_out->DumpImage(
        ("pred_" + std::to_string(group_id) + "_" + std::to_string(chan))
            .c_str(),
        predictor_img);
  }
  return true;
}

Status DecodeModularChannelMAANS(BitReader *br, ANSSymbolReader *reader,
                                 const std::vector<uint8_t> &context_map,
                                 const Tree &global_tree,
                                 const weighted::Header &wp_header,
                                 const ModularOptions &options, pixel_type chan,
                                 size_t group_id, Image *image) {
  Channel &channel = image->channel[chan];

  std::array<pixel_type, kNumStaticProperties> static_props = {chan,
                                                               (int)group_id};
  // TODO(veluca): filter the tree according to static_props.

  // zero pixel channel? could happen
  if (channel.w == 0 || channel.h == 0) return true;

  channel.resize(channel.w, channel.h);
  bool tree_has_wp_prop_or_pred = false;
  FlatTree tree =
      FilterTree(global_tree, static_props, &tree_has_wp_prop_or_pred);

  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Decoded MA tree with %zu nodes", tree.size());
  Properties properties(NumProperties(*image, options));

  // MAANS decode

  if (tree.size() == 1) {
    // special optimized case: no meta-adaptation, so no need
    // to compute properties.
    Predictor predictor = tree[0].predictor;
    int64_t offset = tree[0].predictor_offset;
    size_t ctx_id = tree[0].childID;
    if (predictor == Predictor::Zero) {
      JXL_DEBUG_V(8, "Fast track.");
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        for (size_t x = 0; x < channel.w; x++) {
          uint32_t v = reader->ReadHybridUint(ctx_id, br, context_map);
          r[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), offset);
        }
      }
    } else if (predictor != Predictor::Weighted) {
      // special optimized case: no meta-adaptation, no wp, so no need to
      // compute properties
      JXL_DEBUG_V(8, "Quite fast track.");
      const intptr_t onerow = channel.plane.PixelsPerRow();
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type_w g =
              PredictNoTreeNoWP(channel, r + x, onerow, x, y, predictor).guess +
              offset;
          uint64_t v = reader->ReadHybridUint(ctx_id, br, context_map);
          r[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), g);
        }
      }
    } else {
      // special optimized case: no meta-adaptation, so no need to
      // compute properties
      JXL_DEBUG_V(8, "Somewhat fast track.");
      const intptr_t onerow = channel.plane.PixelsPerRow();
      weighted::State wp_state(wp_header, channel.w, channel.h);
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type_w g = PredictNoTreeWP(channel, r + x, onerow, x, y,
                                           predictor, &wp_state)
                               .guess +
                           offset;
          uint64_t v = reader->ReadHybridUint(ctx_id, br, context_map);
          r[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), g);
          wp_state.UpdateErrors(r[x], x, y, channel.w);
        }
      }
    }
  } else if (!tree_has_wp_prop_or_pred) {
    // special optimized case: the weighted predictor and its properties are not
    // used, so no need to compute weights and properties.
    JXL_DEBUG_V(8, "Slow track.");
    const intptr_t onerow = channel.plane.PixelsPerRow();
    Channel references(properties.size() - kNumNonrefProperties, channel.w);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, *image, chan, options, &references);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeNoWP(&properties, channel, static_props, p + x, onerow,
                            x, y, tree_lookup, references);
        uint64_t v = reader->ReadHybridUint(res.context, br, context_map);
        p[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), res.guess);
      }
    }
  } else {
    JXL_DEBUG_V(8, "Slowest track.");
    const intptr_t onerow = channel.plane.PixelsPerRow();
    Channel references(properties.size() - kNumNonrefProperties, channel.w);
    weighted::State wp_state(wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, *image, chan, options, &references);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeWP(&properties, channel, static_props, p + x, onerow, x,
                          y, tree_lookup, references, &wp_state);
        uint64_t v = reader->ReadHybridUint(res.context, br, context_map);
        p[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), res.guess);
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
    }
  }
  return true;
}

void GatherTreeData(const Image &image, pixel_type chan, size_t group_id,
                    const weighted::Header &wp_header,
                    const std::vector<Predictor> &predictors,
                    const ModularOptions &options,
                    std::vector<std::vector<int32_t>> &props,
                    std::vector<std::vector<int32_t>> &residuals,
                    size_t *total_pixels) {
  const Channel &channel = image.channel[chan];

  JXL_DEBUG_V(7, "Learning %zux%zu channel %d", channel.w, channel.h, chan);

  std::array<pixel_type, kNumStaticProperties> static_props = {chan,
                                                               (int)group_id};
  Properties properties(NumProperties(image, options));
  std::mt19937_64 gen(1);  // deterministic learning (also between threads)
  float pixel_fraction = std::min(1.0f, options.nb_repeats);
  // a fraction of 0 is used to disable learning entirely.
  if (pixel_fraction > 0) {
    pixel_fraction = std::max(
        pixel_fraction, std::min(1.0f, 1024.0f / (channel.w * channel.h)));
  }
  std::bernoulli_distribution dist(pixel_fraction);

  const intptr_t onerow = channel.plane.PixelsPerRow();
  Channel references(properties.size() - kNumNonrefProperties, channel.w);
  weighted::State wp_state(wp_header, channel.w, channel.h);
  if (props.empty()) {
    props.resize(properties.size());
    residuals.resize(predictors.size());
  }
  JXL_ASSERT(props.size() == properties.size());
  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    PrecomputeReferences(channel, y, image, chan, options, &references);
    for (size_t x = 0; x < channel.w; x++) {
      pixel_type_w res[kNumModularPredictors];
      if (predictors.size() != 1) {
        PredictLearnAll(&properties, channel, static_props, p + x, onerow, x, y,
                        references, &wp_state, res);
        for (size_t i = 0; i < predictors.size(); i++) {
          res[i] = p[x] - res[(int)predictors[i]];
        }
      } else {
        PredictionResult pres =
            PredictLearn(&properties, channel, static_props, p + x, onerow, x,
                         y, predictors[0], references, &wp_state);
        res[0] = p[x] - pres.guess;
      }
      (*total_pixels)++;
      if (dist(gen)) {
        for (size_t i = 0; i < predictors.size(); i++) {
          residuals[i].push_back(res[i]);
        }
        for (size_t i = 0; i < properties.size(); i++) {
          props[i].push_back(properties[i]);
        }
      }
      wp_state.UpdateErrors(p[x], x, y, channel.w);
    }
  }
}

Tree LearnTree(std::vector<Predictor> predictors,
               std::vector<std::vector<int32_t>> &&props,
               std::vector<std::vector<int32_t>> &&residuals,
               size_t total_pixels, const ModularOptions &options) {
  int64_t offset = 0;
  if (predictors.size() > 1 && !residuals[0].empty()) {
    int base_pred = 0;
    size_t base_pred_cost = 0;
    for (size_t i = 0; i < predictors.size(); i++) {
      int64_t sum = 0;
      for (size_t j = 0; j < residuals[i].size(); j++) {
        sum += residuals[i][j];
      }
      int64_t tot = residuals[i].size();
      int64_t off = sum > 0 ? (sum + tot / 2) / tot : (sum - tot / 2) / tot;
      size_t cost = 0;
      for (size_t j = 0; j < residuals[i].size(); j++) {
        cost += PackSigned(residuals[i][j] - off);
      }
      if (cost < base_pred_cost || i == 0) {
        base_pred = i;
        offset = off;
        base_pred_cost = cost;
      }
    }
    std::swap(predictors[base_pred], predictors[0]);
    std::swap(residuals[base_pred], residuals[0]);
  }
  if (residuals.empty() || residuals[0].empty()) {
    Tree tree;
    tree.emplace_back();
    tree.back().predictor = predictors.back();
    return tree;
  }
  std::vector<size_t> props_to_use;
  std::vector<std::vector<int>> compact_properties(props.size());
  // TODO(veluca): add an option for max total number of property values.
  ChooseAndQuantizeProperties(options.splitting_heuristics_max_properties,
                              options.splitting_heuristics_max_properties * 256,
                              offset, residuals, &props, &compact_properties,
                              &props_to_use);
  float pixel_fraction = props[0].size() * 1.0f / total_pixels;
  float required_cost = pixel_fraction * 0.9 + 0.1;
  Tree tree;
  ComputeBestTree(residuals, props, predictors, offset, compact_properties,
                  props_to_use,
                  options.splitting_heuristics_node_threshold * required_cost,
                  options.splitting_heuristics_max_properties,
                  options.fast_decode_multiplier, &tree);
  return tree;
}

Status EncodeModularChannelBrotli(const Image &image, pixel_type chan,
                                  const weighted::Header &wp_header,
                                  Predictor predictor, size_t total_pixels,
                                  size_t *JXL_RESTRICT pos,
                                  size_t *JXL_RESTRICT subpred_pos,
                                  PaddedBytes *JXL_RESTRICT data) {
  const Channel &channel = image.channel[chan];
  JXL_ASSERT(channel.w != 0 && channel.h != 0);
  Predictor subpredictor = Predictor::Gradient;
  int min, max;
  channel.compute_minmax(&min, &max);
  if (min == max) {
    predictor = subpredictor = Predictor::Zero;
  }
  const intptr_t onerow = channel.plane.PixelsPerRow();
  weighted::State wp_state(wp_header, channel.w, channel.h);
  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    if (predictor == Predictor::Variable) {
      subpredictor = FindBest(image, chan, p, onerow, y, subpredictor);
    } else {
      subpredictor = predictor;
    }
    (*data)[*subpred_pos] = (uint8_t)subpredictor;
    JXL_ASSERT((*data)[*subpred_pos] < kNumModularPredictors);
    (*subpred_pos)++;
    for (size_t x = 0; x < channel.w; x++) {
      pixel_type_w guess =
          PredictNoTreeWP(channel, p + x, onerow, x, y, subpredictor, &wp_state)
              .guess;
      pixel_type_w v = PackSigned(p[x] - guess);
      (*data)[*pos] = (v & 0xff);
      (*data)[*pos + total_pixels] = ((v >> 8) & 0xff);
      (*data)[*pos + 2 * total_pixels] = ((v >> 16) & 0xff);
      (*data)[*pos + 3 * total_pixels] = ((v >> 24) & 0xff);
      (*pos)++;
      wp_state.UpdateErrors(p[x], x, y, channel.w);
    }
  }
  return true;
}

Status DecodeModularChannelBrotli(const PaddedBytes &data,
                                  const weighted::Header &wp_header,
                                  size_t total_pixels, size_t *JXL_RESTRICT pos,
                                  size_t *JXL_RESTRICT subpred_pos,
                                  Image *image, pixel_type chan) {
  Channel *channel = &image->channel[chan];
  JXL_ASSERT(channel->w != 0 && channel->h != 0);

  channel->resize(channel->w, channel->h);

  bool no_predictor = true;
  bool no_wp = true;
  for (size_t y = 0; y < channel->h; y++) {
    if (data[*subpred_pos + y] != (uint8_t)Predictor::Zero) {
      no_predictor = false;
    }
    if (data[*subpred_pos + y] == (uint8_t)Predictor::Weighted) {
      no_wp = false;
    }
    if (data[*subpred_pos + y] >= (uint8_t)Predictor::Best) {
      return JXL_FAILURE("Invalid predictor");
    }
  }

  if (no_predictor) {
    *subpred_pos += channel->h;
    // special optimized case: no predictor
    JXL_DEBUG_V(8, "Fast track.");
    for (size_t y = 0; y < channel->h; y++) {
      pixel_type *JXL_RESTRICT r = channel->Row(y);
      for (size_t x = 0; x < channel->w; x++) {
        pixel_type_w v = data[*pos];
        v += (pixel_type_w)data[total_pixels + *pos] << 8;
        v += (pixel_type_w)data[2 * total_pixels + *pos] << 16;
        v += (pixel_type_w)data[3 * total_pixels + *pos] << 24;
        (*pos)++;
        r[x] = UnpackSigned(v);
      }
    }
  } else if (no_wp) {  // special optimized case: no weighted predictor
    const intptr_t onerow = channel->plane.PixelsPerRow();
    for (size_t y = 0; y < channel->h; y++) {
      pixel_type *JXL_RESTRICT r = channel->Row(y);
      Predictor predictor = (Predictor)data[*subpred_pos];
      (*subpred_pos)++;
      for (size_t x = 0; x < channel->w; x++) {
        pixel_type_w g =
            PredictNoTreeNoWP(*channel, r + x, onerow, x, y, predictor).guess;
        pixel_type_w v = data[*pos];
        v += (pixel_type_w)data[total_pixels + *pos] << 8;
        v += (pixel_type_w)data[2 * total_pixels + *pos] << 16;
        v += (pixel_type_w)data[3 * total_pixels + *pos] << 24;
        (*pos)++;
        r[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), g);
      }
    }
  } else {
    const intptr_t onerow = channel->plane.PixelsPerRow();
    weighted::State wp_state(wp_header, channel->w, channel->h);
    for (size_t y = 0; y < channel->h; y++) {
      pixel_type *JXL_RESTRICT r = channel->Row(y);
      Predictor predictor = (Predictor)data[*subpred_pos];
      (*subpred_pos)++;
      for (size_t x = 0; x < channel->w; x++) {
        pixel_type_w g =
            PredictNoTreeWP(*channel, r + x, onerow, x, y, predictor, &wp_state)
                .guess;
        pixel_type_w v = data[*pos];
        v += (pixel_type_w)data[total_pixels + *pos] << 8;
        v += (pixel_type_w)data[2 * total_pixels + *pos] << 16;
        v += (pixel_type_w)data[3 * total_pixels + *pos] << 24;
        (*pos)++;
        r[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), g);
        wp_state.UpdateErrors(r[x], x, y, channel->w);
      }
    }
  }

  return true;
}

GroupHeader::GroupHeader() { Bundle::Init(this); }

constexpr bool kPrintTree = false;

void PrintTree(const Tree &tree, size_t cur, FILE *f) {
  if (tree[cur].property < 0) {
    fprintf(f, "n%05zu [label=\"%s%+" PRId64 "\"];\n", cur,
            PredictorName(tree[cur].predictor), tree[cur].predictor_offset);
  } else {
    fprintf(f, "n%05zu [label=\"%s>%d\"];\n", cur,
            PropertyName(tree[cur].property).c_str(), tree[cur].splitval);
    fprintf(f, "n%05zu -- n%05d;\n", cur, tree[cur].childID);
    fprintf(f, "n%05zu -- n%05d;\n", cur, tree[cur].childID + 1);
    PrintTree(tree, tree[cur].childID, f);
    PrintTree(tree, tree[cur].childID + 1, f);
  }
}

void PrintTree(const Tree &tree, const std::string &path) {
  if (!kPrintTree) return;
  FILE *f = fopen((path + ".dot").c_str(), "w");
  fprintf(f, "graph{\n");
  PrintTree(tree, 0, f);
  fprintf(f, "}\n");
  fclose(f);
  JXL_ASSERT(
      system(("dot " + path + ".dot -T svg -o " + path + ".svg").c_str()) == 0);
}

bool ModularEncode(const Image &image, const ModularOptions &options,
                   BitWriter *writer, AuxOut *aux_out, size_t layer,
                   size_t group_id, std::vector<std::vector<int32_t>> *props,
                   std::vector<std::vector<int32_t>> *residuals,
                   size_t *total_pixels, const Tree *tree, GroupHeader *header,
                   std::vector<Token> *tokens, bool want_debug) {
  if (image.error) return JXL_FAILURE("Invalid image");
  size_t nb_channels = image.real_nb_channels;
  int bit_depth = 1, maxval = 1;
  while (maxval < image.maxval) {
    bit_depth++;
    maxval = maxval * 2 + 1;
  }
  JXL_DEBUG_V(2, "Encoding %zu-channel, %i-bit, %zux%zu image.", nb_channels,
              bit_depth, image.w, image.h);

  if (nb_channels < 1) {
    return true;  // is there any use for a zero-channel image?
  }

  // encode transforms
  GroupHeader header_storage;
  if (header == nullptr) header = &header_storage;
  Bundle::Init(header);
  if (options.predictor == Predictor::Weighted) {
    weighted::PredictorMode(options.wp_mode, &header->wp_header);
  }
  header->max_properties = options.max_properties;
  header->use_brotli =
      options.entropy_coder == ModularOptions::EntropyCoder::kBrotli;
  header->transforms = image.transform;
  // This doesn't actually work
  if (tree != nullptr) {
    header->use_global_tree = true;
  }
  if (props == nullptr && tree == nullptr) {
    JXL_RETURN_IF_ERROR(Bundle::Write(*header, writer, layer, aux_out));
  }

  nb_channels = image.channel.size();

  if (header->use_brotli) {
    JXL_ASSERT(!props && !residuals && !tree && !tokens);
    size_t total_pixels = 0;
    size_t total_height = 0;
    for (size_t i = options.skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      total_pixels += image.channel[i].w * image.channel[i].h;
      total_height += image.channel[i].h;
    }
    if (total_pixels == 0) return true;

    PaddedBytes data(4 * total_pixels + total_height);
    size_t subpred_pos = 0;
    size_t pos = total_height;

    for (size_t i = options.skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      JXL_RETURN_IF_ERROR(EncodeModularChannelBrotli(
          image, i, header->wp_header, options.predictor, total_pixels, &pos,
          &subpred_pos, &data));
    }
    writer->ZeroPadToByte();

    PaddedBytes cbuffer;
    JXL_RETURN_IF_ERROR(BrotliCompress(options.brotli_effort, data, &cbuffer));
    if (aux_out) {
      aux_out->layers[layer].total_bits += cbuffer.size() * kBitsPerByte;
    }
    (*writer) += cbuffer;
    return true;
  }

  std::vector<Predictor> predictors;
  if (options.predictor == Predictor::Variable) {
    predictors.resize(kNumModularPredictors);
    for (size_t i = 0; i < kNumModularPredictors; i++) {
      predictors[i] = static_cast<Predictor>(i);
    }
  } else if (options.predictor == Predictor::Best) {
    predictors = {Predictor::Gradient, Predictor::Weighted};
  } else {
    predictors = {options.predictor};
  }

  std::vector<std::vector<int32_t>> props_storage;
  std::vector<std::vector<int32_t>> residuals_storage;
  size_t total_pixels_storage = 0;
  if (!total_pixels) total_pixels = &total_pixels_storage;
  // If there's no tree, compute one (or gather data to).
  if (tree == nullptr) {
    JXL_ASSERT((props == nullptr) == (residuals == nullptr));
    bool gather_data = props != nullptr;
    for (size_t i = options.skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      GatherTreeData(image, i, group_id, header->wp_header, predictors, options,
                     gather_data ? *props : props_storage,
                     gather_data ? *residuals : residuals_storage,
                     total_pixels);
    }
    if (gather_data) return true;
  }

  JXL_ASSERT((tree == nullptr) == (tokens == nullptr));

  Tree tree_storage;
  std::vector<std::vector<Token>> tokens_storage(1);
  // Compute tree.
  if (tree == nullptr) {
    EntropyEncodingData code;
    std::vector<uint8_t> context_map;

    std::vector<std::vector<Token>> tree_tokens(1);
    tree_storage =
        LearnTree(predictors, std::move(props_storage),
                  std::move(residuals_storage), *total_pixels, options);
    tree = &tree_storage;
    tokens = &tokens_storage[0];

    Tree decoded_tree;
    TokenizeTree(*tree, &tree_tokens[0], &decoded_tree);
    JXL_ASSERT(tree->size() == decoded_tree.size());
    tree_storage = std::move(decoded_tree);

    if (want_debug && WantDebugOutput(aux_out)) {
      PrintTree(*tree,
                aux_out->debug_prefix + "/tree_" + std::to_string(group_id));
    }
    // Write tree
    BuildAndEncodeHistograms(HistogramParams(), kNumTreeContexts, tree_tokens,
                             &code, &context_map, writer, kLayerModularTree,
                             aux_out);
    WriteTokens(tree_tokens[0], code, context_map, writer, kLayerModularTree,
                aux_out);
  }

  for (size_t i = options.skipchannels; i < nb_channels; i++) {
    if (!image.channel[i].w || !image.channel[i].h) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    JXL_RETURN_IF_ERROR(
        EncodeModularChannelMAANS(image, i, header->wp_header, *tree, options,
                                  tokens, aux_out, group_id, want_debug));
  }

  // Write data if not using a global tree/ANS stream.
  if (!header->use_global_tree) {
    EntropyEncodingData code;
    std::vector<uint8_t> context_map;
    BuildAndEncodeHistograms(HistogramParams(), (tree->size() + 1) / 2,
                             tokens_storage, &code, &context_map, writer, layer,
                             aux_out);
    WriteTokens(tokens_storage[0], code, context_map, writer, layer, aux_out);
  }
  return true;
}

Status ModularDecode(BitReader *br, Image &image, size_t group_id,
                     ModularOptions *options, const Tree *global_tree,
                     const ANSCode *global_code,
                     const std::vector<uint8_t> *global_ctx_map) {
  if (image.nb_channels < 1) return true;

  // decode transforms
  GroupHeader header;
  JXL_RETURN_IF_ERROR(Bundle::Read(br, &header));
  options->max_properties = header.max_properties;
  JXL_DEBUG_V(4, "Global option: up to %i back-referencing MA properties.",
              options->max_properties);
  JXL_DEBUG_V(3, "Image data underwent %zu transformations: ",
              header.transforms.size());
  image.transform = header.transforms;
  for (Transform &transform : image.transform) {
    JXL_RETURN_IF_ERROR(transform.MetaApply(image));
  }
  if (options->identify) return true;
  if (image.error) {
    return JXL_FAILURE("Corrupt file. Aborting.");
  }

  size_t nb_channels = image.channel.size();

  size_t num_chans = 0;
  for (size_t i = options->skipchannels; i < nb_channels; i++) {
    if (!image.channel[i].w || !image.channel[i].h) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options->max_chan_size ||
         image.channel[i].h > options->max_chan_size)) {
      break;
    }
    num_chans++;
  }
  if (num_chans == 0) return true;

  if (header.use_brotli) {
    size_t total_pixels = 0;
    size_t total_height = 0;
    for (size_t i = options->skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options->max_chan_size ||
           image.channel[i].h > options->max_chan_size)) {
        break;
      }
      total_pixels += image.channel[i].w * image.channel[i].h;
      total_height += image.channel[i].h;
    }

    JXL_RETURN_IF_ERROR(br->JumpToByteBoundary());

    size_t data_size = 4 * total_pixels + total_height;
    PaddedBytes data;
    size_t read_size = 0;
    JXL_RETURN_IF_ERROR(br->AllReadsWithinBounds());
    auto span = br->GetSpan();
    bool decodestatus = BrotliDecompress(span, data_size, &read_size, &data);
    br->SkipBits(kBitsPerByte * read_size);
    JXL_DEBUG_V(4, "   Decoded %zu bytes for %zu pixels", read_size,
                total_pixels);

    if (!decodestatus) {
      return JXL_FAILURE("Problem during Brotli decode");
    }
    if (data_size != data.size()) {
      return JXL_FAILURE("Invalid decoded data size");
    }
    size_t subpred_pos = 0;
    size_t pos = total_height;

    for (size_t i = options->skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options->max_chan_size ||
           image.channel[i].h > options->max_chan_size)) {
        break;
      }
      JXL_RETURN_IF_ERROR(DecodeModularChannelBrotli(
          data, header.wp_header, total_pixels, &pos, &subpred_pos, &image, i));
    }
    return true;
  }

  // Read tree.
  Tree tree_storage;
  std::vector<uint8_t> context_map_storage;
  ANSCode code_storage;
  const Tree *tree = &tree_storage;
  const ANSCode *code = &code_storage;
  const std::vector<uint8_t> *context_map = &context_map_storage;
  if (!header.use_global_tree) {
    std::vector<uint8_t> tree_context_map;
    ANSCode tree_code;
    JXL_RETURN_IF_ERROR(
        DecodeHistograms(br, kNumTreeContexts, &tree_code, &tree_context_map));
    ANSSymbolReader reader(&tree_code, br);
    JXL_RETURN_IF_ERROR(DecodeTree(br, &reader, tree_context_map, &tree_storage,
                                   NumProperties(image, *options)));
    if (!reader.CheckANSFinalState()) {
      return JXL_FAILURE("ANS decode final state failed");
    }
    JXL_RETURN_IF_ERROR(DecodeHistograms(br, (tree_storage.size() + 1) / 2,
                                         &code_storage, &context_map_storage));
  } else {
    if (!global_tree || !global_code || !global_ctx_map ||
        global_tree->empty()) {
      return JXL_FAILURE("No global tree available but one was requested");
    }
    tree = global_tree;
    code = global_code;
    context_map = global_ctx_map;
    for (size_t i = 0; i < tree->size(); i++) {
      if ((*tree)[i].property >= (int)NumProperties(image, *options)) {
        return JXL_FAILURE("Invalid property ID used in tree");
      }
    }
  }

  // Read channels
  ANSSymbolReader reader(code, br);
  for (size_t i = options->skipchannels; i < nb_channels; i++) {
    Channel &channel = image.channel[i];
    if (!channel.w || !channel.h) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels && (channel.w > options->max_chan_size ||
                                        channel.h > options->max_chan_size)) {
      break;
    }
    JXL_RETURN_IF_ERROR(DecodeModularChannelMAANS(
        br, &reader, *context_map, *tree, header.wp_header, *options, i,
        group_id, &image));
  }
  if (!reader.CheckANSFinalState()) {
    return JXL_FAILURE("ANS decode final state failed");
  }
  return true;
}

bool ModularGenericCompress(Image &image, const ModularOptions &opts,
                            BitWriter *writer, AuxOut *aux_out, size_t layer,
                            size_t group_id,
                            std::vector<std::vector<int32_t>> *props,
                            std::vector<std::vector<int32_t>> *residuals,
                            size_t *total_pixels, const Tree *tree,
                            GroupHeader *header, std::vector<Token> *tokens,
                            bool want_debug) {
  if (image.w == 0 || image.h == 0) return true;
  ModularOptions options = opts;  // Make a copy to modify it.

  if (options.predictor == static_cast<Predictor>(-1)) {
    options.predictor = Predictor::Gradient;
  }

  size_t bits = writer ? writer->BitsWritten() : 0;
  JXL_RETURN_IF_ERROR(ModularEncode(image, options, writer, aux_out, layer,
                                    group_id, props, residuals, total_pixels,
                                    tree, header, tokens, want_debug));
  bits = writer ? writer->BitsWritten() - bits : 0;
  if (writer) {
    JXL_DEBUG_V(
        4, "Modular-encoded a %zux%zu maxval=%i nbchans=%zu image in %zu bytes",
        image.w, image.h, image.maxval, image.real_nb_channels, bits / 8);
  }
  (void)bits;
  return true;
}

bool ModularGenericDecompress(BitReader *br, Image &image, size_t group_id,
                              ModularOptions *options, int undo_transforms,
                              const Tree *tree, const ANSCode *code,
                              const std::vector<uint8_t> *ctx_map) {
  JXL_RETURN_IF_ERROR(
      ModularDecode(br, image, group_id, options, tree, code, ctx_map));
  image.undo_transforms(undo_transforms);
  size_t bit_pos = br->TotalBitsConsumed();
  JXL_DEBUG_V(4, "Modular-decoded a %zux%zu nbchans=%zu image from %zu bytes",
              image.w, image.h, image.real_nb_channels,
              (br->TotalBitsConsumed() - bit_pos) / 8);
  (void)bit_pos;
  return true;
}

}  // namespace jxl
