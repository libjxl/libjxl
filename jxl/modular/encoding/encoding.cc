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

struct ChannelHeader {
  ChannelHeader() { Bundle::Init(this); }

  static const char *Name() { return "ChannelHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor *JXL_RESTRICT visitor) {
    visitor->Bool(false, &trivial_channel);
    if (visitor->Conditional(trivial_channel)) {
      uint32_t minval_packed = PackSigned(minv);
      visitor->U32(Val(0), BitsOffset(9, 1), BitsOffset(24, 513),
                   BitsOffset(32, 16777729), 0, &minval_packed);
      minv = UnpackSigned(minval_packed);
    }
    if (visitor->Conditional(!trivial_channel)) {
      JXL_RETURN_IF_ERROR(visitor->VisitNested(&wp_header));
    }
    return true;
  }

  bool trivial_channel;
  int32_t minv = 0;            // trivial_channel.
  weighted::Header wp_header;  // !trivial_channel.
};

// Tries all the predictors, excluding Weighted, per row.
Predictor FindBest(const Image &image, size_t chan,
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

Status EncodeModularChannelMAANS(const Image &image, size_t chan,
                                 const ChannelHeader &header, const Tree &tree,
                                 const ModularOptions &options,
                                 const HybridUintConfig &uint_config,
                                 size_t base_ctx, std::vector<Token> *tokens) {
  const Channel &channel = image.channel[chan];

  JXL_ASSERT(channel.w != 0 && channel.h != 0);

  if (header.trivial_channel) return true;

  JXL_DEBUG_V(6,
              "Encoding %zux%zu channel %zu, "
              "(shift=%i,%i, cshift=%i,%i)",
              channel.w, channel.h, chan, channel.hshift, channel.vshift,
              channel.hcshift, channel.vcshift);

  Properties properties(NumProperties(image, chan, chan, options));
  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Encoding using a MA tree with %zu nodes", tree.size());

  const intptr_t onerow = channel.plane.PixelsPerRow();
  Channel references(properties.size() - kNumNonrefProperties, channel.w);
  weighted::State wp_state(header.wp_header, channel.w, channel.h);
  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    PrecomputeReferences(channel, y, image, chan, options, &references);
    for (size_t x = 0; x < channel.w; x++) {
      PredictionResult res =
          PredictTreeWP(&properties, channel, p + x, onerow, x, y, tree_lookup,
                        references, &wp_state);
      TokenizeWithConfig(uint_config, base_ctx + res.context,
                         PackSigned(p[x] - res.guess), tokens);
      wp_state.UpdateErrors(p[x], x, y, channel.w);
    }
  }
  return true;
}

Status DecodeModularChannelMAANS(BitReader *br, ANSSymbolReader *reader,
                                 const std::vector<uint8_t> &context_map,
                                 const Tree &tree, const ChannelHeader &header,
                                 const ModularOptions &options, size_t base_ctx,
                                 size_t chan, Image *image) {
  Channel &channel = image->channel[chan];

  // zero pixel channel? could happen
  if (channel.w == 0 || channel.h == 0) return true;

  if (header.trivial_channel) {
    channel.plane = Plane<pixel_type>();
    channel.resize(header.minv);  // fill it with the constant value
    return true;
  }

  channel.resize(channel.w, channel.h);

  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Decoded MA tree with %zu nodes", tree.size());
  Properties properties(NumProperties(*image, chan, chan, options));

  // MAANS decode
  bool tree_has_wp_prop_or_pred = false;
  for (size_t i = 0; i < tree.size(); i++) {
    if (tree[i].property < 0) {
      if (tree[i].predictor == Predictor::Weighted) {
        tree_has_wp_prop_or_pred = true;
      }
    } else if (tree[i].property >=
               properties.size() - weighted::kNumProperties) {
      tree_has_wp_prop_or_pred = true;
    }
  }

  if (tree.size() == 1) {
    // special optimized case: no meta-adaptation, so no need
    // to compute properties.
    Predictor predictor = tree[0].predictor;
    int64_t offset = tree[0].predictor_offset;
    if (predictor == Predictor::Zero) {
      JXL_DEBUG_V(8, "Fast track.");
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        for (size_t x = 0; x < channel.w; x++) {
          uint32_t v = reader->ReadHybridUint(base_ctx, br, context_map);
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
          uint64_t v = reader->ReadHybridUint(base_ctx, br, context_map);
          r[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), g);
        }
      }
    } else {
      // special optimized case: no meta-adaptation, so no need to
      // compute properties
      JXL_DEBUG_V(8, "Somewhat fast track.");
      const intptr_t onerow = channel.plane.PixelsPerRow();
      weighted::State wp_state(header.wp_header, channel.w, channel.h);
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type_w g = PredictNoTreeWP(channel, r + x, onerow, x, y,
                                           predictor, &wp_state)
                               .guess +
                           offset;
          uint64_t v = reader->ReadHybridUint(base_ctx, br, context_map);
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
        PredictionResult res = PredictTreeNoWP(
            &properties, channel, p + x, onerow, x, y, tree_lookup, references);
        uint64_t v =
            reader->ReadHybridUint(base_ctx + res.context, br, context_map);
        p[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), res.guess);
      }
    }
  } else {
    JXL_DEBUG_V(8, "Slowest track.");
    const intptr_t onerow = channel.plane.PixelsPerRow();
    Channel references(properties.size() - kNumNonrefProperties, channel.w);
    weighted::State wp_state(header.wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, *image, chan, options, &references);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeWP(&properties, channel, p + x, onerow, x, y,
                          tree_lookup, references, &wp_state);
        uint64_t v =
            reader->ReadHybridUint(base_ctx + res.context, br, context_map);
        p[x] = SaturatingAdd<pixel_type>(UnpackSigned(v), res.guess);
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
    }
  }
  return true;
}

ChannelHeader ComputeChannelHeader(const Channel &channel) {
  ChannelHeader header;

  if (channel.is_trivial) {
    header.trivial_channel = true;
    pixel_type minv;
    pixel_type maxv;
    channel.compute_minmax(&minv, &maxv);
    header.minv = minv;
  }
  return header;
}

Tree LearnTree(const Image &image, size_t chan, const ChannelHeader &header,
               Predictor predictor, const ModularOptions &options,
               const HybridUintConfig &uint_config) {
  if (options.nb_repeats <= 0) {
    Tree t;
    t.push_back(PropertyDecisionNode{});
    return t;
  }
  const Channel &channel = image.channel[chan];

  JXL_DEBUG_V(7, "Learning %zux%zu channel %zu (predictor %i)", channel.w,
              channel.h, chan, predictor);

  Properties properties(NumProperties(image, chan, chan, options));
  std::mt19937_64 gen(1);  // deterministic learning (also between threads)
  float pixel_fraction = std::min(1.0f, options.nb_repeats);
  pixel_fraction = std::max(pixel_fraction,
                            std::min(1.0f, 1024.0f / (channel.w * channel.h)));
  std::bernoulli_distribution dist(pixel_fraction);

  const intptr_t onerow = channel.plane.PixelsPerRow();
  Channel references(properties.size() - kNumNonrefProperties, channel.w);
  weighted::State wp_state(header.wp_header, channel.w, channel.h);
  std::vector<std::vector<int32_t>> props(properties.size());
  std::vector<Predictor> predictors;
  if (predictor == Predictor::Variable) {
    predictors.resize(kNumModularPredictors);
    for (size_t i = 0; i < kNumModularPredictors; i++) {
      predictors[i] = static_cast<Predictor>(i);
    }
  } else {
    predictors = {predictor};
  }
  std::vector<std::vector<int32_t>> residuals(predictors.size());
  std::vector<std::pair<int, int>> prop_range(properties.size());
  for (size_t i = 0; i < properties.size(); i++) {
    prop_range[i].first = std::numeric_limits<int>::max();
    prop_range[i].second = std::numeric_limits<int>::min();
  }
  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    PrecomputeReferences(channel, y, image, chan, options, &references);
    for (size_t x = 0; x < channel.w; x++) {
      pixel_type_w res[kNumModularPredictors];
      if (predictor == Predictor::Variable) {
        PredictLearnAll(&properties, channel, p + x, onerow, x, y, references,
                        &wp_state, res);
        for (size_t i = 0; i < kNumModularPredictors; i++) {
          res[i] = p[x] - res[i];
        }
      } else {
        PredictionResult pres =
            PredictLearn(&properties, channel, p + x, onerow, x, y, predictor,
                         references, &wp_state);
        res[0] = p[x] - pres.guess;
      }
      if (dist(gen)) {
        for (size_t i = 0; i < predictors.size(); i++) {
          residuals[i].push_back(res[i]);
        }
        for (size_t i = 0; i < properties.size(); i++) {
          props[i].push_back(properties[i]);
          prop_range[i].first = std::min(prop_range[i].first, properties[i]);
          prop_range[i].second = std::max(prop_range[i].second, properties[i]);
        }
      }
      wp_state.UpdateErrors(p[x], x, y, channel.w);
    }
  }
  int64_t offset = 0;
  if (predictor == Predictor::Variable) {
    int base_pred = 0;
    size_t base_pred_cost = 0;
    for (size_t i = 0; i < kNumModularPredictors; i++) {
      int64_t sum = 0;
      for (size_t j = 0; j < residuals[i].size(); j++) {
        sum += residuals[i][j];
      }
      int64_t tot = residuals.size();
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
  std::vector<size_t> props_to_use;
  std::vector<std::vector<int>> compact_properties(properties.size());
  // TODO(veluca): add an option for max total number of property values.
  ChooseAndQuantizeProperties(options.splitting_heuristics_max_properties,
                              options.splitting_heuristics_max_properties * 256,
                              uint_config, offset, residuals, &props,
                              &compact_properties, &props_to_use);
  Tree tree;
  ComputeBestTree(residuals, props, predictors, uint_config, offset,
                  compact_properties, props_to_use,
                  options.splitting_heuristics_node_threshold * pixel_fraction,
                  options.splitting_heuristics_max_properties, &tree);
  return tree;
}

Status EncodeModularChannelBrotli(const Image &image, size_t chan,
                                  const ChannelHeader &header,
                                  Predictor predictor, size_t total_pixels,
                                  size_t *JXL_RESTRICT pos,
                                  size_t *JXL_RESTRICT subpred_pos,
                                  PaddedBytes *JXL_RESTRICT data) {
  const Channel &channel = image.channel[chan];
  JXL_ASSERT(channel.w != 0 && channel.h != 0);
  if (header.trivial_channel) return true;
  Predictor subpredictor = Predictor::Gradient;
  const intptr_t onerow = channel.plane.PixelsPerRow();
  weighted::State wp_state(header.wp_header, channel.w, channel.h);
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
                                  const ChannelHeader &header,
                                  size_t total_pixels, size_t *JXL_RESTRICT pos,
                                  size_t *JXL_RESTRICT subpred_pos,
                                  Image *image, size_t chan) {
  Channel *channel = &image->channel[chan];
  JXL_ASSERT(channel->w != 0 && channel->h != 0);
  if (header.trivial_channel) {
    channel->plane = Plane<pixel_type>();
    channel->resize(header.minv);  // fill it with the constant value
    return true;
  }

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
    weighted::State wp_state(header.wp_header, channel->w, channel->h);
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

namespace {
struct GroupHeader {
  GroupHeader() { Bundle::Init(this); }

  static const char *Name() { return "GroupHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor *JXL_RESTRICT visitor) {
    visitor->Bool(false, &use_brotli);
    visitor->U32(Val(0), Val(1), Val(2), BitsOffset(4, 3), 0, &max_properties);
    uint32_t num_transforms = transforms.size();
    visitor->U32(Val(0), Val(1), BitsOffset(4, 2), BitsOffset(8, 18), 0,
                 &num_transforms);
    if (visitor->IsReading()) transforms.resize(num_transforms);
    for (size_t i = 0; i < num_transforms; i++) {
      visitor->Bits(3, 0, &transforms[i].id);
      if (transforms[i].id >= (uint32_t)TransformId::kNumTransforms) {
        return JXL_FAILURE("Invalid transform id: %u", transforms[i].id);
      }
      if (transforms[i].id == (uint32_t)TransformId::kNearLossless) {
        return JXL_FAILURE("Invalid transform id: %u", transforms[i].id);
      }
      uint32_t num_params = transforms[i].params.size();
      visitor->U32(Val(0), Val(1), Val(3), BitsOffset(7, 2), 0, &num_params);
      if (visitor->IsReading()) transforms[i].params.resize(num_params);
      for (size_t j = 0; j < num_params; j++) {
        visitor->U32(Bits(2), BitsOffset(3, 4), BitsOffset(8, 12),
                     BitsOffset(16, 268), 1, &transforms[i].params[j]);
      }
    }
    return true;
  }

  bool use_brotli;
  uint32_t max_properties;

  struct Transform {
    uint32_t id;
    std::vector<uint32_t> params;
  };
  std::vector<Transform> transforms;
};

}  // namespace

bool ModularEncode(const Image &image, const ModularOptions &options,
                   const HybridUintConfig &uint_config, BitWriter *writer,
                   AuxOut *aux_out, size_t layer) {
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
  GroupHeader header;
  header.max_properties = options.max_properties;
  header.use_brotli =
      options.entropy_coder == ModularOptions::EntropyCoder::kBrotli;
  header.transforms.resize(image.transform.size());
  for (size_t i = 0; i < header.transforms.size(); i++) {
    header.transforms[i].id = (uint32_t)image.transform[i].id;
    header.transforms[i].params.assign(image.transform[i].parameters.begin(),
                                       image.transform[i].parameters.end());
  }
  JXL_RETURN_IF_ERROR(Bundle::Write(header, writer, layer, aux_out));

  nb_channels = image.channel.size();

  // Predictors to use for each channel.
  std::vector<Predictor> predictors;
  for (Predictor p : options.predictor) {
    predictors.push_back(p);
  }
  // if nothing at all is specified, use Gradient,
  // seems to be the best general-purpose predictor
  if (predictors.empty()) {
    predictors.push_back(Predictor::Gradient);
  }
  if (predictors.size() < nb_channels) {
    predictors.resize(nb_channels, predictors.back());
  }

  if (header.use_brotli) {
    std::vector<ChannelHeader> headers(nb_channels);

    for (size_t i = options.skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      headers[i] = ComputeChannelHeader(image.channel[i]);
      JXL_RETURN_IF_ERROR(Bundle::Write(headers[i], writer, layer, aux_out));
    }

    size_t total_pixels = 0;
    size_t total_height = 0;
    for (size_t i = options.skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h ||
          headers[i].trivial_channel) {
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
          image, i, headers[i],
          predictors[i] == Predictor::Best ? Predictor::Gradient
                                           : predictors[i],
          total_pixels, &pos, &subpred_pos, &data));
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

  std::vector<ChannelHeader> headers(nb_channels);
  for (size_t i = options.skipchannels; i < nb_channels; i++) {
    if (!image.channel[i].w || !image.channel[i].h) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    headers[i] = ComputeChannelHeader(image.channel[i]);
  }

  std::vector<std::vector<Token>> tokens(1);
  std::vector<std::vector<Token>> tree_tokens(1);
  size_t base_tree_ctx = 0;
  size_t base_ctx = 0;

  size_t num_chans = 0;
  for (size_t i = options.skipchannels; i < nb_channels; i++) {
    if (!image.channel[i].w || !image.channel[i].h ||
        headers[i].trivial_channel) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    num_chans++;
    Predictor predictor = predictors[i];

    std::vector<Token> best_tree_tokens, best_tokens;
    bool has_best = false;
    float best_cost = 0;
    size_t num_ctxs = 0;
    auto try_mode = [&](Predictor pred, size_t wp_mode) {
      std::vector<Token> local_tree_tokens, local_tokens;
      Tree tree;
      ChannelHeader header = ComputeChannelHeader(image.channel[i]);
      if (pred == Predictor::Weighted) {
        weighted::PredictorMode(wp_mode, &header.wp_header);
      }
      tree = LearnTree(image, i, header, pred, options, uint_config);
      Tree decoded_tree;
      TokenizeTree(tree, uint_config, base_tree_ctx, &local_tree_tokens,
                   &decoded_tree);
      JXL_ASSERT(tree.size() == decoded_tree.size());
      tree = std::move(decoded_tree);
      JXL_RETURN_IF_ERROR(EncodeModularChannelMAANS(image, i, header, tree,
                                                    options, uint_config,
                                                    base_ctx, &local_tokens));
      size_t extension_bits, total_bits;
      JXL_RETURN_IF_ERROR(
          Bundle::CanEncode(header, &extension_bits, &total_bits));
      float cost =
          TokenCost(local_tree_tokens) + TokenCost(local_tokens) + total_bits;
      if (!has_best || best_cost > cost) {
        best_tree_tokens = local_tree_tokens;
        best_tokens = local_tokens;
        best_cost = cost;
        has_best = true;
        num_ctxs = (tree.size() + 1) / 2;
        headers[i] = header;
      }
      return true;
    };
    if (predictor != Predictor::Weighted && predictor != Predictor::Best) {
      JXL_RETURN_IF_ERROR(try_mode(predictor, 0));
    }
    if (predictor == Predictor::Best) {
      JXL_RETURN_IF_ERROR(try_mode(Predictor::Gradient, 0));
    }
    if (predictor == Predictor::Weighted || predictor == Predictor::Best) {
      for (size_t wp_mode = 0; wp_mode < options.nb_wp_modes; wp_mode++) {
        JXL_RETURN_IF_ERROR(try_mode(Predictor::Weighted, wp_mode));
      }
    }
    if (!has_best) {
      return JXL_FAILURE("Could not compress channel");
    }
    tree_tokens[0].insert(tree_tokens[0].end(), best_tree_tokens.begin(),
                          best_tree_tokens.end());
    tokens[0].insert(tokens[0].end(), best_tokens.begin(), best_tokens.end());

    // Advance context starts
    base_ctx += num_ctxs;
    base_tree_ctx += kNumTreeContexts;
  }

  // Write headers
  for (size_t i = options.skipchannels; i < nb_channels; i++) {
    if (!image.channel[i].w || !image.channel[i].h) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    JXL_RETURN_IF_ERROR(Bundle::Write(headers[i], writer, layer, aux_out));
  }
  // Write trees
  if (base_tree_ctx != 0) {
    EntropyEncodingData codes;
    std::vector<uint8_t> context_map;
    BuildAndEncodeHistograms(HistogramParams(), base_tree_ctx, tree_tokens,
                             &codes, &context_map, writer, kLayerModularTree,
                             aux_out);
    WriteTokens(tree_tokens[0], codes, context_map, writer, kLayerModularTree,
                aux_out, uint_config);
  }
  if (base_ctx == 0) return true;
  // Write data
  EntropyEncodingData codes;
  std::vector<uint8_t> context_map;
  BuildAndEncodeHistograms(HistogramParams(), base_ctx, tokens, &codes,
                           &context_map, writer, layer, aux_out);
  WriteTokens(tokens[0], codes, context_map, writer, layer, aux_out,
              uint_config);
  return true;
}

Status ModularDecode(BitReader *br, Image &image, ModularOptions *options) {
  if (image.nb_channels < 1) return true;

  // decode transforms
  GroupHeader header;
  JXL_RETURN_IF_ERROR(Bundle::Read(br, &header));
  options->max_properties = header.max_properties;
  JXL_DEBUG_V(4, "Global option: up to %i back-referencing MA properties.",
              options->max_properties);
  JXL_DEBUG_V(3, "Image data underwent %zu transformations: ",
              header.transforms.size());
  for (size_t i = 0; i < header.transforms.size(); i++) {
    image.transform.emplace_back((TransformId)header.transforms[i].id);
    image.transform[i].parameters.assign(header.transforms[i].params.begin(),
                                         header.transforms[i].params.end());
    JXL_RETURN_IF_ERROR(image.transform[i].MetaApply(image));
  }
  if (options->identify) return true;
  if (image.error) {
    return JXL_FAILURE("Corrupt file. Aborting.");
  }

  size_t nb_channels = image.channel.size();

  size_t num_trees = 0;

  std::vector<ChannelHeader> headers(nb_channels);
  for (size_t i = options->skipchannels; i < nb_channels; i++) {
    if (!image.channel[i].w || !image.channel[i].h) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options->max_chan_size ||
         image.channel[i].h > options->max_chan_size)) {
      break;
    }
    JXL_RETURN_IF_ERROR(Bundle::Read(br, &headers[i]));
    if (headers[i].trivial_channel) continue;
    num_trees++;
  }

  if (header.use_brotli) {
    size_t total_pixels = 0;
    size_t total_height = 0;
    for (size_t i = options->skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h ||
          headers[i].trivial_channel) {
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
          data, headers[i], total_pixels, &pos, &subpred_pos, &image, i));
    }
    return true;
  }

  // Read trees and compute starting contexts for each channel.
  size_t num_ctxs = 0;
  std::vector<Tree> trees(nb_channels);
  std::vector<uint32_t> base_context(nb_channels);
  // If no trees, all channels are trivial.
  if (num_trees != 0) {
    std::vector<uint8_t> context_map;
    ANSCode code;
    JXL_RETURN_IF_ERROR(DecodeHistograms(br, num_trees * kNumTreeContexts,
                                         ANS_MAX_ALPHA_SIZE, &code,
                                         &context_map));
    ANSSymbolReader reader(&code, br);
    size_t base_ctx = 0;
    for (size_t i = options->skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h ||
          headers[i].trivial_channel) {
        continue;  // skip empty channels
      }
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options->max_chan_size ||
           image.channel[i].h > options->max_chan_size)) {
        break;
      }
      base_context[i] = num_ctxs;
      JXL_RETURN_IF_ERROR(DecodeTree(br, &reader, context_map, base_ctx,
                                     &trees[i],
                                     NumProperties(image, i, i, *options)));
      num_ctxs += (trees[i].size() + 1) / 2;
      base_ctx += kNumTreeContexts;
    }
    if (!reader.CheckANSFinalState()) {
      return JXL_FAILURE("ANS decode final state failed");
    }
  }

  // Trivial image.
  if (num_ctxs == 0) {
    for (size_t i = options->skipchannels; i < nb_channels; i++) {
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options->max_chan_size ||
           image.channel[i].h > options->max_chan_size)) {
        break;
      }
      Channel &channel = image.channel[i];
      if (headers[i].trivial_channel) {
        channel.plane = Plane<pixel_type>();
        channel.is_trivial = true;
        channel.resize(headers[i].minv);  // fill it with the constant value
      }
    }
    return true;
  }
  // Read channels
  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(
      DecodeHistograms(br, num_ctxs, ANS_MAX_ALPHA_SIZE, &code, &context_map));
  ANSSymbolReader reader(&code, br);
  for (size_t i = options->skipchannels; i < nb_channels; i++) {
    Channel &channel = image.channel[i];
    if (!channel.w || !channel.h) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels && (channel.w > options->max_chan_size ||
                                        channel.h > options->max_chan_size)) {
      break;
    }
    if (headers[i].trivial_channel) {
      channel.plane = Plane<pixel_type>();
      channel.is_trivial = true;
      channel.resize(headers[i].minv);  // fill it with the constant value
      continue;
    }
    JXL_RETURN_IF_ERROR(DecodeModularChannelMAANS(
        br, &reader, context_map, trees[i], headers[i], *options,
        base_context[i], i, &image));
  }
  if (!reader.CheckANSFinalState()) {
    return JXL_FAILURE("ANS decode final state failed");
  }
  return true;
}

bool ModularGenericCompress(Image &image, const ModularOptions &opts,
                            BitWriter *writer, AuxOut *aux_out, size_t layer,
                            const HybridUintConfig &uint_config) {
  if (image.w == 0 || image.h == 0) return true;
  ModularOptions options = opts;  // Make a copy to modify it.

  if (options.predictor.empty()) {
    options.predictor.push_back(Predictor::Gradient);
  }

  size_t bits = writer->BitsWritten();
  image.recompute_minmax();
  JXL_RETURN_IF_ERROR(
      ModularEncode(image, options, uint_config, writer, aux_out, layer));
  bits = writer->BitsWritten() - bits;
  JXL_DEBUG_V(
      4, "Modular-encoded a %zux%zu maxval=%i nbchans=%zu image in %zu bytes",
      image.w, image.h, image.maxval, image.real_nb_channels, bits / 8);
  (void)bits;
  return true;
}

bool ModularGenericDecompress(BitReader *br, Image &image,
                              ModularOptions *options, int undo_transforms) {
  JXL_RETURN_IF_ERROR(ModularDecode(br, image, options));
  image.undo_transforms(undo_transforms);
  size_t bit_pos = br->TotalBitsConsumed();
  JXL_DEBUG_V(4, "Modular-decoded a %zux%zu nbchans=%zu image from %zu bytes",
              image.w, image.h, image.real_nb_channels,
              (br->TotalBitsConsumed() - bit_pos) / 8);
  (void)bit_pos;
  return true;
}

}  // namespace jxl
