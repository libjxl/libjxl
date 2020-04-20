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
#include "jxl/modular/encoding/options.h"
#include "jxl/modular/encoding/weighted_predict.h"
#include "jxl/modular/transform/transform.h"
#include "jxl/toc.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/modular/encoding/encoding.cpp"
#include <hwy/foreach_target.h>

namespace jxl {

#ifndef JXL_MODULAR_ENCODING_ENCODING
#define JXL_MODULAR_ENCODING_ENCODING

struct ChannelHeader {
  ChannelHeader() { Bundle::Init(this); }

  static const char *Name() { return "ChannelHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor *JXL_RESTRICT visitor) {
    visitor->Bool(false, &trivial_channel);
    if (visitor->Conditional(!trivial_channel)) {
      visitor->Bits(3, (uint32_t)Predictor::Gradient, &predictor);
    }
    visitor->Bool(true, &sign2lsb);
    if (visitor->Conditional(!sign2lsb || trivial_channel)) {
      uint32_t minval_packed = PackSigned(minv);
      visitor->U32(Val(0), BitsOffset(9, 1), BitsOffset(24, 513),
                   BitsOffset(32, 16777729), 0, &minval_packed);
      minv = UnpackSigned(minval_packed);
    }
    return true;
  }

  bool trivial_channel;  // otherwise use ANS.
  uint32_t predictor;    // 0-7
  bool sign2lsb;         // otherwise subtract minv.
  int32_t minv = 0;      // trivial_channel or !sign2lsb.
};

#ifdef HAS_ENCODER
Predictor find_best(const Channel &channel, const pixel_type *JXL_RESTRICT p,
                    intptr_t onerow, size_t y, Predictor prev_predictor) {
  uint64_t best = 0;
  uint64_t prev = -1;
  int best_predictor = 0;
  for (uint32_t i = 0; i < 6; i++) {
    uint64_t sum_of_abs_residuals = 0;
    for (size_t x = 0; x < channel.w; x++) {
      pixel_type guess = predict(channel, p + x, onerow, x, y, (Predictor)i);
      sum_of_abs_residuals += abs(p[x] - guess);
    }
    if (i == 0 || sum_of_abs_residuals < best) {
      best = sum_of_abs_residuals;
      best_predictor = i;
    }
    if ((Predictor)i == prev_predictor) prev = sum_of_abs_residuals;
  }
  // only change predictor if residuals are 10% smaller
  if (prev < best * 1.1) return prev_predictor;
  return (Predictor)best_predictor;
}
#endif  // HAS_ENCODER

#endif  // JXL_MODULAR_ENCODING_ENCODING

#include <hwy/begin_target-inl.h>

// Position in the table corresponding to the given properties.
HWY_ATTR size_t CompactTreePosition(const CompactTree::ChunkData &cd,
                                    const Properties &props) {
  int32_t values[CompactTree::kChunkPropertyLimit] = {};
  for (size_t i = 0; i < CompactTree::kChunkPropertyLimit; i++) {
    values[i] = props[cd.properties[i]];
  }
  size_t res = 0;
  HWY_CAPPED(int32_t, CompactTree::kChunkPropertyLimit) di;
  for (size_t i = 0; i < CompactTree::kChunkPropertyLimit; i += di.N) {
    const auto vals = LoadU(di, values + i);
    const auto thres = LoadU(di, cd.thresholds + i);
    res |= BitsFromMask(vals > thres) << i;
  }
  return res + cd.start;
}

class MATreeLookup {
 public:
  explicit MATreeLookup(const Tree &treeIn) : inner_nodes_(treeIn) {
    has_compact_tree_ = CompactifyTree(inner_nodes_, &compact_inner_nodes_);
    nb_contexts_ = 0;
    for (size_t i = 0; i < inner_nodes_.size(); i++) {
      if (inner_nodes_[i].property < 0) nb_contexts_++;
    }
  }
  HWY_ATTR int context_id(const Properties &properties) const ATTRIBUTE_HOT {
    if (has_compact_tree_) {
      int64_t pos = 0;
      do {
        size_t table_idx =
            CompactTreePosition(compact_inner_nodes_.chunks[pos], properties);
        pos = compact_inner_nodes_.table[table_idx];
      } while (pos > 0);
      return -pos;
    }
    // Fallback case for very peculiar trees.
    Tree::size_type pos = 0;
    while (true) {
      const PropertyDecisionNode &node = inner_nodes_[pos];
      if (node.property < 0) {
        return node.childID;
      }
      if (properties[node.property] > node.splitval) {
        pos = node.childID;
      } else {
        pos = node.childID + 1;
      }
    }
  }
  size_t nb_contexts() const { return nb_contexts_; }

 private:
  const Tree &inner_nodes_;
  CompactTree compact_inner_nodes_;
  bool has_compact_tree_ = false;
  size_t nb_contexts_;
};

#ifdef HAS_ENCODER
HWY_ATTR Status EncodeModularChannelMAANS(const Image &image, size_t chan,
                                          const ChannelHeader &header,
                                          const Tree &tree,
                                          const ModularOptions &options,
                                          const HybridUintConfig &uint_config,
                                          size_t base_ctx,
                                          std::vector<Token> *tokens) {
  const Channel &channel = image.channel[chan];

  JXL_ASSERT(channel.w != 0 && channel.h != 0);

  if (header.trivial_channel) return true;

  JXL_DEBUG_V(6,
              "Encoding %zux%zu channel %zu (predictor %i), "
              "(shift=%i,%i, cshift=%i,%i)",
              channel.w, channel.h, chan, header.predictor, channel.hshift,
              channel.vshift, channel.hcshift, channel.vcshift);

  Predictor predictor = (Predictor)header.predictor;
  Predictor subpredictor = predictor;
  Properties properties(num_properties(image, chan, chan, options));
  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Encoding using a MA tree with %zu nodes (%zu contexts)",
              tree.size(), tree_lookup.nb_contexts());
  int nbctx = tree_lookup.nb_contexts();

  const intptr_t onerow = channel.plane.PixelsPerRow();
  Channel references(properties.size() - NB_NONREF_PROPERTIES, channel.w);
  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    precompute_references(channel, y, image, chan, options, references);
    if (predictor == Predictor::Variable) {
      subpredictor = find_best(channel, p, onerow, y, subpredictor);
      TokenizeWithConfig(uint_config, base_ctx + nbctx, (uint32_t)subpredictor,
                         tokens);
    }
    for (size_t x = 0; x < channel.w; x++) {
      pixel_type guess =
          predict_and_compute_properties_with_precomputed_reference(
              properties, channel, p + x, onerow, x, y, subpredictor, image,
              chan, references);
      int ctx = tree_lookup.context_id(properties);
      if (header.sign2lsb) {
        TokenizeWithConfig(uint_config, base_ctx + ctx,
                           PackSigned(p[x] - guess), tokens);
      } else {
        TokenizeWithConfig(uint_config, base_ctx + ctx, p[x] - header.minv,
                           tokens);
      }
    }
  }
  return true;
}

#endif  // HAS_ENCODER

HWY_ATTR Status DecodeModularChannelMAANS(
    BitReader *br, ANSSymbolReader *reader,
    const std::vector<uint8_t> &context_map, const Tree &tree,
    const ChannelHeader &header, const ModularOptions &options, size_t base_ctx,
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
  JXL_DEBUG_V(3, "Decoded MA tree with %zu nodes (%zu contexts)", tree.size(),
              tree_lookup.nb_contexts());
  Properties properties(num_properties(*image, chan, chan, options));

  // MAANS decode
  int nbctx = tree_lookup.nb_contexts();

  Predictor predictor = (Predictor)header.predictor;
  Predictor subpredictor = predictor;

  if (tree.size() == 1 && predictor == Predictor::Zero) {
    // special optimized case: no meta-adaptation, no predictor, so no need
    // to compute properties
    JXL_DEBUG_V(8, "Fast track.");
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT r = channel.Row(y);
      if (header.sign2lsb) {
        for (size_t x = 0; x < channel.w; x++) {
          uint32_t v = reader->ReadHybridUint(base_ctx, br, context_map);
          r[x] = UnpackSigned(v);
        }
      } else {
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type v = reader->ReadHybridUint(base_ctx, br, context_map);
          v += header.minv;
          r[x] = v;
        }
      }
    }
  } else if (tree.size() == 1) {
    // special optimized case: no meta-adaptation, so no need to compute
    // properties
    JXL_DEBUG_V(8, "Quite fast track.");
    const intptr_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT r = channel.Row(y);
      if (predictor == Predictor::Variable) {
        subpredictor = (Predictor)reader->ReadHybridUint(base_ctx + nbctx, br,
                                                         context_map);
      }
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type g = predict(channel, r + x, onerow, x, y, subpredictor);
        uint32_t v = reader->ReadHybridUint(base_ctx, br, context_map);
        r[x] = UnpackSigned(v) + g;
      }
    }
  } else {
    JXL_DEBUG_V(8, "Slow track.");
    const intptr_t onerow = channel.plane.PixelsPerRow();
    Channel references(properties.size() - NB_NONREF_PROPERTIES, channel.w);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT p = channel.Row(y);
      precompute_references(channel, y, *image, chan, options, references);
      if (predictor == Predictor::Variable) {
        subpredictor = (Predictor)reader->ReadHybridUint(base_ctx + nbctx, br,
                                                         context_map);
      }
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type guess =
            predict_and_compute_properties_with_precomputed_reference(
                properties, channel, p + x, onerow, x, y, subpredictor, *image,
                chan, references);
        int ctx = tree_lookup.context_id(properties);
        uint32_t v = reader->ReadHybridUint(base_ctx + ctx, br, context_map);
        if (header.sign2lsb) {
          p[x] = UnpackSigned(v) + guess;
        } else {
          p[x] = header.minv + v;
        }
      }
    }
  }
  return true;
}

#include <hwy/end_target-inl.h>

#if HWY_ONCE
HWY_EXPORT(EncodeModularChannelMAANS)
HWY_EXPORT(DecodeModularChannelMAANS)

#ifdef HAS_ENCODER

ChannelHeader ComputeChannelHeader(const Channel &channel,
                                   Predictor predictor) {
  ChannelHeader header;
  header.predictor = (uint32_t)predictor;

  header.sign2lsb = true;

  if (channel.is_trivial) {
    header.trivial_channel = true;
    pixel_type minv;
    pixel_type maxv;
    channel.compute_minmax(&minv, &maxv);
    header.minv = minv;
  } else if (predictor == Predictor::Zero) {
    pixel_type minv;
    pixel_type maxv;
    channel.compute_minmax(&minv, &maxv);
    header.sign2lsb = !(maxv <= 0 || minv >= 0);
    header.minv = minv;
  }
  return header;
}

Tree LearnTree(const Image &image, size_t chan, Predictor predictor,
               const ModularOptions &options) {
  if (options.nb_repeats <= 0) return {};
  const Channel &channel = image.channel[chan];

  JXL_DEBUG_V(7, "Learning %zux%zu channel %zu (predictor %i)", channel.w,
              channel.h, chan, predictor);

  ChannelHeader header = ComputeChannelHeader(channel, predictor);

  Predictor subpredictor = predictor;
  Properties properties(num_properties(image, chan, chan, options));
  std::mt19937_64 gen(1);  // deterministic learning (also between threads)
  std::vector<size_t> ys(channel.h);
  std::iota(ys.begin(), ys.end(), 0);
  std::shuffle(ys.begin(), ys.end(), gen);
  float pixel_fraction = std::min(1.0f, options.nb_repeats);
  ys.resize(std::ceil(ys.size() * pixel_fraction));

  const intptr_t onerow = channel.plane.PixelsPerRow();
  Channel references(properties.size() - NB_NONREF_PROPERTIES, channel.w);
  std::vector<std::vector<int32_t>> data;  // 0 -> token, 1...: properties.
  data.resize(properties.size() + 1);
  std::vector<std::pair<int, int>> prop_range(properties.size());
  for (size_t i = 0; i < properties.size(); i++) {
    prop_range[i].first = std::numeric_limits<int>::max();
    prop_range[i].second = std::numeric_limits<int>::min();
  }
  for (size_t y : ys) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    precompute_references(channel, y, image, chan, options, references);
    if (predictor == Predictor::Variable) {
      subpredictor = find_best(channel, p, onerow, y, subpredictor);
    }
    for (size_t x = 0; x < channel.w; x++) {
      pixel_type guess =
          predict_and_compute_properties_with_precomputed_reference(
              properties, channel, p + x, onerow, x, y, subpredictor, image,
              chan, references);
      pixel_type diff =
          header.sign2lsb ? PackSigned(p[x] - guess) : p[x] - header.minv;
      uint32_t token, nbits, bits;
      EncodeHybridVarLenUint(diff, &token, &nbits, &bits);
      data[0].push_back(token);
      for (size_t i = 0; i < properties.size(); i++) {
        data[i + 1].push_back(properties[i]);
        prop_range[i].first = std::min(prop_range[i].first, properties[i]);
        prop_range[i].second = std::max(prop_range[i].second, properties[i]);
      }
    }
  }
  std::vector<size_t> props_to_use;
  std::vector<int> multiplicity;
  std::vector<std::vector<int>> compact_properties(data.size() - 1);
  // TODO(veluca): add an option for max total number of property values.
  ChooseAndQuantizeProperties(options.splitting_heuristics_max_properties,
                              options.splitting_heuristics_max_properties * 256,
                              &data, &multiplicity, &compact_properties,
                              &props_to_use);
  Tree tree;
  ComputeBestTree(data, multiplicity, compact_properties, props_to_use,
                  options.splitting_heuristics_node_threshold * pixel_fraction,
                  options.splitting_heuristics_max_properties, &tree);
  return tree;
}

Status EncodeModularChannelBrotli(const Channel &channel,
                                  const ChannelHeader &header,
                                  size_t total_pixels, size_t *JXL_RESTRICT pos,
                                  size_t *JXL_RESTRICT subpred_pos,
                                  PaddedBytes *JXL_RESTRICT data) {
  JXL_ASSERT(channel.w != 0 && channel.h != 0);
  if (header.trivial_channel) return true;
  Predictor subpredictor = (Predictor)header.predictor;
  const intptr_t onerow = channel.plane.PixelsPerRow();
  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    if (header.predictor == (uint32_t)Predictor::Variable) {
      subpredictor = find_best(channel, p, onerow, y, subpredictor);
      (*data)[*subpred_pos] = (uint8_t)subpredictor;
      (*subpred_pos)++;
    }
    for (size_t x = 0; x < channel.w; x++) {
      pixel_type guess = predict(channel, p + x, onerow, x, y, subpredictor);
      pixel_type v;
      if (header.sign2lsb) {
        v = PackSigned(p[x] - guess);
      } else {
        v = p[x] - header.minv;
      }
      (*data)[*pos] = (v & 0xff);
      (*data)[*pos + total_pixels] = ((v >> 8) & 0xff);
      (*data)[*pos + 2 * total_pixels] = ((v >> 16) & 0xff);
      (*data)[*pos + 3 * total_pixels] = ((v >> 24) & 0xff);
      (*pos)++;
    }
  }
  return true;
}

#endif  // HAS_ENCODER

Status DecodeModularChannelBrotli(const PaddedBytes &data,
                                  const ChannelHeader &header,
                                  size_t total_pixels, size_t *JXL_RESTRICT pos,
                                  size_t *JXL_RESTRICT subpred_pos,
                                  Channel *channel) {
  JXL_ASSERT(channel->w != 0 && channel->h != 0);
  if (header.trivial_channel) {
    channel->plane = Plane<pixel_type>();
    channel->resize(header.minv);  // fill it with the constant value
    return true;
  }

  channel->resize(channel->w, channel->h);

  Predictor predictor = (Predictor)header.predictor;
  Predictor subpredictor = predictor;

  if (predictor == Predictor::Zero) {
    // special optimized case: no predictor
    JXL_DEBUG_V(8, "Fast track.");
    for (size_t y = 0; y < channel->h; y++) {
      pixel_type *JXL_RESTRICT r = channel->Row(y);
      if (header.sign2lsb) {
        for (size_t x = 0; x < channel->w; x++) {
          pixel_type v = data[*pos];
          v += data[total_pixels + *pos] << 8;
          v += data[2 * total_pixels + *pos] << 16;
          v += data[3 * total_pixels + *pos] << 24;
          (*pos)++;
          r[x] = UnpackSigned(v);
        }
      } else {
        for (size_t x = 0; x < channel->w; x++) {
          pixel_type v = data[*pos];
          v += data[total_pixels + *pos] << 8;
          v += data[2 * total_pixels + *pos] << 16;
          v += data[3 * total_pixels + *pos] << 24;
          (*pos)++;
          v += header.minv;
          r[x] = v;
        }
      }
    }
  } else {
    const intptr_t onerow = channel->plane.PixelsPerRow();
    for (size_t y = 0; y < channel->h; y++) {
      pixel_type *JXL_RESTRICT r = channel->Row(y);
      if (predictor == Predictor::Variable) {
        subpredictor = (Predictor)data[*subpred_pos];
        (*subpred_pos)++;
      }
      for (size_t x = 0; x < channel->w; x++) {
        pixel_type g = predict(*channel, r + x, onerow, x, y, subpredictor);
        pixel_type v = data[*pos];
        v += data[total_pixels + *pos] << 8;
        v += data[2 * total_pixels + *pos] << 16;
        v += data[3 * total_pixels + *pos] << 24;
        (*pos)++;
        r[x] = UnpackSigned(v) + g;
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
#ifdef HAS_ENCODER
bool modular_encode(const Image &image, const ModularOptions &options,
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
  for (int p : options.predictor) {
    predictors.push_back((Predictor)p);
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
      headers[i] = ComputeChannelHeader(image.channel[i],
                                        predictors[i] == Predictor::Best
                                            ? Predictor::Gradient
                                            : predictors[i]);
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
      if (headers[i].predictor == (uint32_t)Predictor::Variable) {
        total_height += image.channel[i].h;
      }
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
      JXL_RETURN_IF_ERROR(
          EncodeModularChannelBrotli(image.channel[i], headers[i], total_pixels,
                                     &pos, &subpred_pos, &data));
    }
    writer->ZeroPadToByte();

    PaddedBytes cbuffer;
    JXL_RETURN_IF_ERROR(BrotliCompress(options.brotli_effort, data, &cbuffer));
    if (aux_out) {
      aux_out->layers[layer].total_bits += cbuffer.size() * kBitsPerByte;
    }
    (*writer) += std::move(cbuffer);
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
    headers[i] = ComputeChannelHeader(
        image.channel[i],
        predictors[i] == Predictor::Best ? Predictor::Gradient : predictors[i]);
  }

  std::vector<std::vector<Token>> tokens(1);
  std::vector<std::vector<Token>> tree_tokens(1);
  size_t base_tree_ctx = 0;
  size_t base_ctx = 0;

  std::vector<WeightedPredictorHeader> weighted_headers(nb_channels);
  size_t num_chans = 0;
  const auto encode_maans =
      ChooseEncodeModularChannelMAANS(hwy::SupportedTargets());
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
    size_t tree_start = tree_tokens[0].size();

    std::vector<Token> *local_tokens = &tokens[0];
    std::vector<Token> local_tokens_storage;
    std::vector<Token> *weighted_tokens = &tokens[0];
    std::vector<Token> weighted_tokens_storage;

    Predictor learn_predictor = predictor;
    if (predictor == Predictor::Best) {
      local_tokens = &local_tokens_storage;
      weighted_tokens = &weighted_tokens_storage;
      learn_predictor = Predictor::Gradient;
    }

    // MA encode
    Tree tree;
    if (predictor != Predictor::Weighted) {
      tree = LearnTree(image, i, learn_predictor, options);
      Tree decoded_tree;
      TokenizeTree(tree, uint_config, base_tree_ctx, &tree_tokens[0],
                   &decoded_tree);
      JXL_ASSERT(tree.size() == decoded_tree.size());
      tree = std::move(decoded_tree);
      JXL_RETURN_IF_ERROR(encode_maans(image, i, headers[i], tree, options,
                                       uint_config, base_ctx, local_tokens));
    }
    // Weighted encode
    if (predictor == Predictor::Weighted || predictor == Predictor::Best) {
      WeightedPredictorState state(image.channel[i].w, image.channel[i].h);
      JXL_RETURN_IF_ERROR(state.wp_compress(
          image.channel[i], options.nb_wp_modes, base_ctx, uint_config,
          weighted_tokens, &weighted_headers[i]));
    }
    // Choose best
    if (predictor == Predictor::Best) {
      if (TokenCost(local_tokens_storage) <
          TokenCost(weighted_tokens_storage)) {
        tokens[0].insert(tokens[0].end(), local_tokens_storage.begin(),
                         local_tokens_storage.end());
      } else {
        tokens[0].insert(tokens[0].end(), weighted_tokens_storage.begin(),
                         weighted_tokens_storage.end());
        headers[i].predictor = (uint32_t)Predictor::Weighted;
        // Remove tokenized tree.
        tree_tokens[0].resize(tree_start, Token(0, 0, 0, 0));
      }
    }

    // Advance context starts
    if (headers[i].predictor == (uint32_t)Predictor::Weighted) {
      base_ctx += kNumContexts;
    } else {
      base_ctx += (tree.size() + 1) / 2;
      if (headers[i].predictor == (uint32_t)Predictor::Variable) {
        base_ctx++;
      }
      base_tree_ctx += kNumTreeContexts;
    }
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
    if (headers[i].trivial_channel) continue;
    if (headers[i].predictor == (uint32_t)Predictor::Weighted) {
      JXL_RETURN_IF_ERROR(
          Bundle::Write(weighted_headers[i], writer, layer, aux_out));
    }
  }
  // Write trees
  if (base_tree_ctx != 0) {
    EntropyEncodingData codes;
    std::vector<uint8_t> context_map;
    BuildAndEncodeHistograms(HistogramParams(), base_tree_ctx, tree_tokens,
                             &codes, &context_map, writer, layer, aux_out);
    WriteTokens(tree_tokens[0], codes, context_map, writer, layer, aux_out,
                uint_config);
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
#endif

Status modular_decode(BitReader *br, Image &image, ModularOptions *options) {
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
  std::vector<WeightedPredictorHeader> weighted_headers(nb_channels);
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
    if (headers[i].predictor == (uint32_t)Predictor::Weighted) {
      JXL_RETURN_IF_ERROR(Bundle::Read(br, &weighted_headers[i]));
    } else {
      num_trees++;
    }
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
      if ((Predictor)headers[i].predictor == Predictor::Variable) {
        total_height += image.channel[i].h;
      }
    }

    JXL_RETURN_IF_ERROR(br->JumpToByteBoundary());

    size_t data_size = 4 * total_pixels + total_height;
    PaddedBytes data;
    size_t read_size = 0;
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
      JXL_RETURN_IF_ERROR(
          DecodeModularChannelBrotli(data, headers[i], total_pixels, &pos,
                                     &subpred_pos, &image.channel[i]));
    }
    return true;
  }

  // Read trees and compute starting contexts for each channel.
  size_t num_ctxs = 0;
  std::vector<Tree> trees(nb_channels);
  std::vector<uint32_t> base_context(nb_channels);
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
      if (headers[i].predictor != (uint32_t)Predictor::Weighted) {
        JXL_RETURN_IF_ERROR(
            DecodeTree(br, &reader, context_map, base_ctx, &trees[i]));
        num_ctxs += (trees[i].size() + 1) / 2;
        base_ctx += kNumTreeContexts;
        if (headers[i].predictor == (uint32_t)Predictor::Variable) {
          num_ctxs++;
        }
      } else {
        num_ctxs += kNumContexts;
      }
    }
    if (!reader.CheckANSFinalState()) {
      return JXL_FAILURE("ANS decode final state failed");
    }
  }
  if (num_trees == 0) {
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
      num_ctxs += kNumContexts;
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
  const auto decode_maans =
      ChooseDecodeModularChannelMAANS(hwy::SupportedTargets());
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
      channel.resize(headers[i].minv);  // fill it with the constant value
      continue;
    }
    if (headers[i].predictor == (uint32_t)Predictor::Weighted) {
      WeightedPredictorState state(channel.w, channel.h);
      JXL_RETURN_IF_ERROR(state.wp_decompress(br, &reader, context_map,
                                              base_context[i],
                                              weighted_headers[i], &channel));
    } else {
      JXL_RETURN_IF_ERROR(decode_maans(br, &reader, context_map, trees[i],
                                       headers[i], *options, base_context[i], i,
                                       &image));
    }
  }
  if (!reader.CheckANSFinalState()) {
    return JXL_FAILURE("ANS decode final state failed");
  }
  return true;
}

#ifdef HAS_ENCODER

void modular_prepare_encode(Image &image, ModularOptions &options) {
  // ensure that the ranges are correct and tight
  image.recompute_minmax();
}
#endif

#ifdef HAS_ENCODER
bool modular_generic_compress(Image &image, const ModularOptions &opts,
                              BitWriter *writer, AuxOut *aux_out, size_t layer,
                              int loss, bool try_transforms,
                              const HybridUintConfig &uint_config) {
  if (image.w == 0 || image.h == 0) return true;
  ModularOptions options = opts;  // Make a copy to modify it.

  if (options.predictor.empty())
    options.predictor.push_back(5);  // use predictor 5 by default

  if (try_transforms) {
    if (loss > 1) {
      // lossy DC
      image.do_transform(Transform(TransformId::kSqueeze));
      Transform quantize(TransformId::kQuantize);
      for (size_t i = image.nb_meta_channels; i < image.channel.size(); i++) {
        Channel &ch = image.channel[i];
        int shift = ch.hcshift + ch.vcshift;  // number of pixel halvings
        int q;
        q = (loss >> shift);
        if (q < 1) q = 1;
        quantize.parameters.push_back(q);
      }
      image.do_transform(quantize);
    } else if (!options.skipchannels) {
      // simple heuristic: if less than 90 percent of the values in the range
      // actually occur, it is probably worth it to do a compaction
      // unless there are few pixels to encode, then the overhead is not worth
      // it

      for (size_t c = 0; c < image.nb_channels; c++) {
        Transform maybe_palette_1(TransformId::kPalette);
        maybe_palette_1.parameters.push_back(c + image.nb_meta_channels);
        maybe_palette_1.parameters.push_back(c + image.nb_meta_channels);
        int minv, maxv;
        image.channel[c + image.nb_meta_channels].compute_minmax(&minv, &maxv);
        int colors = maxv - minv + 1;
        float factor = 1.0;
        if (image.w * image.h < 1000) {
          factor =
              static_cast<float>(image.w) * static_cast<float>(image.h) / 1000;
        }
        maybe_palette_1.parameters.push_back((int)(0.9 * factor * colors));
        if (image.do_transform(maybe_palette_1)) {
          options.predictor.insert(options.predictor.begin(),
                                   1);  // left predictor for palette encoding
        }
      }
    }
    image.recompute_minmax();
  }

  size_t bits = writer->BitsWritten();
  modular_prepare_encode(image, options);
  JXL_RETURN_IF_ERROR(
      modular_encode(image, options, uint_config, writer, aux_out, layer));
  bits = writer->BitsWritten() - bits;
  JXL_DEBUG_V(
      4, "Modular-encoded a %zux%zu maxval=%i nbchans=%zu image in %zu bytes",
      image.w, image.h, image.maxval, image.real_nb_channels, bits / 8);
  (void)bits;
  return true;
}
#endif

bool modular_generic_decompress(BitReader *br, Image &image,
                                ModularOptions *options, int undo_transforms) {
  JXL_RETURN_IF_ERROR(modular_decode(br, image, options));
  image.undo_transforms(undo_transforms);
  size_t bit_pos = br->TotalBitsConsumed();
  JXL_DEBUG_V(4, "Modular-decoded a %zux%zu nbchans=%zu image from %zu bytes",
              image.w, image.h, image.real_nb_channels,
              (br->TotalBitsConsumed() - bit_pos) / 8);
  (void)bit_pos;
  return true;
}

#endif  // HWY_ONCE

}  // namespace jxl
