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

#include "jxl/enc_modular.h"

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <utility>
#include <vector>

#include "jxl/aux_out.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/status.h"
#include "jxl/dec_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_params.h"
#include "jxl/modular/encoding/encoding.h"
#include "jxl/modular/image/image.h"
#include "jxl/modular/options.h"
#include "jxl/modular/transform/transform.h"
#include "jxl/toc.h"

namespace jxl {

namespace {
// Squeeze default quantization factors
// these quantization factors are for -Q 50  (other qualities simply scale the
// factors; things are rounded down and obviously cannot get below 1)
static const float squeeze_quality_factor =
    0.3;  // for easy tweaking of the quality range (decrease this number for
          // higher quality)
static const float squeeze_luma_factor =
    1.2;  // for easy tweaking of the balance between luma (or anything
          // non-chroma) and chroma (decrease this number for higher quality
          // luma)

static const float squeeze_luma_qtable[16] = {
    163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28,
    0.64,   0.32,  0.16,  0.08,  0.04,  0.02, 0.01, 0.005};
// for 8-bit input, the range of YCoCg chroma is -255..255 so basically this
// does 4:2:0 subsampling (two most fine grained layers get quantized away)
static const float squeeze_chroma_qtable[16] = {
    1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.5, 0.5, 0.5, 0.5};

// XYB multipliers. TODO: signal them.
// Also may want to use chroma_from_luma instead of doing B-Y here...
static const float kEncoderMul2[3] = {32768.0f, 2048.0f, 2048.0f};
}  // namespace

Status ModularFrameEncoder::ComputeEncodingData(
    CompressParams cparams, const FrameHeader& frame_header,
    const ImageBundle& ib, Image3F* JXL_RESTRICT color,
    PassesEncoderState* JXL_RESTRICT enc_state, ThreadPool* pool,
    AuxOut* aux_out, bool do_color) {
  const FrameDimensions& frame_dim = enc_state->shared.frame_dim;

  if (cparams.options.entropy_coder == ModularOptions::kMAANS) {
    if (cparams.speed_tier > SpeedTier::kWombat) {
      cparams.options.splitting_heuristics_node_threshold = 192;
    } else {
      cparams.options.splitting_heuristics_node_threshold = 96;
    }
    switch (cparams.speed_tier) {
      case SpeedTier::kWombat:
        cparams.options.splitting_heuristics_max_properties = 4;
        break;
      case SpeedTier::kSquirrel:
        cparams.options.splitting_heuristics_max_properties = 6;
        break;
      case SpeedTier::kKitten:
        cparams.options.splitting_heuristics_max_properties = 8;
        break;
      case SpeedTier::kTortoise:
        cparams.options.splitting_heuristics_max_properties = 128;
        break;
      default:
        cparams.options.splitting_heuristics_max_properties = 4;
        break;
    }
  }

  float quality = cparams.quality_pair.first;
  float cquality = cparams.quality_pair.second;
  if (cquality > 100) cquality = quality;

  // use a sensible default if nothing explicit is specified:
  // Squeeze for lossy, no squeeze for lossless
  if (cparams.responsive < 0) {
    if (quality == 100) {
      cparams.responsive = 0;
    } else {
      cparams.responsive = 1;
    }
  }

  if (cparams.options.predictor == static_cast<Predictor>(-1)) {
    // no explicit predictor(s) given, set a good default
    if (cparams.speed_tier < SpeedTier::kTortoise) {
      cparams.options.predictor = Predictor::Variable;
    } else if (cparams.near_lossless) {
      // avg(top,left) predictor for near_lossless
      cparams.options.predictor = Predictor::Average;
    } else if (cparams.responsive) {
      // zero predictor for Squeeze residues
      cparams.options.predictor = Predictor::Zero;
    } else if (cparams.speed_tier < SpeedTier::kFalcon) {
      // try median and weighted predictor for anything else
      cparams.options.predictor = Predictor::Best;
    } else {
      // just weighted predictor in fastest mode
      cparams.options.predictor = Predictor::Weighted;
    }
  }

  const bool has_ac_global = !frame_header.IsJpeg();
  size_t num_groups =
      (has_ac_global ? 2 : 1) + frame_dim.num_dc_groups +
      enc_state->shared.multiframe->GetNumPasses() * frame_dim.num_groups;

  cparams.options.max_chan_size = kGroupDim;
  group_images.resize(num_groups);
  group_options.resize(num_groups, cparams.options);

  if (do_color && cparams.speed_tier < SpeedTier::kCheetah) {
    FindBestPatchDictionary(*color, enc_state, nullptr, nullptr,
                            cparams.color_transform == ColorTransform::kXYB);
    enc_state->shared.image_features.patches.SubtractFrom(color);
  }

  // Convert ImageBundle to modular Image object
  const size_t xsize = ib.xsize();
  const size_t ysize = ib.ysize();

  int nb_chans = 3;
  if (ib.IsGray()) nb_chans = 1;
  if (!do_color) nb_chans = 0;

  if (frame_header.HasAlpha()) nb_chans++;
  if (ib.HasDepth()) nb_chans++;
  if (ib.HasExtraChannels() && frame_header.IsDisplayed()) {
    nb_chans += ib.extra_channels().size();
  }

  if (ib.metadata()->bits_per_sample >= 32) {
    if (ib.metadata()->bits_per_sample == 32) {
      // TODO(lode): does modular support uint32_t? maxval is signed int so
      // cannot represent 32 bits.
      return JXL_FAILURE("uint32_t not supported in dec_modular");
    } else {
      return JXL_FAILURE("bits_per_sample > 32 not supported");
    }
  }

  // TODO(lode): must handle decoded->metadata()->floating_point_channel?
  int maxval =
      (1u << static_cast<uint32_t>(ib.metadata()->bits_per_sample)) - 1;

  // Nothing to do if image is trivial.
  if (nb_chans == 0) {
    for (size_t i = 0; i < num_groups; i++) {
      group_images[i] = Image(xsize, ysize, maxval, 0);
    }
    return true;
  }
  if (cparams.color_transform == ColorTransform::kXYB) {
    maxval = 255;  // not true, but bits_per_sample doesn't matter either
  }
  Image& gi = group_images[0];
  gi = Image(xsize, ysize, maxval, nb_chans);
  int c = 0;
  if (do_color) {
    for (; c < 3; c++) {
      if (ib.IsGray() &&
          c != (cparams.color_transform == ColorTransform::kXYB ? 1 : 0))
        continue;
      int c_out = c;
      // XYB is encoded as YX(B-Y)
      if (cparams.color_transform == ColorTransform::kXYB && c < 2)
        c_out = 1 - c_out;
      float factor = maxval / 255.f;
      if (cparams.color_transform == ColorTransform::kXYB)
        factor *= kEncoderMul2[c];
      if (c == 2 && cparams.color_transform == ColorTransform::kXYB) {
        for (size_t y = 0; y < ysize; ++y) {
          const float* const JXL_RESTRICT row_in = color->PlaneRow(c, y);
          pixel_type* const JXL_RESTRICT row_out = gi.channel[c_out].Row(y);
          pixel_type* const JXL_RESTRICT row_Y = gi.channel[0].Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            row_out[x] = row_in[x] * factor + 0.5f;
            row_out[x] -= row_Y[x];
          }
        }
      } else {
        for (size_t y = 0; y < ysize; ++y) {
          const float* const JXL_RESTRICT row_in = color->PlaneRow(c, y);
          pixel_type* const JXL_RESTRICT row_out = gi.channel[c_out].Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            row_out[x] = row_in[x] * factor + 0.5f;
          }
        }
      }
    }
    if (ib.IsGray()) c = 1;
  }
  if (frame_header.HasAlpha()) {
    for (size_t y = 0; y < ysize; ++y) {
      const uint16_t* const JXL_RESTRICT row_in = ib.alpha().Row(y);
      pixel_type* const JXL_RESTRICT row_out = gi.channel[c].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row_in[x];
      }
    }
    c++;
  }
  if (ib.HasDepth()) {
    gi.channel[c].resize(ib.depth().xsize(), ib.depth().ysize());
    gi.channel[c].hshift = ib.metadata()->m2.depth_shift;
    gi.channel[c].vshift = ib.metadata()->m2.depth_shift;
    for (size_t y = 0; y < ib.depth().ysize(); ++y) {
      const uint16_t* const JXL_RESTRICT row_in = ib.depth().Row(y);
      pixel_type* const JXL_RESTRICT row_out = gi.channel[c].Row(y);
      for (size_t x = 0; x < ib.depth().xsize(); ++x) {
        row_out[x] = row_in[x];
      }
    }
    c++;
  }
  if (ib.HasExtraChannels() && frame_header.IsDisplayed()) {
    for (size_t ec = 0; ec < ib.extra_channels().size(); ec++, c++) {
      for (size_t y = 0; y < ysize; ++y) {
        const uint16_t* const JXL_RESTRICT row_in =
            ib.extra_channels()[ec].Row(y);
        pixel_type* const JXL_RESTRICT row_out = gi.channel[c].Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_out[x] = row_in[x];
        }
      }
    }
  }
  JXL_ASSERT(c == nb_chans);

  // Set options and apply transformations

  if (quality < 100 || cparams.near_lossless) {
    if (cparams.palette_colors != 0) {
      JXL_DEBUG_V(3, "Lossy encode, not doing palette transforms");
    }
    cparams.channel_colors_pre_transform_percent = 0;
    cparams.channel_colors_percent = 0;
    cparams.palette_colors = 0;
  }

  // Global channel palette
  if (cparams.channel_colors_pre_transform_percent > 0 && quality == 100) {
    // single channel palette (like FLIF's ChannelCompact)
    for (size_t i = 0; i < gi.nb_channels; i++) {
      int min, max;
      gi.channel[gi.nb_meta_channels + i].compute_minmax(&min, &max);
      int colors = max - min + 1;
      JXL_DEBUG_V(10, "Channel %zu: range=%i..%i", i, min, max);
      Transform maybe_palette_1(TransformId::kPalette);
      maybe_palette_1.begin_c = i + gi.nb_meta_channels;
      maybe_palette_1.num_c = 1;
      // simple heuristic: if less than X percent of the values in the range
      // actually occur, it is probably worth it to do a compaction
      // (but only if the channel palette is less than 80% the size of the
      // image itself)
      maybe_palette_1.nb_colors = std::min(
          (int)(xsize * ysize * 0.8),
          (int)(cparams.channel_colors_pre_transform_percent / 100. * colors));
      gi.do_transform(maybe_palette_1);
    }
  }

  // Global palette
  if (cparams.palette_colors != 0 && cparams.speed_tier < SpeedTier::kFalcon) {
    // all-channel palette (e.g. RGBA)
    if (gi.nb_channels > 1) {
      Transform maybe_palette(TransformId::kPalette);
      maybe_palette.begin_c = gi.nb_meta_channels;
      maybe_palette.num_c = gi.nb_channels;
      maybe_palette.nb_colors = std::abs(cparams.palette_colors);
      maybe_palette.ordered_palette = cparams.palette_colors >= 0;
      gi.do_transform(maybe_palette);
    }
    // all-minus-one-channel palette (RGB with separate alpha, or CMY with
    // separate K)
    if (gi.nb_channels > 3) {
      Transform maybe_palette_3(TransformId::kPalette);
      maybe_palette_3.begin_c = gi.nb_meta_channels;
      maybe_palette_3.num_c = gi.nb_channels - 1;
      maybe_palette_3.nb_colors = std::abs(cparams.palette_colors);
      maybe_palette_3.ordered_palette = cparams.palette_colors >= 0;
      gi.do_transform(maybe_palette_3);
    }
  }

  if (cparams.color_transform == ColorTransform::kNone && do_color) {
    if (cparams.colorspace == 1 ||
        (cparams.colorspace < 0 && (quality < 100 || cparams.near_lossless ||
                                    cparams.speed_tier > SpeedTier::kWombat))) {
      Transform ycocg{TransformId::kRCT};
      ycocg.rct_type = 6;
      ycocg.begin_c = gi.nb_meta_channels;
      gi.do_transform(ycocg);
    } else if (cparams.colorspace >= 2) {
      Transform sg(TransformId::kRCT);
      sg.begin_c = gi.nb_meta_channels;
      sg.rct_type = cparams.colorspace - 2;
      gi.do_transform(sg);
    }
  }

  if (cparams.responsive) {
    gi.do_transform(Transform(TransformId::kSqueeze));  // use default squeezing
  }
  if (quality < 100 || cquality < 100) {
    JXL_DEBUG_V(
        2,
        "Adding quantization constants corresponding to luma quality %.2f "
        "and chroma quality %.2f",
        quality, cquality);
    if (!cparams.responsive) {
      JXL_DEBUG_V(1,
                  "Warning: lossy compression without Squeeze "
                  "transform is just color quantization.");
      quality = (400 + quality) / 5;
      cquality = (400 + cquality) / 5;
    }
    Transform quantize(TransformId::kQuantize);
    for (size_t i = 0; i < gi.nb_meta_channels; i++) {
      quantize.nonserialized_quant_factors.push_back(
          1);  // don't quantize metachannels
    }

    // convert 'quality' to quantization scaling factor
    if (quality > 50)
      quality = 200.0 - quality * 2.0;
    else
      quality = 900.0 - quality * 16.0;
    if (cquality > 50)
      cquality = 200.0 - cquality * 2.0;
    else
      cquality = 900.0 - cquality * 16.0;
    quality *= 0.01f * maxval / 255.f;
    cquality *= 0.01f * maxval / 255.f;

    for (size_t i = gi.nb_meta_channels; i < gi.channel.size(); i++) {
      Channel& ch = gi.channel[i];
      int shift = ch.hcshift + ch.vcshift;  // number of pixel halvings
      if (shift > 15) shift = 15;
      int q;
      // assuming default Squeeze here
      int component = ((i - gi.nb_meta_channels) % gi.real_nb_channels);
      // last 4 channels are final chroma residuals
      if (gi.real_nb_channels > 2 && i >= gi.channel.size() - 4) component = 1;

      if (cparams.colorspace != 0 && component > 0 && component < 3)
        q = cquality * squeeze_quality_factor * squeeze_chroma_qtable[shift];
      else
        q = quality * squeeze_quality_factor * squeeze_luma_factor *
            squeeze_luma_qtable[shift];
      if (q < 1) q = 1;
      quantize.nonserialized_quant_factors.push_back(q);
    }
    gi.do_transform(quantize);
  }

  // Fill other groups.
  struct GroupParams {
    Rect rect;
    int minShift;
    int maxShift;
    size_t id;
  };
  std::vector<GroupParams> group_params;

  // DC
  for (size_t group_index = 0; group_index < frame_dim.num_dc_groups;
       group_index++) {
    const size_t gx = group_index % frame_dim.xsize_dc_groups;
    const size_t gy = group_index / frame_dim.xsize_dc_groups;
    const Rect rect(gx * kDcGroupDim, gy * kDcGroupDim, kDcGroupDim,
                    kDcGroupDim);
    // minShift==3 because kDcGroupDim>>3 == kGroupDim
    // maxShift==1000 is infinity
    group_params.push_back(GroupParams{rect, 3, 1000, group_index + 1});
  }
  // AC global -> nothing.
  // AC
  for (size_t group_index = 0; group_index < frame_dim.num_groups;
       group_index++) {
    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    const Rect mrect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim);
    int maxShift = 2;
    int minShift = 0;
    for (size_t i = 0; i < enc_state->shared.multiframe->GetNumPasses(); i++) {
      for (uint32_t j = 0; j < frame_header.passes.num_downsample; ++j) {
        if (i <= frame_header.passes.last_pass[j]) {
          if (frame_header.passes.downsample[j] == 8) minShift = 3;
          if (frame_header.passes.downsample[j] == 4) minShift = 2;
          if (frame_header.passes.downsample[j] == 2) minShift = 1;
          if (frame_header.passes.downsample[j] == 1) minShift = 0;
        }
      }
      group_params.push_back(
          GroupParams{mrect, minShift, maxShift,
                      AcGroupIndex(i, group_index, frame_dim.num_groups,
                                   frame_dim.num_dc_groups, has_ac_global)});
      maxShift = minShift - 1;
      minShift = 0;
    }
  }

  if (cparams.options.entropy_coder == ModularOptions::EntropyCoder::kBrotli) {
    for (size_t i = 0; i < group_params.size(); i++) {
      JXL_RETURN_IF_ERROR(PrepareGroupParams(
          group_params[i].rect, cparams, group_params[i].minShift,
          group_params[i].maxShift, group_params[i].id, kHybridUint420Config,
          do_color, /*bit_size=*/nullptr));
    }
    return true;
  }

  HybridUintConfig all_uint_config[2] = {kHybridUint420Config,
                                         HybridUintConfig(4, 2, 1)};
  // Try putting the sign bit in the token only at slower settings.
  size_t num_uint_configs = cparams.speed_tier < SpeedTier::kWombat ? 2 : 1;

  size_t best_cost = std::numeric_limits<size_t>::max();
  for (size_t uc = 0; uc < num_uint_configs; uc++) {
    std::atomic<size_t> total_cost{0};
    RunOnPool(
        pool, 0, group_params.size(), ThreadPool::SkipInit(),
        [&](size_t i, size_t _) {
          size_t cost = 0;
          JXL_CHECK(PrepareGroupParams(
              group_params[i].rect, cparams, group_params[i].minShift,
              group_params[i].maxShift, group_params[i].id, all_uint_config[uc],
              do_color, &cost));
          total_cost += cost;
        },
        "ChooseParams");
    if (total_cost < best_cost) {
      best_cost = total_cost;
      uint_config = all_uint_config[uc];
    }
  }

  // Add dummy params for the global info so that tree learning takes that into
  // account too.
  group_params.push_back(GroupParams{Rect{}, 0, 0, /*id=*/0});

  group_headers.resize(num_groups);
  tokens.resize(num_groups);

  // TODO: parallelize.
  std::vector<std::vector<int32_t>> props;
  std::vector<std::vector<int32_t>> residuals;
  size_t total_pixels = 0;
  for (size_t i = 0; i < group_params.size(); i++) {
    size_t group_id = group_params[i].id;
    JXL_RETURN_IF_ERROR(ModularGenericCompress(
        group_images[group_id], group_options[group_id], /*writer=*/nullptr,
        /*aux_out=*/nullptr, 0, group_id, uint_config, &props, &residuals,
        &total_pixels));
  }

  std::vector<Predictor> predictors;
  if (cparams.options.predictor == Predictor::Variable) {
    predictors.resize(kNumModularPredictors);
    for (size_t i = 0; i < kNumModularPredictors; i++) {
      predictors[i] = static_cast<Predictor>(i);
    }
  } else if (cparams.options.predictor == Predictor::Best) {
    predictors = {Predictor::Gradient, Predictor::Weighted};
  } else {
    predictors = {cparams.options.predictor};
  }

  // TODO(veluca): parallelize.
  tree = LearnTree(predictors, std::move(props), std::move(residuals),
                   total_pixels, cparams.options, uint_config);
  tree_tokens.resize(1);
  Tree decoded_tree;
  TokenizeTree(tree, kHybridUint420Config, &tree_tokens[0], &decoded_tree);
  JXL_ASSERT(tree.size() == decoded_tree.size());
  tree = std::move(decoded_tree);

  RunOnPool(
      pool, 0, group_params.size(), ThreadPool::SkipInit(),
      [&](size_t i, size_t _) {
        AuxOut my_aux_out;
        if (aux_out) {
          my_aux_out.testing_aux = aux_out->testing_aux;
          my_aux_out.dump_image = aux_out->dump_image;
          my_aux_out.debug_prefix = aux_out->debug_prefix;
        }
        size_t group_id = group_params[i].id;
        JXL_CHECK(ModularGenericCompress(
            group_images[group_id], group_options[group_id], /*writer=*/nullptr,
            &my_aux_out, 0, group_id, uint_config,
            /*props=*/nullptr,
            /*residuals=*/nullptr,
            /*total_pixels=*/nullptr,
            /*tree=*/&tree, /*header=*/&group_headers[group_id],
            /*tokens=*/&tokens[group_id],
            /*want_debug=*/true));
      },
      "ComputeTokens");

  return true;
}

Status ModularFrameEncoder::EncodeGlobalInfo(BitWriter* writer,
                                             AuxOut* aux_out) {
  BitWriter::Allotment allotment(writer, 1);
  // If we are using brotli, or not using modular mode.
  if (group_options[0].entropy_coder == ModularOptions::EntropyCoder::kBrotli ||
      tree_tokens.empty()) {
    writer->Write(1, 0);
    ReclaimAndCharge(writer, &allotment, kLayerModularTree, aux_out);
    return true;
  }
  writer->Write(1, 1);
  ReclaimAndCharge(writer, &allotment, kLayerModularTree, aux_out);

  if (WantDebugOutput(aux_out)) {
    PrintTree(tree, aux_out->debug_prefix + "/global_tree");
  }
  // Write tree
  BuildAndEncodeHistograms(HistogramParams(), kNumTreeContexts, tree_tokens,
                           &code, &context_map, writer, kLayerModularTree,
                           aux_out);
  WriteTokens(tree_tokens[0], code, context_map, writer, kLayerModularTree,
              aux_out, kHybridUint420Config);
  // Write histograms.
  BuildAndEncodeHistograms(HistogramParams(), (tree.size() + 1) / 2, tokens,
                           &code, &context_map, writer, kLayerModularGlobal,
                           aux_out);
  return true;
}

Status ModularFrameEncoder::EncodeGroup(BitWriter* writer, AuxOut* aux_out,
                                        size_t layer, size_t group_id) {
  if (group_options[group_id].entropy_coder ==
      ModularOptions::EntropyCoder::kBrotli) {
    return ModularGenericCompress(
        group_images[group_id], group_options[group_id], writer, aux_out, layer,
        group_id, kHybridUint420Config,
        /*props=*/nullptr, /*residuals=*/nullptr, /*total_pixels=*/nullptr,
        /*tree=*/nullptr, /*header=*/nullptr, /*tokens=*/nullptr,
        /*want_debug=*/true);
  }
  if (group_images[group_id].real_nb_channels < 1) {
    return true;  // Image with no channels, header never gets decoded.
  }
  JXL_RETURN_IF_ERROR(
      Bundle::Write(group_headers[group_id], writer, layer, aux_out));
  WriteTokens(tokens[group_id], code, context_map, writer, layer, aux_out,
              uint_config);
  return true;
}

Status ModularFrameEncoder::PrepareGroupParams(
    const Rect& rect, const CompressParams& cparams, int minShift, int maxShift,
    size_t group_id, HybridUintConfig uint_config, bool do_color,
    size_t* bit_size) {
  JXL_ASSERT(group_id != 0);
  Image& full_image = group_images[0];
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  int maxval = full_image.maxval;
  Image& gi = group_images[group_id];
  ModularOptions base_options = group_options[group_id];
  gi = Image(xsize, ysize, maxval, 0);
  // start at the first bigger-than-kGroupDim non-metachannel
  int c = full_image.nb_meta_channels;
  for (; c < full_image.channel.size(); c++) {
    Channel& fc = full_image.channel[c];
    if (fc.w > kGroupDim || fc.h > kGroupDim) break;
  }
  for (; c < full_image.channel.size(); c++) {
    Channel& fc = full_image.channel[c];
    int shift = std::min(fc.hshift, fc.vshift);
    if (shift > maxShift) continue;
    if (shift < minShift) continue;
    Rect r(rect.x0() >> fc.hshift, rect.y0() >> fc.vshift,
           rect.xsize() >> fc.hshift, rect.ysize() >> fc.vshift, fc.w, fc.h);
    if (r.xsize() == 0 || r.ysize() == 0) continue;
    Channel gc(r.xsize(), r.ysize());
    gc.hshift = fc.hshift;
    gc.vshift = fc.vshift;
    for (size_t y = 0; y < r.ysize(); ++y) {
      const pixel_type* const JXL_RESTRICT row_in = r.ConstRow(fc.plane, y);
      pixel_type* const JXL_RESTRICT row_out = gc.Row(y);
      for (size_t x = 0; x < r.xsize(); ++x) {
        row_out[x] = row_in[x];
      }
    }
    gi.channel.emplace_back(std::move(gc));
  }
  gi.nb_channels = gi.channel.size();
  gi.real_nb_channels = gi.nb_channels;

  // Do some per-group transforms

  float quality = cparams.quality_pair.first;

  // Local palette
  if (cparams.palette_colors != 0 && cparams.speed_tier < SpeedTier::kCheetah) {
    // all-channel palette (e.g. RGBA)
    if (gi.nb_channels > 1) {
      Transform maybe_palette(TransformId::kPalette);
      maybe_palette.begin_c = gi.nb_meta_channels;
      maybe_palette.num_c = gi.nb_channels;
      maybe_palette.nb_colors = std::abs(cparams.palette_colors);
      maybe_palette.ordered_palette = cparams.palette_colors >= 0;
      gi.do_transform(maybe_palette);
    }
    // all-minus-one-channel palette (RGB with separate alpha, or CMY with
    // separate K)
    if (gi.nb_channels > 3) {
      Transform maybe_palette_3(TransformId::kPalette);
      maybe_palette_3.begin_c = gi.nb_meta_channels;
      maybe_palette_3.num_c = gi.nb_channels - 1;
      maybe_palette_3.nb_colors = std::abs(cparams.palette_colors);
      maybe_palette_3.ordered_palette = cparams.palette_colors >= 0;
      gi.do_transform(maybe_palette_3);
    }
  }

  // Local channel palette
  if (cparams.channel_colors_percent > 0 && quality == 100 &&
      cparams.speed_tier < SpeedTier::kCheetah) {
    // single channel palette (like FLIF's ChannelCompact)
    for (size_t i = 0; i < gi.nb_channels; i++) {
      int min, max;
      gi.channel[gi.nb_meta_channels + i].compute_minmax(&min, &max);
      int colors = max - min + 1;
      JXL_DEBUG_V(10, "Channel %zu: range=%i..%i", i, min, max);
      Transform maybe_palette_1(TransformId::kPalette);
      maybe_palette_1.begin_c = i + gi.nb_meta_channels;
      maybe_palette_1.num_c = 1;
      // simple heuristic: if less than X percent of the values in the range
      // actually occur, it is probably worth it to do a compaction
      // (but only if the channel palette is less than 80% the size of the
      // image itself)
      maybe_palette_1.nb_colors =
          std::min((int)(xsize * ysize * 0.8),
                   (int)(cparams.channel_colors_percent / 100. * colors));
      gi.do_transform(maybe_palette_1);
    }
  }
  if (cparams.near_lossless > 0 && gi.nb_channels != 0) {
    if (cparams.colorspace == 0) {
      Transform nl(TransformId::kNearLossless);
      nl.begin_c = gi.nb_meta_channels;
      nl.num_c = gi.nb_channels;
      nl.max_delta_error = cparams.near_lossless;
      gi.do_transform(nl);
    } else {
      Transform nl(TransformId::kNearLossless);
      nl.begin_c = gi.nb_meta_channels;
      nl.num_c = 1;
      nl.max_delta_error = cparams.near_lossless;
      gi.do_transform(nl);
      nl.begin_c += 1;
      nl.num_c = gi.nb_channels - 1;
      nl.max_delta_error *= 2;  // more loss for chroma
      gi.do_transform(nl);
    }
  }

  size_t compressed_size = std::numeric_limits<size_t>::max();
  size_t best_rct = 0;
  auto try_compress_once = [&](int new_best,
                               const ModularOptions& options) -> Status {
    BitWriter compressed;
    JXL_RETURN_IF_ERROR(ModularGenericCompress(gi, options, &compressed,
                                               /*aux_out=*/nullptr, /*layer=*/0,
                                               group_id, uint_config));
    if (compressed.BitsWritten() < compressed_size) {
      compressed_size = compressed.BitsWritten();
      group_options[group_id] = options;
      best_rct = new_best;
    }
    return true;
  };

  size_t nb_wp_modes = 0;
  switch (cparams.speed_tier) {
    case SpeedTier::kFalcon:
      nb_wp_modes = 1;
      break;
    case SpeedTier::kCheetah:
      nb_wp_modes = 1;
      break;
    case SpeedTier::kHare:
      nb_wp_modes = 1;
      break;
    case SpeedTier::kWombat:
      nb_wp_modes = 1;
      break;
    case SpeedTier::kSquirrel:
      nb_wp_modes = 1;
      break;
    case SpeedTier::kKitten:
      nb_wp_modes = 2;
      break;
    case SpeedTier::kTortoise:
      nb_wp_modes = 5;
      break;
  }
  auto try_compress = [&](int new_best) -> Status {
    if (base_options.predictor != Predictor::Weighted &&
        base_options.predictor != Predictor::Best) {
      return try_compress_once(new_best, base_options);
    }
    if (base_options.predictor == Predictor::Weighted ||
        base_options.predictor == Predictor::Best) {
      ModularOptions options = base_options;
      for (size_t i = 0; i < nb_wp_modes; i++) {
        options.wp_mode = i;
        JXL_RETURN_IF_ERROR(try_compress_once(new_best, options));
      }
    }
    return true;
  };

  // lossless and no specific color transform specified: try Nothing, YCoCg,
  // and 17 RCTs
  if (cparams.color_transform == ColorTransform::kNone && quality == 100 &&
      cparams.colorspace < 0 && gi.nb_channels > 2 && !cparams.near_lossless &&
      cparams.responsive == false && do_color &&
      cparams.speed_tier <= SpeedTier::kWombat) {
    Transform sg(TransformId::kRCT);
    sg.begin_c = gi.nb_meta_channels;

    size_t nb_rcts_to_try = 0;
    switch (cparams.speed_tier) {
      case SpeedTier::kFalcon:
        nb_rcts_to_try = 2;
        break;
      case SpeedTier::kCheetah:
        nb_rcts_to_try = 3;
        break;
      case SpeedTier::kHare:
        nb_rcts_to_try = 4;
        break;
      case SpeedTier::kWombat:
        nb_rcts_to_try = 5;
        break;
      case SpeedTier::kSquirrel:
        nb_rcts_to_try = 7;
        break;
      case SpeedTier::kKitten:
        nb_rcts_to_try = 9;
        break;
      case SpeedTier::kTortoise:
        nb_rcts_to_try = 19;
        break;
    }
    // These should be 19 actually different transforms; the remaining ones
    // are equivalent to one of these (note that the first two are do-nothing
    // and YCoCg) modulo channel reordering (which only matters in the case of
    // MA-with-prev-channels-properties) and/or sign (e.g. RmG vs GmR)
    for (int i : {0 * 7 + 0, 0 * 7 + 6, 0 * 7 + 5, 1 * 7 + 3, 3 * 7 + 5,
                  5 * 7 + 5, 1 * 7 + 5, 2 * 7 + 5, 1 * 7 + 1, 0 * 7 + 4,
                  1 * 7 + 2, 2 * 7 + 1, 2 * 7 + 2, 2 * 7 + 3, 4 * 7 + 4,
                  4 * 7 + 5, 0 * 7 + 2, 0 * 7 + 1, 0 * 7 + 3}) {
      if (nb_rcts_to_try == 0) break;
      int num_transforms_to_keep = gi.transform.size();
      sg.rct_type = i;
      gi.do_transform(sg);
      JXL_RETURN_IF_ERROR(try_compress(/*new_best=*/i));
      nb_rcts_to_try--;
      // Ensure we do not clamp channels to their supposed range, as this
      // otherwise breaks in the presence of patches.
      gi.undo_transforms(num_transforms_to_keep == 0 ? -1
                                                     : num_transforms_to_keep);
    }
    // Apply the best RCT to the image for future encoding.
    sg.rct_type = best_rct;
    gi.do_transform(sg);
  } else {
    // No need to try anything in wombat mode or faster: the default options
    // are already set.
    if (cparams.speed_tier < SpeedTier::kWombat) {
      JXL_RETURN_IF_ERROR(try_compress(/*new_best=*/-1));
    }
  }
  if (bit_size) *bit_size = compressed_size;
  return true;
}
}  // namespace jxl
