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
    const CompressParams& orig_cparams, const FrameHeader& frame_header,
    const ImageBundle& ib, Image3F* JXL_RESTRICT color,
    PassesEncoderState* JXL_RESTRICT enc_state, bool encode_color) {
  cparams = orig_cparams;
  do_color = encode_color;

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

  // TODO(lode): must handle decoded->metadata()->floating_point_sample?
  int maxval = (1 << ib.metadata()->bits_per_sample) - 1;
  if (cparams.color_transform == ColorTransform::kXYB)
    maxval = 255;  // not true, but bits_per_sample doesn't matter either
  Image gi(xsize, ysize, maxval, nb_chans);
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
  // stop here if we have a trivial zero-channel image
  if (c == 0) {
    full_image = std::move(gi);
    return true;
  }

  // Set options and apply transformations

  float quality = cparams.quality_pair.first;
  float cquality = cparams.quality_pair.second;
  if (cquality > 100) cquality = quality;

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

  if (orig_cparams.color_transform == ColorTransform::kNone && do_color) {
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
  // use a sensible default if nothing explicit is specified:
  // Squeeze for lossy, no squeeze for lossless
  if (cparams.responsive < 0) {
    if (quality == 100)
      cparams.responsive = 0;
    else
      cparams.responsive = 1;
  }
  if (cparams.responsive) {
    gi.do_transform(Transform(TransformId::kSqueeze));  // use default squeezing
  }
  if (cparams.options.entropy_coder == ModularOptions::kMAANS) {
    float pixel_fraction = std::min(1.0f, cparams.options.nb_repeats);
    if (cparams.speed_tier > SpeedTier::kWombat) {
      cparams.options.splitting_heuristics_node_threshold =
          192 * pixel_fraction;
    } else {
      cparams.options.splitting_heuristics_node_threshold = 96 * pixel_fraction;
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
  if (cparams.options.predictor == static_cast<Predictor>(-1)) {
    // no explicit predictor(s) given, set a good default
    if (cparams.near_lossless) {
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

  full_image = std::move(gi);
  return true;
}

Status ModularFrameEncoder::EncodeGlobalInfo(BitWriter* writer, AuxOut* aux_out,
                                             size_t group_id) {
  cparams.options.max_chan_size = kGroupDim;
  ModularOptions options = cparams.options;
  if (options.predictor == Predictor::Best) {
    options.predictor = Predictor::Gradient;
  }
  ModularGenericCompress(full_image, options, writer, aux_out,
                         kLayerModularGlobal, group_id, call++);
  return true;
}

Status ModularFrameEncoder::EncodeGroup(const Rect& rect, BitWriter* writer,
                                        AuxOut* aux_out, size_t minShift,
                                        size_t maxShift, size_t layer,
                                        size_t group_id) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  int maxval = full_image.maxval;
  Image gi(xsize, ysize, maxval, 0);
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

  HybridUintConfig uint_config_sign{4, 2, 1};

#if JXL_DEBUG_V_LEVEL >= 5
  // Only used for lossless.
  int best = 0;
#endif

  BitWriter local_compressed;
  AuxOut local_aux_out;
  bool has_compressed = false;
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
  auto try_compress_once = [&](int new_best, const ModularOptions& options,
                               HybridUintConfig uint_config) -> Status {
    BitWriter compressed2;
    AuxOut aux_out2;
    if (aux_out) {
      aux_out2.testing_aux = aux_out->testing_aux;
      aux_out2.dump_image = aux_out->dump_image;
      aux_out2.debug_prefix = aux_out->debug_prefix;
    }
    JXL_RETURN_IF_ERROR(ModularGenericCompress(gi, options, &compressed2,
                                               &aux_out2, layer, group_id,
                                               call++, uint_config));
    if (compressed2.BitsWritten() < local_compressed.BitsWritten() ||
        !has_compressed) {
      has_compressed = true;
      local_compressed = std::move(compressed2);
      local_aux_out = std::move(aux_out2);
#if JXL_DEBUG_V_LEVEL >= 5
      best = new_best;
#endif
    }
    return true;
  };
  auto try_compress = [&](int new_best, HybridUintConfig uint_config =
                                            kHybridUint420Config) -> Status {
    if (cparams.options.predictor != Predictor::Weighted &&
        cparams.options.predictor != Predictor::Best) {
      return try_compress_once(new_best, cparams.options, uint_config);
    }
    if (cparams.options.predictor == Predictor::Best) {
      ModularOptions options = cparams.options;
      options.predictor = Predictor::Gradient;
      JXL_RETURN_IF_ERROR(try_compress_once(new_best, options, uint_config));
    }
    if (cparams.options.predictor == Predictor::Weighted ||
        cparams.options.predictor == Predictor::Best) {
      ModularOptions options = cparams.options;
      options.predictor = Predictor::Weighted;
      for (size_t i = 0; i < nb_wp_modes; i++) {
        options.wp_mode = i;
        JXL_RETURN_IF_ERROR(try_compress_once(new_best, options, uint_config));
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
      // Try putting the sign bit in the token only at slower settings.
      if (cparams.speed_tier < SpeedTier::kWombat) {
        JXL_RETURN_IF_ERROR(try_compress(
            /*new_best=*/i, uint_config_sign));
      }
      JXL_RETURN_IF_ERROR(try_compress(/*new_best=*/i));
      nb_rcts_to_try--;
      // Ensure we do not clamp channels to their supposed range, as this
      // otherwise breaks in the presence of patches.
      gi.undo_transforms(num_transforms_to_keep == 0 ? -1
                                                     : num_transforms_to_keep);
    }
#if JXL_DEBUG_V_LEVEL >= 5
    JXL_DEBUG_V(5, "Best color transform: %i", best);
#endif
  } else {
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

    JXL_RETURN_IF_ERROR(try_compress(/*new_best=*/-1));
    if (cparams.speed_tier < SpeedTier::kWombat) {
      JXL_RETURN_IF_ERROR(try_compress(/*new_best=*/-1, uint_config_sign));
    }
  }
  writer->ZeroPadToByte();
  if (local_compressed.BitsWritten() != 0) {
    local_compressed.ZeroPadToByte();
    writer->AppendByteAligned(std::move(local_compressed));
    if (aux_out) aux_out->Assimilate(local_aux_out);
  }
  return true;
}

}  // namespace jxl
