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

#include "tools/cjxl.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "jxl/aux_out.h"
#include "jxl/base/arch_specific.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/os_specific.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/codec_in_out.h"
#include "jxl/common.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_file.h"
#include "jxl/extras/codec.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/modular/encoding/encoding.h"
#include "tools/args.h"
#include "tools/speed_stats.h"

namespace jpegxl {
namespace tools {
namespace {

static inline bool ParseSpeedTier(const char* arg, jxl::SpeedTier* out) {
  return jxl::ParseSpeedTier(arg, out);
}
static inline bool ParseColorTransform(const char* arg,
                                       jxl::ColorTransform* out) {
  size_t value = 0;
  bool ret = ParseUnsigned(arg, &value);
  if (ret && value > 2) ret = false;
  if (ret) *out = jxl::ColorTransform(value);
  return ret;
}
// Proposes a distance to try for a given bpp target. This could depend
// on the entropy in the image, too, but let's start with something.
static double ApproximateDistanceForBPP(double bpp) {
  return 1.704 * pow(bpp, -0.804);
}

}  // namespace

JxlCompressArgs::JxlCompressArgs() {
  jxl::ProcessorTopology topology;
  JXL_CHECK(jxl::DetectProcessorTopology(&topology));
  num_threads = topology.packages * topology.cores_per_package;
}

jxl::Status JxlCompressArgs::AddCommandLineOptions(CommandLineParser* cmdline) {
  cmdline->AddPositionalOption(
      "SPOT", "spot color channel (optional, for testing)", &spot_in);
  // Flags.
  cmdline->AddOptionFlag('\0', "progressive",
                         "Use the progressive mode for AC.",
                         &params.progressive_mode, &SetBooleanTrue);
  cmdline->AddOptionFlag('\0', "qprogressive",
                         "Use the progressive mode for AC.",
                         &params.qprogressive_mode, &SetBooleanTrue);
  cmdline->AddOptionValue('\0', "progressive_dc", "num_dc_frames",
                          "Use progressive mode for DC.",
                          &params.progressive_dc, &ParseUnsigned);
  cmdline->AddOptionFlag('g', "modular-group",
                         "Use the modular-group mode (lossy / lossless).",
                         &params.modular_group_mode, &SetBooleanTrue);
  cmdline->AddOptionFlag('b', "jpeg-group", "Use the jpeg-group mode.",
                         &params.brunsli_group_mode, &SetBooleanTrue);

  cmdline->AddOptionFlag(
      'j', "jpeg_transcode",
      "Do lossless transcode of input JPEG file (don't decode it to pixels).",
      &jpeg_transcode, &SetBooleanTrue);

  cmdline->AddOptionFlag('v', "verbose", "Verbose output.", &params.verbose,
                         &SetBooleanTrue);

  cmdline->AddOptionValue('\0', "num_threads", "N",
                          "number of worker threads (zero = none).",
                          &num_threads, &ParseUnsigned);
  cmdline->AddOptionValue('\0', "num_reps", "N", "how many times to compress.",
                          &num_reps, &ParseUnsigned);

  cmdline->AddOptionValue('\0', "noise", "0|1",
                          "force enable/disable noise generation.",
                          &params.noise, &ParseOverride);
  cmdline->AddOptionValue('\0', "dots", "0|1",
                          "force enable/disable dots generation.", &params.dots,
                          &ParseOverride);

  cmdline->AddOptionValue('\0', "adaptive_reconstruction", "0|1",
                          "force enable/disable loop filter.",
                          &params.adaptive_reconstruction, &ParseOverride);

  cmdline->AddOptionValue('\0', "gaborish", "0|1", "force disable gaborish.",
                          &params.gaborish, &ParseOverride);

  cmdline->AddOptionValue(
      '\0', "speed", "falcon|cheetah|hare|wombat|squirrel|kitten|tortoise",
      "Speed tier to use for compression. kitten is the default; values are in "
      "order from faster to slower.",
      &params.speed_tier, &ParseSpeedTier);

  // Target distance/size/bpp
  opt_distance_id = cmdline->AddOptionValue(
      '\0', "distance", "maxError",
      ("Max. butteraugli distance, lower = higher quality.\n"
       "    Good default: 1.0. Supported range: 0.5 .. 3.0."),
      &params.butteraugli_distance, &ParseFloat);
  opt_target_size_id = cmdline->AddOptionValue(
      '\0', "target_size", "N",
      ("Aim at file size of N bytes.\n"
       "    Compresses to 1 % of the target size in ideal conditions.\n"
       "    Runs the same algorithm as --target_bpp"),
      &params.target_size, &ParseUnsigned);
  opt_target_bpp_id = cmdline->AddOptionValue(
      '\0', "target_bpp", "BPP",
      ("Aim at file size that has N bits per pixel.\n"
       "    Compresses to 1 % of the target BPP in ideal conditions."),
      &params.target_bitrate, &ParseFloat);

  opt_intensity_target_id = cmdline->AddOptionValue(
      '\0', "intensity_target", "N",
      ("Intensity target of monitor in nits, higher\n"
       "   results in higher quality image. Supported range: 250..50000,\n"
       "   default is 250 for standard images, 4000 for input images known"
       "   to have PQ or HLG transfer function."),
      &params.intensity_target, &ParseFloat);

  cmdline->AddOptionValue('\0', "saliency_num_progressive_steps", "N", nullptr,
                          &params.saliency_num_progressive_steps,
                          &ParseUnsigned);
  cmdline->AddOptionValue('\0', "saliency_map_filename", "STRING", nullptr,
                          &params.saliency_map_filename, &ParseString);
  cmdline->AddOptionValue('\0', "saliency_threshold", "0..1", nullptr,
                          &params.saliency_threshold, &ParseFloat);

  cmdline->AddOptionValue(
      'x', "dec-hints", "key=value",
      "color_space indicates the ColorEncoding, see Description().", &dec_hints,
      &ParseAndAppendKeyValue);

  cmdline->AddOptionValue(
      '\0', "override_bitdepth", "0=use from image, 1-32=override",
      "If nonzero, store the given bit depth in the JPEG XL file metadata"
      " (1-32), instead of using the bit depth from the original input"
      " image.",
      &override_bitdepth, &ParseUnsigned);

  opt_color_id = cmdline->AddOptionValue(
      'c', "colortransform", "0..2", "0=XYB, 1=None, 2=YCbCr",
      &params.color_transform, &ParseColorTransform);

  // modular mode options
  cmdline->AddOptionValue(
      'Q', "quality", "luma_q[,chroma_q]",
      "[modular encoding] lossy 'quality' (100=lossless, lower is more lossy)",
      &params.quality_pair, &ParseFloatPair);

  cmdline->AddOptionValue(
      'I', "iterations", "F",
      "[modular encoding] number of mock encodes to learn MABEGABRAC trees "
      "(default=0.5, try 0 for no MA and fast decode)",
      &params.options.nb_repeats, &ParseFloat);

  cmdline->AddOptionValue('T', "ctx-threshold", "F",
                          ("[modular encoding] number of bits hypothetically "
                           "saved during MABEGABRAC training "
                           "to justify adding another context (default: 16)"),
                          &params.options.ctx_threshold, &ParseFloat);

  cmdline->AddOptionValue(
      'C', "colorspace", "K",
      ("[modular encoding] color transform: 0=RGB, 1=YCoCg, "
       "2-37=RCT (default: YCoCg)"),
      &params.colorspace, &ParseSigned);

  cmdline->AddOptionValue('P', "predictor", "K",
                          "[modular encoding] predictor(s) to use: 0=zero, "
                          "1=avgNW, 2=clampedGradient (default), 3=W, 4=N, "
                          "5=WeightedPredictor (good for photo)",
                          &params.options.predictor, &ParsePredictorsVector);

  cmdline->AddOptionValue(
      'E', "extra-properties", "K",
      "[modular encoding] number of extra MA tree properties to use",
      &params.options.max_properties, &ParseSigned);

  cmdline->AddOptionValue('N', "near-lossless", "max_d",
                          "[modular encoding] apply near-lossless "
                          "preprocessing with maximum delta = max_d",
                          &params.near_lossless, &ParseSigned);

  cmdline->AddOptionValue('p', "palette", "K",
                          "[modular encoding] use a palette if image has at "
                          "most K colors (default: 1024)",
                          &params.palette_colors, &ParseSigned);

  cmdline->AddOptionValue(
      'X', "pre-compact", "PERCENT",
      ("[modular encoding] compact channels (globally) if ratio "
       "used/range is below this (default: 80%)"),
      &params.channel_colors_pre_transform_percent, &ParseFloat);

  cmdline->AddOptionValue(
      'Y', "post-compact", "PERCENT",
      ("[modular encoding] compact channels (per-group) if ratio "
       "used/range is below this (default: 80%)"),
      &params.channel_colors_percent, &ParseFloat);

  opt_brotli_id = cmdline->AddOptionValue(
      'B', "brotli", "effort",
      ("[modular encoding] use Brotli instead of MABEGABRAC/MAANS"
       " (with encode effort=0..11)"),
      &params.options.brotli_effort, &ParseSigned);

  cmdline->AddOptionFlag('A', "ans",
                         "[modular encoding] use MAANS instead of MABEGABRAC",
                         &params.ans, &SetBooleanTrue);

  cmdline->AddOptionValue('R', "responsive", "K",
                          "[modular encoding] do Squeeze transform, 0=false, "
                          "1=true (default: true if lossy, false if lossless)",
                          &params.responsive, &ParseSigned);

  return true;
}

jxl::Status JxlCompressArgs::ValidateArgs(
    const tools::CommandLineParser& cmdline) {
  bool got_distance = cmdline.GetOption(opt_distance_id)->matched();
  bool got_target_size = cmdline.GetOption(opt_target_size_id)->matched();
  bool got_target_bpp = cmdline.GetOption(opt_target_bpp_id)->matched();

  got_intensity_target = cmdline.GetOption(opt_intensity_target_id)->matched();

  if (got_distance) {
    constexpr float butteraugli_min_dist = 0.125f;
    constexpr float butteraugli_max_dist = 15.0f;
    if (!(butteraugli_min_dist <= params.butteraugli_distance &&
          params.butteraugli_distance <= butteraugli_max_dist)) {
      fprintf(stderr, "Invalid/out of range distance, try %g to %g.\n",
              butteraugli_min_dist, butteraugli_max_dist);
      return false;
    }
  }

  if (got_target_bpp + got_target_size + got_distance > 1) {
    fprintf(stderr,
            "You can specify only one of '--distance', "
            "'--target_bpp' and '--target_size'. They are all different ways"
            " to specify the image quality. When in doubt, use --distance."
            " It gives the most visually consistent results.\n");
    return false;
  }

  if (!params.saliency_map_filename.empty()) {
    if (!params.progressive_mode) {
      params.saliency_map_filename.clear();
      fprintf(stderr,
              "Warning: Specifying --saliency_map_filename only makes sense "
              "for --progressive mode.\n");
    }
  }

  if (!params.file_in) {
    fprintf(stderr, "Missing input filename.\n");
    return false;
  }

  if (!cmdline.GetOption(opt_color_id)->matched()) {
    // default to RGB for modular
    if (params.modular_group_mode)
      params.color_transform = jxl::ColorTransform::kNone;
    // default to YCrCb for jpeg
    if (params.brunsli_group_mode)
      params.color_transform = jxl::ColorTransform::kYCbCr;
  }

  if (cmdline.GetOption(opt_brotli_id)->matched())
    params.options.entropy_coder = 1;
  else if (params.ans)
    params.options.entropy_coder = 2;

  if (params.near_lossless) {
    // Near-lossless assumes -R 0
    params.responsive = 0;
  }

  if (override_bitdepth > 32) {
    fprintf(stderr, "override_bitdepth must be <= 32\n");
    return false;
  }

  return true;
}

jxl::Status LoadSaliencyMap(const std::string& filename_heatmap,
                            const jxl::CodecInOut* io, jxl::ThreadPool* pool,
                            jxl::ImageF* out_map) {
  jxl::CodecInOut io_heatmap;
  if (!SetFromFile(filename_heatmap, &io_heatmap, pool)) {
    return JXL_FAILURE("Could not load heatmap.");
  }
  jxl::ImageF heatmap(io_heatmap.xsize(), io_heatmap.ysize());
  for (size_t num_row = 0; num_row < io_heatmap.ysize(); num_row++) {
    const float* JXL_RESTRICT row_src =
        io_heatmap.Main().color().ConstPlaneRow(0, num_row);
    float* JXL_RESTRICT row_dst = heatmap.Row(num_row);
    for (size_t num_col = 0; num_col < io_heatmap.xsize(); num_col++) {
      row_dst[num_col] = row_src[num_col] / 255.0f;
    }
  }
  *out_map = std::move(heatmap);
  return true;
}

jxl::Status CompressJxl(jxl::ThreadPoolInternal* pool, JxlCompressArgs& args,
                        jxl::PaddedBytes* compressed) {
  JXL_CHECK(pool);
  double t0, t1;

  jxl::CodecInOut io;
  io.dec_hints = args.dec_hints;
  t0 = jxl::Now();
  if (!SetFromFile(args.params.file_in, &io, nullptr,
                   (args.jpeg_transcode ? jxl::DecodeTarget::kQuantizedCoeffs
                                        : jxl::DecodeTarget::kPixels))) {
    fprintf(stderr, "Failed to read image %s.\n", args.params.file_in);
    return false;
  }
  if (args.override_bitdepth != 0) {
    io.metadata.bits_per_sample = args.override_bitdepth;
  }
  jxl::ImageF saliency_map;
  if (!args.params.saliency_map_filename.empty()) {
    if (!LoadSaliencyMap(args.params.saliency_map_filename, &io, pool,
                         &saliency_map)) {
      fprintf(stderr, "Failed to read saliency map %s.\n",
              args.params.saliency_map_filename.c_str());
      return false;
    }
    args.params.saliency_map = &saliency_map;
  }
  if (!args.got_intensity_target) {
    args.params.intensity_target = ChooseDefaultIntensityTarget(io.metadata);
  }
  if (args.spot_in != nullptr) {
    jxl::CodecInOut spot_io;
    spot_io.dec_hints = args.dec_hints;
    if (!SetFromFile(args.spot_in, &spot_io)) {
      fprintf(stderr, "Failed to read spot image %s.\n", args.spot_in);
      return false;
    }
    io.metadata.m2.num_extra_channels = 1;
    io.metadata.m2.extra_channel_bits = 8;
    jxl::ExtraChannelInfo example;
    example.rendered = 1;
    example.color[0] = 255.0f;
    example.color[1] = 0.0f;
    example.color[2] = 0.0f;
    example.color[3] = 1.0f;
    io.metadata.m2.extra_channel_info.push_back(example);
    jxl::ImageU sc(spot_io.xsize(), spot_io.ysize());
    for (size_t y = 0; y < spot_io.ysize(); ++y) {
      const float* JXL_RESTRICT from = spot_io.Main().color().PlaneRow(1, y);
      uint16_t* JXL_RESTRICT to = sc.Row(y);
      for (size_t x = 0; x < spot_io.xsize(); ++x) {
        to[x] = from[x];
      }
    }
    std::vector<jxl::ImageU> scv;
    scv.push_back(std::move(sc));
    io.Main().SetExtraChannels(std::move(scv));
  }
  t1 = jxl::Now();
  const double decode_mps =
      io.xsize() * io.ysize() * io.frames.size() * 1E-6 / (t1 - t0);

  const size_t xsize = io.xsize();
  const size_t ysize = io.ysize();
  if (args.params.target_size > 0 || args.params.target_bitrate > 0) {
    // Search algorithm for target bpp / size.
    JxlCompressArgs s = args;  // Args for search.
    if (s.params.target_size > 0) {
      s.params.target_bitrate =
          s.params.target_size * 8.0 / (io.xsize() * io.ysize());
      s.params.target_size = 0;
    }
    double dist = ApproximateDistanceForBPP(s.params.target_bitrate);
    s.params.butteraugli_distance = dist;
    double target_size =
        s.params.target_bitrate * (1 / 8.) * io.xsize() * io.ysize();
    s.params.target_bitrate = 0;
    double best_dist = 1.0;
    double best_loss = 1e99;
    for (int i = 0; i < 7; ++i) {
      s.params.butteraugli_distance = dist;
      jxl::PaddedBytes candidate;
      bool ok = CompressJxl(pool, s, &candidate);
      if (!ok) {
        printf(
            "Compression error occurred during the search for best size."
            " Trying with butteraugli distance %.15g\n",
            best_dist);
        break;
      }
      printf("Butteraugli distance %g yields %zu bytes, %g bpp.\n", dist,
             candidate.size(),
             candidate.size() * 8.0 / (io.xsize() * io.ysize()));
      const double ratio = static_cast<double>(candidate.size()) / target_size;
      const double loss = std::max(ratio, 1.0 / std::max(ratio, 1e-30));
      if (best_loss > loss) {
        best_dist = dist;
        best_loss = loss;
      }
      dist *= ratio;
      if (dist < 0.01) {
        dist = 0.01;
      }
      if (dist >= 16.0) {
        dist = 16.0;
      }
    }
    printf("Choosing butteraugli distance %.15g\n", best_dist);
    args.params.butteraugli_distance = best_dist;
    args.params.target_bitrate = 0;
    args.params.target_size = 0;
  }
  char mode[200];
  if (args.params.speed_tier != jxl::SpeedTier::kKitten) {
    snprintf(mode, sizeof(mode),
             "in %s mode with maximum Butteraugli distance %f",
             SpeedTierName(args.params.speed_tier),
             args.params.butteraugli_distance);
  } else {
    snprintf(mode, sizeof(mode), "with maximum Butteraugli distance %f",
             args.params.butteraugli_distance);
  }
  fprintf(stderr,
          "Read %zu bytes (%zux%zu, %.1f MP/s); compressing %s, %zu threads.\n",
          io.enc_size, xsize, ysize, decode_mps, mode,
          pool->NumWorkerThreads());

  jxl::AuxOut aux_out;
  if (args.inspector_image3f) {
    aux_out.SetInspectorImage3F(args.inspector_image3f);
  }
  SpeedStats stats;
  jxl::PassesEncoderState passes_encoder_state;
  for (size_t i = 0; i < args.num_reps; ++i) {
    t0 = jxl::Now();
    if (!EncodeFile(args.params, &io, &passes_encoder_state, compressed,
                    &aux_out, pool)) {
      fprintf(stderr, "Failed to compress.\n");
      return false;
    }
    t1 = jxl::Now();
    stats.NotifyElapsed(t1 - t0);
  }
  const double bpp =
      static_cast<double>(compressed->size() * jxl::kBitsPerByte) /
      (xsize * ysize);
  fprintf(stderr, "Compressed to %zu bytes (%.3f bpp%s).\n", compressed->size(),
          bpp / io.frames.size(), io.frames.size() == 1 ? "" : "/frame");
  JXL_CHECK(stats.Print(xsize, ysize, args.num_threads));

  if (args.params.verbose) {
    aux_out.Print(1);
  }

  return true;
}

}  // namespace tools
}  // namespace jpegxl
