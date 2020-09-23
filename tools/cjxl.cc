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
#if JPEGXL_ENABLE_JPEG
#include "jxl/extras/codec_jpg.h"
#endif
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/modular/encoding/encoding.h"
#include "tools/args.h"
#include "tools/box/box.h"
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
static inline bool ParseIntensityTarget(const char* arg, float* out) {
  return ParseFloat(arg, out) && *out > 0;
}

// Proposes a distance to try for a given bpp target. This could depend
// on the entropy in the image, too, but let's start with something.
static double ApproximateDistanceForBPP(double bpp) {
  return 1.704 * pow(bpp, -0.804);
}

jxl::Status LoadSaliencyMap(const std::string& filename_heatmap,
                            jxl::ThreadPool* pool, jxl::ImageF* out_map) {
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

jxl::Status LoadSpotColors(const JxlCompressArgs& args, jxl::CodecInOut* io) {
  jxl::CodecInOut spot_io;
  spot_io.target_nits = args.intensity_target;
  spot_io.dec_hints = args.dec_hints;
  if (!SetFromFile(args.spot_in, &spot_io)) {
    fprintf(stderr, "Failed to read spot image %s.\n", args.spot_in);
    return false;
  }
  jxl::ExtraChannelInfo example;
  example.type = jxl::ExtraChannel::kSpotColor;
  example.blend_mode = jxl::BlendMode::kBlend;
  example.bit_depth.bits_per_sample = 8;
  example.dim_shift = 0;
  example.name = "spot";
  example.spot_color[0] = io->metadata.IntensityTarget();  // R
  example.spot_color[1] = 0.0f;                            // G
  example.spot_color[2] = 0.0f;                            // B
  example.spot_color[3] = 1.0f;                            // A
  io->metadata.m2.extra_channel_info.push_back(example);
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
  io->Main().SetExtraChannels(std::move(scv));
  return true;
}

jxl::Status LoadAll(JxlCompressArgs& args, jxl::ThreadPoolInternal* pool,
                    jxl::CodecInOut* io, double* decode_mps) {
  const double t0 = jxl::Now();

  io->target_nits = args.intensity_target;
  io->dec_hints = args.dec_hints;
  io->dec_target = (args.jpeg_transcode ? jxl::DecodeTarget::kQuantizedCoeffs
                                        : jxl::DecodeTarget::kPixels);
  jxl::Codec input_codec;
  if (!SetFromFile(args.params.file_in, io, nullptr, &input_codec)) {
    fprintf(stderr, "Failed to read image %s.\n", args.params.file_in);
    return false;
  }
  if (input_codec != jxl::Codec::kJPG) args.jpeg_transcode = false;

  if (input_codec == jxl::Codec::kGIF && args.default_settings) {
    args.params.modular_group_mode = true;
    args.params.options.predictor = jxl::Predictor::Select;
    args.params.responsive = 0;
    args.params.colorspace = 0;
    args.params.channel_colors_pre_transform_percent = 0;
    args.params.channel_colors_percent = 0;
    args.params.quality_pair.first = args.params.quality_pair.second = 100;
  }

  if (args.override_bitdepth != 0) {
    if (args.override_bitdepth == 32) {
      io->metadata.SetFloat32Samples();
    } else {
      io->metadata.SetUintSamples(args.override_bitdepth);
    }
  }

  jxl::ImageF saliency_map;
  if (!args.saliency_map_filename.empty()) {
    if (!LoadSaliencyMap(args.saliency_map_filename, pool, &saliency_map)) {
      fprintf(stderr, "Failed to read saliency map %s.\n",
              args.saliency_map_filename.c_str());
      return false;
    }
    args.params.saliency_map = &saliency_map;
  }

  if (args.spot_in != nullptr) {
    if (!LoadSpotColors(args, io)) {
      fprintf(stderr, "Failed to read spot colors %s.\n", args.spot_in);
      return false;
    }
  }

  const double t1 = jxl::Now();
  const size_t pixels = io->xsize() * io->ysize();
  *decode_mps = pixels * io->frames.size() * 1E-6 / (t1 - t0);

  return true;
}

// Search algorithm for modular mode instead of Butteraugli distance.
void SetModularQualityForBitrate(jxl::ThreadPoolInternal* pool,
                                 const size_t pixels, const double target_size,
                                 JxlCompressArgs* args) {
  JXL_ASSERT(args->params.modular_group_mode);

  JxlCompressArgs s = *args;  // Args for search.
  // 5 bpp => 100, 0.1 bpp => 2
  float quality = s.params.target_bitrate * 20;
  s.params.target_bitrate = 0;
  double best_loss = 1e99;
  float best_quality = quality;
  for (int i = 0; i < 7; ++i) {
    s.params.quality_pair = std::make_pair(quality, quality);
    jxl::PaddedBytes candidate;
    bool ok = CompressJxl(pool, s, &candidate, /*print_stats=*/false);
    if (!ok) {
      printf(
          "Compression error occurred during the search for best size."
          " Trying with quality %.1f\n",
          quality);
      break;
    }
    printf("Quality %.1f yields %6zu bytes, %.3f bpp.\n", quality,
           candidate.size(), candidate.size() * 8.0 / pixels);
    const double ratio = static_cast<double>(candidate.size()) / target_size;
    const double loss = std::max(ratio, 1.0 / std::max(ratio, 1e-30));
    if (best_loss > loss) {
      best_quality = quality;
      best_loss = loss;
    }
    quality /= ratio;
    if (quality < 1) {
      quality = 1;
    }
    if (quality >= 100) {
      quality = 100;
    }
  }
  args->params.quality_pair = std::make_pair(best_quality, best_quality);
  args->params.target_bitrate = 0;
  args->params.target_size = 0;
}

void SetParametersForSizeOrBitrate(jxl::ThreadPoolInternal* pool,
                                   const size_t pixels, JxlCompressArgs* args) {
  JxlCompressArgs s = *args;  // Args for search.

  // If fixed size, convert to bitrate.
  if (s.params.target_size > 0) {
    s.params.target_bitrate = s.params.target_size * 8.0 / pixels;
    s.params.target_size = 0;
  }
  const double target_size = s.params.target_bitrate * (1 / 8.) * pixels;

  if (args->params.modular_group_mode) {
    SetModularQualityForBitrate(pool, pixels, target_size, args);
    return;
  }

  double dist = ApproximateDistanceForBPP(s.params.target_bitrate);
  s.params.target_bitrate = 0;
  double best_dist = 1.0;
  double best_loss = 1e99;
  for (int i = 0; i < 7; ++i) {
    s.params.butteraugli_distance = static_cast<float>(dist);
    jxl::PaddedBytes candidate;
    bool ok = CompressJxl(pool, s, &candidate, /*print_stats=*/false);
    if (!ok) {
      printf(
          "Compression error occurred during the search for best size. "
          "Trying with butteraugli distance %.15g\n",
          best_dist);
      break;
    }
    printf("Butteraugli distance %.3f yields %6zu bytes, %.3f bpp.\n", dist,
           candidate.size(), candidate.size() * 8.0 / pixels);
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
  args->params.butteraugli_distance = static_cast<float>(best_dist);
  args->params.target_bitrate = 0;
  args->params.target_size = 0;
}

const char* ModeFromArgs(const JxlCompressArgs& args) {
  if (args.jpeg_transcode) return "JPEG";
  if (args.params.modular_group_mode) return "Modular";
  if (args.params.pixels_to_jpeg_mode) return "JPEG(encode)";
  return "VarDCT";
}

std::string QualityFromArgs(const JxlCompressArgs& args) {
  char buf[100];
  if (args.jpeg_transcode) {
    snprintf(buf, sizeof(buf), "lossless transcode");
  } else if (args.params.modular_group_mode) {
    if (args.params.quality_pair.first == 100 &&
        args.params.quality_pair.second == 100) {
      snprintf(buf, sizeof(buf), "lossless");
    } else if (args.params.quality_pair.first !=
               args.params.quality_pair.second) {
      snprintf(buf, sizeof(buf), "Q%.2f,%.2f", args.params.quality_pair.first,
               args.params.quality_pair.second);
    } else {
      snprintf(buf, sizeof(buf), "Q%.2f", args.params.quality_pair.first);
    }
  } else if (args.params.pixels_to_jpeg_mode) {
    snprintf(buf, sizeof(buf), "q%u", args.params.jpeg_quality);
  } else {
    snprintf(buf, sizeof(buf), "d%.3f", args.params.butteraugli_distance);
  }
  return buf;
}

void PrintMode(jxl::ThreadPoolInternal* pool, const jxl::CodecInOut& io,
               const double decode_mps, const JxlCompressArgs& args) {
  const char* mode = ModeFromArgs(args);
  const char* speed = SpeedTierName(args.params.speed_tier);
  const std::string quality = QualityFromArgs(args);
  fprintf(stderr,
          "Read %zu bytes (%zux%zu, %.3f bpp, %.1f MP/s)\n"
          "Encoding [%s, %s, %s], %zu threads.\n",
          io.enc_size, io.xsize(), io.ysize(),
          io.enc_size * 8.0 / (io.xsize() * io.ysize()), decode_mps, mode,
          quality.c_str(), speed, pool->NumWorkerThreads());
}

}  // namespace

JxlCompressArgs::JxlCompressArgs() {}

jxl::Status JxlCompressArgs::AddCommandLineOptions(CommandLineParser* cmdline) {
  cmdline->AddPositionalOption("SPOT", /* required = */ false,
                               "spot color channel (optional, for testing)",
                               &spot_in, 2);

  // Target distance/size/bpp
  opt_distance_id = cmdline->AddOptionValue(
      'd', "distance", "maxError",
      ("Max. butteraugli distance, lower = higher quality. Range: 0 .. 15.\n"
       "    0.0 = mathematically lossless. Default for already-lossy input "
       "(JPEG/GIF).\n"
       "    1.0 = visually lossless. Default for other input.\n"
       "    Recommended range: 0.5 .. 3.0."),
      &params.butteraugli_distance, &ParseFloat);
  opt_target_size_id = cmdline->AddOptionValue(
      '\0', "target_size", "N",
      ("Aim at file size of N bytes.\n"
       "    Compresses to 1 % of the target size in ideal conditions.\n"
       "    Runs the same algorithm as --target_bpp"),
      &params.target_size, &ParseUnsigned, 1);
  opt_target_bpp_id = cmdline->AddOptionValue(
      '\0', "target_bpp", "BPP",
      ("Aim at file size that has N bits per pixel.\n"
       "    Compresses to 1 % of the target BPP in ideal conditions."),
      &params.target_bitrate, &ParseFloat, 1);

  // High-level options
  opt_quality_id = cmdline->AddOptionValue(
      'q', "quality", "QUALITY",
      "Quality setting for modular mode. Range: 0 .. 100.\n"
      "    100 = mathematically lossless. Default for already-lossy input "
      "(JPEG/GIF).\n",
      &quality, &ParseFloat);

  cmdline->AddOptionValue(
      's', "speed", "SPEED",
      "Encoder effort/speed setting. Valid values are:\n"
      "    3|falcon| 4|cheetah| 5|hare| 6|wombat| 7|squirrel| 8|kitten| "
      "9|tortoise\n"
      "    Default: squirrel (7). Values are in order from faster to slower.",
      &params.speed_tier, &ParseSpeedTier);

  cmdline->AddOptionFlag('p', "progressive",
                         "Enable progressive/responsive decoding.",
                         &progressive, &SetBooleanTrue);

  // Flags.
  cmdline->AddOptionFlag('\0', "progressive_ac",
                         "Use the progressive mode for AC.",
                         &params.progressive_mode, &SetBooleanTrue, 1);
  cmdline->AddOptionFlag('\0', "qprogressive_ac",
                         "Use the progressive mode for AC.",
                         &params.qprogressive_mode, &SetBooleanTrue, 1);
  cmdline->AddOptionValue('\0', "progressive_dc", "num_dc_frames",
                          "Use progressive mode for DC.",
                          &params.progressive_dc, &ParseUnsigned, 1);
  cmdline->AddOptionFlag('g', "modular-group",
                         "Use the modular mode (lossy / lossless).",
                         &params.modular_group_mode, &SetBooleanTrue, 1);

  // JPEG modes: parallel Brunsli, pixels to JPEG, or JPEG to Brunsli
  cmdline->AddOptionFlag('\0', "jpeg1", "Compress pixels to JPEG.",
                         &params.pixels_to_jpeg_mode, &SetBooleanTrue, 1);
  cmdline->AddOptionFlag('j', "jpeg_transcode",
                         "Do lossy transcode of input JPEG file (decode to "
                         "pixels instead of doing lossless transcode).",
                         &jpeg_transcode, &SetBooleanFalse, 1);
  cmdline->AddOptionValue('\0', "jpeg_quality", "0-100",
                          "Target JPEG quality .", &params.jpeg_quality,
                          &ParseUint32, 1);
  cmdline->AddOptionFlag('\0', "jpeg_420",
                         "Do 4:2:0 chroma subsampling for JPEG.",
                         &params.jpeg_420, &SetBooleanTrue, 1);

  opt_num_threads_id = cmdline->AddOptionValue(
      '\0', "num_threads", "N", "number of worker threads (zero = none).",
      &num_threads, &ParseUnsigned, 1);
  cmdline->AddOptionValue('\0', "num_reps", "N", "how many times to compress.",
                          &num_reps, &ParseUnsigned, 1);

  cmdline->AddOptionValue('\0', "noise", "0|1",
                          "force enable/disable noise generation.",
                          &params.noise, &ParseOverride, 1);
  cmdline->AddOptionValue('\0', "dots", "0|1",
                          "force enable/disable dots generation.", &params.dots,
                          &ParseOverride, 1);
  cmdline->AddOptionValue('\0', "patches", "0|1",
                          "force enable/disable patches generation.",
                          &params.patches, &ParseOverride, 1);

  cmdline->AddOptionValue('\0', "adaptive_reconstruction", "0|1",
                          "force enable/disable loop filter.",
                          &params.adaptive_reconstruction, &ParseOverride, 1);

  cmdline->AddOptionValue('\0', "gaborish", "0|1", "force disable gaborish.",
                          &params.gaborish, &ParseOverride, 1);

  opt_intensity_target_id = cmdline->AddOptionValue(
      '\0', "intensity_target", "N",
      ("Intensity target of monitor in nits, higher\n"
       "   results in higher quality image. Must be strictly positive.\n"
       "   Default is 255 for standard images, 4000 for input images known to\n"
       "   to have PQ or HLG transfer function."),
      &intensity_target, &ParseIntensityTarget, 1);

  cmdline->AddOptionValue('\0', "saliency_num_progressive_steps", "N", nullptr,
                          &params.saliency_num_progressive_steps,
                          &ParseUnsigned, 2);
  cmdline->AddOptionValue('\0', "saliency_map_filename", "STRING", nullptr,
                          &saliency_map_filename, &ParseString, 2);
  cmdline->AddOptionValue('\0', "saliency_threshold", "0..1", nullptr,
                          &params.saliency_threshold, &ParseFloat, 2);

  cmdline->AddOptionValue(
      'x', "dec-hints", "key=value",
      "color_space indicates the ColorEncoding, see Description();\n"
      "icc_pathname refers to a binary file containing an ICC profile.",
      &dec_hints, &ParseAndAppendKeyValue, 1);

  cmdline->AddOptionValue(
      '\0', "override_bitdepth", "0=use from image, 1-32=override",
      "If nonzero, store the given bit depth in the JPEG XL file metadata"
      " (1-32), instead of using the bit depth from the original input"
      " image.",
      &override_bitdepth, &ParseUnsigned, 2);

  opt_color_id = cmdline->AddOptionValue(
      'c', "colortransform", "0..2", "0=XYB, 1=None, 2=YCbCr",
      &params.color_transform, &ParseColorTransform, 2);

  // modular mode options
  cmdline->AddOptionValue(
      'Q', "mquality", "luma_q[,chroma_q]",
      "[modular encoding] lossy 'quality' (100=lossless, lower is more lossy)",
      &params.quality_pair, &ParseFloatPair, 1);

  cmdline->AddOptionValue(
      'I', "iterations", "F",
      "[modular encoding] number of mock encodes to learn MABEGABRAC trees "
      "(default=0.5, try 0 for no MA and fast decode)",
      &params.options.nb_repeats, &ParseFloat, 2);

  cmdline->AddOptionValue(
      'C', "colorspace", "K",
      ("[modular encoding] color transform: 0=RGB, 1=YCoCg, "
       "2-37=RCT (default: try several, depending on speed)"),
      &params.colorspace, &ParseSigned, 1);

  cmdline->AddOptionValue(
      'P', "predictor", "K",
      "[modular encoding] predictor(s) to use: 0=zero, "
      "1=left, 2=top, 3=avg, 4=select, 5=gradient, 6=variable, "
      "7=weighted (default: best of 5,7)",
      &params.options.predictor, &ParsePredictor, 1);

  cmdline->AddOptionValue(
      'E', "extra-properties", "K",
      "[modular encoding] number of extra MA tree properties to use",
      &params.options.max_properties, &ParseSigned, 2);

  cmdline->AddOptionValue('N', "near-lossless", "max_d",
                          "[modular encoding] apply near-lossless "
                          "preprocessing with maximum delta = max_d",
                          &params.near_lossless, &ParseSigned, 1);

  cmdline->AddOptionValue('\0', "palette", "K",
                          "[modular encoding] use a palette if image has at "
                          "most K colors (default: 1024)",
                          &params.palette_colors, &ParseSigned, 1);

  cmdline->AddOptionValue(
      'X', "pre-compact", "PERCENT",
      ("[modular encoding] compact channels (globally) if ratio "
       "used/range is below this (default: 80%)"),
      &params.channel_colors_pre_transform_percent, &ParseFloat, 2);

  cmdline->AddOptionValue(
      'Y', "post-compact", "PERCENT",
      ("[modular encoding] compact channels (per-group) if ratio "
       "used/range is below this (default: 80%)"),
      &params.channel_colors_percent, &ParseFloat, 2);

  cmdline->AddOptionValue('R', "responsive", "K",
                          "[modular encoding] do Squeeze transform, 0=false, "
                          "1=true (default: true if lossy, false if lossless)",
                          &params.responsive, &ParseSigned, 1);

  cmdline->AddOptionFlag('v', "verbose",
                         "Verbose output (also applies to help).",
                         &params.verbose, &SetBooleanTrue);

  return true;
}

jxl::Status JxlCompressArgs::ValidateArgs(
    const tools::CommandLineParser& cmdline) {
  bool got_distance = cmdline.GetOption(opt_distance_id)->matched();
  bool got_target_size = cmdline.GetOption(opt_target_size_id)->matched();
  bool got_target_bpp = cmdline.GetOption(opt_target_bpp_id)->matched();
  bool got_quality = cmdline.GetOption(opt_quality_id)->matched();
  bool got_intensity_target =
      cmdline.GetOption(opt_intensity_target_id)->matched();

  if (got_quality) {
    params.modular_group_mode = true;
    if (quality < 100) jpeg_transcode = false;
    params.quality_pair.first = params.quality_pair.second = quality;
    default_settings = false;
  }

  if (progressive) {
    params.qprogressive_mode = true;
    params.progressive_dc = 1;
    params.responsive = 1;
    jpeg_transcode = false;
    default_settings = false;
  }
  if (got_target_size || got_target_bpp || got_intensity_target)
    default_settings = false;

  if (got_distance) {
    constexpr float butteraugli_min_dist = 0.1f;
    constexpr float butteraugli_max_dist = 15.0f;
    if (!(0 <= params.butteraugli_distance &&
          params.butteraugli_distance <= butteraugli_max_dist)) {
      fprintf(stderr, "Invalid/out of range distance, try 0 to %g.\n",
              butteraugli_max_dist);
      return false;
    }
    if (params.butteraugli_distance == 0) {
      // Use modular for lossless.
      params.modular_group_mode = true;
    } else if (params.butteraugli_distance < butteraugli_min_dist) {
      params.butteraugli_distance = butteraugli_min_dist;
    }
    default_settings = false;
  }

  if (got_target_bpp + got_target_size + got_distance + got_quality > 1) {
    fprintf(stderr,
            "You can specify only one of '--distance', '-q', "
            "'--target_bpp' and '--target_size'. They are all different ways"
            " to specify the image quality. When in doubt, use --distance."
            " It gives the most visually consistent results.\n");
    return false;
  }

  if (!saliency_map_filename.empty()) {
    if (!params.progressive_mode) {
      saliency_map_filename.clear();
      fprintf(stderr,
              "Warning: Specifying --saliency_map_filename only makes sense "
              "for --progressive_ac mode.\n");
    }
  }

  if (!params.file_in) {
    fprintf(stderr, "Missing input filename.\n");
    return false;
  }

  if (!cmdline.GetOption(opt_color_id)->matched()) {
    // default to RGB for lossless modular
    if (params.modular_group_mode) {
      if (params.quality_pair.first != 100 ||
          params.quality_pair.second != 100) {
        params.color_transform = jxl::ColorTransform::kXYB;
      } else {
        params.color_transform = jxl::ColorTransform::kNone;
      }
    }
  }

  if (params.near_lossless) {
    // Near-lossless assumes -R 0
    params.responsive = 0;
  }

  if (override_bitdepth > 32) {
    fprintf(stderr, "override_bitdepth must be <= 32\n");
    return false;
  }

  if (params.jpeg_quality > 100) {
    fprintf(stderr, "jpeg_quality must be <= 100\n");
    return false;
  }

  // User didn't override num_threads, so we have to compute a default, which
  // might fail, so only do so when necessary. Don't just check num_threads != 0
  // because the user may have set it to that.
  if (!cmdline.GetOption(opt_num_threads_id)->matched()) {
    jxl::ProcessorTopology topology;
    if (!jxl::DetectProcessorTopology(&topology)) {
      // We have seen sporadic failures caused by setaffinity_np.
      fprintf(stderr,
              "Failed to choose default num_threads; you can avoid this "
              "error by specifying a --num_threads N argument.\n");
      return false;
    }
    num_threads = topology.packages * topology.cores_per_package;
  }

  return true;
}

jxl::Status CompressJxl(jxl::ThreadPoolInternal* pool, JxlCompressArgs& args,
                        jxl::PaddedBytes* compressed, bool print_stats) {
  JXL_CHECK(pool);

  jxl::CodecInOut io;
  double decode_mps;
  JXL_RETURN_IF_ERROR(LoadAll(args, pool, &io, &decode_mps));
  const size_t pixels = io.xsize() * io.ysize();

  if (args.params.target_size > 0 || args.params.target_bitrate > 0) {
    // Slow iterative search for parameters that reach target bpp / size.
    SetParametersForSizeOrBitrate(pool, pixels, &args);
  }

  if (print_stats) PrintMode(pool, io, decode_mps, args);

  // Final/actual compression run (possibly repeated for benchmarking).
  jxl::AuxOut aux_out;
  if (args.inspector_image3f) {
    aux_out.SetInspectorImage3F(args.inspector_image3f);
  }
  SpeedStats stats;
  jxl::PassesEncoderState passes_encoder_state;
  for (size_t i = 0; i < args.num_reps; ++i) {
    const double t0 = jxl::Now();
    jxl::Status ok = false;
    if (args.params.pixels_to_jpeg_mode) {
#if JPEGXL_ENABLE_JPEG
      jxl::YCbCrChromaSubsampling subsample;
      if (args.params.jpeg_420) {
        uint8_t ss[3] = {2, 1, 1};
        JXL_CHECK(subsample.Set(ss, ss));
      }
      ok = jxl::EncodeImageJPG(&io, jxl::JpegEncoder::kLibJpeg,
                               args.params.jpeg_quality, subsample, pool,
                               compressed, jxl::DecodeTarget::kPixels);
#endif
    } else {
      if (io.Main().IsJPEG()) {
        // TODO(lode): automate this in the encoder. The encoder must in the
        // beginning choose to either do all in xyb, or all in non-xyb, write
        // that in the xyb_encoded header flag, and persistently keep that state
        // to check if every frame uses an allowed color transform.
        args.params.color_transform = io.Main().color_transform;
        ok = EncodeJpegToJpegXL(args.params, &io, &passes_encoder_state,
                                compressed, &aux_out, pool);
      } else {
        ok = EncodeFile(args.params, &io, &passes_encoder_state, compressed,
                        &aux_out, pool);
      }
    }
    if (!ok) {
      fprintf(stderr, "Failed to compress to %s.\n", ModeFromArgs(args));
      return false;
    }
    const double t1 = jxl::Now();
    stats.NotifyElapsed(t1 - t0);
    stats.SetImageSize(io.xsize(), io.ysize());
  }

  if (print_stats) {
    const double bpp =
        static_cast<double>(compressed->size() * jxl::kBitsPerByte) / pixels;
    fprintf(stderr, "Compressed to %zu bytes (%.3f bpp%s).\n",
            compressed->size(), bpp / io.frames.size(),
            io.frames.size() == 1 ? "" : "/frame");
    JXL_CHECK(stats.Print(args.num_threads));
    if (args.params.verbose) {
      aux_out.Print(1);
    }
  }

  return true;
}

}  // namespace tools
}  // namespace jpegxl
