// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/cjxl.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/codec.h"
#if JPEGXL_ENABLE_JPEG
#include "lib/extras/codec_jpg.h"
#endif

#include "lib/extras/time.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/cache_aligned.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "tools/args.h"
#include "tools/box/box.h"
#include "tools/cpu/cpu.h"
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
static inline bool ParsePhotonNoiseParameter(const char* arg, float* out) {
  return strncmp(arg, "ISO", 3) == 0 && ParseFloat(arg + 3, out) && *out > 0;
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
  *out_map = std::move(io_heatmap.Main().color()->Plane(0));
  return true;
}

// Search algorithm for modular mode instead of Butteraugli distance.
void SetModularQualityForBitrate(jxl::ThreadPoolInternal* pool,
                                 const size_t pixels, const double target_size,
                                 CompressArgs* args) {
  JXL_ASSERT(args->params.modular_mode);

  CompressArgs s = *args;  // Args for search.
  float quality = -100 + target_size * 8.0 / pixels * 50;
  if (quality > 100.f) quality = 100.f;
  s.params.target_size = 0;
  s.params.target_bitrate = 0;
  double best_loss = 1e99;
  float best_quality = quality;
  float best_below = -10000.f;
  float best_below_size = 0;
  float best_above = 200.f;
  float best_above_size = pixels * 15.f;

  jxl::CodecInOut io;
  double decode_mps = 0;

  if (!LoadAll(*args, pool, &io, &decode_mps)) {
    s.params.quality_pair = std::make_pair(quality, quality);
    printf("couldn't load image\n");
    return;
  }

  for (int i = 0; i < 10; ++i) {
    s.params.quality_pair = std::make_pair(quality, quality);
    jxl::PaddedBytes candidate;
    bool ok =
        CompressJxl(io, decode_mps, pool, s, &candidate, /*print_stats=*/false);
    if (!ok) {
      printf(
          "Compression error occurred during the search for best size."
          " Trying with quality %.1f\n",
          quality);
      break;
    }
    printf("Quality %.2f yields %6zu bytes, %.3f bpp.\n", quality,
           candidate.size(), candidate.size() * 8.0 / pixels);
    const double ratio = static_cast<double>(candidate.size()) / target_size;
    const double loss = std::abs(1.0 - ratio);
    if (best_loss > loss) {
      best_quality = quality;
      best_loss = loss;
      if (loss < 0.01f) break;
    }
    if (quality == 100.f && ratio < 1.f) break;  // can't spend more bits
    if (ratio > 1.f && quality < best_above) {
      best_above = quality;
      best_above_size = candidate.size();
    }
    if (ratio < 1.f && quality > best_below) {
      best_below = quality;
      best_below_size = candidate.size();
    }
    float t =
        (target_size - best_below_size) / (best_above_size - best_below_size);
    if (best_above > 100.f && ratio < 1.f) {
      quality = (quality + 105) / 2;
    } else if (best_above - best_below > 1000 && ratio > 1.f) {
      quality -= 1000;
    } else {
      quality = best_above * t + best_below * (1.f - t);
    }
    if (quality >= 100.f) quality = 100.f;
  }
  args->params.quality_pair = std::make_pair(best_quality, best_quality);
  args->params.target_bitrate = 0;
  args->params.target_size = 0;
}

void SetParametersForSizeOrBitrate(jxl::ThreadPoolInternal* pool,
                                   const size_t pixels, CompressArgs* args) {
  CompressArgs s = *args;  // Args for search.

  // If fixed size, convert to bitrate.
  if (s.params.target_size > 0) {
    s.params.target_bitrate = s.params.target_size * 8.0 / pixels;
    s.params.target_size = 0;
  }
  const double target_size = s.params.target_bitrate * (1 / 8.) * pixels;

  if (args->params.modular_mode) {
    SetModularQualityForBitrate(pool, pixels, target_size, args);
    return;
  }

  double dist = ApproximateDistanceForBPP(s.params.target_bitrate);
  s.params.target_bitrate = 0;
  double best_dist = 1.0;
  double best_loss = 1e99;

  jxl::CodecInOut io;
  double decode_mps = 0;
  if (!LoadAll(*args, pool, &io, &decode_mps)) {
    s.params.butteraugli_distance = static_cast<float>(dist);
    printf("couldn't load image\n");
    return;
  }

  for (int i = 0; i < 7; ++i) {
    s.params.butteraugli_distance = static_cast<float>(dist);
    jxl::PaddedBytes candidate;
    bool ok =
        CompressJxl(io, decode_mps, pool, s, &candidate, /*print_stats=*/false);
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

const char* ModeFromArgs(const CompressArgs& args) {
  if (args.jpeg_transcode) return "JPEG";
  if (args.params.modular_mode) return "Modular";
  return "VarDCT";
}

std::string QualityFromArgs(const CompressArgs& args) {
  char buf[100];
  if (args.jpeg_transcode) {
    snprintf(buf, sizeof(buf), "lossless transcode");
  } else if (args.params.modular_mode) {
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
  } else {
    snprintf(buf, sizeof(buf), "d%.3f", args.params.butteraugli_distance);
  }
  return buf;
}

void PrintMode(jxl::ThreadPoolInternal* pool, const jxl::CodecInOut& io,
               const double decode_mps, const CompressArgs& args) {
  const char* mode = ModeFromArgs(args);
  const char* speed = SpeedTierName(args.params.speed_tier);
  const std::string quality = QualityFromArgs(args);
  fprintf(stderr,
          "Read %zux%zu image, %.1f MP/s\n"
          "Encoding [%s%s, %s, %s",
          io.xsize(), io.ysize(), decode_mps,
          (args.use_container ? "Container | " : ""), mode, quality.c_str(),
          speed);
  if (args.use_container) {
    if (args.jpeg_transcode) fprintf(stderr, " | JPEG reconstruction data");
    if (!io.blobs.exif.empty())
      fprintf(stderr, " | %zu-byte Exif", io.blobs.exif.size());
    if (!io.blobs.xmp.empty())
      fprintf(stderr, " | %zu-byte XMP", io.blobs.xmp.size());
    if (!io.blobs.jumbf.empty())
      fprintf(stderr, " | %zu-byte JUMBF", io.blobs.jumbf.size());
  }
  fprintf(stderr, "], %zu threads.\n", pool->NumWorkerThreads());
}

}  // namespace

void CompressArgs::AddCommandLineOptions(CommandLineParser* cmdline) {
  // Positional arguments.
  cmdline->AddPositionalOption("INPUT", /* required = */ true,
                               "the input can be PNG"
#if JPEGXL_ENABLE_APNG
                               ", APNG"
#endif
#if JPEGXL_ENABLE_GIF
                               ", GIF"
#endif
#if JPEGXL_ENABLE_JPEG
                               ", JPEG"
#endif
#if JPEGXL_ENABLE_EXR
                               ", EXR"
#endif
                               ", PPM, PFM, or PGX",
                               &file_in);
  cmdline->AddPositionalOption(
      "OUTPUT", /* required = */ true,
      "the compressed JXL output file (can be omitted for benchmarking)",
      &file_out);

  // Flags.
  // TODO(lode): also add options to add exif/xmp/other metadata in the
  // container.
  // TODO(lode): decide on good name for this flag: box, container, bmff, ...
  cmdline->AddOptionFlag(
      '\0', "container",
      "Always encode using container format (default: only if needed)",
      &use_container, &SetBooleanTrue, 1);

  cmdline->AddOptionFlag('\0', "strip",
                         "Do not encode using container format (strips "
                         "Exif/XMP/JPEG bitstream reconstruction data)",
                         &no_container, &SetBooleanTrue, 2);

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
      "Quality setting (is remapped to --distance). Range: -inf .. 100.\n"
      "    100 = mathematically lossless. Default for already-lossy input "
      "(JPEG/GIF).\n    Positive quality values roughly match libjpeg quality.",
      &quality, &ParseFloat);

  cmdline->AddOptionValue(
      'e', "effort", "EFFORT",
      "Encoder effort setting. Range: 1 .. 9.\n"
      "    Default: 7. Higher number is more effort (slower).",
      &params.speed_tier, &ParseSpeedTier);

  cmdline->AddOptionValue(
      's', "speed", "ANIMAL",
      "Deprecated synonym for --effort. Valid values are:\n"
      "    lightning (1), thunder, falcon, cheetah, hare, wombat, squirrel, "
      "kitten, tortoise (9)\n"
      "    Default: squirrel. Values are in order from faster to slower.\n",
      &params.speed_tier, &ParseSpeedTier, 2);

  cmdline->AddOptionValue('\0', "faster_decoding", "AMOUNT",
                          "Favour higher decoding speed. 0 = default, higher "
                          "values give higher speed at the expense of quality",
                          &params.decoding_speed_tier, &ParseUnsigned, 2);

  cmdline->AddOptionFlag('p', "progressive",
                         "Enable progressive/responsive decoding.",
                         &progressive, &SetBooleanTrue);

  cmdline->AddOptionFlag('\0', "premultiply",
                         "Force premultiplied (associated) alpha.",
                         &force_premultiplied, &SetBooleanTrue, 1);
  cmdline->AddOptionValue('\0', "keep_invisible", "0|1",
                          "force disable/enable preserving color of invisible "
                          "pixels (default: 1 if lossless, 0 if lossy).",
                          &params.keep_invisible, &ParseOverride, 1);

  cmdline->AddOptionFlag('\0', "centerfirst",
                         "Put center groups first in the compressed file.",
                         &params.centerfirst, &SetBooleanTrue, 1);

  cmdline->AddOptionValue('\0', "center_x", "0..XSIZE",
                          "Put center groups first in the compressed file.",
                          &params.center_x, &ParseUnsigned, 1);
  cmdline->AddOptionValue('\0', "center_y", "0..YSIZE",
                          "Put center groups first in the compressed file.",
                          &params.center_y, &ParseUnsigned, 1);

  // Flags.
  cmdline->AddOptionFlag('\0', "progressive_ac",
                         "Use the progressive mode for AC.",
                         &params.progressive_mode, &SetBooleanTrue, 1);
  cmdline->AddOptionFlag('\0', "qprogressive_ac",
                         "Use the progressive mode for AC.",
                         &params.qprogressive_mode, &SetBooleanTrue, 1);
  cmdline->AddOptionValue('\0', "progressive_dc", "num_dc_frames",
                          "Use progressive mode for DC.",
                          &params.progressive_dc, &ParseSigned, 1);
  cmdline->AddOptionFlag('m', "modular",
                         "Use the modular mode (lossy / lossless).",
                         &params.modular_mode, &SetBooleanTrue, 1);
  cmdline->AddOptionFlag('\0', "use_new_heuristics",
                         "use new and not yet ready encoder heuristics",
                         &params.use_new_heuristics, &SetBooleanTrue, 2);

  // JPEG modes: parallel Brunsli, pixels to JPEG, or JPEG to Brunsli
  cmdline->AddOptionFlag('j', "jpeg_transcode",
                         "Do lossy transcode of input JPEG file (decode to "
                         "pixels instead of doing lossless transcode).",
                         &jpeg_transcode, &SetBooleanFalse, 1);

  opt_num_threads_id = cmdline->AddOptionValue(
      '\0', "num_threads", "N", "number of worker threads (zero = none).",
      &num_threads, &ParseUnsigned, 1);
  cmdline->AddOptionValue('\0', "num_reps", "N", "how many times to compress.",
                          &num_reps, &ParseUnsigned, 1);

  cmdline->AddOptionValue('\0', "noise", "0|1",
                          "force disable/enable noise generation.",
                          &params.noise, &ParseOverride, 1);
  cmdline->AddOptionValue(
      '\0', "photon_noise", "ISO3200",
      "Set the noise to approximately what it would be at a given nominal "
      "exposure on a 35mm camera. For formats other than 35mm, or when the "
      "whole sensor was not used, you can multiply the ISO value by the "
      "equivalence ratio squared, for example by 2.25 for an APS-C camera.",
      &params.photon_noise_iso, &ParsePhotonNoiseParameter, 1);
  cmdline->AddOptionValue('\0', "dots", "0|1",
                          "force disable/enable dots generation.", &params.dots,
                          &ParseOverride, 1);
  cmdline->AddOptionValue('\0', "patches", "0|1",
                          "force disable/enable patches generation.",
                          &params.patches, &ParseOverride, 1);
  cmdline->AddOptionValue('\0', "resampling", "1|2|4|8",
                          "Subsample all color channels by this factor.",
                          &params.resampling, &ParseUnsigned, 1);
  cmdline->AddOptionValue(
      '\0', "ec_resampling", "1|2|4|8",
      "Subsample all extra channels by this factor. If this value is smaller "
      "than the resampling of color channels, it will be increased to match.",
      &params.ec_resampling, &ParseUnsigned, 2);
  cmdline->AddOptionFlag('\0', "already_downsampled",
                         "Do not downsample the given input before encoding, "
                         "but still signal that the decoder should upsample.",
                         &params.already_downsampled, &SetBooleanTrue, 2);

  cmdline->AddOptionValue(
      '\0', "epf", "-1..3",
      "Edge preserving filter level (-1 = choose based on quality, default)",
      &params.epf, &ParseSigned, 1);

  cmdline->AddOptionValue('\0', "gaborish", "0|1",
                          "force disable/enable gaborish.", &params.gaborish,
                          &ParseOverride, 1);

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
      "[modular encoding] fraction of pixels used to learn MA trees "
      "(default=0.5, try 0 for no MA and fast decode)",
      &params.options.nb_repeats, &ParseFloat, 2);

  cmdline->AddOptionValue(
      'C', "colorspace", "K",
      ("[modular encoding] color transform: 0=RGB, 1=YCoCg, "
       "2-37=RCT (default: try several, depending on speed)"),
      &params.colorspace, &ParseSigned, 1);

  opt_m_group_size_id = cmdline->AddOptionValue(
      'g', "group-size", "K",
      ("[modular encoding] set group size to 128 << K "
       "(default: 1 or 2)"),
      &params.modular_group_size_shift, &ParseUnsigned, 1);

  cmdline->AddOptionValue(
      'P', "predictor", "K",
      "[modular encoding] predictor(s) to use: 0=zero, "
      "1=left, 2=top, 3=avg0, 4=select, 5=gradient, 6=weighted, "
      "7=topright, 8=topleft, 9=leftleft, 10=avg1, 11=avg2, 12=avg3, "
      "13=toptop predictive average "
      "14=mix 5 and 6, 15=mix everything. Default 14, at slowest speed "
      "default 15",
      &params.options.predictor, &ParsePredictor, 1);

  cmdline->AddOptionValue(
      'E', "extra-properties", "K",
      "[modular encoding] number of extra MA tree properties to use",
      &params.options.max_properties, &ParseSigned, 2);

  cmdline->AddOptionValue('\0', "palette", "K",
                          "[modular encoding] use a palette if image has at "
                          "most K colors (default: 1024)",
                          &params.palette_colors, &ParseSigned, 1);

  cmdline->AddOptionFlag(
      '\0', "lossy-palette",
      "[modular encoding] quantize to a palette that has fewer entries than "
      "would be necessary for perfect preservation; for the time being, it is "
      "recommended to set --palette=0 with this option to use the default "
      "palette only",
      &params.lossy_palette, &SetBooleanTrue, 1);

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

  cmdline->AddOptionFlag('V', "version", "Print version number and exit",
                         &version, &SetBooleanTrue, 1);
  cmdline->AddOptionFlag('\0', "quiet", "Be more silent", &quiet,
                         &SetBooleanTrue, 1);
  cmdline->AddOptionValue('\0', "print_profile", "0|1",
                          "Print timing information before exiting",
                          &print_profile, &ParseOverride, 1);

  cmdline->AddOptionFlag(
      'v', "verbose",
      "Verbose output; can be repeated, also applies to help (!).",
      &params.verbose, &SetBooleanTrue);
}

jxl::Status CompressArgs::ValidateArgs(const CommandLineParser& cmdline) {
  params.file_in = file_in;
  params.file_out = file_out;

  if (file_in == nullptr) {
    fprintf(stderr, "Missing INPUT filename.\n");
    return false;
  }

  bool got_distance = cmdline.GetOption(opt_distance_id)->matched();
  bool got_target_size = cmdline.GetOption(opt_target_size_id)->matched();
  bool got_target_bpp = cmdline.GetOption(opt_target_bpp_id)->matched();
  bool got_quality = cmdline.GetOption(opt_quality_id)->matched();
  bool got_intensity_target =
      cmdline.GetOption(opt_intensity_target_id)->matched();

  if (got_quality) {
    default_settings = false;
    if (quality < 100) jpeg_transcode = false;
    // Quality settings roughly match libjpeg qualities.
    if (quality < 7 || quality == 100 || params.modular_mode) {
      if (jpeg_transcode == false) params.modular_mode = true;
      // Internal modular quality to roughly match VarDCT size.
      if (quality < 7) {
        params.quality_pair.first = params.quality_pair.second =
            std::min(35 + (quality - 7) * 3.0f, 100.0f);
      } else {
        params.quality_pair.first = params.quality_pair.second =
            std::min(35 + (quality - 7) * 65.f / 93.f, 100.0f);
      }
    } else {
      if (quality >= 30) {
        params.butteraugli_distance = 0.1 + (100 - quality) * 0.09;
      } else {
        params.butteraugli_distance =
            6.4 + pow(2.5, (30 - quality) / 5.0f) / 6.25f;
      }
    }
  }
  if (params.resampling > 1 && !params.already_downsampled)
    jpeg_transcode = false;

  if (progressive) {
    params.qprogressive_mode = true;
    params.responsive = 1;
    default_settings = false;
  }
  if (got_target_size || got_target_bpp || got_intensity_target) {
    default_settings = false;
  }

  if (params.progressive_dc < -1 || params.progressive_dc > 2) {
    fprintf(stderr, "Invalid/out of range progressive_dc (%d), try -1 to 2.\n",
            params.progressive_dc);
    return false;
  }

  if (got_distance) {
    constexpr float butteraugli_min_dist = 0.1f;
    constexpr float butteraugli_max_dist = 15.0f;
    if (!(0 <= params.butteraugli_distance &&
          params.butteraugli_distance <= butteraugli_max_dist)) {
      fprintf(stderr, "Invalid/out of range distance, try 0 to %g.\n",
              butteraugli_max_dist);
      return false;
    }
    if (params.butteraugli_distance > 0) jpeg_transcode = false;
    if (params.butteraugli_distance == 0) {
      // Use modular for lossless.
      if (jpeg_transcode == false) params.modular_mode = true;
    } else if (params.butteraugli_distance < butteraugli_min_dist) {
      params.butteraugli_distance = butteraugli_min_dist;
    }
    default_settings = false;
  }

  if (got_target_bpp || got_target_size) {
    jpeg_transcode = false;
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
    if (params.modular_mode) {
      if (params.quality_pair.first != 100 ||
          params.quality_pair.second != 100) {
        params.color_transform = jxl::ColorTransform::kXYB;
      } else {
        params.color_transform = jxl::ColorTransform::kNone;
      }
    }
  }

  if (override_bitdepth > 32) {
    fprintf(stderr, "override_bitdepth must be <= 32\n");
    return false;
  }

  if (params.epf > 3) {
    fprintf(stderr, "--epf must be in the 0..3 range\n");
    return false;
  }

  // User didn't override num_threads, so we have to compute a default, which
  // might fail, so only do so when necessary. Don't just check num_threads != 0
  // because the user may have set it to that.
  if (!cmdline.GetOption(opt_num_threads_id)->matched()) {
    cpu::ProcessorTopology topology;
    if (!cpu::DetectProcessorTopology(&topology)) {
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

jxl::Status CompressArgs::ValidateArgsAfterLoad(
    const CommandLineParser& cmdline, const jxl::CodecInOut& io) {
  if (!ValidateArgs(cmdline)) return false;
  bool got_m_group_size = cmdline.GetOption(opt_m_group_size_id)->matched();
  if (params.modular_mode && !got_m_group_size) {
    // Default modular group size: set to 512 if 256 would be silly
    const size_t kThinImageThr = 256 + 64;
    const size_t kSmallImageThr = 256 + 128;
    if (io.xsize() < kThinImageThr || io.ysize() < kThinImageThr ||
        (io.xsize() < kSmallImageThr && io.ysize() < kSmallImageThr)) {
      params.modular_group_size_shift = 2;
    }
  }
  if (!io.blobs.exif.empty() || !io.blobs.xmp.empty() ||
      !io.blobs.jumbf.empty() || !io.blobs.iptc.empty() || jpeg_transcode) {
    use_container = true;
  }
  if (no_container) use_container = false;
  if (jpeg_transcode && params.modular_mode) {
    fprintf(stderr,
            "Error: cannot do lossless JPEG transcode in modular mode.\n");
    return false;
  }
  if (jpeg_transcode) {
    if (params.progressive_mode || params.qprogressive_mode ||
        params.progressive_dc > 0) {
      fprintf(stderr,
              "Error: progressive lossless JPEG transcode is not yet "
              "implemented.\n");
      return false;
    }
  }
  return true;
}

jxl::Status LoadAll(CompressArgs& args, jxl::ThreadPoolInternal* pool,
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
  if (args.jpeg_transcode) args.params.butteraugli_distance = 0;

  if (input_codec == jxl::Codec::kGIF && args.default_settings) {
    args.params.modular_mode = true;
    args.params.quality_pair.first = args.params.quality_pair.second = 100;
  }
  if (args.params.modular_mode && args.params.quality_pair.first < 100) {
    if (io->metadata.m.bit_depth.floating_point_sample) {
      // for lossy modular, pretend pfm/exr is integer data
      io->metadata.m.SetUintSamples(12);
    }
  }
  if (args.override_bitdepth != 0) {
    if (args.override_bitdepth == 32) {
      io->metadata.m.SetFloat32Samples();
    } else {
      io->metadata.m.SetUintSamples(args.override_bitdepth);
    }
  }
  if (args.force_premultiplied) {
    io->PremultiplyAlpha();
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

  const double t1 = jxl::Now();
  const size_t pixels = io->xsize() * io->ysize();
  *decode_mps = pixels * io->frames.size() * 1E-6 / (t1 - t0);

  return true;
}

jxl::Status CompressJxl(jxl::CodecInOut& io, double decode_mps,
                        jxl::ThreadPoolInternal* pool, CompressArgs& args,
                        jxl::PaddedBytes* compressed, bool print_stats) {
  JXL_CHECK(pool);

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
  if (args.params.use_new_heuristics) {
    passes_encoder_state.heuristics =
        jxl::make_unique<jxl::FastEncoderHeuristics>();
  }
  for (size_t i = 0; i < args.num_reps; ++i) {
    const double t0 = jxl::Now();
    jxl::Status ok = false;
    if (io.Main().IsJPEG()) {
      // TODO(lode): automate this in the encoder. The encoder must in the
      // beginning choose to either do all in xyb, or all in non-xyb, write
      // that in the xyb_encoded header flag, and persistently keep that state
      // to check if every frame uses an allowed color transform.
      args.params.color_transform = io.Main().color_transform;
    }
    ok = EncodeFile(args.params, &io, &passes_encoder_state, compressed,
                    &aux_out, pool);
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
