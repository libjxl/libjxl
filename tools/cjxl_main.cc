// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: This encoder binary does extensive flag-validity checking (in
// order to produce meaningful error messages), and on top of that
// checks all libjxl C API call return values. The downside of this
// vs. libjxl providing meaningful error messages is that a change to
// the accepted range of a flag-specified parameter in libjxl will
// also require a change to the range-check here. The advantage is
// that this minimizes the size of libjxl.

#include <stdint.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "jxl/codestream_header.h"
#include "jxl/encode.h"
#include "jxl/encode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "jxl/types.h"
#include "lib/extras/dec/apng.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/exr.h"
#include "lib/extras/dec/gif.h"
#include "lib/extras/dec/jpg.h"
#include "lib/extras/dec/pgx.h"
#include "lib/extras/dec/pnm.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/exif.h"
#include "lib/jxl/size_constraints.h"
#include "tools/args.h"
#include "tools/cmdline.h"
#include "tools/codec_config.h"
#include "tools/file_io.h"
#include "tools/speed_stats.h"

namespace jpegxl {
namespace tools {

namespace {
inline bool ParsePhotonNoiseParameter(const char* arg, float* out) {
  return ParseFloat(arg, out) && *out >= 0;
}
inline bool ParseIntensityTarget(const char* arg, float* out) {
  return ParseFloat(arg, out) && *out > 0;
}
}  // namespace

enum CjxlRetCode : int {
  OK = 0,
  ERR_PARSE,
  ERR_INVALID_ARG,
  ERR_LOAD_INPUT,
  ERR_INVALID_INPUT,
  ERR_ENCODING,
  ERR_CONTAINER,
  ERR_WRITE,
  DROPPED_JBRD,
};

struct CompressArgs {
  // CompressArgs() = default;
  void AddCommandLineOptions(CommandLineParser* cmdline) {
    // Positional arguments.
    cmdline->AddPositionalOption("INPUT", /* required = */ true,
                                 "the input can be "
#if JPEGXL_ENABLE_APNG
                                 "PNG, APNG, "
#endif
#if JPEGXL_ENABLE_GIF
                                 "GIF, "
#endif
#if JPEGXL_ENABLE_JPEG
                                 "JPEG, "
#else
                                 "JPEG (lossless recompression only), "
#endif
#if JPEGXL_ENABLE_EXR
                                 "EXR, "
#endif
                                 "PPM, PFM, or PGX",
                                 &file_in);
    cmdline->AddPositionalOption("OUTPUT", /* required = */ true,
                                 "the compressed JXL output file", &file_out);

    // Flags.
    // TODO(lode): also add options to add exif/xmp/other metadata in the
    // container.
    cmdline->AddOptionValue('\0', "container", "0|1",
                            "0 = Do not encode using container format (strip "
                            "Exif/XMP/JPEG bitstream reconstruction data)."
                            "1 = Force using container format \n"
                            "(default: use only if needed).\n",
                            &container, &ParseOverride, 1);

    cmdline->AddOptionValue(
        '\0', "jpeg_store_metadata", "0|1",
        ("If --lossless_jpeg=1, store JPEG reconstruction "
         "metadata in the JPEG XL container "
         "(for lossless reconstruction of the JPEG codestream)."
         "(default: 1)"),
        &jpeg_store_metadata, &ParseUnsigned, 2);

    // Target distance/size/bpp
    opt_distance_id = cmdline->AddOptionValue(
        'd', "distance", "maxError",
        "Max. butteraugli distance, lower = higher quality.\n"
        "    0.0 = mathematically lossless. Default for already-lossy input "
        "(JPEG/GIF).\n"
        "    1.0 = visually lossless. Default for other input.\n"
        "    Recommended range: 0.5 .. 3.0. Allowed range: 0.0 ... 25.0.\n"
        "    Mutually exclusive with --quality.",
        &distance, &ParseFloat);

    opt_alpha_distance_id = cmdline->AddOptionValue(
        'a', "alpha_distance", "maxError",
        "Max. butteraugli distance for the alpha channel, lower = higher "
        "quality.\n"
        "    0.0 = mathematically lossless. 1.0 = visually lossless.\n"
        "    Default is to use the same value as for the color image.\n"
        "    Recommended range: 0.5 .. 3.0. Allowed range: 0.0 ... 25.0.",
        &alpha_distance, &ParseFloat);

    // High-level options
    opt_quality_id = cmdline->AddOptionValue(
        'q', "quality", "QUALITY",
        "Quality setting (is remapped to --distance)."
        "    100 = mathematically lossless. Default for already-lossy input "
        "(JPEG/GIF).\n"
        "    Other input gets encoded as per --distance default,\n"
        "    which corresponds to quality 90.\n"
        "    Quality values roughly match libjpeg quality.\n"
        "    Recommended range: 68 .. 96. Allowed range: 0 .. 100.\n"
        "    Mutually exclusive with --distance.",
        &quality, &ParseFloat);

    cmdline->AddOptionValue(
        'e', "effort", "EFFORT",
        "Encoder effort setting. Range: 1 .. 9.\n"
        "     Default: 7. Higher number is more effort (slower).",
        &effort, &ParseUnsigned, -1);

    cmdline->AddOptionValue(
        '\0', "compress_boxes", "0|1",
        "Disable/enable Brotli compression for metadata boxes "
        "(not provided = default, 0 = disable, 1 = enable).",
        &compress_boxes, &ParseOverride, 1);

    cmdline->AddOptionValue(
        '\0', "brotli_effort", "B_EFFORT",
        "Brotli effort setting. Range: 0 .. 11.\n"
        "    Default: 9. Higher number is more effort (slower).",
        &brotli_effort, &ParseUnsigned, -1);

    cmdline->AddOptionValue(
        '\0', "faster_decoding", "0|1|2|3|4",
        "Favour higher decoding speed. 0 = default, higher "
        "values give higher speed at the expense of quality",
        &faster_decoding, &ParseUnsigned, 2);

    cmdline->AddOptionFlag('p', "progressive",
                           "Enable progressive/responsive decoding.",
                           &progressive, &SetBooleanTrue);

    cmdline->AddOptionValue('\0', "premultiply", "-1|0|1",
                            "Force premultiplied (associated) alpha.",
                            &premultiply, &ParseSigned, 1);

    cmdline->AddOptionValue(
        '\0', "keep_invisible", "0|1",
        "force disable/enable preserving color of invisible "
        "pixels (default: 1 if lossless, 0 if lossy).",
        &keep_invisible, &ParseOverride, 1);

    cmdline->AddOptionValue(
        '\0', "group_order", "0|1",
        "Order in which 256x256 groups are stored "
        "in the codestream for progressive rendering. "
        "Value not provided means 'encoder default', 0 means 'scanline order', "
        "1 means 'center-first order'.",
        &group_order, &ParseOverride, 1);

    cmdline->AddOptionValue(
        '\0', "center_x", "0..XSIZE",
        "Determines the horizontal position of center for the center-first "
        "group order. The value -1 means 'use the middle of the image', "
        "other values [0..xsize) set this to a particular coordinate.",
        &center_x, &ParseInt64, 1);

    cmdline->AddOptionValue(
        '\0', "center_y", "0..YSIZE",
        "Determines the vertical position of center for the center-first "
        "group order. The value -1 means 'use the middle of the image', "
        "other values [0..ysize) set this to a particular coordinate.",
        &center_y, &ParseInt64, 1);

    // Flags.
    cmdline->AddOptionFlag('\0', "progressive_ac",
                           "Use the progressive mode for AC.", &progressive_ac,
                           &SetBooleanTrue, 1);

    cmdline->AddOptionFlag(
        '\0', "qprogressive_ac",
        "Use the progressive mode for AC with shift quantization.",
        &qprogressive_ac, &SetBooleanTrue, 1);

    cmdline->AddOptionValue(
        '\0', "progressive_dc", "num_dc_frames",
        "Progressive-DC setting. Valid values are: -1, 0, 1, 2.",
        &progressive_dc, &ParseInt64, 1);

    cmdline->AddOptionValue(
        'm', "modular", "0|1",
        "Use modular mode (not provided = encoder chooses, 0 = enforce VarDCT, "
        "1 = enforce modular mode).",
        &modular, &ParseOverride, 1);

    // JPEG modes: parallel Brunsli, pixels to JPEG, or JPEG to Brunsli
    opt_lossless_jpeg_id = cmdline->AddOptionValue(
        'j', "lossless_jpeg", "0|1",
        "If the input is JPEG, losslessly transcode JPEG, "
        "rather than using reencode pixels.",
        &lossless_jpeg, &ParseUnsigned, 1);

    cmdline->AddOptionValue(
        '\0', "jpeg_reconstruction_cfl", "0|1",
        "Enable/disable chroma-from-luma (CFL) for lossless "
        "JPEG reconstruction.",
        &jpeg_reconstruction_cfl, &ParseOverride, 2);

    cmdline->AddOptionValue(
        '\0', "num_threads", "N",
        "Number of worker threads (-1 == use machine default, "
        "0 == do not use multithreading).",
        &num_threads, &ParseSigned, 1);

    cmdline->AddOptionValue('\0', "num_reps", "N",
                            "How many times to compress. (For benchmarking).",
                            &num_reps, &ParseUnsigned, 1);

    cmdline->AddOptionFlag('\0', "disable_output",
                           "No output file will be written (for benchmarking)",
                           &disable_output, &SetBooleanTrue, 1);

    cmdline->AddOptionValue(
        '\0', "photon_noise_iso", "3200",
        "Adds noise to the image emulating photographic film noise. "
        "The higher the given number, the grainier the image will be. "
        "As an example, a value of 100 gives low noise whereas a value "
        "of 3200 gives a lot of noise. The default value is 0.",
        &photon_noise_iso, &ParsePhotonNoiseParameter, 1);

    cmdline->AddOptionValue(
        '\0', "dots", "0|1",
        "Force disable/enable dots generation. "
        "(not provided = default, 0 = disable, 1 = enable).",
        &dots, &ParseOverride, 1);

    cmdline->AddOptionValue(
        '\0', "patches", "0|1",
        "Force disable/enable patches generation. "
        "(not provided = default, 0 = disable, 1 = enable).",
        &patches, &ParseOverride, 1);

    cmdline->AddOptionValue(
        '\0', "resampling", "-1|1|2|4|8",
        "Resampling for extra channels. Default of -1 applies resampling only "
        "for low quality. Value 1 does no downsampling (1x1), 2 does 2x2 "
        "downsampling, 4 is for 4x4 downsampling, and 8 for 8x8 downsampling.",
        &resampling, &ParseInt64, 0);

    cmdline->AddOptionValue(
        '\0', "ec_resampling", "-1|1|2|4|8",
        "Resampling for extra channels. Default of -1 applies resampling only "
        "for low quality. Value 1 does no downsampling (1x1), 2 does 2x2 "
        "downsampling, 4 is for 4x4 downsampling, and 8 for 8x8 downsampling.",
        &ec_resampling, &ParseInt64, 2);

    cmdline->AddOptionFlag('\0', "already_downsampled",
                           "Do not downsample the given input before encoding, "
                           "but still signal that the decoder should upsample.",
                           &already_downsampled, &SetBooleanTrue, 2);

    cmdline->AddOptionValue(
        '\0', "epf", "-1|0|1|2|3",
        "Edge preserving filter level, -1 to 3. "
        "Value -1 means: default (encoder chooses), 0 to 3 set a strength.",
        &epf, &ParseInt64, 1);

    cmdline->AddOptionValue(
        '\0', "gaborish", "0|1",
        "Force disable/enable the gaborish filter. "
        "(not provided = default, 0 = disable, 1 = enable).",
        &gaborish, &ParseOverride, 1);

    cmdline->AddOptionValue(
        '\0', "intensity_target", "N",
        "Upper bound on the intensity level present in the image in nits. "
        "Leaving this set to its default of 0 lets libjxl choose a sensible "
        "default "
        "value based on the color encoding.",
        &intensity_target, &ParseIntensityTarget, 1);

    cmdline->AddOptionValue(
        'x', "dec-hints", "key=value",
        "color_space indicates the ColorEncoding, see Description();\n"
        "icc_pathname refers to a binary file containing an ICC profile.",
        &color_hints, &ParseAndAppendKeyValue, 1);

    cmdline->AddOptionValue(
        '\0', "override_bitdepth", "0=use from image, 1-32=override",
        "If nonzero, store the given bit depth in the JPEG XL file metadata"
        " (1-32), instead of using the bit depth from the original input"
        " image.",
        &override_bitdepth, &ParseUnsigned, 2);

    // modular mode options
    cmdline->AddOptionValue(
        'I', "iterations", "F",
        "[modular encoding] Fraction of pixels used to learn MA trees as "
        "a percentage. -1 = default, 0 = no MA and fast decode, 50 = "
        "default value, 100 = all."
        "Higher values use more encoder memory.",
        &modular_ma_tree_learning_percent, &ParseFloat, 2);

    cmdline->AddOptionValue(
        'C', "modular_colorspace", "K",
        ("[modular encoding] color transform: -1=default, 0=RGB (none), "
         "1-41=RCT (6=YCoCg, default: try several, depending on speed)"),
        &modular_colorspace, &ParseInt64, 1);

    opt_modular_group_size_id = cmdline->AddOptionValue(
        'g', "modular_group_size", "K",
        "[modular encoding] group size: -1 == default. 0 => 128, "
        "1 => 256, 2 => 512, 3 => 1024",
        &modular_group_size, &ParseInt64, 1);

    cmdline->AddOptionValue(
        'P', "modular_predictor", "K",
        "[modular encoding] predictor(s) to use: 0=zero, "
        "1=left, 2=top, 3=avg0, 4=select, 5=gradient, 6=weighted, "
        "7=topright, 8=topleft, 9=leftleft, 10=avg1, 11=avg2, 12=avg3, "
        "13=toptop predictive average "
        "14=mix 5 and 6, 15=mix everything. If unset, uses default 14, "
        "at slowest speed default 15.",
        &modular_predictor, &ParseInt64, 1);

    cmdline->AddOptionValue(
        'E', "modular_nb_prev_channels", "K",
        "[modular encoding] number of extra MA tree properties to use",
        &modular_nb_prev_channels, &ParseInt64, 2);

    cmdline->AddOptionValue(
        '\0', "modular_palette_colors", "K",
        "[modular encoding] Use color palette if number of colors is smaller "
        "than or equal to this, or -1 to use the encoder default.",
        &modular_palette_colors, &ParseInt64, 1);

    cmdline->AddOptionFlag(
        '\0', "modular_lossy_palette",
        "[modular encoding] quantize to a palette that has fewer entries than "
        "would be necessary for perfect preservation; for the time being, it "
        "is "
        "recommended to set --palette=0 with this option to use the default "
        "palette only",
        &modular_lossy_palette, &SetBooleanTrue, 1);

    cmdline->AddOptionValue(
        'X', "pre-compact", "PERCENT",
        "[modular encoding] Use Global channel palette if the number of "
        "colors is smaller than this percentage of range. "
        "Use 0-100 to set an explicit percentage, -1 to use the encoder "
        "default.",
        &modular_channel_colors_global_percent, &ParseFloat, 2);

    cmdline->AddOptionValue(
        'Y', "post-compact", "PERCENT",
        "[modular encoding] Use Local (per-group) channel palette if the "
        "number "
        "of colors is smaller than this percentage of range. Use 0-100 to set "
        "an explicit percentage, -1 to use the encoder default.",
        &modular_channel_colors_group_percent, &ParseFloat, 2);

    cmdline->AddOptionValue('\0', "codestream_level", "K",
                            "The codestream level. Either `-1`, `5` or `10`.",
                            &codestream_level, &ParseInt64, 2);

    opt_responsive_id = cmdline->AddOptionValue(
        'R', "responsive", "K",
        "[modular encoding] do Squeeze transform, 0=false, "
        "1=true (default: true if lossy, false if lossless)",
        &responsive, &ParseInt64, 1);

    cmdline->AddOptionFlag('V', "version",
                           "Print encoder library version number and exit.",
                           &version, &SetBooleanTrue, 1);

    cmdline->AddOptionFlag('\0', "quiet", "Be more silent", &quiet,
                           &SetBooleanTrue, 1);

    cmdline->AddOptionValue(
        '\0', "frame_indexing", "string",
        // TODO(tfish): Add a more convenient vanilla alternative.
        "If non-empty, a string matching '^(0*|1[01]*)'. If this string has a "
        "'1' in i-th position, then the i-th frame will be indexed in "
        "the frame index box.",
        &frame_indexing, &ParseString, 1);

    cmdline->AddOptionFlag(
        '\0', "allow_expert_options",
        "Allow specifying advanced options; at the moment, this allows setting "
        "effort to 10, for somewhat better lossless compression at the cost of "
        "a massive speed hit.",
        &allow_expert_options, &SetBooleanTrue, 2);

    cmdline->AddOptionFlag(
        'v', "verbose",
        "Verbose output; can be repeated, also applies to help (!).", &verbose,
        &SetBooleanTrue);
  }

  // Common flags.
  bool version = false;
  jxl::Override container = jxl::Override::kDefault;
  bool quiet = false;
  bool disable_output = false;

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  jxl::Override print_profile = jxl::Override::kDefault;

  // Decoding source image flags
  jxl::extras::ColorHints color_hints;

  // JXL flags
  size_t override_bitdepth = 0;
  int32_t num_threads = -1;
  size_t num_reps = 1;
  float intensity_target = 0;

  // Whether to perform lossless transcoding with kVarDCT or kJPEG encoding.
  // If true, attempts to load JPEG coefficients instead of pixels.
  // Reset to false if input image is not a JPEG.
  size_t lossless_jpeg = 1;

  size_t jpeg_store_metadata = 1;

  float quality = -1001.f;  // Default to lossless if input is already lossy,
                            // or to VarDCT otherwise.
  bool verbose = false;
  bool progressive = false;
  bool progressive_ac = false;
  bool qprogressive_ac = false;
  int64_t progressive_dc = -1;
  bool modular_lossy_palette = false;
  int32_t premultiply = -1;
  bool already_downsampled = false;
  jxl::Override jpeg_reconstruction_cfl = jxl::Override::kDefault;
  jxl::Override modular = jxl::Override::kDefault;
  jxl::Override keep_invisible = jxl::Override::kDefault;
  jxl::Override dots = jxl::Override::kDefault;
  jxl::Override patches = jxl::Override::kDefault;
  jxl::Override gaborish = jxl::Override::kDefault;
  jxl::Override group_order = jxl::Override::kDefault;
  jxl::Override compress_boxes = jxl::Override::kDefault;

  size_t faster_decoding = 0;
  int64_t resampling = -1;
  int64_t ec_resampling = -1;
  int64_t epf = -1;
  int64_t center_x = -1;
  int64_t center_y = -1;
  int64_t modular_group_size = -1;
  int64_t modular_predictor = -1;
  int64_t modular_colorspace = -1;
  float modular_channel_colors_global_percent = -1.f;
  float modular_channel_colors_group_percent = -1.f;
  int64_t modular_palette_colors = -1;
  int64_t modular_nb_prev_channels = -1;
  float modular_ma_tree_learning_percent = -1.f;
  float photon_noise_iso = 0;
  int64_t codestream_level = -1;
  int64_t responsive = -1;
  float distance = 1.0;
  float alpha_distance = 1.0;
  size_t effort = 7;
  size_t brotli_effort = 9;
  std::string frame_indexing;

  bool allow_expert_options = false;

  // Will get passed on to AuxOut.
  // jxl::InspectorImage3F inspector_image3f;

  // References (ids) of specific options to check if they were matched.
  CommandLineParser::OptionId opt_lossless_jpeg_id = -1;
  CommandLineParser::OptionId opt_responsive_id = -1;
  CommandLineParser::OptionId opt_distance_id = -1;
  CommandLineParser::OptionId opt_alpha_distance_id = -1;
  CommandLineParser::OptionId opt_quality_id = -1;
  CommandLineParser::OptionId opt_modular_group_size_id = -1;
};

const char* ModeFromArgs(const CompressArgs& args) {
  if (args.lossless_jpeg) return "JPEG";
  if (args.modular == jxl::Override::kOn || args.distance == 0)
    return "Modular";
  return "VarDCT";
}

std::string DistanceFromArgs(const CompressArgs& args) {
  char buf[100];
  if (args.lossless_jpeg) {
    snprintf(buf, sizeof(buf), "lossless transcode");
  } else if (args.distance == 0) {
    snprintf(buf, sizeof(buf), "lossless");
  } else {
    snprintf(buf, sizeof(buf), "d%.3f", args.distance);
  }
  return buf;
}

void PrintMode(jxl::extras::PackedPixelFile& ppf, const double decode_mps,
               size_t num_bytes, const CompressArgs& args) {
  const char* mode = ModeFromArgs(args);
  const std::string distance = DistanceFromArgs(args);
  if (args.lossless_jpeg) {
    fprintf(stderr, "Read JPEG image with %" PRIuS " bytes.\n", num_bytes);
  } else {
    fprintf(stderr,
            "Read %" PRIuS "x%" PRIuS " image, %" PRIuS " bytes, %.1f MP/s\n",
            static_cast<size_t>(ppf.info.xsize),
            static_cast<size_t>(ppf.info.ysize), num_bytes, decode_mps);
  }
  fprintf(stderr, "Encoding [%s%s, %s, effort: %" PRIuS,
          (args.container == jxl::Override::kOn ? "Container | " : ""), mode,
          distance.c_str(), args.effort);
  if (args.container == jxl::Override::kOn) {
    if (args.lossless_jpeg && args.jpeg_store_metadata)
      fprintf(stderr, " | JPEG reconstruction data");
    if (!ppf.metadata.exif.empty())
      fprintf(stderr, " | %" PRIuS "-byte Exif", ppf.metadata.exif.size());
    if (!ppf.metadata.xmp.empty())
      fprintf(stderr, " | %" PRIuS "-byte XMP", ppf.metadata.xmp.size());
    if (!ppf.metadata.jumbf.empty())
      fprintf(stderr, " | %" PRIuS "-byte JUMBF", ppf.metadata.jumbf.size());
  }
  fprintf(stderr, "], \n");
}

bool IsJPG(const std::vector<uint8_t>& image_data) {
  return (image_data.size() >= 2 && image_data[0] == 0xFF &&
          image_data[1] == 0xD8);
}

// TODO(tfish): Replace with non-C-API library function.
// Implementation is in extras/.
jxl::Status GetPixeldata(const std::vector<uint8_t>& image_data,
                         const jxl::extras::ColorHints& color_hints,
                         jxl::extras::PackedPixelFile& ppf,
                         jxl::extras::Codec& codec) {
  // Any valid encoding is larger (ensures codecs can read the first few bytes).
  constexpr size_t kMinBytes = 9;

  if (image_data.size() < kMinBytes) return JXL_FAILURE("Input too small.");
  jxl::Span<const uint8_t> encoded(image_data);

  ppf.info.orientation = JXL_ORIENT_IDENTITY;
  jxl::SizeConstraints size_constraints;

  const auto choose_codec = [&]() {
#if JPEGXL_ENABLE_APNG
    if (jxl::extras::DecodeImageAPNG(encoded, color_hints, size_constraints,
                                     &ppf)) {
      return jxl::extras::Codec::kPNG;
    }
#endif
    if (jxl::extras::DecodeImagePGX(encoded, color_hints, size_constraints,
                                    &ppf)) {
      return jxl::extras::Codec::kPGX;
    } else if (jxl::extras::DecodeImagePNM(encoded, color_hints,
                                           size_constraints, &ppf)) {
      return jxl::extras::Codec::kPNM;
    }
#if JPEGXL_ENABLE_GIF
    if (jxl::extras::DecodeImageGIF(encoded, color_hints, size_constraints,
                                    &ppf)) {
      return jxl::extras::Codec::kGIF;
    }
#endif
#if JPEGXL_ENABLE_JPEG
    if (jxl::extras::DecodeImageJPG(encoded, color_hints, size_constraints,
                                    &ppf)) {
      return jxl::extras::Codec::kJPG;
    }
#endif
#if JPEGXL_ENABLE_EXR
    if (jxl::extras::DecodeImageEXR(encoded, color_hints, size_constraints,
                                    &ppf)) {
      return jxl::extras::Codec::kEXR;
    }
#endif
    // TODO(tfish): Bring back PSD.
    return jxl::extras::Codec::kUnknown;
  };
  codec = choose_codec();
  if (codec == jxl::extras::Codec::kUnknown) {
    return JXL_FAILURE("Codecs failed to decode input.");
  }
  return true;
}

using flag_check_fn = std::function<std::string(int64_t)>;
using flag_check_float_fn = std::function<std::string(float)>;

template <typename T>
void ProcessFlag(
    const char* flag_name, T flag_value,
    JxlEncoderFrameSettingId encoder_option,
    jxl::extras::JXLCompressParams* params,
    flag_check_fn flag_check = [](T x) { return std::string(); }) {
  std::string error = flag_check(flag_value);
  if (!error.empty()) {
    std::cerr << "Invalid flag value for --" << flag_name << ": " << error
              << std::endl;
    exit(EXIT_FAILURE);
  }
  params->options.emplace_back(
      jxl::extras::JXLOption(encoder_option, flag_value, 0));
}

void ProcessBoolFlag(jxl::Override flag_value,
                     JxlEncoderFrameSettingId encoder_option,
                     jxl::extras::JXLCompressParams* params) {
  if (flag_value != jxl::Override::kDefault) {
    int64_t value = flag_value == jxl::Override::kOn ? 1 : 0;
    params->options.emplace_back(
        jxl::extras::JXLOption(encoder_option, value, 0));
  }
}

void SetDistanceFromFlags(CommandLineParser* cmdline, CompressArgs* args,
                          jxl::extras::JXLCompressParams* params,
                          const jxl::extras::Codec& codec) {
  bool distance_set = cmdline->GetOption(args->opt_distance_id)->matched();
  bool alpha_distance_set =
      cmdline->GetOption(args->opt_alpha_distance_id)->matched();
  bool quality_set = cmdline->GetOption(args->opt_quality_id)->matched();
  if (((distance_set && (args->distance != 0.0)) ||
       (quality_set && (args->quality != 100))) &&
      args->lossless_jpeg && args->lossless_jpeg) {
    std::cerr << "Must not set quality below 100 nor non-zero distance in "
                 "combination with --lossless_jpeg=1."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (quality_set) {
    if (distance_set) {
      std::cerr << "Must not set both --distance and --quality." << std::endl;
      exit(EXIT_FAILURE);
    }
    double distance = args->quality >= 100 ? 0.0
                      : args->quality >= 30
                          ? 0.1 + (100 - args->quality) * 0.09
                          : 53.0 / 3000.0 * args->quality * args->quality -
                                23.0 / 20.0 * args->quality + 25.0;
    args->distance = distance;
    distance_set = true;
  }
  if (!distance_set) {
    bool lossy_input = (codec == jxl::extras::Codec::kJPG ||
                        codec == jxl::extras::Codec::kGIF);
    args->distance = lossy_input ? 0.0 : 1.0;
  } else if (args->distance > 0) {
    args->lossless_jpeg = 0;
  }
  params->distance = args->distance;
  params->alpha_distance =
      alpha_distance_set ? args->alpha_distance : params->distance;
}

void ProcessFlags(const jxl::extras::Codec codec,
                  const jxl::extras::PackedPixelFile& ppf,
                  const std::vector<uint8_t>* jpeg_bytes,
                  CommandLineParser* cmdline, CompressArgs* args,
                  jxl::extras::JXLCompressParams* params) {
  // Tuning flags.
  ProcessBoolFlag(args->modular, JXL_ENC_FRAME_SETTING_MODULAR, params);
  ProcessBoolFlag(args->keep_invisible, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE,
                  params);
  ProcessBoolFlag(args->dots, JXL_ENC_FRAME_SETTING_DOTS, params);
  ProcessBoolFlag(args->patches, JXL_ENC_FRAME_SETTING_PATCHES, params);
  ProcessBoolFlag(args->gaborish, JXL_ENC_FRAME_SETTING_GABORISH, params);
  ProcessBoolFlag(args->group_order, JXL_ENC_FRAME_SETTING_GROUP_ORDER, params);

  params->allow_expert_options = args->allow_expert_options;

  if (!args->frame_indexing.empty()) {
    bool must_be_all_zeros = args->frame_indexing[0] != '1';
    for (char c : args->frame_indexing) {
      if (c == '1') {
        if (must_be_all_zeros) {
          std::cerr << "Invalid --frame_indexing. If the first character is "
                       "'0', all must be '0'."
                    << std::endl;
          exit(EXIT_FAILURE);
        }
      } else if (c != '0') {
        std::cerr << "Invalid --frame_indexing. Must match the pattern "
                     "'^(0*|1[01]*)$'."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  ProcessFlag(
      "effort", static_cast<int64_t>(args->effort),
      JXL_ENC_FRAME_SETTING_EFFORT, params, [args](int64_t x) -> std::string {
        if (args->allow_expert_options) {
          return (1 <= x && x <= 10) ? "" : "Valid range is {1, 2, ..., 10}.";
        } else {
          return (1 <= x && x <= 9) ? "" : "Valid range is {1, 2, ..., 9}.";
        }
      });
  ProcessFlag("brotli_effort", static_cast<int64_t>(args->brotli_effort),
              JXL_ENC_FRAME_SETTING_BROTLI_EFFORT, params,
              [](int64_t x) -> std::string {
                return (-1 <= x && x <= 11)
                           ? ""
                           : "Valid range is {-1, 0, 1, ..., 11}.";
              });
  ProcessFlag(
      "epf", args->epf, JXL_ENC_FRAME_SETTING_EPF, params,
      [](int64_t x) -> std::string {
        return (-1 <= x && x <= 3) ? "" : "Valid range is {-1, 0, 1, 2, 3}.\n";
      });
  ProcessFlag("faster_decoding", static_cast<int64_t>(args->faster_decoding),
              JXL_ENC_FRAME_SETTING_DECODING_SPEED, params,
              [](int64_t x) -> std::string {
                return (0 <= x && x <= 4) ? ""
                                          : "Valid range is {0, 1, 2, 3, 4}.\n";
              });
  ProcessFlag("resampling", args->resampling, JXL_ENC_FRAME_SETTING_RESAMPLING,
              params, [](int64_t x) -> std::string {
                return (x == -1 || x == 1 || x == 2 || x == 4 || x == 8)
                           ? ""
                           : "Valid values are {-1, 1, 2, 4, 8}.\n";
              });
  ProcessFlag("ec_resampling", args->ec_resampling,
              JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, params,
              [](int64_t x) -> std::string {
                return (x == -1 || x == 1 || x == 2 || x == 4 || x == 8)
                           ? ""
                           : "Valid values are {-1, 1, 2, 4, 8}.\n";
              });
  ProcessFlag("photon_noise_iso", args->photon_noise_iso,
              JXL_ENC_FRAME_SETTING_PHOTON_NOISE, params);
  ProcessFlag("already_downsampled",
              static_cast<int64_t>(args->already_downsampled),
              JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED, params);

  SetDistanceFromFlags(cmdline, args, params, codec);

  if (args->group_order != jxl::Override::kOn &&
      (args->center_x != -1 || args->center_y != -1)) {
    std::cerr << "Invalid flag combination. Setting --center_x or --center_y "
              << "requires setting --group_order=1" << std::endl;
    exit(EXIT_FAILURE);
  }
  ProcessFlag("center_x", args->center_x,
              JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X, params,
              [](int64_t x) -> std::string {
                if (x < -1) {
                  return "Valid values are: -1 or [0 .. xsize).";
                }
                return "";
              });
  ProcessFlag("center_y", args->center_y,
              JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y, params,
              [](int64_t x) -> std::string {
                if (x < -1) {
                  return "Valid values are: -1 or [0 .. ysize).";
                }
                return "";
              });

  // Progressive/responsive mode settings.
  bool responsive_set = cmdline->GetOption(args->opt_responsive_id)->matched();

  ProcessFlag("progressive_dc", args->progressive_dc,
              JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, params,
              [](int64_t x) -> std::string {
                return (-1 <= x && x <= 2) ? ""
                                           : "Valid range is {-1, 0, 1, 2}.\n";
              });
  ProcessFlag("progressive_ac", static_cast<int64_t>(args->progressive_ac),
              JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, params);

  if (args->progressive) {
    args->qprogressive_ac = true;
    args->responsive = 1;
    responsive_set = true;
  }
  if (responsive_set) {
    ProcessFlag("responsive", args->responsive,
                JXL_ENC_FRAME_SETTING_RESPONSIVE, params);
  }
  if (args->qprogressive_ac) {
    ProcessFlag("qprogressive_ac", static_cast<int64_t>(1),
                JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC, params);
  }

  // Modular mode related.
  // TODO(firsching): consider doing more validation after image size is
  // known, i.e. set to 512 if 256 would be silly using
  // opt_modular_group_size_id.
  ProcessFlag("modular_group_size", args->modular_group_size,
              JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, params,
              [](int64_t x) -> std::string {
                return (-1 <= x && x <= 3)
                           ? ""
                           : "Invalid --modular_group_size. Valid "
                             "range is {-1, 0, 1, 2, 3}.\n";
              });
  ProcessFlag("modular_predictor", args->modular_predictor,
              JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, params,
              [](int64_t x) -> std::string {
                return (-1 <= x && x <= 15)
                           ? ""
                           : "Invalid --modular_predictor. Valid "
                             "range is {-1, 0, 1, ..., 15}.\n";
              });
  ProcessFlag("modular_colorspace", args->modular_colorspace,
              JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE, params,
              [](int64_t x) -> std::string {
                return (-1 <= x && x <= 41)
                           ? ""
                           : "Invalid --modular_colorspace. Valid range is "
                             "{-1, 0, 1, ..., 41}.\n";
              });
  ProcessFlag("modular_ma_tree_learning_percent",
              args->modular_ma_tree_learning_percent,
              JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT, params,
              [](float x) -> std::string {
                return -1 <= x && x <= 100
                           ? ""
                           : "Invalid --modular_ma_tree_learning_percent, Valid"
                             "rang is [-1, 100].\n";
              });
  ProcessFlag("modular_nb_prev_channels", args->modular_nb_prev_channels,
              JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS, params,
              [](int64_t x) -> std::string {
                return (-1 <= x && x <= 11)
                           ? ""
                           : "Invalid --modular_nb_prev_channels. Valid "
                             "range is {-1, 0, 1, ..., 11}.\n";
              });
  if (args->modular_lossy_palette) {
    if (args->progressive || args->qprogressive_ac) {
      fprintf(stderr,
              "WARNING: --modular_lossy_palette is ignored in "
              "progressive mode.\n");
      args->modular_lossy_palette = false;
    }
  }
  ProcessFlag("modular_lossy_palette",
              static_cast<int64_t>(args->modular_lossy_palette),
              JXL_ENC_FRAME_SETTING_LOSSY_PALETTE, params);
  ProcessFlag("modular_palette_colors", args->modular_palette_colors,
              JXL_ENC_FRAME_SETTING_PALETTE_COLORS, params,
              [](int64_t x) -> std::string {
                return -1 <= x ? ""
                               : "Invalid --modular_palette_colors, must "
                                 "be -1 or non-negative\n";
              });
  ProcessFlag("modular_channel_colors_global_percent",
              args->modular_channel_colors_global_percent,
              JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT, params,
              [](float x) -> std::string {
                return (-1 <= x && x <= 100)
                           ? ""
                           : "Invalid --modular_channel_colors_global_percent. "
                             "Valid "
                             "range is [-1, 100].\n";
              });
  ProcessFlag("modular_channel_colors_group_percent",
              args->modular_channel_colors_group_percent,
              JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT, params,
              [](float x) -> std::string {
                return (-1 <= x && x <= 100)
                           ? ""
                           : "Invalid --modular_channel_colors_group_percent. "
                             "Valid "
                             "range is [-1, 100].\n";
              });

  if (args->num_threads < -1) {
    std::cerr
        << "Invalid flag value for --num_threads: must be -1, 0 or postive."
        << std::endl;
    exit(EXIT_FAILURE);
  }
  // JPEG specific options.
  if (jpeg_bytes) {
    ProcessBoolFlag(args->jpeg_reconstruction_cfl,
                    JXL_ENC_FRAME_SETTING_JPEG_RECON_CFL, params);
    ProcessBoolFlag(args->compress_boxes,
                    JXL_ENC_FRAME_SETTING_JPEG_COMPRESS_BOXES, params);
  }
  // Set per-frame options.
  for (size_t num_frame = 0; num_frame < ppf.frames.size(); ++num_frame) {
    if (num_frame < args->frame_indexing.size() &&
        args->frame_indexing[num_frame] == '1') {
      int64_t value = 1;
      params->options.emplace_back(
          jxl::extras::JXLOption(JXL_ENC_FRAME_INDEX_BOX, value, num_frame));
    }
  }
  // Copy over the rest of the non-option params.
  params->use_container = args->container == jxl::Override::kOn;
  params->jpeg_store_metadata = args->jpeg_store_metadata;
  params->intensity_target = args->intensity_target;
  params->override_bitdepth = args->override_bitdepth;
  params->codestream_level = args->codestream_level;
  params->premultiply = args->premultiply;
  params->compress_boxes = args->compress_boxes != jxl::Override::kOff;
  if (codec == jxl::extras::Codec::kPNM) {
    params->input_bitdepth.type = JXL_BIT_DEPTH_FROM_CODESTREAM;
  }
}

}  // namespace tools
}  // namespace jpegxl

int main(int argc, char** argv) {
  std::string version = jpegxl::tools::CodecConfigString(JxlEncoderVersion());
  jpegxl::tools::CompressArgs args;
  jpegxl::tools::CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, const_cast<const char**>(argv))) {
    // Parse already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return jpegxl::tools::CjxlRetCode::ERR_PARSE;
  }

  if (args.version) {
    fprintf(stdout, "cjxl %s\n", version.c_str());
    fprintf(stdout, "Copyright (c) the JPEG XL Project\n");
    return jpegxl::tools::CjxlRetCode::OK;
  }

  if (!args.quiet) {
    fprintf(stderr, "JPEG XL encoder %s\n", version.c_str());
  }

  if (cmdline.HelpFlagPassed() || !args.file_in) {
    cmdline.PrintHelp();
    return jpegxl::tools::CjxlRetCode::OK;
  }

  if (!args.file_out && !args.disable_output) {
    std::cerr
        << "No output file specified and --disable_output flag not passed."
        << std::endl;
    exit(EXIT_FAILURE);
  }

  if (args.file_out && args.disable_output && !args.quiet) {
    fprintf(stderr,
            "Encoding will be performed, but the result will be discarded.\n");
  }

  // Loading the input.
  // Depending on flags-settings, we want to either load a JPEG and
  // faithfully convert it to JPEG XL, or load (JPEG or non-JPEG)
  // pixel data.
  std::vector<uint8_t> image_data;
  jxl::extras::PackedPixelFile ppf;
  jxl::extras::Codec codec = jxl::extras::Codec::kUnknown;
  std::vector<uint8_t>* jpeg_bytes = nullptr;
  double decode_mps = 0;
  size_t pixels = 0;
  if (!jpegxl::tools::ReadFile(args.file_in, &image_data)) {
    std::cerr << "Reading image data failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!jpegxl::tools::IsJPG(image_data)) args.lossless_jpeg = 0;
  jxl::extras::JXLCompressParams params;
  ProcessFlags(codec, ppf, jpeg_bytes, &cmdline, &args, &params);
  if (!args.lossless_jpeg) {
    const double t0 = jxl::Now();
    jxl::Status status =
        jpegxl::tools::GetPixeldata(image_data, args.color_hints, ppf, codec);
    if (!status) {
      std::cerr << "Getting pixel data failed." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (ppf.frames.empty()) {
      std::cerr << "No frames on input file." << std::endl;
      exit(EXIT_FAILURE);
    }

    const double t1 = jxl::Now();
    pixels = ppf.info.xsize * ppf.info.ysize;
    decode_mps = pixels * ppf.info.num_color_channels * 1E-6 / (t1 - t0);
  }
  if (args.lossless_jpeg && jpegxl::tools::IsJPG(image_data)) {
    if (!cmdline.GetOption(args.opt_lossless_jpeg_id)->matched()) {
      std::cerr << "Note: Implicit-default for JPEG is lossless-transcoding. "
                << "To silence this message, set --lossless_jpeg=(1|0)."
                << std::endl;
    }
    jpeg_bytes = &image_data;
  }

  ProcessFlags(codec, ppf, jpeg_bytes, &cmdline, &args, &params);

  if (!ppf.metadata.exif.empty() || !ppf.metadata.xmp.empty() ||
      !ppf.metadata.jumbf.empty() || !ppf.metadata.iptc.empty() ||
      (args.lossless_jpeg && args.jpeg_store_metadata)) {
    args.container = jxl::Override::kOn;
  }

  if (!ppf.metadata.exif.empty()) {
    jxl::InterpretExif(ppf.metadata.exif, &ppf.info.orientation);
  }

  if (!args.quiet) {
    PrintMode(ppf, decode_mps, image_data.size(), args);
  }

  size_t num_worker_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
  int64_t flag_num_worker_threads = args.num_threads;
  if (flag_num_worker_threads > -1) {
    num_worker_threads = flag_num_worker_threads;
  }
  JxlThreadParallelRunnerPtr runner = JxlThreadParallelRunnerMake(
      /*memory_manager=*/nullptr, num_worker_threads);
  params.runner = JxlThreadParallelRunner;
  params.runner_opaque = runner.get();

  jpegxl::tools::SpeedStats stats;
  std::vector<uint8_t> compressed;
  for (size_t num_rep = 0; num_rep < args.num_reps; ++num_rep) {
    const double t0 = jxl::Now();
    if (!EncodeImageJXL(params, ppf, jpeg_bytes, &compressed)) {
      fprintf(stderr, "EncodeImageJXL() failed.\n");
      return EXIT_FAILURE;
    }
    const double t1 = jxl::Now();
    stats.NotifyElapsed(t1 - t0);
    stats.SetImageSize(ppf.info.xsize, ppf.info.ysize);
  }

  if (args.file_out && !args.disable_output) {
    if (!jpegxl::tools::WriteFile(args.file_out, compressed)) {
      std::cerr << "Could not write jxl file." << std::endl;
      return EXIT_FAILURE;
    }
  }
  if (!args.quiet) {
    fprintf(stderr, "Compressed to %" PRIuS " bytes ", compressed.size());
    // For lossless jpeg-reconstruction, we don't print some stats, since we
    // don't have easy access to the image dimensions.
    if (args.container == jxl::Override::kOn) {
      fprintf(stderr, "including container ");
    }
    if (!args.lossless_jpeg) {
      const double bpp =
          static_cast<double>(compressed.size() * jxl::kBitsPerByte) / pixels;
      fprintf(stderr, "(%.3f bpp%s).\n", bpp / ppf.frames.size(),
              ppf.frames.size() == 1 ? "" : "/frame");
      JXL_CHECK(stats.Print(num_worker_threads));
    } else {
      fprintf(stderr, "\n");
    }
  }
  return EXIT_SUCCESS;
}
