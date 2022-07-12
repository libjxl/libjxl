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
#include <vector>

#include "jxl/codestream_header.h"
#include "jxl/encode.h"
#include "jxl/encode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "jxl/types.h"
#include "lib/extras/dec/apng.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/gif.h"
#include "lib/extras/dec/jpg.h"
#include "lib/extras/dec/pgx.h"
#include "lib/extras/dec/pnm.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/size_constraints.h"
#include "tools/args.h"
#include "tools/cmdline.h"
#include "tools/codec_config.h"

namespace jpegxl {
namespace tools {

namespace {
inline bool ParsePhotonNoiseParameter(const char* arg, float* out) {
  return strncmp(arg, "ISO", 3) == 0 && ParseFloat(arg + 3, out) && *out > 0;
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
    cmdline->AddPositionalOption(
        "OUTPUT", /* required = */ true,
        "the compressed JXL output file (can be omitted for benchmarking)",
        &file_out);

    // Flags.
    // TODO(lode): also add options to add exif/xmp/other metadata in the
    // container.
    cmdline->AddOptionFlag(
        '\0', "container",
        "Force using container format (default: use only if needed).",
        &container, &SetBooleanTrue, 1);

    cmdline->AddOptionFlag('\0', "strip",
                           "Do not encode using container format (strips "
                           "Exif/XMP/JPEG bitstream reconstruction data).",
                           &strip, &SetBooleanTrue, 2);

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
        "    Recommended range: 0.5 .. 3.0. Mutually exclusive with --quality.",
        &distance, &ParseFloat);

    // High-level options
    opt_quality_id = cmdline->AddOptionValue(
        'q', "quality", "QUALITY",
        "Quality setting (is remapped to --distance). Range: -inf .. 100.\n"
        "    100 = mathematically lossless. Default for already-lossy input "
        "(JPEG/GIF).\n"
        "    Other input gets encoded as per --distance default.\n"
        "    Positive quality values roughly match libjpeg quality.\n"
        "    Mutually exclusive with --distance.",
        &quality, &ParseFloat);

    cmdline->AddOptionValue(
        'e', "effort", "EFFORT",
        "Encoder effort setting. Range: 1 .. 9.\n"
        "     Default: 3. Higher number is more effort (slower).",
        &effort, &ParseUnsigned, -1);

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

    opt_qprogressive_ac_id = cmdline->AddOptionFlag(
        '\0', "qprogressive_ac",
        "Use the progressive mode for AC with shift quantization.",
        &qprogressive_ac, &SetBooleanTrue, 1);

    cmdline->AddOptionValue(
        '\0', "progressive_dc", "num_dc_frames",
        "Progressive-DC setting. Valid values are: -1, 0, 1, 2.",
        &progressive_dc, &ParseSigned, 1);

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

    cmdline->AddOptionValue(
        '\0', "photon_noise", "ISO3200",
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
        &resampling, &ParseSigned, 0);

    cmdline->AddOptionValue(
        '\0', "ec_resampling", "-1|1|2|4|8",
        "Resampling for extra channels. Default of -1 applies resampling only "
        "for low quality. Value 1 does no downsampling (1x1), 2 does 2x2 "
        "downsampling, 4 is for 4x4 downsampling, and 8 for 8x8 downsampling.",
        &ec_resampling, &ParseSigned, 2);

    cmdline->AddOptionFlag('\0', "already_downsampled",
                           "Do not downsample the given input before encoding, "
                           "but still signal that the decoder should upsample.",
                           &already_downsampled, &SetBooleanTrue, 2);

    cmdline->AddOptionValue(
        '\0', "epf", "-1|0|1|2|3",
        "Edge preserving filter level, -1 to 3. "
        "Value -1 means: default (encoder chooses), 0 to 3 set a strength.",
        &epf, &ParseSigned, 1);

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

    // TODO(firsching): wire this up.
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
        "default value, 100 = all, values above 100 are also permitted. "
        "Higher values use more encoder memory.",
        &modular_ma_tree_learning_percent, &ParseSigned, 2);

    cmdline->AddOptionValue(
        'C', "modular_colorspace", "K",
        ("[modular encoding] color transform: -1=default, 0=RGB (none), "
         "1-41=RCT (6=YCoCg, default: try several, depending on speed)"),
        &modular_colorspace, &ParseSigned, 1);

    opt_modular_group_size_id = cmdline->AddOptionValue(
        'g', "modular_group_size", "K",
        "[modular encoding] group size: -1 == default. 0 => 128, "
        "1 => 256, 2 => 512, 3 => 1024",
        &modular_group_size, &ParseSigned, 1);

    cmdline->AddOptionValue(
        'P', "modular_predictor", "K",
        "[modular encoding] predictor(s) to use: 0=zero, "
        "1=left, 2=top, 3=avg0, 4=select, 5=gradient, 6=weighted, "
        "7=topright, 8=topleft, 9=leftleft, 10=avg1, 11=avg2, 12=avg3, "
        "13=toptop predictive average "
        "14=mix 5 and 6, 15=mix everything. If unset, uses default 14, "
        "at slowest speed default 15.",
        &modular_predictor, &ParseSigned, 1);

    cmdline->AddOptionValue(
        'E', "modular_nb_prev_channels", "K",
        "[modular encoding] number of extra MA tree properties to use",
        &modular_nb_prev_channels, &ParseSigned, 2);

    cmdline->AddOptionValue(
        '\0', "modular_palette_colors", "K",
        "[modular encoding] Use color palette if number of colors is smaller "
        "than or equal to this, or -1 to use the encoder default.",
        &modular_palette_colors, &ParseSigned, 1);

    cmdline->AddOptionFlag(
        '\0', "moular_lossy_palette",
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
        &modular_channel_colors_global_percent, &ParseSigned, 2);

    cmdline->AddOptionValue(
        'Y', "post-compact", "PERCENT",
        "[modular encoding] Use Local (per-group) channel palette if the "
        "number "
        "of colors is smaller than this percentage of range. Use 0-100 to set "
        "an explicit percentage, -1 to use the encoder default.",
        &modular_channel_colors_group_percent, &ParseSigned, 2);

    cmdline->AddOptionValue('\0', "codestream_level", "K",
                            "The codestream level. Either `-1`, `5` or `10`.",
                            &codestream_level, &ParseSigned, 2);

    opt_responsive_id = cmdline->AddOptionValue(
        'R', "responsive", "K",
        "[modular encoding] do Squeeze transform, 0=false, "
        "1=true (default: true if lossy, false if lossless)",
        &responsive, &ParseSigned, 1);

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
        'v', "verbose",
        "Verbose output; can be repeated, also applies to help (!).", &verbose,
        &SetBooleanTrue);
  }

  // Common flags.
  bool version = false;
  bool container = false;
  bool strip = false;
  bool quiet = false;

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

  // Filename for the user provided saliency-map.
  std::string saliency_map_filename;

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
  int32_t progressive_dc = -1;
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

  size_t faster_decoding = 0;
  int32_t resampling = -1;
  int32_t ec_resampling = -1;
  int32_t epf = -1;
  int64_t center_x = -1;
  int64_t center_y = -1;
  int32_t modular_group_size = -1;
  int32_t modular_predictor = -1;
  int32_t modular_colorspace = -1;
  int32_t modular_channel_colors_global_percent = -1;
  int32_t modular_channel_colors_group_percent = -1;
  int32_t modular_palette_colors = -1;
  int32_t modular_nb_prev_channels = -1;
  int32_t modular_ma_tree_learning_percent = -1;
  float photon_noise_iso = 0;
  int32_t codestream_level = -1;
  int32_t responsive = -1;
  float distance = 1.0;
  size_t effort = 3;
  size_t brotli_effort = 9;
  std::string frame_indexing;

  // Will get passed on to AuxOut.
  // jxl::InspectorImage3F inspector_image3f;

  // References (ids) of specific options to check if they were matched.
  CommandLineParser::OptionId opt_lossless_jpeg_id = -1;
  CommandLineParser::OptionId opt_responsive_id = -1;
  CommandLineParser::OptionId opt_distance_id = -1;
  CommandLineParser::OptionId opt_quality_id = -1;
  CommandLineParser::OptionId opt_qprogressive_ac_id = -1;
  CommandLineParser::OptionId opt_modular_group_size_id = -1;
};

}  // namespace tools
}  // namespace jpegxl

namespace {
/**
 * Writes bytes to file.
 */
bool WriteFile(const std::vector<uint8_t>& bytes, const char* filename) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    std::cerr << "Could not open file: " << filename << " for writing"
              << std::endl
              << "Error: " << strerror(errno) << std::endl;
    return false;
  }
  if (fwrite(bytes.data(), sizeof(uint8_t), bytes.size(), file) !=
      bytes.size()) {
    std::cerr << "Could not write bytes to file: " << filename << std::endl
              << "Error: " << strerror(errno) << std::endl;
    return false;
  }
  if (fclose(file) != 0) {
    std::cerr << "Could not close file: " << filename << std::endl
              << "Error: " << strerror(errno) << std::endl;
    return false;
  }
  return true;
}

void SetFlagFrameOptionOrDie(const char* flag_name, int32_t flag_value,
                             JxlEncoderFrameSettings* frame_settings,
                             JxlEncoderFrameSettingId encoder_option) {
  if (JXL_ENC_SUCCESS != JxlEncoderFrameSettingsSetOption(
                             frame_settings, encoder_option, flag_value)) {
    std::cerr << "Setting encoder option from flag -- " << flag_name
              << "failed." << std::endl;
    exit(EXIT_FAILURE);
  }
}

void SetDistanceFromFlags(JxlEncoderFrameSettings* jxl_encoder_frame_settings,
                          jpegxl::tools::CommandLineParser* cmdline,
                          jpegxl::tools::CompressArgs* args,
                          const jxl::extras::Codec& codec) {
  bool distance_set = cmdline->GetOption(args->opt_distance_id)->matched();
  bool quality_set = cmdline->GetOption(args->opt_quality_id)->matched();

  if (distance_set && quality_set) {
    std::cerr << "Must not set both --distance and --quality." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (distance_set) {
    if (JXL_ENC_SUCCESS != JxlEncoderSetFrameDistance(
                               jxl_encoder_frame_settings, args->distance)) {
      std::cerr << "Setting --distance parameter failed." << std::endl;
      exit(EXIT_FAILURE);
    }
    return;
  }
  if (quality_set) {
    double distance = args->quality >= 100 ? 0.0
                      : args->quality >= 30
                          ? 0.1 + (100 - args->quality) * 0.09
                          : 6.4 + pow(2.5, (30 - args->quality) / 5.0) / 6.25;
    if (JXL_ENC_SUCCESS !=
        JxlEncoderSetFrameDistance(jxl_encoder_frame_settings, distance)) {
      std::cerr << "Setting --quality parameter failed." << std::endl;
      exit(EXIT_FAILURE);
    }
    return;
  }
  // No flag set, but input is JPG or GIF: Use distance 0 default.
  if (codec == jxl::extras::Codec::kJPG || codec == jxl::extras::Codec::kGIF) {
    if (JXL_ENC_SUCCESS ==
        JxlEncoderSetFrameDistance(jxl_encoder_frame_settings, 0.0)) {
      std::cerr << "Setting 'lossless' default for GIF or JPEG input."
                << std::endl;
    }
  }
}

using flag_check_fn = std::function<std::string(int32_t)>;

bool IsJPG(const jxl::PaddedBytes& image_data) {
  return (image_data.size() >= 2 && image_data[0] == 0xFF &&
          image_data[1] == 0xD8);
}

// TODO(tfish): Replace with non-C-API library function.
// Implementation is in extras/.
jxl::Status GetPixeldata(const jxl::PaddedBytes& image_data,
                         jxl::extras::PackedPixelFile& ppf,
                         jxl::extras::Codec& codec) {
  // Any valid encoding is larger (ensures codecs can read the first few bytes).
  constexpr size_t kMinBytes = 9;

  if (image_data.size() < kMinBytes) return JXL_FAILURE("Input too small.");
  jxl::Span<const uint8_t> encoded(image_data);

  ppf.info.orientation = JXL_ORIENT_IDENTITY;
  jxl::extras::ColorHints color_hints;
  jxl::SizeConstraints size_constraints;

#if JPEGXL_ENABLE_APNG
  if (jxl::extras::DecodeImageAPNG(encoded, color_hints, size_constraints,
                                   &ppf)) {
    codec = jxl::extras::Codec::kPNG;
  } else
#endif
      if (jxl::extras::DecodeImagePGX(encoded, color_hints, size_constraints,
                                      &ppf)) {
    codec = jxl::extras::Codec::kPGX;
  } else if (jxl::extras::DecodeImagePNM(encoded, color_hints, size_constraints,
                                         &ppf)) {
    codec = jxl::extras::Codec::kPNM;
  }
#if JPEGXL_ENABLE_GIF
  else if (jxl::extras::DecodeImageGIF(encoded, color_hints, size_constraints,
                                       &ppf)) {
    codec = jxl::extras::Codec::kGIF;
  }
#endif
#if JPEGXL_ENABLE_JPEG
  else if (jxl::extras::DecodeImageJPG(encoded, color_hints, size_constraints,
                                       &ppf)) {
    codec = jxl::extras::Codec::kJPG;
  }
#endif
  else {  // TODO(tfish): Bring back EXR and PSD.
    return JXL_FAILURE("Codecs failed to decode input.");
  }
  // TODO(tfish): Migrate this:
  // if (!skip_ppf_conversion) {
  //   JXL_RETURN_IF_ERROR(ConvertPackedPixelFileToCodecInOut(ppf, pool, io));
  // }
  return true;
}

}  // namespace

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

  if (!args.file_out && !args.quiet) {
    fprintf(stderr,
            "No output file specified.\n"
            "Encoding will be performed, but the result will be discarded.\n");
  }

  // Loading the input.
  // Depending on flags-settings, we want to either load a JPEG and
  // faithfully convert it to JPEG XL, or load (JPEG or non-JPEG)
  // pixel data. For benchmarking, we want to be able to do
  // N repetitions of image-compression, but the input should
  // not get reloaded as part of that.
  // Since we do not want to load the input before we decided that
  // flag-settings are valid, we need a mechanism to lazy-load the image.
  bool input_image_loaded = false;
  jxl::PaddedBytes image_data;
  jxl::extras::PackedPixelFile ppf;
  jxl::extras::Codec codec = jxl::extras::Codec::kUnknown;
  auto ensure_image_loaded = [&input_image_loaded, &image_data, &ppf, &codec,
                              &args]() {
    if (input_image_loaded) return;
    if (!ReadFile(args.file_in, &image_data)) {
      std::cerr << "Reading image data failed." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!(args.lossless_jpeg && IsJPG(image_data))) {
      jxl::Status status = GetPixeldata(image_data, ppf, codec);
      if (!status) {
        std::cerr << "Getting pixel data." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (ppf.frames.empty()) {
        std::cerr << "No frames on input file." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    input_image_loaded = true;
  };

  JxlEncoderPtr enc = JxlEncoderMake(/*memory_manager=*/nullptr);
  JxlEncoder* jxl_encoder = enc.get();
  JxlThreadParallelRunnerPtr runner;
  for (size_t num_rep = 0; num_rep < args.num_reps; ++num_rep) {
    JxlEncoderReset(jxl_encoder);
    if (args.num_threads != 0) {
      size_t num_worker_threads =
          JxlThreadParallelRunnerDefaultNumWorkerThreads();
      {
        int64_t flag_num_worker_threads = args.num_threads;
        if (flag_num_worker_threads > -1) {
          num_worker_threads = flag_num_worker_threads;
        }
      }
      if (runner == nullptr) {
        runner = JxlThreadParallelRunnerMake(
            /*memory_manager=*/nullptr, num_worker_threads);
      }
      if (JXL_ENC_SUCCESS !=
          JxlEncoderSetParallelRunner(jxl_encoder, JxlThreadParallelRunner,
                                      runner.get())) {
        std::cerr << "JxlEncoderSetParallelRunner failed." << std::endl;
        return EXIT_FAILURE;
      }
    }

    JxlEncoderFrameSettings* jxl_encoder_frame_settings =
        JxlEncoderFrameSettingsCreate(jxl_encoder, nullptr);

    auto process_flag = [&jxl_encoder_frame_settings](
                            const char* flag_name, int32_t flag_value,
                            JxlEncoderFrameSettingId encoder_option,
                            const flag_check_fn& flag_check) {
      std::string error = flag_check(flag_value);
      if (!error.empty()) {
        std::cerr << "Invalid flag value for --" << flag_name << ": " << error
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      SetFlagFrameOptionOrDie(flag_name, flag_value, jxl_encoder_frame_settings,
                              encoder_option);
    };

    auto process_bool_flag = [&jxl_encoder_frame_settings](
                                 const char* flag_name,
                                 jxl::Override flag_value,
                                 JxlEncoderFrameSettingId encoder_option) {
      if (flag_value != jxl::Override::kDefault) {
        SetFlagFrameOptionOrDie(flag_name,
                                flag_value == jxl::Override::kOn ? 1 : 0,
                                jxl_encoder_frame_settings, encoder_option);
      }
    };

    {  // Processing tuning flags.
      bool use_container = args.container;
      // TODO(tfish): Set use_container according to need of encoded data.
      // This will likely require moving this piece out of flags-processing.
      if (args.strip) {
        use_container = false;
      }
      if (JXL_ENC_SUCCESS !=
          JxlEncoderUseContainer(jxl_encoder,
                                 static_cast<int>(use_container))) {
        std::cerr << "JxlEncoderUseContainer failed." << std::endl;
        return EXIT_FAILURE;
      }

      process_bool_flag("modular", args.modular, JXL_ENC_FRAME_SETTING_MODULAR);
      process_bool_flag("keep_invisible", args.keep_invisible,
                        JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE);
      process_bool_flag("dots", args.dots, JXL_ENC_FRAME_SETTING_DOTS);
      process_bool_flag("patches", args.patches, JXL_ENC_FRAME_SETTING_PATCHES);
      process_bool_flag("gaborish", args.gaborish,
                        JXL_ENC_FRAME_SETTING_GABORISH);
      process_bool_flag("group_order", args.group_order,
                        JXL_ENC_FRAME_SETTING_GROUP_ORDER);

      if (!args.frame_indexing.empty()) {
        bool must_be_all_zeros = args.frame_indexing[0] != '1';
        for (char c : args.frame_indexing) {
          if (c == '1') {
            if (must_be_all_zeros) {
              std::cerr
                  << "Invalid --frame_indexing. If the first character is "
                     "'0', all must be '0'."
                  << std::endl;
              return EXIT_FAILURE;
            }
          } else if (c != '0') {
            std::cerr << "Invalid --frame_indexing. Must match the pattern "
                         "'^(0*|1[01]*)$'."
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
      }

      process_flag(
          "effort", args.effort, JXL_ENC_FRAME_SETTING_EFFORT,
          [](int32_t x) -> std::string {
            return (1 <= x && x <= 9) ? "" : "Valid range is {1, 2, ..., 9}.";
          });
      process_flag(
          "brotli_effort", args.brotli_effort,
          JXL_ENC_FRAME_SETTING_BROTLI_EFFORT, [](int32_t x) -> std::string {
            return (-1 <= x && x <= 11) ? ""
                                        : "Valid range is {-1, 0, 1, ..., 11}.";
          });
      process_flag("epf", args.epf, JXL_ENC_FRAME_SETTING_EPF,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 3)
                                ? ""
                                : "Valid range is {-1, 0, 1, 2, 3}.\n";
                   });
      process_flag(
          "faster_decoding", args.faster_decoding,
          JXL_ENC_FRAME_SETTING_DECODING_SPEED, [](int32_t x) -> std::string {
            return (0 <= x && x <= 4) ? ""
                                      : "Valid range is {0, 1, 2, 3, 4}.\n";
          });
      process_flag("resampling", args.resampling,
                   JXL_ENC_FRAME_SETTING_RESAMPLING,
                   [](int32_t x) -> std::string {
                     return (x == -1 || x == 1 || x == 4 || x == 8)
                                ? ""
                                : "Valid values are {-1, 1, 2, 4, 8}.\n";
                   });
      process_flag("ec_resampling", args.ec_resampling,
                   JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING,
                   [](int32_t x) -> std::string {
                     return (x == -1 || x == 1 || x == 4 || x == 8)
                                ? ""
                                : "Valid values are {-1, 1, 2, 4, 8}.\n";
                   });
      // TODO(firsching): change JxlEncoderFrameSettingsSetOption to take float
      // for JXL_ENC_FRAME_SETTING_PHOTON_NOISE.
      SetFlagFrameOptionOrDie("photon_noise_iso", args.photon_noise_iso,
                              jxl_encoder_frame_settings,
                              JXL_ENC_FRAME_SETTING_PHOTON_NOISE);
      SetFlagFrameOptionOrDie("already_downsampled",
                              static_cast<int32_t>(args.already_downsampled),
                              jxl_encoder_frame_settings,
                              JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED);
      SetDistanceFromFlags(jxl_encoder_frame_settings, &cmdline, &args, codec);

      if (args.group_order != jxl::Override::kOn &&
          (args.center_x != -1 || args.center_y != -1)) {
        std::cerr
            << "Invalid flag combination. Setting --center_x or --center_y "
            << "requires setting --group_order=1" << std::endl;
        return EXIT_FAILURE;
      }
      // TODO(firsching): change JxlEncoderFrameSettingsSetOption fo take
      // int64_t for JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_[X|Y].
      process_flag("center_x", args.center_x,
                   JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X,
                   [](int32_t x) -> std::string {
                     if (x < -1) {
                       return "Valid values are: -1 or [0 .. xsize).";
                     }
                     return "";
                   });
      process_flag("center_y", args.center_y,
                   JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y,
                   [](int32_t x) -> std::string {
                     if (x < -1) {
                       return "Valid values are: -1 or [0 .. ysize).";
                     }
                     return "";
                   });
    }
    {  // Progressive/responsive mode settings.
      bool qprogressive_ac_set =
          cmdline.GetOption(args.opt_qprogressive_ac_id)->matched();
      int32_t qprogressive_ac = args.qprogressive_ac ? 1 : 0;
      bool responsive_set =
          cmdline.GetOption(args.opt_responsive_id)->matched();
      int32_t responsive = args.responsive ? 1 : 0;

      process_flag(
          "progressive_dc", args.progressive_dc,
          JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, [](int32_t x) -> std::string {
            return (-1 <= x && x <= 2) ? "" : "Valid range is {-1, 0, 1, 2}.\n";
          });
      SetFlagFrameOptionOrDie(
          "progressive_ac", static_cast<int32_t>(args.progressive_ac),
          jxl_encoder_frame_settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC);

      if (args.progressive) {
        qprogressive_ac = 1;
        qprogressive_ac_set = true;
        responsive = 1;
        responsive_set = true;
      }
      if (responsive_set) {
        SetFlagFrameOptionOrDie("responsive", responsive,
                                jxl_encoder_frame_settings,
                                JXL_ENC_FRAME_SETTING_RESPONSIVE);
      }
      if (qprogressive_ac_set) {
        SetFlagFrameOptionOrDie("qprogressive_ac", qprogressive_ac,
                                jxl_encoder_frame_settings,
                                JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC);
      }
    }
    {  // Modular mode related.
      // TODO(firsching): consider doing more validation after image size is
      // known, i.e. set to 512 if 256 would be silly using
      // opt_modular_group_size_id.
      process_flag("modular_group_size", args.modular_group_size,
                   JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 3)
                                ? ""
                                : "Invalid --modular_group_size. Valid "
                                  "range is {-1, 0, 1, 2, 3}.\n";
                   });
      process_flag("modular_predictor", args.modular_predictor,
                   JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 15)
                                ? ""
                                : "Invalid --modular_predictor. Valid "
                                  "range is {-1, 0, 1, ..., 15}.\n";
                   });
      process_flag(
          "modular_colorspace", args.modular_colorspace,
          JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE,
          [](int32_t x) -> std::string {
            return (-1 <= x && x <= 41)
                       ? ""
                       : "Invalid --modular_colorspace. Valid range is "
                         "{-1, 0, 1, ..., 41}.\n";
          });
      process_flag(
          "modular_ma_tree_learning_percent",
          args.modular_ma_tree_learning_percent,
          JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT,
          [](int32_t x) -> std::string {
            return -1 <= x ? ""
                           : "Invalid --modular_ma_tree_learning_percent, must "
                             "be -1 or non-negative\n";
          });
      process_flag("modular_nb_prev_channels", args.modular_nb_prev_channels,
                   JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 11)
                                ? ""
                                : "Invalid --modular_nb_prev_channels. Valid "
                                  "range is {-1, 0, 1, ..., 11}.\n";
                   });
      SetFlagFrameOptionOrDie("modular_lossy_palette",
                              static_cast<int32_t>(args.modular_lossy_palette),
                              jxl_encoder_frame_settings,
                              JXL_ENC_FRAME_SETTING_LOSSY_PALETTE);
      process_flag("modular_palette_colors", args.modular_palette_colors,
                   JXL_ENC_FRAME_SETTING_PALETTE_COLORS,
                   [](int32_t x) -> std::string {
                     return -1 <= x ? ""
                                    : "Invalid --modular_palette_colors, must "
                                      "be -1 or non-negative\n";
                   });
      process_flag(
          "modular_channel_colors_global_percent",
          args.modular_channel_colors_global_percent,
          JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT,
          [](int32_t x) -> std::string {
            return (-1 <= x && x <= 100)
                       ? ""
                       : "Invalid --modular_channel_colors_global_percent. "
                         "Valid "
                         "range is {-1, 0, 1, ..., 100}.\n";
          });
      process_flag(
          "modular_channel_colors_group_percent",
          args.modular_channel_colors_group_percent,
          JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT,
          [](int32_t x) -> std::string {
            return (-1 <= x && x <= 100)
                       ? ""
                       : "Invalid --modular_channel_colors_group_percent. "
                         "Valid "
                         "range is {-1, 0, 1, ..., 100}.\n";
          });
    }
    ensure_image_loaded();
    if (args.lossless_jpeg && IsJPG(image_data)) {
      if (!cmdline.GetOption(args.opt_lossless_jpeg_id)->matched()) {
        std::cerr << "Note: Implicit-default for JPEG is lossless-transcoding. "
                  << "To silence this message, set --lossless_jpeg=(1|0)."
                  << std::endl;
      }
      if (args.jpeg_store_metadata) {
        if (JXL_ENC_SUCCESS !=
            JxlEncoderStoreJPEGMetadata(jxl_encoder, JXL_TRUE)) {
          std::cerr << "Storing JPEG metadata failed. " << std::endl;
          return EXIT_FAILURE;
        }
      }
      process_bool_flag("jpeg_reconstruction_cfl", args.jpeg_reconstruction_cfl,
                        JXL_ENC_FRAME_SETTING_JPEG_RECON_CFL);
      if (JXL_ENC_SUCCESS != JxlEncoderAddJPEGFrame(jxl_encoder_frame_settings,
                                                    image_data.data(),
                                                    image_data.size())) {
        std::cerr << "JxlEncoderAddJPEGFrame() failed." << std::endl;
        return EXIT_FAILURE;
      }
    } else {                          // Do JxlEncoderAddImageFrame().
      size_t num_alpha_channels = 0;  // Adjusted below.
      {
        JxlBasicInfo basic_info = ppf.info;
        if (basic_info.alpha_bits > 0) num_alpha_channels = 1;
        basic_info.intensity_target = args.intensity_target;
        basic_info.num_extra_channels = num_alpha_channels;
        basic_info.num_color_channels = ppf.info.num_color_channels;
        const bool lossless =
            args.distance == 0 ||
            (cmdline.GetOption(args.opt_quality_id)->matched() &&
             args.quality == 100);
        basic_info.uses_original_profile = lossless;
        if (args.override_bitdepth != 0) {
          basic_info.bits_per_sample = args.override_bitdepth;
          basic_info.exponent_bits_per_sample =
              args.override_bitdepth == 32 ? 8 : 0;
        }
        if (JXL_ENC_SUCCESS !=
            JxlEncoderSetCodestreamLevel(jxl_encoder, args.codestream_level)) {
          std::cerr << "Setting --codestream_level failed." << std::endl;
          return EXIT_FAILURE;
        }
        if (JXL_ENC_SUCCESS !=
            JxlEncoderSetBasicInfo(jxl_encoder, &basic_info)) {
          std::cerr << "JxlEncoderSetBasicInfo() failed." << std::endl;
          return EXIT_FAILURE;
        }
        if (lossless &&
            JXL_ENC_SUCCESS != JxlEncoderSetFrameLossless(
                                   jxl_encoder_frame_settings, JXL_TRUE)) {
          std::cerr << "JxlEncoderSetFrameLossless() failed." << std::endl;
          return EXIT_FAILURE;
        }
      }

      if (!ppf.icc.empty()) {
        if (JXL_ENC_SUCCESS != JxlEncoderSetICCProfile(jxl_encoder,
                                                       ppf.icc.data(),
                                                       ppf.icc.size())) {
          std::cerr << "JxlEncoderSetICCProfile() failed." << std::endl;
          return EXIT_FAILURE;
        }
      } else {
        if (JXL_ENC_SUCCESS !=
            JxlEncoderSetColorEncoding(jxl_encoder, &ppf.color_encoding)) {
          std::cerr << "JxlEncoderSetColorEncoding() failed." << std::endl;
          return EXIT_FAILURE;
        }
      }

      for (size_t num_frame = 0; num_frame < ppf.frames.size(); ++num_frame) {
        const jxl::extras::PackedFrame& pframe = ppf.frames[num_frame];
        const jxl::extras::PackedImage& pimage = pframe.color;
        JxlPixelFormat ppixelformat = pimage.format;
        {
          if (JXL_ENC_SUCCESS !=
              JxlEncoderSetFrameHeader(jxl_encoder_frame_settings,
                                       &pframe.frame_info)) {
            std::cerr << "JxlEncoderSetFrameHeader() failed." << std::endl;
            return EXIT_FAILURE;
          }
        }
        if (num_frame < args.frame_indexing.size() &&
            args.frame_indexing[num_frame] == '1') {
          if (JXL_ENC_SUCCESS !=
              JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                               JXL_ENC_FRAME_INDEX_BOX, 1)) {
            std::cerr << "Setting option JXL_ENC_FRAME_INDEX_BOX failed."
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
        JxlEncoderStatus enc_status;
        {
          if (num_alpha_channels > 0) {
            JxlExtraChannelInfo extra_channel_info;
            JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_ALPHA,
                                           &extra_channel_info);
            enc_status = JxlEncoderSetExtraChannelInfo(jxl_encoder, 0,
                                                       &extra_channel_info);
            if (JXL_ENC_SUCCESS != enc_status) {
              std::cerr << "JxlEncoderSetExtraChannelInfo() failed."
                        << std::endl;
              return EXIT_FAILURE;
            }
            if (args.premultiply != -1) {
              if (!(args.premultiply == 0 || args.premultiply == 1)) {
                std::cerr << "Flag --premultiply must be one of: -1, 0, 1."
                          << std::endl;
                return EXIT_FAILURE;
              }
              extra_channel_info.alpha_premultiplied = args.premultiply;
            }
            // We take the extra channel blend info frame_info, but don't do
            // clamping.
            JxlBlendInfo extra_channel_blend_info =
                pframe.frame_info.layer_info.blend_info;
            extra_channel_blend_info.clamp = JXL_FALSE;
            JxlEncoderSetExtraChannelBlendInfo(jxl_encoder_frame_settings, 0,
                                               &extra_channel_blend_info);
          }
          enc_status =
              JxlEncoderAddImageFrame(jxl_encoder_frame_settings, &ppixelformat,
                                      pimage.pixels(), pimage.pixels_size);
          if (JXL_ENC_SUCCESS != enc_status) {
            std::cerr << "JxlEncoderAddImageFrame() failed." << std::endl;
            return EXIT_FAILURE;
          }
          // Only set extra channel buffer if is is provided non-interleaved.
          if (!pframe.extra_channels.empty()) {
            enc_status = JxlEncoderSetExtraChannelBuffer(
                jxl_encoder_frame_settings, &ppixelformat,
                pframe.extra_channels[0].pixels(),
                pframe.extra_channels[0].stride *
                    pframe.extra_channels[0].ysize,
                0);
            if (JXL_ENC_SUCCESS != enc_status) {
              std::cerr << "JxlEncoderSetExtraChannelBuffer() failed."
                        << std::endl;
              return EXIT_FAILURE;
            }
          }
        }
      }
    }
    JxlEncoderCloseInput(jxl_encoder);
  }
  // Reading compressed output
  std::vector<uint8_t> compressed;
  compressed.resize(4096);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size() - (next_out - compressed.data());
  JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
  while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
    process_result =
        JxlEncoderProcessOutput(jxl_encoder, &next_out, &avail_out);
    if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() * 2);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    }
  }
  compressed.resize(next_out - compressed.data());
  if (JXL_ENC_SUCCESS != process_result) {
    std::cerr << "JxlEncoderProcessOutput failed." << std::endl;
    return EXIT_FAILURE;
  }

  // TODO(firsching): print info about compressed size and other image stats
  // here and in the beginning, like is done in current cjxl.
  if (args.file_out) {
    if (!WriteFile(compressed, args.file_out)) {
      std::cerr << "Could not write jxl file." << std::endl;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
