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
#include <vector>

#include "gflags/gflags.h"
#include "jxl/codestream_header.h"
#include "jxl/encode.h"
#include "jxl/encode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "jxl/types.h"
#include "lib/extras/codec.h"
#include "lib/extras/dec/apng.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/gif.h"
#include "lib/extras/dec/jpg.h"
#include "lib/extras/dec/pgx.h"
#include "lib/extras/dec/pnm.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/size_constraints.h"

DECLARE_bool(help);
DECLARE_bool(helpshort);
// The flag --version is owned by gflags itself.
DEFINE_bool(encoder_version, false,
            "Print encoder library version number and exit.");

DEFINE_bool(lossless_jpeg, true,
            "If the input is JPEG, use JxlEncoderAddJPEGFrame "
            "to add a JPEG frame  (i.e. losslessly transcoding JPEG), "
            "rather than using JxlEncoderAddImageFrame to reencode pixels.");

DEFINE_bool(jpeg_store_metadata, true,
            "If --lossless_jpeg is set, store JPEG reconstruction "
            "metadata in the JPEG XL container "
            "(for lossless reconstruction of the JPEG codestream).");

DEFINE_bool(jpeg_reconstruction_cfl, true,
            "Enable/disable chroma-from-luma (CFL) for lossless "
            "JPEG reconstruction.");

DEFINE_bool(container, false,
            "Force using container format (default: use only if needed).");

DEFINE_bool(strip, false,
            "Do not encode using container format (strips "
            "Exif/XMP/JPEG bitstream reconstruction data).");

DEFINE_bool(responsive, false, "[modular encoding] Do Squeeze transform");

DEFINE_bool(progressive, false, "Enable progressive/responsive decoding.");

DEFINE_bool(progressive_ac, false, "Use progressive mode for AC.");

DEFINE_bool(qprogressive_ac, false, "Use progressive mode for AC.");

DEFINE_bool(modular_lossy_palette, false, "Use delta-palette.");

DEFINE_int32(premultiply, -1,
             "Force premultiplied (associated) alpha. "
             "-1 = Do what the input does, 0 = Do not premultiply, "
             "1 = force premultiply.");

DEFINE_bool(already_downsampled, false,
            "Do not downsample the given input before encoding, "
            "but still signal that the decoder should upsample.");

DEFINE_bool(
    modular, false,
    "Use modular mode (not provided = encoder chooses, 0 = enforce VarDCT, "
    "1 = enforce modular mode).");

DEFINE_bool(keep_invisible, false,
            "Force disable/enable preserving color of invisible "
            "pixels. (not provided = default, 0 = disable, 1 = enable).");

DEFINE_bool(dots, false,
            "Force disable/enable dots generation. "
            "(not provided = default, 0 = disable, 1 = enable).");

DEFINE_bool(patches, false,
            "Force disable/enable patches generation. "
            "(not provided = default, 0 = disable, 1 = enable).");

DEFINE_bool(gaborish, false,
            "Force disable/enable the gaborish filter. "
            "(not provided = default, 0 = disable, 1 = enable).");

DEFINE_bool(
    group_order, false,
    "Order in which 256x256 regions are stored "
    "in the codestream for progressive rendering. "
    "Value not provided means 'encoder default', 0 means 'scanline order', "
    "1 means 'center-first order'.");

DEFINE_double(
    intensity_target, 0.0,
    "Upper bound on the intensity level present in the image in nits. "
    "Leaving this set to its default of 0 lets libjxl choose a sensible "
    "default "
    "value based on the color encoding.");

// TODO(tfish):
// --dec-hints, -- NEED (passed to image decoders, via extras, tweaks decoding)
// --override_bitdepth, -- NEED

DEFINE_int32(progressive_dc, -1,
             "Progressive-DC setting. Valid values are: -1, 0, 1, 2.");

DEFINE_int32(faster_decoding, 0,
             "Favour higher decoding speed. 0 = default, higher "
             "values give higher speed at the expense of quality");

DEFINE_int32(
    resampling, -1,
    "Resampling. Default of -1 applies resampling only for low quality. "
    "Value 1 does no downsampling (1x1), 2 does 2x2 downsampling, "
    "4 is for 4x4 downsampling, and 8 for 8x8 downsampling.");

DEFINE_int32(
    ec_resampling, -1,
    "Resampling for extra channels. Default of -1 applies resampling only "
    "for low quality. Value 1 does no downsampling (1x1), 2 does 2x2 "
    "downsampling, 4 is for 4x4 downsampling, and 8 for 8x8 downsampling.");

DEFINE_int32(
    epf, -1,
    "Edge preserving filter level, -1 to 3. "
    "Value -1 means: default (encoder chooses), 0 to 3 set a strength.");

DEFINE_int64(
    center_x, -1,
    "Determines the horizontal position of center for the center-first "
    "group order. The value -1 means 'use the middle of the image', "
    "other values [0..xsize) set this to a particular coordinate.");

DEFINE_int64(center_y, -1,
             "Determines the vertical position of center for the center-first "
             "group order. The value -1 means 'use the middle of the image', "
             "other values [0..ysize) set this to a particular coordinate.");

DEFINE_int64(num_threads, -1,
             "Number of worker threads (-1 == use machine default, "
             "0 == do not use multithreading).");

DEFINE_int64(num_reps, 1, "How many times to compress. (For benchmarking).");

DEFINE_int32(modular_group_size, -1,
             "[modular encoding] group size: -1 == default. 0 => 128, "
             "1 => 256, 2 => 512, 3 => 1024");

DEFINE_int32(modular_predictor, -1,
             "[modular encoding] predictor(s) to use: 0=zero, "
             "1=left, 2=top, 3=avg0, 4=select, 5=gradient, 6=weighted, "
             "7=topright, 8=topleft, 9=leftleft, 10=avg1, 11=avg2, 12=avg3, "
             "13=toptop predictive average "
             "14=mix 5 and 6, 15=mix everything. If unset, uses default 14, "
             "at slowest speed default 15.");

DEFINE_int32(modular_colorspace, -1,
             "[modular encoding] color transform: -1=default, 0=RGB (none), "
             "1-48=RCT (6=YCoCg, default: try several, depending on speed)");

DEFINE_int32(
    modular_channel_colors_global_percent, -1,
    "[modular encoding] Use Global channel palette if the number of "
    "colors is smaller than this percentage of range. "
    "Use 0-100 to set an explicit percentage, -1 to use the encoder default.");

DEFINE_int32(
    modular_channel_colors_group_percent, -1,
    "[modular encoding] Use Local (per-group) channel palette if the number "
    "of colors is smaller than this percentage of range. Use 0-100 to set "
    "an explicit percentage, -1 to use the encoder default.");

DEFINE_int32(
    modular_palette_colors, -1,
    "[modular encoding] Use color palette if number of colors is smaller "
    "than or equal to this, or -1 to use the encoder default.");

DEFINE_int32(modular_nb_prev_channels, -1,
             "[modular encoding] number of extra MA tree properties to use");

DEFINE_int32(modular_ma_tree_learning_percent, -1,
             "[modular encoding] Fraction of pixels used to learn MA trees as "
             "a percentage. -1 = default, 0 = no MA and fast decode, 50 = "
             "default value, 100 = all, values above 100 are also permitted. "
             "Higher values use more encoder memory.");

DEFINE_int32(photon_noise_iso, 0,
             "Adds noise to the image emulating photographic film noise. "
             "The higher the given number, the grainier the image will be. "
             "As an example, a value of 100 gives low noise whereas a value "
             "of 3200 gives a lot of noise. The default value is 0.");

DEFINE_int32(codestream_level, 5, "The codestream level. Either `5` or `10`.");

DEFINE_double(
    distance, 1.0,
    "Max. butteraugli distance, lower = higher quality.\n"
    "    0.0 = mathematically lossless. Default for already-lossy input "
    "(JPEG/GIF).\n"
    "    1.0 = visually lossless. Default for other input.\n"
    "    Recommended range: 0.5 .. 3.0. Mutually exclusive with --quality.");

DEFINE_double(
    quality, 100.0,
    "Quality setting (is remapped to --distance). Range: -inf .. 100.\n"
    "    100 = mathematically lossless. Default for already-lossy input "
    "(JPEG/GIF).\n"
    "    Other input gets encoded as per --distance default.\n"
    "    Positive quality values roughly match libjpeg quality.\n"
    "    Mutually exclusive with --distance.");

DEFINE_int64(effort, 3,
             "Encoder effort setting. Range: 1 .. 9.\n"
             "     Higher number is more effort (slower).");

DEFINE_int32(brotli_effort, 9,
             "Brotli effort setting. Range: 0 .. 11.\n"
             "    Default: 9. Higher number is more effort (slower).");

DEFINE_string(frame_indexing, "",
              // TODO(tfish): Add a more convenient vanilla alternative.
              "If non-empty, a string matching '^[01]*$'. If this string has a "
              "'1' in i-th position, then the i-th frame will be indexed in "
              "the frame index box.");

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
                          const jxl::extras::Codec& codec) {
  bool distance_set =
      !gflags::GetCommandLineFlagInfoOrDie("distance").is_default;
  bool quality_set = !gflags::GetCommandLineFlagInfoOrDie("quality").is_default;

  if (distance_set && quality_set) {
    std::cerr << "Must not set both --distance and --quality." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (distance_set) {
    if (JXL_ENC_SUCCESS != JxlEncoderSetFrameDistance(
                               jxl_encoder_frame_settings, FLAGS_distance)) {
      std::cerr << "Setting --distance parameter failed." << std::endl;
      exit(EXIT_FAILURE);
    }
    return;
  }
  if (quality_set) {
    double distance = FLAGS_quality >= 100 ? 0.0
                      : FLAGS_quality >= 30
                          ? 0.1 + (100 - FLAGS_quality) * 0.09
                          : 6.4 + pow(2.5, (30 - FLAGS_quality) / 5.0) / 6.25;
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

typedef std::function<std::string(int32_t)> flag_check_fn;

bool IsJPG(const jxl::PaddedBytes& image_data) {
  return (image_data.size() >= 2 && image_data[0] == 0xFF &&
          image_data[1] == 0xD8);
}

void SetCodestreamLevel(JxlEncoder* jxl_encoder, bool for_lossless_jpeg) {
  bool flag_set =
      !gflags::GetCommandLineFlagInfoOrDie("codestream_level").is_default;
  int32_t codestream_level = FLAGS_codestream_level;
  auto set_codestream_level = [&jxl_encoder, &codestream_level]() {
    if (JXL_ENC_SUCCESS !=
        JxlEncoderSetCodestreamLevel(jxl_encoder, codestream_level)) {
      std::cerr << "Setting --codestream_level failed." << std::endl;
      exit(EXIT_FAILURE);
    }
  };
  if (for_lossless_jpeg) {
    if (!flag_set) {
      set_codestream_level();
    }
  } else {
    if (!flag_set) {
      codestream_level = static_cast<int32_t>(
          JxlEncoderGetRequiredCodestreamLevel(jxl_encoder));
      if (codestream_level == -1) {
        std::cerr << "No codestream_level supports the given image parameters."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    set_codestream_level();
  }
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

void set_usage_message_and_version(const char* argv0) {
  gflags::SetUsageMessage(
      "JPEG XL-encodes an image.\n"
      " Input format can be one of: "
#if JPEGXL_ENABLE_APNG
      "PNG, APNG, "
#endif
#if JPEGXL_ENABLE_GIF
      "GIF, "
#endif
#if JPEGXL_ENABLE_JPEG
      "JPEG, "
#endif
      "PPM, PFM, PGX.\n  Sample usage:\n" +
      std::string(argv0) + " <source_image_filename> <target_image_filename>");
  uint32_t version = JxlEncoderVersion();

  gflags::SetVersionString(std::to_string(version / 1000000) + "." +
                           std::to_string((version / 1000) % 1000) + "." +
                           std::to_string(version % 1000));
}

}  // namespace

int main(int argc, char** argv) {
  std::cerr << "Warning: This is work in progress, consider using cjxl instead!"
            << std::endl;
  set_usage_message_and_version(argv[0]);
  // TODO(firsching): rethink --help handling
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, /*remove_flags=*/true);
  if (FLAGS_help) {
    FLAGS_help = false;
    FLAGS_helpshort = true;
  }
  gflags::HandleCommandLineHelpFlags();

  if (argc != 3) {
    FLAGS_help = false;
    FLAGS_helpshort = true;
    gflags::HandleCommandLineHelpFlags();
    return EXIT_FAILURE;
  }
  const char* filename_in = argv[1];
  const char* filename_out = argv[2];

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
  auto ensure_image_loaded = [&filename_in, &input_image_loaded, &image_data,
                              &ppf, &codec]() {
    if (input_image_loaded) return;
    if (!ReadFile(filename_in, &image_data)) {
      std::cerr << "Reading image data failed." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!(FLAGS_lossless_jpeg && IsJPG(image_data))) {
      jxl::Status status = GetPixeldata(image_data, ppf, codec);
      if (!status) {
        std::cerr << "Getting pixel data." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (ppf.frames.size() < 1) {
        std::cerr << "No frames on input file." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    input_image_loaded = true;
  };

  JxlEncoderPtr enc = JxlEncoderMake(/*memory_manager=*/nullptr);
  JxlEncoder* jxl_encoder = enc.get();
  JxlThreadParallelRunnerPtr runner;
  for (int num_rep = 0; num_rep < FLAGS_num_reps; ++num_rep) {
    JxlEncoderReset(jxl_encoder);
    if (FLAGS_num_threads != 0) {
      size_t num_worker_threads =
          JxlThreadParallelRunnerDefaultNumWorkerThreads();
      {
        int64_t flag_num_worker_threads = FLAGS_num_threads;
        if (flag_num_worker_threads != -1) {
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
                            flag_check_fn flag_check) {
      gflags::CommandLineFlagInfo flag_info =
          gflags::GetCommandLineFlagInfoOrDie(flag_name);
      if (!flag_info.is_default) {
        std::string error = flag_check(flag_value);
        if (!error.empty()) {
          std::cerr << "Invalid flag value for --" << flag_name << ": " << error
                    << std::endl;
          exit(EXIT_FAILURE);
        }
        SetFlagFrameOptionOrDie(flag_name, flag_value,
                                jxl_encoder_frame_settings, encoder_option);
      }
    };

    auto process_bool_flag = [&process_flag](
                                 const char* flag_name, int32_t flag_value,
                                 JxlEncoderFrameSettingId encoder_option) {
      process_flag(flag_name, static_cast<int32_t>(flag_value), encoder_option,
                   [](int32_t x) { return ""; });
    };

    {  // Processing tuning flags.
      bool use_container = FLAGS_container;
      // TODO(tfish): Set use_container according to need of encoded data.
      // This will likely require moving this piece out of flags-processing.
      if (FLAGS_strip) {
        use_container = false;
      }
      JxlEncoderUseContainer(jxl_encoder, use_container);

      process_bool_flag("modular", FLAGS_modular,
                        JXL_ENC_FRAME_SETTING_MODULAR);
      process_bool_flag("keep_invisible", FLAGS_keep_invisible,
                        JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE);
      process_bool_flag("dots", FLAGS_dots, JXL_ENC_FRAME_SETTING_DOTS);
      process_bool_flag("patches", FLAGS_patches,
                        JXL_ENC_FRAME_SETTING_PATCHES);
      process_bool_flag("gaborish", FLAGS_gaborish,
                        JXL_ENC_FRAME_SETTING_GABORISH);
      process_bool_flag("group_order", FLAGS_group_order,
                        JXL_ENC_FRAME_SETTING_GROUP_ORDER);

      if (!FLAGS_frame_indexing.empty()) {
        bool must_be_all_zeros = FLAGS_frame_indexing[0] != '1';
        for (char c : FLAGS_frame_indexing) {
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
          "effort", FLAGS_effort, JXL_ENC_FRAME_SETTING_EFFORT,
          [](int32_t x) -> std::string {
            return (1 <= x && x <= 9) ? "" : "Valid range is {1, 2, ..., 9}.";
          });
      process_flag(
          "brotli_effort", FLAGS_brotli_effort,
          JXL_ENC_FRAME_SETTING_BROTLI_EFFORT, [](int32_t x) -> std::string {
            return (-1 <= x && x <= 11) ? ""
                                        : "Valid range is {-1, 0, 1, ..., 11}.";
          });
      process_flag("epf", FLAGS_epf, JXL_ENC_FRAME_SETTING_EPF,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 3)
                                ? ""
                                : "Valid range is {-1, 0, 1, 2, 3}.\n";
                   });
      process_flag(
          "faster_decoding", FLAGS_faster_decoding,
          JXL_ENC_FRAME_SETTING_DECODING_SPEED, [](int32_t x) -> std::string {
            return (0 <= x && x <= 4) ? ""
                                      : "Valid range is {0, 1, 2, 3, 4}.\n";
          });
      process_flag("resampling", FLAGS_resampling,
                   JXL_ENC_FRAME_SETTING_RESAMPLING,
                   [](int32_t x) -> std::string {
                     return (x == -1 || x == 1 || x == 4 || x == 8)
                                ? ""
                                : "Valid values are {-1, 1, 2, 4, 8}.\n";
                   });
      process_flag("ec_resampling", FLAGS_ec_resampling,
                   JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING,
                   [](int32_t x) -> std::string {
                     return (x == -1 || x == 1 || x == 4 || x == 8)
                                ? ""
                                : "Valid values are {-1, 1, 2, 4, 8}.\n";
                   });
      process_flag("photon_noise_iso", FLAGS_photon_noise_iso,
                   JXL_ENC_FRAME_SETTING_PHOTON_NOISE,
                   [](int32_t x) -> std::string {
                     return x >= 0 ? "" : "Must be >= 0.";
                   });
      process_bool_flag("already_downsampled", FLAGS_already_downsampled,
                        JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED);
      SetDistanceFromFlags(jxl_encoder_frame_settings, codec);

      if (!FLAGS_group_order &&
          (FLAGS_center_x != -1 || FLAGS_center_y != -1)) {
        std::cerr
            << "Invalid flag combination. Setting --center_x or --center_y "
            << "requires setting --group_order=1" << std::endl;
        return EXIT_FAILURE;
      }
      process_flag("center_x", FLAGS_center_x,
                   JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X,
                   [](int32_t x) -> std::string {
                     if (x < -1) {
                       return "Valid values are: -1 or [0 .. xsize).";
                     }
                     return "";
                   });
      process_flag("center_y", FLAGS_center_y,
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
          !gflags::GetCommandLineFlagInfoOrDie("qprogressive_ac").is_default;
      int32_t qprogressive_ac = FLAGS_qprogressive_ac ? 1 : 0;
      bool responsive_set =
          !gflags::GetCommandLineFlagInfoOrDie("responsive").is_default;
      int32_t responsive = FLAGS_responsive ? 1 : 0;

      process_flag(
          "progressive_dc", FLAGS_progressive_dc,
          JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, [](int32_t x) -> std::string {
            return (-1 <= x && x <= 2) ? "" : "Valid range is {-1, 0, 1, 2}.\n";
          });
      process_bool_flag("progressive_ac", FLAGS_progressive_ac,
                        JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC);

      if (FLAGS_progressive) {
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
      process_flag("modular_group_size", FLAGS_modular_group_size,
                   JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 3)
                                ? ""
                                : "Invalid --modular_group_size. Valid "
                                  "range is {-1, 0, 1, 2, 3}.\n";
                   });
      process_flag("modular_predictor", FLAGS_modular_predictor,
                   JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 15)
                                ? ""
                                : "Invalid --modular_predictor. Valid "
                                  "range is {-1, 0, 1, ..., 15}.\n";
                   });
      process_flag(
          "modular_colorspace", FLAGS_modular_colorspace,
          JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE,
          [](int32_t x) -> std::string {
            return (-1 <= x && x <= 41)
                       ? ""
                       : "Invalid --modular_colorspace. Valid range is "
                         "{-1, 0, 1, ..., 41}.\n";
          });
      process_flag("modular_ma_tree_learning_percent",
                   FLAGS_modular_ma_tree_learning_percent,
                   JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 100)
                                ? ""
                                : "Invalid --modular_ma_tree_learning_percent. "
                                  "Valid range is {-1, 0, 1, ..., 100}.\n";
                   });
      process_flag("modular_nb_prev_channels", FLAGS_modular_nb_prev_channels,
                   JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS,
                   [](int32_t x) -> std::string {
                     return (-1 <= x && x <= 11)
                                ? ""
                                : "Invalid --modular_nb_prev_channels. Valid "
                                  "range is {-1, 0, 1, ..., 11}.\n";
                   });
      process_bool_flag("modular_lossy_palette", FLAGS_modular_lossy_palette,
                        JXL_ENC_FRAME_SETTING_LOSSY_PALETTE);
      process_flag("modular_palette_colors", FLAGS_modular_palette_colors,
                   JXL_ENC_FRAME_SETTING_PALETTE_COLORS,
                   [](int32_t x) -> std::string { return ""; });
      process_flag(
          "modular_channel_colors_global_percent",
          FLAGS_modular_channel_colors_global_percent,
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
          FLAGS_modular_channel_colors_group_percent,
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
    if (FLAGS_lossless_jpeg && IsJPG(image_data)) {
      if (gflags::GetCommandLineFlagInfoOrDie("lossless_jpeg").is_default) {
        std::cerr << "Note: Implicit-default for JPEG is lossless-transcoding. "
                  << "To silence this message, set --lossless_jpeg=(1|0)."
                  << std::endl;
      }
      if (FLAGS_jpeg_store_metadata) {
        if (JXL_ENC_SUCCESS != JxlEncoderStoreJPEGMetadata(jxl_encoder, true)) {
          std::cerr << "Storing JPEG metadata failed. " << std::endl;
          return EXIT_FAILURE;
        }
      }
      process_bool_flag("jpeg_reconstruction_cfl",
                        FLAGS_jpeg_reconstruction_cfl,
                        JXL_ENC_FRAME_SETTING_JPEG_RECON_CFL);
      SetCodestreamLevel(jxl_encoder, /*for_lossless_jpeg=*/true);
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
        basic_info.intensity_target =
            static_cast<float>(FLAGS_intensity_target);
        basic_info.num_extra_channels = num_alpha_channels;
        basic_info.num_color_channels = ppf.info.num_color_channels;
        basic_info.uses_original_profile = JXL_FALSE;
        if (JXL_ENC_SUCCESS !=
            JxlEncoderSetBasicInfo(jxl_encoder, &basic_info)) {
          std::cerr << "JxlEncoderSetBasicInfo() failed." << std::endl;
          return EXIT_FAILURE;
        }
        SetCodestreamLevel(jxl_encoder, /*for_lossless_jpeg=*/false);
      }

      if (!ppf.icc.empty()) {
        JxlEncoderSetICCProfile(jxl_encoder, ppf.icc.data(), ppf.icc.size());
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
        if (num_frame < FLAGS_frame_indexing.size() &&
            FLAGS_frame_indexing[num_frame] == '1') {
          if (JXL_ENC_SUCCESS !=
              JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                               JXL_ENC_FRAME_INDEX_BOX, 1)) {
            std::cerr << "Setting option JXL_ENC_FRAME_INDEX_BOX failed."
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
        jxl::Status enc_status(true);
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
            if (FLAGS_premultiply != -1) {
              if (!(FLAGS_premultiply == 0 || FLAGS_premultiply == 1)) {
                std::cerr << "Flag --premultiply must be one of: -1, 0, 1."
                          << std::endl;
                return EXIT_FAILURE;
              }
              extra_channel_info.alpha_premultiplied = FLAGS_premultiply;
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
  if (!WriteFile(compressed, filename_out)) {
    std::cerr << "Could not write jxl file." << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
