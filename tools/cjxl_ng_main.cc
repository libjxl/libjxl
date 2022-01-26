// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <stdint.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "jxl/encode.h"
#include "jxl/encode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "lib/extras/codec.h"
#include "lib/extras/dec/apng.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/gif.h"
#include "lib/extras/dec/jpg.h"
#include "lib/extras/dec/pgx.h"
#include "lib/extras/dec/pnm.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/size_constraints.h"

DECLARE_bool(help);
DECLARE_bool(helpshort);
// The flag --version is owned by gflags itself.
DEFINE_bool(encoder_version, false,
            "Print encoder library version number and exit.");

DEFINE_bool(add_jpeg_frame, false,
            // This supersedes --jpeg_transcode
            "Use JxlEncoderAddJPEGFrame to add a JPEG frame, "
            "rather than JxlEncoderAddImageFrame.");

DEFINE_bool(jpeg_store_metadata, false,
            "If --add_jpeg_frame is set, store JPEG reconstruction "
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

DEFINE_bool(responsive, false, "[modular encoding] do Squeeze transform");

DEFINE_bool(progressive, false, "Enable progressive/responsive decoding.");

DEFINE_bool(progressive_ac, false, "Use progressive mode for AC.");

DEFINE_bool(qprogressive_ac, false, "Use progressive mode for AC.");

DEFINE_bool(modular_lossy_palette, false, "Use delta-palette.");

DEFINE_bool(jpeg_transcode, false,  // TODO(tfish): Wire this up.
            "Do lossy transcode of input JPEG file (decode to "
            "pixels instead of doing lossless transcode).");

DEFINE_bool(premultiply, false,  // TODO(tfish): Wire this up.
            "Force premultiplied (associated) alpha.");

DEFINE_bool(verbose, false,
            // TODO(tfish): Should be a verbosity-level.
            // Original cjxl also makes --help more verbose if this is on,
            // but with gflags, we do that differently...?
            "Verbose output.");

DEFINE_bool(already_downsampled, false,
            "Do not downsample the given input before encoding, "
            "but still signal that the decoder should upsample.");

DEFINE_bool(
    modular, false,
    // TODO(tfish): Flag up parameter meaning change.
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
    // TODO(tfish): This is a new flag. Check with team.
    "Order in which 256x256 regions are stored "
    "in the codestream for progressive rendering. "
    "Value not provided means 'encoder default', 0 means 'scanline order', "
    "1 means 'center-first order'.");
// TODO(tfish):
// --intensity_target,
// --saliency_num_progressive_steps, --saliency_map_filename,
// --saliency_threshold, --dec-hints, --override_bitdepth,
// --mquality, --iterations,
// --extra-properties, --lossy-palette, --pre-compact,
// --post-compact

DEFINE_int32(progressive_dc, -1,
             "Progressive-DC setting. Valid values are: -1, 0, 1, 2.");

DEFINE_int32(faster_decoding, 0,
             "Favour higher decoding speed. 0 = default, higher "
             "values give higher speed at the expense of quality");

DEFINE_int32(
    resampling, -1,
    // TODO(tfish): Discuss with team. The new docstring is from the C API
    // documentation. This differs from what the old docstring said.
    "Resampling. Default of -1 applies resampling only for low quality. "
    "Value 1 does no downsampling (1x1), 2 does 2x2 downsampling, "
    "4 is for 4x4 downsampling, and 8 for 8x8 downsampling.");

DEFINE_int32(
    ec_resampling, -1,
    // TODO(tfish): Discuss with team. The new docstring is from the C API
    // documentation. This differs from what the old docstring said.
    "Resampling for extra channels. Default of -1 applies resampling only "
    "for low quality. Value 1 does no downsampling (1x1), 2 does 2x2 "
    "downsampling, 4 is for 4x4 downsampling, and 8 for 8x8 downsampling.");

DEFINE_int32(
    epf, -1,
    "Edge preserving filter level, -1 to 3. "
    "Value -1 means: default (encoder chooses), 0 to 3 set a strength.");

DEFINE_int64(
    center_x, -1,
    // TODO(tfish): Clarify if this is really the comment we want here.
    "Determines the horizontal position of center for the center-first "
    "group order. The value -1 means 'use the middle of the image', "
    // TODO(tfish): Clarify if encode.h has an off-by-one in the
    // upper limit here.
    "other values 0..(xsize-1) set this to a particular coordinate.");

DEFINE_int64(center_y, -1,
             // TODO(tfish): Clarify if this is really the comment we want here.
             "Determines the vertical position of center for the center-first "
             "group order. The value -1 means 'use the middle of the image', "
             // TODO(tfish): Clarify if encode.h has an off-by-one in the
             // upper limit here.
             "other values 0..(ysize-1) set this to a particular coordinate.");

DEFINE_int64(num_threads, 0,
             // TODO(tfish): Sync with team about changed meaning of 0 -
             // was: No multithreaded workers. Is: use default number.
             "Number of worker threads (0 == use machine default).");

DEFINE_int64(num_reps, 1,  // TODO(tfish): wire this up.
                           // TODO(tfish): Clarify meaning of this docstring.
                           // Is this simply for benchmarking?
             "How many times to compress.");

DEFINE_int32(modular_group_size, -1,
             // TODO(tfish): Clarify with team if renaming group_size
             // -> modular_group_size (to align with C API names) is
             // ok.
             "[modular encoding] group size: -1 == default. 0 => 128, "
             "1 => 256, 2 => 512, 3 => 1024");

DEFINE_int32(modular_predictor, 15,
             // TODO(tfish): Clarify renaming, as for modular_group_size
             "[modular encoding] predictor(s) to use: 0=zero, "
             "1=left, 2=top, 3=avg0, 4=select, 5=gradient, 6=weighted, "
             "7=topright, 8=topleft, 9=leftleft, 10=avg1, 11=avg2, 12=avg3, "
             "13=toptop predictive average "
             "14=mix 5 and 6, 15=mix everything. Default 14, at slowest speed "
             "default 15");

DEFINE_int32(modular_colorspace, -1,
             // TODO(tfish): Clarify renaming, as for modular_group_size
             "[modular encoding] color transform: 0=RGB, 1=YCoCg, "
             "2-37=RCT (default: try several, depending on speed)");

DEFINE_int32(modular_nb_prev_channels, -1,
             // TODO(tfish): Clarify renaming (from --extra-properties),
             // as for --modular_group_size. Is this actually the
             // correct parameter?
             "[modular encoding] number of extra MA tree properties to use");

DEFINE_int32(photon_noise, 0,
             // TODO(tfish): Discuss docstring change with team.
             // Also: This now is an int, no longer a float.
             "Adds noise to the image emulating photographic film noise. "
             "The higher the given number, the grainier the image will be. "
             "As an example, a value of 100 gives low noise whereas a value "
             "of 3200 gives a lot of noise. The default value is 0.");

DEFINE_int32(codestream_level, 5, "The codestream level. Either `5` or `10`.");

DEFINE_double(
    distance, 1.0,  // TODO(tfish): wire this up.
    "Max. butteraugli distance, lower = higher quality. Range: 0 .. 25.\n"
    "    0.0 = mathematically lossless. Default for already-lossy input "
    "(JPEG/GIF).\n"
    "    1.0 = visually lossless. Default for other input.\n"
    "    Recommended range: 0.5 .. 3.0.");

DEFINE_int64(target_size, 0,  // TODO(tfish): wire this up.
             "Aim at file size of N bytes.\n"
             "    Compresses to 1 % of the target size in ideal conditions.\n"
             "    Runs the same algorithm as --target_bpp");

DEFINE_double(target_bpp, 0.0,  // TODO(tfish): wire this up.
              "Aim at file size that has N bits per pixel.\n"
              "    Compresses to 1 % of the target BPP in ideal conditions.");

DEFINE_double(
    quality, 100.0,  // TODO(tfish): wire this up.
    "Quality setting (is remapped to --distance). Range: -inf .. 100.\n"
    "    100 = mathematically lossless. Default for already-lossy input "
    "(JPEG/GIF).\n    Positive quality values roughly match libjpeg "
    "quality.");

DEFINE_int64(
    effort, 7,
    // TODO(tfish): Clarify discrepancy with team:
    // Documentation says default==squirrel(7) here:
    // https://libjxl.readthedocs.io/en/latest/api_encoder.html#_CPPv424JxlEncoderFrameSettingId
    // but enc_params.h has kFalcon=7.
    "Encoder effort setting. Range: 1 .. 9.\n"
    "    Default: 7. Higher number is more effort (slower).");

DEFINE_string(
    // TODO(tfish): Clarify with team whether changing from int-param to string
    // is OK here.
    colortransform, "",
    "The color transform to use. Valid values are: '' (= \"use default\"), "
    "'RGB', 'XYB', 'YCbCr'.");

DEFINE_string(
    mquality, "",  // TODO(tfish): Wire this up.
    "[modular encoding] lossy 'quality', in the form luma_q[,chroma_q] "
    "(100=lossless, lower is more lossy)");

namespace {
/**
 * Writes bytes to file.
 */
bool WriteFile(const std::vector<uint8_t>& bytes, const char* filename) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr,
            "Could not open %s for writing\n"
            "Error: %s",
            filename, strerror(errno));
    return false;
  }
  if (fwrite(bytes.data(), sizeof(uint8_t), bytes.size(), file) !=
      bytes.size()) {
    fprintf(stderr,
            "Could not write bytes to %s\n"
            "Error: %s",
            filename, strerror(errno));
    return false;
  }
  if (fclose(file) != 0) {
    fprintf(stderr,
            "Could not close %s\n"
            "Error: %s",
            filename, strerror(errno));
    return false;
  }
  return true;
}

bool ProcessTristateFlag(const char* flag_name, const bool flag_value,
                         JxlEncoderFrameSettings* frame_settings,
                         JxlEncoderFrameSettingId encoder_option) {
  gflags::CommandLineFlagInfo flag_info =
      gflags::GetCommandLineFlagInfoOrDie(flag_name);
  if (!flag_info.is_default) {
    JxlEncoderFrameSettingsSetOption(frame_settings, encoder_option,
                                     static_cast<int32_t>(flag_value));
  }
  return true;
}

jxl::Status LoadInput(const char* filename_in,
                      jxl::extras::PackedPixelFile& ppf) {
  // Any valid encoding is larger (ensures codecs can read the first few bytes).
  constexpr size_t kMinBytes = 9;

  jxl::PaddedBytes image_data;
  jxl::Status status = ReadFile(filename_in, &image_data);
  if (!status) {
    return status;
  }
  if (image_data.size() < kMinBytes) return JXL_FAILURE("Input too small.");
  jxl::Span<const uint8_t> encoded(image_data);

  // Default values when not set by decoders.
  ppf.info.uses_original_profile = true;
  ppf.info.orientation = JXL_ORIENT_IDENTITY;
  jxl::extras::ColorHints color_hints;
  jxl::SizeConstraints size_constraints;

  jxl::extras::Codec codec;
  (void)codec;
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
  std::cerr << "Warning: This is work in progress, consider using cjxl "
               "instead!\n";
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
      std::string(argv[0]) +
      " <source_image_filename> <target_image_filename>");
  uint32_t version = JxlEncoderVersion();

  gflags::SetVersionString(std::to_string(version / 1000000) + "." +
                           std::to_string((version / 1000) % 1000) + "." +
                           std::to_string(version % 1000));
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

  size_t num_worker_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
  {
    int64_t flag_num_worker_threads = FLAGS_num_threads;
    if (flag_num_worker_threads != 0) {
      num_worker_threads = flag_num_worker_threads;
    }
  }
  auto enc = JxlEncoderMake(/*memory_manager=*/nullptr);
  auto runner = JxlThreadParallelRunnerMake(
      /*memory_manager=*/nullptr, num_worker_threads);
  JxlEncoder* jxl_encoder = enc.get();
  if (JXL_ENC_SUCCESS != JxlEncoderSetParallelRunner(jxl_encoder,
                                                     JxlThreadParallelRunner,
                                                     runner.get())) {
    fprintf(stderr, "JxlEncoderSetParallelRunner failed\n");
    return EXIT_FAILURE;
  }

  JxlEncoderFrameSettings* jxl_encoder_frame_settings =
      JxlEncoderFrameSettingsCreate(jxl_encoder, nullptr);

  {  // Processing tuning flags.
    bool use_container = FLAGS_container;
    // TODO(tfish): Set use_container according to need of encoded data.
    // This will likely require moving this piece out of flags-processing.
    if (FLAGS_strip) {
      use_container = false;
    }
    JxlEncoderUseContainer(jxl_encoder, use_container);

    ProcessTristateFlag("modular", FLAGS_modular, jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_MODULAR);
    ProcessTristateFlag("keep_invisible", FLAGS_keep_invisible,
                        jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE);
    ProcessTristateFlag("dots", FLAGS_dots, jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_DOTS);
    ProcessTristateFlag("patches", FLAGS_patches, jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_PATCHES);
    ProcessTristateFlag("gaborish", FLAGS_gaborish, jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_GABORISH);
    ProcessTristateFlag("group_order", FLAGS_group_order,
                        jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_GROUP_ORDER);

    if (!gflags::GetCommandLineFlagInfoOrDie("codestream_level").is_default) {
      JxlEncoderSetCodestreamLevel(jxl_encoder, FLAGS_codestream_level);
    }

    const int32_t flag_effort = FLAGS_effort;
    // TODO(firsching): rethink if we might want to have a validator with a
    // (template?) parameter for the list of valid values.
    if (!(1 <= flag_effort && flag_effort <= 9)) {
      // Strictly speaking, custom gflags parsing would integrate
      // more nicely with gflags, but the boilerplate cost of
      // handling invalid calls is substantially higher than
      // this lightweight approach here.
      std::cerr << "Invalid --effort. Valid range is {1, 2, ..., 9}.\n";
      return EXIT_FAILURE;
    }
    JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                     JXL_ENC_FRAME_SETTING_EFFORT, flag_effort);

    const int32_t flag_epf = FLAGS_epf;
    if (!(-1 <= flag_epf && flag_epf <= 3)) {
      std::cerr << "Invalid --epf. Valid range is {-1, 0, 1, 2, 3}.\n";
      return EXIT_FAILURE;
    }
    if (flag_epf != -1) {
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_EPF, flag_epf);
    }

    const int32_t flag_faster_decoding = FLAGS_faster_decoding;
    if (!(0 <= flag_faster_decoding && flag_faster_decoding <= 4)) {
      std::cerr << "Invalid --faster_decoding. "
                   "Valid range is {0, 1, 2, 3, 4}.\n";
      return EXIT_FAILURE;
    }
    JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                     JXL_ENC_FRAME_SETTING_DECODING_SPEED,
                                     flag_faster_decoding);
    if (FLAGS_resampling != -1) {
      if (!(((FLAGS_resampling & (FLAGS_resampling - 1)) == 0) &&
            FLAGS_resampling <= 8)) {
        std::cerr << "Invalid --resampling. "
                     "Valid values are {-1, 1, 2, 4, 8}.\n";
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_RESAMPLING,
                                       FLAGS_resampling);
    }
    if (FLAGS_ec_resampling != -1) {
      if (!(((FLAGS_ec_resampling & (FLAGS_ec_resampling - 1)) == 0) &&
            FLAGS_ec_resampling <= 8)) {
        std::cerr << "Invalid --ec_resampling. "
                     "Valid values are {-1, 1, 2, 4, 8}.\n";
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings,
          JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, FLAGS_ec_resampling);
    }
    JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                     JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED,
                                     FLAGS_already_downsampled);

    JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                     JXL_ENC_FRAME_SETTING_PHOTON_NOISE,
                                     FLAGS_photon_noise);

    JxlEncoderSetFrameDistance(jxl_encoder_frame_settings, FLAGS_distance);
    if (FLAGS_center_x != -1) {
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings,
          JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X, FLAGS_center_x);
    }
    if (FLAGS_center_y != -1) {
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings,
          JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y, FLAGS_center_y);
    }
  }
  // Progressive/responsive mode settings.
  {
    // Are the corresponding flag-values explicitly or implicitly set?
    bool progressive_ac_set =
        !gflags::GetCommandLineFlagInfoOrDie("progressive_ac").is_default;
    bool qprogressive_ac_set =
        !gflags::GetCommandLineFlagInfoOrDie("qprogressive_ac").is_default;
    bool progressive_dc_set =
        !gflags::GetCommandLineFlagInfoOrDie("progressive_dc").is_default;
    bool responsive_set =
        !gflags::GetCommandLineFlagInfoOrDie("responsive").is_default;
    // Quantized-progressive mode.
    int32_t qprogressive_ac = FLAGS_qprogressive_ac ? 1 : 0;
    int32_t responsive = FLAGS_responsive ? 1 : 0;

    if (progressive_dc_set) {
      if (!(-1 <= FLAGS_progressive_dc && FLAGS_progressive_dc <= 2)) {
        std::cerr << "Invalid --progressive_dc. "
                     "Valid range is {-1, 0, 1, 2}.\n";
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC,
                                       FLAGS_progressive_dc);
    }
    if (FLAGS_progressive) {
      qprogressive_ac = 1;
      qprogressive_ac_set = true;
      responsive = 1;
      responsive_set = true;
    }
    if (progressive_ac_set) {
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC,
                                       FLAGS_progressive_ac);
    }
    if (responsive_set) {
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_RESPONSIVE,
                                       responsive);
    }
    if (qprogressive_ac_set) {
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC,
                                       qprogressive_ac);
    }
  }
  // Modular mode related
  {
    bool modular_group_size_set =
        !gflags::GetCommandLineFlagInfoOrDie("modular_group_size").is_default;
    bool modular_predictor_set =
        !gflags::GetCommandLineFlagInfoOrDie("modular_predictor").is_default;
    bool modular_colorspace_set =
        !gflags::GetCommandLineFlagInfoOrDie("modular_colorspace").is_default;
    bool modular_nb_prev_channels_set =
        !gflags::GetCommandLineFlagInfoOrDie("modular_nb_prev_channels")
             .is_default;
    bool modular_lossy_palette_set =
        !gflags::GetCommandLineFlagInfoOrDie("modular_lossy_palette")
             .is_default;

    if (modular_group_size_set) {
      if (!(FLAGS_modular_group_size == -1 ||
            (0 <= FLAGS_modular_group_size && FLAGS_modular_group_size <= 3))) {
        std::cerr << "Invalid --modular_group_size: "
                  << FLAGS_modular_group_size << std::endl;
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE,
                                       FLAGS_modular_group_size);
    }
    if (modular_predictor_set) {
      if (!(0 <= FLAGS_modular_predictor && FLAGS_modular_predictor <= 3)) {
        std::cerr << "Invalid --modular_predictor: " << FLAGS_modular_predictor
                  << std::endl;
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR,
                                       FLAGS_modular_predictor);
    }
    if (modular_colorspace_set) {
      if (!(-1 <= FLAGS_modular_colorspace && FLAGS_modular_colorspace <= 35)) {
        std::cerr << "Invalid --modular_colorspace: "
                  << FLAGS_modular_colorspace << std::endl;
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings, JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE,
          FLAGS_modular_colorspace);
    }
    if (modular_nb_prev_channels_set) {
      if (!(-1 <= FLAGS_modular_nb_prev_channels &&
            FLAGS_modular_nb_prev_channels <= 11)) {
        std::cerr << "Invalid --modular_nb_prev_channels: "
                  << FLAGS_modular_nb_prev_channels << std::endl;
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings,
          JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS,
          FLAGS_modular_nb_prev_channels);
    }
    if (modular_lossy_palette_set) {
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_LOSSY_PALETTE,
                                       FLAGS_modular_lossy_palette);
    }
  }
  // Color related (not for modular-mode)
  {
    // TODO(tfish): Clarify with team - old `cjxl` had some extra
    // "if quality is 100%" logic which has not been ported here.
    // Overall, the new rule "set it if provided" is more
    // straightforward than the old one, which needed the caller to
    // understand subtle dependencies of the "this flag is ignored
    // if those other flags are as follows" dependencies.
    // Should we nevertheless introduce the old logic?
    bool colortransform_set =
        !gflags::GetCommandLineFlagInfoOrDie("colortransform").is_default;

    if (colortransform_set) {
      int32_t colortransform = -1;
      if (FLAGS_colortransform == "XYB") {
        colortransform = 0;
      } else if (FLAGS_colortransform == "RGB") {
        colortransform = 1;
      } else if (FLAGS_colortransform == "YCbCr") {
        colortransform = 2;
      } else {
        std::cerr << "Invalid --colortransform: " << FLAGS_colortransform
                  << std::endl;
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM,
                                       colortransform);
    }
  }  // Processing flags.

  if (FLAGS_add_jpeg_frame) {
    jxl::PaddedBytes jpeg_data;
    if (!ReadFile(filename_in, &jpeg_data)) {
      std::cerr << "Reading image data failed.\n";
      return EXIT_FAILURE;
    }
    if (FLAGS_jpeg_store_metadata) {
      JxlEncoderStoreJPEGMetadata(jxl_encoder, true);
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("jpeg_reconstruction_cfl")
             .is_default) {
      JxlEncoderFrameSettingsSetOption(jxl_encoder_frame_settings,
                                       JXL_ENC_FRAME_SETTING_JPEG_RECON_CFL,
                                       FLAGS_jpeg_reconstruction_cfl ? 1 : 0);
    }
    if (JXL_ENC_SUCCESS != JxlEncoderAddJPEGFrame(jxl_encoder_frame_settings,
                                                  jpeg_data.data(),
                                                  jpeg_data.size())) {
      std::cerr << "JxlEncoderAddJPEGFrame() failed.\n";
      return EXIT_FAILURE;
    }
  } else {  // Do JxlEncoderAddImageFrame().
    jxl::extras::PackedPixelFile ppf;
    jxl::Status status = LoadInput(filename_in, ppf);
    if (!status) {
      // TODO(tfish): Fix such status handling throughout.  We should
      // have more detail available about what went wrong than what we
      // currently share with the caller.
      std::cerr << "Loading input file failed.\n";
      return EXIT_FAILURE;
    }
    if (ppf.frames.size() < 1) {
      std::cerr << "No frames on input file.\n";
      return EXIT_FAILURE;
    }

    if (ppf.icc.size() > 0) {
      JxlEncoderStatus enc_status =
          JxlEncoderSetICCProfile(jxl_encoder, ppf.icc.data(), ppf.icc.size());
      if (JXL_ENC_SUCCESS != enc_status) {
        std::cerr << "JxlEncoderSetICCProfile() failed.\n";
        return EXIT_FAILURE;
      }
    } else {
      if (JXL_ENC_SUCCESS !=
          JxlEncoderSetColorEncoding(jxl_encoder, &ppf.color_encoding)) {
        std::cerr << "JxlEncoderSetColorEncoding() failed.\n";
        return EXIT_FAILURE;
      }
    }

    for (const jxl::extras::PackedFrame& pframe : ppf.frames) {
      const jxl::extras::PackedImage& pimage = pframe.color;
      JxlPixelFormat ppixelformat = pimage.format;

      {  // JxlEncoderSetBasicInfo
        JxlBasicInfo basic_info;
        JxlEncoderInitBasicInfo(&basic_info);
        basic_info.xsize = pimage.xsize;
        basic_info.ysize = pimage.ysize;
        basic_info.bits_per_sample = 32;
        basic_info.exponent_bits_per_sample = 8;
        basic_info.num_extra_channels =
            (pimage.format.num_channels == 2 || pimage.format.num_channels == 4)
                ? 1
                : 0;
        basic_info.uses_original_profile = JXL_FALSE;
        if (JXL_ENC_SUCCESS !=
            JxlEncoderSetBasicInfo(jxl_encoder, &basic_info)) {
          std::cerr << "JxlEncoderSetBasicInfo() failed.\n";
          return EXIT_FAILURE;
        }
      }
      {
        jxl::Status enc_status = JxlEncoderSetFrameHeader(
            jxl_encoder_frame_settings, &pframe.frame_info);
        if (JXL_ENC_SUCCESS != enc_status) {
          std::cerr << "JxlEncoderSetFrameHeader() failed.\n";
          return EXIT_FAILURE;
        }
      }
      {
        jxl::Status enc_status =
            JxlEncoderAddImageFrame(jxl_encoder_frame_settings, &ppixelformat,
                                    pimage.pixels(), pimage.pixels_size);
        if (JXL_ENC_SUCCESS != enc_status) {
          // TODO(tfish): Fix such status handling throughout.  We should
          // have more detail available about what went wrong than what we
          // currently share with the caller.
          std::cerr << "JxlEncoderAddImageFrame() failed.\n";
          return EXIT_FAILURE;
        }
      }
    }
  }
  JxlEncoderCloseInput(jxl_encoder);

  // Reading compressed output
  std::vector<uint8_t> compressed;
  compressed.resize(64);
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
    fprintf(stderr, "JxlEncoderProcessOutput failed\n");
    return EXIT_FAILURE;
  }

  // TODO(firsching): print info about compressed size and other image stats
  // here and in the beginning, like is done in current cjxl.
  if (!WriteFile(compressed, filename_out)) {
    fprintf(stderr, "Couldn't write jxl file\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
