// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <iostream>
#include <vector>

#include <stdint.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/str_cat.h"

#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/padded_bytes.h"
#include "jxl/codestream_header.h"
#include "jxl/color_encoding.h"
#include "jxl/encode.h"
#include "jxl/types.h"

#include "jxl/thread_parallel_runner.h"

#include "fetch_encoded.h"


ABSL_FLAG(bool, version, false,
          "Print encoder library version number and exit.");

ABSL_FLAG(bool, container, false,
          "Force using container format (default: use only if needed).");

ABSL_FLAG(bool, strip, false,
          "Do not encode using container format (strips "
          "Exif/XMP/JPEG bitstream reconstruction data).");

ABSL_FLAG(bool, progressive, false,  // TODO(tfish): Wire this up.
          "Enable progressive/responsive decoding.");

ABSL_FLAG(bool, progressive_ac, false,  // TODO(tfish): Wire this up.
          "Use progressive mode for AC.");

ABSL_FLAG(bool, qprogressive_ac, false,  // TODO(tfish): Wire this up.
          // TODO(tfish): Clarify what this flag is about.
          "Use progressive mode for AC.");

ABSL_FLAG(bool, progressive_dc, false,  // TODO(tfish): Wire this up.
          "Use progressive mode for DC.");

ABSL_FLAG(bool, use_experimental_encoder_heuristics, false,  // TODO(tfish): Wire this up.
          "Use new and not yet ready encoder heuristics");

ABSL_FLAG(bool, jpeg_transcode, false,  // TODO(tfish): Wire this up.
          "Do lossy transcode of input JPEG file (decode to "
          "pixels instead of doing lossless transcode).");

ABSL_FLAG(bool, jpeg_transcode_disable_cfl, false,  // TODO(tfish): Wire this up.
          "Disable CFL for lossless JPEG recompression");

ABSL_FLAG(bool, premultiply, false,  // TODO(tfish): Wire this up.
          "Force premultiplied (associated) alpha.");

ABSL_FLAG(bool, centerfirst, false,  // TODO(tfish): Wire this up.
          "Put center groups first in the compressed file.");

// TODO(tfish): Clarify if this is indeed deprecated. Remove if it is.
// ABSL_FLAG(bool, noise, false,
//           "force disable/enable noise generation.");

ABSL_FLAG(bool, verbose, false,
          // TODO(tfish): Should be a verbosity-level.
          // Original cjxl also makes --help more verbose if this is on,
          // but with absl, we do that differently...?
          "Verbose output.");

ABSL_FLAG(bool, already_downsampled, false,
          "Do not downsample the given input before encoding, "
          "but still signal that the decoder should upsample.");


// TODO(tfish):
// --intensity_target,
// --saliency_num_progressive_steps, --saliency_map_filename,
// --saliency_threshold, --dec-hints, --override_bitdepth,
// --colortransform, --mquality, --iterations, --colorspace, --group-size,
// --predictor, --extra-properties, --lossy-palette, --pre-compact,
// --post-compact, --responsive, --quiet, --print_profile,


ABSL_FLAG(int32_t, faster_decoding, 0,
          "Favour higher decoding speed. 0 = default, higher "
          "values give higher speed at the expense of quality");

ABSL_FLAG(int32_t, resampling, -1,
          // TODO(tfish): Discuss with team. The new docstring is from the C API
          // documentation. This differs from what the old docstring said.
          "Resampling. Default of -1 applies resampling only for low quality. "
          "Value 1 does no downsampling (1x1), 2 does 2x2 downsampling, "
          "4 is for 4x4 downsampling, and 8 for 8x8 downsampling.");

ABSL_FLAG(int32_t, ec_resampling, -1,
          // TODO(tfish): Discuss with team. The new docstring is from the C API
          // documentation. This differs from what the old docstring said.
          "Resampling for extra channels. Default of -1 applies resampling only "
          "for low quality. Value 1 does no downsampling (1x1), 2 does 2x2 "
          "downsampling, 4 is for 4x4 downsampling, and 8 for 8x8 downsampling."
          );

ABSL_FLAG(int32_t, modular, -1,
          // TODO(tfish): Flag up parameter meaning change.
          "Use modular mode (-1 = encoder chooses, 0 = enforce VarDCT, "
          "1 = enforce modular mode).");

ABSL_FLAG(int32_t, keep_invisible, -1,
          "Force disable/enable preserving color of invisible "
          "pixels. (-1 = default, 0 = disable, 1 = enable).");

ABSL_FLAG(int32_t, dots, -1,
          "Force disable/enable dots generation. "
          "(-1 = default, 0 = disable, 1 = enable).");

ABSL_FLAG(int32_t, patches, -1,
          "Force disable/enable patches generation. "
          "(-1 = default, 0 = disable, 1 = enable).");

ABSL_FLAG(int32_t, gaborish, -1,
          "Force disable/enable the gaborish filter. "
          "(-1 = default, 0 = disable, 1 = enable).");

ABSL_FLAG(int32_t, group_order, -1,
          // TODO(tfish): This is a new flag. Check with team.
          "Order in which 256x256 regions are stored "
          "in the codestream for progressive rendering. "
          "Value -1 means 'encoder default', 0 means 'scanline order', "
          "1 means 'center-first order'.");

ABSL_FLAG(int32_t, epf, -1,
          "Edge preserving filter level, -1 to 3. "
          "Value -1 means: default (encoder chooses), 0 to 3 set a strength.");

ABSL_FLAG(int64_t, center_x, -1,
          // TODO(tfish): Clarify if this is really the comment we want here.
          "Determines the horizontal position of center for the center-first "
          "group order. The value -1 means 'use the middle of the image', "
          // TODO(tfish): Clarify if encode.h has an off-by-one in the
          // upper limit here.
          "other values 0..(xsize-1) set this to a particular coordinate.");

ABSL_FLAG(int64_t, center_y, -1,
          // TODO(tfish): Clarify if this is really the comment we want here.
          "Determines the vertical position of center for the center-first "
          "group order. The value -1 means 'use the middle of the image', "
          // TODO(tfish): Clarify if encode.h has an off-by-one in the
          // upper limit here.
          "other values 0..(ysize-1) set this to a particular coordinate.");

ABSL_FLAG(int64_t, num_threads, 0,
          // TODO(tfish): Sync with team about changed meaning of 0 -
          // was: No multithreaded workers. Is: use default number.
          "Number of worker threads (0 == use machine default).");

ABSL_FLAG(int64_t, num_reps, 1,  // TODO(tfish): wire this up.
          // TODO(tfish): Clarify meaning of this docstring.
          // Is this simply for benchmarking?
          "How many times to compress.");

ABSL_FLAG(int32_t, photon_noise, 0,
          // TODO(tfish): Discuss docstring change with team.
          // Also: This now is an int, no longer a float.
          "Adds noise to the image emulating photographic film noise. "
          "The higher the given number, the grainier the image will be. "
          "As an example, a value of 100 gives low noise whereas a value "
          "of 3200 gives a lot of noise. The default value is 0.");

ABSL_FLAG(float, distance, 1.0,  // TODO(tfish): wire this up.
          "Max. butteraugli distance, lower = higher quality. Range: 0 .. 25.\n"
          "    0.0 = mathematically lossless. Default for already-lossy input "
          "(JPEG/GIF).\n"
          "    1.0 = visually lossless. Default for other input.\n"
          "    Recommended range: 0.5 .. 3.0.");

ABSL_FLAG(int64_t, target_size, 0,  // TODO(tfish): wire this up.
          "Aim at file size of N bytes.\n"
          "    Compresses to 1 % of the target size in ideal conditions.\n"
          "    Runs the same algorithm as --target_bpp");

ABSL_FLAG(float, target_bpp, 0,  // TODO(tfish): wire this up.
          "Aim at file size that has N bits per pixel.\n"
          "    Compresses to 1 % of the target BPP in ideal conditions.");

ABSL_FLAG(float, quality, 100.0,  // TODO(tfish): wire this up.
          "Quality setting (is remapped to --distance). Range: -inf .. 100.\n"
          "    100 = mathematically lossless. Default for already-lossy input "
          "(JPEG/GIF).\n    Positive quality values roughly match libjpeg "
          "quality.");

ABSL_FLAG(int64_t, effort, 7,
          // TODO(tfish): Clarify discrepancy with team:
          // Documentation says default==squirrel(7) here:
          // https://libjxl.readthedocs.io/en/latest/api_encoder.html#_CPPv424JxlEncoderFrameSettingId
          // but enc_params.h has kFalcon=7.
          "Encoder effort setting. Range: 1 .. 9.\n"
          "    Default: 7. Higher number is more effort (slower).");


namespace {

// RAII-wraps the C-API encoder.
class ManagedJxlEncoder {
public:
  ManagedJxlEncoder(size_t num_worker_threads) :
    encoder_(JxlEncoderCreate(NULL)),
    encoder_frame_settings_(JxlEncoderFrameSettingsCreate(encoder_, NULL)) {
    if (num_worker_threads > 1) {
      parallel_runner_ = JxlThreadParallelRunnerCreate(
          /*memory_manager=*/nullptr, num_worker_threads);
    }

  }
  ~ManagedJxlEncoder() {
    if (parallel_runner_ != nullptr) {
      JxlThreadParallelRunnerDestroy(parallel_runner_);
    }
    JxlEncoderDestroy(encoder_);
    if (compressed_buffer_) {
      free(compressed_buffer_);
    }
  }

  JxlEncoder* encoder_;
  JxlEncoderFrameSettings *encoder_frame_settings_;
  uint8_t *compressed_buffer_ = nullptr;
  size_t compressed_buffer_size_ = 0;
  size_t compressed_buffer_used_ = 0;
  void* parallel_runner_ = nullptr;  // TODO(tfish): fix type.
};


bool ProcessTristateFlag(const char* flag_name, int32_t absl_flag_value,
                         JxlEncoderFrameSettings* frame_settings,
                         JxlEncoderFrameSettingId encoder_option) {
  if (! (absl_flag_value == -1 || absl_flag_value == 0 ||
         absl_flag_value == 1)) {
    std::cerr << "Invalid flag --" << flag_name <<
      ". Should be one of: -1, 0, 1.\n";
    return false;
  }
  if (absl_flag_value != -1) {
    JxlEncoderFrameSettingsSetOption(frame_settings, encoder_option,
        absl_flag_value);
  }
  return true;
}

}  // namespace


int main(int argc, char **argv) {
  absl::SetProgramUsageMessage(
      absl::StrCat("JPEG XL-encodes an image.  Sample usage:\n", argv[0],
                   " <source_image_filename> <target_image_filename>"));
  const std::vector<char*>& positional_args =
      absl::ParseCommandLine(argc, argv);

  // Handle --version.
  if (absl::GetFlag(FLAGS_version)) {
    uint32_t version = JxlEncoderVersion();
    std::cout << version / 1000000 << "." << (version / 1000) % 1000 <<
      "." << version % 1000 << std::endl;
    return EXIT_SUCCESS;
  }

  if (positional_args.size() != 3) {
    std::cerr << absl::ProgramUsageMessage() << std::endl;
    return EXIT_FAILURE;
  }
  const char* filename_in = positional_args[1];
  const char* filename_out = positional_args[2];

  size_t num_worker_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
  {
    int64_t flag_num_worker_threads = absl::GetFlag(FLAGS_num_threads);
    if (flag_num_worker_threads != 0) {
      num_worker_threads = flag_num_worker_threads;
    }
  }
  ManagedJxlEncoder managed_jxl_encoder = ManagedJxlEncoder(num_worker_threads);
  if (managed_jxl_encoder.parallel_runner_ != nullptr) {
    if (JXL_ENC_SUCCESS !=
        JxlEncoderSetParallelRunner(
            managed_jxl_encoder.encoder_,
            // TODO(tfish): Flag up the need to have the parameter below
            // documented better in the encode.h API docs.
            JxlThreadParallelRunner,
            managed_jxl_encoder.parallel_runner_)) {
      std::cerr << "JxlEncoderSetParallelRunner failed\n";
      return EXIT_FAILURE;
    }
  }

  JxlEncoder* jxl_encoder = managed_jxl_encoder.encoder_;
  JxlEncoderFrameSettings* jxl_encoder_frame_settings =
    managed_jxl_encoder.encoder_frame_settings_;

  {  // Processing tuning flags.
    bool use_container = absl::GetFlag(FLAGS_container);
    // TODO(tfish): Set use_container according to need of encoded data.
    // This will likely require moving this piece out of flags-processing.
    if (absl::GetFlag(FLAGS_strip)) {
      use_container = false;
    }
    JxlEncoderUseContainer(jxl_encoder, use_container);
    ProcessTristateFlag("modular", absl::GetFlag(FLAGS_modular),
                        jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_MODULAR);
    ProcessTristateFlag("keep_invisible", absl::GetFlag(FLAGS_keep_invisible),
                        jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE);
    ProcessTristateFlag("dots", absl::GetFlag(FLAGS_dots),
                        jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_DOTS);
    ProcessTristateFlag("patches", absl::GetFlag(FLAGS_patches),
                        jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_PATCHES);
    ProcessTristateFlag("gaborish", absl::GetFlag(FLAGS_gaborish),
                        jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_GABORISH);
    ProcessTristateFlag("group_order", absl::GetFlag(FLAGS_group_order),
                        jxl_encoder_frame_settings,
                        JXL_ENC_FRAME_SETTING_GROUP_ORDER);

    const int32_t flag_effort = absl::GetFlag(FLAGS_effort);
    if (! (1 <= flag_effort && flag_effort <= 9)) {
      // Strictly speaking, custom absl flags-parsing would integrate
      // more nicely with abseil flags, but the boilerplate cost of
      // handling invalid calls is substantially higher than
      // this lightweight approach here.
      std::cerr << "Invalid --effort. Valid range is {1, 2, ..., 9}.\n";
      return EXIT_FAILURE;
    }
    JxlEncoderFrameSettingsSetOption(
        jxl_encoder_frame_settings,
        JXL_ENC_FRAME_SETTING_EFFORT,
        flag_effort);

    const int32_t flag_epf = absl::GetFlag(FLAGS_epf);
    if (! (-1 <= flag_epf && flag_epf <= 3)) {
      std::cerr << "Invalid --epf. Valid range is {-1, 0, 1, 2, 3}.\n";
      return EXIT_FAILURE;
    }
    if (flag_epf != -1) {
      JxlEncoderFrameSettingsSetOption(
        jxl_encoder_frame_settings,
        JXL_ENC_FRAME_SETTING_EPF,
        flag_epf);
    }

    const int32_t flag_faster_decoding = absl::GetFlag(FLAGS_faster_decoding);
    if (! (0 <= flag_faster_decoding && flag_faster_decoding <= 4)) {
      std::cerr << "Invalid --faster_decoding. "
          "Valid range is {0, 1, 2, 3, 4}.\n";
      return EXIT_FAILURE;
    }
    JxlEncoderFrameSettingsSetOption(
        jxl_encoder_frame_settings,
        JXL_ENC_FRAME_SETTING_DECODING_SPEED,
        flag_faster_decoding);

    const int32_t flag_resampling = absl::GetFlag(FLAGS_resampling);
    if (flag_resampling != -1) {
      if (! (((flag_resampling & (flag_resampling - 1)) == 0) &&
             flag_resampling <= 8)) {
        std::cerr << "Invalid --resampling. "
            "Valid values are {-1, 1, 2, 4, 8}.\n";
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings,
          JXL_ENC_FRAME_SETTING_RESAMPLING,
          flag_resampling);
    }
    const int32_t flag_ec_resampling = absl::GetFlag(FLAGS_ec_resampling);
    if (flag_ec_resampling != -1) {
      if (! (((flag_ec_resampling & (flag_ec_resampling - 1)) == 0) &&
             flag_ec_resampling <= 8)) {
        std::cerr << "Invalid --ec_resampling. "
            "Valid values are {-1, 1, 2, 4, 8}.\n";
        return EXIT_FAILURE;
      }
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings,
          JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING,
          flag_ec_resampling);
    }

    JxlEncoderFrameSettingsSetOption(
        jxl_encoder_frame_settings,
        JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED,
        absl::GetFlag(FLAGS_already_downsampled));

    JxlEncoderFrameSettingsSetOption(
        jxl_encoder_frame_settings,
        JXL_ENC_FRAME_SETTING_PHOTON_NOISE,
        absl::GetFlag(FLAGS_photon_noise));
    // Removed: --noise (superseded by: --photon_noise).

    JxlEncoderSetFrameDistance(
        jxl_encoder_frame_settings,
        absl::GetFlag(FLAGS_distance));

    const int32_t flag_center_x = absl::GetFlag(FLAGS_center_x);
    if (flag_center_x != -1) {
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings,
          JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X,
          flag_center_x);
    }
    const int32_t flag_center_y = absl::GetFlag(FLAGS_center_y);
    if (flag_center_y != -1) {
      JxlEncoderFrameSettingsSetOption(
          jxl_encoder_frame_settings,
          JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y,
          flag_center_y);
    }
  }  // Processing flags.

  jxl::PaddedBytes jpeg_data;
  JXL_RETURN_IF_ERROR(ReadFile(filename_in, &jpeg_data));

  if (JXL_ENC_SUCCESS !=
      JxlEncoderAddJPEGFrame(jxl_encoder_frame_settings,
                             jpeg_data.data(), jpeg_data.size())) {
    std::cerr << "JxlEncoderAddJPEGFrame() failed.\n";
    return EXIT_FAILURE;
  }

  if (!fetch_jxl_encoded_image(jxl_encoder,
                               &managed_jxl_encoder.compressed_buffer_,
                               &managed_jxl_encoder.compressed_buffer_size_,
                               &managed_jxl_encoder.compressed_buffer_used_)) {
    std::cerr << "Fetching encoded image failed.\n";
    return EXIT_FAILURE;
  }

  if(!write_jxl_file(managed_jxl_encoder.compressed_buffer_,
                     managed_jxl_encoder.compressed_buffer_used_,
                     filename_out)) {
    std::cerr << "Writing output file failed: " << filename_out << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
