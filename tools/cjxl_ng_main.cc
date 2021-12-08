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

#include "fetch_encoded.h"


ABSL_FLAG(bool, container, false,
          "Always encode using container format");

ABSL_FLAG(bool, strip, false,
          "Do not encode using container format (strips "
          "Exif/XMP/JPEG bitstream reconstruction data)");

ABSL_FLAG(bool, progressive, false,
          "Enable progressive/responsive decoding.");

ABSL_FLAG(bool, progressive_ac, false,
          "Use progressive mode for AC.");

ABSL_FLAG(bool, qprogressive_ac, false,
          // TODO(tfish): Clarify what this flag is about.
          "Use progressive mode for AC.");

ABSL_FLAG(bool, progressive_dc, false,
          "Use progressive mode for DC.");

ABSL_FLAG(bool, modular, false,
          "Use modular mode (lossy / lossless).");

ABSL_FLAG(bool, use_experimental_encoder_heuristics, false,
          "Use new and not yet ready encoder heuristics");

ABSL_FLAG(bool, jpeg_transcode, false,
          "Do lossy transcode of input JPEG file (decode to "
          "pixels instead of doing lossless transcode).");

ABSL_FLAG(bool, jpeg_transcode_disable_cfl, false,
          "Disable CFL for lossless JPEG recompression");

ABSL_FLAG(bool, premultiply, false,
          "Force premultiplied (associated) alpha.");

ABSL_FLAG(bool, keep_invisible, false,
          "force disable/enable preserving color of invisible "
          "pixels..");

ABSL_FLAG(bool, centerfirst, false,
          "Put center groups first in the compressed file.");

ABSL_FLAG(bool, noise, false,
          "force disable/enable noise generation.");

ABSL_FLAG(bool, verbose, false,
          // TODO(tfish): Should be a verbosity-level.
          // Original cjxl also makes --help more verbose if this is on,
          // but with absl, we do that differently...?
          "Verbose output.");


ABSL_FLAG(std::string, photon_noise, "ISO3200",
          "Set the noise to approximately what it would be at a given nominal "
          "exposure on a 35mm camera. For formats other than 35mm, or when the "
          "whole sensor was not used, you can multiply the ISO value by the "
          "equivalence ratio squared, for example by 2.25 for an APS-C "
          "camera.");

// TODO(tfish): --dots, --patches, --resampling, --ec_resampling,
// --already_downsampled, --epf, --gaborish, --intensity_target,
// --saliency_num_progressive_steps, --saliency_map_filename,
// --saliency_threshold, --dec-hints, --override_bitdepth,
// --colortransform, --mquality, --iterations, --colorspace, --group-size,
// --predictor, --extra-properties, --lossy-palette, --pre-compact,
// --post-compact, --responsive, --version, --quiet, --print_profile,

ABSL_FLAG(int64_t, center_x, 0,
          // TODO(tfish): Clarify if this is really the comment we want here.
          "Put center groups first in the compressed file.");

ABSL_FLAG(int64_t, center_y, 0,
          // TODO(tfish): Clarify if this is really the comment we want here.
          "Put center groups first in the compressed file.");

ABSL_FLAG(int64_t, num_threads, 0,
          "number of worker threads (zero = none).");

ABSL_FLAG(int64_t, num_reps, 1,
          // TODO(tfish): Clarify meaning of this docstring.
          // Is this simply for benchmarking?
          "how many times to compress.");



ABSL_FLAG(float, distance, 1.0,
          "Max. butteraugli distance, lower = higher quality. Range: 0 .. 25.\n"
          "    0.0 = mathematically lossless. Default for already-lossy input "
          "(JPEG/GIF).\n"
          "    1.0 = visually lossless. Default for other input.\n"
          "    Recommended range: 0.5 .. 3.0.");

ABSL_FLAG(int64_t, target_size, 0,
          "Aim at file size of N bytes.\n"
          "    Compresses to 1 % of the target size in ideal conditions.\n"
          "    Runs the same algorithm as --target_bpp");

ABSL_FLAG(float, target_bpp, 0,
          "Aim at file size that has N bits per pixel.\n"
          "    Compresses to 1 % of the target BPP in ideal conditions.");

ABSL_FLAG(float, quality, 100.0,
          "Quality setting (is remapped to --distance). Range: -inf .. 100.\n"
          "    100 = mathematically lossless. Default for already-lossy input "
          "(JPEG/GIF).\n    Positive quality values roughly match libjpeg "
          "quality.");

ABSL_FLAG(int64_t, effort, 7,
          "Encoder effort setting. Range: 1 .. 9.\n"
          "    Default: 7. Higher number is more effort (slower).");

// TODO(tfish): --speed, --faster_decoding



namespace {

// RAII-wraps the C-API encoder.
class ManagedJxlEncoder {
public:    
  ManagedJxlEncoder() :
    encoder_(JxlEncoderCreate(NULL)),
    encoder_options_(JxlEncoderOptionsCreate(encoder_, NULL)) {}
  ~ManagedJxlEncoder() {
    JxlEncoderDestroy(encoder_);
    encoder_ = nullptr;
    encoder_options_ = nullptr;
    if (compressed_buffer_) {
      free(compressed_buffer_);
    }
  }
  
  JxlEncoder* encoder_;
  JxlEncoderOptions *encoder_options_;
  uint8_t *compressed_buffer_ = nullptr;
  size_t compressed_buffer_size_ = 0;
  size_t compressed_buffer_used_ = 0;
};
  
}  // namespace


int main(int argc, char **argv) {
  absl::SetProgramUsageMessage(
      absl::StrCat("JPEG XL-encodes an image.  Sample usage:\n", argv[0],
                   " <source_image_filename> <target_image_filename>"));
  
  const std::vector<char*>& positional_args =
      absl::ParseCommandLine(argc, argv);

  if (positional_args.size() != 3) {
    std::cerr << absl::ProgramUsageMessage() << std::endl;
    return EXIT_FAILURE;
  }
  const char* filename_in = positional_args[1];
  const char* filename_out = positional_args[2];
  
  ManagedJxlEncoder managed_jxl_encoder = ManagedJxlEncoder();
  JxlEncoder* jxl_encoder = managed_jxl_encoder.encoder_;
  JxlEncoderOptions *jxl_encoder_options = managed_jxl_encoder.encoder_options_;


  jxl::PaddedBytes jpeg_data;
  JXL_RETURN_IF_ERROR(ReadFile(filename_in, &jpeg_data));

  if (absl::GetFlag(FLAGS_container)) {
    std::cout << "TODO(tfish): container=true.\n";
  }
  
  if (JXL_ENC_SUCCESS !=
      JxlEncoderAddJPEGFrame(jxl_encoder_options,
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
