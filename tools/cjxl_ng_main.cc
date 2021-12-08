// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <iostream>
#include <vector>

// XXX Check which ones are needed.
#include <stdint.h>
#include <stdlib.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

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

ABSL_FLAG(float, distance, 1.0,
          "Max. butteraugli distance, lower = higher quality. Range: 0 .. 25.\n"
          "    0.0 = mathematically lossless. Default for already-lossy input "
          "(JPEG/GIF).\n"
          "    1.0 = visually lossless. Default for other input.\n"
          "    Recommended range: 0.5 .. 3.0.");

ABSL_FLAG(int64_t, size, 0,
          "Aim at file size of N bytes.\n"
          "    Compresses to 1 % of the target size in ideal conditions.\n"
          "    Runs the same algorithm as --target_bpp");

ABSL_FLAG(int64_t, target_bpp, 0,
          "Aim at file size that has N bits per pixel.\n"
          "    Compresses to 1 % of the target BPP in ideal conditions.");


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
  }
  
  JxlEncoder* encoder_ = nullptr;
  JxlEncoderOptions *encoder_options_ = nullptr;
};
  
}  // namespace


int main(int argc, char **argv) {
  uint8_t *compressed_buffer = NULL;
  size_t compressed_buffer_size = 0, compressed_buffer_used = 0;

  absl::SetProgramUsageMessage(
      absl::StrCat("JPEG XL-encodes an image.  Sample usage:\n", argv[0],
                   " <source_image_filename> <target_image_filename>"));
  
  const std::vector<char*>& positional_args = absl::ParseCommandLine(
      argc, argv);

  int success = EXIT_FAILURE;
  if (positional_args.size() != 3) {
    std::cerr << absl::ProgramUsageMessage() << std::endl;
    return EXIT_FAILURE;
  }
  const char* filename_in = positional_args[0];
  const char* filename_out = positional_args[1];
  
  ManagedJxlEncoder managed_jxl_encoder = ManagedJxlEncoder();
  JxlEncoder* jxl_encoder = managed_jxl_encoder.encoder_;
  JxlEncoderOptions *jxl_encoder_options = managed_jxl_encoder.encoder_options_;


  jxl::PaddedBytes jpeg_data;
  JXL_RETURN_IF_ERROR(ReadFile(filename_in, &jpeg_data));

  if (JXL_ENC_SUCCESS !=
      JxlEncoderAddJPEGFrame(jxl_encoder_options,
                             jpeg_data.data(), jpeg_data.size())) {
    goto cleanup;
  }

  if (!fetch_jxl_encoded_image(jxl_encoder,
                               &compressed_buffer,
                               &compressed_buffer_size,
                               &compressed_buffer_used)) {
    goto cleanup;
  }
  
  if(!write_jxl_file(compressed_buffer,
                     compressed_buffer_used,
                     filename_out)) {
    std::cerr << "Writing output file failed: " << filename_out << std::endl;
    success = EXIT_FAILURE;
  }
  
 cleanup:
  if (compressed_buffer) {
    free(compressed_buffer);
    compressed_buffer = NULL;
  }
  return success;
}
