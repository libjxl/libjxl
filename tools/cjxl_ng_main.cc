// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstdio>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

// #include "jxl/encode.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/padded_bytes.h"

// #include "lib/jxl/base/profiler.h"
// #include "lib/jxl/jpeg/enc_jpeg_data.h"
// #include "tools/box/box.h"
// #include "tools/cjxl.h"
// #include "tools/codec_config.h"

/* ======

namespace jpegxl {
namespace tools {

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

int CompressJpegXlMain(int argc, const char* argv[]) {
  CommandLineParser cmdline;
  CompressArgs args;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, argv)) {
    // Parse already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return CjxlRetCode::ERR_PARSE;
  }

  if (args.version) {
    fprintf(stdout, "cjxl %s\n",
            CodecConfigString(JxlEncoderVersion()).c_str());
    fprintf(stdout, "Copyright (c) the JPEG XL Project\n");
    return CjxlRetCode::OK;
  }

  if (!args.quiet) {
    fprintf(stderr, "JPEG XL encoder %s\n",
            CodecConfigString(JxlEncoderVersion()).c_str());
  }

  if (cmdline.HelpFlagPassed()) {
    cmdline.PrintHelp();
    return CjxlRetCode::OK;
  }

  if (!args.ValidateArgs(cmdline)) {
    // ValidateArgs already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return CjxlRetCode::ERR_INVALID_ARG;
  }

  jxl::PaddedBytes compressed;

  jxl::ThreadPoolInternal pool(args.num_threads);
  jxl::CodecInOut io;
  double decode_mps = 0;
  if (!LoadAll(args, &pool, &io, &decode_mps)) {
    return CjxlRetCode::ERR_LOAD_INPUT;
  }

  // need to validate again because now we know the input
  if (!args.ValidateArgsAfterLoad(cmdline, io)) {
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return CjxlRetCode::ERR_INVALID_INPUT;
  }
  if (!args.file_out && !args.quiet) {
    fprintf(stderr,
            "No output file specified.\n"
            "Encoding will be performed, but the result will be discarded.\n");
  }
  if (!CompressJxl(io, decode_mps, &pool, args, &compressed, !args.quiet)) {
    return CjxlRetCode::ERR_ENCODING;
  }

  int ret = CjxlRetCode::OK;
  if (args.use_container) {
    JpegXlContainer container;
    container.codestream = compressed.data();
    container.codestream_size = compressed.size();
    if (!io.blobs.exif.empty()) {
      container.exif = io.blobs.exif.data();
      container.exif_size = io.blobs.exif.size();
    }
    auto append_xml = [&container](const jxl::PaddedBytes& bytes) {
      if (bytes.empty()) return;
      container.xml.emplace_back(bytes.data(), bytes.size());
    };
    append_xml(io.blobs.xmp);
    if (!io.blobs.jumbf.empty()) {
      container.jumb = io.blobs.jumbf.data();
      container.jumb_size = io.blobs.jumbf.size();
    }
    jxl::PaddedBytes jpeg_data;
    if (io.Main().IsJPEG()) {
      jxl::jpeg::JPEGData data_in = *io.Main().jpeg_data;
      if (EncodeJPEGData(data_in, &jpeg_data)) {
        container.jpeg_reconstruction = jpeg_data.data();
        container.jpeg_reconstruction_size = jpeg_data.size();
      } else {
        fprintf(stderr, "Warning: failed to create JPEG reconstruction data\n");
        ret = CjxlRetCode::DROPPED_JBRD;
      }
    }
    jxl::PaddedBytes container_file;
    if (!EncodeJpegXlContainerOneShot(container, &container_file)) {
      fprintf(stderr, "Failed to encode container format\n");
      return CjxlRetCode::ERR_CONTAINER;
    }
    compressed.swap(container_file);
    if (!args.quiet) {
      const size_t pixels = io.xsize() * io.ysize();
      const double bpp =
          static_cast<double>(compressed.size() * jxl::kBitsPerByte) / pixels;
      fprintf(stderr, "Including container: %llu bytes (%.3f bpp%s).\n",
              static_cast<long long unsigned>(compressed.size()),
              bpp / io.frames.size(), io.frames.size() == 1 ? "" : "/frame");
    }
  }
  if (args.file_out) {
    if (!jxl::WriteFile(compressed, args.file_out)) {
      fprintf(stderr, "Failed to write to \"%s\"\n", args.file_out);
      return CjxlRetCode::ERR_WRITE;
    }
  }

  if (args.print_profile == jxl::Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  if (!args.quiet && cmdline.verbosity > 0) {
    jxl::CacheAligned::PrintStats();
  }
  return ret;
}

}  // namespace tools
}  // namespace jpegxl

int main0(int argc, const char** argv) {
  return jpegxl::tools::CompressJpegXlMain(argc, argv);
}
====== */

//////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "jxl/codestream_header.h"
#include "jxl/color_encoding.h"
#include "jxl/encode.h"
#include "jxl/types.h"

#include "fetch_encoded.h"



#define DDD_DUMMY_WIDTH 640
#define DDD_DUMMY_HEIGHT 480


ABSL_FLAG(bool, dummy_testonly, false,
          "Dummy-test-encode-only");

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

// TODO(tfish): cjxl.cc - onwards from "quality".



/* This corresponds to: lib/jxl/encode.cc:JxlEncoderInitBasicInfo
   in the C++ API.
 */

static void JxlBasicInfoInit(JxlBasicInfo* info) {
  info->have_container = JXL_FALSE;
  info->xsize = 0;
  info->ysize = 0;
  info->bits_per_sample = 8;
  info->exponent_bits_per_sample = 0;
  info->intensity_target = 255.f;
  info->min_nits = 0.f;
  info->relative_to_max_display = JXL_FALSE;
  info->linear_below = 0.f;
  info->uses_original_profile = JXL_FALSE;
  info->have_preview = JXL_FALSE;
  info->have_animation = JXL_FALSE;
  info->orientation = JXL_ORIENT_IDENTITY;
  info->num_color_channels = 3;
  info->num_extra_channels = 0;
  info->alpha_bits = 0;
  info->alpha_exponent_bits = 0;
  info->alpha_premultiplied = JXL_FALSE;
  info->preview.xsize = 0;
  info->preview.ysize = 0;
  info->animation.tps_numerator = 10;
  info->animation.tps_denominator = 1;
  info->animation.num_loops = 0;
  info->animation.have_timecodes = JXL_FALSE;
}



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
  /* TODO(tfish): Replace allocating dummy-data here with actual image data. */
  float* dummy_src = NULL;
  uint8_t *compressed = NULL;
  size_t compressed_size;
  
  const std::vector<char*>& rest_args = absl::ParseCommandLine(argc, argv);

  int success = EXIT_SUCCESS;
  JxlPixelFormat pixel_format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  /* This is owned by the encoder. */

  if (rest_args.size() != 3) {
    fprintf(stderr,
            "Usage: %s {input_jpeg_filename} {output_jxl_filename}\n",
            rest_args[0]);
    return EXIT_FAILURE;
  }
  
  fprintf(stderr, "Creating encoder.\n");
  ManagedJxlEncoder managed_jxl_encoder = ManagedJxlEncoder();
  JxlEncoder* jxl_encoder = managed_jxl_encoder.encoder_;
  JxlEncoderOptions *jxl_encoder_options = managed_jxl_encoder.encoder_options_;
  fprintf(stderr, "Encoder is at %p.\n", (void *)jxl_encoder);


  jxl::PaddedBytes jpeg_data;
  JXL_RETURN_IF_ERROR(ReadFile(rest_args[1], &jpeg_data));
  fprintf(stderr, "DDD loaded jpeg, size=%zu\n", jpeg_data.size());

  if (absl::GetFlag(FLAGS_dummy_testonly)) {  
    JxlBasicInfo jxl_basic_info;
    JxlBasicInfoInit(&jxl_basic_info);
    jxl_basic_info.xsize = DDD_DUMMY_WIDTH;
    jxl_basic_info.ysize = DDD_DUMMY_HEIGHT;
    jxl_basic_info.bits_per_sample = 32;
    jxl_basic_info.exponent_bits_per_sample = 8;
    jxl_basic_info.uses_original_profile = JXL_FALSE;
    if (JXL_ENC_SUCCESS != JxlEncoderSetBasicInfo(jxl_encoder, &jxl_basic_info)) {
      fprintf(stderr, "JxlEncoderSetBasicInfo failed\n");
      goto cleanup;
    }
    fprintf(stderr, "JxlEncoderSetBasicInfo done.\n");
    
    JxlColorEncoding jxl_color_encoding;
    JxlColorEncodingSetToSRGB(&jxl_color_encoding,/*is_gray=*/JXL_FALSE);
    if (JXL_ENC_SUCCESS != JxlEncoderSetColorEncoding(jxl_encoder,
                                                      &jxl_color_encoding)) {
      fprintf(stderr, "JxlEncoderSetColorEncoding failed\n");
      goto cleanup;
    }
    fprintf(stderr, "JxlEncoderSetColorEncoding done.\n");
    
    if (NULL == (dummy_src = static_cast<float*>(aligned_alloc(sizeof(float),
                                                               DDD_DUMMY_WIDTH * DDD_DUMMY_HEIGHT * 3 * sizeof(float))))) {
      goto cleanup;
    }
    fprintf(stderr, "Allocated dummy data at %p.\n", (void *)dummy_src);
  
    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddImageFrame(jxl_encoder_options,
                                &pixel_format, (void*)dummy_src,
                                sizeof(float) * DDD_DUMMY_WIDTH * DDD_DUMMY_HEIGHT * 3)) {
      fprintf(stderr, "JxlEncoderAddImageFrame failed\n");
      goto cleanup;
    }
  }
  else {
    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddJPEGFrame(jxl_encoder_options,
                               jpeg_data.data(), jpeg_data.size())) {
      fprintf(stderr, "JxlEncoderAddJPEGFrame failed\n");
      goto cleanup;
    }
    fprintf(stderr, "JxlEncoderAddJPEGFrame done.\n");    
  }

  if (!fetch_jxl_encoded_image(jxl_encoder, &compressed, &compressed_size)) {
    goto cleanup;
  }

  if(!write_jxl_file(compressed, compressed_size, rest_args[2])) {
    fprintf(stderr, "Writing output file failed: %s\n", rest_args[2]);
    success = EXIT_FAILURE;
  }
  
 cleanup:
  if (dummy_src) {
    free(dummy_src);
    dummy_src = NULL;
  }
  if (compressed) {
    free(compressed);
    compressed = NULL;
  }
  return success;
}
