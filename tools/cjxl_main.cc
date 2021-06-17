// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>

#include "jxl/encode.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/jpeg/enc_jpeg_data.h"
#include "tools/box/box.h"
#include "tools/cjxl.h"
#include "tools/codec_config.h"

namespace jpegxl {
namespace tools {

int CompressJpegXlMain(int argc, const char* argv[]) {
  CommandLineParser cmdline;
  CompressArgs args;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, argv)) {
    // Parse already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return 1;
  }

  if (args.version) {
    fprintf(stdout, "cjxl %s\n",
            CodecConfigString(JxlEncoderVersion()).c_str());
    fprintf(stdout, "Copyright (c) the JPEG XL Project\n");
    return 0;
  }

  if (!args.quiet) {
    fprintf(stderr, "JPEG XL encoder %s\n",
            CodecConfigString(JxlEncoderVersion()).c_str());
  }

  if (cmdline.HelpFlagPassed()) {
    cmdline.PrintHelp();
    return 0;
  }

  if (!args.ValidateArgs(cmdline)) {
    // ValidateArgs already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return 1;
  }

  jxl::PaddedBytes compressed;

  jxl::ThreadPoolInternal pool(args.num_threads);
  jxl::CodecInOut io;
  double decode_mps = 0;
  JXL_RETURN_IF_ERROR(LoadAll(args, &pool, &io, &decode_mps));

  // need to validate again because now we know the input
  if (!args.ValidateArgsAfterLoad(cmdline, io)) {
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return 1;
  }
  if (!args.file_out && !args.quiet) {
    fprintf(stderr,
            "No output file specified.\n"
            "Encoding will be performed, but the result will be discarded.\n");
  }
  if (!CompressJxl(io, decode_mps, &pool, args, &compressed, !args.quiet)) {
    return 1;
  }

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
      JXL_RETURN_IF_ERROR(EncodeJPEGData(data_in, &jpeg_data));
      container.jpeg_reconstruction = jpeg_data.data();
      container.jpeg_reconstruction_size = jpeg_data.size();
    }
    jxl::PaddedBytes container_file;
    if (!EncodeJpegXlContainerOneShot(container, &container_file)) {
      fprintf(stderr, "Failed to encode container format\n");
      return 1;
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
      return 1;
    }
  }

  if (args.print_profile == jxl::Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  if (!args.quiet && cmdline.verbosity > 0) {
    jxl::CacheAligned::PrintStats();
  }
  return 0;
}

}  // namespace tools
}  // namespace jpegxl

int main(int argc, const char** argv) {
  return jpegxl::tools::CompressJpegXlMain(argc, argv);
}
