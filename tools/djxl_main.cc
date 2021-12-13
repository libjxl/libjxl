// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "jxl/decode.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/cache_aligned.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "tools/box/box.h"
#include "tools/cmdline.h"
#include "tools/codec_config.h"
#include "tools/djxl.h"
#include "tools/speed_stats.h"

namespace jpegxl {
namespace tools {
namespace {

int DecompressMain(int argc, const char* argv[]) {
  DecompressArgs args;
  CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, argv)) {
    // ValidateArgs already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return 1;
  }

  if (args.version) {
    fprintf(stdout, "djxl %s\n",
            CodecConfigString(JxlDecoderVersion()).c_str());
    fprintf(stdout, "Copyright (c) the JPEG XL Project\n");
    return 0;
  }
  if (!args.quiet) {
    fprintf(stderr, "JPEG XL decoder %s\n",
            CodecConfigString(JxlDecoderVersion()).c_str());
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
  if (!jxl::ReadFile(args.file_in, &compressed)) {
    fprintf(stderr, "Failed to read file: %s.\n", args.file_in);
    return 1;
  }
  if (!args.quiet) {
    fprintf(stderr, "Read %" PRIuS " compressed bytes.\n", compressed.size());
  }

  // If the file uses the box format container, unpack the boxes into
  // `container`. Otherwise, fill `container.codestream` accordingly.
  JpegXlContainer container;
  if (IsContainerHeader(compressed.data(), compressed.size())) {
    if (!DecodeJpegXlContainerOneShot(compressed.data(), compressed.size(),
                                      &container)) {
      fprintf(stderr, "Decoding container format failed.\n");
      return 1;
    }
  } else {
    container.codestream = std::move(compressed);
  }

  jxl::ThreadPoolInternal pool(args.num_threads);
  SpeedStats stats;

  // Quick test that this looks like a valid JXL file.
  JxlSignature signature = JxlSignatureCheck(container.codestream.data(),
                                             container.codestream.size());
  if (signature == JXL_SIG_NOT_ENOUGH_BYTES || signature == JXL_SIG_INVALID) {
    fprintf(stderr, "Unknown compressed image format (%u)\n", signature);
    return 1;
  }

  if (!args.file_out && !args.quiet) {
    fprintf(stderr,
            "No output file specified.\n"
            "Decoding will be performed, but the result will be discarded.\n");
  }

  jxl::AuxOut aux_out;

  if (!args.decode_to_pixels) {
    args.params.keep_dct = true;

    jxl::PaddedBytes jpg_output;
    bool success = true;
    for (size_t i = 0; i < args.num_reps; ++i) {
      success = success && DecompressJxlToJPEG(container, args, &pool,
                                               &jpg_output, &stats);
    }
    if (!args.quiet && success) fprintf(stderr, "Reconstructed to JPEG.\n");

    if (success && args.file_out != nullptr) {
      if (!jxl::WriteFile(jpg_output, args.file_out)) {
        fprintf(stderr, "Failed to write to \"%s\"\n", args.file_out);
        return 1;
      }
    }
    if (!success) {
      if (!args.quiet) {
        fprintf(stderr,
                "Warning: could not decode losslessly to JPEG. Retrying with "
                "--pixels_to_jpeg...\n");
      }
      args.decode_to_pixels = true;
    }
  }
  if (args.decode_to_pixels) {
    args.params.keep_dct = false;
    jxl::CodecInOut io;
    auto assign = [](const uint8_t* bytes, size_t size,
                     jxl::PaddedBytes& target) {
      target.assign(bytes, bytes + size);
    };
    if (container.exif_size) {
      assign(container.exif, container.exif_size, io.blobs.exif);
    }
    if (!container.xml.empty()) {
      assign(container.xml[0].first, container.xml[0].second, io.blobs.xmp);
    }
    if (container.xml.size() > 1) {
      fprintf(stderr,
              "Warning: more than one XML box found, assuming first one is XMP "
              "and ignoring others\n");
    }
    // Set JPEG quality.
    // TODO(veluca): the decoder should set this value, and the argument should
    // be an override.
    // TODO(veluca): the decoder should directly produce a JPEG file, and this
    // should not be necessary.
    io.use_sjpeg = args.use_sjpeg;
    io.jpeg_quality = args.jpeg_quality;

    // Decode to pixels.
    for (size_t i = 0; i < args.num_reps; ++i) {
      if (!DecompressJxlToPixels(jxl::Span<const uint8_t>(container.codestream),
                                 args.params, &pool, &io, &stats)) {
        // Error is already reported by DecompressJxlToPixels.
        return 1;
      }
    }
    if (!args.quiet) fprintf(stderr, "Decoded to pixels.\n");
    if (!WriteJxlOutput(args, args.file_out, io, &pool)) {
      // Error is already reported by WriteJxlOutput.
      return 1;
    }

    if (args.print_read_bytes) {
      fprintf(stderr, "Decoded bytes: %" PRIuS "\n", io.Main().decoded_bytes());
    }
  }

  if (!args.quiet) JXL_CHECK(stats.Print(pool.NumWorkerThreads()));

  if (args.print_profile == jxl::Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  if (!args.quiet) jxl::CacheAligned::PrintStats();
  return 0;
}

}  // namespace
}  // namespace tools
}  // namespace jpegxl

int main(int argc, const char* argv[]) {
  return jpegxl::tools::DecompressMain(argc, argv);
}
