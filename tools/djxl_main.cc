// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "jxl/decode.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/cache_aligned.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/os_specific.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
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

int DecompressMain(int argc, const char *argv[]) {
  DecompressArgs args;
  CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, argv)) {
    // ValidateArgs already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return 1;
  }

  if (args.version) {
    fprintf(stdout, "djxl [%s]\n",
            CodecConfigString(JxlDecoderVersion()).c_str());
    fprintf(stdout, "Copyright (c) the JPEG XL Project\n");
    return 0;
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
  if (!jxl::ReadFile(args.file_in, &compressed)) return 1;
  fprintf(stderr, "Read %zu compressed bytes [%s]\n", compressed.size(),
          CodecConfigString(JxlDecoderVersion()).c_str());

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
    container.codestream = compressed.data();
    container.codestream_size = compressed.size();
  }

  jxl::ThreadPoolInternal pool(args.num_threads);
  SpeedStats stats;

  // Quick test that this looks like a valid JXL file.
  JxlSignature signature =
      JxlSignatureCheck(container.codestream, container.codestream_size);
  if (signature == JXL_SIG_NOT_ENOUGH_BYTES || signature == JXL_SIG_INVALID) {
    fprintf(stderr, "Unknown compressed image format\n");
    return 1;
  }

  const std::vector<int> cpus = jxl::AvailableCPUs();
  pool.RunOnEachThread([&cpus](const int task, const size_t thread) {
    // 1.1-1.2x speedup (36 cores) from pinning.
    if (thread < cpus.size()) {
      if (!jxl::PinThreadToCPU(cpus[thread])) {
        fprintf(stderr, "WARNING: failed to pin thread %zu.\n", thread);
      }
    }
  });

  jxl::AuxOut aux_out;

  if (args.decode_to_jpeg) {
    // --jpeg flag passed, decode to JPEG.
    args.params.keep_dct = true;

    jxl::PaddedBytes jpg_output;
    for (size_t i = 0; i < args.num_reps; ++i) {
      if (!DecompressJxlToJPEG(container, args, &pool, &jpg_output, &aux_out,
                               &stats)) {
        return 1;
      }
    }

    if (args.file_out != nullptr) {
      if (!jxl::WriteFile(jpg_output, args.file_out)) {
        fprintf(stderr, "Failed to write to \"%s\"\n", args.file_out);
        return 1;
      }
    }
  } else {
    jxl::CodecInOut io;
    auto assign = [](const uint8_t* bytes, size_t size,
        jxl::PaddedBytes& target) {
      target.assign(bytes, bytes + size);
    };
    if (container.exif_size) {
      assign(container.exif, container.exif_size, io.blobs.exif);
    }
    for (const auto& span : container.xml) {
      std::string xml(span.first, span.first + span.second);
      bool is_xmp = strstr(xml.c_str(), "XML:com.adobe.xmp");
      assign(span.first, span.second, is_xmp ? io.blobs.xmp : io.blobs.iptc);
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
      if (!DecompressJxlToPixels(
              jxl::Span<const uint8_t>(container.codestream,
                                       container.codestream_size),
              args.params, &pool, &io, &aux_out, &stats)) {
        return 1;
      }
    }

    if (!WriteJxlOutput(args, args.file_out, io)) return 1;

    if (args.print_read_bytes) {
      fprintf(stderr, "Decoded bytes: %zu\n", io.Main().decoded_bytes());
    }
  }

  if (args.print_info == jxl::Override::kOn) {
    aux_out.Print(args.num_reps);
  }

  JXL_CHECK(stats.Print(pool.NumWorkerThreads()));

  if (args.print_profile == jxl::Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  jxl::CacheAligned::PrintStats();
  return 0;
}

}  // namespace
}  // namespace tools
}  // namespace jpegxl

int main(int argc, const char *argv[]) {
  return jpegxl::tools::DecompressMain(argc, argv);
}
