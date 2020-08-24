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

#include "jpegxl/decode.h"
#include "jxl/aux_out.h"
#include "jxl/base/cache_aligned.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/file_io.h"
#include "jxl/base/os_specific.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/brunsli.h"
#include "jxl/codec_in_out.h"
#include "tools/box/box.h"
#include "tools/cmdline.h"
#include "tools/codec_config.h"
#include "tools/djpegxl.h"
#include "tools/speed_stats.h"

namespace jpegxl {
namespace tools {
namespace {

int DecompressMain(int argc, const char *argv[]) {
  DecompressArgs args;
  CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, argv) || !args.ValidateArgs(cmdline)) {
    cmdline.PrintHelp();
    return 1;
  }
  if (args.version) {
    fprintf(stderr, "djpegxl [%s]\n", CodecConfigString().c_str());
    fprintf(stderr, "Copyright (c) the JPEG XL Project\n");
    return 0;
  }

  jxl::PaddedBytes compressed;
  if (!jxl::ReadFile(args.file_in, &compressed)) return 1;
  fprintf(stderr, "Read %zu compressed bytes [%s]\n", compressed.size(),
          CodecConfigString().c_str());

  // Detect whether the file uses the box format container. If so, extract the
  // primary codestream, and continue with only the codestream.
  const uint8_t box_header[] = {0,   0,   0,   0xc, 'J',  'X',
                                'L', ' ', 0xd, 0xa, 0x87, 0xa};
  if (compressed.size() >= 12 && !memcmp(box_header, compressed.data(), 12)) {
    JpegXlContainer container;
    if (!DecodeJpegXlContainerOneShot(compressed.data(), compressed.size(),
                                      &container)) {
      fprintf(stderr, "Decoding container format failed.\n");
      return 1;
    }
    compressed.assign(container.codestream,
                      container.codestream + container.codestream_size);
  }

  jxl::ThreadPoolInternal pool(args.num_threads);
  SpeedStats stats;

  // Quick test that this looks like a valid JXL file.
  if (JpegxlSignatureCheck(compressed.data(), compressed.size()) !=
      JPEGXL_SIG_VALID) {
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
      if (!DecompressJxlToJPEG(jxl::Span<const uint8_t>(compressed), args,
                               &pool, &jpg_output, &aux_out, &stats)) {
        return 1;
      }
    }

    if (args.file_out != nullptr) {
      if (!jxl::WriteFile(jpg_output, args.file_out)) return 1;
    }
  } else {
    jxl::CodecInOut io;
    // Set JPEG quality.
    // TODO(veluca): the decoder should set this value, and the argument should
    // be an override.
    // TODO(veluca): the decoder should directly produce a JPEG file, and this
    // should not be necessary.
    io.use_sjpeg = args.use_sjpeg;
    io.jpeg_quality = args.jpeg_quality;

    // Decode to pixels.
    for (size_t i = 0; i < args.num_reps; ++i) {
      if (!DecompressJxlToPixels(jxl::Span<const uint8_t>(compressed),
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
