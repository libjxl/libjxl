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

#include "tools/cjpegxl.h"

#include <stdio.h>

#include <algorithm>
#include <hwy/runtime_dispatch.h>
#include <hwy/static_targets.h>

#include "jxl/base/cache_aligned.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/file_io.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/enc_params.h"
#include "tools/args.h"

namespace jpegxl {
namespace tools {

void CompressArgs::AddCommandLineOptions(CommandLineParser* cmdline,
                                         CompressionMode mode) {
  // Positional arguments.
  cmdline->AddPositionalOption("INPUT", "the input can be PNG PPM or PFM.",
                               &file_in);
  cmdline->AddPositionalOption(
      "OUTPUT", "the compressed output file (optional)", &file_out);

  // Flags.
  cmdline->AddOptionFlag('V', "version", "print version number and exit",
                         &version, &SetBooleanTrue);

  cmdline->AddOptionValue('\0', "print_profile", "0|1",
                          "print timing information before exiting",
                          &print_profile, &ParseOverride);

  switch (mode) {
    case CompressionMode::kJpegXL:
      JXL_ASSERT(cjxl_args.AddCommandLineOptions(cmdline));
      break;
    case CompressionMode::kBrunsli:
      JXL_ASSERT(cbrunsli_args.AddCommandLineOptions(cmdline));
      break;
  }
}

jxl::Status CompressArgs::ValidateArgs(const CommandLineParser& cmdline,
                                       CompressionMode mode) {
  cjxl_args.params.file_in = file_in;
  cjxl_args.params.file_out = file_out;
  cbrunsli_args.file_in = file_in;
  cbrunsli_args.file_out = file_out;

  if (file_in == nullptr) {
    fprintf(stderr, "Missing INPUT filename.\n");
    return false;
  }

  switch (mode) {
    case CompressionMode::kJpegXL:
      return cjxl_args.ValidateArgs(cmdline);
    case CompressionMode::kBrunsli:
      return cbrunsli_args.ValidateArgs(cmdline);
  }

  return true;
}

int CompressJpegXlMain(CompressionMode mode, int argc, const char* argv[]) {
  CommandLineParser cmdline;
  CompressArgs args;
  args.AddCommandLineOptions(&cmdline, mode);

  if (!cmdline.Parse(argc, argv) || !args.ValidateArgs(cmdline, mode)) {
    cmdline.PrintHelp();
    return 1;
  }

  if (args.version) {
    fprintf(stderr, "cjpegxl - version " JPEGXL_VERSION "\n");
    fprintf(stderr, "Copyright (c) the JPEG XL Project\n");
    return 0;
  }

  const int bits = hwy::TargetBitfield().Bits();
  if ((bits & HWY_STATIC_TARGETS) != HWY_STATIC_TARGETS) {
    fprintf(stderr, "CPU does not support all enabled targets => exiting.\n");
    return 1;
  }

  jxl::PaddedBytes compressed;

  switch (mode) {
    case CompressionMode::kJpegXL: {
      jxl::ThreadPoolInternal pool(args.cjxl_args.num_threads);
      if (!CompressJxl(&pool, args.cjxl_args, &compressed)) return 1;
      if (args.file_out) {
        if (!jxl::WriteFile(compressed, args.file_out)) return 1;
      }
    } break;
    case CompressionMode::kBrunsli: {
      // TODO(eustas): add num_threads parameter.
      jxl::ThreadPoolInternal pool(0);
      if (!CompressBrunsli(&pool, args.cbrunsli_args, &compressed)) return 1;
      if (args.file_out) {
        if (!jxl::WriteFile(compressed, args.file_out)) return 1;
      }
    } break;
  }

  if (args.print_profile == jxl::Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  jxl::CacheAligned::PrintStats();
  return 0;
}

}  // namespace tools
}  // namespace jpegxl
