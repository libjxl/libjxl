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

#include <stdio.h>

#include "jxl/encode.h"
#include "lib/jxl/base/file_io.h"
#include "tools/box/box.h"
#include "tools/cjxl.h"
#include "tools/codec_config.h"

namespace jpegxl {
namespace tools {

int CompressJpegXlMain(int argc, const char* argv[]) {
  CommandLineParser cmdline;
  CompressArgs args;
  args.AddCommandLineOptions(&cmdline);

  bool printhelp = false;
  if (!cmdline.Parse(argc, argv)) {
    printhelp = true;
  }

  if (args.version) {
    fprintf(stderr, "cjxl [%s]\n",
            CodecConfigString(JxlEncoderVersion()).c_str());
    fprintf(stderr, "Copyright (c) the JPEG XL Project\n");
    return 0;
  }

  if (!args.quiet) {
    fprintf(stderr, "  J P E G   \\/ |\n");
    fprintf(stderr, "            /\\ |_   e n c o d e r    [%s]\n\n",
            CodecConfigString(JxlEncoderVersion()).c_str());
  }

  if (printhelp || !args.ValidateArgs(cmdline)) {
    cmdline.PrintHelp();
    return 1;
  }

  jxl::PaddedBytes compressed;

  jxl::ThreadPoolInternal pool(args.num_threads);
  if (!CompressJxl(&pool, args, &compressed, !args.quiet)) return 1;

  if (args.use_container &&
      !IsContainerHeader(compressed.data(), compressed.size())) {
    JpegXlContainer container;
    container.codestream = compressed.data();
    container.codestream_size = compressed.size();
    jxl::PaddedBytes container_file;
    if (!EncodeJpegXlContainerOneShot(container, &container_file)) {
      fprintf(stderr, "Failed to encode container format\n");
      return 1;
    }
    compressed.swap(container_file);
  }

  if (args.file_out) {
    if (!jxl::WriteFile(compressed, args.file_out)) return 1;
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
