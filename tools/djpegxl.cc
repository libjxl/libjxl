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

#include "tools/djpegxl.h"

#include <stdio.h>

#include "jxl/base/arch_specific.h"
#include "tools/args.h"

namespace jpegxl {
namespace tools {

DecompressArgs::DecompressArgs() {}

void DecompressArgs::AddCommandLineOptions(CommandLineParser* cmdline) {
  // Positional arguments.
  cmdline->AddPositionalOption("INPUT", /* required = */ true,
                               "the compressed input file", &file_in);

  cmdline->AddPositionalOption(
      "OUTPUT", /* required = */ true,
      "the output can be PNG with ICC, JPG, or PPM/PFM.", &file_out);

  cmdline->AddOptionFlag('V', "version", "print version number and exit",
                         &version, &SetBooleanTrue);

  cmdline->AddOptionValue('\0', "num_reps", "N", nullptr, &num_reps,
                          &ParseUnsigned);

#if JPEGXL_ENABLE_SJPEG
  cmdline->AddOptionFlag('\0', "use_sjpeg",
                         "use sjpeg instead of libjpeg for JPEG output",
                         &use_sjpeg, &SetBooleanTrue);
#endif

  cmdline->AddOptionValue('\0', "jpeg_quality", "N", "JPEG output quality",
                          &jpeg_quality, &ParseUnsigned);

  opt_num_threads_id = cmdline->AddOptionValue('\0', "num_threads", "N",
                                               "The number of threads to use",
                                               &num_threads, &ParseUnsigned);

  cmdline->AddOptionValue('\0', "print_profile", "0|1",
                          "print timing information before exiting",
                          &print_profile, &ParseOverride);

  cmdline->AddOptionValue('\0', "print_info", "0|1",
                          "print AuxOut before exiting", &print_info,
                          &ParseOverride);

  djxl_args.AddCommandLineOptions(cmdline);
}

jxl::Status DecompressArgs::ValidateArgs(const CommandLineParser& cmdline) {
  if (file_in == nullptr) {
    fprintf(stderr, "Missing INPUT filename.\n");
    return false;
  }

  // User didn't override num_threads, so we have to compute a default, which
  // might fail, so only do so when necessary. Don't just check num_threads != 0
  // because the user may have set it to that.
  if (!cmdline.GetOption(opt_num_threads_id)->matched()) {
    jxl::ProcessorTopology topology;
    if (!jxl::DetectProcessorTopology(&topology)) {
      // We have seen sporadic failures caused by setaffinity_np.
      fprintf(stderr,
              "Failed to choose default num_threads; you can avoid this "
              "error by specifying a --num_threads N argument.\n");
      return false;
    }
    num_threads = topology.packages * topology.cores_per_package;
  }

  return djxl_args.ValidateArgs();
}

}  // namespace tools
}  // namespace jpegxl
