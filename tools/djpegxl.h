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

#ifndef TOOLS_DJPEGXL_H_
#define TOOLS_DJPEGXL_H_

#include <stddef.h>

#include "jxl/base/override.h"
#include "jxl/base/status.h"
#include "tools/args.h"
#include "tools/cmdline.h"
#include "tools/djxl.h"

namespace jpegxl {
namespace tools {

// Common JPEG XL decompress arguments.
struct DecompressArgs {
  // Initialize non-static default options.
  DecompressArgs();

  // Add all the command line options to the CommandLineParser. Note that the
  // options are tied to the instance that this was called on.
  void AddCommandLineOptions(CommandLineParser* cmdline);

  // Validate the passed arguments, checking whether all passed options are
  // compatible. Returns whether the validation was successful.
  jxl::Status ValidateArgs(const CommandLineParser& cmdline);

  // Common djpegxl parameters.
  const char* file_in = nullptr;
  const char* file_out = nullptr;
  size_t num_threads;
  bool use_sjpeg = false;
  size_t jpeg_quality = 95;
  bool version = false;
  jxl::Override print_profile = jxl::Override::kDefault;
  jxl::Override print_info = jxl::Override::kDefault;

  size_t num_reps = 1;

  JxlDecompressArgs djxl_args;

  // References (ids) of specific options to check if they were matched.
  CommandLineParser::OptionId opt_num_threads_id = -1;
};

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_DJPEGXL_H_
