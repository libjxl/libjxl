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

#ifndef TOOLS_CJPEGXL_H_
#define TOOLS_CJPEGXL_H_

#include <stddef.h>

#include "jxl/base/override.h"
#include "jxl/base/status.h"
#include "tools/cjxl.h"
#include "tools/cmdline.h"

namespace jpegxl {
namespace tools {

struct CompressArgs {
  // Add all the command line options to the CommandLineParser. Note that the
  // options are tied to the instance that this was called on.
  void AddCommandLineOptions(CommandLineParser* cmdline);

  // Validate the passed arguments, checking whether all passed options are
  // compatible. Returns whether the validation was successful.
  jxl::Status ValidateArgs(const CommandLineParser& cmdline);

  // Common flags.
  // TODO(deymo): Move more flags here.
  bool version = false;
  bool use_container = false;
  bool quiet = false;

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  jxl::Override print_profile = jxl::Override::kDefault;

  // Algorithm-specific flags.
  JxlCompressArgs cjxl_args;
};

int CompressJpegXlMain(int argc, const char* argv[]);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_CJPEGXL_H_
