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

#ifndef TOOLS_CBRUNSLI_H_
#define TOOLS_CBRUNSLI_H_

#include <stddef.h>

#include "jxl/base/data_parallel.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/status.h"
#include "jxl/brunsli.h"
#include "jxl/codec_in_out.h"
#include "tools/cmdline.h"

namespace jpegxl {
namespace tools {

struct BrunsliCompressArgs {
  // Initialize non-static default options.
  BrunsliCompressArgs();

  // Add all the command line options to the CommandLineParser. Note that the
  // options are tied to the instance that this was called on.
  jxl::Status AddCommandLineOptions(CommandLineParser* cmdline);

  // Validate the passed arguments, checking whether all passed options are
  // compatible. Returns whether the validation was successful.
  jxl::Status ValidateArgs(const CommandLineParser& cmdline);

  jxl::BrunsliEncoderOptions options;

  jxl::DecoderHints dec_hints;

  size_t quant_scale = 0;

  const char* file_in = nullptr;
  const char* file_out = nullptr;
};

jxl::Status CompressBrunsli(jxl::ThreadPool* pool,
                            const BrunsliCompressArgs& args,
                            jxl::PaddedBytes* compressed);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_CBRUNSLI_H_
