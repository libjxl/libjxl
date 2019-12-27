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

#ifndef TOOLS_DJXL_H_
#define TOOLS_DJXL_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/codec_in_out.h"
#include "jxl/dec_params.h"
#include "tools/cmdline.h"
#include "tools/speed_stats.h"

namespace jpegxl {
namespace tools {

struct JxlDecompressArgs {
  // Add all the command line options to the CommandLineParser. Note that the
  // options are tied to the instance that this was called on.
  void AddCommandLineOptions(tools::CommandLineParser* cmdline);

  // Validate the passed arguments, checking whether all passed options are
  // compatible. Returns whether the validation was successful.
  jxl::Status ValidateArgs();

  // The parameters.
  size_t bits_per_sample = 0;
  std::string color_space;  // description

  bool brunsli_fix_dc_staircase = false;
  bool brunsli_gaborish = false;

  jxl::DecompressParams params;

  // If true, print the effective amount of bytes read from the bitstream.
  bool print_read_bytes = false;

  bool coalesce = false;
};

// Decompresses and notifies SpeedStats of elapsed time.
jxl::Status DecompressJxl(const jxl::Span<const uint8_t> compressed,
                          const jxl::DecompressParams& params,
                          jxl::ThreadPool* pool,
                          jxl::CodecInOut* JXL_RESTRICT io,
                          jxl::AuxOut* aux_out, SpeedStats* JXL_RESTRICT stats);

jxl::Status WriteJxlOutput(const JxlDecompressArgs& args, const char* file_out,
                           const jxl::CodecInOut& io);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_DJXL_H_
