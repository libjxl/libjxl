// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_DJXL_H_
#define TOOLS_DJXL_H_

#include <stddef.h>

#include <thread>

#include "jxl/decode.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/dec_params.h"
#include "tools/args.h"
#include "tools/box/box.h"
#include "tools/cmdline.h"
#include "tools/speed_stats.h"

namespace jpegxl {
namespace tools {

// Common JPEG XL decompress arguments.
struct DecompressArgs {
  // Initialize non-static default options.
  DecompressArgs() = default;

  // Add all the command line options to the CommandLineParser. Note that the
  // options are tied to the instance that this was called on.
  void AddCommandLineOptions(CommandLineParser* cmdline);

  // Validate the passed arguments, checking whether all passed options are
  // compatible. Returns whether the validation was successful.
  jxl::Status ValidateArgs(const CommandLineParser& cmdline);

  // Common djxl parameters.
  const char* file_in = nullptr;
  const char* file_out = nullptr;
  size_t num_threads = std::thread::hardware_concurrency();
  bool use_sjpeg = false;
  size_t jpeg_quality = 95;
  bool decode_to_pixels = false;
  bool version = false;
  jxl::Override print_profile = jxl::Override::kDefault;

  size_t num_reps = 1;

  // Format parameters:

  size_t bits_per_sample = 0;
  bool tone_map = false;
  std::pair<float, float> display_nits = {0.f, jxl::kDefaultIntensityTarget};
  float preserve_saturation = .1f;
  std::string color_space;  // description or path to ICC profile

  jxl::DecompressParams params;

  // If true, print the effective amount of bytes read from the bitstream.
  bool print_read_bytes = false;
  bool quiet = false;

  // References (ids) of specific options to check if they were matched.
  CommandLineParser::OptionId opt_jpeg_quality_id = -1;
};

// Decompresses and notifies SpeedStats of elapsed time.
jxl::Status DecompressJxlToPixels(const jxl::Span<const uint8_t> compressed,
                                  const jxl::DecompressParams& params,
                                  jxl::ThreadPool* pool,
                                  jxl::CodecInOut* JXL_RESTRICT io,
                                  SpeedStats* JXL_RESTRICT stats);

jxl::Status DecompressJxlToJPEG(const JpegXlContainer& container,
                                const DecompressArgs& args,
                                jxl::ThreadPool* pool, jxl::PaddedBytes* output,
                                SpeedStats* JXL_RESTRICT stats);

jxl::Status WriteJxlOutput(const DecompressArgs& args, const char* file_out,
                           jxl::CodecInOut& io,
                           jxl::ThreadPool* pool = nullptr);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_DJXL_H_
