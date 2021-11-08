// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_CJXL_H_
#define TOOLS_CJXL_H_

#include <stddef.h>

#include <string>
#include <thread>
#include <utility>

#include "lib/extras/color_hints.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/jxl_inspection.h"
#include "tools/cmdline.h"

namespace jpegxl {
namespace tools {

struct CompressArgs {
  void SetInspectorImage3F(const jxl::InspectorImage3F& inspector) {
    inspector_image3f = inspector;
  }

  // Add all the command line options to the CommandLineParser. Note that the
  // options are tied to the instance that this was called on.
  void AddCommandLineOptions(CommandLineParser* cmdline);

  // Post-processes and validates the passed arguments, checking whether all
  // passed options are compatible. Returns whether the validation was
  // successful.
  jxl::Status ValidateArgs(const CommandLineParser& cmdline);

  // Validates the arguments again, having loaded the input so sensible defaults
  // can be chosen based on e.g. dimensions.
  jxl::Status ValidateArgsAfterLoad(const CommandLineParser& cmdline,
                                    const jxl::CodecInOut& io);

  // Common flags.
  bool version = false;
  bool use_container = false;
  bool no_container = false;
  bool quiet = false;

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  jxl::Override print_profile = jxl::Override::kDefault;

  // Decoding source image flags
  jxl::ColorHints color_hints;

  // JXL flags
  size_t override_bitdepth = 0;
  jxl::CompressParams params;
  size_t num_threads = std::thread::hardware_concurrency();
  size_t num_reps = 1;
  float intensity_target = 0;

  // Filename for the user provided saliency-map.
  std::string saliency_map_filename;

  // Whether to perform lossless transcoding with kVarDCT or kJPEG encoding.
  // If true, attempts to load JPEG coefficients instead of pixels.
  // Reset to false if input image is not a JPEG.
  bool jpeg_transcode = true;

  float quality = -1001.f;  // Default to lossless if input is already lossy,
                            // or to VarDCT otherwise.
  bool progressive = false;
  bool default_settings = true;
  bool force_premultiplied = false;

  // Will get passed on to AuxOut.
  jxl::InspectorImage3F inspector_image3f;

  // References (ids) of specific options to check if they were matched.
  CommandLineParser::OptionId opt_distance_id = -1;
  CommandLineParser::OptionId opt_target_size_id = -1;
  CommandLineParser::OptionId opt_target_bpp_id = -1;
  CommandLineParser::OptionId opt_quality_id = -1;
  CommandLineParser::OptionId opt_near_lossless_id = -1;
  CommandLineParser::OptionId opt_intensity_target_id = -1;
  CommandLineParser::OptionId opt_color_id = -1;
  CommandLineParser::OptionId opt_m_group_size_id = -1;
};

jxl::Status LoadAll(CompressArgs& args, jxl::ThreadPoolInternal* pool,
                    jxl::CodecInOut* io, double* decode_mps);

// The input image must already have been loaded into io using LoadAll.
jxl::Status CompressJxl(jxl::CodecInOut& io, double decode_mps,
                        jxl::ThreadPoolInternal* pool, CompressArgs& args,
                        jxl::PaddedBytes* compressed, bool print_stats = true);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_CJXL_H_
