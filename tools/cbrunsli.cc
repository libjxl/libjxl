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

#include "tools/cbrunsli.h"

#include <brunsli/brunsli_encode.h>
#include <brunsli/jpeg_data_reader.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "jxl/base/data_parallel.h"
#include "jxl/base/file_io.h"
#include "jxl/base/os_specific.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/brunsli.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/extras/codec.h"
#include "jxl/extras/codec_jpg.h"
#include "jxl/image_bundle.h"
#include "tools/args.h"
#include "tools/cmdline.h"
#include "tools/speed_stats.h"

namespace jpegxl {
namespace tools {

BrunsliCompressArgs::BrunsliCompressArgs() {
  // TODO(eustas): find a way to make it more elegant.
  // Linear transform is important for HDR PFM files to avoid banding.
  dec_hints.Add("color_space", "RGB_D65_SRG_Rel_Lin");
}

jxl::Status BrunsliCompressArgs::AddCommandLineOptions(
    CommandLineParser* cmdline) {
  cmdline->AddOptionValue('q', "quant", "Q",
                          "quant scale (0=best quality, 256=worst)",
                          &quant_scale, &ParseUnsigned);

  cmdline->AddOptionValue(
      'x', "dec-hints", "key=value",
      "color_space indicates the ColorEncoding, see Description().", &dec_hints,
      &ParseAndAppendKeyValue);

  cmdline->AddOptionValue('\0', "num_reps", "N", "how many times to compress.",
                          &num_reps, &ParseUnsigned);

  return true;
}

jxl::Status BrunsliCompressArgs::ValidateArgs(
    const tools::CommandLineParser& cmdline) {
  // TODO(deymo): Make this optional for benchmarking.
  if (file_out == nullptr) {
    fprintf(stderr, "Missing OUTPUT filename.\n");
    return false;
  }

  if (quant_scale < 0 || quant_scale > 1024) {
    fprintf(stderr, "quant must be in range [0..1024].\n");
    return false;
  } else {
    options.quant_scale = quant_scale / 64.0f;
  }

  return true;
}

jxl::Status CompressBrunsli(jxl::ThreadPool* pool,
                            const BrunsliCompressArgs& args,
                            jxl::PaddedBytes* compressed) {
  jxl::CodecInOut io;
  io.dec_hints = args.dec_hints;

  jxl::PaddedBytes file_content;
  if (!jxl::ReadFile(args.file_in, &file_content)) {
    return JXL_FAILURE("Could not read input file");
  }

  if (jxl::IsJPG(jxl::Span<const uint8_t>(file_content))) {
    SpeedStats stats;
    size_t xsize = 0;
    size_t ysize = 0;
    for (size_t i = 0; i < args.num_reps; ++i) {
      compressed->clear();
      const double t0 = jxl::Now();
      brunsli::JPEGData jpg;
      const uint8_t* input_data = file_content.data();
      if (!brunsli::ReadJpeg(input_data, file_content.size(),
                             brunsli::JPEG_READ_ALL, &jpg)) {
        return JXL_FAILURE("Could not parse JPEG file");
      }

      size_t output_size = ::brunsli::GetMaximumBrunsliEncodedSize(jpg);
      std::vector<uint8_t> output(output_size);
      // TODO(eustas): introduce streaming API?
      if (!brunsli::BrunsliEncodeJpeg(jpg, output.data(), &output_size)) {
        return JXL_FAILURE("Could not encode recompressed JPEG file");
      }
      compressed->append(jxl::Span<uint8_t>(output.data(), output_size));

      const double t1 = jxl::Now();
      stats.NotifyElapsed(t1 - t0);
      xsize = jpg.width;
      ysize = jpg.height;
    }
    JXL_CHECK(stats.Print(xsize, ysize, /* num_threads */ 1));

    return true;
  }

  if (!SetFromBytes(jxl::Span<const uint8_t>(file_content), &io)) {
    fprintf(stderr, "Failed to decode image %s.\n", args.file_in);
    return false;
  }

  jxl::BrunsliEncoderOptions options = args.options;
  if (io.metadata.bit_depth.bits_per_sample > 8) {
    options.hdr_orig_colorspace = jxl::Description(io.Main().c_current());
    jxl::ColorEncoding hdr;
    std::string hdr_description = "RGB_D65_202_Rel_PeQ";
    JXL_CHECK(jxl::ParseDescription(hdr_description, &hdr));
    JXL_CHECK(hdr.CreateICC());
    JXL_CHECK(io.Main().TransformTo(hdr, pool));
    io.metadata.color_encoding = hdr;
  }

  if (!PixelsToBrunsli(&io, compressed, options, pool)) {
    fprintf(stderr, "Failed to compress.\n");
    return false;
  }

  return true;
}

}  // namespace tools
}  // namespace jpegxl
