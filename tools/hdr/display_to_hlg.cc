// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstdio>
#include <cstdlib>

#include "lib/extras/codec.h"
#include "lib/extras/hlg.h"
#include "lib/extras/tone_mapping.h"
#include "lib/jxl/base/span.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"
#include "tools/hdr/image_utils.h"
#include "tools/no_memory_manager.h"
#include "tools/thread_pool_internal.h"

int main(int argc, const char** argv) {
  jpegxl::tools::ThreadPoolInternal pool;

  jpegxl::tools::CommandLineParser parser;
  float max_nits = 0;
  auto max_nits_option = parser.AddOptionValue(
      'm', "max_nits", "nits", "maximum luminance of the display", &max_nits,
      &jpegxl::tools::ParseFloat, 0);
  float surround_nits = 5;
  parser.AddOptionValue(
      's', "surround_nits", "nits",
      "surround luminance of the viewing environment (default: 5)",
      &surround_nits, &jpegxl::tools::ParseFloat, 0);
  float preserve_saturation = .1f;
  parser.AddOptionValue(
      '\0', "preserve_saturation", "0..1",
      "to what extent to try and preserve saturation over luminance if an "
      "inverse gamma < 1 generates out-of-gamut colors",
      &preserve_saturation, &jpegxl::tools::ParseFloat, 0);
  const char* input_filename = nullptr;
  auto input_filename_option = parser.AddPositionalOption(
      "input", true, "input image", &input_filename, 0);
  const char* output_filename = nullptr;
  auto output_filename_option = parser.AddPositionalOption(
      "output", true, "output image", &output_filename, 0);

  if (!parser.Parse(argc, argv)) {
    fprintf(stderr, "See -h for help.\n");
    return EXIT_FAILURE;
  }

  if (parser.HelpFlagPassed()) {
    parser.PrintHelp();
    return EXIT_SUCCESS;
  }

  if (!parser.GetOption(max_nits_option)->matched()) {
    fprintf(stderr,
            "Missing required argument --max_nits.\nSee -h for help.\n");
    return EXIT_FAILURE;
  }
  if (!parser.GetOption(input_filename_option)->matched()) {
    fprintf(stderr, "Missing input filename.\nSee -h for help.\n");
    return EXIT_FAILURE;
  }
  if (!parser.GetOption(output_filename_option)->matched()) {
    fprintf(stderr, "Missing output filename.\nSee -h for help.\n");
    return EXIT_FAILURE;
  }

  std::vector<uint8_t> encoded;
  JPEGXL_TOOLS_CHECK(jpegxl::tools::ReadFile(input_filename, &encoded));
  jxl::CodecInOut image{jpegxl::tools::NoMemoryManager()};
  JPEGXL_TOOLS_CHECK(jxl::SetFromBytes(
      jxl::Bytes(encoded), jxl::extras::ColorHints(), &image, pool.get()));
  image.metadata.m.SetIntensityTarget(max_nits);
  JPEGXL_TOOLS_CHECK(jxl::HlgInverseOOTF(
      &image.Main(), jxl::GetHlgGamma(max_nits, surround_nits), pool.get()));
  JPEGXL_TOOLS_CHECK(jxl::GamutMap(&image, preserve_saturation, pool.get()));
  image.metadata.m.SetIntensityTarget(301);

  jxl::ColorEncoding hlg;
  hlg.SetColorSpace(jxl::ColorSpace::kRGB);
  JPEGXL_TOOLS_CHECK(hlg.SetPrimariesType(jxl::Primaries::k2100));
  JPEGXL_TOOLS_CHECK(hlg.SetWhitePointType(jxl::WhitePoint::kD65));
  hlg.Tf().SetTransferFunction(jxl::TransferFunction::kHLG);
  JPEGXL_TOOLS_CHECK(hlg.CreateICC());
  JPEGXL_TOOLS_CHECK(
      jpegxl::tools::TransformCodecInOutTo(image, hlg, pool.get()));
  image.metadata.m.color_encoding = hlg;
  JPEGXL_TOOLS_CHECK(
      jpegxl::tools::Encode(image, output_filename, &encoded, pool.get()));
  JPEGXL_TOOLS_CHECK(jpegxl::tools::WriteFile(output_filename, encoded));
}
