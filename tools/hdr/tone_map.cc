// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/extras/tone_mapping.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"
#include "tools/hdr/image_utils.h"
#include "tools/no_memory_manager.h"
#include "tools/thread_pool_internal.h"

int main(int argc, const char** argv) {
  jpegxl::tools::ThreadPoolInternal pool;

  jpegxl::tools::CommandLineParser parser;
  float max_nits = 0;
  parser.AddOptionValue('m', "max_nits", "nits",
                        "maximum luminance in the image", &max_nits,
                        &jpegxl::tools::ParseFloat, 0);
  float target_nits = 0;
  auto target_nits_option = parser.AddOptionValue(
      't', "target_nits", "nits",
      "peak luminance of the display for which to tone map", &target_nits,
      &jpegxl::tools::ParseFloat, 0);
  float preserve_saturation = .1f;
  parser.AddOptionValue(
      's', "preserve_saturation", "0..1",
      "to what extent to try and preserve saturation over luminance",
      &preserve_saturation, &jpegxl::tools::ParseFloat, 0);
  bool pq = false;
  parser.AddOptionFlag('p', "pq",
                       "write the output with absolute luminance using PQ", &pq,
                       &jpegxl::tools::SetBooleanTrue, 0);
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

  if (!parser.GetOption(target_nits_option)->matched()) {
    fprintf(stderr,
            "Missing required argument --target_nits.\nSee -h for help.\n");
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

  jxl::CodecInOut image{jpegxl::tools::NoMemoryManager()};
  jxl::extras::ColorHints color_hints;
  color_hints.Add("color_space", "RGB_D65_202_Rel_PeQ");
  std::vector<uint8_t> encoded;
  JPEGXL_TOOLS_CHECK(jpegxl::tools::ReadFile(input_filename, &encoded));
  JPEGXL_TOOLS_CHECK(
      jxl::SetFromBytes(jxl::Bytes(encoded), color_hints, &image, pool.get()));
  if (max_nits > 0) {
    image.metadata.m.SetIntensityTarget(max_nits);
  }
  JPEGXL_TOOLS_CHECK(jxl::ToneMapTo({0, target_nits}, &image, pool.get()));
  JPEGXL_TOOLS_CHECK(jxl::GamutMap(&image, preserve_saturation, pool.get()));

  jxl::ColorEncoding c_out = image.metadata.m.color_encoding;
  jxl::cms::TransferFunction tf =
      pq ? jxl::TransferFunction::kPQ : jxl::TransferFunction::kSRGB;

  if (jxl::extras::CodecFromPath(output_filename) == jxl::extras::Codec::kEXR) {
    tf = jxl::TransferFunction::kLinear;
    image.metadata.m.SetFloat16Samples();
  }
  c_out.Tf().SetTransferFunction(tf);

  JPEGXL_TOOLS_CHECK(c_out.CreateICC());
  JPEGXL_TOOLS_CHECK(
      jpegxl::tools::TransformCodecInOutTo(image, c_out, pool.get()));
  image.metadata.m.color_encoding = c_out;
  JPEGXL_TOOLS_CHECK(
      jpegxl::tools::Encode(image, output_filename, &encoded, pool.get()));
  JPEGXL_TOOLS_CHECK(jpegxl::tools::WriteFile(output_filename, encoded));
  return EXIT_SUCCESS;
}
