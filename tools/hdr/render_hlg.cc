// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>

#include <cmath>

#include "lib/extras/codec.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "tools/args.h"
#include "tools/cmdline.h"

namespace jxl {
namespace {

float GetSystemGamma(const float peak_luminance,
                     const float surround_luminance) {
  return 1.2f * std::pow(1.111f, std::log2(peak_luminance / 1000.f)) *
         std::pow(0.98f, std::log2(surround_luminance / 5));
}

Status HlgOOTF(ImageBundle* ib, const float gamma, ThreadPool* pool) {
  ColorEncoding linear_rec2020;
  linear_rec2020.SetColorSpace(ColorSpace::kRGB);
  linear_rec2020.primaries = Primaries::k2100;
  linear_rec2020.white_point = WhitePoint::kD65;
  linear_rec2020.tf.SetTransferFunction(TransferFunction::kLinear);
  JXL_RETURN_IF_ERROR(linear_rec2020.CreateICC());
  JXL_RETURN_IF_ERROR(ib->TransformTo(linear_rec2020, pool));

  return RunOnPool(
      pool, 0, ib->ysize(), ThreadPool::SkipInit(),
      [&](const int y, const int thread) {
        float* const JXL_RESTRICT rows[3] = {ib->color()->PlaneRow(0, y),
                                             ib->color()->PlaneRow(1, y),
                                             ib->color()->PlaneRow(2, y)};
        for (size_t x = 0; x < ib->xsize(); ++x) {
          float& red = rows[0][x];
          float& green = rows[1][x];
          float& blue = rows[2][x];
          const float luminance =
              0.2627f * red + 0.6780f * green + 0.0593f * blue;
          const float ratio = std::pow(luminance, gamma - 1);
          if (std::isfinite(ratio)) {
            red *= ratio;
            green *= ratio;
            blue *= ratio;
          }
        }
      },
      "HlgOOTF");
}

}  // namespace
}  // namespace jxl

int main(int argc, const char** argv) {
  jxl::ThreadPoolInternal pool;

  jpegxl::tools::CommandLineParser parser;
  float target_nits = 0;
  auto target_nits_option = parser.AddOptionValue(
      't', "target_nits", "nits", "peak luminance of the target display",
      &target_nits, &jpegxl::tools::ParseFloat, 0);
  float surround_nits = 5;
  parser.AddOptionValue(
      's', "surround_nits", "nits",
      "surround luminance of the viewing environment (default: 5)",
      &surround_nits, &jpegxl::tools::ParseFloat, 0);
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

  jxl::CodecInOut image;
  jxl::ColorHints color_hints;
  color_hints.Add("color_space", "RGB_D65_202_Rel_HLG");
  JXL_CHECK(jxl::SetFromFile(input_filename, color_hints, &image, &pool));
  const float gamma = jxl::GetSystemGamma(target_nits, surround_nits);
  fprintf(stderr, "Using a system gamma of %g\n", gamma);
  JXL_CHECK(jxl::HlgOOTF(&image.Main(), gamma, &pool));
  image.metadata.m.SetIntensityTarget(target_nits);

  jxl::ColorEncoding c_out = image.metadata.m.color_encoding;
  if (pq) {
    c_out.tf.SetTransferFunction(jxl::TransferFunction::kPQ);
  } else {
    c_out.tf.SetTransferFunction(jxl::TransferFunction::kSRGB);
  }
  JXL_CHECK(c_out.CreateICC());
  JXL_CHECK(image.TransformTo(c_out, &pool));
  image.metadata.m.color_encoding = c_out;
  JXL_CHECK(jxl::EncodeToFile(image, output_filename, &pool));
}
