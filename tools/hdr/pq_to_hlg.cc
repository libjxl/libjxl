// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>

#include <cmath>

#include "lib/extras/codec.h"
#include "lib/extras/tone_mapping.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "tools/args.h"
#include "tools/cmdline.h"

namespace jxl {
namespace {

Status HlgInverseOOTF(ImageBundle* ib, ThreadPool* pool) {
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
          const float ratio = std::pow(luminance, 1 / 1.2f - 1);
          if (std::isfinite(ratio)) {
            red *= ratio;
            green *= ratio;
            blue *= ratio;
          }
        }
      },
      "HlgInverseOOTF");
}

}  // namespace
}  // namespace jxl

int main(int argc, const char** argv) {
  jxl::ThreadPoolInternal pool;

  jpegxl::tools::CommandLineParser parser;
  float max_nits = 0;
  parser.AddOptionValue('m', "max_nits", "nits",
                        "maximum luminance in the image", &max_nits,
                        &jpegxl::tools::ParseFloat, 0);
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
  color_hints.Add("color_space", "RGB_D65_202_Rel_PeQ");
  JXL_CHECK(jxl::SetFromFile(input_filename, color_hints, &image, &pool));
  if (max_nits > 0) {
    image.metadata.m.SetIntensityTarget(max_nits);
  }
  JXL_CHECK(jxl::ToneMapTo({0, 1000}, &image, &pool));
  JXL_CHECK(jxl::HlgInverseOOTF(&image.Main(), &pool));

  jxl::ColorEncoding hlg;
  hlg.SetColorSpace(jxl::ColorSpace::kRGB);
  hlg.primaries = jxl::Primaries::k2100;
  hlg.white_point = jxl::WhitePoint::kD65;
  hlg.tf.SetTransferFunction(jxl::TransferFunction::kHLG);
  JXL_CHECK(hlg.CreateICC());
  JXL_CHECK(image.TransformTo(hlg, &pool));
  image.metadata.m.color_encoding = hlg;
  JXL_CHECK(jxl::EncodeToFile(image, output_filename, &pool));
}
