// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tool for adding noise to images using the photon noise model in the JPEG XL
// encoder.
//
// Example usage, which would add noise equivalent to
// `cjxl --photon_noise=ISO6400`:
//
//    tools/add_noise 6400 input.png output.png

#include <stdio.h>
#include <stdlib.h>

#include "lib/extras/codec.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/dec_noise.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_photon_noise.h"
#include "lib/jxl/enc_xyb.h"

int main(int argc, const char** argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <iso> <input.png> <output.png>\n", argv[0]);
    return EXIT_FAILURE;
  }

  char* end;
  const float iso = strtof(argv[1], &end);
  if (end == argv[1] || *end != '\0') {
    fprintf(stderr, "Failed to parse \"%s\" as an ISO setting\n", argv[1]);
    return EXIT_FAILURE;
  }

  jxl::ThreadPoolInternal pool;

  jxl::CodecInOut io;
  JXL_CHECK(jxl::SetFromFile(argv[2], jxl::ColorHints(), &io, &pool));

  jxl::Image3F xyb(io.Main().xsize(), io.Main().ysize());
  jxl::ToXYB(io.Main(), &pool, &xyb, jxl::GetJxlCms());

  jxl::Image3F noise_image(xyb.xsize(), xyb.ysize());
  {
    // TODO(sboukortt): perhaps add a flag to set the seed.
    jxl::RandomImage3(1337, jxl::Rect(noise_image), &noise_image);
    // TODO(sboukortt): and maybe one to use Gaussian noise instead of our
    // high-pass-filtered uniform noise.
    jxl::WeightsSymmetric5 weights{{HWY_REP4(-3.84)}, {HWY_REP4(0.16)},
                                   {HWY_REP4(0.16)},  {HWY_REP4(0.16)},
                                   {HWY_REP4(0.16)},  {HWY_REP4(0.16)}};
    jxl::ImageF noise_tmp(noise_image.xsize(), noise_image.ysize());
    for (size_t c = 0; c < 3; c++) {
      jxl::Symmetric5(noise_image.Plane(c), jxl::Rect(noise_image), weights,
                      &pool, &noise_tmp);
      std::swap(noise_image.Plane(c), noise_tmp);
    }
  }

  const jxl::NoiseParams noise_params =
      jxl::SimulatePhotonNoise(xyb.xsize(), xyb.ysize(), iso);
  jxl::AddNoise(noise_params, jxl::Rect(noise_image), noise_image,
                jxl::Rect(xyb), jxl::ColorCorrelationMap(), &xyb);

  jxl::OpsinParams opsin_params;
  opsin_params.Init(io.metadata.m.IntensityTarget());
  jxl::OpsinToLinearInplace(&xyb, &pool, opsin_params);
  io.SetFromImage(std::move(xyb), jxl::ColorEncoding::LinearSRGB());

  JXL_CHECK(jxl::EncodeToFile(io, argv[3], &pool));
}
