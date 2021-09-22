// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>

#include "lib/extras/codec.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "tools/args.h"
#include "tools/cmdline.h"
#include "tools/epf.h"

int main(int argc, const char** argv) {
  float distance = 1.f;
  size_t sharpness = 1;
  size_t epf_iters = 2;
  size_t num_threads = 8;
  const char* input_filename = nullptr;
  const char* output_filename = nullptr;

  jpegxl::tools::CommandLineParser cmdline;
  cmdline.AddOptionValue(
      'd', "distance", "1..",
      "Butteraugli distance to assume for quantization values", &distance,
      &jpegxl::tools::ParseFloat);
  cmdline.AddOptionValue('s', "sharpness", "1..7",
                         "EPF sharpness value from 1 to 7", &sharpness,
                         &jpegxl::tools::ParseUnsigned);
  cmdline.AddOptionValue('\0', "epf", "1..3", "number of epf iterations",
                         &epf_iters, &jpegxl::tools::ParseUnsigned);
  cmdline.AddOptionValue('\0', "num_threads", "N",
                         "The number of threads to use", &num_threads,
                         &jpegxl::tools::ParseUnsigned);
  cmdline.AddPositionalOption("INPUT", /* required = */ true, "Input image",
                              &input_filename);
  cmdline.AddPositionalOption("OUTPUT", /* required = */ true, "Output image",
                              &output_filename);
  if (!cmdline.Parse(argc, argv) || input_filename == nullptr ||
      output_filename == nullptr) {
    cmdline.PrintHelp();
    return EXIT_FAILURE;
  }
  if (!epf_iters || epf_iters > 3) {
    fprintf(stderr, "epf_iters value (%zu) is out of range, must be 1..3.\n",
            epf_iters);
    return EXIT_FAILURE;
  }

  jxl::ThreadPoolInternal pool(num_threads);
  jxl::CodecInOut io;
  if (!jxl::SetFromFile(input_filename, jxl::ColorHints(), &io, &pool)) {
    fprintf(stderr, "Failed to read from \"%s\".\n", input_filename);
    return EXIT_FAILURE;
  }

  if (!jpegxl::tools::RunEPF(epf_iters, distance, sharpness, &io, &pool)) {
    fprintf(stderr, "Failed to run the EPF\n");
    return EXIT_FAILURE;
  }

  if (!jxl::EncodeToFile(io, output_filename, &pool)) {
    fprintf(stderr, "Failed to write the result to \"%s\".\n", output_filename);
    return EXIT_FAILURE;
  }
}
