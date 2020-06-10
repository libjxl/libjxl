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

#include <stdio.h>

#include "jxl/base/thread_pool_internal.h"
#include "jxl/enc_adaptive_quantization.h"
#include "jxl/extras/codec.h"
#include "tools/args.h"
#include "tools/cmdline.h"
#include "tools/epf.h"

int main(int argc, const char** argv) {
  float distance = 1.f;
  size_t sharpness = 1;
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
  cmdline.AddPositionalOption("INPUT", /* required = */ true, "Input image",
                              &input_filename);
  cmdline.AddPositionalOption("OUTPUT", /* required = */ true, "Output image",
                              &output_filename);
  if (!cmdline.Parse(argc, argv) || input_filename == nullptr ||
      output_filename == nullptr) {
    cmdline.PrintHelp();
    return EXIT_FAILURE;
  }

  jxl::ThreadPoolInternal pool;
  jxl::CodecInOut io;
  if (!jxl::SetFromFile(input_filename, &io, &pool)) {
    fprintf(stderr, "Failed to read from \"%s\".\n", input_filename);
    return EXIT_FAILURE;
  }

  if (!jpegxl::tools::RunEPF(distance, sharpness, &io, &pool)) {
    fprintf(stderr, "Failed to run the EPF\n");
    return EXIT_FAILURE;
  }

  if (!jxl::EncodeToFile(io, output_filename, &pool)) {
    fprintf(stderr, "Failed to write the result to \"%s\".\n", output_filename);
    return EXIT_FAILURE;
  }
}
