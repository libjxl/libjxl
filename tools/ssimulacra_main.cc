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

#include "lib/extras/codec.h"
#include "lib/jxl/color_management.h"
#include "tools/ssimulacra.h"

namespace ssimulacra {
namespace {

int PrintUsage(char** argv) {
  fprintf(stderr, "Usage: %s [-v] orig.png distorted.png\n", argv[0]);
  return 1;
}

int Run(int argc, char** argv) {
  if (argc < 2) return PrintUsage(argv);

  bool verbose = false;
  int input_arg = 1;
  if (argv[1][0] == '-' && argv[1][1] == 'v') {
    verbose = true;
    input_arg = 2;
  }
  if (argc < input_arg + 2) return PrintUsage(argv);

  jxl::CodecInOut io1;
  jxl::CodecInOut io2;
  JXL_CHECK(SetFromFile(argv[input_arg], &io1));
  JXL_CHECK(SetFromFile(argv[input_arg + 1], &io2));
  JXL_CHECK(
      io1.TransformTo(jxl::ColorEncoding::LinearSRGB(io1.Main().IsGray())));
  JXL_CHECK(
      io2.TransformTo(jxl::ColorEncoding::LinearSRGB(io2.Main().IsGray())));

  if (io1.xsize() != io2.xsize() || io1.ysize() != io2.ysize()) {
    fprintf(stderr, "Image size mismatch\n");
    return 1;
  }
  if (io1.xsize() < 8 || io1.ysize() < 8) {
    fprintf(stderr, "Minimum image size is 8x8 pixels\n");
    return 1;
  }

  Ssimulacra ssimulacra = ComputeDiff(*io1.Main().color(), *io2.Main().color());

  if (verbose) {
    ssimulacra.PrintDetails();
  }
  printf("%.8f\n", ssimulacra.Score());
  return 0;
}

}  // namespace
}  // namespace ssimulacra

int main(int argc, char** argv) { return ssimulacra::Run(argc, argv); }
