// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>

#include "lib/extras/codec.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/image_bundle.h"
#include "tools/image_utils.h"
#include "tools/ssimulacra.h"

namespace ssimulacra {
namespace {

int PrintUsage(char** argv) {
  fprintf(stderr, "Usage: %s [-v] [-s] orig.png distorted.png\n", argv[0]);
  return 1;
}

int Run(int argc, char** argv) {
  if (argc < 2) return PrintUsage(argv);

  bool verbose = false, simple = false;
  int input_arg = 1;
  if (!strcmp(argv[input_arg], "-v")) {
    verbose = true;
    input_arg++;
  }
  if (!strcmp(argv[input_arg], "-s")) {
    simple = true;
    input_arg++;
  }
  if (argc < input_arg + 2) return PrintUsage(argv);

  jxl::CodecInOut io1;
  jxl::CodecInOut io2;
  JXL_CHECK(SetFromFile(argv[input_arg], jxl::extras::ColorHints(), &io1));
  JXL_CHECK(SetFromFile(argv[input_arg + 1], jxl::extras::ColorHints(), &io2));
  JXL_CHECK(jpegxl::tools::TransformCodecInOutTo(
      io1, jxl::ColorEncoding::LinearSRGB(io1.Main().IsGray()),
      jxl::GetJxlCms(), nullptr));
  JXL_CHECK(jpegxl::tools::TransformCodecInOutTo(
      io2, jxl::ColorEncoding::LinearSRGB(io2.Main().IsGray()),
      jxl::GetJxlCms(), nullptr));

  if (io1.xsize() != io2.xsize() || io1.ysize() != io2.ysize()) {
    fprintf(stderr, "Image size mismatch\n");
    return 1;
  }
  if (io1.xsize() < 8 || io1.ysize() < 8) {
    fprintf(stderr, "Minimum image size is 8x8 pixels\n");
    return 1;
  }

  Ssimulacra ssimulacra =
      ComputeDiff(*io1.Main().color(), *io2.Main().color(), simple);

  if (verbose) {
    ssimulacra.PrintDetails();
  }
  printf("%.8f\n", ssimulacra.Score());
  return 0;
}

}  // namespace
}  // namespace ssimulacra

int main(int argc, char** argv) { return ssimulacra::Run(argc, argv); }
