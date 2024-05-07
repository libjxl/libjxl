// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "lib/extras/codec.h"
// TODO(eustas): we should, but we can't?
// #include "lib/jxl/base/span.h"
#include <jxl/cms.h>

#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/image_bundle.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/ssimulacra.h"

namespace ssimulacra {
namespace {

int PrintUsage(char** argv) {
  fprintf(stderr, "Usage: %s [-v] [-s] orig.png distorted.png\n", argv[0]);
  return 1;
}

int Run(int argc, char** argv) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  if (argc < 2) return PrintUsage(argv);

  bool verbose = false;
  bool simple = false;
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

  jxl::CodecInOut io1{memory_manager};
  jxl::CodecInOut io2{memory_manager};
  jxl::CodecInOut* io[2] = {&io1, &io2};
  for (size_t i = 0; i < 2; ++i) {
    std::vector<uint8_t> encoded;
    JXL_CHECK(jpegxl::tools::ReadFile(argv[input_arg + i], &encoded));
    JXL_CHECK(jxl::SetFromBytes(jxl::Bytes(encoded), jxl::extras::ColorHints(),
                                io[i]));
  }
  jxl::ImageBundle& ib1 = io1.Main();
  jxl::ImageBundle& ib2 = io2.Main();
  JXL_CHECK(ib1.TransformTo(jxl::ColorEncoding::LinearSRGB(ib1.IsGray()),
                            *JxlGetDefaultCms(), nullptr));
  JXL_CHECK(ib2.TransformTo(jxl::ColorEncoding::LinearSRGB(ib2.IsGray()),
                            *JxlGetDefaultCms(), nullptr));
  jxl::Image3F& img1 = *ib1.color();
  jxl::Image3F& img2 = *ib2.color();
  if (img1.xsize() != img2.xsize() || img1.ysize() != img2.ysize()) {
    fprintf(stderr, "Image size mismatch\n");
    return 1;
  }
  if (img1.xsize() < 8 || img1.ysize() < 8) {
    fprintf(stderr, "Minimum image size is 8x8 pixels\n");
    return 1;
  }

  jxl::StatusOr<Ssimulacra> ssimulacra_or = ComputeDiff(img1, img2, simple);
  if (!ssimulacra_or.ok()) {
    fprintf(stderr, "ComputeDiff failed\n");
    return 1;
  }

  Ssimulacra ssimulacra = std::move(ssimulacra_or).value();

  if (verbose) {
    ssimulacra.PrintDetails();
  }
  printf("%.8f\n", ssimulacra.Score());
  return 0;
}

}  // namespace
}  // namespace ssimulacra

int main(int argc, char** argv) { return ssimulacra::Run(argc, argv); }
