// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/ssimulacra2.h"

int PrintUsage(char** argv) {
  fprintf(stderr, "Usage: %s orig.png distorted.png\n", argv[0]);
  fprintf(stderr,
          "Returns a score in range -inf..100, which correlates to subjective "
          "visual quality:\n");
  fprintf(stderr,
          "     30 = low quality (p10 worst output of mozjpeg -quality 30)\n");
  fprintf(stderr,
          "     50 = medium quality (average output of cjxl -q 40 or mozjpeg "
          "-quality 40,\n");
  fprintf(stderr,
          "                          p10 output of cjxl -q 50 or mozjpeg "
          "-quality 60)\n");
  fprintf(stderr,
          "     70 = high quality (average output of cjxl -q 70 or mozjpeg "
          "-quality 70,\n");
  fprintf(stderr,
          "                        p10 output of cjxl -q 75 or mozjpeg "
          "-quality 80)\n");
  fprintf(stderr,
          "     90 = very high quality (impossible to distinguish from "
          "original at 1:1,\n");
  fprintf(stderr,
          "                             average output of cjxl -q 90 or "
          "mozjpeg -quality 90)\n");
  return 1;
}

int main(int argc, char** argv) {
  if (argc != 3) return PrintUsage(argv);
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();

  jxl::CodecInOut io1{memory_manager};
  jxl::CodecInOut io2{memory_manager};
  jxl::CodecInOut* io[2] = {&io1, &io2};
  const char* purpose[] = {"original", "distorted"};
  for (size_t i = 0; i < 2; ++i) {
    std::vector<uint8_t> encoded;
    if (!jpegxl::tools::ReadFile(argv[1 + i], &encoded)) {
      fprintf(stderr, "Could not load %s image: %s\n", purpose[i], argv[1 + i]);
      return 1;
    }
    if (!jxl::SetFromBytes(jxl::Bytes(encoded), jxl::extras::ColorHints(),
                           io[i])) {
      fprintf(stderr, "Could not decode %s image: %s\n", purpose[i],
              argv[1 + i]);
      return 1;
    }
    if (io[i]->xsize() < 8 || io[i]->ysize() < 8) {
      fprintf(stderr, "Minimum image size is 8x8 pixels\n");
      return 1;
    }
  }

  if (io1.xsize() != io2.xsize() || io1.ysize() != io2.ysize()) {
    fprintf(stderr, "Image size mismatch\n");
    return 1;
  }

  if (!io1.Main().HasAlpha()) {
    jxl::StatusOr<Msssim> msssim_or =
        ComputeSSIMULACRA2(io1.Main(), io2.Main());
    if (!msssim_or.ok()) {
      fprintf(stderr, "ComputeSSIMULACRA2 failed\n");
      return 1;
    }
    Msssim msssim = std::move(msssim_or).value();
    printf("%.8f\n", msssim.Score());
  } else {
    // in case of alpha transparency: blend against dark and bright backgrounds
    // and return the worst of both scores
    jxl::StatusOr<Msssim> msssim0_or =
        ComputeSSIMULACRA2(io1.Main(), io2.Main(), 0.1f);
    if (!msssim0_or.ok()) {
      fprintf(stderr, "ComputeSSIMULACRA2 failed\n");
      return 1;
    }
    Msssim msssim0 = std::move(msssim0_or).value();
    jxl::StatusOr<Msssim> msssim1_or =
        ComputeSSIMULACRA2(io1.Main(), io2.Main(), 0.9f);
    if (!msssim1_or.ok()) {
      fprintf(stderr, "ComputeSSIMULACRA2 failed\n");
      return 1;
    }
    Msssim msssim1 = std::move(msssim1_or).value();
    printf("%.8f\n", std::min(msssim0.Score(), msssim1.Score()));
  }
  return 0;
}
