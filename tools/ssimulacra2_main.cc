// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/extras/codec_in_out.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/ssimulacra2.h"

#define QUIT(M)               \
  fprintf(stderr, "%s\n", M); \
  return EXIT_FAILURE;

int PrintUsage(char** argv) {
  fprintf(stderr, "Usage: %s original.png distorted.png\n", argv[0]);
  fprintf(stderr,
          "Returns a score in range -inf..100, which correlates to subjective "
          "visual quality:\n");
  fprintf(
      stderr,
      "     negative scores: extremely low quality, very strong distortion\n");
  fprintf(stderr,
          "     10 = very low quality (average output of cjxl -d 14 / -q 12 or "
          "libjpeg-turbo quality 14)\n");
  fprintf(stderr,
          "     30 = low quality (average output of cjxl -d 9 / -q 20 or "
          "libjpeg-turbo quality 20)\n");
  fprintf(stderr,
          "     50 = medium quality (average output of cjxl -d 5 / -q 45 or "
          "libjpeg-turbo quality 35)\n");
  fprintf(stderr,
          "     70 = high quality (hard to notice artifacts without comparison "
          "to the original,\n");
  fprintf(stderr,
          "                        average output of cjxl -d 2.5 / -q 73 or "
          "libjpeg-turbo quality 70)\n");
  fprintf(stderr,
          "     80 = very high quality (impossible to distinguish from the "
          "original in a side-by-side comparison at 1:1,\n");
  fprintf(stderr,
          "                             average output of cjxl -d 1.5 / -q 85 "
          "or libjpeg-turbo quality 85 (4:2:2))\n");
  fprintf(stderr,
          "     85 = excellent quality (impossible to distinguish from the "
          "original in a flip test at 1:1,\n");
  fprintf(stderr,
          "                             average output of cjxl -d 1 / -q 90 or "
          "libjpeg-turbo quality 90 (4:4:4))\n");
  fprintf(stderr,
          "     90 = visually lossless (impossible to distinguish from the "
          "original in a flicker test at 1:1,\n");

  fprintf(stderr,
          "                             average output of cjxl -d 0.5 / -q 95 "
          "or libjpeg-turbo quality 95 (4:4:4)\n");
  fprintf(stderr, "     100 = mathematically lossless\n");

  return 1;
}

int main(int argc, char** argv) {
  if (argc != 3) return PrintUsage(argv);
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();

  auto io1 = jxl::make_unique<jxl::CodecInOut>(memory_manager);
  auto io2 = jxl::make_unique<jxl::CodecInOut>(memory_manager);
  jxl::CodecInOut* io[2] = {io1.get(), io2.get()};
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
      QUIT("Minimum image size is 8x8 pixels.");
    }
  }

  if (io1->xsize() != io2->xsize() || io1->ysize() != io2->ysize()) {
    QUIT("Image size mismatch.");
  }

  if (!io1->Main().HasAlpha()) {
    JXL_ASSIGN_OR_QUIT(Msssim msssim,
                       ComputeSSIMULACRA2(io1->Main(), io2->Main()),
                       "ComputeSSIMULACRA2 failed.");
    printf("%.8f\n", msssim.Score());
  } else {
    // in case of alpha transparency: blend against dark and bright backgrounds
    // and return the worst of both scores
    JXL_ASSIGN_OR_QUIT(Msssim msssim0,
                       ComputeSSIMULACRA2(io1->Main(), io2->Main(), 0.1f),
                       "ComputeSSIMULACRA2 failed.");
    JXL_ASSIGN_OR_QUIT(Msssim msssim1,
                       ComputeSSIMULACRA2(io1->Main(), io2->Main(), 0.9f),
                       "ComputeSSIMULACRA2 failed.");
    printf("%.8f\n", std::min(msssim0.Score(), msssim1.Score()));
  }
  return EXIT_SUCCESS;
}
