// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstdio>
#include <cstdlib>

#include "lib/extras/codec.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/jxl/base/status.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/thread_pool_internal.h"

using ::jxl::Image3F;

#define QUIT(M) JPEGXL_TOOLS_ABORT(M)

int main(int argc, const char** argv) {
  jpegxl::tools::ThreadPoolInternal pool;

  jpegxl::tools::CommandLineParser parser;
  size_t N = 64;
  parser.AddOptionValue('N', "lut_size", "N", "linear size of the LUT", &N,
                        &jpegxl::tools::ParseUnsigned, 0);
  const char* output_filename = nullptr;
  auto output_filename_option = parser.AddPositionalOption(
      "output", true, "output LUT", &output_filename, 0);

  if (!parser.Parse(argc, argv)) {
    fprintf(stderr, "See -h for help.\n");
    return EXIT_FAILURE;
  }

  if (parser.HelpFlagPassed()) {
    parser.PrintHelp();
    return EXIT_SUCCESS;
  }

  if (!parser.GetOption(output_filename_option)->matched()) {
    fprintf(stderr, "Missing output filename.\nSee -h for help.\n");
    return EXIT_FAILURE;
  }

  JXL_ASSIGN_OR_RETURN(
      Image3F image,
      Image3F::Create(jpegxl::tools::NoMemoryManager(), N * N, N));
  const float scale = 1.0 / (N - 1);
  const auto process_row = [&](const uint32_t y,
                               size_t /* thread */) -> jxl::Status {
    const float g = y * scale;
    float* const JXL_RESTRICT rows[3] = {
        image.PlaneRow(0, y), image.PlaneRow(1, y), image.PlaneRow(2, y)};
    for (size_t x = 0; x < N * N; ++x) {
      size_t r = x % N;
      size_t q = x / N;
      rows[0][x] = r * scale;
      rows[1][x] = g;
      rows[2][x] = q * scale;
    }
    return true;
  };
  JPEGXL_TOOLS_CHECK(jxl::RunOnPool(pool.get(), 0, N, jxl::ThreadPool::NoInit,
                                    process_row, "GenerateTemplate"));

  JxlPixelFormat format = {3, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  JXL_ASSIGN_OR_QUIT(jxl::extras::PackedPixelFile ppf,
                     jxl::extras::ConvertImage3FToPackedPixelFile(
                         image, jxl::ColorEncoding::SRGB(), format, pool.get()),
                     "ConvertImage3FToPackedPixelFile failed.");
  std::vector<uint8_t> encoded;
  JPEGXL_TOOLS_CHECK(jxl::Encode(ppf, output_filename, &encoded, pool.get()));
  JPEGXL_TOOLS_CHECK(jpegxl::tools::WriteFile(output_filename, encoded));
  return EXIT_SUCCESS;
}
