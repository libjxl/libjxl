// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>

#include <cstdio>
#include <cstdlib>

#include "lib/extras/codec.h"
#include "lib/extras/dec/color_description.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_encoding_internal.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/thread_pool_internal.h"

int main(int argc, const char** argv) {
  jpegxl::tools::ThreadPoolInternal pool;

  jpegxl::tools::CommandLineParser parser;
  const char* input_filename = nullptr;
  auto input_filename_option = parser.AddPositionalOption(
      "INPUT", true,
      "input ICC profile, image file containing ICC profile or tags, or "
      "description string\n"
      "    Description string syntax: "
      "ColorModel_WhitePoint_Primaries_RenderingIntent_TransferFunction\n"
      "              {RGB,Gra,XYB}_{D65,EER,DCI}_{SRG,202,DCI}_"
      "{Per,Rel,Sat,Abs}_{SRG,Lin,709,PeQ,HLG}\n",
      &input_filename, 0);
  const char* output_filename = nullptr;
  auto output_filename_option = parser.AddPositionalOption(
      "OUTPUT.icc", true, "output ICC profile filename", &output_filename, 0);

  if (!parser.Parse(argc, argv)) {
    fprintf(stderr, "See -h for help.\n");
    return EXIT_FAILURE;
  }

  if (parser.HelpFlagPassed()) {
    parser.PrintHelp();
    return EXIT_SUCCESS;
  }

  if (!parser.GetOption(input_filename_option)->matched()) {
    fprintf(stderr, "Missing input filename/string.\nSee -h for help.\n");
    return EXIT_FAILURE;
  }
  if (!parser.GetOption(output_filename_option)->matched()) {
    fprintf(stderr, "Missing output filename.\nSee -h for help.\n");
    return EXIT_FAILURE;
  }

  jxl::CodecInOut io{jpegxl::tools::NoMemoryManager()};
  std::vector<uint8_t> encoded;
  JxlColorEncoding c_descr;
  if (jpegxl::tools::ReadFile(input_filename, &encoded)) {
    bool icc_signature = true;
    if (encoded.size() < 128) {
      icc_signature = false;
    } else {
      if (encoded[36] != 'a') icc_signature = false;
      if (encoded[37] != 'c') icc_signature = false;
      if (encoded[38] != 's') icc_signature = false;
      if (encoded[39] != 'p') icc_signature = false;
    }

    if (!icc_signature || !io.metadata.m.color_encoding.SetICC(
                              jxl::IccBytes(encoded), JxlGetDefaultCms())) {
      JPEGXL_TOOLS_CHECK(
          jxl::SetFromBytes(jxl::Bytes(encoded), {}, &io, pool.get()));
    }
  } else if (jxl::ParseDescription(input_filename, &c_descr)) {
    JPEGXL_TOOLS_CHECK(io.metadata.m.color_encoding.FromExternal(c_descr));
  }

  jxl::ColorEncoding c_out = io.metadata.m.color_encoding;
  JPEGXL_TOOLS_CHECK(c_out.CreateICC());
  JPEGXL_TOOLS_CHECK(jpegxl::tools::WriteFile(output_filename, c_out.ICC()));
  return EXIT_SUCCESS;
}
