// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <string>
#include <vector>

#include "lib/include/jxl/decode.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"

namespace jpegxl {
namespace tools {
namespace {

struct Args {
  void AddCommandLineOptions(CommandLineParser* cmdline) {
    cmdline->AddPositionalOption("INPUT", /* required = */ true,
                                 "The JPEG XL input file.", &file_in);

    cmdline->AddPositionalOption("OUTPUT", /* required = */ true,
                                 "The JPEG XL output file.", &file_out);
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;
};

}  // namespace

int JxlTranMain(int argc, const char* argv[]) {
  Args args;
  CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, const_cast<const char**>(argv))) {
    // Parse already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information.\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (cmdline.HelpFlagPassed() || !args.file_in) {
    cmdline.PrintHelp();
    return EXIT_SUCCESS;
  }

  if (!args.file_out) {
    fprintf(stderr, "No output file specified.\n");
    return EXIT_FAILURE;
  }

  std::vector<uint8_t> jxl_bytes;
  if (!ReadFile(args.file_in, &jxl_bytes)) {
    fprintf(stderr, "Failed to read input image %s\n", args.file_in);
    return EXIT_FAILURE;
  }

  JxlSignature signature;
  signature = JxlSignatureCheck(jxl_bytes.data(), jxl_bytes.size());
  if (signature != JXL_SIG_CODESTREAM && signature != JXL_SIG_CONTAINER) {
    fprintf(stderr, "Input file is not a JPEG XL file.\n");
    return EXIT_FAILURE;
  }

  std::string filename_out = std::string(args.file_out);

  if (!WriteFile(filename_out, jxl_bytes)) {
    fprintf(stderr, "Failed to write output file %s\n", filename_out.c_str());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

}  // namespace tools
}  // namespace jpegxl

int main(int argc, const char* argv[]) {
  return jpegxl::tools::JxlTranMain(argc, argv);
}

