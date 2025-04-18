// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/decode.h>
#include <jxl/decode_cxx.h>

#include <string>
#include <vector>

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

    cmdline->AddHelpText("\nFile format options:", 0);

    cmdline->AddOptionFlag('\0', "extract",
                           "Extract the JPEG XL codestream"
                           " from a file in the container file format.",
                           &extract, &SetBooleanTrue);
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;

  bool extract = false;
};

JxlDecoderStatus apply_file_format_options(
    std::vector<uint8_t> const& input_bytes,
    std::shared_ptr<std::vector<uint8_t>>& output_bytes, Args const& args,
    JxlSignature signature) {
  JxlDecoderPtr dec = JxlDecoderMake(nullptr);
  if (!dec) {
    fprintf(stderr, "JxlDecoderMake failed\n");
    return JXL_DEC_ERROR;
  }

  JxlDecoderStatus status =
      JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BOX | JXL_DEC_BOX_COMPLETE);
  if (status != JXL_DEC_SUCCESS) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    return JXL_DEC_ERROR;
  }

  JxlDecoderSetInput(dec.get(), input_bytes.data(), input_bytes.size());
  JxlDecoderCloseInput(dec.get());

  if (args.extract) {
    if (signature != JXL_SIG_CONTAINER) {
      fprintf(stderr, "Input file is not a container file.\n");
      return JXL_DEC_ERROR;
    }

    static constexpr uint32_t kChunkSize = 65536;
    output_bytes = std::make_shared<std::vector<uint8_t>>();
    auto& codestream = *output_bytes;
    size_t already_written_bytes = 0;
    size_t start_of_last_codestream_box = 0;

    JxlBoxType type;

    uint32_t jxlc, jxll, jxlp;
    // these aren't hardcoded because
    // they're in native endian order
    memcpy(&jxlc, "jxlc", sizeof(jxlc));
    memcpy(&jxll, "jxll", sizeof(jxll));
    memcpy(&jxlp, "jxlp", sizeof(jxlp));

    uint32_t box = 0;
    uint8_t level = 5;

    for (;;) {
      status = JxlDecoderProcessInput(dec.get());
      if (status == JXL_DEC_ERROR) {
        fprintf(stderr, "Decoder error\n");
        return JXL_DEC_ERROR;
      } else if (status == JXL_DEC_NEED_MORE_INPUT) {
        fprintf(stderr, "Error, already provided all input\n");
        return JXL_DEC_ERROR;
      } else if (status == JXL_DEC_BOX) {
        status = JxlDecoderGetBoxType(dec.get(), type, /*decompressed=*/false);
        if (status != JXL_DEC_SUCCESS) {
          fprintf(stderr, "Error, failed to get box type\n");
          return JXL_DEC_ERROR;
        }
        memcpy(&box, type, sizeof(box));
        if (box == jxlc || box == jxlp) {
          codestream.resize(codestream.size() + kChunkSize);
          JxlDecoderSetBoxBuffer(dec.get(),
                                 codestream.data() + already_written_bytes,
                                 codestream.size() - already_written_bytes);
          start_of_last_codestream_box = already_written_bytes;
        } else if (box == jxll) {
          JxlDecoderSetBoxBuffer(dec.get(), &level, 1);
        }
      } else if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
        if (box == jxll) {
          fprintf(stderr, "jxll box too large");
          JxlDecoderReleaseBoxBuffer(dec.get());
          continue;
        }
        size_t remaining = JxlDecoderReleaseBoxBuffer(dec.get());
        already_written_bytes += kChunkSize - remaining;
        codestream.resize(codestream.size() + kChunkSize);
        JxlDecoderSetBoxBuffer(dec.get(),
                               codestream.data() + already_written_bytes,
                               codestream.size() - already_written_bytes);
      } else if (status == JXL_DEC_BOX_COMPLETE) {
        if (box == jxlc || box == jxlp) {
          size_t remaining = JxlDecoderReleaseBoxBuffer(dec.get());
          codestream.resize(codestream.size() - remaining);
          if (box == jxlp) {
            // Remove jxlp's indices from the codestream
            // TODO: Warn about malformed files, like missing or unordered jxlp
            //       or even mixed jxlp and jxlc boxes.
            codestream.erase(
                codestream.begin() + start_of_last_codestream_box,
                codestream.begin() + start_of_last_codestream_box + 4);
          }
          already_written_bytes = codestream.size();
        } else if (box == jxll) {
          JxlDecoderReleaseBoxBuffer(dec.get());
          if (level > 5) {
            fprintf(stderr,
                    "Warning: extracting raw codestream that"
                    "is greater than level 5\n");
          }
        }
      } else if (status == JXL_DEC_SUCCESS) {
        return JXL_DEC_SUCCESS;
      } else {
        fprintf(stderr, "Unknown decoder status\n");
        return JXL_DEC_ERROR;
      }
    }
  }

  return JXL_DEC_SUCCESS;
}

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

  auto jxl_bytes = std::make_shared<std::vector<uint8_t>>();
  if (!ReadFile(args.file_in, jxl_bytes.get())) {
    fprintf(stderr, "Failed to read input image %s\n", args.file_in);
    return EXIT_FAILURE;
  }

  JxlSignature signature;
  signature = JxlSignatureCheck(jxl_bytes->data(), jxl_bytes->size());
  if (signature != JXL_SIG_CODESTREAM && signature != JXL_SIG_CONTAINER) {
    fprintf(stderr, "Input file is not a JPEG XL file.\n");
    return EXIT_FAILURE;
  }

  std::string filename_out = std::string(args.file_out);

  std::shared_ptr<std::vector<uint8_t>> out_file(jxl_bytes);

  JxlDecoderStatus status =
      apply_file_format_options(*jxl_bytes, out_file, args, signature);
  if (status != JXL_DEC_SUCCESS) return EXIT_FAILURE;

  if (!WriteFile(filename_out, *out_file)) {
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
