// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "jxl/decode.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "jxl/types.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/dec/jxl.h"
#include "lib/extras/enc/encode.h"
#include "lib/extras/enc/pnm.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/printf_macros.h"
#include "tools/cmdline.h"
#include "tools/codec_config.h"
#include "tools/file_io.h"
#include "tools/speed_stats.h"

namespace jpegxl {
namespace tools {

struct DecompressArgs {
  DecompressArgs() = default;

  void AddCommandLineOptions(CommandLineParser* cmdline) {
    cmdline->AddPositionalOption("INPUT", /* required = */ true,
                                 "The compressed input file.", &file_in);

    cmdline->AddPositionalOption("OUTPUT", /* required = */ true,
                                 "The output can be (A)PNG with ICC, JPG, or "
                                 "PPM/PFM.",
                                 &file_out);

    cmdline->AddOptionFlag('V', "version", "Print version number and exit.",
                           &version, &SetBooleanTrue);

    cmdline->AddOptionValue('\0', "num_reps", "N",
                            "Sets the number of times to decompress the image. "
                            "Used for benchmarking, the default is 1.",
                            &num_reps, &ParseUnsigned);

    cmdline->AddOptionFlag('\0', "disable_output",
                           "No output file will be written (for benchmarking)",
                           &disable_output, &SetBooleanTrue);

    cmdline->AddOptionValue('\0', "num_threads", "N",
                            "Number of worker threads (-1 == use machine "
                            "default, 0 == do not use multithreading).",
                            &num_threads, &ParseSigned);

    opt_bits_per_sample_id = cmdline->AddOptionValue(
        '\0', "bits_per_sample", "N",
        "Sets the output bit depth. The 0 value (default for PNM output) "
        "means the original (input) bit depth. The -1 value (default for "
        "other codecs) means the full bit depth of the output pixel "
        "format.",
        &bits_per_sample, &ParseSigned);

    cmdline->AddOptionValue('\0', "display_nits", "N",
                            "If set to a non-zero value, tone maps the image "
                            "the given peak display luminance.",
                            &display_nits, &ParseDouble);

    cmdline->AddOptionValue('\0', "color_space", "COLORSPACE_DESC",
                            "Sets the output color space of the image. This "
                            "flag has no effect if the image is not XYB "
                            "encoded.",
                            &color_space, &ParseString);

    cmdline->AddOptionValue('s', "downsampling", "N",
                            "If set and the input JXL stream is progressive "
                            "and contains hints for target downsampling "
                            "ratios, the decoder will skip any progressive "
                            "passes that are not needed to produce a partially "
                            "decoded image intended for this downsampling "
                            "ratio.",
                            &downsampling, &ParseUint32);

    cmdline->AddOptionFlag('\0', "allow_partial_files",
                           "Allow decoding of truncated files.",
                           &allow_partial_files, &SetBooleanTrue);

#if JPEGXL_ENABLE_JPEG
    cmdline->AddOptionFlag(
        'j', "pixels_to_jpeg",
        "By default, if the input JPEG XL contains a recompressed JPEG file, "
        "djxl reconstructs the exact original JPEG file. This flag causes the "
        "decoder to instead decode the image to pixels and encode a new "
        "(lossy) JPEG. The output file if provided must be a .jpg or .jpeg "
        "file.",
        &pixels_to_jpeg, &SetBooleanTrue);

    opt_jpeg_quality_id = cmdline->AddOptionValue(
        'q', "jpeg_quality", "N",
        "Sets the JPEG output quality, default is 95. Setting an output "
        "quality implies --pixels_to_jpeg.",
        &jpeg_quality, &ParseUnsigned);
#endif

#if JPEGXL_ENABLE_SJPEG
    cmdline->AddOptionFlag('\0', "use_sjpeg",
                           "Use sjpeg instead of libjpeg for JPEG output.",
                           &use_sjpeg, &SetBooleanTrue);
#endif

    cmdline->AddOptionFlag('\0', "norender_spotcolors",
                           "Disables rendering spot colors.",
                           &render_spotcolors, &SetBooleanFalse);

    cmdline->AddOptionValue('\0', "preview_out", "FILENAME",
                            "If specified, writes the preview image to this "
                            "file.",
                            &preview_out, &ParseString);

    cmdline->AddOptionValue(
        '\0', "icc_out", "FILENAME",
        "If specified, writes the ICC profile of the decoded image to "
        "this file.",
        &icc_out, &ParseString);

    cmdline->AddOptionValue(
        '\0', "orig_icc_out", "FILENAME",
        "If specified, writes the ICC profile of the original image to "
        "this file. This can be different from the ICC profile of the "
        "decoded image if --color_space was specified, or if the image "
        "was XYB encoded and the color conversion to the original "
        "profile was not supported by the decoder.",
        &orig_icc_out, &ParseString);

    cmdline->AddOptionValue(
        '\0', "metadata_out", "FILENAME",
        "If specified, writes decoded metadata info to this file in "
        "JSON format. Used by the conformance test script",
        &metadata_out, &ParseString);

    cmdline->AddOptionFlag('\0', "print_read_bytes",
                           "Print total number of decoded bytes.",
                           &print_read_bytes, &SetBooleanTrue);

    cmdline->AddOptionFlag('\0', "quiet", "Silence output (except for errors).",
                           &quiet, &SetBooleanTrue);
  }

  // Validate the passed arguments, checking whether all passed options are
  // compatible. Returns whether the validation was successful.
  bool ValidateArgs(const CommandLineParser& cmdline) {
    if (file_in == nullptr) {
      fprintf(stderr, "Missing INPUT filename.\n");
      return false;
    }
    if (num_threads < -1) {
      fprintf(
          stderr,
          "Invalid flag value for --num_threads: must be -1, 0 or postive.\n");
      return false;
    }
    return true;
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  bool version = false;
  size_t num_reps = 1;
  bool disable_output = false;
  int32_t num_threads = -1;
  int bits_per_sample = -1;
  double display_nits = 0.0;
  std::string color_space;
  uint32_t downsampling = 0;
  bool allow_partial_files = false;
  bool pixels_to_jpeg = false;
  size_t jpeg_quality = 95;
  bool use_sjpeg = false;
  bool render_spotcolors = true;
  std::string preview_out;
  std::string icc_out;
  std::string orig_icc_out;
  std::string metadata_out;
  bool print_read_bytes = false;
  bool quiet = false;
  // References (ids) of specific options to check if they were matched.
  CommandLineParser::OptionId opt_bits_per_sample_id = -1;
  CommandLineParser::OptionId opt_jpeg_quality_id = -1;
};

}  // namespace tools
}  // namespace jpegxl

namespace {

bool WriteOptionalOutput(const std::string& filename,
                         const std::vector<uint8_t>& bytes) {
  if (filename.empty() || bytes.empty()) {
    return true;
  }
  return jpegxl::tools::WriteFile(filename.data(), bytes);
}

std::string Filename(const std::string& base, const std::string& extension,
                     int layer_index, int frame_index, int num_layers,
                     int num_frames) {
  auto digits = [](int n) { return 1 + static_cast<int>(std::log10(n)); };
  std::string out = base;
  if (num_frames > 1) {
    std::vector<char> buf(2 + digits(num_frames));
    snprintf(buf.data(), buf.size(), "-%0*d", digits(num_frames), frame_index);
    out.append(buf.data());
  }
  if (num_layers > 1) {
    std::vector<char> buf(4 + digits(num_layers));
    snprintf(buf.data(), buf.size(), "-ec%0*d", digits(num_layers),
             layer_index);
    out.append(buf.data());
  }
  if (extension == ".ppm" && layer_index > 0) {
    out.append(".pgm");
  } else {
    out.append(extension);
  }
  return out;
}

bool DecompressJxlReconstructJPEG(const jpegxl::tools::DecompressArgs& args,
                                  const std::vector<uint8_t>& compressed,
                                  void* runner,
                                  std::vector<uint8_t>* jpeg_bytes,
                                  jpegxl::tools::SpeedStats* stats) {
  const double t0 = jxl::Now();
  jxl::extras::PackedPixelFile ppf;  // for JxlBasicInfo
  jxl::extras::JXLDecompressParams dparams;
  dparams.allow_partial_input = args.allow_partial_files;
  dparams.runner = JxlThreadParallelRunner;
  dparams.runner_opaque = runner;
  if (!jxl::extras::DecodeImageJXL(compressed.data(), compressed.size(),
                                   dparams, nullptr, &ppf, jpeg_bytes)) {
    return false;
  }
  const double t1 = jxl::Now();
  if (stats) {
    stats->NotifyElapsed(t1 - t0);
    stats->SetImageSize(ppf.info.xsize, ppf.info.ysize);
    stats->SetFileSize(jpeg_bytes->size());
  }
  return true;
}

bool DecompressJxlToPackedPixelFile(
    const jpegxl::tools::DecompressArgs& args,
    const std::vector<uint8_t>& compressed,
    const std::vector<JxlPixelFormat>& accepted_formats, void* runner,
    jxl::extras::PackedPixelFile* ppf, size_t* decoded_bytes,
    jpegxl::tools::SpeedStats* stats) {
  jxl::extras::JXLDecompressParams dparams;
  dparams.max_downsampling = args.downsampling;
  dparams.accepted_formats = accepted_formats;
  dparams.display_nits = args.display_nits;
  dparams.color_space = args.color_space;
  dparams.render_spotcolors = args.render_spotcolors;
  dparams.runner = JxlThreadParallelRunner;
  dparams.runner_opaque = runner;
  dparams.allow_partial_input = args.allow_partial_files;
  if (args.bits_per_sample == 0) {
    dparams.output_bitdepth.type = JXL_BIT_DEPTH_FROM_CODESTREAM;
  } else if (args.bits_per_sample > 0) {
    dparams.output_bitdepth.type = JXL_BIT_DEPTH_CUSTOM;
    dparams.output_bitdepth.bits_per_sample = args.bits_per_sample;
  }
  const double t0 = jxl::Now();
  if (!jxl::extras::DecodeImageJXL(compressed.data(), compressed.size(),
                                   dparams, decoded_bytes, ppf)) {
    return false;
  }
  const double t1 = jxl::Now();
  if (stats) {
    stats->NotifyElapsed(t1 - t0);
    stats->SetImageSize(ppf->info.xsize, ppf->info.ysize);
  }
  return true;
}

}  // namespace

int main(int argc, const char* argv[]) {
  std::string version = jpegxl::tools::CodecConfigString(JxlDecoderVersion());
  jpegxl::tools::DecompressArgs args;
  jpegxl::tools::CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, argv)) {
    // Parse already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (args.version) {
    fprintf(stdout, "djxl %s\n", version.c_str());
    fprintf(stdout, "Copyright (c) the JPEG XL Project\n");
    return EXIT_SUCCESS;
  }
  if (!args.quiet) {
    fprintf(stderr, "JPEG XL decoder %s\n", version.c_str());
  }

  if (cmdline.HelpFlagPassed()) {
    cmdline.PrintHelp();
    return EXIT_SUCCESS;
  }

  if (!args.ValidateArgs(cmdline)) {
    // ValidateArgs already printed the actual error cause.
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::vector<uint8_t> compressed;
  // Reading compressed JPEG XL input
  if (!jpegxl::tools::ReadFile(args.file_in, &compressed)) {
    fprintf(stderr, "couldn't load %s\n", args.file_in);
    return EXIT_FAILURE;
  }
  if (!args.quiet) {
    fprintf(stderr, "Read %" PRIuS " compressed bytes.\n", compressed.size());
  }

  if (!args.file_out && !args.disable_output) {
    std::cerr
        << "No output file specified and --disable_output flag not passed."
        << std::endl;
    return EXIT_FAILURE;
  }

  if (args.file_out && args.disable_output && !args.quiet) {
    fprintf(stderr,
            "Decoding will be performed, but the result will be discarded.\n");
  }

  std::string filename_out;
  std::string base;
  std::string extension;
  if (args.file_out && !args.disable_output) {
    filename_out = std::string(args.file_out);
    size_t pos = filename_out.find_last_of('.');
    if (pos < filename_out.size()) {
      base = filename_out.substr(0, pos);
      extension = filename_out.substr(pos);
    } else {
      base = filename_out;
    }
  }
  const jxl::extras::Codec codec = jxl::extras::CodecFromExtension(extension);
  if (codec == jxl::extras::Codec::kEXR) {
    std::string force_colorspace = "RGB_D65_SRG_Rel_Lin";
    if (!args.color_space.empty() && args.color_space != force_colorspace) {
      fprintf(stderr, "Warning: colorspace ignored for EXR output\n");
    }
    args.color_space = force_colorspace;
  }
  if (codec == jxl::extras::Codec::kPNM && extension != ".pfm" &&
      !cmdline.GetOption(args.opt_jpeg_quality_id)->matched()) {
    args.bits_per_sample = 0;
  }

  jpegxl::tools::SpeedStats stats;
  size_t num_worker_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
  {
    int64_t flag_num_worker_threads = args.num_threads;
    if (flag_num_worker_threads > -1) {
      num_worker_threads = flag_num_worker_threads;
    }
  }
  auto runner = JxlThreadParallelRunnerMake(
      /*memory_manager=*/nullptr, num_worker_threads);

  bool decode_to_pixels = (codec != jxl::extras::Codec::kJPG);
#if JPEGXL_ENABLE_JPEG
  if (args.pixels_to_jpeg ||
      cmdline.GetOption(args.opt_jpeg_quality_id)->matched()) {
    decode_to_pixels = true;
  }
#endif

  size_t num_reps = args.num_reps;
  if (!decode_to_pixels) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < num_reps; ++i) {
      if (!DecompressJxlReconstructJPEG(args, compressed, runner.get(), &bytes,
                                        &stats)) {
        if (bytes.empty()) {
          if (!args.quiet) {
            fprintf(stderr,
                    "Warning: could not decode losslessly to JPEG. Retrying "
                    "with --pixels_to_jpeg...\n");
          }
          decode_to_pixels = true;
          break;
        }
        return EXIT_FAILURE;
      }
    }
    if (!bytes.empty()) {
      if (!args.quiet) fprintf(stderr, "Reconstructed to JPEG.\n");
      if (!filename_out.empty() &&
          !jpegxl::tools::WriteFile(filename_out.c_str(), bytes)) {
        return EXIT_FAILURE;
      }
    }
  }
  if (decode_to_pixels) {
    std::vector<JxlPixelFormat> accepted_formats;
    std::unique_ptr<jxl::extras::Encoder> encoder;
    if (!filename_out.empty()) {
      encoder = jxl::extras::Encoder::FromExtension(extension);
      if (encoder == nullptr) {
        fprintf(stderr, "can't decode to the file extension '%s'\n",
                extension.c_str());
        return EXIT_FAILURE;
      }
      accepted_formats = encoder->AcceptedFormats();
    }
    jxl::extras::PackedPixelFile ppf;
    size_t decoded_bytes = 0;
    for (size_t i = 0; i < num_reps; ++i) {
      if (!DecompressJxlToPackedPixelFile(args, compressed, accepted_formats,
                                          runner.get(), &ppf, &decoded_bytes,
                                          &stats)) {
        fprintf(stderr, "DecompressJxlToPackedPixelFile failed\n");
        return EXIT_FAILURE;
      }
    }
    if (!args.quiet) fprintf(stderr, "Decoded to pixels.\n");
    if (args.print_read_bytes) {
      fprintf(stderr, "Decoded bytes: %" PRIuS "\n", decoded_bytes);
    }
#if JPEGXL_ENABLE_JPEG
    if (encoder) {
      std::ostringstream os;
      os << args.jpeg_quality;
      encoder->SetOption("q", os.str());
    }
#endif
#if JPEGXL_ENABLE_SJPEG
    if (encoder && args.use_sjpeg) {
      encoder->SetOption("jpeg_encoder", "sjpeg");
    }
#endif
    jxl::extras::EncodedImage encoded_image;
    if (encoder) {
      if (!encoder->Encode(ppf, &encoded_image)) {
        fprintf(stderr, "Encode failed\n");
        return EXIT_FAILURE;
      }
    }
    size_t nlayers = 1 + encoded_image.extra_channel_bitstreams.size();
    size_t nframes = encoded_image.bitstreams.size();
    for (size_t i = 0; i < nlayers; ++i) {
      for (size_t j = 0; j < nframes; ++j) {
        const std::vector<uint8_t>& bitstream =
            (i == 0 ? encoded_image.bitstreams[j]
                    : encoded_image.extra_channel_bitstreams[i - 1][j]);
        std::string fn = Filename(base, extension, i, j, nlayers, nframes);
        if (!jpegxl::tools::WriteFile(fn.c_str(), bitstream)) {
          return EXIT_FAILURE;
        }
      }
    }
    if (!WriteOptionalOutput(args.preview_out,
                             encoded_image.preview_bitstream) ||
        !WriteOptionalOutput(args.icc_out, ppf.icc) ||
        !WriteOptionalOutput(args.orig_icc_out, ppf.orig_icc) ||
        !WriteOptionalOutput(args.metadata_out, encoded_image.metadata)) {
      return EXIT_FAILURE;
    }
  }
  if (!args.quiet) {
    stats.Print(num_worker_threads);
  }
  return EXIT_SUCCESS;
}
