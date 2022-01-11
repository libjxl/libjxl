// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "gflags/gflags.h"
#include "jxl/codestream_header.h"
#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/resizable_parallel_runner_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "lib/extras/dec/decode.h"
#include "lib/jxl/base/printf_macros.h"

DECLARE_bool(help);
DECLARE_bool(helpshort);

DEFINE_int64(num_reps, 1,  // TODO(firsching): Clarify meaning of this
                           // docstring. Is this simply for benchmarking?
             "How many times to decompress.");

DEFINE_int64(num_threads, 0,
             // TODO(firsching): Sync with team about changed meaning of 0 -
             // was: No multithreaded workers. Is: use default number.
             "Number of worker threads (0 == use machine default).");

// TODO(firsching): wire this up.
DEFINE_bool(print_profile, false, "Print timing information before exiting.");

// TODO(firsching): wire this up.
DEFINE_int32(bits_per_sample, 0, "0 = original (input) bit depth");

// TODO(firsching): wire this up.
DEFINE_bool(
    tone_map, true,
    "tone map the image to the luminance range indicated by --display_nits "
    "instead of performing a naive 0-1 -> 0-1 conversion");

// TODO(firsching): wire this up.
DEFINE_string(display_nits, "0.f-255.",
              "luminance range of the display to which to "
              "tone-map; the lower bound can be omitted");

// TODO(firsching): wire this up.
DEFINE_double(preserve_saturation, 0.1,
              "with --tone_map, how much to favor saturation over luminance");

// TODO(firsching): wire this up.
DEFINE_string(color_space, "RGB_D65_SRG_Rel_Lin",
              "defaults to original (input) color space");

// TODO(firsching): wire this up.
DEFINE_uint32(downsampling, 0,
              "maximum permissible downsampling factor (values "
              "greater than 16 will return the LQIP if available");

// TODO(firsching): wire this up.
DEFINE_bool(allow_partial_files, false, "allow decoding of truncated files");

// TODO(firsching): wire this up.
DEFINE_bool(allow_more_progressive_steps, false,
            "allow decoding more progressive steps in truncated "
            "files. No effect without --allow_partial_files");

#if JPEGXL_ENABLE_JPEG
// TODO(firsching): wire this up.
DEFINE_bool(
    pixels_to_jpeg, false,
    "By default, if the input JPEG XL contains a recompressed JPEG file, "
    "djxl "
    "reconstructs the exact original JPEG file. This flag causes the decoder "
    "to instead decode the image to pixels and encode a new (lossy) JPEG. "
    "The output file if provided must be a .jpg or .jpeg file.");

// TODO(firsching): wire this up.
DEFINE_uint32(jpeg_quality, 95,
              "JPEG output quality. Setting an output quality "
              "implies --pixels_to_jpeg.");
#endif

#if JPEGXL_ENABLE_SJPEG
// TODO(firsching): wire this up.
DEFINE_bool(use_sjpeg, false, "use sjpeg instead of libjpeg for JPEG output");
#endif

// TODO(firsching): wire this up.
DEFINE_bool(print_read_bytes, false, "print total number of decoded bytes");

// TODO(firsching): wire this up.
DEFINE_bool(quiet, false, "silence output (except for errors)");

bool WriteFile(const char* filename, const std::vector<uint8_t> bytes) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr,
            "Could not open %s for writing\n"
            "Error: %s",
            filename, strerror(errno));
    return false;
  }
  if (fwrite(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
    fprintf(stderr,
            "Could not write to file\n"
            "Error: %s",
            strerror(errno));
    return false;
  }
  if (fclose(file) != 0) {
    fprintf(stderr,
            "Could not close file\n"
            "Error: %s",
            strerror(errno));
    return false;
  }
  return true;
}

int DecompressJxlReconstructJPEG(const char* filename,
                                 std::vector<uint8_t>& jpeg_bytes,
                                 JxlDecoderPtr dec,
                                 JxlThreadParallelRunnerPtr runner,
                                 JxlBasicInfo* info) {
  FILE* file_in = fopen(filename, "rb");
  if (!file_in) {
    fprintf(stderr,
            "Could not open %s for reading\n"
            "Error: %s",
            filename, strerror(errno));
    return EXIT_FAILURE;
  }
  if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(),
                                                     JxlThreadParallelRunner,
                                                     runner.get())) {
    fprintf(stderr, "JxlEncoderSetParallelRunner failed\n");
    return EXIT_FAILURE;
  }

  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(
          dec.get(), JXL_DEC_JPEG_RECONSTRUCTION | JXL_DEC_FULL_IMAGE)) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    return EXIT_FAILURE;
  }

  // In how large chunks to read from the file.
  const constexpr size_t kInputChunkSize = 65536;
  const constexpr size_t kOutputChunkSize = 16384;
  // Reading compressed JPEG XL input and decoding to pixels
  std::vector<uint8_t> compressed;
  size_t data_size = 0;
  bool dec_successful = false;
  bool can_reconstruct_jpeg = false;
  std::vector<uint8_t> jpeg_data_chunk(kOutputChunkSize);
  jpeg_bytes.resize(0);
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Failed to decode image\n");
      break;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      size_t remaining = JxlDecoderReleaseInput(dec.get());
      if (remaining != 0) {
        compressed.erase(compressed.begin(),
                         compressed.begin() + data_size - remaining);
      }
      compressed.resize(remaining + kInputChunkSize);
      size_t read_size =
          fread(compressed.data() + remaining, 1, kInputChunkSize, file_in);
      if (read_size == 0 && feof(file_in)) {
        fprintf(stderr, "Unexpected EOF\n");
        break;
      }
      data_size = remaining + read_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderSetInput(dec.get(), compressed.data(), data_size)) {
        fprintf(stderr, "Decoder failed to set input\n");
        break;
      };
      if (feof(file_in)) {
        JxlDecoderCloseInput(dec.get());
      }
    } else if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
      can_reconstruct_jpeg = true;
      // Decoding to JPEG.
      if (JXL_DEC_SUCCESS != JxlDecoderSetJPEGBuffer(dec.get(),
                                                     jpeg_data_chunk.data(),
                                                     jpeg_data_chunk.size())) {
        fprintf(stderr, "Decoder failed to set JPEG Buffer\n");
        break;
      }
    } else if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
      // Decoded a chunk to JPEG.
      size_t used_jpeg_output =
          jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec.get());
      jpeg_bytes.insert(jpeg_bytes.end(), jpeg_data_chunk.data(),
                        jpeg_data_chunk.data() + used_jpeg_output);
      if (used_jpeg_output == 0) {
        // Chunk is too small.
        jpeg_data_chunk.resize(jpeg_data_chunk.size() * 2);
      }
      if (JXL_DEC_SUCCESS != JxlDecoderSetJPEGBuffer(dec.get(),
                                                     jpeg_data_chunk.data(),
                                                     jpeg_data_chunk.size())) {
        fprintf(stderr, "Decoder failed to set JPEG Buffer\n");
        break;
      };
    } else if (status == JXL_DEC_SUCCESS) {
      // Finished all processing.
      dec_successful = true;
      break;
    } else if (status == JXL_DEC_FULL_IMAGE) {
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      break;
    } else {
      fprintf(stderr, "Error: unexpected status: %d\n",
              static_cast<int>(status));
      break;
    }
  }
  if (fclose(file_in) != 0) return EXIT_FAILURE;
  if (!can_reconstruct_jpeg) return EXIT_FAILURE;
  size_t used_jpeg_output =
      jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec.get());
  jpeg_bytes.insert(jpeg_bytes.end(), jpeg_data_chunk.data(),
                    jpeg_data_chunk.data() + used_jpeg_output);
  return (dec_successful) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv) {
  std::cerr << "Warning: This is work in progress, consider using djxl "
               "instead!\n";

  gflags::SetUsageMessage("JPEG XL decoder");
  uint32_t version = JxlDecoderVersion();
  gflags::SetVersionString(std::to_string(version / 1000000) + "." +
                           std::to_string((version / 1000) % 1000) + "." +
                           std::to_string(version % 1000));
  // TODO(firsching): rethink --help handling
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, /*remove_flags=*/true);
  if (FLAGS_help) {
    FLAGS_help = false;
    FLAGS_helpshort = true;
  }
  gflags::HandleCommandLineHelpFlags();

  if (argc != 3) {
    FLAGS_help = false;
    FLAGS_helpshort = true;
    gflags::HandleCommandLineHelpFlags();
    return EXIT_FAILURE;
  }
  const char* filename_in = argv[1];
  const char* filename_out = argv[2];
  size_t num_reps = FLAGS_num_reps;

  const char* extension = strrchr(filename_out, '.');
  std::string base = extension == nullptr
                         ? std::string(filename_out)
                         : std::string(filename_out, extension - filename_out);
  if (extension == nullptr) extension = "";
  const jxl::extras::Codec codec = jxl::extras::CodecFromExtension(extension);

  size_t num_worker_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
  {
    int64_t flag_num_worker_threads = FLAGS_num_threads;
    if (flag_num_worker_threads != 0) {
      num_worker_threads = flag_num_worker_threads;
    }
  }
  auto dec = JxlDecoderMake(/*memory_manager=*/nullptr);
  auto runner = JxlThreadParallelRunnerMake(
      /*memory_manager=*/nullptr, num_worker_threads);
  JxlBasicInfo info;
  if (codec == jxl::extras::Codec::kJPG && !FLAGS_pixels_to_jpeg) {
    std::vector<uint8_t> jpeg_bytes;
    for (size_t i = 0; i < num_reps; ++i) {
      if (DecompressJxlReconstructJPEG(filename_in, jpeg_bytes, std::move(dec),
                                       std::move(runner), &info) != 0) {
        return EXIT_FAILURE;
      }
    }
    if (WriteFile(filename_out, jpeg_bytes)) {
      return EXIT_FAILURE;
    };
    // TODO(firsching): handle non-reconstruct JPEG
  } else {
    // TODO(firsching): handle other formats
  }

  return EXIT_SUCCESS;
}
