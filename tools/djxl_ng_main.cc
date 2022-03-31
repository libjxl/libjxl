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
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "jxl/codestream_header.h"
#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/resizable_parallel_runner_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "jxl/types.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/pnm.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"

DECLARE_bool(help);
DECLARE_bool(helpshort);

DEFINE_int64(num_reps, 1, "How many times to decompress.");

DEFINE_int64(num_threads, 0,
             // TODO(firsching): Sync with team about changed meaning of 0 -
             // was: No multithreaded workers. Is: use default number.
             "Number of worker threads (0 == use machine default).");

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

// TODO(firsching): wire this up; consider making empty string the default.
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
    "By default, if the input JPEG XL contains a recompressed JPEG file, djxl "
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

bool ReadFile(const char* filename, std::vector<uint8_t>* out) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    return false;
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return false;
  }

  long size = ftell(file);
  // Avoid invalid file or directory.
  if (size >= LONG_MAX || size < 0) {
    fclose(file);
    return false;
  }

  if (fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return false;
  }

  out->resize(size);
  size_t readsize = fread(out->data(), 1, size, file);
  if (fclose(file) != 0) {
    return false;
  }

  return readsize == static_cast<size_t>(size);
}

bool WriteFile(const char* filename, const std::vector<uint8_t>& bytes) {
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

int DecompressJxlReconstructJPEG(const std::vector<uint8_t>& compressed,
                                 std::vector<uint8_t>& jpeg_bytes,
                                 JxlDecoderPtr dec,
                                 JxlThreadParallelRunnerPtr runner) {
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
  bool can_reconstruct_jpeg = false;
  std::vector<uint8_t> jpeg_data_chunk(16384);
  jpeg_bytes.resize(0);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetInput(dec.get(), compressed.data(), compressed.size())) {
    fprintf(stderr, "Decoder failed to set input\n");
    return EXIT_FAILURE;
  }
  JxlDecoderCloseInput(dec.get());

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Failed to decode image\n");
      return EXIT_FAILURE;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      fprintf(stderr, "Error, already provided all input\n");
      return EXIT_FAILURE;
    } else if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
      can_reconstruct_jpeg = true;
      // Decoding to JPEG.
      if (JXL_DEC_SUCCESS != JxlDecoderSetJPEGBuffer(dec.get(),
                                                     jpeg_data_chunk.data(),
                                                     jpeg_data_chunk.size())) {
        fprintf(stderr, "Decoder failed to set JPEG Buffer\n");
        return EXIT_FAILURE;
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
        return EXIT_FAILURE;
      }
    } else if (status == JXL_DEC_SUCCESS) {
      // Decoding finished successfully.
      break;
    } else if (status == JXL_DEC_FULL_IMAGE) {
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      break;
    } else {
      fprintf(stderr, "Error: unexpected status: %d\n",
              static_cast<int>(status));
      return EXIT_FAILURE;
    }
  }
  if (!can_reconstruct_jpeg) return EXIT_FAILURE;
  size_t used_jpeg_output =
      jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec.get());
  jpeg_bytes.insert(jpeg_bytes.end(), jpeg_data_chunk.data(),
                    jpeg_data_chunk.data() + used_jpeg_output);
  return EXIT_SUCCESS;
}

int DecompressJxlToPackedPixelFile(const std::vector<uint8_t>& compressed,
                                   jxl::extras::PackedPixelFile& ppf,
                                   JxlPixelFormat& format, JxlDecoderPtr dec,
                                   JxlThreadParallelRunnerPtr runner) {
  if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(),
                                                     JxlThreadParallelRunner,
                                                     runner.get())) {
    fprintf(stderr, "JxlEncoderSetParallelRunner failed\n");
    return EXIT_FAILURE;
  }
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(dec.get(),
                                JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING |
                                    JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE)) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    return EXIT_FAILURE;
  }

  // Reading compressed JPEG XL input and decoding to pixels
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetInput(dec.get(), compressed.data(), compressed.size())) {
    fprintf(stderr, "Decoder failed to set input\n");
    return EXIT_FAILURE;
  }
  // TODO(firsching): handle boxes as well (exif, iptc, jumbf and xmp).
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Failed to decode image\n");
      return EXIT_FAILURE;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      fprintf(stderr, "Error, already provided all input\n");
      return EXIT_FAILURE;
    } else if (status == JXL_DEC_BASIC_INFO) {
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &ppf.info)) {
        fprintf(stderr, "JxlDecoderGetBasicInfo failed\n");
        return EXIT_FAILURE;
      }
      // Make some modifications to the format if the decoded data requires it.
      if (ppf.info.num_color_channels != format.num_channels) {
        format.num_channels = ppf.info.num_color_channels;
      }
      if (ppf.info.bits_per_sample > 8 &&
          ppf.info.exponent_bits_per_sample == 0) {
        format.data_type = JXL_TYPE_UINT16;
      }
      // TODO(firsching): handle extra channels
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      size_t icc_size = 0;
      // TODO(firsching) handle other targets as well.
      JxlColorProfileTarget target = JXL_COLOR_PROFILE_TARGET_ORIGINAL;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderGetICCProfileSize(dec.get(), &format, target, &icc_size)) {
        fprintf(stderr, "JxlDecoderGetICCProfileSize failed\n");
      }
      if (icc_size != 0) {
        ppf.icc.resize(icc_size);
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetColorAsICCProfile(dec.get(), &format, target,
                                           ppf.icc.data(), icc_size)) {
          fprintf(stderr, "JxlDecoderGetColorAsICCProfile failed\n");
          return EXIT_FAILURE;
        }
      } else {
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetColorAsEncodedProfile(dec.get(), &format, target,
                                               &ppf.color_encoding)) {
          fprintf(stderr, "JxlDecoderGetColorAsEncodedProfile failed\n");
          return EXIT_FAILURE;
        }
      }
    } else if (status == JXL_DEC_FRAME) {
      jxl::extras::PackedFrame frame(ppf.info.xsize, ppf.info.ysize, format);
      ppf.frames.emplace_back(std::move(frame));
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size)) {
        fprintf(stderr, "JxlDecoderImageOutBufferSize failed\n");
        return EXIT_FAILURE;
      }
      if (buffer_size != ppf.frames.back().color.pixels_size) {
        fprintf(stderr, "Invalid out buffer size %" PRIuS " %" PRIuS "\n",
                buffer_size, ppf.frames.back().color.pixels_size);
        return EXIT_FAILURE;
      }

      auto callback = [](void* opaque, size_t x, size_t y, size_t num_pixels,
                         const void* pixels) {
        jxl::extras::PackedPixelFile* ppf =
            reinterpret_cast<jxl::extras::PackedPixelFile*>(opaque);
        uint8_t* pixels_buffer =
            reinterpret_cast<uint8_t*>(ppf->frames.back().color.pixels());
        size_t sample_size = ppf->frames.back().color.format.num_channels *
                             ppf->frames.back().color.BitsPerChannel(
                                 ppf->frames.back().color.format.data_type) /
                             8;
        // TODO(firsching): take color profile into account and transform if
        // needed here.
        memcpy(pixels_buffer +
                   (ppf->frames.back().color.stride * y + sample_size * x),
               pixels, num_pixels * sample_size);
      };
      if (JXL_DEC_SUCCESS !=
          JxlDecoderSetImageOutCallback(dec.get(), &format, callback, &ppf)) {
        fprintf(stderr, "JxlDecoderSetImageOutCallback failed\n");
        return EXIT_FAILURE;
      }
    } else if (status == JXL_DEC_SUCCESS) {
      // Decoding finished successfully.
      break;
    } else if (status == JXL_DEC_FULL_IMAGE) {
    } else {
      fprintf(stderr, "Error: unexpected status: %d\n",
              static_cast<int>(status));
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
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

  std::vector<uint8_t> compressed;
  // Reading compressed JPEG XL input
  if (!ReadFile(filename_in, &compressed)) {
    fprintf(stderr, "couldn't load %s\n", filename_in);
    return EXIT_FAILURE;
  }

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
  std::vector<uint8_t> bytes;
  if (codec == jxl::extras::Codec::kJPG
#if JPEGXL_ENABLE_JPEG
      && !FLAGS_pixels_to_jpeg
#endif
  ) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < num_reps; ++i) {
      if (DecompressJxlReconstructJPEG(compressed, bytes, std::move(dec),
                                       std::move(runner)) != 0) {
        return EXIT_FAILURE;
      }
    }
    if (WriteFile(filename_out, bytes)) {
      return EXIT_FAILURE;
    };
    // TODO(firsching): handle non-reconstruct JPEG
  } else if (codec == jxl::extras::Codec::kPNM) {
    JxlDataType datatype = JXL_TYPE_UINT8;
    uint32_t num_channels = 3;
    if (std::string(extension) == ".pfm") {
      datatype = JXL_TYPE_FLOAT;
    }
    if (std::string(extension) == ".pgm") {
      num_channels = 1;
    }

    JxlPixelFormat format = {num_channels, datatype, JXL_NATIVE_ENDIAN, 0};
    jxl::extras::PackedPixelFile ppf;
    if (DecompressJxlToPackedPixelFile(compressed, ppf, format, std::move(dec),
                                       std::move(runner)) != 0) {
      return EXIT_FAILURE;
    }
    if (ppf.info.exponent_bits_per_sample != 0) {
      if (num_channels == 1 && ppf.info.num_color_channels == 3) {
        JXL_WARNING("For color images, the filename should end with .ppm.\n");
      }
      if (num_channels == 3 && ppf.info.num_color_channels == 1) {
        JXL_WARNING(
            "For grayscale images, the filename should end with .pgm.\n");
      }
      if (ppf.info.bits_per_sample > 16) {
        JXL_WARNING("PPM only supports up to 16 bits per sample");
      }
    }
    const int digits = 1 + static_cast<int>(std::log10(std::max(
                               1, static_cast<int>(ppf.frames.size() - 1))));
    std::vector<char> output_filename;
    output_filename.resize(base.size() + 1 + digits + strlen(extension) + 1);
    for (size_t i = 0; i < ppf.frames.size(); i++) {
      JXL_RETURN_IF_ERROR(jxl::extras::EncodeImagePNM(
          ppf, ppf.frames[i].color.BitsPerChannel(format.data_type), nullptr, i,
          &bytes));
      snprintf(output_filename.data(), output_filename.size(), "%s-%0*zu%s",
               base.c_str(), digits, i, extension);
      if (!WriteFile(
              ppf.frames.size() > 1 ? output_filename.data() : filename_out,
              bytes)) {
        return EXIT_FAILURE;
      }
    }
  } else {
    // TODO(firsching): handle other formats
  }
  return EXIT_SUCCESS;
}
