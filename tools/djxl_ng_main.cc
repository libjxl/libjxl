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

#include "gflags/gflags.h"
#include "jxl/codestream_header.h"
#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/resizable_parallel_runner_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "jxl/types.h"
#include "lib/extras/dec/color_description.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/encode.h"
#include "lib/extras/enc/pnm.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/printf_macros.h"

DECLARE_bool(help);
DECLARE_bool(helpshort);

DEFINE_int64(num_reps, 1, "How many times to decompress.");

DEFINE_int64(num_threads, 0,
             // TODO(firsching): Sync with team about changed meaning of 0 -
             // was: No multithreaded workers. Is: use default number.
             "Number of worker threads (0 == use machine default).");

DEFINE_int32(bits_per_sample, 0, "0 = original (input) bit depth");

DEFINE_double(display_nits, 0.,
              "tone map the image to the peak display luminance given");

DEFINE_string(color_space, "",
              "Sets the output color space of the image. This flag has no "
              "effect if the image is not XYB encoded.");

DEFINE_uint32(downsampling, 0,
              "If set and the input JXL stream is progressive and contains "
              "hints for target downsampling ratios, the decoder will skip any "
              "progressive passes that are not needed to produce a partially "
              "decoded image intended for this downsampling ratio.");

DEFINE_bool(allow_partial_files, false, "allow decoding of truncated files");

#if JPEGXL_ENABLE_JPEG
DEFINE_bool(
    pixels_to_jpeg, false,
    "By default, if the input JPEG XL contains a recompressed JPEG file, djxl "
    "reconstructs the exact original JPEG file. This flag causes the decoder "
    "to instead decode the image to pixels and encode a new (lossy) JPEG. "
    "The output file if provided must be a .jpg or .jpeg file.");

DEFINE_uint32(jpeg_quality, 95,
              "JPEG output quality. Setting an output quality "
              "implies --pixels_to_jpeg.");
#endif

#if JPEGXL_ENABLE_SJPEG
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

bool DecompressJxlReconstructJPEG(const std::vector<uint8_t>& compressed,
                                  JxlDecoder* dec, void* runner,
                                  std::vector<uint8_t>* jpeg_bytes,
                                  bool* can_reconstruct_jpeg) {
  JxlDecoderReset(dec);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner)) {
    fprintf(stderr, "JxlEncoderSetParallelRunner failed\n");
    return false;
  }

  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO |
                                         JXL_DEC_JPEG_RECONSTRUCTION |
                                         JXL_DEC_FULL_IMAGE)) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    return false;
  }
  *can_reconstruct_jpeg = false;
  std::vector<uint8_t> jpeg_data_chunk(16384);
  jpeg_bytes->resize(0);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetInput(dec, compressed.data(), compressed.size())) {
    fprintf(stderr, "Decoder failed to set input\n");
    return false;
  }
  JxlDecoderCloseInput(dec);

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Failed to decode image\n");
      return false;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      fprintf(stderr, "Error, already provided all input\n");
      return false;
    } else if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
      *can_reconstruct_jpeg = true;
      // Decoding to JPEG.
      if (JXL_DEC_SUCCESS != JxlDecoderSetJPEGBuffer(dec,
                                                     jpeg_data_chunk.data(),
                                                     jpeg_data_chunk.size())) {
        fprintf(stderr, "Decoder failed to set JPEG Buffer\n");
        return false;
      }
    } else if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
      // Decoded a chunk to JPEG.
      size_t used_jpeg_output =
          jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec);
      jpeg_bytes->insert(jpeg_bytes->end(), jpeg_data_chunk.data(),
                         jpeg_data_chunk.data() + used_jpeg_output);
      if (used_jpeg_output == 0) {
        // Chunk is too small.
        jpeg_data_chunk.resize(jpeg_data_chunk.size() * 2);
      }
      if (JXL_DEC_SUCCESS != JxlDecoderSetJPEGBuffer(dec,
                                                     jpeg_data_chunk.data(),
                                                     jpeg_data_chunk.size())) {
        fprintf(stderr, "Decoder failed to set JPEG Buffer\n");
        return false;
      }
    } else if (status == JXL_DEC_BASIC_INFO) {
      if (!*can_reconstruct_jpeg) return false;
    } else if (status == JXL_DEC_SUCCESS) {
      // Decoding finished successfully.
      break;
    } else if (status == JXL_DEC_FULL_IMAGE) {
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      break;
    } else {
      fprintf(stderr, "Error: unexpected status: %d\n",
              static_cast<int>(status));
      return false;
    }
  }
  if (!*can_reconstruct_jpeg) return false;
  size_t used_jpeg_output =
      jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec);
  jpeg_bytes->insert(jpeg_bytes->end(), jpeg_data_chunk.data(),
                     jpeg_data_chunk.data() + used_jpeg_output);
  return true;
}

struct BoxProcessor {
  BoxProcessor(JxlDecoder* dec) : dec_(dec) { Reset(); }

  void InitializeOutput(std::vector<uint8_t>* out) {
    box_data_ = out;
    AddMoreOutput();
  }

  bool AddMoreOutput() {
    Flush();
    static const size_t kBoxOutputChunkSize = 1 << 16;
    box_data_->resize(box_data_->size() + kBoxOutputChunkSize);
    next_out_ = box_data_->data() + total_size_;
    avail_out_ = box_data_->size() - total_size_;
    if (JXL_DEC_SUCCESS !=
        JxlDecoderSetBoxBuffer(dec_, next_out_, avail_out_)) {
      fprintf(stderr, "JxlDecoderSetBoxBuffer failed\n");
      return false;
    }
    return true;
  }

  void FinalizeOutput() {
    if (box_data_ == nullptr) return;
    Flush();
    box_data_->resize(total_size_);
    Reset();
  }

 private:
  JxlDecoder* dec_;
  std::vector<uint8_t>* box_data_;
  uint8_t* next_out_;
  size_t avail_out_;
  size_t total_size_;

  void Reset() {
    box_data_ = nullptr;
    next_out_ = nullptr;
    avail_out_ = 0;
    total_size_ = 0;
  }
  void Flush() {
    if (box_data_ == nullptr) return;
    size_t remaining = JxlDecoderReleaseBoxBuffer(dec_);
    size_t bytes_written = avail_out_ - remaining;
    next_out_ += bytes_written;
    avail_out_ -= bytes_written;
    total_size_ += bytes_written;
  }
};

bool DecompressJxlToPackedPixelFile(
    const std::vector<uint8_t>& compressed,
    const std::vector<JxlPixelFormat>& accepted_formats, JxlDecoder* dec,
    void* runner, jxl::extras::PackedPixelFile* ppf) {
  JxlDecoderReset(dec);
  ppf->frames.clear();
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner)) {
    fprintf(stderr, "JxlEncoderSetParallelRunner failed\n");
    return false;
  }
  int events = (JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME |
                JXL_DEC_FULL_IMAGE | JXL_DEC_BOX);
  if (FLAGS_downsampling > 1) {
    events |= JXL_DEC_FRAME_PROGRESSION;
    JxlDecoderSetProgressiveDetail(dec, JxlProgressiveDetail::kLastPasses);
  }
  if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec, events)) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    return false;
  }
  JxlPixelFormat format;
  // Reading compressed JPEG XL input and decoding to pixels
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetInput(dec, compressed.data(), compressed.size())) {
    fprintf(stderr, "Decoder failed to set input\n");
    return false;
  }
  if (FLAGS_display_nits > 0 &&
      JXL_DEC_SUCCESS !=
          JxlDecoderSetDesiredIntensityTarget(dec, FLAGS_display_nits)) {
    fprintf(stderr, "Decoder failed to set desired intensity target\n");
    return false;
  }
  if (JXL_DEC_SUCCESS != JxlDecoderSetDecompressBoxes(dec, JXL_TRUE)) {
    fprintf(stderr, "JxlDecoderSetDecompressBoxes failed\n");
    return false;
  }
  bool codestream_done = false;
  BoxProcessor boxes(dec);
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Failed to decode image\n");
      return false;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      if (codestream_done) {
        break;
      }
      if (FLAGS_allow_partial_files) {
        if (JXL_DEC_SUCCESS != JxlDecoderFlushImage(dec)) {
          fprintf(stderr,
                  "Input file is truncated and there is no preview "
                  "available yet.\n");
          return false;
        }
        break;
      }
      fprintf(stderr,
              "Input file is truncated and --allow_partial_files was "
              "not used\n");
      return false;
    } else if (status == JXL_DEC_BOX) {
      boxes.FinalizeOutput();
      JxlBoxType box_type;
      if (JXL_DEC_SUCCESS != JxlDecoderGetBoxType(dec, box_type, JXL_TRUE)) {
        fprintf(stderr, "JxlDecoderGetBoxType failed\n");
        return false;
      }
      std::vector<uint8_t>* box_data = nullptr;
      if (memcmp(box_type, "Exif", 4) == 0) {
        box_data = &ppf->metadata.exif;
      } else if (memcmp(box_type, "iptc", 4) == 0) {
        box_data = &ppf->metadata.iptc;
      } else if (memcmp(box_type, "jumb", 4) == 0) {
        box_data = &ppf->metadata.jumbf;
      } else if (memcmp(box_type, "xml ", 4) == 0) {
        box_data = &ppf->metadata.xmp;
      }
      if (box_data) {
        boxes.InitializeOutput(box_data);
      }
    } else if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
      boxes.AddMoreOutput();
    } else if (status == JXL_DEC_BASIC_INFO) {
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec, &ppf->info)) {
        fprintf(stderr, "JxlDecoderGetBasicInfo failed\n");
        return false;
      }
      // Select format according to accepted formats.
      if (!jxl::extras::SelectFormat(accepted_formats, ppf->info, &format)) {
        fprintf(stderr, "SelectFormat failed\n");
        return false;
      }
      bool have_alpha = (format.num_channels == 2 || format.num_channels == 4);
      bool alpha_found = false;
      for (uint32_t i = 0; i < ppf->info.num_extra_channels; ++i) {
        JxlExtraChannelInfo eci;
        if (JXL_DEC_SUCCESS != JxlDecoderGetExtraChannelInfo(dec, i, &eci)) {
          fprintf(stderr, "JxlDecoderGetExtraChannelInfo failed\n");
          return false;
        }
        if (eci.type == JXL_CHANNEL_ALPHA && have_alpha && !alpha_found) {
          // Skip the first alpha channels because it is already present in the
          // interleaved image.
          alpha_found = true;
          continue;
        }
        std::string name(eci.name_length + 1, 0);
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetExtraChannelName(dec, i, &name[0], name.size())) {
          fprintf(stderr, "JxlDecoderGetExtraChannelName failed\n");
          return false;
        }
        ppf->extra_channels_info.push_back({eci, i, name});
      }
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      if (!FLAGS_color_space.empty()) {
        if (ppf->info.uses_original_profile) {
          fprintf(stderr,
                  "Warning: --color_space ignored because the image is "
                  "not XYB encoded.\n");
        } else {
          JxlColorEncoding color_encoding;
          if (!jxl::ParseDescription(FLAGS_color_space, &color_encoding)) {
            fprintf(stderr, "Failed to parse color space.\n");
            return false;
          }
          if (JXL_DEC_SUCCESS !=
              JxlDecoderSetPreferredColorProfile(dec, &color_encoding)) {
            fprintf(stderr, "Failed to set color space.\n");
            return false;
          }
        }
      }
      size_t icc_size = 0;
      JxlColorProfileTarget target = JXL_COLOR_PROFILE_TARGET_DATA;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderGetICCProfileSize(dec, &format, target, &icc_size)) {
        fprintf(stderr, "JxlDecoderGetICCProfileSize failed\n");
      }
      if (icc_size != 0) {
        ppf->icc.resize(icc_size);
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetColorAsICCProfile(dec, &format, target,
                                           ppf->icc.data(), icc_size)) {
          fprintf(stderr, "JxlDecoderGetColorAsICCProfile failed\n");
          return false;
        }
      } else {
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetColorAsEncodedProfile(dec, &format, target,
                                               &ppf->color_encoding)) {
          fprintf(stderr, "JxlDecoderGetColorAsEncodedProfile failed\n");
          return false;
        }
      }
    } else if (status == JXL_DEC_FRAME) {
      jxl::extras::PackedFrame frame(ppf->info.xsize, ppf->info.ysize, format);
      if (JXL_DEC_SUCCESS != JxlDecoderGetFrameHeader(dec, &frame.frame_info)) {
        fprintf(stderr, "JxlDecoderGetFrameHeader failed\n");
        return false;
      }
      frame.name.resize(frame.frame_info.name_length + 1, 0);
      if (JXL_DEC_SUCCESS !=
          JxlDecoderGetFrameName(dec, &frame.name[0], frame.name.size())) {
        fprintf(stderr, "JxlDecoderGetFrameName failed\n");
        return false;
      }
      ppf->frames.emplace_back(std::move(frame));
    } else if (status == JXL_DEC_FRAME_PROGRESSION) {
      size_t downsampling = JxlDecoderGetIntendedDownsamplingRatio(dec);
      if (downsampling <= FLAGS_downsampling) {
        if (JXL_DEC_SUCCESS != JxlDecoderFlushImage(dec)) {
          fprintf(stderr, "JxlDecoderFlushImage failed\n");
          return false;
        }
        if (ppf->frames.back().frame_info.is_last) {
          break;
        }
        if (JXL_DEC_SUCCESS != JxlDecoderSkipCurrentFrame(dec)) {
          fprintf(stderr, "JxlDecoderSkipCurrentFrame failed\n");
          return false;
        }
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderImageOutBufferSize(dec, &format, &buffer_size)) {
        fprintf(stderr, "JxlDecoderImageOutBufferSize failed\n");
        return false;
      }
      jxl::extras::PackedFrame& frame = ppf->frames.back();
      if (buffer_size != frame.color.pixels_size) {
        fprintf(stderr, "Invalid out buffer size %" PRIuS " %" PRIuS "\n",
                buffer_size, frame.color.pixels_size);
        return false;
      }
      auto callback = [](void* opaque, size_t x, size_t y, size_t num_pixels,
                         const void* pixels) {
        auto* ppf = reinterpret_cast<jxl::extras::PackedPixelFile*>(opaque);
        jxl::extras::PackedImage& color = ppf->frames.back().color;
        uint8_t* pixels_buffer = reinterpret_cast<uint8_t*>(color.pixels());
        size_t sample_size = color.pixel_stride();
        memcpy(pixels_buffer + (color.stride * y + sample_size * x), pixels,
               num_pixels * sample_size);
      };
      if (JXL_DEC_SUCCESS !=
          JxlDecoderSetImageOutCallback(dec, &format, callback, ppf)) {
        fprintf(stderr, "JxlDecoderSetImageOutCallback failed\n");
        return false;
      }
      JxlPixelFormat ec_format = format;
      ec_format.num_channels = 1;
      for (const auto& eci : ppf->extra_channels_info) {
        frame.extra_channels.emplace_back(jxl::extras::PackedImage(
            ppf->info.xsize, ppf->info.ysize, ec_format));
        auto& ec = frame.extra_channels.back();
        size_t buffer_size;
        if (JXL_DEC_SUCCESS != JxlDecoderExtraChannelBufferSize(
                                   dec, &ec_format, &buffer_size, eci.index)) {
          fprintf(stderr, "JxlDecoderExtraChannelBufferSize failed\n");
          return false;
        }
        if (buffer_size != ec.pixels_size) {
          fprintf(stderr,
                  "Invalid extra channel buffer size"
                  " %" PRIuS " %" PRIuS "\n",
                  buffer_size, ec.pixels_size);
          return false;
        }
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetExtraChannelBuffer(dec, &ec_format, ec.pixels(),
                                            buffer_size, eci.index)) {
          fprintf(stderr, "JxlDecoderSetExtraChannelBuffer failed\n");
          return false;
        }
      }
    } else if (status == JXL_DEC_SUCCESS) {
      // Decoding finished successfully.
      break;
    } else if (status == JXL_DEC_FULL_IMAGE) {
      if (ppf->frames.back().frame_info.is_last) {
        codestream_done = true;
      }
    } else {
      fprintf(stderr, "Error: unexpected status: %d\n",
              static_cast<int>(status));
      return false;
    }
  }
  boxes.FinalizeOutput();
  return true;
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

  bool decode_to_pixels = (codec != jxl::extras::Codec::kJPG);
#if JPEGXL_ENABLE_JPEG
  if (FLAGS_pixels_to_jpeg ||
      !gflags::GetCommandLineFlagInfoOrDie("jpeg_quality").is_default) {
    decode_to_pixels = true;
  }
#endif

  if (!decode_to_pixels) {
    std::vector<uint8_t> bytes;
    bool can_reconstruct_jpeg = false;
    for (size_t i = 0; i < num_reps; ++i) {
      if (!DecompressJxlReconstructJPEG(compressed, dec.get(), runner.get(),
                                        &bytes, &can_reconstruct_jpeg)) {
        if (!can_reconstruct_jpeg) {
          decode_to_pixels = true;
          break;
        }
        return EXIT_FAILURE;
      }
    }
    if (can_reconstruct_jpeg && !WriteFile(filename_out, bytes)) {
      return EXIT_FAILURE;
    };
  }
  if (decode_to_pixels) {
    std::unique_ptr<jxl::extras::Encoder> encoder =
        jxl::extras::Encoder::FromExtension(extension);
    if (encoder == nullptr) {
      fprintf(stderr, "can't decode to the file extension '%s'\n", extension);
      return EXIT_FAILURE;
    }
#if JPEGXL_ENABLE_JPEG
    std::ostringstream os;
    os << FLAGS_jpeg_quality;
    encoder->SetOption("q", os.str());
#endif
#if JPEGXL_ENABLE_SJPEG
    if (FLAGS_use_sjpeg) {
      encoder->SetOption("jpeg_encoder", "sjpeg");
    }
#endif
    jxl::extras::PackedPixelFile ppf;
    for (size_t i = 0; i < num_reps; ++i) {
      if (!DecompressJxlToPackedPixelFile(compressed,
                                          encoder->AcceptedFormats(), dec.get(),
                                          runner.get(), &ppf)) {
        fprintf(stderr, "DecompressJxlToPackedPixelFile failed\n");
        return EXIT_FAILURE;
      }
    }
    if (strcmp(extension, ".pfm") == 0) {
      ppf.info.bits_per_sample = 32;
    } else if (FLAGS_bits_per_sample > 0) {
      ppf.info.bits_per_sample = FLAGS_bits_per_sample;
    }
    jxl::extras::EncodedImage encoded_image;
    if (!encoder->Encode(ppf, &encoded_image)) {
      fprintf(stderr, "Encode failed\n");
      return EXIT_FAILURE;
    }
    size_t nlayers = 1 + encoded_image.extra_channel_bitstreams.size();
    size_t nframes = encoded_image.bitstreams.size();
    for (size_t i = 0; i < nlayers; ++i) {
      for (size_t j = 0; j < nframes; ++j) {
        const std::vector<uint8_t>& bitstream =
            (i == 0 ? encoded_image.bitstreams[j]
                    : encoded_image.extra_channel_bitstreams[i - 1][j]);
        std::string fn = Filename(base, extension, i, j, nlayers, nframes);
        if (!WriteFile(fn.c_str(), bitstream)) {
          return EXIT_FAILURE;
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
