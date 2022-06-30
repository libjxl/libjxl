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
#include "lib/extras/time.h"
#include "lib/jxl/base/printf_macros.h"
#include "tools/cmdline.h"
#include "tools/codec_config.h"
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

    cmdline->AddOptionValue('\0', "num_threads", "N",
                            "Sets the number of threads to use. The default 0 "
                            "value means the machine default.",
                            &num_threads, &ParseUnsigned);

    cmdline->AddOptionValue('\0', "bits_per_sample", "N",
                            "Sets the output bit depth. The default 0 value "
                            "means the original (input) bit depth.",
                            &bits_per_sample, &ParseUnsigned);

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
    return true;
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  bool version = false;
  size_t num_reps = 1;
  size_t num_threads = 0;
  size_t bits_per_sample = 0;
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
  CommandLineParser::OptionId opt_jpeg_quality_id = -1;
};

}  // namespace tools
}  // namespace jpegxl

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

bool WriteOptionalOutput(const std::string& filename,
                         const std::vector<uint8_t>& bytes) {
  if (filename.empty() || bytes.empty()) {
    return true;
  }
  return WriteFile(filename.data(), bytes);
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
                                  JxlDecoder* dec, void* runner,
                                  std::vector<uint8_t>* jpeg_bytes,
                                  bool* can_reconstruct_jpeg,
                                  jpegxl::tools::SpeedStats* stats) {
  const double t0 = jxl::Now();
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

  JxlBasicInfo info;
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
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec, &info)) {
        fprintf(stderr, "JxlDecoderGetBasicInfo failed\n");
        return false;
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
      return false;
    }
  }
  if (!*can_reconstruct_jpeg) return false;
  size_t used_jpeg_output =
      jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec);
  jpeg_bytes->insert(jpeg_bytes->end(), jpeg_data_chunk.data(),
                     jpeg_data_chunk.data() + used_jpeg_output);
  const double t1 = jxl::Now();
  if (stats) {
    stats->NotifyElapsed(t1 - t0);
    stats->SetImageSize(info.xsize, info.ysize);
    stats->SetFileSize(jpeg_bytes->size());
  }
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
    const jpegxl::tools::DecompressArgs& args,
    const std::vector<uint8_t>& compressed,
    const std::vector<JxlPixelFormat>& accepted_formats, JxlDecoder* dec,
    void* runner, jxl::extras::PackedPixelFile* ppf, size_t* decoded_bytes,
    jpegxl::tools::SpeedStats* stats) {
  const double t0 = jxl::Now();
  JxlDecoderReset(dec);
  ppf->frames.clear();
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner)) {
    fprintf(stderr, "JxlEncoderSetParallelRunner failed\n");
    return false;
  }
  int events = (JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME |
                JXL_DEC_FULL_IMAGE | JXL_DEC_PREVIEW_IMAGE | JXL_DEC_BOX);
  if (args.downsampling > 1) {
    events |= JXL_DEC_FRAME_PROGRESSION;
    JxlDecoderSetProgressiveDetail(dec, JxlProgressiveDetail::kLastPasses);
  }
  if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec, events)) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    return false;
  }
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetRenderSpotcolors(dec, args.render_spotcolors)) {
    fprintf(stderr, "JxlDecoderSetRenderSpotColors failed\n");
    return false;
  }
  JxlPixelFormat format;
  // Reading compressed JPEG XL input and decoding to pixels
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetInput(dec, compressed.data(), compressed.size())) {
    fprintf(stderr, "Decoder failed to set input\n");
    return false;
  }
  if (args.display_nits > 0 &&
      JXL_DEC_SUCCESS !=
          JxlDecoderSetDesiredIntensityTarget(dec, args.display_nits)) {
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
      if (args.allow_partial_files) {
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
        name.resize(eci.name_length);
        ppf->extra_channels_info.push_back({eci, i, name});
      }
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      if (!args.color_space.empty()) {
        if (ppf->info.uses_original_profile) {
          fprintf(stderr,
                  "Warning: --color_space ignored because the image is "
                  "not XYB encoded.\n");
        } else {
          JxlColorEncoding color_encoding;
          if (!jxl::ParseDescription(args.color_space, &color_encoding)) {
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
      icc_size = 0;
      target = JXL_COLOR_PROFILE_TARGET_ORIGINAL;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderGetICCProfileSize(dec, &format, target, &icc_size)) {
        fprintf(stderr, "JxlDecoderGetICCProfileSize failed\n");
      }
      if (icc_size != 0) {
        ppf->orig_icc.resize(icc_size);
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetColorAsICCProfile(dec, &format, target,
                                           ppf->orig_icc.data(), icc_size)) {
          fprintf(stderr, "JxlDecoderGetColorAsICCProfile failed\n");
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
      frame.name.resize(frame.frame_info.name_length);
      ppf->frames.emplace_back(std::move(frame));
    } else if (status == JXL_DEC_FRAME_PROGRESSION) {
      size_t downsampling = JxlDecoderGetIntendedDownsamplingRatio(dec);
      if (downsampling <= args.downsampling) {
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
    } else if (status == JXL_DEC_NEED_PREVIEW_OUT_BUFFER) {
      size_t buffer_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderPreviewOutBufferSize(dec, &format, &buffer_size)) {
        fprintf(stderr, "JxlDecoderPreviewOutBufferSize failed\n");
        return false;
      }
      ppf->preview_frame = std::unique_ptr<jxl::extras::PackedFrame>(
          new jxl::extras::PackedFrame(ppf->info.preview.xsize,
                                       ppf->info.preview.ysize, format));
      if (buffer_size != ppf->preview_frame->color.pixels_size) {
        fprintf(stderr, "Invalid out buffer size %" PRIuS " %" PRIuS "\n",
                buffer_size, ppf->preview_frame->color.pixels_size);
        return false;
      }
      if (JXL_DEC_SUCCESS !=
          JxlDecoderSetPreviewOutBuffer(
              dec, &format, ppf->preview_frame->color.pixels(), buffer_size)) {
        fprintf(stderr, "JxlDecoderSetPreviewOutBuffer failed\n");
        return false;
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
    } else if (status == JXL_DEC_PREVIEW_IMAGE) {
      // Nothing to do.
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
  if (decoded_bytes) {
    *decoded_bytes = compressed.size() - JxlDecoderReleaseInput(dec);
  }
  const double t1 = jxl::Now();
  if (stats) {
    stats->NotifyElapsed(t1 - t0);
    stats->SetImageSize(ppf->info.xsize, ppf->info.ysize);
  }
  return true;
}

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
  if (!ReadFile(args.file_in, &compressed)) {
    fprintf(stderr, "couldn't load %s\n", args.file_in);
    return EXIT_FAILURE;
  }
  if (!args.quiet) {
    fprintf(stderr, "Read %" PRIuS " compressed bytes.\n", compressed.size());
  }

  if (!args.file_out && !args.quiet) {
    fprintf(stderr,
            "No output file specified.\n"
            "Decoding will be performed, but the result will be discarded.\n");
  }

  std::string filename_out;
  std::string base;
  std::string extension;
  if (args.file_out) {
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

  jpegxl::tools::SpeedStats stats;
  size_t num_worker_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
  {
    int64_t flag_num_worker_threads = args.num_threads;
    if (flag_num_worker_threads != 0) {
      num_worker_threads = flag_num_worker_threads;
    }
  }
  auto dec = JxlDecoderMake(/*memory_manager=*/nullptr);
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
    bool can_reconstruct_jpeg = false;
    for (size_t i = 0; i < num_reps; ++i) {
      if (!DecompressJxlReconstructJPEG(args, compressed, dec.get(),
                                        runner.get(), &bytes,
                                        &can_reconstruct_jpeg, &stats)) {
        if (!can_reconstruct_jpeg) {
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
    if (can_reconstruct_jpeg) {
      if (!args.quiet) fprintf(stderr, "Reconstructed to JPEG.\n");
      if (!filename_out.empty() && !WriteFile(filename_out.c_str(), bytes)) {
        return EXIT_FAILURE;
      }
    }
  }
  if (decode_to_pixels) {
    std::vector<JxlPixelFormat> accepted_formats;
    for (const uint32_t num_channels : {1, 2, 3, 4}) {
      accepted_formats.push_back(
          {num_channels, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, /*align=*/0});
    }
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
                                          dec.get(), runner.get(), &ppf,
                                          &decoded_bytes, &stats)) {
        fprintf(stderr, "DecompressJxlToPackedPixelFile failed\n");
        return EXIT_FAILURE;
      }
    }
    if (!args.quiet) fprintf(stderr, "Decoded to pixels.\n");
    if (args.print_read_bytes) {
      fprintf(stderr, "Decoded bytes: %" PRIuS "\n", decoded_bytes);
    }
    if (extension == ".pfm") {
      ppf.info.bits_per_sample = 32;
    } else if (args.bits_per_sample > 0) {
      ppf.info.bits_per_sample = args.bits_per_sample;
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
        if (!WriteFile(fn.c_str(), bitstream)) {
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
