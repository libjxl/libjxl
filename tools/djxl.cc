// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tools/djxl.h"

#include <stdio.h>

#include "lib/extras/codec.h"
#include "lib/extras/tone_mapping.h"
#include "lib/jxl/alpha.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/time.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "tools/args.h"
#include "tools/box/box.h"
#include "tools/cpu/cpu.h"

#if JPEGXL_ENABLE_JPEG
#include "lib/extras/codec_jpg.h"
#endif

namespace jpegxl {
namespace tools {

static inline bool ParseLuminanceRange(const char* arg,
                                       std::pair<float, float>* out) {
  char* end;
  out->first = static_cast<float>(strtod(arg, &end));
  if (*end == '\0') {
    // That was actually the upper bound.
    out->second = out->first;
    out->first = 0;
    return true;
  }
  if (*end != '-') {
    fprintf(stderr, "Unable to interpret as luminance range: %s.\n", arg);
    return JXL_FAILURE("Args");
  }
  const char* second = end + 1;
  out->second = static_cast<float>(strtod(second, &end));
  if (*end != '\0') {
    fprintf(stderr, "Unable to interpret as luminance range: %s.\n", arg);
    return JXL_FAILURE("Args");
  }
  return true;
}

void DecompressArgs::AddCommandLineOptions(CommandLineParser* cmdline) {
  // Positional arguments.
  cmdline->AddPositionalOption("INPUT", /* required = */ true,
                               "the compressed input file", &file_in);

  cmdline->AddPositionalOption(
      "OUTPUT", /* required = */ true,
      "the output can be PNG with ICC, JPG, or PPM/PFM.", &file_out);

  cmdline->AddOptionFlag('V', "version", "print version number and exit",
                         &version, &SetBooleanTrue);

  cmdline->AddOptionValue('\0', "num_reps", "N", nullptr, &num_reps,
                          &ParseUnsigned);

  opt_num_threads_id = cmdline->AddOptionValue('\0', "num_threads", "N",
                                               "The number of threads to use",
                                               &num_threads, &ParseUnsigned);

  cmdline->AddOptionValue('\0', "print_profile", "0|1",
                          "print timing information before exiting",
                          &print_profile, &ParseOverride);

  cmdline->AddOptionValue('\0', "bits_per_sample", "N",
                          "defaults to original (input) bit depth",
                          &bits_per_sample, &ParseUnsigned);

  cmdline->AddOptionFlag(
      '\0', "tone_map",
      "tone map the image to the luminance range indicated by --display_nits "
      "instead of performing a naive 0-1 -> 0-1 conversion",
      &tone_map, &SetBooleanTrue);

  cmdline->AddOptionValue('\0', "display_nits", "0.3-250",
                          "luminance range of the display to which to "
                          "tone-map; the lower bound can be omitted",
                          &display_nits, &ParseLuminanceRange);

  cmdline->AddOptionValue('\0', "color_space", "RGB_D65_SRG_Rel_Lin",
                          "defaults to original (input) color space",
                          &color_space, &ParseString);

  cmdline->AddOptionValue('s', "downsampling", "1,2,4,8,16",
                          "maximum permissible downsampling factor (values "
                          "greater than 16 will return the LQIP if available)",
                          &params.max_downsampling, &ParseUnsigned);

  cmdline->AddOptionFlag('\0', "allow_partial_files",
                         "allow decoding of truncated files",
                         &params.allow_partial_files, &SetBooleanTrue);

  cmdline->AddOptionFlag('\0', "allow_more_progressive_steps",
                         "allow decoding more progressive steps in truncated "
                         "files. No effect without --allow_partial_files",
                         &params.allow_more_progressive_steps, &SetBooleanTrue);

#if JPEGXL_ENABLE_JPEG
  cmdline->AddOptionFlag(
      'j', "pixels_to_jpeg",
      "By default, if the input JPEG XL contains a recompressed JPEG file, "
      "djxl "
      "reconstructs the exact original JPEG file. This flag causes the decoder "
      "to instead decode the image to pixels and encode a new (lossy) JPEG. "
      "The output file if provided must be a .jpg or .jpeg file.",
      &decode_to_pixels, &SetBooleanTrue);

  opt_jpeg_quality_id =
      cmdline->AddOptionValue('q', "jpeg_quality", "N",
                              "JPEG output quality. Setting an output quality "
                              "implies --pixels_to_jpeg.",
                              &jpeg_quality, &ParseUnsigned);
#endif

#if JPEGXL_ENABLE_SJPEG
  cmdline->AddOptionFlag('\0', "use_sjpeg",
                         "use sjpeg instead of libjpeg for JPEG output",
                         &use_sjpeg, &SetBooleanTrue);
#endif

  cmdline->AddOptionFlag('\0', "print_read_bytes",
                         "print total number of decoded bytes",
                         &print_read_bytes, &SetBooleanTrue);

  cmdline->AddOptionFlag('\0', "quiet", "silence output (except for errors)",
                         &quiet, &SetBooleanTrue);
}

jxl::Status DecompressArgs::ValidateArgs(const CommandLineParser& cmdline) {
  if (file_in == nullptr) {
    fprintf(stderr, "Missing INPUT filename.\n");
    return false;
  }

  // User didn't override num_threads, so we have to compute a default, which
  // might fail, so only do so when necessary. Don't just check num_threads != 0
  // because the user may have set it to that.
  if (!cmdline.GetOption(opt_num_threads_id)->matched()) {
    cpu::ProcessorTopology topology;
    if (!cpu::DetectProcessorTopology(&topology)) {
      // We have seen sporadic failures caused by setaffinity_np.
      fprintf(stderr,
              "Failed to choose default num_threads; you can avoid this "
              "error by specifying a --num_threads N argument.\n");
      return false;
    }
    num_threads = topology.packages * topology.cores_per_package;
  }

#if JPEGXL_ENABLE_JPEG
  if (cmdline.GetOption(opt_jpeg_quality_id)->matched()) {
    decode_to_pixels = true;
  }
#endif
  if (file_out) {
    const std::string extension = jxl::Extension(file_out);
    const jxl::Codec codec =
        jxl::CodecFromExtension(extension, &bits_per_sample);
    if (codec != jxl::Codec::kJPG) {
      // when decoding to anything-but-JPEG, we'll need pixels
      decode_to_pixels = true;
    }
  } else {
    decode_to_pixels = true;
  }
  return true;
}

jxl::Status DecompressJxlToPixels(const jxl::Span<const uint8_t> compressed,
                                  const jxl::DecompressParams& params,
                                  jxl::ThreadPool* pool,
                                  jxl::CodecInOut* JXL_RESTRICT io,
                                  SpeedStats* JXL_RESTRICT stats) {
  const double t0 = jxl::Now();
  if (!jxl::DecodeFile(params, compressed, io, pool)) {
    fprintf(stderr, "Failed to decompress to pixels.\n");
    return false;
  }
  const double t1 = jxl::Now();
  stats->NotifyElapsed(t1 - t0);
  stats->SetImageSize(io->xsize(), io->ysize());
  return true;
}

jxl::Status DecompressJxlToJPEG(const JpegXlContainer& container,
                                const DecompressArgs& args,
                                jxl::ThreadPool* pool, jxl::PaddedBytes* output,
                                SpeedStats* JXL_RESTRICT stats) {
  output->clear();
  const double t0 = jxl::Now();

  jxl::Span<const uint8_t> compressed(container.codestream,
                                      container.codestream_size);

  JXL_RETURN_IF_ERROR(compressed.size() >= 2);

  // JXL case
  // Decode to DCT when possible and generate a JPG file.
  jxl::CodecInOut io;
  // Set JPEG quality.
  // TODO(deymo): We should probably fail to give a JPEG file if the
  // original image can't be transcoded to a JPEG file without passing
  // through pixels, or at least signal this to the user.
  io.use_sjpeg = args.use_sjpeg;
  io.jpeg_quality = args.jpeg_quality;

  if (!DecodeJpegXlToJpeg(args.params, container, &io, pool)) {
    return JXL_FAILURE("Failed to decode JXL to JPEG");
  }
#if JPEGXL_ENABLE_JPEG
  if (!EncodeImageJPG(
          &io,
          io.use_sjpeg ? jxl::JpegEncoder::kSJpeg : jxl::JpegEncoder::kLibJpeg,
          io.jpeg_quality, jxl::YCbCrChromaSubsampling(), pool, output,
          jxl::DecodeTarget::kQuantizedCoeffs)) {
    return JXL_FAILURE("Failed to generate JPEG");
  }
#else   // JPEGXL_ENABLE_JPEG
  fprintf(
      stderr,
      "ERROR: Support for decoding to JPEG was not compiled in this tool.\n");
  return false;
#endif  // JPEGXL_ENABLE_JPEG
  stats->SetImageSize(io.xsize(), io.ysize());

  const double t1 = jxl::Now();
  stats->NotifyElapsed(t1 - t0);
  stats->SetFileSize(output->size());
  return true;
}

void RenderSpotColor(jxl::Image3F& img, const jxl::ImageF& sc,
                     const float color[4]) {
  float scale = color[3];
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < img.ysize(); y++) {
      float* JXL_RESTRICT p = img.Plane(c).Row(y);
      const float* JXL_RESTRICT s = sc.ConstRow(y);
      for (size_t x = 0; x < img.xsize(); x++) {
        float mix = scale * s[x];
        p[x] = mix * color[c] + (1.0 - mix) * p[x];
      }
    }
  }
}

jxl::Status WriteJxlOutput(const DecompressArgs& args, const char* file_out,
                           jxl::CodecInOut& io, jxl::ThreadPool* pool) {
  // Can only write if we decoded and have an output filename.
  // (Writing large PNGs is slow, so allow skipping it for benchmarks.)
  if (file_out == nullptr) return true;

  for (size_t i = 0; i < io.metadata.m.num_extra_channels; i++) {
    // Don't use Find() because there may be multiple spot color channels.
    const jxl::ExtraChannelInfo& eci = io.metadata.m.extra_channel_info[i];
    if (eci.type == jxl::ExtraChannel::kOptional) {
      continue;
    }
    if (eci.type == jxl::ExtraChannel::kUnknown ||
        (int(jxl::ExtraChannel::kReserved0) <= int(eci.type) &&
         int(eci.type) <= int(jxl::ExtraChannel::kReserved7))) {
      fprintf(stderr, "Unknown extra channel (bits %u, shift %u, name '%s')\n",
              eci.bit_depth.bits_per_sample, eci.dim_shift, eci.name.c_str());
      continue;
    }
    if (eci.type == jxl::ExtraChannel::kSpotColor) {
      for (size_t fr = 0; fr < io.frames.size(); fr++) {
        RenderSpotColor(*io.frames[fr].color(),
                        io.frames[fr].extra_channels()[i], eci.spot_color);
      }
    }
  }

  // Override original color space with arg if specified.
  jxl::ColorEncoding c_out = io.metadata.m.color_encoding;
  if (!args.color_space.empty()) {
    bool color_space_applied = false;
    if (jxl::ParseDescription(args.color_space, &c_out) && c_out.CreateICC()) {
      color_space_applied = true;
    } else {
      jxl::PaddedBytes icc;
      if (jxl::ReadFile(args.color_space, &icc) &&
          c_out.SetICC(std::move(icc))) {
        color_space_applied = true;
      }
    }

    if (!color_space_applied) {
      fprintf(stderr, "Failed to apply color_space.\n");
      return false;
    }
  }

  // Override original #bits with arg if specified.
  size_t bits_per_sample = io.metadata.m.bit_depth.bits_per_sample;
  if (args.bits_per_sample != 0) bits_per_sample = args.bits_per_sample;

  if (args.tone_map) {
    JXL_RETURN_IF_ERROR(jxl::ToneMapTo(args.display_nits, &io, pool));
    if (c_out.tf.IsPQ() && args.color_space.empty()) {
      // Prevent writing the tone-mapped image to PQ output unless explicitly
      // requested. The result would look even dimmer than it would have without
      // tone mapping.
      c_out.tf.SetTransferFunction(jxl::TransferFunction::kSRGB);
      JXL_RETURN_IF_ERROR(c_out.CreateICC());
    }
  }

  if (!io.metadata.m.have_animation) {
    if (!EncodeToFile(io, c_out, bits_per_sample, file_out, pool)) {
      fprintf(stderr, "Failed to write decoded image.\n");
      return false;
    }
  } else {
    const char* extension = strrchr(file_out, '.');
    std::string base = extension == nullptr
                           ? std::string(file_out)
                           : std::string(file_out, extension - file_out);
    if (extension == nullptr) extension = "";
    const int digits = 1 + static_cast<int>(std::log10(std::max(
                               1, static_cast<int>(io.frames.size() - 1))));
    std::vector<char> output_filename;
    output_filename.resize(base.size() + 1 + digits + strlen(extension) + 1);

    for (size_t i = 0; i < io.frames.size(); ++i) {
      jxl::CodecInOut frame_io;
      frame_io.SetFromImage(jxl::CopyImage(*io.frames[i].color()),
                            io.frames[i].c_current());
      frame_io.metadata.m = *io.frames[i].metadata();
      if (io.frames[i].HasAlpha()) {
        frame_io.Main().SetAlpha(
            jxl::CopyImage(*io.frames[i].alpha()),
            /*alpha_is_premultiplied=*/io.frames[i].AlphaIsPremultiplied());
      }
      snprintf(output_filename.data(), output_filename.size(), "%s-%0*zu%s",
               base.c_str(), digits, i, extension);
      if (!EncodeToFile(frame_io, c_out, bits_per_sample,
                        output_filename.data(), pool)) {
        fprintf(stderr, "Failed to write decoded image for frame %zu/%zu.\n",
                i + 1, io.frames.size());
      }
    }
  }
  if (!args.quiet) fprintf(stderr, "Done.\n");
  return true;
}

}  // namespace tools
}  // namespace jpegxl
