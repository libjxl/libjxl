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

#include "tools/djpegxl.h"

#include <stdio.h>

#include "jxl/alpha.h"
#include "jxl/aux_out.h"
#include "jxl/base/arch_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/file_io.h"
#include "jxl/base/os_specific.h"
#include "jxl/base/override.h"
#include "jxl/brunsli.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/dec_file.h"
#include "jxl/extras/codec.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "tools/args.h"

#if JPEGXL_ENABLE_JPEG
#include "jxl/extras/codec_jpg.h"
#endif

#include <brunsli/brunsli_decode.h>
#include <brunsli/jpeg_data.h>
#include <brunsli/jpeg_data_writer.h>
#include <brunsli/status.h>
#include <brunsli/types.h>

namespace jpegxl {
namespace tools {

DecompressArgs::DecompressArgs() {}

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

#if JPEGXL_ENABLE_SJPEG
  cmdline->AddOptionFlag('\0', "use_sjpeg",
                         "use sjpeg instead of libjpeg for JPEG output",
                         &use_sjpeg, &SetBooleanTrue);
#endif

  cmdline->AddOptionValue('\0', "jpeg_quality", "N", "JPEG output quality",
                          &jpeg_quality, &ParseUnsigned);

  opt_num_threads_id = cmdline->AddOptionValue('\0', "num_threads", "N",
                                               "The number of threads to use",
                                               &num_threads, &ParseUnsigned);

  cmdline->AddOptionValue('\0', "print_profile", "0|1",
                          "print timing information before exiting",
                          &print_profile, &ParseOverride);

  cmdline->AddOptionValue('\0', "print_info", "0|1",
                          "print AuxOut before exiting", &print_info,
                          &ParseOverride);

  cmdline->AddOptionValue('\0', "bits_per_sample", "N",
                          "defaults to original (input) bit depth",
                          &bits_per_sample, &ParseUnsigned);

  cmdline->AddOptionValue('\0', "color_space", "RGB_D65_SRG_Rel_Lin",
                          "defaults to original (input) color space",
                          &color_space, &ParseString);

  cmdline->AddOptionValue('\0', "noise", "0", "disables noise generation",
                          &params.noise, &ParseOverride);

  cmdline->AddOptionValue('\0', "adaptive_reconstruction", "0|1",
                          "disables/enables extra filtering",
                          &params.adaptive_reconstruction, &ParseOverride);

  cmdline->AddOptionValue('s', "downsampling", "1,2,4,8,16",
                          "maximum permissible downsampling factor (values "
                          "greater than 16 will return the LQIP if available)",
                          &params.max_downsampling, &ParseUnsigned);

  cmdline->AddOptionFlag('c', "coalesce", "decode coalesced animation frames",
                         &coalesce, &SetBooleanTrue);

  cmdline->AddOptionFlag(
      'j', "jpeg",
      "decode directly to JPEG when possible. Depending on the JPEG XL mode "
      "used when encoding this will produce an exact original JPEG file, a "
      "lossless pixel image data in a JPEG file or just a similar JPEG than "
      "the original image. The output file if provided must be a .jpg or .jpeg "
      "file.",
      &decode_to_jpeg, &SetBooleanTrue);

  cmdline->AddOptionFlag('\0', "print_read_bytes",
                         "print total number of decoded bytes",
                         &print_read_bytes, &SetBooleanTrue);

  cmdline->AddOptionFlag(
      't', "fix_dc_staircase",
      "Fix DC staircase, for recompressed JPEG1 files (brunsli) only",
      &brunsli_fix_dc_staircase, &SetBooleanTrue);
  cmdline->AddOptionFlag(
      'g', "gaborish",
      "Gaborish deblocking, for recompressed JPEG1 files (brunsli) only",
      &brunsli_gaborish, &SetBooleanTrue);
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
    jxl::ProcessorTopology topology;
    if (!jxl::DetectProcessorTopology(&topology)) {
      // We have seen sporadic failures caused by setaffinity_np.
      fprintf(stderr,
              "Failed to choose default num_threads; you can avoid this "
              "error by specifying a --num_threads N argument.\n");
      return false;
    }
    num_threads = topology.packages * topology.cores_per_package;
  }

  if (params.noise == jxl::Override::kOn) {
    fprintf(stderr, "Noise can only be enabled by the encoder.\n");
    return JXL_FAILURE("Cannot force noise on");
  }
  params.brunsli.fix_dc_staircase = brunsli_fix_dc_staircase;
  params.brunsli.gaborish = brunsli_gaborish;

  if (!decode_to_jpeg && file_out) {
    const std::string extension = jxl::Extension(file_out);
    const jxl::Codec codec =
        jxl::CodecFromExtension(extension, &bits_per_sample);
    if (codec == jxl::Codec::kJPG) {
      fprintf(stderr,
              "Notice: Decoding to pixels and re-encoding to JPEG file. To "
              "decode a losslessly recompressed JPEG back to JPEG pass --jpeg "
              "to djpegxl.\n");
    }
  }

  return true;
}

jxl::Status DecompressJxlToPixels(const jxl::Span<const uint8_t> compressed,
                                  const jxl::DecompressParams& params,
                                  jxl::ThreadPool* pool,
                                  jxl::CodecInOut* JXL_RESTRICT io,
                                  jxl::AuxOut* aux_out,
                                  SpeedStats* JXL_RESTRICT stats) {
  const double t0 = jxl::Now();

  jxl::Status ok = false;
  // JPEG1, not JXL nor Brunsli
  if (compressed[0] == 0xff && compressed[1] != jxl::kCodestreamMarker) {
#if JPEGXL_ENABLE_JPEG
    ok = DecodeImageJPG(compressed, pool, io);
#endif
  } else {
    ok = DecodeFile(params, compressed, io, aux_out, pool);
  }
  if (!ok) {
    fprintf(stderr, "Failed to decompress to pixels.\n");
    return false;
  }

  const double t1 = jxl::Now();
  stats->NotifyElapsed(t1 - t0);
  stats->SetImageSize(io->xsize(), io->ysize());
  return true;
}

namespace {

// Writer function needed for the Brunsli API.
size_t PaddedBytesWriter(void* data, const uint8_t* buf, size_t count) {
  jxl::PaddedBytes* output = reinterpret_cast<jxl::PaddedBytes*>(data);
  output->append(buf, buf + count);
  return count;
}

}  // namespace

jxl::Status DecompressJxlToJPEG(const jxl::Span<const uint8_t> compressed,
                                const DecompressArgs& args,
                                jxl::ThreadPool* pool, jxl::PaddedBytes* output,
                                jxl::AuxOut* aux_out,
                                SpeedStats* JXL_RESTRICT stats) {
  const double t0 = jxl::Now();
  JXL_RETURN_IF_ERROR(compressed.size() >= 2);

  if (compressed[0] == 0xff && compressed[1] != jxl::kCodestreamMarker) {
    // JPEG1 case, just copy the file.
    // TODO: We should signal that the file was not compressed.
    output->assign(compressed.data(), compressed.data() + compressed.size());
    // TODO: In this case we don't know the size of the pixel data so we can't
    // report stats on it.
  } else if (jxl::IsBrunsliFile(compressed) ==
             jxl::BrunsliFileSignature::kBrunsli) {
    // Brunsli file.
    brunsli::JPEGData jpg;
    bool ok = false;

#ifdef BRUNSLI_EXPERIMENTAL_GROUPS
    {
      brunsli::Executor executor = [&](const brunsli::Runnable& runnable,
                                       size_t num_tasks) {
        RunOnPool(pool, 0, num_tasks, runnable);
      };
      ok = brunsli::DecodeGroups(compressed.data(), compressed.size(), &jpg, 32,
                                 128, &executor);
    }
#else  // BRUNSLI_EXPERIMENTAL_GROUPS
    brunsli::BrunsliStatus status =
        brunsli::BrunsliDecodeJpeg(compressed.data(), compressed.size(), &jpg);
    ok = (status == brunsli::BRUNSLI_OK);
#endif  // BRUNSLI_EXPERIMENTAL_GROUPS
    if (!ok) {
      return JXL_FAILURE("Failed to parse Brunsli input.");
    }

    output->clear();
    brunsli::JPEGOutput writer(PaddedBytesWriter, output);
    if (!brunsli::WriteJpeg(jpg, writer)) {
      return JXL_FAILURE("Failed to generate JPEG from Brunsli input.");
    }
    stats->SetImageSize(jpg.width, jpg.height);
  } else {
    // JXL case
    // Decode to DCT when possible and generate a JPG file.
    jxl::CodecInOut io;
    // Set JPEG quality.
    // TODO(deymo): We should probably fail to give a JPEG file if the
    // original image can't be transcoded to a JPEG file without passing
    // through pixels, or at least signal this to the user.
    io.use_sjpeg = args.use_sjpeg;
    io.jpeg_quality = args.jpeg_quality;

    if (!jxl::DecodeFile(args.params, compressed, &io, aux_out, pool)) {
      return JXL_FAILURE("Failed to decode JXL to JPEG");
    }
    if (!EncodeImageJPG(&io,
                        io.use_sjpeg ? jxl::JpegEncoder::kSJpeg
                                     : jxl::JpegEncoder::kLibJpeg,
                        io.jpeg_quality,
                        io.use_sjpeg ? jxl::YCbCrChromaSubsampling::kAuto
                                     : jxl::YCbCrChromaSubsampling::k444,
                        pool, output,
                        io.Main().jpeg_xsize
                            ? jxl::DecodeTarget::kQuantizedCoeffs
                            : jxl::DecodeTarget::kPixels)) {
      return JXL_FAILURE("Failed to generate JPEG");
    }
    stats->SetImageSize(io.xsize(), io.ysize());
  }

  const double t1 = jxl::Now();
  stats->NotifyElapsed(t1 - t0);
  stats->SetFileSize(output->size());
  return true;
}

void RenderSpotColor(const jxl::Image3F& img, const jxl::ImageU& sc,
                     const float color[4], int ec_bit_depth) {
  float scale = color[3] / ((1 << ec_bit_depth) - 1.0f);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < img.ysize(); y++) {
      float* JXL_RESTRICT p = img.Plane(c).MutableRow(y);
      const uint16_t* JXL_RESTRICT s = sc.ConstRow(y);
      for (size_t x = 0; x < img.xsize(); x++) {
        float mix = scale * s[x];
        p[x] = mix * color[c] + (1.0 - mix) * p[x];
      }
    }
  }
}

jxl::Status WriteJxlOutput(const DecompressArgs& args, const char* file_out,
                           const jxl::CodecInOut& io) {
  // Can only write if we decoded and have an output filename.
  // (Writing large PNGs is slow, so allow skipping it for benchmarks.)
  if (file_out == nullptr) return true;

  for (size_t i = 0; i < io.metadata.m2.num_extra_channels; i++) {
    // Don't use Find() because there may be multiple spot color channels.
    const jxl::ExtraChannelInfo& eci = io.metadata.m2.extra_channel_info[i];
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
      for (size_t fr = 0; fr < io.frames.size(); fr++)
        RenderSpotColor(io.frames[fr].color(),
                        io.frames[fr].extra_channels()[i], eci.spot_color,
                        eci.bit_depth.bits_per_sample);
    }
  }

  // Override original color space with arg if specified.
  jxl::ColorEncoding c_out = io.metadata.color_encoding;
  if (!args.color_space.empty()) {
    if (!jxl::ParseDescription(args.color_space, &c_out) ||
        !c_out.CreateICC()) {
      fprintf(stderr, "Failed to apply color_space.\n");
      return false;
    }
  }

  // Override original #bits with arg if specified.
  size_t bits_per_sample = io.metadata.bit_depth.bits_per_sample;
  if (args.bits_per_sample != 0) bits_per_sample = args.bits_per_sample;

  if (!io.metadata.m2.have_animation) {
    if (!EncodeToFile(io, c_out, bits_per_sample, file_out)) {
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

    jxl::CodecInOut frame_io;
    if (args.coalesce) {
      frame_io.SetFromImage(jxl::CopyImage(io.frames[0].color()),
                            io.frames[0].c_current());
      frame_io.metadata = *io.frames[0].metadata();
      if (io.frames[0].HasAlpha())
        frame_io.Main().SetAlpha(
            jxl::CopyImage(io.frames[0].alpha()),
            /*alpha_is_premultiplied=*/io.frames[0].AlphaIsPremultiplied());
    }

    // TODO: take NewBase into account
    for (size_t i = 0; i < io.frames.size(); ++i) {
      if (args.coalesce) {
        if (i > 0) {
          const jxl::AnimationFrame& af = io.animation_frames[i];
          jxl::Rect cropbox(frame_io.Main().color());
          if (af.have_crop)
            cropbox = jxl::Rect(af.x0, af.y0, af.xsize, af.ysize);
          if (af.blend_mode == jxl::BlendMode::kAdd) {
            for (int p = 0; p < 3; p++) {
              jxl::AddTo(jxl::Rect(io.frames[i].color()),
                         io.frames[i].color().Plane(p), cropbox,
                         &frame_io.Main().color().Plane(p));
            }
            if (frame_io.Main().HasAlpha()) {
              jxl::AddTo(jxl::Rect(io.frames[i].alpha()), io.frames[i].alpha(),
                         cropbox, &frame_io.Main().alpha());
            }
          } else if (af.blend_mode == jxl::BlendMode::kBlend
                     // blend without alpha is just replace
                     && io.frames[i].HasAlpha()) {
            if (io.frames[i].AlphaIsPremultiplied()) {
              // The whole frame needs to be converted to premultiplied alpha,
              // not just the part corresponding to the crop of the new frame.
              frame_io.Main().PremultiplyAlphaIfNeeded();
            }
            for (size_t y = 0; y < cropbox.ysize(); y++) {
              const uint16_t* JXL_RESTRICT a1 = io.frames[i].alpha().Row(y);
              const float* JXL_RESTRICT r1 =
                  io.frames[i].color().PlaneRow(0, y);
              const float* JXL_RESTRICT g1 =
                  io.frames[i].color().PlaneRow(1, y);
              const float* JXL_RESTRICT b1 =
                  io.frames[i].color().PlaneRow(2, y);
              uint16_t* JXL_RESTRICT a =
                  cropbox.MutableRow(&frame_io.Main().alpha(), y);
              float* JXL_RESTRICT r =
                  cropbox.MutableRow(&frame_io.Main().color().Plane(0), y);
              float* JXL_RESTRICT g =
                  cropbox.MutableRow(&frame_io.Main().color().Plane(1), y);
              float* JXL_RESTRICT b =
                  cropbox.MutableRow(&frame_io.Main().color().Plane(2), y);
              jxl::PerformAlphaBlending(
                  /*bg=*/{r, g, b, a, io.metadata.GetAlphaBits(),
                          frame_io.Main().AlphaIsPremultiplied()},
                  /*fg=*/
                  {r1, g1, b1, a1, io.metadata.GetAlphaBits(),
                   io.frames[i].AlphaIsPremultiplied()},
                  /*out=*/
                  {r, g, b, a, io.metadata.GetAlphaBits(),
                   frame_io.Main().AlphaIsPremultiplied()},
                  cropbox.xsize());
            }
          } else {  // kReplace
            if (!frame_io.Main().HasAlpha() ||
                !frame_io.Main().AlphaIsPremultiplied() ||
                io.frames[i].AlphaIsPremultiplied()) {
              if (io.frames[i].AlphaIsPremultiplied()) {
                frame_io.Main().PremultiplyAlphaIfNeeded();
              }
              jxl::CopyImageTo(
                  io.frames[i].color(), cropbox,
                  const_cast<jxl::Image3F*>(&frame_io.Main().color()));
              if (frame_io.Main().HasAlpha())
                jxl::CopyImageTo(
                    io.frames[i].alpha(), cropbox,
                    const_cast<jxl::ImageU*>(&frame_io.Main().alpha()));
            } else {
              JXL_ASSERT(frame_io.Main().AlphaIsPremultiplied() &&
                         !io.frames[i].AlphaIsPremultiplied());
              float max_alpha = jxl::MaxAlpha(io.metadata.GetAlphaBits());
              float rmax_alpha = 1.0f / max_alpha;
              for (size_t y = 0; y < cropbox.ysize(); ++y) {
                const uint16_t* JXL_RESTRICT a1 = io.frames[i].alpha().Row(y);
                const float* JXL_RESTRICT r1 =
                    io.frames[i].color().PlaneRow(0, y);
                const float* JXL_RESTRICT g1 =
                    io.frames[i].color().PlaneRow(1, y);
                const float* JXL_RESTRICT b1 =
                    io.frames[i].color().PlaneRow(2, y);
                uint16_t* JXL_RESTRICT a =
                    cropbox.MutableRow(&frame_io.Main().alpha(), y);
                float* JXL_RESTRICT r =
                    cropbox.MutableRow(&frame_io.Main().color().Plane(0), y);
                float* JXL_RESTRICT g =
                    cropbox.MutableRow(&frame_io.Main().color().Plane(1), y);
                float* JXL_RESTRICT b =
                    cropbox.MutableRow(&frame_io.Main().color().Plane(2), y);
                for (size_t x = 0; x < cropbox.xsize(); ++x) {
                  const float normalized_a1 = a1[x] * rmax_alpha;
                  r[x] = r1[x] * normalized_a1;
                  g[x] = g1[x] * normalized_a1;
                  b[x] = b1[x] * normalized_a1;
                  a[x] = a1[x];
                }
              }
            }
          }
        }

        snprintf(output_filename.data(), output_filename.size(), "%s-%0*zu%s",
                 base.c_str(), digits, i, extension);
        if (!EncodeToFile(frame_io, c_out, bits_per_sample,
                          output_filename.data())) {
          fprintf(stderr, "Failed to write decoded image for frame %zu/%zu.\n",
                  i + 1, io.frames.size());
        }

      } else {
        jxl::CodecInOut frame_io;
        frame_io.SetFromImage(jxl::CopyImage(io.frames[i].color()),
                              io.frames[i].c_current());
        frame_io.metadata = *io.frames[i].metadata();
        if (io.frames[i].HasAlpha())
          frame_io.Main().SetAlpha(
              jxl::CopyImage(io.frames[i].alpha()),
              /*alpha_is_premultiplied=*/io.frames[i].AlphaIsPremultiplied());
        snprintf(output_filename.data(), output_filename.size(), "%s-%0*zu%s",
                 base.c_str(), digits, i, extension);
        if (!EncodeToFile(frame_io, c_out, bits_per_sample,
                          output_filename.data())) {
          fprintf(stderr, "Failed to write decoded image for frame %zu/%zu.\n",
                  i + 1, io.frames.size());
        }
      }
    }
  }
  fprintf(stderr, "Wrote %zu bytes; done.\n", io.enc_size);
  return true;
}

}  // namespace tools
}  // namespace jpegxl
