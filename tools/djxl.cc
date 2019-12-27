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

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "jxl/aux_out.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/os_specific.h"
#include "jxl/base/override.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/dec_file.h"
#include "jxl/extras/codec.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "tools/args.h"

namespace jpegxl {
namespace tools {

void JxlDecompressArgs::AddCommandLineOptions(
    tools::CommandLineParser* cmdline) {
  // Flags.
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

  cmdline->AddOptionFlag('j', "jpeg", "decode losslessly recompressed JPEG",
                         &params.keep_dct, &SetBooleanTrue);

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

jxl::Status JxlDecompressArgs::ValidateArgs() {
  if (params.noise == jxl::Override::kOn) {
    fprintf(stderr, "Noise can only be enabled by the encoder.\n");
    return JXL_FAILURE("Cannot force noise on");
  }
  params.brunsli.fix_dc_staircase = brunsli_fix_dc_staircase;
  params.brunsli.gaborish = brunsli_gaborish;

  return true;
}

jxl::Status DecompressJxl(const jxl::Span<const uint8_t> compressed,
                          const jxl::DecompressParams& params,
                          jxl::ThreadPool* pool,
                          jxl::CodecInOut* JXL_RESTRICT io,
                          jxl::AuxOut* aux_out,
                          SpeedStats* JXL_RESTRICT stats) {
  const double t0 = jxl::Now();
  if (!DecodeFile(params, compressed, io, aux_out, pool)) {
    fprintf(stderr, "Failed to decompress.\n");
    return false;
  }
  const double t1 = jxl::Now();
  stats->NotifyElapsed(t1 - t0);
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

jxl::Status WriteJxlOutput(const JxlDecompressArgs& args, const char* file_out,
                           const jxl::CodecInOut& io) {
  // Can only write if we decoded and have an output filename.
  // (Writing large PNGs is slow, so allow skipping it for benchmarks.)
  if (file_out == nullptr) return true;

  for (size_t ec = 0; ec < io.metadata.m2.num_extra_channels; ec++) {
    if (io.metadata.m2.extra_channel_info[ec].rendered != 0) {
      if (io.metadata.m2.extra_channel_info[ec].rendered == 1) {
        for (size_t fr = 0; fr < io.frames.size(); fr++)
          RenderSpotColor(io.frames[fr].color(),
                          io.frames[fr].extra_channels()[ec],
                          io.metadata.m2.extra_channel_info[ec].color,
                          io.metadata.m2.extra_channel_bits);
      } else {
        fprintf(stderr,
                "Warning: ignoring extra channel which is supposed to be "
                "rendered but I don't know how.\n");
      }
    }
  }

  // Override original color space with arg if specified.
  jxl::ColorEncoding c_out = io.metadata.color_encoding;
  if (!args.color_space.empty()) {
    if (!jxl::ParseDescription(args.color_space, &c_out) ||
        !jxl::ColorManagement::CreateProfile(&c_out)) {
      fprintf(stderr, "Failed to apply color_space.\n");
      return false;
    }
  }

  // Override original #bits with arg if specified.
  size_t bits_per_sample = io.metadata.bits_per_sample;
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
        frame_io.Main().SetAlpha(jxl::CopyImage(io.frames[0].alpha()));
    }

    // TODO: take NewBase into account
    for (size_t i = 0; i < io.frames.size(); ++i) {
      if (args.coalesce) {
        if (i > 0) {
          const jxl::AnimationFrame& af = io.animation_frames[i];
          jxl::Rect cropbox(frame_io.Main().color());
          if (af.have_crop)
            cropbox = jxl::Rect(af.x0, af.y0, af.xsize, af.ysize);
          if (af.blend_mode() == jxl::AnimationFrame::BlendMode::kAdd) {
            for (int p = 0; p < 3; p++) {
              jxl::AddTo(jxl::Rect(io.frames[i].color()),
                         io.frames[i].color().Plane(p), cropbox,
                         &frame_io.Main().color().Plane(p));
            }
            if (frame_io.Main().HasAlpha()) {
              jxl::AddTo(jxl::Rect(io.frames[i].alpha()), io.frames[i].alpha(),
                         cropbox, &frame_io.Main().alpha());
            }
          } else if (af.blend_mode() ==
                         jxl::AnimationFrame::BlendMode::kBlend &&
                     io.frames[i].HasAlpha()) {
            // blend without alpha is just replace
            float max_alpha = (1 << io.metadata.alpha_bits) - 1;
            float rmax_alpha = 1.0f / max_alpha;
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
              for (size_t x = 0; x < cropbox.xsize(); x++) {
                if (a1[x] == 0) continue;
                float new_a = a1[x] + (a[x] * (max_alpha - a1[x])) * rmax_alpha;
                float rnew_a = 1.0f / new_a;
                r[x] = (r1[x] * a1[x] + r[x] * a[x] * (max_alpha - a1[x])) *
                       rnew_a;
                g[x] = (g1[x] * a1[x] + g[x] * a[x] * (max_alpha - a1[x])) *
                       rnew_a;
                b[x] = (b1[x] * a1[x] + b[x] * a[x] * (max_alpha - a1[x])) *
                       rnew_a;
                a[x] = new_a;
              }
            }
          } else {  // kReplace
            jxl::CopyImageTo(
                io.frames[i].color(), cropbox,
                const_cast<jxl::Image3F*>(&frame_io.Main().color()));
            if (frame_io.Main().HasAlpha())
              jxl::CopyImageTo(
                  io.frames[i].alpha(), cropbox,
                  const_cast<jxl::ImageU*>(&frame_io.Main().alpha()));
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
          frame_io.Main().SetAlpha(jxl::CopyImage(io.frames[i].alpha()));
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
