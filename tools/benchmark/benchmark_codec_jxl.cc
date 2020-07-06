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
#include "tools/benchmark/benchmark_codec_jxl.h"

#include <brunsli/brunsli_decode.h>
#include <brunsli/brunsli_encode.h>
#include <brunsli/jpeg_data.h>
#include <brunsli/jpeg_data_reader.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "jxl/aux_out.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/os_specific.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/span.h"
#include "jxl/brunsli.h"
#include "jxl/codec_in_out.h"
#include "jxl/dec_file.h"
#include "jxl/dec_params.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_file.h"
#include "jxl/enc_params.h"
#include "jxl/extras/codec.h"
#include "jxl/image_bundle.h"
#include "jxl/modular/encoding/encoding.h"
#include "tools/benchmark/benchmark_file_io.h"
#include "tools/benchmark/benchmark_stats.h"
#include "tools/cmdline.h"

namespace jxl {

// Output function for EncodeBrunsli.
size_t OutputToBytes(void* data, const uint8_t* buf, size_t count) {
  PaddedBytes* output = reinterpret_cast<PaddedBytes*>(data);
  output->append(buf, buf + count);
  return count;
}

// TODO(lode): This is copied from brunsli/c/enc/encode.cc, use that one, once
// the latest version of brunsli is checked out as submodule.
int EncodeBrunsli(size_t insize, const unsigned char* in, void* outdata,
                  size_t (*outfun)(void* outdata, const unsigned char* buf,
                                   size_t size)) {
  std::vector<uint8_t> output;
  brunsli::JPEGData jpg;
  if (!brunsli::ReadJpeg(in, insize, brunsli::JPEG_READ_ALL, &jpg)) {
    return 0;
  }
  size_t output_size = brunsli::GetMaximumBrunsliEncodedSize(jpg);
  output.resize(output_size);
  if (!brunsli::BrunsliEncodeJpeg(jpg, output.data(), &output_size)) {
    return 0;
  }
  output.resize(output_size);
  if (!outfun(outdata, reinterpret_cast<const unsigned char*>(output.data()),
              output.size())) {
    return 0;
  }
  return 1; /* ok */
}

struct JxlArgs {
  double xmul;
  double quant_bias;

  bool use_ac_strategy;
  bool qprogressive;  // progressive with shift-quantization.
  bool progressive;
  size_t progressive_dc;

  Override noise;
  Override adaptive_reconstruction;
  Override dots;
  Override patches;
  Override gaborish;

  std::string debug_image_dir;
};

static JxlArgs* const jxlargs = new JxlArgs;

Status AddCommandLineOptionsJxlCodec(BenchmarkArgs* args) {
  args->AddDouble(&jxlargs->xmul, "xmul",
                  "Multiplier for the difference in X channel in Butteraugli.",
                  1.0);
  args->AddDouble(&jxlargs->quant_bias, "quant_bias",
                  "Bias border pixels during quantization by this ratio.", 0.0);
  args->AddFlag(&jxlargs->use_ac_strategy, "use_ac_strategy",
                "If true, AC strategy will be used.", false);
  args->AddFlag(&jxlargs->qprogressive, "qprogressive",
                "Enable quantized progressive mode for AC.", false);
  args->AddFlag(&jxlargs->progressive, "progressive",
                "Enable progressive mode for AC.", false);
  args->AddUnsigned(&jxlargs->progressive_dc, "progressive_dc",
                    "Enable progressive mode for DC.", 0);

  args->AddOverride(&jxlargs->noise, "noise",
                    "Enable(1)/disable(0) noise generation.");
  args->AddOverride(
      &jxlargs->adaptive_reconstruction, "adaptive_reconstruction",
      "Enable(1)/disable(0) adaptive reconstruction (deringing).");
  args->AddOverride(&jxlargs->dots, "dots",
                    "Enable(1)/disable(0) dots generation.");
  args->AddOverride(&jxlargs->patches, "patches",
                    "Enable(1)/disable(0) patch dictionary.");
  args->AddOverride(&jxlargs->gaborish, "gaborish", "Disable gaborish if 0.");

  args->AddString(
      &jxlargs->debug_image_dir, "debug_image_dir",
      "If not empty, saves debug images for each "
      "input image and each codec that provides it to this directory.");

  return true;
}

Status ValidateArgsJxlCodec(BenchmarkArgs* args) { return true; }

class JxlCodec : public ImageCodec {
 public:
  explicit JxlCodec(const BenchmarkArgs& args) : ImageCodec(args) {}

  Status ParseParam(const std::string& param) override {
    const std::string kMaxPassesPrefix = "max_passes=";
    const std::string kDownsamplingPrefix = "downsampling=";

    if (ImageCodec::ParseParam(param)) {
      // Nothing to do.
    } else if (param[0] == 'u') {
      cparams_.uniform_quant = strtof(param.substr(1).c_str(), nullptr);
      ba_params_.hf_asymmetry = args_.ba_params.hf_asymmetry;
    } else if (param[0] == 'Q') {
      brunsli_params_.quant_scale = strtof(param.substr(1).c_str(), nullptr);
    } else if (param.substr(0, kMaxPassesPrefix.size()) == kMaxPassesPrefix) {
      std::istringstream parser(param.substr(kMaxPassesPrefix.size()));
      parser >> dparams_.max_passes;
    } else if (param.substr(0, kDownsamplingPrefix.size()) ==
               kDownsamplingPrefix) {
      std::istringstream parser(param.substr(kDownsamplingPrefix.size()));
      parser >> dparams_.max_downsampling;
    } else if (param[0] == 'n' && param[1] == 'l') {
      cparams_.color_transform = jxl::ColorTransform::kNone;
      cparams_.near_lossless = strtol(param.substr(2).c_str(), nullptr, 10);
      if (cparams_.near_lossless == 0) cparams_.near_lossless = 2;
      cparams_.responsive = 0;
    } else if (ParseSpeedTier(param, &cparams_.speed_tier)) {
      // Nothing to do.
    } else if (param[0] == 'X') {
      cparams_.channel_colors_pre_transform_percent =
          strtol(param.substr(1).c_str(), nullptr, 10);
    } else if (param[0] == 'Y') {
      cparams_.channel_colors_percent =
          strtol(param.substr(1).c_str(), nullptr, 10);
    } else if (param[0] == 'p') {
      cparams_.palette_colors = strtol(param.substr(1).c_str(), nullptr, 10);
    } else if (param[0] == 'N') {
      cparams_.near_lossless = strtol(param.substr(1).c_str(), nullptr, 10);
    } else if (param[0] == 'C') {
      cparams_.colorspace = strtol(param.substr(1).c_str(), nullptr, 10);
    } else if (param[0] == 'c') {
      cparams_.color_transform =
          (jxl::ColorTransform)strtol(param.substr(1).c_str(), nullptr, 10);
    } else if (param[0] == 'I') {
      cparams_.options.nb_repeats = strtof(param.substr(1).c_str(), nullptr);
    } else if (param == "Brotli") {
      cparams_.options.entropy_coder = ModularOptions::kBrotli;
    } else if (param[0] == 'E') {
      cparams_.options.max_properties =
          strtof(param.substr(1).c_str(), nullptr);
    } else if (param[0] == 'P') {
      cparams_.options.predictor =
          static_cast<Predictor>(strtof(param.substr(1).c_str(), nullptr));
    } else if (param == "slow") {
      cparams_.options.nb_repeats = 2;
    } else if (param == "R") {
      cparams_.responsive = 1;
    } else if (param == "mg") {
      cparams_.modular_group_mode = true;
      cparams_.color_transform = jxl::ColorTransform::kNone;
    } else if (param == "bg") {
      cparams_.brunsli_group_mode = true;
      cparams_.color_transform = jxl::ColorTransform::kYCbCr;
    } else if (param == "plt") {
      cparams_.options.entropy_coder = ModularOptions::kBrotli;
      cparams_.options.brotli_effort = 11;
      cparams_.options.max_properties = 0;
      cparams_.options.nb_repeats = 0;
      cparams_.options.predictor = Predictor::Zero;
      cparams_.responsive = 0;
      cparams_.colorspace = 0;
      cparams_.channel_colors_pre_transform_percent = 0;
      cparams_.channel_colors_percent = 0;
    } else if (param == "b") {
      brunsli_mode_ = true;
    } else if (param == "file") {
      brunsli_file_ = true;  // Use jxl:b:file
    } else {
      return JXL_FAILURE("Unrecognized param");
    }
    return true;
  }

  bool IsColorAware() const override {
    // Can't deal with negative values from color space conversion.
    if (cparams_.brunsli_group_mode || cparams_.modular_group_mode)
      return false;
    // Otherwise, input may be in any color space.
    return true;
  }

  bool IsJpegTranscoder() const override {
    if (cparams_.brunsli_group_mode) return true;
    return false;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPool* pool, PaddedBytes* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    if (brunsli_mode_) {
      if (brunsli_file_) {
        // Encode the original JPG file (or get failure if it was not JPG),
        // rather than the CodecInOut pixels.
        PaddedBytes bytes;
        JXL_RETURN_IF_ERROR(ReadFile(filename, &bytes));
        const double start = Now();
        if (!EncodeBrunsli(bytes.size(), bytes.data(), compressed,
                           OutputToBytes)) {
          return JXL_FAILURE("Failed to encode file to brunsli");
        }
        const double end = Now();
        speed_stats->NotifyElapsed(end - start);
        return true;
      } else {
        if (io->metadata.HasAlpha()) {
          // Prevent Abort in ImageBundle::VerifyMetadata when decompressing.
          return JXL_FAILURE("Alpha not supported for brunsli");
        }
        const double start = Now();
        JXL_RETURN_IF_ERROR(
            PixelsToBrunsli(io, compressed, brunsli_params_, pool));
        const double end = Now();
        speed_stats->NotifyElapsed(end - start);
        return true;
      }
    }
    if (!jxlargs->debug_image_dir.empty()) {
      cinfo_.dump_image = [](const CodecInOut& io, const std::string& path) {
        return EncodeToFile(io, path);
      };
      cinfo_.debug_prefix =
          JoinPath(jxlargs->debug_image_dir, FileBaseName(filename)) +
          ".jxl:" + params_ + ".dbg/";
      JXL_RETURN_IF_ERROR(MakeDir(cinfo_.debug_prefix));
    }
    cparams_.butteraugli_distance = butteraugli_target_;
    cparams_.target_bitrate = bitrate_target_;

    cparams_.dots = jxlargs->dots;
    cparams_.patches = jxlargs->patches;

    cparams_.progressive_mode = jxlargs->progressive;
    cparams_.qprogressive_mode = jxlargs->qprogressive;
    cparams_.progressive_dc = jxlargs->progressive_dc;

    cparams_.noise = jxlargs->noise;
    cparams_.adaptive_reconstruction = jxlargs->adaptive_reconstruction;
    cparams_.gaborish = jxlargs->gaborish;

    cparams_.quant_border_bias = static_cast<float>(jxlargs->quant_bias);
    cparams_.ba_params.hf_asymmetry = ba_params_.hf_asymmetry;
    cparams_.ba_params.xmul = static_cast<float>(jxlargs->xmul);

    cparams_.quality_pair.first = q_target_;
    cparams_.quality_pair.second = q_target_;

    const double start = Now();
    PassesEncoderState passes_encoder_state;
    JXL_RETURN_IF_ERROR(EncodeFile(cparams_, io, &passes_encoder_state,
                                   compressed, &cinfo_, pool));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed, ThreadPool* pool,
                    CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    if (!jxlargs->debug_image_dir.empty()) {
      dinfo_.dump_image = [](const CodecInOut& io, const std::string& path) {
        return EncodeToFile(io, path);
      };
      dinfo_.debug_prefix =
          JoinPath(jxlargs->debug_image_dir, FileBaseName(filename)) +
          ".jxl:" + params_ + ".dbg/";
      JXL_RETURN_IF_ERROR(MakeDir(dinfo_.debug_prefix));
    }
    dparams_.noise = jxlargs->noise;
    dparams_.adaptive_reconstruction = jxlargs->adaptive_reconstruction;
    const double start = Now();
    JXL_RETURN_IF_ERROR(DecodeFile(dparams_, compressed, io, &dinfo_, pool));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  void GetMoreStats(BenchmarkStats* stats) override {
    JxlStats jxl_stats;
    jxl_stats.num_inputs = 1;
    jxl_stats.aux_out = cinfo_;
    stats->jxl_stats.Assimilate(jxl_stats);
  }

 protected:
  AuxOut cinfo_;
  AuxOut dinfo_;
  CompressParams cparams_;
  DecompressParams dparams_;
  BrunsliEncoderOptions brunsli_params_;
  bool brunsli_mode_{false};
  bool brunsli_file_{false};  // Brunsli on original JPG file instead of pixels
};

ImageCodec* CreateNewJxlCodec(const BenchmarkArgs& args) {
  return new JxlCodec(args);
}

}  // namespace jxl
