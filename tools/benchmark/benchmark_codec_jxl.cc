// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "tools/benchmark/benchmark_codec_jxl.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "jxl/decode_cxx.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "lib/extras/codec.h"
#include "lib/extras/time.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/dec_params.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/modular/encoding/encoding.h"
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

struct JxlArgs {
  double xmul;
  double quant_bias;

  bool use_ac_strategy;
  bool qprogressive;  // progressive with shift-quantization.
  bool progressive;
  int progressive_dc;

  Override noise;
  Override dots;
  Override patches;

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
  args->AddSigned(&jxlargs->progressive_dc, "progressive_dc",
                  "Enable progressive mode for DC.", -1);

  args->AddOverride(&jxlargs->noise, "noise",
                    "Enable(1)/disable(0) noise generation.");
  args->AddOverride(&jxlargs->dots, "dots",
                    "Enable(1)/disable(0) dots generation.");
  args->AddOverride(&jxlargs->patches, "patches",
                    "Enable(1)/disable(0) patch dictionary.");

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
    const std::string kResamplingPrefix = "resampling=";
    const std::string kEcResamplingPrefix = "ec_resampling=";

    if (param.substr(0, kResamplingPrefix.size()) == kResamplingPrefix) {
      std::istringstream parser(param.substr(kResamplingPrefix.size()));
      parser >> cparams_.resampling;
    } else if (param.substr(0, kEcResamplingPrefix.size()) ==
               kEcResamplingPrefix) {
      std::istringstream parser(param.substr(kEcResamplingPrefix.size()));
      parser >> cparams_.ec_resampling;
    } else if (ImageCodec::ParseParam(param)) {
      // Nothing to do.
    } else if (param == "uint8") {
      uint8_ = true;
    } else if (param[0] == 'u') {
      char* end;
      cparams_.uniform_quant = strtof(param.c_str() + 1, &end);
      if (end == param.c_str() + 1 || *end != '\0') {
        return JXL_FAILURE("failed to parse uniform quant parameter %s",
                           param.c_str());
      }
      ba_params_.hf_asymmetry = args_.ba_params.hf_asymmetry;
    } else if (param.substr(0, kMaxPassesPrefix.size()) == kMaxPassesPrefix) {
      std::istringstream parser(param.substr(kMaxPassesPrefix.size()));
      parser >> dparams_.max_passes;
    } else if (param.substr(0, kDownsamplingPrefix.size()) ==
               kDownsamplingPrefix) {
      std::istringstream parser(param.substr(kDownsamplingPrefix.size()));
      parser >> dparams_.max_downsampling;
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
    } else if (param == "lp") {
      cparams_.lossy_palette = true;
    } else if (param[0] == 'C') {
      cparams_.colorspace = strtol(param.substr(1).c_str(), nullptr, 10);
    } else if (param[0] == 'c') {
      cparams_.color_transform =
          (jxl::ColorTransform)strtol(param.substr(1).c_str(), nullptr, 10);
      has_ctransform_ = true;
    } else if (param[0] == 'I') {
      cparams_.options.nb_repeats = strtof(param.substr(1).c_str(), nullptr);
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
    } else if (param[0] == 'R') {
      cparams_.responsive = strtol(param.substr(1).c_str(), nullptr, 10);
    } else if (param == "m") {
      cparams_.modular_mode = true;
      cparams_.color_transform = jxl::ColorTransform::kNone;
    } else if (param.substr(0, 3) == "gab") {
      long gab = strtol(param.substr(3).c_str(), nullptr, 10);
      if (gab != 0 && gab != 1) {
        return JXL_FAILURE("Invalid gab value");
      }
      cparams_.gaborish = static_cast<Override>(gab);
    } else if (param[0] == 'g') {
      long gsize = strtol(param.substr(1).c_str(), nullptr, 10);
      if (gsize < 0 || gsize > 3) {
        return JXL_FAILURE("Invalid group size shift value");
      }
      cparams_.modular_group_size_shift = gsize;
    } else if (param == "new_heuristics") {
      cparams_.use_new_heuristics = true;
    } else if (param == "plt") {
      cparams_.options.max_properties = 0;
      cparams_.options.nb_repeats = 0;
      cparams_.options.predictor = Predictor::Zero;
      cparams_.responsive = 0;
      cparams_.colorspace = 0;
      cparams_.channel_colors_pre_transform_percent = 0;
      cparams_.channel_colors_percent = 0;
    } else if (param.substr(0, 3) == "epf") {
      cparams_.epf = strtol(param.substr(3).c_str(), nullptr, 10);
      if (cparams_.epf > 3) {
        return JXL_FAILURE("Invalid epf value");
      }
    } else if (param.substr(0, 16) == "faster_decoding=") {
      cparams_.decoding_speed_tier =
          strtol(param.substr(16).c_str(), nullptr, 10);
    } else {
      return JXL_FAILURE("Unrecognized param");
    }
    return true;
  }

  bool IsColorAware() const override {
    // Can't deal with negative values from color space conversion.
    if (cparams_.modular_mode) return false;
    // Otherwise, input may be in any color space.
    return true;
  }

  bool IsJpegTranscoder() const override {
    // TODO(veluca): figure out when to turn this on.
    return false;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPoolInternal* pool, PaddedBytes* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
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

    cparams_.quant_border_bias = static_cast<float>(jxlargs->quant_bias);
    cparams_.ba_params.hf_asymmetry = ba_params_.hf_asymmetry;
    cparams_.ba_params.xmul = static_cast<float>(jxlargs->xmul);

    cparams_.quality_pair.first = q_target_;
    cparams_.quality_pair.second = q_target_;
    if (q_target_ != 100 && cparams_.color_transform == ColorTransform::kNone &&
        cparams_.modular_mode && !has_ctransform_) {
      cparams_.color_transform = ColorTransform::kXYB;
    }

    const double start = Now();
    PassesEncoderState passes_encoder_state;
    if (cparams_.use_new_heuristics) {
      passes_encoder_state.heuristics =
          jxl::make_unique<jxl::FastEncoderHeuristics>();
    }
    JXL_RETURN_IF_ERROR(EncodeFile(cparams_, io, &passes_encoder_state,
                                   compressed, GetJxlCms(), &cinfo_, pool));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    io->frames.clear();
    if (dparams_.max_passes != DecompressParams().max_passes ||
        dparams_.max_downsampling != DecompressParams().max_downsampling) {
      // Must use the C++ API to honor non-default dparams.
      if (uint8_) {
        return JXL_FAILURE(
            "trying to use decompress params that are not all available in "
            "either decoding API");
      }
      const double start = Now();
      JXL_RETURN_IF_ERROR(DecodeFile(dparams_, compressed, io, pool));
      const double end = Now();
      speed_stats->NotifyElapsed(end - start);
      return true;
    }

    double elapsed_convert_image = 0;
    const double start = Now();
    {
      std::vector<uint8_t> pixel_data;
      PaddedBytes icc_profile;
      auto runner = JxlThreadParallelRunnerMake(nullptr, pool->NumThreads());
      auto dec = JxlDecoderMake(nullptr);
      // By default, the decoder will undo exif orientation, giving an image
      // with identity exif rotation as result. However, the benchmark does
      // not undo exif orientation of the originals, and compares against the
      // originals, so we must set the option to keep the original orientation
      // instead.
      JxlDecoderSetKeepOrientation(dec.get(), JXL_TRUE);
      JXL_RETURN_IF_ERROR(
          JXL_DEC_SUCCESS ==
          JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO |
                                                   JXL_DEC_COLOR_ENCODING |
                                                   JXL_DEC_FULL_IMAGE));
      JxlBasicInfo info{};
      JxlPixelFormat format = {/*num_channels=*/3,
                               /*data_type=*/JXL_TYPE_FLOAT,
                               /*endianness=*/JXL_NATIVE_ENDIAN,
                               /*align=*/0};
      if (uint8_) {
        format.data_type = JXL_TYPE_UINT8;
      }
      JxlDecoderSetInput(dec.get(), compressed.data(), compressed.size());
      JxlDecoderStatus status;
      while ((status = JxlDecoderProcessInput(dec.get())) != JXL_DEC_SUCCESS) {
        switch (status) {
          case JXL_DEC_ERROR:
            return JXL_FAILURE("decoder error");
          case JXL_DEC_NEED_MORE_INPUT:
            return JXL_FAILURE("decoder requests more input");
          case JXL_DEC_BASIC_INFO:
            JXL_RETURN_IF_ERROR(JXL_DEC_SUCCESS ==
                                JxlDecoderGetBasicInfo(dec.get(), &info));
            format.num_channels = info.num_color_channels;
            if (info.alpha_bits != 0) {
              ++format.num_channels;
              io->metadata.m.extra_channel_info.resize(1);
              io->metadata.m.extra_channel_info[0].type =
                  jxl::ExtraChannel::kAlpha;
            }
            break;
          case JXL_DEC_COLOR_ENCODING: {
            size_t icc_size;
            JXL_RETURN_IF_ERROR(JXL_DEC_SUCCESS ==
                                JxlDecoderGetICCProfileSize(
                                    dec.get(), &format,
                                    JXL_COLOR_PROFILE_TARGET_DATA, &icc_size));
            icc_profile.resize(icc_size);
            JXL_RETURN_IF_ERROR(JXL_DEC_SUCCESS ==
                                JxlDecoderGetColorAsICCProfile(
                                    dec.get(), &format,
                                    JXL_COLOR_PROFILE_TARGET_DATA,
                                    icc_profile.data(), icc_profile.size()));
            break;
          }
          case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
            size_t buffer_size;
            JXL_RETURN_IF_ERROR(
                JXL_DEC_SUCCESS ==
                JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size));
            JXL_RETURN_IF_ERROR(buffer_size ==
                                info.xsize * info.ysize * format.num_channels *
                                    (uint8_ ? sizeof(uint8_t) : sizeof(float)));
            pixel_data.resize(buffer_size);
            JXL_RETURN_IF_ERROR(JXL_DEC_SUCCESS ==
                                JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                                            pixel_data.data(),
                                                            buffer_size));
            break;
          }
          case JXL_DEC_FULL_IMAGE: {
            const double start_convert_image = Now();
            {
              ColorEncoding color_encoding;
              JXL_RETURN_IF_ERROR(
                  color_encoding.SetICC(PaddedBytes(icc_profile)));
              ImageBundle frame(&io->metadata.m);
              JXL_RETURN_IF_ERROR(BufferToImageBundle(
                  format, info.xsize, info.ysize, pixel_data.data(),
                  pixel_data.size(), pool, color_encoding, &frame));
              io->frames.push_back(std::move(frame));
              io->dec_pixels += info.xsize * info.ysize;
            }
            const double end_convert_image = Now();
            elapsed_convert_image += end_convert_image - start_convert_image;
            break;
          }
          default:
            return JXL_FAILURE("unrecognized status %d",
                               static_cast<int>(status));
        }
      }
    }
    const double end = Now();
    speed_stats->NotifyElapsed(end - start - elapsed_convert_image);
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
  CompressParams cparams_;
  bool has_ctransform_ = false;
  DecompressParams dparams_;
  bool uint8_ = false;
};

ImageCodec* CreateNewJxlCodec(const BenchmarkArgs& args) {
  return new JxlCodec(args);
}

}  // namespace jxl
