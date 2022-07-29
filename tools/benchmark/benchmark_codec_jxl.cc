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

#include "jxl/thread_parallel_runner_cxx.h"
#include "lib/extras/codec.h"
#include "lib/extras/dec/jxl.h"
#if JPEGXL_ENABLE_JPEG
#include "lib/extras/enc/jpg.h"
#endif
#include "lib/extras/packed_image_convert.h"
#include "lib/extras/time.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/codec_in_out.h"
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

  bool log_search_state;
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

  args->AddFlag(&jxlargs->log_search_state, "log_search_state",
                "Print out debug info for tortoise mode AQ loop.", false);

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
      if (param[0] == 'd' && butteraugli_target_ == 0.0) {
        cparams_.SetLossless();
      }
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
    } else if (param.substr(0, 2) == "nr") {
      normalize_bitrate_ = true;
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
    if (normalize_bitrate_) return false;
    // Otherwise, input may be in any color space.
    return true;
  }

  bool IsJpegTranscoder() const override {
    // TODO(veluca): figure out when to turn this on.
    return false;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPoolInternal* pool, std::vector<uint8_t>* compressed,
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

    if (cparams_.butteraugli_distance > 0.f &&
        cparams_.color_transform == ColorTransform::kNone &&
        cparams_.modular_mode && !has_ctransform_) {
      cparams_.color_transform = ColorTransform::kXYB;
    }

    cparams_.log_search_state = jxlargs->log_search_state;

#if JPEGXL_ENABLE_JPEG
    if (normalize_bitrate_ && cparams_.butteraugli_distance > 0.0f) {
      extras::PackedPixelFile ppf;
      JxlPixelFormat format = {0, JXL_TYPE_UINT8, JXL_BIG_ENDIAN, 0};
      JXL_RETURN_IF_ERROR(ConvertCodecInOutToPackedPixelFile(
          *io, format, io->metadata.m.color_encoding, pool, &ppf));
      extras::EncodedImage encoded;
      std::unique_ptr<extras::Encoder> encoder = extras::GetJPEGEncoder();
      encoder->SetOption("q", "95");
      JXL_RETURN_IF_ERROR(encoder->Encode(ppf, &encoded, pool));
      float jpeg_bits = encoded.bitstreams.back().size() * kBitsPerByte;
      float jpeg_bitrate = jpeg_bits / (io->xsize() * io->ysize());
      // Formula fitted on jyrki31 corpus for distances between 1.0 and 8.0.
      cparams_.target_bitrate = (jpeg_bitrate * 0.36f /
                                 (0.6f * cparams_.butteraugli_distance + 0.4f));
    }
#endif

    const double start = Now();
    PassesEncoderState passes_encoder_state;
    PaddedBytes compressed_padded;
    JXL_RETURN_IF_ERROR(EncodeFile(cparams_, io, &passes_encoder_state,
                                   &compressed_padded, GetJxlCms(), &cinfo_,
                                   pool));
    const double end = Now();
    compressed->assign(compressed_padded.begin(), compressed_padded.end());
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    dparams_.runner = pool->runner();
    dparams_.runner_opaque = pool->runner_opaque();
    JxlDataType data_type = uint8_ ? JXL_TYPE_UINT8 : JXL_TYPE_FLOAT;
    dparams_.accepted_formats = {{3, data_type, JXL_NATIVE_ENDIAN, 0},
                                 {4, data_type, JXL_NATIVE_ENDIAN, 0}};
    // By default, the decoder will undo exif orientation, giving an image
    // with identity exif rotation as result. However, the benchmark does
    // not undo exif orientation of the originals, and compares against the
    // originals, so we must set the option to keep the original orientation
    // instead.
    dparams_.keep_orientation = true;
    extras::PackedPixelFile ppf;
    size_t decoded_bytes;
    const double start = Now();
    JXL_RETURN_IF_ERROR(DecodeImageJXL(compressed.data(), compressed.size(),
                                       dparams_, &decoded_bytes, &ppf));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    JXL_RETURN_IF_ERROR(ConvertPackedPixelFileToCodecInOut(ppf, pool, io));
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
  extras::JXLDecompressParams dparams_;
  bool uint8_ = false;
  bool normalize_bitrate_ = false;
};

ImageCodec* CreateNewJxlCodec(const BenchmarkArgs& args) {
  return new JxlCodec(args);
}

}  // namespace jxl
