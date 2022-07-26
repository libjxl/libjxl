// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/benchmark/benchmark_codec.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "lib/extras/time.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "tools/benchmark/benchmark_args.h"
#include "tools/benchmark/benchmark_codec_custom.h"
#ifdef JPEGXL_ENABLE_JPEG
#include "tools/benchmark/benchmark_codec_jpeg.h"
#endif  // JPEG_ENABLE_JPEG
#include "tools/benchmark/benchmark_codec_jxl.h"
#include "tools/benchmark/benchmark_codec_png.h"
#include "tools/benchmark/benchmark_stats.h"

#ifdef BENCHMARK_WEBP
#include "tools/benchmark/benchmark_codec_webp.h"
#endif  // BENCHMARK_WEBP

#ifdef BENCHMARK_AVIF
#include "tools/benchmark/benchmark_codec_avif.h"
#endif  // BENCHMARK_AVIF

namespace jxl {

void ImageCodec::ParseParameters(const std::string& parameters) {
  params_ = parameters;
  std::vector<std::string> parts = SplitString(parameters, ':');
  for (size_t i = 0; i < parts.size(); ++i) {
    if (!ParseParam(parts[i])) {
      JXL_ABORT("Invalid parameter %s", parts[i].c_str());
    }
  }
}

Status ImageCodec::ParseParam(const std::string& param) {
  if (param[0] == 'q') {  // libjpeg-style quality, [0,100]
    const std::string quality_param = param.substr(1);
    char* end;
    const float q_target = strtof(quality_param.c_str(), &end);
    if (end == quality_param.c_str() ||
        end != quality_param.c_str() + quality_param.size()) {
      return false;
    }
    q_target_ = q_target;
    return true;
  }
  if (param[0] == 'd') {  // butteraugli distance
    const std::string distance_param = param.substr(1);
    char* end;
    const float butteraugli_target = strtof(distance_param.c_str(), &end);
    if (end == distance_param.c_str() ||
        end != distance_param.c_str() + distance_param.size()) {
      return false;
    }
    butteraugli_target_ = butteraugli_target;

    // full hf asymmetry at high distance
    static const double kHighDistance = 2.5;

    // no hf asymmetry at low distance
    static const double kLowDistance = 0.6;

    if (butteraugli_target_ >= kHighDistance) {
      ba_params_.hf_asymmetry = args_.ba_params.hf_asymmetry;
    } else if (butteraugli_target_ >= kLowDistance) {
      float w =
          (butteraugli_target_ - kLowDistance) / (kHighDistance - kLowDistance);
      ba_params_.hf_asymmetry =
          args_.ba_params.hf_asymmetry * w + 1.0f * (1.0f - w);
    } else {
      ba_params_.hf_asymmetry = 1.0f;
    }
    return true;
  } else if (param[0] == 'r') {
    ba_params_.hf_asymmetry = args_.ba_params.hf_asymmetry;
    bitrate_target_ = strtof(param.substr(1).c_str(), nullptr);
    return true;
  }
  return false;
}

// Low-overhead "codec" for measuring benchmark overhead.
class NoneCodec : public ImageCodec {
 public:
  explicit NoneCodec(const BenchmarkArgs& args) : ImageCodec(args) {}
  Status ParseParam(const std::string& param) override { return true; }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPoolInternal* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    PROFILER_ZONE("NoneCompress");
    const double start = Now();
    // Encode image size so we "decompress" something of the same size, as
    // required by butteraugli.
    const uint32_t xsize = io->xsize();
    const uint32_t ysize = io->ysize();
    compressed->resize(8);
    memcpy(compressed->data(), &xsize, 4);
    memcpy(compressed->data() + 4, &ysize, 4);
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    PROFILER_ZONE("NoneDecompress");
    const double start = Now();
    JXL_ASSERT(compressed.size() == 8);
    uint32_t xsize, ysize;
    memcpy(&xsize, compressed.data(), 4);
    memcpy(&ysize, compressed.data() + 4, 4);
    Image3F image(xsize, ysize);
    ZeroFillImage(&image);
    io->metadata.m.SetFloat32Samples();
    io->metadata.m.color_encoding = ColorEncoding::SRGB();
    io->SetFromImage(std::move(image), io->metadata.m.color_encoding);
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  void GetMoreStats(BenchmarkStats* stats) override {}
};

ImageCodecPtr CreateImageCodec(const std::string& description) {
  std::string name = description;
  std::string parameters = "";
  size_t colon = description.find(':');
  if (colon < description.size()) {
    name = description.substr(0, colon);
    parameters = description.substr(colon + 1);
  }
  ImageCodecPtr result;
  if (name == "jxl") {
    result.reset(CreateNewJxlCodec(*Args()));
#if !defined(__wasm__)
  } else if (name == "custom") {
    result.reset(CreateNewCustomCodec(*Args()));
#endif
#ifdef JPEGXL_ENABLE_JPEG
  } else if (name == "jpeg") {
    result.reset(CreateNewJPEGCodec(*Args()));
#endif  // BENCHMARK_JPEG
#if JPEGXL_ENABLE_APNG
  } else if (name == "png") {
    result.reset(CreateNewPNGCodec(*Args()));
#endif
  } else if (name == "none") {
    result.reset(new NoneCodec(*Args()));
#ifdef BENCHMARK_WEBP
  } else if (name == "webp") {
    result.reset(CreateNewWebPCodec(*Args()));
#endif  // BENCHMARK_WEBP
#ifdef BENCHMARK_AVIF
  } else if (name == "avif") {
    result.reset(CreateNewAvifCodec(*Args()));
#endif  // BENCHMARK_AVIF
  } else {
    JXL_ABORT("Unknown image codec: %s", name.c_str());
  }
  result->set_description(description);
  if (!parameters.empty()) result->ParseParameters(parameters);
  return result;
}

}  // namespace jxl
