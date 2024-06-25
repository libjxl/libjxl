// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/benchmark/benchmark_codec.h"

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/packed_image_convert.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "tools/benchmark/benchmark_args.h"
#include "tools/benchmark/benchmark_codec_custom.h"
#include "tools/benchmark/benchmark_codec_jpeg.h"
#include "tools/benchmark/benchmark_codec_jxl.h"
#include "tools/benchmark/benchmark_stats.h"
#include "tools/cmdline.h"
#include "tools/no_memory_manager.h"
#include "tools/speed_stats.h"
#include "tools/thread_pool_internal.h"

#ifdef BENCHMARK_PNG
#include "tools/benchmark/benchmark_codec_png.h"
#endif  // BENCHMARK_PNG

#ifdef BENCHMARK_WEBP
#include "tools/benchmark/benchmark_codec_webp.h"
#endif  // BENCHMARK_WEBP

#ifdef BENCHMARK_AVIF
#include "tools/benchmark/benchmark_codec_avif.h"
#endif  // BENCHMARK_AVIF

namespace jpegxl {
namespace tools {

using ::jxl::Image3F;
using ::jxl::Status;

Status ImageCodec::ParseParameters(const std::string& parameters) {
  params_ = parameters;
  std::vector<std::string> parts = SplitString(parameters, ':');
  for (const auto& part : parts) {
    if (!ParseParam(part)) {
      return JXL_FAILURE("Invalid parameter %s", part.c_str());
    }
  }
  return true;
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
    return true;
  } else if (param[0] == 'r') {
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

  Status Compress(const std::string& filename, const PackedPixelFile& ppf,
                  ThreadPool* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    const double start = jxl::Now();
    // Encode image size so we "decompress" something of the same size, as
    // required by butteraugli.
    const uint32_t xsize = ppf.xsize();
    const uint32_t ysize = ppf.ysize();
    compressed->resize(8);
    memcpy(compressed->data(), &xsize, 4);
    memcpy(compressed->data() + 4, &ysize, 4);
    const double end = jxl::Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed, ThreadPool* pool,
                    PackedPixelFile* ppf,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    CodecInOut io{jpegxl::tools::NoMemoryManager()};
    JXL_RETURN_IF_ERROR(
        Decompress(filename, compressed, pool, &io, speed_stats));
    JxlPixelFormat format{0, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0};
    return jxl::extras::ConvertCodecInOutToPackedPixelFile(
        io, format, io.Main().c_current(), pool, ppf);
  };

  static Status Decompress(const std::string& filename,
                           const Span<const uint8_t> compressed,
                           ThreadPool* pool, CodecInOut* io,
                           jpegxl::tools::SpeedStats* speed_stats) {
    const double start = jxl::Now();
    JXL_ENSURE(compressed.size() == 8);
    uint32_t xsize;
    uint32_t ysize;
    memcpy(&xsize, compressed.data(), 4);
    memcpy(&ysize, compressed.data() + 4, 4);
    JXL_ASSIGN_OR_RETURN(
        Image3F image,
        Image3F::Create(jpegxl::tools::NoMemoryManager(), xsize, ysize));
    ZeroFillImage(&image);
    io->metadata.m.SetFloat32Samples();
    io->metadata.m.color_encoding = ColorEncoding::SRGB();
    JXL_RETURN_IF_ERROR(
        io->SetFromImage(std::move(image), io->metadata.m.color_encoding));
    const double end = jxl::Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  void GetMoreStats(BenchmarkStats* stats) override {}
};

ImageCodecPtr CreateImageCodec(const std::string& description,
                               JxlMemoryManager* memory_manager) {
  std::string name = description;
  std::string parameters;
  size_t colon = description.find(':');
  if (colon < description.size()) {
    name = description.substr(0, colon);
    parameters = description.substr(colon + 1);
  }
  ImageCodecPtr result;
  if (name == "jxl") {
    result.reset(CreateNewJxlCodec(*Args(), memory_manager));
#if !defined(__wasm__)
  } else if (name == "custom") {
    result.reset(CreateNewCustomCodec(*Args()));
#endif
  } else if (name == "jpeg") {
    result.reset(CreateNewJPEGCodec(*Args()));
#ifdef BENCHMARK_PNG
  } else if (name == "png") {
    result.reset(CreateNewPNGCodec(*Args()));
#endif  // BENCHMARK_PNG
  } else if (name == "none") {
    result = jxl::make_unique<NoneCodec>(*Args());
#ifdef BENCHMARK_WEBP
  } else if (name == "webp") {
    result.reset(CreateNewWebPCodec(*Args()));
#endif  // BENCHMARK_WEBP
#ifdef BENCHMARK_AVIF
  } else if (name == "avif") {
    result.reset(CreateNewAvifCodec(*Args()));
#endif  // BENCHMARK_AVIF
  }
  if (!result.get()) {
    fprintf(stderr, "Unknown image codec: %s", name.c_str());
    JPEGXL_TOOLS_CHECK(false);
  }
  result->set_description(description);
  if (!parameters.empty()) {
    JPEGXL_TOOLS_CHECK(result->ParseParameters(parameters));
  }
  return result;
}

}  // namespace tools
}  // namespace jpegxl
