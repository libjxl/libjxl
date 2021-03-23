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
#include "tools/benchmark/benchmark_codec_jpeg.h"

#include <stddef.h>
#include <stdio.h>
// After stddef/stdio
#include <jpeglib.h>
#include <stdint.h>
#include <string.h>

#include <numeric>  // partial_sum
#include <string>

#include "lib/extras/codec_jpg.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/base/time.h"
#include "lib/jxl/codec_in_out.h"
#include "tools/cmdline.h"

#ifdef MEMORY_SANITIZER
#include "sanitizer/msan_interface.h"
#endif

namespace jxl {

namespace {

struct JPEGArgs {
  JpegEncoder encoder = JpegEncoder::kLibJpeg;
  YCbCrChromaSubsampling chroma_subsampling;
};

JPEGArgs* const jpegargs = new JPEGArgs;

bool ParseChromaSubsampling(const char* param,
                            YCbCrChromaSubsampling* subsampling) {
  std::vector<std::pair<
      std::string, std::pair<std::array<uint8_t, 3>, std::array<uint8_t, 3>>>>
      options = {{"444", {{1, 1, 1}, {1, 1, 1}}},
                 {"420", {{2, 1, 1}, {2, 1, 1}}},
                 {"422", {{2, 1, 1}, {1, 1, 1}}},
                 {"440", {{1, 1, 1}, {2, 1, 1}}}};
  for (const auto& option : options) {
    if (param == option.first) {
      JXL_CHECK(subsampling->Set(option.second.first.data(),
                                 option.second.second.data()));
      return true;
    }
  }
  return false;
}

}  // namespace

Status AddCommandLineOptionsJPEGCodec(BenchmarkArgs* args) {
  args->cmdline.AddOptionValue(
      '\0', "chroma_subsampling", "444/422/420/411",
      "default JPEG chroma subsampling (default: 444).",
      &jpegargs->chroma_subsampling, &ParseChromaSubsampling);
  return true;
}

class JPEGCodec : public ImageCodec {
 public:
  explicit JPEGCodec(const BenchmarkArgs& args) : ImageCodec(args) {
    encoder_ = jpegargs->encoder;
    chroma_subsampling_ = jpegargs->chroma_subsampling;
  }

  Status ParseParam(const std::string& param) override {
    if (ImageCodec::ParseParam(param)) {
      return true;
    }
    if (param == "sjpeg") {
      encoder_ = JpegEncoder::kSJpeg;
      return true;
    }
    if (param.compare(0, 3, "yuv") == 0) {
      if (param.size() != 6) return false;
      return ParseChromaSubsampling(param.c_str() + 3, &chroma_subsampling_);
    }
    return false;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPoolInternal* pool, PaddedBytes* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    const double start = Now();
    JXL_RETURN_IF_ERROR(EncodeImageJPG(io, encoder_,
                                       static_cast<int>(std::round(q_target_)),
                                       chroma_subsampling_, pool, compressed));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    double elapsed_deinterleave;
    const double start = Now();
    JXL_RETURN_IF_ERROR(
        DecodeImageJPG(compressed, pool, io, &elapsed_deinterleave));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start - elapsed_deinterleave);
    return true;
  }

 protected:
  JpegEncoder encoder_;
  YCbCrChromaSubsampling chroma_subsampling_;
};

ImageCodec* CreateNewJPEGCodec(const BenchmarkArgs& args) {
  return new JPEGCodec(args);
}

}  // namespace jxl
