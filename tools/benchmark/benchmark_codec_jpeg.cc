// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "tools/benchmark/benchmark_codec_jpeg.h"

#include <stddef.h>
#include <stdio.h>
// After stddef/stdio
#include <stdint.h>
#include <string.h>

#include <numeric>  // partial_sum
#include <string>

#include "lib/extras/dec/jpg.h"
#include "lib/extras/enc/jpg.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "tools/cmdline.h"

using jxl::extras::JpegEncoder;

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
      options = {{"444", {{{1, 1, 1}}, {{1, 1, 1}}}},
                 {"420", {{{2, 1, 1}}, {{2, 1, 1}}}},
                 {"422", {{{2, 1, 1}}, {{1, 1, 1}}}},
                 {"440", {{{1, 1, 1}}, {{2, 1, 1}}}}};
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
                  ThreadPoolInternal* pool, std::vector<uint8_t>* compressed,
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
    extras::PackedPixelFile ppf;
    const double start = Now();
    JXL_RETURN_IF_ERROR(DecodeImageJPG(compressed, extras::ColorHints(),
                                       SizeConstraints(), &ppf));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    JXL_RETURN_IF_ERROR(ConvertPackedPixelFileToCodecInOut(ppf, pool, io));
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
