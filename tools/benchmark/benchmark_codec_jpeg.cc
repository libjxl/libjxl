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

namespace jxl {

namespace {

struct JPEGArgs {
  std::string jpeg_encoder = "libjpeg";
  std::string chroma_subsampling = "444";
};

JPEGArgs* const jpegargs = new JPEGArgs;

}  // namespace

Status AddCommandLineOptionsJPEGCodec(BenchmarkArgs* args) {
  args->cmdline.AddOptionValue(
      '\0', "chroma_subsampling", "444/422/420/411",
      "default JPEG chroma subsampling (default: 444).",
      &jpegargs->chroma_subsampling, &jpegxl::tools::ParseString);
  return true;
}

class JPEGCodec : public ImageCodec {
 public:
  explicit JPEGCodec(const BenchmarkArgs& args) : ImageCodec(args) {
    jpeg_encoder_ = jpegargs->jpeg_encoder;
    chroma_subsampling_ = jpegargs->chroma_subsampling;
  }

  Status ParseParam(const std::string& param) override {
    if (ImageCodec::ParseParam(param)) {
      return true;
    }
    if (param == "sjpeg") {
      jpeg_encoder_ = param;
      return true;
    }
    if (param.compare(0, 3, "yuv") == 0) {
      if (param.size() != 6) return false;
      chroma_subsampling_ = param.substr(3);
      return true;
    }
    return false;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPoolInternal* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    extras::PackedPixelFile ppf;
    JxlPixelFormat format = {0, JXL_TYPE_UINT8, JXL_BIG_ENDIAN, 0};
    JXL_RETURN_IF_ERROR(ConvertCodecInOutToPackedPixelFile(
        *io, format, io->metadata.m.color_encoding, pool, &ppf));
    extras::EncodedImage encoded;
    std::unique_ptr<extras::Encoder> encoder = extras::GetJPEGEncoder();
    std::ostringstream os;
    os << static_cast<int>(std::round(q_target_));
    encoder->SetOption("q", os.str());
    encoder->SetOption("jpeg_encoder", jpeg_encoder_);
    encoder->SetOption("chroma_subsampling", chroma_subsampling_);
    const double start = Now();
    JXL_RETURN_IF_ERROR(encoder->Encode(ppf, &encoded, pool));
    const double end = Now();
    *compressed = encoded.bitstreams.back();
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
  std::string jpeg_encoder_;
  std::string chroma_subsampling_;
};

ImageCodec* CreateNewJPEGCodec(const BenchmarkArgs& args) {
  return new JPEGCodec(args);
}

}  // namespace jxl
