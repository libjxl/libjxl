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
#include "lib/extras/dec/jxl.h"
#include "lib/extras/decode_jpeg.h"
#include "lib/extras/enc/jpg.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/encode_jpeg.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "tools/benchmark/benchmark_utils.h"
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
    if (param == "sjpeg" || param == "libjxl" ||
        param.find("cjpeg") != std::string::npos) {
      jpeg_encoder_ = param;
      return true;
    }
    if (param == "djxl8") {
      use_jxl_decoder_ = true;
      jxl_decoder_data_type_ = JXL_TYPE_UINT8;
      return true;
    }
    if (param == "djxl16") {
      use_jxl_decoder_ = true;
      jxl_decoder_data_type_ = JXL_TYPE_UINT16;
      return true;
    }
    if (param.compare(0, 3, "yuv") == 0) {
      if (param.size() != 6) return false;
      chroma_subsampling_ = param.substr(3);
      return true;
    }
    if (param.substr(0, 2) == "bd") {
      bitdepth_ = strtol(param.substr(2).c_str(), nullptr, 10);
      return true;
    }
    if (param.substr(0, 2) == "nr") {
      normalize_bitrate_ = true;
      return true;
    }
    if (param[0] == 'p') {
      progressive_id_ = strtol(param.substr(1).c_str(), nullptr, 10);
      return true;
    }
    return false;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPoolInternal* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    if (jpeg_encoder_.find("cjpeg") != std::string::npos) {
// Not supported on Windows due to Linux-specific functions.
// Not supported in Android NDK before API 28.
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__) && \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 28)
      const std::string basename = GetBaseName(filename);
      TemporaryFile in_file(basename, "pnm");
      TemporaryFile encoded_file(basename, "jpg");
      std::string in_filename, encoded_filename;
      JXL_RETURN_IF_ERROR(in_file.GetFileName(&in_filename));
      JXL_RETURN_IF_ERROR(encoded_file.GetFileName(&encoded_filename));
      const size_t bits = io->metadata.m.bit_depth.bits_per_sample;
      ColorEncoding c_enc = io->Main().c_current();
      JXL_RETURN_IF_ERROR(EncodeToFile(*io, c_enc, bits, in_filename, pool));
      std::string compress_command = jpeg_encoder_;
      std::vector<std::string> arguments;
      arguments.push_back("-outfile");
      arguments.push_back(encoded_filename);
      arguments.push_back("-quality");
      arguments.push_back(std::to_string(static_cast<int>(q_target_)));
      arguments.push_back("-sample");
      if (chroma_subsampling_ == "444") {
        arguments.push_back("1x1");
      } else if (chroma_subsampling_ == "420") {
        arguments.push_back("2x2");
      } else {
        return JXL_FAILURE("Unsupported chroma subsampling");
      }
      arguments.push_back("-optimize");
      arguments.push_back(in_filename);
      const double start = Now();
      JXL_RETURN_IF_ERROR(RunCommand(compress_command, arguments, false));
      const double end = Now();
      speed_stats->NotifyElapsed(end - start);
      return ReadFile(encoded_filename, compressed);
#else
      return JXL_FAILURE("Not supported on this build");
#endif
    }
    double elapsed = 0.0;
    if (jpeg_encoder_ != "libjxl" || normalize_bitrate_) {
      extras::PackedPixelFile ppf;
      JxlPixelFormat format = {0, JXL_TYPE_UINT8, JXL_BIG_ENDIAN, 0};
      JXL_RETURN_IF_ERROR(ConvertCodecInOutToPackedPixelFile(
          *io, format, io->metadata.m.color_encoding, pool, &ppf));
      extras::EncodedImage encoded;
      std::unique_ptr<extras::Encoder> encoder = extras::GetJPEGEncoder();
      std::ostringstream os;
      os << static_cast<int>(std::round(q_target_));
      encoder->SetOption("q", os.str());
      std::string jpeg_encoder = normalize_bitrate_ ? "libjpeg" : jpeg_encoder_;
      encoder->SetOption("jpeg_encoder", jpeg_encoder);
      encoder->SetOption("chroma_subsampling", chroma_subsampling_);
      if (progressive_id_ >= 0) {
        encoder->SetOption("progressive", std::to_string(progressive_id_));
      }
      const double start = Now();
      JXL_RETURN_IF_ERROR(encoder->Encode(ppf, &encoded, pool));
      const double end = Now();
      elapsed = end - start;
      *compressed = encoded.bitstreams.back();
    }
    if (jpeg_encoder_ == "libjxl") {
      size_t target_size = normalize_bitrate_ ? compressed->size() : 0;
      compressed->clear();
      const double start = Now();
      extras::JpegSettings settings;
      settings.xyb = true;
      settings.distance = butteraugli_target_;
      settings.target_size = target_size;
      JXL_RETURN_IF_ERROR(
          extras::EncodeJpeg(io->Main(), settings, pool, compressed));
      const double end = Now();
      elapsed = end - start;
    }
    speed_stats->NotifyElapsed(elapsed);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    extras::PackedPixelFile ppf;
    if (use_jxl_decoder_) {
      std::vector<uint8_t> jpeg_bytes(compressed.data(),
                                      compressed.data() + compressed.size());
      const double start = Now();
      JXL_RETURN_IF_ERROR(
          extras::DecodeJpeg(jpeg_bytes, jxl_decoder_data_type_, pool, &ppf));
      const double end = Now();
      speed_stats->NotifyElapsed(end - start);
    } else {
      const double start = Now();
      JXL_RETURN_IF_ERROR(DecodeImageJPG(compressed, extras::ColorHints(),
                                         SizeConstraints(), bitdepth_, &ppf));
      const double end = Now();
      speed_stats->NotifyElapsed(end - start);
    }
    JXL_RETURN_IF_ERROR(ConvertPackedPixelFileToCodecInOut(ppf, pool, io));
    return true;
  }

 protected:
  bool normalize_bitrate_ = false;
  std::string jpeg_encoder_;
  std::string chroma_subsampling_;
  bool use_jxl_decoder_ = false;
  int progressive_id_ = -1;
  JxlDataType jxl_decoder_data_type_ = JXL_TYPE_UINT8;
  size_t bitdepth_ = 8;
};

ImageCodec* CreateNewJPEGCodec(const BenchmarkArgs& args) {
  return new JPEGCodec(args);
}

}  // namespace jxl
