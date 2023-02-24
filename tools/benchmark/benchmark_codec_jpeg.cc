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

#if JPEGXL_ENABLE_JPEGLI
#include "lib/extras/dec/jpegli.h"
#endif
#include "lib/extras/dec/jpg.h"
#if JPEGXL_ENABLE_JPEGLI
#include "lib/extras/enc/jpegli.h"
#endif
#include "lib/extras/enc/jpg.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/extras/time.h"
#if JPEGXL_ENABLE_JPEGLI
#include "lib/jpegli/encode.h"
#endif
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/image_bundle.h"
#include "tools/benchmark/benchmark_utils.h"
#include "tools/cmdline.h"

namespace jxl {

class JPEGCodec : public ImageCodec {
 public:
  explicit JPEGCodec(const BenchmarkArgs& args) : ImageCodec(args) {}

  Status ParseParam(const std::string& param) override {
    if (param[0] == 'q' && ImageCodec::ParseParam(param)) {
      enc_quality_set_ = true;
      return true;
    }
    if (ImageCodec::ParseParam(param)) {
      return true;
    }
    if (param == "sjpeg" || param.find("cjpeg") != std::string::npos) {
      jpeg_encoder_ = param;
      return true;
    }
#if JPEGXL_ENABLE_JPEGLI
    if (param == "enc-jpegli") {
      jpeg_encoder_ = "jpegli";
      return true;
    }
#endif
    if (param.compare(0, 3, "yuv") == 0) {
      if (param.size() != 6) return false;
      chroma_subsampling_ = param.substr(3);
      return true;
    }
    if (param[0] == 'p') {
      progressive_id_ = strtol(param.substr(1).c_str(), nullptr, 10);
      return true;
    }
#if JPEGXL_ENABLE_JPEGLI
    if (param == "xyb") {
      xyb_mode_ = true;
      return true;
    }
    if (param == "std") {
      use_std_tables_ = true;
      return true;
    }
    if (param[0] == 'Q') {
      libjpeg_quality_ = strtol(param.substr(1).c_str(), nullptr, 10);
      return true;
    }
    if (param.compare(0, 3, "YUV") == 0) {
      if (param.size() != 6) return false;
      libjpeg_chroma_subsampling_ = param.substr(3);
      return true;
    }
    if (param == "dec-jpegli") {
      jpeg_decoder_ = "jpegli";
      return true;
    }
    if (param.substr(0, 2) == "bd") {
      bitdepth_ = strtol(param.substr(2).c_str(), nullptr, 10);
      return true;
    }
    if (param == "noaq") {
      enable_adaptive_quant_ = false;
      return true;
    }
#endif
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
      } else if (!chroma_subsampling_.empty()) {
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
    if (jpeg_encoder_ == "jpegli") {
#if JPEGXL_ENABLE_JPEGLI
      extras::PackedPixelFile ppf;
      size_t bits_per_sample = io->metadata.m.bit_depth.bits_per_sample;
      JxlPixelFormat format = {
          0,  // num_channels is ignored by the converter
          bits_per_sample <= 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16,
          JXL_BIG_ENDIAN, 0};
      JXL_RETURN_IF_ERROR(ConvertCodecInOutToPackedPixelFile(
          *io, format, io->metadata.m.color_encoding, pool, &ppf));
      extras::JpegSettings settings;
      settings.xyb = xyb_mode_;
      if (!xyb_mode_) {
        settings.use_std_quant_tables = use_std_tables_;
      }
      if (enc_quality_set_) {
        settings.distance = jpegli_quality_to_distance(q_target_);
      } else {
        settings.distance = butteraugli_target_;
      }
      if (progressive_id_ >= 0) {
        settings.progressive_level = progressive_id_;
      }
      settings.chroma_subsampling = chroma_subsampling_;
      settings.use_adaptive_quantization = enable_adaptive_quant_;
      settings.libjpeg_quality = libjpeg_quality_;
      settings.libjpeg_chroma_subsampling = libjpeg_chroma_subsampling_;
      const double start = Now();
      JXL_RETURN_IF_ERROR(extras::EncodeJpeg(ppf, settings, pool, compressed));
      const double end = Now();
      elapsed = end - start;
#endif
    } else {
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
      if (!chroma_subsampling_.empty()) {
        encoder->SetOption("chroma_subsampling", chroma_subsampling_);
      }
      if (progressive_id_ >= 0) {
        encoder->SetOption("progressive", std::to_string(progressive_id_));
      }
      const double start = Now();
      JXL_RETURN_IF_ERROR(encoder->Encode(ppf, &encoded, pool));
      const double end = Now();
      elapsed = end - start;
      *compressed = encoded.bitstreams.back();
    }
    speed_stats->NotifyElapsed(elapsed);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    extras::PackedPixelFile ppf;
    if (jpeg_decoder_ == "jpegli") {
#if JPEGXL_ENABLE_JPEGLI
      std::vector<uint8_t> jpeg_bytes(compressed.data(),
                                      compressed.data() + compressed.size());
      const double start = Now();
      JxlDataType data_type = bitdepth_ > 8 ? JXL_TYPE_UINT16 : JXL_TYPE_UINT8;
      JXL_RETURN_IF_ERROR(
          extras::DecodeJpeg(jpeg_bytes, data_type, pool, &ppf));
      const double end = Now();
      speed_stats->NotifyElapsed(end - start);
#endif
    } else {
      const double start = Now();
      JXL_RETURN_IF_ERROR(DecodeImageJPG(compressed, extras::ColorHints(),
                                         SizeConstraints(), &ppf));
      const double end = Now();
      speed_stats->NotifyElapsed(end - start);
    }
    JXL_RETURN_IF_ERROR(ConvertPackedPixelFileToCodecInOut(ppf, pool, io));
    return true;
  }

 protected:
  // JPEG encoder and its parameters
  std::string jpeg_encoder_ = "libjpeg";
  std::string chroma_subsampling_;
  int progressive_id_ = -1;
  bool enc_quality_set_ = false;
#if JPEGXL_ENABLE_JPEGLI
  bool xyb_mode_ = false;
  bool use_std_tables_ = false;
  int libjpeg_quality_ = 0;
  std::string libjpeg_chroma_subsampling_;
#endif
  // JPEG decoder and its parameters
  std::string jpeg_decoder_ = "libjpeg";
#if JPEGXL_ENABLE_JPEGLI
  size_t bitdepth_ = 8;
  bool enable_adaptive_quant_ = true;
#endif
};

ImageCodec* CreateNewJPEGCodec(const BenchmarkArgs& args) {
  return new JPEGCodec(args);
}

}  // namespace jxl
