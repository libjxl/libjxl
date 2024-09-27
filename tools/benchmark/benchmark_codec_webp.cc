// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "tools/benchmark/benchmark_codec_webp.h"

#include <jxl/cms.h>
#include <jxl/types.h>
#include <webp/decode.h>
#include <webp/encode.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "lib/extras/packed_image_convert.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_external_image.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/enc_image_bundle.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "tools/benchmark/benchmark_args.h"
#include "tools/benchmark/benchmark_codec.h"
#include "tools/no_memory_manager.h"
#include "tools/speed_stats.h"
#include "tools/thread_pool_internal.h"

namespace jpegxl {
namespace tools {

using ::jxl::ImageBundle;
using ::jxl::ImageMetadata;
using ::jxl::ThreadPool;

// Sets image data from 8-bit sRGB pixel array in bytes.
// Amount of input bytes per pixel must be:
// (is_gray ? 1 : 3) + (has_alpha ? 1 : 0)
Status FromSRGB(const size_t xsize, const size_t ysize, const bool is_gray,
                const bool has_alpha, const bool is_16bit,
                const JxlEndianness endianness, const uint8_t* pixels,
                const uint8_t* end, ThreadPool* pool, ImageBundle* ib) {
  const ColorEncoding& c = ColorEncoding::SRGB(is_gray);
  const size_t bits_per_sample = (is_16bit ? 2 : 1) * jxl::kBitsPerByte;
  const uint32_t num_channels = (is_gray ? 1 : 3) + (has_alpha ? 1 : 0);
  JxlDataType data_type = is_16bit ? JXL_TYPE_UINT16 : JXL_TYPE_UINT8;
  JxlPixelFormat format = {num_channels, data_type, endianness, 0};
  const Span<const uint8_t> span(pixels, end - pixels);
  return ConvertFromExternal(span, xsize, ysize, c, bits_per_sample, format,
                             pool, ib);
}

struct WebPArgs {
  // Empty, no WebP-specific args currently.
};

static WebPArgs* const webpargs = new WebPArgs;

Status AddCommandLineOptionsWebPCodec(BenchmarkArgs* args) { return true; }

class WebPCodec : public ImageCodec {
 public:
  explicit WebPCodec(const BenchmarkArgs& args) : ImageCodec(args) {}

  Status ParseParam(const std::string& param) override {
    // Ensure that the 'q' parameter is not used up by ImageCodec.
    if (param[0] == 'q') {
      if (near_lossless_) {
        return ParseIntParam(param, 0, 99, near_lossless_quality_);
      } else {
        return ParseIntParam(param, 1, 100, quality_);
      }
    } else if (ImageCodec::ParseParam(param)) {
      return true;
    } else if (param == "ll") {
      lossless_ = true;
      JXL_ENSURE(!near_lossless_);
      return true;
    } else if (param == "nl") {
      near_lossless_ = true;
      JXL_ENSURE(!lossless_);
      return true;
    } else if (param[0] == 'm') {
      return ParseIntParam(param, 1, 6, method_);
    }
    return false;
  }

  Status Compress(const std::string& filename, const PackedPixelFile& ppf,
                  ThreadPool* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    CodecInOut io{jpegxl::tools::NoMemoryManager()};
    JXL_RETURN_IF_ERROR(
        jxl::extras::ConvertPackedPixelFileToCodecInOut(ppf, pool, &io));
    return Compress(filename, &io, pool, compressed, speed_stats);
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPool* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) {
    const double start = jxl::Now();
    const ImageBundle& ib = io->Main();

    if (ib.HasAlpha() && ib.metadata()->GetAlphaBits() > 8) {
      return JXL_FAILURE("WebP alpha must be 8-bit");
    }

    size_t num_chans = (ib.HasAlpha() ? 4 : 3);
    ImageMetadata metadata = io->metadata.m;
    ImageBundle store(jpegxl::tools::NoMemoryManager(), &metadata);
    const ImageBundle* transformed;
    const ColorEncoding& c_desired = ColorEncoding::SRGB(false);
    JXL_RETURN_IF_ERROR(jxl::TransformIfNeeded(
        ib, c_desired, *JxlGetDefaultCms(), pool, &store, &transformed));
    size_t xsize = ib.oriented_xsize();
    size_t ysize = ib.oriented_ysize();
    size_t stride = xsize * num_chans;
    std::vector<uint8_t> srgb(stride * ysize);
    JXL_RETURN_IF_ERROR(ConvertToExternal(
        *transformed, 8, /*float_out=*/false, num_chans, JXL_BIG_ENDIAN, stride,
        pool, srgb.data(), srgb.size(),
        /*out_callback=*/{}, metadata.GetOrientation()));

    if (lossless_ || near_lossless_) {
      // The lossless codec does not support 16-bit channels.
      // Color models are currently not supported here and the sRGB 8-bit
      // conversion causes loss due to clipping.
      if (!ib.IsSRGB() || ib.metadata()->bit_depth.bits_per_sample > 8 ||
          ib.metadata()->bit_depth.exponent_bits_per_sample > 0) {
        return JXL_FAILURE("%s: webp:ll/nl requires 8-bit sRGB",
                           filename.c_str());
      }
      JXL_RETURN_IF_ERROR(
          CompressInternal(srgb, xsize, ysize, num_chans, 100, compressed));
    } else if (bitrate_target_ > 0.0) {
      int quality_bad = 100;
      int quality_good = 92;
      size_t target_size = xsize * ysize * bitrate_target_ / 8.0;
      while (quality_good > 0 &&
             CompressInternal(srgb, xsize, ysize, num_chans, quality_good,
                              compressed) &&
             compressed->size() > target_size) {
        quality_bad = quality_good;
        quality_good -= 8;
      }
      if (quality_good <= 0) quality_good = 1;
      while (quality_good + 1 < quality_bad) {
        int quality = (quality_bad + quality_good) / 2;
        if (!CompressInternal(srgb, xsize, ysize, num_chans, quality,
                              compressed)) {
          break;
        }
        if (compressed->size() <= target_size) {
          quality_good = quality;
        } else {
          quality_bad = quality;
        }
      }
      JXL_RETURN_IF_ERROR(CompressInternal(srgb, xsize, ysize, num_chans,
                                           quality_good, compressed));
    } else if (quality_ > 0) {
      JXL_RETURN_IF_ERROR(CompressInternal(srgb, xsize, ysize, num_chans,
                                           quality_, compressed));
    } else {
      return false;
    }
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
    JxlPixelFormat format{0, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    return jxl::extras::ConvertCodecInOutToPackedPixelFile(
        io, format, io.Main().c_current(), pool, ppf);
  };

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed, ThreadPool* pool,
                    CodecInOut* io, jpegxl::tools::SpeedStats* speed_stats) {
    WebPDecoderConfig config;
#ifdef MEMORY_SANITIZER
    // config is initialized by libwebp, which we are not instrumenting with
    // msan, therefore we need to initialize it here.
    memset(&config, 0, sizeof(config));
#endif
    JXL_RETURN_IF_ERROR(WebPInitDecoderConfig(&config) == 1);
    config.options.use_threads = 0;
    config.options.dithering_strength = 0;
    config.options.bypass_filtering = 0;
    config.options.no_fancy_upsampling = 0;
    WebPDecBuffer* const buf = &config.output;
    buf->colorspace = MODE_RGBA;
    const uint8_t* webp_data = compressed.data();
    const int webp_size = compressed.size();
    const double start = jxl::Now();
    if (WebPDecode(webp_data, webp_size, &config) != VP8_STATUS_OK) {
      return JXL_FAILURE("WebPDecode failed");
    }
    const double end = jxl::Now();
    speed_stats->NotifyElapsed(end - start);
    JXL_ENSURE(buf->u.RGBA.stride == buf->width * 4);

    const bool is_gray = false;
    const bool has_alpha = true;
    const uint8_t* data_begin = &buf->u.RGBA.rgba[0];
    const uint8_t* data_end = data_begin + buf->width * buf->height * 4;
    // The image data is initialized by libwebp, which we are not instrumenting
    // with msan.
    jxl::msan::UnpoisonMemory(data_begin, data_end - data_begin);
    if (io->metadata.m.color_encoding.IsGray() != is_gray) {
      // TODO(lode): either ensure is_gray matches what the color profile says,
      // or set a correct color profile, e.g.
      // io->metadata.m.color_encoding = ColorEncoding::SRGB(is_gray);
      // Return a standard failure because SetFromSRGB triggers a fatal assert
      // for this instead.
      return JXL_FAILURE("Color profile is-gray mismatch");
    }
    io->metadata.m.SetAlphaBits(8);
    JXL_RETURN_IF_ERROR(io->SetSize(buf->width, buf->height));
    const Status ok = FromSRGB(buf->width, buf->height, is_gray, has_alpha,
                               /*is_16bit=*/false, JXL_LITTLE_ENDIAN,
                               data_begin, data_end, pool, &io->Main());
    WebPFreeDecBuffer(buf);
    JXL_RETURN_IF_ERROR(ok);
    return true;
  }

 private:
  static int WebPStringWrite(const uint8_t* data, size_t data_size,
                             const WebPPicture* const picture) {
    if (data_size) {
      std::vector<uint8_t>* const out =
          static_cast<std::vector<uint8_t>*>(picture->custom_ptr);
      const size_t pos = out->size();
      out->resize(pos + data_size);
      memcpy(out->data() + pos, data, data_size);
    }
    return 1;
  }
  Status CompressInternal(const std::vector<uint8_t>& srgb, size_t xsize,
                          size_t ysize, size_t num_chans, int quality,
                          std::vector<uint8_t>* compressed) const {
    compressed->clear();
    WebPConfig config;
    if (!WebPConfigInit(&config)) {
      return JXL_FAILURE("WebPConfigInit failed");
    }
    JXL_ENSURE(!lossless_ || !near_lossless_);  // can't have both
    config.lossless = lossless_ ? 1 : 0;
    config.quality = quality;
    config.method = method_;
#if WEBP_ENCODER_ABI_VERSION >= 0x020a
    config.near_lossless = near_lossless_ ? near_lossless_quality_ : 100;
#else
    if (near_lossless_) {
      JXL_WARNING("Near lossless not supported by this WebP version");
    }
#endif
    JXL_ENSURE(WebPValidateConfig(&config));

    WebPPicture pic;
    if (!WebPPictureInit(&pic)) {
      return JXL_FAILURE("WebPPictureInit failed");
    }
    pic.width = static_cast<int>(xsize);
    pic.height = static_cast<int>(ysize);
    pic.writer = &WebPStringWrite;
    if (lossless_ || near_lossless_) pic.use_argb = 1;
    pic.custom_ptr = compressed;

    if (num_chans == 3) {
      if (!WebPPictureImportRGB(&pic, srgb.data(), 3 * xsize)) {
        return JXL_FAILURE("WebPPictureImportRGB failed");
      }
    } else {
      if (!WebPPictureImportRGBA(&pic, srgb.data(), 4 * xsize)) {
        return JXL_FAILURE("WebPPictureImportRGBA failed");
      }
    }

    // WebP encoding may fail, for example, if the image is more than 16384
    // pixels high or wide.
    bool ok = FROM_JXL_BOOL(WebPEncode(&config, &pic));
    WebPPictureFree(&pic);
    // Compressed image data is initialized by libwebp, which we are not
    // instrumenting with msan.
    jxl::msan::UnpoisonMemory(compressed->data(), compressed->size());
    return ok;
  }

  int quality_ = 90;
  bool lossless_ = false;
  bool near_lossless_ = false;
  int near_lossless_quality_ = 40;   // only used if near_lossless_
  int method_ = 6;                   // smallest, some speed cost
};

ImageCodec* CreateNewWebPCodec(const BenchmarkArgs& args) {
  return new WebPCodec(args);
}

}  // namespace tools
}  // namespace jpegxl
