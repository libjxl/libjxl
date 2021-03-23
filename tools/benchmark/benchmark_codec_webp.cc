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
#include "tools/benchmark/benchmark_codec_webp.h"

#include <stdint.h>
#include <string.h>
#include <webp/decode.h>
#include <webp/encode.h>

#include <string>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/base/time.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

#ifdef MEMORY_SANITIZER
#include "sanitizer/msan_interface.h"
#endif

namespace jxl {

// Sets image data from 8-bit sRGB pixel array in bytes.
// Amount of input bytes per pixel must be:
// (is_gray ? 1 : 3) + (has_alpha ? 1 : 0)
Status FromSRGB(const size_t xsize, const size_t ysize, const bool is_gray,
                const bool has_alpha, const bool alpha_is_premultiplied,
                const bool is_16bit, const JxlEndianness endianness,
                const uint8_t* pixels, const uint8_t* end, ThreadPool* pool,
                ImageBundle* ib) {
  const ColorEncoding& c = ColorEncoding::SRGB(is_gray);
  const size_t bits_per_sample = (is_16bit ? 2 : 1) * kBitsPerByte;
  const Span<const uint8_t> span(pixels, end - pixels);
  return ConvertFromExternal(span, xsize, ysize, c, has_alpha,
                             alpha_is_premultiplied, bits_per_sample,
                             endianness, /*flipped_y=*/false, pool, ib);
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
        near_lossless_quality_ = ParseIntParam(param, 0, 99);
      } else {
        quality_ = ParseIntParam(param, 1, 100);
      }
      return true;
    } else if (ImageCodec::ParseParam(param)) {
      return true;
    } else if (param == "ll") {
      lossless_ = true;
      JXL_CHECK(!near_lossless_);
      return true;
    } else if (param == "nl") {
      near_lossless_ = true;
      JXL_CHECK(!lossless_);
      return true;
    } else if (param[0] == 'm') {
      method_ = ParseIntParam(param, 1, 6);
      return true;
    }
    return false;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPoolInternal* pool, PaddedBytes* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    const double start = Now();
    const ImageBundle& ib = io->Main();

    const ImageF* alpha = ib.HasAlpha() ? &ib.alpha() : nullptr;
    if (ib.HasAlpha() && ib.metadata()->GetAlphaBits() > 8) {
      return JXL_FAILURE("WebP alpha must be 8-bit");
    }

    // TODO: ib.CopyToSRGB to Image3B assert-fails if 16-bit alpha. Fix that,
    // since it may be bug in external_image (alpha isn't requested here).
    Image3B srgb;
    JXL_RETURN_IF_ERROR(ib.CopyToSRGB(Rect(ib), &srgb, pool));

    if (lossless_ || near_lossless_) {
      // The lossless codec does not support 16-bit channels.
      // Color models are currently not supported here and the sRGB 8-bit
      // conversion causes loss due to clipping.
      if (!ib.IsSRGB() || ib.metadata()->bit_depth.bits_per_sample > 8 ||
          ib.metadata()->bit_depth.exponent_bits_per_sample > 0) {
        return JXL_FAILURE("%s: webp:ll/nl requires 8-bit sRGB",
                           filename.c_str());
      }
      JXL_RETURN_IF_ERROR(CompressInternal(srgb, alpha, 100, compressed));
    } else if (bitrate_target_ > 0.0) {
      int quality_bad = 100;
      int quality_good = 92;
      size_t target_size = srgb.xsize() * srgb.ysize() * bitrate_target_ / 8.0;
      while (quality_good > 0 &&
             CompressInternal(srgb, alpha, quality_good, compressed) &&
             compressed->size() > target_size) {
        quality_bad = quality_good;
        quality_good -= 8;
      }
      if (quality_good <= 0) quality_good = 1;
      while (quality_good + 1 < quality_bad) {
        int quality = (quality_bad + quality_good) / 2;
        if (!CompressInternal(srgb, alpha, quality, compressed)) {
          break;
        }
        if (compressed->size() <= target_size) {
          quality_good = quality;
        } else {
          quality_bad = quality;
        }
      }
      JXL_RETURN_IF_ERROR(
          CompressInternal(srgb, alpha, quality_good, compressed));
    } else if (quality_ > 0) {
      JXL_RETURN_IF_ERROR(CompressInternal(srgb, alpha, quality_, compressed));
    } else {
      return false;
    }
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
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
    const double start = Now();
    if (WebPDecode(webp_data, webp_size, &config) != VP8_STATUS_OK) {
      return JXL_FAILURE("WebPDecode failed");
    }
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    JXL_CHECK(buf->u.RGBA.stride == buf->width * 4);

    const bool is_gray = false;
    const bool has_alpha = true;
    const uint8_t* data_begin = &buf->u.RGBA.rgba[0];
    const uint8_t* data_end = data_begin + buf->width * buf->height * 4;
#ifdef MEMORY_SANITIZER
    // The image data is initialized by libwebp, which we are not instrumenting
    // with msan.
    __msan_unpoison(data_begin, data_end - data_begin);
#endif
    if (io->metadata.m.color_encoding.IsGray() != is_gray) {
      // TODO(lode): either ensure is_gray matches what the color profile says,
      // or set a correct color profile, e.g.
      // io->metadata.m.color_encoding = ColorEncoding::SRGB(is_gray);
      // Return a standard failure becuase SetFromSRGB triggers a fatal assert
      // for this instead.
      return JXL_FAILURE("Color profile is-gray mismatch");
    }
    io->metadata.m.SetAlphaBits(8);
    const Status ok =
        FromSRGB(buf->width, buf->height, is_gray, has_alpha,
                 /*alpha_is_premultiplied=*/false, /*is_16bit=*/false,
                 JXL_LITTLE_ENDIAN, data_begin, data_end, pool, &io->Main());
    WebPFreeDecBuffer(buf);
    JXL_RETURN_IF_ERROR(ok);
    io->dec_pixels = buf->width * buf->height;
    return true;
  }

 private:
  static int WebPStringWrite(const uint8_t* data, size_t data_size,
                             const WebPPicture* const picture) {
    if (data_size) {
      PaddedBytes* const out = static_cast<PaddedBytes*>(picture->custom_ptr);
      const size_t pos = out->size();
      out->resize(pos + data_size);
      memcpy(out->data() + pos, data, data_size);
    }
    return 1;
  }

  static void Import(const Image3B& srgb, WebPPicture* pic) {
    const size_t xsize = srgb.xsize();
    const size_t ysize = srgb.ysize();
    std::vector<uint8_t> rgb(xsize * ysize * 3);
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* JXL_RESTRICT row0 = srgb.ConstPlaneRow(0, y);
      const uint8_t* JXL_RESTRICT row1 = srgb.ConstPlaneRow(1, y);
      const uint8_t* JXL_RESTRICT row2 = srgb.ConstPlaneRow(2, y);
      uint8_t* const JXL_RESTRICT row_rgb = &rgb[y * xsize * 3];
      for (size_t x = 0; x < xsize; ++x) {
        row_rgb[3 * x + 0] = row0[x];
        row_rgb[3 * x + 1] = row1[x];
        row_rgb[3 * x + 2] = row2[x];
      }
    }
    WebPPictureImportRGB(pic, &rgb[0], 3 * srgb.xsize());
  }

  static void Import(const Image3B& srgb, const ImageF& alpha,
                     WebPPicture* pic) {
    const size_t xsize = srgb.xsize();
    const size_t ysize = srgb.ysize();
    std::vector<uint8_t> rgba(xsize * ysize * 4);
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* JXL_RESTRICT row0 = srgb.ConstPlaneRow(0, y);
      const uint8_t* JXL_RESTRICT row1 = srgb.ConstPlaneRow(1, y);
      const uint8_t* JXL_RESTRICT row2 = srgb.ConstPlaneRow(2, y);
      const float* JXL_RESTRICT rowa = alpha.ConstRow(y);
      uint8_t* const JXL_RESTRICT row_rgba = &rgba[y * xsize * 4];
      for (size_t x = 0; x < xsize; ++x) {
        row_rgba[4 * x + 0] = row0[x];
        row_rgba[4 * x + 1] = row1[x];
        row_rgba[4 * x + 2] = row2[x];
        row_rgba[4 * x + 3] = rowa[x] * 255 + .5f;
      }
    }
    WebPPictureImportRGBA(pic, &rgba[0], 4 * srgb.xsize());
  }

  Status CompressInternal(const Image3B& srgb, const ImageF* alpha, int quality,
                          PaddedBytes* compressed) {
    *compressed = PaddedBytes();
    WebPConfig config;
    WebPConfigInit(&config);
    JXL_ASSERT(!lossless_ || !near_lossless_);  // can't have both
    config.lossless = lossless_;
    config.quality = quality;
    config.method = method_;
#if WEBP_ENCODER_ABI_VERSION >= 0x020a
    config.near_lossless = near_lossless_ ? near_lossless_quality_ : 100;
#else
    if (near_lossless_) {
      JXL_WARNING("Near lossless not supported by this WebP version");
    }
#endif
    JXL_CHECK(WebPValidateConfig(&config));

    WebPPicture pic;
    WebPPictureInit(&pic);
    pic.width = static_cast<int>(srgb.xsize());
    pic.height = static_cast<int>(srgb.ysize());
    pic.writer = &WebPStringWrite;
    if (lossless_ || near_lossless_) pic.use_argb = 1;
    pic.custom_ptr = compressed;

    if (alpha == nullptr) {
      Import(srgb, &pic);
    } else {
      Import(srgb, *alpha, &pic);
    }

    // WebP encoding may fail, for example, if the image is more than 16384
    // pixels high or wide.
    bool ok = WebPEncode(&config, &pic);
    WebPPictureFree(&pic);
#ifdef MEMORY_SANITIZER
    // Compressed image data is initialized by libwebp, which we are not
    // instrumenting with msan.
    __msan_unpoison(compressed->data(), compressed->size());
#endif
    return ok;
  }

  int quality_ = 90;
  bool lossless_ = false;
  bool near_lossless_ = false;
  bool near_lossless_quality_ = 40;  // only used if near_lossless_
  int method_ = 6;                   // smallest, some speed cost
};

ImageCodec* CreateNewWebPCodec(const BenchmarkArgs& args) {
  return new WebPCodec(args);
}

}  // namespace jxl
