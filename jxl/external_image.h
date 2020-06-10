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

#ifndef JXL_EXTERNAL_IMAGE_H_
#define JXL_EXTERNAL_IMAGE_H_

// Interleaved image for color transforms and Codec.

#include <stddef.h>
#include <stdint.h>

#include "jxl/base/data_parallel.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/status.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"

namespace jxl {

// Packed (no row padding), interleaved (RGBRGB) u1/u8/u16/f32.
struct PackedImage {
  PackedImage(size_t xsize, size_t ysize, const ColorEncoding& c_current,
              bool has_alpha, bool alpha_is_premultiplied,
              size_t bits_per_alpha, size_t bits_per_sample, bool big_endian,
              bool flipped_y)
      : xsize(xsize),
        ysize(ysize),
        c_current(c_current),
        channels(c_current.Channels() + has_alpha),
        alpha_is_premultiplied(alpha_is_premultiplied),
        bits_per_alpha(bits_per_alpha),
        bits_per_sample(bits_per_sample),
        row_size(xsize * channels * DivCeil(bits_per_sample, kBitsPerByte)),
        big_endian(big_endian),
        flipped_y(flipped_y) {
    if (bits_per_sample == 1) {
      row_size = DivCeil(xsize, kBitsPerByte);
    }
  }

  bool HasAlpha() const { return channels == 2 || channels == 4; }

  // Return whether the passed buffer size in bytes would be enough to hold the
  // PackagedImage data.
  Status ValidBufferSize(size_t buffer_size) const {
    if (ysize && buffer_size / ysize < row_size) {
      return JXL_FAILURE("Buffer size is too small");
    }
    return true;
  }

  size_t xsize;
  size_t ysize;
  ColorEncoding c_current;
  size_t channels;
  bool alpha_is_premultiplied;
  // Per alpha channel value
  size_t bits_per_alpha;
  // Per color channel
  size_t bits_per_sample;
  size_t row_size;
  bool big_endian;
  bool flipped_y;
};

// Packed (no row padding), interleaved (RGBRGB) u1/u8/u16/f32.
class ExternalImage {
 public:
  // Copies from existing interleaved image. Called by decoders. "big_endian"
  // only matters for bits_per_sample > 8 (single-bit are always big endian to
  // match PBM). "end" is the STL-style end of "bytes" for range checks.
  //
  // DEPRECATED, use ::CopyTo instead
  ExternalImage(size_t xsize, size_t ysize, const ColorEncoding& c_current,
                bool has_alpha, bool alpha_is_premultiplied,
                size_t bits_per_alpha, size_t bits_per_sample, bool big_endian,
                const uint8_t* bytes, const uint8_t* end);

  // Copies pixels from rect and converts from c_current to c_desired. Called by
  // encoders and ImageBundle::CopyTo. alpha is nullptr iff !has_alpha.
  // If temp_intervals != null, fills them such that CopyTo can rescale to that
  // range. Otherwise, clamps temp to [0, 1].
  ExternalImage(ThreadPool* pool, const Image3F& color, const Rect& rect,
                const ColorEncoding& c_current, const ColorEncoding& c_desired,
                bool has_alpha, bool alpha_is_premultiplied,
                const ImageU* alpha, size_t bits_per_alpha,
                size_t bits_per_sample, bool big_endian,
                CodecIntervals* temp_intervals);

  // Indicates whether the ctor succeeded; if not, do not use this instance.
  Status IsHealthy() const { return is_healthy_; }

  // Sets "ib" to a newly allocated copy with c_current color space.
  // Uses temp_intervals for rescaling if not null (NOTE: temp_intervals is
  // given as if a range of [0.0f-1.0f] would be used, even though it uses
  // [0.0f-255.0f] internally, to match the same parameter given to the
  // color converting constructor).
  //
  // DEPRECATED, use ::CopyTo instead
  Status CopyTo(const CodecIntervals* temp_intervals, ThreadPool* pool,
                ImageBundle* ib) const;

  // Packed, interleaved pixels, for passing to encoders.
  const PaddedBytes& Bytes() const { return bytes_; }

  const PackedImage& Desc() const { return desc_; }

  size_t xsize() const { return desc_.xsize; }
  size_t ysize() const { return desc_.ysize; }
  const ColorEncoding& c_current() const { return desc_.c_current; }
  bool IsGray() const { return desc_.c_current.IsGray(); }
  bool HasAlpha() const { return desc_.channels == 2 || desc_.channels == 4; }
  bool AlphaIsPremultiplied() const { return desc_.alpha_is_premultiplied; }
  size_t BitsPerAlpha() const { return desc_.bits_per_alpha; }
  size_t BitsPerSample() const { return desc_.bits_per_sample; }
  bool BigEndian() const { return desc_.big_endian; }

  uint8_t* Row(size_t y) { return bytes_.data() + y * desc_.row_size; }
  const uint8_t* ConstRow(size_t y) const {
    return bytes_.data() + y * desc_.row_size;
  }

 private:
  ExternalImage(size_t xsize, size_t ysize, const ColorEncoding& c_current,
                bool has_alpha, bool alpha_is_premultiplied,
                size_t bits_per_alpha, size_t bits_per_sample, bool big_endian);

  PackedImage desc_;
  PaddedBytes bytes_;
  bool is_healthy_;
};

// Copies from packed u8/u16/f32 to floating-point, keeping the same color
// space. No clipping; assumes the range of unsigned integers maps to [0, 1].
Status CopyTo(const PackedImage& desc, Span<const uint8_t> bytes,
              ThreadPool* pool, ImageBundle* ib);

}  // namespace jxl

#endif  // JXL_EXTERNAL_IMAGE_H_
