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

#include "lib/jxl/external_image.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/external_image.cc"
#include <hwy/foreach_target.h>
// ^ must come before highway.h and any *-inl.h.
#include <string.h>

#include <algorithm>
#include <array>
#include <hwy/highway.h>
#include <utility>
#include <vector>

#include "hwy/base.h"  // EnableIf
#include "lib/jxl/alpha.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/cache_aligned.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/transfer_functions-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// Input/output uses the codec.h scaling: nominally 0-255 if in-gamut.
template <class V>
V LinearToSRGB(V v255) {
  const HWY_FULL(float) d;
  const auto encoded = v255 * Set(d, 1.0f / 255);
  const auto display = TF_SRGB().EncodedFromDisplay(encoded);
  return display * Set(d, 255.0f);
}

void LinearToSRGBInPlace(jxl::ThreadPool* pool, Image3F* image,
                         size_t color_channels) {
  size_t xsize = image->xsize();
  size_t ysize = image->ysize();
  const HWY_FULL(float) d;
  for (size_t c = 0; c < color_channels; ++c) {
    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          float* JXL_RESTRICT row = image->PlaneRow(c, y);
          for (size_t x = 0; x < xsize; x += Lanes(d)) {
            const auto v = LinearToSRGB(Load(d, row + x));
            Store(v, d, row + x);
          }
        },
        "LinearToSRGB");
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {
namespace {

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

#define JXL_EXT_VERBOSE 0

#if JXL_EXT_VERBOSE >= 2
// For printing RGB values at this X within each line.
constexpr size_t kX = 1;
#endif

// Encoding ImageBundle using other codecs requires format conversions to their
// "External" representation:
// ImageBundle -[1]-> Temp01 -[CMS]-> Temp01 -[2dt]-> External
// For External -> ImageBundle, we need only demux and rescale.
//
// "Temp01" and "Temp255" are interleaved and have 1 or 3 non-alpha channels.
// Alpha is included in External but not Temp because it is neither color-
// transformed nor included in Image3F.
// "ImageBundle" is Image3F (range [0, 255]) + ImageU alpha.
//
// "Temp01" is in range float [0, 1] as required by the CMS, but cannot
// losslessly represent 8-bit integer values [0, 255] due to floating point
// precision, which will reflect as a loss in Image3F which uses float range
// [0, 255] instead, which may cause effects on butteraugli score. Therefore,
// only use Temp01 if CMS transformation to different color space is required.
//
// "Temp255" is in range float [0, 255] and can losslessly represent 8-bit
// integer values [0, 255], but has floating point loss for 16-bit integer
// values [0, 65535]. The latter is not an issue however since Image3F uses
// float [0, 255] so has the same loss (so no butteraugli score effect), and
// the loss is gone when outputting to external integer again.
//
// Summary of formats:
// Name        |    Bits   |    Max   | Channels |   Layout    |  Alpha
// ------------+-----------+----------+----------+-------------+---------
// External    | 1,8,16,32 | 2^Bits-1 |  1,2,3,4 | Interleaved | Included
// Temp01      |     32    |     1    |    1,3   | Interleaved | Separate
// Temp255     |     32    |    255   |    1,3   | Interleaved | Separate
// ImageBundle |     32    |    255   |    3,4   |   Planar    |  ImageU

// Number of external channels including alpha.
struct Channels1 {
  static const char* Name() { return "1"; }
};
struct Channels2 {
  static const char* Name() { return "2"; }
};
struct Channels3 {
  static const char* Name() { return "3"; }
};
struct Channels4 {
  static const char* Name() { return "4"; }
};

// Step 1: interleaved <-> planar and rescale [0, 1] <-> [0, 255]
struct Interleave {
  static JXL_INLINE void Image3ToTemp01(Channels1 /*tag*/, const size_t y,
                                        const Image3F& image, const Rect& rect,
                                        float* JXL_RESTRICT row_temp) {
    const float* JXL_RESTRICT row_image1 = rect.ConstPlaneRow(image, 1, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      row_temp[x] = row_image1[x] * (1.0f / 255);
    }
  }

  static JXL_INLINE void Image3ToTemp01(Channels3 /*tag*/, const size_t y,
                                        const Image3F& image, const Rect& rect,
                                        float* JXL_RESTRICT row_temp) {
    const float* JXL_RESTRICT row_image0 = rect.ConstPlaneRow(image, 0, y);
    const float* JXL_RESTRICT row_image1 = rect.ConstPlaneRow(image, 1, y);
    const float* JXL_RESTRICT row_image2 = rect.ConstPlaneRow(image, 2, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      row_temp[3 * x + 0] = row_image0[x] * (1.0f / 255);
      row_temp[3 * x + 1] = row_image1[x] * (1.0f / 255);
      row_temp[3 * x + 2] = row_image2[x] * (1.0f / 255);
    }
  }

  // Same implementation for 2/4 because neither Image3 nor Temp have alpha.
  static JXL_INLINE void Image3ToTemp01(Channels2 /*tag*/, const size_t y,
                                        const Image3F& image, const Rect& rect,
                                        float* JXL_RESTRICT row_temp) {
    Image3ToTemp01(Channels1(), y, image, rect, row_temp);
  }

  static JXL_INLINE void Image3ToTemp01(Channels4 /*tag*/, const size_t y,
                                        const Image3F& image, const Rect& rect,
                                        float* JXL_RESTRICT row_temp) {
    Image3ToTemp01(Channels3(), y, image, rect, row_temp);
  }

  static JXL_INLINE void Temp255ToImage3(Channels1 /*tag*/,
                                         const float* JXL_RESTRICT row_temp,
                                         size_t y,
                                         Image3F* JXL_RESTRICT image) {
    const size_t xsize = image->xsize();
    float* JXL_RESTRICT row0 = image->PlaneRow(0, y);
    for (size_t x = 0; x < xsize; ++x) {
      row0[x] = row_temp[x];
    }

    for (size_t c = 1; c < 3; ++c) {
      float* JXL_RESTRICT row = image->PlaneRow(c, y);
      memcpy(row, row0, xsize * sizeof(float));
    }
  }

  static JXL_INLINE void Temp255ToImage3(Channels3 /*tag*/,
                                         const float* JXL_RESTRICT row_temp,
                                         size_t y,
                                         Image3F* JXL_RESTRICT image) {
    float* JXL_RESTRICT row_image0 = image->PlaneRow(0, y);
    float* JXL_RESTRICT row_image1 = image->PlaneRow(1, y);
    float* JXL_RESTRICT row_image2 = image->PlaneRow(2, y);
    for (size_t x = 0; x < image->xsize(); ++x) {
      row_image0[x] = row_temp[3 * x + 0];
      row_image1[x] = row_temp[3 * x + 1];
      row_image2[x] = row_temp[3 * x + 2];
    }
  }

  static JXL_INLINE void Temp255ToImage3(Channels2 /*tag*/,
                                         const float* JXL_RESTRICT row_temp,
                                         size_t y,
                                         Image3F* JXL_RESTRICT image) {
    Temp255ToImage3(Channels1(), row_temp, y, image);
  }

  static JXL_INLINE void Temp255ToImage3(Channels4 /*tag*/,
                                         const float* JXL_RESTRICT row_temp,
                                         size_t y,
                                         Image3F* JXL_RESTRICT image) {
    Temp255ToImage3(Channels3(), row_temp, y, image);
  }
};

// Step 2t: type conversion

// Same naming convention as Image: B=u8, U=u16, F=f32. kSize enables generic
// functions with Type and Order template arguments. 1=PBM.
struct Type1 {
  static const char* Name() { return "1"; }
  static constexpr size_t kSize = 0;
};
struct TypeB {
  static const char* Name() { return "B"; }
  static constexpr size_t kSize = 1;
};
struct TypeU {
  static const char* Name() { return "U"; }
  static constexpr size_t kSize = 2;
};
struct TypeF {
  static const char* Name() { return "F"; }
  static constexpr size_t kSize = 4;
};

// Load/stores float "sample" (gray/color) from/to u8/u16/float.
struct Sample {
  template <class Order>
  static JXL_INLINE float FromExternal(TypeB /*tag*/, const uint8_t* external) {
    return *external;
  }

  template <class Order>
  static JXL_INLINE float FromExternal(TypeU /*tag*/, const uint8_t* external) {
    return Load16(Order(), external);
  }

  template <class Order>
  static JXL_INLINE float FromExternal(TypeF /*tag*/, const uint8_t* external) {
    const int32_t bits = Load32(Order(), external);
    float sample;
    memcpy(&sample, &bits, 4);
    return sample;
  }

  template <class Order>
  static JXL_INLINE void ToExternal(TypeB /*tag*/, const float sample,
                                    uint8_t* external) {
    JXL_ASSERT(0 <= sample && sample < 256);
    // Don't need std::round since sample value is positive.
    *external = static_cast<int>(sample + 0.5f);
  }

  template <class Order>
  static JXL_INLINE void ToExternal(TypeU /*tag*/, const float sample,
                                    uint8_t* external) {
    JXL_ASSERT(0 <= sample && sample < 65536);
    // Don't need std::round since sample value is positive.
    Store16(Order(), static_cast<int>(sample + 0.5f), external);
  }

  template <class Order>
  static JXL_INLINE void ToExternal(TypeF /*tag*/, const float sample,
                                    uint8_t* external) {
    int32_t bits;
    memcpy(&bits, &sample, 4);
    Store32(Order(), bits, external);
  }
};

// Load/stores uint32_t (8/16-bit range) "alpha" from/to u8/u16. Lossless.
struct Alpha {
  // Per-thread alpha statistics.
  struct Stats {
    // Bitwise AND of all alpha values; used to detect all-opaque alpha.
    uint32_t and_bits = 0xFFFF;

    // Bitwise OR; used to detect out of bounds values (i.e. > 255 for 8-bit).
    uint32_t or_bits = 0;

    // Prevents false sharing.
    uint8_t pad[CacheAligned::kAlignment - sizeof(and_bits) - sizeof(or_bits)];
  };

  static JXL_INLINE uint32_t FromExternal(TypeB /*tag*/, OrderLE /*tag*/,
                                          const uint8_t* external) {
    return *external;
  }

  // Any larger type implies 16-bit alpha. NOTE: if TypeF, the alpha is smaller
  // than other external values (subsequent bytes are uninitialized/ignored).
  template <typename Type, class Order>
  static JXL_INLINE uint32_t FromExternal(Type /*tag*/, Order /*tag*/,
                                          const uint8_t* external) {
    const uint32_t alpha = Load16(Order(), external);
    return alpha;
  }

  static JXL_INLINE void ToExternal(TypeB /*tag*/, OrderLE /*tag*/,
                                    const uint32_t alpha, uint8_t* external) {
    JXL_ASSERT(alpha < 256);
    *external = alpha;
  }

  // Any larger type implies 16-bit alpha. NOTE: if TypeF, the alpha is smaller
  // than other external values (subsequent bytes are uninitialized/ignored).
  template <typename Type, class Order>
  static JXL_INLINE void ToExternal(Type /*tag*/, Order /*tag*/,
                                    const uint32_t alpha, uint8_t* external) {
    Store16(Order(), alpha, external);
  }
};

#define JXL_IF_NOT_PBM hwy::EnableIf<Type::kSize != 0>* = nullptr

// Step 2d: demux external into separate (type-converted) color and alpha.
// Supports Temp01 and Temp255, the Cast decides this.
struct Demux {
  // PBM, one plane
  template <class Order, class Channels, class Cast>
  static JXL_INLINE void ExternalToTemp(Type1 /*type*/, Order /*order*/,
                                        Channels /*channels*/,
                                        const size_t xsize,
                                        const uint8_t* external,
                                        const Cast /*cast*/,
                                        float* JXL_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const uint32_t byte = external[x / kBitsPerByte];
      const size_t idx_bit = x % kBitsPerByte;  // 0 = MSB!
      // 1 is black, and bit order is MSB to LSB.
      row_temp[x] = (byte & (0x80 >> idx_bit)) ? 0 : 255;
    }
  }
  template <class Order, class Channels, class Cast>
  static JXL_INLINE void TempToExternal(Type1 /*type*/, Order /*order*/,
                                        Channels /*channels*/,
                                        const size_t xsize,
                                        const float* JXL_RESTRICT row_temp,
                                        const Cast /*cast*/,
                                        uint8_t* row_external) {
    memset(row_external, 0, DivCeil(xsize, kBitsPerByte));
    for (size_t x = 0; x < xsize; ++x) {
      const size_t idx_byte = x / kBitsPerByte;
      const size_t idx_bit = x % kBitsPerByte;  // 0 = MSB!
      // 1 is black, and bit order is MSB to LSB.
      const uint32_t bit = (row_temp[x] == 0.0f) ? (0x80 >> idx_bit) : 0;
      row_external[idx_byte] =
          static_cast<uint8_t>(row_external[idx_byte] | bit);
    }
  }

  // 1 plane - copy all.
  template <class Type, class Order, class Cast, JXL_IF_NOT_PBM>
  static JXL_INLINE void ExternalToTemp(Type type, Order /*order*/,
                                        Channels1 /*tag*/, const size_t xsize,
                                        const uint8_t* external,
                                        const Cast cast,
                                        float* JXL_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const float rounded =
          Sample::FromExternal<Order>(type, external + x * Type::kSize);
      row_temp[x] = cast.FromExternal(rounded, 0);
    }
  }
  template <class Type, class Order, class Cast, JXL_IF_NOT_PBM>
  static JXL_INLINE void TempToExternal(Type type, Order /*order*/,
                                        Channels1 /*tag*/, const size_t xsize,
                                        const float* JXL_RESTRICT row_temp,
                                        const Cast cast,
                                        uint8_t* row_external) {
    for (size_t x = 0; x < xsize; ++x) {
      const float sample = cast.FromTemp(row_temp[x], 0);
      Sample::ToExternal<Order>(type, sample, row_external + x * Type::kSize);
    }
  }

  // 2 planes - ignore alpha.
  template <class Type, class Order, class Cast, JXL_IF_NOT_PBM>
  static JXL_INLINE void ExternalToTemp(Type type, Order /*order*/,
                                        Channels2 /*tag*/, const size_t xsize,
                                        const uint8_t* external,
                                        const Cast cast,
                                        float* JXL_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const float rounded = Sample::FromExternal<Order>(
          type, external + (2 * x + 0) * Type::kSize);
      row_temp[x] = cast.FromExternal(rounded, 0);
    }
  }
  template <class Type, class Order, class Cast, JXL_IF_NOT_PBM>
  static JXL_INLINE void TempToExternal(Type type, Order /*order*/,
                                        Channels2 /*tag*/, const size_t xsize,
                                        const float* JXL_RESTRICT row_temp,
                                        const Cast cast,
                                        uint8_t* row_external) {
    for (size_t x = 0; x < xsize; ++x) {
      const float sample = cast.FromTemp(row_temp[x], 0);
      Sample::ToExternal<Order>(type, sample,
                                row_external + (2 * x + 0) * Type::kSize);
    }
  }

  // 3 planes - copy all.
  template <class Type, class Order, class Cast, JXL_IF_NOT_PBM>
  static JXL_INLINE void ExternalToTemp(Type type, Order /*order*/,
                                        Channels3 /*tag*/, const size_t xsize,
                                        const uint8_t* external,
                                        const Cast cast,
                                        float* JXL_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const float rounded0 = Sample::FromExternal<Order>(
          type, external + (3 * x + 0) * Type::kSize);
      const float rounded1 = Sample::FromExternal<Order>(
          type, external + (3 * x + 1) * Type::kSize);
      const float rounded2 = Sample::FromExternal<Order>(
          type, external + (3 * x + 2) * Type::kSize);
      row_temp[3 * x + 0] = cast.FromExternal(rounded0, 0);
      row_temp[3 * x + 1] = cast.FromExternal(rounded1, 1);
      row_temp[3 * x + 2] = cast.FromExternal(rounded2, 2);
    }
  }
  template <class Type, class Order, class Cast, JXL_IF_NOT_PBM>
  static JXL_INLINE void TempToExternal(Type type, Order /*order*/,
                                        Channels3 /*tag*/, const size_t xsize,
                                        const float* JXL_RESTRICT row_temp,
                                        const Cast cast,
                                        uint8_t* row_external) {
    for (size_t x = 0; x < xsize; ++x) {
      const float sample0 = cast.FromTemp(row_temp[3 * x + 0], 0);
      const float sample1 = cast.FromTemp(row_temp[3 * x + 1], 1);
      const float sample2 = cast.FromTemp(row_temp[3 * x + 2], 2);
      Sample::ToExternal<Order>(type, sample0,
                                row_external + (3 * x + 0) * Type::kSize);
      Sample::ToExternal<Order>(type, sample1,
                                row_external + (3 * x + 1) * Type::kSize);
      Sample::ToExternal<Order>(type, sample2,
                                row_external + (3 * x + 2) * Type::kSize);
    }
  }

  // 4 planes - ignore alpha.
  template <class Type, class Order, class Cast, JXL_IF_NOT_PBM>
  static JXL_INLINE void ExternalToTemp(Type type, Order /*order*/,
                                        Channels4 /*tag*/, const size_t xsize,
                                        const uint8_t* external,
                                        const Cast cast,
                                        float* JXL_RESTRICT row_temp) {
    for (size_t x = 0; x < xsize; ++x) {
      const float rounded0 = Sample::FromExternal<Order>(
          type, external + (4 * x + 0) * Type::kSize);
      const float rounded1 = Sample::FromExternal<Order>(
          type, external + (4 * x + 1) * Type::kSize);
      const float rounded2 = Sample::FromExternal<Order>(
          type, external + (4 * x + 2) * Type::kSize);
      row_temp[3 * x + 0] = cast.FromExternal(rounded0, 0);
      row_temp[3 * x + 1] = cast.FromExternal(rounded1, 1);
      row_temp[3 * x + 2] = cast.FromExternal(rounded2, 2);
    }
  }
  template <class Type, class Order, class Cast, JXL_IF_NOT_PBM>
  static JXL_INLINE void TempToExternal(Type type, Order /*order*/,
                                        Channels4 /*tag*/, const size_t xsize,
                                        const float* JXL_RESTRICT row_temp,
                                        const Cast cast,
                                        uint8_t* row_external) {
    for (size_t x = 0; x < xsize; ++x) {
      const float sample0 = cast.FromTemp(row_temp[3 * x + 0], 0);
      const float sample1 = cast.FromTemp(row_temp[3 * x + 1], 1);
      const float sample2 = cast.FromTemp(row_temp[3 * x + 2], 2);
      Sample::ToExternal<Order>(type, sample0,
                                row_external + (4 * x + 0) * Type::kSize);
      Sample::ToExternal<Order>(type, sample1,
                                row_external + (4 * x + 1) * Type::kSize);
      Sample::ToExternal<Order>(type, sample2,
                                row_external + (4 * x + 2) * Type::kSize);
    }
  }

  // Gray only, no alpha.
  template <class Type, class Order>
  static JXL_INLINE void ExternalToAlpha(
      Type /*type*/, Order /*order*/, Channels1 /*tag*/, const size_t /*xsize*/,
      const uint8_t* /*external*/, uint16_t* JXL_RESTRICT /*row_alpha*/,
      const size_t /*thread*/, std::vector<Alpha::Stats>* /*stats*/) {}
  template <class Type, class Order>
  static JXL_INLINE void AlphaToExternal(
      Type /*type*/, Order /*order*/, Channels1 /*tag*/, const size_t /*xsize*/,
      const uint16_t* JXL_RESTRICT /*row_alpha*/, uint8_t* /*row_external*/) {}

  // Gray + alpha.
  template <class Type, class Order>
  static JXL_INLINE void ExternalToAlpha(Type type, Order order,
                                         Channels2 /*tag*/, const size_t xsize,
                                         const uint8_t* external,
                                         uint16_t* JXL_RESTRICT row_alpha,
                                         const size_t thread,
                                         std::vector<Alpha::Stats>* stats) {
    if (row_alpha == nullptr) return;
    uint32_t and_bits = 0xFFFF;
    uint32_t or_bits = 0;
    for (size_t x = 0; x < xsize; ++x) {
      const uint32_t alpha = Alpha::FromExternal(
          type, order, external + (2 * x + 1) * Type::kSize);
      and_bits &= alpha;
      or_bits |= alpha;
      row_alpha[x] = alpha;
    }
    (*stats)[thread].and_bits &= and_bits;
    (*stats)[thread].or_bits |= or_bits;
  }
  template <class Type, class Order>
  static JXL_INLINE void AlphaToExternal(Type type, Order order,
                                         Channels2 /*tag*/, const size_t xsize,
                                         const uint16_t* JXL_RESTRICT row_alpha,
                                         uint8_t* row_external) {
    if (row_alpha == nullptr) {
      for (size_t x = 0; x < xsize; ++x) {
        Alpha::ToExternal(type, order, type.kMaxAlpha,
                          row_external + (2 * x + 1) * Type::kSize);
      }
    } else {
      for (size_t x = 0; x < xsize; ++x) {
        Alpha::ToExternal(type, order, row_alpha[x],
                          row_external + (2 * x + 1) * Type::kSize);
      }
    }
  }

  // RGB only, no alpha.
  template <class Type, class Order>
  static JXL_INLINE void ExternalToAlpha(
      Type /*type*/, Order /*order*/, Channels3 /*tag*/, const size_t /*xsize*/,
      const uint8_t* /*external*/, uint16_t* JXL_RESTRICT /*row_alpha*/,
      const size_t /*thread*/, std::vector<Alpha::Stats>* /*stats*/) {}
  template <class Type, class Order>
  static JXL_INLINE void AlphaToExternal(
      Type /*type*/, Order /*order*/, Channels3 /*tag*/, const size_t /*xsize*/,
      const uint16_t* JXL_RESTRICT /*row_alpha*/, uint8_t* /*row_external*/) {}

  // RGBA.
  template <class Type, class Order>
  static JXL_INLINE void ExternalToAlpha(Type type, Order order,
                                         Channels4 /*tag*/, const size_t xsize,
                                         const uint8_t* external,
                                         uint16_t* JXL_RESTRICT row_alpha,
                                         const size_t thread,
                                         std::vector<Alpha::Stats>* stats) {
    if (row_alpha == nullptr) return;
    uint32_t and_bits = 0xFFFF;
    uint32_t or_bits = 0;
    for (size_t x = 0; x < xsize; ++x) {
      const uint32_t alpha = Alpha::FromExternal(
          type, order, external + (4 * x + 3) * Type::kSize);
      and_bits &= alpha;
      or_bits |= alpha;
      row_alpha[x] = alpha;
    }
    (*stats)[thread].and_bits &= and_bits;
    (*stats)[thread].or_bits |= or_bits;
  }
  template <class Type, class Order>
  static JXL_INLINE void AlphaToExternal(Type type, Order order,
                                         Channels4 /*tag*/, const size_t xsize,
                                         const uint16_t* JXL_RESTRICT row_alpha,
                                         uint8_t* row_external) {
    if (row_alpha == nullptr) {
      for (size_t x = 0; x < xsize; ++x) {
        Alpha::ToExternal(type, order, type.kMaxAlpha,
                          row_external + (4 * x + 3) * Type::kSize);
      }
    } else {
      for (size_t x = 0; x < xsize; ++x) {
        Alpha::ToExternal(type, order, row_alpha[x],
                          row_external + (4 * x + 3) * Type::kSize);
      }
    }
  }
};

// Used to select the Transformer::DoRow overload to call.
struct ToExternal1 {};  // first phase: store to temp and compute min/max.
struct ToExternal2 {};  // second phase: rescale temp to external.
struct ToExternal {};   // single-pass, only usable with CastClip.

// For ToExternal - assumes known/static extents of temp values.
struct ExtentsStatic {
  static Status SetNumThreads(size_t /* num_threads */) { return true; }
};

// For ToExternal1 - computes extents of temp values.
class ExtentsDynamic {
 public:
  ExtentsDynamic(const size_t xsize, const size_t ysize,
                 const ColorEncoding& c_desired)
      : temp_intervals_(c_desired.Channels()) {
    // Store all temp pixels here, convert to external in a second phase after
    // Finalize computes ChannelIntervals from min_max_.
    temp_ = ImageF(xsize * temp_intervals_, ysize);
  }

  Status SetNumThreads(size_t num_threads) {
    min_max_.resize(num_threads);
    return true;
  }

  float* RowTemp(const size_t y) { return temp_.Row(y); }

  // Row size is obtained from temp_. NOTE: clamps temp values to kMax.
  JXL_INLINE void Update(const size_t thread, float* JXL_RESTRICT row_temp) {
    // row_temp is interleaved - keep track of current channel.
    size_t c = 0;
    for (size_t i = 0; i < temp_.xsize(); ++i, ++c) {
      if (c == temp_intervals_) c = 0;
      if (row_temp[i] > min_max_[thread].max[c]) {
        if (row_temp[i] > kMax) row_temp[i] = kMax;
        min_max_[thread].max[c] = row_temp[i];
      }
      if (row_temp[i] < min_max_[thread].min[c]) {
        if (row_temp[i] < -kMax) row_temp[i] = -kMax;
        min_max_[thread].min[c] = row_temp[i];
      }
    }
  }

  void Finalize(CodecIntervals* temp_intervals) const {
    // Any other ChannelInterval remains default-initialized.
    for (size_t c = 0; c < temp_intervals_; ++c) {
      float min = min_max_[0].min[c];
      float max = min_max_[0].max[c];
      for (size_t i = 1; i < min_max_.size(); ++i) {
        min = std::min(min, min_max_[i].min[c]);
        max = std::max(max, min_max_[i].max[c]);
      }
      // Update ensured these are clamped.
      JXL_ASSERT(-kMax <= min && min <= max && max <= kMax);
      (*temp_intervals)[c] = CodecInterval(min, max);
    }
  }

 private:
  // Larger values are probably invalid, so clamp to preserve some precision.
  static constexpr float kMax = 1E10;

  struct MinMax {
    MinMax() {
      for (size_t c = 0; c < 4; ++c) {
        min[c] = kMax;
        max[c] = -kMax;
      }
    }

    float min[4];
    float max[4];
    // Prevents false sharing.
    uint8_t pad[CacheAligned::kAlignment - sizeof(min) - sizeof(max)];
  };

  const size_t temp_intervals_;
  ImageF temp_;
  std::vector<MinMax> min_max_;
};

// For ToExternal1, which updates ExtentsDynamic without casting.
struct CastUnused {};

// Returns range of valid values for all channel.
CodecInterval GetInterval(const size_t bits_per_sample) {
  if (bits_per_sample == 32) {
    // This ensures ConvertImage produces an image with the same [0, 255]
    // range as its input, but increases round trip error by ~2x vs [0, 1].
    return CodecInterval(0.0f, 255.0f);
  } else {
    const float max = (1U << bits_per_sample) - 1;
    return CodecInterval(0, max);
  }
}

// Lossless conversion between [0, 1] and [min, min+width]. Width is 1 or
// > 1 ("unbounded", useful for round trip testing). This is used to scale to
// the external type and back to the arbitrary interval.
class CastRescale01 {
 public:
  static const char* Name() { return "Rescale01"; }
  CastRescale01(const CodecIntervals& temp_intervals,
                const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      temp_min_[c] = temp_intervals[c].min;
      temp_mul_[c] = ext_interval.width / temp_intervals[c].width;
      external_min_[c] = ext_interval.min;
      external_mul_[c] = temp_intervals[c].width / ext_interval.width;
    }
#if JXL_EXT_VERBOSE >= 2
    printf("CastRescale01 min %f width %f %f\n", temp_intervals[0].min,
           temp_intervals[0].width, ext_interval.width);
#endif
  }

  JXL_INLINE float FromExternal(const float external, const size_t c) const {
    return (external - external_min_[c]) * external_mul_[c] + temp_min_[c];
  }
  JXL_INLINE float FromTemp(const float temp, const size_t c) const {
    return (temp - temp_min_[c]) * temp_mul_[c] + external_min_[c];
  }

 private:
  float temp_min_[4];
  float temp_mul_[4];
  float external_min_[4];
  float external_mul_[4];
};

// Lossless conversion between [0, 255] and [min, min+width]. Width is 255 or
// > 255 ("unbounded", useful for round trip testing). This is used to scale to
// the external type and back to the arbitrary interval.
// NOTE: this rescaler exists to make CopyTo match the convention of
// "temp_intervals" used by the color converting constructor. In the external to
// IO case without color conversion, one normally does not use this parameter.
class CastRescale255 {
 public:
  static const char* Name() { return "Rescale255"; }
  CastRescale255(const CodecIntervals& temp_intervals,
                 const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      temp_min_[c] = 255.0f * temp_intervals[c].min;
      temp_mul_[c] =
          ext_interval.width / temp_intervals[c].width * (1.0f / 255);
      external_min_[c] = ext_interval.min * (1.0f / 255);
      external_mul_[c] = 255.0f * temp_intervals[c].width / ext_interval.width;
    }
#if JXL_EXT_VERBOSE >= 2
    printf("CastRescale255 min %f width %f %f\n", temp_intervals[0].min,
           temp_intervals[0].width, ext_interval.width);
#endif
  }

  JXL_INLINE float FromExternal(const float external, const size_t c) const {
    return (external - external_min_[c]) * external_mul_[c] + temp_min_[c];
  }
  JXL_INLINE float FromTemp(const float temp, const size_t c) const {
    return (temp - temp_min_[c]) * temp_mul_[c] + external_min_[c];
  }

 private:
  float temp_min_[4];
  float temp_mul_[4];
  float external_min_[4];
  float external_mul_[4];
};

// Converts between [0, 1] and the external type's range. Lossy because values
// outside [0, 1] are clamped - this is necessary for codecs that are not able
// to store min/width metadata.
class CastClip01 {
 public:
  static const char* Name() { return "Clip01"; }
  explicit CastClip01(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      temp_mul_[c] = ext_interval.width;
      external_min_[c] = ext_interval.min;
      external_mul_[c] = 1.0f / ext_interval.width;
    }
#if JXL_EXT_VERBOSE >= 2
    printf("CastClip01 width %f\n", ext_interval.width);
#endif
  }

  JXL_INLINE float FromExternal(const float external, const size_t c) const {
    const float temp01 = (external - external_min_[c]) * external_mul_[c];
    return temp01;
  }
  JXL_INLINE float FromTemp(const float temp, const size_t c) const {
    return Clamp01(temp) * temp_mul_[c] + external_min_[c];
  }

 private:
  static JXL_INLINE float Clamp01(const float temp) {
    return std::min(std::max(0.0f, temp), 1.0f);
  }

  float temp_mul_[4];
  float external_min_[4];
  float external_mul_[4];
};

struct CastFloat {
  static const char* Name() { return "Float"; }
  explicit CastFloat(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      JXL_CHECK(ext_interval.min == 0.0f);
      JXL_CHECK(ext_interval.width == 255.0f);
    }
#if JXL_EXT_VERBOSE >= 2
    printf("CastFloat\n");
#endif
  }

  JXL_INLINE float FromExternal(const float external, const size_t c) const {
    const float temp01 = external * (1.0f / 255);
    return temp01;
  }
  JXL_INLINE float FromTemp(const float temp, const size_t c) const {
    return temp * 255.0f;
  }
};

// Converts between [0, 255] and the external type's range. Lossy because values
// outside [0, 255] are clamped - this is necessary for codecs that are not able
// to store min/width metadata.
class CastClip255 {
 public:
  static const char* Name() { return "Clip255"; }
  explicit CastClip255(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      temp_mul_[c] = ext_interval.width;
      external_min_[c] = ext_interval.min;
      external_mul_[c] = 255.0f / ext_interval.width;
    }
#if JXL_EXT_VERBOSE >= 2
    printf("CastClip255 width %f\n", ext_interval.width);
#endif
  }

  JXL_INLINE float FromExternal(const float external, const size_t c) const {
    const float temp255 = (external - external_min_[c]) * external_mul_[c];
    return temp255;
  }
  JXL_INLINE float FromTemp(const float temp, const size_t c) const {
    return Clamp255(temp) * temp_mul_[c] + external_min_[c];
  }

 private:
  static JXL_INLINE float Clamp255(const float temp) {
    return std::min(std::max(0.0f, temp), 255.0f);
  }

  float temp_mul_[4];
  float external_min_[4];
  float external_mul_[4];
};

struct CastFloat01 {
  static const char* Name() { return "Float01"; }
  explicit CastFloat01(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      JXL_CHECK(ext_interval.min == 0.0f);
      JXL_CHECK(ext_interval.width == 255.0f);
    }
#if JXL_EXT_VERBOSE >= 2
    printf("CastFloat01\n");
#endif
  }

  JXL_INLINE float FromExternal(const float external, const size_t c) const {
    const float temp01 = external * (1.0f / 255);
    return temp01;
  }
  JXL_INLINE float FromTemp(const float temp, const size_t c) const {
    return temp * 255.0f;
  }
};

// No-op
struct CastFloat255 {
  static const char* Name() { return "Float255"; }
  explicit CastFloat255(const CodecInterval ext_interval) {
    for (size_t c = 0; c < 4; ++c) {
      JXL_CHECK(ext_interval.min == 0.0f);
      JXL_CHECK(ext_interval.width == 255.0f);
    }
#if JXL_EXT_VERBOSE >= 2
    printf("CastFloat255\n");
#endif
  }

  JXL_INLINE float FromExternal(const float external, const size_t c) const {
    return external;
  }
  JXL_INLINE float FromTemp(const float temp, const size_t c) const {
    return temp;
  }
};

// Multithreaded deinterleaving/conversion from ExternalImage to Image3.
class Converter {
 public:
  Converter(ThreadPool* pool, const PackedImage& desc, const uint8_t* bytes)
      : pool_(pool),
        desc_(&desc),
        bytes_(bytes),
        color_(desc.xsize, desc.ysize) {}

  // Run can only be called once per instance.
  template <class Cast>
  Status Run(const Cast& cast) {
    const size_t bytes = DivCeil(desc_->bits_per_sample, kBitsPerByte);
    const bool big_endian = desc_->big_endian;
    if (desc_->bits_per_sample == 1) {
      return DispatchType<Type1, OrderLE>(cast);
    } else if (bytes == 1) {
      return DispatchType<TypeB, OrderLE>(cast);
    } else if (bytes == 2 && big_endian) {
      return DispatchType<TypeU, OrderBE>(cast);
    } else if (bytes == 2) {
      return DispatchType<TypeU, OrderLE>(cast);
    } else if (bytes == 4 && big_endian) {
      return DispatchType<TypeF, OrderBE>(cast);
    } else if (bytes == 4) {
      return DispatchType<TypeF, OrderLE>(cast);
    } else {
      return JXL_FAILURE("Unsupported BitsPerSample");
    }
    return true;
  }

  Status MoveTo(ImageBundle* ib) {
    ib->SetFromImage(std::move(color_), desc_->c_current);

    // Don't have alpha; during TransformTo, don't remove existing alpha.
    if (alpha_stats_.empty()) return true;

    // Also don't remove alpha for animations, since a single frame is not
    // enough to know that it is safe to do so.
    if (ib->metadata()->have_animation) return true;

    const size_t max_alpha = MaxAlpha(bits_per_alpha_);

    // Reduce per-thread statistics.
    uint32_t and_bits = alpha_stats_[0].and_bits;
    uint32_t or_bits = alpha_stats_[0].or_bits;
    for (size_t i = 1; i < alpha_stats_.size(); ++i) {
      and_bits &= alpha_stats_[i].and_bits;
      or_bits |= alpha_stats_[i].or_bits;
    }

    if (or_bits > max_alpha) {
      return JXL_FAILURE("Alpha out of range");
    }

    // Always set, so we can properly remove below.
    ib->SetAlpha(std::move(alpha_), desc_->alpha_is_premultiplied);

    return true;
  }

 private:
  Status InitBeforeRun(size_t num_threads) {
    if (init_called_) return true;
    init_called_ = true;

    temp_buf_ = ImageF(desc_->xsize * desc_->c_current.Channels(), num_threads);

    if (desc_->HasAlpha()) {
      alpha_ = ImageU(desc_->xsize, desc_->ysize);
      bits_per_alpha_ = desc_->bits_per_alpha;
      alpha_stats_.resize(num_threads);
    }
    return true;
  }

  template <class Type, class Order, class Channels, class Cast>
  JXL_INLINE void DoRow(const Cast& cast, const size_t y, const size_t thread) {
    const size_t iy = desc_->flipped_y ? (desc_->ysize - 1 - y) : y;
    const uint8_t* JXL_RESTRICT row_external = bytes_ + iy * desc_->row_size;

    if (!alpha_stats_.empty()) {
      // No-op if Channels1/3.
      Demux::ExternalToAlpha(Type(), Order(), Channels(), desc_->xsize,
                             row_external, alpha_.Row(y), thread,
                             &alpha_stats_);
    }

    float* JXL_RESTRICT row_temp = temp_buf_.Row(thread);
    Demux::ExternalToTemp(Type(), Order(), Channels(), desc_->xsize,
                          row_external, cast, row_temp);

#if JXL_EXT_VERBOSE >= 2
    printf("ToIO(%s%s %s): ext %3d %3d %3d  tmp %.4f %.4f %.4f\n",
           Channels::Name(), Type::Name(), Cast::Name(),
           row_external[3 * kX + 0], row_external[3 * kX + 1],
           row_external[3 * kX + 2], row_temp[3 * kX + 0], row_temp[3 * kX + 1],
           row_temp[3 * kX + 2]);
#endif

    Interleave::Temp255ToImage3(Channels(), row_temp, y, &color_);
  }

  // Closure callable by ThreadPool.
  template <class Type, class Order, class Channels, class Cast>
  class Bind {
   public:
    explicit Bind(Converter* converter, const Cast& cast)
        : converter_(converter), cast_(cast) {}

    JXL_INLINE void operator()(const int task, const int thread) const {
      converter_->DoRow<Type, Order, Channels>(cast_, task, thread);
    }

   private:
    Converter* converter_;  // not owned
    const Cast cast_;
  };

  template <class Type, class Order, class Channels, class Cast>
  Status DoRows(const Cast& cast) {
    return RunOnPool(
        pool_, 0, desc_->ysize,
        [this](const size_t num_threads) {
          return this->InitBeforeRun(num_threads);
        },
        Bind<Type, Order, Channels, Cast>(this, cast), "ExtImg cvt");
  }

  // Calls the instantiation with the matching Type and Order.
  template <class Type, class Order, class Cast>
  Status DispatchType(const Cast& cast) {
    if (desc_->c_current.IsGray()) {
      if (desc_->HasAlpha()) {
        return DoRows<Type, Order, Channels2>(cast);
      } else {
        return DoRows<Type, Order, Channels1>(cast);
      }
    } else {
      if (desc_->HasAlpha()) {
        return DoRows<Type, Order, Channels4>(cast);
      } else {
        return DoRows<Type, Order, Channels3>(cast);
      }
    }
  }

  ThreadPool* pool_;         // not owned
  const PackedImage* desc_;  // not owned
  const uint8_t* bytes_;     // not owned
  Image3F color_;

  ImageF temp_buf_;

  // Only initialized if external_->HasAlpha() && want_alpha:
  std::vector<Alpha::Stats> alpha_stats_;
  ImageU alpha_;
  size_t bits_per_alpha_;

  // Whether InitBeforeRun() was already called.
  bool init_called_ = false;
};

// Copies from packed u8/u16/f32 to floating-point, keeping the same color
// space. No clipping; assumes the range of unsigned integers maps to [0, 1].
Status CopyTo(const PackedImage& desc, Span<const uint8_t> bytes,
              ThreadPool* pool, ImageBundle* ib) {
  JXL_RETURN_IF_ERROR(desc.ValidBufferSize(bytes.size()));
  Converter converter(pool, desc, bytes.data());

  const CodecInterval ext_interval = GetInterval(desc.bits_per_sample);

  if (desc.bits_per_sample == 32) {
    const CastFloat255 cast(ext_interval);
    JXL_RETURN_IF_ERROR(converter.Run(cast));
  } else {
    const CastClip255 cast(ext_interval);
    JXL_RETURN_IF_ERROR(converter.Run(cast));
  }

  return converter.MoveTo(ib);
}

// Stores a float in big endian
void StoreBEFloat(float value, uint8_t* p) {
  uint32_t u;
  memcpy(&u, &value, 4);
  StoreBE32(u, p);
}

// Stores a float in little endian
void StoreLEFloat(float value, uint8_t* p) {
  uint32_t u;
  memcpy(&u, &value, 4);
  StoreLE32(u, p);
}

void ConvertAlpha(size_t bits_in, const jxl::ImageU& in, size_t bits_out,
                  jxl::ImageU* out, jxl::ThreadPool* pool) {
  size_t xsize = in.xsize();
  size_t ysize = in.ysize();

  // Error checked elsewhere, but ensure clang-tidy does not report division
  // through zero.
  if (bits_in == 0 || bits_out == 0) return;

  if (bits_in < bits_out) {
    // Multiplier such that bits are duplicated, e.g. when going from 4 bits
    // to 16 bits, converts 0x5 into 0x5555.
    const uint16_t mul =
        ((1ull << bits_out) - 1ull) / ((1ull << bits_in) - 1ull);

    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const uint16_t* JXL_RESTRICT row_in = in.Row(y);
          uint16_t* JXL_RESTRICT row_out = out->Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            row_out[x] = row_in[x] * mul;
          }
        },
        "ConvertAlphaU");

  } else {
    // E.g. divide through 257 when converting 16-bit to 8-bit
    const uint16_t div =
        ((1ull << bits_in) - 1ull) / ((1ull << bits_out) - 1ull);
    // Add for round to nearest division.
    const uint16_t add = 1 << (bits_out - 1);

    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const uint16_t* JXL_RESTRICT row_in = in.Row(y);
          uint16_t* JXL_RESTRICT row_out = out->Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            row_out[x] = (row_in[x] + add) / div;
          }
        },
        "ConvertAlphaU");
  }
}

// The orientation may not be identity.
// TODO(lode): SIMDify where possible
template <typename T>
void UndoOrientation(jxl::Orientation undo_orientation, const Plane<T>& image,
                     Plane<T>& out, jxl::ThreadPool* pool) {
  const size_t xsize = image.xsize();
  const size_t ysize = image.ysize();

  if (undo_orientation == Orientation::kFlipHorizontal) {
    out = Plane<T>(xsize, ysize);
    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const T* JXL_RESTRICT row_in = image.Row(y);
          T* JXL_RESTRICT row_out = out.Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            row_out[xsize - x - 1] = row_in[x];
          }
        },
        "UndoOrientation");
  } else if (undo_orientation == Orientation::kRotate180) {
    out = Plane<T>(xsize, ysize);
    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const T* JXL_RESTRICT row_in = image.Row(y);
          T* JXL_RESTRICT row_out = out.Row(ysize - y - 1);
          for (size_t x = 0; x < xsize; ++x) {
            row_out[xsize - x - 1] = row_in[x];
          }
        },
        "UndoOrientation");
  } else if (undo_orientation == Orientation::kFlipVertical) {
    out = Plane<T>(xsize, ysize);
    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const T* JXL_RESTRICT row_in = image.Row(y);
          T* JXL_RESTRICT row_out = out.Row(ysize - y - 1);
          for (size_t x = 0; x < xsize; ++x) {
            row_out[x] = row_in[x];
          }
        },
        "UndoOrientation");
  } else if (undo_orientation == Orientation::kTranspose) {
    out = Plane<T>(ysize, xsize);
    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const T* JXL_RESTRICT row_in = image.Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            out.Row(x)[y] = row_in[x];
          }
        },
        "UndoOrientation");
  } else if (undo_orientation == Orientation::kRotate90) {
    out = Plane<T>(ysize, xsize);
    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const T* JXL_RESTRICT row_in = image.Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            out.Row(x)[ysize - y - 1] = row_in[x];
          }
        },
        "UndoOrientation");
  } else if (undo_orientation == Orientation::kAntiTranspose) {
    out = Plane<T>(ysize, xsize);
    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const T* JXL_RESTRICT row_in = image.Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            out.Row(xsize - x - 1)[ysize - y - 1] = row_in[x];
          }
        },
        "UndoOrientation");
  } else if (undo_orientation == Orientation::kRotate270) {
    out = Plane<T>(ysize, xsize);
    RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t y = task;
          const T* JXL_RESTRICT row_in = image.Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            out.Row(xsize - x - 1)[y] = row_in[x];
          }
        },
        "UndoOrientation");
  }
}
}  // namespace

HWY_EXPORT(LinearToSRGBInPlace);

Status ConvertImage(const jxl::ImageBundle& ib, size_t bits_per_sample,
                    bool float_out, bool lossless_float, bool apply_srgb_tf,
                    size_t num_channels, bool little_endian, size_t stride,
                    jxl::ThreadPool* pool, void* out_image, size_t out_size,
                    jxl::Orientation undo_orientation) {
  size_t xsize = ib.xsize();
  size_t ysize = ib.ysize();

  uint8_t* out = reinterpret_cast<uint8_t*>(out_image);

  bool want_alpha = num_channels == 2 || num_channels == 4;
  size_t color_channels = num_channels <= 2 ? 1 : 3;

  // Increment per output pixel
  const size_t inc = num_channels * bits_per_sample / jxl::kBitsPerByte;

  if (stride < inc * xsize) {
    return JXL_FAILURE("stride is smaller than scanline width in bytes");
  }

  const Image3F* color = &ib.color();
  Image3F temp_color;
  const ImageU* alpha = ib.HasAlpha() ? &ib.alpha() : nullptr;
  ImageU temp_alpha;
  if (apply_srgb_tf) {
    temp_color = CopyImage(*color);
    HWY_DYNAMIC_DISPATCH(LinearToSRGBInPlace)
    (pool, &temp_color, color_channels);
    color = &temp_color;
  }

  if (undo_orientation != Orientation::kIdentity) {
    Image3F transformed;
    for (size_t c = 0; c < color_channels; ++c) {
      UndoOrientation(undo_orientation, color->Plane(c), transformed.Plane(c),
                      pool);
    }
    transformed.Swap(temp_color);
    color = &temp_color;
    if (ib.HasAlpha()) {
      UndoOrientation(undo_orientation, *alpha, temp_alpha, pool);
      alpha = &temp_alpha;
    }

    xsize = color->xsize();
    ysize = color->ysize();
  }

  if (float_out) {
    if (bits_per_sample != 32) {
      return JXL_FAILURE("non-32-bit float not supported");
    }
    for (size_t c = 0; c < color_channels; ++c) {
      // JXL uses range 0-255 internally, but externally we use range 0-1
      float mul = 1.0f / 255.0f;

      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const int64_t y = task;
            size_t i = stride * y + (c * bits_per_sample / jxl::kBitsPerByte);
            const float* JXL_RESTRICT row_in = color->PlaneRow(c, y);
            if (lossless_float) {
              // for lossless PFM, we need to avoid the * (1./255.) * 255
              // so just interleave and don't touch
              if (little_endian) {
                for (size_t x = 0; x < xsize; ++x) {
                  StoreLEFloat(row_in[x], out + i);
                  i += inc;
                }
              } else {
                for (size_t x = 0; x < xsize; ++x) {
                  StoreBEFloat(row_in[x], out + i);
                  i += inc;
                }
              }
            } else {
              if (little_endian) {
                for (size_t x = 0; x < xsize; ++x) {
                  StoreLEFloat(row_in[x] * mul, out + i);
                  i += inc;
                }
              } else {
                for (size_t x = 0; x < xsize; ++x) {
                  StoreBEFloat(row_in[x] * mul, out + i);
                  i += inc;
                }
              }
            }
          },
          "ConvertRGBFloat");
    }
  } else {
    // Multiplier to convert from floating point 0-255 range to the integer
    // range.
    float mul = (bits_per_sample == 32)
                    ? 16843009.0f  // 4294967295 / 255.0f
                    : (((1ull << bits_per_sample) - 1) * (1 / 255.0f));
    for (size_t c = 0; c < color_channels; ++c) {
      if (bits_per_sample != 8 && bits_per_sample != 16) {
        return JXL_FAILURE("32-bit and 1-bit not yet implemented");
      }

      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const int64_t y = task;
            size_t i = stride * y + (c * bits_per_sample / jxl::kBitsPerByte);
            const float* JXL_RESTRICT row_in = color->PlaneRow(c, y);
            if (bits_per_sample == 8) {
              for (size_t x = 0; x < xsize; ++x) {
                float v = row_in[x];
                v = (v < 0) ? 0 : (v > 255 ? 255 * mul : (v * mul));
                uint32_t value = static_cast<uint32_t>(v + 0.5);
                out[i] = value;
                i += inc;
              }
            } else if (bits_per_sample == 16 && little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                float v = row_in[x];
                v = (v < 0) ? 0 : (v > 255 ? 255 * mul : (v * mul));
                uint32_t value = static_cast<uint32_t>(v + 0.5);
                StoreLE16(value, out + i);
                i += inc;
              }
            } else if (bits_per_sample == 16 && !little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                float v = row_in[x];
                v = (v < 0) ? 0 : (v > 255 ? 255 * mul : (v * mul));
                uint32_t value = static_cast<uint32_t>(v + 0.5);
                StoreBE16(value, out + i);
                i += inc;
              }
            }
          },
          "ConvertRGBUint");
    }
  }

  if (want_alpha) {
    // Alpha is stored as a 16-bit ImageU, rather than a floating point Image3F,
    // in the CodecInOut.
    size_t alpha_bits = 0;
    jxl::ImageU alpha_temp;
    if (ib.HasAlpha()) {
      alpha_bits = ib.metadata()->GetAlphaBits();
      if (alpha_bits == 0) {
        return JXL_FAILURE("invalid alpha bit depth");
      }
    } else {
      alpha_temp = jxl::ImageU(xsize, ysize);
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t* JXL_RESTRICT row = alpha_temp.Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row[x] = 255;
        }
      }
      alpha = &alpha_temp;
      alpha_bits = 8;
    }

    if (float_out) {
      if (bits_per_sample != 32) {
        return JXL_FAILURE("non-32-bit float not supported");
      }
      // Multiplier for 0.0-1.0 nominal range.
      float mul = 1.0 / ((1ull << alpha_bits) - 1ull);
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const int64_t y = task;
            size_t i = stride * y +
                       (color_channels * bits_per_sample / jxl::kBitsPerByte);
            const uint16_t* JXL_RESTRICT row_in = alpha->Row(y);
            if (little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                float alpha = row_in[x] * mul;
                StoreLEFloat(alpha, out + i);
                i += inc;
              }
            } else {
              for (size_t x = 0; x < xsize; ++x) {
                float alpha = row_in[x] * mul;
                StoreBEFloat(alpha, out + i);
                i += inc;
              }
            }
          },
          "ConvertAlphaFloat");
    } else {
      if (alpha_bits != 8 && alpha_bits != 16) {
        return JXL_FAILURE("32-bit and 1-bit not yet implemented");
      }

      if (alpha_bits != bits_per_sample) {
        alpha_temp = jxl::ImageU(xsize, ysize);
        // Since both the input and output alpha can have multiple possible
        // bit-depths, this is implemented as a 2-step process: convert to an
        // ImageU with the target bit depth, then store it in the output buffer.
        ConvertAlpha(alpha_bits, *alpha, bits_per_sample, &alpha_temp, pool);
        alpha_bits = bits_per_sample;
        alpha = &alpha_temp;
      }
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const int64_t y = task;
            size_t i = stride * y +
                       (color_channels * bits_per_sample / jxl::kBitsPerByte);
            const uint16_t* JXL_RESTRICT row_in = alpha->Row(y);
            if (alpha_bits == 8) {
              for (size_t x = 0; x < xsize; ++x) {
                out[i] = row_in[x];
                i += inc;
              }
            } else if (alpha_bits == 16 && little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                StoreLE16(row_in[x], out + i);
                i += inc;
              }
            } else if (alpha_bits == 16 && !little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                StoreBE16(row_in[x], out + i);
                i += inc;
              }
            }
          },
          "ConvertAlphaUint");
    }
  }

  return true;
}

Status ConvertImage(Span<const uint8_t> bytes, size_t xsize, size_t ysize,
                    const ColorEncoding& c_current, bool has_alpha,
                    bool alpha_is_premultiplied, size_t bits_per_alpha,
                    size_t bits_per_sample, bool big_endian, bool flipped_y,
                    ThreadPool* pool, ImageBundle* ib) {
  PackedImage desc(xsize, ysize, c_current, has_alpha, alpha_is_premultiplied,
                   bits_per_alpha, bits_per_sample, big_endian, flipped_y);
  return CopyTo(desc, bytes, pool, ib);
}
}  // namespace jxl
#endif  // HWY_ONCE
