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

#include "jxl/external_image.h"

#include <string.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include <third_party/highway/hwy/base.h>  // EnableIf
#include "jxl/alpha.h"
#include "jxl/base/byte_order.h"
#include "jxl/base/cache_aligned.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/color_management.h"
#include "jxl/common.h"

namespace jxl {
namespace {

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
                                         const Image3F* JXL_RESTRICT image) {
    const size_t xsize = image->xsize();
    float* JXL_RESTRICT row0 = const_cast<float*>(image->PlaneRow(0, y));
    for (size_t x = 0; x < xsize; ++x) {
      row0[x] = row_temp[x];
    }

    for (size_t c = 1; c < 3; ++c) {
      float* JXL_RESTRICT row = const_cast<float*>(image->PlaneRow(c, y));
      memcpy(row, row0, xsize * sizeof(float));
    }
  }

  static JXL_INLINE void Temp255ToImage3(Channels3 /*tag*/,
                                         const float* JXL_RESTRICT row_temp,
                                         size_t y,
                                         const Image3F* JXL_RESTRICT image) {
    float* JXL_RESTRICT row_image0 = const_cast<float*>(image->PlaneRow(0, y));
    float* JXL_RESTRICT row_image1 = const_cast<float*>(image->PlaneRow(1, y));
    float* JXL_RESTRICT row_image2 = const_cast<float*>(image->PlaneRow(2, y));
    for (size_t x = 0; x < image->xsize(); ++x) {
      row_image0[x] = row_temp[3 * x + 0];
      row_image1[x] = row_temp[3 * x + 1];
      row_image2[x] = row_temp[3 * x + 2];
    }
  }

  static JXL_INLINE void Temp255ToImage3(Channels2 /*tag*/,
                                         const float* JXL_RESTRICT row_temp,
                                         size_t y,
                                         const Image3F* JXL_RESTRICT image) {
    Temp255ToImage3(Channels1(), row_temp, y, image);
  }

  static JXL_INLINE void Temp255ToImage3(Channels4 /*tag*/,
                                         const float* JXL_RESTRICT row_temp,
                                         size_t y,
                                         const Image3F* JXL_RESTRICT image) {
    Temp255ToImage3(Channels3(), row_temp, y, image);
  }
};

// Step 2t: type conversion

// Same naming convention as Image: B=u8, U=u16, F=f32. kSize enables generic
// functions with Type and Order template arguments. 1=PBM.
struct Type1 {
  static const char* Name() { return "1"; }
  static constexpr size_t kSize = 0;
  static constexpr uint16_t kMaxAlpha = 0xFF;
};
struct TypeB {
  static const char* Name() { return "B"; }
  static constexpr size_t kSize = 1;
  static constexpr uint16_t kMaxAlpha = 0xFF;
};
struct TypeU {
  static const char* Name() { return "U"; }
  static constexpr size_t kSize = 2;
  static constexpr uint16_t kMaxAlpha = 0xFFFF;
};
struct TypeF {
  static const char* Name() { return "F"; }
  static constexpr size_t kSize = 4;
  static constexpr uint16_t kMaxAlpha = 0xFFFF;
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

// Multithreaded color space transform from IO to ExternalImage.
class Transformer {
 public:
  Transformer(ThreadPool* pool, const Image3F& color, const Rect& rect,
              const bool has_alpha, const ImageU* alpha,
              ExternalImage* external, const ColorEncoding& c_src,
              const ColorEncoding& c_dst)
      : pool_(pool),
        color_(color),
        rect_(rect),
        alpha_(alpha),
        external_(external),
        want_alpha_(has_alpha && external->HasAlpha()),
        c_src_(c_src),
        c_dst_(c_dst) {
    JXL_ASSERT(rect.IsInside(color));
    JXL_ASSERT(SameSize(rect, *external));
  }

  // Converts in the specified direction (To*).
  template <class To, class Extent, class Cast>
  Status Run(Extent* extents, const Cast& cast) {
    const size_t bytes = DivCeil(external_->BitsPerSample(), kBitsPerByte);
    const bool big_endian = external_->BigEndian();
    if (external_->BitsPerSample() == 1) {
      return DispatchType<To, Type1, OrderLE>(extents, cast);
    } else if (bytes == 1) {
      return DispatchType<To, TypeB, OrderLE>(extents, cast);
    } else if (bytes == 2 && big_endian) {
      return DispatchType<To, TypeU, OrderBE>(extents, cast);
    } else if (bytes == 2) {
      return DispatchType<To, TypeU, OrderLE>(extents, cast);
    } else if (bytes == 4 && big_endian) {
      return DispatchType<To, TypeF, OrderBE>(extents, cast);
    } else if (bytes == 4) {
      return DispatchType<To, TypeF, OrderLE>(extents, cast);
    } else {
      return JXL_FAILURE("Unsupported BitsPerSample");
    }
    return true;
  }

 private:
  Status InitBeforeRun(size_t num_threads) {
    if (init_called_) return true;
    init_called_ = true;
#if JXL_EXT_VERBOSE >= 1
    printf("ExtImg Transformer %s->%s\n", Description(c_src_).c_str(),
           Description(c_dst_).c_str());
#endif
    return transform_.Init(c_src_, c_dst_, rect_.xsize(), num_threads);
  }

  // First pass: only needed for ExtentsDynamic/CastUnused.
  template <class Type, class Order, class Channels>
  JXL_INLINE void DoRow(ToExternal1 /*tag*/, ExtentsDynamic* extents,
                        const CastUnused /*unused*/, const size_t y,
                        const size_t thread) {
    float* JXL_RESTRICT row_temp = extents->RowTemp(y);

    Interleave::Image3ToTemp01(Channels(), y, color_, rect_, row_temp);

#if JXL_EXT_VERBOSE >= 2
    const float in0 = row_temp[3 * kX + 0], in1 = row_temp[3 * kX + 1];
    const float in2 = row_temp[3 * kX + 2];
#endif

    DoColorSpaceTransform(&transform_, thread, row_temp, row_temp);

#if JXL_EXT_VERBOSE >= 2
    printf("ToExt1: in %.4f %.4f %.4f; xform %.4f %.4f %.4f\n", in0, in1, in2,
           row_temp[3 * kX + 0], row_temp[3 * kX + 1], row_temp[3 * kX + 2]);
#endif

    extents->Update(thread, row_temp);
  }

  // Second pass: only needed for ExtentsDynamic/CastRescale.
  template <class Type, class Order, class Channels>
  JXL_INLINE void DoRow(ToExternal2 /*tag*/, ExtentsDynamic* extents,
                        const CastRescale01& cast, const size_t y,
                        const size_t /*thread*/) {
    const float* JXL_RESTRICT row_temp = extents->RowTemp(y);
    uint8_t* JXL_RESTRICT row_external = external_->Row(y);
    Demux::TempToExternal(Type(), Order(), Channels(), rect_.xsize(), row_temp,
                          cast, row_external);

#if JXL_EXT_VERBOSE >= 2
    printf("ToExt2: ext %3d %3d %3d\n", row_external[3 * kX + 0],
           row_external[3 * kX + 1], row_external[3 * kX + 2]);
#endif

    const uint16_t* JXL_RESTRICT row_alpha =
        want_alpha_ ? alpha_->ConstRow(y) : nullptr;
    Demux::AlphaToExternal(Type(), Order(), Channels(), rect_.xsize(),
                           row_alpha, row_external);
  }

  // Single-pass: only works for ExtentsStatic.
  template <class Type, class Order, class Channels, class Cast>
  JXL_INLINE void DoRow(ToExternal /*tag*/, ExtentsStatic* /*unused*/,
                        const Cast& cast, const size_t y, const size_t thread) {
    float* JXL_RESTRICT row_temp = transform_.BufDst(thread);
    Interleave::Image3ToTemp01(Channels(), y, color_, rect_, row_temp);

#if JXL_EXT_VERBOSE >= 2
    // Save inputs for printing before in-place transform overwrites them.
    const float in0 = row_temp[3 * kX + 0];
    const float in1 = row_temp[3 * kX + 1];
    const float in2 = row_temp[3 * kX + 2];
#endif
    DoColorSpaceTransform(&transform_, thread, row_temp, row_temp);

    uint8_t* JXL_RESTRICT row_external = external_->Row(y);
    Demux::TempToExternal(Type(), Order(), Channels(), rect_.xsize(), row_temp,
                          cast, row_external);

#if JXL_EXT_VERBOSE >= 2
    const float tmp0 = row_temp[3 * kX + 0];
    const float tmp1 = row_temp[3 * kX + 1];
    const float tmp2 = row_temp[3 * kX + 2];
    // Convert back so we can print the external values
    Demux::ExternalToTemp(Type(), Order(), Channels(), rect_.xsize(),
                          row_external, cast, row_temp);
    printf("ToExt(%s%s %s): tmp %.4f %.4f %.4f|%.4f %.4f %.4f|%.4f %.4f %.4f\n",
           Channels::Name(), Type::Name(), Cast::Name(), in0, in1, in2, tmp0,
           tmp1, tmp2, row_temp[3 * kX + 0], row_temp[3 * kX + 1],
           row_temp[3 * kX + 2]);
#endif

    const uint16_t* JXL_RESTRICT row_alpha =
        want_alpha_ ? alpha_->ConstRow(y) : nullptr;
    Demux::AlphaToExternal(Type(), Order(), Channels(), rect_.xsize(),
                           row_alpha, row_external);
  }

  // Closure callable by ThreadPool.
  template <class To, class Type, class Order, class Channels, class Extent,
            class Cast>
  class Bind {
   public:
    explicit Bind(Transformer* converter, Extent* extents, const Cast& cast)
        : xform_(converter), extents_(extents), cast_(cast) {}

    JXL_INLINE void operator()(const int task, const int thread) const {
      xform_->DoRow<Type, Order, Channels>(To(), extents_, cast_, task, thread);
    }

   private:
    Transformer* xform_;  // not owned
    Extent* extents_;     // not owned
    const Cast cast_;
  };

  template <class To, class Type, class Order, class Channels, class Extent,
            class Cast>
  Status DoRows(Extent* extents, const Cast& cast) {
    return RunOnPool(
        pool_, 0, rect_.ysize(),
        [this, extents](const size_t num_threads) {
          return InitBeforeRun(num_threads) &&
                 extents->SetNumThreads(num_threads);
        },
        Bind<To, Type, Order, Channels, Extent, Cast>(this, extents, cast),
        "ExtImg xform");
  }

  // Calls the instantiation with the matching Type and Order.
  template <class To, class Type, class Order, class Extent, class Cast>
  Status DispatchType(Extent* extents, const Cast& cast) {
    if (external_->IsGray()) {
      if (external_->HasAlpha()) {
        return DoRows<To, Type, Order, Channels2>(extents, cast);
      } else {
        return DoRows<To, Type, Order, Channels1>(extents, cast);
      }
    } else {
      if (external_->HasAlpha()) {
        return DoRows<To, Type, Order, Channels4>(extents, cast);
      } else {
        return DoRows<To, Type, Order, Channels3>(extents, cast);
      }
    }
  }

  ThreadPool* pool_;  // not owned
  const Image3F& color_;
  const Rect rect_;          // whence in color_ to copy, and output size.
  const ImageU* alpha_;      // not owned
  ExternalImage* external_;  // not owned

  bool want_alpha_;
  // Whether InitBeforeRun() was already called.
  bool init_called_ = false;

  ColorSpaceTransform transform_;
  const ColorEncoding& c_src_;
  const ColorEncoding& c_dst_;
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
    if (ib->metadata()->m2.have_animation) return true;

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

    // Keep alpha if at least one value is (semi)transparent.
    if (and_bits != max_alpha) {
      ib->SetAlpha(std::move(alpha_), desc_->alpha_is_premultiplied);
    } else {
      ib->RemoveAlpha();
    }

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

}  // namespace

ExternalImage::ExternalImage(const size_t xsize, const size_t ysize,
                             const ColorEncoding& c_current,
                             const bool has_alpha,
                             const bool alpha_is_premultiplied,
                             const size_t bits_per_alpha,
                             const size_t bits_per_sample,
                             const bool big_endian)
    : desc_(xsize, ysize, c_current, has_alpha, alpha_is_premultiplied,
            bits_per_alpha, bits_per_sample, big_endian, /*flipped_y=*/false),
      is_healthy_(true) {
  JXL_ASSERT(1 <= desc_.channels && desc_.channels <= 4);
  JXL_ASSERT(1 <= bits_per_sample && bits_per_sample <= 32);
  if (has_alpha) JXL_ASSERT(1 <= bits_per_alpha && bits_per_alpha <= 32);
  // NOTE: codec_* are responsible for ensuring xsize/ysize are small enough.
  bytes_.resize(desc_.ysize * desc_.row_size);
  if (bytes_.empty()) {
    is_healthy_ = false;
    JXL_NOTIFY_ERROR("Zero-dimensioned image");
  }
}

ExternalImage::ExternalImage(
    const size_t xsize, const size_t ysize, const ColorEncoding& c_current,
    const bool has_alpha, const bool alpha_is_premultiplied,
    const size_t bits_per_alpha, const size_t bits_per_sample,
    const bool big_endian, const uint8_t* bytes, const uint8_t* end)
    : ExternalImage(xsize, ysize, c_current, has_alpha, alpha_is_premultiplied,
                    bits_per_alpha, bits_per_sample, big_endian) {
  JXL_ASSERT(end > bytes);
  if (!is_healthy_) return;
  if (bytes + bytes_.size() > end) {
    is_healthy_ = false;
    JXL_NOTIFY_ERROR("Not enough bytes given to fill image");
    return;
  }
  memcpy(bytes_.data(), bytes, bytes_.size());
}

ExternalImage::ExternalImage(ThreadPool* pool, const Image3F& color,
                             const Rect& rect, const ColorEncoding& c_current,
                             const ColorEncoding& c_desired,
                             const bool has_alpha,
                             const bool alpha_is_premultiplied,
                             const ImageU* alpha, size_t bits_per_alpha,
                             size_t bits_per_sample, bool big_endian,
                             CodecIntervals* temp_intervals)
    : ExternalImage(rect.xsize(), rect.ysize(), c_desired, has_alpha,
                    alpha_is_premultiplied, bits_per_alpha, bits_per_sample,
                    big_endian) {
  if (!is_healthy_) return;
  Transformer transformer(pool, color, rect, has_alpha, alpha, this, c_current,
                          c_desired);

  const CodecInterval ext_interval = GetInterval(bits_per_sample);

  if (bits_per_sample == 32) {
    ExtentsStatic extents;
    const CastFloat01 cast(ext_interval);  // only multiply by const
    if (!transformer.Run<ToExternal>(&extents, cast)) {
      is_healthy_ = false;
      JXL_NOTIFY_ERROR("Color transform with CastFloat01 failed");
    }
  } else if (temp_intervals != nullptr) {
    // Store temp to separate image and obtain per-channel intervals.
    ExtentsDynamic extents(desc_.xsize, desc_.ysize, c_desired);
    const CastUnused unused;
    if (!transformer.Run<ToExternal1>(&extents, unused)) {
      is_healthy_ = false;
      JXL_NOTIFY_ERROR("Color transform with CastUnused failed");
      return;
    }
    extents.Finalize(temp_intervals);

    // Rescale based on temp_intervals.
    const CastRescale01 cast(*temp_intervals, ext_interval);
    if (!transformer.Run<ToExternal2>(&extents, cast)) {
      is_healthy_ = false;
      JXL_NOTIFY_ERROR("Color transform with CastRescale01 failed");
    }
  } else {
    ExtentsStatic extents;
    const CastClip01 cast(ext_interval);  // clip
    if (!transformer.Run<ToExternal>(&extents, cast)) {
      is_healthy_ = false;
      JXL_NOTIFY_ERROR("Color transform with CastClip01 failed");
    }
  }
}

Status ExternalImage::CopyTo(const CodecIntervals* temp_intervals,
                             ThreadPool* pool, ImageBundle* ib) const {
  JXL_ASSERT(IsHealthy());  // Caller should have checked beforehand.
  JXL_RETURN_IF_ERROR(desc_.ValidBufferSize(bytes_.size()));

  Converter converter(pool, desc_, bytes_.data());

  const CodecInterval ext_interval = GetInterval(desc_.bits_per_sample);

  if (desc_.bits_per_sample == 32) {
    const CastFloat255 cast(ext_interval);
    JXL_RETURN_IF_ERROR(converter.Run(cast));
  } else if (temp_intervals != nullptr) {
    const CastRescale255 cast(*temp_intervals, ext_interval);
    JXL_RETURN_IF_ERROR(converter.Run(cast));
  } else {
    const CastClip255 cast(ext_interval);
    JXL_RETURN_IF_ERROR(converter.Run(cast));
  }

  return converter.MoveTo(ib);
}

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

}  // namespace jxl
