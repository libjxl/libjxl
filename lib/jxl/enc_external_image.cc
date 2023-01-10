// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_external_image.h"

#include <string.h>

#include <algorithm>
#include <array>
#include <functional>
#include <utility>
#include <vector>

#include "jxl/types.h"
#include "lib/jxl/alpha.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"

namespace jxl {
namespace {

// Based on highway scalar implementation, for testing
float LoadFloat16(uint16_t bits16) {
  const uint32_t sign = bits16 >> 15;
  const uint32_t biased_exp = (bits16 >> 10) & 0x1F;
  const uint32_t mantissa = bits16 & 0x3FF;

  // Subnormal or zero
  if (biased_exp == 0) {
    const float subnormal = (1.0f / 16384) * (mantissa * (1.0f / 1024));
    return sign ? -subnormal : subnormal;
  }

  // Normalized: convert the representation directly (faster than ldexp/tables).
  const uint32_t biased_exp32 = biased_exp + (127 - 15);
  const uint32_t mantissa32 = mantissa << (23 - 10);
  const uint32_t bits32 = (sign << 31) | (biased_exp32 << 23) | mantissa32;

  float result;
  memcpy(&result, &bits32, 4);
  return result;
}

float LoadLEFloat16(const uint8_t* p) {
  uint16_t bits16 = LoadLE16(p);
  return LoadFloat16(bits16);
}

float LoadBEFloat16(const uint8_t* p) {
  uint16_t bits16 = LoadBE16(p);
  return LoadFloat16(bits16);
}

typedef uint32_t(LoadFuncType)(const uint8_t* p);
template <LoadFuncType LoadFunc>
void JXL_INLINE LoadFloatRow(float* JXL_RESTRICT row_out, const uint8_t* in,
                             float mul, size_t xsize, size_t bytes_per_pixel) {
  size_t i = 0;
  for (size_t x = 0; x < xsize; ++x) {
    row_out[x] = mul * LoadFunc(in + i);
    i += bytes_per_pixel;
  }
}

uint32_t JXL_INLINE Load8(const uint8_t* p) { return *p; }

size_t JxlDataTypeBytes(JxlDataType data_type) {
  switch (data_type) {
    case JXL_TYPE_UINT8:
      return 1;
    case JXL_TYPE_UINT16:
      return 2;
    case JXL_TYPE_FLOAT16:
      return 2;
    case JXL_TYPE_FLOAT:
      return 4;
    default:
      return 0;
  }
}

}  // namespace

Status ConvertFromExternal(Span<const uint8_t> bytes, size_t xsize,
                           size_t ysize, size_t bits_per_sample,
                           JxlPixelFormat format, size_t c, ThreadPool* pool,
                           ImageF* channel) {
  if (format.data_type == JXL_TYPE_UINT8) {
    JXL_RETURN_IF_ERROR(bits_per_sample > 0 && bits_per_sample <= 8);
  } else if (format.data_type == JXL_TYPE_UINT16) {
    JXL_RETURN_IF_ERROR(bits_per_sample > 8 && bits_per_sample <= 16);
  } else if (format.data_type == JXL_TYPE_FLOAT16) {
    JXL_RETURN_IF_ERROR(bits_per_sample == 16);
  } else if (format.data_type == JXL_TYPE_FLOAT) {
    JXL_RETURN_IF_ERROR(bits_per_sample == 32);
  } else {
    JXL_FAILURE("unsupported pixel format data type %d", format.data_type);
  }
  size_t bytes_per_channel = JxlDataTypeBytes(format.data_type);
  size_t bytes_per_pixel = format.num_channels * bytes_per_channel;
  size_t pixel_offset = c * bytes_per_channel;

  const size_t last_row_size = xsize * bytes_per_pixel;
  const size_t align = format.align;
  const size_t row_size =
      (align > 1 ? jxl::DivCeil(last_row_size, align) * align : last_row_size);
  const size_t bytes_to_read = row_size * (ysize - 1) + last_row_size;
  if (xsize == 0 || ysize == 0) return JXL_FAILURE("Empty image");
  if (bytes.size() < bytes_to_read) {
    return JXL_FAILURE("Buffer size is too small, expected: %" PRIuS
                       " got: %" PRIuS " (Image: %" PRIuS "x%" PRIuS
                       "x%u, bytes_per_channel: %" PRIuS ")",
                       bytes_to_read, bytes.size(), xsize, ysize,
                       format.num_channels, bytes_per_channel);
  }
  JXL_ASSERT(channel->xsize() == xsize);
  JXL_ASSERT(channel->ysize() == ysize);
  // Too large buffer is likely an application bug, so also fail for that.
  // Do allow padding to stride in last row though.
  if (bytes.size() > row_size * ysize) {
    return JXL_FAILURE("Buffer size is too large");
  }

  const bool little_endian =
      format.endianness == JXL_LITTLE_ENDIAN ||
      (format.endianness == JXL_NATIVE_ENDIAN && IsLittleEndian());

  const uint8_t* const in = bytes.data();
  if (format.data_type == JXL_TYPE_FLOAT ||
      format.data_type == JXL_TYPE_FLOAT16) {
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::NoInit,
        [&](const uint32_t task, size_t /*thread*/) {
          const size_t y = task;
          size_t i = row_size * task + pixel_offset;
          float* JXL_RESTRICT row_out = channel->Row(y);
          if (format.data_type == JXL_TYPE_FLOAT16) {
            if (little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = LoadLEFloat16(in + i);
                i += bytes_per_pixel;
              }
            } else {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = LoadBEFloat16(in + i);
                i += bytes_per_pixel;
              }
            }
          } else {
            if (little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = LoadLEFloat(in + i);
                i += bytes_per_pixel;
              }
            } else {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = LoadBEFloat(in + i);
                i += bytes_per_pixel;
              }
            }
          }
        },
        "ConvertExtraChannelFloat"));
  } else {
    float mul = 1. / ((1ull << bits_per_sample) - 1);
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, static_cast<uint32_t>(ysize), ThreadPool::NoInit,
        [&](const uint32_t task, size_t /*thread*/) {
          const size_t y = task;
          size_t i = row_size * task + pixel_offset;
          float* JXL_RESTRICT row_out = channel->Row(y);
          if (format.data_type == JXL_TYPE_UINT8) {
            LoadFloatRow<Load8>(row_out, in + i, mul, xsize, bytes_per_pixel);
          } else {
            if (little_endian) {
              LoadFloatRow<LoadLE16>(row_out, in + i, mul, xsize,
                                     bytes_per_pixel);
            } else {
              LoadFloatRow<LoadBE16>(row_out, in + i, mul, xsize,
                                     bytes_per_pixel);
            }
          }
        },
        "ConvertExtraChannelUint"));
  }

  return true;
}
Status ConvertFromExternal(Span<const uint8_t> bytes, size_t xsize,
                           size_t ysize, const ColorEncoding& c_current,
                           size_t bits_per_sample, JxlPixelFormat format,
                           ThreadPool* pool, ImageBundle* ib) {
  const size_t color_channels = c_current.Channels();
  bool has_alpha = format.num_channels == 2 || format.num_channels == 4;
  if (format.num_channels < color_channels) {
    return JXL_FAILURE("Expected %" PRIuS
                       " color channels, received only %u channels",
                       color_channels, format.num_channels);
  }

  Image3F color(xsize, ysize);
  for (size_t c = 0; c < color_channels; ++c) {
    JXL_RETURN_IF_ERROR(ConvertFromExternal(bytes, xsize, ysize,
                                            bits_per_sample, format, c, pool,
                                            &color.Plane(c)));
  }
  if (color_channels == 1) {
    CopyImageTo(color.Plane(0), &color.Plane(1));
    CopyImageTo(color.Plane(0), &color.Plane(2));
  }
  ib->SetFromImage(std::move(color), c_current);

  // Passing an interleaved image with an alpha channel to an image that doesn't
  // have alpha channel just discards the passed alpha channel.
  if (has_alpha && ib->HasAlpha()) {
    ImageF alpha(xsize, ysize);
    JXL_RETURN_IF_ERROR(
        ConvertFromExternal(bytes, xsize, ysize, bits_per_sample, format,
                            format.num_channels - 1, pool, &alpha));
    ib->SetAlpha(std::move(alpha));
  } else if (!has_alpha && ib->HasAlpha()) {
    // if alpha is not passed, but it is expected, then assume
    // it is all-opaque
    ImageF alpha(xsize, ysize);
    FillImage(1.0f, &alpha);
    ib->SetAlpha(std::move(alpha));
  }

  return true;
}

Status BufferToImageF(const JxlPixelFormat& pixel_format, size_t xsize,
                      size_t ysize, const void* buffer, size_t size,
                      ThreadPool* pool, ImageF* channel) {
  size_t bitdepth = JxlDataTypeBytes(pixel_format.data_type) * kBitsPerByte;
  return ConvertFromExternal(
      jxl::Span<const uint8_t>(static_cast<const uint8_t*>(buffer), size),
      xsize, ysize, bitdepth, pixel_format, 0, pool, channel);
}

Status BufferToImageBundle(const JxlPixelFormat& pixel_format, uint32_t xsize,
                           uint32_t ysize, const void* buffer, size_t size,
                           jxl::ThreadPool* pool,
                           const jxl::ColorEncoding& c_current,
                           jxl::ImageBundle* ib) {
  size_t bitdepth = JxlDataTypeBytes(pixel_format.data_type) * kBitsPerByte;
  JXL_RETURN_IF_ERROR(ConvertFromExternal(
      jxl::Span<const uint8_t>(static_cast<const uint8_t*>(buffer), size),
      xsize, ysize, c_current, bitdepth, pixel_format, pool, ib));
  ib->VerifyMetadata();

  return true;
}

}  // namespace jxl
