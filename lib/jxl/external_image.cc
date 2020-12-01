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

#include <string.h>

#include <algorithm>
#include <array>
#include <functional>
#include <utility>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/external_image.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

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

// Loads a float in big endian
float LoadBEFloat(const uint8_t* p) {
  float value;
  const uint32_t u = LoadBE32(p);
  memcpy(&value, &u, 4);
  return value;
}

// Loads a float in little endian
float LoadLEFloat(const uint8_t* p) {
  float value;
  const uint32_t u = LoadLE32(p);
  memcpy(&value, &u, 4);
  return value;
}

// `out` is allowed to be the same image as `in`, in which case the transform is
// done in-place.
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
          const uint16_t* row_in = in.Row(y);
          uint16_t* row_out = out->Row(y);
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
          const uint16_t* row_in = in.Row(y);
          uint16_t* row_out = out->Row(y);
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

namespace {

typedef void(StoreFuncType)(uint32_t value, uint8_t* dest);
template <StoreFuncType StoreFunc>
void JXL_INLINE StoreFloatRow(const float* JXL_RESTRICT row_in, uint8_t* out,
                              float mul, size_t xsize, size_t bytes_per_pixel) {
  size_t i = 0;
  for (size_t x = 0; x < xsize; ++x) {
    float v = row_in[x];
    v = (v < 0) ? 0 : (v > 255 ? 255 * mul : (v * mul));
    uint32_t value = static_cast<uint32_t>(v + 0.5);
    StoreFunc(value, out + i);
    i += bytes_per_pixel;
  }
}

void JXL_INLINE Store8(uint32_t value, uint8_t* dest) { *dest = value & 0xff; }

}  // namespace

Status ConvertImage(const jxl::ImageBundle& ib, size_t bits_per_sample,
                    bool float_out, bool lossless_float, bool apply_srgb_tf,
                    size_t num_channels, bool little_endian, size_t stride,
                    jxl::ThreadPool* pool, void* out_image, size_t out_size,
                    jxl::Orientation undo_orientation) {
  if (bits_per_sample < 1 || bits_per_sample > 32) {
    return JXL_FAILURE("Invalid bits_per_sample value.");
  }
  // TODO(deymo): Implement 1-bit per pixel packed in 8 samples per byte.
  if (bits_per_sample == 1) {
    return JXL_FAILURE("packed 1-bit per sample is not yet supported");
  }
  size_t xsize = ib.xsize();
  size_t ysize = ib.ysize();

  uint8_t* out = reinterpret_cast<uint8_t*>(out_image);

  bool want_alpha = num_channels == 2 || num_channels == 4;
  size_t color_channels = num_channels <= 2 ? 1 : 3;

  // bytes_per_channel and bytes_per_pixel are only valid for
  // bits_per_sample > 1.
  const size_t bytes_per_channel = DivCeil(bits_per_sample, jxl::kBitsPerByte);
  const size_t bytes_per_pixel = num_channels * bytes_per_channel;

  if (stride < bytes_per_pixel * xsize) {
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
                  i += bytes_per_pixel;
                }
              } else {
                for (size_t x = 0; x < xsize; ++x) {
                  StoreBEFloat(row_in[x], out + i);
                  i += bytes_per_pixel;
                }
              }
            } else {
              if (little_endian) {
                for (size_t x = 0; x < xsize; ++x) {
                  StoreLEFloat(row_in[x] * mul, out + i);
                  i += bytes_per_pixel;
                }
              } else {
                for (size_t x = 0; x < xsize; ++x) {
                  StoreBEFloat(row_in[x] * mul, out + i);
                  i += bytes_per_pixel;
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
    // TODO(deymo): Move the for(c) inside the StoreFloatRow() function so it is
    // more write-cache friendly.
    for (size_t c = 0; c < color_channels; ++c) {
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const int64_t y = task;
            size_t i = stride * y + (c * bits_per_sample / jxl::kBitsPerByte);
            const float* JXL_RESTRICT row_in = color->PlaneRow(c, y);
            // TODO(deymo): add bits_per_sample == 1 case here.
            if (bits_per_sample <= 8) {
              StoreFloatRow<Store8>(row_in, out + i, mul, xsize,
                                    bytes_per_pixel);
            } else if (bits_per_sample <= 16) {
              if (little_endian) {
                StoreFloatRow<StoreLE16>(row_in, out + i, mul, xsize,
                                         bytes_per_pixel);
              } else {
                StoreFloatRow<StoreBE16>(row_in, out + i, mul, xsize,
                                         bytes_per_pixel);
              }
            } else if (bits_per_sample <= 24) {
              if (little_endian) {
                StoreFloatRow<StoreLE24>(row_in, out + i, mul, xsize,
                                         bytes_per_pixel);
              } else {
                StoreFloatRow<StoreBE24>(row_in, out + i, mul, xsize,
                                         bytes_per_pixel);
              }
            } else {
              if (little_endian) {
                StoreFloatRow<StoreLE32>(row_in, out + i, mul, xsize,
                                         bytes_per_pixel);
              } else {
                StoreFloatRow<StoreBE32>(row_in, out + i, mul, xsize,
                                         bytes_per_pixel);
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
                i += bytes_per_pixel;
              }
            } else {
              for (size_t x = 0; x < xsize; ++x) {
                float alpha = row_in[x] * mul;
                StoreBEFloat(alpha, out + i);
                i += bytes_per_pixel;
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
                i += bytes_per_pixel;
              }
            } else if (alpha_bits == 16 && little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                StoreLE16(row_in[x], out + i);
                i += bytes_per_pixel;
              }
            } else if (alpha_bits == 16 && !little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                StoreBE16(row_in[x], out + i);
                i += bytes_per_pixel;
              }
            }
          },
          "ConvertAlphaUint");
    }
  }

  return true;
}

namespace {

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

}  // namespace

Status ConvertImage(Span<const uint8_t> bytes, size_t xsize, size_t ysize,
                    const ColorEncoding& c_current, bool has_alpha,
                    bool alpha_is_premultiplied, size_t bits_per_alpha,
                    size_t bits_per_sample, bool big_endian, bool flipped_y,
                    ThreadPool* pool, ImageBundle* ib) {
  if (bits_per_sample < 1 || bits_per_sample > 32) {
    return JXL_FAILURE("Invalid bits_per_sample value.");
  }
  // TODO(deymo): Implement 1-bit per sample as 8 samples per byte. In
  // any other case we use DivCeil(bits_per_sample, 8) bytes per pixel per
  // channel.
  if (bits_per_sample == 1) {
    return JXL_FAILURE("packed 1-bit per sample is not yet supported");
  }

  const size_t color_channels = c_current.Channels();
  const size_t channels = color_channels + has_alpha;

  // bytes_per_channel and bytes_per_pixel are only valid for
  // bits_per_sample > 1.
  const size_t bytes_per_channel = DivCeil(bits_per_sample, jxl::kBitsPerByte);
  const size_t bytes_per_pixel = channels * bytes_per_channel;

  const size_t row_size = xsize * bytes_per_pixel;
  if (ysize && bytes.size() / ysize < row_size) {
    return JXL_FAILURE("Buffer size is too small");
  }

  const bool little_endian = !big_endian;

  const uint8_t* const in = bytes.data();

  Image3F color(xsize, ysize);
  ImageU alpha;
  if (has_alpha) {
    alpha = ImageU(xsize, ysize);
  }

  // Matches the old behavior of PackedImage.
  // TODO(sboukortt): make this a parameter.
  const bool float_in = bits_per_sample == 32;

  // TODO(sboukortt): and remove this once we use 0-1 instead of 0-255.
  const bool lossless_float = false;

  const auto get_y = [flipped_y, ysize](const size_t y) {
    return flipped_y ? ysize - 1 - y : y;
  };

  if (float_in) {
    if (bits_per_sample != 32) {
      return JXL_FAILURE("non-32-bit float not supported");
    }
    for (size_t c = 0; c < color_channels; ++c) {
      // JXL uses range 0-255 internally, but externally we use range 0-1
      float mul = 255.0f;

      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const size_t y = get_y(task);
            size_t i = row_size * y + (c * bits_per_sample / jxl::kBitsPerByte);
            float* JXL_RESTRICT row_out = color.PlaneRow(c, y);
            if (lossless_float) {
              // for lossless PFM, we need to avoid the * (1./255.) * 255
              // so just interleave and don't touch
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
            } else {
              if (little_endian) {
                for (size_t x = 0; x < xsize; ++x) {
                  row_out[x] = mul * LoadLEFloat(in + i);
                  i += bytes_per_pixel;
                }
              } else {
                for (size_t x = 0; x < xsize; ++x) {
                  row_out[x] = mul * LoadBEFloat(in + i);
                  i += bytes_per_pixel;
                }
              }
            }
          },
          "ConvertRGBFloat");
    }
  } else {
    // Multiplier to convert from the integer range to floating point 0-255
    // range.
    float mul = (bits_per_sample == 32)
                    ? 5.937181414556033e-08  // 255.f / 4294967295
                    : 255.f / ((1ull << bits_per_sample) - 1);
    for (size_t c = 0; c < color_channels; ++c) {
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const size_t y = get_y(task);
            size_t i = row_size * y + c * bytes_per_channel;
            float* JXL_RESTRICT row_out = color.PlaneRow(c, y);
            // TODO(deymo): add bits_per_sample == 1 case here. Also maybe
            // implement masking if bits_per_sample is not a multiple of 8.
            if (bits_per_sample <= 8) {
              LoadFloatRow<Load8>(row_out, in + i, mul, xsize, bytes_per_pixel);
            } else if (bits_per_sample <= 16) {
              if (little_endian) {
                LoadFloatRow<LoadLE16>(row_out, in + i, mul, xsize,
                                       bytes_per_pixel);
              } else {
                LoadFloatRow<LoadBE16>(row_out, in + i, mul, xsize,
                                       bytes_per_pixel);
              }
            } else if (bits_per_sample <= 24) {
              if (little_endian) {
                LoadFloatRow<LoadLE24>(row_out, in + i, mul, xsize,
                                       bytes_per_pixel);
              } else {
                LoadFloatRow<LoadBE24>(row_out, in + i, mul, xsize,
                                       bytes_per_pixel);
              }
            } else {
              if (little_endian) {
                LoadFloatRow<LoadLE32>(row_out, in + i, mul, xsize,
                                       bytes_per_pixel);
              } else {
                LoadFloatRow<LoadBE32>(row_out, in + i, mul, xsize,
                                       bytes_per_pixel);
              }
            }
          },
          "ConvertRGBUint");
    }
  }

  if (color_channels == 1) {
    CopyImageTo(color.Plane(0), &color.Plane(1));
    CopyImageTo(color.Plane(0), &color.Plane(2));
  }

  ib->SetFromImage(std::move(color), c_current);

  if (has_alpha) {
    // Alpha is stored as a 16-bit ImageU, rather than a floating point Image3F,
    // in the CodecInOut.
    // TODO(sboukortt): change that.
    if (float_in) {
      if (bits_per_sample != 32) {
        return JXL_FAILURE("non-32-bit float not supported");
      }
      float mul = (1ull << bits_per_alpha) - 1ull;
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const size_t y = get_y(task);
            size_t i = row_size * y +
                       (color_channels * bits_per_sample / jxl::kBitsPerByte);
            uint16_t* JXL_RESTRICT row_out = alpha.Row(y);
            if (little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = mul * LoadLEFloat(in + i);
                i += bytes_per_pixel;
              }
            } else {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = mul * LoadBEFloat(in + i);
                i += bytes_per_pixel;
              }
            }
          },
          "ConvertAlphaFloat");
    } else {
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const size_t y = get_y(task);
            size_t i = row_size * y +
                       (color_channels * bits_per_sample / jxl::kBitsPerByte);
            uint16_t* JXL_RESTRICT row_out = alpha.Row(y);
            if (bits_per_sample == 8) {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = in[i];
                i += bytes_per_pixel;
              }
            } else if (bits_per_sample == 16 && little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = LoadLE16(in + i);
                i += bytes_per_pixel;
              }
            } else if (bits_per_sample == 16 && !little_endian) {
              for (size_t x = 0; x < xsize; ++x) {
                row_out[x] = LoadBE16(in + i);
                i += bytes_per_pixel;
              }
            }
          },
          "ConvertAlphaUint");

      if (bits_per_sample != bits_per_alpha) {
        ConvertAlpha(bits_per_sample, alpha, bits_per_alpha, &alpha, pool);
      }
    }

    ib->SetAlpha(std::move(alpha), alpha_is_premultiplied);
  }

  return true;
}
}  // namespace jxl
#endif  // HWY_ONCE
