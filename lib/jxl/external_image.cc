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

// Input/output uses the codec.h scaling: nominally 0-1 if in-gamut.
template <class V>
V LinearToSRGB(V encoded) {
  return TF_SRGB().EncodedFromDisplay(encoded);
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
    v = (v < 0) ? 0 : (v > 1 ? mul : (v * mul));
    uint32_t value = static_cast<uint32_t>(v + 0.5);
    StoreFunc(value, out + i);
    i += bytes_per_pixel;
  }
}

void JXL_INLINE Store8(uint32_t value, uint8_t* dest) { *dest = value & 0xff; }

}  // namespace

void LinearToSRGBInPlace(jxl::ThreadPool* pool, Image3F* image,
                         size_t color_channels) {
  return HWY_DYNAMIC_DISPATCH(LinearToSRGBInPlace)(pool, image, color_channels);
}

Status ConvertImage(const jxl::ImageBundle& ib, size_t bits_per_sample,
                    bool float_out, bool apply_srgb_tf, size_t num_channels,
                    JxlEndianness endianness, size_t stride,
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
    return JXL_FAILURE(
        "stride is smaller than scanline width in bytes: %zu vs %zu", stride,
        bytes_per_pixel * xsize);
  }

  const Image3F* color = &ib.color();
  Image3F temp_color;
  const ImageF* alpha = ib.HasAlpha() ? &ib.alpha() : nullptr;
  ImageF temp_alpha;
  if (apply_srgb_tf) {
    temp_color = CopyImage(*color);
    LinearToSRGBInPlace(pool, &temp_color, color_channels);
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

  const bool little_endian =
      endianness == JXL_LITTLE_ENDIAN ||
      (endianness == JXL_NATIVE_ENDIAN && IsLittleEndian());

  if (float_out) {
    if (bits_per_sample != 32) {
      return JXL_FAILURE("non-32-bit float not supported");
    }
    for (size_t c = 0; c < color_channels; ++c) {
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const int64_t y = task;
            size_t i = stride * y + (c * bits_per_sample / jxl::kBitsPerByte);
            const float* JXL_RESTRICT row_in = color->PlaneRow(c, y);
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
          },
          "ConvertRGBFloat");
    }
  } else {
    // Multiplier to convert from floating point 0-1 range to the integer
    // range.
    float mul = (1ull << bits_per_sample) - 1;
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
    jxl::ImageF alpha_temp;
    if (ib.HasAlpha()) {
      if (ib.metadata()->GetAlphaBits() == 0) {
        return JXL_FAILURE("invalid alpha bit depth");
      }
    } else {
      // TODO(sboukortt): have a different path altogether below instead of
      // filling an image with ones just to read from it immediately afterwards.
      alpha_temp = jxl::ImageF(xsize, ysize);
      FillImage(1.f, &alpha_temp);
      alpha = &alpha_temp;
    }

    if (float_out) {
      if (bits_per_sample != 32) {
        return JXL_FAILURE("non-32-bit float not supported");
      }
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const int64_t y = task;
            size_t i = stride * y +
                       (color_channels * bits_per_sample / jxl::kBitsPerByte);
            const float* JXL_RESTRICT row_in = alpha->Row(y);
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
          },
          "ConvertAlphaFloat");
    } else {
      if (bits_per_sample != 8 && bits_per_sample != 16) {
        return JXL_FAILURE("32-bit and 1-bit not yet implemented");
      }

      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const int64_t y = task;
            size_t i = stride * y +
                       (color_channels * bits_per_sample / jxl::kBitsPerByte);
            const float* JXL_RESTRICT row_in = alpha->Row(y);
            if (bits_per_sample == 8) {
              StoreFloatRow<Store8>(row_in, out + i, 255.f, xsize,
                                    bytes_per_pixel);
            } else if (bits_per_sample == 16) {
              if (little_endian) {
                StoreFloatRow<StoreLE16>(row_in, out + i, 65535.f, xsize,
                                         bytes_per_pixel);
              } else {
                StoreFloatRow<StoreBE16>(row_in, out + i, 65535.f, xsize,
                                         bytes_per_pixel);
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
                    bool alpha_is_premultiplied, size_t bits_per_sample,
                    JxlEndianness endianness, bool flipped_y, ThreadPool* pool,
                    ImageBundle* ib) {
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

  const bool little_endian =
      endianness == JXL_LITTLE_ENDIAN ||
      (endianness == JXL_NATIVE_ENDIAN && IsLittleEndian());

  const uint8_t* const in = bytes.data();

  Image3F color(xsize, ysize);
  ImageF alpha;
  if (has_alpha) {
    alpha = ImageF(xsize, ysize);
  }

  // Matches the old behavior of PackedImage.
  // TODO(sboukortt): make this a parameter.
  const bool float_in = bits_per_sample == 32;

  const auto get_y = [flipped_y, ysize](const size_t y) {
    return flipped_y ? ysize - 1 - y : y;
  };

  if (float_in) {
    if (bits_per_sample != 32) {
      return JXL_FAILURE("non-32-bit float not supported");
    }
    for (size_t c = 0; c < color_channels; ++c) {
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const size_t y = get_y(task);
            size_t i =
                row_size * task + (c * bits_per_sample / jxl::kBitsPerByte);
            float* JXL_RESTRICT row_out = color.PlaneRow(c, y);
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
          },
          "ConvertRGBFloat");
    }
  } else {
    // Multiplier to convert from the integer range to floating point 0-1 range.
    float mul = 1. / ((1ull << bits_per_sample) - 1);
    for (size_t c = 0; c < color_channels; ++c) {
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const size_t y = get_y(task);
            size_t i = row_size * task + c * bytes_per_channel;
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
    if (float_in) {
      if (bits_per_sample != 32) {
        return JXL_FAILURE("non-32-bit float not supported");
      }
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const size_t y = get_y(task);
            size_t i = row_size * task +
                       (color_channels * bits_per_sample / jxl::kBitsPerByte);
            float* JXL_RESTRICT row_out = alpha.Row(y);
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
          },
          "ConvertAlphaFloat");
    } else {
      float mul = 1. / ((1ull << bits_per_sample) - 1);
      RunOnPool(
          pool, 0, static_cast<uint32_t>(ysize), ThreadPool::SkipInit(),
          [&](const int task, int /*thread*/) {
            const size_t y = get_y(task);
            size_t i = row_size * task + color_channels * bytes_per_channel;
            float* JXL_RESTRICT row_out = alpha.Row(y);
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
          "ConvertAlphaUint");
    }

    ib->SetAlpha(std::move(alpha), alpha_is_premultiplied);
  }

  return true;
}
}  // namespace jxl
#endif  // HWY_ONCE
