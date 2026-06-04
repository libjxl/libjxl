// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/tiff.h"

#include <jxl/codestream_header.h>
#include <jxl/types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "lib/extras/size_constraints.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"

#if JPEGXL_ENABLE_TIFF
#include <tiffio.h>
#endif

namespace jxl {
namespace extras {

#if JPEGXL_ENABLE_TIFF
namespace {

struct TiffMemoryReader {
  const uint8_t* data;
  toff_t size;
  toff_t pos;
};

tmsize_t ReadProc(thandle_t handle, void* out, tmsize_t size) {
  if (size <= 0) return 0;
  TiffMemoryReader* reader = reinterpret_cast<TiffMemoryReader*>(handle);
  const toff_t remaining = reader->size - reader->pos;
  const tmsize_t to_read = static_cast<tmsize_t>(std::min<toff_t>(
      remaining, static_cast<toff_t>(std::numeric_limits<tmsize_t>::max())));
  const tmsize_t clamped = std::min(size, to_read);
  memcpy(out, reader->data + reader->pos, static_cast<size_t>(clamped));
  reader->pos += static_cast<toff_t>(clamped);
  return clamped;
}

tmsize_t WriteProc(thandle_t handle, void* data, tmsize_t size) {
  (void)handle;
  (void)data;
  (void)size;
  return 0;
}

int CloseProc(thandle_t handle) {
  (void)handle;
  return 0;
}

toff_t SeekProc(thandle_t handle, toff_t offset, int whence) {
  TiffMemoryReader* reader = reinterpret_cast<TiffMemoryReader*>(handle);
  toff_t base = 0;
  if (whence == SEEK_SET) {
    base = 0;
  } else if (whence == SEEK_CUR) {
    base = reader->pos;
  } else if (whence == SEEK_END) {
    base = reader->size;
  } else {
    return static_cast<toff_t>(-1);
  }
  if (offset > 0 && base > std::numeric_limits<toff_t>::max() - offset) {
    return static_cast<toff_t>(-1);
  }
  const toff_t new_pos = base + offset;
  if (new_pos > reader->size) return static_cast<toff_t>(-1);
  reader->pos = new_pos;
  return reader->pos;
}

toff_t SizeProc(thandle_t handle) {
  TiffMemoryReader* reader = reinterpret_cast<TiffMemoryReader*>(handle);
  return reader->size;
}

int MapProc(thandle_t handle, void** base, toff_t* size) {
  (void)handle;
  *base = nullptr;
  *size = 0;
  return 0;
}

void UnmapProc(thandle_t handle, void* base, toff_t size) {
  (void)handle;
  (void)base;
  (void)size;
}

bool IsTIFF(const Span<const uint8_t> bytes) {
  if (bytes.size() < 4) return false;
  return (bytes[0] == 'I' && bytes[1] == 'I' && bytes[2] == 42 &&
          bytes[3] == 0) ||
         (bytes[0] == 'M' && bytes[1] == 'M' && bytes[2] == 0 &&
          bytes[3] == 42);
}

uint16_t LoadNativeU16(const uint8_t* p) {
  uint16_t value;
  memcpy(&value, p, sizeof(value));
  return value;
}

float LoadNativeFloat(const uint8_t* p) {
  float value;
  memcpy(&value, p, sizeof(value));
  return value;
}

double LoadNativeDouble(const uint8_t* p) {
  double value;
  memcpy(&value, p, sizeof(value));
  return value;
}

void StoreNativeU16(uint16_t value, uint8_t* p) { memcpy(p, &value, 2); }

void StoreNativeFloat(float value, uint8_t* p) { memcpy(p, &value, 4); }

bool HasAlpha(TIFF* tif, uint16_t samples, uint16_t color_channels,
              bool* premultiplied) {
  *premultiplied = false;
  if (samples <= color_channels) return false;
  uint16_t extra_count = 0;
  const uint16_t* extra_samples = nullptr;
  if (TIFFGetField(tif, TIFFTAG_EXTRASAMPLES, &extra_count, &extra_samples) !=
          1 ||
      extra_count == 0 || extra_samples == nullptr) {
    return false;
  }
  if (extra_samples[0] == EXTRASAMPLE_ASSOCALPHA) {
    *premultiplied = true;
    return true;
  }
  return extra_samples[0] == EXTRASAMPLE_UNASSALPHA;
}

void StoreSample(const uint8_t* src, uint16_t bits, uint16_t sample_format,
                 bool invert, uint8_t* dst) {
  if (sample_format == SAMPLEFORMAT_UINT && bits == 8) {
    dst[0] = invert ? 255u - src[0] : src[0];
  } else if (sample_format == SAMPLEFORMAT_UINT && bits == 16) {
    uint16_t value = LoadNativeU16(src);
    if (invert) value = 65535u - value;
    StoreNativeU16(value, dst);
  } else if (sample_format == SAMPLEFORMAT_IEEEFP && bits == 32) {
    float value = LoadNativeFloat(src);
    if (invert) value = 1.0f - value;
    StoreNativeFloat(value, dst);
  } else {
    double value = LoadNativeDouble(src);
    if (invert) value = 1.0 - value;
    StoreNativeFloat(static_cast<float>(value), dst);
  }
}

Status SetColorAndMetadata(TIFF* tif, bool is_gray,
                           const ColorHints& color_hints,
                           PackedPixelFile* ppf) {
  uint32_t icc_size = 0;
  void* icc_data = nullptr;
  if (TIFFGetField(tif, TIFFTAG_ICCPROFILE, &icc_size, &icc_data) == 1 &&
      icc_size != 0 && icc_data != nullptr) {
    const uint8_t* icc = reinterpret_cast<const uint8_t*>(icc_data);
    ppf->icc.assign(icc, icc + icc_size);
    ppf->primary_color_representation = PackedPixelFile::kIccIsPrimary;
    JXL_RETURN_IF_ERROR(
        ApplyColorHints(color_hints, /*color_already_set=*/true, is_gray, ppf));
  } else {
    JXL_RETURN_IF_ERROR(ApplyColorHints(color_hints,
                                        /*color_already_set=*/false, is_gray,
                                        ppf));
  }
  return true;
}

Status ReadTIFFDirect(TIFF* tif, uint32_t width, uint32_t height,
                      const ColorHints& color_hints, PackedPixelFile* ppf,
                      const SizeConstraints* constraints) {
  uint16_t samples = 0;
  uint16_t bits = 0;
  uint16_t photometric = 0;
  uint16_t planar = PLANARCONFIG_CONTIG;
  uint16_t sample_format = SAMPLEFORMAT_UINT;
  TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
  TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
  TIFFGetFieldDefaulted(tif, TIFFTAG_PHOTOMETRIC, &photometric);
  TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);
  TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);

  if (planar != PLANARCONFIG_CONTIG) return false;
  const bool is_gray = photometric == PHOTOMETRIC_MINISBLACK ||
                       photometric == PHOTOMETRIC_MINISWHITE;
  const bool is_rgb = photometric == PHOTOMETRIC_RGB;
  const uint16_t color_channels = is_gray ? 1 : 3;
  if ((!is_gray && !is_rgb) || samples < color_channels) return false;
  if (!((sample_format == SAMPLEFORMAT_UINT && (bits == 8 || bits == 16)) ||
        (sample_format == SAMPLEFORMAT_IEEEFP && (bits == 32 || bits == 64)))) {
    return false;
  }

  JXL_RETURN_IF_ERROR(VerifyDimensions<uint32_t>(constraints, width, height));

  bool alpha_premultiplied = false;
  const bool has_alpha =
      HasAlpha(tif, samples, color_channels, &alpha_premultiplied);
  const uint32_t output_channels = color_channels + (has_alpha ? 1 : 0);
  JxlDataType data_type = JXL_TYPE_UINT8;
  if (sample_format == SAMPLEFORMAT_IEEEFP) {
    data_type = JXL_TYPE_FLOAT;
  } else if (bits > 8) {
    data_type = JXL_TYPE_UINT16;
  }
  const JxlPixelFormat format{
      /*num_channels=*/output_channels,
      /*data_type=*/data_type,
      /*endianness=*/JXL_NATIVE_ENDIAN,
      /*align=*/0,
  };

  ppf->info.xsize = width;
  ppf->info.ysize = height;
  ppf->info.bits_per_sample = sample_format == SAMPLEFORMAT_IEEEFP ? 32 : bits;
  ppf->info.exponent_bits_per_sample =
      sample_format == SAMPLEFORMAT_IEEEFP ? 8 : 0;
  ppf->info.orientation = JXL_ORIENT_IDENTITY;
  ppf->info.alpha_bits = has_alpha ? ppf->info.bits_per_sample : 0;
  ppf->info.alpha_exponent_bits =
      has_alpha ? ppf->info.exponent_bits_per_sample : 0;
  ppf->info.alpha_premultiplied = TO_JXL_BOOL(alpha_premultiplied);
  ppf->info.num_color_channels = color_channels;
  ppf->info.num_extra_channels = has_alpha ? 1 : 0;
  JXL_RETURN_IF_ERROR(SetColorAndMetadata(tif, is_gray, color_hints, ppf));

  ppf->frames.clear();
  JXL_ASSIGN_OR_RETURN(PackedFrame frame,
                       PackedFrame::Create(width, height, format));
  ppf->frames.emplace_back(std::move(frame));
  PackedImage& image = ppf->frames.back().color;

  const tmsize_t scanline_size = TIFFScanlineSize(tif);
  if (scanline_size <= 0) return JXL_FAILURE("Invalid TIFF scanline size");
  std::vector<uint8_t> scanline(static_cast<size_t>(scanline_size));
  const uint32_t input_sample_bytes = bits / 8;
  const uint32_t output_sample_bytes =
      PackedImage::BitsPerChannel(data_type) / jxl::kBitsPerByte;

  for (uint32_t y = 0; y < height; ++y) {
    if (TIFFReadScanline(tif, scanline.data(), y, 0) != 1) {
      return JXL_FAILURE("Failed to read TIFF scanline");
    }
    for (uint32_t x = 0; x < width; ++x) {
      const uint8_t* src =
          scanline.data() +
          (static_cast<size_t>(x) * samples) * input_sample_bytes;
      for (uint32_t c = 0; c < color_channels; ++c) {
        StoreSample(src + c * input_sample_bytes, bits, sample_format,
                    is_gray && photometric == PHOTOMETRIC_MINISWHITE,
                    image.pixels(y, x, c));
      }
      if (has_alpha) {
        const uint8_t* alpha = src + color_channels * input_sample_bytes;
        uint8_t* dst = image.pixels(y, x, color_channels);
        if (input_sample_bytes == output_sample_bytes) {
          memcpy(dst, alpha, output_sample_bytes);
        } else {
          StoreSample(alpha, bits, sample_format, false, dst);
        }
      }
    }
  }
  return true;
}

Status ReadTIFFRGBA(TIFF* tif, uint32_t width, uint32_t height,
                    const ColorHints& color_hints, PackedPixelFile* ppf,
                    const SizeConstraints* constraints) {
  uint16_t samples = 0;
  TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
  bool alpha_premultiplied = false;
  const bool has_alpha = HasAlpha(tif, samples, 3, &alpha_premultiplied);

  JXL_RETURN_IF_ERROR(VerifyDimensions<uint32_t>(constraints, width, height));
  if (height != 0 && width > std::numeric_limits<size_t>::max() / height) {
    return JXL_FAILURE("TIFF image is too large");
  }
  const size_t num_pixels = static_cast<size_t>(width) * height;
  std::vector<uint32_t> raster(num_pixels);
  if (TIFFReadRGBAImageOriented(tif, width, height, raster.data(),
                                ORIENTATION_TOPLEFT, 0) != 1) {
    return false;
  }

  const uint32_t output_channels = has_alpha ? 4 : 3;
  const JxlPixelFormat format{
      /*num_channels=*/output_channels,
      /*data_type=*/JXL_TYPE_UINT8,
      /*endianness=*/JXL_NATIVE_ENDIAN,
      /*align=*/0,
  };
  ppf->info.xsize = width;
  ppf->info.ysize = height;
  ppf->info.bits_per_sample = 8;
  ppf->info.exponent_bits_per_sample = 0;
  ppf->info.orientation = JXL_ORIENT_IDENTITY;
  ppf->info.alpha_bits = has_alpha ? 8 : 0;
  ppf->info.alpha_exponent_bits = 0;
  ppf->info.alpha_premultiplied = TO_JXL_BOOL(alpha_premultiplied);
  ppf->info.num_color_channels = 3;
  ppf->info.num_extra_channels = has_alpha ? 1 : 0;
  JXL_RETURN_IF_ERROR(
      SetColorAndMetadata(tif, /*is_gray=*/false, color_hints, ppf));

  ppf->frames.clear();
  JXL_ASSIGN_OR_RETURN(PackedFrame frame,
                       PackedFrame::Create(width, height, format));
  ppf->frames.emplace_back(std::move(frame));
  PackedImage& image = ppf->frames.back().color;
  for (uint32_t y = 0; y < height; ++y) {
    const uint32_t* row = raster.data() + static_cast<size_t>(y) * width;
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t pixel = row[x];
      image.pixels(y, x, 0)[0] = TIFFGetR(pixel);
      image.pixels(y, x, 1)[0] = TIFFGetG(pixel);
      image.pixels(y, x, 2)[0] = TIFFGetB(pixel);
      if (has_alpha) image.pixels(y, x, 3)[0] = TIFFGetA(pixel);
    }
  }
  return true;
}

}  // namespace
#endif  // JPEGXL_ENABLE_TIFF

bool CanDecodeTIFF() {
#if JPEGXL_ENABLE_TIFF
  return true;
#else
  return false;
#endif
}

Status DecodeImageTIFF(Span<const uint8_t> bytes, const ColorHints& color_hints,
                       PackedPixelFile* ppf,
                       const SizeConstraints* constraints) {
#if JPEGXL_ENABLE_TIFF
  if (!IsTIFF(bytes)) return false;
  if (bytes.size() > std::numeric_limits<toff_t>::max()) {
    return JXL_FAILURE("TIFF input is too large");
  }
  TiffMemoryReader reader = {bytes.data(), static_cast<toff_t>(bytes.size()),
                             0};
  TIFF* tif = TIFFClientOpen("MemoryTIFF", "r", &reader, ReadProc, WriteProc,
                             SeekProc, CloseProc, SizeProc, MapProc, UnmapProc);
  if (tif == nullptr) return JXL_FAILURE("Failed to open TIFF input");

  uint32_t width = 0;
  uint32_t height = 0;
  Status status = true;
  if (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) != 1 ||
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height) != 1) {
    status = JXL_FAILURE("Missing TIFF image dimensions");
  } else {
    status = ReadTIFFDirect(tif, width, height, color_hints, ppf, constraints);
    if (!status) {
      TIFFSetDirectory(tif, 0);
      status = ReadTIFFRGBA(tif, width, height, color_hints, ppf, constraints);
    }
  }
  TIFFClose(tif);
  JXL_RETURN_IF_ERROR(status);
  ppf->metadata.exif.assign(bytes.data(), bytes.data() + bytes.size());
  return true;
#else
  return false;
#endif
}

}  // namespace extras
}  // namespace jxl
