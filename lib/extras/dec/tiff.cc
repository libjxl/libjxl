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

uint16_t LoadTIFFU16(const Span<const uint8_t> bytes, size_t offset,
                     bool bigendian) {
  return bigendian
             ? static_cast<uint16_t>((bytes[offset] << 8) | bytes[offset + 1])
             : static_cast<uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
}

uint32_t LoadTIFFU32(const Span<const uint8_t> bytes, size_t offset,
                     bool bigendian) {
  if (bigendian) {
    return (static_cast<uint32_t>(bytes[offset]) << 24) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 8) |
           static_cast<uint32_t>(bytes[offset + 3]);
  }
  return static_cast<uint32_t>(bytes[offset]) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

void StoreTIFFU16(uint16_t value, bool bigendian, std::vector<uint8_t>* out) {
  if (bigendian) {
    out->push_back(static_cast<uint8_t>(value >> 8));
    out->push_back(static_cast<uint8_t>(value));
  } else {
    out->push_back(static_cast<uint8_t>(value));
    out->push_back(static_cast<uint8_t>(value >> 8));
  }
}

void StoreTIFFU32(uint32_t value, bool bigendian, std::vector<uint8_t>* out) {
  if (bigendian) {
    out->push_back(static_cast<uint8_t>(value >> 24));
    out->push_back(static_cast<uint8_t>(value >> 16));
    out->push_back(static_cast<uint8_t>(value >> 8));
    out->push_back(static_cast<uint8_t>(value));
  } else {
    out->push_back(static_cast<uint8_t>(value));
    out->push_back(static_cast<uint8_t>(value >> 8));
    out->push_back(static_cast<uint8_t>(value >> 16));
    out->push_back(static_cast<uint8_t>(value >> 24));
  }
}

void StoreTIFFU16At(uint16_t value, bool bigendian, size_t offset,
                    std::vector<uint8_t>* out) {
  if (bigendian) {
    (*out)[offset + 0] = static_cast<uint8_t>(value >> 8);
    (*out)[offset + 1] = static_cast<uint8_t>(value);
  } else {
    (*out)[offset + 0] = static_cast<uint8_t>(value);
    (*out)[offset + 1] = static_cast<uint8_t>(value >> 8);
  }
}

void StoreTIFFU32At(uint32_t value, bool bigendian, size_t offset,
                    std::vector<uint8_t>* out) {
  if (bigendian) {
    (*out)[offset + 0] = static_cast<uint8_t>(value >> 24);
    (*out)[offset + 1] = static_cast<uint8_t>(value >> 16);
    (*out)[offset + 2] = static_cast<uint8_t>(value >> 8);
    (*out)[offset + 3] = static_cast<uint8_t>(value);
  } else {
    (*out)[offset + 0] = static_cast<uint8_t>(value);
    (*out)[offset + 1] = static_cast<uint8_t>(value >> 8);
    (*out)[offset + 2] = static_cast<uint8_t>(value >> 16);
    (*out)[offset + 3] = static_cast<uint8_t>(value >> 24);
  }
}

size_t TIFFTypeSize(uint16_t type) {
  switch (type) {
    case 1:  // BYTE
    case 2:  // ASCII
    case 6:  // SBYTE
    case 7:  // UNDEFINED
      return 1;
    case 3:  // SHORT
    case 8:  // SSHORT
      return 2;
    case 4:   // LONG
    case 9:   // SLONG
    case 11:  // FLOAT
    case 13:  // IFD
      return 4;
    case 5:   // RATIONAL
    case 10:  // SRATIONAL
    case 12:  // DOUBLE
      return 8;
    default:
      return 0;
  }
}

bool IsTIFFImageDataTag(uint16_t tag) {
  (void)tag;
  return false;
}

bool IsTIFFIFDPointerTag(uint16_t tag) {
  return tag == 330 || tag == 34665 || tag == 34853 || tag == 40965;
}

struct CopiedIFD {
  uint32_t old_offset;
  uint32_t new_offset;
};

StatusOr<uint32_t> CopyTIFFMetadataIFD(const Span<const uint8_t> bytes,
                                       bool bigendian, uint32_t old_offset,
                                       std::vector<uint8_t>* out,
                                       std::vector<CopiedIFD>* copied) {
  for (const CopiedIFD& ifd : *copied) {
    if (ifd.old_offset == old_offset) return uint32_t{ifd.new_offset};
  }
  if (old_offset > bytes.size() || bytes.size() - old_offset < 2) {
    return JXL_FAILURE("Invalid TIFF metadata IFD offset");
  }
  const uint16_t entry_count = LoadTIFFU16(bytes, old_offset, bigendian);
  const size_t entries_offset = static_cast<size_t>(old_offset) + 2;
  if (entry_count > (bytes.size() - entries_offset) / 12 ||
      bytes.size() - entries_offset - entry_count * 12 < 4) {
    return JXL_FAILURE("Invalid TIFF metadata IFD");
  }

  struct Entry {
    uint16_t tag;
    uint16_t type;
    uint32_t count;
    uint32_t value;
    size_t offset;
    size_t size;
  };
  std::vector<Entry> entries;
  entries.reserve(entry_count);
  for (uint16_t i = 0; i < entry_count; ++i) {
    const size_t entry_offset = entries_offset + i * 12;
    const uint16_t tag = LoadTIFFU16(bytes, entry_offset, bigendian);
    const uint16_t type = LoadTIFFU16(bytes, entry_offset + 2, bigendian);
    const uint32_t count = LoadTIFFU32(bytes, entry_offset + 4, bigendian);
    const uint32_t value = LoadTIFFU32(bytes, entry_offset + 8, bigendian);
    const size_t type_size = TIFFTypeSize(type);
    if (type_size == 0 ||
        count > std::numeric_limits<size_t>::max() / type_size) {
      return JXL_FAILURE("Invalid TIFF metadata field type");
    }
    const size_t size = static_cast<size_t>(count) * type_size;
    if (IsTIFFImageDataTag(tag)) continue;
    if (size > 4 && (value > bytes.size() || bytes.size() - value < size)) {
      return JXL_FAILURE("Invalid TIFF metadata field offset");
    }
    entries.push_back({tag, type, count, value, entry_offset, size});
  }

  const uint32_t new_offset = static_cast<uint32_t>(out->size());
  copied->push_back({old_offset, new_offset});
  StoreTIFFU16(static_cast<uint16_t>(entries.size()), bigendian, out);
  const size_t new_entries_offset = out->size();
  out->resize(out->size() + entries.size() * 12 + 4);

  for (size_t i = 0; i < entries.size(); ++i) {
    const Entry& entry = entries[i];
    const size_t dst = new_entries_offset + i * 12;
    StoreTIFFU16At(entry.tag, bigendian, dst, out);
    StoreTIFFU16At(entry.type, bigendian, dst + 2, out);
    StoreTIFFU32At(entry.count, bigendian, dst + 4, out);
    if (IsTIFFIFDPointerTag(entry.tag)) {
      if (entry.size <= 4) {
        JXL_ASSIGN_OR_RETURN(
            uint32_t child_offset,
            CopyTIFFMetadataIFD(bytes, bigendian, entry.value, out, copied));
        StoreTIFFU32At(child_offset, bigendian, dst + 8, out);
      } else {
        const uint32_t data_offset = static_cast<uint32_t>(out->size());
        for (uint32_t n = 0; n < entry.count; ++n) {
          const uint32_t child_old = LoadTIFFU32(
              bytes, static_cast<size_t>(entry.value) + n * 4, bigendian);
          JXL_ASSIGN_OR_RETURN(
              uint32_t child_offset,
              CopyTIFFMetadataIFD(bytes, bigendian, child_old, out, copied));
          StoreTIFFU32(child_offset, bigendian, out);
        }
        StoreTIFFU32At(data_offset, bigendian, dst + 8, out);
      }
    } else if (entry.size <= 4) {
      memcpy(out->data() + dst + 8, bytes.data() + entry.offset + 8, 4);
    } else {
      const uint32_t data_offset = static_cast<uint32_t>(out->size());
      out->insert(out->end(), bytes.data() + entry.value,
                  bytes.data() + entry.value + entry.size);
      if (out->size() & 1) out->push_back(0);
      StoreTIFFU32At(data_offset, bigendian, dst + 8, out);
    }
  }

  const size_t next_offset = entries_offset + entry_count * 12;
  const uint32_t next_old = LoadTIFFU32(bytes, next_offset, bigendian);
  uint32_t next_new = 0;
  if (next_old != 0) {
    JXL_ASSIGN_OR_RETURN(
        next_new, CopyTIFFMetadataIFD(bytes, bigendian, next_old, out, copied));
  }
  StoreTIFFU32At(next_new, bigendian, new_entries_offset + entries.size() * 12,
                 out);
  return uint32_t{new_offset};
}

Status ExtractTIFFMetadata(const Span<const uint8_t> bytes,
                           std::vector<uint8_t>* metadata) {
  const bool bigendian = bytes[0] == 'M';
  const uint32_t first_ifd = LoadTIFFU32(bytes, 4, bigendian);
  metadata->clear();
  metadata->reserve(bytes.size() / 1024);
  metadata->push_back(bytes[0]);
  metadata->push_back(bytes[1]);
  StoreTIFFU16(42, bigendian, metadata);
  StoreTIFFU32(8, bigendian, metadata);
  std::vector<CopiedIFD> copied;
  JXL_ASSIGN_OR_RETURN(
      uint32_t copied_first,
      CopyTIFFMetadataIFD(bytes, bigendian, first_ifd, metadata, &copied));
  if (copied_first != 8) return JXL_FAILURE("Invalid TIFF metadata layout");
  return true;
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

Status ValidateOrientation(uint16_t orientation) {
  if (orientation < ORIENTATION_TOPLEFT || orientation > ORIENTATION_LEFTBOT) {
    return JXL_FAILURE("Invalid TIFF orientation");
  }
  return true;
}

bool OrientationSwapsDimensions(uint16_t orientation) {
  return orientation >= ORIENTATION_LEFTTOP;
}

uint32_t OrientedWidth(uint32_t width, uint32_t height, uint16_t orientation) {
  return OrientationSwapsDimensions(orientation) ? height : width;
}

uint32_t OrientedHeight(uint32_t width, uint32_t height, uint16_t orientation) {
  return OrientationSwapsDimensions(orientation) ? width : height;
}

void OrientedPixel(uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                   uint16_t orientation, uint32_t* out_x, uint32_t* out_y) {
  switch (orientation) {
    case ORIENTATION_TOPRIGHT:
      *out_x = width - 1 - x;
      *out_y = y;
      break;
    case ORIENTATION_BOTRIGHT:
      *out_x = width - 1 - x;
      *out_y = height - 1 - y;
      break;
    case ORIENTATION_BOTLEFT:
      *out_x = x;
      *out_y = height - 1 - y;
      break;
    case ORIENTATION_LEFTTOP:
      *out_x = y;
      *out_y = x;
      break;
    case ORIENTATION_RIGHTTOP:
      *out_x = height - 1 - y;
      *out_y = x;
      break;
    case ORIENTATION_RIGHTBOT:
      *out_x = height - 1 - y;
      *out_y = width - 1 - x;
      break;
    case ORIENTATION_LEFTBOT:
      *out_x = y;
      *out_y = width - 1 - x;
      break;
    case ORIENTATION_TOPLEFT:
    default:
      *out_x = x;
      *out_y = y;
      break;
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

  uint16_t orientation = ORIENTATION_TOPLEFT;
  TIFFGetFieldDefaulted(tif, TIFFTAG_ORIENTATION, &orientation);
  JXL_RETURN_IF_ERROR(ValidateOrientation(orientation));
  const uint32_t oriented_width = OrientedWidth(width, height, orientation);
  const uint32_t oriented_height = OrientedHeight(width, height, orientation);

  JXL_RETURN_IF_ERROR(
      VerifyDimensions<uint32_t>(constraints, oriented_width, oriented_height));

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

  ppf->info.xsize = oriented_width;
  ppf->info.ysize = oriented_height;
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
  JXL_ASSIGN_OR_RETURN(
      PackedFrame frame,
      PackedFrame::Create(oriented_width, oriented_height, format));
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
      uint32_t out_x;
      uint32_t out_y;
      OrientedPixel(x, y, width, height, orientation, &out_x, &out_y);
      for (uint32_t c = 0; c < color_channels; ++c) {
        StoreSample(src + c * input_sample_bytes, bits, sample_format,
                    is_gray && photometric == PHOTOMETRIC_MINISWHITE,
                    image.pixels(out_y, out_x, c));
      }
      if (has_alpha) {
        const uint8_t* alpha = src + color_channels * input_sample_bytes;
        uint8_t* dst = image.pixels(out_y, out_x, color_channels);
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
  uint16_t orientation = ORIENTATION_TOPLEFT;
  TIFFGetFieldDefaulted(tif, TIFFTAG_ORIENTATION, &orientation);
  JXL_RETURN_IF_ERROR(ValidateOrientation(orientation));
  if (OrientationSwapsDimensions(orientation)) {
    return JXL_FAILURE(
        "TIFF orientations requiring rotation are not supported by RGBA "
        "fallback");
  }

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
  ppf->info.alpha_premultiplied = TO_JXL_BOOL(has_alpha);
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

Status DecodeTIFFDirectory(TIFF* tif, tdir_t dir, const ColorHints& color_hints,
                           PackedPixelFile* page_ppf,
                           const SizeConstraints* constraints) {
  if (TIFFSetDirectory(tif, dir) != 1) {
    return JXL_FAILURE("Failed to select TIFF directory");
  }
  uint32_t width = 0;
  uint32_t height = 0;
  if (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) != 1 ||
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height) != 1) {
    return JXL_FAILURE("Missing TIFF image dimensions");
  }
  Status status =
      ReadTIFFDirect(tif, width, height, color_hints, page_ppf, constraints);
  if (!status) {
    if (TIFFSetDirectory(tif, dir) != 1) {
      return JXL_FAILURE("Failed to reselect TIFF directory");
    }
    status =
        ReadTIFFRGBA(tif, width, height, color_hints, page_ppf, constraints);
  }
  return status;
}

Status AppendTIFFPage(PackedPixelFile page_ppf, PackedPixelFile* ppf) {
  if (page_ppf.frames.empty()) return JXL_FAILURE("No TIFF frame decoded");
  if (ppf->frames.empty()) {
    *ppf = std::move(page_ppf);
    return true;
  }

  if (ppf->info.xsize != page_ppf.info.xsize ||
      ppf->info.ysize != page_ppf.info.ysize ||
      ppf->info.bits_per_sample != page_ppf.info.bits_per_sample ||
      ppf->info.exponent_bits_per_sample !=
          page_ppf.info.exponent_bits_per_sample ||
      ppf->info.alpha_bits != page_ppf.info.alpha_bits ||
      ppf->info.alpha_exponent_bits != page_ppf.info.alpha_exponent_bits ||
      ppf->info.alpha_premultiplied != page_ppf.info.alpha_premultiplied ||
      ppf->info.num_color_channels != page_ppf.info.num_color_channels) {
    return JXL_FAILURE(
        "TIFF directories with different image formats are not "
        "supported");
  }
  if (ppf->frames.size() == 1 && page_ppf.frames.size() == 1 &&
      ppf->info.num_color_channels == 1 &&
      page_ppf.info.num_color_channels == 1 && ppf->info.alpha_bits == 0 &&
      page_ppf.info.alpha_bits == 0 &&
      ppf->frames[0].color.format.num_channels == 1 &&
      page_ppf.frames[0].color.format.num_channels == 1) {
    PackedExtraChannel extra_channel;
    extra_channel.ec_info = {};
    extra_channel.ec_info.type = JXL_CHANNEL_OPTIONAL;
    extra_channel.ec_info.bits_per_sample = ppf->info.bits_per_sample;
    extra_channel.ec_info.exponent_bits_per_sample =
        ppf->info.exponent_bits_per_sample;
    extra_channel.index = ppf->extra_channels_info.size();
    ppf->extra_channels_info.emplace_back(std::move(extra_channel));
    ppf->frames[0].extra_channels.emplace_back(
        std::move(page_ppf.frames[0].color));
    ppf->info.num_extra_channels = ppf->extra_channels_info.size();
    return true;
  }

  for (PackedFrame& frame : page_ppf.frames) {
    ppf->frames.emplace_back(std::move(frame));
  }
  return true;
}

void SetTIFFPageFrameMetadata(PackedPixelFile* ppf) {
  if (ppf->frames.size() <= 1) return;
  ppf->info.have_animation = JXL_TRUE;
  ppf->info.animation.tps_numerator = 1;
  ppf->info.animation.tps_denominator = 1;
  ppf->info.animation.num_loops = 0;
  for (size_t i = 0; i < ppf->frames.size(); ++i) {
    PackedFrame& frame = ppf->frames[i];
    frame.frame_info.duration = 1;
    frame.frame_info.layer_info.xsize = ppf->info.xsize;
    frame.frame_info.layer_info.ysize = ppf->info.ysize;
    frame.frame_info.layer_info.blend_info.blendmode = JXL_BLEND_REPLACE;
    frame.frame_info.layer_info.blend_info.source = i == 0 ? 0 : 1;
    frame.frame_info.layer_info.save_as_reference = 1;
  }
  ppf->frames.back().frame_info.is_last = JXL_TRUE;
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

  const tdir_t directory_count = TIFFNumberOfDirectories(tif);
  Status status = true;
  for (tdir_t dir = 0; dir < directory_count; ++dir) {
    PackedPixelFile page_ppf;
    status = DecodeTIFFDirectory(tif, dir, color_hints, &page_ppf, constraints);
    if (!status) break;
    status = AppendTIFFPage(std::move(page_ppf), ppf);
    if (!status) break;
  }
  if (status && ppf->frames.empty()) {
    status = JXL_FAILURE("No TIFF directories decoded");
  }
  if (status) {
    SetTIFFPageFrameMetadata(ppf);
    status = ExtractTIFFMetadata(bytes, &ppf->metadata.exif);
  }
  TIFFClose(tif);
  JXL_RETURN_IF_ERROR(status);
  return true;
#else
  return false;
#endif
}

}  // namespace extras
}  // namespace jxl
