// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdlib.h>
#include <string.h>

#include "lib/extras/dec/pnm.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {

namespace {

bool IsDigit(const uint8_t c) { return '0' <= c && c <= '9'; }
Status ParseUnsigned(size_t* number, size_t& pos,
                     const Span<const uint8_t> bytes) {
  if (pos == bytes.size()) return JXL_FAILURE("NPY: reached end before number");
  if (!IsDigit(bytes[pos])) return JXL_FAILURE("NPY: expected unsigned number");

  *number = 0;
  while (pos < bytes.size() && IsDigit(bytes[pos])) {
    *number *= 10;
    *number += bytes[pos] - '0';
    ++pos;
  }

  return true;
}
Status ParseConstantString(const char* expected, size_t& pos,
                           const Span<const uint8_t> bytes) {
  if (memcmp(bytes.data() + pos, expected, strlen(expected)) != 0) {
    return JXL_FAILURE("Expected string %s not found", expected);
  }
  pos += strlen(expected);
  return true;
}
}  // namespace

Status DecodeImageNPY(const Span<const uint8_t> bytes,
                      const ColorHints& color_hints,
                      const SizeConstraints& constraints,
                      PackedPixelFile* ppf) {
  constexpr const char kMagic[] = "\x93NUMPY\x01\00";
  if (memcmp(bytes.data(), kMagic, sizeof(kMagic) - 1) != 0) {
    return JXL_FAILURE("Not a NPY array");
  }
  size_t pos = sizeof(kMagic) - 1;
  if (bytes.size() < pos + 2) {
    return JXL_FAILURE("Not a NPY array");
  }
  size_t hlen = bytes[pos] | (uint32_t(bytes[pos + 1]) << 8);
  if (bytes.size() < pos + hlen + 2) {
    return JXL_FAILURE("Not a NPY array");
  }
  pos += 2;
  size_t maxpos = pos + hlen;

  JXL_RETURN_IF_ERROR(ParseConstantString(
      "{'descr': '<f4', 'fortran_order': False, 'shape': (", pos, bytes));
  size_t frames, width, height, channels;
  JXL_RETURN_IF_ERROR(ParseUnsigned(&frames, pos, bytes));
  JXL_RETURN_IF_ERROR(ParseConstantString(", ", pos, bytes));
  JXL_RETURN_IF_ERROR(ParseUnsigned(&height, pos, bytes));
  JXL_RETURN_IF_ERROR(ParseConstantString(", ", pos, bytes));
  JXL_RETURN_IF_ERROR(ParseUnsigned(&width, pos, bytes));
  JXL_RETURN_IF_ERROR(ParseConstantString(", ", pos, bytes));
  JXL_RETURN_IF_ERROR(ParseUnsigned(&channels, pos, bytes));
  if (pos > maxpos) {
    return JXL_FAILURE("Wrong header length");
  }
  pos = maxpos;
  size_t data_bytes = 4 * frames * height * width * channels;
  if (data_bytes + pos > bytes.size()) {
    return JXL_FAILURE("File too short");
  }
  if (channels == 0 || frames == 0 || height == 0 || width == 0) {
    return JXL_FAILURE("Invalid dimensions");
  }

  ppf->info.xsize = width;
  ppf->info.ysize = height;
  ppf->info.bits_per_sample = 8;
  ppf->info.exponent_bits_per_sample = 0;
  ppf->color_encoding.color_space = JXL_COLOR_SPACE_GRAY;
  ppf->color_encoding.primaries = JXL_PRIMARIES_SRGB;
  ppf->color_encoding.white_point = JXL_WHITE_POINT_D65;
  ppf->color_encoding.transfer_function = JXL_TRANSFER_FUNCTION_SRGB;
  JxlExtraChannelInfo info = {};
  info.bits_per_sample = 8;
  info.exponent_bits_per_sample = 0;
  info.type = JXL_CHANNEL_OPTIONAL;
  PackedPixelFile::PackedExtraChannel ec(info, "");
  ppf->extra_channels_info.resize(channels - 1, ec);

  auto next_float = [&]() {
    float f;
    memcpy(&f, bytes.data() + pos, 4);
    pos += 4;
    return f / 255.0;
  };

  ppf->frames.clear();
  for (size_t fid = 0; fid < frames; fid++) {
    JxlPixelFormat format = {};
    format.num_channels = 1;
    format.data_type = JXL_TYPE_FLOAT;
    PackedFrame frame(width, height, format);
    frame.extra_channels.clear();
    for (size_t c = 1; c < channels; c++) {
      frame.extra_channels.push_back(PackedImage(width, height, format));
    }
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        for (size_t c = 0; c < channels; c++) {
          float f = next_float();
          PackedImage& img = c == 0 ? frame.color : frame.extra_channels[c - 1];
          memcpy(reinterpret_cast<char*>(img.pixels()) + img.stride * y + x * 4,
                 &f, 4);
        }
      }
    }
    ppf->frames.push_back(std::move(frame));
  }

  return true;

#if 0
  Parser parser(bytes);
  HeaderPNM header = {};
  const uint8_t* pos = nullptr;
  if (!parser.ParseHeader(&header, &pos)) return false;
  JXL_RETURN_IF_ERROR(
      VerifyDimensions(&constraints, header.xsize, header.ysize));

  if (header.bits_per_sample == 0 || header.bits_per_sample > 32) {
    return JXL_FAILURE("PNM: bits_per_sample invalid");
  }

  JXL_RETURN_IF_ERROR(ApplyColorHints(color_hints, /*color_already_set=*/false,
                                      header.is_gray, ppf));

  ppf->info.xsize = header.xsize;
  ppf->info.ysize = header.ysize;
  if (header.floating_point) {
    ppf->info.bits_per_sample = 32;
    ppf->info.exponent_bits_per_sample = 8;
  } else {
    ppf->info.bits_per_sample = header.bits_per_sample;
    ppf->info.exponent_bits_per_sample = 0;
  }

  ppf->info.orientation = JXL_ORIENT_IDENTITY;

  // No alpha in PNM and PFM
  ppf->info.alpha_bits = (header.has_alpha ? ppf->info.bits_per_sample : 0);
  ppf->info.alpha_exponent_bits = 0;
  ppf->info.num_color_channels = (header.is_gray ? 1 : 3);
  ppf->info.num_extra_channels = (header.has_alpha ? 1 : 0);

  JxlDataType data_type;
  if (header.floating_point) {
    // There's no float16 pnm version.
    data_type = JXL_TYPE_FLOAT;
  } else {
    if (header.bits_per_sample > 16) {
      data_type = JXL_TYPE_UINT32;
    } else if (header.bits_per_sample > 8) {
      data_type = JXL_TYPE_UINT16;
    } else if (header.is_bit) {
      data_type = JXL_TYPE_BOOLEAN;
    } else {
      data_type = JXL_TYPE_UINT8;
    }
  }

  const JxlPixelFormat format{
      /*num_channels=*/ppf->info.num_color_channels +
          ppf->info.num_extra_channels,
      /*data_type=*/data_type,
      /*endianness=*/header.big_endian ? JXL_BIG_ENDIAN : JXL_LITTLE_ENDIAN,
      /*align=*/0,
  };
  ppf->frames.clear();
  ppf->frames.emplace_back(header.xsize, header.ysize, format);
  auto* frame = &ppf->frames.back();

  frame->color.flipped_y = header.bits_per_sample == 32;  // PFMs are flipped
  size_t pnm_remaining_size = bytes.data() + bytes.size() - pos;
  if (pnm_remaining_size < frame->color.pixels_size) {
    return JXL_FAILURE("PNM file too small");
  }
  memcpy(frame->color.pixels(), pos, frame->color.pixels_size);
  return true;
#endif
}

}  // namespace extras
}  // namespace jxl
