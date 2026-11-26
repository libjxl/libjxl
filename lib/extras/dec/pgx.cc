// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/pgx.h"

#include <jxl/codestream_header.h>
#include <jxl/types.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "lib/extras/dec/byte_reader.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/size_constraints.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {
namespace {

struct HeaderPGX {
  // NOTE: PGX is always grayscale
  size_t xsize;
  size_t ysize;
  size_t bits_per_sample;
  bool big_endian;
  bool is_signed;
};

class Parser {
 public:
  explicit Parser(const Span<const uint8_t> bytes) : reader_(bytes) {}

  // Sets "pos" to the first non-header byte/pixel on success.
  Status ParseHeader(HeaderPGX* header, const uint8_t** pos) {
    if (reader_.remaining() < 2) return false;
    uint8_t p;
    uint8_t g;
    JXL_RETURN_IF_ERROR(reader_.ReadU8(&p));
    JXL_RETURN_IF_ERROR(reader_.ReadU8(&g));
    if (p != 'P' || g != 'G') return false;
    return ParseHeaderPGX(header, pos);
  }

  // Exposed for testing
  Status ParseUnsigned(size_t* number) {
    uint8_t c;
    if (reader_.AtEnd()) return JXL_FAILURE("PGX: reached end before number");
    JXL_RETURN_IF_ERROR(reader_.Peek(&c));
    if (!IsDigit(c)) return JXL_FAILURE("PGX: expected unsigned number");

    *number = 0;
    while (!reader_.AtEnd()) {
      JXL_RETURN_IF_ERROR(reader_.Peek(&c));
      if (!IsDigit(c)) break;
      const size_t digit = static_cast<size_t>(c - '0');
      if (*number > (std::numeric_limits<size_t>::max() - digit) / 10) {
        return JXL_FAILURE("PGX: number too large");
      }
      *number = *number * 10 + digit;
      JXL_RETURN_IF_ERROR(reader_.Skip(1));
    }

    return true;
  }

 private:
  static bool IsDigit(const uint8_t c) { return '0' <= c && c <= '9'; }
  static bool IsLineBreak(const uint8_t c) { return c == '\r' || c == '\n'; }
  static bool IsWhitespace(const uint8_t c) {
    return IsLineBreak(c) || c == '\t' || c == ' ';
  }

  Status SkipSpace() {
    uint8_t c;
    if (reader_.AtEnd()) return JXL_FAILURE("PGX: reached end before space");
    JXL_RETURN_IF_ERROR(reader_.Peek(&c));
    if (c != ' ') return JXL_FAILURE("PGX: expected space");
    return reader_.Skip(1);
  }

  Status SkipLineBreak() {
    uint8_t c;
    if (reader_.AtEnd()) {
      return JXL_FAILURE("PGX: reached end before line break");
    }
    JXL_RETURN_IF_ERROR(reader_.Peek(&c));
    // Line break can be either "\n" (0a) or "\r\n" (0d 0a).
    if (c == '\n') return reader_.Skip(1);
    if (c == '\r') {
      Span<const uint8_t> line_break;
      if (reader_.remaining() < 2) {
        return JXL_FAILURE("PGX: expected line break");
      }
      JXL_RETURN_IF_ERROR(reader_.Read(2, &line_break));
      if (line_break[1] != '\n') {
        return JXL_FAILURE("PGX: expected line break");
      }
      return true;
    }
    return JXL_FAILURE("PGX: expected line break");
  }

  Status SkipSingleWhitespace() {
    uint8_t c;
    if (reader_.AtEnd()) {
      return JXL_FAILURE("PGX: reached end before whitespace");
    }
    JXL_RETURN_IF_ERROR(reader_.Peek(&c));
    if (!IsWhitespace(c)) return JXL_FAILURE("PGX: expected whitespace");
    return reader_.Skip(1);
  }

  Status ParseHeaderPGX(HeaderPGX* header, const uint8_t** pos) {
    JXL_RETURN_IF_ERROR(SkipSpace());
    if (reader_.remaining() < 2) return JXL_FAILURE("PGX: header too small");
    uint8_t endianness[2];
    JXL_RETURN_IF_ERROR(reader_.ReadU8(&endianness[0]));
    JXL_RETURN_IF_ERROR(reader_.ReadU8(&endianness[1]));
    if (endianness[0] == 'M' && endianness[1] == 'L') {
      header->big_endian = true;
    } else if (endianness[0] == 'L' && endianness[1] == 'M') {
      header->big_endian = false;
    } else {
      return JXL_FAILURE("PGX: invalid endianness");
    }
    JXL_RETURN_IF_ERROR(SkipSpace());
    if (reader_.AtEnd()) return JXL_FAILURE("PGX: header too small");
    uint8_t signedness;
    JXL_RETURN_IF_ERROR(reader_.ReadU8(&signedness));
    if (signedness == '+') {
      header->is_signed = false;
    } else if (signedness == '-') {
      header->is_signed = true;
    } else {
      return JXL_FAILURE("PGX: invalid signedness");
    }
    // Skip optional space
    uint8_t c;
    if (!reader_.AtEnd()) {
      JXL_RETURN_IF_ERROR(reader_.Peek(&c));
      if (c == ' ') {
        JXL_RETURN_IF_ERROR(reader_.Skip(1));
      }
    }
    JXL_RETURN_IF_ERROR(ParseUnsigned(&header->bits_per_sample));
    JXL_RETURN_IF_ERROR(SkipSingleWhitespace());
    JXL_RETURN_IF_ERROR(ParseUnsigned(&header->xsize));
    JXL_RETURN_IF_ERROR(SkipSingleWhitespace());
    JXL_RETURN_IF_ERROR(ParseUnsigned(&header->ysize));
    // 0xa, or 0xd 0xa.
    JXL_RETURN_IF_ERROR(SkipLineBreak());

    // TODO(jon): could do up to 24-bit by converting the values to
    // JXL_TYPE_FLOAT.
    if (header->bits_per_sample > 16) {
      return JXL_FAILURE("PGX: >16 bits not yet supported");
    }
    // TODO(lode): support signed integers. This may require changing the way
    // external_image works.
    if (header->is_signed) {
      return JXL_FAILURE("PGX: signed not yet supported");
    }

    const size_t bytes_per_pixel = header->bits_per_sample <= 8 ? 1 : 2;
    size_t numpixels;
    size_t bytes_needed;
    if (!SafeMul(header->xsize, header->ysize, numpixels) ||
        !SafeMul(numpixels, bytes_per_pixel, bytes_needed)) {
      return JXL_FAILURE("PGX: image dimensions are too large");
    }
    if (reader_.remaining() < bytes_needed) {
      return JXL_FAILURE("PGX: data too small");
    }

    *pos = reader_.current();
    return true;
  }

  ByteReader reader_;
};

}  // namespace

Status DecodeImagePGX(const Span<const uint8_t> bytes,
                      const ColorHints& color_hints, PackedPixelFile* ppf,
                      const SizeConstraints* constraints) {
  Parser parser(bytes);
  HeaderPGX header = {};
  const uint8_t* pos = nullptr;
  if (!parser.ParseHeader(&header, &pos)) return false;
  JXL_RETURN_IF_ERROR(
      VerifyDimensions(constraints, header.xsize, header.ysize));
  if (header.bits_per_sample == 0 || header.bits_per_sample > 32) {
    return JXL_FAILURE("PGX: bits_per_sample invalid");
  }

  JXL_RETURN_IF_ERROR(ApplyColorHints(color_hints, /*color_already_set=*/false,
                                      /*is_gray=*/true, ppf));
  ppf->info.xsize = header.xsize;
  ppf->info.ysize = header.ysize;
  // Original data is uint, so exponent_bits_per_sample = 0.
  ppf->info.bits_per_sample = header.bits_per_sample;
  ppf->info.exponent_bits_per_sample = 0;
  ppf->info.uses_original_profile = JXL_TRUE;

  // No alpha in PGX
  ppf->info.alpha_bits = 0;
  ppf->info.alpha_exponent_bits = 0;
  ppf->info.num_color_channels = 1;  // Always grayscale
  ppf->info.orientation = JXL_ORIENT_IDENTITY;

  JxlDataType data_type;
  if (header.bits_per_sample > 8) {
    data_type = JXL_TYPE_UINT16;
  } else {
    data_type = JXL_TYPE_UINT8;
  }

  const JxlPixelFormat format{
      /*num_channels=*/1,
      /*data_type=*/data_type,
      /*endianness=*/header.big_endian ? JXL_BIG_ENDIAN : JXL_LITTLE_ENDIAN,
      /*align=*/0,
  };
  ppf->frames.clear();
  // Allocates the frame buffer.
  {
    JXL_ASSIGN_OR_RETURN(
        PackedFrame frame,
        PackedFrame::Create(header.xsize, header.ysize, format));
    ppf->frames.emplace_back(std::move(frame));
  }
  const auto& frame = ppf->frames.back();
  size_t pgx_remaining_size = bytes.data() + bytes.size() - pos;
  if (pgx_remaining_size < frame.color.pixels_size) {
    return JXL_FAILURE("PGX file too small");
  }
  memcpy(frame.color.pixels(), pos, frame.color.pixels_size);
  return true;
}

}  // namespace extras
}  // namespace jxl
