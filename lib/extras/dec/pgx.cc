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
  explicit Parser(const Span<const uint8_t> bytes) : data_(bytes), pos_(0) {}

  // Sets "rest" to the bytes following the header on success.
  Status ParseHeader(HeaderPGX* header, Bytes* rest) {
    if (Remaining() < 2) return false;
    if (Peek(0) != 'P' || Peek(1) != 'G') return false;
    pos_ += 2;
    return ParseHeaderPGX(header, rest);
  }

  // Exposed for testing
  Status ParseUnsigned(size_t* number) {
    if (AtEnd()) return JXL_FAILURE("PGX: reached end before number");
    if (!IsDigit(Peek())) return JXL_FAILURE("PGX: expected unsigned number");

    *number = 0;
    while (!AtEnd() && IsDigit(Peek())) {
      const size_t digit = static_cast<size_t>(Peek() - '0');
      if (*number > (std::numeric_limits<size_t>::max() - digit) / 10) {
        return JXL_FAILURE("PGX: number too large");
      }
      *number = *number * 10 + digit;
      ++pos_;
    }

    return true;
  }

 private:
  static bool IsDigit(const uint8_t c) { return '0' <= c && c <= '9'; }
  static bool IsLineBreak(const uint8_t c) { return c == '\r' || c == '\n'; }
  static bool IsWhitespace(const uint8_t c) {
    return IsLineBreak(c) || c == '\t' || c == ' ';
  }

  // Bytes not yet consumed. These are size_t comparisons: the previous code
  // compared pointers, and an expression such as `pos_ + 2 > end_` forms a
  // pointer more than one past the end of the array, which is undefined.
  size_t Remaining() const { return data_.size() - pos_; }
  bool AtEnd() const { return pos_ == data_.size(); }
  // Callers must have established Remaining() > i.
  uint8_t Peek(size_t i = 0) const { return data_[pos_ + i]; }

  Status SkipSpace() {
    if (AtEnd()) return JXL_FAILURE("PGX: reached end before space");
    if (Peek() != ' ') return JXL_FAILURE("PGX: expected space");
    ++pos_;
    return true;
  }

  Status SkipLineBreak() {
    if (AtEnd()) return JXL_FAILURE("PGX: reached end before line break");
    // Line break can be either "\n" (0a) or "\r\n" (0d 0a).
    if (Peek() == '\n') {
      pos_++;
      return true;
    } else if (Peek() == '\r' && Remaining() >= 2 && Peek(1) == '\n') {
      pos_ += 2;
      return true;
    }
    return JXL_FAILURE("PGX: expected line break");
  }

  Status SkipSingleWhitespace() {
    if (AtEnd()) return JXL_FAILURE("PGX: reached end before whitespace");
    if (!IsWhitespace(Peek())) return JXL_FAILURE("PGX: expected whitespace");
    ++pos_;
    return true;
  }

  Status ParseHeaderPGX(HeaderPGX* header, Bytes* rest) {
    JXL_RETURN_IF_ERROR(SkipSpace());
    if (Remaining() < 2) return JXL_FAILURE("PGX: header too small");
    if (Peek(0) == 'M' && Peek(1) == 'L') {
      header->big_endian = true;
    } else if (Peek(0) == 'L' && Peek(1) == 'M') {
      header->big_endian = false;
    } else {
      return JXL_FAILURE("PGX: invalid endianness");
    }
    pos_ += 2;
    JXL_RETURN_IF_ERROR(SkipSpace());
    if (AtEnd()) return JXL_FAILURE("PGX: header too small");
    if (Peek() == '+') {
      header->is_signed = false;
    } else if (Peek() == '-') {
      header->is_signed = true;
    } else {
      return JXL_FAILURE("PGX: invalid signedness");
    }
    pos_++;
    // Skip optional space
    if (!AtEnd() && Peek() == ' ') pos_++;
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
    if (Remaining() < bytes_needed) {
      return JXL_FAILURE("PGX: data too small");
    }

    JXL_ASSIGN_OR_RETURN(*rest, data_.subspan(pos_));
    return true;
  }

  Bytes data_;
  size_t pos_;
};

}  // namespace

Status DecodeImagePGX(const Span<const uint8_t> bytes,
                      const ColorHints& color_hints, PackedPixelFile* ppf,
                      const SizeConstraints* constraints) {
  Parser parser(bytes);
  HeaderPGX header = {};
  Bytes rest;
  if (!parser.ParseHeader(&header, &rest)) return false;
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
  if (rest.size() < frame.color.pixels_size) {
    return JXL_FAILURE("PGX file too small");
  }
  memcpy(frame.color.pixels(), rest.data(), frame.color.pixels_size);
  return true;
}

}  // namespace extras
}  // namespace jxl
