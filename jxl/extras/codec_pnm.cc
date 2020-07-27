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

#include "jxl/extras/codec_pnm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "jxl/base/bits.h"
#include "jxl/base/byte_order.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/file_io.h"
#include "jxl/color_management.h"
#include "jxl/external_image.h"
#include "jxl/fields.h"  // AllDefault
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/luminance.h"

namespace jxl {
namespace {

struct HeaderPNM {
  size_t xsize;
  size_t ysize;
  bool is_bit;   // PBM
  bool is_gray;  // PGM
  size_t bits_per_sample;
  bool floating_point;
  bool big_endian;
};

class Parser {
 public:
  explicit Parser(const Span<const uint8_t> bytes)
      : pos_(bytes.data()), end_(pos_ + bytes.size()) {}

  // Sets "pos" to the first non-header byte/pixel on success.
  Status ParseHeader(HeaderPNM* header, const uint8_t** pos) {
    // codec.cc ensures we have at least two bytes => no range check here.
    if (pos_[0] != 'P') return false;
    const uint8_t type = pos_[1];
    pos_ += 2;

    header->is_bit = false;

    switch (type) {
      case '4':
        header->is_bit = true;
        header->is_gray = true;
        header->bits_per_sample = 1;
        return ParseHeaderPNM(header, pos);

      case '5':
        header->is_gray = true;
        return ParseHeaderPNM(header, pos);

      case '6':
        header->is_gray = false;
        return ParseHeaderPNM(header, pos);

      case 'F':
        header->is_gray = false;
        return ParseHeaderPFM(header, pos);

      case 'f':
        header->is_gray = true;
        return ParseHeaderPFM(header, pos);
    }
    return false;
  }

  // Exposed for testing
  Status ParseUnsigned(size_t* number) {
    if (pos_ == end_) return JXL_FAILURE("PNM: reached end before number");
    if (!IsDigit(*pos_)) return JXL_FAILURE("PNM: expected unsigned number");

    *number = 0;
    while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
      *number *= 10;
      *number += *pos_ - '0';
      ++pos_;
    }

    return true;
  }

  Status ParseSigned(double* number) {
    if (pos_ == end_) return JXL_FAILURE("PNM: reached end before signed");

    if (*pos_ != '-' && *pos_ != '+' && !IsDigit(*pos_)) {
      return JXL_FAILURE("PNM: expected signed number");
    }

    // Skip sign
    const bool is_neg = *pos_ == '-';
    if (is_neg || *pos_ == '+') {
      ++pos_;
      if (pos_ == end_) return JXL_FAILURE("PNM: reached end before digits");
    }

    // Leading digits
    *number = 0.0;
    while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
      *number *= 10;
      *number += *pos_ - '0';
      ++pos_;
    }

    // Decimal places?
    if (pos_ < end_ && *pos_ == '.') {
      ++pos_;
      double place = 0.1;
      while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
        *number += (*pos_ - '0') * place;
        place *= 0.1;
        ++pos_;
      }
    }

    if (is_neg) *number = -*number;
    return true;
  }

 private:
  static bool IsDigit(const uint8_t c) { return '0' <= c && c <= '9'; }
  static bool IsLineBreak(const uint8_t c) { return c == '\r' || c == '\n'; }
  static bool IsWhitespace(const uint8_t c) {
    return IsLineBreak(c) || c == '\t' || c == ' ';
  }

  Status SkipBlank() {
    if (pos_ == end_) return JXL_FAILURE("PNM: reached end before blank");
    const uint8_t c = *pos_;
    if (c != ' ' && c != '\n') return JXL_FAILURE("PNM: expected blank");
    ++pos_;
    return true;
  }

  Status SkipSingleWhitespace() {
    if (pos_ == end_) return JXL_FAILURE("PNM: reached end before whitespace");
    if (!IsWhitespace(*pos_)) return JXL_FAILURE("PNM: expected whitespace");
    ++pos_;
    return true;
  }

  Status SkipWhitespace() {
    if (pos_ == end_) return JXL_FAILURE("PNM: reached end before whitespace");
    if (!IsWhitespace(*pos_) && *pos_ != '#') {
      return JXL_FAILURE("PNM: expected whitespace/comment");
    }

    while (pos_ < end_ && IsWhitespace(*pos_)) {
      ++pos_;
    }

    // Comment(s)
    while (pos_ != end_ && *pos_ == '#') {
      while (pos_ != end_ && !IsLineBreak(*pos_)) {
        ++pos_;
      }
      // Newline(s)
      while (pos_ != end_ && IsLineBreak(*pos_)) pos_++;
    }

    while (pos_ < end_ && IsWhitespace(*pos_)) {
      ++pos_;
    }
    return true;
  }

  Status ParseHeaderPNM(HeaderPNM* header, const uint8_t** pos) {
    JXL_RETURN_IF_ERROR(SkipWhitespace());
    JXL_RETURN_IF_ERROR(ParseUnsigned(&header->xsize));

    JXL_RETURN_IF_ERROR(SkipWhitespace());
    JXL_RETURN_IF_ERROR(ParseUnsigned(&header->ysize));

    if (!header->is_bit) {
      JXL_RETURN_IF_ERROR(SkipWhitespace());
      size_t max_val;
      JXL_RETURN_IF_ERROR(ParseUnsigned(&max_val));
      if (max_val == 0 || max_val >= 65536) {
        return JXL_FAILURE("PNM: bad MaxVal");
      }
      header->bits_per_sample = CeilLog2Nonzero(max_val);
    }
    header->floating_point = false;
    header->big_endian = true;

    JXL_RETURN_IF_ERROR(SkipSingleWhitespace());

    *pos = pos_;
    return true;
  }

  Status ParseHeaderPFM(HeaderPNM* header, const uint8_t** pos) {
    JXL_RETURN_IF_ERROR(SkipSingleWhitespace());
    JXL_RETURN_IF_ERROR(ParseUnsigned(&header->xsize));

    JXL_RETURN_IF_ERROR(SkipBlank());
    JXL_RETURN_IF_ERROR(ParseUnsigned(&header->ysize));

    JXL_RETURN_IF_ERROR(SkipSingleWhitespace());
    double scale;
    JXL_RETURN_IF_ERROR(ParseSigned(&scale));
    header->big_endian = scale >= 0.0;
    header->bits_per_sample = 32;
    header->floating_point = true;

    JXL_RETURN_IF_ERROR(SkipSingleWhitespace());

    *pos = pos_;
    return true;
  }

  const uint8_t* pos_;
  const uint8_t* const end_;
};

constexpr size_t kMaxHeaderSize = 200;

Status EncodeHeader(const ExternalImage& external, char* header,
                    int* JXL_RESTRICT chars_written) {
  if (external.HasAlpha()) return JXL_FAILURE("PNM: can't store alpha");

  if (external.BitsPerSample() == 32) {  // PFM
    const char type = external.IsGray() ? 'f' : 'F';
    const double scale = external.BigEndian() ? 1.0 : -1.0;
    snprintf(header, kMaxHeaderSize, "P%c %zu %zu\n%f\n%n", type,
             external.xsize(), external.ysize(), scale, chars_written);
  } else if (external.BitsPerSample() == 1) {  // PBM
    if (!external.IsGray()) {
      return JXL_FAILURE("Cannot encode color as PBM");
    }
    snprintf(header, kMaxHeaderSize, "P4\n%zu %zu\n%n", external.xsize(),
             external.ysize(), chars_written);
  } else {  // PGM/PPM
    const uint32_t max_val = (1U << external.BitsPerSample()) - 1;
    if (max_val >= 65536) return JXL_FAILURE("PNM cannot have > 16 bits");
    const char type = external.IsGray() ? '5' : '6';
    snprintf(header, kMaxHeaderSize, "P%c\n%zu %zu\n%u\n%n", type,
             external.xsize(), external.ysize(), max_val, chars_written);
  }
  return true;
}

Status ApplyHints(const bool is_gray, CodecInOut* io) {
  bool got_color_space = false;

  JXL_RETURN_IF_ERROR(io->dec_hints.Foreach(
      [is_gray, io, &got_color_space](const std::string& key,
                                      const std::string& value) -> Status {
        ColorEncoding* c_original = &io->metadata.color_encoding;
        if (key == "color_space") {
          if (!ParseDescription(value, c_original) ||
              !c_original->CreateICC()) {
            return JXL_FAILURE("PNM: Failed to apply color_space");
          }

          if (is_gray != io->metadata.color_encoding.IsGray()) {
            return JXL_FAILURE(
                "PNM: mismatch between file and color_space hint");
          }

          got_color_space = true;
        } else if (key == "icc_pathname") {
          PaddedBytes icc;
          JXL_RETURN_IF_ERROR(ReadFile(value, &icc));
          JXL_RETURN_IF_ERROR(c_original->SetICC(std::move(icc)));
          got_color_space = true;
        } else {
          JXL_WARNING("PNM decoder ignoring %s hint", key.c_str());
        }
        return true;
      }));

  if (!got_color_space) {
    JXL_WARNING("PNM: no color_space/icc_pathname given, assuming sRGB");
    JXL_RETURN_IF_ERROR(io->metadata.color_encoding.SetSRGB(
        is_gray ? ColorSpace::kGray : ColorSpace::kRGB));
  }

  return true;
}

Span<const uint8_t> MakeSpan(const char* str) {
  return Span<const uint8_t>(reinterpret_cast<const uint8_t*>(str),
                             strlen(str));
}

// Flip the image vertically for loading/saving PFM files which have the
// scanlines inverted.
Image3F VerticallyFlipImage(const Image3F& image) {
  Image3F flipped = Image3F(image.xsize(), image.ysize());
  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < image.ysize(); y++) {
      float* row_out = flipped.PlaneRow(c, y);
      const float* row_in = image.PlaneRow(c, image.ysize() - y - 1);
      memcpy(row_out, row_in, image.xsize() * sizeof(float));
    }
  }
  return flipped;
}

}  // namespace

Status DecodeImagePNM(const Span<const uint8_t> bytes, ThreadPool* pool,
                      CodecInOut* io) {
  io->enc_size = bytes.size();

  Parser parser(bytes);
  HeaderPNM header = {};
  const uint8_t* pos;
  if (!parser.ParseHeader(&header, &pos)) return false;
  JXL_RETURN_IF_ERROR(io->VerifyDimensions(header.xsize, header.ysize));

  if (header.bits_per_sample == 0 || header.bits_per_sample > 32) {
    return JXL_FAILURE("PNM: bits_per_sample invalid");
  }

  JXL_RETURN_IF_ERROR(ApplyHints(header.is_gray, io));
  if (header.floating_point) {
    io->metadata.SetFloat32Samples();
  } else {
    io->metadata.SetUintSamples(header.bits_per_sample);
  }
  io->metadata.SetAlphaBits(0);
  io->dec_pixels = header.xsize * header.ysize;
  io->frames.clear();
  io->frames.reserve(1);
  ImageBundle ib(&io->metadata);

  const bool flipped_y = header.bits_per_sample == 32;  // PFMs are flipped
  const PackedImage desc(
      header.xsize, header.ysize, io->metadata.color_encoding,
      /*has_alpha=*/false, /*alpha_is_premultiplied=*/false,
      io->metadata.GetAlphaBits(), io->metadata.bit_depth.bits_per_sample,
      header.big_endian, flipped_y);
  const Span<const uint8_t> span(pos, bytes.data() + bytes.size() - pos);
  if (!CopyTo(desc, span, pool, &ib)) return false;
  if (!header.floating_point) {
    io->metadata.bit_depth.bits_per_sample = ib.DetectRealBitdepth();
  }
  io->frames.push_back(std::move(ib));
  return Map255ToTargetNits(io, pool);
}

Status EncodeImagePNM(const CodecInOut* io, const ColorEncoding& c_desired,
                      size_t bits_per_sample, ThreadPool* pool,
                      PaddedBytes* bytes) {
  const bool floating_point = bits_per_sample > 16;
  io->enc_bits_per_sample = floating_point ? 32 : bits_per_sample;
  // Choose native for PFM; PGM/PPM require big-endian (N/A for PBM)
  const bool big_endian = floating_point ? !IsLittleEndian() : true;

  if (!Bundle::AllDefault(io->metadata)) {
    JXL_WARNING("PNM encoder ignoring metadata - use a different codec");
  }
  if (!c_desired.IsSRGB()) {
    JXL_WARNING(
        "PNM encoder cannot store custom ICC profile; decoder\n"
        "will need hint key=color_space to get the same values");
  }

  ImageBundle ib = io->Main().Copy();
  JXL_RETURN_IF_ERROR(MapTargetNitsTo255(&ib, pool));
  const ImageU* alpha = ib.HasAlpha() ? &ib.alpha() : nullptr;
  const size_t alpha_bits = ib.HasAlpha() ? io->metadata.GetAlphaBits() : 0;
  CodecIntervals* temp_intervals = nullptr;  // Can't store min/max.

  // TODO(lode): Implement the flipping in external_image.cc instead.
  // In case of PFM the image must be flipped upside down since that format
  // is designed that way.
  const Image3F* to_external_image = &ib.color();
  Image3F flipped;
  if (floating_point) {
    flipped = VerticallyFlipImage(ib.color());
    to_external_image = &flipped;
  }
  ExternalImage external(pool, *to_external_image, Rect(ib), ib.c_current(),
                         c_desired, ib.HasAlpha(), ib.AlphaIsPremultiplied(),
                         alpha, alpha_bits, io->enc_bits_per_sample, big_endian,
                         temp_intervals);
  JXL_RETURN_IF_ERROR(external.IsHealthy());

  char header[kMaxHeaderSize];
  int header_size = 0;
  JXL_RETURN_IF_ERROR(EncodeHeader(external, header, &header_size));

  const PaddedBytes& pixels = external.Bytes();
  io->enc_size = static_cast<size_t>(header_size) + pixels.size();
  bytes->resize(io->enc_size);
  memcpy(bytes->data(), header, static_cast<size_t>(header_size));
  memcpy(bytes->data() + header_size, pixels.data(), pixels.size());

  return true;
}

void TestCodecPNM() {
  size_t u = 77777;  // Initialized to wrong value.
  double d = 77.77;
// Failing to parse invalid strings results in a crash if `JXL_CRASH_ON_ERROR`
// is defined and hence the tests fail. Therefore we only run these tests if
// `JXL_CRASH_ON_ERROR` is not defined.
#ifndef JXL_CRASH_ON_ERROR
  JXL_CHECK(false == Parser(MakeSpan("")).ParseUnsigned(&u));
  JXL_CHECK(false == Parser(MakeSpan("+")).ParseUnsigned(&u));
  JXL_CHECK(false == Parser(MakeSpan("-")).ParseUnsigned(&u));
  JXL_CHECK(false == Parser(MakeSpan("A")).ParseUnsigned(&u));

  JXL_CHECK(false == Parser(MakeSpan("")).ParseSigned(&d));
  JXL_CHECK(false == Parser(MakeSpan("+")).ParseSigned(&d));
  JXL_CHECK(false == Parser(MakeSpan("-")).ParseSigned(&d));
  JXL_CHECK(false == Parser(MakeSpan("A")).ParseSigned(&d));
#endif
  JXL_CHECK(true == Parser(MakeSpan("1")).ParseUnsigned(&u));
  JXL_CHECK(u == 1);

  JXL_CHECK(true == Parser(MakeSpan("32")).ParseUnsigned(&u));
  JXL_CHECK(u == 32);

  JXL_CHECK(true == Parser(MakeSpan("1")).ParseSigned(&d));
  JXL_CHECK(d == 1.0);
  JXL_CHECK(true == Parser(MakeSpan("+2")).ParseSigned(&d));
  JXL_CHECK(d == 2.0);
  JXL_CHECK(true == Parser(MakeSpan("-3")).ParseSigned(&d));
  JXL_CHECK(std::abs(d - -3.0) < 1E-15);
  JXL_CHECK(true == Parser(MakeSpan("3.141592")).ParseSigned(&d));
  JXL_CHECK(std::abs(d - 3.141592) < 1E-15);
  JXL_CHECK(true == Parser(MakeSpan("-3.141592")).ParseSigned(&d));
  JXL_CHECK(std::abs(d - -3.141592) < 1E-15);
}

}  // namespace jxl
