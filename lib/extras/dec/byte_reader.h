// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_EXTRAS_DEC_BYTE_READER_H_
#define LIB_EXTRAS_DEC_BYTE_READER_H_

// Small bounds-checked cursor for byte-oriented image parsers. It keeps the
// "never read past the end" invariant in one place instead of re-deriving it
// at every call site.

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {

class ByteReader {
 public:
  explicit constexpr ByteReader(Span<const uint8_t> bytes) : bytes_(bytes) {}

  constexpr bool AtEnd() const { return pos_ >= bytes_.size(); }
  constexpr size_t remaining() const { return bytes_.size() - pos_; }
  constexpr size_t offset() const { return pos_; }
  constexpr const uint8_t* current() const {
    return pos_ == 0 ? bytes_.data() : bytes_.data() + pos_;
  }

  Status Peek(uint8_t* value) const {
    if (AtEnd()) return JXL_FAILURE("ByteReader: read past end");
    *value = bytes_[pos_];
    return true;
  }

  Status ReadU8(uint8_t* value) {
    if (AtEnd()) return JXL_FAILURE("ByteReader: read past end");
    *value = bytes_[pos_++];
    return true;
  }

  Status Read(size_t n, Span<const uint8_t>* out) {
    if (n > remaining()) return JXL_FAILURE("ByteReader: read past end");
    *out = Span<const uint8_t>(current(), n);
    pos_ += n;
    return true;
  }

  Status Skip(size_t n) {
    if (n > remaining()) return JXL_FAILURE("ByteReader: skip past end");
    pos_ += n;
    return true;
  }

 private:
  Span<const uint8_t> bytes_;
  size_t pos_ = 0;
};

}  // namespace extras
}  // namespace jxl

#endif  // LIB_EXTRAS_DEC_BYTE_READER_H_
