// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_JPEG_JPEG_BYTE_CURSOR_H_
#define LIB_JXL_JPEG_JPEG_BYTE_CURSOR_H_

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace jpeg {

// A non-owning reader for byte-aligned JPEG marker data. The input must remain
// valid for the lifetime of the cursor and any views returned by Take().
//
// Every operation checks its complete byte count before advancing the cursor.
// A failed operation leaves both the offset and its output unchanged. Output
// pointers must be non-null.
class JpegByteCursor {
 public:
  explicit JpegByteCursor(Bytes bytes) : bytes_(bytes) {}

  Status U8(uint8_t* value) {
    JXL_RETURN_IF_ERROR(CheckAvailable(1));
    *value = bytes_[offset_];
    ++offset_;
    return true;
  }

  Status U16BE(uint16_t* value) {
    JXL_RETURN_IF_ERROR(CheckAvailable(2));
    *value = (static_cast<uint16_t>(bytes_[offset_]) << 8) |
             static_cast<uint16_t>(bytes_[offset_ + 1]);
    offset_ += 2;
    return true;
  }

  Status Skip(size_t count) {
    JXL_RETURN_IF_ERROR(CheckAvailable(count));
    offset_ += count;
    return true;
  }

  Status Take(size_t count, Bytes* bytes) {
    JXL_RETURN_IF_ERROR(CheckAvailable(count));
    // Avoid pointer arithmetic on a null pointer for an empty input.
    const uint8_t* data = bytes_.data();
    if (offset_ != 0) data += offset_;
    *bytes = Bytes(data, count);
    offset_ += count;
    return true;
  }

  size_t offset() const { return offset_; }
  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  Status CheckAvailable(size_t count) const {
    // offset_ <= bytes_.size() is preserved by all advancing operations. Check
    // against the remaining size instead of forming offset_ + count, which
    // could wrap when count comes from an invalid length.
    if (count > remaining()) {
      return JXL_FAILURE("Unexpected end of JPEG byte input");
    }
    return true;
  }

  Bytes bytes_;
  size_t offset_ = 0;
};

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_JPEG_BYTE_CURSOR_H_
