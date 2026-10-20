// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/exif.h"

#include <cstdint>
#include <vector>

#include "lib/jxl/base/byte_order.h"

namespace jxl {

constexpr uint16_t kExifOrientationTag = 274;

void ResetExifOrientation(std::vector<uint8_t>& exif) {
  if (exif.size() < 12) return;  // not enough bytes for a valid exif blob
  bool bigendian;
  // Walk the blob with a bounds-aware offset into `exif` instead of a raw
  // pointer cursor; every byte handed to the Load*/Store* helpers is derived
  // as `&exif[pos]` after the existing size checks, so no pointer arithmetic is
  // performed on the raw data pointer.
  size_t pos = 0;
  if (LoadLE32(&exif[pos]) == 0x2A004D4D) {
    bigendian = true;
  } else if (LoadLE32(&exif[pos]) == 0x002A4949) {
    bigendian = false;
  } else {
    return;  // not a valid tiff header
  }
  pos += 4;
  uint64_t offset = (bigendian ? LoadBE32(&exif[pos]) : LoadLE32(&exif[pos]));
  if (exif.size() < 12 + offset + 2 || offset < 8) return;
  pos += offset - 4;
  uint16_t nb_tags = (bigendian ? LoadBE16(&exif[pos]) : LoadLE16(&exif[pos]));
  pos += 2;
  while (nb_tags > 0) {
    if (pos + 12 >= exif.size()) return;
    uint16_t tag = (bigendian ? LoadBE16(&exif[pos]) : LoadLE16(&exif[pos]));
    pos += 2;
    if (tag == kExifOrientationTag) {
      uint16_t type = (bigendian ? LoadBE16(&exif[pos]) : LoadLE16(&exif[pos]));
      pos += 2;
      uint32_t count = (bigendian ? LoadBE32(&exif[pos]) : LoadLE32(&exif[pos]));
      pos += 4;
      if (type == 3 && count == 1) {
        if (bigendian) {
          StoreBE16(1, &exif[pos]);
        } else {
          StoreLE16(1, &exif[pos]);
        }
      }
      return;
    } else {
      pos += 10;
      nb_tags--;
    }
  }
}

}  // namespace jxl
