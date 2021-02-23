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

#include "lib/jxl/box.h"

#include <cstring>

#include "lib/jxl/base/byte_order.h"

namespace jxl {

namespace {
// Checks if a + b > size, taking possible integer overflow into account.
bool OutOfBounds(size_t a, size_t b, size_t size) {
  size_t pos = a + b;
  if (pos > size) return true;
  if (pos < a) return true;  // overflow happened
  return false;
}
}  // namespace

void JxlBox::Encode(std::vector<uint8_t>* out) {
  bool use_extended = !std::memcmp("uuid", type, 4);

  uint64_t box_size = 0;
  bool large_size = false;
  if (data_size_given) {
    box_size = data.size() + 8 + (use_extended ? 16 : 0);
    if (box_size >= 0x100000000ull) {
      large_size = true;
    }
  }

  out->resize(out->size() + 4);
  StoreBE32(large_size ? 1 : box_size, &out->back() - 4 + 1);

  out->resize(out->size() + 4);
  memcpy(&out->back() - 4 + 1, type, 4);

  if (large_size) {
    out->resize(out->size() + 8);
    StoreBE64(box_size, &out->back() - 8 + 1);
  }

  if (use_extended) {
    out->resize(out->size() + 16);
    memcpy(&out->back() - 16 + 1, extended_type, 16);
  }

  out->insert(out->end(), data.data(), data.data() + data.size());
}

Status JxlBox::Decode(Span<uint8_t>* in) {
  if (OutOfBounds(0, 8, in->size())) return JXL_FAILURE("Out of bounds");

  // Total box_size including this header itself.
  uint64_t box_size = LoadBE32(in->data());
  size_t pos = 4;

  memcpy(type, in->data() + pos, 4);
  pos += 4;

  if (box_size == 1) {
    // If the size is 1, it indicates extended size read from 64-bit integer.
    if (OutOfBounds(pos, 8, in->size())) return JXL_FAILURE("Out of bounds");
    box_size = LoadBE64(in->data() + pos);
    pos += 8;
  }

  if (!memcmp("uuid", type, 4)) {
    if (OutOfBounds(pos, 16, in->size())) return JXL_FAILURE("Out of bounds");
    memcpy(extended_type, in->data() + pos, 16);
    pos += 16;
  }

  // This is the end of the box header, the box data begins here. Handle
  // the data size now.
  const size_t header_size = pos;

  if (box_size != 0) {
    if (box_size < header_size) {
      return JXL_FAILURE("Invalid box size");
    }
    data_size_given = true;
    data =
        Span<const uint8_t>(in->data() + header_size, box_size - header_size);
  } else {
    data =
        Span<const uint8_t>(in->data() + header_size, in->size() - header_size);
  }

  *in = Span<uint8_t>(in->data() + header_size + data.size(),
                      in->size() - header_size - data.size());
  return true;
}

void JxlContainer::Encode(std::vector<uint8_t>* out) {
  const unsigned char header[] = {0,   0,   0,    0xc, 'J', 'X', 'L', ' ',
                                  0xd, 0xa, 0x87, 0xa, 0,   0,   0,   0x14,
                                  'f', 't', 'y',  'p', 'j', 'x', 'l', ' ',
                                  0,   0,   0,    0,   'j', 'x', 'l', ' '};
  out->insert(out->end(), header, header + sizeof(header));
  for (JxlBox& box : boxes) {
    box.Encode(out);
  }
}

Status JxlContainer::Decode(Span<uint8_t>* in) {
  boxes.clear();

  JxlBox signature_box;
  JXL_RETURN_IF_ERROR(signature_box.Decode(in));
  if (memcmp("JXL ", signature_box.type, 4) != 0) {
    return JXL_FAILURE("Invalid magic signature");
  }
  if (signature_box.data.size() != 4)
    return JXL_FAILURE("Invalid magic signature");
  if (signature_box.data[0] != 0xd || signature_box.data[1] != 0xa ||
      signature_box.data[2] != 0x87 || signature_box.data[3] != 0xa) {
    return JXL_FAILURE("Invalid magic signature");
  }

  JxlBox ftyp_box;
  JXL_RETURN_IF_ERROR(ftyp_box.Decode(in));
  if (memcmp("ftyp", ftyp_box.type, 4) != 0) {
    return JXL_FAILURE("Invalid ftyp");
  }
  if (ftyp_box.data.size() != 12) return JXL_FAILURE("Invalid ftyp");
  const char* expected = "jxl \0\0\0\0jxl ";
  if (memcmp(expected, ftyp_box.data.data(), 12) != 0)
    return JXL_FAILURE("Invalid ftyp");

  while (in->size() > 0) {
    JxlBox box = {};
    JXL_RETURN_IF_ERROR(box.Decode(in));
    boxes.emplace_back(box);
  }

  return true;
}

}  // namespace jxl
