// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/gain_map.h>
#include <jxl/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/fields.h"

JXL_BOOL JxlGainMapReadBundle(JxlGainMapBundle* map_bundle,
                              const uint8_t* input_buffer,
                              const size_t input_buffer_size,
                              size_t* bytes_read) {
  if (map_bundle == nullptr || input_buffer == nullptr ||
      input_buffer_size == 0) {
    return JXL_FALSE;
  }

  uint64_t cursor = 0;
  uint64_t next_cursor = 0;

#define SAFE_CURSOR_UPDATE(n)                              \
  do {                                                     \
    cursor = next_cursor;                                  \
    if (!jxl::SafeAdd<uint64_t>(cursor, n, next_cursor) || \
        next_cursor > input_buffer_size) {                 \
      return JXL_FALSE;                                    \
    }                                                      \
  } while (false)

  // Read the version byte
  SAFE_CURSOR_UPDATE(1);
  map_bundle->jhgm_version = input_buffer[cursor];

  // Read gain_map_metadata_size
  SAFE_CURSOR_UPDATE(2);
  uint16_t gain_map_metadata_size = LoadBE16(input_buffer + cursor);

  SAFE_CURSOR_UPDATE(gain_map_metadata_size);
  map_bundle->gain_map_metadata_size = gain_map_metadata_size;
  map_bundle->gain_map_metadata = input_buffer + cursor;

  // Read compressed_color_encoding_size
  SAFE_CURSOR_UPDATE(1);
  uint8_t compressed_color_encoding_size;
  memcpy(&compressed_color_encoding_size, input_buffer + cursor, 1);

  map_bundle->has_color_encoding = (compressed_color_encoding_size > 0);
  if (map_bundle->has_color_encoding) {
    SAFE_CURSOR_UPDATE(compressed_color_encoding_size);
    // Decode color encoding
    jxl::Span<const uint8_t> color_encoding_span(
        input_buffer + cursor, compressed_color_encoding_size);
    jxl::BitReader color_encoding_reader(color_encoding_span);
    jxl::ColorEncoding internal_color_encoding;
    if (!jxl::Bundle::Read(&color_encoding_reader, &internal_color_encoding)) {
      return JXL_FALSE;
    }
    JXL_RETURN_IF_ERROR(color_encoding_reader.Close());
    map_bundle->color_encoding = internal_color_encoding.ToExternal();
  }

  // Read compressed_icc_size
  SAFE_CURSOR_UPDATE(4);
  uint32_t compressed_icc_size = LoadBE32(input_buffer + cursor);

  SAFE_CURSOR_UPDATE(compressed_icc_size);
  map_bundle->alt_icc_size = compressed_icc_size;
  map_bundle->alt_icc = input_buffer + cursor;

  // Calculate remaining bytes for gain map
  cursor = next_cursor;
  // Compute remaining bytes and ensure they fit into the public
  // `uint32_t gain_map_size` field to avoid silent truncation on platforms
  // where `size_t` is wider than 32 bits.
  size_t remaining = input_buffer_size - cursor;
  if (remaining > std::numeric_limits<uint32_t>::max()) {
    return JXL_FALSE;
  }
  map_bundle->gain_map_size = static_cast<uint32_t>(remaining);
  SAFE_CURSOR_UPDATE(map_bundle->gain_map_size);
  map_bundle->gain_map = input_buffer + cursor;

#undef SAFE_CURSOR_UPDATE

  cursor = next_cursor;

  if (bytes_read != nullptr) {
    *bytes_read = cursor;
  }
  return JXL_TRUE;
}
