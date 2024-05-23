// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/gain_map.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"

namespace jxl {

struct JxlGainMapBundleInternal : public Fields {
  JxlGainMapBundle bundle;

  JXL_FIELDS_NAME(JxlGainMapBundleInternal)

  Status VisitFields(Visitor *JXL_RESTRICT visitor) override {
    uint32_t temp_u32;
    // Promote uint8_t to uint32_t for the visitor
    temp_u32 = bundle.jhgm_version;
    JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &temp_u32));
    bundle.jhgm_version = static_cast<uint8_t>(temp_u32);

    // Promote uint16_t to uint32_t for the visitor
    temp_u32 = bundle.gain_map_metadata_size;
    JXL_RETURN_IF_ERROR(visitor->Bits(16, 0, &temp_u32));
    bundle.gain_map_metadata_size = static_cast<uint16_t>(temp_u32);

    // Assume gain_map_metadata is a binary blob that we visit as raw bits
    if (bundle.gain_map_metadata_size > 0) {
      // Visitor doesn't directly handle raw bit arrays, so we need a workaround
      for (size_t i = 0; i < bundle.gain_map_metadata_size; ++i) {
        temp_u32 = bundle.gain_map_metadata[i];
        JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &temp_u32));
        bundle.gain_map_metadata[i] = static_cast<uint8_t>(temp_u32);
      }
    }

    jxl::ColorEncoding internal_color_encoding;
    JXL_RETURN_IF_ERROR(
        internal_color_encoding.FromExternal(bundle.color_encoding));
    // Visit the color_encoding as a whole block
    JXL_RETURN_IF_ERROR(visitor->VisitNested(&internal_color_encoding));

    // Assume alt_icc is a binary blob that we visit as raw bits
    if (bundle.alt_icc) {
      size_t alt_icc_size = strlen(reinterpret_cast<char *>(bundle.alt_icc));
      for (size_t i = 0; i < alt_icc_size; ++i) {
        temp_u32 = bundle.alt_icc[i];
        JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &temp_u32));
        bundle.alt_icc[i] = static_cast<uint8_t>(temp_u32);
      }
    }

    // Assume gain_map is a binary blob that we visit as raw bits
    if (bundle.gain_map) {
      for (size_t i = 0; i < bundle.gain_map_size; ++i) {
        temp_u32 = bundle.gain_map[i];
        JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &temp_u32));
        bundle.gain_map[i] = static_cast<uint8_t>(temp_u32);
      }
    }

    return true;
  }
};
}  // namespace jxl

size_t JxlGainMapGetBundleSize(const JxlGainMapBundle *map_bundle) {
  if (map_bundle == nullptr) return 0;

  jxl::JxlGainMapBundleInternal internal_bundle;
  internal_bundle.bundle = *map_bundle;

  return jxl::Bundle::MaxBits(internal_bundle) / 8;
}

size_t JxlGainMapWriteBundle(const JxlGainMapBundle *map_bundle,
                             uint8_t *output_buffer,
                             size_t output_buffer_size) {
  if (map_bundle == nullptr || output_buffer == nullptr) return 0;

  jxl::JxlGainMapBundleInternal internal_bundle;
  internal_bundle.bundle = *map_bundle;

  size_t required_size = jxl::Bundle::MaxBits(internal_bundle) / 8;
  if (output_buffer_size < required_size)
    return 0;  // Not enough space in the output buffer
  jxl::BitWriter writer;
  jxl::BitWriter::Allotment allotment(&writer, output_buffer_size);
  if (!jxl::Bundle::Write(internal_bundle, &writer, /*layer=*/0, nullptr)) {
    return 0;  // Failed to write the bundle
  }
  return writer.BitsWritten() / 8;  // Return the number of bytes written
}

size_t JxlGainMapReadBundle(JxlGainMapBundle *map_bundle,
                            const uint8_t *input_buffer,
                            const size_t input_buffer_size) {
  if (map_bundle == nullptr || input_buffer == nullptr) return 0;

  jxl::BitReader reader(
      jxl::Span<const uint8_t>(input_buffer, input_buffer_size));
  jxl::JxlGainMapBundleInternal internal_bundle;

  if (!jxl::Bundle::Read(&reader, &internal_bundle)) {
    return 0;  // Failed to read the bundle
  }

  // Copy the internal bundle to the output bundle
  *map_bundle = internal_bundle.bundle;

  // Return the number of bytes read
  return reader.TotalBitsConsumed() / 8;
}