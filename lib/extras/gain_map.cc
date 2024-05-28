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
#include "lib/jxl/enc_icc_codec.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/icc_codec.h"
namespace {
// Function to swap the byte order of uint32_t
inline uint32_t SwapByteOrder(uint32_t value) {
  return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) |
         ((value & 0x00FF0000) >> 8) | ((value & 0xFF000000) >> 24);
}

// Function to swap the byte order of uint16_t
inline uint16_t SwapByteOrder(uint16_t value) {
  return ((value & 0x00FF) << 8) | ((value & 0xFF00) >> 8);
}
}  // namespace
namespace jxl {

struct JxlGainMapBundleInternal : public Fields {
  uint8_t jhgm_version;
  bool has_color_encoding;
  ColorEncoding color_encoding;
  std::vector<uint8_t> gain_map_metadata;
  IccBytes compressed_icc;
  std::vector<uint8_t> gain_map;

  JXL_FIELDS_NAME(JxlGainMapBundleInternal)

  Status VisitFields(Visitor *JXL_RESTRICT visitor) override {
    uint32_t temp_u32;
    // uint32_t temp_u32_le;
    //  Promote uint8_t to uint32_t for the visitor
    temp_u32 = jhgm_version;
    JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &temp_u32));
    jhgm_version = static_cast<uint8_t>(temp_u32);

    // Byte order is little endian
    uint16_t temp_gain_map_metadata_size =
        SwapByteOrder(static_cast<uint16_t>(gain_map_metadata.size()));
    // Promote uint16_t to uint32_t for the visitor
    temp_u32 = temp_gain_map_metadata_size;
    JXL_RETURN_IF_ERROR(visitor->Bits(16, 0, &temp_u32));
    // Restore byte order
    temp_gain_map_metadata_size =
        SwapByteOrder(static_cast<uint16_t>(temp_u32));
    gain_map_metadata.resize(temp_gain_map_metadata_size);

    // Assume gain_map_metadata is a binary blob that we visit as raw bits
    // Visitor doesn't directly handle raw bit arrays, so we need a workaround
    for (size_t i = 0; i < gain_map_metadata.size(); ++i) {
      temp_u32 = gain_map_metadata[i];
      JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &temp_u32));
      gain_map_metadata[i] = static_cast<uint8_t>(temp_u32);
    }

    JXL_RETURN_IF_ERROR(visitor->Bool(false, &has_color_encoding));

    if (visitor->Conditional(has_color_encoding)) {
      // Visit the color_encoding as a whole block
      JXL_RETURN_IF_ERROR(visitor->VisitNested(&color_encoding));
    }

    temp_u32 = SwapByteOrder(static_cast<uint32_t>(compressed_icc.size()));
    JXL_RETURN_IF_ERROR(visitor->Bits(32, 0, &temp_u32));
    compressed_icc.resize(SwapByteOrder(temp_u32));

    for (size_t i = 0; i < compressed_icc.size(); ++i) {
      temp_u32 = compressed_icc[i];
      JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &temp_u32));
      compressed_icc[i] = static_cast<uint8_t>(temp_u32);
    }

    temp_u32 = SwapByteOrder(static_cast<uint32_t>(gain_map.size()));
    JXL_RETURN_IF_ERROR(visitor->Bits(32, 0, &temp_u32));
    gain_map.resize(SwapByteOrder(temp_u32));
    // Assume gain_map is a binary blob that we visit as raw bits
    for (size_t i = 0; i < gain_map.size(); ++i) {
      temp_u32 = gain_map[i];
      JXL_RETURN_IF_ERROR(visitor->Bits(8, 0, &temp_u32));
      gain_map[i] = static_cast<uint8_t>(temp_u32);
    }
    return true;
  }
};
}  // namespace jxl

size_t JxlGainMapGetBundleSize(const JxlGainMapBundle *map_bundle) {
  if (map_bundle == nullptr) return 0;

  jxl::JxlGainMapBundleInternal internal_bundle;
  internal_bundle.jhgm_version = map_bundle->jhgm_version;
  internal_bundle.has_color_encoding = map_bundle->has_color_encoding;
  JXL_RETURN_IF_ERROR(
      internal_bundle.color_encoding.FromExternal(map_bundle->color_encoding));

  // Initialize vectors from raw data and sizes
  internal_bundle.gain_map_metadata = std::vector<uint8_t>(
      map_bundle->gain_map_metadata,
      map_bundle->gain_map_metadata + map_bundle->gain_map_metadata_size);
  internal_bundle.gain_map = std::vector<uint8_t>(
      map_bundle->gain_map, map_bundle->gain_map + map_bundle->gain_map_size);

  jxl::BitWriter icc_writer;
  if (map_bundle->alt_icc) {
    jxl::IccBytes icc_bytes(map_bundle->alt_icc,
                            map_bundle->alt_icc + map_bundle->alt_icc_size);
    JXL_RETURN_IF_ERROR(jxl::WriteICC(icc_bytes, &icc_writer, 0, nullptr));
    icc_writer.ZeroPadToByte();
    internal_bundle.compressed_icc.assign(icc_writer.GetSpan().begin(),
                                          icc_writer.GetSpan().end());
  }

  jxl::BitWriter writer;
  if (!jxl::Bundle::Write(internal_bundle, &writer, /*layer=*/0, nullptr)) {
    return 0;  // Failed to write the bundle
  }
  return jxl::DivCeil(writer.BitsWritten(), 8);
}

size_t JxlGainMapWriteBundle(const JxlGainMapBundle *map_bundle,
                             uint8_t *output_buffer,
                             size_t output_buffer_size) {
  jxl::JxlGainMapBundleInternal internal_bundle;
  if (map_bundle == nullptr) return 0;
  internal_bundle.jhgm_version = map_bundle->jhgm_version;
  internal_bundle.has_color_encoding = map_bundle->has_color_encoding;
  JXL_RETURN_IF_ERROR(
      internal_bundle.color_encoding.FromExternal(map_bundle->color_encoding));

  // Initialize vectors from raw data and sizes
  internal_bundle.gain_map_metadata = std::vector<uint8_t>(
      map_bundle->gain_map_metadata,
      map_bundle->gain_map_metadata + map_bundle->gain_map_metadata_size);
  internal_bundle.gain_map = std::vector<uint8_t>(
      map_bundle->gain_map, map_bundle->gain_map + map_bundle->gain_map_size);

  jxl::BitWriter icc_writer;
  if (map_bundle->alt_icc) {
    jxl::IccBytes icc_bytes(map_bundle->alt_icc,
                            map_bundle->alt_icc + map_bundle->alt_icc_size);
    JXL_RETURN_IF_ERROR(jxl::WriteICC(icc_bytes, &icc_writer, 0, nullptr));
    icc_writer.ZeroPadToByte();
    internal_bundle.compressed_icc.assign(icc_writer.GetSpan().begin(),
                                          icc_writer.GetSpan().end());
  }

  jxl::BitWriter writer;
  if (!jxl::Bundle::Write(internal_bundle, &writer, /*layer=*/0, nullptr)) {
    return 0;  // Failed to write the bundle
  }
  writer.ZeroPadToByte();
  size_t size = jxl::DivCeil(writer.BitsWritten(), 8);
  memcpy(output_buffer, writer.GetSpan().data(), size);
  return size;
}

void JxlGainMapGetBufferSizes(JxlGainMapBundle *map_bundle,
                              const uint8_t *input_buffer,
                              const size_t input_buffer_size) {
  if (input_buffer == nullptr) {
    map_bundle->alt_icc_size = 0;
    map_bundle->gain_map_metadata_size = 0;
    map_bundle->has_color_encoding = false;
    map_bundle->gain_map_size = 0;
  }

  jxl::BitReader reader(
      jxl::Span<const uint8_t>(input_buffer, input_buffer_size));
  jxl::JxlGainMapBundleInternal internal_bundle;
  if (!jxl::Bundle::Read(&reader, &internal_bundle)) {
    fprintf(stderr, "Failed to read the bundle\n");
    return;
  }
  (void)(reader.Close());

  jxl::BitReader bit_reader(internal_bundle.compressed_icc);
  jxl::ICCReader icc_reader;
  jxl::PaddedBytes icc_buffer;
  (void)icc_reader.Init(&bit_reader, 0UL);
  (void)icc_reader.Process(&bit_reader, &icc_buffer);
  (void)bit_reader.Close();
  map_bundle->alt_icc_size = icc_buffer.size();
  map_bundle->gain_map_metadata_size = internal_bundle.gain_map_metadata.size();
  map_bundle->has_color_encoding = internal_bundle.has_color_encoding;
  map_bundle->gain_map_size = internal_bundle.gain_map.size();
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
  map_bundle->has_color_encoding = internal_bundle.has_color_encoding;
  map_bundle->jhgm_version = internal_bundle.jhgm_version;

  if (map_bundle->gain_map_metadata_size ==
      internal_bundle.gain_map_metadata.size()) {
    std::memcpy(map_bundle->gain_map_metadata,
                internal_bundle.gain_map_metadata.data(),
                internal_bundle.gain_map_metadata.size());
  }
  if (map_bundle->gain_map_size == internal_bundle.gain_map.size()) {
    std::memcpy(map_bundle->gain_map, internal_bundle.gain_map.data(),
                internal_bundle.gain_map.size());
  }
  map_bundle->color_encoding = internal_bundle.color_encoding.ToExternal();
  // Return the number of bytes read
  (void)reader.Close();

  jxl::BitReader bit_reader(internal_bundle.compressed_icc);
  jxl::ICCReader icc_reader;
  jxl::PaddedBytes icc_buffer;
  (void)icc_reader.Init(&bit_reader, 0UL);
  (void)icc_reader.Process(&bit_reader, &icc_buffer);
  (void)bit_reader.Close();
  if (map_bundle->alt_icc_size == icc_buffer.size()) {
    std::memcpy(map_bundle->alt_icc, icc_buffer.data(), icc_buffer.size());
  }

  return reader.TotalBitsConsumed() / 8;
}