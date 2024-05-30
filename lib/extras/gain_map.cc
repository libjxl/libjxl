// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/gain_map.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_icc_codec.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/icc_codec.h"

JXL_BOOL JxlGainMapGetBundleSize(JxlMemoryManager* memory_manager,
                                 const JxlGainMapBundle* map_bundle,
                                 size_t* bundle_size) {
  if (map_bundle == nullptr) return 0;

  jxl::ColorEncoding internal_color_encoding;
  jxl::BitWriter color_encoding_writer(memory_manager);
  if (map_bundle->has_color_encoding) {
    JXL_RETURN_IF_ERROR(
        internal_color_encoding.FromExternal(map_bundle->color_encoding));
    if (!jxl::Bundle::Write(internal_color_encoding, &color_encoding_writer,
                            /*layer=*/0, nullptr)) {
      return JXL_FALSE;
    }
  }
  color_encoding_writer.ZeroPadToByte();
  std::vector<uint8_t> compressed_color_encoding(
      color_encoding_writer.GetSpan().data(),
      color_encoding_writer.GetSpan().data() +
          color_encoding_writer.GetSpan().size());
  // Initialize vectors from raw data and sizes
  std::vector<uint8_t> gain_map_metadata(
      map_bundle->gain_map_metadata,
      map_bundle->gain_map_metadata + map_bundle->gain_map_metadata_size);
  std::vector<uint8_t> gain_map(
      map_bundle->gain_map, map_bundle->gain_map + map_bundle->gain_map_size);

  jxl::BitWriter icc_writer(memory_manager);
  std::vector<uint8_t> compressed_icc;
  if (map_bundle->alt_icc) {
    jxl::IccBytes icc_bytes(map_bundle->alt_icc,
                            map_bundle->alt_icc + map_bundle->alt_icc_size);
    JXL_RETURN_IF_ERROR(jxl::WriteICC(icc_bytes, &icc_writer, 0, nullptr));
    icc_writer.ZeroPadToByte();
    compressed_icc.assign(icc_writer.GetSpan().begin(),
                          icc_writer.GetSpan().end());
  }

  *bundle_size =
      1 +                                 // size of jhgm_version
      2 +                                 // size_of gain_map_metadata_size
      gain_map_metadata.size() +          // size of gain_map_metadata
      4 +                                 // size of color_encoding_size
      compressed_color_encoding.size() +  // size of the color_encoding
      4 +                                 // size of compressed_icc_size
      compressed_icc.size() +             // size of compressed_icc
      map_bundle->gain_map_size;          // size of gain map
  return JXL_TRUE;
}

JXL_BOOL JxlGainMapWriteBundle(JxlMemoryManager* memory_manager,
                               const JxlGainMapBundle* map_bundle,
                               uint8_t* output_buffer,
                               size_t output_buffer_size,
                               size_t* bytes_written) {
  // TODO avoid code duplication with JxlGainMapGetBundleSize
  if (map_bundle == nullptr) return 0;

  uint8_t jhgm_version = map_bundle->jhgm_version;
  size_t gain_map_size = map_bundle->gain_map_size;

  jxl::ColorEncoding internal_color_encoding;
  jxl::BitWriter color_encoding_writer(memory_manager);
  if (map_bundle->has_color_encoding) {
    JXL_RETURN_IF_ERROR(
        internal_color_encoding.FromExternal(map_bundle->color_encoding));
    if (!jxl::Bundle::Write(internal_color_encoding, &color_encoding_writer,
                            /*layer=*/0, nullptr)) {
      return JXL_FALSE;
    }
  }

  color_encoding_writer.ZeroPadToByte();
  std::vector<uint8_t> compressed_color_encoding(
      color_encoding_writer.GetSpan().data(),
      color_encoding_writer.GetSpan().data() +
          color_encoding_writer.GetSpan().size());

  // Initialize vectors from raw data and sizes
  std::vector<uint8_t> gain_map_metadata(
      map_bundle->gain_map_metadata,
      map_bundle->gain_map_metadata + map_bundle->gain_map_metadata_size);
  std::vector<uint8_t> gain_map(
      map_bundle->gain_map, map_bundle->gain_map + map_bundle->gain_map_size);

  jxl::BitWriter icc_writer(memory_manager);
  std::vector<uint8_t> compressed_icc;

  if (map_bundle->alt_icc) {
    jxl::IccBytes icc_bytes(map_bundle->alt_icc,
                            map_bundle->alt_icc + map_bundle->alt_icc_size);
    JXL_RETURN_IF_ERROR(jxl::WriteICC(icc_bytes, &icc_writer, 0, nullptr));
    icc_writer.ZeroPadToByte();
    compressed_icc.assign(icc_writer.GetSpan().begin(),
                          icc_writer.GetSpan().end());
  }
  size_t cursor = 0;
  if (cursor + 1 <= output_buffer_size) {
    memcpy(output_buffer + cursor, &jhgm_version, 1);
    cursor += 1;
  }

  uint16_t metadata_size = gain_map_metadata.size();
  uint16_t metadata_size_le = JXL_BSWAP16(metadata_size);
  if (cursor + 2 <= output_buffer_size) {
    memcpy(output_buffer + cursor, &metadata_size_le, 2);
    cursor += 2;
  }

  if (cursor + metadata_size <= output_buffer_size) {
    memcpy(output_buffer + cursor, gain_map_metadata.data(), metadata_size);
    cursor += metadata_size;
  }

  uint32_t color_enc_size = compressed_color_encoding.size();
  uint32_t color_enc_size_le = JXL_BSWAP32(color_enc_size);
  if (cursor + 4 <= output_buffer_size) {
    memcpy(output_buffer + cursor, &color_enc_size_le, 4);
    cursor += 4;
  }

  if (cursor + color_enc_size <= output_buffer_size) {
    memcpy(output_buffer + cursor, compressed_color_encoding.data(),
           color_enc_size);
    cursor += color_enc_size;
  }

  uint32_t icc_size = compressed_icc.size();
  uint32_t icc_size_le = JXL_BSWAP32(icc_size);
  if (cursor + 4 <= output_buffer_size) {
    memcpy(output_buffer + cursor, &icc_size_le, 4);
    cursor += 4;
  }

  if (cursor + icc_size <= output_buffer_size) {
    memcpy(output_buffer + cursor, compressed_icc.data(), icc_size);
    cursor += icc_size;
  }

  if (cursor + gain_map_size <= output_buffer_size) {
    memcpy(output_buffer + cursor, gain_map.data(), gain_map_size);
    cursor += gain_map_size;
  }

  if (bytes_written != nullptr) *bytes_written = cursor;
  return cursor == output_buffer_size ? JXL_TRUE : JXL_FALSE;
}

JXL_BOOL JxlGainMapGetBufferSizes(JxlMemoryManager* memory_manager,
                                  JxlGainMapBundle* map_bundle,
                                  const uint8_t* input_buffer,
                                  const size_t input_buffer_size) {
  if (input_buffer == nullptr || input_buffer_size < 1 + 2 + 4 + 4) {
    return JXL_FALSE;
  }

  size_t cursor = 0;

  uint8_t jhgm_version = input_buffer[cursor];
  cursor += 1;

  // Read the gain_map_metadata_size (2 bytes, needs endian swap)
  uint16_t gain_map_metadata_size;
  memcpy(&gain_map_metadata_size, input_buffer + cursor, 2);
  gain_map_metadata_size = JXL_BSWAP16(gain_map_metadata_size);
  cursor += 2;

  if (input_buffer_size < cursor + gain_map_metadata_size + 4 + 4) {
    return JXL_FALSE;
  }

  cursor += gain_map_metadata_size;

  // Read compressed_color_encoding size (4 bytes, needs endian swap)
  uint32_t compressed_color_encoding_size;
  memcpy(&compressed_color_encoding_size, input_buffer + cursor, 4);
  compressed_color_encoding_size = JXL_BSWAP32(compressed_color_encoding_size);
  cursor += 4;

  if (input_buffer_size < cursor + compressed_color_encoding_size + 4) {
    return JXL_FALSE;
  }

  cursor += compressed_color_encoding_size;

  // Read compressed_icc size (4 bytes, needs endian swap)
  uint32_t compressed_icc_size;
  memcpy(&compressed_icc_size, input_buffer + cursor, 4);
  compressed_icc_size = JXL_BSWAP32(compressed_icc_size);
  cursor += 4;

  if (input_buffer_size < cursor + compressed_icc_size) {
    return JXL_FALSE;
  }
  jxl::PaddedBytes icc_buffer(memory_manager);
  if (0 < compressed_icc_size) {
    // Decode ICC data to determine its size when uncompressed
    jxl::Span<const uint8_t> icc_span(input_buffer + cursor,
                                      compressed_icc_size);

    jxl::BitReader icc_reader(icc_span);
    jxl::ICCReader icc_decoder(memory_manager);
    JXL_RETURN_IF_ERROR(icc_decoder.Init(&icc_reader, 0UL));
    JXL_RETURN_IF_ERROR(icc_decoder.Process(&icc_reader, &icc_buffer));
    JXL_RETURN_IF_ERROR(icc_reader.Close());
  }
  // Set sizes in the map bundle
  map_bundle->jhgm_version = jhgm_version;
  map_bundle->gain_map_metadata_size = gain_map_metadata_size;
  map_bundle->alt_icc_size = icc_buffer.size();
  map_bundle->gain_map_size = input_buffer_size - cursor - compressed_icc_size;

  return JXL_TRUE;
}

JXL_BOOL JxlGainMapReadBundle(JxlMemoryManager* memory_manager,
                              JxlGainMapBundle* map_bundle,
                              const uint8_t* input_buffer,
                              const size_t input_buffer_size,
                              size_t* bytes_read) {
  if (map_bundle == nullptr || input_buffer == nullptr ||
      input_buffer_size == 0) {
    return JXL_FALSE;
  }

  size_t cursor = 0;
  // Read the version byte
  map_bundle->jhgm_version = input_buffer[cursor];
  cursor += 1;

  // Read and swap gain_map_metadata_size
  uint16_t gain_map_metadata_size_le;
  memcpy(&gain_map_metadata_size_le, input_buffer + cursor, 2);
  uint16_t gain_map_metadata_size = JXL_BSWAP16(gain_map_metadata_size_le);
  cursor += 2;
  // Copy gain_map_metadata
  if (input_buffer_size < cursor + gain_map_metadata_size) return JXL_FALSE;
  memcpy(map_bundle->gain_map_metadata, input_buffer + cursor,
         gain_map_metadata_size);
  cursor += gain_map_metadata_size;

  // Read and swap compressed_color_encoding_size
  uint32_t compressed_color_encoding_size_le;
  memcpy(&compressed_color_encoding_size_le, input_buffer + cursor, 4);
  uint32_t compressed_color_encoding_size =
      JXL_BSWAP32(compressed_color_encoding_size_le);
  cursor += 4;

  // Decode color encoding
  jxl::Span<const uint8_t> color_encoding_span(input_buffer + cursor,
                                               compressed_color_encoding_size);
  jxl::BitReader color_encoding_reader(color_encoding_span);
  jxl::ColorEncoding internal_color_encoding;
  if (compressed_color_encoding_size > 0) {
    if (!jxl::Bundle::Read(&color_encoding_reader, &internal_color_encoding)) {
      return JXL_FALSE;
    }
  }
  JXL_RETURN_IF_ERROR(color_encoding_reader.Close());
  map_bundle->color_encoding = internal_color_encoding.ToExternal();
  cursor += compressed_color_encoding_size;
  // Read and swap compressed_icc_size
  uint32_t compressed_icc_size_le;
  memcpy(&compressed_icc_size_le, input_buffer + cursor, 4);
  uint32_t compressed_icc_size = JXL_BSWAP32(compressed_icc_size_le);
  cursor += 4;

  // Decode ICC
  jxl::Span<const uint8_t> icc_span(input_buffer + cursor, compressed_icc_size);
  jxl::BitReader icc_reader(icc_span);
  jxl::ICCReader icc_decoder(memory_manager);
  jxl::PaddedBytes icc_buffer(memory_manager);
  JXL_RETURN_IF_ERROR(icc_decoder.Init(&icc_reader, 0UL));
  JXL_RETURN_IF_ERROR(icc_decoder.Process(&icc_reader, &icc_buffer));
  if (map_bundle->alt_icc_size == icc_buffer.size()) {
    memcpy(map_bundle->alt_icc, icc_buffer.data(), icc_buffer.size());
  }
  JXL_RETURN_IF_ERROR(icc_reader.Close());
  cursor += compressed_icc_size;

  // Remaining bytes are gain map
  map_bundle->gain_map_size = input_buffer_size - cursor;
  if (input_buffer_size < cursor + map_bundle->gain_map_size) return JXL_FALSE;
  memcpy(map_bundle->gain_map, input_buffer + cursor,
         map_bundle->gain_map_size);
  cursor += map_bundle->gain_map_size;

  if (bytes_read != nullptr) *bytes_read = cursor;
  return JXL_TRUE;
}
