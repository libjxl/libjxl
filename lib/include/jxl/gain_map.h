/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_metadata
 * @{
 * @file gain_map.h
 * @brief Utility functions to manipulate jhgm (gain map) boxes.
 */

#ifndef JXL_GAIN_MAP_H_
#define JXL_GAIN_MAP_H_

#include <jxl/color_encoding.h>
#include <jxl/jxl_gain_map_export.h>
#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/fields.h"

/** Gain map bundle */
typedef struct {
  /** version number */
  uint8_t jhgm_version;
  /** size of the gain map metadata */
  uint16_t gain_map_metadata_size;
  /** pointer to binary blob of gain map metadata (ISO 21496-1) */
  uint8_t* gain_map_metadata;
  /** indicate if it has a color encoding*/
  bool has_color_encoding;
  /** uncompressed color encoding */
  JxlColorEncoding color_encoding;
  /** size of the alt_icc profile */
  uint32_t alt_icc_size;
  /** pointer to the uncompressed icc profile */
  uint8_t* alt_icc;
  /** size of the gain map */
  uint32_t gain_map_size;
  /** pointer to the gain map (a JPEG XL naked codestream) */
  uint8_t* gain_map;
} JxlGainMapBundle;

/**
 * Calculates the total size required to serialize the gain map bundle into a
 * binary buffer. This function accounts for all the necessary space to
 * serialize fields such as gain map metadata, color encoding, compressed ICC
 * profile data, and the gain map itself.
 *
 * @param[in] memory_manager A memory manager.
 * @param[in] map_bundle Pointer to the JxlGainMapBundle containing all
 * necessary data to compute the size.
 * @param[out] bundle_size The size in bytes required to serialize the bundle.
 * @return Whether setting the size was successful.
 */
JXL_GAIN_MAP_EXPORT JXL_BOOL JxlGainMapGetBundleSize(
    JxlMemoryManager* memory_manager, const JxlGainMapBundle* map_bundle,
    size_t* bundle_size);

/**
 * Serializes the gain map bundle into a preallocated buffer. The function
 * ensures that all parts of the bundle such as metadata, color encoding,
 * compressed ICC profile, and the gain map are correctly encoded into the
 * buffer. First call `JxlGainMapGetBundleSize` to get the size needed for
 * the buffer.
 *
 * @param[in] memory_manager A memory manager.
 * @param[in] map_bundle Pointer to the `JxlGainMapBundle` to serialize.
 * @param[out] output_buffer Pointer to the buffer where the serialized data
 * will be written.
 * @param[in] output_buffer_size The size of the output buffer in bytes. Must be
 * large enough to hold the entire serialized data.
 * @param[out] bytes_written The number of bytes written to the output buffer.
 * @return Whether writing the bundle was successful.
 */
JXL_GAIN_MAP_EXPORT JXL_BOOL JxlGainMapWriteBundle(
    JxlMemoryManager* memory_manager, const JxlGainMapBundle* map_bundle,
    uint8_t* output_buffer, size_t output_buffer_size, size_t* bytes_written);

/**
 * Determines the sizes of various components within a gain map bundle from a
 * serialized buffer. This function parses the buffer to extract sets the
 * following fields of `JxlGainMapBundle`:
 *  - gain_map_metadata_size
 *  - alt_icc_size
 *  - gain_map_size
 * allowing buffer allocation for deserialization, preparing the call to
 * `JxlGainMapReadBundle`.
 *
 * @param[in] memory_manager A memory manager.
 * @param[in,out] map_bundle Pointer to the `JxlGainMapBundle` where the sizes
 * will be stored. Must be preallocated.
 * @param[in] input_buffer Pointer to the buffer containing the serialized gain
 * map bundle data.
 * @param[in] input_buffer_size The size of the input buffer in bytes.
 * @return Whether the sizes could be successfully determined.
 */
JXL_GAIN_MAP_EXPORT JXL_BOOL JxlGainMapGetBufferSizes(
    JxlMemoryManager* memory_manager, JxlGainMapBundle* map_bundle,
    const uint8_t* input_buffer, size_t input_buffer_size);

/**
 * Deserializes a gain map bundle from a given buffer, populating the provided
 * `JxlGainMapBundle` structure with data extracted from the buffer. Assumes
 * that the buffer contains a valid serialized gain map bundle and that the
 * `map_bundle` has preallocated memory for the pointers
 *  - gain_map_metadata
 *  - alt_icc
 *  - gain
 * based on sizes obtained from `JxlGainMapGetBufferSizes`.
 *
 * @param[in] memory_manager A memory manager.
 * @param[in,out] map_bundle Pointer to a preallocated `JxlGainMapBundle` where
 * the deserialized data will be stored.
 * @param[in] input_buffer Pointer to the buffer containing the serialized gain
 * map bundle data.
 * @param[in] input_buffer_size The size of the input buffer in bytes.
 * @param[out] bytes_read The number of bytes read from the input buffer.
 * @return Whether reading the bundle was successful.
 */
JXL_GAIN_MAP_EXPORT JXL_BOOL JxlGainMapReadBundle(
    JxlMemoryManager* memory_manager, JxlGainMapBundle* map_bundle,
    const uint8_t* input_buffer, size_t input_buffer_size, size_t* bytes_read);

#endif /* JXL_GAIN_MAP_H_ */

/** @} */