// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <jxl/color_encoding.h>
#include <jxl/jxl_gain_map_export.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/fields.h"
using ::jxl::Visitor;

/** Gain map bundle */
typedef struct {
  /** version number */
  uint8_t jhgm_version;
  /** size of the gain map metadata */
  uint16_t gain_map_metadata_size;
  /** pointer to binary blob of gain map metadata (ISO 21496-1) */
  uint8_t *gain_map_metadata;
  /** indicate if it has a color encoding*/
  bool has_color_encoding;
  /** uncompressed color encoding */
  JxlColorEncoding color_encoding;
  /** size of the alt_icc profile */
  uint32_t alt_icc_size;
  /** pointer to the uncompressed icc profile */
  uint8_t *alt_icc;
  /** size of the gain map */
  uint32_t gain_map_size;
  /** pointer to the gain map (a JPEG XL naked codestream) */
  uint8_t *gain_map;
} JxlGainMapBundle;

/**
 * Calculates the total size required to serialize the gain map bundle into a
 * binary buffer. This function accounts for all the necessary space to
 * serialize fields such as gain map metadata, color encoding, compressed ICC
 * profile data, and the gain map itself.
 *
 * @param map_bundle Pointer to the JxlGainMapBundle containing all necessary
 * data to compute the size.
 * @return The size in bytes required to serialize the bundle. Returns 0 if the
 * input bundle is null.
 */
JXL_GAIN_MAP_EXPORT size_t
JxlGainMapGetBundleSize(const JxlGainMapBundle *map_bundle);

/**
 * Serializes the gain map bundle into a preallocated buffer. The function
 * ensures that all parts of the bundle such as metadata, color encoding,
 * compressed ICC profile, and the gain map are correctly encoded into the
 * buffer. First call @ref JxlGainMapGetBundleSize to get the size needed for
 * the buffer.
 *
 * @param map_bundle Pointer to the JxlGainMapBundle to serialize.
 * @param output_buffer Pointer to the buffer where the serialized data will be
 * written.
 * @param output_buffer_size The size of the output buffer in bytes. Must be
 * large enough to hold the entire serialized data.
 * @return The number of bytes written to the output buffer. Returns 0 if the
 * serialization fails due to insufficient buffer size or null inputs.
 */
JXL_GAIN_MAP_EXPORT size_t
JxlGainMapWriteBundle(const JxlGainMapBundle *map_bundle,
                      uint8_t *output_buffer, const size_t output_buffer_size);

/**
 * Determines the sizes of various components within a gain map bundle from a
 * serialized buffer. This function parses the buffer to extract sets the
 * following fields of @ref JxlGainMapBundle:
 *  - gain_map_metadata_size
 *  - alt_icc_size
 *  - gain_map_size
 * allowing buffer allocation for deserialization, preparing the call to @ref
 * JxlGainMapReadBundle.
 *
 * @param map_bundle Pointer to the JxlGainMapBundle where the sizes will be
 * stored. Must be preallocated.
 * @param input_buffer Pointer to the buffer containing the serialized gain map
 * bundle data.
 * @param input_buffer_size The size of the input buffer in bytes.
 */
JXL_GAIN_MAP_EXPORT void JxlGainMapGetBufferSizes(
    JxlGainMapBundle *map_bundle, const uint8_t *input_buffer,
    const size_t input_buffer_size);

/**
 * Deserializes a gain map bundle from a given buffer, populating the provided
 * JxlGainMapBundle structure with data extracted from the buffer. Assumes that
 * the buffer contains a valid serialized gain map bundle and that the
 * `map_bundle` has preallocated memory for the pointers
 *  - gain_map_metadata
 *  - alt_icc
 *  - gain
 * based on sizes obtained from @ref JxlGainMapGetBufferSizes.
 *
 * @param map_bundle Pointer to a preallocated JxlGainMapBundle where the
 * deserialized data will be stored.
 * @param input_buffer Pointer to the buffer containing the serialized gain map
 * bundle data.
 * @param input_buffer_size The size of the input buffer in bytes.
 * @return The number of bytes read from the input buffer. Returns 0 if
 * deserialization fails due to invalid data or buffer inconsistencies.
 */
JXL_GAIN_MAP_EXPORT size_t JxlGainMapReadBundle(JxlGainMapBundle *map_bundle,
                                                const uint8_t *input_buffer,
                                                const size_t input_buffer_size);