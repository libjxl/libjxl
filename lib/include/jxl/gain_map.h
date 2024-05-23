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
  /** size of the gain map metadata*/
  uint16_t gain_map_metadata_size;
  /** pointer to binary blob of gain map metadata (ISO 21496-1) */
  uint8_t *gain_map_metadata;
  /** uncompressed color encoding */
  JxlColorEncoding color_encoding;
  /** pointer to the uncompressed icc profile */
  uint8_t *alt_icc;
  /** size of the gain map */
  uint8_t gain_map_size;
  /** pointer to the gain map (a JPEG XL naked codestream) */
  uint8_t *gain_map;
} JxlGainMapBundle;

// Function to get the size of the gain map bundle
JXL_GAIN_MAP_EXPORT size_t
JxlGainMapGetBundleSize(const JxlGainMapBundle *map_bundle);

// Function to write the map bundle
JXL_GAIN_MAP_EXPORT size_t
JxlGainMapWriteBundle(const JxlGainMapBundle *map_bundle,
                      uint8_t *output_buffer, const size_t output_buffer_size);

// Function to read the map bundle from a buffer
JXL_GAIN_MAP_EXPORT size_t JxlGainMapReadBundle(JxlGainMapBundle *map_bundle,
                                                const uint8_t *input_buffer,
                                                const size_t input_buffer_size);