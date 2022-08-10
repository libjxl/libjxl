/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_common
 * @{
 * @file types.h
 * @brief Data types for the JPEG XL API, for both encoding and decoding.
 */

#ifndef JXL_TYPES_H_
#define JXL_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#include "jxl/jxl_export.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * A portable @c bool replacement.
 *
 * ::JXL_BOOL is a "documentation" type: actually it is @c int, but in API it
 * denotes a type, whose only values are ::JXL_TRUE and ::JXL_FALSE.
 */
#define JXL_BOOL int
/** Portable @c true replacement. */
#define JXL_TRUE 1
/** Portable @c false replacement. */
#define JXL_FALSE 0

/** Data type for the sample values per channel per pixel.
 */
typedef enum {
  /** Use 32-bit single-precision floating point values, with range 0.0-1.0
   * (within gamut, may go outside this range for wide color gamut). Floating
   * point output, either JXL_TYPE_FLOAT or JXL_TYPE_FLOAT16, is recommended
   * for HDR and wide gamut images when color profile conversion is required. */
  JXL_TYPE_FLOAT = 0,

  /** Use type uint8_t. May clip wide color gamut data.
   */
  JXL_TYPE_UINT8 = 2,

  /** Use type uint16_t. May clip wide color gamut data.
   */
  JXL_TYPE_UINT16 = 3,

  /** Use 16-bit IEEE 754 half-precision floating point values */
  JXL_TYPE_FLOAT16 = 5,
} JxlDataType;

/* DEPRECATED: bit-packed 1-bit data type. Use JXL_TYPE_UINT8 instead.
 */
JXL_DEPRECATED static const int JXL_TYPE_BOOLEAN = 1;

/* DEPRECATED: uint32_t data type. Use JXL_TYPE_FLOAT instead.
 */
JXL_DEPRECATED static const int JXL_TYPE_UINT32 = 4;

/** Ordering of multi-byte data.
 */
typedef enum {
  /** Use the endianness of the system, either little endian or big endian,
   * without forcing either specific endianness. Do not use if pixel data
   * should be exported to a well defined format.
   */
  JXL_NATIVE_ENDIAN = 0,
  /** Force little endian */
  JXL_LITTLE_ENDIAN = 1,
  /** Force big endian */
  JXL_BIG_ENDIAN = 2,
} JxlEndianness;

/** Data type for the sample values per channel per pixel for the output buffer
 * for pixels. This is not necessarily the same as the data type encoded in the
 * codestream. The channels are interleaved per pixel. The pixels are
 * organized row by row, left to right, top to bottom.
 * TODO(lode): implement padding / alignment (row stride)
 * TODO(lode): support different channel orders if needed (RGB, BGR, ...)
 */
typedef struct {
  /** Amount of channels available in a pixel buffer.
   * 1: single-channel data, e.g. grayscale or a single extra channel
   * 2: single-channel + alpha
   * 3: trichromatic, e.g. RGB
   * 4: trichromatic + alpha
   * TODO(lode): this needs finetuning. It is not yet defined how the user
   * chooses output color space. CMYK+alpha needs 5 channels.
   */
  uint32_t num_channels;

  /** Data type of each channel.
   */
  JxlDataType data_type;

  /** Whether multi-byte data types are represented in big endian or little
   * endian format. This applies to JXL_TYPE_UINT16, JXL_TYPE_UINT32
   * and JXL_TYPE_FLOAT.
   */
  JxlEndianness endianness;

  /** Align scanlines to a multiple of align bytes, or 0 to require no
   * alignment at all (which has the same effect as value 1)
   */
  size_t align;
} JxlPixelFormat;

/** Data type holding the 4-character type name of an ISOBMFF box.
 */
typedef char JxlBoxType[4];

/** Types of progressive detail.
 * Setting a progressive detail with value N implies all progressive details
 * with smaller or equal value. Currently only the following level of
 * progressive detail is implemented:
 *  - kDC (which implies kFrames)
 *  - kLastPasses (which implies kDC and kFrames)
 *  - kPasses (which implies kLastPasses, kDC and kFrames)
 */
typedef enum {
  // after completed kRegularFrames
  kFrames = 0,
  // after completed DC (1:8)
  kDC = 1,
  // after completed AC passes that are the last pass for their resolution
  // target.
  kLastPasses = 2,
  // after completed AC passes that are not the last pass for their resolution
  // target.
  kPasses = 3,
  // during DC frame when lower resolution are completed (1:32, 1:16)
  kDCProgressive = 4,
  // after completed groups
  kDCGroups = 5,
  // after completed groups
  kGroups = 6,
} JxlProgressiveDetail;

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JXL_TYPES_H_ */

/** @}*/
