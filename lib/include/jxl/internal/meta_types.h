// only called meta_types because the C-API Generator complained about the name

#include "default_type_builder.h"

/** Data type for the sample values per channel per pixel.
 */
Enum(JxlDataType,
  /** Use 32-bit single-precision floating point values, with range 0.0-1.0
   * (within gamut, may go outside this range for wide color gamut). Floating
   * point output, either JXL_TYPE_FLOAT or JXL_TYPE_FLOAT16, is recommended
   * for HDR and wide gamut images when color profile conversion is required. */
  EnumDefinedValue(JXL_TYPE_FLOAT, 0)

  /** Use 1-bit packed in uint8_t, first pixel in LSB, padded to uint8_t per
   * row.
   * TODO(lode): support first in MSB, other padding.
   */
  EnumValue(JXL_TYPE_BOOLEAN)

  /** Use type uint8_t. May clip wide color gamut data.
   */
  EnumValue(JXL_TYPE_UINT8)

  /** Use type uint16_t. May clip wide color gamut data.
   */
  EnumValue(JXL_TYPE_UINT16)

  /** Use type uint32_t. May clip wide color gamut data.
   */
  EnumValue(JXL_TYPE_UINT32)

  /** Use 16-bit IEEE 754 half-precision floating point values */
  EnumValue(JXL_TYPE_FLOAT16)
)

/** Ordering of multi-byte data.
 */
Enum(JxlEndianness,
  /** Use the endianness of the system, either little endian or big endian,
   * without forcing either specific endianness. Do not use if pixel data
   * should be exported to a well defined format.
   */
  EnumDefinedValue(JXL_NATIVE_ENDIAN, 0)
  /** Force little endian */
  EnumDefinedValue(JXL_LITTLE_ENDIAN, 1)
  /** Force big endian */
  EnumDefinedValue(JXL_BIG_ENDIAN, 2)
)

/** Data type for the sample values per channel per pixel for the output buffer
 * for pixels. This is not necessarily the same as the data type encoded in the
 * codestream. The channels are interleaved per pixel. The pixels are
 * organized row by row, left to right, top to bottom.
 * TODO(lode): implement padding / alignment (row stride)
 * TODO(lode): support different channel orders if needed (RGB, BGR, ...)
 */
Struct(JxlPixelFormat,
  /** Amount of channels available in a pixel buffer.
   * 1: single-channel data, e.g. grayscale or a single extra channel
   * 2: single-channel + alpha
   * 3: trichromatic, e.g. RGB
   * 4: trichromatic + alpha
   * TODO(lode): this needs finetuning. It is not yet defined how the user
   * chooses output color space. CMYK+alpha needs 5 channels.
   */
  Member(uint32_t, num_channels)

  /** Data type of each channel.
   */
  Member(JxlDataType, data_type)

  /** Whether multi-byte data types are represented in big endian or little
   * endian format. This applies to JXL_TYPE_UINT16, JXL_TYPE_UINT32
   * and JXL_TYPE_FLOAT.
   */
  Member(JxlEndianness, endianness)

  /** Align scanlines to a multiple of align bytes, or 0 to require no
   * alignment at all (which has the same effect as value 1)
   */
  Member(size_t, align)
)

#include "clear_type_builder.h"

