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

#if !defined(JXL_TYPES_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
#	ifndef DOC_GENERATOR
#		define JXL_TYPES_H_
#		include <stddef.h>
#		include <stdint.h>
#		define CLEAR_GENERATOR true
#		include "typebuilder/type_generator.h"
#	else
	ESCAPE(#ifndef JXL_TYPES_H_)
	ESCAPE(#define JXL_TYPES_H_)
	ESCAPE(#include <stddef.h>)
	ESCAPE(#include <stdint.h>)
#	endif
#endif

EXTERN_C(
	/**
	 * A portable @c bool replacement.
	 *
	 * ::JXL_BOOL is a "documentation" type: actually it is @c int, but in API it
	 * denotes a type, whose only values are ::JXL_TRUE and ::JXL_FALSE.
	 */
	#ifndef DOC_GENERATOR
		#define JXL_BOOL int
	#else
		ESCAPE(#define JXL_BOOL int)
	#endif

	/** Portable @c true replacement. */
	#ifndef DOC_GENERATOR
		#define JXL_TRUE 1
	#else
		ESCAPE(#define JXL_TRUE 1)
	#endif

	/** Portable @c false replacement. */
	#ifndef DOC_GENERATOR
		#define JXL_FALSE 0
	#else
		ESCAPE(#define JXL_FALSE 0)
	#endif

	/** Data type for the sample values per channel per pixel.
	 */
	EnumDef(JxlDataType,
		/** Use 32-bit single-precision floating point values, with range 0.0-1.0
		 * (within gamut, may go outside this range for wide color gamut). Floating
		 * point output, either JXL_TYPE_FLOAT or JXL_TYPE_FLOAT16, is recommended
		 * for HDR and wide gamut images when color profile conversion is required. */
		DefinedValue(JXL_TYPE_FLOAT, 0)

		/** Use 1-bit packed in uint8_t, first pixel in LSB, padded to uint8_t per
		 * row.
		 * TODO(lode): support first in MSB, other padding.
		 */
		Value(JXL_TYPE_BOOLEAN)

		/** Use type uint8_t. May clip wide color gamut data.
		 */
		Value(JXL_TYPE_UINT8)

		/** Use type uint16_t. May clip wide color gamut data.
		 */
		Value(JXL_TYPE_UINT16)

		/** Use type uint32_t. May clip wide color gamut data.
		 */
		Value(JXL_TYPE_UINT32)

		/** Use 16-bit IEEE 754 half-precision floating point values */
		Value(JXL_TYPE_FLOAT16)
	 )

	/** Ordering of multi-byte data.
	 */
	EnumDef(JxlEndianness,
		/** Use the endianness of the system, either little endian or big endian,
		 * without forcing either specific endianness. Do not use if pixel data
		 * should be exported to a well defined format.
		 */
		DefinedValue(JXL_NATIVE_ENDIAN, 0)
		/** Force little endian */
		DefinedValue(JXL_LITTLE_ENDIAN, 1)
		/** Force big endian */
		DefinedValue(JXL_BIG_ENDIAN, 2)
	)

	/** Data type for the sample values per channel per pixel for the output buffer
	 * for pixels. This is not necessarily the same as the data type encoded in the
	 * codestream. The channels are interleaved per pixel. The pixels are
	 * organized row by row, left to right, top to bottom.
	 * TODO(lode): implement padding / alignment (row stride)
	 * TODO(lode): support different channel orders if needed (RGB, BGR, ...)
	 */
	StructDef(JxlPixelFormat,
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

	RawCode(
		/** Data type holding the 4-character type name of an ISOBMFF box.
		 */
		typedef char JxlBoxType[4]; 
	)
)

#if CLEAR_GENERATOR
#	undef CLEAR_GENERATOR
#	include "typebuilder/clear_generator.h"
#endif

#endif /* JXL_TYPES_H_ */

/** @}*/
