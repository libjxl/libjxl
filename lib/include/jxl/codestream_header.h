/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_common
 * @{
 * @file codestream_header.h
 * @brief Definitions of structs and enums for the metadata from the JPEG XL
 * codestream headers (signature, metadata, preview dimensions, ...), excluding
 * color encoding which is in color_encoding.h.
 */

#if !defined(JXL_CODESTREAM_HEADER_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
#	ifndef DOC_GENERATOR
#		define JXL_CODESTREAM_HEADER_H_
#		include <stddef.h>
#		include <stdint.h>
#		include "jxl/color_encoding.h"
#		include "jxl/types.h"
#		define CLEAR_GENERATOR true
#		include "typebuilder/type_generator.h"
#	else
		ESCAPE(#ifndef JXL_CODESTREAM_HEADER_H_)
		ESCAPE(#define JXL_CODESTREAM_HEADER_H_)
		ESCAPE(#include <stddef.h>)
		ESCAPE(#include <stdint.h>)
		ESCAPE(#include "jxl/color_encoding.h")
		ESCAPE(#include "jxl/types.h")
#	endif
#endif

EXTERN_C(
	/** Image orientation metadata.
	 * Values 1..8 match the EXIF definitions.
	 * The name indicates the operation to perform to transform from the encoded
	 * image to the display image.
	 */
	EnumDef(JxlOrientation,
		DefinedValue(JXL_ORIENT_IDENTITY, 1)
		DefinedValue(JXL_ORIENT_FLIP_HORIZONTAL, 2)
		DefinedValue(JXL_ORIENT_ROTATE_180, 3)
		DefinedValue(JXL_ORIENT_FLIP_VERTICAL, 4)
		DefinedValue(JXL_ORIENT_TRANSPOSE, 5)
		DefinedValue(JXL_ORIENT_ROTATE_90_CW, 6)
		DefinedValue(JXL_ORIENT_ANTI_TRANSPOSE, 7)
		DefinedValue(JXL_ORIENT_ROTATE_90_CCW, 8)

	)

	/** Given type of an extra channel.
	 */
	EnumDef(JxlExtraChannelType,
		Value(JXL_CHANNEL_ALPHA)
		Value(JXL_CHANNEL_DEPTH)
		Value(JXL_CHANNEL_SPOT_COLOR)
		Value(JXL_CHANNEL_SELECTION_MASK)
		Value(JXL_CHANNEL_BLACK)
		Value(JXL_CHANNEL_CFA)
		Value(JXL_CHANNEL_THERMAL)
		Value(JXL_CHANNEL_RESERVED0)
		Value(JXL_CHANNEL_RESERVED1)
		Value(JXL_CHANNEL_RESERVED2)
		Value(JXL_CHANNEL_RESERVED3)
		Value(JXL_CHANNEL_RESERVED4)
		Value(JXL_CHANNEL_RESERVED5)
		Value(JXL_CHANNEL_RESERVED6)
		Value(JXL_CHANNEL_RESERVED7)
		Value(JXL_CHANNEL_UNKNOWN)
		Value(JXL_CHANNEL_OPTIONAL)
	)


	/** The codestream preview header */
	StructDef(JxlPreviewHeader,
		/** Preview width in pixels */
		Member(uint32_t, xsize)

		/** Preview height in pixels */
		Member(uint32_t, ysize)
	)

	/** The intrinsic size header */
	StructDef(JxlIntrinsicSizeHeader,
		/** Intrinsic width in pixels */
		Member(uint32_t, xsize)

		/** Intrinsic height in pixels */
		Member(uint32_t, ysize)
	)

	/** The codestream animation header, optionally present in the beginning of
	 * the codestream, and if it is it applies to all animation frames, unlike
	 * JxlFrameHeader which applies to an individual frame.
	 */
	StructDef(JxlAnimationHeader,
		/** Numerator of ticks per second of a single animation frame time unit */
		Member(uint32_t, tps_numerator)

		/** Denominator of ticks per second of a single animation frame time unit */
		Member(uint32_t, tps_denominator)

		/** Amount of animation loops, or 0 to repeat infinitely */
		Member(uint32_t, num_loops)

		/** Whether animation time codes are present at animation frames in the
		 * codestream */
		Member(JXL_BOOL, have_timecodes)
	)

	/** Basic image information. This information is available from the file
	 * signature and first part of the codestream header.
	 */
	StructDef(JxlBasicInfo,

		/* TODO(lode): need additional fields for (transcoded) JPEG? For reusable
		* fields orientation must be read from Exif APP1. For has_icc_profile: must
		* look up where ICC profile is guaranteed to be in a JPEG file to be able to
		* indicate this. */

		/* TODO(lode): make struct packed, and/or make this opaque struct with getter
		* functions (still separate struct from opaque decoder) */

		/** Whether the codestream is embedded in the container format. If true,
		 * metadata information and extensions may be available in addition to the
		 * codestream.
		 */
		Member(JXL_BOOL, have_container)

		/** Width of the image in pixels, before applying orientation.
		 */
		Member(uint32_t, xsize)

		/** Height of the image in pixels, before applying orientation.
		 */
		Member(uint32_t, ysize)

		/** Original image color channel bit depth.
		 */
		Member(uint32_t, bits_per_sample)

		/** Original image color channel floating point exponent bits, or 0 if they
		 * are unsigned integer. For example, if the original data is half-precision
		 * (binary16) floating point, bits_per_sample is 16 and
		 * exponent_bits_per_sample is 5, and so on for other floating point
		 * precisions.
		 */
		Member(uint32_t, exponent_bits_per_sample)

		/** Upper bound on the intensity level present in the image in nits. For
		 * unsigned integer pixel encodings, this is the brightness of the largest
		 * representable value. The image does not necessarily contain a pixel
		 * actually this bright. An encoder is allowed to set 255 for SDR images
		 * without computing a histogram.
		 */
		Member(float, intensity_target)

		/** Lower bound on the intensity level present in the image. This may be
		 * loose, i.e. lower than the actual darkest pixel. When tone mapping, a
		 * decoder will map [min_nits, intensity_target] to the display range.
		 */
		Member(float, min_nits)

		/** See the description of @see linear_below.
		 */
		Member(JXL_BOOL, relative_to_max_display)

		/** The tone mapping will leave unchanged (linear mapping) any pixels whose
		 * brightness is strictly below this. The interpretation depends on
		 * relative_to_max_display. If true, this is a ratio [0, 1] of the maximum
		 * display brightness [nits], otherwise an absolute brightness [nits].
		 */
		Member(float, linear_below)

		/** Whether the data in the codestream is encoded in the original color
		 * profile that is attached to the codestream metadata header, or is
		 * encoded in an internally supported absolute color space (which the decoder
		 * can always convert to linear or non-linear sRGB or to XYB). If the original
		 * profile is used, the decoder outputs pixel data in the color space matching
		 * that profile, but doesn't convert it to any other color space. If the
		 * original profile is not used, the decoder only outputs the data as sRGB
		 * (linear if outputting to floating point, nonlinear with standard sRGB
		 * transfer function if outputting to unsigned integers) but will not convert
		 * it to to the original color profile. The decoder also does not convert to
		 * the target display color profile, but instead will always indicate which
		 * color profile the returned pixel data is encoded in when using @see
		 * JXL_COLOR_PROFILE_TARGET_DATA so that a CMS can be used to convert the
		 * data.
		 */
		Member(JXL_BOOL, uses_original_profile)

		/** Indicates a preview image exists near the beginning of the codestream.
		 * The preview itself or its dimensions are not included in the basic info.
		 */
		Member(JXL_BOOL, have_preview)

		/** Indicates animation frames exist in the codestream. The animation
		 * information is not included in the basic info.
		 */
		Member(JXL_BOOL, have_animation)

		/** Image orientation, value 1-8 matching the values used by JEITA CP-3451C
		 * (Exif version 2.3).
		 */
		Member(JxlOrientation, orientation)

		/** Number of color channels encoded in the image, this is either 1 for
		 * grayscale data, or 3 for colored data. This count does not include
		 * the alpha channel or other extra channels. To check presence of an alpha
		 * channel, such as in the case of RGBA color, check alpha_bits != 0.
		 * If and only if this is 1, the JxlColorSpace in the JxlColorEncoding is
		 * JXL_COLOR_SPACE_GRAY.
		 */
		Member(uint32_t, num_color_channels)

		/** Number of additional image channels. This includes the main alpha channel,
		 * but can also include additional channels such as depth, additional alpha
		 * channels, spot colors, and so on. Information about the extra channels
		 * can be queried with JxlDecoderGetExtraChannelInfo. The main alpha channel,
		 * if it exists, also has its information available in the alpha_bits,
		 * alpha_exponent_bits and alpha_premultiplied fields in this JxlBasicInfo.
		 */
		Member(uint32_t, num_extra_channels)

		/** Bit depth of the encoded alpha channel, or 0 if there is no alpha channel.
		 * If present, matches the alpha_bits value of the JxlExtraChannelInfo
		 * associated with this alpha channel.
		 */
		Member(uint32_t, alpha_bits)

		/** Alpha channel floating point exponent bits, or 0 if they are unsigned. If
		 * present, matches the alpha_bits value of the JxlExtraChannelInfo associated
		 * with this alpha channel. integer.
		 */
		Member(uint32_t, alpha_exponent_bits)

		/** Whether the alpha channel is premultiplied. Only used if there is a main
		 * alpha channel. Matches the alpha_premultiplied value of the
		 * JxlExtraChannelInfo associated with this alpha channel.
		 */
		Member(JXL_BOOL, alpha_premultiplied)

		/** Dimensions of encoded preview image, only used if have_preview is
		 * JXL_TRUE.
		 */
		Member(JxlPreviewHeader, preview)

		/** Animation header with global animation properties for all frames, only
		 * used if have_animation is JXL_TRUE.
		 */
		Member(JxlAnimationHeader, animation)

		/** Intrinsic width of the image.
		 * The intrinsic size can be different from the actual size in pixels
		 * (as given by xsize and ysize) and it denotes the recommended dimensions
		 * for displaying the image, i.e. applications are advised to resample the
		 * decoded image to the intrinsic dimensions.
		 */
		Member(uint32_t, intrinsic_xsize)

		/** Intrinsic heigth of the image.
		 * The intrinsic size can be different from the actual size in pixels
		 * (as given by xsize and ysize) and it denotes the recommended dimensions
		 * for displaying the image, i.e. applications are advised to resample the
		 * decoded image to the intrinsic dimensions.
		 */
		Member(uint32_t, intrinsic_ysize)

		/** Padding for forwards-compatibility, in case more fields are exposed
		 * in a future version of the library.
		 */
		FixedArray(uint8_t, padding, 100)
	)

	/** Information for a single extra channel.
	 */
	StructDef(JxlExtraChannelInfo,
		/** Given type of an extra channel.
		 */
		Member(JxlExtraChannelType, type)

		/** Total bits per sample for this channel.
		 */
		Member(uint32_t, bits_per_sample)

		/** Floating point exponent bits per channel, or 0 if they are unsigned
		 * integer.
		 */
		Member(uint32_t, exponent_bits_per_sample)

		/** The exponent the channel is downsampled by on each axis.
		 * TODO(lode): expand this comment to match the JPEG XL specification,
		 * specify how to upscale, how to round the size computation, and to which
		 * extra channels this field applies.
		 */
		Member(uint32_t, dim_shift)

		/** Length of the extra channel name in bytes, or 0 if no name.
		 * Excludes null termination character.
		 */
		Member(uint32_t, name_length)

		/** Whether alpha channel uses premultiplied alpha. Only applicable if
		 * type is JXL_CHANNEL_ALPHA.
		 */
		Member(JXL_BOOL, alpha_premultiplied)

		/** Spot color of the current spot channel in linear RGBA. Only applicable if
		 * type is JXL_CHANNEL_SPOT_COLOR.
		 */
		FixedArray(float, spot_color, 4)

		/** Only applicable if type is JXL_CHANNEL_CFA.
		 * TODO(lode): add comment about the meaning of this field.
		 */
		Member(uint32_t, cfa_channel)
	)

	/* TODO(lode): add API to get the codestream header extensions. */
	/** Extensions in the codestream header. */
	StructDef(JxlHeaderExtensions,
		/** Extension bits. */
		Member(uint64_t, extensions)
	)

	/** Frame blend modes.
	 * When decoding, if coalescing is enabled (default), this can be ignored.
	 */
	EnumDef(JxlBlendMode,
		DefinedValue(JXL_BLEND_REPLACE, 0)
		DefinedValue(JXL_BLEND_ADD, 1)
		DefinedValue(JXL_BLEND_BLEND, 2)
		DefinedValue(JXL_BLEND_MULADD, 3)
		DefinedValue(JXL_BLEND_MUL, 4)
	)

	/** The information about blending the color channels or a single extra channel.
	 * When decoding, if coalescing is enabled (default), this can be ignored and
	 * the blend mode is considered to be JXL_BLEND_REPLACE.
	 * When encoding, these settings apply to the pixel data given to the encoder.
	 */
	StructDef(JxlBlendInfo,
		/** Blend mode.
		 */
		Member(JxlBlendMode, blendmode)
		/** Reference frame ID to use as the 'bottom' layer (0-3).
		 */
		Member(uint32_t, source)
		/** Which extra channel to use as the 'alpha' channel for blend modes
		 * JXL_BLEND_BLEND and JXL_BLEND_MULADD.
		 */
		Member(uint32_t, alpha)
		/** Clamp values to [0,1] for the purpose of blending.
		 */
		Member(JXL_BOOL, clamp)
	)

	/** The information about layers.
	 * When decoding, if coalescing is enabled (default), this can be ignored.
	 * When encoding, these settings apply to the pixel data given to the encoder,
	 * the encoder could choose an internal representation that differs.
	 */
	StructDef(JxlLayerInfo,
		/** Whether cropping is applied for this frame. When decoding, if false,
		 * crop_x0 and crop_y0 are set to zero, and xsize and ysize to the main
		 * image dimensions. When encoding and this is false, those fields are
		 * ignored. When decoding, if coalescing is enabled (default), this is always
		 * false, regardless of the internal encoding in the JPEG XL codestream.
		 */
		Member(JXL_BOOL, have_crop)

		/** Horizontal offset of the frame (can be negative).
		 */
		Member(int32_t, crop_x0)

		/** Vertical offset of the frame (can be negative).
		 */
		Member(int32_t, crop_y0)

		/** Width of the frame (number of columns).
		 */
		Member(uint32_t, xsize)

		/** Height of the frame (number of rows).
		 */
		Member(uint32_t, ysize)

		/** The blending info for the color channels. Blending info for extra channels
		 * has to be retrieved separately using JxlDecoderGetExtraChannelBlendInfo.
		 */
		Member(JxlBlendInfo, blend_info)

		/** After blending, save the frame as reference frame with this ID (0-3).
		 * Special case: if the frame duration is nonzero, ID 0 means "will not be
		 * referenced in the future". This value is not used for the last frame.
		 */
		Member(uint32_t, save_as_reference)
	)

	/** The header of one displayed frame or non-coalesced layer. */
	StructDef(JxlFrameHeader,
		/** How long to wait after rendering in ticks. The duration in seconds of a
		 * tick is given by tps_numerator and tps_denominator in JxlAnimationHeader.
		 */
		Member(uint32_t, duration)

		/** SMPTE timecode of the current frame in form 0xHHMMSSFF, or 0. The bits are
		 * interpreted from most-significant to least-significant as hour, minute,
		 * second, and frame. If timecode is nonzero, it is strictly larger than that
		 * of a previous frame with nonzero duration. These values are only available
		 * if have_timecodes in JxlAnimationHeader is JXL_TRUE.
		 * This value is only used if have_timecodes in JxlAnimationHeader is
		 * JXL_TRUE.
		 */
		Member(uint32_t, timecode)

		/** Length of the frame name in bytes, or 0 if no name.
		 * Excludes null termination character. This value is set by the decoder.
		 * For the encoder, this value is ignored and @ref JxlEncoderSetFrameName is
		 * used instead to set the name and the length.
		 */
		Member(uint32_t, name_length)

		/** Indicates this is the last animation frame. This value is set by the
		 * decoder to indicate no further frames follow. For the encoder, it is not
		 * required to set this value and it is ignored, @ref JxlEncoderCloseFrames is
		 * used to indicate the last frame to the encoder instead.
		 */
		Member(JXL_BOOL, is_last)

		/** Information about the layer in case of no coalescing.
		 */
		Member(JxlLayerInfo, layer_info)
	)
)

#if CLEAR_GENERATOR
#	include "typebuilder/clear_generator.h"
#	undef CLEAR_GENERATOR
#endif

#ifdef DOC_GENERATOR
    ESCAPE(#endif)
#endif

#endif /* JXL_CODESTREAM_HEADER_H_ */

/** @}*/
