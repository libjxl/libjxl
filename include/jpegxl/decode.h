/* Copyright (c) the JPEG XL Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef JPEGXL_DECODE_H_
#define JPEGXL_DECODE_H_

#include <stddef.h>
#include <stdint.h>

#include "jpegxl/jpegxl_export.h"
#include "jpegxl/memory_manager.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Decoder library version.
 *
 * @returns the decoder library version as an integer:
 * MAJOR_VERSION * 1000000 + MINOR_VERSION * 1000 + PATCH_VERSION. For example,
 * version 1.2.3 would return 1002003.
 */
JPEGXL_EXPORT uint32_t JpegxlDecoderVersion(void);

enum JpegxlSignature {
  /** Not enough bytes were passed to determine if a valid signature was found.
   */
  JPEGXL_SIG_NOT_ENOUGH_BYTES = 0,

  /** No valid JPEGXL header was found. */
  JPEGXL_SIG_INVALID = 1,

  /** A valid JPEG XL image signature was found, which could be a JPEG XL
   * codestream, a transcoded JPEG image, or a JPEG XL container. This also
   * includes the case of a JPEG codestream, which would preferably also be
   * decoded using this decoder in case the codestream contains JPEG XL
   * extensions (marker segments).
   */
  JPEGXL_SIG_VALID = 2,
};

/**
 * JPEG XL signature identification.
 *
 * Checks if the passed buffer contains a valid JPEG XL signature. The passed @p
 * buf of size
 * @p size doesn't need to be a full image, only the beginning of the file.
 *
 * @returns a flag indicating if a JPEG XL signature was found and what type.
 *   - JPEGXL_SIG_INVALID: no valid signature found for JPEG XL decoding.
 *   - JPEGXL_SIG_VALID a valid JPEG XL signature was found.
 *   - JPEGXL_SIG_NOT_ENOUGH_BYTES not enough bytes were passed to determine
 *       if a valid signature is there.
 */
JPEGXL_EXPORT enum JpegxlSignature JpegxlSignatureCheck(const uint8_t* buf,
                                                        size_t len);

/**
 * Opaque structure that holds the JPEGXL decoder.
 *
 * Allocated and initialized with JpegxlDecoderCreate().
 * Cleaned up and deallocated with JpegxlDecoderDestroy().
 */
typedef struct JpegxlDecoderStruct JpegxlDecoder;

/**
 * Creates an instance of JpegxlDecoder and initializes it.
 *
 * @p memory_manager will be used for all the library dynamic allocations made
 * from this instance. The parameter may be NULL, in which case the default
 * allocator will be used. See jpegxl/memory_manager.h for details.
 *
 * @param memory_manager custom allocator function. It may be NULL. The memory
 *        manager will be copied internally.
 * @returns @c NULL if the instance can not be allocated or initialized
 * @returns pointer to initialized JpegxlDecoder otherwise
 */
JPEGXL_EXPORT JpegxlDecoder* JpegxlDecoderCreate(
    const JpegxlMemoryManager* memory_manager);

/**
 * Deinitializes and frees JpegxlDecoder instance.
 *
 * @param dec instance to be cleaned up and deallocated.
 */
JPEGXL_EXPORT void JpegxlDecoderDestroy(JpegxlDecoder* dec);

/**
 * Return value for JpegxlDecoderProcessInput.
 */
typedef enum {
  /** Decoding has finished, the end of the input file is reached and all
   * output has been delivered.
   */
  JPEGXL_DEC_FINISHED = 0,

  /** An error occured, for example invalid input file or out of memory.
   * TODO(lode): add function to get error information from decoder.
   */
  JPEGXL_DEC_ERROR,

  /** The decoder needs more input bytes to continue. In the next
   * JpegxlDecoderProcessInput call, next_in and avail_in must point to more
   * bytes to continue. If *avail_in is not 0, the new bytes must be appended to
   * the *avail_in last previous bytes.
   */
  JPEGXL_DEC_NEED_MORE_INPUT,
} JpegxlDecoderStatus;

/** Signature type of the codestream.
 */
typedef enum {
  /** JPEG XL codestream.
   */
  JPEGXL_SIG_TYPE_JPEGXL = 0,

  /** Transcoded JPEG image signature was found. The decoder will be
   * able to transcode back to the JPEG codestream passed to the encoder.
   */
  JPEGXL_SIG_TYPE_TRANSCODED_JPEG = 1,

  /** JPEG codestream, which would preferably also be decoded using this
   * decoder in case the codestream contains JPEG XL extensions (marker
   * segments).
   */
  JPEGXL_SIG_TYPE_JPEG = 2,
} JpegxlSignatureType;

typedef enum {
  // Values 1..8 match the EXIF definitions.
  // The name indicates the operation to perform to transform from the encoded
  // image to the display image.
  JPEGXL_ORIENT_IDENTITY = 1,
  JPEGXL_ORIENT_FLIP_HORIZONTAL = 2,
  JPEGXL_ORIENT_ROTATE_180 = 3,
  JPEGXL_ORIENT_FLIP_VERTICAL = 4,
  JPEGXL_ORIENT_TRANSPOSE = 5,
  JPEGXL_ORIENT_ROTATE_90_CW = 6,
  JPEGXL_ORIENT_ANTI_TRANSPOSE = 7,
  JPEGXL_ORIENT_ROTATE_90_CCW = 8,
} JpegxlOrientation;

/** Basic image information. This information is available from the file
 * signature and first part of the codestream header.
 */
typedef struct JpegxlBasicInfo {
  // TODO(lode): need additional fields for (transcoded) JPEG? For reusable
  // fields orientation must be read from Exif APP1. For has_icc_profile: must
  // look up where ICC profile is guaranteed to be in a JPEG file to be able to
  // indicate this.

  // TODO(lode): make struct packed, and/or make this opaque struct with getter
  // functions (still separate struct from opaque decoder)

  /** Whether the codestream is embedded in the container format. If true,
   * metadata information and extensions may be available in addition to the
   * codestream.
   */
  uint8_t have_container;

  /** Signature of the codestream.
   */
  JpegxlSignatureType signature_type;

  /** Width of the image in pixels, before applying orientation.
   */
  uint32_t xsize;

  /** Height of the image in pixels, before applying orientation.
   */
  uint32_t ysize;

  /** Original image color channel data type: 0 = unsigned integer, 1 = floating
   * point.
   */
  uint8_t floating_point_sample;

  /** Original image color channel bit depth.
   */
  uint32_t bits_per_sample;

  /** Original image color channel exponent bits, only used if
   * floating_point_sample is true. If the original data is single precision
   * floating point, bits_per_sample is 32 and exponent_bits_per_sample is 8,
   * and so on for other floating point precisions.
   */
  uint32_t exponent_bits_per_sample;

  /** If true, an ICC profile must be decoded after the headers. If false,
   * the color space is defined by descriptors instead. The ICC profile or
   * color descriptors themselves are not included in the basic info.
   */
  uint8_t have_icc;

  /** Bit depth of the encoded alpha channel, or 0 if there is no alpha channel.
   */
  uint32_t alpha_bits;

  /** Intensity target: intended display luminance in nits (candelas per square
   * meter).
   */
  uint32_t target_nits;

  /** Indicates a preview image exists near the beginning of the codestream.
   * The preview itself or its dimensions are not included in the basic info.
   */
  uint8_t have_preview;

  /** Indicates animation frames exist in the codestream. The animation
   * information is not included in the basic info.
   */
  uint8_t have_animation;

  /** Image orientation, value 1-8 matching the values used by JEITA CP-3451C
   * (Exif version 2.3).
   */
  JpegxlOrientation orientation;

  /** Bit depth of depth image, or 0 if there is no depth image.
   */
  uint32_t depth_bits;

  /** Base-2 logarithm of the downsampling factor of the dimensions of the depth
   * image (if any) with respect to the main image dimensions defined by xsize,
   * ysize. The depth image size is rounded up.
   * This field is invalid if depth_bits is 0.
   */
  uint32_t depth_shift;

  /** Number of additional image channels. Information of the individual
   * extra channels is not included in the basic info.
   */
  uint32_t num_extra_channels;

  /** Bit depth of all additional image channels.
   * This field is invalid if num_extra_channels is 0.
   */
  uint32_t extra_channel_bits;
} JpegxlBasicInfo;

/**
 * Returns a hint indicating how many more bytes the decoder is expected to
 * need to make JpegxlDecoderGetBasicInfo available after the next
 * JpegxlDecoderProcessInput call. This is a suggested large enough value for
 * the *avail_in parameter, but it is not guaranteed to be an upper bound nor
 * a lower bound.
 * Can be used before the first JpegxlDecoderProcessInput call, and is correct
 * the first time in most cases. If not, JpegxlDecoderSizeHintBasicInfo can be
 * called again to get an updated hint.
 *
 * @returns the size hint in bytes if the basic info is not yet fully decoded.
 * @returns 0 when the basic info is already available.
 */
JPEGXL_EXPORT size_t JpegxlDecoderSizeHintBasicInfo(const JpegxlDecoder* dec);

/**
 * Decodes JPEG XL file using the available bytes. @p *avail_in indicates how
 * many input bytes are available, and @p *next_in points to the input bytes.
 * *avail_in will be decremented by the amount of bytes that have been processed
 * by the decoder and *next_in will be incremented by the same amount, so
 * *next_in will now point at the amount of *avail_in unprocessed bytes. For the
 * next call to this function, all unprocessed bytes must be provided again (the
 * address need not match, but the contents must), and more bytes may be
 * concatenated after the unprocessed bytes.
 *
 * The returned status indicates whether the decoder needs more input bytes, or
 * more output buffer for a certain type of output data. No matter what the
 * returned status is (other than JPEGXL_DEC_ERROR), new information, such as
 * JpegxlDecoderGetBasicInfo, may have become available after this call.
 *
 * @returns status indicating the decoding needs more input or output bytes to
 * continue, encountered an error, or successfully finished, See
 * JpegxlDecoderStatus for the description of each possible status.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderProcessInput(
    JpegxlDecoder* dec, const uint8_t** next_in, size_t* avail_in);

/**
 * Outputs the basic image information, such as image dimensions, bit depth and
 * all other JpegxlBasicInfo fields, if available.
 *
 * @param info struct to copy the information into, or NULL to only check
 * whether the information is available through the return value.
 * @returns 0 if the value is available, 1 if not available.
 */
JPEGXL_EXPORT int JpegxlDecoderGetBasicInfo(const JpegxlDecoder* dec,
                                            JpegxlBasicInfo* info);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JPEGXL_DECODE_H_ */
