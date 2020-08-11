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

/** @file jpegxl/decode.h
 * @brief Decoding API for JPEG XL.
 */

#ifndef JPEGXL_DECODE_H_
#define JPEGXL_DECODE_H_

#include <stddef.h>
#include <stdint.h>

#include "jpegxl/codestream_header.h"
#include "jpegxl/color_encoding.h"
#include "jpegxl/jpegxl_export.h"
#include "jpegxl/memory_manager.h"
#include "jpegxl/parallel_runner.h"
#include "jpegxl/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Decoder library version.
 *
 * @return the decoder library version as an integer:
 * MAJOR_VERSION * 1000000 + MINOR_VERSION * 1000 + PATCH_VERSION. For example,
 * version 1.2.3 would return 1002003.
 */
JPEGXL_EXPORT uint32_t JpegxlDecoderVersion(void);

/** The result of JpegxlSignatureCheck.
 */
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
 * @return a flag indicating if a JPEG XL signature was found and what type.
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
 * @return @c NULL if the instance can not be allocated or initialized
 * @return pointer to initialized JpegxlDecoder otherwise
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
 * The values above 64 are optional informal events that can be subscribed to.
 */
typedef enum {
  /** Function call finished sucessfully, or decoding is finished and there is
   * nothing more to be done.
   */
  JPEGXL_DEC_SUCCESS = 0,

  /** An error occured, for example invalid input file or out of memory.
   * TODO(lode): add function to get error information from decoder.
   */
  JPEGXL_DEC_ERROR = 1,

  /** The decoder needs more input bytes to continue. In the next
   * JpegxlDecoderProcessInput call, next_in and avail_in must point to more
   * bytes to continue. If *avail_in is not 0, the new bytes must be appended to
   * the *avail_in last previous bytes.
   */
  JPEGXL_DEC_NEED_MORE_INPUT = 2,

  /** The decoder needs an output buffer to continue. Which output buffers it
   * needs depends on events subscribed to with JpegxlDecoderSubscribeEvents.
   */
  JPEGXL_DEC_NEED_MORE_OUTPUT = 3,

  /** Informative event: basic information such as image dimensions and extra
   * channels.
   */
  JPEGXL_DEC_BASIC_INFO = 64,

  /** Informative event: user extensions of the codestream header.
   */
  JPEGXL_DEC_EXTENSIONS = 128,

  /** Informative event: preview header from the codestream header.
   */
  JPEGXL_DEC_PREVIEW_HEADER = 256,

  /** Informative event: animation header from the codestream header.
   */
  JPEGXL_DEC_ANIMATION_HEADER = 512,

  /** Informative event: color encoding or ICC profile from the codestream
   * header.
   */
  JPEGXL_DEC_COLOR_ENCODING = 1024,

  /** Informative event: DC image, 8x8 sub-sampled image.
   */
  JPEGXL_DEC_DC = 2048,

  /** Informative event: full image decoded.
   */
  JPEGXL_DEC_FULL_IMAGE = 2048,
} JpegxlDecoderStatus;

/** Data type for the sample values per channel per pixel.
 */
typedef enum {
  /** Use 1-bit packed in uint8_t, first pixel in LSB, padded to uint8_t per
   * row.
   * TODO(lode): support first in MSB, other padding.
   */
  JPEGXL_TYPE_BOOLEAN,

  /** Use type uint8_t.
   */
  JPEGXL_TYPE_UINT8,

  /** Use type uint16_t.
   */
  JPEGXL_TYPE_UINT16,

  /** Use type uint32_t.
   */
  JPEGXL_TYPE_UINT32,

  /** use type float.
   */
  JPEGXL_TYPE_FLOAT,
} JpegxlDataType;

/** Data type for the sample values per channel per pixel for the output buffer
 * for pixels. This is not necessarily the same as the data type encoded in the
 * codestream. The channels are interleaved per pixel. The pixels are
 * organized row by row, left to right, top to bottom.
 * TODO(lode): support padding / alignment (row stride)
 * TODO(lode): support outputting >8-bit data into uint8_t (and endianness)
 * TODO(lode): support non-interleaved (may be a no-op here, involves getting
 *     single channels separately instead)
 * TODO(lode): support different channel orders if needed (RGB, BGR, ...)
 */
typedef struct {
  /** Amount of channels available in a pixel buffer.
   * 1: single-channel data, e.g. grayscale
   * 2: single-channel + alpha
   * 3: trichromatic, e.g. RGB
   * 4: trichromatic + alpha
   * TODO(lode): this needs finetuning. It is not yet defined how the user
   * chooses output color space. CMYK+alpha needs 5 channels.
   */
  size_t num_channels;

  /** Data type of each channel.
   */
  JpegxlDataType data_type;
} JpegxlPixelFormat;

/**
 * Set the parallel runner for multithreading. May only be set before starting
 * decoding.
 *
 * @param dec decoder object
 * @param parallel_runner function pointer to runner for multithreading. It may
 *        be NULL to use the default, single-threaded, runner. A multithreaded
 *        runner should be set to reach fast performance.
 * @param parallel_runner_opaque opaque pointer for parallel_runner.
 * @return JPEGXL_DEC_SUCCESS if the runner was set, JPEGXL_DEC_ERROR
 * otherwise (the previous runner remains set).
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderSetParallelRunner(
    JpegxlDecoder* dec, JpegxlParallelRunner parallel_runner,
    void* parallel_runner_opaque);

/** Specifies how the color profile information of the image can be retrieved,
 * Either through ICC profile, or through JpegxlColorEncoding.
 *
 * The following scenarios are possible:
 * - The ICC profile is accurate and the color encoding is invalid. Only the
 * ICC profile may be requested.
 * - the ICC profile is inaccurate and the color encoding is valid. Both can
 * be requested, but only the color encoding correctly specifies the color, the
 * ICC profile is an approximation that can be used when only ICC profile is
 * supported.
 * - Both are valid and accurate, any can be used equally well.
 *
 * Example: for PQ or HLG color, JpegxlDecoderGetColorEncoding describes the
 * transfer function more correctly and the ICC profile will not be accurate. On
 * the other hand, there exist transfer functions that can only be described by
 * an ICC profile, in that case icc_profile_accurate is JPEGXL_TRUE and
 * color_encoding_valid is JPEGXL_FALSE.
 */
typedef struct {
  /** If JPEGXL_TRUE, the ICC profile describes the image color encoding, once
   * the ICC profile has been decoded. If JPEGXL_FALSE, the ICC profile is not
   * accurate and a JpegxlDecoderGetColorEncoding must be used instead to
   * accurately describe the color profile of the image. An ICC profile can
   * still be requested, but it will only be an approximation.
   */
  JPEGXL_BOOL icc_profile_accurate;

  /** If JPEGXL_TRUE, JpegxlDecoderGetColorEncoding describes the image color
   * encoding, once its information is available. If JPEGXL_FALSE, only the ICC
   * profile can be used.
   */
  JPEGXL_BOOL color_encoding_valid;

  /**
   * ICC profile size in bytes.
   */
  size_t icc_profile_size;
} JpegxlColorProfileSource;

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
 * @param dec decoder object
 * @return the size hint in bytes if the basic info is not yet fully decoded.
 * @return 0 when the basic info is already available.
 */
JPEGXL_EXPORT size_t JpegxlDecoderSizeHintBasicInfo(const JpegxlDecoder* dec);

/** Select for which informative events (JPEGXL_DEC_BASIC_INFO, etc...) the
 * decoder should return with a status. It is not required to subscribe to any
 * events, data can still be requested from the decoder as soon as it available.
 * By default, the decoder is subscribed to no events (events_wanted == 0), and
 * the decoder will then only return when it cannot continue because it needs
 * more input data or more output buffer. This function may only be be called
 * before using JpegxlDecoderProcessInput
 *
 * @param dec decoder object
 * @param events_wanted bitfield of desired events.
 * @return JPEGXL_DEC_SUCCESS if no error, JPEGXL_DEC_ERROR otherwise.
 */
JPEGXL_EXPORT JpegxlDecoderStatus
JpegxlDecoderSubscribeEvents(JpegxlDecoder* dec, int events_wanted);

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
 * JpegxlDecoderGetBasicInfo, may have become available after this call. When
 * the return value is not JPEGXL_DEC_ERROR or JPEGXL_DEC_SUCCESS, the decoding
 * requires more JpegxlDecoderProcessInput calls to continue.
 *
 * @param dec decoder object
 * @param next_in pointer to next bytes to read from
 * @param avail_in amount of bytes available starting from *next_in
 * @return JPEGXL_DEC_SUCCESS when decoding finished and all events handled.
 * @return JPEGXL_DEC_ERROR when decoding failed, e.g. invalid codestream.
 * TODO(lode) document the input data mechanism
 * @return JPEGXL_DEC_NEED_MORE_INPUT more input data is necessary.
 * @return JPEGXL_DEC_BASIC_INFO when basic info such as image dimensions is
 * available and this informative event is subscribed to.
 * @return JPEGXL_DEC_EXTENSIONS when JPEG XL codestream user extensions are
 * available and this informative event is subscribed to.
 * @return JPEGXL_DEC_PREVIEW_HEADER when preview dimensions are available and
 * this informative event is subscribed to.
 * @return JPEGXL_DEC_ANIMATION_HEADER when animation information is available
 * and this informative event is subscribed to.
 * @return JPEGXL_DEC_COLOR_ENCODING when color profile information is
 * available and this informative event is subscribed to.
 * @return JPEGXL_DEC_DC when DC pixel information is available and output in
 * the DC buffer.
 * @return JPEGXL_DEC_FULL_IMAGE when all pixel information at highest detail is
 * available and has been output in the pixel buffer.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderProcessInput(
    JpegxlDecoder* dec, const uint8_t** next_in, size_t* avail_in);

/**
 * Outputs the basic image information, such as image dimensions, bit depth and
 * all other JpegxlBasicInfo fields, if available.
 *
 * @param dec decoder object
 * @param info struct to copy the information into, or NULL to only check
 * whether the information is available through the return value.
 * @return JPEGXL_DEC_SUCCESS if the value is available,
 *    JPEGXL_DEC_NEED_MORE_INPUT if not yet available, JPEGXL_DEC_ERROR in case
 *    of other error conditions.
 */
JPEGXL_EXPORT JpegxlDecoderStatus
JpegxlDecoderGetBasicInfo(const JpegxlDecoder* dec, JpegxlBasicInfo* info);

/**
 * Outputs information for extra channel at the given index. The index must be
 * smaller than num_extra_channels in the associated JpegxlBasicInfo.
 *
 * @param dec decoder object
 * @param index index of the extra channel to query.
 * @param info struct to copy the information into, or NULL to only check
 * whether the information is available through the return value.
 * @return JPEGXL_DEC_SUCCESS if the value is available,
 *    JPEGXL_DEC_NEED_MORE_INPUT if not yet available, JPEGXL_DEC_ERROR in case
 *    of other error conditions.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderGetExtraChannelInfo(
    const JpegxlDecoder* dec, size_t index, JpegxlExtraChannelInfo* info);

/**
 * Outputs name for extra channel at the given index in UTF-8. The index must be
 * smaller than num_extra_channels in the associated JpegxlBasicInfo. The buffer
 * for name must have at least name_length + 1 bytes allocated, gotten from
 * the associated JpegxlExtraChannelInfo.
 *
 * @param dec decoder object
 * @param index index of the extra channel to query.
 * @param name buffer to copy the name into
 * @param size size of the name buffer in bytes
 * @return JPEGXL_DEC_SUCCESS if the value is available,
 *    JPEGXL_DEC_NEED_MORE_INPUT if not yet available, JPEGXL_DEC_ERROR in case
 *    of other error conditions.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderGetExtraChannelName(
    const JpegxlDecoder* dec, size_t index, char* name, size_t size);

/**
 * Outputs the preview header, if available.
 *
 * @param dec decoder object
 * @param preview_header struct to copy the information into, or NULL to only
 * check whether the information is available through the return value.
 * @return JPEGXL_DEC_SUCCESS if the value is available,
 *    JPEGXL_DEC_NEED_MORE_INPUT if not yet available, JPEGXL_DEC_ERROR in case
 *    of other error conditions.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderGetPreviewHeader(
    const JpegxlDecoder* dec, JpegxlPreviewHeader* preview_header);

/**
 * Outputs the animation header, if available.
 *
 * @param dec decoder object
 * @param animation_header struct to copy the information into, or NULL to only
 * check whether the information is available through the return value.
 * @return JPEGXL_DEC_SUCCESS if the value is available,
 *    JPEGXL_DEC_NEED_MORE_INPUT if not yet available, JPEGXL_DEC_ERROR in case
 *    of other error conditions.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderGetAnimationHeader(
    const JpegxlDecoder* dec, JpegxlAnimationHeader* animation_header);

/**
 * Outputs the color information, if available. The color info tells whether
 * JpegxlDecoderGetColorEncoding and/or JpegxlDecoderGetICCProfile should be
 * used to get the correct image color profile.
 *
 * @param dec decoder object
 * @param color_info struct to copy the information into, or NULL to only
 * check whether the information is available through the return value.
 * @return JPEGXL_DEC_SUCCESS if the value is available,
 *    JPEGXL_DEC_NEED_MORE_INPUT if not yet available, JPEGXL_DEC_ERROR in case
 *    of other error conditions.
 *
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderGetColorProfileSource(
    const JpegxlDecoder* dec, JpegxlColorProfileSource* color_info);

/**
 * Outputs the color profile information, if available. Only may be used if
 * JpegxlColorProfileSource.color_encoding_valid.
 *
 * @param dec decoder object
 * @param color_encoding struct to copy the information into, or NULL to only
 * check whether the information is available through the return value.
 * @return JPEGXL_DEC_SUCCESS if the value is available,
 *    JPEGXL_DEC_NEED_MORE_INPUT if not yet available, JPEGXL_DEC_ERROR in case
 *    of other error conditions.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderGetColorEncoding(
    const JpegxlDecoder* dec, JpegxlColorEncoding* color_encoding);

/**
 * Outputs ICC profile. The output buffer should have enough bytes allocated
 * to contain the icc_size given in the JpegxlColorEncoding.
 *
 * @param dec decoder object
 * @param icc_profile buffer to copy the ICC profile into
 * @param size size of the icc_profile buffer in bytes
 * @return JPEGXL_DEC_SUCCESS if the value is available,
 *    JPEGXL_DEC_NEED_MORE_INPUT if not yet available, JPEGXL_DEC_ERROR in case
 *    of other error conditions.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderGetICCProfile(
    const JpegxlDecoder* dec, uint8_t* icc_profile, size_t size);

/**
 * Returns the size in bytes the DC image output pixel buffer requires at least
 * to contain the DC image pixels in the given format. This is the minumum size
 * of the buffer for JpegxlDecoderSetDCOutBuffer. Requires the basic image
 * information is available in the decoder.
 *
 * @param dec decoder object
 * @param format format of pixelsformat of pixels.
 * @param size output value, buffer size in bytes
 * @return JPEGXL_DEC_SUCCESS on success, JPEGXL_DEC_ERROR on error, such as
 *    information not available yet.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderDCOutBufferSize(
    const JpegxlDecoder* dec, const JpegxlPixelFormat* format, size_t* size);

/**
 * Sets the buffer to write the lower resolution (8x8 sub-sampled) DC image
 * to. The size of the buffer must be at least as large as given by
 * JpegxlDecoderDCOutBufferSize. The buffer follows the format described by
 * JpegxlPixelFormat. The DC image has dimensions ceil(sizex / 8) * ceil(sizey /
 * 8). The buffer is owned by the caller.
 *
 * @param dec decoder object
 * @param format format of pixels. Object owned by user and its contents are
 * copied internally.
 * @param buffer buffer type to output the pixel data to
 * @param size size of buffer in bytes
 * @return JPEGXL_DEC_SUCCESS on success, JPEGXL_DEC_ERROR on error, such as
 * size too small.
 */
JPEGXL_EXPORT JpegxlDecoderStatus
JpegxlDecoderSetDCOutBuffer(JpegxlDecoder* dec, const JpegxlPixelFormat* format,
                            void* buffer, size_t size);

/**
 * Returns the size in bytes the image output pixel buffer requires at least to
 * contain all pixels in the given format. This is the minumum size of the
 * buffer for JpegxlDecoderSetImageOutBuffer. Requires the basic image
 * information is available in the decoder.
 *
 * @param dec decoder object
 * @param format format of pixelsformat of pixels.
 * @param size output value, buffer size in bytes
 * @return JPEGXL_DEC_SUCCESS on success, JPEGXL_DEC_ERROR on error, such as
 *    information not available yet.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderImageOutBufferSize(
    const JpegxlDecoder* dec, const JpegxlPixelFormat* format, size_t* size);

/**
 * Sets the buffer to write the full resolution image to. The size of the
 * buffer must be at least as large as given by JpegxlDecoderImageOutBufferSize.
 * The buffer follows the format described by JpegxlPixelFormat. The buffer is
 * owned by the caller.
 *
 * @param dec decoder object
 * @param format format of pixelsformat of pixels. Object owned by user and its
 * contents are copied internally.
 * @param buffer buffer type to output the pixel data to
 * @param size size of buffer in bytes
 * @return JPEGXL_DEC_SUCCESS on success, JPEGXL_DEC_ERROR on error, such as
 * size too small.
 */
JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderSetImageOutBuffer(
    JpegxlDecoder* dec, const JpegxlPixelFormat* format, void* buffer,
    size_t size);

// TODO(lode): add way to output extra channels

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JPEGXL_DECODE_H_ */
