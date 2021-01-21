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

/** @file encode.h
 * @brief Encoding API for JPEG XL.
 */

#ifndef JXL_ENCODE_H_
#define JXL_ENCODE_H_

#include "jxl/decode.h"
#include "jxl/jxl_export.h"
#include "jxl/memory_manager.h"
#include "jxl/parallel_runner.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Encoder library version.
 *
 * @return the encoder library version as an integer:
 * MAJOR_VERSION * 1000000 + MINOR_VERSION * 1000 + PATCH_VERSION. For example,
 * version 1.2.3 would return 1002003.
 */
JXL_EXPORT uint32_t JxlEncoderVersion(void);

/**
 * Opaque structure that holds the JPEG XL encoder.
 *
 * Allocated and initialized with JxlEncoderCreate().
 * Cleaned up and deallocated with JxlEncoderDestroy().
 */
typedef struct JxlEncoderStruct JxlEncoder;

/**
 * Opaque structure that holds encoding options for a JPEG XL encoder.
 *
 * Allocated and initialized with JxlEncoderOptionsCreate().
 * Cleaned up and deallocated when the encoder is destroyed with
 * JxlEncoderDestroy().
 */
typedef struct JxlEncoderOptionsStruct JxlEncoderOptions;

/**
 * Return value for multiple encoder functions.
 */
typedef enum {
  /** Function call finished successfully, or encoding is finished and there is
   * nothing more to be done.
   */
  JXL_ENC_SUCCESS = 0,

  /** An error occurred, for example out of memory.
   */
  JXL_ENC_ERROR = 1,

  /** The encoder needs more output buffer to continue encoding.
   */
  JXL_ENC_NEED_MORE_OUTPUT = 2,

} JxlEncoderStatus;

/**
 * Creates an instance of JxlEncoder and initializes it.
 *
 * @p memory_manager will be used for all the library dynamic allocations made
 * from this instance. The parameter may be NULL, in which case the default
 * allocator will be used. See jpegxl/memory_manager.h for details.
 *
 * @param memory_manager custom allocator function. It may be NULL. The memory
 *        manager will be copied internally.
 * @return @c NULL if the instance can not be allocated or initialized
 * @return pointer to initialized JxlEncoder otherwise
 */
JXL_EXPORT JxlEncoder* JxlEncoderCreate(const JxlMemoryManager* memory_manager);

/**
 * Deinitializes and frees JxlEncoder instance.
 *
 * @param enc instance to be cleaned up and deallocated.
 */
JXL_EXPORT void JxlEncoderDestroy(JxlEncoder* enc);

/**
 * Set the parallel runner for multithreading. May only be set before starting
 * encoding.
 *
 * @param enc encoder object.
 * @param parallel_runner function pointer to runner for multithreading. It may
 *        be NULL to use the default, single-threaded, runner. A multithreaded
 *        runner should be set to reach fast performance.
 * @param parallel_runner_opaque opaque pointer for parallel_runner.
 * @return JXL_ENC_SUCCESS if the runner was set, JXL_ENC_ERROR
 * otherwise (the previous runner remains set).
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderSetParallelRunner(JxlEncoder* enc, JxlParallelRunner parallel_runner,
                            void* parallel_runner_opaque);

/**
 * Encodes JPEG XL file using the available bytes. @p *avail_out indicates how
 * many output bytes are available, and @p *next_out points to the input bytes.
 * *avail_out will be decremented by the amount of bytes that have been
 * processed by the encoder and *next_out will be incremented by the same
 * amount, so *next_out will now point at the amount of *avail_out unprocessed
 * bytes.
 *
 * The returned status indicates whether the encoder needs more output bytes.
 * When the return value is not JXL_ENC_ERROR or JXL_ENC_SUCCESS, the encoding
 * requires more JxlEncoderProcessOutput calls to continue.
 *
 * @param enc encoder object.
 * @param next_out pointer to next bytes to write to.
 * @param avail_out amount of bytes available starting from *next_out.
 * @return JXL_ENC_SUCCESS when encoding finished and all events handled.
 * @return JXL_ENC_ERROR when encoding failed, e.g. invalid input.
 * @return JXL_ENC_NEED_MORE_OUTPUT more output buffer is necessary.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderProcessOutput(JxlEncoder* enc,
                                                    uint8_t** next_out,
                                                    size_t* avail_out);

/**
 * Sets the buffer to read JPEG encoded bytes from for the next image to encode
 * losslessly.
 *
 * Currently only supports adding JPEG frames losslessly if there is only a
 * single frame in the image.
 *
 * @param options set of encoder options to use when encoding the frame.
 * @param buffer bytes to read JPEG from. Owned by the caller and its contents
 * are copied internally.
 * @param size size of buffer in bytes.
 * @return JXL_ENC_SUCCESS on success, JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderAddJPEGFrame(JxlEncoderOptions* options,
                                                   const uint8_t* buffer,
                                                   size_t size);

/**
 * Sets the buffer to read pixels from for the next image to encode. Must call
 * JxlEncoderSetDimensions before JxlEncoderAddImageFrame.
 *
 * Currently only some pixel formats are supported:
 * - JXL_TYPE_UINT8, input pixels assumed to be nonlinear SRGB encoded
 * - JXL_TYPE_UINT16, input pixels assumed to be nonlinear SRGB encoded
 * - JXL_TYPE_FLOAT, input pixels are assumed to be linear SRGB encoded
 *
 * @param options set of encoder options to use when encoding the frame
 * @param pixel_format format for pixels. Object owned by the caller and its
 * contents are copied internally.
 * @param buffer buffer type to input the pixel data from. Owned by the caller
 * and its contents are copied internally.
 * @param size size of buffer in bytes
 * @return JXL_ENC_SUCCESS on success, JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderAddImageFrame(
    JxlEncoderOptions* options, const JxlPixelFormat* pixel_format,
    const void* buffer, size_t size);

/**
 * Declares that this encoder will not encode anything further.
 *
 * Must be called between JxlEncoderAddImageFrame of the last frame and the next
 * call to JxlEncoderProcessOutput, or JxlEncoderProcessOutput won't output the
 * last frame correctly.
 *
 * @param enc encoder object
 */
JXL_EXPORT void JxlEncoderCloseInput(JxlEncoder* enc);

/**
 * Sets the dimensions of the image encoded by this encoder.
 *
 * @param enc encoder object
 * @param xsize width of image
 * @param ysize height of image
 * @return JXL_ENC_SUCCESS if the dimensions are within jxl spec limitations,
 * JXL_ENC_ERROR otherwise
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderSetDimensions(JxlEncoder* enc,
                                                    size_t xsize, size_t ysize);

/**
 * Sets lossless/lossy mode for the provided options. Default is lossy.
 *
 * @param options set of encoder options to update with the new mode
 * @param lossless whether the options should be lossless
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderOptionsSetLossless(JxlEncoderOptions* options, JXL_BOOL lossless);

/**
 * Sets encoder effort/speed level. Valid values are, from faster to slower
 * speed:
 * 3:falcon 4:cheetah 5:hare 6:wombat 7:squirrel 8:kitten 9:tortoise
 * Default: squirrel (7).
 *
 * @param options set of encoder options to update with the new mode
 * @param effort the effort value to set
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderOptionsSetEffort(JxlEncoderOptions* options, int effort);

/**
 * Sets the distance level for lossy compression: target max butteraugli
 *  distance, lower = higher quality. Range: 0 .. 15.
 *  0.0 = mathematically lossless (however, use JxlEncoderOptionsSetLossless to
 *  use true lossless).
 *  1.0 = visually lossless.
 *  Recommended range: 0.5 .. 3.0.
 *  Default value: 1.0.
 *  If JxlEncoderOptionsSetLossless is used, this value is unused and implied
 *  to be 0.
 *
 * @param options set of encoder options to update with the new mode
 * @param distance the distance value to set
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderOptionsSetDistance(JxlEncoderOptions* options, float distance);

/**
 * Create a new set of encoder options, with all values initially copied from
 * the @p source options, or set to default if @p source is NULL.
 *
 * The returned pointer is an opaque struct tied to the encoder and it will be
 * deallocated by the encoder when JxlEncoderDestroy() is called. For functions
 * taking both a @ref JxlEncoder and a @ref JxlEncoderOptions, only
 * JxlEncoderOptions created with this function for the same encoder instance
 * can be used.
 *
 * @param enc encoder object
 * @param source source options to copy initial values from, or NULL to get
 * defaults initialized to defaults
 * @return the opaque struct pointer identifying a new set of encoder options.
 */
JXL_EXPORT JxlEncoderOptions* JxlEncoderOptionsCreate(
    JxlEncoder* enc, const JxlEncoderOptions* source);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JXL_ENCODE_H_ */
