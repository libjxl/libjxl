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
 * Return value for multiple encoder functions.
 */
typedef enum {
  /** Function call finished sucessfully, or encoding is finished and there is
   * nothing more to be done.
   */
  JXL_ENC_SUCCESS = 0,

  /** An error occured, for example out of memory.
   */
  JXL_ENC_ERROR = 1,

  /** The encoder needs more output buffer to continue encoding.
   */
  JXL_ENC_NEED_MORE_OUTPUT = 2,

} JxlEncoderStatus;

/**
 * Data type for the minimum format necessary to encode a single frame.
 */
typedef struct {
  /**
   * Pixel format of frame.
   */
  JxlPixelFormat pixel_format;

  /**
   * Pixel width of frame.
   */
  uint32_t frame_width;

  /**
   * Pixel height of frame.
   */
  uint32_t frame_height;
} JxlFrameFormat;

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
 * @param enc encoder object
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
 * *avail_out will be decremented by the amount of bytes that have been processed
 * by the encoder and *next_out will be incremented by the same amount, so
 * *next_out will now point at the amount of *avail_out unprocessed bytes.
 *
 * The returned status indicates whether the encoder needs more output bytes.
 * When the return value is not JXL_ENC_ERROR or JXL_ENC_SUCCESS, the encoding
 * requires more JxlEncoderProcessOutput calls to continue.
 *
 * @param enc encoder object
 * @param next_out pointer to next bytes to write to
 * @param avail_out amount of bytes available starting from *next_out
 * @return JXL_ENC_SUCCESS when encoding finished and all events handled.
 * @return JXL_ENC_ERROR when encoding failed, e.g. invalid input.
 * @return JXL_ENC_NEED_MORE_OUTPUT more output buffer is necessary.
 */
JXL_EXPORT JxlEncoderStatus JxlEncoderProcessOutput(
    JxlEncoder* enc, uint8_t** next_out, size_t* avail_out);

/**
 * Sets the buffer to read from for the next image to encode.
 * The buffer is owned by the caller.
 *
 * @param enc encoder object
 * @param frame_format frame format for pixels. Object owned by user and its
 * contents are copied internally.
 * @param buffer buffer type to input the pixel data from
 * @param size size of buffer in bytes
 * @return JXL_ENC_SUCCESS on success, JXL_ENC_ERROR on error
 */
JXL_EXPORT JxlEncoderStatus
JxlEncoderAddImageFrame(JxlEncoder* enc, const JxlFrameFormat* frame_format,
                        void* buffer, size_t size);

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

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JXL_ENCODE_H_ */
