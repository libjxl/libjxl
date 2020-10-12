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

/** @file jpegxl/encode.h
 * @brief Encoding API for JPEG XL.
 */

#ifndef JPEGXL_ENCODE_H_
#define JPEGXL_ENCODE_H_

#include "jpegxl/jpegxl_export.h"
#include "jpegxl/memory_manager.h"
#include "jpegxl/parallel_runner.h"

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
JPEGXL_EXPORT uint32_t JpegxlEncoderVersion(void);

/**
 * Opaque structure that holds the JPEGXL encoder.
 *
 * Allocated and initialized with JpegxlEncoderCreate().
 * Cleaned up and deallocated with JpegxlEncoderDestroy().
 */
typedef struct JpegxlEncoderStruct JpegxlEncoder;

/**
 * Return value for multiple encoder functions.
 */
typedef enum {
  /** Function call finished sucessfully, or encoding is finished and there is
   * nothing more to be done.
   */
  JPEGXL_ENC_SUCCESS = 0,

  /** An error occured, for example out of memory.
   */
  JPEGXL_ENC_ERROR = 1,

} JpegxlEncoderStatus;

/**
 * Creates an instance of JpegxlEncoder and initializes it.
 *
 * @p memory_manager will be used for all the library dynamic allocations made
 * from this instance. The parameter may be NULL, in which case the default
 * allocator will be used. See jpegxl/memory_manager.h for details.
 *
 * @param memory_manager custom allocator function. It may be NULL. The memory
 *        manager will be copied internally.
 * @return @c NULL if the instance can not be allocated or initialized
 * @return pointer to initialized JpegxlEncoder otherwise
 */
JPEGXL_EXPORT JpegxlEncoder* JpegxlEncoderCreate(
    const JpegxlMemoryManager* memory_manager);

/**
 * Deinitializes and frees JpegxlEncoder instance.
 *
 * @param enc instance to be cleaned up and deallocated.
 */
JPEGXL_EXPORT void JpegxlEncoderDestroy(JpegxlEncoder* enc);

/**
 * Set the parallel runner for multithreading. May only be set before starting
 * encoding.
 *
 * @param enc encoder object
 * @param parallel_runner function pointer to runner for multithreading. It may
 *        be NULL to use the default, single-threaded, runner. A multithreaded
 *        runner should be set to reach fast performance.
 * @param parallel_runner_opaque opaque pointer for parallel_runner.
 * @return JPEGXL_ENC_SUCCESS if the runner was set, JPEGXL_ENC_ERROR
 * otherwise (the previous runner remains set).
 */
JPEGXL_EXPORT JpegxlEncoderStatus JpegxlEncoderSetParallelRunner(
    JpegxlEncoder* enc, JpegxlParallelRunner parallel_runner,
    void* parallel_runner_opaque);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JPEGXL_ENCODE_H_ */
