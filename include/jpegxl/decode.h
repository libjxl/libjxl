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

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JPEGXL_DECODE_H_ */
