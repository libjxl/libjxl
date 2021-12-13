/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 *
 *
 * Auxiliary functions for fetching and processing JPEG XL decoder
 * output that might be useful in a no-C++ context, such as when
 * implementing a JPEG XL library via some other language's C foreign
 * function interface.
 */

#ifndef TOOLS_FETCH_ENCODED_H_
#define TOOLS_FETCH_ENCODED_H_


#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#include <stdint.h>

#include "jxl/types.h"


/* Fetches the encoded data from `jxl_encoder`.  If
   `*compressed_buffer` is non-NULL, *compressed_buffer_size must
   indicate the allocated size of the buffer (in bytes), and ownership
   of the allocated buffer is handed over to the callee.

   At return time, if `*compressed_buffer` is non-NULL, it is a
   pointer to an allocated buffer of size `*compressed_buffer_size`
   bytes, and ownership of this buffer is handed over to the caller.

   If the return value is JXL_TRUE, the first
   `*compressed_buffer_used` bytes in the buffer contain the encoded
   data.
 */
JXL_BOOL fetch_jxl_encoded_image(JxlEncoder *jxl_encoder,
                                 uint8_t **compressed_buffer,
                                 size_t *compressed_buffer_size,
                                 size_t *compressed_buffer_used);


/* Writes `size` many bytes from buffer `bytes` to the file named `filename`. */
JXL_BOOL write_jxl_file(const uint8_t* bytes,
                        size_t size,
                        const char* filename);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* TOOLS_FETCH_ENCODED_H_ */
