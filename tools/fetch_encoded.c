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
#include "fetch_encoded.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jxl/encode.h"

#define DEBUG_BUFFER_REALLOCATION 0

JXL_BOOL fetch_jxl_encoded_image(JxlEncoder *jxl_encoder,
                                 uint8_t **compressed_buffer,
                                 size_t *compressed_buffer_size,
                                 size_t *compressed_buffer_used) {
  JXL_BOOL success = JXL_FALSE;
  uint8_t *compressed_buffer2 = NULL, *compressed_ptr = NULL;
  size_t compressed_buffer2_size = 0;
  size_t compressed_available;
  JxlEncoderStatus process_result;

  /* `max_ok_buffer_size` is half the maximal size_t value, rounded down:
     we do not grow buffers this large or larger.
   */
  const size_t max_ok_buffer_size = (~(size_t)0) / 2;

  if (*compressed_buffer == NULL) {
    /* Caller did not pass us a buffer to use, so we need to allocate one. */
    *compressed_buffer_size = 64;
    if (NULL == (*compressed_buffer = malloc(*compressed_buffer_size))) {
      goto cleanup;
    }
  }
  *compressed_buffer_used = 0;
  compressed_available = *compressed_buffer_size;
  compressed_ptr = *compressed_buffer;
  do {
    process_result = JxlEncoderProcessOutput(jxl_encoder, &compressed_ptr,
                                             &compressed_available);
    *compressed_buffer_used = compressed_ptr - *compressed_buffer;
    if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      if (*compressed_buffer_size >= max_ok_buffer_size) {
        goto cleanup;
      }
      compressed_buffer2_size = 2 * *compressed_buffer_size;
      if (NULL == (compressed_buffer2 = malloc(compressed_buffer2_size))) {
        goto cleanup;
      }
      memcpy(compressed_buffer2, *compressed_buffer, *compressed_buffer_used);
      free(*compressed_buffer);
      *compressed_buffer = compressed_buffer2;
      *compressed_buffer_size = compressed_buffer2_size;
      compressed_buffer2 = NULL;
      compressed_ptr = &(*compressed_buffer)[*compressed_buffer_used];
      compressed_available = *compressed_buffer_size - *compressed_buffer_used;
#if DEBUG_BUFFER_REALLOCATION
      fprintf(stderr, "Re-allocated to %d bytes, %d used.\n",
              (int)*compressed_buffer_size, (int)*compressed_buffer_used);
#endif
    }
  } while (process_result == JXL_ENC_NEED_MORE_OUTPUT);

  if (JXL_ENC_SUCCESS != process_result) {
    goto cleanup;
  } else {
    success = JXL_TRUE;
  }

cleanup:
  if (compressed_buffer2 != NULL) {
    free(compressed_buffer2);
  }
  return success;
}

JXL_BOOL write_jxl_file(const uint8_t *bytes, size_t size,
                        const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Could not open file for writing: %s\n", filename);
    return JXL_FALSE;
  }
  size_t bytes_written = fwrite(bytes, sizeof(uint8_t), size, file);
  if (bytes_written != size) {
    fprintf(stderr,
            "Failure: Wanted to write %d bytes, "
            "but did write %d bytes to file: %s\n"
            "Error: %s\n",
            (int)size, (int)bytes_written, filename, strerror(errno));
    return JXL_FALSE;
  }
  if (fclose(file)) {
    fprintf(stderr, "Could not close file: %s\n", filename);
    return JXL_FALSE;
  }
  return JXL_TRUE;
}
