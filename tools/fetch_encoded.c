
#include "jxl/encode.h"
#include "fetch_encoded.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Fetches the encoded data from `jxl_encoder` and sets
   `*compressed_out` to a pointer to that freshly allocated data,
   transfering ownership to caller. Data size is returned via
   `compressed_size_out`. On success, returns JXL_TRUE.
   On failure, returns JXL_FALSE and sets `*compressed_out`
   to NULL.
 */
JXL_BOOL fetch_jxl_encoded_image(JxlEncoder *jxl_encoder,
                                 uint8_t **compressed_out,
                                 size_t *compressed_size_out) {
  JXL_BOOL success = JXL_TRUE;  
  uint8_t *compressed = NULL, *compressed2 = NULL, *compressed_ptr = NULL;
  size_t compressed_size = 64, compressed2_size = 0;
  size_t compressed_available = compressed_size;
  size_t compressed_used = 0;
  const size_t max_ok_size = (~(size_t)0) / 2;
  
  *compressed_out = NULL;
  *compressed_size_out = 0;
  
  if (NULL == (compressed = compressed_ptr = malloc(compressed_size))) {
    goto cleanup;
  }
  
  JxlEncoderStatus process_result;
  do {
    process_result = JxlEncoderProcessOutput(jxl_encoder,
                                             &compressed_ptr,
                                             &compressed_available);
    compressed_used = compressed_ptr - compressed;
    if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      if (compressed_used >= max_ok_size) {
        success = JXL_FALSE;
        goto cleanup;
      }
      compressed2_size = 2 * compressed_size;
      if (NULL == (compressed2 = malloc(compressed2_size))) {
        success = JXL_FALSE;
        goto cleanup;
      }
      memcpy(compressed2, compressed, compressed_used);
      free(compressed);
      compressed = compressed2;
      compressed_size = compressed2_size;
      compressed2 = NULL;
      compressed_ptr = &compressed[compressed_used];
      compressed_available = compressed_size - compressed_used;
      fprintf(stderr, "Re-allocated to %zu bytes, %zu used.\n",
              compressed_size, compressed_used);
    }
  } while (process_result == JXL_ENC_NEED_MORE_OUTPUT);

  if (JXL_ENC_SUCCESS != process_result) {
    fprintf(stderr, "JxlEncoderProcessOutput failed\n");
    goto cleanup;
  }
  
  *compressed_out = compressed;
  compressed = NULL;
  *compressed_size_out = compressed_used;
  
 cleanup:
  if (compressed) {
    free(compressed);
    compressed = NULL;
  }
  if (compressed2) {
    free(compressed2);
    compressed2 = NULL;
  }
  return success;
}


JXL_BOOL write_jxl_file(const uint8_t* bytes,
                        size_t size,
                        const char* filename) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Could not open file for writing: %s\n", filename);
    return JXL_FALSE;
  }
  size_t bytes_written = fwrite(bytes, sizeof(uint8_t), size, file);
  if (bytes_written != size) {
    fprintf(stderr,
            "Failure: Wanted to write %zu bytes, "
            "but did write %zu bytes to file: %s\n",
            size, bytes_written, filename);
    return JXL_FALSE;
  }
  if (fclose(file)) {
    fprintf(stderr, "Could not close file: %s\n", filename);
    return JXL_FALSE;
  }
  return JXL_TRUE;
}
