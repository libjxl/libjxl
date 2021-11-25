/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
*/

/* Program to test that we can use the libjpegxl public C API for encoding. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jxl/codestream_header.h"
#include "jxl/color_encoding.h"
#include "jxl/encode.h"
#include "jxl/types.h"


#define DDD_WIDTH 640
#define DDD_HEIGHT 480


/* This corresponds to: lib/jxl/encode.cc:JxlEncoderInitBasicInfo
   in the C++ API.
 */
static void JxlBasicInfoInit(JxlBasicInfo* info) {
  info->have_container = JXL_FALSE;
  info->xsize = 0;
  info->ysize = 0;
  info->bits_per_sample = 8;
  info->exponent_bits_per_sample = 0;
  info->intensity_target = 255.f;
  info->min_nits = 0.f;
  info->relative_to_max_display = JXL_FALSE;
  info->linear_below = 0.f;
  info->uses_original_profile = JXL_FALSE;
  info->have_preview = JXL_FALSE;
  info->have_animation = JXL_FALSE;
  info->orientation = JXL_ORIENT_IDENTITY;
  info->num_color_channels = 3;
  info->num_extra_channels = 0;
  info->alpha_bits = 0;
  info->alpha_exponent_bits = 0;
  info->alpha_premultiplied = JXL_FALSE;
  info->preview.xsize = 0;
  info->preview.ysize = 0;
  info->animation.tps_numerator = 10;
  info->animation.tps_denominator = 1;
  info->animation.num_loops = 0;
  info->animation.have_timecodes = JXL_FALSE;
}

/* Fetches the encoded data from `jxl_encoder` and sets
   `*compressed_out` to a pointer to that freshly allocated data,
   transfering ownership to caller. Data size is returned via
   `compressed_size_out`. On success, returns JXL_TRUE.
   On failure, returns JXL_FALSE and sets `*compressed_out`
   to NULL.
 */
static JXL_BOOL fetch_encoded(JxlEncoder *jxl_encoder,
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
  *compressed_size_out = compressed_size;
  
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


static JXL_BOOL write_file(const uint8_t* bytes,
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


int main(int argc, char **argv) {
  int success = EXIT_SUCCESS;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s {output_jxl_filename}\n", argv[0]);
    return EXIT_FAILURE;
  }
  
  fprintf(stderr, "Creating encoder.\n");
  JxlEncoder *jxl_encoder = JxlEncoderCreate(NULL);

  fprintf(stderr, "Encoder is at %p.\n", (void *)jxl_encoder);


  /* TODO(tfish): Replace allocating dummy-data here with actual image data. */
  float* dummy_src = NULL;
  uint8_t *compressed = NULL;
  size_t compressed_size;

  if (NULL == (dummy_src = calloc(DDD_WIDTH * DDD_HEIGHT * 3,
                                   sizeof(float)))) {
    goto cleanup;
  }
  fprintf(stderr, "Allocated dummy data at %p.\n", (void *)dummy_src);
  
  JxlBasicInfo jxl_basic_info;
  JxlBasicInfoInit(&jxl_basic_info);
  jxl_basic_info.xsize = DDD_WIDTH;
  jxl_basic_info.ysize = DDD_HEIGHT;
  jxl_basic_info.bits_per_sample = 32;
  jxl_basic_info.exponent_bits_per_sample = 8;
  jxl_basic_info.uses_original_profile = JXL_FALSE;
  if (JXL_ENC_SUCCESS != JxlEncoderSetBasicInfo(jxl_encoder, &jxl_basic_info)) {
    fprintf(stderr, "JxlEncoderSetBasicInfo failed\n");
    goto cleanup;
  }
  fprintf(stderr, "JxlEncoderSetBasicInfo done.\n");
  
  JxlColorEncoding jxl_color_encoding;
  JxlColorEncodingSetToSRGB(&jxl_color_encoding,/*is_gray=*/JXL_FALSE);
  if (JXL_ENC_SUCCESS != JxlEncoderSetColorEncoding(jxl_encoder,
                                                    &jxl_color_encoding)) {
    fprintf(stderr, "JxlEncoderSetColorEncoding failed\n");
    goto cleanup;
  }
  fprintf(stderr, "JxlEncoderSetColorEncoding done.\n");
  
  JxlPixelFormat pixel_format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  /* This is owned by the encoder. */
  JxlEncoderOptions *jxl_encoder_options = JxlEncoderOptionsCreate(
      jxl_encoder, NULL);
  fprintf(stderr, "JxlEncoderOptionsCreate done.\n");
  
  if (JXL_ENC_SUCCESS !=
      JxlEncoderAddImageFrame(jxl_encoder_options,
                              &pixel_format, (void*)dummy_src,
                              sizeof(float) * DDD_WIDTH * DDD_HEIGHT * 3)) {
    fprintf(stderr, "JxlEncoderAddImageFrame failed\n");
    goto cleanup;
  }

  fprintf(stderr, "JxlEncoderAddImageFrame done.\n");

  if (!fetch_encoded(jxl_encoder, &compressed, &compressed_size)) {
    goto cleanup;
  }

  if(!write_file(compressed, compressed_size, argv[1])) {
    fprintf(stderr, "Writing output file failed: %s\n", argv[1]);
    success = EXIT_FAILURE;
  }
  
 cleanup:
  if (dummy_src) {
    free(dummy_src);
    dummy_src = NULL;
  }
  if (compressed) {
    free(compressed);
    compressed = NULL;
  }
  if (jxl_encoder) {
    printf("Destroying encoder at %p.\n", (void *)jxl_encoder);
    JxlEncoderDestroy(jxl_encoder);
    jxl_encoder = NULL;
  }
  return success;
}
