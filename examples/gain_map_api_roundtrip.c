/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 *
 * This is a linkage and serialization smoke test with synthetic payloads. It
 * does not decode image data or use a valid color profile.
 */

#include <jxl/color_encoding.h>
#include <jxl/compressed_icc.h>
#include <jxl/encode.h>
#include <jxl/gain_map.h>
#include <jxl/memory_manager.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void* Allocate(void* opaque, size_t size) {
  (void)opaque;
  return malloc(size);
}

static void Release(void* opaque, void* address) {
  (void)opaque;
  free(address);
}

int main(void) {
  /* The ICC codec preserves bytes; these are not a valid color profile. */
  static const uint8_t icc_data[] = {
      0x00, 0x00, 0x00, 0x3c, 's',  'y',  'n',  't',  'h',  'e',  't',  'i',
      'c',  '-',  'I',  'C',  'C',  '-',  't',  'e',  's',  't',  '-',  'd',
      'a',  't',  'a',  '-',  'f',  'o',  'r',  '-',  'l',  'i',  'b',  'j',
      'x',  'l',  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
      0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
  };
  static const uint8_t metadata[] = {0x00, 0x01, 0x02, 0x03, 0x7f, 0xff};
  static const uint8_t gain_map[] = {0xff, 0x00, 0x80, 0x40, 0x20};

  JxlMemoryManager memory_manager = {NULL, Allocate, Release};
  uint8_t* compressed_icc = NULL;
  size_t compressed_icc_size = 0;
  uint8_t* decoded_icc = NULL;
  size_t decoded_icc_size = 0;
  uint8_t* bundle_buffer = NULL;
  size_t bundle_size = 0;
  size_t bytes_written = 0;
  size_t bytes_read = 0;
  int result = EXIT_FAILURE;
  JxlGainMapBundle input_bundle = {0};
  JxlGainMapBundle output_bundle = {0};

  if (!JxlICCProfileEncode(&memory_manager, icc_data, sizeof(icc_data),
                           &compressed_icc, &compressed_icc_size)) {
    fprintf(stderr, "JxlICCProfileEncode failed\n");
    goto cleanup;
  }
  if (compressed_icc_size > UINT32_MAX) {
    fprintf(stderr, "compressed ICC data does not fit in a gain map bundle\n");
    goto cleanup;
  }

  input_bundle.jhgm_version = 0;
  input_bundle.gain_map_metadata_size = sizeof(metadata);
  input_bundle.gain_map_metadata = metadata;
  input_bundle.has_color_encoding = JXL_TRUE;
  JxlColorEncodingSetToLinearSRGB(&input_bundle.color_encoding, JXL_FALSE);
  input_bundle.alt_icc_size = (uint32_t)compressed_icc_size;
  input_bundle.alt_icc = compressed_icc;
  input_bundle.gain_map_size = sizeof(gain_map);
  input_bundle.gain_map = gain_map;

  if (!JxlGainMapGetBundleSize(&input_bundle, &bundle_size)) {
    fprintf(stderr, "JxlGainMapGetBundleSize failed\n");
    goto cleanup;
  }
  bundle_buffer = (uint8_t*)malloc(bundle_size);
  if (bundle_buffer == NULL) {
    fprintf(stderr, "allocating gain map bundle failed\n");
    goto cleanup;
  }
  if (!JxlGainMapWriteBundle(&input_bundle, bundle_buffer, bundle_size,
                             &bytes_written)) {
    fprintf(stderr, "JxlGainMapWriteBundle failed\n");
    goto cleanup;
  }
  if (bytes_written != bundle_size) {
    fprintf(stderr, "JxlGainMapWriteBundle wrote an unexpected size\n");
    goto cleanup;
  }
  if (!JxlGainMapReadBundle(&output_bundle, bundle_buffer, bundle_size,
                            &bytes_read)) {
    fprintf(stderr, "JxlGainMapReadBundle failed\n");
    goto cleanup;
  }
  if (bytes_read != bundle_size || output_bundle.jhgm_version != 0 ||
      output_bundle.gain_map_metadata_size != sizeof(metadata) ||
      memcmp(output_bundle.gain_map_metadata, metadata, sizeof(metadata)) !=
          0 ||
      !output_bundle.has_color_encoding ||
      output_bundle.color_encoding.color_space !=
          input_bundle.color_encoding.color_space ||
      output_bundle.color_encoding.white_point !=
          input_bundle.color_encoding.white_point ||
      output_bundle.color_encoding.primaries !=
          input_bundle.color_encoding.primaries ||
      output_bundle.color_encoding.transfer_function !=
          input_bundle.color_encoding.transfer_function ||
      output_bundle.color_encoding.rendering_intent !=
          input_bundle.color_encoding.rendering_intent ||
      output_bundle.alt_icc_size != compressed_icc_size ||
      memcmp(output_bundle.alt_icc, compressed_icc, compressed_icc_size) != 0 ||
      output_bundle.gain_map_size != sizeof(gain_map) ||
      memcmp(output_bundle.gain_map, gain_map, sizeof(gain_map)) != 0) {
    fprintf(stderr, "gain map bundle roundtrip mismatch\n");
    goto cleanup;
  }
  if (!JxlICCProfileDecode(&memory_manager, output_bundle.alt_icc,
                           output_bundle.alt_icc_size, &decoded_icc,
                           &decoded_icc_size)) {
    fprintf(stderr, "ICC byte-stream decode failed\n");
    goto cleanup;
  }
  if (decoded_icc_size != sizeof(icc_data) ||
      memcmp(decoded_icc, icc_data, sizeof(icc_data)) != 0) {
    fprintf(stderr, "ICC byte-stream roundtrip mismatch\n");
    goto cleanup;
  }

  result = EXIT_SUCCESS;

cleanup:
  memory_manager.free(memory_manager.opaque, decoded_icc);
  memory_manager.free(memory_manager.opaque, compressed_icc);
  free(bundle_buffer);
  return result;
}
