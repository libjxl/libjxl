/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#include <jxl/compressed_icc.h>
#include <jxl/gain_map.h>
#include <jxl/memory_manager.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* These fixed bytes were generated once from synthetic data with the public
 * bundle and ICC helpers. This test calls only their decoder counterparts.
 * The ICC payload is opaque test data, not a color profile. */
static const uint8_t kSyntheticIccBytes[] = {
    0x00, 0x00, 0x00, 0x3c, 0x73, 0x79, 0x6e, 0x74, 0x68, 0x65, 0x74, 0x69,
    0x63, 0x2d, 0x49, 0x43, 0x43, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x2d, 0x64,
    0x61, 0x74, 0x61, 0x2d, 0x66, 0x6f, 0x72, 0x2d, 0x6c, 0x69, 0x62, 0x6a,
    0x78, 0x6c, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
    0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
};

static const uint8_t kGainMapBundleBytes[] = {
    0x00, 0x00, 0x03, 0x10, 0x20, 0x30, 0x03, 0x50, 0xb4, 0x00, 0x00,
    0x00, 0x00, 0x46, 0xb6, 0xc8, 0xf1, 0x7a, 0xd6, 0xe4, 0x3e, 0xc0,
    0x80, 0xcd, 0x5a, 0x55, 0x93, 0xa8, 0xbb, 0x5f, 0x4b, 0x9b, 0x5b,
    0x11, 0x11, 0x49, 0x49, 0xc4, 0x55, 0x10, 0xa5, 0x22, 0xeb, 0x7e,
    0x5f, 0xb4, 0xe0, 0x8f, 0x75, 0xae, 0x5e, 0x85, 0x61, 0xc9, 0x47,
    0x84, 0x8a, 0x88, 0x62, 0xbc, 0x26, 0x0a, 0x93, 0x08, 0xea, 0x9b,
    0x6c, 0x19, 0x9b, 0xb6, 0xeb, 0x87, 0x71, 0x32, 0xf3, 0xb2, 0x6e,
    0xfb, 0x61, 0x39, 0xb6, 0x7b, 0xf2, 0x00, 0xff, 0x00, 0x80,
};

static void* Allocate(void* opaque, size_t size) {
  (void)opaque;
  return malloc(size);
}

static void Release(void* opaque, void* address) {
  (void)opaque;
  free(address);
}

int main(void) {
  JxlMemoryManager memory_manager = {NULL, Allocate, Release};
  JxlGainMapBundle bundle = {0};
  uint8_t* icc_bytes = NULL;
  size_t icc_size = 0;
  size_t bytes_read = 0;
  int result = EXIT_FAILURE;

  if (!JxlGainMapReadBundle(&bundle, kGainMapBundleBytes,
                            sizeof(kGainMapBundleBytes), &bytes_read)) {
    fprintf(stderr, "JxlGainMapReadBundle failed\n");
    goto cleanup;
  }
  if (bytes_read != sizeof(kGainMapBundleBytes) || bundle.jhgm_version != 0 ||
      bundle.gain_map_metadata_size != 3 ||
      memcmp(bundle.gain_map_metadata, "\x10\x20\x30", 3) != 0 ||
      !bundle.has_color_encoding ||
      bundle.color_encoding.color_space != JXL_COLOR_SPACE_RGB ||
      bundle.color_encoding.white_point != JXL_WHITE_POINT_D65 ||
      bundle.color_encoding.primaries != JXL_PRIMARIES_SRGB ||
      bundle.color_encoding.transfer_function != JXL_TRANSFER_FUNCTION_LINEAR ||
      bundle.gain_map_size != 3 ||
      memcmp(bundle.gain_map, "\xff\x00\x80", 3) != 0) {
    fprintf(stderr, "gain map reader returned unexpected data\n");
    goto cleanup;
  }
  if (!JxlICCProfileDecode(&memory_manager, bundle.alt_icc, bundle.alt_icc_size,
                           &icc_bytes, &icc_size)) {
    fprintf(stderr, "JxlICCProfileDecode failed\n");
    goto cleanup;
  }
  if (icc_size != sizeof(kSyntheticIccBytes) ||
      memcmp(icc_bytes, kSyntheticIccBytes, sizeof(kSyntheticIccBytes)) != 0) {
    fprintf(stderr, "ICC decoder returned unexpected bytes\n");
    goto cleanup;
  }

  result = EXIT_SUCCESS;

cleanup:
  memory_manager.free(memory_manager.opaque, icc_bytes);
  return result;
}
