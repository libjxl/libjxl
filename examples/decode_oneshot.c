// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This C++ example decodes a JPEG XL image in one shot (all input bytes
// available at once). The example outputs the pixels and color information to a
// floating point image and an ICC profile on disk.

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include <inttypes.h>
#include <jxl/codestream_header.h>
#include <jxl/decode.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/types.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct FloatBuffer {
  float* data;
  size_t size;
};

struct ByteBuffer {
  uint8_t* data;
  size_t size;
};

/** Decodes JPEG XL image to floating point pixels and ICC Profile. Pixel are
 * stored as floating point, as interleaved RGBA (4 floating point values per
 * pixel), line per line from top to bottom.  Pixel values have nominal range
 * 0..1 but may go beyond this range for HDR or wide gamut. The ICC profile
 * describes the color format of the pixel data.
 */
JXL_BOOL DecodeJpegXlOneShot(const uint8_t* jxl, size_t size,
                             struct FloatBuffer* pixels, size_t* xsize,
                             size_t* ysize, struct ByteBuffer* icc_profile) {
  // Multi-threaded parallel runner.
  void* runner = JxlResizableParallelRunnerCreate(NULL);
  JXL_BOOL ret = JXL_FALSE;

  JxlDecoder* dec = JxlDecoderCreate(NULL);
  if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(
                             dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING |
                                      JXL_DEC_FULL_IMAGE)) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    goto cleanup;
  }

  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetParallelRunner(dec, JxlResizableParallelRunner, runner)) {
    fprintf(stderr, "JxlDecoderSetParallelRunner failed\n");
    goto cleanup;
  }

  JxlBasicInfo info;
  JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};

  JxlDecoderSetInput(dec, jxl, size);
  JxlDecoderCloseInput(dec);

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);

    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Decoder error\n");
      goto cleanup;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      fprintf(stderr, "Error, already provided all input\n");
      goto cleanup;
    } else if (status == JXL_DEC_BASIC_INFO) {
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec, &info)) {
        fprintf(stderr, "JxlDecoderGetBasicInfo failed\n");
        goto cleanup;
      }
      *xsize = info.xsize;
      *ysize = info.ysize;
      JxlResizableParallelRunnerSetThreads(
          runner,
          JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      // Get the ICC color profile of the pixel data
      size_t icc_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderGetICCProfileSize(dec, JXL_COLOR_PROFILE_TARGET_DATA,
                                      &icc_size)) {
        fprintf(stderr, "JxlDecoderGetICCProfileSize failed\n");
        goto cleanup;
      }
      icc_profile->size = icc_size;
      icc_profile->data = realloc(icc_profile->data, icc_size);
      if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
                                 dec, JXL_COLOR_PROFILE_TARGET_DATA,
                                 icc_profile->data, icc_profile->size)) {
        fprintf(stderr, "JxlDecoderGetColorAsICCProfile failed\n");
        goto cleanup;
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderImageOutBufferSize(dec, &format, &buffer_size)) {
        fprintf(stderr, "JxlDecoderImageOutBufferSize failed\n");
        goto cleanup;
      }
      if (buffer_size != *xsize * *ysize * 16) {
        fprintf(stderr, "Invalid out buffer size %" PRIu64 " %" PRIu64 "\n",
                (uint64_t)buffer_size, (uint64_t)(*xsize * *ysize * 16));
        goto cleanup;
      }
      pixels->size = *xsize * *ysize * 4;
      size_t pixels_buffer_size = pixels->size * sizeof(float);
      pixels->data = realloc(pixels->data, pixels_buffer_size);
      if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec, &format,
                                                         pixels->data,
                                                         pixels_buffer_size)) {
        fprintf(stderr, "JxlDecoderSetImageOutBuffer failed\n");
        goto cleanup;
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      // Nothing to do. Do not yet return. If the image is an animation, more
      // full frames may be decoded. This example only keeps the last one.
    } else if (status == JXL_DEC_SUCCESS) {
      // All decoding successfully finished.
      // It's not required to call JxlDecoderReleaseInput(dec.get()) here since
      // the decoder will be destroyed.
      ret = JXL_TRUE;
      goto cleanup;
    } else {
      fprintf(stderr, "Unknown decoder status\n");
      goto cleanup;
    }
  }

cleanup:
  JxlDecoderDestroy(dec);
  JxlResizableParallelRunnerDestroy(runner);
  return ret;
}

/** Writes to .pfm file (Portable FloatMap). Gimp, tev viewer and ImageMagick
 * support viewing this format.
 * The input pixels are given as 32-bit floating point with 4-channel RGBA.
 * The alpha channel will not be written since .pfm does not support it.
 */
JXL_BOOL WritePFM(const char* filename, const float* pixels, size_t xsize,
                  size_t ysize) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Could not open %s for writing", filename);
    return JXL_FALSE;
  }
  uint32_t endian_test = 1;
  uint8_t little_endian[4];
  memcpy(little_endian, &endian_test, 4);

  fprintf(file, "PF\n%d %d\n%s\n", (int)xsize, (int)ysize,
          little_endian[0] ? "-1.0" : "1.0");
  for (int y = ysize - 1; y >= 0; y--) {
    for (size_t x = 0; x < xsize; x++) {
      for (size_t c = 0; c < 3; c++) {
        const float* f = &pixels[(y * xsize + x) * 4 + c];
        fwrite(f, 4, 1, file);
      }
    }
  }
  if (fclose(file) != 0) {
    return JXL_FALSE;
  }
  return JXL_TRUE;
}

JXL_BOOL LoadFile(const char* filename, struct ByteBuffer* out) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    return JXL_FALSE;
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return JXL_FALSE;
  }

  long size = ftell(file);
  // Avoid invalid file or directory.
  if (size >= LONG_MAX || size < 0) {
    fclose(file);
    return JXL_FALSE;
  }

  if (fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return JXL_FALSE;
  }

  out->size = size;
  out->data = realloc(out->data, size);
  size_t readsize = fread(out->data, 1, size, file);
  if (fclose(file) != 0) {
    return JXL_FALSE;
  }

  return readsize == (size_t)size;
}

JXL_BOOL WriteFile(const char* filename, const uint8_t* data, size_t size) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Could not open %s for writing", filename);
    return JXL_FALSE;
  }
  fwrite(data, 1, size, file);
  if (fclose(file) != 0) {
    return JXL_FALSE;
  }
  return JXL_TRUE;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: %s <jxl> <pfm> <icc>\n"
            "Where:\n"
            "  jxl = input JPEG XL image filename\n"
            "  pfm = output Portable FloatMap image filename\n"
            "  icc = output ICC color profile filename\n"
            "Output files will be overwritten.\n",
            argv[0]);
    return 1;
  }

  const char* jxl_filename = argv[1];
  const char* pfm_filename = argv[2];
  const char* icc_filename = argv[3];

  struct ByteBuffer jxl = {};
  if (!LoadFile(jxl_filename, &jxl)) {
    fprintf(stderr, "couldn't load %s\n", jxl_filename);
    free(jxl.data);
    return 1;
  }

  struct FloatBuffer pixels = {};
  struct ByteBuffer icc_profile = {};
  size_t xsize = 0, ysize = 0;
  int ret = 1;
  if (!DecodeJpegXlOneShot(jxl.data, jxl.size, &pixels, &xsize, &ysize,
                           &icc_profile)) {
    fprintf(stderr, "Error while decoding the jxl file\n");
    goto cleanup;
  }
  if (!WritePFM(pfm_filename, pixels.data, xsize, ysize)) {
    fprintf(stderr, "Error while writing the PFM image file\n");
    goto cleanup;
  }
  if (!WriteFile(icc_filename, icc_profile.data, icc_profile.size)) {
    fprintf(stderr, "Error while writing the ICC profile file\n");
    goto cleanup;
  }
  printf("Successfully wrote %s and %s\n", pfm_filename, icc_filename);
  ret = 0;
cleanup:
  free(icc_profile.data);
  free(pixels.data);
  free(jxl.data);
  return ret;
}
