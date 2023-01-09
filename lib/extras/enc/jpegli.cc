// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/enc/jpegli.h"

#include <setjmp.h>
#include <stdint.h>

#include "lib/jpegli/encode.h"

namespace jxl {
namespace extras {

namespace {

void MyErrorExit(j_common_ptr cinfo) {
  jmp_buf* env = static_cast<jmp_buf*>(cinfo->client_data);
  (*cinfo->err->output_message)(cinfo);
  jpegli_destroy_compress(reinterpret_cast<j_compress_ptr>(cinfo));
  longjmp(*env, 1);
}

}  // namespace

Status EncodeJpeg(const ImageBundle& input, const JpegSettings& jpeg_settings,
                  ThreadPool* pool, std::vector<uint8_t>* compressed) {
  // We need to declare all the non-trivial destructor local variables before
  // the call to setjmp().
  PaddedBytes icc;
  std::vector<uint8_t> pixels;
  unsigned char* output_buffer = nullptr;
  unsigned long output_size = 0;

  const auto try_catch_block = [&]() -> bool {
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    jmp_buf env;
    cinfo.err = jpegli_std_error(&jerr);
    jerr.error_exit = &MyErrorExit;
    if (setjmp(env)) {
      if (output_buffer) free(output_buffer);
      return false;
    }
    cinfo.client_data = static_cast<void*>(&env);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &output_buffer, &output_size);
    cinfo.image_width = input.xsize();
    cinfo.image_height = input.ysize();
    cinfo.input_components = input.IsGray() ? 1 : 3;
    cinfo.in_color_space = input.IsGray() ? JCS_GRAYSCALE : JCS_RGB;
    if (jpeg_settings.xyb) {
      jpegli_set_xyb_mode(&cinfo);
    }
    jpegli_set_defaults(&cinfo);
    jpegli_set_distance(&cinfo, jpeg_settings.distance);
    jpegli_start_compress(&cinfo, TRUE);
    icc = input.c_current().ICC();
    if (!icc.empty()) {
      jpegli_write_icc_profile(&cinfo, icc.data(), icc.size());
    }
    jpegli_set_input_format(&cinfo, JPEGLI_TYPE_FLOAT, JPEGLI_NATIVE_ENDIAN);
    size_t stride = input.xsize() * cinfo.input_components * 4;
    pixels.resize(stride);
    for (size_t y = 0; y < input.ysize(); ++y) {
      const float* JXL_RESTRICT row0 = input.color().ConstPlaneRow(0, y);
      const float* JXL_RESTRICT row1 = input.color().ConstPlaneRow(1, y);
      const float* JXL_RESTRICT row2 = input.color().ConstPlaneRow(2, y);
      for (size_t x = 0; x < input.xsize(); ++x) {
        memcpy(&pixels[x * 12 + 0], row0 + x, sizeof(float));
        memcpy(&pixels[x * 12 + 4], row1 + x, sizeof(float));
        memcpy(&pixels[x * 12 + 8], row2 + x, sizeof(float));
      }
      JSAMPROW row[] = {pixels.data()};
      jpegli_write_scanlines(&cinfo, row, 1);
    }
    jpegli_finish_compress(&cinfo);
    jpegli_destroy_compress(&cinfo);
    compressed->resize(output_size);
    std::copy_n(output_buffer, output_size, compressed->data());
    std::free(output_buffer);
    return true;
  };
  return try_catch_block();
}

}  // namespace extras
}  // namespace jxl
