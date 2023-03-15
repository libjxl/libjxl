// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/encode.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/test_utils.h"
#include "lib/jpegli/testing.h"
#include "lib/jxl/sanitizers.h"

namespace jpegli {
namespace {

TEST(ErrorHandlingTest, MinimalSuccess) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  {
    jpeg_compress_struct cinfo;
    const auto try_catch_block = [&]() -> bool {
      ERROR_HANDLER_SETUP(jpegli);
      jpegli_create_compress(&cinfo);
      jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
      cinfo.image_width = 1;
      cinfo.image_height = 1;
      cinfo.input_components = 1;
      jpegli_set_defaults(&cinfo);
      jpegli_start_compress(&cinfo, TRUE);
      JSAMPLE image[1] = {0};
      JSAMPROW row[] = {image};
      jpegli_write_scanlines(&cinfo, row, 1);
      jpegli_finish_compress(&cinfo);
      return true;
    };
    EXPECT_TRUE(try_catch_block());
    jpegli_destroy_compress(&cinfo);
  }
  {
    jpeg_decompress_struct cinfo = {};
    const auto try_catch_block = [&]() -> bool {
      ERROR_HANDLER_SETUP(jpeg);
      jpeg_create_decompress(&cinfo);
      jpeg_mem_src(&cinfo, buffer, buffer_size);
      jpeg_read_header(&cinfo, TRUE);
      EXPECT_EQ(1, cinfo.image_width);
      EXPECT_EQ(1, cinfo.image_height);
      jpeg_start_decompress(&cinfo);
      JSAMPLE image[1];
      JSAMPROW row[] = {image};
      jpeg_read_scanlines(&cinfo, row, 1);
      jxl::msan::UnpoisonMemory(image, 1);
      EXPECT_EQ(0, image[0]);
      jpeg_finish_decompress(&cinfo);
      return true;
    };
    EXPECT_TRUE(try_catch_block());
    jpeg_destroy_decompress(&cinfo);
  }
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NoDestination) {
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
}

TEST(ErrorHandlingTest, NoImageDimensions) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, ImageTooBig) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 100000;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NoInputComponents) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, TooManyInputComponents) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1000;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NoSetDefaults) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[1] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NoStartCompress) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    JSAMPLE image[1] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NoWriteScanlines) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NoWriteAllScanlines) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 2;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[1] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidQuantValue) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    cinfo.quant_tbl_ptrs[0] = jpegli_alloc_quant_table((j_common_ptr)&cinfo);
    for (size_t k = 0; k < DCTSIZE2; ++k) {
      cinfo.quant_tbl_ptrs[0]->quantval[k] = 0;
    }
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[1] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidQuantTableIndex) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    cinfo.comp_info[0].quant_tbl_no = 3;
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[1] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NumberOfComponentsMismatch1) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    cinfo.num_components = 100;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NumberOfComponentsMismatch2) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    cinfo.num_components = 2;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NumberOfComponentsMismatch3) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    cinfo.num_components = 2;
    cinfo.comp_info[1].h_samp_factor = cinfo.comp_info[1].v_samp_factor = 1;
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[1] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NumberOfComponentsMismatch4) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_RGB;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[1] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NumberOfComponentsMismatch5) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_GRAYSCALE;
    jpegli_set_defaults(&cinfo);
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[3] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NumberOfComponentsMismatch6) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpegli_set_defaults(&cinfo);
    cinfo.num_components = 2;
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[3] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidColorTransform) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_YCbCr;
    jpegli_set_defaults(&cinfo);
    cinfo.jpeg_color_space = JCS_RGB;
    jpegli_start_compress(&cinfo, TRUE);
    JSAMPLE image[3] = {0};
    JSAMPROW row[] = {image};
    jpegli_write_scanlines(&cinfo, row, 1);
    jpegli_finish_compress(&cinfo);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, DuplicateComponentIds) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    jpegli_set_defaults(&cinfo);
    cinfo.comp_info[0].component_id = 0;
    cinfo.comp_info[1].component_id = 0;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidComponentIndex) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    jpegli_set_defaults(&cinfo);
    cinfo.comp_info[0].component_index = 17;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, ArithmeticCoding) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    jpegli_set_defaults(&cinfo);
    cinfo.arith_code = TRUE;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, CCIR601Sampling) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    jpegli_set_defaults(&cinfo);
    cinfo.CCIR601_sampling = TRUE;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript1) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {{1, {0}, 0, 63, 0, 0}};  //
    cinfo.scan_info = kScript;
    cinfo.num_scans = 0;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript2) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {{2, {0, 1}, 0, 63, 0, 0}};  //
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript3) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {{5, {0}, 0, 63, 0, 0}};  //
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript4) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 2;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {{2, {0, 0}, 0, 63, 0, 0}};  //
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript5) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 2;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {{2, {1, 0}, 0, 63, 0, 0}};  //
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript6) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {{1, {0}, 0, 64, 0, 0}};  //
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript7) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {{1, {0}, 2, 1, 0, 0}};  //
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript8) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 2;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {
        {1, {0}, 0, 63, 0, 0}, {1, {1}, 0, 0, 0, 0}, {1, {1}, 1, 63, 0, 0}  //
    };
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript9) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {
        {1, {0}, 0, 1, 0, 0}, {1, {0}, 2, 63, 0, 0},  //
    };
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript10) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 2;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {
        {2, {0, 1}, 0, 0, 0, 0}, {2, {0, 1}, 1, 63, 0, 0}  //
    };
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript11) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {
        {1, {0}, 1, 63, 0, 0}, {1, {0}, 0, 0, 0, 0}  //
    };
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript12) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {
        {1, {0}, 0, 0, 10, 1}, {1, {0}, 0, 0, 1, 0}, {1, {0}, 1, 63, 0, 0}  //
    };
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, InvalidScanScript13) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    static constexpr jpeg_scan_info kScript[] = {
        {1, {0}, 0, 0, 0, 2},
        {1, {0}, 0, 0, 1, 0},
        {1, {0}, 0, 0, 2, 1},  //
        {1, {0}, 1, 63, 0, 0}  //
    };
    cinfo.scan_info = kScript;
    cinfo.num_scans = ARRAY_SIZE(kScript);
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, RestartIntervalTooBig) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 1;
    jpegli_set_defaults(&cinfo);
    cinfo.restart_interval = 1000000;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, SamplingFactorTooBig) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    jpegli_set_defaults(&cinfo);
    cinfo.comp_info[0].h_samp_factor = 5;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

TEST(ErrorHandlingTest, NonIntegralSamplingRatio) {
  uint8_t* buffer = nullptr;
  unsigned long buffer_size = 0;
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    jpegli_mem_dest(&cinfo, &buffer, &buffer_size);
    cinfo.image_width = 1;
    cinfo.image_height = 1;
    cinfo.input_components = 3;
    jpegli_set_defaults(&cinfo);
    cinfo.comp_info[0].h_samp_factor = 3;
    cinfo.comp_info[1].h_samp_factor = 2;
    jpegli_start_compress(&cinfo, TRUE);
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
  if (buffer) free(buffer);
}

constexpr const char* kAddOnTable[] = {"First message",
                                       "Second message with int param %d",
                                       "Third message with string param %s"};

TEST(ErrorHandlingTest, AddOnTableNoParam) {
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    cinfo.err->addon_message_table = kAddOnTable;
    cinfo.err->first_addon_message = 10000;
    cinfo.err->last_addon_message = 10002;
    cinfo.err->msg_code = 10000;
    (*cinfo.err->error_exit)(reinterpret_cast<j_common_ptr>(&cinfo));
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
}

TEST(ErrorHandlingTest, AddOnTableIntParam) {
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    cinfo.err->addon_message_table = kAddOnTable;
    cinfo.err->first_addon_message = 10000;
    cinfo.err->last_addon_message = 10002;
    cinfo.err->msg_code = 10001;
    cinfo.err->msg_parm.i[0] = 17;
    (*cinfo.err->error_exit)(reinterpret_cast<j_common_ptr>(&cinfo));
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
}

TEST(ErrorHandlingTest, AddOnTableNoStringParam) {
  jpeg_compress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_compress(&cinfo);
    cinfo.err->addon_message_table = kAddOnTable;
    cinfo.err->first_addon_message = 10000;
    cinfo.err->last_addon_message = 10002;
    cinfo.err->msg_code = 10002;
    memcpy(cinfo.err->msg_parm.s, "MESSAGE PARAM", 14);
    (*cinfo.err->error_exit)(reinterpret_cast<j_common_ptr>(&cinfo));
    return true;
  };
  EXPECT_FALSE(try_catch_block());
  jpegli_destroy_compress(&cinfo);
}

}  // namespace
}  // namespace jpegli
