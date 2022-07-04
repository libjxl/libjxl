// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstring>
#include <memory>
#include <vector>

#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/thread_parallel_runner_cxx.h"

#if !defined(__wasm__)
#include "lib/jxl/base/file_io.h"
#endif

namespace {

struct DecoderInstance {
  uint32_t width = 0;
  uint32_t height = 0;
  uint8_t* pixels = nullptr;
  uint32_t color_space = 0;

  size_t pixels_size = 0;
  bool want_sdr;
  uint32_t display_nits;
  JxlPixelFormat format;
  JxlDecoderPtr decoder;
  JxlThreadParallelRunnerPtr thread_pool;

  std::vector<uint8_t> tail;
};

}  // namespace

extern "C" {

void* jxlCreateInstance(bool want_sdr, uint32_t display_nits) {
  DecoderInstance* instance = new DecoderInstance();
  instance->want_sdr = want_sdr;
  instance->display_nits = display_nits;
  JxlDataType storageFormat = want_sdr ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16;
  instance->format = {4, storageFormat, JXL_NATIVE_ENDIAN, 0};
  instance->decoder = JxlDecoderMake(nullptr);

  JxlDecoder* dec = instance->decoder.get();

  auto report_error = [&](uint32_t code, const char* text) {
    fprintf(stderr, "%s\n", text);
    // instance->result = code;
    return instance;
  };

  instance->thread_pool = JxlThreadParallelRunnerMake(nullptr, 4);
  void* runner = instance->thread_pool.get();

  auto status =
      JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner);

  if (status != JXL_DEC_SUCCESS) {
    return report_error(1, "JxlDecoderSetParallelRunner failed");
  }

  status = JxlDecoderSubscribeEvents(
      dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE |
               JXL_DEC_FRAME_PROGRESSION);
  if (JXL_DEC_SUCCESS != status) {
    return report_error(2, "JxlDecoderSubscribeEvents failed");
  }

  status = JxlDecoderSetProgressiveDetail(dec, kPasses);
  if (JXL_DEC_SUCCESS != status) {
    return report_error(3, "JxlDecoderSetProgressiveDetail failed");
  }
  return instance;
}

void jxlDestroyInstance(void* opaque_instance) {
  if (opaque_instance == nullptr) return;
  DecoderInstance* instance =
      reinterpret_cast<DecoderInstance*>(opaque_instance);
  if (instance->pixels) {
    free(instance->pixels);
  }
  delete instance;
}

uint32_t jxlProcessInput(void* opaque_instance, const uint8_t* input,
                         size_t input_size) {
  if (opaque_instance == nullptr) return static_cast<uint32_t>(-1);
  DecoderInstance* instance =
      reinterpret_cast<DecoderInstance*>(opaque_instance);
  JxlDecoder* dec = instance->decoder.get();

  auto report_error = [&](int code, const char* text) {
    fprintf(stderr, "%s\n", text);
    // instance->result = code;
    return static_cast<uint32_t>(code);
  };

  std::vector<uint8_t>& tail = instance->tail;
  if (!tail.empty()) {
    tail.reserve(tail.size() + input_size);
    tail.insert(tail.end(), input, input + input_size);
    input = tail.data();
    input_size = tail.size();
  }

  auto status = JxlDecoderSetInput(dec, input, input_size);
  if (JXL_DEC_SUCCESS != status) {
    return report_error(-2, "JxlDecoderSetInput failed");
  }

  auto release_input = [&]() {
    size_t unused_input = JxlDecoderReleaseInput(dec);
    if (unused_input == 0) {
      tail.clear();
      return;
    }
    if (tail.empty()) {
      tail.insert(tail.end(), input + input_size - unused_input,
                  input + input_size);
    } else {
      memmove(tail.data(), tail.data() + tail.size() - unused_input,
              unused_input);
      tail.resize(unused_input);
    }
  };

  while (true) {
    status = JxlDecoderProcessInput(dec);
    if (JXL_DEC_SUCCESS == status) {
      release_input();
      return 0;  // ¯\_(ツ)_/¯
    } else if (JXL_DEC_FRAME_PROGRESSION == status) {
      release_input();
      return 1;  // ready to flush; client will decide whether it is necessary
    } else if (JXL_DEC_NEED_MORE_INPUT == status) {
      release_input();
      return 2;
    } else if (JXL_DEC_FULL_IMAGE == status) {
      release_input();
      return 0;  // final image is ready
    } else if (JXL_DEC_BASIC_INFO == status) {
      JxlBasicInfo info;
      status = JxlDecoderGetBasicInfo(dec, &info);
      if (status != JXL_DEC_SUCCESS) {
        release_input();
        return report_error(-4, "JxlDecoderGetBasicInfo failed");
      }
      instance->width = info.xsize;
      instance->height = info.ysize;
      status = JxlDecoderImageOutBufferSize(dec, &instance->format,
                                            &instance->pixels_size);
      if (status != JXL_DEC_SUCCESS) {
        release_input();
        return report_error(-6, "JxlDecoderImageOutBufferSize failed");
      }
      if (instance->pixels) {
        release_input();
        return report_error(-7, "Tried to realloc pixels");
      }
      instance->pixels =
          reinterpret_cast<uint8_t*>(malloc(instance->pixels_size));
    } else if (JXL_DEC_NEED_IMAGE_OUT_BUFFER == status) {
      if (!instance->pixels) {
        release_input();
        return report_error(-8, "Out buffer not allocated");
      }
      status = JxlDecoderSetImageOutBuffer(
          dec, &instance->format, instance->pixels, instance->pixels_size);
      if (status != JXL_DEC_SUCCESS) {
        release_input();
        return report_error(-9, "JxlDecoderSetImageOutBuffer failed");
      }
    } else if (JXL_DEC_COLOR_ENCODING == status) {
      JxlColorEncoding color_encoding;
      color_encoding.color_space = JXL_COLOR_SPACE_RGB;
      color_encoding.white_point = JXL_WHITE_POINT_D65;
      color_encoding.primaries =
          instance->want_sdr ? JXL_PRIMARIES_SRGB : JXL_PRIMARIES_2100;
      color_encoding.transfer_function = instance->want_sdr
                                             ? JXL_TRANSFER_FUNCTION_SRGB
                                             : JXL_TRANSFER_FUNCTION_PQ;
      color_encoding.rendering_intent = JXL_RENDERING_INTENT_PERCEPTUAL;
      status = JxlDecoderSetPreferredColorProfile(dec, &color_encoding);
      if (status != JXL_DEC_SUCCESS) {
        release_input();
        return report_error(-5, "JxlDecoderSetPreferredColorProfile failed");
      }
    } else {
      release_input();
      return report_error(-3, "Unexpected decoder status");
    }
  }

  release_input();
  return 0;
}

uint32_t jxlFlush(void* opaque_instance) {
  if (opaque_instance == nullptr) return static_cast<uint32_t>(-1);
  DecoderInstance* instance =
      reinterpret_cast<DecoderInstance*>(opaque_instance);
  JxlDecoder* dec = instance->decoder.get();

  auto report_error = [&](int code, const char* text) {
    fprintf(stderr, "%s\n", text);
    // instance->result = code;
    return static_cast<uint32_t>(code);
  };

  if (!instance->pixels) {
    return report_error(-2, "Not ready to flush");
  }

  auto status = JxlDecoderFlushImage(dec);
  if (status != JXL_DEC_SUCCESS) {
    return report_error(-3, "Failed to flush");
  }

  return 0;
}

#if !defined(__wasm__)
int main(int argc, char* argv[]) {
  std::vector<uint8_t> data;
  JXL_RETURN_IF_ERROR(jxl::ReadFile(argv[1], &data));
  fprintf(stderr, "File size: %d\n", (int)data.size());

  void* instance = jxlCreateInstance(true, 100);
  uint32_t status = jxlProcessInput(instance, data.data(), data.size());
  fprintf(stderr, "Process result: %d\n", status);
  jxlFlush(instance);
  status = jxlProcessInput(instance, nullptr, 0);
  fprintf(stderr, "Process result: %d\n", status);
  jxlDestroyInstance(instance);
}
#endif

}  // extern "C"
