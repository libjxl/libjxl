// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <hwy/targets.h>
#include <random>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/test_image.h"

namespace {

struct FuzzSpec {
  uint32_t xsize;
  uint32_t ysize;
  bool grayscale;
  uint8_t bit_depth;  // 1 - 16
  // JXL_ENC_FRAME_SETTING_EFFORT (1 - 6)
  uint8_t effort;
  // JXL_ENC_FRAME_SETTING_DECODING_SPEED (0 - 4)
  uint8_t decode_effort;
  // JXL_ENC_FRAME_SETTING_PHOTON_NOISE
  bool photon_noise;
  // JXL_ENC_FRAME_SETTING_EPF (-1 - 1)
  int8_t epf;
  // JXL_ENC_FRAME_SETTING_GABORISH (-1 - 1)
  int8_t gaborish;
  // JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC (-1 - 1)
  int8_t progressive_ac;
  // JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC (-1 - 1)
  int8_t qprogressive_ac;

  uint8_t num_threads;

  float distance;  // 0.01 - 25

  // Tiled to cover the entire image area.
  uint16_t pixel_data[3][64][64];

  static FuzzSpec FromData(const uint8_t* data, size_t len) {
    size_t pos = 0;
    auto u8 = [&]() -> uint8_t {
      if (pos == len) return 0;
      return data[pos++];
    };
    auto u16 = [&]() -> uint16_t { return (uint16_t{u8()} << 8) | u8(); };
    FuzzSpec spec;
    spec.xsize = uint32_t{u16()} + 1;
    spec.ysize = uint32_t{u16()} + 1;
    constexpr uint64_t kMaxSize = 1 << 24;
    if (spec.xsize * uint64_t{spec.ysize} > kMaxSize) {
      spec.ysize = kMaxSize / spec.xsize;
    }
    spec.grayscale = u8() % 2;
    spec.bit_depth = u8() % 16 + 1;
    spec.effort = u8() % 6 + 1;
    spec.decode_effort = u8() % 5;
    spec.photon_noise = u8() % 2;
    spec.epf = u8() % 3 - 1;
    spec.gaborish = u8() % 3 - 1;
    spec.progressive_ac = u8() % 3 - 1;
    spec.qprogressive_ac = u8() % 3 - 1;
    spec.num_threads = u8();
    // constants chosen so to cover the entire 0.01 - 25 range.
    spec.distance = 0.01 + 0.00038132 * u16();

    for (auto& x : spec.pixel_data) {
      for (auto& y : x) {
        for (auto& p : y) {
          p = u16();
        }
      }
    }

    return spec;
  }
};

std::vector<uint8_t> Encode(const FuzzSpec& spec, bool streaming) {
  auto runner = JxlThreadParallelRunnerMake(nullptr, spec.num_threads);
  JxlEncoderPtr enc_ptr = JxlEncoderMake(/*memory_manager=*/nullptr);
  JxlEncoder* enc = enc_ptr.get();

  JXL_CHECK(JxlEncoderSetParallelRunner(enc, JxlThreadParallelRunner,
                                        runner.get()) == JXL_ENC_SUCCESS);
  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc, nullptr);

  JXL_CHECK(JxlEncoderSetFrameDistance(frame_settings, spec.distance) ==
            JXL_ENC_SUCCESS);

  JXL_CHECK(JxlEncoderFrameSettingsSetOption(frame_settings,
                                             JXL_ENC_FRAME_SETTING_EFFORT,
                                             spec.effort) == JXL_ENC_SUCCESS);
  JXL_CHECK(JxlEncoderFrameSettingsSetOption(
                frame_settings, JXL_ENC_FRAME_SETTING_DECODING_SPEED,
                spec.decode_effort) == JXL_ENC_SUCCESS);
  if (spec.photon_noise) {
    JXL_CHECK(JxlEncoderFrameSettingsSetFloatOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_PHOTON_NOISE, 1600) ==
              JXL_ENC_SUCCESS);
  }
  JXL_CHECK(JxlEncoderFrameSettingsSetOption(frame_settings,
                                             JXL_ENC_FRAME_SETTING_EPF,
                                             spec.epf) == JXL_ENC_SUCCESS);
  JXL_CHECK(JxlEncoderFrameSettingsSetOption(frame_settings,
                                             JXL_ENC_FRAME_SETTING_GABORISH,
                                             spec.gaborish) == JXL_ENC_SUCCESS);
  JXL_CHECK(JxlEncoderFrameSettingsSetOption(
                frame_settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC,
                spec.progressive_ac) == JXL_ENC_SUCCESS);
  JXL_CHECK(JxlEncoderFrameSettingsSetOption(
                frame_settings, JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC,
                spec.qprogressive_ac) == JXL_ENC_SUCCESS);

  if (streaming) {
    JXL_CHECK(JxlEncoderFrameSettingsSetOption(frame_settings,
                                               JXL_ENC_FRAME_SETTING_BUFFERING,
                                               3) == JXL_ENC_SUCCESS);
  }

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.num_color_channels = spec.grayscale ? 1 : 3;
  basic_info.xsize = spec.xsize;
  basic_info.ysize = spec.ysize;
  basic_info.bits_per_sample = spec.bit_depth;
  basic_info.uses_original_profile = false;
  JXL_CHECK(JxlEncoderSetBasicInfo(enc, &basic_info) == JXL_ENC_SUCCESS);
  JxlColorEncoding color_encoding;
  memset(&color_encoding, 0, sizeof(color_encoding));
  color_encoding.color_space = spec.grayscale
                                   ? JxlColorSpace::JXL_COLOR_SPACE_GRAY
                                   : JxlColorSpace::JXL_COLOR_SPACE_RGB;
  color_encoding.transfer_function =
      JxlTransferFunction::JXL_TRANSFER_FUNCTION_SRGB;
  color_encoding.primaries = JxlPrimaries::JXL_PRIMARIES_2100;
  color_encoding.white_point = JxlWhitePoint::JXL_WHITE_POINT_D65;
  color_encoding.rendering_intent =
      JxlRenderingIntent::JXL_RENDERING_INTENT_RELATIVE;
  JXL_CHECK(JxlEncoderSetColorEncoding(enc, &color_encoding) ==
            JXL_ENC_SUCCESS);

  JxlFrameHeader frame_header;
  JxlEncoderInitFrameHeader(&frame_header);
  // TODO(szabadka) Add more frame header options.
  JXL_CHECK(JxlEncoderSetFrameHeader(frame_settings, &frame_header) ==
            JXL_ENC_SUCCESS);
  JxlPixelFormat pixelformat = {basic_info.num_color_channels, JXL_TYPE_UINT16,
                                JXL_LITTLE_ENDIAN, 0};
  std::vector<uint16_t> pixels(spec.xsize * (uint64_t)spec.ysize *
                               pixelformat.num_channels);
  for (size_t y = 0; y < spec.ysize; y++) {
    for (size_t x = 0; x < spec.xsize; x++) {
      for (size_t c = 0; c < pixelformat.num_channels; c++) {
        pixels[(y * spec.xsize + x) * pixelformat.num_channels + c] =
            spec.pixel_data[c][y % 64][x % 64];
      }
    }
  }
  JXL_CHECK(JxlEncoderAddImageFrame(frame_settings, &pixelformat, pixels.data(),
                                    pixels.size() * sizeof(uint16_t)) ==
            JXL_ENC_SUCCESS);
  JxlEncoderCloseInput(enc);
  // Reading compressed output
  JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
  std::vector<uint8_t> buf(1024);
  size_t written = 0;
  while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
    buf.resize(buf.size() * 2);
    uint8_t* next_out = buf.data() + written;
    size_t avail_out = buf.size() - written;
    process_result = JxlEncoderProcessOutput(enc, &next_out, &avail_out);
    written = next_out - buf.data();
  }
  JXL_CHECK(process_result == JXL_ENC_SUCCESS);
  buf.resize(written);

  return buf;
}

std::vector<float> Decode(const std::vector<uint8_t>& data) {
  // Multi-threaded parallel runner.
  auto dec = JxlDecoderMake(nullptr);
  JXL_CHECK(JxlDecoderSubscribeEvents(
                dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) ==
            JXL_DEC_SUCCESS);

  JxlBasicInfo info;
  JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};

  std::vector<float> pixels;

  JxlDecoderSetInput(dec.get(), data.data(), data.size());
  JxlDecoderCloseInput(dec.get());

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

    if (status == JXL_DEC_BASIC_INFO) {
      JXL_CHECK(JxlDecoderGetBasicInfo(dec.get(), &info) == JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size;
      JXL_CHECK(JxlDecoderImageOutBufferSize(dec.get(), &format,
                                             &buffer_size) == JXL_DEC_SUCCESS);
      pixels.resize(buffer_size / sizeof(float));
      void* pixels_buffer = (void*)pixels.data();
      size_t pixels_buffer_size = pixels.size() * sizeof(float);
      JXL_CHECK(JxlDecoderSetImageOutBuffer(dec.get(), &format, pixels_buffer,
                                            pixels_buffer_size) ==
                JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_FULL_IMAGE || status == JXL_DEC_SUCCESS) {
      return pixels;
    } else {
      // Unexpected status
      JXL_CHECK(false);
    }
  }
}

int TestOneInput(const uint8_t* data, size_t size) {
  auto spec = FuzzSpec::FromData(data, size);
  auto enc_default = Encode(spec, false);
  auto enc_streaming = Encode(spec, true);
  auto dec_default = Decode(enc_default);
  auto dec_streaming = Decode(enc_streaming);
  if (dec_default != dec_streaming) {
    return 1;
  }
  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return TestOneInput(data, size);
}
