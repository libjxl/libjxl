// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>
#include <jxl/types.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/fuzztest.h"
#include "tools/tracking_memory_manager.h"

namespace {

using ::jpegxl::tools::kGiB;
using ::jpegxl::tools::TrackingMemoryManager;
using ::jxl::Status;
using ::jxl::StatusOr;

void Check(bool ok) {
  if (!ok) {
    JXL_CRASH();
  }
}

struct FuzzSpec {
  uint32_t xsize;
  uint32_t ysize;
  bool grayscale;
  bool alpha;
  uint8_t bit_depth;  // 1 - 16

  struct IntOptionSpec {
    JxlEncoderFrameSettingId flag;
    int min;
    int max;
    int value;
  };

  std::vector<IntOptionSpec> int_options = {
      IntOptionSpec{JXL_ENC_FRAME_SETTING_EFFORT, 1, 9, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_DECODING_SPEED, 0, 4, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_NOISE, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_DOTS, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_PATCHES, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_EPF, -1, 3, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_GABORISH, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_MODULAR, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_RESPONSIVE, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_PALETTE_COLORS, -1, 255, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_LOSSY_PALETTE, -1, 1, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, -1, 2, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE, -1, 41, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, -1, 3, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, -1, 15, 0},
      IntOptionSpec{JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS, -1, 11, 0},
  };

  struct FloatOptionSpec {
    JxlEncoderFrameSettingId flag;
    float possible_values[4];
    float value;
  };

  std::vector<FloatOptionSpec> float_options = {
      FloatOptionSpec{JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT,
                      {-1, 0, 50, 100},
                      -1},
      FloatOptionSpec{JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT,
                      {-1, 0, 50, 100},
                      -1},
      FloatOptionSpec{JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT,
                      {-1, 0, 50, 100},
                      -1},
      FloatOptionSpec{
          JXL_ENC_FRAME_SETTING_PHOTON_NOISE, {-1, 200, 1600, 10000}, -1},
  };

  uint8_t num_threads;

  float distance;  // 0.01 - 25

  // Tiled to cover the entire image area.
  uint16_t pixel_data[4][64][64];

  static FuzzSpec FromData(const uint8_t* data, size_t len) {
    size_t pos = 0;
    auto u8 = [&]() -> uint8_t {
      if (pos == len) return 0;
      return data[pos++];
    };
    auto b1 = [&]() -> bool { return static_cast<bool>(u8() % 2); };
    auto u16 = [&]() -> uint16_t { return (uint16_t{u8()} << 8) | u8(); };
    FuzzSpec spec;
    spec.xsize = uint32_t{u16()} + 1;
    spec.ysize = uint32_t{u16()} + 1;
    constexpr uint64_t kMaxSize = 1 << 24;
    if (spec.xsize * uint64_t{spec.ysize} > kMaxSize) {
      spec.ysize = kMaxSize / spec.xsize;
    }
    spec.grayscale = b1();
    spec.alpha = b1();
    spec.bit_depth = u8() % 16 + 1;
    // constants chosen so to cover the entire 0.01 - 25 range.
    spec.distance = u8() % 2 ? 0.0 : 0.01 + 0.00038132 * u16();

    Check(spec.float_options[2].flag ==
          JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT);
    Check(spec.int_options[15].flag == JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM);
    if (spec.distance != 0 || spec.int_options[15].value == 0) {
      spec.float_options[2].possible_values[1] = 1;
    }

    spec.num_threads = u8();

    for (auto& int_opt : spec.int_options) {
      int_opt.value = u8() % (int_opt.max - int_opt.min + 1) + int_opt.min;
    }
    for (auto& float_opt : spec.float_options) {
      float_opt.value = float_opt.possible_values[u8() % 4];
    }

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

StatusOr<std::vector<uint8_t>> Encode(const FuzzSpec& spec,
                                      TrackingMemoryManager& memory_manager,
                                      bool streaming) {
  auto runner = JxlThreadParallelRunnerMake(nullptr, spec.num_threads);
  JxlEncoderPtr enc_ptr = JxlEncoderMake(memory_manager.get());
  JxlEncoder* enc = enc_ptr.get();

  Check(JxlEncoderSetParallelRunner(enc, JxlThreadParallelRunner,
                                    runner.get()) == JXL_ENC_SUCCESS);
  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc, nullptr);

  Check(JxlEncoderSetFrameDistance(frame_settings, spec.distance) ==
        JXL_ENC_SUCCESS);

  for (const auto& opt : spec.int_options) {
    Check(JxlEncoderFrameSettingsSetOption(frame_settings, opt.flag,
                                           opt.value) == JXL_ENC_SUCCESS);
  }
  for (const auto& opt : spec.float_options) {
    if (opt.value != -1) {
      Check(JxlEncoderFrameSettingsSetFloatOption(
                frame_settings, opt.flag, opt.value) == JXL_ENC_SUCCESS);
    }
  }

  Check(JxlEncoderFrameSettingsSetOption(frame_settings,
                                         JXL_ENC_FRAME_SETTING_BUFFERING,
                                         streaming ? 3 : 0) == JXL_ENC_SUCCESS);

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.num_color_channels = spec.grayscale ? 1 : 3;
  basic_info.xsize = spec.xsize;
  basic_info.ysize = spec.ysize;
  basic_info.bits_per_sample = spec.bit_depth;
  basic_info.uses_original_profile = JXL_FALSE;
  uint32_t nchan = basic_info.num_color_channels;
  if (spec.alpha) {
    nchan += 1;
    basic_info.alpha_bits = spec.bit_depth;
    basic_info.num_extra_channels = 1;
  }
  Check(JxlEncoderSetBasicInfo(enc, &basic_info) == JXL_ENC_SUCCESS);
  if (spec.alpha) {
    JxlExtraChannelInfo info;
    memset(&info, 0, sizeof(info));
    info.type = JxlExtraChannelType::JXL_CHANNEL_ALPHA;
    info.bits_per_sample = spec.bit_depth;
    JxlEncoderSetExtraChannelInfo(enc, 0, &info);
  }
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
  Check(JxlEncoderSetColorEncoding(enc, &color_encoding) == JXL_ENC_SUCCESS);

  JxlFrameHeader frame_header;
  JxlEncoderInitFrameHeader(&frame_header);
  // TODO(szabadka) Add more frame header options.
  Check(JxlEncoderSetFrameHeader(frame_settings, &frame_header) ==
        JXL_ENC_SUCCESS);
  JxlPixelFormat pixelformat = {nchan, JXL_TYPE_UINT16, JXL_LITTLE_ENDIAN, 0};
  std::vector<uint16_t> pixels(spec.xsize * static_cast<uint64_t>(spec.ysize) *
                               nchan);
  for (size_t y = 0; y < spec.ysize; y++) {
    for (size_t x = 0; x < spec.xsize; x++) {
      for (size_t c = 0; c < nchan; c++) {
        pixels[(y * spec.xsize + x) * nchan + c] =
            spec.pixel_data[c][y % 64][x % 64];
      }
    }
  }
  JxlEncoderStatus status =
      JxlEncoderAddImageFrame(frame_settings, &pixelformat, pixels.data(),
                              pixels.size() * sizeof(uint16_t));
  // TODO(eustas): update when API will provide OOM status.
  if (memory_manager.seen_oom) {
    // Actually, that is fine.
    return JXL_FAILURE("OOM");
  }
  Check(status == JXL_ENC_SUCCESS);
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
  // TODO(eustas): update when API will provide OOM status.
  if (memory_manager.seen_oom) {
    // Actually, that is fine.
    return JXL_FAILURE("OOM");
  }
  Check(process_result == JXL_ENC_SUCCESS);
  buf.resize(written);

  return buf;
}

StatusOr<std::vector<float>> Decode(const std::vector<uint8_t>& data,
                                    TrackingMemoryManager& memory_manager) {
  // Multi-threaded parallel runner.
  auto dec = JxlDecoderMake(memory_manager.get());
  Check(JxlDecoderSubscribeEvents(dec.get(),
                                  JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) ==
        JXL_DEC_SUCCESS);

  JxlBasicInfo info;
  JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};

  std::vector<float> pixels;

  JxlDecoderSetInput(dec.get(), data.data(), data.size());
  JxlDecoderCloseInput(dec.get());

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

    if (status == JXL_DEC_BASIC_INFO) {
      Check(JxlDecoderGetBasicInfo(dec.get(), &info) == JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size;
      Check(JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size) ==
            JXL_DEC_SUCCESS);
      pixels.resize(buffer_size / sizeof(float));
      void* pixels_buffer = static_cast<void*>(pixels.data());
      size_t pixels_buffer_size = pixels.size() * sizeof(float);
      Check(JxlDecoderSetImageOutBuffer(dec.get(), &format, pixels_buffer,
                                        pixels_buffer_size) == JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_FULL_IMAGE || status == JXL_DEC_SUCCESS) {
      return pixels;
    } else {
      // TODO(eustas): update when API will provide OOM status.
      if (memory_manager.seen_oom) {
        // Actually, that is fine.
        return JXL_FAILURE("OOM");
      }
      // Unexpected status
      Check(false);
    }
  }
}

Status Run(const FuzzSpec& spec, TrackingMemoryManager& memory_manager) {
  std::vector<uint8_t> enc_default;
  std::vector<uint8_t> enc_streaming;

  const auto encode = [&]() -> Status {
    // It is not clear, which approach eats more memory.
    JXL_ASSIGN_OR_RETURN(enc_default, Encode(spec, memory_manager, false));
    Check(memory_manager.Reset());
    JXL_ASSIGN_OR_RETURN(enc_streaming, Encode(spec, memory_manager, true));
    Check(memory_manager.Reset());
    return true;
  };
  // It is fine, if encoder OOMs.
  if (!encode()) return true;

  // It is NOT OK, if decoder OOMs - it should not consume more than encoder.
  JXL_ASSIGN_OR_RETURN(auto dec_default, Decode(enc_default, memory_manager));
  Check(memory_manager.Reset());
  JXL_ASSIGN_OR_RETURN(auto dec_streaming,
                       Decode(enc_streaming, memory_manager));
  Check(memory_manager.Reset());

  Check(dec_default.size() == dec_streaming.size());
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < dec_default.size(); ++i) {
    float abs_diff = std::abs(dec_default[i] - dec_streaming[i]);
    if (abs_diff > max_abs_diff) {
      max_abs_diff = abs_diff;
    }
  }

  Check(spec.int_options[0].flag == JXL_ENC_FRAME_SETTING_EFFORT);
  int effort = spec.int_options[0].value;
  std::array<float, 10> kThreshold = {0.00f, 0.05f, 0.05f, 0.05f, 0.05f,
                                      0.05f, 0.05f, 0.05f, 0.10f, 0.10f};
  Check(max_abs_diff <= kThreshold[effort]);

  return true;
}

int DoTestOneInput(const uint8_t* data, size_t size) {
  auto spec = FuzzSpec::FromData(data, size);

  TrackingMemoryManager memory_manager{/* cap */ 1 * kGiB,
                                       /* total_cap */ 5 * kGiB};
  Check(Run(spec, memory_manager));
  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return DoTestOneInput(data, size);
}

void TestOneInput(const std::vector<uint8_t>& data) {
  DoTestOneInput(data.data(), data.size());
}

FUZZ_TEST(StreamingFuzzTest, TestOneInput);
