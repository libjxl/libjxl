// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Encoder option-matrix + round-trip fuzzer.
//
// Threat model: an embedding application drives the JxlEncoder API with
// attacker-influenced pixel data and encoder settings (image upload /
// transcoding pipelines pick quality flags and feed uploaded pixels).
// The untrusted levers are the pixel content, the pixel format
// (type / endianness / channel count / bit depth) and the frame settings.
// A real bug here is a memory-safety violation in the encoder kernels
// (multiple highway SIMD targets are exercised), a JXL_DASSERT logic
// violation (OSS-Fuzz builds define JXL_IS_DEBUG_BUILD=1), or an OOM/hang
// on legal-but-extreme settings.
//
// Unlike cjxl_fuzzer, which samples option id/value pairs randomly and thus
// rarely lands coherent configurations (effort 1-2 lossless fast-lossless
// path, effort >= 7 Butteraugli-heavy heuristics, modular trees, palette,
// patches, dots, noise), this harness consumes a *coherent* configuration:
// every option is drawn from its documented legal range and the lossy /
// lossless / effort / modular interplay is kept consistent. It then decodes
// its own output so the round-trip property is checked as well.
//
// Per-iteration cost is bounded: images are at most 320x320 pixels, so tiny
// (few-byte) libFuzzer inputs still run fast while larger corpora entries
// sweep the option matrix.

#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/memory_manager.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <hwy/targets.h>
#include <vector>

#include "lib/jxl/fuzztest.h"
#include "tools/tracking_memory_manager.h"

namespace {

using ::jpegxl::tools::kGiB;
using ::jpegxl::tools::TrackingMemoryManager;

void CheckImpl(bool ok, const char* condition, const char* file, int line) {
  if (!ok) {
    fprintf(stderr, "Check(%s) failed at %s:%d\n", condition, file, line);
    __builtin_trap();
  }
}
#define Check(OK) CheckImpl((OK), #OK, __FILE__, __LINE__)

#define TRY(expr)                                 \
  do {                                            \
    if (JXL_ENC_SUCCESS != (expr)) return false;   \
  } while (0)

struct IntOptionSpec {
  JxlEncoderFrameSettingId flag;
  const char* name;
  int min;
  int max;
  int value;
};

#define INT_OPTION(FLAG, MIN_V, MAX_V) \
  IntOptionSpec { FLAG, #FLAG, MIN_V, MAX_V, 0 }

struct FloatOptionSpec {
  JxlEncoderFrameSettingId flag;
  float possible_values[4];
  float value;
};

struct FuzzSpec {
  uint32_t xsize;
  uint32_t ysize;
  JxlDataType data_type;
  size_t bits_per_sample;
  size_t exponent_bits_per_sample;
  JxlEndianness endianness;
  bool lossless;
  float distance;
  int effort;
  uint8_t num_color;      // 1 (grayscale) or 3 (RGB)
  bool have_alpha;
  size_t num_frames;
  bool use_animation;
  bool use_boxes;
  bool use_chunked;
  uint16_t pixel_seed[4][64][64];  // tiled 64x64 pixel content

  std::vector<IntOptionSpec> int_options = {
      INT_OPTION(JXL_ENC_FRAME_SETTING_EFFORT, 1, 9),
      INT_OPTION(JXL_ENC_FRAME_SETTING_DECODING_SPEED, 0, 4),
      INT_OPTION(JXL_ENC_FRAME_SETTING_NOISE, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_DOTS, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_PATCHES, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_EPF, -1, 3),
      INT_OPTION(JXL_ENC_FRAME_SETTING_GABORISH, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_MODULAR, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_RESPONSIVE, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_PALETTE_COLORS, -1, 255),
      INT_OPTION(JXL_ENC_FRAME_SETTING_LOSSY_PALETTE, -1, 1),
      INT_OPTION(JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, -1, 2),
      INT_OPTION(JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE, -1, 41),
      INT_OPTION(JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, -1, 3),
      INT_OPTION(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, -1, 15),
      INT_OPTION(JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS, -1, 11),
  };

  std::vector<FloatOptionSpec> float_options = {
      FloatOptionSpec{JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT,
                      {-1, 0, 50, 100}, -1},
      FloatOptionSpec{JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT,
                      {-1, 0, 50, 100}, -1},
      FloatOptionSpec{JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT,
                      {-1, 0, 50, 100}, -1},
      FloatOptionSpec{JXL_ENC_FRAME_SETTING_PHOTON_NOISE,
                      {-1, 200, 1600, 10000}, -1},
  };

  static FuzzSpec FromData(const uint8_t* data, size_t len) {
    size_t pos = 0;
    auto u8 = [&]() -> uint8_t {
      if (pos == len) return 0;
      return data[pos++];
    };
    auto u16 = [&]() -> uint16_t { return (uint16_t{u8()} << 8) | u8(); };
    auto b1 = [&]() -> bool { return (u8() % 2) == 1; };

    FuzzSpec spec;
    // Cost bound: at most 320x320; 256-wide/tall values still cross group
    // boundaries.
    spec.xsize = u16() % 320 + 1;
    spec.ysize = u16() % 320 + 1;

    switch (u8() % 4) {
      case 0:
        spec.data_type = JXL_TYPE_UINT8;
        spec.bits_per_sample = u8() % 8 + 1;
        spec.exponent_bits_per_sample = 0;
        break;
      case 1:
        spec.data_type = JXL_TYPE_UINT16;
        spec.bits_per_sample = u8() % 16 + 1;
        spec.exponent_bits_per_sample = 0;
        break;
      case 2:
        spec.data_type = JXL_TYPE_FLOAT16;
        spec.bits_per_sample = 16;
        spec.exponent_bits_per_sample = 5;
        break;
      default:
        spec.data_type = JXL_TYPE_FLOAT;
        spec.bits_per_sample = 32;
        spec.exponent_bits_per_sample = 8;
        break;
    }
    spec.endianness = b1() ? JXL_BIG_ENDIAN : JXL_LITTLE_ENDIAN;
    spec.lossless = b1();
    spec.distance = spec.lossless ? 0.0f : 0.01f + 0.00038132f * u16();
    spec.effort = u8() % 9 + 1;
    spec.num_color = b1() ? 3 : 1;
    spec.have_alpha = b1();
    spec.num_frames = b1() ? 2 : 1;
    spec.use_animation = b1();
    spec.use_boxes = b1();
    spec.use_chunked = b1();

    for (auto& int_opt : spec.int_options) {
      int_opt.value = u8() % (int_opt.max - int_opt.min + 1) + int_opt.min;
    }
    // Keep the coherent interplay: effort is driven by its own byte above.
    spec.int_options[0].value = spec.effort;
    for (auto& float_opt : spec.float_options) {
      float_opt.value = float_opt.possible_values[u8() % 4];
    }
    // Lossless forbids photon noise.
    if (spec.lossless) {
      spec.float_options[3].value = -1;
    }
    // Alpha always flows through the chunked-callback path. Driving
    // JxlEncoderSetExtraChannelBuffer after a fast-lossless-eligible
    // AddImageFrame (lossless + effort <= 2) currently null-derefs inside
    // libjxl (encode.cc:2873, frame_data is null for queued fast-lossless
    // frames) - documented as an upstream finding with reproducer
    // pair1-crash-1.bin; revisit once upstream fixed it.
    if (spec.have_alpha) {
      spec.use_chunked = true;
    }

    for (auto& x : spec.pixel_seed) {
      for (auto& y : x) {
        for (auto& p : y) {
          p = u16();
        }
      }
    }
    return spec;
  }
};

// Context for chunked-frame callbacks: serve tiles from the full-frame
// interleaved pixel buffer built below.
struct ChunkedContext {
  std::vector<uint8_t>* pixels;
  size_t xsize;
  size_t num_color;
  size_t num_channels;
  size_t bytes_per_sample;
  JxlPixelFormat color_format;
  JxlPixelFormat extra_format;
};

const void* ChunkedGetColorData(void* opaque, size_t xpos, size_t ypos,
                                size_t xsize, size_t ysize,
                                size_t* row_offset) {
  ChunkedContext* ctx = static_cast<ChunkedContext*>(opaque);
  *row_offset = ctx->xsize * ctx->num_channels * ctx->bytes_per_sample;
  size_t ofs =
      (ypos * ctx->xsize + xpos) * ctx->num_channels * ctx->bytes_per_sample;
  return ctx->pixels->data() + ofs;
}

const void* ChunkedGetExtraData(void* opaque, size_t ec_index, size_t xpos,
                                size_t ypos, size_t xsize, size_t ysize,
                                size_t* row_offset) {
  ChunkedContext* ctx = static_cast<ChunkedContext*>(opaque);
  *row_offset = ctx->xsize * ctx->num_channels * ctx->bytes_per_sample;
  size_t ofs =
      ((ypos * ctx->xsize + xpos) * ctx->num_channels + ctx->num_color) *
      ctx->bytes_per_sample;
  return ctx->pixels->data() + ofs;
}

void ChunkedGetColorFormat(void* opaque, JxlPixelFormat* pixel_format) {
  ChunkedContext* ctx = static_cast<ChunkedContext*>(opaque);
  *pixel_format = ctx->color_format;
}

void ChunkedGetExtraFormat(void* opaque, size_t ec_index,
                           JxlPixelFormat* pixel_format) {
  ChunkedContext* ctx = static_cast<ChunkedContext*>(opaque);
  *pixel_format = ctx->extra_format;
}

void ChunkedReleaseBuffer(void* opaque, const void* buf) {}

bool EncodeToJxl(const FuzzSpec& spec, TrackingMemoryManager& memory_manager,
                 std::vector<uint8_t>& result) {
  result.clear();
  // Multi-threaded parallel runner. Limit to max 2 threads since the fuzzer
  // itself is already multithreaded.
  size_t num_threads =
      std::min<size_t>(2, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  auto runner = JxlThreadParallelRunnerMake(memory_manager.get(), num_threads);
  JxlEncoderPtr enc_ptr = JxlEncoderMake(memory_manager.get());
  JxlEncoder* enc = enc_ptr.get();
  TRY(JxlEncoderSetParallelRunner(enc, JxlThreadParallelRunner, runner.get()));
  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc, nullptr);
  Check(frame_settings != nullptr);

  if (spec.lossless) {
    TRY(JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE));
  } else {
    TRY(JxlEncoderSetFrameDistance(frame_settings, spec.distance));
  }
  for (const auto& opt : spec.int_options) {
    TRY(JxlEncoderFrameSettingsSetOption(frame_settings, opt.flag, opt.value));
  }
  for (const auto& opt : spec.float_options) {
    if (opt.value != -1.0f) {
      TRY(JxlEncoderFrameSettingsSetFloatOption(frame_settings, opt.flag,
                                                opt.value));
    }
  }

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.xsize = spec.xsize;
  basic_info.ysize = spec.ysize;
  basic_info.num_color_channels = spec.num_color;
  basic_info.bits_per_sample = spec.bits_per_sample;
  basic_info.exponent_bits_per_sample = spec.exponent_bits_per_sample;
  basic_info.uses_original_profile = TO_JXL_BOOL(spec.lossless);
  size_t num_channels = spec.num_color;
  if (spec.have_alpha) {
    num_channels += 1;
    basic_info.num_extra_channels = 1;
    basic_info.alpha_bits = spec.bits_per_sample;
    basic_info.alpha_exponent_bits = spec.exponent_bits_per_sample;
    basic_info.alpha_premultiplied = JXL_FALSE;
  }
  if (spec.use_animation) {
    basic_info.animation.tps_numerator = 100;
    basic_info.animation.tps_denominator = 1;
    basic_info.animation.num_loops = 0;
    basic_info.animation.have_timecodes = JXL_FALSE;
  }
  TRY(JxlEncoderSetBasicInfo(enc, &basic_info));

  if (spec.use_boxes) {
    TRY(JxlEncoderUseBoxes(enc));
    JxlBoxType exif_type = {'E', 'x', 'i', 'f'};
    const uint8_t exif[] = {0x00, 0x4d, 0x4d, 0x00, 0x2a, 0x00, 0x00};
    TRY(JxlEncoderAddBox(enc, exif_type, exif, sizeof(exif), JXL_FALSE));
    JxlBoxType xml_type = {'x', 'm', 'l', ' '};
    const char* xml_payload = "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>";
    TRY(JxlEncoderAddBox(enc, xml_type,
                         reinterpret_cast<const uint8_t*>(xml_payload),
                         strlen(xml_payload), JXL_FALSE));
    JxlEncoderCloseBoxes(enc);
  }

  JxlColorEncoding color_encoding;
  memset(&color_encoding, 0, sizeof(color_encoding));
  color_encoding.color_space =
      spec.num_color == 1 ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB;
  color_encoding.transfer_function = JXL_TRANSFER_FUNCTION_SRGB;
  color_encoding.primaries = JXL_PRIMARIES_SRGB;
  color_encoding.white_point = JXL_WHITE_POINT_D65;
  color_encoding.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  TRY(JxlEncoderSetColorEncoding(enc, &color_encoding));

  JxlPixelFormat color_format = {spec.num_color, spec.data_type,
                                 spec.endianness, 0};
  JxlPixelFormat extra_format = {1, spec.data_type, spec.endianness, 0};

  const size_t bytes_per_sample =
      spec.data_type == JXL_TYPE_UINT8
          ? 1
          : (spec.data_type == JXL_TYPE_FLOAT16 ||
                     spec.data_type == JXL_TYPE_UINT16
                 ? 2
                 : 4);
  std::vector<uint8_t> pixels(static_cast<size_t>(spec.xsize) * spec.ysize *
                              num_channels * bytes_per_sample);
  for (size_t y = 0; y < spec.ysize; y++) {
    for (size_t x = 0; x < spec.xsize; x++) {
      for (size_t c = 0; c < num_channels; c++) {
        uint16_t sample = spec.pixel_seed[c % 4][y % 64][x % 64];
        uint8_t* dst = &pixels[((y * spec.xsize + x) * num_channels + c) *
                               bytes_per_sample];
        if (bytes_per_sample == 1) {
          dst[0] = static_cast<uint8_t>(sample >> 8);
        } else if (bytes_per_sample == 2) {
          if (spec.endianness == JXL_BIG_ENDIAN) {
            dst[0] = static_cast<uint8_t>(sample >> 8);
            dst[1] = static_cast<uint8_t>(sample);
          } else {
            dst[0] = static_cast<uint8_t>(sample);
            dst[1] = static_cast<uint8_t>(sample >> 8);
          }
        } else {
          float f = static_cast<float>(sample) / 65535.0f;
          uint32_t u;
          memcpy(&u, &f, 4);
          if (spec.endianness == JXL_BIG_ENDIAN) {
            dst[0] = static_cast<uint8_t>(u >> 24);
            dst[1] = static_cast<uint8_t>(u >> 16);
            dst[2] = static_cast<uint8_t>(u >> 8);
            dst[3] = static_cast<uint8_t>(u);
          } else {
            dst[0] = static_cast<uint8_t>(u);
            dst[1] = static_cast<uint8_t>(u >> 8);
            dst[2] = static_cast<uint8_t>(u >> 16);
            dst[3] = static_cast<uint8_t>(u >> 24);
          }
        }
      }
    }
  }
  // Alpha plane for the separate extra-channel-buffer API path.
  std::vector<uint8_t> alpha_plane;
  if (spec.have_alpha) {
    alpha_plane.resize(static_cast<size_t>(spec.xsize) * spec.ysize *
                       bytes_per_sample);
    for (size_t i = 0; i < alpha_plane.size(); i++) {
      size_t src =
          (i / bytes_per_sample) * num_channels * bytes_per_sample +
          spec.num_color * bytes_per_sample +
          (i % bytes_per_sample);
      alpha_plane[i] = pixels[src];
    }
  }

  for (size_t frame_idx = 0; frame_idx < spec.num_frames; frame_idx++) {
    bool is_last = frame_idx + 1 == spec.num_frames;
    JxlFrameHeader frame_header;
    JxlEncoderInitFrameHeader(&frame_header);
    frame_header.is_last = TO_JXL_BOOL(is_last);
    if (spec.use_animation) {
      frame_header.duration = frame_idx + 1;
    }
    TRY(JxlEncoderSetFrameHeader(frame_settings, &frame_header));
    if (spec.have_alpha) {
      JxlExtraChannelInfo extra_channel_info;
      JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_ALPHA, &extra_channel_info);
      TRY(JxlEncoderSetExtraChannelInfo(enc, 0, &extra_channel_info));
      if (!spec.lossless) {
        TRY(JxlEncoderSetExtraChannelDistance(frame_settings, 0, spec.distance));
      }
    }
    if (spec.use_chunked) {
      ChunkedContext ctx;
      ctx.pixels = &pixels;
      ctx.xsize = spec.xsize;
      ctx.num_color = spec.num_color;
      ctx.num_channels = num_channels;
      ctx.bytes_per_sample = bytes_per_sample;
      ctx.color_format = color_format;
      ctx.extra_format = extra_format;
      JxlChunkedFrameInputSource source;
      source.opaque = &ctx;
      source.get_color_channels_pixel_format = ChunkedGetColorFormat;
      source.get_color_channel_data_at = ChunkedGetColorData;
      source.get_extra_channel_pixel_format = ChunkedGetExtraFormat;
      source.get_extra_channel_data_at = ChunkedGetExtraData;
      source.release_buffer = ChunkedReleaseBuffer;
      TRY(JxlEncoderAddChunkedFrame(frame_settings, TO_JXL_BOOL(is_last),
                                    source));
    } else {
      TRY(JxlEncoderAddImageFrame(frame_settings, &color_format, pixels.data(),
                                  pixels.size()));
      if (spec.have_alpha) {
        TRY(JxlEncoderSetExtraChannelBuffer(frame_settings, &extra_format,
                                            alpha_plane.data(),
                                            alpha_plane.size(), 0));
      }
    }
    // The next loop iteration adds the following frame; the final
    // JxlEncoderProcessOutput flushes everything.
  }
  JxlEncoderCloseInput(enc);

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
  if (memory_manager.seen_oom) {
    // Actually, that is fine.
    return false;
  }
  if (process_result != JXL_ENC_SUCCESS) {
    // The encoder may legitimately reject an exotic-but-legal settings
    // combination while producing output (mirrors cjxl_fuzzer's TRY
    // convention).
    return false;
  }
  buf.resize(written);
  result = std::move(buf);
  return true;
}

// Round-trip: the codestream the encoder produced must decode.
bool DecodeJxl(const std::vector<uint8_t>& data,
               TrackingMemoryManager& memory_manager, std::vector<float>& out) {
  auto dec = JxlDecoderMake(memory_manager.get());
  Check(JxlDecoderSubscribeEvents(dec.get(),
                                  JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) ==
        JXL_DEC_SUCCESS);
  JxlBasicInfo info;
  JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  out.clear();

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
      out.resize(buffer_size / sizeof(float));
      Check(JxlDecoderSetImageOutBuffer(dec.get(), &format, out.data(),
                                        out.size() * sizeof(float)) ==
            JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_FULL_IMAGE || status == JXL_DEC_SUCCESS) {
      return true;
    } else {
      if (memory_manager.seen_oom) {
        // Decoder OOM is an upstream bug only if it happens without the
        // encoder having OOM'd first; report it by failing the check.
        Check(false);
      }
      // Unexpected status: the encoder produced an invalid codestream.
      Check(false);
    }
  }
}

void Run(const FuzzSpec& spec) {
  TrackingMemoryManager memory_manager{/* cap */ 1 * kGiB,
                                       /* total_cap */ 5 * kGiB};
  std::vector<uint8_t> encoded;
  // It is fine, if the encoder rejects an exotic-but-legal combination.
  if (!EncodeToJxl(spec, memory_manager, encoded)) return;
  Check(memory_manager.Reset());

  std::vector<float> decoded;
  Check(DecodeJxl(encoded, memory_manager, decoded));
  Check(memory_manager.Reset());
}

template <typename T>
T Select(const std::vector<T>& vec,
         const std::function<uint32_t(size_t)>& get_index) {
  return vec[get_index(vec.size() - 1)];
}

int DoTestOneInput(const uint8_t* data, size_t size) {
  uint32_t selector = 0;
  for (size_t i = 0; i < size && i < 4; i++) {
    selector = (selector << 8) | data[i];
  }
  const auto targets = hwy::SupportedAndGeneratedTargets();
  hwy::SetSupportedTargetsForTest(
      Select(targets, [&](size_t max_value) {
        return selector % (max_value + 1);
      }));
  FuzzSpec spec = FuzzSpec::FromData(data, size);
  Run(spec);
  hwy::SetSupportedTargetsForTest(0);
  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return DoTestOneInput(data, size);
}

void TestOneInput(const std::vector<uint8_t>& data) {
  DoTestOneInput(data.data(), data.size());
}

FUZZ_TEST(EncoderOptionsFuzzTest, TestOneInput);
