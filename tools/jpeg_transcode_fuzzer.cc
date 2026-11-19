// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// JPEG -> JXL -> JPEG transcode fuzzer.
//
// Threat model: JPEG files are classic untrusted input. libjxl accepts raw
// JPEG bytes through JxlEncoderAddJPEGFrame for lossless transcoding (the
// `cjxl input.jpg out.jxl` path) and reconstructs the original JPEG byte
// stream on decode (JXL_DEC_JPEG_RECONSTRUCTION). The untrusted bytes cross
// the trust boundary directly: the marker parser in
// lib/jxl/jpeg/enc_jpeg_data_reader.cc (marker byte 0xff and big-endian
// segment-length checks at lines 41-50, SOF handling at 65-74, SOS handling
// at 129-135) and the reconstruction writer in
// lib/jxl/jpeg/dec_jpeg_data_writer.cc.
//
// Mutation-resistant gates justify the dictionary and the real-JPEG seed
// corpus: a random byte stream almost never satisfies the 0xFF marker byte +
// segment-length + SOI/SOF/DHT/SOS ordering constraints, so coverage-guided
// mutation alone cannot reach the deep JPEG parsing code. The last four
// bytes of the input are the encoder option vector, so real JPEG seed files
// stay parseable when 4 config bytes are appended by the build's seed
// packaging step.
//
// Properties checked:
// - a JPEG that the encoder accepts must produce a decodable codestream;
// - the reconstructed byte stream must start with the JPEG SOI marker.

#include <jxl/codestream_header.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// Walks the JPEG marker chain and returns the pixel count declared by the
// first SOF marker (0 if none found). Mutated SOF dimensions can declare
// gigapixel images whose proportional allocations OOM the fuzzer, so the
// harness rejects anything beyond the same 16.7 MPix budget other libjxl
// fuzz targets use.
uint64_t JpegSofPixelCount(const uint8_t* data, size_t size) {
  if (size < 4 || data[0] != 0xFF || data[1] != 0xD8) return 0;
  size_t pos = 2;
  while (pos + 4 <= size) {
    if (data[pos] != 0xFF) {
      pos++;
      continue;
    }
    uint8_t marker = data[pos + 1];
    if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
      pos += 2;
      continue;
    }
    if (marker == 0xD9) break;
    if (pos + 4 > size) break;
    size_t seglen = (static_cast<size_t>(data[pos + 2]) << 8) | data[pos + 3];
    if (seglen < 2) break;
    if (0xC0 <= marker && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 &&
        marker != 0xCC) {
      if (pos + 9 > size) return 0;
      uint64_t h = (static_cast<uint64_t>(data[pos + 5]) << 8) | data[pos + 6];
      uint64_t w = (static_cast<uint64_t>(data[pos + 7]) << 8) | data[pos + 8];
      return w * h;
    }
    pos += 2 + seglen;
  }
  return 0;
}

bool EncodeToJxl(TrackingMemoryManager& memory_manager, int effort, int cfl,
                 int compress_boxes, int brotli_effort, const uint8_t* bytes,
                 size_t len, std::vector<uint8_t>* result) {
  result->clear();
  auto enc = JxlEncoderMake(memory_manager.get());
  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  Check(frame_settings != nullptr);
  TRY(JxlEncoderFrameSettingsSetOption(
      frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, effort));
  TRY(JxlEncoderFrameSettingsSetOption(
      frame_settings, JXL_ENC_FRAME_SETTING_JPEG_RECON_CFL, cfl));
  TRY(JxlEncoderFrameSettingsSetOption(
      frame_settings, JXL_ENC_FRAME_SETTING_JPEG_COMPRESS_BOXES,
      compress_boxes));
  TRY(JxlEncoderFrameSettingsSetOption(
      frame_settings, JXL_ENC_FRAME_SETTING_BROTLI_EFFORT, brotli_effort));
  TRY(JxlEncoderAddJPEGFrame(frame_settings, bytes, len));
  JxlEncoderCloseInput(enc.get());

  JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
  std::vector<uint8_t> buf(1024);
  size_t written = 0;
  while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
    buf.resize(buf.size() * 2);
    uint8_t* next_out = buf.data() + written;
    size_t avail_out = buf.size() - written;
    process_result = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    written = next_out - buf.data();
  }
  if (memory_manager.seen_oom) {
    // Actually, that is fine.
    return false;
  }
  if (process_result != JXL_ENC_SUCCESS) {
    // The encoder may legitimately fail on a mutated JPEG whose parsed
    // properties are unsupported (mirrors cjxl_fuzzer's TRY convention).
    return false;
  }
  buf.resize(written);
  *result = std::move(buf);
  return true;
}

int DoTestOneInput(const uint8_t* data, size_t size) {
  if (size < 8) return 0;
  if (size > (1u << 20)) return 0;  // bound per-iteration work
  const uint8_t* jpeg = data;
  size_t jpeg_size = size - 4;
  uint64_t sof_pixels = JpegSofPixelCount(jpeg, jpeg_size);
  if (sof_pixels != 0 && sof_pixels > (1u << 24)) {
    // Mutated SOF dimensions would drive proportional allocations far past
    // the fuzzing memory budget; skip instead of OOMing.
    return 0;
  }
  TrackingMemoryManager memory_manager{/* cap */ 1 * kGiB,
                                       /* total_cap */ 5 * kGiB};
  const uint8_t* cfg = data + size - 4;

  int effort = cfg[0] % 9 + 1;
  int cfl = static_cast<int>(cfg[1] % 3) - 1;
  int compress_boxes = static_cast<int>(cfg[2] % 3) - 1;
  int brotli_effort = static_cast<int>(cfg[3] % 12) - 1;
  // Exercise the JXL_DEC_JPEG_NEED_MORE_OUTPUT growth path by starting with
  // a too-small reconstruction buffer.
  bool small_jpeg_buffer = (cfg[1] & 4) != 0;
  bool retranscode = (cfg[2] & 4) != 0;

  std::vector<uint8_t> encoded;
  if (!EncodeToJxl(memory_manager, effort, cfl, compress_boxes, brotli_effort,
                   jpeg, jpeg_size, &encoded)) {
    return 0;
  }
  Check(memory_manager.Reset());

  auto dec = JxlDecoderMake(memory_manager.get());
  Check(JxlDecoderSubscribeEvents(dec.get(),
                                  JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE |
                                      JXL_DEC_JPEG_RECONSTRUCTION) ==
        JXL_DEC_SUCCESS);
  std::vector<uint8_t> reconstructed;
  std::vector<uint8_t> pixels;
  bool seen_recon = false;

  JxlDecoderSetInput(dec.get(), encoded.data(), encoded.size());
  JxlDecoderCloseInput(dec.get());

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      Check(JxlDecoderGetBasicInfo(dec.get(), &info) == JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
      size_t buffer_size;
      Check(JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size) ==
            JXL_DEC_SUCCESS);
      pixels.resize(buffer_size);
      Check(JxlDecoderSetImageOutBuffer(dec.get(), &format, pixels.data(),
                                        pixels.size()) == JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
      Check(!seen_recon);
      seen_recon = true;
      reconstructed.resize(small_jpeg_buffer ? 1 : (1 << 16));
      Check(JxlDecoderSetJPEGBuffer(dec.get(), reconstructed.data(),
                                    reconstructed.size()) == JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
      Check(seen_recon);
      size_t used =
          reconstructed.size() - JxlDecoderReleaseJPEGBuffer(dec.get());
      reconstructed.resize(std::max<size_t>(4096, reconstructed.size() * 2));
      Check(JxlDecoderSetJPEGBuffer(dec.get(), reconstructed.data() + used,
                                    reconstructed.size() - used) ==
            JXL_DEC_SUCCESS);
    } else if (status == JXL_DEC_FULL_IMAGE || status == JXL_DEC_SUCCESS) {
      if (seen_recon) {
        reconstructed.resize(reconstructed.size() -
                             JxlDecoderReleaseJPEGBuffer(dec.get()));
        // The reconstruction must be a JPEG byte stream (SOI marker).
        Check(reconstructed.size() >= 2 && reconstructed[0] == 0xFF &&
              reconstructed[1] == 0xD8);
        Check(memory_manager.Reset());
        if (retranscode) {
          std::vector<uint8_t> encoded_again;
          if (EncodeToJxl(memory_manager, effort, cfl, compress_boxes,
                          brotli_effort, reconstructed.data(),
                          reconstructed.size(), &encoded_again)) {
            Check(memory_manager.Reset());
          }
        }
      }
      return 0;
    } else {
      // A codestream the encoder produced must decode without error.
      Check(false);
    }
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return DoTestOneInput(data, size);
}

void TestOneInput(const std::vector<uint8_t>& data) {
  DoTestOneInput(data.data(), data.size());
}

FUZZ_TEST(JpegTranscodeFuzzTest, TestOneInput);
