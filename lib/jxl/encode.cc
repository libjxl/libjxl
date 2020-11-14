// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "jxl/encode.h"

#include <algorithm>
#include <vector>

#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/external_image.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/memory_manager_internal.h"

// Debug-printing failure macro similar to JXL_FAILURE, but for the status code
// JXL_ENC_ERROR
#ifdef JXL_CRASH_ON_ERROR
#define JXL_API_ERROR(format, ...)                                           \
  (::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__), \
   ::jxl::Abort(), JXL_ENC_ERROR)
#else  // JXL_CRASH_ON_ERROR
#define JXL_API_ERROR(format, ...)                                             \
  (((JXL_DEBUG_ON_ERROR) &&                                                    \
    ::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__)), \
   JXL_ENC_ERROR)
#endif  // JXL_CRASH_ON_ERROR

uint32_t JxlEncoderVersion(void) {
  return JPEGXL_MAJOR_VERSION * 1000000 + JPEGXL_MINOR_VERSION * 1000 +
         JPEGXL_PATCH_VERSION;
}

typedef struct JxlEncoderQueuedFrame {
  JxlFrameFormat frame_format;
  std::vector<uint8_t> buffer;
} JxlEncoderQueuedFrame;

struct JxlEncoderStruct {
  JxlMemoryManager memory_manager;
  std::unique_ptr<jxl::ThreadPool> thread_pool;
  std::vector<std::unique_ptr<JxlEncoderQueuedFrame>> input_frame_queue;
  std::vector<uint8_t> output_byte_queue;
  bool wrote_headers;
  jxl::CodecMetadata metadata;

  JxlEncoderStatus RefillOutputByteQueue() {
    std::unique_ptr<JxlEncoderQueuedFrame> input_frame =
        std::move(this->input_frame_queue[0]);
    this->input_frame_queue.erase(this->input_frame_queue.begin());

    jxl::CodecInOut io;
    jxl::ColorEncoding c_current;
    bool has_alpha;
    // TODO(zond): Make this accept more than bitdepth 16.
    size_t bitdepth = 16;

    io.metadata.m.SetUintSamples(bitdepth);
    // TODO(zond): Make this accept more than sRGB, and also gray+alpha.
    if (input_frame->frame_format.pixel_format.num_channels == 1) {
      c_current = jxl::ColorEncoding::SRGB(/*is_gray=*/true);
      has_alpha = false;
    } else if (input_frame->frame_format.pixel_format.num_channels == 3) {
      c_current = jxl::ColorEncoding::SRGB(/*is_gray=*/false);
      has_alpha = false;
    } else if (input_frame->frame_format.pixel_format.num_channels == 4) {
      c_current = jxl::ColorEncoding::SRGB(/*is_gray=*/false);
      has_alpha = true;
      io.metadata.m.SetAlphaBits(bitdepth);
    } else {
      return JXL_ENC_ERROR;
    }

    if (!ConvertImage(jxl::Span<const uint8_t>(input_frame->buffer.data(),
                                               input_frame->buffer.size()),
                      input_frame->frame_format.frame_width,
                      input_frame->frame_format.frame_height, c_current,
                      has_alpha, /*alpha_is_premultiplied=*/false,
                      /*bits_per_alpha=*/has_alpha ? bitdepth : 0, bitdepth,
                      /*big_endian=*/true,
                      /*flipped_y=*/false, this->thread_pool.get(),
                      &io.Main())) {
      return JXL_ENC_ERROR;
    }
    io.SetSize(io.Main().xsize(), io.Main().ysize());
    io.CheckMetadata();

    jxl::CompressParams cparams;
    jxl::BitWriter writer;

    if (!wrote_headers) {
      if (!WriteHeaders(cparams, &io, &this->metadata, &writer, nullptr)) {
        return JXL_ENC_ERROR;
      }
      // Only send ICC (at least several hundred bytes) if fields aren't enough.
      if (this->metadata.m.color_encoding.WantICC()) {
        if (!jxl::WriteICC(this->metadata.m.color_encoding.ICC(), &writer,
                           jxl::kLayerHeader, nullptr)) {
          return JXL_ENC_ERROR;
        }
      }
      if (this->metadata.m.have_preview) {
        if (!jxl::EncodePreview(cparams, io.preview_frame, &this->metadata,
                                this->thread_pool.get(), &writer)) {
          return JXL_ENC_ERROR;
        }
      }
      wrote_headers = true;
    }

    // Each frame should start on byte boundaries.
    writer.ZeroPadToByte();

    // TODO(zond): Handle progressive mode like EncodeFile does it.
    // TODO(zond): Handle animation like EncodeFile does it, by checking if
    //             JxlEncoderCloseInput has been called (to see if it's the
    //             last animation frame).

    jxl::PassesEncoderState enc_state;
    if (!jxl::EncodeFrame(cparams, jxl::FrameInfo{}, &this->metadata,
                          io.frames[0], &enc_state, this->thread_pool.get(),
                          &writer, /*aux_out=*/nullptr)) {
      return JXL_ENC_ERROR;
    }

    jxl::PaddedBytes bytes = std::move(writer).TakeBytes();
    this->output_byte_queue =
        std::vector<uint8_t>(bytes.data(), bytes.data() + bytes.size());
    return JXL_ENC_SUCCESS;
  }
};

JxlEncoder* JxlEncoderCreate(const JxlMemoryManager* memory_manager) {
  JxlMemoryManager local_memory_manager;
  if (!jxl::MemoryManagerInit(&local_memory_manager, memory_manager)) {
    return nullptr;
  }

  void* alloc =
      jxl::MemoryManagerAlloc(&local_memory_manager, sizeof(JxlEncoder));
  if (!alloc) return nullptr;
  // Placement new constructor on allocated memory
  JxlEncoder* enc = new (alloc) JxlEncoder();
  enc->memory_manager = local_memory_manager;
  enc->wrote_headers = false;

  return enc;
}

void JxlEncoderDestroy(JxlEncoder* enc) {
  if (enc) {
    // Call destructor directly since custom free function is used.
    enc->~JxlEncoder();
    jxl::MemoryManagerFree(&enc->memory_manager, enc);
  }
}

JxlEncoderStatus JxlEncoderSetParallelRunner(JxlEncoder* enc,
                                             JxlParallelRunner parallel_runner,
                                             void* parallel_runner_opaque) {
  if (enc->thread_pool) return JXL_API_ERROR("parallel runner already set");
  enc->thread_pool.reset(
      new jxl::ThreadPool(parallel_runner, parallel_runner_opaque));
  return JXL_ENC_SUCCESS;
}

JxlEncoderStatus JxlEncoderAddImageFrame(JxlEncoder* enc,
                                         const JxlFrameFormat* frame_format,
                                         void* buffer, size_t size) {
  // TODO(zond): Return error if the input has been closed.
  enc->input_frame_queue.push_back(
      std::unique_ptr<JxlEncoderQueuedFrame>(new JxlEncoderQueuedFrame{
          *frame_format,
          std::vector<uint8_t>(static_cast<uint8_t*>(buffer),
                               static_cast<uint8_t*>(buffer) + size)}));
  return JXL_ENC_SUCCESS;
}

void JxlEncoderCloseInput(JxlEncoder* enc) {
  // TODO(zond): Make this function mark the most recent frame as the last.
}

JxlEncoderStatus JxlEncoderProcessOutput(JxlEncoder* enc, uint8_t** next_out,
                                         size_t* avail_out) {
  while (*avail_out > 0 && (enc->output_byte_queue.size() > 0 ||
                            enc->input_frame_queue.size() > 0)) {
    if (enc->output_byte_queue.size() > 0) {
      size_t to_copy = std::min(*avail_out, enc->output_byte_queue.size());
      memcpy(static_cast<void*>(*next_out), enc->output_byte_queue.data(),
             to_copy);
      *next_out += to_copy;
      *avail_out -= to_copy;
      enc->output_byte_queue.erase(enc->output_byte_queue.begin(),
                                   enc->output_byte_queue.begin() + to_copy);
    } else if (enc->input_frame_queue.size() > 0) {
      if (enc->RefillOutputByteQueue() != JXL_ENC_SUCCESS) {
        return JXL_ENC_ERROR;
      }
    }
  }

  if (enc->output_byte_queue.size() > 0 || enc->input_frame_queue.size() > 0) {
    return JXL_ENC_NEED_MORE_OUTPUT;
  }
  return JXL_ENC_SUCCESS;
}
