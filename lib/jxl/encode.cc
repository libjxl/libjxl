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

namespace {

typedef struct JxlEncoderOptionsValuesStruct {
  // lossless is a separate setting from cparams because it is a combination
  // setting that overrides multiple settings inside of cparams.
  bool lossless;
  jxl::CompressParams cparams;
} JxlEncoderOptionsValues;

typedef struct JxlEncoderQueuedFrame {
  JxlPixelFormat pixel_format;
  JxlEncoderOptionsValues option_values;
  std::vector<uint8_t> buffer;
} JxlEncoderQueuedFrame;

}  // namespace

struct JxlEncoderStruct {
  JxlMemoryManager memory_manager;
  jxl::MemoryManagerUniquePtr<jxl::ThreadPool> thread_pool{
      nullptr, jxl::MemoryManagerDeleteHelper(&memory_manager)};
  std::vector<jxl::MemoryManagerUniquePtr<JxlEncoderQueuedFrame>>
      input_frame_queue;
  std::vector<jxl::MemoryManagerUniquePtr<JxlEncoderOptions>> encoder_options;
  std::vector<uint8_t> output_byte_queue;
  bool wrote_headers;
  jxl::CodecMetadata metadata;

  JxlEncoderStatus RefillOutputByteQueue() {
    jxl::MemoryManagerUniquePtr<JxlEncoderQueuedFrame> input_frame =
        std::move(this->input_frame_queue[0]);
    this->input_frame_queue.erase(this->input_frame_queue.begin());

    jxl::CodecInOut io;
    jxl::ColorEncoding c_current;
    bool has_alpha;
    bool is_gray;
    size_t bitdepth;

    // TODO(zond): Make this accept more than float and uint8/16.
    if (input_frame->pixel_format.data_type == JXL_TYPE_FLOAT) {
      bitdepth = 32;
      io.metadata.m.SetFloat32Samples();
    } else if (input_frame->pixel_format.data_type == JXL_TYPE_UINT8) {
      bitdepth = 8;
      io.metadata.m.SetUintSamples(bitdepth);
    } else if (input_frame->pixel_format.data_type == JXL_TYPE_UINT16) {
      bitdepth = 16;
      io.metadata.m.SetUintSamples(bitdepth);
    } else {
      return JXL_ENC_ERROR;
    }

    if (input_frame->pixel_format.num_channels == 1) {
      has_alpha = false;
      is_gray = true;
    } else if (input_frame->pixel_format.num_channels == 2) {
      is_gray = true;
      has_alpha = true;
      io.metadata.m.SetAlphaBits(bitdepth == 32 ? 16 : bitdepth);
    } else if (input_frame->pixel_format.num_channels == 3) {
      is_gray = false;
      has_alpha = false;
    } else if (input_frame->pixel_format.num_channels == 4) {
      is_gray = false;
      has_alpha = true;
      io.metadata.m.SetAlphaBits(bitdepth == 32 ? 16 : bitdepth);
    } else {
      return JXL_ENC_ERROR;
    }

    if (input_frame->pixel_format.data_type == JXL_TYPE_FLOAT) {
      c_current = jxl::ColorEncoding::LinearSRGB(is_gray);
    } else {
      c_current = jxl::ColorEncoding::SRGB(is_gray);
    }
    io.metadata.m.color_encoding = c_current;

    if (!ConvertImage(jxl::Span<const uint8_t>(input_frame->buffer.data(),
                                               input_frame->buffer.size()),
                      metadata.xsize(), metadata.ysize(), c_current, has_alpha,
                      /*alpha_is_premultiplied=*/false, bitdepth,
                      input_frame->pixel_format.endianness, /*flipped_y=*/false,
                      this->thread_pool.get(), &io.Main())) {
      return JXL_ENC_ERROR;
    }
    io.SetSize(io.Main().xsize(), io.Main().ysize());
    io.CheckMetadata();

    jxl::CompressParams cparams = input_frame->option_values.cparams;
    if (input_frame->option_values.lossless) {
      cparams.SetLossless();
    }

    jxl::BitWriter writer;

    if (!wrote_headers) {
      if (!WriteHeaders(cparams, &io, &metadata, &writer, nullptr)) {
        return JXL_ENC_ERROR;
      }
      // Only send ICC (at least several hundred bytes) if fields aren't enough.
      if (metadata.m.color_encoding.WantICC()) {
        if (!jxl::WriteICC(metadata.m.color_encoding.ICC(), &writer,
                           jxl::kLayerHeader, nullptr)) {
          return JXL_ENC_ERROR;
        }
      }
      if (metadata.m.have_preview) {
        if (!jxl::EncodePreview(cparams, io.preview_frame, &metadata,
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
    if (!jxl::EncodeFrame(cparams, jxl::FrameInfo{}, &metadata, io.frames[0],
                          &enc_state, this->thread_pool.get(), &writer,
                          /*aux_out=*/nullptr)) {
      return JXL_ENC_ERROR;
    }

    jxl::PaddedBytes bytes = std::move(writer).TakeBytes();
    this->output_byte_queue =
        std::vector<uint8_t>(bytes.data(), bytes.data() + bytes.size());
    return JXL_ENC_SUCCESS;
  }
};

struct JxlEncoderOptionsStruct {
  JxlEncoder* enc;
  JxlEncoderOptionsValues values;
};

JxlEncoderStatus JxlEncoderSetDimensions(JxlEncoder* enc, const size_t xsize,
                                         const size_t ysize) {
  if (enc->metadata.size.Set(xsize, ysize)) {
    return JXL_ENC_SUCCESS;
  }
  return JXL_ENC_ERROR;
}

JxlEncoderOptions* JxlEncoderOptionsCreate(JxlEncoder* enc,
                                           const JxlEncoderOptions* source) {
  auto opts =
      jxl::MemoryManagerMakeUnique<JxlEncoderOptions>(&enc->memory_manager);
  if (!opts) return nullptr;
  opts->enc = enc;
  if (source != nullptr) {
    opts->values = source->values;
  } else {
    opts->values.lossless = false;
  }
  JxlEncoderOptions* ret = opts.get();
  enc->encoder_options.emplace_back(std::move(opts));
  return ret;
}

JxlEncoderStatus JxlEncoderOptionsSetLossless(JxlEncoderOptions* options,
                                              const JXL_BOOL lossless) {
  options->values.lossless = lossless;
  return JXL_ENC_SUCCESS;
}

JxlEncoderStatus JxlEncoderOptionsSetEffort(JxlEncoderOptions* options,
                                            const int effort) {
  if (effort < 3 || effort > 9) {
    return JXL_ENC_ERROR;
  }
  options->values.cparams.speed_tier = static_cast<jxl::SpeedTier>(10 - effort);
  return JXL_ENC_SUCCESS;
}

JxlEncoderStatus JxlEncoderOptionsSetDistance(JxlEncoderOptions* options,
                                              float distance) {
  if (distance < 0 || distance > 15) {
    return JXL_ENC_ERROR;
  }
  options->values.cparams.butteraugli_distance = distance;
  return JXL_ENC_SUCCESS;
}

JxlEncoder* JxlEncoderCreate(const JxlMemoryManager* memory_manager) {
  JxlMemoryManager local_memory_manager;
  if (!jxl::MemoryManagerInit(&local_memory_manager, memory_manager)) {
    return nullptr;
  }

  void* alloc =
      jxl::MemoryManagerAlloc(&local_memory_manager, sizeof(JxlEncoder));
  if (!alloc) return nullptr;
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
  enc->thread_pool = jxl::MemoryManagerMakeUnique<jxl::ThreadPool>(
      &enc->memory_manager, parallel_runner, parallel_runner_opaque);
  if (!enc->thread_pool) {
    return JXL_ENC_ERROR;
  }
  return JXL_ENC_SUCCESS;
}

JxlEncoderStatus JxlEncoderAddImageFrame(JxlEncoderOptions* options,
                                         const JxlPixelFormat* pixel_format,
                                         void* buffer, size_t size) {
  // TODO(zond): Return error if the input has been closed.
  auto frame = jxl::MemoryManagerMakeUnique<JxlEncoderQueuedFrame>(
      &options->enc->memory_manager,
      // JxlEncoderQueuedFrame is a struct with no constructors, so we use the
      // default move constructor there.
      JxlEncoderQueuedFrame{
          *pixel_format, options->values,
          std::vector<uint8_t>(static_cast<uint8_t*>(buffer),
                               static_cast<uint8_t*>(buffer) + size)});
  if (!frame) {
    return JXL_ENC_ERROR;
  }
  options->enc->input_frame_queue.emplace_back(std::move(frame));
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
