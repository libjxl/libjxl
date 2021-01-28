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

#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/external_image.h"
#include "lib/jxl/icc_codec.h"

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

JxlEncoderStatus JxlEncoderStruct::RefillOutputByteQueue() {
  jxl::MemoryManagerUniquePtr<jxl::JxlEncoderQueuedFrame> input_frame =
      std::move(this->input_frame_queue[0]);
  this->input_frame_queue.erase(this->input_frame_queue.begin());

  jxl::BitWriter writer;

  if (!wrote_headers) {
    if (!WriteHeaders(&metadata, &writer, nullptr)) {
      return JXL_ENC_ERROR;
    }
    // Only send ICC (at least several hundred bytes) if fields aren't enough.
    if (metadata.m.color_encoding.WantICC()) {
      if (!jxl::WriteICC(metadata.m.color_encoding.ICC(), &writer,
                         jxl::kLayerHeader, nullptr)) {
        return JXL_ENC_ERROR;
      }
    }

    // TODO(lode): preview should be added here if a preview image is added

    wrote_headers = true;
  }

  // Each frame should start on byte boundaries.
  writer.ZeroPadToByte();

  // TODO(zond): Handle progressive mode like EncodeFile does it.
  // TODO(zond): Handle animation like EncodeFile does it, by checking if
  //             JxlEncoderCloseInput has been called (to see if it's the
  //             last animation frame).

  jxl::PassesEncoderState enc_state;
  if (!jxl::EncodeFrame(input_frame->option_values.cparams, jxl::FrameInfo{}, &metadata,
                        input_frame->frame, &enc_state, this->thread_pool.get(),
                        &writer,
                        /*aux_out=*/nullptr)) {
    return JXL_ENC_ERROR;
  }

  jxl::PaddedBytes bytes = std::move(writer).TakeBytes();
  this->output_byte_queue =
      std::vector<uint8_t>(bytes.data(), bytes.data() + bytes.size());
  last_used_cparams = input_frame->option_values.cparams;
  return JXL_ENC_SUCCESS;
}

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

void JxlEncoderReset(JxlEncoder* enc) {
  enc->thread_pool.reset();
  enc->input_frame_queue.clear();
  enc->encoder_options.clear();
  enc->output_byte_queue.clear();
  enc->wrote_headers = false;
  enc->metadata = jxl::CodecMetadata();
  enc->last_used_cparams = jxl::CompressParams();
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

JxlEncoderStatus JxlEncoderAddJPEGFrame(const JxlEncoderOptions* options,
                                        const uint8_t* buffer, size_t size) {
  return JXL_ENC_SUCCESS;
}

JxlEncoderStatus JxlEncoderAddImageFrame(const JxlEncoderOptions* options,
                                         const JxlPixelFormat* pixel_format,
                                         const void* buffer,
                                         const size_t size) {
  // TODO(zond): Return error if the input has been closed.
  auto queued_frame = jxl::MemoryManagerMakeUnique<jxl::JxlEncoderQueuedFrame>(
      &options->enc->memory_manager,
      // JxlEncoderQueuedFrame is a struct with no constructors, so we use the
      // default move constructor there.
      jxl::JxlEncoderQueuedFrame{options->values,
                                 jxl::ImageBundle(&options->enc->metadata.m)});
  if (!queued_frame) {
    return JXL_ENC_ERROR;
  }

  jxl::ColorEncoding c_current;
  bool has_alpha;
  bool is_gray;
  size_t bitdepth;

  // TODO(zond): Make this accept more than float and uint8/16.
  if (pixel_format->data_type == JXL_TYPE_FLOAT) {
    bitdepth = 32;
    if (!options->enc->wrote_headers)
      options->enc->metadata.m.SetFloat32Samples();
  } else if (pixel_format->data_type == JXL_TYPE_UINT8) {
    bitdepth = 8;
    if (!options->enc->wrote_headers)
      options->enc->metadata.m.SetUintSamples(bitdepth);
  } else if (pixel_format->data_type == JXL_TYPE_UINT16) {
    bitdepth = 16;
    if (!options->enc->wrote_headers)
      options->enc->metadata.m.SetUintSamples(bitdepth);
  } else {
    return JXL_ENC_ERROR;
  }

  if (pixel_format->num_channels == 1) {
    has_alpha = false;
    is_gray = true;
  } else if (pixel_format->num_channels == 2) {
    is_gray = true;
    has_alpha = true;
    if (!options->enc->wrote_headers)
      options->enc->metadata.m.SetAlphaBits(bitdepth == 32 ? 16 : bitdepth);
  } else if (pixel_format->num_channels == 3) {
    is_gray = false;
    has_alpha = false;
  } else if (pixel_format->num_channels == 4) {
    is_gray = false;
    has_alpha = true;
    if (!options->enc->wrote_headers)
      options->enc->metadata.m.SetAlphaBits(bitdepth == 32 ? 16 : bitdepth);
  } else {
    return JXL_ENC_ERROR;
  }

  if (pixel_format->data_type == JXL_TYPE_FLOAT) {
    c_current = jxl::ColorEncoding::LinearSRGB(is_gray);
  } else {
    c_current = jxl::ColorEncoding::SRGB(is_gray);
  }

  if (options->values.lossless) {
    queued_frame->option_values.cparams.SetLossless();
  }

  if (!options->enc->wrote_headers) {
    options->enc->metadata.m.color_encoding = c_current;
    options->enc->metadata.m.xyb_encoded =
        queued_frame->option_values.cparams.color_transform == jxl::ColorTransform::kXYB;
  }

  if (!ConvertImage(jxl::Span<const uint8_t>(
                        static_cast<uint8_t*>(const_cast<void*>(buffer)), size),
                    options->enc->metadata.xsize(),
                    options->enc->metadata.ysize(), c_current, has_alpha,
                    /*alpha_is_premultiplied=*/false, bitdepth,
                    pixel_format->endianness, /*flipped_y=*/false,
                    options->enc->thread_pool.get(), &(queued_frame->frame))) {
    return JXL_ENC_ERROR;
  }
  queued_frame->frame.VerifyMetadata();

  options->enc->input_frame_queue.emplace_back(std::move(queued_frame));
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
