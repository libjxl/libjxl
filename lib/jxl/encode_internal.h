/* Copyright (c) the JPEG XL Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef JXL_ENCODE_INTERNAL_H_
#define JXL_ENCODE_INTERNAL_H_

#include <vector>

#include "jxl/encode.h"
#include "jxl/memory_manager.h"
#include "jxl/parallel_runner.h"
#include "jxl/types.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

typedef struct JxlEncoderOptionsValuesStruct {
  // lossless is a separate setting from cparams because it is a combination
  // setting that overrides multiple settings inside of cparams.
  bool lossless;
  jxl::CompressParams cparams;
} JxlEncoderOptionsValues;

typedef struct JxlEncoderQueuedFrame {
  JxlEncoderOptionsValues option_values;
  jxl::ImageBundle frame;
} JxlEncoderQueuedFrame;

Status ConvertExternalToInternalColorEncoding(const JxlColorEncoding& external,
                                              jxl::ColorEncoding* internal);

JxlEncoderStatus BufferToImageBundle(const JxlPixelFormat& pixel_format,
                                     uint32_t xsize, uint32_t ysize,
                                     const void* buffer, size_t size,
                                     jxl::ThreadPool* pool,
                                     const jxl::ColorEncoding& c_current,
                                     jxl::ImageBundle* ib);

}  // namespace jxl

struct JxlEncoderStruct {
  JxlMemoryManager memory_manager;
  jxl::MemoryManagerUniquePtr<jxl::ThreadPool> thread_pool{
      nullptr, jxl::MemoryManagerDeleteHelper(&memory_manager)};
  std::vector<jxl::MemoryManagerUniquePtr<jxl::JxlEncoderQueuedFrame>>
      input_frame_queue;
  std::vector<jxl::MemoryManagerUniquePtr<JxlEncoderOptions>> encoder_options;
  std::vector<uint8_t> output_byte_queue;
  bool wrote_headers;
  jxl::CodecMetadata metadata;
  jxl::CompressParams last_used_cparams;

  JxlEncoderStatus RefillOutputByteQueue();
};

struct JxlEncoderOptionsStruct {
  JxlEncoder* enc;
  jxl::JxlEncoderOptionsValues values;
};

#endif /* JXL_ENCODE_INTERNAL_H_ */
