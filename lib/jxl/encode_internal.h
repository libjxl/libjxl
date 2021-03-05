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

#ifndef LIB_JXL_ENCODE_INTERNAL_H_
#define LIB_JXL_ENCODE_INTERNAL_H_

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

typedef std::array<uint8_t, 4> BoxType;

// Utility function that makes a BoxType from a null terminated string literal.
constexpr BoxType MakeBoxType(const char (&type)[5]) {
  return BoxType({static_cast<uint8_t>(type[0]), static_cast<uint8_t>(type[1]),
                  static_cast<uint8_t>(type[2]),
                  static_cast<uint8_t>(type[3])});
}

}  // namespace jxl

struct JxlEncoderStruct {
  JxlMemoryManager memory_manager;
  jxl::MemoryManagerUniquePtr<jxl::ThreadPool> thread_pool{
      nullptr, jxl::MemoryManagerDeleteHelper(&memory_manager)};
  std::vector<jxl::MemoryManagerUniquePtr<JxlEncoderOptions>> encoder_options;

  std::vector<jxl::MemoryManagerUniquePtr<jxl::JxlEncoderQueuedFrame>>
      input_frame_queue;
  std::vector<uint8_t> output_byte_queue;

  bool use_container = false;
  bool store_jpeg_metadata = false;
  jxl::CodecMetadata metadata;
  std::vector<uint8_t> jpeg_metadata;

  bool wrote_headers = false;
  jxl::CompressParams last_used_cparams;

  // Takes the first frame in the input_frame_queue, encodes it, and appends the
  // bytes to the output_byte_queue.
  JxlEncoderStatus RefillOutputByteQueue();

  // Appends the bytes of a JXL box header with the provided type and size to
  // the end of the output_byte_queue. If unbounded is true, the size won't be
  // added to the header and the box will be assumed to continue until EOF.
  void AppendBoxHeader(const jxl::BoxType& type, size_t size, bool unbounded);
};

struct JxlEncoderOptionsStruct {
  JxlEncoder* enc;
  jxl::JxlEncoderOptionsValues values;
};

#endif  // LIB_JXL_ENCODE_INTERNAL_H_
