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

#include "jpegxl/encode.h"

#include "jxl/base/data_parallel.h"
#include "jxl/memory_manager_internal.h"

// Debug-printing failure macro similar to JXL_FAILURE, but for the status code
// JXL_ENC_ERROR
#ifdef JXL_CRASH_ON_ERROR
#define JXL_API_ERROR(format, ...)                                           \
  (::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__), \
   ::jxl::Abort(), JPEGXL_ENC_ERROR)
#else  // JXL_CRASH_ON_ERROR
#define JXL_API_ERROR(format, ...)                                             \
  (((JXL_DEBUG_ON_ERROR) &&                                                    \
    ::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__)), \
   JPEGXL_ENC_ERROR)
#endif  // JXL_CRASH_ON_ERROR

uint32_t JpegxlEncoderVersion(void) {
  return JPEGXL_MAJOR_VERSION * 1000000 + JPEGXL_MINOR_VERSION * 1000 +
         JPEGXL_PATCH_VERSION;
}

struct JpegxlEncoderStruct {
  JpegxlMemoryManager memory_manager;
  std::unique_ptr<jxl::ThreadPool> thread_pool;
};

JpegxlEncoder* JpegxlEncoderCreate(const JpegxlMemoryManager* memory_manager) {
  JpegxlMemoryManager local_memory_manager;
  if (!jxl::MemoryManagerInit(&local_memory_manager, memory_manager))
    return nullptr;

  void* alloc =
      jxl::MemoryManagerAlloc(&local_memory_manager, sizeof(JpegxlEncoder));
  if (!alloc) return nullptr;
  // Placement new constructor on allocated memory
  JpegxlEncoder* enc = new (alloc) JpegxlEncoder();
  enc->memory_manager = local_memory_manager;

  return enc;
}

void JpegxlEncoderDestroy(JpegxlEncoder* enc) {
  if (enc) {
    // Call destructor directly since custom free function is used.
    enc->~JpegxlEncoder();
    jxl::MemoryManagerFree(&enc->memory_manager, enc);
  }
}

JPEGXL_EXPORT JpegxlEncoderStatus JpegxlEncoderSetParallelRunner(
    JpegxlEncoder* enc, JpegxlParallelRunner parallel_runner,
    void* parallel_runner_opaque) {
  if (enc->thread_pool) return JXL_API_ERROR("parallel runner already set");
  enc->thread_pool.reset(
      new jxl::ThreadPool(parallel_runner, parallel_runner_opaque));
  return JPEGXL_ENC_SUCCESS;
}

