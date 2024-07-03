// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_TRACKING_MEMORY_MANAGER_H_
#define TOOLS_TRACKING_MEMORY_MANAGER_H_

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace jpegxl {
namespace tools {

class TrackingMemoryManager {
 public:
  explicit TrackingMemoryManager(JxlMemoryManager* inner, size_t cap = 0);

  JxlMemoryManager* get() { return &outer_; }

  uint64_t max_bytes_in_use = 0;
  uint64_t num_allocations = 0;

 private:
  static void* Alloc(void* opaque, size_t size);
  static void Free(void* opaque, void* address);

  std::unordered_map<void*, size_t> allocations_;
  std::mutex numbers_mutex_;
  std::mutex map_mutex_;
  uint64_t cap_;
  uint64_t bytes_in_use_ = 0;
  JxlMemoryManager outer_;
  JxlMemoryManager* inner_;
};

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_TRACKING_MEMORY_MANAGER_H_
