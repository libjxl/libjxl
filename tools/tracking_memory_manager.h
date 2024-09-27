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

#include "lib/jxl/base/status.h"

namespace jpegxl {
namespace tools {

const uint64_t kGiB = 1u << 30;

class TrackingMemoryManager {
 public:
  explicit TrackingMemoryManager(uint64_t cap = 0, uint64_t total_cap = 0);

  // void setInner(JxlMemoryManager* inner) { inner_ = inner; }

  JxlMemoryManager* get() { return &outer_; }

  jxl::Status Reset();

  bool seen_oom = false;
  uint64_t max_bytes_in_use = 0;
  uint64_t total_allocations = 0;
  uint64_t total_bytes_allocated = 0;

 private:
  static void* Alloc(void* opaque, size_t size);
  static void Free(void* opaque, void* address);

  std::unordered_map<void*, size_t> allocations_;
  std::mutex numbers_mutex_;
  std::mutex map_mutex_;
  uint64_t cap_;
  uint64_t total_cap_;
  uint64_t bytes_in_use_ = 0;
  uint64_t num_allocations_ = 0;
  JxlMemoryManager outer_;
  JxlMemoryManager default_;
  JxlMemoryManager* inner_;
};

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_TRACKING_MEMORY_MANAGER_H_
