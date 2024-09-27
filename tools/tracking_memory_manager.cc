// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/tracking_memory_manager.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "lib/jxl/base/status.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jpegxl {
namespace tools {

TrackingMemoryManager::TrackingMemoryManager(uint64_t cap, uint64_t total_cap)
    : cap_(cap), total_cap_(total_cap) {
  jxl::Status status = jxl::MemoryManagerInit(&default_, nullptr);
  JXL_DASSERT(status);
  (void)status;
  inner_ = &default_;

  outer_.opaque = reinterpret_cast<void*>(this);
  outer_.alloc = &Alloc;
  outer_.free = &Free;
}

void* TrackingMemoryManager::Alloc(void* opaque, size_t size) {
  if (opaque == nullptr) {
    JXL_DEBUG_ABORT("Internal logic error");
    return nullptr;
  }
  TrackingMemoryManager* self =
      reinterpret_cast<TrackingMemoryManager*>(opaque);
  {
    std::lock_guard<std::mutex> guard(self->numbers_mutex_);
    uint64_t new_total = self->total_bytes_allocated + size;
    if (new_total < size ||
        (self->total_cap_ && new_total > self->total_cap_)) {
      // Brittle "OOM" - even freeing memory won't undo it.
      self->total_bytes_allocated = self->total_cap_;
      self->seen_oom = true;
      return nullptr;
    }
    uint64_t new_bytes_in_use = self->bytes_in_use_ + size;
    if (new_bytes_in_use < size ||
        (self->cap_ && new_bytes_in_use > self->cap_)) {
      // Soft "OOM"
      self->seen_oom = true;
      return nullptr;
    }
    self->num_allocations_++;
    self->total_allocations++;
    self->bytes_in_use_ = new_bytes_in_use;
    self->max_bytes_in_use = std::max(self->max_bytes_in_use, new_bytes_in_use);
    self->total_bytes_allocated = new_total;
  }
  void* result = self->inner_->alloc(self->inner_->opaque, size);
  if (result != nullptr) {
    std::lock_guard<std::mutex> guard(self->map_mutex_);
    self->allocations_[result] = size;
  } else {
    // Hard "OOM" - rollback accounting.
    std::lock_guard<std::mutex> guard(self->numbers_mutex_);
    self->seen_oom = true;
    self->num_allocations_--;
    self->bytes_in_use_ -= size;
  }
  return result;
}

void TrackingMemoryManager::Free(void* opaque, void* address) {
  if (opaque == nullptr) {
    JXL_DEBUG_ABORT("Internal logic error");
    return;
  }
  if (address == nullptr) return;
  TrackingMemoryManager* self =
      reinterpret_cast<TrackingMemoryManager*>(opaque);
  bool found = false;
  size_t size = 0;
  {
    std::lock_guard<std::mutex> guard(self->map_mutex_);
    auto entry = self->allocations_.find(address);
    if (entry != self->allocations_.end()) {
      found = true;
      size = entry->second;
      self->allocations_.erase(entry);
    } else {
      JXL_DEBUG_ABORT("Internal logic error");
    }
  }

  if (found) {
    std::lock_guard<std::mutex> guard(self->numbers_mutex_);
    self->num_allocations_--;
    self->bytes_in_use_ -= size;
  }
  self->inner_->free(self->inner_->opaque, address);
}

jxl::Status TrackingMemoryManager::Reset() {
  if (num_allocations_ != 0) {
    return JXL_FAILURE("Memory leak");
  }
  if (!allocations_.empty()) {
    return JXL_FAILURE("Internal logic error");
  }
  if (bytes_in_use_ != 0) {
    return JXL_FAILURE("Internal logic error");
  }
  seen_oom = false;
  max_bytes_in_use = 0;
  total_allocations = 0;
  total_bytes_allocated = 0;
  return true;
}

}  // namespace tools
}  // namespace jpegxl
