// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/tracking_memory_manager.h"

#include <jxl/memory_manager.h>

#include <mutex>

#include "lib/jxl/base/status.h"

namespace jpegxl {
namespace tools {

TrackingMemoryManager::TrackingMemoryManager(JxlMemoryManager* inner,
                                             size_t cap)
    : cap_(cap), inner_(inner) {
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
    size_t new_bytes_in_use = self->bytes_in_use_ + size;
    if (self->cap_ && new_bytes_in_use > self->cap_) {
      // Soft "OOM"
      return nullptr;
    }
    self->num_allocations++;
    self->bytes_in_use_ = new_bytes_in_use;
    self->max_bytes_in_use = std::max(self->max_bytes_in_use, new_bytes_in_use);
  }
  void* result = self->inner_->alloc(self->inner_->opaque, size);
  if (result != nullptr) {
    std::lock_guard<std::mutex> guard(self->map_mutex_);
    self->allocations_[result] = size;
  } else {
    // Hard "OOM" - rollback accounting.
    std::lock_guard<std::mutex> guard(self->numbers_mutex_);
    self->num_allocations--;
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
    self->num_allocations--;
    self->bytes_in_use_ -= size;
  }
  self->inner_->free(self->inner_->opaque, address);
}

}  // namespace tools
}  // namespace jpegxl
