// Copyright 2020 Google LLC
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

#ifndef HWY_ALIGNED_ALLOCATOR_H_
#define HWY_ALIGNED_ALLOCATOR_H_

// Memory allocator with support for alignment and offsets.

#include <stddef.h>
#include <memory>

namespace hwy {

// Pointers to functions equivalent to malloc/free.
using AllocPtr = void* (*)(size_t bytes);
using FreePtr = void (*)(void* memory);

// Returns null or a pointer to at least `payload_size` (which can be zero)
// bytes of newly allocated memory suitable for Load/Store. Calls `alloc` to
// obtain memory; if null, uses the default malloc.
void* AllocateAlignedBytes(size_t payload_size, AllocPtr alloc_ptr);

// Frees all memory. No effect if `aligned_pointer` == nullptr, otherwise it
// must have been returned from a previous call to `AllocateAlignedBytes`.
// Calls `free` to free the memory; if null, uses the default free().
void FreeAlignedBytes(const void* aligned_pointer, FreePtr free_ptr);

class AlignedDeleter {
 public:
  explicit AlignedDeleter(FreePtr free) : free_(free) {}

  template <typename T>
  void operator()(T* aligned_pointer) const {
    return FreeAlignedBytes(aligned_pointer, free_);
  }

 private:
  FreePtr free_;
};

// Unique pointer to single POD, or (if T is U[]) an array of POD.
template <typename T>
using AlignedUniquePtr = std::unique_ptr<T, AlignedDeleter>;

template <typename T>
AlignedUniquePtr<T[]> AllocateAligned(const size_t items, AllocPtr alloc,
                                      FreePtr free) {
  return AlignedUniquePtr<T[]>(
      static_cast<T*>(AllocateAlignedBytes(items * sizeof(T), alloc)),
      AlignedDeleter(free));
}
template <typename T>
AlignedUniquePtr<T> AllocateSingleAligned(AllocPtr alloc, FreePtr free) {
  return AlignedUniquePtr<T>(
      static_cast<T*>(AllocateAlignedBytes(sizeof(T), alloc)),
      AlignedDeleter(free));
}

// Same, using default allocate/free functions.
template <typename T>
AlignedUniquePtr<T[]> AllocateAligned(const size_t items) {
  return AllocateAligned<T>(items, nullptr, nullptr);
}
template <typename T>
AlignedUniquePtr<T> AllocateSingleAligned() {
  return AllocateSingleAligned<T>(nullptr, nullptr);
}

}  // namespace hwy

#endif  // HWY_ALIGNED_ALLOCATOR_H_
