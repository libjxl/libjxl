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

#include "jxl/base/padded_bytes.h"

#include <string.h>

namespace jxl {

void PaddedBytes::IncreaseCapacityTo(size_t capacity) {
  JXL_ASSERT(capacity > capacity_);

  // BitWriter writes up to 7 bytes past the end.
  CacheAlignedUniquePtr new_data = AllocateArray(capacity + 8);
  if (new_data == nullptr) {
    // Allocation failed, discard all data to ensure this is noticed.
    size_ = capacity_ = 0;
    return;
  }

  if (data_ == nullptr) {
    // First allocation: ensure first byte is initialized (won't be copied).
    new_data[0] = 0;
  } else {
    // Subsequent resize: copy existing data to new location.
    memcpy(new_data.get(), data_.get(), size_);
    // Ensure that the first new byte is initialized, to allow write_bits to
    // safely append to the newly-resized PaddedBytes.
    new_data[size_] = 0;
  }

  capacity_ = capacity;
  std::swap(new_data, data_);
}

}  // namespace jxl
