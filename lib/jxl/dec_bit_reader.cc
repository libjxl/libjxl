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

#include "lib/jxl/dec_bit_reader.h"

#include <algorithm>
#include <limits>

namespace jxl {

size_t BitReader::Suspend() {
  JXL_ASSERT(AllReadsWithinBounds());
  size_t reclaim_bytes = bits_in_buf_ / 8;
  bits_in_buf_ %= 8;

  // Clear out the higher bits in the buf_. These will be | with the next byte
  // after resume which might be different.
  buf_ &= ~(std::numeric_limits<uint64_t>::max() << bits_in_buf_);

  if (overread_bytes_ > 0) {
    size_t reclaim_overread = std::min<size_t>(reclaim_bytes, overread_bytes_);
    overread_bytes_ -= reclaim_overread;
    reclaim_bytes -= reclaim_overread;
  }

  next_byte_ -= reclaim_bytes;
  // Truncate the source buffer.
  size_t unused_bytes = end_minus_8_ + 8 - next_byte_;
  end_minus_8_ -= unused_bytes;
  return unused_bytes;
}

}  // namespace jxl
