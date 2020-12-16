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

// Library to compute the Lehmer code of a permutation and to reconstruct the
// permutation from its Lehmer code. For more details on Lehmer codes, see
// http://en.wikipedia.org/wiki/Lehmer_code

#ifndef LIB_JXL_JPEG_JPEG_LEHMER_CODE_H_
#define LIB_JXL_JPEG_JPEG_LEHMER_CODE_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace jpeg {

// This class is an optimized Lehmer-like coder that takes the remaining
// number of possible values into account to reduce the bit usage.
// TODO(eustas): in worst case (always removing the first element), O(N^2)
// elements are moved; "Fenwick tree" is simple to implement and could reduce
// the complexity to O(N * log(N)).
class PermutationCoder {
 public:
  PermutationCoder() {}

  void Init(std::vector<uint8_t> values) {
    values_ = std::move(values);
  }

  void Clear() {
    std::vector<uint8_t>().swap(values_);
  }

  // number of bits needed to represent the next code.
  int num_bits() const {
    size_t num_values = values_.size();
    JXL_DASSERT(num_values > 0);
    return num_values <= 1 ? 0 : (FloorLog2Nonzero(num_values - 1) + 1);
  }

  // Copy value at position 'code' and remove it. Returns false in
  // case of error (invalid slot).
  bool Remove(size_t code, uint8_t* value) {
    if (code >= values_.size()) {
      return false;
    }
    *value = values_[code];
    values_.erase(values_.begin() + code);
    return true;
  }

  // Removes 'value' from the list and assign a code + number-of-bits
  // for it. Returns false if value could not be encoded.
  bool RemoveValue(uint8_t value, int* code, int* nbits) {
    std::vector<uint8_t>::iterator it =
        std::find(values_.begin(), values_.end(), value);
    if (it == values_.end()) {
      return false;  // invalid/non-existing value was passed.
    }
    *code = it - values_.begin();
    *nbits = num_bits();
    values_.erase(it);
    return true;
  }

 private:
  std::vector<uint8_t> values_;
};

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_JPEG_LEHMER_CODE_H_
