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

#include "jxl/entropy_coder.h"

#include <stdint.h>
#include "gtest/gtest.h"

namespace jxl {
namespace {

TEST(EntropyCoderTest, PackUnpack) {
  for (int32_t i = -31; i < 32; ++i) {
    uint32_t packed = PackSigned(i);
    EXPECT_LT(packed, 63);
    int32_t unpacked = UnpackSigned(packed);
    EXPECT_EQ(i, unpacked);
  }
}

TEST(EntropyCoderTest, EncodeDecodeVarUint) {
  // When n == 0 there is only one, but most important case 0 <-> (0, 0).
  for (int n = 0; n < 6; ++n) {
    uint32_t count = 1 << n;
    uint32_t base = count - 1;
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t nbits = 0xFFFFFFFF;
      uint32_t bits = 0xFFFFFFFF;
      uint32_t value = base + i;
      EncodeVarLenUint(value, &nbits, &bits);
      EXPECT_EQ(n, nbits);
      EXPECT_EQ(i, bits);
      EXPECT_EQ(value, DecodeVarLenUint(nbits, bits));
    }
  }
}

}  // namespace
}  // namespace jxl
