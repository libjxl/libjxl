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

HWY_ATTR void HybridUintRoundtrip(HybridUintConfig config,
                                  size_t limit = 1 << 24) {
  std::mt19937 rng(0);
  std::uniform_int_distribution<uint32_t> dist(0, limit);
  constexpr size_t kNumIntegers = 1 << 20;
  std::vector<uint32_t> integers(kNumIntegers);
  std::vector<Token> tokens;
  for (size_t i = 0; i < kNumIntegers; i++) {
    integers[i] = dist(rng);
    TokenizeWithConfig(config, 0, integers[i], &tokens);
  }
  BitWriter writer;

  std::vector<uint8_t> context_map;
  EntropyEncodingData codes;

  BuildAndEncodeHistograms(HistogramParams(), 1, {tokens}, &codes, &context_map,
                           &writer, 0, nullptr);
  WriteTokens(tokens, codes, context_map, &writer, 0, nullptr, config);
  writer.ZeroPadToByte();

  BitReader br(writer.GetSpan());

  std::vector<uint8_t> dec_context_map;
  ANSCode decoded_codes;
  ASSERT_TRUE(DecodeHistograms(&br, 1, ANS_MAX_ALPHA_SIZE, &decoded_codes,
                               &dec_context_map));
  ASSERT_EQ(dec_context_map, context_map);
  ANSSymbolReader reader(&decoded_codes, &br);

  for (size_t i = 0; i < kNumIntegers; i++) {
    EXPECT_EQ(integers[i], reader.ReadHybridUint(0, &br, context_map));
  }
  EXPECT_TRUE(br.Close());
}

TEST(HybridUintTest, Test000) {
  HybridUintRoundtrip(HybridUintConfig{0, 0, 0});
}
TEST(HybridUintTest, Test411) {
  HybridUintRoundtrip(HybridUintConfig{4, 1, 1});
}
TEST(HybridUintTest, Test420) {
  HybridUintRoundtrip(HybridUintConfig{4, 2, 0});
}
TEST(HybridUintTest, Test421) {
  HybridUintRoundtrip(HybridUintConfig{4, 2, 1}, 256);
}

}  // namespace
}  // namespace jxl
