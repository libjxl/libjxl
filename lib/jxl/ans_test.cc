// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_ans_simd.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

void RoundtripTestcase(int n_histograms, int alphabet_size,
                       const std::vector<Token>& input_values) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr uint16_t kMagic1 = 0x9e33;
  constexpr uint16_t kMagic2 = 0x8b04;

  BitWriter writer{memory_manager};
  // Space for magic bytes.
  ASSERT_TRUE(writer.WithMaxBits(16, LayerType::Header, nullptr, [&] {
    writer.Write(16, kMagic1);
    return true;
  }));

  EntropyEncodingData codes;
  std::vector<std::vector<Token>> input_values_vec;
  input_values_vec.push_back(input_values);

  JXL_TEST_ASSIGN_OR_DIE(
      size_t cost,
      BuildAndEncodeHistograms(memory_manager, HistogramParams(), n_histograms,
                               input_values_vec, &codes, &writer,
                               LayerType::Header, nullptr));
  (void)cost;
  ASSERT_TRUE(WriteTokens(input_values_vec[0], codes, 0, &writer,
                          LayerType::Header, nullptr));

  // Magic bytes + padding
  ASSERT_TRUE(writer.WithMaxBits(24, LayerType::Header, nullptr, [&] {
    writer.Write(16, kMagic2);
    writer.ZeroPadToByte();
    return true;
  }));

  // We do not truncate the output. Reading past the end reads out zeroes
  // anyway.
  BitReader br(writer.GetSpan());

  ASSERT_EQ(br.ReadBits(16), kMagic1);

  std::vector<uint8_t> dec_context_map;
  ANSCode decoded_codes;
  ASSERT_TRUE(DecodeHistograms(memory_manager, &br, n_histograms,
                               &decoded_codes, &dec_context_map));
  ASSERT_EQ(dec_context_map, codes.context_map);
  JXL_TEST_ASSIGN_OR_DIE(ANSSymbolReader reader,
                         ANSSymbolReader::Create(&decoded_codes, &br));

  for (const Token& symbol : input_values) {
    uint32_t read_symbol =
        reader.ReadHybridUint(symbol.context, &br, dec_context_map);
    ASSERT_EQ(read_symbol, symbol.value);
  }
  ASSERT_TRUE(reader.CheckANSFinalState());

  ASSERT_EQ(br.ReadBits(16), kMagic2);
  EXPECT_TRUE(br.Close());
}

TEST(ANSTest, EmptyRoundtrip) {
  RoundtripTestcase(2, ANS_MAX_ALPHABET_SIZE, std::vector<Token>());
}

TEST(ANSTest, SingleSymbolRoundtrip) {
  for (uint32_t i = 0; i < ANS_MAX_ALPHABET_SIZE; i++) {
    RoundtripTestcase(2, ANS_MAX_ALPHABET_SIZE, {{0, i}});
  }
  for (uint32_t i = 0; i < ANS_MAX_ALPHABET_SIZE; i++) {
    RoundtripTestcase(2, ANS_MAX_ALPHABET_SIZE,
                      std::vector<Token>(1024, {0, i}));
  }
}

#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
constexpr size_t kReps = 3;
#else
constexpr size_t kReps = 10;
#endif

void RoundtripRandomStream(int alphabet_size, size_t reps = kReps,
                           size_t num = 1 << 18) {
  constexpr int kNumHistograms = 3;
  Rng rng(0);
  for (size_t i = 0; i < reps; i++) {
    std::vector<Token> symbols;
    for (size_t j = 0; j < num; j++) {
      int context = rng.UniformI(0, kNumHistograms);
      int value = rng.UniformU(0, alphabet_size);
      symbols.emplace_back(context, value);
    }
    RoundtripTestcase(kNumHistograms, alphabet_size, symbols);
  }
}

void RoundtripRandomUnbalancedStream(int alphabet_size) {
  constexpr int kNumHistograms = 3;
  constexpr int kPrecision = 1 << 10;
  Rng rng(0);
  for (size_t i = 0; i < kReps; i++) {
    std::vector<int> distributions[kNumHistograms] = {};
    for (auto& distr : distributions) {
      distr.resize(kPrecision);
      int symbol = 0;
      int remaining = 1;
      for (int k = 0; k < kPrecision; k++) {
        if (remaining == 0) {
          if (symbol < alphabet_size - 1) symbol++;
          // There is no meaning behind this distribution: it's anything that
          // will create a nonuniform distribution and won't have too few
          // symbols usually. Also we want different distributions we get to be
          // sufficiently dissimilar.
          remaining = rng.UniformU(0, kPrecision - k + 1);
        }
        distr[k] = symbol;
        remaining--;
      }
    }
    std::vector<Token> symbols;
    for (int j = 0; j < 1 << 18; j++) {
      int context = rng.UniformI(0, kNumHistograms);
      int value = rng.UniformU(0, kPrecision);
      symbols.emplace_back(context, value);
    }
    RoundtripTestcase(kNumHistograms + 1, alphabet_size, symbols);
  }
}

TEST(ANSTest, RandomStreamRoundtrip3Small) { RoundtripRandomStream(3, 1, 16); }

TEST(ANSTest, RandomStreamRoundtrip3) { RoundtripRandomStream(3); }

TEST(ANSTest, RandomStreamRoundtripBig) {
  RoundtripRandomStream(ANS_MAX_ALPHABET_SIZE);
}

TEST(ANSTest, RandomUnbalancedStreamRoundtrip3) {
  RoundtripRandomUnbalancedStream(3);
}

TEST(ANSTest, RandomUnbalancedStreamRoundtripBig) {
  RoundtripRandomUnbalancedStream(ANS_MAX_ALPHABET_SIZE);
}

TEST(ANSTest, UintConfigRoundtrip) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  for (size_t log_alpha_size = 5; log_alpha_size <= 8; log_alpha_size++) {
    std::vector<HybridUintConfig> uint_config;
    std::vector<HybridUintConfig> uint_config_dec;
    for (size_t i = 0; i < log_alpha_size; i++) {
      for (size_t j = 0; j <= i; j++) {
        for (size_t k = 0; k <= i - j; k++) {
          uint_config.emplace_back(i, j, k);
        }
      }
    }
    uint_config.emplace_back(log_alpha_size, 0, 0);
    uint_config_dec.resize(uint_config.size());
    BitWriter writer{memory_manager};
    ASSERT_TRUE(writer.WithMaxBits(
        10 * uint_config.size(), LayerType::Header, nullptr, [&] {
          EncodeUintConfigs(uint_config, &writer, log_alpha_size);
          return true;
        }));
    writer.ZeroPadToByte();
    BitReader br(writer.GetSpan());
    EXPECT_TRUE(DecodeUintConfigs(log_alpha_size, &uint_config_dec, &br));
    EXPECT_TRUE(br.Close());
    for (size_t i = 0; i < uint_config.size(); i++) {
      EXPECT_EQ(uint_config[i].split_token, uint_config_dec[i].split_token);
      EXPECT_EQ(uint_config[i].msb_in_token, uint_config_dec[i].msb_in_token);
      EXPECT_EQ(uint_config[i].lsb_in_token, uint_config_dec[i].lsb_in_token);
    }
  }
}

void TestCheckpointing(bool ans, bool lz77) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  std::vector<std::vector<Token>> input_values(1);
  for (size_t i = 0; i < 1024; i++) {
    input_values[0].emplace_back(0, i % 4);
  }
  // up to lz77 window size.
  for (size_t i = 0; i < (1 << 20) - 1022; i++) {
    input_values[0].emplace_back(0, (i % 5) + 4);
  }
  // Ensure that when the window wraps around, new values are different.
  input_values[0].emplace_back(0, 0);
  for (size_t i = 0; i < 1024; i++) {
    input_values[0].emplace_back(0, i % 4);
  }

  EntropyEncodingData codes;
  HistogramParams params;
  params.lz77_method = lz77 ? HistogramParams::LZ77Method::kLZ77
                            : HistogramParams::LZ77Method::kNone;
  params.force_huffman = !ans;

  BitWriter writer{memory_manager};
  {
    auto input_values_copy = input_values;
    JXL_TEST_ASSIGN_OR_DIE(
        size_t cost,
        BuildAndEncodeHistograms(memory_manager, params, 1, input_values_copy,
                                 &codes, &writer, LayerType::Header, nullptr));
    (void)cost;
    ASSERT_TRUE(WriteTokens(input_values_copy[0], codes, 0, &writer,
                            LayerType::Header, nullptr));
    writer.ZeroPadToByte();
  }

  // We do not truncate the output. Reading past the end reads out zeroes
  // anyway.
  BitReader br(writer.GetSpan());
  Status status = true;
  {
    BitReaderScopedCloser bc(br, status);

    std::vector<uint8_t> dec_context_map;
    ANSCode decoded_codes;
    ASSERT_TRUE(DecodeHistograms(memory_manager, &br, 1, &decoded_codes,
                                 &dec_context_map));
    ASSERT_EQ(dec_context_map, codes.context_map);
    JXL_TEST_ASSIGN_OR_DIE(ANSSymbolReader reader,
                           ANSSymbolReader::Create(&decoded_codes, &br));

    auto checkpoint = jxl::make_unique<ANSSymbolReader::Checkpoint>();
    size_t br_pos = 0;
    constexpr size_t kInterval = ANSSymbolReader::kMaxCheckpointInterval - 2;
    for (size_t i = 0; i < input_values[0].size(); i++) {
      if (i % kInterval == 0 && i > 0) {
        reader.Restore(*checkpoint);
        ASSERT_TRUE(br.Close());
        br = BitReader(writer.GetSpan());
        br.SkipBits(br_pos);
        for (size_t j = i - kInterval; j < i; j++) {
          Token symbol = input_values[0][j];
          uint32_t read_symbol =
              reader.ReadHybridUint(symbol.context, &br, dec_context_map);
          ASSERT_EQ(read_symbol, symbol.value) << "j = " << j;
        }
      }
      if (i % kInterval == 0) {
        reader.Save(checkpoint.get());
        br_pos = br.TotalBitsConsumed();
      }
      Token symbol = input_values[0][i];
      uint32_t read_symbol =
          reader.ReadHybridUint(symbol.context, &br, dec_context_map);
      ASSERT_EQ(read_symbol, symbol.value) << "i = " << i;
    }
    ASSERT_TRUE(reader.CheckANSFinalState());
  }
  EXPECT_TRUE(status);
}

TEST(ANSTest, TestCheckpointingANS) {
  TestCheckpointing(/*ans=*/true, /*lz77=*/false);
}

TEST(ANSTest, TestCheckpointingPrefix) {
  TestCheckpointing(/*ans=*/false, /*lz77=*/false);
}

TEST(ANSTest, TestCheckpointingANSLZ77) {
  TestCheckpointing(/*ans=*/true, /*lz77=*/true);
}

TEST(ANSTest, TestCheckpointingPrefixLZ77) {
  TestCheckpointing(/*ans=*/false, /*lz77=*/true);
}

std::string AnsSimdTestDescription(
    const testing::TestParamInfo<HybridUintConfig>& info) {
  std::stringstream name;
  const HybridUintConfig& cfg = info.param;
  name << cfg.split_exponent << "_" << cfg.msb_in_token << "_" << cfg.lsb_in_token;
  return name.str();
}

std::vector<HybridUintConfig> GenerateAnsSimdTests() {
  return {HybridUintConfig{0, 0, 0},  HybridUintConfig{2, 0, 1},
          HybridUintConfig{3, 1, 0},  HybridUintConfig{3, 1, 2},
          HybridUintConfig{3, 2, 0},  HybridUintConfig{3, 2, 1},
          HybridUintConfig{4, 1, 0},  HybridUintConfig{4, 1, 2},
          HybridUintConfig{4, 1, 3},  HybridUintConfig{4, 2, 0},
          HybridUintConfig{4, 2, 1},  HybridUintConfig{4, 2, 2},
          HybridUintConfig{5, 1, 0},  HybridUintConfig{5, 1, 2},
          HybridUintConfig{5, 1, 4},  HybridUintConfig{5, 2, 0},
          HybridUintConfig{5, 2, 1},  HybridUintConfig{5, 2, 2},
          HybridUintConfig{5, 2, 3},  HybridUintConfig{6, 0, 0},
          HybridUintConfig{6, 2, 4},  HybridUintConfig{7, 0, 0},
          HybridUintConfig{8, 0, 0},  HybridUintConfig{9, 0, 0},
          HybridUintConfig{10, 0, 0}, HybridUintConfig{11, 0, 0},
          HybridUintConfig{12, 0, 0}};
}

class AnsSimdTest : public ::testing::TestWithParam<HybridUintConfig> {};

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(AnsSimdTestAll,
                                   AnsSimdTest,
                                   testing::ValuesIn(GenerateAnsSimdTests()),
                                   AnsSimdTestDescription);

TEST_P(AnsSimdTest, EstimateTokenCost) {
  const HybridUintConfig& cfg = GetParam();
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr size_t kNumItems = 8 * 1024 * 1024;
  std::vector<uint32_t> in;
  std::vector<uint32_t> expected_out;
  constexpr size_t kTail = 1024;
  in.reserve(kNumItems + 311306 + kTail);
  uint32_t expected_eb = 0;
  for (size_t i = 0; i < kNumItems; ++i) {
    in.push_back(i);
  }
  for (size_t b = 23; b <= 31; ++b) {
    size_t base = 1 << b;
    for (size_t k = base - 16384; k <= base + 16384; ++k) {
      in.push_back(k);
    }
  }
  for (size_t n = 0; n <= 16384; ++n) {
    in.push_back(static_cast<uint32_t>(~0u) - n);
  }
  expected_out.reserve(in.size() + 1024);
  for (size_t i = 0; i < in.size(); ++i) {
    uint32_t v = in[i];
    uint32_t tok, nbits, bits;
    cfg.Encode(v, &tok, &nbits, &bits);
    expected_out.push_back(tok);
    expected_eb += nbits;
  }
  size_t data_size = in.size();
  in.resize(data_size + kTail);

  JXL_ASSIGN_OR_QUIT(
      AlignedMemory out,
      AlignedMemory::Create(memory_manager, data_size * sizeof(uint32_t)),
      "Failed to rallocate memory.");
  uint32_t eb = EstimateTokenCost(in.data(), data_size, cfg, out);
  ASSERT_EQ(eb, expected_eb);
  uint32_t* actual_out = out.address<uint32_t>();
  for (size_t i = 0; i < data_size; ++i) {
    ASSERT_EQ(actual_out[i], expected_out[i]) << i;
  }
}
}  // namespace
}  // namespace jxl
