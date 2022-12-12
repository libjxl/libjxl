// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_fast_lossless.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <vector>

#if (!defined(__BYTE_ORDER__) || (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__))
#error "system not known to be little endian"
#endif

#ifdef _MSC_VER
#define FJXL_INLINE __forceinline
#else
#define FJXL_INLINE inline __attribute__((always_inline))
#endif

namespace {

constexpr size_t kNumRawSymbols = 19;

struct BitWriter {
  void Allocate(size_t maximum_bit_size) {
    assert(data == nullptr);
    // Leave some padding.
    data.reset(static_cast<uint8_t*>(malloc(maximum_bit_size / 8 + 32)));
  }

  void Write(uint32_t count, uint64_t bits) {
    buffer |= bits << bits_in_buffer;
    bits_in_buffer += count;
    memcpy(data.get() + bytes_written, &buffer, 8);
    size_t bytes_in_buffer = bits_in_buffer / 8;
    bits_in_buffer -= bytes_in_buffer * 8;
    buffer >>= bytes_in_buffer * 8;
    bytes_written += bytes_in_buffer;
  }

  void ZeroPadToByte() {
    if (bits_in_buffer != 0) {
      Write(8 - bits_in_buffer, 0);
    }
  }

  std::unique_ptr<uint8_t[], void (*)(void*)> data = {nullptr, free};
  size_t bytes_written = 0;
  size_t bits_in_buffer = 0;
  uint64_t buffer = 0;
};

constexpr size_t kLZ77Offset = 224;
constexpr size_t kLZ77MinLength = 16;

struct PrefixCode {
  static constexpr size_t kNumLZ77 = 17;

  uint8_t raw_nbits[kNumRawSymbols] = {};
  uint8_t raw_bits[kNumRawSymbols] = {};

  alignas(32) uint8_t raw_nbits_simd[16] = {};
  alignas(32) uint8_t raw_bits_simd[16] = {};

  uint8_t lz77_nbits[kNumLZ77] = {};
  uint16_t lz77_bits[kNumLZ77] = {};

  static uint16_t BitReverse(size_t nbits, uint16_t bits) {
    constexpr uint16_t kNibbleLookup[16] = {
        0b0000, 0b1000, 0b0100, 0b1100, 0b0010, 0b1010, 0b0110, 0b1110,
        0b0001, 0b1001, 0b0101, 0b1101, 0b0011, 0b1011, 0b0111, 0b1111,
    };
    uint16_t rev16 = (kNibbleLookup[bits & 0xF] << 12) |
                     (kNibbleLookup[(bits >> 4) & 0xF] << 8) |
                     (kNibbleLookup[(bits >> 8) & 0xF] << 4) |
                     (kNibbleLookup[bits >> 12]);
    return rev16 >> (16 - nbits);
  }

  // Create the prefix codes given the code lengths.
  // Supports the code lengths being split into two halves.
  static void ComputeCanonicalCode(const uint8_t* first_chunk_nbits,
                                   uint8_t* first_chunk_bits,
                                   size_t first_chunk_size,
                                   const uint8_t* second_chunk_nbits,
                                   uint16_t* second_chunk_bits,
                                   size_t second_chunk_size) {
    constexpr size_t kMaxCodeLength = 15;
    uint8_t code_length_counts[kMaxCodeLength + 1] = {};
    for (size_t i = 0; i < first_chunk_size; i++) {
      code_length_counts[first_chunk_nbits[i]]++;
      assert(first_chunk_nbits[i] <= kMaxCodeLength);
      assert(first_chunk_nbits[i] <= 8);
      assert(first_chunk_nbits[i] > 0);
    }
    for (size_t i = 0; i < second_chunk_size; i++) {
      code_length_counts[second_chunk_nbits[i]]++;
      assert(second_chunk_nbits[i] <= kMaxCodeLength);
    }

    uint16_t next_code[kMaxCodeLength + 1] = {};

    uint16_t code = 0;
    for (size_t i = 1; i < kMaxCodeLength + 1; i++) {
      code = (code + code_length_counts[i - 1]) << 1;
      next_code[i] = code;
    }

    for (size_t i = 0; i < first_chunk_size; i++) {
      first_chunk_bits[i] =
          BitReverse(first_chunk_nbits[i], next_code[first_chunk_nbits[i]]++);
    }
    for (size_t i = 0; i < second_chunk_size; i++) {
      second_chunk_bits[i] =
          BitReverse(second_chunk_nbits[i], next_code[second_chunk_nbits[i]]++);
    }
  }

  // Computes nbits[i] for i <= n, subject to min_limit[i] <= nbits[i] <=
  // max_limit[i], so to minimize sum(nbits[i] * freqs[i]).
  static void ComputeCodeLengthsNonZero(const uint64_t* freqs, size_t n,
                                        uint8_t* min_limit, uint8_t* max_limit,
                                        uint8_t* nbits) {
    size_t precision = 0;
    uint64_t freqsum = 0;
    for (size_t i = 0; i < n; i++) {
      assert(freqs[i] != 0);
      freqsum += freqs[i];
      if (min_limit[i] < 1) min_limit[i] = 1;
      assert(min_limit[i] <= max_limit[i]);
      precision = std::max<size_t>(max_limit[i], precision);
    }
    uint64_t infty = freqsum * precision;
    std::vector<uint64_t> dynp(((1U << precision) + 1) * (n + 1), infty);
    auto d = [&](size_t sym, size_t off) -> uint64_t& {
      return dynp[sym * ((1 << precision) + 1) + off];
    };
    d(0, 0) = 0;
    for (size_t sym = 0; sym < n; sym++) {
      for (size_t bits = min_limit[sym]; bits <= max_limit[sym]; bits++) {
        size_t off_delta = 1U << (precision - bits);
        for (size_t off = 0; off + off_delta <= (1U << precision); off++) {
          d(sym + 1, off + off_delta) = std::min(
              d(sym, off) + freqs[sym] * bits, d(sym + 1, off + off_delta));
        }
      }
    }

    size_t sym = n;
    size_t off = 1U << precision;

    assert(d(sym, off) != infty);

    while (sym-- > 0) {
      assert(off > 0);
      for (size_t bits = min_limit[sym]; bits <= max_limit[sym]; bits++) {
        size_t off_delta = 1U << (precision - bits);
        if (off_delta <= off &&
            d(sym + 1, off) == d(sym, off - off_delta) + freqs[sym] * bits) {
          off -= off_delta;
          nbits[sym] = bits;
          break;
        }
      }
    }
  }
  static void ComputeCodeLengths(const uint64_t* freqs, size_t n,
                                 const uint8_t* min_limit_in,
                                 const uint8_t* max_limit_in, uint8_t* nbits) {
    assert(n <= kNumRawSymbols + 1);
    uint64_t compact_freqs[kNumRawSymbols + 1];
    uint8_t min_limit[kNumRawSymbols + 1];
    uint8_t max_limit[kNumRawSymbols + 1];
    size_t ni = 0;
    for (size_t i = 0; i < n; i++) {
      if (freqs[i]) {
        compact_freqs[ni] = freqs[i];
        min_limit[ni] = min_limit_in[i];
        max_limit[ni] = max_limit_in[i];
        ni++;
      }
    }
    uint8_t num_bits[kNumRawSymbols + 1] = {};
    ComputeCodeLengthsNonZero(compact_freqs, ni, min_limit, max_limit,
                              num_bits);
    ni = 0;
    for (size_t i = 0; i < n; i++) {
      nbits[i] = 0;
      if (freqs[i]) {
        nbits[i] = num_bits[ni++];
      }
    }
  }

  template <typename BitDepth>
  PrefixCode(BitDepth, uint64_t* raw_counts, uint64_t* lz77_counts) {
    // "merge" together all the lz77 counts in a single symbol for the level 1
    // table (containing just the raw symbols, up to length 7).
    uint64_t level1_counts[kNumRawSymbols + 1];
    memcpy(level1_counts, raw_counts, kNumRawSymbols * sizeof(uint64_t));
    size_t numraw = kNumRawSymbols;
    while (numraw > 0 && level1_counts[numraw - 1] == 0) numraw--;

    level1_counts[numraw] = 0;
    for (size_t i = 0; i < kNumLZ77; i++) {
      level1_counts[numraw] += lz77_counts[i];
    }
    uint8_t level1_nbits[kNumRawSymbols + 1] = {};
    ComputeCodeLengths(level1_counts, numraw + 1, BitDepth::kMinRawLength,
                       BitDepth::kMaxRawLength, level1_nbits);

    uint8_t level2_nbits[kNumLZ77] = {};
    uint8_t min_lengths[kNumLZ77] = {};
    uint8_t l = 15 - level1_nbits[numraw];
    uint8_t max_lengths[kNumLZ77] = {
        l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l,
    };
    size_t num_lz77 = kNumLZ77;
    while (num_lz77 > 0 && lz77_counts[num_lz77 - 1] == 0) num_lz77--;
    ComputeCodeLengths(lz77_counts, num_lz77, min_lengths, max_lengths,
                       level2_nbits);
    for (size_t i = 0; i < numraw; i++) {
      raw_nbits[i] = level1_nbits[i];
    }
    for (size_t i = 0; i < num_lz77; i++) {
      lz77_nbits[i] =
          level2_nbits[i] ? level1_nbits[numraw] + level2_nbits[i] : 0;
    }

    ComputeCanonicalCode(raw_nbits, raw_bits, numraw, lz77_nbits, lz77_bits,
                         kNumLZ77);
    BitDepth::PrepareForSimd(raw_nbits, raw_bits, numraw, raw_nbits_simd,
                             raw_bits_simd);
  }

  void WriteTo(BitWriter* writer) const {
    uint64_t code_length_counts[18] = {};
    code_length_counts[17] = 3 + 2 * (kNumLZ77 - 1);
    for (size_t i = 0; i < kNumRawSymbols; i++) {
      code_length_counts[raw_nbits[i]]++;
    }
    for (size_t i = 0; i < kNumLZ77; i++) {
      code_length_counts[lz77_nbits[i]]++;
    }
    uint8_t code_length_nbits[18] = {};
    uint8_t code_length_nbits_min[18] = {};
    uint8_t code_length_nbits_max[18] = {
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    };
    ComputeCodeLengths(code_length_counts, 18, code_length_nbits_min,
                       code_length_nbits_max, code_length_nbits);
    writer->Write(2, 0b00);  // HSKIP = 0, i.e. don't skip code lengths.

    // As per Brotli RFC.
    uint8_t code_length_order[18] = {1, 2, 3, 4,  0,  5,  17, 6,  16,
                                     7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8_t code_length_length_nbits[] = {2, 4, 3, 2, 2, 4};
    uint8_t code_length_length_bits[] = {0, 7, 3, 2, 1, 15};

    // Encode lengths of code lengths.
    size_t num_code_lengths = 18;
    while (code_length_nbits[code_length_order[num_code_lengths - 1]] == 0) {
      num_code_lengths--;
    }
    for (size_t i = 0; i < num_code_lengths; i++) {
      int symbol = code_length_nbits[code_length_order[i]];
      writer->Write(code_length_length_nbits[symbol],
                    code_length_length_bits[symbol]);
    }

    // Compute the canonical codes for the codes that represent the lengths of
    // the actual codes for data.
    uint16_t code_length_bits[18] = {};
    ComputeCanonicalCode(nullptr, nullptr, 0, code_length_nbits,
                         code_length_bits, 18);
    // Encode raw bit code lengths.
    for (size_t i = 0; i < kNumRawSymbols; i++) {
      writer->Write(code_length_nbits[raw_nbits[i]],
                    code_length_bits[raw_nbits[i]]);
    }
    size_t num_lz77 = kNumLZ77;
    while (lz77_nbits[num_lz77 - 1] == 0) {
      num_lz77--;
    }
    // Encode 0s until 224 (start of LZ77 symbols). This is in total 224-19 =
    // 205.
    static_assert(kLZ77Offset == 224, "");
    static_assert(kNumRawSymbols == 19, "");
    writer->Write(code_length_nbits[17], code_length_bits[17]);
    writer->Write(3, 0b010);  // 5
    writer->Write(code_length_nbits[17], code_length_bits[17]);
    writer->Write(3, 0b000);  // (5-2)*8 + 3 = 27
    writer->Write(code_length_nbits[17], code_length_bits[17]);
    writer->Write(3, 0b010);  // (27-2)*8 + 5 = 205
    // Encode LZ77 symbols, with values 224+i*16.
    for (size_t i = 0; i < num_lz77; i++) {
      writer->Write(code_length_nbits[lz77_nbits[i]],
                    code_length_bits[lz77_nbits[i]]);
      if (i != num_lz77 - 1) {
        // Encode gap between LZ77 symbols: 15 zeros.
        writer->Write(code_length_nbits[17], code_length_bits[17]);
        writer->Write(3, 0b000);  // 3
        writer->Write(code_length_nbits[17], code_length_bits[17]);
        writer->Write(3, 0b100);  // (3-2)*8+7 = 15
      }
    }
  }
};

#ifdef FASTLL_ENABLE_AVX2_INTRINSICS
#include <immintrin.h>
void EncodeChunkAVX2(const uint16_t* residuals, const PrefixCode& code,
                     BitWriter& output) {
  auto value = _mm256_load_si256((__m256i*)residuals);

  // we know that residuals[i] has at most 12 bits, so we just need 3 nibbles
  // and don't need to mask the third. However we do need to set the high
  // byte to 0xFF, which will make table lookups return 0.
  auto lo_nibble =
      _mm256_or_si256(_mm256_and_si256(value, _mm256_set1_epi16(0xF)),
                      _mm256_set1_epi16(0xFF00));
  auto mi_nibble = _mm256_or_si256(
      _mm256_and_si256(_mm256_srli_epi16(value, 4), _mm256_set1_epi16(0xF)),
      _mm256_set1_epi16(0xFF00));
  auto hi_nibble =
      _mm256_or_si256(_mm256_srli_epi16(value, 8), _mm256_set1_epi16(0xFF00));

  auto lo_lut = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4));
  auto mi_lut = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8));
  auto hi_lut = _mm256_broadcastsi128_si256(_mm_setr_epi8(
      0, 9, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12));

  auto lo_token = _mm256_shuffle_epi8(lo_lut, lo_nibble);
  auto mi_token = _mm256_shuffle_epi8(mi_lut, mi_nibble);
  auto hi_token = _mm256_shuffle_epi8(hi_lut, hi_nibble);

  auto token = _mm256_max_epi16(lo_token, _mm256_max_epi16(mi_token, hi_token));
  auto nbits = _mm256_subs_epu16(token, _mm256_set1_epi16(1));

  // Compute 1<<nbits.
  auto pow2_lo_lut = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6,
                    1u << 7, 0, 0, 0, 0, 0, 0, 0, 0));
  auto pow2_hi_lut = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1 << 0, 1 << 1, 1 << 2, 1 << 3,
                    1 << 4, 1 << 5, 1 << 6, 1u << 7));

  auto nbits_masked = _mm256_or_si256(nbits, _mm256_set1_epi16(0xFF00));

  auto nbits_pow2_lo = _mm256_shuffle_epi8(pow2_lo_lut, nbits_masked);
  auto nbits_pow2_hi = _mm256_shuffle_epi8(pow2_hi_lut, nbits_masked);

  auto nbits_pow2 =
      _mm256_or_si256(_mm256_slli_epi16(nbits_pow2_hi, 8), nbits_pow2_lo);

  auto bits = _mm256_subs_epu16(value, nbits_pow2);

  auto token_masked = _mm256_or_si256(token, _mm256_set1_epi16(0xFF00));

  // huff_nbits <= 6.
  auto huff_nbits =
      _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(
                              _mm_load_si128((__m128i*)code.raw_nbits_simd)),
                          token_masked);

  auto huff_bits = _mm256_shuffle_epi8(
      _mm256_broadcastsi128_si256(_mm_load_si128((__m128i*)code.raw_bits_simd)),
      token_masked);

  auto huff_nbits_masked =
      _mm256_or_si256(huff_nbits, _mm256_set1_epi16(0xFF00));

  auto bits_shifted = _mm256_mullo_epi16(
      bits, _mm256_shuffle_epi8(pow2_lo_lut, huff_nbits_masked));

  nbits = _mm256_add_epi16(nbits, huff_nbits);
  bits = _mm256_or_si256(bits_shifted, huff_bits);

  // Merge nbits and bits from 16-bit to 32-bit lanes.
  auto nbits_hi16 = _mm256_srli_epi32(nbits, 16);
  auto nbits_lo16 = _mm256_and_si256(nbits, _mm256_set1_epi32(0xFFFF));
  auto bits_hi16 = _mm256_srli_epi32(bits, 16);
  auto bits_lo16 = _mm256_and_si256(bits, _mm256_set1_epi32(0xFFFF));

  nbits = _mm256_add_epi32(nbits_hi16, nbits_lo16);
  bits = _mm256_or_si256(_mm256_sllv_epi32(bits_hi16, nbits_lo16), bits_lo16);

  // Merge 32 -> 64 bit lanes.
  auto nbits_hi32 = _mm256_srli_epi64(nbits, 32);
  auto nbits_lo32 = _mm256_and_si256(nbits, _mm256_set1_epi64x(0xFFFFFFFF));
  auto bits_hi32 = _mm256_srli_epi64(bits, 32);
  auto bits_lo32 = _mm256_and_si256(bits, _mm256_set1_epi64x(0xFFFFFFFF));

  nbits = _mm256_add_epi64(nbits_hi32, nbits_lo32);
  bits = _mm256_or_si256(_mm256_sllv_epi64(bits_hi32, nbits_lo32), bits_lo32);

  alignas(32) uint64_t nbits_simd[4] = {};
  alignas(32) uint64_t bits_simd[4] = {};

  _mm256_store_si256((__m256i*)nbits_simd, nbits);
  _mm256_store_si256((__m256i*)bits_simd, bits);

  // Manually merge the buffer bits with the SIMD bits.
  // Necessary because Write() is only guaranteed to work with <=56 bits.
  // Trying to SIMD-fy this code results in slower speed (and definitely less
  // clarity).
  {
    for (size_t i = 0; i < 4; i++) {
      output.buffer |= bits_simd[i] << output.bits_in_buffer;
      memcpy(output.data.get() + output.bytes_written, &output.buffer, 8);
      // If >> 64, next_buffer is unused.
      uint64_t next_buffer = bits_simd[i] >> (64 - output.bits_in_buffer);
      output.bits_in_buffer += nbits_simd[i];
      // This `if` seems to be faster than using ternaries.
      if (output.bits_in_buffer >= 64) {
        output.buffer = next_buffer;
        output.bits_in_buffer -= 64;
        output.bytes_written += 8;
      }
    }
    memcpy(output.data.get() + output.bytes_written, &output.buffer, 8);
    size_t bytes_in_buffer = output.bits_in_buffer / 8;
    output.bits_in_buffer -= bytes_in_buffer * 8;
    output.buffer >>= bytes_in_buffer * 8;
    output.bytes_written += bytes_in_buffer;
  }
}
#endif

#ifdef FASTLL_ENABLE_NEON_INTRINSICS
#include <arm_neon.h>

FJXL_INLINE void TokenizeNeon(const uint16_t* residuals, uint16_t* token_out,
                              uint16_t* nbits_out, uint16_t* bits_out) {
  uint16x8_t res = vld1q_u16(residuals);
  uint16x8_t token = vsubq_u16(vdupq_n_u16(16), vclzq_u16(res));
  uint16x8_t nbits = vqsubq_u16(token, vdupq_n_u16(1));
  uint16x8_t bits =
      vqsubq_u16(res, vshlq_u16(vdupq_n_u16(1), vreinterpretq_s16_u16(nbits)));
  vst1q_u16(token_out, token);
  vst1q_u16(nbits_out, nbits);
  vst1q_u16(bits_out, bits);
}

FJXL_INLINE void TokenizeNeon(const uint32_t* residuals, uint16_t* token_out,
                              uint32_t* nbits_out, uint32_t* bits_out) {
  uint32x4_t res_lo = vld1q_u32(residuals);
  uint32x4_t res_hi = vld1q_u32(residuals + 4);
  uint32x4_t token_lo = vsubq_u32(vdupq_n_u32(32), vclzq_u32(res_lo));
  uint32x4_t token_hi = vsubq_u32(vdupq_n_u32(32), vclzq_u32(res_hi));
  uint32x4_t nbits_lo = vqsubq_u32(token_lo, vdupq_n_u32(1));
  uint32x4_t nbits_hi = vqsubq_u32(token_hi, vdupq_n_u32(1));
  uint32x4_t bits_lo = vqsubq_u32(
      res_lo, vshlq_u32(vdupq_n_u32(1), vreinterpretq_s32_u32(nbits_lo)));
  uint32x4_t bits_hi = vqsubq_u32(
      res_hi, vshlq_u32(vdupq_n_u32(1), vreinterpretq_s32_u32(nbits_hi)));
  uint16x8_t token = vmovn_high_u32(vmovn_u32(token_lo), token_hi);
  vst1q_u16(token_out, token);
  vst1q_u32(nbits_out, nbits_lo);
  vst1q_u32(nbits_out + 4, nbits_hi);
  vst1q_u32(bits_out, bits_lo);
  vst1q_u32(bits_out + 4, bits_hi);
}

FJXL_INLINE void HuffmanNeonUpTo13(const uint16_t* tokens,
                                   const PrefixCode& code, uint16_t* nbits_out,
                                   uint16_t* bits_out) {
  uint8x16_t tok8x16 =
      vreinterpretq_u8_u16(vorrq_u16(vld1q_u16(tokens), vdupq_n_u16(0xFF00)));
  uint16x8_t huff_bits =
      vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(code.raw_bits_simd), tok8x16));
  uint16x8_t huff_nbits =
      vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(code.raw_nbits_simd), tok8x16));
  vst1q_u16(nbits_out, huff_nbits);
  vst1q_u16(bits_out, huff_bits);
}

FJXL_INLINE void HuffmanNeon14(const uint16_t* tokens, const PrefixCode& code,
                               uint16_t* nbits_out, uint16_t* bits_out) {
  uint16x8_t tok_cap = vdupq_n_u16(15);
  uint16x8_t tok = vld1q_u16(tokens);
  uint8x16_t tokindex = vreinterpretq_u8_u16(
      vorrq_u16(vminq_u16(tok, tok_cap), vdupq_n_u16(0xFF00)));
  uint16x8_t huff_bits_pre =
      vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(code.raw_bits_simd), tokindex));
  // Set the highest bit when token == 16; the Huffman code is constructed in
  // such a way that the code for token 15 is the same as the code for 16,
  // except for the highest bit.
  uint16x8_t huff_bits = vorrq_u16(
      vandq_u16(vcgtq_u16(tok, tok_cap), vdupq_n_u16(128)), huff_bits_pre);
  uint16x8_t huff_nbits =
      vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(code.raw_nbits_simd), tokindex));
  vst1q_u16(nbits_out, huff_nbits);
  vst1q_u16(bits_out, huff_bits);
}

FJXL_INLINE void HuffmanNeonAbove14(const uint16_t* tokens,
                                    const PrefixCode& code, uint16_t* nbits_out,
                                    uint16_t* bits_out) {
  uint16x8_t tok = vld1q_u16(tokens);

  uint16x8_t above = vcgtq_u16(tok, vdupq_n_u16(12));
  // 13, 14 -> 13
  // 15, 16 -> 14
  // 17, 18 -> 15
  uint16x8_t remap_tok =
      vbslq_u16(above, vshrq_n_u16(vaddq_u16(tok, vdupq_n_u16(13)), 1), tok);

  uint8x16_t tokindex =
      vreinterpretq_u8_u16(vorrq_u16(remap_tok, vdupq_n_u16(0xFF00)));
  uint16x8_t huff_bits_pre =
      vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(code.raw_bits_simd), tokindex));
  // Set the highest bit when token == 14, 16, 18.
  uint16x8_t needs_high_bit =
      vandq_u16(above, vceqq_u16(tok, vandq_u16(tok, vdupq_n_u16(0xFFFE))));
  uint16x8_t huff_bits =
      vorrq_u16(vandq_u16(needs_high_bit, vdupq_n_u16(128)), huff_bits_pre);
  uint16x8_t huff_nbits =
      vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(code.raw_nbits_simd), tokindex));
  vst1q_u16(nbits_out, huff_nbits);
  vst1q_u16(bits_out, huff_bits);
}

FJXL_INLINE uint32x4_t Merge16To32Neon(uint16x8_t nbits, uint16x8_t bits,
                                       uint32x4_t* nbits32) {
  uint32x4_t nbits_lo16 =
      vandq_u32(vreinterpretq_u32_u16(nbits), vdupq_n_u32(0xFFFF));
  uint32x4_t bits_hi16 = vshlq_u32(vshrq_n_u32(vreinterpretq_u32_u16(bits), 16),
                                   vreinterpretq_s32_u32(nbits_lo16));
  uint32x4_t bits_lo16 =
      vandq_u32(vreinterpretq_u32_u16(bits), vdupq_n_u32(0xFFFF));

  *nbits32 = vsraq_n_u32(nbits_lo16, vreinterpretq_u32_u16(nbits), 16);
  return vorrq_u32(bits_hi16, bits_lo16);
}

FJXL_INLINE void StoreNeonUpTo8(const uint16_t* nbits_tok,
                                const uint16_t* bits_tok,
                                const uint16_t* nbits_huff,
                                const uint16_t* bits_huff, BitWriter& out) {
  uint16x8_t bits = vld1q_u16(bits_tok);
  uint16x8_t nbits = vld1q_u16(nbits_tok);
  uint16x8_t huff_bits = vld1q_u16(bits_huff);
  uint16x8_t huff_nbits = vld1q_u16(nbits_huff);
  bits =
      vorrq_u16(vshlq_u16(bits, vreinterpretq_s16_u16(huff_nbits)), huff_bits);
  nbits = vaddq_u16(nbits, huff_nbits);

  uint32x4_t nbits32;
  auto bits32 = Merge16To32Neon(nbits, bits, &nbits32);

  // Merging up to 64 bits is not faster.
  for (size_t i = 0; i < 4; i++) {
    out.Write(nbits32[i], bits32[i]);
  }
}

// Huffman and raw bits don't necessarily fit in a single u16 here.
FJXL_INLINE void StoreNeonUpTo14(const uint16_t* nbits_tok,
                                 const uint16_t* bits_tok,
                                 const uint16_t* nbits_huff,
                                 const uint16_t* bits_huff, BitWriter& out) {
  uint16x8_t bits = vld1q_u16(bits_tok);
  uint16x8_t nbits = vld1q_u16(nbits_tok);
  uint16x8_t huff_bits = vld1q_u16(bits_huff);
  uint16x8_t huff_nbits = vld1q_u16(nbits_huff);

  uint16x8_t lbits = vzip1q_u16(huff_bits, bits);
  uint16x8_t hbits = vzip2q_u16(huff_bits, bits);
  uint16x8_t lnbits = vzip1q_u16(huff_nbits, nbits);
  uint16x8_t hnbits = vzip2q_u16(huff_nbits, nbits);

  // Merging up to 64 bits is not faster.
  uint32x4_t nbits32;
  auto bits32 = Merge16To32Neon(lnbits, lbits, &nbits32);
  for (size_t i = 0; i < 4; i++) {
    out.Write(nbits32[i], bits32[i]);
  }
  bits32 = Merge16To32Neon(hnbits, hbits, &nbits32);
  for (size_t i = 0; i < 4; i++) {
    out.Write(nbits32[i], bits32[i]);
  }
}

FJXL_INLINE void StoreNeonAbove14(const uint32_t* nbits_tok,
                                  const uint32_t* bits_tok,
                                  const uint16_t* nbits_huff,
                                  const uint16_t* bits_huff, BitWriter& out) {
  uint32x4_t bits_lo = vld1q_u32(bits_tok);
  uint32x4_t nbits_lo = vld1q_u32(nbits_tok);
  uint32x4_t bits_hi = vld1q_u32(bits_tok + 4);
  uint32x4_t nbits_hi = vld1q_u32(nbits_tok + 4);
  uint16x8_t huff_bits = vld1q_u16(bits_huff);
  uint16x8_t huff_nbits = vld1q_u16(nbits_huff);
  uint32x4_t huff_nbits_lo = vmovl_u16(vget_low_u16(huff_nbits));
  uint32x4_t huff_nbits_hi = vmovl_high_u16(huff_nbits);
  uint32x4_t huff_bits_lo = vmovl_u16(vget_low_u16(huff_bits));
  uint32x4_t huff_bits_hi = vmovl_high_u16(huff_bits);
  bits_lo = vorrq_u32(vshlq_u32(bits_lo, vreinterpretq_s32_u32(huff_nbits_lo)),
                      huff_bits_lo);
  nbits_lo = vaddq_u32(nbits_lo, huff_nbits_lo);

  // Merging up to 64 bits is not faster.
  for (size_t i = 0; i < 4; i++) {
    out.Write(nbits_lo[i], bits_lo[i]);
  }
  bits_hi = vorrq_u32(vshlq_u32(bits_hi, vreinterpretq_s32_u32(huff_nbits_hi)),
                      huff_bits_hi);
  nbits_hi = vaddq_u32(nbits_hi, huff_nbits_hi);
  for (size_t i = 0; i < 4; i++) {
    out.Write(nbits_hi[i], bits_hi[i]);
  }
}
#endif

void EncodeHybridUint000(uint32_t value, uint32_t* token, uint32_t* nbits,
                         uint32_t* bits) {
  uint32_t n = 31 - __builtin_clz(value);
  *token = value ? n + 1 : 0;
  *nbits = value ? n : 0;
  *bits = value ? value - (1 << n) : 0;
}

// NOTE: the encoding of lz77 lengths relies on the chunk size being 16.
constexpr size_t kChunkSize = 16;

template <typename Residual>
void GenericEncodeChunk(const Residual* residuals, const PrefixCode& code,
                        BitWriter& output) {
  for (size_t ix = 0; ix < kChunkSize; ix++) {
    unsigned token, nbits, bits;
    EncodeHybridUint000(residuals[ix], &token, &nbits, &bits);
    output.Write(code.raw_nbits[token] + nbits,
                 code.raw_bits[token] | bits << code.raw_nbits[token]);
  }
}

struct UpTo8Bits {
  size_t bitdepth;
  explicit UpTo8Bits(size_t bitdepth) : bitdepth(bitdepth) {
    assert(bitdepth <= 8);
  }
  // Here we can fit up to 9 extra bits + 7 Huffman bits in a u16.
  // Last symbol is used for LZ77 lengths and has no limitations except allowing
  // to represent 32 symbols in total.
  static constexpr uint8_t kMinRawLength[12] = {};
  static constexpr uint8_t kMaxRawLength[12] = {
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 10,
  };
  static size_t MaxEncodedBitsPerSample() { return 16; }
  static constexpr size_t kInputBytes = 1;
  using pixel_t = int16_t;
  using upixel_t = uint16_t;

  static void PrepareForSimd(const uint8_t* nbits, const uint8_t* bits,
                             size_t n, uint8_t* nbits_simd,
                             uint8_t* bits_simd) {
    assert(n <= 16);
    memcpy(nbits_simd, nbits, 16);
    memcpy(bits_simd, bits, 16);
  }

  static void EncodeChunk(upixel_t* residuals, const PrefixCode& code,
                          BitWriter& output) {
#if defined(FASTLL_ENABLE_AVX2_INTRINSICS) && FASTLL_ENABLE_AVX2_INTRINSICS
    EncodeChunkAVX2(residuals, code, output);
    return;
#elif defined(FASTLL_ENABLE_NEON_INTRINSICS) && FASTLL_ENABLE_NEON_INTRINSICS
    for (int i : {0, 8}) {
      uint16_t bits[8];
      uint16_t nbits[8];
      uint16_t bits_huff[8];
      uint16_t nbits_huff[8];
      uint16_t token[8];
      TokenizeNeon(residuals + i, token, nbits, bits);
      HuffmanNeonUpTo13(token, code, nbits_huff, bits_huff);
      StoreNeonUpTo8(nbits, bits, nbits_huff, bits_huff, output);
    }
    return;
#endif
    GenericEncodeChunk(residuals, code, output);
  }

  size_t NumSymbols(bool doing_ycocg) const {
    // values gain 1 bit for YCoCg, 1 bit for prediction.
    // Maximum symbol is 1 + effective bit depth of residuals.
    if (doing_ycocg) {
      return bitdepth + 3;
    } else {
      return bitdepth + 2;
    }
  }
};
constexpr uint8_t UpTo8Bits::kMinRawLength[];
constexpr uint8_t UpTo8Bits::kMaxRawLength[];

struct From9To13Bits {
  size_t bitdepth;
  explicit From9To13Bits(size_t bitdepth) : bitdepth(bitdepth) {
    assert(bitdepth <= 13 && bitdepth >= 9);
  }
  // Last symbol is used for LZ77 lengths and has no limitations except allowing
  // to represent 32 symbols in total.
  // We cannot fit all the bits in a u16, so do not even try and use up to 8
  // bits per raw symbol.
  // There are at most 16 raw symbols, so Huffman coding can be SIMDfied without
  // any special tricks.
  static constexpr uint8_t kMinRawLength[17] = {};
  static constexpr uint8_t kMaxRawLength[17] = {
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10,
  };
  static size_t MaxEncodedBitsPerSample() { return 21; }
  static constexpr size_t kInputBytes = 2;
  using pixel_t = int16_t;
  using upixel_t = uint16_t;

  static void PrepareForSimd(const uint8_t* nbits, const uint8_t* bits,
                             size_t n, uint8_t* nbits_simd,
                             uint8_t* bits_simd) {
    assert(n <= 16);
    memcpy(nbits_simd, nbits, 16);
    memcpy(bits_simd, bits, 16);
  }

  static void EncodeChunk(upixel_t* residuals, const PrefixCode& code,
                          BitWriter& output) {
#if defined(FASTLL_ENABLE_NEON_INTRINSICS) && FASTLL_ENABLE_NEON_INTRINSICS
    for (int i : {0, 8}) {
      uint16_t bits[8];
      uint16_t nbits[8];
      uint16_t bits_huff[8];
      uint16_t nbits_huff[8];
      uint16_t token[8];
      TokenizeNeon(residuals + i, token, nbits, bits);
      HuffmanNeonUpTo13(token, code, nbits_huff, bits_huff);
      StoreNeonUpTo14(nbits, bits, nbits_huff, bits_huff, output);
    }
    return;
#endif
    GenericEncodeChunk(residuals, code, output);
  }

  size_t NumSymbols(bool doing_ycocg) const {
    // values gain 1 bit for YCoCg, 1 bit for prediction.
    // Maximum symbol is 1 + effective bit depth of residuals.
    if (doing_ycocg) {
      return bitdepth + 3;
    } else {
      return bitdepth + 2;
    }
  }
};
constexpr uint8_t From9To13Bits::kMinRawLength[];
constexpr uint8_t From9To13Bits::kMaxRawLength[];

void CheckHuffmanBitsSIMD(int bits1, int nbits1, int bits2, int nbits2) {
  assert(nbits1 == 8);
  assert(nbits2 == 8);
  assert(bits2 == (bits1 | 128));
}

struct Exactly14Bits {
  explicit Exactly14Bits(size_t bitdepth) { assert(bitdepth == 14); }
  // Force LZ77 symbols to have at least 8 bits, and raw symbols 15 and 16 to
  // have exactly 8, and no other symbol to have 8 or more. This ensures that
  // the representation for 15 and 16 is identical up to one bit.
  static constexpr uint8_t kMinRawLength[18] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 7,
  };
  static constexpr uint8_t kMaxRawLength[18] = {
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 10,
  };
  static constexpr size_t bitdepth = 14;
  static size_t MaxEncodedBitsPerSample() { return 22; }
  static constexpr size_t kInputBytes = 2;
  using pixel_t = int16_t;
  using upixel_t = uint16_t;

  static void PrepareForSimd(const uint8_t* nbits, const uint8_t* bits,
                             size_t n, uint8_t* nbits_simd,
                             uint8_t* bits_simd) {
    assert(n == 17);
    CheckHuffmanBitsSIMD(bits[15], nbits[15], bits[16], nbits[16]);
    memcpy(nbits_simd, nbits, 16);
    memcpy(bits_simd, bits, 16);
  }

  static void EncodeChunk(upixel_t* residuals, const PrefixCode& code,
                          BitWriter& output) {
#if defined(FASTLL_ENABLE_NEON_INTRINSICS) && FASTLL_ENABLE_NEON_INTRINSICS
    for (int i : {0, 8}) {
      uint16_t bits[8];
      uint16_t nbits[8];
      uint16_t bits_huff[8];
      uint16_t nbits_huff[8];
      uint16_t token[8];
      TokenizeNeon(residuals + i, token, nbits, bits);
      HuffmanNeon14(token, code, nbits_huff, bits_huff);
      StoreNeonUpTo14(nbits, bits, nbits_huff, bits_huff, output);
    }
    return;
#endif
    GenericEncodeChunk(residuals, code, output);
  }

  size_t NumSymbols(bool) const { return 17; }
};
constexpr uint8_t Exactly14Bits::kMinRawLength[];
constexpr uint8_t Exactly14Bits::kMaxRawLength[];

struct MoreThan14Bits {
  size_t bitdepth;
  explicit MoreThan14Bits(size_t bitdepth) : bitdepth(bitdepth) {
    assert(bitdepth > 14);
    assert(bitdepth <= 16);
  }
  // Force LZ77 symbols to have at least 8 bits, and raw symbols 13 to 18 to
  // have exactly 8, and no other symbol to have 8 or more. This ensures that
  // the representation for (13, 14), (15, 16), (17, 18) is identical up to one
  // bit.
  static constexpr uint8_t kMinRawLength[20] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 7,
  };
  static constexpr uint8_t kMaxRawLength[20] = {
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 10,
  };
  static size_t MaxEncodedBitsPerSample() { return 24; }
  static constexpr size_t kInputBytes = 2;
  using pixel_t = int32_t;
  using upixel_t = uint32_t;

  static void PrepareForSimd(const uint8_t* nbits, const uint8_t* bits,
                             size_t n, uint8_t* nbits_simd,
                             uint8_t* bits_simd) {
    assert(n == 19);
    CheckHuffmanBitsSIMD(bits[13], nbits[13], bits[14], nbits[14]);
    CheckHuffmanBitsSIMD(bits[15], nbits[15], bits[16], nbits[16]);
    CheckHuffmanBitsSIMD(bits[17], nbits[17], bits[18], nbits[18]);
    for (size_t i = 0; i < 14; i++) {
      nbits_simd[i] = nbits[i];
      bits_simd[i] = bits[i];
    }
    nbits_simd[14] = nbits[15];
    bits_simd[14] = bits[15];
    nbits_simd[15] = nbits[17];
    bits_simd[15] = bits[17];
  }

  static void EncodeChunk(upixel_t* residuals, const PrefixCode& code,
                          BitWriter& output) {
#if defined(FASTLL_ENABLE_NEON_INTRINSICS) && FASTLL_ENABLE_NEON_INTRINSICS
    for (int i : {0, 8}) {
      uint32_t bits[8];
      uint32_t nbits[8];
      uint16_t bits_huff[8];
      uint16_t nbits_huff[8];
      uint16_t token[8];
      TokenizeNeon(residuals + i, token, nbits, bits);
      HuffmanNeonAbove14(token, code, nbits_huff, bits_huff);
      StoreNeonAbove14(nbits, bits, nbits_huff, bits_huff, output);
    }
    return;
#endif
    GenericEncodeChunk(residuals, code, output);
  }
  size_t NumSymbols(bool) const { return 19; }
};
constexpr uint8_t MoreThan14Bits::kMinRawLength[];
constexpr uint8_t MoreThan14Bits::kMaxRawLength[];

void AppendWriter(BitWriter* dest, const BitWriter* src) {
  if (dest->bits_in_buffer == 0) {
    memcpy(dest->data.get() + dest->bytes_written, src->data.get(),
           src->bytes_written);
    dest->bytes_written += src->bytes_written;
  } else {
    size_t i = 0;
    uint64_t buf = dest->buffer;
    uint64_t bits_in_buffer = dest->bits_in_buffer;
    uint8_t* dest_buf = dest->data.get() + dest->bytes_written;
    // Copy 8 bytes at a time until we reach the border.
    for (; i + 8 < src->bytes_written; i += 8) {
      uint64_t chunk;
      memcpy(&chunk, src->data.get() + i, 8);
      uint64_t out = buf | (chunk << bits_in_buffer);
      memcpy(dest_buf + i, &out, 8);
      buf = chunk >> (64 - bits_in_buffer);
    }
    dest->buffer = buf;
    dest->bytes_written += i;
    for (; i < src->bytes_written; i++) {
      dest->Write(8, src->data[i]);
    }
  }
  dest->Write(src->bits_in_buffer, src->buffer);
}

void AssembleFrame(size_t width, size_t height, size_t nb_chans,
                   size_t bitdepth, size_t chunk_size,
                   const std::vector<std::array<BitWriter, 4>>& group_data,
                   BitWriter* output) {
  size_t total_size_groups = 0;
  std::vector<size_t> group_sizes(group_data.size());
  for (size_t i = 0; i < group_data.size(); i++) {
    size_t sz = 0;
    for (size_t j = 0; j < nb_chans; j++) {
      const auto& writer = group_data[i][j];
      sz += writer.bytes_written * 8 + writer.bits_in_buffer;
    }
    sz = (sz + 7) / 8;
    group_sizes[i] = sz;
    total_size_groups += sz * 8;
  }
  output->Allocate(1000 + group_data.size() * 32 + total_size_groups);

  // Signature
  output->Write(16, 0x0AFF);

  // Size header, hand-crafted.
  // Not small
  output->Write(1, 0);

  auto wsz = [output](size_t size) {
    if (size - 1 < (1 << 9)) {
      output->Write(2, 0b00);
      output->Write(9, size - 1);
    } else if (size - 1 < (1 << 13)) {
      output->Write(2, 0b01);
      output->Write(13, size - 1);
    } else if (size - 1 < (1 << 18)) {
      output->Write(2, 0b10);
      output->Write(18, size - 1);
    } else {
      output->Write(2, 0b11);
      output->Write(30, size - 1);
    }
  };

  wsz(height);

  // No special ratio.
  output->Write(3, 0);

  wsz(width);

  // Hand-crafted ImageMetadata.
  output->Write(1, 0);  // all_default
  output->Write(1, 0);  // extra_fields
  output->Write(1, 0);  // bit_depth.floating_point_sample
  if (bitdepth == 8) {
    output->Write(2, 0b00);  // bit_depth.bits_per_sample = 8
  } else if (bitdepth == 10) {
    output->Write(2, 0b01);  // bit_depth.bits_per_sample = 10
  } else if (bitdepth == 12) {
    output->Write(2, 0b10);  // bit_depth.bits_per_sample = 12
  } else {
    output->Write(2, 0b11);  // 1 + u(6)
    output->Write(6, bitdepth - 1);
  }
  if (bitdepth <= 14) {
    output->Write(1, 1);  // 16-bit-buffer sufficient
  } else {
    output->Write(1, 0);  // 16-bit-buffer NOT sufficient
  }
  bool have_alpha = (nb_chans == 2 || nb_chans == 4);
  if (have_alpha) {
    output->Write(2, 0b01);  // One extra channel
    output->Write(1, 1);     // ... all_default (ie. 8-bit alpha)
  } else {
    output->Write(2, 0b00);  // No extra channel
  }
  output->Write(1, 0);  // Not XYB
  if (nb_chans > 1) {
    output->Write(1, 1);  // color_encoding.all_default (sRGB)
  } else {
    output->Write(1, 0);     // color_encoding.all_default false
    output->Write(1, 0);     // color_encoding.want_icc false
    output->Write(2, 1);     // grayscale
    output->Write(2, 1);     // D65
    output->Write(1, 0);     // no gamma transfer function
    output->Write(2, 0b10);  // tf: 2 + u(4)
    output->Write(4, 11);    // tf of sRGB
    output->Write(2, 1);     // relative rendering intent
  }
  output->Write(2, 0b00);  // No extensions.

  output->Write(1, 1);  // all_default transform data

  // No ICC, no preview. Frame should start at byte boundery.
  output->ZeroPadToByte();

  auto wsz_fh = [output](size_t size) {
    if (size < (1 << 8)) {
      output->Write(2, 0b00);
      output->Write(8, size);
    } else if (size - 256 < (1 << 11)) {
      output->Write(2, 0b01);
      output->Write(11, size - 256);
    } else if (size - 2304 < (1 << 14)) {
      output->Write(2, 0b10);
      output->Write(14, size - 2304);
    } else {
      output->Write(2, 0b11);
      output->Write(30, size - 18688);
    }
  };

  // Handcrafted frame header.
  output->Write(1, 0);     // all_default
  output->Write(2, 0b00);  // regular frame
  output->Write(1, 1);     // modular
  output->Write(2, 0b00);  // default flags
  output->Write(1, 0);     // not YCbCr
  output->Write(2, 0b00);  // no upsampling
  if (have_alpha) {
    output->Write(2, 0b00);  // no alpha upsampling
  }
  output->Write(2, 0b01);  // default group size
  output->Write(2, 0b00);  // exactly one pass
  if (width % chunk_size == 0) {
    output->Write(1, 0);  // no custom size or origin
  } else {
    output->Write(1, 1);  // custom size
    wsz_fh(0);            // x0 = 0
    wsz_fh(0);            // y0 = 0
    wsz_fh((width + chunk_size - 1) / chunk_size *
           chunk_size);  // xsize rounded up to chunk size
    wsz_fh(height);      // ysize same
  }
  output->Write(2, 0b00);  // kReplace blending mode
  if (have_alpha) {
    output->Write(2, 0b00);  // kReplace blending mode for alpha channel
  }
  output->Write(1, 1);     // is_last
  output->Write(2, 0b00);  // a frame has no name
  output->Write(1, 0);     // loop filter is not all_default
  output->Write(1, 0);     // no gaborish
  output->Write(2, 0);     // 0 EPF iters
  output->Write(2, 0b00);  // No LF extensions
  output->Write(2, 0b00);  // No FH extensions

  output->Write(1, 0);      // No TOC permutation
  output->ZeroPadToByte();  // TOC is byte-aligned.
  for (size_t i = 0; i < group_data.size(); i++) {
    size_t sz = group_sizes[i];
    if (sz < (1 << 10)) {
      output->Write(2, 0b00);
      output->Write(10, sz);
    } else if (sz - 1024 < (1 << 14)) {
      output->Write(2, 0b01);
      output->Write(14, sz - 1024);
    } else if (sz - 17408 < (1 << 22)) {
      output->Write(2, 0b10);
      output->Write(22, sz - 17408);
    } else {
      output->Write(2, 0b11);
      output->Write(30, sz - 4211712);
    }
  }
  output->ZeroPadToByte();  // Groups are byte-aligned.

  for (size_t i = 0; i < group_data.size(); i++) {
    for (size_t j = 0; j < nb_chans; j++) {
      AppendWriter(output, &group_data[i][j]);
    }
    output->ZeroPadToByte();
  }
}

void PrepareDCGlobalCommon(bool is_single_group, size_t width, size_t height,
                           const PrefixCode& code, BitWriter* output) {
  output->Allocate(100000 + (is_single_group ? width * height * 16 : 0));
  // No patches, spline or noise.
  output->Write(1, 1);  // default DC dequantization factors (?)
  output->Write(1, 1);  // use global tree / histograms
  output->Write(1, 0);  // no lz77 for the tree

  output->Write(1, 1);   // simple code for the tree's context map
  output->Write(2, 0);   // all contexts clustered together
  output->Write(1, 1);   // use prefix code for tree
  output->Write(4, 15);  // don't do hybriduint for tree - 2 symbols anyway
  output->Write(7, 0b0100101);  // Alphabet size is 6: we need 0 and 5 (var16)
  output->Write(2, 1);          // simple prefix code
  output->Write(2, 1);          // with two symbols
  output->Write(3, 0);          // 0
  output->Write(3, 5);          // 5
  output->Write(5, 0b00010);    // tree repr: predictor is 5, all else 0

  output->Write(1, 1);     // Enable lz77 for the main bitstream
  output->Write(2, 0b00);  // lz77 offset 224
  static_assert(kLZ77Offset == 224, "");
  output->Write(10, 0b0000011111);  // lz77 min length 16
  static_assert(kLZ77MinLength == 16, "");
  output->Write(4, 4);  // 404 hybrid uint config for lz77: 4
  output->Write(3, 0);  // 0
  output->Write(3, 4);  // 4
  output->Write(1, 1);  // simple code for the context map
  output->Write(2, 1);  // two clusters
  output->Write(1, 1);  // raw/lz77 length histogram last
  output->Write(1, 0);  // distance histogram first
  output->Write(1, 1);  // use prefix codes
  output->Write(4, 0);  // 000 hybrid uint config for distances (only need 0)
  output->Write(4, 0);  // 000 hybrid uint config for symbols (only <= 10)
  // Distance alphabet size:
  output->Write(5, 0b00001);  // 2: just need 1 for RLE (i.e. distance 1)
  // Symbol + LZ77 alphabet size:
  output->Write(1, 1);    // > 1
  output->Write(4, 8);    // <= 512
  output->Write(8, 255);  // == 512

  // Distance histogram:
  output->Write(2, 1);  // simple prefix code
  output->Write(2, 0);  // with one symbol
  output->Write(1, 1);  // 1

  // Symbol + lz77 histogram:
  code.WriteTo(output);

  // Group header for global modular image.
  output->Write(1, 1);  // Global tree
  output->Write(1, 1);  // All default wp
}

void PrepareDCGlobal(bool is_single_group, size_t width, size_t height,
                     size_t nb_chans, const PrefixCode& code,
                     BitWriter* output) {
  PrepareDCGlobalCommon(is_single_group, width, height, code, output);
  if (nb_chans > 2) {
    output->Write(2, 0b01);     // 1 transform
    output->Write(2, 0b00);     // RCT
    output->Write(5, 0b00000);  // Starting from ch 0
    output->Write(2, 0b00);     // YCoCg
  } else {
    output->Write(2, 0b00);  // no transforms
  }
  if (!is_single_group) {
    output->ZeroPadToByte();
  }
}

void EncodeHybridUint404_Mul16(uint32_t value, uint32_t* token_div16,
                               uint32_t* nbits, uint32_t* bits) {
  // NOTE: token in libjxl is actually << 4.
  uint32_t n = 31 - __builtin_clz(value);
  *token_div16 = value < 16 ? 0 : n - 3;
  *nbits = value < 16 ? 0 : n - 4;
  *bits = value < 16 ? 0 : (value >> 4) - (1 << *nbits);
}

template <typename BitDepth>
struct ChunkEncoder {
  static void EncodeRle(size_t count, const PrefixCode& code,
                        BitWriter& output) {
    if (count == 0) return;
    count -= kLZ77MinLength;
    unsigned token_div16, nbits, bits;
    EncodeHybridUint404_Mul16(count, &token_div16, &nbits, &bits);
    output.Write(
        code.lz77_nbits[token_div16] + nbits,
        (bits << code.lz77_nbits[token_div16]) | code.lz77_bits[token_div16]);
  }

  inline void Chunk(size_t run, typename BitDepth::upixel_t* residuals) {
    EncodeRle(run, *code, *output);
    BitDepth::EncodeChunk(residuals, *code, *output);
  }

  inline void Finalize(size_t run) { EncodeRle(run, *code, *output); }

  const PrefixCode* code;
  BitWriter* output;
};

template <typename BitDepth>
struct ChunkSampleCollector {
  void Rle(size_t count, uint64_t* lz77_counts) {
    if (count == 0) return;
    count -= kLZ77MinLength;
    unsigned token_div16, nbits, bits;
    EncodeHybridUint404_Mul16(count, &token_div16, &nbits, &bits);
    lz77_counts[token_div16]++;
  }

  inline void Chunk(size_t run, typename BitDepth::upixel_t* residuals) {
    // Run is broken. Encode the run and encode the individual vector.
    Rle(run, lz77_counts);
    for (size_t ix = 0; ix < kChunkSize; ix++) {
      unsigned token, nbits, bits;
      EncodeHybridUint000(residuals[ix], &token, &nbits, &bits);
      raw_counts[token]++;
    }
  }

  // don't count final run since we don't know how long it really is
  void Finalize(size_t run) {}

  uint64_t* raw_counts;
  uint64_t* lz77_counts;
};

constexpr uint32_t PackSigned(int32_t value) {
  return (static_cast<uint32_t>(value) << 1) ^
         ((static_cast<uint32_t>(~value) >> 31) - 1);
}

template <typename T, typename BitDepth>
struct ChannelRowProcessor {
  using upixel_t = typename BitDepth::upixel_t;
  using pixel_t = typename BitDepth::pixel_t;
  T* t;
  inline void ProcessChunk(const pixel_t* row, const pixel_t* row_left,
                           const pixel_t* row_top, const pixel_t* row_topleft) {
    bool continue_rle = true;
    alignas(32) upixel_t residuals[kChunkSize] = {};
    for (size_t ix = 0; ix < kChunkSize; ix++) {
      pixel_t px = row[ix];
      pixel_t left = row_left[ix];
      pixel_t top = row_top[ix];
      pixel_t topleft = row_topleft[ix];
      pixel_t ac = left - topleft;
      pixel_t ab = left - top;
      pixel_t bc = top - topleft;
      pixel_t grad = static_cast<pixel_t>(static_cast<upixel_t>(ac) +
                                          static_cast<upixel_t>(top));
      pixel_t d = ab ^ bc;
      pixel_t clamp = d < 0 ? top : left;
      pixel_t s = ac ^ bc;
      pixel_t pred = s < 0 ? grad : clamp;
      residuals[ix] = PackSigned(px - pred);
      continue_rle &= residuals[ix] == last;
    }
    // Run continues, nothing to do.
    if (continue_rle) {
      run += kChunkSize;
    } else {
      // Run is broken. Encode the run and encode the individual vector.
      t->Chunk(run, residuals);
      run = 0;
    }
    last = residuals[kChunkSize - 1];
  }
  void ProcessRow(const pixel_t* row, const pixel_t* row_left,
                  const pixel_t* row_top, const pixel_t* row_topleft,
                  size_t xs) {
    for (size_t x = 0; x + kChunkSize <= xs; x += kChunkSize) {
      ProcessChunk(row + x, row_left + x, row_top + x, row_topleft + x);
    }
  }

  void Finalize() { t->Finalize(run); }
  size_t run = 0;
  upixel_t last = std::numeric_limits<upixel_t>::max();  // Can never appear
};

template <typename Processor, size_t nb_chans, typename BitDepth,
          bool big_endian>
void ProcessImageArea(const unsigned char* rgba, size_t x0, size_t y0,
                      size_t oxs, size_t xs, size_t yskip, size_t ys,
                      size_t row_stride, BitDepth bitdepth,
                      Processor* processors) {
  constexpr size_t kPadding = 16;

  using pixel_t = typename BitDepth::pixel_t;
  using upixel_t = typename BitDepth::upixel_t;

  // Could use nb_chans, but clang-tidy complains otherwise.
  pixel_t group_data[4][2][256 + kPadding * 2] = {};
  upixel_t allzero[4] = {};
  upixel_t allone[4];
  auto get_pixel = [&](size_t x, size_t y, size_t channel) {
    pixel_t p = rgba[row_stride * (y0 + y) +
                     (x0 + x) * nb_chans * BitDepth::kInputBytes +
                     channel * BitDepth::kInputBytes];
    if (BitDepth::kInputBytes == 2) {
      if (big_endian) {
        p <<= 8;
        p |= rgba[row_stride * (y0 + y) + (x0 + x) * nb_chans * 2 +
                  channel * 2 + 1];
      } else {
        p |= rgba[row_stride * (y0 + y) + (x0 + x) * nb_chans * 2 +
                  channel * 2 + 1]
             << 8;
      }
    }
    return p;
  };

  size_t one_mask = (1 << bitdepth.bitdepth) - 1;
  for (size_t c = 0; c < nb_chans; c++) {
    allone[c] = one_mask;
  }
  for (size_t y = 0; y < ys; y++) {
    // Pre-fill rows with YCoCg converted pixels.
    for (size_t x = 0; x < oxs; x++) {
      if (nb_chans < 3) {
        pixel_t luma = get_pixel(x, y, 0);
        group_data[0][y & 1][x + kPadding] = luma;
        if (nb_chans == 2) {
          pixel_t a = get_pixel(x, y, 1);
          group_data[1][y & 1][x + kPadding] = a;
        }
      } else {
        pixel_t r = get_pixel(x, y, 0);
        pixel_t g = get_pixel(x, y, 1);
        pixel_t b = get_pixel(x, y, 2);
        if (nb_chans == 4) {
          pixel_t a = get_pixel(x, y, 3);
          group_data[3][y & 1][x + kPadding] = a;
          group_data[1][y & 1][x + kPadding] = a ? r - b : 0;
          pixel_t tmp = b + (group_data[1][y & 1][x + kPadding] >> 1);
          group_data[2][y & 1][x + kPadding] = a ? g - tmp : 0;
          group_data[0][y & 1][x + kPadding] =
              a ? tmp + (group_data[2][y & 1][x + kPadding] >> 1) : 0;
        } else {
          group_data[1][y & 1][x + kPadding] = r - b;
          pixel_t tmp = b + (group_data[1][y & 1][x + kPadding] >> 1);
          group_data[2][y & 1][x + kPadding] = g - tmp;
          group_data[0][y & 1][x + kPadding] =
              tmp + (group_data[2][y & 1][x + kPadding] >> 1);
        }
      }
      for (size_t c = 0; c < nb_chans; c++) {
        allzero[c] |= group_data[c][y & 1][x + kPadding];
        allone[c] &= group_data[c][y & 1][x + kPadding];
      }
    }
    // Deal with x == 0.
    for (size_t c = 0; c < nb_chans; c++) {
      group_data[c][y & 1][kPadding - 1] =
          y > 0 ? group_data[c][(y - 1) & 1][kPadding] : 0;
      // Fix topleft.
      group_data[c][(y - 1) & 1][kPadding - 1] =
          y > 0 ? group_data[c][(y - 1) & 1][kPadding] : 0;
    }
    // Fill in padding.
    for (size_t c = 0; c < nb_chans; c++) {
      for (size_t x = oxs; x < xs; x++) {
        group_data[c][y & 1][kPadding + x] =
            group_data[c][y & 1][kPadding + oxs - 1];
      }
    }
    if (y < yskip) continue;
    for (size_t c = 0; c < nb_chans; c++) {
      if (y > 0 && (allzero[c] == 0 || allone[c] == one_mask)) {
        processors[c].run += xs;
        continue;
      }

      // Get pointers to px/left/top/topleft data to speedup loop.
      const pixel_t* row = &group_data[c][y & 1][kPadding];
      const pixel_t* row_left = &group_data[c][y & 1][kPadding - 1];
      const pixel_t* row_top =
          y == 0 ? row_left : &group_data[c][(y - 1) & 1][kPadding];
      const pixel_t* row_topleft =
          y == 0 ? row_left : &group_data[c][(y - 1) & 1][kPadding - 1];

      processors[c].ProcessRow(row, row_left, row_top, row_topleft, xs);
    }
  }
  for (size_t c = 0; c < nb_chans; c++) {
    processors[c].Finalize();
  }
}

template <typename Processor, size_t nb_chans, typename BitDepth>
void ProcessImageArea(const unsigned char* rgba, size_t x0, size_t y0,
                      size_t oxs, size_t xs, size_t yskip, size_t ys,
                      size_t row_stride, BitDepth bitdepth, bool big_endian,
                      Processor* processors) {
  if (big_endian) {
    ProcessImageArea<Processor, nb_chans, BitDepth, /*big_endian=*/true>(
        rgba, x0, y0, oxs, xs, yskip, ys, row_stride, bitdepth, processors);
  } else {
    ProcessImageArea<Processor, nb_chans, BitDepth, /*big_endian=*/false>(
        rgba, x0, y0, oxs, xs, yskip, ys, row_stride, bitdepth, processors);
  }
}

template <size_t nb_chans, typename BitDepth>
void WriteACSection(const unsigned char* rgba, size_t x0, size_t y0, size_t oxs,
                    size_t ys, size_t row_stride, bool is_single_group,
                    BitDepth bitdepth, bool big_endian, const PrefixCode& code,
                    std::array<BitWriter, 4>& output) {
  size_t xs = (oxs + kChunkSize - 1) / kChunkSize * kChunkSize;
  for (size_t i = 0; i < nb_chans; i++) {
    if (is_single_group && i == 0) continue;
    output[i].Allocate(xs * ys * bitdepth.MaxEncodedBitsPerSample() + 4);
  }
  if (!is_single_group) {
    // Group header for modular image.
    // When the image is single-group, the global modular image is the one that
    // contains the pixel data, and there is no group header.
    output[0].Write(1, 1);     // Global tree
    output[0].Write(1, 1);     // All default wp
    output[0].Write(2, 0b00);  // 0 transforms
  }

  ChunkEncoder<BitDepth> encoders[nb_chans];
  ChannelRowProcessor<ChunkEncoder<BitDepth>, BitDepth> row_encoders[nb_chans];
  for (size_t c = 0; c < nb_chans; c++) {
    row_encoders[c].t = &encoders[c];
    encoders[c].output = &output[c];
    encoders[c].code = &code;
  }
  ProcessImageArea<ChannelRowProcessor<ChunkEncoder<BitDepth>, BitDepth>,
                   nb_chans>(rgba, x0, y0, oxs, xs, 0, ys, row_stride, bitdepth,
                             big_endian, row_encoders);
}

constexpr int kHashExp = 16;
constexpr uint32_t kHashSize = 1 << kHashExp;
constexpr uint32_t kHashMultiplier = 2654435761;
constexpr int kMaxColors = 512;

// can be any function that returns a value in 0 .. kHashSize-1
// has to map 0 to 0
inline uint32_t pixel_hash(uint32_t p) {
  return (p * kHashMultiplier) >> (32 - kHashExp);
}

template <typename Processor, size_t nb_chans>
void ProcessImageAreaPalette(const unsigned char* rgba, size_t x0, size_t y0,
                             size_t oxs, size_t xs, size_t yskip, size_t ys,
                             size_t row_stride, const int16_t* lookup,
                             Processor* processors) {
  constexpr size_t kPadding = 16;

  int16_t group_data[2][256 + kPadding * 2] = {};
  Processor& row_encoder = processors[0];

  for (size_t y = 0; y < ys; y++) {
    // Pre-fill rows with palette converted pixels.
    const unsigned char* inrow = rgba + row_stride * (y0 + y) + x0 * nb_chans;
    for (size_t x = 0; x < oxs; x++) {
      uint32_t p = 0;
      memcpy(&p, inrow + x * nb_chans, nb_chans);
      group_data[y & 1][x + kPadding] = lookup[pixel_hash(p)];
    }
    // Deal with x == 0.
    group_data[y & 1][kPadding - 1] =
        y > 0 ? group_data[(y - 1) & 1][kPadding] : 0;
    // Fix topleft.
    group_data[(y - 1) & 1][kPadding - 1] =
        y > 0 ? group_data[(y - 1) & 1][kPadding] : 0;
    // Fill in padding.
    for (size_t x = oxs; x < xs; x++) {
      group_data[y & 1][kPadding + x] = group_data[y & 1][kPadding + oxs - 1];
    }
    // Get pointers to px/left/top/topleft data to speedup loop.
    const int16_t* row = &group_data[y & 1][kPadding];
    const int16_t* row_left = &group_data[y & 1][kPadding - 1];
    const int16_t* row_top =
        y == 0 ? row_left : &group_data[(y - 1) & 1][kPadding];
    const int16_t* row_topleft =
        y == 0 ? row_left : &group_data[(y - 1) & 1][kPadding - 1];

    row_encoder.ProcessRow(row, row_left, row_top, row_topleft, xs);
  }
  row_encoder.Finalize();
}

template <size_t nb_chans>
void WriteACSectionPalette(const unsigned char* rgba, size_t x0, size_t y0,
                           size_t oxs, size_t ys, size_t row_stride,
                           bool is_single_group, const PrefixCode& code,
                           const int16_t* lookup, BitWriter& output) {
  size_t xs = (oxs + kChunkSize - 1) / kChunkSize * kChunkSize;

  if (!is_single_group) {
    output.Allocate(16 * xs * ys + 4);
    // Group header for modular image.
    // When the image is single-group, the global modular image is the one that
    // contains the pixel data, and there is no group header.
    output.Write(1, 1);     // Global tree
    output.Write(1, 1);     // All default wp
    output.Write(2, 0b00);  // 0 transforms
  }

  ChunkEncoder<UpTo8Bits> encoder;
  ChannelRowProcessor<ChunkEncoder<UpTo8Bits>, UpTo8Bits> row_encoder;

  row_encoder.t = &encoder;
  encoder.output = &output;
  encoder.code = &code;
  ProcessImageAreaPalette<
      ChannelRowProcessor<ChunkEncoder<UpTo8Bits>, UpTo8Bits>, nb_chans>(
      rgba, x0, y0, oxs, xs, 0, ys, row_stride, lookup, &row_encoder);
}

template <size_t nb_chans, typename BitDepth>
void CollectSamples(const unsigned char* rgba, size_t x0, size_t y0, size_t xs,
                    size_t row_stride, size_t row_count, uint64_t* raw_counts,
                    uint64_t* lz77_counts, bool palette, BitDepth bitdepth,
                    bool big_endian, const int16_t* lookup) {
  if (palette) {
    ChunkSampleCollector<UpTo8Bits> sample_collectors[nb_chans];
    ChannelRowProcessor<ChunkSampleCollector<UpTo8Bits>, UpTo8Bits>
        row_sample_collectors[nb_chans];
    for (size_t c = 0; c < nb_chans; c++) {
      row_sample_collectors[c].t = &sample_collectors[c];
      sample_collectors[c].raw_counts = raw_counts;
      sample_collectors[c].lz77_counts = lz77_counts;
    }
    ProcessImageAreaPalette<
        ChannelRowProcessor<ChunkSampleCollector<UpTo8Bits>, UpTo8Bits>,
        nb_chans>(rgba, x0, y0, xs, xs, 1, 1 + row_count, row_stride, lookup,
                  row_sample_collectors);
  } else {
    ChunkSampleCollector<BitDepth> sample_collectors[nb_chans];
    ChannelRowProcessor<ChunkSampleCollector<BitDepth>, BitDepth>
        row_sample_collectors[nb_chans];
    for (size_t c = 0; c < nb_chans; c++) {
      row_sample_collectors[c].t = &sample_collectors[c];
      sample_collectors[c].raw_counts = raw_counts;
      sample_collectors[c].lz77_counts = lz77_counts;
    }
    ProcessImageArea<
        ChannelRowProcessor<ChunkSampleCollector<BitDepth>, BitDepth>,
        nb_chans>(rgba, x0, y0, xs, xs, 1, 1 + row_count, row_stride, bitdepth,
                  big_endian, row_sample_collectors);
  }
}

void PrepareDCGlobalPalette(bool is_single_group, size_t width, size_t height,
                            const PrefixCode& code,
                            const std::vector<uint32_t>& palette,
                            size_t pcolors_real, BitWriter* output) {
  PrepareDCGlobalCommon(is_single_group, width, height, code, output);
  output->Write(2, 0b01);     // 1 transform
  output->Write(2, 0b01);     // Palette
  output->Write(5, 0b00000);  // Starting from ch 0
  output->Write(2, 0b10);     // 4-channel palette (RGBA)
  size_t pcolors = (pcolors_real + kChunkSize - 1) / kChunkSize * kChunkSize;
  // pcolors <= kMaxColors + kChunkSize - 1
  static_assert(kMaxColors + kChunkSize < 1281,
                "add code to signal larger palette sizes");
  if (pcolors < 256) {
    output->Write(2, 0b00);
    output->Write(8, pcolors);
  } else {
    output->Write(2, 0b01);
    output->Write(10, pcolors - 256);
  }

  output->Write(2, 0b00);  // nb_deltas == 0
  output->Write(4, 0);     // Zero predictor for delta palette
  // Encode palette
  ChunkEncoder<UpTo8Bits> encoder;
  ChannelRowProcessor<ChunkEncoder<UpTo8Bits>, UpTo8Bits> row_encoder;
  row_encoder.t = &encoder;
  encoder.output = output;
  encoder.code = &code;
  int16_t p[4][32 + 1024] = {};
  uint8_t prgba[4];
  size_t i = 0;
  size_t have_zero = 0;
  if (palette[pcolors_real - 1] == 0) have_zero = 1;
  for (; i < pcolors; i++) {
    if (i < pcolors_real) {
      memcpy(prgba, &palette[i], 4);
    }
    p[0][16 + i + have_zero] = prgba[0];
    p[1][16 + i + have_zero] = prgba[1];
    p[2][16 + i + have_zero] = prgba[2];
    p[3][16 + i + have_zero] = prgba[3];
  }
  p[0][15] = 0;
  row_encoder.ProcessRow(p[0] + 16, p[0] + 15, p[0] + 15, p[0] + 15, pcolors);
  p[1][15] = p[0][16];
  p[0][15] = p[0][16];
  row_encoder.ProcessRow(p[1] + 16, p[1] + 15, p[0] + 16, p[0] + 15, pcolors);
  p[2][15] = p[1][16];
  p[1][15] = p[1][16];
  row_encoder.ProcessRow(p[2] + 16, p[2] + 15, p[1] + 16, p[1] + 15, pcolors);
  p[3][15] = p[2][16];
  p[2][15] = p[2][16];
  row_encoder.ProcessRow(p[3] + 16, p[3] + 15, p[2] + 16, p[2] + 15, pcolors);
  row_encoder.Finalize();

  if (!is_single_group) {
    output->ZeroPadToByte();
  }
}

template <size_t nb_chans, typename BitDepth>
size_t LLEnc(const unsigned char* rgba, size_t width, size_t stride,
             size_t height, BitDepth bitdepth, bool big_endian, int effort,
             unsigned char** output) {
  assert(width != 0);
  assert(height != 0);
  assert(stride >= nb_chans * BitDepth::kInputBytes * width);

  // Count colors to try palette
  std::vector<uint32_t> palette(kHashSize);
  palette[0] = 1;
  int16_t lookup[kHashSize];
  lookup[0] = 0;
  int pcolors = 0;
  bool collided = effort < 2 || bitdepth.bitdepth != 8 ||
                  nb_chans < 4;  // todo: also do rgb palette
  for (size_t y = 0; y < height && !collided; y++) {
    const unsigned char* r = rgba + stride * y;
    size_t x = 0;
    if (nb_chans == 4) {
      // this is just an unrolling of the next loop
      for (; x + 7 < width; x += 8) {
        uint32_t p[8], index[8];
        memcpy(p, r + x * 4, 32);
        for (int i = 0; i < 8; i++) index[i] = pixel_hash(p[i]);
        for (int i = 0; i < 8; i++) {
          uint32_t init_entry = index[i] ? 0 : 1;
          if (init_entry != palette[index[i]] && p[i] != palette[index[i]]) {
            collided = true;
          }
        }
        for (int i = 0; i < 8; i++) palette[index[i]] = p[i];
      }
      for (; x < width; x++) {
        uint32_t p;
        memcpy(&p, r + x * 4, 4);
        uint32_t index = pixel_hash(p);
        uint32_t init_entry = index ? 0 : 1;
        if (init_entry != palette[index] && p != palette[index]) {
          collided = true;
        }
        palette[index] = p;
      }
    } else {
      for (; x < width; x++) {
        uint32_t p = 0;
        memcpy(&p, r + x * nb_chans, nb_chans);
        uint32_t index = pixel_hash(p);
        uint32_t init_entry = index ? 0 : 1;
        if (init_entry != palette[index] && p != palette[index]) {
          collided = true;
        }
        palette[index] = p;
      }
    }
  }

  int nb_entries = 0;
  if (!collided) {
    if (palette[0] == 0) pcolors = 1;
    if (palette[0] == 1) palette[0] = 0;
    bool have_color = false;
    uint8_t minG = 255, maxG = 0;
    for (uint32_t k = 0; k < kHashSize; k++) {
      if (palette[k] == 0) continue;
      uint8_t p[4];
      memcpy(p, &palette[k], 4);
      // move entries to front so sort has less work
      palette[nb_entries] = palette[k];
      if (p[0] != p[1] || p[0] != p[2]) have_color = true;
      if (p[1] < minG) minG = p[1];
      if (p[1] > maxG) maxG = p[1];
      nb_entries++;
      // don't do palette if too many colors are needed
      if (nb_entries + pcolors > kMaxColors) {
        collided = true;
        break;
      }
    }
    if (!have_color) {
      // don't do palette if it's just grayscale without many holes
      if (maxG - minG < nb_entries * 1.4f) collided = true;
    }
  }
  if (!collided) {
    std::sort(
        palette.begin(), palette.begin() + nb_entries,
        [](uint32_t ap, uint32_t bp) {
          if (ap == 0) return false;
          if (bp == 0) return true;
          uint8_t a[4], b[4];
          memcpy(a, &ap, 4);
          memcpy(b, &bp, 4);
          float ay, by;
          ay = (0.299f * a[0] + 0.587f * a[1] + 0.114f * a[2] + 0.01f) * a[3];
          by = (0.299f * b[0] + 0.587f * b[1] + 0.114f * b[2] + 0.01f) * b[3];
          return ay < by;  // sort on alpha*luma
        });
    for (int k = 0; k < nb_entries; k++) {
      if (palette[k] == 0) break;
      lookup[pixel_hash(palette[k])] = pcolors++;
    }
  }

  // Width gets padded to kChunkSize, but this computation doesn't change
  // because of that.
  size_t num_groups_x = (width + 255) / 256;
  size_t num_groups_y = (height + 255) / 256;
  size_t num_dc_groups_x = (width + 2047) / 2048;
  size_t num_dc_groups_y = (height + 2047) / 2048;

  uint64_t raw_counts[kNumRawSymbols] = {};
  uint64_t lz77_counts[17] = {};

  // sample the middle (effort * 2) rows of every group
  for (size_t g = 0; g < num_groups_y * num_groups_x; g++) {
    size_t xg = g % num_groups_x;
    size_t yg = g / num_groups_x;
    int y_offset = yg * 256;
    int y_max = std::min<size_t>(height - yg * 256, 256);
    int y_begin = y_offset + std::max<int>(0, y_max - 2 * effort) / 2;
    int y_count =
        std::min<int>(2 * effort * y_max / 256, y_offset + y_max - y_begin - 1);
    int x_max =
        std::min<size_t>(width - xg * 256, 256) / kChunkSize * kChunkSize;
    CollectSamples<nb_chans>(rgba, xg * 256, y_begin, x_max, stride, y_count,
                             raw_counts, lz77_counts, !collided, bitdepth,
                             big_endian, lookup);
  }

  // TODO(veluca): can probably improve this and make it bitdepth-dependent.
  uint64_t base_raw_counts[kNumRawSymbols] = {
      3843, 852, 1270, 1214, 1014, 727, 481, 300, 159, 51,
      5,    1,   1,    1,    1,    1,   1,   1,   1};

  bool doing_ycocg = nb_chans > 2 && collided;
  for (size_t i = bitdepth.NumSymbols(doing_ycocg); i < kNumRawSymbols; i++) {
    base_raw_counts[i] = 0;
  }
  uint64_t base_lz77_counts[17] = {
      // short runs will be sampled, but long ones won't.
      // near full-group run is quite common (e.g. all-opaque alpha)
      18, 12, 9, 11, 15, 2, 2, 1, 1, 1, 1, 2, 300, 0, 0, 0, 0};

  for (size_t i = 0; i < kNumRawSymbols; i++) {
    raw_counts[i] = (raw_counts[i] << 8) + base_raw_counts[i];
  }

  if (!collided) {
    unsigned token, nbits, bits;
    EncodeHybridUint000(PackSigned(pcolors - 1), &token, &nbits, &bits);
    // ensure all palette indices can actually be encoded
    for (size_t i = 0; i < token + 1; i++)
      raw_counts[i] = std::max<uint64_t>(raw_counts[i], 1);
    // these tokens are only used for the palette itself so they can get a bad
    // code
    for (size_t i = token + 1; i < 10; i++) raw_counts[i] = 1;
  }
  for (size_t i = 0; i < 17; i++) {
    lz77_counts[i] = (lz77_counts[i] << 8) + base_lz77_counts[i];
  }

  alignas(32) PrefixCode hcode(bitdepth, raw_counts, lz77_counts);

  BitWriter writer;

  bool onegroup = num_groups_x == 1 && num_groups_y == 1;

  size_t num_groups = onegroup ? 1
                               : (2 + num_dc_groups_x * num_dc_groups_y +
                                  num_groups_x * num_groups_y);

  std::vector<std::array<BitWriter, 4>> group_data(num_groups);
  if (collided) {
    PrepareDCGlobal(onegroup, width, height, nb_chans, hcode,
                    &group_data[0][0]);
  } else {
    PrepareDCGlobalPalette(onegroup, width, height, hcode, palette, pcolors,
                           &group_data[0][0]);
  }
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t g = 0; g < num_groups_y * num_groups_x; g++) {
    size_t xg = g % num_groups_x;
    size_t yg = g / num_groups_x;
    size_t group_id =
        onegroup ? 0 : (2 + num_dc_groups_x * num_dc_groups_y + g);
    size_t xs = std::min<size_t>(width - xg * 256, 256);
    size_t ys = std::min<size_t>(height - yg * 256, 256);
    size_t x0 = xg * 256;
    size_t y0 = yg * 256;
    auto& gd = group_data[group_id];
    if (collided) {
      WriteACSection<nb_chans>(rgba, x0, y0, xs, ys, stride, onegroup, bitdepth,
                               big_endian, hcode, gd);

    } else {
      WriteACSectionPalette<nb_chans>(rgba, x0, y0, xs, ys, stride, onegroup,
                                      hcode, lookup, gd[0]);
    }
  }

  AssembleFrame(width, height, nb_chans, bitdepth.bitdepth, kChunkSize,
                group_data, &writer);

  *output = writer.data.release();
  return writer.bytes_written;
}

template <typename BitDepth>
size_t LLEnc(const unsigned char* rgba, size_t width, size_t stride,
             size_t height, size_t nb_chans, BitDepth bitdepth, bool big_endian,
             int effort, unsigned char** output) {
  assert(nb_chans <= 4);
  assert(nb_chans != 0);
  if (nb_chans == 1) {
    return LLEnc<1>(rgba, width, stride, height, bitdepth, big_endian, effort,
                    output);
  }
  if (nb_chans == 2) {
    return LLEnc<2>(rgba, width, stride, height, bitdepth, big_endian, effort,
                    output);
  }
  if (nb_chans == 3) {
    return LLEnc<3>(rgba, width, stride, height, bitdepth, big_endian, effort,
                    output);
  }
  return LLEnc<4>(rgba, width, stride, height, bitdepth, big_endian, effort,
                  output);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

size_t JxlFastLosslessEncode(const unsigned char* rgba, size_t width,
                             size_t stride, size_t height, size_t nb_chans,
                             size_t bitdepth, bool big_endian, int effort,
                             unsigned char** output) {
  assert(bitdepth > 0);
  if (bitdepth <= 8) {
    return LLEnc(rgba, width, stride, height, nb_chans, UpTo8Bits(bitdepth),
                 big_endian, effort, output);
  }
  if (bitdepth <= 13) {
    return LLEnc(rgba, width, stride, height, nb_chans, From9To13Bits(bitdepth),
                 big_endian, effort, output);
  }
  if (bitdepth == 14) {
    return LLEnc(rgba, width, stride, height, nb_chans, Exactly14Bits(bitdepth),
                 big_endian, effort, output);
  }
  return LLEnc(rgba, width, stride, height, nb_chans, MoreThan14Bits(bitdepth),
               big_endian, effort, output);
}

#ifdef __cplusplus
}  // extern "C"
#endif
