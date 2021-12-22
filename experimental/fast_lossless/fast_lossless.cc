// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "fast_lossless.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <array>
#include <memory>
#include <vector>

#if (!defined(__BYTE_ORDER__) || (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__))
#error "system not known to be little endian"
#endif

struct BitWriter {
  void Allocate(size_t maximum_bit_size) {
    assert(data == nullptr);
    // Leave some padding.
    data.reset((uint8_t*)malloc(maximum_bit_size / 8 + 32));
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

/*
// Run this in the libjxl codebase to compute the prefix code tables and code:
#include "lib/jxl/enc_huffman.h"
__attribute__((constructor)) void f() {
  uint32_t histo[512] = {};
  histo[0] = 103741937;
  histo[1] = 63368045;
  histo[2] = 95396302;
  histo[3] = 82611295;
  histo[4] = 56681795;
  histo[5] = 27357516;
  // These are bumped up as they otherwise go above the maximum bit count.
  histo[6] = 30602258;
  histo[7] = 20354502;
  histo[8] = 20042520;
  histo[9] = 20059400;
  histo[10] = 20004000;

  for (size_t i = 0; i < 17; i++) {
    histo[kLZ77Offset + i * 16] = 1;
  }
  histo[kLZ77Offset + 0 * 16] = 119345;
  histo[kLZ77Offset + 1 * 16] = 51104;
  histo[kLZ77Offset + 2 * 16] = 44302;
  histo[kLZ77Offset + 3 * 16] = 33556;
  histo[kLZ77Offset + 4 * 16] = 26230;
  histo[kLZ77Offset + 5 * 16] = 1059;
  histo[kLZ77Offset + 6 * 16] = 364;
  histo[kLZ77Offset + 7 * 16] = 288;
  histo[kLZ77Offset + 8 * 16] = 183;
  histo[kLZ77Offset + 9 * 16] = 146;
  histo[kLZ77Offset + 10 * 16] = 210;
  histo[kLZ77Offset + 11 * 16] = 87;
  histo[kLZ77Offset + 12 * 16] = 2374;

  uint8_t depth[512] = {};
  uint16_t bits[512] = {};
  BitWriter w;
  BitWriter::Allotment allotment(&w, 1000);
  BuildAndStoreHuffmanTree(histo, 512, depth, bits, &w);
  ReclaimAndCharge(&w, &allotment, 0, nullptr);

  unsigned wbits = w.BitsWritten();
  w.ZeroPadToByte();
  fprintf(stderr, "constexpr uint8_t kRawNBits[16] = {\n");
  for (size_t i = 0; i < 11; i++) {
    fprintf(stderr, "%d,", depth[i]);
  }
  fprintf(stderr, "};\nconstexpr uint8_t kRawBits[16] = {\n");
  for (size_t i = 0; i < 11; i++) {
    fprintf(stderr, "0x%x,", bits[i]);
  }
  fprintf(stderr, "};\nconstexpr uint8_t kLZ77NBits[17] = {\n");
  for (size_t i = 0; i < 17; i++) {
    fprintf(stderr, "%d,", depth[kLZ77Offset + i * 16]);
  }
  fprintf(stderr, "};\nconstexpr uint16_t kLZ77Bits[17] = {\n");
  for (size_t i = 0; i < 17; i++) {
    fprintf(stderr, "0x%x,", bits[kLZ77Offset + i * 16]);
  }
  fprintf(stderr, "};\nconstexpr uint8_t kHistoCode[] = {");
  auto wspan = w.GetSpan();
  for (size_t i = 0; i * 8 < w.BitsWritten(); i++) {
    fprintf(stderr, "0x%x, ", wspan[i]);
  }
  fprintf(stderr, "};\nconstexpr size_t kHistoBits = %u;\n", wbits);
  exit(1);
}

*/

struct PrefixCode {
  alignas(32) uint8_t raw_nbits[16] = {
      2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6,
  };
  alignas(32) uint8_t raw_bits[16] = {};
  alignas(32) uint8_t lz77_nbits[17] = {
      7, 9, 9, 9, 10, 13, 14, 14, 15, 15, 15, 15, 11, 15, 15, 15, 15,
  };

  alignas(32) uint16_t lz77_bits[17] = {};

  uint16_t BitReverse(size_t nbits, uint16_t bits) {
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

  PrefixCode() {
    constexpr size_t kNumRaw = 11;
    constexpr size_t kNumLZ77 = 17;

    // Create the prefix codes given the code lengths.
    {
      uint8_t code_length_counts[16] = {};
      for (size_t i = 0; i < kNumRaw; i++) {
        code_length_counts[raw_nbits[i]]++;
        assert(raw_nbits[i] <= 7);
        assert(raw_nbits[i] > 0);
      }
      for (size_t i = 0; i < kNumLZ77; i++) {
        code_length_counts[lz77_nbits[i]]++;
      }

      uint16_t next_code[16] = {};

      uint16_t code = 0;
      for (size_t i = 1; i < 16; i++) {
        code = (code + code_length_counts[i - 1]) << 1;
        next_code[i] = code;
      }

      for (size_t i = 0; i < kNumRaw; i++) {
        raw_bits[i] = BitReverse(raw_nbits[i], next_code[raw_nbits[i]]++);
      }
      for (size_t i = 0; i < kNumLZ77; i++) {
        lz77_bits[i] = BitReverse(lz77_nbits[i], next_code[lz77_nbits[i]]++);
      }
    }
  }

  void WriteTo(BitWriter* writer) {
    // TODO
  }
};

constexpr uint8_t kHistoCode[] = {
    0xf0, 0x3d, 0x7f, 0xcf, 0xf3, 0xff, 0xfc, 0xdb, 0xaa, 0xaa,
    0x7a, 0x1f, 0x63, 0x4c, 0x12, 0x5a, 0x80, 0x1b, 0x70, 0x3,
    0x6e, 0xc0, 0x3,  0xf8, 0x0,  0x3f, 0xe0, 0x7,  0xc,  0x60,
    0x0,  0x3,  0x18, 0xc0, 0xb,  0x18, 0xc0, 0x0,  0x6,  0x30,
};
constexpr size_t kHistoBits = 320;

constexpr size_t kLZ77Offset = 224;
constexpr size_t kLZ77MinLength = 16;

constexpr size_t kChunkSize = 16;

void EncodeHybridUint000(uint32_t value, uint32_t* token, uint32_t* nbits,
                         uint32_t* bits) {
  uint32_t n = 31 - __builtin_clz(value);
  *token = value ? n + 1 : 0;
  *nbits = value ? n : 0;
  *bits = value ? value - (1 << n) : 0;
}

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

void AssembleFrame(size_t width, size_t height,
                   const std::vector<std::array<BitWriter, 4>>& group_data,
                   BitWriter* output) {
  size_t total_size_groups = 0;
  std::vector<size_t> group_sizes(group_data.size());
  for (size_t i = 0; i < group_data.size(); i++) {
    size_t sz = 0;
    for (size_t j = 0; j < 4; j++) {
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
  output->Write(1, 0);     // all_default
  output->Write(1, 0);     // extra_fields
  output->Write(1, 0);     // bit_depth.floating_point_sample
  output->Write(2, 0b00);  // bit_depth.bits_per_sample = 8
  output->Write(1, 1);     // 16-bit-buffer sufficient
  output->Write(2, 0b01);  // One extra channel
  output->Write(1, 1);     // ... all_default (ie. 8-bit alpha)
  output->Write(1, 0);     // Not XYB
  output->Write(1, 1);     // color_encoding.all_default (sRGB)
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
  output->Write(2, 0b00);  // no alpha upsampling
  output->Write(2, 0b01);  // default group size
  output->Write(2, 0b00);  // exactly one pass
  if (width % kChunkSize == 0) {
    output->Write(1, 0);  // no custom size or origin
  } else {
    output->Write(1, 1);  // custom size
    wsz_fh(0);            // x0 = 0
    wsz_fh(0);            // y0 = 0
    wsz_fh((width + kChunkSize - 1) / kChunkSize *
           kChunkSize);  // xsize rounded up to chunk size
    wsz_fh(height);      // ysize same
  }
  output->Write(2, 0b00);  // kReplace blending mode
  output->Write(2, 0b00);  // kReplace blending mode for alpha channel
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
    for (size_t j = 0; j < 4; j++) {
      AppendWriter(output, &group_data[i][j]);
    }
    output->ZeroPadToByte();
  }
}

void PrepareDCGlobal(bool is_single_group, size_t width, size_t height,
                     const PrefixCode& code, BitWriter* output) {
  output->Allocate(1000 + (is_single_group ? width * height * 15 : 0));
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
  for (size_t i = 0; i < (kHistoBits + 7) / 8 * 8; i += 8) {
    output->Write(std::min<size_t>(kHistoBits - i, 8), kHistoCode[i / 8]);
  }

  // Group header for global modular image.
  output->Write(1, 1);        // Global tree
  output->Write(1, 1);        // All default wp
  output->Write(2, 0b01);     // 1 transform
  output->Write(2, 0b00);     // RCT
  output->Write(5, 0b00000);  // Starting from ch 0
  output->Write(2, 0b00);     // YCoCg

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

void EncodeRle(uint16_t residual, size_t count, const PrefixCode& code,
               BitWriter& output) {
  if (count == 0) return;
  count -= kLZ77MinLength;
  unsigned token_div16, nbits, bits;
  EncodeHybridUint404_Mul16(count, &token_div16, &nbits, &bits);
  output.Write(
      code.lz77_nbits[token_div16] + nbits,
      (bits << code.lz77_nbits[token_div16]) | code.lz77_bits[token_div16]);
}

#ifdef FASTLL_ENABLE_AVX2_INTRINSICS
#include <immintrin.h>
void EncodeChunk(const uint16_t* residuals, const PrefixCode& prefix_code,
                 BitWriter& output) {
  static_assert(kChunkSize == 16, "Chunk size must be 16");
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
                              _mm_load_si128((__m128i*)prefix_code.raw_nbits)),
                          token_masked);

  auto huff_bits =
      _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(
                              _mm_load_si128((__m128i*)prefix_code.raw_bits)),
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

void EncodeChunk(const uint16_t* residuals, const PrefixCode& code,
                 BitWriter& output) {
  uint16x8_t res = vld1q_u16(residuals);
  uint16x8_t token = vsubq_u16(vdupq_n_u16(16), vclzq_u16(res));
  uint16x8_t nbits = vqsubq_u16(token, vdupq_n_u16(1));
  uint16x8_t bits = vqsubq_u16(res, vshlq_s16(vdupq_n_s16(1), nbits));
  uint16x8_t huff_bits =
      vandq_u16(vdupq_n_u16(0xFF), vqtbl1q_u8(vld1q_u8(code.raw_bits), token));
  uint16x8_t huff_nbits =
      vandq_u16(vdupq_n_u16(0xFF), vqtbl1q_u8(vld1q_u8(code.raw_nbits), token));
  bits = vorrq_u16(vshlq_u16(bits, huff_nbits), huff_bits);
  nbits = vaddq_u16(nbits, huff_nbits);

  // Merge nbits and bits from 16-bit to 32-bit lanes.
  uint32x4_t nbits_lo16 = vandq_u32(nbits, vdupq_n_u32(0xFFFF));
  uint32x4_t bits_hi16 = vshlq_u32(vshrq_n_u32(bits, 16), nbits_lo16);
  uint32x4_t bits_lo16 = vandq_u32(bits, vdupq_n_u32(0xFFFF));

  uint32x4_t nbits32 = vsraq_n_u32(nbits_lo16, nbits, 16);
  uint32x4_t bits32 = vorrq_u32(bits_hi16, bits_lo16);

  // Merging up to 64 bits is not faster.

  // Manually merge the buffer bits with the SIMD bits.
  // A bit faster.
  for (size_t i = 0; i < 4; i++) {
    output.buffer |= bits32[i] << output.bits_in_buffer;
    memcpy(output.data.get() + output.bytes_written, &output.buffer, 8);
    output.bits_in_buffer += nbits32[i];
    size_t bytes_in_buffer = output.bits_in_buffer / 8;
    output.bits_in_buffer -= bytes_in_buffer * 8;
    output.buffer >>= bytes_in_buffer * 8;
    output.bytes_written += bytes_in_buffer;
  }
}
#endif

constexpr uint16_t PackSigned(int16_t value) {
  return (static_cast<uint16_t>(value) << 1) ^
         ((static_cast<uint16_t>(~value) >> 15) - 1);
}

struct ChannelRowEncoder {
  inline void ProcessChunk(const int16_t* row, const int16_t* row_left,
                           const int16_t* row_top, const int16_t* row_topleft,
                           const PrefixCode& code, BitWriter& output) {
    bool continue_rle = true;
    alignas(32) uint16_t residuals[kChunkSize] = {};
    for (size_t ix = 0; ix < kChunkSize; ix++) {
      int16_t px = row[ix];
      int16_t left = row_left[ix];
      int16_t top = row_top[ix];
      int16_t topleft = row_topleft[ix];

      int16_t m = std::min(top, left);
      int16_t M = std::max(top, left);
      int16_t grad = static_cast<int16_t>(static_cast<uint16_t>(top) +
                                          static_cast<uint16_t>(left) -
                                          static_cast<uint16_t>(topleft));
      int16_t grad_clamp_M = (topleft < m) ? M : grad;
      int16_t pred = (topleft > M) ? m : grad_clamp_M;
      residuals[ix] = PackSigned(px - pred);
      continue_rle &= residuals[ix] == last;
    }
    // Run continues, nothing to do.
    if (continue_rle) {
      run += kChunkSize;
    } else {
      // Run is broken. Encode the run and encode the individual vector.
      EncodeRle(last, run, code, output);
      run = 0;
#ifdef FASTLL_ENABLE_AVX2_INTRINSICS
      EncodeChunk(residuals, code, output);
#elif FASTLL_ENABLE_NEON_INTRINSICS
      EncodeChunk(residuals, code, output);
      if (kChunkSize > 8) {
        EncodeChunk(residuals + 8, code, output);
      }
#else
      for (size_t ix = 0; ix < kChunkSize; ix++) {
        unsigned token, nbits, bits;
        EncodeHybridUint000(residuals[ix], &token, &nbits, &bits);

        output.Write(code.raw_nbits[token] + nbits,
                     code.raw_bits[token] | bits << code->raw_nbits[token]);
      }
#endif
    }
    last = residuals[kChunkSize - 1];
  }
  void ProcessRow(const int16_t* row, const int16_t* row_left,
                  const int16_t* row_top, const int16_t* row_topleft, size_t xs,
                  const PrefixCode& code, BitWriter& output) {
    size_t x = 0;
    for (; x + kChunkSize <= xs; x += kChunkSize) {
      ProcessChunk(row + x, row_left + x, row_top + x, row_topleft + x, code,
                   output);
    }
  }

  void Finalize(const PrefixCode& code, BitWriter& output) {
    EncodeRle(last, run, code, output);
  }
  size_t run = 0;
  uint16_t last = 0xFFFF;  // Can never appear
};

void WriteACSection(const unsigned char* rgba, size_t x0, size_t y0, size_t oxs,
                    size_t ys, size_t row_stride, bool is_single_group,
                    const PrefixCode& code, std::array<BitWriter, 4>& output) {
  size_t xs = (oxs + kChunkSize - 1) / kChunkSize * kChunkSize;
  for (size_t i = 0; i < 4; i++) {
    if (is_single_group && i == 0) continue;
    output[i].Allocate(15 * xs * ys + 4);
  }
  if (!is_single_group) {
    // Group header for modular image.
    // When the image is single-group, the global modular image is the one that
    // contains the pixel data, and there is no group header.
    output[0].Write(1, 1);     // Global tree
    output[0].Write(1, 1);     // All default wp
    output[0].Write(2, 0b00);  // 0 transforms
  }

  constexpr size_t kPadding = 16;

  int16_t group_data[4][2][256 + kPadding * 2] = {};

  ChannelRowEncoder row_encoders[4];

  for (size_t y = 0; y < ys; y++) {
    // Pre-fill rows with YCoCg converted pixels.
    for (size_t x = 0; x < oxs; x++) {
      int16_t r = rgba[row_stride * (y0 + y) + (x0 + x) * 4 + 0];
      int16_t g = rgba[row_stride * (y0 + y) + (x0 + x) * 4 + 1];
      int16_t b = rgba[row_stride * (y0 + y) + (x0 + x) * 4 + 2];
      int16_t a = rgba[row_stride * (y0 + y) + (x0 + x) * 4 + 3];
      group_data[3][y & 1][x + kPadding] = a;
      group_data[1][y & 1][x + kPadding] = r - b;
      int16_t tmp = b + (group_data[1][y & 1][x + kPadding] >> 1);
      group_data[2][y & 1][x + kPadding] = g - tmp;
      group_data[0][y & 1][x + kPadding] =
          tmp + (group_data[2][y & 1][x + kPadding] >> 1);
    }
    // Deal with x == 0.
    for (size_t c = 0; c < 4; c++) {
      group_data[c][y & 1][kPadding - 1] =
          y > 0 ? group_data[c][(y - 1) & 1][kPadding] : 0;
    }
    // Fill in padding.
    for (size_t c = 0; c < 4; c++) {
      for (size_t x = oxs; x < xs; x++) {
        group_data[c][y & 1][kPadding + x] =
            group_data[c][y & 1][kPadding + oxs - 1];
      }
    }
    for (size_t c = 0; c < 4; c++) {
      // Get pointers to px/left/top/topleft data to speedup loop.
      const int16_t* row = &group_data[c][y & 1][kPadding];
      const int16_t* row_left = &group_data[c][y & 1][kPadding - 1];
      const int16_t* row_top =
          y == 0 ? row_left : &group_data[c][(y - 1) & 1][kPadding];
      const int16_t* row_topleft =
          y == 0 ? row_left : &group_data[c][(y - 1) & 1][kPadding - 1];

      row_encoders[c].ProcessRow(row, row_left, row_top, row_topleft, xs, code,
                                 output[c]);
    }
  }
  for (size_t c = 0; c < 4; c++) {
    row_encoders[c].Finalize(code, output[c]);
  }
}

size_t FastLosslessEncode(const unsigned char* rgba, size_t width,
                          size_t row_stride, size_t height,
                          unsigned char** output) {
  assert(width != 0);
  assert(height != 0);
  assert(row_stride >= 4 * width);

  alignas(32) PrefixCode prefix_code;

  BitWriter writer;

  // Width gets padded to kChunkSize, but this computation doesn't change
  // because of that.
  size_t num_groups_x = (width + 255) / 256;
  size_t num_groups_y = (height + 255) / 256;
  size_t num_dc_groups_x = (width + 2047) / 2048;
  size_t num_dc_groups_y = (height + 2047) / 2048;

  bool is_single_group = num_groups_x == 1 && num_groups_y == 1;

  size_t num_groups = is_single_group ? 1
                                      : (2 + num_dc_groups_x * num_dc_groups_y +
                                         num_groups_x * num_groups_y);

  std::vector<std::array<BitWriter, 4>> group_data(num_groups);

  PrepareDCGlobal(is_single_group, width, height, prefix_code,
                  &group_data[0][0]);

#pragma omp parallel for
  for (size_t g = 0; g < num_groups_y * num_groups_x; g++) {
    size_t xg = g % num_groups_x;
    size_t yg = g / num_groups_x;
    size_t group_id =
        is_single_group ? 0 : (2 + num_dc_groups_x * num_dc_groups_y + g);
    WriteACSection(rgba, xg * 256, yg * 256,
                   std::min<size_t>(width - xg * 256, 256),
                   std::min<size_t>(height - yg * 256, 256), row_stride,
                   is_single_group, prefix_code, group_data[group_id]);
  }

  AssembleFrame(width, height, group_data, &writer);

  *output = writer.data.release();
  return writer.bytes_written;
}
