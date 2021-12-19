#include "fast_lossless.h"

#include <assert.h>
#include <endian.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <vector>

#if __BYTE_ORDER != __LITTLE_ENDIAN
#error "little endian only"
#endif

/*
// Run this in the libjxl codebase to compute the prefix code tables and code:
#include "lib/jxl/enc_huffman.h"
__attribute__((constructor)) void f() {
  uint32_t histo[256] = {};
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
    histo[kLZ77Offset + i] = 1;
  }
  histo[228] = 176674;
  histo[229] = 69920;
  histo[230] = 53489;
  histo[231] = 37537;
  histo[232] = 28236;
  histo[233] = 1107;
  histo[234] = 371;
  histo[235] = 293;
  histo[236] = 181;
  histo[237] = 147;
  histo[238] = 208;
  histo[239] = 87;
  histo[240] = 2374;

  uint8_t depth[256] = {};
  uint16_t bits[256] = {};
  BitWriter w;
  BitWriter::Allotment allotment(&w, 1000);
  BuildAndStoreHuffmanTree(histo, 256, depth, bits, &w);
  ReclaimAndCharge(&w, &allotment, 0, nullptr);

  unsigned wbits = w.BitsWritten();
  w.ZeroPadToByte();
  fprintf(stderr, "constexpr uint8_t kRawNBits[11] = {\n");
  for (size_t i = 0; i < 11; i++) {
    fprintf(stderr, "%d,", depth[i]);
  }
  fprintf(stderr, "};\nconstexpr uint8_t kRawBits[11] = {\n");
  for (size_t i = 0; i < 11; i++) {
    fprintf(stderr, "0x%x,", bits[i]);
  }
  fprintf(stderr, "};\nconstexpr uint8_t kLZ77NBits[17] = {\n");
  for (size_t i = 0; i < 17; i++) {
    fprintf(stderr, "%d,", depth[kLZ77Offset + i]);
  }
  fprintf(stderr, "};\nconstexpr uint16_t kLZ77Bits[17] = {\n");
  for (size_t i = 0; i < 17; i++) {
    fprintf(stderr, "0x%x,", bits[kLZ77Offset + i]);
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

constexpr uint8_t kRawNBits[11] = {
    2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6,
};
constexpr uint8_t kRawBits[11] = {
    0x0, 0x2, 0x6, 0x1, 0x5, 0x3, 0xb, 0x7, 0x17, 0xf, 0x1f,
};
constexpr uint8_t kLZ77NBits[17] = {
    15, 15, 15, 15, 7, 9, 9, 9, 10, 13, 14, 14, 15, 15, 15, 15, 11,
};
constexpr uint16_t kLZ77Bits[17] = {
    0xfff, 0x4fff, 0x2fff, 0x6fff, 0x3f,   0x7f,   0x17f,  0xff,  0x1ff,
    0x7ff, 0x17ff, 0x37ff, 0x1fff, 0x5fff, 0x3fff, 0x7fff, 0x3ff,
};
constexpr uint8_t kHistoCode[] = {
    0x50, 0xa1, 0xf9, 0xf8, 0xcf, 0x57, 0xa4, 0xa8, 0x2,  0xd0,
    0x96, 0x63, 0xad, 0x38, 0x24, 0xef, 0x9f, 0x59, 0xf1, 0x0,
};
constexpr size_t kHistoBits = 153;

constexpr size_t kLZ77Offset = 224;
constexpr size_t kLZ77MinLength = 3;

void EncodeHybridUint000(uint32_t value, uint32_t* token, uint32_t* nbits,
                         uint32_t* bits) {
  uint32_t n = 31 - __builtin_clz(value);
  *token = value ? n + 1 : 0;
  *nbits = value ? n : 0;
  *bits = value ? value - (1 << n) : 0;
}

constexpr uint32_t PackSigned(int32_t value) {
  return (static_cast<uint32_t>(value) << 1) ^
         ((static_cast<uint32_t>(~value) >> 31) - 1);
}

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

void AssembleFrame(size_t width, size_t height,
                   const std::vector<BitWriter>& group_data,
                   BitWriter* output) {
  std::vector<size_t> group_offsets(group_data.size());
  size_t total_size_groups = 0;
  for (size_t i = 0; i < group_offsets.size(); i++) {
    const auto& writer = group_data[i];
    if (i != group_offsets.size() - 1) {
      group_offsets[i + 1] = group_offsets[i] + writer.bytes_written;
    }
    assert(writer.bits_in_buffer == 0);
    total_size_groups += 8 * writer.bytes_written;
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
  output->Write(1, 0);     // no custom size or origin
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
    size_t sz = group_data[i].bytes_written;
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

  for (size_t i = 0; i < group_offsets.size(); i++) {
    const auto& writer = group_data[i];
    memcpy(output->data.get() + output->bytes_written + group_offsets[i],
           writer.data.get(), writer.bytes_written);
  }
  output->bytes_written += total_size_groups / 8;
}

void PrepareDCGlobal(BitWriter* output) {
  output->Allocate(1000);
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
  output->Write(2, 0b00);  // lz77 min length 3 (TODO)
  static_assert(kLZ77MinLength == 3, "");
  output->Write(4, 0);  // 000 hybrid uint config for lz77
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
  output->Write(4, 7);    // <= 256
  output->Write(7, 127);  // == 256

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

  output->ZeroPadToByte();
}

__attribute__((always_inline)) void EncodeRle(uint16_t residual, size_t count,
                                              size_t rle_count,
                                              BitWriter* output) {
  if (count == 0) return;
  unsigned token, nbits, bits;
  EncodeHybridUint000(residual, &token, &nbits, &bits);
  output->Write(kRawNBits[token] + nbits,
                (bits << kRawNBits[token]) | kRawBits[token]);
  count -= 1;
  if (rle_count >= kLZ77MinLength + 1) {
    rle_count -= 1;
    count -= rle_count;
    rle_count -= kLZ77MinLength;
    unsigned token, nbits, bits;
    EncodeHybridUint000(rle_count, &token, &nbits, &bits);
    output->Write(kLZ77NBits[token] + nbits,
                  (bits << kLZ77NBits[token]) | kLZ77Bits[token]);
    // No need to encode distance: it uses 0 bits.
  }
  for (int i = 0; i < count; i++) {
    output->Write(kRawNBits[token] + nbits,
                  (bits << kRawNBits[token]) | kRawBits[token]);
  }
};

void WriteACSection(const unsigned char* rgba, size_t x0, size_t y0, size_t xs,
                    size_t ys, size_t row_stride, BitWriter* output) {
  output->Allocate(15 * xs * ys * 4 + 4);
  // Group header for modular image.
  output->Write(1, 1);     // Global tree
  output->Write(1, 1);     // All default wp
  output->Write(2, 0b00);  // 0 transforms

  int16_t group_data[4][256][256];

  for (size_t y = 0; y < ys; y++) {
    for (size_t x = 0; x < xs; x++) {
      int16_t r = rgba[row_stride * (y0 + y) + (x0 + x) * 4 + 0];
      int16_t g = rgba[row_stride * (y0 + y) + (x0 + x) * 4 + 1];
      int16_t b = rgba[row_stride * (y0 + y) + (x0 + x) * 4 + 2];
      int16_t a = rgba[row_stride * (y0 + y) + (x0 + x) * 4 + 3];
      group_data[3][y][x] = a;
      group_data[1][y][x] = r - b;
      int16_t tmp = b + (group_data[1][y][x] >> 1);
      group_data[2][y][x] = g - tmp;
      group_data[0][y][x] = tmp + (group_data[2][y][x] >> 1);
    }
  }

  for (size_t c = 0; c < 4; c++) {
    size_t run = 0;
    size_t idx = 0;
    size_t last = 0;
    for (size_t y = 0; y < ys; y++) {
      for (size_t x = 0; x < xs; x++) {
        int16_t px = group_data[c][y][x];
        int16_t left = x > 0 ? group_data[c][y][x - 1]
                             : (y > 0 ? group_data[c][y - 1][x] : 0);
        int16_t top = y > 0 ? group_data[c][y - 1][x] : left;
        int16_t topleft = x > 0 && y > 0 ? group_data[c][y - 1][x - 1] : left;

        int16_t m = std::min(top, left);
        int16_t M = std::max(top, left);
        int16_t grad = static_cast<int32_t>(static_cast<uint32_t>(top) +
                                            static_cast<uint32_t>(left) -
                                            static_cast<uint32_t>(topleft));
        int16_t grad_clamp_M = (topleft < m) ? M : grad;
        int16_t pred = (topleft > M) ? m : grad_clamp_M;
        uint16_t residual = PackSigned(px - pred);

        if (idx % 16 == 0 && run == 0) {
          last = residual;
        }
        idx++;
        if (residual == last) {
          run++;
          continue;
        } else {
          EncodeRle(last, run, run / 16 * 16, output);
          run = 0;
        }

        unsigned token, nbits, bits;
        EncodeHybridUint000(residual, &token, &nbits, &bits);

        output->Write(kRawNBits[token] + nbits,
                      kRawBits[token] | bits << kRawNBits[token]);
      }
    }
    EncodeRle(last, run, run, output);
  }

  output->ZeroPadToByte();
}

size_t FastLosslessEncode(const unsigned char* rgba, size_t width,
                          size_t row_stride, size_t height, size_t num_threads,
                          unsigned char** output) {
  assert(width != 0);
  assert(height != 0);
  assert(row_stride >= 4 * width);

  BitWriter writer;

  size_t num_groups_x = (width + 255) / 256;
  size_t num_groups_y = (height + 255) / 256;
  size_t num_dc_groups_x = (width + 2047) / 2048;
  size_t num_dc_groups_y = (height + 2047) / 2048;

  size_t num_groups = num_groups_x == 1 && num_groups_y == 1
                          ? 1
                          : (2 + num_dc_groups_x * num_dc_groups_y +
                             num_groups_x * num_groups_y);

  std::vector<BitWriter> group_data(num_groups);

  PrepareDCGlobal(&group_data[0]);

  for (size_t g = 0; g < num_groups_y * num_groups_x; g++) {
    size_t xg = g % num_groups_x;
    size_t yg = g / num_groups_x;
    size_t group_id =
        num_groups == 0 ? 0 : (2 + num_dc_groups_x * num_dc_groups_y + g);
    WriteACSection(rgba, xg * 256, yg * 256,
                   std::min<size_t>(width - xg * 256, 256),
                   std::min<size_t>(height - yg * 256, 256), row_stride,
                   &group_data[group_id]);
  }

  AssembleFrame(width, height, group_data, &writer);

  *output = writer.data.release();
  return writer.bytes_written;
}
