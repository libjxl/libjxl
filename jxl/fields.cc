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

#include "jxl/fields.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>

#include "jxl/base/bits.h"

namespace jxl {

size_t U32Coder::MaxEncodedBits(const U32Enc enc) {
  size_t extra_bits = 0;
  for (uint32_t selector = 0; selector < 4; ++selector) {
    const U32Distr d = enc.GetDistr(selector);
    if (d.IsDirect()) {
      continue;
    } else {
      extra_bits = std::max<size_t>(extra_bits, d.ExtraBits());
    }
  }
  return 2 + extra_bits;
}

Status U32Coder::CanEncode(const U32Enc enc, const uint32_t value,
                           size_t* JXL_RESTRICT encoded_bits) {
  uint32_t selector;
  size_t total_bits;
  const Status ok = ChooseSelector(enc, value, &selector, &total_bits);
  *encoded_bits = ok ? total_bits : 0;
  return ok;
}

HWY_ATTR uint32_t U32Coder::Read(const U32Enc enc,
                                 BitReader* JXL_RESTRICT reader) {
  const uint32_t selector = reader->ReadFixedBits<2>();
  const U32Distr d = enc.GetDistr(selector);
  if (d.IsDirect()) {
    return d.Direct();
  } else {
    return reader->ReadBits(d.ExtraBits()) + d.Offset();
  }
}

// Returns false if the value is too large to encode.
Status U32Coder::Write(const U32Enc enc, const uint32_t value,
                       BitWriter* JXL_RESTRICT writer) {
  uint32_t selector;
  size_t total_bits;
  JXL_RETURN_IF_ERROR(ChooseSelector(enc, value, &selector, &total_bits));

  writer->Write(2, selector);

  const U32Distr d = enc.GetDistr(selector);
  if (!d.IsDirect()) {  // Nothing more to write for direct encoding
    const uint32_t offset = d.Offset();
    JXL_ASSERT(value >= offset);
    writer->Write(total_bits - 2, value - offset);
  }

  return true;
}

Status U32Coder::ChooseSelector(const U32Enc enc, const uint32_t value,
                                uint32_t* JXL_RESTRICT selector,
                                size_t* JXL_RESTRICT total_bits) {
#if JXL_ENABLE_ASSERT
  const size_t bits_required = 32 - NumZeroBitsAboveMSB(value);
#endif  // JXL_ENABLE_ASSERT
  JXL_ASSERT(bits_required <= 32);

  *selector = 0;
  *total_bits = 0;

  // It is difficult to verify whether Dist32Byte are sorted, so check all
  // selectors and keep the one with the fewest total_bits.
  *total_bits = 64;  // more than any valid encoding
  for (uint32_t s = 0; s < 4; ++s) {
    const U32Distr d = enc.GetDistr(s);
    if (d.IsDirect()) {
      if (d.Direct() == value) {
        *selector = s;
        *total_bits = 2;
        return true;  // Done, direct is always the best possible.
      }
      continue;
    }
    const size_t extra_bits = d.ExtraBits();
    const uint32_t offset = d.Offset();
    if (value < offset || value >= offset + (1ULL << extra_bits)) continue;

    // Better than prior encoding, remember it:
    if (2 + extra_bits < *total_bits) {
      *selector = s;
      *total_bits = 2 + extra_bits;
    }
  }

  if (*total_bits == 64) {
    return JXL_FAILURE("No feasible selector for %u", value);
  }

  return true;
}

HWY_ATTR uint64_t U64Coder::Read(BitReader* JXL_RESTRICT reader) {
  uint64_t selector = reader->ReadFixedBits<2>();
  if (selector == 0) {
    return 0;
  }
  if (selector == 1) {
    return 1 + reader->ReadFixedBits<4>();
  }
  if (selector == 2) {
    return 17 + reader->ReadFixedBits<8>();
  }

  // selector 3, varint, groups have first 12, then 8, and last 4 bits.
  uint64_t result = reader->ReadFixedBits<12>();

  uint64_t shift = 12;
  while (reader->ReadFixedBits<1>()) {
    if (shift == 60) {
      result |= static_cast<uint64_t>(reader->ReadFixedBits<4>()) << shift;
      break;
    }
    result |= static_cast<uint64_t>(reader->ReadFixedBits<8>()) << shift;
    shift += 8;
  }

  return result;
}

// Returns false if the value is too large to encode.
Status U64Coder::Write(uint64_t value, BitWriter* JXL_RESTRICT writer) {
  if (value == 0) {
    // Selector: use 0 bits, value 0
    writer->Write(2, 0);
  } else if (value <= 16) {
    // Selector: use 4 bits, value 1..16
    writer->Write(2, 1);
    writer->Write(4, value - 1);
  } else if (value <= 272) {
    // Selector: use 8 bits, value 17..272
    writer->Write(2, 2);
    writer->Write(8, value - 17);
  } else {
    // Selector: varint, first a 12-bit group, after that per 8-bit group.
    writer->Write(2, 3);
    writer->Write(12, value & 4095);
    value >>= 12;
    int shift = 12;
    while (value > 0 && shift < 60) {
      // Indicate varint not done
      writer->Write(1, 1);
      writer->Write(8, value & 255);
      value >>= 8;
      shift += 8;
    }
    if (value > 0) {
      // This only could happen if shift == N - 4.
      writer->Write(1, 1);
      writer->Write(4, value & 15);
      // Implicitly closed sequence, no extra stop bit is required.
    } else {
      // Indicate end of varint
      writer->Write(1, 0);
    }
  }

  return true;
}

// Can always encode, but useful because it also returns bit size.
Status U64Coder::CanEncode(uint64_t value, size_t* JXL_RESTRICT encoded_bits) {
  if (value == 0) {
    *encoded_bits = 2;  // 2 selector bits
  } else if (value <= 16) {
    *encoded_bits = 2 + 4;  // 2 selector bits + 4 payload bits
  } else if (value <= 272) {
    *encoded_bits = 2 + 8;  // 2 selector bits + 8 payload bits
  } else {
    *encoded_bits = 2 + 12;  // 2 selector bits + 12 payload bits
    value >>= 12;
    int shift = 12;
    while (value > 0 && shift < 60) {
      *encoded_bits += 1 + 8;  // 1 continuation bit + 8 payload bits
      value >>= 8;
      shift += 8;
    }
    if (value > 0) {
      // This only could happen if shift == N - 4.
      *encoded_bits += 1 + 4;  // 1 continuation bit + 4 payload bits
    } else {
      *encoded_bits += 1;  // 1 stop bit
    }
  }

  return true;
}

HWY_ATTR Status F16Coder::Read(BitReader* JXL_RESTRICT reader,
                               float* JXL_RESTRICT value) {
  const uint32_t bits16 = reader->ReadFixedBits<16>();
  const uint32_t sign = bits16 >> 15;
  const uint32_t biased_exp = (bits16 >> 10) & 0x1F;
  const uint32_t mantissa = bits16 & 0x3FF;

  if (JXL_UNLIKELY(biased_exp == 31)) {
    return JXL_FAILURE("F16 infinity or NaN are not supported");
  }

  // Subnormal or zero
  if (JXL_UNLIKELY(biased_exp == 0)) {
    *value = (1.0f / 16384) * (mantissa * (1.0f / 1024));
    return true;
  }

  // Normalized: convert the representation directly (faster than ldexp/tables).
  const uint32_t biased_exp32 = biased_exp + (127 - 15);
  const uint32_t mantissa32 = mantissa << (23 - 10);
  const uint32_t bits32 = (sign << 31) | (biased_exp32 << 23) | mantissa32;
  memcpy(value, &bits32, sizeof(bits32));
  return true;
}

Status F16Coder::Write(float value, BitWriter* JXL_RESTRICT writer) {
  uint32_t bits32;
  memcpy(&bits32, &value, sizeof(bits32));
  const uint32_t sign = bits32 >> 31;
  const uint32_t biased_exp32 = (bits32 >> 23) & 0xFF;
  const uint32_t mantissa32 = bits32 & 0x7FFFFF;

  const int32_t exp = static_cast<int32_t>(biased_exp32) - 127;
  if (JXL_UNLIKELY(exp > 15)) {
    return JXL_FAILURE("Too big to encode, CanEncode should return false");
  }

  // Tiny or zero => zero.
  if (exp < -24) {
    writer->Write(16, 0);
    return true;
  }

  uint32_t biased_exp16, mantissa16;

  // exp = [-24, -15] => subnormal
  if (JXL_UNLIKELY(exp < -14)) {
    biased_exp16 = 0;
    const uint32_t sub_exp = static_cast<uint32_t>(-14 - exp);
    JXL_ASSERT(1 <= sub_exp && sub_exp < 11);
    mantissa16 = (1 << (10 - sub_exp)) + (mantissa32 >> (13 + sub_exp));
  } else {
    // exp = [-14, 15]
    biased_exp16 = static_cast<uint32_t>(exp + 15);
    JXL_ASSERT(1 <= biased_exp16 && biased_exp16 < 31);
    mantissa16 = mantissa32 >> 13;
  }

  JXL_ASSERT(mantissa16 < 1024);
  const uint32_t bits16 = (sign << 15) | (biased_exp16 << 10) | mantissa16;
  JXL_ASSERT(bits16 < 0x10000);
  writer->Write(16, bits16);
  return true;
}

Status F16Coder::CanEncode(float value, size_t* JXL_RESTRICT encoded_bits) {
  *encoded_bits = MaxEncodedBits();
  if (std::isnan(value) || std::isinf(value)) {
    return JXL_FAILURE("Should not attempt to store NaN and infinity");
  }
  return std::abs(value) <= 65504.0f;
}

}  // namespace jxl
