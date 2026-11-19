// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_EXTRAS_DEC_SAMPLE_RANGE_H_
#define LIB_EXTRAS_DEC_SAMPLE_RANGE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {

// Integer sample formats that declare a bit depth smaller than their storage
// type (e.g. 12-bit samples in a uint16 raster) must have every sample in
// [0, 2^bits_per_sample - 1]. The bit depth is copied into the image metadata
// and nothing downstream enforces the range, so a sample with a bit set at or
// above `bits_per_sample` ends up in a codestream that promises a smaller
// depth; decoders that use 16-bit buffers when the header says they are
// sufficient then read such samples as wrapped negative values.
//
// The declared maximum is 2^bits_per_sample - 1, so a sample is out of range
// iff its most significant byte has a bit set at or above the depth. That
// byte is the first byte of each sample for big-endian rasters and the second
// byte for little-endian rasters; rasters of 8 bits or less per sample have
// one byte per sample and no endianness. `data` holds `size` bytes of samples
// and must start at a sample boundary. Returns true when every sample fits.
inline Status VerifySampleRange(const uint8_t* data, size_t size,
                                size_t bits_per_sample, bool big_endian) {
  // Every value of the storage type is a valid sample.
  if (bits_per_sample == 8 || bits_per_sample == 16) return true;
  // OR the input 8 bytes at a time: byte k of `acc` is then the OR of all
  // bytes at offsets equal to k modulo 8.
  uint64_t acc = 0;
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    uint64_t v;
    memcpy(&v, data + i, 8);
    acc |= v;
  }
  uint8_t acc_bytes[8];
  memcpy(acc_bytes, &acc, 8);
  for (; i < size; ++i) acc_bytes[i % 8] |= data[i];
  const size_t step = bits_per_sample > 8 ? 2 : 1;
  const size_t first =
      (bits_per_sample > 8 && !big_endian) ? 1 : 0;  // most significant byte
  const size_t shift =
      bits_per_sample > 8 ? bits_per_sample - 8 : bits_per_sample;
  uint8_t bits = 0;
  for (size_t k = first; k < 8; k += step) bits |= acc_bytes[k];
  if (bits >> shift) {
    return JXL_FAILURE("sample value exceeds the declared bit depth");
  }
  return true;
}

}  // namespace extras
}  // namespace jxl

#endif  // LIB_EXTRAS_DEC_SAMPLE_RANGE_H_
