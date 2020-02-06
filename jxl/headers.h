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

#ifndef JXL_HEADERS_H_
#define JXL_HEADERS_H_

// Codestream headers, also stored in CodecInOut.

#include <stddef.h>
#include <stdint.h>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/status.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/field_encodings.h"

namespace jxl {

// Reserved by ISO/IEC 10918-1. LF causes files opened in text mode to be
// rejected because the marker changes to 0x0D instead. The 0xFF prefix also
// ensures there were no 7-bit transmission limitations.
static constexpr uint8_t kCodestreamMarker = 0x0A;

// Compact representation of image dimensions (best case: 9 bits) so decoders
// can preallocate early.
class SizeHeader {
 public:
  // All fields are valid after reading at most this many bits. WriteSizeHeader
  // verifies this matches Bundle::MaxBits(SizeHeader).
  static constexpr size_t kMaxBits = 78;

  SizeHeader();
  static const char* Name() { return "SizeHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    visitor->Bool(false, &small_);

    if (visitor->Conditional(small_)) {
      visitor->Bits(5, 0, &ysize_div8_minus_1_);
    }
    if (visitor->Conditional(!small_)) {
      // (Could still be small, but non-multiple of 8.)
      visitor->U32(Bits(9), Bits(13), Bits(18), Bits(30), 0, &ysize_minus_1_);
    }

    visitor->Bits(3, 0, &ratio_);
    if (visitor->Conditional(ratio_ == 0 && small_)) {
      visitor->Bits(5, 0, &xsize_div8_minus_1_);
    }
    if (visitor->Conditional(ratio_ == 0 && !small_)) {
      visitor->U32(Bits(9), Bits(13), Bits(18), Bits(30), 0, &xsize_minus_1_);
    }

    return true;
  }

  Status Set(size_t xsize, size_t ysize);

  size_t xsize() const;
  size_t ysize() const {
    return small_ ? ((ysize_div8_minus_1_ + 1) * 8) : ysize_minus_1_ + 1;
  }

 private:
  bool small_;  // xsize and ysize <= 256 and divisible by 8.

  uint32_t ysize_div8_minus_1_;
  uint32_t ysize_minus_1_;

  uint32_t ratio_;
  uint32_t xsize_div8_minus_1_;
  uint32_t xsize_minus_1_;
};

// (Similar to SizeHeader but different encoding because previews are smaller)
class PreviewHeader {
 public:
  PreviewHeader();
  static const char* Name() { return "PreviewHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    visitor->Bool(false, &div8_);

    if (visitor->Conditional(div8_)) {
      visitor->U32(Val(15), Val(31), Bits(5), BitsOffset(9, 32), 0,
                   &ysize_div8_minus_1_);
    }
    if (visitor->Conditional(!div8_)) {
      visitor->U32(Bits(6), BitsOffset(8, 64), BitsOffset(10, 320),
                   BitsOffset(12, 1344), 0, &ysize_minus_1_);
    }

    visitor->Bits(3, 0, &ratio_);
    if (visitor->Conditional(ratio_ == 0 && div8_)) {
      visitor->U32(Val(15), Val(31), Bits(5), BitsOffset(9, 32), 0,
                   &xsize_div8_minus_1_);
    }
    if (visitor->Conditional(ratio_ == 0 && !div8_)) {
      visitor->U32(Bits(6), BitsOffset(8, 64), BitsOffset(10, 320),
                   BitsOffset(12, 1344), 0, &xsize_minus_1_);
    }

    return true;
  }

  Status Set(size_t xsize, size_t ysize);

  size_t xsize() const;
  size_t ysize() const {
    return div8_ ? ((ysize_div8_minus_1_ + 1) * 8) : ysize_minus_1_ + 1;
  }

 private:
  bool div8_;  // xsize and ysize divisible by 8.

  uint32_t ysize_div8_minus_1_;
  uint32_t ysize_minus_1_;

  uint32_t ratio_;
  uint32_t xsize_div8_minus_1_;
  uint32_t xsize_minus_1_;
};

struct AnimationHeader {
  AnimationHeader();
  static const char* Name() { return "AnimationHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    visitor->Bool(false, &composite_still);
    if (visitor->Conditional(!composite_still)) {
      visitor->U32(Val(99), Val(999), Bits(6), Bits(18), 0,
                   &tps_numerator_minus_1);
      visitor->U32(Val(0), Val(1000), Bits(8), Bits(10), 0,
                   &tps_denominator_minus_1);

      visitor->U32(Val(0), Bits(3), Bits(16), Bits(32), 0, &num_loops);

      visitor->Bool(false, &have_timecodes);
    }
    return true;
  }

  bool composite_still;

  // Ticks per second (expressed as rational number to support NTSC)
  uint32_t tps_numerator_minus_1;
  uint32_t tps_denominator_minus_1;

  uint32_t num_loops;  // 0 means to repeat infinitely.

  bool have_timecodes;
};

Status ReadSizeHeader(BitReader* JXL_RESTRICT reader,
                      SizeHeader* JXL_RESTRICT size);
Status ReadPreviewHeader(BitReader* JXL_RESTRICT reader,
                         PreviewHeader* JXL_RESTRICT preview);
Status ReadAnimationHeader(BitReader* JXL_RESTRICT reader,
                           AnimationHeader* JXL_RESTRICT animation);

Status WriteSizeHeader(const SizeHeader& size, BitWriter* JXL_RESTRICT writer,
                       size_t layer, AuxOut* aux_out);
Status WritePreviewHeader(const PreviewHeader& preview,
                          BitWriter* JXL_RESTRICT writer, size_t layer,
                          AuxOut* aux_out);
Status WriteAnimationHeader(const AnimationHeader& animation,
                            BitWriter* JXL_RESTRICT writer, size_t layer,
                            AuxOut* aux_out);

}  // namespace jxl

#endif  // JXL_HEADERS_H_
