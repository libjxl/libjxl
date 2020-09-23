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
class SizeHeader : public Fields {
 public:
  // All fields are valid after reading at most this many bits. WriteSizeHeader
  // verifies this matches Bundle::MaxBits(SizeHeader).
  static constexpr size_t kMaxBits = 78;

  SizeHeader();
  const char* Name() const override { return "SizeHeader"; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  Status Set(size_t xsize, size_t ysize);

  size_t xsize() const;
  size_t ysize() const {
    return small_ ? ((ysize_div8_minus_1_ + 1) * 8) : ysize_;
  }

 private:
  bool small_;  // xsize and ysize <= 256 and divisible by 8.

  uint32_t ysize_div8_minus_1_;
  uint32_t ysize_;

  uint32_t ratio_;
  uint32_t xsize_div8_minus_1_;
  uint32_t xsize_;
};

// (Similar to SizeHeader but different encoding because previews are smaller)
class PreviewHeader : public Fields {
 public:
  PreviewHeader();
  const char* Name() const override { return "PreviewHeader"; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  Status Set(size_t xsize, size_t ysize);

  size_t xsize() const;
  size_t ysize() const { return div8_ ? (ysize_div8_ * 8) : ysize_; }

 private:
  bool div8_;  // xsize and ysize divisible by 8.

  uint32_t ysize_div8_;
  uint32_t ysize_;

  uint32_t ratio_;
  uint32_t xsize_div8_;
  uint32_t xsize_;
};

struct AnimationHeader : public Fields {
  AnimationHeader();
  const char* Name() const override { return "AnimationHeader"; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  bool composite_still;

  // Ticks per second (expressed as rational number to support NTSC)
  uint32_t tps_numerator;
  uint32_t tps_denominator;

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
