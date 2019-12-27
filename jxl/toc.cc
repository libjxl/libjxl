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

#include "jxl/toc.h"

#include <stdint.h>

#include "jxl/aux_out_fwd.h"
#include "jxl/coeff_order.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/common.h"
#include "jxl/field_encodings.h"
#include "jxl/fields.h"

namespace jxl {
namespace {
// (2+bits) = 2,3,4 bytes so encoders can patch TOC after encoding.
// 30 is sufficient for 4K channels of uncompressed 16-bit samples.
constexpr U32Enc kDist(Bits(10), BitsOffset(14, 1024), BitsOffset(22, 17408),
                       BitsOffset(30, 4211712));

static size_t MaxBits(const size_t num_sizes) {
  const size_t entry_bits = U32Coder::MaxEncodedBits(kDist) * num_sizes;
  // permutation bit (not its tokens!), padding, entries, padding.
  return 1 + kBitsPerByte + entry_bits + kBitsPerByte;
}

}  // namespace

HWY_ATTR Status ReadGroupOffsets(size_t toc_entries,
                                 BitReader* JXL_RESTRICT reader,
                                 std::vector<uint64_t>* JXL_RESTRICT offsets) {
  std::vector<coeff_order_t> permutation;
  if (reader->ReadFixedBits<1>() == 1) {
    permutation.resize(toc_entries);
    JXL_RETURN_IF_ERROR(
        DecodePermutation(/*skip=*/0, toc_entries, permutation.data(), reader));
  }

  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  std::vector<uint32_t> sizes;
  sizes.reserve(toc_entries);
  for (size_t i = 0; i < toc_entries; ++i) {
    sizes.push_back(U32Coder::Read(kDist, reader));
  }
  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  // Prefix sum starting with 0 and ending with total_size
  offsets->reserve(toc_entries + 1);
  offsets->push_back(0);
  uint64_t offset = 0;
  for (size_t i = 0; i < toc_entries; ++i) {
    offset += sizes[i];
    offsets->push_back(offset);
  }

  if (!permutation.empty()) {
    std::vector<uint64_t> permuted_offsets;
    permuted_offsets.reserve(offsets->size());
    for (coeff_order_t index : permutation) {
      permuted_offsets.push_back((*offsets)[index]);
    }
    std::swap(*offsets, permuted_offsets);
  }

  return true;
}

Status WriteGroupOffsets(const std::vector<BitWriter>& group_codes,
                         BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  BitWriter::Allotment allotment(writer, MaxBits(group_codes.size()));
  writer->Write(1, 0);  // no permutation
  // TODO(janwas): WriteTokens(permutation), already has an allotment.
  writer->ZeroPadToByte();  // before TOC entries

  for (size_t i = 0; i < group_codes.size(); i++) {
    JXL_ASSERT(group_codes[i].BitsWritten() % kBitsPerByte == 0);
    const size_t group_size = group_codes[i].BitsWritten() / kBitsPerByte;
    JXL_RETURN_IF_ERROR(U32Coder::Write(kDist, group_size, writer));
  }
  writer->ZeroPadToByte();  // before first group
  ReclaimAndCharge(writer, &allotment, kLayerTOC, aux_out);
  return true;
}

Status WriteResponsiveOffsets(const std::vector<int>& offsets,
                              BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  BitWriter::Allotment allotment(writer, MaxBits(offsets.size()));
  writer->Write(1, 0);      // no permutation
  writer->ZeroPadToByte();  // before TOC entries
  for (size_t i = 0; i < offsets.size(); i++) {
    JXL_RETURN_IF_ERROR(U32Coder::Write(kDist, offsets[i], writer));
  }
  writer->ZeroPadToByte();  // before first group
  ReclaimAndCharge(writer, &allotment, kLayerTOC, aux_out);
  return true;
}

}  // namespace jxl
