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

#include "lib/jxl/toc.h"

#include <stdint.h>

#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"

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

Status ReadGroupOffsets(size_t toc_entries, BitReader* JXL_RESTRICT reader,
                        std::vector<uint64_t>* JXL_RESTRICT offsets,
                        std::vector<uint32_t>* JXL_RESTRICT sizes,
                        uint64_t* total_size) {
  JXL_DASSERT(offsets != nullptr && sizes != nullptr);
  std::vector<coeff_order_t> permutation;
  if (reader->ReadFixedBits<1>() == 1 && toc_entries > 0) {
    // Skip permutation description if the toc_entries is 0.
    permutation.resize(toc_entries);
    JXL_RETURN_IF_ERROR(
        DecodePermutation(/*skip=*/0, toc_entries, permutation.data(), reader));
  }

  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  sizes->clear();
  sizes->reserve(toc_entries);
  for (size_t i = 0; i < toc_entries; ++i) {
    sizes->push_back(U32Coder::Read(kDist, reader));
  }
  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  // Prefix sum starting with 0 and ending with the offset of the last group
  offsets->clear();
  offsets->reserve(toc_entries);
  uint64_t offset = 0;
  for (size_t i = 0; i < toc_entries; ++i) {
    if (offset + (*sizes)[i] < offset) {
      return JXL_FAILURE("group offset overflow");
    }
    offsets->push_back(offset);
    offset += (*sizes)[i];
  }
  if (total_size) {
    *total_size = offset;
  }

  if (!permutation.empty()) {
    std::vector<uint64_t> permuted_offsets;
    std::vector<uint32_t> permuted_sizes;
    permuted_offsets.reserve(toc_entries);
    permuted_sizes.reserve(toc_entries);
    for (coeff_order_t index : permutation) {
      permuted_offsets.push_back((*offsets)[index]);
      permuted_sizes.push_back((*sizes)[index]);
    }
    std::swap(*offsets, permuted_offsets);
    std::swap(*sizes, permuted_sizes);
  }

  return true;
}

Status WriteGroupOffsets(const std::vector<BitWriter>& group_codes,
                         const std::vector<coeff_order_t>* permutation,
                         BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  BitWriter::Allotment allotment(writer, MaxBits(group_codes.size()));
  if (permutation && !group_codes.empty()) {
    // Don't write a permutation at all for an empty group_codes.
    writer->Write(1, 1);  // permutation
    JXL_DASSERT(permutation->size() == group_codes.size());
    EncodePermutation(permutation->data(), /*skip=*/0, permutation->size(),
                      writer, /* layer= */ 0, aux_out);

  } else {
    writer->Write(1, 0);  // no permutation
  }
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

}  // namespace jxl
