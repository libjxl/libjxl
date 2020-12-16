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

#ifndef LIB_JXL_JPEG_DEC_JPEG_STATE_H_
#define LIB_JXL_JPEG_DEC_JPEG_STATE_H_

#include <array>
#include <memory>
#include <vector>

#include "lib/jxl/dec_ans.h"
#include "lib/jxl/jpeg/brunsli_status.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {

typedef std::array<int32_t, kDCTBlockSize> BlockI32;

struct DecComponentMeta {
  size_t context_offset;
  int32_t h_samp;
  int32_t v_samp;
  int32_t context_bits;
  int32_t ac_stride;
  int32_t b_stride;
  int32_t width_in_blocks;
  int32_t height_in_blocks;
  coeff_t* ac_coeffs;
  // TODO(eustas): investigate bit fields.
  uint8_t* block_state;
  BlockI32 quant;
};

enum struct Stage {
  SIGNATURE = 0,
  HEADER,
  FALLBACK,
  SECTION,
  SECTION_BODY,
  DONE,
  ERROR
};

enum struct SerializationStatus {
  NEEDS_MORE_INPUT,
  NEEDS_MORE_OUTPUT,
  ERROR,
  DONE
};

struct InternalState;

class DecState {
 public:
  DecState();
  DecState(DecState&&);
  ~DecState();

  // Public workflow knobs.
  Stage stage = Stage::SIGNATURE;
  // NB: this |tags_met| is not updated by decoder.
  uint32_t tags_met = 0;
  uint32_t skip_tags = 0;

  // Public input knobs.
  const uint8_t* data = nullptr;
  size_t len = 0;
  size_t pos = 0;

  // "JPEGDecodingState" view.
  bool use_legacy_context_model = false;

  bool is_storage_allocated = false;
  std::vector<DecComponentMeta> meta;

  // Private state parts.
  std::unique_ptr<InternalState> internal;
};

// Use in "headerless" mode, after jpg is filled, but before decoding.
bool UpdateSubsamplingDerivatives(JPEGData* jpg);

// Use in "headerless" mode, after UpdateSubsamplingDerivatives.
void PrepareMeta(const JPEGData* jpg, DecState* state);

bool HasSection(const DecState* state, uint32_t tag);

// Core decoding loop.
BrunsliStatus ProcessJpeg(DecState* state, JPEGData* jpg);

// Core serialization loop.
SerializationStatus SerializeJpeg(DecState* state, const JPEGData& jpg,
                                  size_t* available_out, uint8_t** next_out);

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_DEC_JPEG_STATE_H_
