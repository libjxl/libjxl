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

#include "lib/jxl/jpeg/dec_jpeg_state.h"

#include <brotli/decode.h>

#include "lib/jxl/jpeg/dec_jpeg_state_internal.h"

namespace jxl {
namespace jpeg {

DecState::DecState() : internal(new InternalState()) {}

DecState::DecState(DecState&&) = default;

DecState::~DecState() {}

MetadataState::~MetadataState() {
  if (brotli != nullptr) {
    BrotliDecoderDestroyInstance(brotli);
    brotli = nullptr;
  }
}

bool HasSection(const DecState* state, uint32_t tag) {
  return state->internal->section.tags_met & (1u << tag);
}

}  // namespace jpeg
}  // namespace jxl
