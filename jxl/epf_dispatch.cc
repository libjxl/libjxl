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

#include "jxl/epf_dispatch.h"

#include <hwy/runtime_dispatch.h>

#include "jxl/epf.h"

namespace jxl {

void DoAdaptiveReconstruction(Image3F* JXL_RESTRICT out,
                              PassesDecoderState* JXL_RESTRICT decoder_state,
                              ThreadPool* pool) {
  const hwy::Target target = hwy::TargetBitfield().Best();
  Dispatch(target, AdaptiveReconstruction(), out, decoder_state, pool);
}

}  // namespace jxl
