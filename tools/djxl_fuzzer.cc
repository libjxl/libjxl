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

#include <stdint.h>
#include <stdio.h>

#include <hwy/runtime_dispatch.h>
#include <hwy/static_targets.h>

#include "jxl/aux_out.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/span.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/codec_in_out.h"
#include "jxl/dec_file.h"
#include "jxl/dec_params.h"

namespace jxl {

int TestOneInput(const uint8_t* data, size_t size) {
  // TODO(b/65240090): Remove
  const int bits = hwy::TargetBitfield().Bits();
  if ((bits & HWY_STATIC_TARGETS) != HWY_STATIC_TARGETS) {
    static bool warned = false;
    if (!warned) {
      fprintf(stderr, "CPU doesn't support all enabled targets; exiting.\n");
      warned = true;
    }
    return 0;
  }

  DecompressParams params;

  CodecInOut io;
  io.dec_max_xsize = 1u << 16;
  io.dec_max_ysize = 1u << 16;
  io.dec_max_pixels = 1 << 21;

  ThreadPoolInternal pool(2);
  AuxOut aux_out;
  (void)DecodeFile(params, Span<const uint8_t>(data, size), &io, &aux_out,
                   &pool);
  return 0;
}

}  // namespace jxl

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return jxl::TestOneInput(data, size);
}
