// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stddef.h>
#include <stdint.h>

#include "lib/extras/codec.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"

namespace {

int TestOneInput(const uint8_t* data, size_t size) {
  jxl::CodecInOut io;
  io.constraints.dec_max_xsize = 1u << 16;
  io.constraints.dec_max_ysize = 1u << 16;
  io.constraints.dec_max_pixels = 1u << 22;
  jxl::ThreadPoolInternal pool(0);

  (void)jxl::SetFromBytes(jxl::Span<const uint8_t>(data, size), &io, &pool);

  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return TestOneInput(data, size);
}
