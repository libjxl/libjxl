// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstddef>
#include <cstdint>

#include "lib/extras/codec.h"
#include "lib/extras/size_constraints.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/fuzztest.h"
#include "tools/no_memory_manager.h"
#include "tools/thread_pool_internal.h"

namespace {

int DoTestOneInput(const uint8_t* data, size_t size) {
  jxl::CodecInOut io{jpegxl::tools::NoMemoryManager()};
  jxl::SizeConstraints constraints;
  constraints.dec_max_xsize = 1u << 16;
  constraints.dec_max_ysize = 1u << 16;
  constraints.dec_max_pixels = 1u << 22;
  jpegxl::tools::ThreadPoolInternal pool(0);

  (void)jxl::SetFromBytes(jxl::Bytes(data, size), &io, pool.get(),
                          &constraints);

  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return DoTestOneInput(data, size);
}

void TestOneInput(const std::vector<uint8_t>& data) {
  DoTestOneInput(data.data(), data.size());
}

FUZZ_TEST(SetFromBytesFuzzTest, TestOneInput);
