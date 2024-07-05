// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/fuzztest.h"
#include "tools/tracking_memory_manager.h"

namespace {

using ::jpegxl::tools::kGiB;
using ::jpegxl::tools::TrackingMemoryManager;
using ::jxl::ANSCode;
using ::jxl::ANSSymbolReader;
using ::jxl::BitReader;
using ::jxl::BitReaderScopedCloser;
using ::jxl::Bytes;
using ::jxl::Status;

void Check(bool ok) {
  if (!ok) {
    JXL_CRASH();
  }
}

Status Run(const uint8_t* data, size_t size, JxlMemoryManager* memory_manager,
           size_t num_contexts) {
  std::vector<uint8_t> context_map;
  Status ret = true;
  {
    BitReader br(Bytes(data, size));
    BitReaderScopedCloser br_closer(br, ret);
    ANSCode code;
    JXL_RETURN_IF_ERROR(DecodeHistograms(memory_manager, &br, num_contexts,
                                         &code, &context_map));
    JXL_ASSIGN_OR_RETURN(ANSSymbolReader ansreader,
                         ANSSymbolReader::Create(&code, &br));

    // Limit the maximum amount of reads to avoid (valid) infinite loops.
    const size_t maxreads = size * 8;
    size_t numreads = 0;
    int context = 0;
    while (jxl::DivCeil(br.TotalBitsConsumed(), jxl::kBitsPerByte) < size &&
           numreads <= maxreads) {
      int code = ansreader.ReadHybridUint(context, &br, context_map);
      context = code % num_contexts;
      numreads++;
    }
  }
  return true;
}

int DoTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) return 0;
  size_t numContexts = data[0] * 256 * data[1] + 1;
  data += 2;
  size -= 2;

  TrackingMemoryManager memory_manager{/* cap */ 1 * kGiB,
                                       /* total_cap */ 5 * kGiB};
  // It is OK to fail.
  (void)Run(data, size, memory_manager.get(), numContexts);
  Check(memory_manager.Reset());

  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return DoTestOneInput(data, size);
}

void TestOneInput(const std::vector<uint8_t>& data) {
  DoTestOneInput(data.data(), data.size());
}

FUZZ_TEST(RansFuzzTest, TestOneInput);
