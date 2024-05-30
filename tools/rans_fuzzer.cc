// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/fuzztest.h"
#include "lib/jxl/test_memory_manager.h"

namespace {

using ::jxl::ANSCode;
using ::jxl::ANSSymbolReader;
using ::jxl::BitReader;
using ::jxl::BitReaderScopedCloser;
using ::jxl::Bytes;
using ::jxl::Status;

int DoTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) return 0;
  size_t numContexts = data[0] * 256 * data[1] + 1;
  data += 2;
  size -= 2;

  std::vector<uint8_t> context_map;
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  Status ret = true;
  {
    BitReader br(Bytes(data, size));
    BitReaderScopedCloser br_closer(&br, &ret);
    ANSCode code;
    JXL_RETURN_IF_ERROR(DecodeHistograms(memory_manager, &br, numContexts,
                                         &code, &context_map));
    JXL_ASSIGN_OR_DIE(ANSSymbolReader ansreader,
                      ANSSymbolReader::Create(&code, &br));

    // Limit the maximum amount of reads to avoid (valid) infinite loops.
    const size_t maxreads = size * 8;
    size_t numreads = 0;
    int context = 0;
    while (jxl::DivCeil(br.TotalBitsConsumed(), jxl::kBitsPerByte) < size &&
           numreads <= maxreads) {
      int code = ansreader.ReadHybridUint(context, &br, context_map);
      context = code % numContexts;
      numreads++;
    }
  }

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
