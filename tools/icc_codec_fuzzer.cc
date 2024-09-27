// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/enc_icc_codec.h"
#include "tools/tracking_memory_manager.h"

#ifdef JXL_ICC_FUZZER_SLOW_TEST
#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_bit_reader.h"
#endif

#include "lib/jxl/base/status.h"
#include "lib/jxl/fuzztest.h"
#include "lib/jxl/padded_bytes.h"

namespace jxl {
Status PredictICC(const uint8_t* icc, size_t size, PaddedBytes* result);
Status UnpredictICC(const uint8_t* enc, size_t size, PaddedBytes* result);
}  // namespace jxl

namespace {

using ::jpegxl::tools::kGiB;
using ::jpegxl::tools::TrackingMemoryManager;
using ::jxl::PaddedBytes;

#ifdef JXL_ICC_FUZZER_SLOW_TEST
using ::jxl::BitReader;
using ::jxl::Span;
#endif

void Check(bool ok) {
  if (!ok) {
    JXL_CRASH();
  }
}

int DoTestOneInput(const uint8_t* data, size_t size) {
#if defined(JXL_ICC_FUZZER_ONLY_WRITE)
  bool read = false;
#elif defined(JXL_ICC_FUZZER_ONLY_READ)
  bool read = true;
#else
  // Decide whether to test the reader or the writer (both use parsing)
  if (!size) return 0;
  bool read = data[0] == 0;
  data++;
  size--;
#endif
  TrackingMemoryManager memory_manager{/* cap */ 1 * kGiB,
                                       /* total_cap */ 5 * kGiB};

#ifdef JXL_ICC_FUZZER_SLOW_TEST
  // Including JPEG XL LZ77 and ANS compression. These are already fuzzed
  // separately, so it is better to disable JXL_ICC_FUZZER_SLOW_TEST to focus on
  // the ICC parsing.
  if (read) {
    // Reading parses the compressed format.
    BitReader br(Bytes(data, size));
    std::vector<uint8_t> result;
    (void)jxl::test::ReadICC(&br, &result);
    (void)br.Close();
  } else {
    // Writing parses the original ICC profile.
    PaddedBytes icc{memory_manager.get()};
    icc.assign(data, data + size);
    BitWriter writer{memory_manager.get()};
    // Writing should support any random bytestream so must succeed, make
    // fuzzer fail if not.
    Check(jxl::WriteICC(icc, &writer, jxl::LayerType::Header, nullptr));
  }
#else  // JXL_ICC_FUZZER_SLOW_TEST
  if (read) {
    // Reading (unpredicting) parses the compressed format.
    PaddedBytes result{memory_manager.get()};
    (void)jxl::UnpredictICC(data, size, &result);
  } else {
    // Writing (predicting) parses the original ICC profile.
    PaddedBytes result{memory_manager.get()};
    // Writing should support any random bytestream so must succeed, make
    // fuzzer fail if not.
    Check(jxl::PredictICC(data, size, &result));
    PaddedBytes reconstructed{memory_manager.get()};
    Check(jxl::UnpredictICC(result.data(), result.size(), &reconstructed));
    Check(reconstructed.size() == size);
    Check(memcmp(data, reconstructed.data(), size) == 0);
  }
#endif  // JXL_ICC_FUZZER_SLOW_TEST

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

FUZZ_TEST(IccCodecFuzzTest, TestOneInput);
