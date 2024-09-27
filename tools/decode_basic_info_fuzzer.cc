// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/decode.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/fuzztest.h"
#include "tools/tracking_memory_manager.h"

namespace {

using ::jpegxl::tools::kGiB;
using ::jpegxl::tools::TrackingMemoryManager;

void Check(bool ok) {
  if (!ok) {
    JXL_CRASH();
  }
}

int DoTestOneInput(const uint8_t* data, size_t size) {
  JxlDecoderStatus status;
  TrackingMemoryManager memory_manager{/* cap */ 1 * kGiB,
                                       /* total_cap */ 5 * kGiB};
  JxlDecoder* dec = JxlDecoderCreate(memory_manager.get());
  const auto finish = [&]() -> int {
    JxlDecoderDestroy(dec);
    Check(memory_manager.Reset());
    return 0;
  };

  JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING);
  JxlDecoderSetInput(dec, data, size);

  status = JxlDecoderProcessInput(dec);

  if (status != JXL_DEC_BASIC_INFO) return finish();

  JxlBasicInfo info;
  status = JxlDecoderGetBasicInfo(dec, &info);
  bool have_basic_info = (status == JXL_DEC_SUCCESS);

  if (have_basic_info) {
    if (info.alpha_bits != 0) {
      for (int i = 0; i < info.num_extra_channels; ++i) {
        JxlExtraChannelInfo extra;
        JxlDecoderGetExtraChannelInfo(dec, 0, &extra);
      }
    }
  }
  status = JxlDecoderProcessInput(dec);

  if (status != JXL_DEC_COLOR_ENCODING) return finish();

  JxlDecoderGetColorAsEncodedProfile(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                     nullptr);
  size_t dec_profile_size;
  JxlDecoderGetICCProfileSize(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                              &dec_profile_size);

  return finish();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return DoTestOneInput(data, size);
}

void TestOneInput(const std::vector<uint8_t>& data) {
  DoTestOneInput(data.data(), data.size());
}

FUZZ_TEST(DecodeBasiInfoFuzzTest, TestOneInput);
