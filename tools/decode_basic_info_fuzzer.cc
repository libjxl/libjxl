// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/decode.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/fuzztest.h"

namespace {

int DoTestOneInput(const uint8_t* data, size_t size) {
  JxlDecoderStatus status;
  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING);
  JxlDecoderSetInput(dec, data, size);

  status = JxlDecoderProcessInput(dec);

  if (status != JXL_DEC_BASIC_INFO) {
    JxlDecoderDestroy(dec);
    return 0;
  }

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

  if (status != JXL_DEC_COLOR_ENCODING) {
    JxlDecoderDestroy(dec);
    return 0;
  }

  JxlDecoderGetColorAsEncodedProfile(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                     nullptr);
  size_t dec_profile_size;
  JxlDecoderGetICCProfileSize(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                              &dec_profile_size);

  JxlDecoderDestroy(dec);
  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return DoTestOneInput(data, size);
}

void TestOneInput(const std::vector<uint8_t>& data) {
  DoTestOneInput(data.data(), data.size());
}

FUZZ_TEST(DecodeBasiInfoFuzzTest, TestOneInput);
