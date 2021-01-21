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

#include "jxl/decode.h"

namespace jxl {

int TestOneInput(const uint8_t* data, size_t size) {
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
  bool have_basic_info = !JxlDecoderGetBasicInfo(dec, &info);

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

  JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, 0};
  JxlDecoderGetColorAsEncodedProfile(
      dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL, nullptr);
  size_t dec_profile_size;
  JxlDecoderGetICCProfileSize(dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                              &dec_profile_size);

  JxlDecoderDestroy(dec);
  return 0;
}

}  // namespace jxl

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return jxl::TestOneInput(data, size);
}
