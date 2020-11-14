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

#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/loop_filter.h"

namespace jxl {

int TestOneInput(const uint8_t* data, size_t size) {
  // First byte controls which header to parse.
  if (size == 0) return 0;
  BitReader reader(Span<const uint8_t>(data + 1, size - 1));
  switch (data[0]) {
    case 0: {
      SizeHeader size_header;
      (void)ReadSizeHeader(&reader, &size_header);
      break;
    }

    case 1: {
      ImageMetadata metadata;
      (void)ReadImageMetadata(&reader, &metadata);
      break;
    }

    case 2: {
      CodecMetadata metadata;
      FrameHeader frame(&metadata);
      (void)ReadFrameHeader(&reader, &frame);
      break;
    }

    default: {
      CustomTransformData transform_data;
      transform_data.nonserialized_xyb_encoded = true;
      (void)Bundle::Read(&reader, &transform_data);
      break;
    }
  }
  (void)reader.Close();

  return 0;
}

}  // namespace jxl

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return jxl::TestOneInput(data, size);
}
