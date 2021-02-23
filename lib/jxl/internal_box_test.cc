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

#include "gtest/gtest.h"
#include "lib/jxl/box.h"

namespace {
jxl::Span<const uint8_t> MakeSpan(const char* str) {
  return jxl::Span<const uint8_t>((const uint8_t*)(str), strlen(str));
}
}  // namespace

TEST(BoxTest, RoundtripTest) {
  jxl::JxlBox reconstruction_box = {};
  memcpy(reconstruction_box.type, "jbrd", 4);
  reconstruction_box.data_size_given = true;
  reconstruction_box.data = MakeSpan("reconstruction_box");
  jxl::JxlBox box_with_extended_type = {};
  memcpy(box_with_extended_type.type, "uuid", 4);
  memcpy(box_with_extended_type.extended_type, "012345679abcdef", 16);
  box_with_extended_type.data = MakeSpan("extended_type_box");
  jxl::JxlBox codestream_box = {};
  memcpy(codestream_box.type, "jxlc", 4);
  codestream_box.data_size_given = false;
  codestream_box.data = MakeSpan("codestream_box");
  jxl::JxlContainer container = {};
  container.boxes.emplace_back(reconstruction_box);
  container.boxes.emplace_back(codestream_box);
  std::vector<uint8_t> encoded;
  container.Encode(&encoded);

  EXPECT_EQ(80, encoded.size());

  jxl::JxlContainer decoded_container = {};
  jxl::Span<uint8_t> encoded_span =
      jxl::Span<uint8_t>(encoded.data(), encoded.size());
  EXPECT_TRUE(decoded_container.Decode(&encoded_span));
  EXPECT_EQ(0, encoded_span.size());
  EXPECT_EQ(decoded_container.boxes.size(), container.boxes.size());
  for (size_t i = 0; i < decoded_container.boxes.size(); i++) {
    EXPECT_EQ(
        0, memcmp(decoded_container.boxes[i].type, container.boxes[i].type, 4));
    EXPECT_EQ(0, memcmp(decoded_container.boxes[i].extended_type,
                        container.boxes[i].extended_type, 16));
    EXPECT_EQ(decoded_container.boxes[i].data_size_given,
              container.boxes[i].data_size_given);
    EXPECT_EQ(decoded_container.boxes[i].data.size(),
              container.boxes[i].data.size());
    EXPECT_EQ(0, memcmp(decoded_container.boxes[i].data.data(),
                        container.boxes[i].data.data(),
                        container.boxes[i].data.size()));
  }
}
