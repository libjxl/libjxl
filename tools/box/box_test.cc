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

#include "tools/box/box.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "gtest/gtest.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"

TEST(BoxTest, BoxTest) {
  size_t test_size = 256;
  jxl::PaddedBytes exif(test_size);
  jxl::PaddedBytes xml0(test_size);
  jxl::PaddedBytes xml1(test_size);
  jxl::PaddedBytes jumb(test_size);
  jxl::PaddedBytes codestream(test_size);
  // Generate arbitrary data for the codestreams: the test is not testing
  // the contents of them but whether they are preserved in the container.
  uint8_t v = 0;
  for (size_t i = 0; i < test_size; ++i) {
    exif[i] = v++;
    xml0[i] = v++;
    xml1[i] = v++;
    jumb[i] = v++;
    codestream[i] = v++;
  }

  jpegxl::tools::JpegXlContainer container;
  container.exif = exif.data();
  container.exif_size = exif.size();
  container.xml.emplace_back(xml0.data(), xml0.size());
  container.xml.emplace_back(xml1.data(), xml1.size());
  container.xmlc.emplace_back(xml1.data(), xml1.size());
  container.jumb = jumb.data();
  container.jumb_size = jumb.size();
  container.codestream = codestream.data();
  container.codestream_size = codestream.size();

  jxl::PaddedBytes file;
  EXPECT_EQ(true,
            jpegxl::tools::EncodeJpegXlContainerOneShot(container, &file));

  jpegxl::tools::JpegXlContainer container2;
  EXPECT_EQ(true, jpegxl::tools::DecodeJpegXlContainerOneShot(
                      file.data(), file.size(), &container2));

  EXPECT_EQ(exif.size(), container2.exif_size);
  EXPECT_EQ(0, memcmp(exif.data(), container2.exif, container2.exif_size));
  EXPECT_EQ(2, container2.xml.size());
  if (container2.xml.size() == 2) {
    EXPECT_EQ(xml0.size(), container2.xml[0].second);
    EXPECT_EQ(0, memcmp(xml0.data(), container2.xml[0].first,
                        container2.xml[0].second));
    EXPECT_EQ(xml1.size(), container2.xml[1].second);
    EXPECT_EQ(0, memcmp(xml1.data(), container2.xml[1].first,
                        container2.xml[1].second));
  }
  EXPECT_EQ(1, container2.xmlc.size());
  if (container2.xmlc.size() == 1) {
    EXPECT_EQ(xml1.size(), container2.xmlc[0].second);
    EXPECT_EQ(0, memcmp(xml1.data(), container2.xmlc[0].first,
                        container2.xmlc[0].second));
  }
  EXPECT_EQ(jumb.size(), container2.jumb_size);
  EXPECT_EQ(0, memcmp(jumb.data(), container2.jumb, container2.jumb_size));
  EXPECT_EQ(codestream.size(), container2.codestream_size);
  EXPECT_EQ(0, memcmp(codestream.data(), container2.codestream,
                      container2.codestream_size));
}
