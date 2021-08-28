// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
  container.addBlob("Exif", exif.data(), exif.size(), true);
  container.addBlob("xml ", xml0.data(), xml0.size(), true);
  container.addBlob("xml ", xml1.data(), xml1.size(), false);
  container.addBlob("jumb", jumb.data(), jumb.size(), false);
  container.codestream = codestream.data();
  container.codestream_size = codestream.size();

  jxl::PaddedBytes file;
  EXPECT_EQ(true,
            jpegxl::tools::EncodeJpegXlContainerOneShot(container, &file));

  jpegxl::tools::JpegXlContainer container2;
  EXPECT_EQ(true, jpegxl::tools::DecodeJpegXlContainerOneShot(
                      file.data(), file.size(), &container2));

  const jpegxl::tools::BrobBlob* b = container2.getBlob("Exif");
  EXPECT_EQ(true, b != nullptr);
  EXPECT_EQ(exif.size(), b->udata_size);
  EXPECT_EQ(0, memcmp(exif.data(), b->udata, b->udata_size));
  b = container2.getBlob("xml ");
  EXPECT_EQ(true, b != nullptr);
  EXPECT_EQ(xml0.size(), b->udata_size);
  EXPECT_EQ(0, memcmp(xml0.data(), b->udata, b->udata_size));
  b = container2.getBlob("xml ", 1);
  EXPECT_EQ(true, b != nullptr);
  EXPECT_EQ(xml1.size(), b->udata_size);
  EXPECT_EQ(0, memcmp(xml1.data(), b->udata, b->udata_size));
  b = container2.getBlob("xml ", 2);
  EXPECT_EQ(true, b == nullptr);
  b = container2.getBlob("jumb");
  EXPECT_EQ(true, b != nullptr);
  EXPECT_EQ(jumb.size(), b->udata_size);
  EXPECT_EQ(0, memcmp(jumb.data(), b->udata, b->udata_size));
  EXPECT_EQ(codestream.size(), container2.codestream_size);
  EXPECT_EQ(0, memcmp(codestream.data(), container2.codestream,
                      container2.codestream_size));
}
