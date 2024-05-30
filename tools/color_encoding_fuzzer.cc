// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/color_encoding.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lib/extras/dec/color_description.h"
#include "lib/jxl/fuzztest.h"

namespace {

int DoTestOneInput(const uint8_t* data, size_t size) {
  std::string description(reinterpret_cast<const char*>(data), size);
  JxlColorEncoding c;
  (void)jxl::ParseDescription(description, &c);

  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return DoTestOneInput(data, size);
}

void TestOneInput(const std::vector<uint8_t>& data) {
  DoTestOneInput(data.data(), data.size());
}

FUZZ_TEST(CjxlFuzzTest, TestOneInput);
