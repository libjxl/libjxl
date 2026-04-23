// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/benchmark/benchmark_file_io.h"

#include "lib/jxl/testing.h"

namespace jpegxl {
namespace tools {
namespace {

TEST(BenchmarkFileIoTest, FileDirName) {
  EXPECT_EQ(FileDirName("file.png"), ".");
  EXPECT_EQ(FileDirName("dir/file.png"), "dir");
  EXPECT_EQ(FileDirName("/abs/file.png"), "/abs");
}

}  // namespace
}  // namespace tools
}  // namespace jpegxl
