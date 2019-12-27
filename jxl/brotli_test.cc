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

#include "jxl/brotli.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>

#include "gtest/gtest.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/thread_pool_internal.h"

namespace jxl {
namespace {

TEST(BrotliTest, TestCompressEmpty) {
  PaddedBytes in;
  PaddedBytes out;
  EXPECT_TRUE(BrotliCompress(6, in, &out));
  EXPECT_TRUE(in.empty());
}

TEST(BrotliTest, TestDecompressEmpty) {
  size_t bytes_read = 0;
  PaddedBytes in;
  PaddedBytes out;
  EXPECT_FALSE(BrotliDecompress(Span<const uint8_t>(in), 1, &bytes_read, &out));
  EXPECT_EQ(0, bytes_read);
  EXPECT_TRUE(in.empty());
  EXPECT_TRUE(out.empty());
}

TEST(BrotliTest, TestRoundTrip) {
  ThreadPoolInternal pool(0);
  pool.Run(1, 65, ThreadPool::SkipInit(), [](const int task, const int thread) {
    const size_t size = task;

    PaddedBytes in(size);
    std::mt19937_64 rng(thread * 65537 + task * 129);
    std::generate(in.begin(), in.end(), rng);
    PaddedBytes compressed;
    PaddedBytes out;

    for (int quality = 1; quality < 7; ++quality) {
      compressed.clear();
      EXPECT_TRUE(BrotliCompress(quality, in, &compressed));
      size_t bytes_read = 0;
      out.clear();
      EXPECT_TRUE(BrotliDecompress(Span<const uint8_t>(compressed), size,
                                   &bytes_read, &out));
      EXPECT_EQ(compressed.size(), bytes_read);
      EXPECT_EQ(in.size(), out.size());
      for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] != out[i]) {
          printf("Mismatch at %zu (%zu)\n", i, size);
          exit(1);
        }
      }
    }
  });
}

}  // namespace
}  // namespace jxl
