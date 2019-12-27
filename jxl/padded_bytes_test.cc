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

#include "jxl/base/padded_bytes.h"

#include "gtest/gtest.h"

namespace jxl {
namespace {

TEST(PaddedBytesTest, TestNonEmptyFirstByteZero) {
  PaddedBytes pb(1);
  EXPECT_EQ(0, pb[0]);
  // Even after resizing..
  pb.resize(20);
  EXPECT_EQ(0, pb[0]);
  // And reserving.
  pb.reserve(200);
  EXPECT_EQ(0, pb[0]);
}

TEST(PaddedBytesTest, TestEmptyFirstByteZero) {
  PaddedBytes pb(0);
  // After resizing - new zero is written despite there being nothing to copy.
  pb.resize(20);
  EXPECT_EQ(0, pb[0]);
}

TEST(PaddedBytesTest, TestFillWithoutReserve) {
  PaddedBytes pb;
  for (size_t i = 0; i < 170; ++i) {
    pb.push_back(i);
  }
  EXPECT_EQ(170, pb.size());
  EXPECT_GE(pb.capacity(), 170);
}

TEST(PaddedBytesTest, TestFillWithExactReserve) {
  PaddedBytes pb;
  pb.reserve(170);
  for (size_t i = 0; i < 170; ++i) {
    pb.push_back(i);
  }
  EXPECT_EQ(170, pb.size());
  EXPECT_EQ(pb.capacity(), 170);
}

TEST(PaddedBytesTest, TestFillWithMoreReserve) {
  PaddedBytes pb;
  pb.reserve(171);
  for (size_t i = 0; i < 170; ++i) {
    pb.push_back(i);
  }
  EXPECT_EQ(170, pb.size());
  EXPECT_GT(pb.capacity(), 170);
}

}  // namespace
}  // namespace jxl
