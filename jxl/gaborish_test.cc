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

#include "jxl/gaborish.h"

#include "gtest/gtest.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"

namespace jxl {
namespace {

void TestRoundTrip(const Image3F& in, double max_l1) {
  Image3F fwd(in.xsize(), in.ysize());
  ThreadPool* null_pool = nullptr;
  ConvolveGaborish(in.Plane(0), 0, 0, null_pool,
                   const_cast<ImageF*>(&fwd.Plane(0)));
  ConvolveGaborish(in.Plane(1), 0, 0, null_pool,
                   const_cast<ImageF*>(&fwd.Plane(1)));
  ConvolveGaborish(in.Plane(2), 0, 0, null_pool,
                   const_cast<ImageF*>(&fwd.Plane(2)));
  const Image3F rev = GaborishInverse(fwd, 0.92718927264540152f, null_pool);
  VerifyRelativeError(in, rev, max_l1, 1E-4);
}

TEST(GaborishTest, TestZero) {
  Image3F in(20, 20);
  ZeroFillImage(&in);
  TestRoundTrip(in, 0.0);
}

// Disabled: large difference.
#if 0
TEST(GaborishTest, TestDirac) {
  Image3F in(20, 20);
  ZeroFillImage(&in);
  in.PlaneRow(1, 10)[10] = 10.0f;
  TestRoundTrip(in, 0.26);
}
#endif

TEST(GaborishTest, TestFlat) {
  Image3F in(20, 20);
  FillImage(1.0f, &in);
  TestRoundTrip(in, 1E-5);
}

}  // namespace
}  // namespace jxl
