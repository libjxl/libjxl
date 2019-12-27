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

#include "jxl/predictor.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <hwy/static_targets.h>
#include <limits>
#include <random>

#include "gtest/gtest.h"
#include "jxl/aux_out.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/status.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/common.h"
#include "jxl/entropy_coder.h"
#include "jxl/image.h"

namespace jxl {
namespace predictor {
namespace {

TEST(PredictorTest, AverageTest) {
  EXPECT_EQ(0, Average(0, 1));
  EXPECT_EQ(0, Average(-1, 1));
  EXPECT_EQ(0, Average(-2, 2));
  EXPECT_EQ(3, Average(2, 4));
  EXPECT_EQ(3, Average(2, 5));
  EXPECT_EQ(3, Average(1, 5));
  EXPECT_EQ(-3, Average(-1, -5));
  EXPECT_EQ(0x7ffffffe, Average(0x7fffffff, 0x7ffffffe));
  EXPECT_EQ(-0x7ffffffe, Average(-0x7fffffff, -0x7ffffffe));
  EXPECT_EQ(0, Average(-0x7fffffff, 0x7ffffffe));  // -0.5
  EXPECT_EQ(0, Average(-0x7ffffffe, 0x7fffffff));  // 0.5
}

HWY_ATTR void TestPackSignedRange() {
  using Pack = PackSignedRange<0, 255>;
  for (int32_t pred = 0; pred <= 255; pred++) {
    for (int32_t i = 0; i <= 255; i++) {
      HWY_ALIGN int32_t predv[kNumPredictors];
      std::fill(predv, predv + kNumPredictors, pred);
      HWY_ALIGN uint32_t res_simd[kNumPredictors];
      Pack::Compute(i, predv, res_simd);
      uint32_t res = Pack::Residual(i, pred);
      EXPECT_EQ(i, Pack::Original(res, pred))
          << "v: " << i << " pred: " << pred;
      for (size_t p = 0; p < kNumPredictors; p++) {
        EXPECT_EQ(res, res_simd[p]) << "v: " << i << " pred: " << pred;
      }
    }
  }
}
TEST(PredictorTest, TestPackSignedRange) { TestPackSignedRange(); }

HWY_ATTR void TestPackSigned() {
  for (int32_t i = std::numeric_limits<int16_t>::min();
       i <= std::numeric_limits<int16_t>::max(); i++) {
    HWY_ALIGN int32_t zeros[kNumPredictors] = {};
    HWY_ALIGN uint32_t cost_simd[kNumPredictors];
    PackSigned::Compute(i, zeros, cost_simd);
    uint32_t cost = jxl::PackSigned(i);
    for (size_t i = 0; i < kNumPredictors; i++) {
      EXPECT_EQ(cost, cost_simd[i]);
    }
  }
}
TEST(PredictorTest, TestPackSigned) { TestPackSigned(); }

template <typename Pred>
HWY_ATTR void TestPredictor() {
  std::mt19937_64 rng;
  // While the range fits in int32_t, the internal calculations in
  // uniform_int_distribution overflow when using integer values, which is
  // undefined. That's why we use int64_t here instead.
  std::uniform_int_distribution<int64_t> dist(
      std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());

  for (int i = 0; i < 1000000; i++) {
    int32_t w = dist(rng);
    int32_t l = dist(rng);
    int32_t n = dist(rng);
    int32_t r = dist(rng);
    int32_t predictions[kNumPredictors];
    Pred::Predict(n, w, l, r, predictions);
    const uint32_t* mask_r0 = Pred::Row0Mask();
    const uint32_t* mask_c0 = Pred::Col0Mask();
    const uint32_t* mask_lc = Pred::LastColMask();
    // Set to 0 each pixel value and check that predictions that claim not to
    // use that pixel are still the same.
    // n
    int32_t altered_predictions[kNumPredictors];
    Pred::Predict(0, w, l, r, altered_predictions);
    for (size_t p = 0; p < kNumPredictors; p++) {
      if (mask_r0[p] == 0) EXPECT_EQ(altered_predictions[p], predictions[p]);
    }
    // w
    Pred::Predict(n, 0, l, r, altered_predictions);
    for (size_t p = 0; p < kNumPredictors; p++) {
      if (mask_c0[p] == 0) EXPECT_EQ(altered_predictions[p], predictions[p]);
    }
    // l
    Pred::Predict(n, w, 0, r, altered_predictions);
    for (size_t p = 0; p < kNumPredictors; p++) {
      if (mask_r0[p] == 0 && mask_c0[p] == 0)
        EXPECT_EQ(altered_predictions[p], predictions[p]);
    }
    // r
    Pred::Predict(n, w, l, 0, altered_predictions);
    for (size_t p = 0; p < kNumPredictors; p++) {
      if (mask_r0[p] == 0 && mask_lc[p] == 0)
        EXPECT_EQ(altered_predictions[p], predictions[p]);
    }
  }
}

TEST(PredictorTest, TestYPredictor) { TestPredictor<YPredictor>(); }

TEST(PredictorTest, TestXBPredictor) { TestPredictor<XBPredictor>(); }

// Simple set of predictors used only for testing.
struct PredictorForTesting {
  static JXL_INLINE void Predict(const int32_t n, const int32_t w,
                                 const int32_t l, const int32_t r,
                                 int32_t* JXL_RESTRICT pred) {
    pred[0] = pred[1] = n;
    pred[2] = pred[3] = w;
    pred[4] = pred[5] = l;
    pred[6] = pred[7] = r;
  }

  static JXL_INLINE const uint32_t* Row0Mask() {
    HWY_ALIGN static constexpr uint32_t kMask[kNumPredictors] = {
        kMaxError, kMaxError, 0, 0, kMaxError, kMaxError, kMaxError, kMaxError};
    return kMask;
  }

  static JXL_INLINE const uint32_t* Col0Mask() {
    HWY_ALIGN static constexpr uint32_t kMask[kNumPredictors] = {
        0, 0, kMaxError, kMaxError, kMaxError, kMaxError, 0, 0};
    return kMask;
  }

  static JXL_INLINE const uint32_t* LastColMask() {
    HWY_ALIGN static constexpr uint32_t kMask[kNumPredictors] = {
        0, 0, 0, 0, 0, 0, kMaxError, kMaxError};
    return kMask;
  }

  // Returns true if predictor is available.
  static bool RunPredictor(size_t idx, size_t c, const Image3I& img, size_t x,
                           size_t y, int32_t* pred) {
    if (idx == 0 || idx == 1) {
      if (y == 0) return false;
      *pred = img.ConstPlaneRow(c, y - 1)[x];
      return true;
    }
    if (idx == 2 || idx == 3) {
      if (x == 0) return false;
      *pred = img.ConstPlaneRow(c, y)[x - 1];
      return true;
    }
    if (idx == 4 || idx == 5) {
      if (y == 0) return false;
      if (x == 0) return false;
      *pred = img.ConstPlaneRow(c, y - 1)[x - 1];
      return true;
    }
    if (idx == 6 || idx == 7) {
      if (y == 0) return false;
      if (x == img.xsize() - 1) return false;
      *pred = img.ConstPlaneRow(c, y - 1)[x + 1];
      return true;
    }
    JXL_ASSERT(idx < 8);
    return false;
  }
};

TEST(PredictorTest, TestPredictorForTesting) {
  TestPredictor<PredictorForTesting>();
}

class PredictorTester;

using PredictorToTest = ComputeResiduals<
    PredictorTester, PackSigned,
    Predictors3<PredictorForTesting, PredictorForTesting, PredictorForTesting>>;

class PredictorTester : public PredictorToTest {
 public:
  explicit PredictorTester(const Image3I& test_img) : test_img_(test_img) {}

  void Run() {
    AuxOut aux_out;
    PredictorToTest::Run(test_img_.xsize(), test_img_.ysize(), &aux_out);
    EXPECT_EQ(pred_count_, test_img_.xsize() * test_img_.ysize());
  }

  JXL_INLINE void Prediction(size_t x, size_t y,
                             const int32_t* JXL_RESTRICT predictions,
                             const uint32_t* JXL_RESTRICT num_correct,
                             const uint32_t* JXL_RESTRICT min_error,
                             int32_t* JXL_RESTRICT decoded) {
    pred_count_++;
    for (size_t c = 0; c < 3; c++) {
      decoded[c] = test_img_.ConstPlaneRow(c, y)[x];

      // Verify prediction.
      uint32_t max_error[kNumPredictors] = {};
      bool has_error_information[kNumPredictors] = {};
      bool has_any_error_information = false;
      static constexpr int kErrorPixels[][2] = {{0, -1}, {-1, 0}, {-1, -1}};

      // Compute max error and availability of error information.
      for (auto px : kErrorPixels) {
        if (x < -px[0] || y < -px[1]) continue;
        for (size_t i = 0; i < kNumPredictors; i++) {
          int32_t pred;
          if (PredictorForTesting::RunPredictor(i, c, test_img_, x + px[0],
                                                y + px[1], &pred)) {
            uint32_t error = PackSigned::Residual(
                test_img_.ConstPlaneRow(c, y + px[1])[x + px[0]], pred);
            max_error[i] = std::max(error, max_error[i]);
            has_error_information[i] = true;
            has_any_error_information = true;
          }
        }
      }

      // Compute prediction for this pixel.
      int32_t prediction[kNumPredictors] = {};
      bool has_prediction[kNumPredictors] = {};
      bool has_any_prediction = false;
      for (size_t i = 0; i < kNumPredictors; i++) {
        has_prediction[i] = PredictorForTesting::RunPredictor(
            i, c, test_img_, x, y, &prediction[i]);
        if (has_prediction[i]) {
          has_any_prediction = true;
        }
      }

      // Compute best predictor.
      uint32_t best_predictor = 0;
      uint32_t merr = kMaxError;
      uint32_t corr = 0;
      bool has_predictor = false;
      // Implementation detail: if no error information is available, but at
      // least one predictor is, we give max_error-1 as the minimum error value.
      if (!has_any_error_information && has_any_prediction)
        merr = kMaxError - 1;
      for (size_t i = 0; i < kNumPredictors; i++) {
        // Count the predictors for which there is error information, the
        // prediction is valid, and max error is 0.
        if (has_prediction[i] && has_error_information[i] &&
            max_error[i] == 0) {
          corr++;
        }
        // Find the prediction with minimum error among the available ones.
        if (has_error_information[i] && has_prediction[i] &&
            max_error[i] < merr) {
          merr = max_error[i];
          best_predictor = i;
          has_predictor = true;
        }
        // If no error information is available, and this predictor is
        // available, use it if no other predictor was chosen.
        if (!has_predictor && !has_any_error_information && has_prediction[i]) {
          has_predictor = true;
          best_predictor = i;
        }
      }

      // Avoid spamming the output with failure once the test has already
      // failed.
      if (!::testing::Test::HasNonfatalFailure()) {
        EXPECT_EQ(predictions[c], prediction[best_predictor])
            << "c: " << c << " x: " << x << " y: " << y;
        EXPECT_EQ(num_correct[c], corr)
            << "c: " << c << " x: " << x << " y: " << y;
        EXPECT_EQ(min_error[c], merr)
            << "c: " << c << " x: " << x << " y: " << y;
      }
      if (::testing::Test::HasNonfatalFailure()) {
        JXL_CHECK(false);
      }
    }
  }

 private:
  const Image3I& test_img_;
  size_t pred_count_ = 0;
};

Image3I Stripes(int32_t low, int32_t high) {
  Image3I ret(kDcGroupDimInBlocks, kDcGroupDimInBlocks);
  for (size_t y = 0; y < ret.ysize(); y++) {
    int32_t* JXL_RESTRICT row = ret.PlaneRow(0, y);
    for (size_t x = 0; x < ret.xsize(); x++) {
      row[x] = x & 1 ? low : high;
    }
  }
  for (size_t y = 0; y < ret.ysize(); y++) {
    int32_t* JXL_RESTRICT row = ret.PlaneRow(1, y);
    for (size_t x = 0; x < ret.xsize(); x++) {
      row[x] = y & 1 ? low : high;
    }
  }
  for (size_t y = 0; y < ret.ysize(); y++) {
    int32_t* JXL_RESTRICT row = ret.PlaneRow(2, y);
    for (size_t x = 0; x < ret.xsize(); x++) {
      row[x] = (x + y) & 1 ? low : high;
    }
  }
  return ret;
}

Image3I Constant(int32_t low, int32_t high) {
  Image3I ret(kDcGroupDimInBlocks, kDcGroupDimInBlocks);
  for (size_t y = 0; y < ret.ysize(); y++) {
    int32_t* JXL_RESTRICT row = ret.PlaneRow(0, y);
    for (size_t x = 0; x < ret.xsize(); x++) {
      row[x] = low;
    }
  }
  for (size_t y = 0; y < ret.ysize(); y++) {
    int32_t* JXL_RESTRICT row = ret.PlaneRow(1, y);
    for (size_t x = 0; x < ret.xsize(); x++) {
      row[x] = high;
    }
  }
  for (size_t y = 0; y < ret.ysize(); y++) {
    int32_t* JXL_RESTRICT row = ret.PlaneRow(2, y);
    for (size_t x = 0; x < ret.xsize(); x++) {
      row[x] = (low + high) >> 1;
    }
  }
  return ret;
}

Image3I Random(int32_t low, int32_t high) {
  Image3I ret(kDcGroupDimInBlocks, kDcGroupDimInBlocks);
  std::mt19937_64 rng;
  std::uniform_int_distribution<int32_t> dist(low, high);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ret.ysize(); y++) {
      int32_t* JXL_RESTRICT row = ret.PlaneRow(c, y);
      for (size_t x = 0; x < ret.xsize(); x++) {
        row[x] = dist(rng);
      }
    }
  }
  return ret;
}

TEST(PredictorTest, TestStripes8) {
  Image3I img = Stripes(0, 255);
  PredictorTester tester(img);
  tester.Run();
}

TEST(PredictorTest, TestConstant8) {
  Image3I img = Constant(0, 255);
  PredictorTester tester(img);
  tester.Run();
}

TEST(PredictorTest, TestRandom8) {
  Image3I img = Random(0, 255);
  PredictorTester tester(img);
  tester.Run();
}

TEST(PredictorTest, TestStripes16) {
  Image3I img = Stripes(std::numeric_limits<int16_t>::min(),
                        std::numeric_limits<int16_t>::max());
  PredictorTester tester(img);
  tester.Run();
}

TEST(PredictorTest, TestConstant16) {
  Image3I img = Constant(std::numeric_limits<int16_t>::min(),
                         std::numeric_limits<int16_t>::max());
  PredictorTester tester(img);
  tester.Run();
}

TEST(PredictorTest, TestRandom16) {
  Image3I img = Random(std::numeric_limits<int16_t>::min(),
                       std::numeric_limits<int16_t>::max());
  PredictorTester tester(img);
  tester.Run();
}
}  // namespace
}  // namespace predictor
}  // namespace jxl
