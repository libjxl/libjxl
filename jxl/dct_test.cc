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

#include "jxl/dct.h"

#include <string.h>

#include <cmath>

#include "gtest/gtest.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/block.h"
#include "jxl/common.h"
#include "jxl/image.h"
#include "jxl/test_utils.h"

namespace jxl {
namespace {

using jxl::slow_dct::DCTSlow;
using jxl::slow_dct::IDCTSlow;

// Tests in the DctShardedTest class are sharded in 32 cases.
class DctShardedTest : public ::testing::TestWithParam<uint32_t> {};
std::vector<uint32_t> ShardRange(uint32_t n) {
#ifdef JXL_DISABLE_SLOW_TESTS
  JXL_ASSERT(n > 6);
  std::vector<uint32_t> ret = {0, 1, 3, 5, n - 1};
#else
  std::vector<uint32_t> ret(n);
  std::iota(ret.begin(), ret.end(), 0);
#endif  // JXL_DISABLE_SLOW_TESTS
  return ret;
}
INSTANTIATE_TEST_CASE_P(SlowDctTestInstantiation, DctShardedTest,
                        ::testing::ValuesIn(ShardRange(32)));

namespace slow_dct {

// These functions should be equivalent to ComputeTransposedScaledDCT in the jxl
// namespace (but slower and implemented for more transform sizes).
template <size_t N, class From, class To>
HWY_ATTR static void ComputeTransposedScaledDCT(const From& from,
                                                const To& to) {
  double blockd[N * N] = {};
  for (size_t y = 0; y < N; y++) {
    for (size_t x = 0; x < N; x++) {
      blockd[y * N + x] = from.Read(y, x);
    }
  }
  DCTSlow<N>(blockd);

  // Scale.
  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      to.Write(
          blockd[N * x + y] * N * N * IDCTScales<N>()[x] * IDCTScales<N>()[y],
          y, x);
    }
  }
}

template <size_t N, class From, class To>
HWY_ATTR static void ComputeTransposedScaledIDCT(const From& from,
                                                 const To& to) {
  // Scale.
  double blockd[N * N] = {};
  for (size_t y = 0; y < N; y++) {
    for (size_t x = 0; x < N; x++) {
      blockd[x * N + y] =
          from.Read(y, x) * N * N * DCTScales<N>()[x] * DCTScales<N>()[y];
    }
  }
  IDCTSlow<N>(blockd);

  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      to.Write(blockd[N * y + x], y, x);
    }
  }
}
}  // namespace slow_dct

// Computes the in-place NxN DCT of block.
// Requires that block is HWY_ALIGN'ed.
//
// Performs ComputeTransposedScaledDCT and then transposes and scales it to
// obtain "vanilla" DCT.
template <size_t N>
HWY_ATTR void ComputeDCT(float block[N * N]) {
  ComputeTransposedScaledDCT<N>()(FromBlock<N>(block), ToBlock<N>(block));

  // Untranspose.
  GenericTransposeBlockInplace<N>(FromBlock<N>(block), ToBlock<N>(block));

  // Unscale.
  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      block[N * y + x] *= DCTScales<N>()[x] * DCTScales<N>()[y];
    }
  }
}

// Computes the in-place 8x8 iDCT of block.
// Requires that block is HWY_ALIGN'ed.
template <int N>
HWY_ATTR void ComputeIDCT(float block[N * N]) {
  // Unscale.
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      block[N * y + x] *= IDCTScales<N>()[x] * IDCTScales<N>()[y];
    }
  }

  // Untranspose.
  GenericTransposeBlockInplace<N>(FromBlock<N>(block), ToBlock<N>(block));

  ComputeTransposedScaledIDCT<N>()(FromBlock<N>(block), ToBlock<N>(block));
}

template <size_t N, class From, class To>
HWY_ATTR JXL_INLINE void TransposeBlock(const From& from, const To& to) {
  if (N == 8) {
    TransposeBlock8(from, to);
  } else if (N == 16) {
    TransposeBlock16(from, to);
  } else {
    TransposeBlock32(from, to);
  }
}

// Cannot just use `if N == 8 / else` because the generated code will be invalid
// for the other codepath.
template <size_t N>
struct SpecializedColumn {};  // primary

template <>
struct SpecializedColumn<8> {
  template <class From, class To>
  static HWY_ATTR JXL_INLINE void DCT(const From& from, const To& to) {
    ColumnDCT8(from, to);
  }

  template <class From, class To>
  static HWY_ATTR JXL_INLINE void IDCT(const From& from, const To& to) {
    ColumnIDCT8(from, to);
  }
};

template <>
struct SpecializedColumn<16> {
  template <class From, class To>
  static HWY_ATTR JXL_INLINE void DCT(const From& from, const To& to) {
    ColumnDCT16(from, to);
  }

  template <class From, class To>
  static HWY_ATTR JXL_INLINE void IDCT(const From& from, const To& to) {
    ColumnIDCT16(from, to);
  }
};

template <>
struct SpecializedColumn<32> {
  template <class From, class To>
  static HWY_ATTR JXL_INLINE void DCT(const From& from, const To& to) {
    ColumnDCT32(from, to);
  }

  template <class From, class To>
  static HWY_ATTR JXL_INLINE void IDCT(const From& from, const To& to) {
    ColumnIDCT32(from, to);
  }
};

template <size_t N>
HWY_ATTR void TransposeTest(float accuracy) {
  constexpr size_t kBlockSize = N * N;
  HWY_ALIGN float src[kBlockSize];
  ToBlock<N> to_src(src);
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      to_src.Write(y * N + x, y, x);
    }
  }
  HWY_ALIGN float dst[kBlockSize];
  TransposeBlock<N>(FromBlock<N>(src), ToBlock<N>(dst));
  FromBlock<N> from_dst(dst);
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      float expected = x * N + y;
      float actual = from_dst.Read(y, x);
      EXPECT_NEAR(expected, actual, accuracy) << "x = " << x << ", y = " << y;
    }
  }
}

TEST(TransposeTest, Transpose8) { TransposeTest<8>(1e-7); }

TEST(TransposeTest, Transpose16) { TransposeTest<16>(1e-7); }

TEST(TransposeTest, Transpose32) { TransposeTest<32>(1e-7); }

template <size_t N>
HWY_ATTR void TestDctAccuracy(float accuracy, size_t start = 0,
                              size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  for (size_t i = start; i < end; i++) {
    HWY_ALIGN float fast[kBlockSize] = {0.0f};
    double slow[kBlockSize] = {0.0};
    fast[i] = 1.0;
    slow[i] = 1.0;
    DCTSlow<N>(slow);
    ComputeDCT<N>(fast);
    for (size_t k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(fast[k], slow[k], accuracy) << "i = " << i << ", k = " << k;
    }
  }
}

template <size_t N>
HWY_ATTR void TestDctAccuracyShard(float accuracy, size_t shard) {
  TestDctAccuracy<N>(accuracy, N * shard, N * (shard + 1));
}

TEST(DctTest, Accuracy8) { TestDctAccuracy<8>(1.1e-7); }
TEST(DctTest, Accuracy16) { TestDctAccuracy<16>(1.1e-7); }
TEST_P(DctShardedTest, DctTest_Accuracy32) {
  TestDctAccuracyShard<32>(1.1e-7, GetParam());
}

template <size_t N>
HWY_ATTR void TestIdctAccuracy(float accuracy, size_t start = 0,
                               size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  for (size_t i = start; i < end; i++) {
    HWY_ALIGN float fast[kBlockSize] = {0.0f};
    double slow[kBlockSize] = {0.0};
    fast[i] = 1.0;
    slow[i] = 1.0;
    IDCTSlow<N>(slow);
    ComputeIDCT<N>(fast);
    for (size_t k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(fast[k], slow[k], accuracy) << "i = " << i << ", k = " << k;
    }
  }
}

template <size_t N>
HWY_ATTR void TestIdctAccuracyShard(float accuracy, size_t shard) {
  TestIdctAccuracy<N>(accuracy, N * shard, N * (shard + 1));
}

TEST(IdctTest, Accuracy8) { TestIdctAccuracy<8>(1e-7); }
TEST(IdctTest, Accuracy16) { TestIdctAccuracy<16>(1e-7); }
TEST_P(DctShardedTest, IdctTest_Accuracy32) {
  TestIdctAccuracyShard<32>(1e-7, GetParam());
}

template <size_t N>
HWY_ATTR void TestInverse(float accuracy) {
  ThreadPoolInternal pool(N < 32 ? 0 : 8);
  enum { kBlockSize = N * N };
  RunOnPool(
      &pool, 0, kBlockSize, ThreadPool::SkipInit(),
      [accuracy](const int task, int /*thread*/) {
        const size_t i = static_cast<size_t>(task);
        HWY_ALIGN float x[kBlockSize] = {0.0f};
        x[i] = 1.0;

        ComputeIDCT<N>(x);
        ComputeDCT<N>(x);

        for (size_t k = 0; k < kBlockSize; ++k) {
          EXPECT_NEAR(x[k], (k == i) ? 1.0f : 0.0f, accuracy)
              << "i = " << i << ", k = " << k;
        }
      },
      "TestInverse");
}

TEST(IdctTest, Inverse8) { TestInverse<8>(1e-6f); }

TEST(IdctTest, Inverse16) { TestInverse<16>(1e-6f); }

TEST(IdctTest, Inverse32) { TestInverse<32>(3e-6f); }

template <size_t N>
HWY_ATTR void TestIdctOrthonormal(float accuracy) {
  constexpr size_t kBlockSize = N * N;
  ImageF xs(kBlockSize, kBlockSize);
  for (int i = 0; i < kBlockSize; ++i) {
    float* x = xs.Row(i);
    for (int j = 0; j < kBlockSize; ++j) x[j] = (i == j) ? 1.0f : 0.0f;
    ComputeIDCT<N>(x);
  }
  for (int i = 0; i < kBlockSize; ++i) {
    for (int j = 0; j < kBlockSize; ++j) {
      float product = 0.0f;
      for (int k = 0; k < kBlockSize; ++k) {
        product += xs.Row(i)[k] * xs.Row(j)[k];
      }
      EXPECT_NEAR(product, (i == j) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", j = " << j;
    }
  }
}

TEST(IdctTest, IDCTOrthonormal8) { TestIdctOrthonormal<8>(1e-6f); }

TEST(IdctTest, IDCTOrthonormal16) { TestIdctOrthonormal<16>(1.2e-6f); }

TEST(IdctTest, IDCTOrthonormal32) { TestIdctOrthonormal<32>(5e-6f); }

template <size_t N>
HWY_ATTR void TestDctTranspose(float accuracy, size_t start = 0,
                               size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  for (size_t i = start; i < end; i++) {
    for (size_t j = 0; j < kBlockSize; ++j) {
      // We check that <e_i, Me_j> = <M^\dagger{}e_i, e_j>.
      // That means (Me_j)_i = (M^\dagger{}e_i)_j

      // x := Me_j
      HWY_ALIGN float x[kBlockSize] = {0.0f};
      x[j] = 1.0;
      ComputeIDCT<N>(x);
      // y := M^\dagger{}e_i
      HWY_ALIGN float y[kBlockSize] = {0.0f};
      y[i] = 1.0;
      ComputeDCT<N>(y);

      EXPECT_NEAR(x[i], y[j], accuracy) << "i = " << i << ", j = " << j;
    }
  }
}

TEST(IdctTest, Transpose8) { TestDctTranspose<8>(1e-6); }

TEST(IdctTest, Transpose16) { TestDctTranspose<16>(1e-6); }

template <size_t N>
HWY_ATTR void TestDctTransposeShard(float accuracy, size_t shard) {
  TestDctTranspose<N>(accuracy, N * shard, N * (shard + 1));
}
TEST_P(DctShardedTest, IdctTest_Transpose32) {
  TestDctTransposeShard<32>(3e-6, GetParam());
}

template <size_t N>
HWY_ATTR void TestDctOrthonormal(float accuracy) {
  constexpr size_t kBlockSize = N * N;
  ImageF xs(kBlockSize, kBlockSize);
  for (int i = 0; i < kBlockSize; ++i) {
    float* x = xs.Row(i);
    for (int j = 0; j < kBlockSize; ++j) x[j] = (i == j) ? 1.0f : 0.0f;
    ComputeDCT<N>(x);
  }
  for (int i = 0; i < kBlockSize; ++i) {
    for (int j = 0; j < kBlockSize; ++j) {
      float product = 0.0f;
      for (int k = 0; k < kBlockSize; ++k) {
        product += xs.Row(i)[k] * xs.Row(j)[k];
      }
      EXPECT_NEAR(product, (i == j) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", j = " << j;
    }
  }
}

TEST(IdctTest, DCTOrthonormal8) { TestDctOrthonormal<8>(1e-6f); }

TEST(IdctTest, DCTOrthonormal16) { TestDctOrthonormal<16>(1e-6f); }

TEST(IdctTest, DCTOrthonormal32) { TestDctOrthonormal<32>(1e-6f); }

template <size_t N>
HWY_ATTR void ColumnDctRoundtrip(float accuracy) {
  constexpr size_t kBlockSize = N * N;
  // Though we are only interested in single column result, dct.h has built-in
  // limit on minimal number of columns processed. So, to be safe, we do
  // regular 8x8 block transformation. On the bright side - we could check all
  // 8 basis vectors at once.
  HWY_ALIGN float block[kBlockSize];
  ToBlock<N> to(block);
  FromBlock<N> from(block);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      to.Write((i == j) ? 1.0f : 0.0f, i, j);
    }
  }

  // Running (I)DCT on the same memory block seems to trigger a compiler bug on
  // ARMv7 with clang6.
  HWY_ALIGN float tmp[kBlockSize];
  ToBlock<N> to_tmp(tmp);
  FromBlock<N> from_tmp(tmp);

  SpecializedColumn<N>::DCT(from, to_tmp);
  SpecializedColumn<N>::IDCT(from_tmp, to);
  constexpr float scale = 1.0f / N;

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      float actual = from.Read(i, j) * scale;
      EXPECT_NEAR(expected, actual, accuracy) << " i=" << i << ", j=" << j;
    }
  }
}

TEST(IdctTest, ColumnDctRoundtrip8) { ColumnDctRoundtrip<8>(1e-6f); }

TEST(IdctTest, ColumnDctRoundtrip16) { ColumnDctRoundtrip<16>(1e-6f); }

TEST(IdctTest, ColumnDctRoundtrip32) { ColumnDctRoundtrip<32>(1e-6f); }

// See "Steerable Discrete Cosine Transform", Fracastoro G., Fosson S., Magli
// E., https://arxiv.org/pdf/1610.09152.pdf
template <int N>
void RotateDCT(float angle, float block[N * N]) {
  float a2a = std::cos(angle);
  float a2b = -std::sin(angle);
  float b2a = std::sin(angle);
  float b2b = std::cos(angle);
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < y; x++) {
      if (x >= 2 || y >= 2) continue;
      float a = block[N * y + x];
      float b = block[N * x + y];
      block[N * y + x] = a2a * a + b2a * b;
      block[N * x + y] = a2b * a + b2b * b;
    }
  }
}

TEST(RotateTest, ZeroIdentity) {
  for (int i = 0; i < 64; i++) {
    float x[64] = {0.0f};
    x[i] = 1.0;
    RotateDCT<8>(0.0f, x);
    for (int j = 0; j < 64; j++) {
      EXPECT_NEAR(x[j], (i == j) ? 1.0f : 0.0f, 1e-6f);
    }
  }
}

TEST(RotateTest, InverseRotation) {
  const float angle = 0.1f;
  for (int i = 0; i < 64; i++) {
    float x[64] = {0.0f};
    x[i] = 1.0;
    RotateDCT<8>(angle, x);
    RotateDCT<8>(-angle, x);
    for (int j = 0; j < 64; j++) {
      EXPECT_NEAR(x[j], (i == j) ? 1.0f : 0.0f, 1e-6f);
    }
  }
}

template <size_t N>
HWY_ATTR void TestSlowIsSameDCT(float accuracy, size_t start = 0,
                                size_t end = N * N) {
  for (size_t i = start; i < end; i++) {
    HWY_ALIGN float block1[N * N] = {};
    HWY_ALIGN float block2[N * N] = {};
    block1[i] = 1.0;
    block2[i] = 1.0;
    ComputeTransposedScaledDCT<N>()(FromBlock<N>(block1), ToBlock<N>(block1));
    slow_dct::ComputeTransposedScaledDCT<N>(FromBlock<N>(block2),
                                            ToBlock<N>(block2));
    for (int j = 0; j < N * N; j++) {
      EXPECT_NEAR(block1[j], block2[j], accuracy);
    }
  }
}

template <size_t N>
HWY_ATTR void TestSlowIsSameDCTShard(float accuracy, int shard) {
  TestSlowIsSameDCT<N>(accuracy, N * shard, N * (shard + 1));
}

TEST(SlowDctTest, DCTIsSame2) { TestSlowIsSameDCT<2>(1e-5); }
TEST(SlowDctTest, DCTIsSame4) { TestSlowIsSameDCT<4>(1e-5); }
TEST(SlowDctTest, DCTIsSame8) { TestSlowIsSameDCT<8>(1e-5); }
TEST(SlowDctTest, DCTIsSame16) { TestSlowIsSameDCT<16>(3e-5); }
TEST_P(DctShardedTest, DCTIsSame32) {
  TestSlowIsSameDCTShard<32>(1e-4, GetParam());
}

template <size_t N>
HWY_ATTR void TestSlowIsSameIDCT(float accuracy, size_t start = 0,
                                 size_t end = N * N) {
  for (size_t i = start; i < end; i++) {
    HWY_ALIGN float block1[N * N] = {};
    HWY_ALIGN float block2[N * N] = {};
    block1[i] = 1.0;
    block2[i] = 1.0;
    ComputeTransposedScaledIDCT<N>()(FromBlock<N>(block1), ToBlock<N>(block1));
    slow_dct::ComputeTransposedScaledIDCT<N>(FromBlock<N>(block2),
                                             ToBlock<N>(block2));
    for (int j = 0; j < N * N; j++) {
      EXPECT_NEAR(block1[j], block2[j], accuracy);
    }
  }
}

template <size_t N>
HWY_ATTR void TestSlowIsSameIDCTShard(float accuracy, int shard) {
  TestSlowIsSameIDCT<N>(accuracy, N * shard, N * (shard + 1));
}

TEST(SlowDctTest, IDCTIsSame2) { TestSlowIsSameIDCT<2>(1e-5); }
TEST(SlowDctTest, IDCTIsSame4) { TestSlowIsSameIDCT<4>(1e-5); }
TEST(SlowDctTest, IDCTIsSame8) { TestSlowIsSameIDCT<8>(1e-5); }
TEST(SlowDctTest, IDCTIsSame16) { TestSlowIsSameIDCT<16>(2e-5); }
TEST_P(DctShardedTest, IDCTIsSame32) {
  TestSlowIsSameIDCTShard<32>(1e-4, GetParam());
}

template <size_t N>
HWY_ATTR void TestSlowInverse(float accuracy, size_t start = 0,
                              size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  for (size_t i = start; i < end; i++) {
    float x[kBlockSize] = {0.0f};
    x[i] = 1.0;

    slow_dct::ComputeTransposedScaledDCT<N>(FromBlock<N>(x),
                                            ScaleToBlock<N>(x));
    slow_dct::ComputeTransposedScaledIDCT<N>(FromBlock<N>(x), ToBlock<N>(x));

    for (int k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(x[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", k = " << k;
    }
  }
}

template <size_t N>
HWY_ATTR void TestSlowInverseShard(float accuracy, int shard) {
  TestSlowInverse<N>(accuracy, N * shard, N * (shard + 1));
}

TEST(SlowDctTest, SlowInverse2) { TestSlowInverse<2>(1e-5); }
TEST(SlowDctTest, SlowInverse4) { TestSlowInverse<4>(1e-5); }
TEST(SlowDctTest, SlowInverse8) { TestSlowInverse<8>(1e-5); }
TEST(SlowDctTest, SlowInverse16) { TestSlowInverse<16>(1e-5); }
TEST_P(DctShardedTest, SlowInverse32) {
  TestSlowInverseShard<32>(1e-5, GetParam());
}

template <size_t ROWS, size_t COLS>
HWY_ATTR void TestRectInverse(float accuracy) {
  constexpr size_t kBlockSize = ROWS * COLS;
  for (int i = 0; i < kBlockSize; ++i) {
    HWY_ALIGN float x[kBlockSize] = {0.0f};
    HWY_ALIGN float out[kBlockSize] = {0.0f};
    x[i] = 1.0;
    HWY_ALIGN float coeffs[kBlockSize] = {0.0f};

    constexpr size_t OUT_ROWS = ROWS < COLS ? ROWS : COLS;
    constexpr size_t OUT_COLS = ROWS < COLS ? COLS : ROWS;

    ComputeScaledDCT<ROWS, COLS>()(FromBlock<ROWS, COLS>(x),
                                   ScaleToBlock<OUT_ROWS, OUT_COLS>(coeffs));
    ComputeScaledIDCT<ROWS, COLS>()(FromBlock<OUT_ROWS, OUT_COLS>(coeffs),
                                    ToBlock<ROWS, COLS>(out));

    for (int k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(out[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", k = " << k;
    }
  }
}

TEST(RectDctTest, Inverse1632) { TestRectInverse<16, 32>(1e-6f); }
TEST(RectDctTest, Inverse832) { TestRectInverse<8, 32>(1e-6f); }
TEST(RectDctTest, Inverse816) { TestRectInverse<8, 16>(1e-6f); }
TEST(RectDctTest, Inverse48) { TestRectInverse<4, 8>(1e-6f); }
TEST(RectDctTest, Inverse24) { TestRectInverse<2, 4>(1e-6f); }
TEST(RectDctTest, Inverse14) { TestRectInverse<1, 4>(1e-6f); }
TEST(RectDctTest, Inverse12) { TestRectInverse<1, 2>(1e-6f); }

TEST(RectDctTest, Inverse3216) { TestRectInverse<32, 16>(1e-6f); }
TEST(RectDctTest, Inverse328) { TestRectInverse<32, 8>(1e-6f); }
TEST(RectDctTest, Inverse168) { TestRectInverse<16, 8>(1e-6f); }
TEST(RectDctTest, Inverse84) { TestRectInverse<8, 4>(1e-6f); }
TEST(RectDctTest, Inverse42) { TestRectInverse<4, 2>(1e-6f); }
TEST(RectDctTest, Inverse41) { TestRectInverse<4, 1>(1e-6f); }
TEST(RectDctTest, Inverse21) { TestRectInverse<2, 1>(1e-6f); }

template <size_t ROWS, size_t COLS>
HWY_ATTR void TestRectTranspose(float accuracy) {
  constexpr size_t kBlockSize = ROWS * COLS;
  for (int px = 0; px < COLS; ++px) {
    for (int py = 0; py < ROWS; ++py) {
      HWY_ALIGN float x1[kBlockSize] = {0.0f};
      HWY_ALIGN float x2[kBlockSize] = {0.0f};
      HWY_ALIGN float coeffs1[kBlockSize] = {0.0f};
      HWY_ALIGN float coeffs2[kBlockSize] = {0.0f};
      x1[py * COLS + px] = 1;
      x2[px * ROWS + py] = 1;

      constexpr size_t OUT_ROWS = ROWS < COLS ? ROWS : COLS;
      constexpr size_t OUT_COLS = ROWS < COLS ? COLS : ROWS;

      ComputeScaledDCT<ROWS, COLS>()(FromBlock<ROWS, COLS>(x1),
                                     ScaleToBlock<OUT_ROWS, OUT_COLS>(coeffs1));
      ComputeScaledDCT<COLS, ROWS>()(FromBlock<COLS, ROWS>(x2),
                                     ScaleToBlock<OUT_ROWS, OUT_COLS>(coeffs2));

      for (int x = 0; x < OUT_COLS; ++x) {
        for (int y = 0; y < OUT_ROWS; ++y) {
          EXPECT_NEAR(coeffs1[y * OUT_COLS + x], coeffs2[y * OUT_COLS + x],
                      accuracy)
              << " px = " << px << ", py = " << py << ", x = " << x
              << ", y = " << y;
        }
      }
    }
  }
}

TEST(RectDctTest, TransposeTest1632) { TestRectTranspose<16, 32>(1e-6f); }
TEST(RectDctTest, TransposeTest832) { TestRectTranspose<8, 32>(1e-6f); }
TEST(RectDctTest, TransposeTest816) { TestRectTranspose<8, 16>(1e-6f); }
TEST(RectDctTest, TransposeTest48) { TestRectTranspose<4, 8>(1e-6f); }
TEST(RectDctTest, TransposeTest24) { TestRectTranspose<2, 4>(1e-6f); }
TEST(RectDctTest, TransposeTest14) { TestRectTranspose<1, 4>(1e-6f); }
TEST(RectDctTest, TransposeTest12) { TestRectTranspose<1, 2>(1e-6f); }

// This would be identical to TransposeTest816.
// TEST(RectDctTest, TransposeTest168) { TestRectTranspose<16, 8>(1e-6f); }
}  // namespace
}  // namespace jxl
