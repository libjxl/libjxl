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

#include <string.h>

#include <cmath>
#include <numeric>

#include "jxl/base/thread_pool_internal.h"
#include "jxl/common.h"
#include "jxl/dct_for_test.h"
#include "jxl/dct_scales.h"
#include "jxl/image.h"
#include "jxl/test_utils.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/dct_test.cc"
#include <hwy/foreach_target.h>

#include "jxl/dec_dct-inl.h"
#include "jxl/enc_dct-inl.h"

#include <hwy/tests/test_util-inl.h>

// SIMD code
#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

// Computes the in-place NxN DCT of block.
// Requires that block is HWY_ALIGN'ed.
//
// Performs ComputeTransposedScaledDCT and then transposes and scales it to
// obtain "vanilla" DCT.
template <size_t N>
void ComputeDCT(float block[N * N]) {
  HWY_ALIGN float tmp_block[N * N];
  ComputeTransposedScaledDCT<N>()(FromBlock(N, N, block),
                                  ToBlock(N, N, tmp_block));

  // Untranspose.
  Transpose<N, N>::Run(FromBlock(N, N, tmp_block), ToBlock(N, N, block));

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
void ComputeIDCT(float block[N * N]) {
  // Unscale.
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      block[N * y + x] *= IDCTScales<N>()[x] * IDCTScales<N>()[y];
    }
  }

  HWY_ALIGN float tmp_block[N * N];
  // Untranspose.
  Transpose<N, N>::Run(FromBlock(N, N, block), ToBlock(N, N, tmp_block));

  ComputeTransposedScaledIDCT<N>()(FromBlock(N, N, tmp_block),
                                   ToBlock(N, N, block));
}

template <size_t N>
void TransposeTestT(float accuracy) {
  constexpr size_t kBlockSize = N * N;
  HWY_ALIGN float src[kBlockSize];
  ToBlock to_src(N, N, src);
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      to_src.Write(y * N + x, y, x);
    }
  }
  HWY_ALIGN float dst[kBlockSize];
  Transpose<N, N>::Run(FromBlock(N, N, src), ToBlock(N, N, dst));
  FromBlock from_dst(N, N, dst);
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      float expected = x * N + y;
      float actual = from_dst.Read(y, x);
      EXPECT_NEAR(expected, actual, accuracy) << "x = " << x << ", y = " << y;
    }
  }
}

void TransposeTest() {
  TransposeTestT<8>(1e-7f);
  TransposeTestT<16>(1e-7f);
  TransposeTestT<32>(1e-7f);
}

template <size_t N>
void ColumnDctRoundtripT(float accuracy) {
  constexpr size_t kBlockSize = N * N;
  // Though we are only interested in single column result, dct.h has built-in
  // limit on minimal number of columns processed. So, to be safe, we do
  // regular 8x8 block transformation. On the bright side - we could check all
  // 8 basis vectors at once.
  HWY_ALIGN float block[kBlockSize];
  ToBlock to(N, N, block);
  FromBlock from(N, N, block);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      to.Write((i == j) ? 1.0f : 0.0f, i, j);
    }
  }

  // Running (I)DCT on the same memory block seems to trigger a compiler bug on
  // ARMv7 with clang6.
  HWY_ALIGN float tmp[kBlockSize];
  ToBlock to_tmp(N, N, tmp);
  FromBlock from_tmp(N, N, tmp);

  ColumnDCT(DCTSizeTag<N>(), from, to_tmp);
  ColumnIDCT(DCTSizeTag<N>(), from_tmp, to);
  constexpr float scale = 1.0f / N;

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      float actual = from.Read(i, j) * scale;
      EXPECT_NEAR(expected, actual, accuracy) << " i=" << i << ", j=" << j;
    }
  }
}

void ColumnDctRoundtrip() {
  ColumnDctRoundtripT<8>(1e-6f);
  ColumnDctRoundtripT<16>(1e-6f);
  ColumnDctRoundtripT<32>(1e-6f);
}

// (inside begin/end because From/To are from dct_block-inl.h)
namespace slow_dct {

// These functions should be equivalent to ComputeTransposedScaledDCT in the jxl
// namespace (but slower and implemented for more transform sizes).
template <size_t N, class From, class To>
void ComputeTransposedScaledDCT(const From& from, const To& to) {
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
void ComputeTransposedScaledIDCT(const From& from, const To& to) {
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

template <size_t N>
void TestDctAccuracy(float accuracy, size_t start = 0, size_t end = N * N) {
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
void TestIdctAccuracy(float accuracy, size_t start = 0, size_t end = N * N) {
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
void TestInverseT(float accuracy) {
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

void InverseTest() {
  TestInverseT<8>(1e-6f);
  TestInverseT<16>(1e-6f);
  TestInverseT<32>(3e-6f);
}

template <size_t N>
void TestIdctOrthonormal(float accuracy) {
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

void IDCTOrthonormalTest() {
  TestIdctOrthonormal<8>(1e-6f);
  TestIdctOrthonormal<16>(1.2e-6f);
  TestIdctOrthonormal<32>(5e-6f);
}

template <size_t N>
void TestDctOrthonormal(float accuracy) {
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

void DCTOrthonormalTest() {
  TestDctOrthonormal<8>(1e-6f);
  TestDctOrthonormal<16>(1e-6f);
  TestDctOrthonormal<32>(1e-6f);
}

template <size_t N>
void TestDctTranspose(float accuracy, size_t start = 0, size_t end = N * N) {
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

template <size_t N>
void TestSlowIsSameDCT(float accuracy, size_t start = 0, size_t end = N * N) {
  for (size_t i = start; i < end; i++) {
    HWY_ALIGN float block1[N * N] = {};
    HWY_ALIGN float block2[N * N] = {};
    block1[i] = 1.0;
    block2[i] = 1.0;
    ComputeTransposedScaledDCT<N>()(FromBlock(N, N, block1),
                                    ToBlock(N, N, block1));
    slow_dct::ComputeTransposedScaledDCT<N>(FromBlock(N, N, block2),
                                            ToBlock(N, N, block2));
    for (int j = 0; j < N * N; j++) {
      EXPECT_NEAR(block1[j], block2[j], accuracy);
    }
  }
}

template <size_t N>
void TestSlowIsSameIDCT(float accuracy, size_t start = 0, size_t end = N * N) {
  for (size_t i = start; i < end; i++) {
    HWY_ALIGN float block1[N * N] = {};
    HWY_ALIGN float block2[N * N] = {};
    block1[i] = 1.0;
    block2[i] = 1.0;
    ComputeTransposedScaledIDCT<N>()(FromBlock(N, N, block1),
                                     ToBlock(N, N, block1));
    slow_dct::ComputeTransposedScaledIDCT<N>(FromBlock(N, N, block2),
                                             ToBlock(N, N, block2));
    for (int j = 0; j < N * N; j++) {
      EXPECT_NEAR(block1[j], block2[j], accuracy);
    }
  }
}

template <size_t N>
void TestSlowInverse(float accuracy, size_t start = 0, size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  for (size_t i = start; i < end; i++) {
    float x[kBlockSize] = {0.0f};
    x[i] = 1.0;

    slow_dct::ComputeTransposedScaledDCT<N>(FromBlock(N, N, x),
                                            ScaleToBlock(N, N, x));
    slow_dct::ComputeTransposedScaledIDCT<N>(FromBlock(N, N, x),
                                             ToBlock(N, N, x));

    for (int k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(x[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", k = " << k;
    }
  }
}

template <size_t ROWS, size_t COLS>
void TestRectInverseT(float accuracy) {
  constexpr size_t kBlockSize = ROWS * COLS;
  for (int i = 0; i < kBlockSize; ++i) {
    HWY_ALIGN float x[kBlockSize] = {0.0f};
    HWY_ALIGN float out[kBlockSize] = {0.0f};
    x[i] = 1.0;
    HWY_ALIGN float coeffs[kBlockSize] = {0.0f};

    constexpr size_t OUT_ROWS = ROWS < COLS ? ROWS : COLS;
    constexpr size_t OUT_COLS = ROWS < COLS ? COLS : ROWS;

    ComputeScaledDCT<ROWS, COLS>()(FromBlock(ROWS, COLS, x),
                                   ScaleToBlock(OUT_ROWS, OUT_COLS, coeffs));
    ComputeScaledIDCT<ROWS, COLS>()(FromBlock(OUT_ROWS, OUT_COLS, coeffs),
                                    ToBlock(ROWS, COLS, out));

    for (int k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(out[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", k = " << k;
    }
  }
}

void TestRectInverse() {
  TestRectInverseT<16, 32>(1e-6f);
  TestRectInverseT<8, 32>(1e-6f);
  TestRectInverseT<8, 16>(1e-6f);
  TestRectInverseT<4, 8>(1e-6f);
  TestRectInverseT<2, 4>(1e-6f);
  TestRectInverseT<1, 4>(1e-6f);
  TestRectInverseT<1, 2>(1e-6f);

  TestRectInverseT<32, 16>(1e-6f);
  TestRectInverseT<32, 8>(1e-6f);
  TestRectInverseT<16, 8>(1e-6f);
  TestRectInverseT<8, 4>(1e-6f);
  TestRectInverseT<4, 2>(1e-6f);
  TestRectInverseT<4, 1>(1e-6f);
  TestRectInverseT<2, 1>(1e-6f);
}

template <size_t ROWS, size_t COLS>
void TestRectTransposeT(float accuracy) {
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

      ComputeScaledDCT<ROWS, COLS>()(FromBlock(ROWS, COLS, x1),
                                     ScaleToBlock(OUT_ROWS, OUT_COLS, coeffs1));
      ComputeScaledDCT<COLS, ROWS>()(FromBlock(COLS, ROWS, x2),
                                     ScaleToBlock(OUT_ROWS, OUT_COLS, coeffs2));

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

void TestRectTranspose() {
  TestRectTransposeT<16, 32>(1e-6f);
  TestRectTransposeT<8, 32>(1e-6f);
  TestRectTransposeT<8, 16>(1e-6f);
  TestRectTransposeT<4, 8>(1e-6f);
  TestRectTransposeT<2, 4>(1e-6f);
  TestRectTransposeT<1, 4>(1e-6f);
  TestRectTransposeT<1, 2>(1e-6f);

  // Identical to 8, 16
  //  TestRectTranspose<16, 8>(1e-6f);
}

void TestDctAccuracyShard(size_t shard) {
  if (shard == 0) {
    TestDctAccuracy<8>(1.1E-7f);
    TestDctAccuracy<16>(1.1E-7f);
  }
  TestDctAccuracy<32>(1.1E-7f, 32 * shard, 32 * (shard + 1));
}

void TestIdctAccuracyShard(size_t shard) {
  if (shard == 0) {
    TestIdctAccuracy<8>(1E-7f);
    TestIdctAccuracy<16>(1E-7f);
  }
  TestIdctAccuracy<32>(1E-7f, 32 * shard, 32 * (shard + 1));
}

void TestDctTransposeShard(size_t shard) {
  if (shard == 0) {
    TestDctTranspose<8>(1E-6f);
    TestDctTranspose<16>(1E-6f);
  }
  TestDctTranspose<32>(3E-6f, 32 * shard, 32 * (shard + 1));
}

void TestSlowIsSameDCTShard(size_t shard) {
  if (shard == 0) {
    TestSlowIsSameDCT<2>(1E-5f);
    TestSlowIsSameDCT<4>(1E-5f);
    TestSlowIsSameDCT<8>(1E-5f);
    TestSlowIsSameDCT<16>(3E-5f);
  }
  TestSlowIsSameDCT<32>(1E-4f, 32 * shard, 32 * (shard + 1));
}

void TestSlowIsSameIDCTShard(size_t shard) {
  if (shard == 0) {
    TestSlowIsSameIDCT<2>(1E-5f);
    TestSlowIsSameIDCT<4>(1E-5f);
    TestSlowIsSameIDCT<8>(1E-5f);
    TestSlowIsSameIDCT<16>(2E-5f);
  }
  TestSlowIsSameIDCT<32>(1E-4f, 32 * shard, 32 * (shard + 1));
}

void TestSlowInverseShard(size_t shard) {
  if (shard == 0) {
    TestSlowInverse<2>(1E-5f);
    TestSlowInverse<4>(1E-5f);
    TestSlowInverse<8>(1E-5f);
    TestSlowInverse<16>(1E-5f);
  }
  TestSlowInverse<32>(1E-5f, 32 * shard, 32 * (shard + 1));
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {

class TransposeTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(TransposeTest);

HWY_EXPORT_AND_TEST_P(TransposeTest, TransposeTest)
HWY_EXPORT_AND_TEST_P(TransposeTest, InverseTest)
HWY_EXPORT_AND_TEST_P(TransposeTest, IDCTOrthonormalTest)
HWY_EXPORT_AND_TEST_P(TransposeTest, DCTOrthonormalTest)
HWY_EXPORT_AND_TEST_P(TransposeTest, ColumnDctRoundtrip)
HWY_EXPORT_AND_TEST_P(TransposeTest, TestRectInverse)
HWY_EXPORT_AND_TEST_P(TransposeTest, TestRectTranspose)

// Tests in the DctShardedTest class are sharded for N=32.
class DctShardedTest : public ::hwy::TestWithParamTargetAndT<uint32_t> {};

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

HWY_TARGET_INSTANTIATE_TEST_SUITE_P_T(DctShardedTest,
                                      ::testing::ValuesIn(ShardRange(32)));

HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestDctAccuracyShard)
HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestIdctAccuracyShard)
HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestDctTransposeShard)
HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestSlowIsSameDCTShard)
HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestSlowIsSameIDCTShard)
HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestSlowInverseShard)

}  // namespace jxl
#endif  // HWY_ONCE
