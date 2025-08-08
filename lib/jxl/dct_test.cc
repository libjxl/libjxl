// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/test_memory_manager.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/dct_test.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>
#include <hwy/tests/hwy_gtest.h>

#include "lib/jxl/dct-inl.h"
#include "lib/jxl/dct_for_test.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// Computes the in-place NxN DCT of block.
// Requires that block is HWY_ALIGN'ed.
//
// Performs ComputeTransposedScaledDCT and then transposes and scales it to
// obtain "vanilla" DCT.
template <size_t N>
void ComputeDCT(float block[N * N]) {
  constexpr size_t kBlockSize = N * N;
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory tmp_block_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* tmp_block = tmp_block_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory scratch_space_mem,
      AlignedMemory::Create(memory_manager, 4 * kBlockSize * sizeof(float)));
  float* scratch_space = scratch_space_mem.address<float>();

  ComputeScaledDCT<N, N>()(DCTFrom(block, N), tmp_block, scratch_space);

  // Untranspose.
  Transpose<N, N>::Run(DCTFrom(tmp_block, N), DCTTo(block, N));
}

// Computes the in-place 8x8 iDCT of block.
// Requires that block is HWY_ALIGN'ed.
template <int N>
void ComputeIDCT(float block[N * N]) {
  constexpr size_t kBlockSize = N * N;
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory tmp_block_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* tmp_block = tmp_block_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory scratch_space_mem,
      AlignedMemory::Create(memory_manager, 4 * kBlockSize * sizeof(float)));
  float* scratch_space = scratch_space_mem.address<float>();
  // Untranspose.
  Transpose<N, N>::Run(DCTFrom(block, N), DCTTo(tmp_block, N));

  ComputeScaledIDCT<N, N>()(tmp_block, DCTTo(block, N), scratch_space);
}

template <size_t N>
void TransposeTestT(float accuracy) {
  constexpr size_t kBlockSize = N * N;
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory src_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* src = src_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory dst_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* dst = dst_mem.address<float>();

  DCTTo to_src(src, N);
  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      to_src.Write(y * N + x, y, x);
    }
  }
  Transpose<N, N>::Run(DCTFrom(src, N), DCTTo(dst, N));
  DCTFrom from_dst(dst, N);
  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
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
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  // Though we are only interested in single column result, dct.h has built-in
  // limit on minimal number of columns processed. So, to be safe, we do
  // regular 8x8 block transformation. On the bright side - we could check all
  // 8 basis vectors at once.
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory block_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* block = block_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory tmp_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* tmp = tmp_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory scratch_mem,
      AlignedMemory::Create(memory_manager, 3 * kBlockSize * sizeof(float)));
  float* scratch = scratch_mem.address<float>();

  DCTTo to(block, N);
  DCTFrom from(block, N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      to.Write((i == j) ? 1.0f : 0.0f, i, j);
    }
  }

  // Running (I)DCT on the same memory block seems to trigger a compiler bug on
  // ARMv7 with clang6.
  DCTTo to_tmp(tmp, N);
  DCTFrom from_tmp(tmp, N);

  DCT1D<N, N>()(from, to_tmp, scratch);
  IDCT1D<N, N>()(from_tmp, to, scratch);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      float actual = from.Read(i, j);
      EXPECT_NEAR(expected, actual, accuracy) << " i=" << i << ", j=" << j;
    }
  }
}

void ColumnDctRoundtrip() {
  ColumnDctRoundtripT<8>(1e-6f);
  ColumnDctRoundtripT<16>(1e-6f);
  ColumnDctRoundtripT<32>(1e-6f);
}

template <size_t N>
void TestDctAccuracy(float accuracy, size_t start = 0, size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory fast_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* fast = fast_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory slow_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(double)));
  double* slow = slow_mem.address<double>();
  for (size_t i = start; i < end; i++) {
    memset(fast, 0, kBlockSize * sizeof(float));
    memset(slow, 0, kBlockSize * sizeof(double));
    fast[i] = 1.0;
    slow[i] = 1.0;
    DCTSlow<N>(slow);
    ComputeDCT<N>(fast);
    for (size_t k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(fast[k], slow[k], accuracy / N)
          << "i = " << i << ", k = " << k << ", N = " << N;
    }
  }
}

template <size_t N>
void TestIdctAccuracy(float accuracy, size_t start = 0, size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory fast_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* fast = fast_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory slow_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(double)));
  double* slow = slow_mem.address<double>();
  for (size_t i = start; i < end; i++) {
    memset(fast, 0, kBlockSize * sizeof(float));
    memset(slow, 0, kBlockSize * sizeof(double));
    fast[i] = 1.0;
    slow[i] = 1.0;
    IDCTSlow<N>(slow);
    ComputeIDCT<N>(fast);
    for (size_t k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(fast[k], slow[k], accuracy * N)
          << "i = " << i << ", k = " << k << ", N = " << N;
    }
  }
}

template <size_t N>
void TestInverseT(float accuracy) {
  constexpr size_t kBlockSize = N * N;
  test::ThreadPoolForTests pool(N < 32 ? 0 : 8);
  const auto process_block = [accuracy](const uint32_t task,
                                        size_t /*thread*/) -> Status {
    JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
    JXL_TEST_ASSIGN_OR_DIE(
        AlignedMemory mem,
        AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
    float* x = mem.address<float>();
    const size_t i = static_cast<size_t>(task);
    memset(x, 0, kBlockSize * sizeof(float));
    x[i] = 1.0;

    ComputeIDCT<N>(x);
    ComputeDCT<N>(x);

    for (size_t k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(x[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", k = " << k;
    }
    return true;
  };
  EXPECT_TRUE(RunOnPool(pool.get(), 0, kBlockSize, ThreadPool::NoInit,
                        process_block, "TestInverse"));
}

void InverseTest() {
  TestInverseT<8>(1e-6f);
  TestInverseT<16>(1e-6f);
  TestInverseT<32>(3e-6f);
}

template <size_t N>
void TestDctTranspose(float accuracy, size_t start = 0, size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  static_assert(kBlockSize >= 64, "Unsupported block size");  // for alignment
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory mem,
      AlignedMemory::Create(memory_manager, 2 * kBlockSize * sizeof(float)));
  float* x = mem.address<float>();
  float* y = x + kBlockSize;
  for (size_t i = start; i < end; i++) {
    for (size_t j = 0; j < kBlockSize; ++j) {
      // We check that <e_i, Me_j> = <M^\dagger{}e_i, e_j>.
      // That means (Me_j)_i = (M^\dagger{}e_i)_j

      // x := Me_j
      memset(x, 0, kBlockSize * sizeof(float));
      x[j] = 1.0;
      ComputeIDCT<N>(x);
      // y := M^\dagger{}e_i
      memset(y, 0, kBlockSize * sizeof(float));
      y[i] = 1.0;
      ComputeDCT<N>(y);

      EXPECT_NEAR(x[i] / N, y[j] * N, accuracy) << "i = " << i << ", j = " << j;
    }
  }
}

template <size_t N>
void TestSlowInverse(float accuracy, size_t start = 0, size_t end = N * N) {
  constexpr size_t kBlockSize = N * N;
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory x_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(double)));
  double* x = x_mem.address<double>();
  for (size_t i = start; i < end; i++) {
    memset(x, 0, kBlockSize * sizeof(double));
    x[i] = 1.0;

    DCTSlow<N>(x);
    IDCTSlow<N>(x);

    for (size_t k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(x[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", k = " << k;
    }
  }
}

template <size_t ROWS, size_t COLS>
JXL_NOINLINE void NoinlineScaledDCT(float* x, float* coeffs,
                                    float* scratch_space) {
  ComputeScaledDCT<ROWS, COLS>()(DCTFrom(x, COLS), coeffs, scratch_space);
}

template <size_t ROWS, size_t COLS>
JXL_NOINLINE void NoinlineScaledIDCT(float* coeffs, float* out,
                                     float* scratch_space) {
  ComputeScaledIDCT<ROWS, COLS>()(coeffs, DCTTo(out, COLS), scratch_space);
}

template <size_t ROWS, size_t COLS>
void TestRectInverseT(float accuracy) {
  constexpr size_t kBlockSize = ROWS * COLS;
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory x_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* x = x_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory out_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* out = out_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory coeffs_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* coeffs = coeffs_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory scratch_space_mem,
      AlignedMemory::Create(memory_manager, 5 * kBlockSize * sizeof(float)));
  float* scratch_space = scratch_space_mem.address<float>();
  for (size_t i = 0; i < kBlockSize; ++i) {
    memset(x, 0, kBlockSize * sizeof(float));
    memset(out, 0, kBlockSize * sizeof(float));
    x[i] = 1.0;
    memset(coeffs, 0, kBlockSize * sizeof(float));
    memset(scratch_space, 0, 5 * kBlockSize * sizeof(float));

    NoinlineScaledDCT<ROWS, COLS>(x, coeffs, scratch_space);
    NoinlineScaledIDCT<ROWS, COLS>(coeffs, out, scratch_space);

    for (size_t k = 0; k < kBlockSize; ++k) {
      EXPECT_NEAR(out[k], (k == i) ? 1.0f : 0.0f, accuracy)
          << "i = " << i << ", k = " << k << " ROWS = " << ROWS
          << " COLS = " << COLS;
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
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory scratch_space_mem,
      AlignedMemory::Create(memory_manager, 5 * kBlockSize * sizeof(float)));
  float* scratch_space = scratch_space_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory x1_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* x1 = x1_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory x2_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* x2 = x2_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory coeffs1_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* coeffs1 = coeffs1_mem.address<float>();
  JXL_TEST_ASSIGN_OR_DIE(
      AlignedMemory coeffs2_mem,
      AlignedMemory::Create(memory_manager, kBlockSize * sizeof(float)));
  float* coeffs2 = coeffs2_mem.address<float>();

  for (size_t px = 0; px < COLS; ++px) {
    for (size_t py = 0; py < ROWS; ++py) {
      memset(x1, 0, sizeof(float) * kBlockSize);
      memset(x2, 0, sizeof(float) * kBlockSize);
      memset(coeffs1, 0, sizeof(float) * kBlockSize);
      memset(coeffs2, 0, sizeof(float) * kBlockSize);
      x1[py * COLS + px] = 1;
      x2[px * ROWS + py] = 1;

      constexpr size_t OUT_ROWS = ROWS < COLS ? ROWS : COLS;
      constexpr size_t OUT_COLS = ROWS < COLS ? COLS : ROWS;

      NoinlineScaledDCT<ROWS, COLS>(x1, coeffs1, scratch_space);
      NoinlineScaledDCT<COLS, ROWS>(x2, coeffs2, scratch_space);

      for (size_t x = 0; x < OUT_COLS; ++x) {
        for (size_t y = 0; y < OUT_ROWS; ++y) {
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
    TestDctAccuracy<1>(1.1E-7f);
    TestDctAccuracy<2>(1.1E-7f);
    TestDctAccuracy<4>(1.1E-7f);
    TestDctAccuracy<8>(1.1E-7f);
    TestDctAccuracy<16>(1.3E-7f);
  }
  TestDctAccuracy<32>(1.1E-7f, 32 * shard, 32 * (shard + 1));
}

void TestIdctAccuracyShard(size_t shard) {
  if (shard == 0) {
    TestIdctAccuracy<1>(1E-7f);
    TestIdctAccuracy<2>(1E-7f);
    TestIdctAccuracy<4>(1E-7f);
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

void TestSlowInverseShard(size_t shard) {
  if (shard == 0) {
    TestSlowInverse<1>(1E-5f);
    TestSlowInverse<2>(1E-5f);
    TestSlowInverse<4>(1E-5f);
    TestSlowInverse<8>(1E-5f);
    TestSlowInverse<16>(1E-5f);
  }
  TestSlowInverse<32>(1E-5f, 32 * shard, 32 * (shard + 1));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

class TransposeTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(TransposeTest);

HWY_EXPORT_AND_TEST_P(TransposeTest, TransposeTest);
HWY_EXPORT_AND_TEST_P(TransposeTest, InverseTest);
HWY_EXPORT_AND_TEST_P(TransposeTest, ColumnDctRoundtrip);
HWY_EXPORT_AND_TEST_P(TransposeTest, TestRectInverse);
HWY_EXPORT_AND_TEST_P(TransposeTest, TestRectTranspose);

// Tests in the DctShardedTest class are sharded for N=32.
class DctShardedTest : public ::hwy::TestWithParamTargetAndT<uint32_t> {};

template <size_t n>
std::vector<uint32_t> ShardRange() {
#ifdef JXL_DISABLE_SLOW_TESTS
  static_assert(n > 6, "'large' range is too small");
  std::vector<uint32_t> ret = {0, 1, 3, 5, n - 1};
#else
  std::vector<uint32_t> ret(n);
  std::iota(ret.begin(), ret.end(), 0);
#endif  // JXL_DISABLE_SLOW_TESTS
  return ret;
}

HWY_TARGET_INSTANTIATE_TEST_SUITE_P_T(DctShardedTest,
                                      ::testing::ValuesIn(ShardRange<32>()));

HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestDctAccuracyShard);
HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestIdctAccuracyShard);
HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestDctTransposeShard);
HWY_EXPORT_AND_TEST_P_T(DctShardedTest, TestSlowInverseShard);

}  // namespace jxl
#endif  // HWY_ONCE
