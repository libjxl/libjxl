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

#ifndef JXL_BLOCK_H_
#define JXL_BLOCK_H_

// Adapters for DCT input/output: from/to contiguous blocks or image rows.

#include <hwy/static_targets.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/common.h"

namespace jxl {

// Adapters for source/destination.
//
// Block: (x, y) <-> (N * y + x)
// Lines: (x, y) <-> (stride * y + x)
//
// I.e. Block is a specialization of Lines with fixed stride.
//
// FromXXX should implement Read and Load (Read vector).
// ToXXX should implement Write and Store (Write vector).

template <size_t N>
using BlockDesc = HWY_CAPPED(float, N);

// Here and in the following, the SZ template parameter specifies the number of
// values to load/store. Needed because we want to handle 4x4 sub-blocks of
// 16x16 blocks.
template <size_t ROWS, size_t COLS = ROWS>
class FromBlock {
 public:
  explicit FromBlock(const float* block) : block_(block) {}

  FromBlock View(size_t dx, size_t dy) const {
    return FromBlock<ROWS, COLS>(Address(dx, dy));
  }

  template <size_t SZ>
  HWY_ATTR HWY_INLINE hwy::VT<BlockDesc<SZ>> LoadPart(const size_t row,
                                                      size_t i) const {
    static_assert(SZ <= COLS, "Requesting more values that are present!");
    return Load(BlockDesc<SZ>(), block_ + row * COLS + i);
  }

  HWY_ATTR HWY_INLINE hwy::VT<BlockDesc<COLS>> LoadVec(const size_t row,
                                                       size_t i) const {
    return LoadPart<COLS>(row, i);
  }

  HWY_ATTR HWY_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  constexpr HWY_INLINE const float* Address(const size_t row,
                                            const size_t i) const {
    return block_ + row * COLS + i;
  }

 private:
  const float* block_;
};

template <size_t ROWS, size_t COLS = ROWS>
class ToBlock {
 public:
  explicit ToBlock(float* block) : block_(block) {}

  ToBlock View(size_t dx, size_t dy) const {
    return ToBlock<ROWS, COLS>(Address(dx, dy));
  }

  template <size_t SZ>
  HWY_ATTR HWY_INLINE void StorePart(const hwy::VT<BlockDesc<SZ>>& v,
                                     const size_t row, const size_t i) const {
    static_assert(SZ <= COLS, "Requesting more values that are present!");
    Store(v, BlockDesc<SZ>(), Address(row, i));
  }

  HWY_ATTR HWY_INLINE void StoreVec(const hwy::VT<BlockDesc<COLS>>& v,
                                    const size_t row, size_t i) const {
    return StorePart<COLS>(v, row, i);
  }

  HWY_ATTR HWY_INLINE void Write(float v, const size_t row,
                                 const size_t i) const {
    *Address(row, i) = v;
  }

  constexpr HWY_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * COLS + i;
  }

 private:
  float* block_;
};

// Same as ToBlock, but multiplies result by (N * N)
template <size_t ROWS, size_t COLS = ROWS>
class ScaleToBlock {
 public:
  explicit HWY_ATTR ScaleToBlock(float* block) : block_(block) {}

  ScaleToBlock View(size_t dx, size_t dy) const {
    return ScaleToBlock<ROWS, COLS>(Address(dx, dy));
  }

  template <size_t SZ>
  HWY_ATTR HWY_INLINE void StorePart(const hwy::VT<BlockDesc<SZ>>& v,
                                     const size_t row, const size_t i) const {
    using BlockDesc = jxl::BlockDesc<SZ>;
    const auto mul_ = Set(BlockDesc(), 1.0f / (COLS * ROWS));
    Store(v * mul_, BlockDesc(), Address(row, i));
  }

  HWY_ATTR HWY_INLINE void StoreVec(const hwy::VT<BlockDesc<COLS>>& v,
                                    const size_t row, size_t i) const {
    return StorePart<COLS>(v, row, i);
  }

  HWY_ATTR HWY_INLINE void Write(float v, const size_t row,
                                 const size_t i) const {
    constexpr float mul_ = 1.0f / (COLS * ROWS);
    *Address(row, i) = v * mul_;
  }

  constexpr HWY_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * COLS + i;
  }

 private:
  float* block_;
};

template <size_t N>
class FromLines {
 public:
  FromLines(const float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  FromLines View(size_t dx, size_t dy) const {
    return FromLines(Address(dx, dy), stride_);
  }

  template <size_t SZ>
  HWY_ATTR HWY_INLINE hwy::VT<BlockDesc<SZ>> LoadPart(const size_t row,
                                                      const size_t i) const {
    return Load(BlockDesc<SZ>(), Address(row, i));
  }

  HWY_ATTR HWY_INLINE hwy::VT<BlockDesc<N>> LoadVec(const size_t row,
                                                    size_t i) const {
    return LoadPart<N>(row, i);
  }

  HWY_ATTR HWY_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  HWY_INLINE const float* Address(const size_t row, const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  const float* HWY_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

// Pointers are restrict-qualified: assumes we don't use both FromLines and
// ToLines in the same DCT. NOTE: Transpose uses From/ToBlock, not *Lines.
template <size_t N>
class ToLines {
 public:
  ToLines(float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  ToLines View(const ToLines& other, size_t dx, size_t dy) const {
    return ToLines(Address(dx, dy), stride_);
  }

  template <size_t SZ>
  HWY_ATTR HWY_INLINE void StorePart(const hwy::VT<BlockDesc<SZ>>& v,
                                     const size_t row, const size_t i) const {
    Store(v, BlockDesc<SZ>(), Address(row, i));
  }

  HWY_ATTR HWY_INLINE void StoreVec(const hwy::VT<BlockDesc<N>>& v,
                                    const size_t row, size_t i) const {
    return StorePart<N>(v, row, i);
  }

  HWY_ATTR HWY_INLINE void Write(float v, const size_t row,
                                 const size_t i) const {
    *Address(row, i) = v;
  }

  HWY_INLINE float* Address(const size_t row, const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  float* HWY_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

}  // namespace jxl

#endif  // JXL_BLOCK_H_
