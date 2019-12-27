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

#ifndef JXL_IMAGE_H_
#define JXL_IMAGE_H_

// SIMD/multicore-friendly planar image representation with row accessors.

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "jxl/base/cache_aligned.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/profiler.h"
#include "jxl/base/robust_statistics.h"
#include "jxl/base/status.h"

namespace jxl {

// Each row address is a multiple of this - enables aligned loads.
static constexpr size_t kImageAlign = CacheAligned::kAlignment;
static_assert(kImageAlign >= kMaxVectorSize, "Insufficient alignment");

// Returns distance [bytes] between the start of two consecutive rows, a
// multiple of kAlign but NOT CacheAligned::kAlias - see below.
//
// Differing "kAlign" make sense for:
// - Image: 128 to avoid false sharing/RFOs between multiple threads processing
//   rows independently;
// - TileFlow: no cache line alignment needed because buffers are per-thread;
//   just need kMaxVectorSize=16..64 for SIMD.
//
// "valid_bytes" is xsize * sizeof(T).
template <size_t kAlign>
static inline size_t BytesPerRow(size_t valid_bytes) {
  static_assert((kAlign & (kAlign - 1)) == 0, "kAlign should be power of two");

  // Extra two vectors allow *writing* a partial or full vector on the right AND
  // left border (for convolve.h) without disturbing the next/previous row.
  const size_t row_size = valid_bytes + 2 * kMaxVectorSize;

  // Round up.
  size_t bytes_per_row = (row_size + kAlign - 1) & ~(kAlign - 1);

  // During the lengthy window before writes are committed to memory, CPUs
  // guard against read after write hazards by checking the address, but
  // only the lower 11 bits. We avoid a false dependency between writes to
  // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
  // Avoid2K prevents the same problem for the planes of an Image3.
  if (bytes_per_row % CacheAligned::kAlias == 0) {
    bytes_per_row += kImageAlign;
  }

  return bytes_per_row;
}

// Factored out of Image<> to avoid dependency on profiler.h and <atomic>.
CacheAlignedUniquePtr AllocateImageBytes(size_t size, size_t xsize,
                                         size_t ysize);

// Single channel, aligned rows separated by padding. T must be POD.
//
// Rationale: vectorization benefits from aligned operands - unaligned loads and
// especially stores are expensive when the address crosses cache line
// boundaries. Introducing padding after each row ensures the start of a row is
// aligned, and that row loops can process entire vectors (writes to the padding
// are allowed and ignored).
//
// We prefer a planar representation, where channels are stored as separate
// 2D arrays, because that simplifies vectorization (repeating the same
// operation on multiple adjacent components) without the complexity of a
// hybrid layout (8 R, 8 G, 8 B, ...). In particular, clients can easily iterate
// over all components in a row and Image requires no knowledge of the pixel
// format beyond the component type "T".
//
// This image layout could also be achieved with a vector and a row accessor
// function, but a class wrapper with support for "deleter" allows wrapping
// existing memory allocated by clients without copying the pixels. It also
// provides convenient accessors for xsize/ysize, which shortens function
// argument lists. Supports move-construction so it can be stored in containers.
template <typename ComponentType>
class Plane {
 public:
  using T = ComponentType;
  static constexpr size_t kNumPlanes = 1;

  Plane() : xsize_(0), ysize_(0), bytes_per_row_(0), bytes_(nullptr) {}

  Plane(const size_t xsize, const size_t ysize)
      : xsize_(static_cast<uint32_t>(xsize)),
        ysize_(static_cast<uint32_t>(ysize)),
        bytes_per_row_(BytesPerRow<kImageAlign>(xsize * sizeof(T))),
        bytes_(nullptr) {
    JXL_ASSERT(bytes_per_row_ % kImageAlign == 0);
    // xsize and/or ysize can legitimately be zero, in which case we don't
    // want to allocate.
    if (xsize != 0 && ysize != 0) {
      bytes_ = AllocateImageBytes(bytes_per_row_ * ysize + kMaxVectorSize,
                                  xsize, ysize);
    }

#ifdef MEMORY_SANITIZER
    // Only in MSAN builds: ensure full vectors are initialized.
    if (xsize != 0 && ysize != 0) {
      const size_t partial = (xsize_ * sizeof(T)) % kMaxVectorSize;
      const size_t remainder = (partial == 0) ? 0 : (kMaxVectorSize - partial);
      for (size_t y = 0; y < ysize_; ++y) {
#if defined(__clang__) && (__clang_major__ <= 6)
        // There's a bug in msan in clang-6 when handling AVX2 operations. This
        // workaround allows pass tests on msan, although it is slower and
        // prevents msan warnings from uninitialized images.
        memset(Row(y), 0, (xsize_ * sizeof(T)) + remainder);
#else
        memset(Row(y) + xsize_, 0, remainder);
#endif
      }
    }
#endif
  }

  // Copy construction/assignment is forbidden to avoid inadvertent copies,
  // which can be very expensive. Use CopyImageTo() instead.
  Plane(const Plane& other) = delete;
  Plane& operator=(const Plane& other) = delete;

  // Move constructor (required for returning Image from function)
  Plane(Plane&& other) noexcept = default;

  // Move assignment (required for std::vector)
  Plane& operator=(Plane&& other) noexcept = default;

  void Swap(Plane& other) {
    std::swap(xsize_, other.xsize_);
    std::swap(ysize_, other.ysize_);
    std::swap(bytes_per_row_, other.bytes_per_row_);
    std::swap(bytes_, other.bytes_);
  }

  // Useful for pre-allocating image with some padding for alignment purposes
  // and later reporting the actual valid dimensions. Caller is responsible
  // for ensuring xsize/ysize are <= the original dimensions.
  void ShrinkTo(const size_t xsize, const size_t ysize) {
    xsize_ = static_cast<uint32_t>(xsize);
    ysize_ = static_cast<uint32_t>(ysize);
    // NOTE: we can't recompute bytes_per_row for more compact storage and
    // better locality because that would invalidate the image contents.
  }

  // How many pixels.
  JXL_INLINE size_t xsize() const { return xsize_; }
  JXL_INLINE size_t ysize() const { return ysize_; }

  // Returns pointer to the start of a row, with at least xsize (rounded up to
  // kImageAlign bytes) accessible values.
  JXL_INLINE T* JXL_RESTRICT Row(const size_t y) {
    RowBoundsCheck(y);
    void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<T*>(JXL_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to non-const - required for writing to individual planes
  // of an Image3.
  JXL_INLINE T* JXL_RESTRICT MutableRow(const size_t y) const {
    RowBoundsCheck(y);
    void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<T*>(JXL_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to const (see above).
  JXL_INLINE const T* JXL_RESTRICT Row(const size_t y) const {
    RowBoundsCheck(y);
    const void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<const T*>(JXL_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to const (see above), even if called on a non-const Image.
  JXL_INLINE const T* JXL_RESTRICT ConstRow(const size_t y) const {
    RowBoundsCheck(y);
    const void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<const T*>(JXL_ASSUME_ALIGNED(row, 64));
  }

  // Raw access to byte contents, for interfacing with other libraries.
  // Unsigned char instead of char to avoid surprises (sign extension).
  JXL_INLINE uint8_t* bytes() {
    void* p = bytes_.get();
    return static_cast<uint8_t * JXL_RESTRICT>(JXL_ASSUME_ALIGNED(p, 64));
  }
  JXL_INLINE const uint8_t* bytes() const {
    const void* p = bytes_.get();
    return static_cast<const uint8_t * JXL_RESTRICT>(JXL_ASSUME_ALIGNED(p, 64));
  }

  // NOTE: do not use this for copying rows - the valid xsize may be much less.
  JXL_INLINE size_t bytes_per_row() const { return bytes_per_row_; }

  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic. WARNING: this must
  // NOT be used to determine xsize. NOTE: this is less efficient than
  // ByteOffset(row, bytes_per_row).
  JXL_INLINE intptr_t PixelsPerRow() const {
    static_assert(kImageAlign % sizeof(T) == 0,
                  "Padding must be divisible by the pixel size.");
    return static_cast<intptr_t>(bytes_per_row_ / sizeof(T));
  }

  // Extra metadata about this plane.
  struct Extra {};

  Extra& GetExtra() {
    if (extra_ == nullptr) {
      extra_.reset(new Extra);
    }
    return *extra_;
  }

 private:
  void RowBoundsCheck(const size_t y) const {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
    if (y >= ysize_) {
      Abort(__FILE__, __LINE__, "Row(%zu) >= %u\n", y, ysize_);
    }
#endif
  }

  // (Members are non-const to enable assignment during move-assignment.)
  uint32_t xsize_;  // In valid pixels, not including any padding.
  uint32_t ysize_;
  size_t bytes_per_row_;  // Includes padding.
  CacheAlignedUniquePtr bytes_;
  std::unique_ptr<Extra> extra_;
};

using ImageB = Plane<uint8_t>;
using ImageS = Plane<int16_t>;  // signed integer or half-float
using ImageU = Plane<uint16_t>;
using ImageI = Plane<int32_t>;
using ImageF = Plane<float>;
using ImageD = Plane<double>;

// We omit unnecessary fields and choose smaller representations to reduce L1
// cache pollution.
#pragma pack(push, 1)

// Size of an image in pixels. POD.
struct ImageSize {
  static ImageSize Make(const size_t xsize, const size_t ysize) {
    ImageSize ret;
    ret.xsize = static_cast<uint32_t>(xsize);
    ret.ysize = static_cast<uint32_t>(ysize);
    return ret;
  }

  bool operator==(const ImageSize& other) const {
    return xsize == other.xsize && ysize == other.ysize;
  }

  uint32_t xsize;
  uint32_t ysize;
};

#pragma pack(pop)

template <typename T>
void CopyImageTo(const Plane<T>& from, Plane<T>* JXL_RESTRICT to) {
  PROFILER_ZONE("CopyImage1");
  JXL_ASSERT(SameSize(from, *to));
  for (size_t y = 0; y < from.ysize(); ++y) {
    const T* JXL_RESTRICT row_from = from.ConstRow(y);
    T* JXL_RESTRICT row_to = to->Row(y);
    memcpy(row_to, row_from, from.xsize() * sizeof(T));
  }
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Plane<T> CopyImage(const Plane<T>& from) {
  Plane<T> to(from.xsize(), from.ysize());
  CopyImageTo(from, &to);
  return to;
}

// Also works for Image3 and mixed argument types.
template <class Image1, class Image2>
bool SameSize(const Image1& image1, const Image2& image2) {
  return image1.xsize() == image2.xsize() && image1.ysize() == image2.ysize();
}

template <typename T>
class Image3;

// Rectangular region in image(s). Factoring this out of Image instead of
// shifting the pointer by x0/y0 allows this to apply to multiple images with
// different resolutions (e.g. color transform and quantization field).
// Can compare using SameSize(rect1, rect2).
class Rect {
 public:
  // Most windows are xsize_max * ysize_max, except those on the borders where
  // begin + size_max > end.
  constexpr Rect(size_t xbegin, size_t ybegin, size_t xsize_max,
                 size_t ysize_max, size_t xend, size_t yend)
      : x0_(xbegin),
        y0_(ybegin),
        xsize_(ClampedSize(xbegin, xsize_max, xend)),
        ysize_(ClampedSize(ybegin, ysize_max, yend)) {}

  // Construct with origin and known size (typically from another Rect).
  constexpr Rect(size_t xbegin, size_t ybegin, size_t xsize, size_t ysize)
      : x0_(xbegin), y0_(ybegin), xsize_(xsize), ysize_(ysize) {}

  // Construct a rect that covers a whole image/plane/ImageBundle etc.
  template <typename Image>
  explicit Rect(const Image& image)
      : Rect(0, 0, image.xsize(), image.ysize()) {}

  Rect() : Rect(0, 0, 0, 0) {}

  Rect(const Rect&) = default;
  Rect& operator=(const Rect&) = default;

  Rect Subrect(size_t xbegin, size_t ybegin, size_t xsize_max,
               size_t ysize_max) {
    return Rect(x0_ + xbegin, y0_ + ybegin, xsize_max, ysize_max, x0_ + xsize_,
                y0_ + ysize_);
  }

  template <typename T>
  T* Row(Plane<T>* image, size_t y) const {
    return image->Row(y + y0_) + x0_;
  }

  template <typename T>
  T* MutableRow(const Plane<T>* image, size_t y) const {
    return image->MutableRow(y + y0_) + x0_;
  }

  template <typename T>
  T* PlaneRow(Image3<T>* image, const size_t c, size_t y) const {
    return image->PlaneRow(c, y + y0_) + x0_;
  }

  template <typename T>
  const T* ConstRow(const Plane<T>& image, size_t y) const {
    return image.ConstRow(y + y0_) + x0_;
  }

  template <typename T>
  const T* ConstPlaneRow(const Image3<T>& image, size_t c, size_t y) const {
    return image.ConstPlaneRow(c, y + y0_) + x0_;
  }

  // Returns true if this Rect fully resides in the given image. ImageT could be
  // Image<T> or Image3<T>; however if ImageT is Rect, results are nonsensical.
  template <class ImageT>
  bool IsInside(const ImageT& image) const {
    return (x0_ + xsize_ <= image.xsize()) && (y0_ + ysize_ <= image.ysize());
  }

  size_t x0() const { return x0_; }
  size_t y0() const { return y0_; }
  size_t xsize() const { return xsize_; }
  size_t ysize() const { return ysize_; }

 private:
  // Returns size_max, or whatever is left in [begin, end).
  static constexpr size_t ClampedSize(size_t begin, size_t size_max,
                                      size_t end) {
    return (begin + size_max <= end) ? size_max
                                     : (end > begin ? end - begin : 0);
  }

  size_t x0_;
  size_t y0_;

  size_t xsize_;
  size_t ysize_;
};

// Copies `from:rect_from` to `to`.
template <typename T>
void CopyImageTo(const Rect& rect_from, const Plane<T>& from,
                 const Plane<T>* JXL_RESTRICT to) {
  PROFILER_ZONE("CopyImageR");
  JXL_ASSERT(SameSize(rect_from, *to));
  for (size_t y = 0; y < rect_from.ysize(); ++y) {
    const T* JXL_RESTRICT row_from = rect_from.ConstRow(from, y);
    T* JXL_RESTRICT row_to = to->MutableRow(y);
    memcpy(row_to, row_from, rect_from.xsize() * sizeof(T));
  }
}

template <typename T>
void CopyImageTo(const Plane<T>& from, const Rect& rect_to,
                 const Plane<T>* JXL_RESTRICT to) {
  PROFILER_ZONE("CopyImageR");
  JXL_ASSERT(SameSize(from, rect_to));
  for (size_t y = 0; y < rect_to.ysize(); ++y) {
    const T* JXL_RESTRICT row_from = from.Row(y);
    T* JXL_RESTRICT row_to = rect_to.MutableRow(to, y);
    memcpy(row_to, row_from, rect_to.xsize() * sizeof(T));
  }
}

// DEPRECATED - Returns a copy of the "image" pixels that lie in "rect".
template <typename T>
Plane<T> CopyImage(const Rect& rect, const Plane<T>& image) {
  Plane<T> copy(rect.xsize(), rect.ysize());
  CopyImageTo(rect, image, &copy);
  return copy;
}

// Currently, we abuse Image to either refer to an image that owns its storage
// or one that doesn't. In similar vein, we abuse Image* function parameters to
// either mean "assign to me" or "fill the provided image with data".
// Hopefully, the "assign to me" meaning will go away and most images in the
// codebase will not be backed by own storage. When this happens we can redesign
// Image to be a non-storage-holding view class and introduce BackedImage in
// those places that actually need it.

// NOTE: we can't use Image as a view because invariants are violated
// (alignment and the presence of padding before/after each "row").

// A bundle of 3 same-sized images. Typically constructed by moving from three
// rvalue references to Image. To overwrite an existing Image3 using
// single-channel producers, we also need access to Image*. Constructing
// temporary non-owning Image pointing to one plane of an existing Image3 risks
// dangling references, especially if the wrapper is moved. Therefore, we
// store an array of Image (which are compact enough that size is not a concern)
// and provide a Plane+MutableRow accessors.
template <typename ComponentType>
class Image3 {
 public:
  using T = ComponentType;
  using PlaneT = Plane<T>;
  static constexpr size_t kNumPlanes = 3;

  Image3() : planes_{PlaneT(), PlaneT(), PlaneT()} {}

  Image3(const size_t xsize, const size_t ysize)
      : planes_{PlaneT(xsize, ysize), PlaneT(xsize, ysize),
                PlaneT(xsize, ysize)} {}

  Image3(Image3&& other) noexcept {
    for (size_t i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
  }

  Image3(PlaneT&& plane0, PlaneT&& plane1, PlaneT&& plane2) {
    JXL_CHECK(SameSize(plane0, plane1));
    JXL_CHECK(SameSize(plane0, plane2));
    planes_[0] = std::move(plane0);
    planes_[1] = std::move(plane1);
    planes_[2] = std::move(plane2);
  }

  // Copy construction/assignment is forbidden to avoid inadvertent copies,
  // which can be very expensive. Use CopyImageTo instead.
  Image3(const Image3& other) = delete;
  Image3& operator=(const Image3& other) = delete;

  Image3& operator=(Image3&& other) noexcept {
    for (size_t i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
    return *this;
  }

  // Returns row pointer; usage: PlaneRow(idx_plane, y)[x] = val.
  JXL_INLINE T* JXL_RESTRICT PlaneRow(const size_t c, const size_t y) {
    // Custom implementation instead of calling planes_[c].Row ensures only a
    // single multiplication is needed for PlaneRow(0..2, y).
    PlaneRowBoundsCheck(c, y);
    const size_t row_offset = y * planes_[0].bytes_per_row();
    void* row = planes_[c].bytes() + row_offset;
    return static_cast<T*>(JXL_ASSUME_ALIGNED(row, 64));
  }

  // Returns const row pointer; usage: val = PlaneRow(idx_plane, y)[x].
  JXL_INLINE const T* JXL_RESTRICT PlaneRow(const size_t c,
                                            const size_t y) const {
    PlaneRowBoundsCheck(c, y);
    const size_t row_offset = y * planes_[0].bytes_per_row();
    const void* row = planes_[c].bytes() + row_offset;
    return static_cast<const T*>(JXL_ASSUME_ALIGNED(row, 64));
  }

  // Returns const row pointer, even if called from a non-const Image3.
  JXL_INLINE const T* JXL_RESTRICT ConstPlaneRow(const size_t c,
                                                 const size_t y) const {
    return PlaneRow(c, y);
  }

  JXL_INLINE const PlaneT& Plane(size_t idx) const { return planes_[idx]; }

  void Swap(Image3& other) {
    for (size_t c = 0; c < 3; ++c) {
      other.planes_[c].Swap(planes_[c]);
    }
  }

  void ShrinkTo(const size_t xsize, const size_t ysize) {
    for (PlaneT& plane : planes_) {
      plane.ShrinkTo(xsize, ysize);
    }
  }

  // Sizes of all three images are guaranteed to be equal.
  JXL_INLINE size_t xsize() const { return planes_[0].xsize(); }
  JXL_INLINE size_t ysize() const { return planes_[0].ysize(); }
  // Returns offset [bytes] from one row to the next row of the same plane.
  // WARNING: this must NOT be used to determine xsize, nor for copying rows -
  // the valid xsize may be much less.
  JXL_INLINE size_t bytes_per_row() const { return planes_[0].bytes_per_row(); }
  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic. WARNING: this must NOT be used
  // to determine xsize. NOTE: this is less efficient than
  // ByteOffset(row, bytes_per_row).
  JXL_INLINE intptr_t PixelsPerRow() const { return planes_[0].PixelsPerRow(); }

 private:
  void PlaneRowBoundsCheck(const size_t c, const size_t y) const {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
    if (c >= kNumPlanes || y >= ysize()) {
      Abort(__FILE__, __LINE__, "PlaneRow(%zu, %zu) >= %zu\n", c, y, ysize());
    }
#endif
  }

 private:
  PlaneT planes_[kNumPlanes];
};

using Image3B = Image3<uint8_t>;
using Image3S = Image3<int16_t>;
using Image3U = Image3<uint16_t>;
using Image3I = Image3<int32_t>;
using Image3F = Image3<float>;
using Image3D = Image3<double>;

// Copies `from:rect_from` to `to:rect_to`.
template <typename T>
void CopyImageTo(const Rect& rect_from, const Image3<T>& from,
                 const Rect& rect_to, Image3<T>* JXL_RESTRICT to) {
  PROFILER_ZONE("CopyImageR");
  JXL_ASSERT(SameSize(rect_from, rect_to));
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < rect_to.ysize(); ++y) {
      const T* JXL_RESTRICT row_from = rect_from.ConstPlaneRow(from, c, y);
      T* JXL_RESTRICT row_to = rect_to.PlaneRow(to, c, y);
      memcpy(row_to, row_from, rect_to.xsize() * sizeof(T));
    }
  }
}

template <typename T>
void CopyImageTo(const Image3<T>& from, Image3<T>* JXL_RESTRICT to) {
  return CopyImageTo(Rect(from), from, Rect(*to), to);
}

template <typename T>
void CopyImageTo(const Rect& rect_from, const Image3<T>& from,
                 Image3<T>* JXL_RESTRICT to) {
  return CopyImageTo(rect_from, from, Rect(*to), to);
}

// Copies `from` to `to:rect_to`.
template <typename T>
void CopyImageTo(const Image3<T>& from, const Rect& rect_to,
                 Image3<T>* JXL_RESTRICT to) {
  return CopyImageTo(Rect(from), from, rect_to, to);
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Image3<T> CopyImage(const Image3<T>& from) {
  Image3<T> copy(from.xsize(), from.ysize());
  CopyImageTo(from, &copy);
  return copy;
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Image3<T> CopyImage(const Rect& rect, const Image3<T>& from) {
  Image3<T> to(rect.xsize(), rect.ysize());
  CopyImageTo(rect, from.Plane(0), const_cast<ImageF*>(&to.Plane(0)));
  CopyImageTo(rect, from.Plane(1), const_cast<ImageF*>(&to.Plane(1)));
  CopyImageTo(rect, from.Plane(2), const_cast<ImageF*>(&to.Plane(2)));
  return to;
}

}  // namespace jxl

#endif  // JXL_IMAGE_H_
