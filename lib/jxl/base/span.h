// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_SPAN_H_
#define LIB_JXL_BASE_SPAN_H_

// Span (array view) is a non-owning container that provides cheap "cut"
// operations and could be used as "ArrayLike" data source for PaddedBytes.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

template <typename T>
class Span {
 public:
  constexpr Span() noexcept : Span(nullptr, 0) {}

  constexpr Span(T* array, size_t length) noexcept
      : ptr_(array), len_(length) {}

  template <size_t N>
  explicit constexpr Span(T (&a)[N]) noexcept : Span(a, N) {}

  template <typename U>
  constexpr Span(U* array, size_t length) noexcept
      : ptr_(reinterpret_cast<T*>(array)), len_(length) {
    static_assert(sizeof(U) == sizeof(T), "Incompatible type of source.");
  }

  template <typename ArrayLike>
  explicit constexpr Span(const ArrayLike& other) noexcept
      : Span(reinterpret_cast<T*>(other.data()), other.size()) {
    static_assert(sizeof(*other.data()) == sizeof(T),
                  "Incompatible type of source.");
  }

  using NCT = typename std::remove_const<T>::type;

  constexpr T* data() const noexcept { return ptr_; }

  constexpr size_t size() const noexcept { return len_; }

  constexpr bool empty() const noexcept { return len_ == 0; }

  constexpr T* begin() const noexcept { return data(); }

  // The pointer arithmetic backing the bounds-carrying accessors below is the
  // single audited place where it is allowed; it is wrapped so that callers can
  // be compiled with -Wunsafe-buffer-usage while treating Span as safe.
  constexpr T* end() const noexcept {
    JXL_UNSAFE_BUFFER_USAGE_BEGIN
    return data() + size();
    JXL_UNSAFE_BUFFER_USAGE_END
  }

  constexpr T& operator[](size_t i) const noexcept {
    // MSVC 2015 accepts this as constexpr, but not ptr_[i]
    JXL_UNSAFE_BUFFER_USAGE_BEGIN
    return *(data() + i);
    JXL_UNSAFE_BUFFER_USAGE_END
  }

  // Returns a view of the elements starting at `offset`. The caller must ensure
  // `offset <= size()` (and, for the 2-arg form, `offset + count <= size()`).
  Span subspan(size_t offset) const noexcept {
    JXL_UNSAFE_BUFFER_USAGE_BEGIN
    return Span(data() + offset, size() - offset);
    JXL_UNSAFE_BUFFER_USAGE_END
  }

  Span subspan(size_t offset, size_t count) const noexcept {
    JXL_UNSAFE_BUFFER_USAGE_BEGIN
    return Span(data() + offset, count);
    JXL_UNSAFE_BUFFER_USAGE_END
  }

  Status remove_prefix(size_t n) noexcept {
    JXL_ENSURE(size() >= n);
    JXL_UNSAFE_BUFFER_USAGE_BEGIN
    ptr_ += n;
    JXL_UNSAFE_BUFFER_USAGE_END
    len_ -= n;
    return true;
  }

  void AppendTo(std::vector<NCT>& dst) const {
    dst.insert(dst.end(), begin(), end());
  }

  std::vector<NCT> Copy() const { return std::vector<NCT>(begin(), end()); }

 private:
  T* ptr_;
  size_t len_;
};

using Bytes = Span<const uint8_t>;

}  // namespace jxl

#endif  // LIB_JXL_BASE_SPAN_H_
