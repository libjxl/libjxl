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

#ifndef JXL_MODULAR_MEMIO_H_
#define JXL_MODULAR_MEMIO_H_

#include "jxl/base/compiler_specific.h"
#include "jxl/base/padded_bytes.h"

namespace jxl {

/*!
 * Read-only IO interface for a constant memory block
 */
class BlobReader {
 private:
  const uint8_t* const JXL_RESTRICT data;
  const size_t data_array_size;
  size_t seek_pos;

 public:
  const int EOS = -1;

  BlobReader(const uint8_t* _data, size_t _data_array_size)
      : data(_data), data_array_size(_data_array_size), seek_pos(0) {}

  bool isEOF() const { return seek_pos > data_array_size; }
  long ftell() const { return seek_pos; }
  const uint8_t* ptr() const { return data; }
  size_t size() const { return data_array_size; }

  int get_c() {
    if (JXL_UNLIKELY(seek_pos >= data_array_size)) {
      if (seek_pos == data_array_size) seek_pos++;
      return EOS;
    }
    return data[seek_pos++];
  }
  char* gets(char* buf, int n) {
    int i = 0;
    const int max_write = n - 1;
    while (seek_pos < data_array_size && i < max_write)
      buf[i++] = data[seek_pos++];
    buf[n - 1] = '\0';

    if (i < max_write)
      return nullptr;
    else
      return buf;
  }
  void fseek(long offset, int where) {
    switch (where) {
      case SEEK_SET:
        seek_pos = offset;
        break;
      case SEEK_CUR:
        seek_pos += offset;
        break;
      case SEEK_END:
        seek_pos = long(data_array_size) + offset;
        break;
    }
  }
};

/*!
 * IO interface for a growable memory block
 */
class BlobWriter {
 private:
  size_t pos = 0;

 public:
  jxl::PaddedBytes blob;
  BlobWriter() = default;

  int ftell() const { return pos; }
  const uint8_t* ptr() const { return blob.data(); }
  size_t size() const { return blob.size(); }
  void append(const BlobWriter& other) {
    blob.resize(pos);
    pos = size() + other.size();
    blob.append(other.blob);
  }
  void append(const jxl::PaddedBytes& other) {
    blob.resize(pos);
    pos = size() + other.size();
    blob.append(other);
  }
  int fputs(const char* s) {
    size_t i = 0;
    // null-terminated string
    while (s[i]) {
      fputc(s[i]);
      i++;
    }
    return 0;
  }
  int fputc(int c) {
    if (pos == size())
      blob.push_back(static_cast<uint8_t>(c));
    else
      blob[pos] = static_cast<uint8_t>(c);
    pos++;
    return c;
  }
  void fseek(long offset, int where) {
    switch (where) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos += offset;
        break;
      case SEEK_END:
        pos = size() + offset;
        break;
    }
  }
};

/*!
 * Dummy writer
 */
class DummyWriter {
 public:
  const int EOS = -1;

  DummyWriter() = default;

  int ftell() const { return -1; }
  const uint8_t* ptr() const { return nullptr; }
  void append(const jxl::PaddedBytes& other) {}
  int fputs(const char* s) { return 0; }
  int fputc(int c) { return c; }
  void fseek(long offset, int where) {}
};

}  // namespace jxl

#endif  // JXL_MODULAR_MEMIO_H_
