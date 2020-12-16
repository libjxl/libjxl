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

// Functions for writing a JPEGData object into a jpeg byte stream.

#ifndef LIB_JXL_JPEG_DEC_JPEG_DATA_WRITER_H_
#define LIB_JXL_JPEG_DEC_JPEG_DATA_WRITER_H_

#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {

// Function pointer type used to write len bytes into buf. Returns the
// number of bytes written.
typedef size_t (*JPEGOutputHook)(void* data, const uint8_t* buf, size_t len);

// Output callback function with associated data.
struct JPEGOutput {
  JPEGOutput(JPEGOutputHook cb, void* data) : cb(cb), data(data) {}
  bool Write(const uint8_t* buf, size_t len) const {
    if (len == 0) return true;
    size_t bytes_written = cb(data, buf, len);
    return (bytes_written == len);
  }

 private:
  JPEGOutputHook cb;
  void* data;
};

bool WriteJpeg(const JPEGData& jpg, JPEGOutput out);

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_DEC_JPEG_DATA_WRITER_H_
