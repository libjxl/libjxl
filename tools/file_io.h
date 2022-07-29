// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_FILE_IO_H_
#define TOOLS_FILE_IO_H_

#include <stdint.h>

#include <vector>

namespace jpegxl {
namespace tools {

bool ReadFile(const char* filename, std::vector<uint8_t>* out);

bool WriteFile(const char* filename, const std::vector<uint8_t>& bytes);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_FILE_IO_H_
