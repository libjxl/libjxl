// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/file_io.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

namespace jpegxl {
namespace tools {

bool ReadFile(const char* filename, std::vector<uint8_t>* out) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    return false;
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return false;
  }

  long size = ftell(file);
  // Avoid invalid file or directory.
  if (size >= LONG_MAX || size < 0) {
    fclose(file);
    return false;
  }

  if (fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return false;
  }

  out->resize(size);
  size_t readsize = fread(out->data(), 1, size, file);
  if (fclose(file) != 0) {
    return false;
  }

  return readsize == static_cast<size_t>(size);
}

bool WriteFile(const char* filename, const std::vector<uint8_t>& bytes) {
  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr,
            "Could not open %s for writing\n"
            "Error: %s",
            filename, strerror(errno));
    return false;
  }
  if (fwrite(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
    fprintf(stderr,
            "Could not write to file\n"
            "Error: %s",
            strerror(errno));
    return false;
  }
  if (fclose(file) != 0) {
    fprintf(stderr,
            "Could not close file\n"
            "Error: %s",
            strerror(errno));
    return false;
  }
  return true;
}

}  // namespace tools
}  // namespace jpegxl
