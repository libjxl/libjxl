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

#ifndef JXL_BASE_FILE_IO_H_
#define JXL_BASE_FILE_IO_H_

// Helper functions for reading/writing files.

#include <stdio.h>

#include <string>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/status.h"

namespace jxl {

// Returns extension including the dot, or empty string if none. Assumes
// filename is not a hidden file (e.g. ".bashrc"). May be called with a pathname
// if the filename contains a dot and/or no other path component does.
static inline std::string Extension(const std::string& filename) {
  const size_t pos = filename.rfind('.');
  if (pos == std::string::npos) return std::string();
  return filename.substr(pos);
}

// RAII, ensures files are closed even when returning early.
class FileWrapper {
 public:
  FileWrapper(const FileWrapper& other) = delete;
  FileWrapper& operator=(const FileWrapper& other) = delete;

  explicit FileWrapper(const std::string& pathname, const char* mode)
      : file_(fopen(pathname.c_str(), mode)) {}

  ~FileWrapper() {
    if (file_ != nullptr) {
      const int err = fclose(file_);
      JXL_CHECK(err == 0);
    }
  }

  // We intend to use FileWrapper as a replacement of FILE.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator FILE*() const { return file_; }

 private:
  FILE* const file_;
};

template <typename ContainerType>
static inline Status ReadFile(const std::string& pathname,
                              ContainerType* JXL_RESTRICT bytes) {
  FileWrapper f(pathname, "rb");
  if (f == nullptr) return JXL_FAILURE("Failed to open file for reading");

  if (fseek(f, 0, SEEK_END) != 0) return JXL_FAILURE("Failed to seek end");
  bytes->resize(ftell(f));
  if (bytes->empty()) return JXL_FAILURE("Zero-length file");
  if (fseek(f, 0, SEEK_SET) != 0) return JXL_FAILURE("Failed to seek set");

  size_t pos = 0;
  while (pos < bytes->size()) {
    // Needed in case ContainerType is std::string, whose data() is const.
    char* bytes_writable = reinterpret_cast<char*>(&(*bytes)[0]);
    const size_t bytes_read =
        fread(bytes_writable + pos, 1, bytes->size() - pos, f);
    if (bytes_read == 0) return JXL_FAILURE("Failed to read");
    pos += bytes_read;
  }
  JXL_ASSERT(pos == bytes->size());
  return true;
}

template <typename ContainerType>
static inline Status WriteFile(const ContainerType& bytes,
                               const std::string& pathname) {
  FileWrapper f(pathname, "wb");
  if (f == nullptr) return JXL_FAILURE("Failed to open file for writing");

  size_t pos = 0;
  while (pos < bytes.size()) {
    const size_t bytes_written =
        fwrite(bytes.data() + pos, 1, bytes.size() - pos, f);
    if (bytes_written == 0) return JXL_FAILURE("Failed to write");
    pos += bytes_written;
  }
  JXL_ASSERT(pos == bytes.size());

  return true;
}

}  // namespace jxl

#endif  // JXL_BASE_FILE_IO_H_
