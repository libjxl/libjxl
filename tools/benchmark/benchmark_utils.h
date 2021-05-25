// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_BENCHMARK_BENCHMARK_UTILS_H_
#define TOOLS_BENCHMARK_BENCHMARK_UTILS_H_

#include <string>
#include <vector>

#include "lib/jxl/base/status.h"

namespace jxl {

class TemporaryFile final {
 public:
  explicit TemporaryFile(std::string basename, std::string extension);
  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;
  ~TemporaryFile();
  Status GetFileName(std::string* output) const;

 private:
  bool ok_ = true;

  std::string temp_filename_;
};

Status RunCommand(const std::string& command,
                  const std::vector<std::string>& arguments);

}  // namespace jxl

#endif  // TOOLS_BENCHMARK_BENCHMARK_UTILS_H_
