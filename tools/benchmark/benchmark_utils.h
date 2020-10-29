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
