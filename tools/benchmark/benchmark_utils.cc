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

#define _DEFAULT_SOURCE  // for mkstemps().

#include "tools/benchmark/benchmark_utils.h"

// Not supported on Windows due to Linux-specific functions.
// Not supported in Android NDK before API 28.
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__) && \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 28)

#include <libgen.h>
#include <spawn.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fstream>

#include "lib/extras/codec_png.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/image_bundle.h"

extern char** environ;

namespace jxl {
TemporaryFile::TemporaryFile(std::string basename, std::string extension) {
  const auto extension_size = 1 + extension.size();
  temp_filename_ = std::move(basename) + "_XXXXXX." + std::move(extension);
  const int fd = mkstemps(&temp_filename_[0], extension_size);
  if (fd == -1) {
    ok_ = false;
    return;
  }
  close(fd);
}
TemporaryFile::~TemporaryFile() {
  if (ok_) {
    unlink(temp_filename_.c_str());
  }
}

Status TemporaryFile::GetFileName(std::string* const output) const {
  JXL_RETURN_IF_ERROR(ok_);
  *output = temp_filename_;
  return true;
}

Status RunCommand(const std::string& command,
                  const std::vector<std::string>& arguments) {
  std::vector<char*> args;
  args.reserve(arguments.size() + 2);
  args.push_back(const_cast<char*>(command.c_str()));
  for (const std::string& argument : arguments) {
    args.push_back(const_cast<char*>(argument.c_str()));
  }
  args.push_back(nullptr);
  pid_t pid;
  JXL_RETURN_IF_ERROR(posix_spawnp(&pid, command.c_str(), nullptr, nullptr,
                                   args.data(), environ) == 0);
  int wstatus;
  waitpid(pid, &wstatus, 0);
  return WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == EXIT_SUCCESS;
}

}  // namespace jxl

#else

namespace jxl {

TemporaryFile::TemporaryFile(std::string basename, std::string extension) {}
TemporaryFile::~TemporaryFile() {}
Status TemporaryFile::GetFileName(std::string* const output) const {
  (void)ok_;
  return JXL_FAILURE("Not supported on this build");
}

Status RunCommand(const std::string& command,
                  const std::vector<std::string>& arguments) {
  return JXL_FAILURE("Not supported on this build");
}

}  // namespace jxl

#endif  // _MSC_VER
