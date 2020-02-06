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

#include "tools/benchmark/benchmark_codec_custom.h"

#include <libgen.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "jxl/base/file_io.h"
#include "jxl/codec_in_out.h"
#include "jxl/extras/codec_png.h"
#include "jxl/image_bundle.h"

extern char** environ;

namespace jxl {
namespace {

class TemporaryFile final {
 public:
  explicit TemporaryFile(std::string basename, std::string extension)
      : temp_filename_(std::move(basename) + "_XXXXXX"),
        extension_('.' + std::move(extension)) {
    const int fd = mkstemp(&temp_filename_[0]);
    if (fd == -1) {
      ok_ = false;
      return;
    }
    close(fd);
  }
  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;
  ~TemporaryFile() {
    if (ok_) {
      unlink(temp_filename_.c_str());
      unlink((temp_filename_ + extension_).c_str());
    }
  }

  Status GetFileName(std::string* const output) const {
    JXL_RETURN_IF_ERROR(ok_);
    *output = temp_filename_ + extension_;
    return true;
  }

 private:
  bool ok_ = true;

  // Name of a file that is kept existing to ensure name uniqueness.
  std::string temp_filename_;

  // Extension to add to the filename when giving it to the clients of this
  // class.
  const std::string extension_;
};

std::string GetBaseName(std::string filename) {
  std::string result = std::move(filename);
  result = basename(&result[0]);
  const size_t dot = result.rfind('.');
  if (dot != std::string::npos) {
    result.resize(dot);
  }
  return result;
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

class CustomCodec : public ImageCodec {
 public:
  explicit CustomCodec(const BenchmarkArgs& args) : ImageCodec(args) {}

  Status ParseParam(const std::string& param) override {
    switch (param_index_) {
      case 0:
        extension_ = param;
        break;

      case 1:
        compress_command_ = param;
        break;

      case 2:
        decompress_command_ = param;
        break;

      default:
        compress_args_.push_back(param);
        break;
    }
    ++param_index_;
    return true;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPool* pool, PaddedBytes* compressed) override {
    JXL_RETURN_IF_ERROR(param_index_ > 2);

    const std::string basename = GetBaseName(filename);
    TemporaryFile png_file(basename, "png"), encoded_file(basename, extension_);
    std::string png_filename, encoded_filename;
    JXL_RETURN_IF_ERROR(png_file.GetFileName(&png_filename));
    JXL_RETURN_IF_ERROR(encoded_file.GetFileName(&encoded_filename));

    const size_t bits = io->metadata.bits_per_sample;
    PaddedBytes png;
    JXL_RETURN_IF_ERROR(
        EncodeImagePNG(io, io->Main().c_current(), bits, pool, &png));
    JXL_RETURN_IF_ERROR(WriteFile(png, png_filename));
    std::vector<std::string> arguments = compress_args_;
    arguments.push_back(png_filename);
    arguments.push_back(encoded_filename);
    JXL_RETURN_IF_ERROR(RunCommand(compress_command_, arguments));
    return ReadFile(encoded_filename, compressed);
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed, ThreadPool* pool,
                    CodecInOut* io) override {
    const std::string basename = GetBaseName(filename);
    TemporaryFile encoded_file(basename, extension_), png_file(basename, "png");
    std::string encoded_filename, png_filename;
    JXL_RETURN_IF_ERROR(encoded_file.GetFileName(&encoded_filename));
    JXL_RETURN_IF_ERROR(png_file.GetFileName(&png_filename));

    JXL_RETURN_IF_ERROR(WriteFile(compressed, encoded_filename));
    JXL_RETURN_IF_ERROR(
        RunCommand(decompress_command_,
                   std::vector<std::string>{encoded_filename, png_filename}));
    PaddedBytes png;
    JXL_RETURN_IF_ERROR(ReadFile(png_filename, &png));
    return DecodeImagePNG(Span<const uint8_t>(png), pool, io);
  }

  void GetMoreStats(BenchmarkStats* const stats) override {
    // Time measurements are unreliable because of the intermediary PNG step.
    stats->total_time_encode = 0;
    stats->total_time_decode = 0;
  }

 private:
  std::string extension_;
  std::string compress_command_;
  std::string decompress_command_;
  std::vector<std::string> compress_args_;
  int param_index_ = 0;
};

}  // namespace

ImageCodec* CreateNewCustomCodec(const BenchmarkArgs& args) {
  return new CustomCodec(args);
}

}  // namespace jxl
