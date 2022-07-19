// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/benchmark/benchmark_codec_custom.h"

// Not supported on Windows due to Linux-specific functions.
#ifndef _WIN32

#include <libgen.h>

#include <fstream>

#include "lib/extras/codec.h"
#include "lib/extras/enc/apng.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/image_bundle.h"
#include "tools/benchmark/benchmark_utils.h"

namespace jxl {
namespace {

std::string GetBaseName(std::string filename) {
  std::string result = std::move(filename);
  result = basename(&result[0]);
  const size_t dot = result.rfind('.');
  if (dot != std::string::npos) {
    result.resize(dot);
  }
  return result;
}

// This uses `output_filename` to determine the name of the corresponding
// `.time` file.
template <typename F>
Status ReportCodecRunningTime(F&& function, std::string output_filename,
                              jpegxl::tools::SpeedStats* const speed_stats) {
  const double start = Now();
  JXL_RETURN_IF_ERROR(function());
  const double end = Now();
  const std::string time_filename =
      GetBaseName(std::move(output_filename)) + ".time";
  std::ifstream time_stream(time_filename);
  double time;
  if (time_stream >> time) {
    // Report the time measured by the external codec itself.
    speed_stats->NotifyElapsed(time);
  } else {
    // Fall back to the less accurate time that we measured.
    speed_stats->NotifyElapsed(end - start);
  }
  if (time_stream.is_open()) {
    remove(time_filename.c_str());
  }
  return true;
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
                  ThreadPoolInternal* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    JXL_RETURN_IF_ERROR(param_index_ > 2);

    const std::string basename = GetBaseName(filename);
    TemporaryFile png_file(basename, "png"), encoded_file(basename, extension_);
    std::string png_filename, encoded_filename;
    JXL_RETURN_IF_ERROR(png_file.GetFileName(&png_filename));
    JXL_RETURN_IF_ERROR(encoded_file.GetFileName(&encoded_filename));
    saved_intensity_target_ = io->metadata.m.IntensityTarget();

    const size_t bits = io->metadata.m.bit_depth.bits_per_sample;
    JXL_RETURN_IF_ERROR(
        EncodeToFile(*io, io->Main().c_current(), bits, png_filename, pool));
    std::vector<std::string> arguments = compress_args_;
    arguments.push_back(png_filename);
    arguments.push_back(encoded_filename);
    JXL_RETURN_IF_ERROR(ReportCodecRunningTime(
        [&, this] { return RunCommand(compress_command_, arguments); },
        encoded_filename, speed_stats));
    return ReadFile(encoded_filename, compressed);
  }

  Status Decompress(const std::string& filename,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    const std::string basename = GetBaseName(filename);
    TemporaryFile encoded_file(basename, extension_), png_file(basename, "png");
    std::string encoded_filename, png_filename;
    JXL_RETURN_IF_ERROR(encoded_file.GetFileName(&encoded_filename));
    JXL_RETURN_IF_ERROR(png_file.GetFileName(&png_filename));

    JXL_RETURN_IF_ERROR(WriteFile(compressed, encoded_filename));
    JXL_RETURN_IF_ERROR(ReportCodecRunningTime(
        [&, this] {
          return RunCommand(
              decompress_command_,
              std::vector<std::string>{encoded_filename, png_filename});
        },
        png_filename, speed_stats));
    JXL_RETURN_IF_ERROR(
        SetFromFile(png_filename, extras::ColorHints(), io, pool));
    io->metadata.m.SetIntensityTarget(saved_intensity_target_);
    return true;
  }

 private:
  std::string extension_;
  std::string compress_command_;
  std::string decompress_command_;
  std::vector<std::string> compress_args_;
  int param_index_ = 0;
  int saved_intensity_target_ = 255;
};

}  // namespace

ImageCodec* CreateNewCustomCodec(const BenchmarkArgs& args) {
  return new CustomCodec(args);
}

}  // namespace jxl

#else

namespace jxl {

ImageCodec* CreateNewCustomCodec(const BenchmarkArgs& args) { return nullptr; }

}  // namespace jxl

#endif  // _MSC_VER
