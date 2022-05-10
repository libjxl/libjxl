// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_BENCHMARK_BENCHMARK_CODEC_H_
#define TOOLS_BENCHMARK_BENCHMARK_CODEC_H_

#include <stdint.h>

#include <deque>
#include <string>
#include <vector>

#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/image.h"
#include "tools/args.h"
#include "tools/benchmark/benchmark_args.h"
#include "tools/benchmark/benchmark_stats.h"
#include "tools/cmdline.h"
#include "tools/speed_stats.h"

namespace jxl {

// Thread-compatible.
class ImageCodec {
 public:
  explicit ImageCodec(const BenchmarkArgs& args)
      : args_(args),
        butteraugli_target_(1.0f),
        q_target_(100.0f),
        bitrate_target_(0.0f) {}

  virtual ~ImageCodec() = default;

  void set_description(const std::string& desc) { description_ = desc; }
  const std::string& description() const { return description_; }

  const ButteraugliParams& BaParams() const { return ba_params_; }

  virtual void ParseParameters(const std::string& parameters);

  virtual Status ParseParam(const std::string& param);

  // Returns true iff the codec instance (including parameters) can tolerate
  // ImageBundle c_current() != metadata()->color_encoding, and the possibility
  // of negative (out of gamut) pixel values.
  virtual bool IsColorAware() const { return false; }

  // Returns true iff the codec instance (including parameters) will operate
  // only with quantized DCT (JPEG) coefficients in input.
  virtual bool IsJpegTranscoder() const { return false; }

  virtual Status Compress(const std::string& filename, const CodecInOut* io,
                          ThreadPoolInternal* pool,
                          std::vector<uint8_t>* compressed,
                          jpegxl::tools::SpeedStats* speed_stats) = 0;

  virtual Status Decompress(const std::string& filename,
                            const Span<const uint8_t> compressed,
                            ThreadPoolInternal* pool, CodecInOut* io,
                            jpegxl::tools::SpeedStats* speed_stats) = 0;

  virtual void GetMoreStats(BenchmarkStats* stats) {}

  virtual Status CanRecompressJpeg() const { return false; }
  virtual Status RecompressJpeg(const std::string& filename,
                                const std::string& data,
                                std::vector<uint8_t>* compressed,
                                jpegxl::tools::SpeedStats* speed_stats) {
    return false;
  }

  virtual std::string GetErrorMessage() const { return error_message_; }

 protected:
  const BenchmarkArgs& args_;
  std::string params_;
  std::string description_;
  float butteraugli_target_;
  float q_target_;
  float bitrate_target_;
  ButteraugliParams ba_params_;
  std::string error_message_;
};

using ImageCodecPtr = std::unique_ptr<ImageCodec>;

// Creates an image codec by name, e.g. "jxl" to get a new instance of the
// jxl codec. Optionally, behind a colon, parameters can be specified,
// then ParseParameters of the codec gets called with the part behind the colon.
ImageCodecPtr CreateImageCodec(const std::string& description);

}  // namespace jxl

#endif  // TOOLS_BENCHMARK_BENCHMARK_CODEC_H_
