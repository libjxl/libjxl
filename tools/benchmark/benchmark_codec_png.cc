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
#include "tools/benchmark/benchmark_codec_png.h"

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "lib/extras/codec_png.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/base/time.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {

struct PNGArgs {
  // Empty, no PNG-specific args currently.
};

static PNGArgs* const pngargs = new PNGArgs;

Status AddCommandLineOptionsPNGCodec(BenchmarkArgs* args) { return true; }

// Lossless.
class PNGCodec : public ImageCodec {
 public:
  explicit PNGCodec(const BenchmarkArgs& args) : ImageCodec(args) {}

  Status ParseParam(const std::string& param) override { return true; }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPoolInternal* pool, PaddedBytes* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    const size_t bits = io->metadata.m.bit_depth.bits_per_sample;
    const double start = Now();
    JXL_RETURN_IF_ERROR(
        EncodeImagePNG(io, io->Main().c_current(), bits, pool, compressed));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& /*filename*/,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    const double start = Now();
    JXL_RETURN_IF_ERROR(DecodeImagePNG(compressed, pool, io));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }
};

ImageCodec* CreateNewPNGCodec(const BenchmarkArgs& args) {
  return new PNGCodec(args);
}

}  // namespace jxl
