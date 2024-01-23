// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/benchmark/benchmark_codec_png.h"

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/extras/dec/apng.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/apng.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "tools/benchmark/benchmark_args.h"
#include "tools/benchmark/benchmark_codec.h"
#include "tools/speed_stats.h"
#include "tools/thread_pool_internal.h"

namespace jpegxl {
namespace tools {

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

  Status Compress(const std::string& filename, const PackedPixelFile& ppf,
                  ThreadPool* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    const double start = jxl::Now();
    JXL_RETURN_IF_ERROR(
        jxl::Encode(ppf, jxl::extras::Codec::kPNG, compressed, pool));
    const double end = jxl::Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& /*filename*/,
                    const Span<const uint8_t> compressed, ThreadPool* pool,
                    PackedPixelFile* ppf,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    const double start = jxl::Now();
    JXL_RETURN_IF_ERROR(jxl::extras::DecodeImageAPNG(
        compressed, jxl::extras::ColorHints(), ppf));
    const double end = jxl::Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }
};

ImageCodec* CreateNewPNGCodec(const BenchmarkArgs& args) {
  if (jxl::extras::GetAPNGEncoder() &&
      jxl::extras::CanDecode(jxl::extras::Codec::kPNG)) {
    return new PNGCodec(args);
  } else {
    return nullptr;
  }
}

}  // namespace tools
}  // namespace jpegxl
