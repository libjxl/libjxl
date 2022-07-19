// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#if JPEGXL_ENABLE_APNG

#include "tools/benchmark/benchmark_codec_png.h"

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "lib/extras/codec.h"
#include "lib/extras/dec/apng.h"
#include "lib/extras/enc/apng.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/extras/time.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"

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
                  ThreadPoolInternal* pool, std::vector<uint8_t>* compressed,
                  jpegxl::tools::SpeedStats* speed_stats) override {
    const size_t bits = io->metadata.m.bit_depth.bits_per_sample;
    const double start = Now();
    JXL_RETURN_IF_ERROR(Encode(*io, extras::Codec::kPNG, io->Main().c_current(),
                               bits, compressed, pool));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    return true;
  }

  Status Decompress(const std::string& /*filename*/,
                    const Span<const uint8_t> compressed,
                    ThreadPoolInternal* pool, CodecInOut* io,
                    jpegxl::tools::SpeedStats* speed_stats) override {
    extras::PackedPixelFile ppf;
    const double start = Now();
    JXL_RETURN_IF_ERROR(extras::DecodeImageAPNG(
        compressed, extras::ColorHints(), SizeConstraints(), &ppf));
    const double end = Now();
    speed_stats->NotifyElapsed(end - start);
    JXL_RETURN_IF_ERROR(ConvertPackedPixelFileToCodecInOut(ppf, pool, io));
    return true;
  }
};

ImageCodec* CreateNewPNGCodec(const BenchmarkArgs& args) {
  return new PNGCodec(args);
}

}  // namespace jxl

#endif
