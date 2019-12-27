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

#include "jxl/base/data_parallel.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/span.h"
#include "jxl/codec_in_out.h"
#include "jxl/extras/codec_png.h"
#include "jxl/image_bundle.h"

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
                  ThreadPool* pool, PaddedBytes* compressed) override {
    const size_t bits = io->metadata.bits_per_sample;
    return EncodeImagePNG(io, io->Main().c_current(), bits, pool, compressed);
  }

  Status Decompress(const std::string& /*filename*/,
                    const Span<const uint8_t> compressed, ThreadPool* pool,
                    CodecInOut* io) override {
    return DecodeImagePNG(compressed, pool, io);
  }
};

ImageCodec* CreateNewPNGCodec(const BenchmarkArgs& args) {
  return new PNGCodec(args);
}

}  // namespace jxl
