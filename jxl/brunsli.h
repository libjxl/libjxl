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

#ifndef JXL_BRUNSLI_H_
#define JXL_BRUNSLI_H_

#include <cstdint>
#include <functional>
#include <string>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/codec_in_out.h"
#include "jxl/dec_params.h"

// Utilities for rasterization of intermediate Brunsli representation.

namespace jxl {

enum class BrunsliFileSignature {
  kBrunsli,
  kNotEnoughData,
  kInvalid,
};

BrunsliFileSignature IsBrunsliFile(jxl::Span<const uint8_t> compressed);

struct BrunsliEncoderOptions {
  float quant_scale = 1.0f;
  std::string hdr_orig_colorspace;
};

struct BrunsliDecoderMeta {
  std::string hdr_orig_colorspace;
};

Status BrunsliToPixels(jxl::Span<const uint8_t> compressed,
                       jxl::CodecInOut* JXL_RESTRICT io,
                       const BrunsliDecoderOptions& options,
                       BrunsliDecoderMeta* metadata, jxl::ThreadPool* pool);

Status PixelsToBrunsli(const jxl::CodecInOut* JXL_RESTRICT io,
                       jxl::PaddedBytes* compressed,
                       const BrunsliEncoderOptions& options,
                       jxl::ThreadPool* pool);

// Actual encoder has a lot of brunsli-specific state. Let's not expose it.
class BrunsliFrameEncoderInternal;

class BrunsliFrameEncoder {
 public:
  BrunsliFrameEncoder(const FrameDimensions& frame_dim, ThreadPool* pool);
  ~BrunsliFrameEncoder();
  bool ReadSourceImage(const ImageBundle* src,
                       const std::vector<int>& quant_table,
                       YCbCrChromaSubsampling subsampling);
  bool DoEncode();
  bool SerializeHeader(BitWriter* out);
  bool SerializeDcGroup(size_t index, BitWriter* out, AuxOut* aux_out);
  bool SerializeAcGroup(size_t index, BitWriter* out, AuxOut* aux_out);

 private:
  std::unique_ptr<BrunsliFrameEncoderInternal> impl_;
};

// Actual decoder has a lot of brunsli-specific state. Let's not expose it.
class BrunsliFrameDecoderInternal;

class BrunsliFrameDecoder {
 public:
  explicit BrunsliFrameDecoder(ThreadPool* pool);
  ~BrunsliFrameDecoder();

  bool ReadHeader(const FrameDimensions* frame_dim, BitReader* src,
                  YCbCrChromaSubsampling subsampling);
  bool DecodeDcGroup(int idx, BitReader* src);
  bool DecodeAcGroup(int idx, BitReader* src, Image3F* img, const Rect& rect);
  bool FinalizeDecoding(const FrameHeader& frame_header, Image3F&& opsin,
                        ImageBundle* decoded);

 private:
  std::unique_ptr<BrunsliFrameDecoderInternal> impl_;
};

}  // namespace jxl

#endif  // JXL_BRUNSLI_H_
