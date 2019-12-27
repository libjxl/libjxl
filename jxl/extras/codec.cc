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

#include "jxl/extras/codec.h"

#include "jxl/base/file_io.h"
#include "jxl/extras/codec_apng.h"
#if JPEGXL_ENABLE_EXR
#include "jxl/extras/codec_exr.h"
#endif
#include "jxl/extras/codec_gif.h"
#include "jxl/extras/codec_jpg.h"
#include "jxl/extras/codec_pgx.h"
#include "jxl/extras/codec_png.h"
#include "jxl/extras/codec_pnm.h"
#include "jxl/image_bundle.h"

namespace jxl {
namespace {

// Any valid encoding is larger (ensures codecs can read the first few bytes)
constexpr size_t kMinBytes = 9;

}  // namespace

std::string ExtensionFromCodec(Codec codec, const bool is_gray,
                               const size_t bits_per_sample) {
  switch (codec) {
    case Codec::kJPG:
      return ".jpg";
    case Codec::kPGX:
      return ".pgx";
    case Codec::kPNG:
      return ".png";
    case Codec::kPNM:
      if (is_gray) return ".pgm";
      return (bits_per_sample == 32) ? ".pfm" : ".ppm";
    case Codec::kGIF:
      return ".gif";
    case Codec::kEXR:
      return ".exr";
    case Codec::kUnknown:
      return std::string();
  }
  JXL_UNREACHABLE;
  return std::string();
}

Codec CodecFromExtension(const std::string& extension,
                         size_t* JXL_RESTRICT bits_per_sample) {
  if (extension == ".png") return Codec::kPNG;

  if (extension == ".jpg") return Codec::kJPG;
  if (extension == ".jpeg") return Codec::kJPG;

  if (extension == ".pgx") return Codec::kPGX;

  if (extension == ".pgm") return Codec::kPNM;
  if (extension == ".ppm") return Codec::kPNM;
  if (extension == ".pfm") {
    *bits_per_sample = 32;
    return Codec::kPNM;
  }

  if (extension == ".gif") return Codec::kGIF;

  if (extension == ".exr") return Codec::kEXR;

  return Codec::kUnknown;
}

Status SetFromBytes(const Span<const uint8_t> bytes, CodecInOut* io,
                    ThreadPool* pool, const DecodeTarget decode_target) {
  if (bytes.size() < kMinBytes) return JXL_FAILURE("Too few bytes");

  io->metadata.bits_per_sample = 0;  // (For is-set check below)

  if (!DecodeImagePNG(bytes, pool, io) && !DecodeImageAPNG(bytes, io) &&
      !DecodeImagePGX(bytes, pool, io) && !DecodeImagePNM(bytes, pool, io) &&
      !DecodeImageGIF(bytes, io) && !DecodeImageJPG(bytes, io, decode_target)
#if JPEGXL_ENABLE_EXR
      && !DecodeImageEXR(bytes, io)
#endif
  ) {
    return JXL_FAILURE("Codecs failed to decode");
  }

  io->CheckMetadata();
  return true;
}

Status SetFromFile(const std::string& pathname, CodecInOut* io,
                   ThreadPool* pool, const DecodeTarget decode_target) {
  PaddedBytes encoded;
  JXL_RETURN_IF_ERROR(ReadFile(pathname, &encoded));
  JXL_RETURN_IF_ERROR(
      SetFromBytes(Span<const uint8_t>(encoded), io, pool, decode_target));
  return true;
}

Status Encode(const CodecInOut& io, const Codec codec,
              const ColorEncoding& c_desired, size_t bits_per_sample,
              PaddedBytes* bytes, ThreadPool* pool) {
  JXL_CHECK(!io.Main().c_current().icc.empty());
  JXL_CHECK(!c_desired.icc.empty());
  io.CheckMetadata();
  if (io.Main().jpeg_xsize && codec != Codec::kJPG)
    return JXL_FAILURE(
        "Output format has to be JPEG for losslessly recompressed JPEG "
        "reconstruction");

  switch (codec) {
    case Codec::kPNG:
      return EncodeImagePNG(&io, c_desired, bits_per_sample, pool, bytes);
    case Codec::kJPG:
      return EncodeImageJPG(
          &io, io.use_sjpeg ? JpegEncoder::kSJpeg : JpegEncoder::kLibJpeg,
          io.jpeg_quality,
          io.use_sjpeg ? YCbCrChromaSubsampling::kAuto
                       : YCbCrChromaSubsampling::k444,
          pool, bytes,
          io.Main().jpeg_xsize ? DecodeTarget::kQuantizedCoeffs
                               : DecodeTarget::kPixels);
    case Codec::kPNM:
      return EncodeImagePNM(&io, c_desired, bits_per_sample, pool, bytes);
    case Codec::kPGX:
      return EncodeImagePGX(&io, c_desired, bits_per_sample, pool, bytes);
    case Codec::kGIF:
      return JXL_FAILURE("Encoding to GIF is not implemented");
    case Codec::kEXR:
#if JPEGXL_ENABLE_EXR
      return EncodeImageEXR(&io, c_desired, pool, bytes);
#else
      return JXL_FAILURE("JPEG XL was built without OpenEXR support");
#endif
    case Codec::kUnknown:
      return JXL_FAILURE("Cannot encode using Codec::kUnknown");
  }

  return JXL_FAILURE("Invalid codec");
}

Status EncodeToFile(const CodecInOut& io, const ColorEncoding& c_desired,
                    size_t bits_per_sample, const std::string& pathname,
                    ThreadPool* pool) {
  const Codec codec = CodecFromExtension(Extension(pathname), &bits_per_sample);

  PaddedBytes encoded;
  return Encode(io, codec, c_desired, bits_per_sample, &encoded, pool) &&
         WriteFile(encoded, pathname);
}

Status EncodeToFile(const CodecInOut& io, const std::string& pathname,
                    ThreadPool* pool) {
  return EncodeToFile(io, io.metadata.color_encoding,
                      io.metadata.bits_per_sample, pathname, pool);
}

}  // namespace jxl
