// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/codec.h"

#include "jxl/decode.h"
#include "jxl/types.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"

#if JPEGXL_ENABLE_APNG
#include "lib/extras/enc/apng.h"
#endif
#if JPEGXL_ENABLE_JPEG
#include "lib/extras/enc/jpg.h"
#endif
#if JPEGXL_ENABLE_EXR
#include "lib/extras/enc/exr.h"
#endif

#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/pgx.h"
#include "lib/extras/enc/pnm.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {
namespace {

// Any valid encoding is larger (ensures codecs can read the first few bytes)
constexpr size_t kMinBytes = 9;

}  // namespace

Status SetFromBytes(const Span<const uint8_t> bytes,
                    const extras::ColorHints& color_hints, CodecInOut* io,
                    ThreadPool* pool, extras::Codec* orig_codec) {
  if (bytes.size() < kMinBytes) return JXL_FAILURE("Too few bytes");

  extras::PackedPixelFile ppf;
  if (extras::DecodeBytes(bytes, color_hints, io->constraints, &ppf,
                          orig_codec)) {
    return ConvertPackedPixelFileToCodecInOut(ppf, pool, io);
  }
  return JXL_FAILURE("Codecs failed to decode");
}

Status SetFromFile(const std::string& pathname,
                   const extras::ColorHints& color_hints, CodecInOut* io,
                   ThreadPool* pool, extras::Codec* orig_codec) {
  std::vector<uint8_t> encoded;
  JXL_RETURN_IF_ERROR(ReadFile(pathname, &encoded));
  JXL_RETURN_IF_ERROR(SetFromBytes(Span<const uint8_t>(encoded), color_hints,
                                   io, pool, orig_codec));
  return true;
}

Status Encode(const CodecInOut& io, const extras::Codec codec,
              const ColorEncoding& c_desired, size_t bits_per_sample,
              std::vector<uint8_t>* bytes, ThreadPool* pool) {
  JXL_CHECK(!io.Main().c_current().ICC().empty());
  JXL_CHECK(!c_desired.ICC().empty());
  io.CheckMetadata();
  if (io.Main().IsJPEG()) {
    JXL_WARNING("Writing JPEG data as pixels");
  }
  JxlPixelFormat format = {
      0,  // num_channels is ignored by the converter
      bits_per_sample <= 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16, JXL_BIG_ENDIAN,
      0};
  const bool floating_point = bits_per_sample > 16;
  std::unique_ptr<extras::Encoder> encoder;
  std::ostringstream os;
  switch (codec) {
    case extras::Codec::kPNG:
#if JPEGXL_ENABLE_APNG
      encoder = extras::GetAPNGEncoder();
      break;
#else
      return JXL_FAILURE("JPEG XL was built without (A)PNG support");
#endif
    case extras::Codec::kJPG:
#if JPEGXL_ENABLE_JPEG
      format.data_type = JXL_TYPE_UINT8;
      encoder = extras::GetJPEGEncoder();
      os << io.jpeg_quality;
      encoder->SetOption("q", os.str());
      break;
#else
      return JXL_FAILURE("JPEG XL was built without JPEG support");
#endif
    case extras::Codec::kPNM:
      if (io.Main().HasAlpha()) {
        encoder = extras::GetPAMEncoder();
      } else if (io.Main().IsGray()) {
        encoder = extras::GetPGMEncoder();
      } else if (!floating_point) {
        encoder = extras::GetPPMEncoder();
      } else {
        format.data_type = JXL_TYPE_FLOAT;
        format.endianness = JXL_NATIVE_ENDIAN;
        encoder = extras::GetPFMEncoder();
      }
      if (!c_desired.IsSRGB()) {
        JXL_WARNING(
            "PNM encoder cannot store custom ICC profile; decoder "
            "will need hint key=color_space to get the same values");
      }
      break;
    case extras::Codec::kPGX:
      encoder = extras::GetPGXEncoder();
      break;
    case extras::Codec::kGIF:
      return JXL_FAILURE("Encoding to GIF is not implemented");
    case extras::Codec::kEXR:
#if JPEGXL_ENABLE_EXR
      format.data_type = JXL_TYPE_FLOAT;
      encoder = extras::GetEXREncoder();
      break;
#else
      return JXL_FAILURE("JPEG XL was built without OpenEXR support");
#endif
    case extras::Codec::kUnknown:
      return JXL_FAILURE("Cannot encode using Codec::kUnknown");
  }

  if (!encoder) {
    return JXL_FAILURE("Invalid codec.");
  }

  extras::PackedPixelFile ppf;
  JXL_RETURN_IF_ERROR(
      ConvertCodecInOutToPackedPixelFile(io, format, c_desired, pool, &ppf));
  extras::EncodedImage encoded_image;
  JXL_RETURN_IF_ERROR(encoder->Encode(ppf, &encoded_image, pool));
  JXL_ASSERT(encoded_image.bitstreams.size() == 1);
  *bytes = encoded_image.bitstreams[0];

  return true;
}

Status EncodeToFile(const CodecInOut& io, const ColorEncoding& c_desired,
                    size_t bits_per_sample, const std::string& pathname,
                    ThreadPool* pool) {
  const std::string extension = Extension(pathname);
  const extras::Codec codec =
      extras::CodecFromExtension(extension, &bits_per_sample);

  // Warn about incorrect usage of PGM/PGX/PPM - only the latter supports
  // color, but CodecFromExtension lumps them all together.
  if (codec == extras::Codec::kPNM && extension != ".pfm") {
    if (io.Main().HasAlpha() && extension != ".pam") {
      JXL_WARNING(
          "For images with alpha, the filename should end with .pam.\n");
    } else if (!io.Main().IsGray() && extension == ".pgm") {
      JXL_WARNING("For color images, the filename should end with .ppm.\n");
    } else if (io.Main().IsGray() && extension == ".ppm") {
      JXL_WARNING(
          "For grayscale images, the filename should not end with .ppm.\n");
    }
    if (bits_per_sample > 16) {
      JXL_WARNING("PPM only supports up to 16 bits per sample");
      bits_per_sample = 16;
    }
  } else if (codec == extras::Codec::kPGX && !io.Main().IsGray()) {
    JXL_WARNING("Storing color image to PGX - use .ppm extension instead.\n");
  }
  if (bits_per_sample > 16 && codec == extras::Codec::kPNG) {
    JXL_WARNING("PNG only supports up to 16 bits per sample");
    bits_per_sample = 16;
  }

  std::vector<uint8_t> encoded;
  return Encode(io, codec, c_desired, bits_per_sample, &encoded, pool) &&
         WriteFile(encoded, pathname);
}

Status EncodeToFile(const CodecInOut& io, const std::string& pathname,
                    ThreadPool* pool) {
  // TODO(lode): need to take the floating_point_sample field into account
  return EncodeToFile(io, io.metadata.m.color_encoding,
                      io.metadata.m.bit_depth.bits_per_sample, pathname, pool);
}

}  // namespace jxl
