// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_HDR_IMAGE_UTILS_H_
#define TOOLS_HDR_IMAGE_UTILS_H_

#include <jxl/cms.h>
#include <jxl/cms_interface.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "lib/extras/dec/decode.h"  // Codec enum
#include "lib/extras/enc/apng.h"
#include "lib/extras/enc/encode.h"
#include "lib/extras/enc/exr.h"
#include "lib/extras/enc/jpg.h"
#include "lib/extras/enc/pgx.h"
#include "lib/extras/enc/pnm.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/image_bundle.h"

namespace jpegxl {
namespace tools {

static JXL_MAYBE_UNUSED jxl::Status TransformCodecInOutTo(
    jxl::CodecInOut& io, const jxl::ColorEncoding& c_desired,
    jxl::ThreadPool* pool) {
  const JxlCmsInterface& cms = *JxlGetDefaultCms();
  if (io.metadata.m.have_preview) {
    JXL_RETURN_IF_ERROR(io.preview_frame.TransformTo(c_desired, cms, pool));
  }
  for (jxl::ImageBundle& ib : io.frames) {
    JXL_RETURN_IF_ERROR(ib.TransformTo(c_desired, cms, pool));
  }
  return true;
}

static inline jxl::Status Encode(const jxl::CodecInOut& io,
                                 const jxl::extras::Codec codec,
                                 const jxl::ColorEncoding& c_desired,
                                 size_t bits_per_sample,
                                 std::vector<uint8_t>* bytes,
                                 jxl::ThreadPool* pool) {
  bytes->clear();
  JXL_ENSURE(!io.Main().c_current().ICC().empty());
  JXL_ENSURE(!c_desired.ICC().empty());
  JXL_RETURN_IF_ERROR(io.CheckMetadata());
  if (io.Main().IsJPEG()) {
    JXL_WARNING("Writing JPEG data as pixels");
  }
  JxlPixelFormat format = {
      0,  // num_channels is ignored by the converter
      bits_per_sample <= 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16, JXL_BIG_ENDIAN,
      0};
  const bool floating_point = bits_per_sample > 16;
  std::unique_ptr<jxl::extras::Encoder> encoder;
  std::ostringstream os;
  switch (codec) {
    case jxl::extras::Codec::kPNG:
      encoder = jxl::extras::GetAPNGEncoder();
      if (encoder) {
        break;
      } else {
        return JXL_FAILURE("JPEG XL was built without (A)PNG support");
      }
    case jxl::extras::Codec::kJPG:
      format.data_type = JXL_TYPE_UINT8;
      encoder = jxl::extras::GetJPEGEncoder();
      if (encoder) {
        os << io.jpeg_quality;
        encoder->SetOption("q", os.str());
        break;
      } else {
        return JXL_FAILURE("JPEG XL was built without JPEG support");
      }
    case jxl::extras::Codec::kPNM:
      if (io.Main().HasAlpha()) {
        encoder = jxl::extras::GetPAMEncoder();
      } else if (io.Main().IsGray()) {
        encoder = jxl::extras::GetPGMEncoder();
      } else if (!floating_point) {
        encoder = jxl::extras::GetPPMEncoder();
      } else {
        format.data_type = JXL_TYPE_FLOAT;
        format.endianness = JXL_LITTLE_ENDIAN;
        encoder = jxl::extras::GetPFMEncoder();
      }
      break;
    case jxl::extras::Codec::kPGX:
      encoder = jxl::extras::GetPGXEncoder();
      break;
    case jxl::extras::Codec::kGIF:
      return JXL_FAILURE("Encoding to GIF is not implemented");
    case jxl::extras::Codec::kEXR:
      format.data_type = JXL_TYPE_FLOAT;
      encoder = jxl::extras::GetEXREncoder();
      if (encoder) {
        break;
      } else {
        return JXL_FAILURE("JPEG XL was built without OpenEXR support");
      }
    case jxl::extras::Codec::kJXL:
      // TODO(user): implement
      return JXL_FAILURE("Codec::kJXL is not supported yet");

    case jxl::extras::Codec::kUnknown:
      return JXL_FAILURE("Cannot encode using Codec::kUnknown");
  }

  if (!encoder) {
    return JXL_FAILURE("Invalid codec.");
  }

  jxl::extras::PackedPixelFile ppf;
  JXL_RETURN_IF_ERROR(
      ConvertCodecInOutToPackedPixelFile(io, format, c_desired, pool, &ppf));
  ppf.info.bits_per_sample = bits_per_sample;
  if (format.data_type == JXL_TYPE_FLOAT) {
    ppf.info.bits_per_sample = 32;
    ppf.info.exponent_bits_per_sample = 8;
  }
  jxl::extras::EncodedImage encoded_image;
  JXL_RETURN_IF_ERROR(encoder->Encode(ppf, &encoded_image, pool));
  JXL_ENSURE(encoded_image.bitstreams.size() == 1);
  *bytes = encoded_image.bitstreams[0];

  return true;
}

static inline jxl::Status Encode(const jxl::CodecInOut& io,
                                 const jxl::ColorEncoding& c_desired,
                                 size_t bits_per_sample,
                                 const std::string& pathname,
                                 std::vector<uint8_t>* bytes,
                                 jxl::ThreadPool* pool) {
  std::string extension;
  const jxl::extras::Codec codec =
      jxl::extras::CodecFromPath(pathname, &bits_per_sample, &extension);

  // Warn about incorrect usage of PGM/PGX/PPM - only the latter supports
  // color, but CodecFromPath lumps them all together.
  if (codec == jxl::extras::Codec::kPNM && extension != ".pfm") {
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
  } else if (codec == jxl::extras::Codec::kPGX && !io.Main().IsGray()) {
    JXL_WARNING("Storing color image to PGX - use .ppm extension instead.\n");
  }
  if (bits_per_sample > 16 && codec == jxl::extras::Codec::kPNG) {
    JXL_WARNING("PNG only supports up to 16 bits per sample");
    bits_per_sample = 16;
  }

  return Encode(io, codec, c_desired, bits_per_sample, bytes, pool);
}

static inline jxl::Status Encode(const jxl::CodecInOut& io,
                                 const std::string& pathname,
                                 std::vector<uint8_t>* bytes,
                                 jxl::ThreadPool* pool) {
  // TODO(lode): need to take the floating_point_sample field into account
  return Encode(io, io.metadata.m.color_encoding,
                io.metadata.m.bit_depth.bits_per_sample, pathname, bytes, pool);
}

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_HDR_IMAGE_UTILS_H_
