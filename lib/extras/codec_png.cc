// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/codec_png.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Lodepng library:
#include <lodepng.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_external_image.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/enc_image_bundle.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/luminance.h"

namespace jxl {
namespace extras {
namespace {

#define JXL_PNG_VERBOSE 0

// Stores XMP and EXIF/IPTC into itext and text.
class BlobsWriterPNG {
 public:
  static Status Encode(const Blobs& blobs, LodePNGInfo* JXL_RESTRICT info) {
    if (!blobs.exif.empty()) {
      JXL_RETURN_IF_ERROR(EncodeBase16("exif", blobs.exif, info));
    }
    if (!blobs.iptc.empty()) {
      JXL_RETURN_IF_ERROR(EncodeBase16("iptc", blobs.iptc, info));
    }

    if (!blobs.xmp.empty()) {
      JXL_RETURN_IF_ERROR(EncodeBase16("xmp", blobs.xmp, info));

      // Below is the official way, but it does not seem to work in ImageMagick.
      // Exiv2 and exiftool are OK with either way of encoding XMP.
      if (/* DISABLES CODE */ (0)) {
        const char* key = "XML:com.adobe.xmp";
        const std::string text(reinterpret_cast<const char*>(blobs.xmp.data()),
                               blobs.xmp.size());
        if (lodepng_add_itext(info, key, "", "", text.c_str()) != 0) {
          return JXL_FAILURE("Failed to add itext");
        }
      }
    }

    return true;
  }

 private:
  static JXL_INLINE char EncodeNibble(const uint8_t nibble) {
    JXL_ASSERT(nibble < 16);
    return (nibble < 10) ? '0' + nibble : 'a' + nibble - 10;
  }

  static Status EncodeBase16(const std::string& type, const PaddedBytes& bytes,
                             LodePNGInfo* JXL_RESTRICT info) {
    // Encoding: base16 with newline after 72 chars.
    const size_t base16_size =
        2 * bytes.size() + DivCeil(bytes.size(), size_t(36)) + 1;
    std::string base16;
    base16.reserve(base16_size);
    for (size_t i = 0; i < bytes.size(); ++i) {
      if (i % 36 == 0) base16.push_back('\n');
      base16.push_back(EncodeNibble(bytes[i] >> 4));
      base16.push_back(EncodeNibble(bytes[i] & 0x0F));
    }
    base16.push_back('\n');
    JXL_ASSERT(base16.length() == base16_size);

    char key[30];
    snprintf(key, sizeof(key), "Raw profile type %s", type.c_str());

    char header[30];
    snprintf(header, sizeof(header), "\n%s\n%8" PRIuS, type.c_str(),
             bytes.size());

    const std::string& encoded = std::string(header) + base16;
    if (lodepng_add_text(info, key, encoded.c_str()) != 0) {
      return JXL_FAILURE("Failed to add text");
    }

    return true;
  }
};

// Stores ColorEncoding into PNG chunks.
class ColorEncodingWriterPNG {
 public:
  static Status Encode(const ColorEncoding& c, LodePNGInfo* JXL_RESTRICT info) {
    // Prefer to only write sRGB - smaller.
    if (c.IsSRGB()) {
      JXL_RETURN_IF_ERROR(AddSRGB(c, info));
      // PNG recommends not including both sRGB and iCCP, so skip the latter.
    } else if (!c.HaveFields() || !c.tf.IsGamma()) {
      // Having a gamma value means that the source was a PNG with gAMA and
      // without iCCP.
      JXL_ASSERT(!c.ICC().empty());
      JXL_RETURN_IF_ERROR(AddICC(c.ICC(), info));
    }

    // gAMA and cHRM are always allowed but will be overridden by sRGB/iCCP.
    JXL_RETURN_IF_ERROR(MaybeAddGAMA(c, info));
    JXL_RETURN_IF_ERROR(MaybeAddCHRM(c, info));
    return true;
  }

 private:
  static Status AddChunk(const char* type, const PaddedBytes& payload,
                         LodePNGInfo* JXL_RESTRICT info) {
    // Ignore original location/order of chunks; place them in the first group.
    if (lodepng_chunk_create(&info->unknown_chunks_data[0],
                             &info->unknown_chunks_size[0], payload.size(),
                             type, payload.data()) != 0) {
      return JXL_FAILURE("Failed to add chunk");
    }
    return true;
  }

  static Status AddICC(const PaddedBytes& icc, LodePNGInfo* JXL_RESTRICT info) {
    LodePNGCompressSettings settings;
    lodepng_compress_settings_init(&settings);
    unsigned char* out = nullptr;
    size_t out_size = 0;
    if (lodepng_zlib_compress(&out, &out_size, icc.data(), icc.size(),
                              &settings) != 0) {
      return JXL_FAILURE("Failed to compress ICC");
    }

    PaddedBytes payload;
    payload.resize(3 + out_size);
    // TODO(janwas): use special name if PQ
    payload[0] = '1';  // profile name
    payload[1] = '\0';
    payload[2] = 0;  // compression method (zlib)
    memcpy(&payload[3], out, out_size);
    free(out);

    return AddChunk("iCCP", payload, info);
  }

  static Status AddSRGB(const ColorEncoding& c,
                        LodePNGInfo* JXL_RESTRICT info) {
    PaddedBytes payload;
    payload.push_back(static_cast<uint8_t>(c.rendering_intent));
    return AddChunk("sRGB", payload, info);
  }

  // Returns PNG encoding of floating-point value (times 10^5).
  static uint32_t U32FromF64(const double x) {
    return static_cast<int32_t>(roundf(x * 1E5));
  }

  static Status MaybeAddGAMA(const ColorEncoding& c,
                             LodePNGInfo* JXL_RESTRICT info) {
    double gamma;
    if (c.tf.IsGamma()) {
      gamma = c.tf.GetGamma();
    } else if (c.tf.IsLinear()) {
      gamma = 1;
    } else if (c.tf.IsSRGB()) {
      gamma = 0.45455;
    } else {
      return true;
    }

    PaddedBytes payload(4);
    StoreBE32(U32FromF64(gamma), payload.data());
    return AddChunk("gAMA", payload, info);
  }

  static Status MaybeAddCHRM(const ColorEncoding& c,
                             LodePNGInfo* JXL_RESTRICT info) {
    CIExy white_point = c.GetWhitePoint();
    // A PNG image stores both whitepoint and primaries in the cHRM chunk, but
    // for grayscale images we don't have primaries. It does not matter what
    // values are stored in the PNG though (all colors are a multiple of the
    // whitepoint), so choose default ones. See
    // http://www.libpng.org/pub/png/spec/1.2/PNG-Chunks.html section 4.2.2.1.
    PrimariesCIExy primaries =
        c.IsGray() ? ColorEncoding().GetPrimaries() : c.GetPrimaries();

    if (c.primaries == Primaries::kSRGB && c.white_point == WhitePoint::kD65) {
      // For sRGB, the cHRM chunk is supposed to have very specific values which
      // don't quite match the pre-quantized ones we have (red is off by
      // 0.00010). Technically, this is only required for full sRGB, but for
      // consistency, we might as well use them whenever the primaries and white
      // point are sRGB's.
      white_point.x = 0.31270;
      white_point.y = 0.32900;
      primaries.r.x = 0.64000;
      primaries.r.y = 0.33000;
      primaries.g.x = 0.30000;
      primaries.g.y = 0.60000;
      primaries.b.x = 0.15000;
      primaries.b.y = 0.06000;
    }

    PaddedBytes payload(32);
    StoreBE32(U32FromF64(white_point.x), &payload[0]);
    StoreBE32(U32FromF64(white_point.y), &payload[4]);
    StoreBE32(U32FromF64(primaries.r.x), &payload[8]);
    StoreBE32(U32FromF64(primaries.r.y), &payload[12]);
    StoreBE32(U32FromF64(primaries.g.x), &payload[16]);
    StoreBE32(U32FromF64(primaries.g.y), &payload[20]);
    StoreBE32(U32FromF64(primaries.b.x), &payload[24]);
    StoreBE32(U32FromF64(primaries.b.y), &payload[28]);
    return AddChunk("cHRM", payload, info);
  }
};

// RAII - ensures state is freed even if returning early.
struct PNGState {
  PNGState() { lodepng_state_init(&s); }
  ~PNGState() { lodepng_state_cleanup(&s); }

  LodePNGState s;
};

Status CheckGray(const LodePNGColorMode& mode, bool has_icc, bool* is_gray) {
  switch (mode.colortype) {
    case LCT_GREY:
    case LCT_GREY_ALPHA:
      *is_gray = true;
      return true;

    case LCT_RGB:
    case LCT_RGBA:
      *is_gray = false;
      return true;

    case LCT_PALETTE: {
      if (has_icc) {
        // If an ICC profile is present, the PNG specification requires
        // palette to be interpreted as RGB colored, not grayscale, so we must
        // output color in that case and unfortunately can't optimize it to
        // gray if the palette only has gray entries.
        *is_gray = false;
        return true;
      } else {
        *is_gray = true;
        for (size_t i = 0; i < mode.palettesize; i++) {
          if (mode.palette[i * 4] != mode.palette[i * 4 + 1] ||
              mode.palette[i * 4] != mode.palette[i * 4 + 2]) {
            *is_gray = false;
            break;
          }
        }
        return true;
      }
    }

    default:
      *is_gray = false;
      return JXL_FAILURE("Unexpected PNG color type");
  }
}

Status CheckAlpha(const LodePNGColorMode& mode, bool* has_alpha) {
  if (mode.key_defined) {
    // Color key marks a single color as transparent.
    *has_alpha = true;
    return true;
  }

  switch (mode.colortype) {
    case LCT_GREY:
    case LCT_RGB:
      *has_alpha = false;
      return true;

    case LCT_GREY_ALPHA:
    case LCT_RGBA:
      *has_alpha = true;
      return true;

    case LCT_PALETTE: {
      *has_alpha = false;
      for (size_t i = 0; i < mode.palettesize; i++) {
        // PNG palettes are always 8-bit.
        if (mode.palette[i * 4 + 3] != 255) {
          *has_alpha = true;
          break;
        }
      }
      return true;
    }

    default:
      *has_alpha = false;
      return JXL_FAILURE("Unexpected PNG color type");
  }
}

LodePNGColorType MakeType(const bool is_gray, const bool has_alpha) {
  if (is_gray) {
    return has_alpha ? LCT_GREY_ALPHA : LCT_GREY;
  }
  return has_alpha ? LCT_RGBA : LCT_RGB;
}

// Inspects first chunk of the given type and updates state with the information
// when the chunk is relevant and present in the file.
Status InspectChunkType(const Span<const uint8_t> bytes,
                        const std::string& type, LodePNGState* state) {
  const unsigned char* chunk = lodepng_chunk_find_const(
      bytes.data(), bytes.data() + bytes.size(), type.c_str());
  if (chunk && lodepng_inspect_chunk(state, chunk - bytes.data(), bytes.data(),
                                     bytes.size()) != 0) {
    return JXL_FAILURE("Invalid chunk \"%s\" in PNG image", type.c_str());
  }
  return true;
}

}  // namespace

Status EncodeImagePNG(const CodecInOut* io, const ColorEncoding& c_desired,
                      size_t bits_per_sample, ThreadPool* pool,
                      PaddedBytes* bytes) {
  if (bits_per_sample > 8) {
    bits_per_sample = 16;
  } else if (bits_per_sample < 8) {
    // PNG can also do 4, 2, and 1 bits per sample, but it isn't implemented
    bits_per_sample = 8;
  }
  ImageBundle ib = io->Main().Copy();
  const size_t alpha_bits = ib.HasAlpha() ? bits_per_sample : 0;
  ImageMetadata metadata = io->metadata.m;
  ImageBundle store(&metadata);
  const ImageBundle* transformed;
  JXL_RETURN_IF_ERROR(TransformIfNeeded(ib, c_desired, GetJxlCms(), pool,
                                        &store, &transformed));
  size_t stride = ib.oriented_xsize() *
                  DivCeil(c_desired.Channels() * bits_per_sample + alpha_bits,
                          kBitsPerByte);
  PaddedBytes raw_bytes(stride * ib.oriented_ysize());
  JXL_RETURN_IF_ERROR(ConvertToExternal(
      *transformed, bits_per_sample, /*float_out=*/false,
      c_desired.Channels() + (ib.HasAlpha() ? 1 : 0), JXL_BIG_ENDIAN, stride,
      pool, raw_bytes.data(), raw_bytes.size(), /*out_callback=*/nullptr,
      /*out_opaque=*/nullptr, metadata.GetOrientation()));

  PNGState state;
  // For maximum compatibility, still store 8-bit even if pixels are all zero.
  state.s.encoder.auto_convert = 0;

  LodePNGInfo* info = &state.s.info_png;
  info->color.bitdepth = bits_per_sample;
  info->color.colortype = MakeType(ib.IsGray(), ib.HasAlpha());
  state.s.info_raw = info->color;

  JXL_RETURN_IF_ERROR(ColorEncodingWriterPNG::Encode(c_desired, info));
  JXL_RETURN_IF_ERROR(BlobsWriterPNG::Encode(io->blobs, info));

  unsigned char* out = nullptr;
  size_t out_size = 0;
  const unsigned err =
      lodepng_encode(&out, &out_size, raw_bytes.data(), ib.oriented_xsize(),
                     ib.oriented_ysize(), &state.s);
  // Automatically call free(out) on return.
  std::unique_ptr<unsigned char, void (*)(void*)> out_ptr{out, free};
  if (err != 0) {
    return JXL_FAILURE("Failed to encode PNG: %s", lodepng_error_text(err));
  }
  bytes->resize(out_size);
  memcpy(bytes->data(), out, out_size);
  return true;
}

}  // namespace extras
}  // namespace jxl
