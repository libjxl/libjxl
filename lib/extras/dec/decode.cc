// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/decode.h"

#include <jxl/codestream_header.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <locale>
#include <string>

#include "lib/extras/dec/apng.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/exr.h"
#include "lib/extras/dec/gif.h"
#include "lib/extras/dec/jpg.h"
#include "lib/extras/dec/jxl.h"
#include "lib/extras/dec/pgx.h"
#include "lib/extras/dec/pnm.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {
namespace {

// Any valid encoding is larger (ensures codecs can read the first few bytes)
constexpr size_t kMinBytes = 9;

std::string GetExtension(const std::string& path) {
  // Pattern: "name.png"
  size_t pos = path.find_last_of('.');
  if (pos != std::string::npos) {
    return path.substr(pos);
  }

  // Extension not found
  return "";
}

}  // namespace

Codec CodecFromPath(const std::string& path,
                    size_t* JXL_RESTRICT bits_per_sample,
                    std::string* extension) {
  std::string ext = GetExtension(path);
  if (extension) {
    if (extension->empty()) {
      *extension = ext;
    } else {
      ext = *extension;
    }
  }
  std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) {
    return std::tolower(c, std::locale::classic());
  });
  if (ext == ".png") return Codec::kPNG;

  if (ext == ".jpg") return Codec::kJPG;
  if (ext == ".jpeg") return Codec::kJPG;

  if (ext == ".pgx") return Codec::kPGX;

  if (ext == ".pam") return Codec::kPNM;
  if (ext == ".pnm") return Codec::kPNM;
  if (ext == ".pgm") return Codec::kPNM;
  if (ext == ".ppm") return Codec::kPNM;
  if (ext == ".pfm") {
    if (bits_per_sample != nullptr) *bits_per_sample = 32;
    return Codec::kPNM;
  }

  if (ext == ".gif") return Codec::kGIF;

  if (ext == ".exr") return Codec::kEXR;

  return Codec::kUnknown;
}

bool CanDecode(Codec codec) {
  switch (codec) {
    case Codec::kEXR:
      return CanDecodeEXR();
    case Codec::kGIF:
      return CanDecodeGIF();
    case Codec::kJPG:
      return CanDecodeJPG();
    case Codec::kPNG:
      return CanDecodeAPNG();
    case Codec::kPNM:
    case Codec::kPGX:
    case Codec::kJXL:
      return true;
    default:
      return false;
  }
}

std::string ListOfDecodeCodecs() {
  std::string list_of_codecs("JXL, PPM, PNM, PFM, PAM, PGX");
  if (CanDecode(Codec::kPNG)) list_of_codecs.append(", PNG, APNG");
  if (CanDecode(Codec::kGIF)) list_of_codecs.append(", GIF");
  if (CanDecode(Codec::kJPG)) list_of_codecs.append(", JPEG");
  if (CanDecode(Codec::kEXR)) list_of_codecs.append(", EXR");
  return list_of_codecs;
}

Status DecodeBytes(const Span<const uint8_t> bytes,
                   const ColorHints& color_hints, extras::PackedPixelFile* ppf,
                   const SizeConstraints* constraints, Codec* orig_codec,
                   JxlMemoryManager* memory_manager, bool coalescing) {
  if (bytes.size() < kMinBytes) return JXL_FAILURE("Too few bytes");

  *ppf = extras::PackedPixelFile();

  // Default values when not set by decoders.
  ppf->info.uses_original_profile = JXL_TRUE;
  ppf->info.orientation = JXL_ORIENT_IDENTITY;

  Codec codec = DetectCodec(bytes);
  bool ok = false;
  switch (codec) {
    case Codec::kEXR:
      ok = DecodeImageEXR(bytes, color_hints, ppf, constraints);
      break;

    case Codec::kGIF:
      ok = DecodeImageGIF(bytes, color_hints, ppf, constraints);
      break;

    case Codec::kJPG:
      ok = DecodeImageJPG(bytes, color_hints, ppf, constraints);
      break;

    case Codec::kJXL: {
      JXLDecompressParams dparams = {};
      dparams.memory_manager = memory_manager;
      dparams.coalescing = coalescing;
      for (const uint32_t num_channels : {1, 2, 3, 4}) {
        dparams.accepted_formats.push_back(
            {num_channels, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, /*align=*/0});
      }
      dparams.output_bitdepth.type = JXL_BIT_DEPTH_FROM_CODESTREAM;
      size_t decoded_bytes;
      ok = DecodeImageJXL(bytes.data(), bytes.size(), dparams, &decoded_bytes,
                          ppf, nullptr, constraints) &&
           ApplyColorHints(color_hints, true, ppf->info.num_color_channels == 1,
                           ppf);
      break;
    }

    case Codec::kPGX:
      ok = DecodeImagePGX(bytes, color_hints, ppf, constraints);
      break;

    case Codec::kPNG:
      ok = DecodeImageAPNG(bytes, color_hints, ppf, constraints);
      break;

    case Codec::kPNM:
      ok = DecodeImagePNM(bytes, color_hints, ppf, constraints);
      break;

    case Codec::kUnknown:
      return JXL_FAILURE("Unrecognized codec");
  }

  if (!ok) {
    return JXL_FAILURE("Codecs failed to decode");
  }
  if (orig_codec) *orig_codec = codec;

  return true;
}

template <size_t N, size_t L>
bool CheckSignatures(const Span<const uint8_t>& bytes,
                     const std::array<std::array<uint8_t, L>, N>& signatures) {
  static_assert(L <= kMinBytes, "Signature too long");
  if (bytes.size() < L) return false;
  for (auto signature : signatures) {
    if (memcmp(bytes.data(), signature.data(), signature.size()) == 0) {
      return true;
    }
  }
  return false;
}

Codec DetectCodec(const Span<const uint8_t>& bytes) {
  constexpr std::array<std::array<uint8_t, 4>, 1> kExrSignatures = {{
      {'v', '/', '1', 0x01},
  }};
  constexpr std::array<std::array<uint8_t, 6>, 2> kGifSignatures = {{
      {'G', 'I', 'F', '8', '7', 'a'},
      {'G', 'I', 'F', '8', '9', 'a'},
  }};
  constexpr std::array<std::array<uint8_t, 7>, 4> kPgxSignatures = {
      {{'P', 'G', ' ', 'L', 'M', ' ', '+'},
       {'P', 'G', ' ', 'L', 'M', ' ', '-'},
       {'P', 'G', ' ', 'M', 'L', ' ', '+'},
       {'P', 'G', ' ', 'M', 'L', ' ', '-'}}};
  constexpr std::array<std::array<uint8_t, 8>, 1> kPngSignatures = {
      {{137, 'P', 'N', 'G', '\r', '\n', 26, '\n'}}};
  static const std::array<std::array<uint8_t, 2>, 9> kPnmSignatures = {
      {{'P', '1'},
       {'P', '2'},
       {'P', '3'},
       {'P', '4'},
       {'P', '5'},
       {'P', '6'},
       {'P', '7'},
       {'P', 'F'},
       {'P', 'f'}}};
  static const std::array<std::array<uint8_t, 4>, 5> kJpgSignatures = {{
      {0xFF, 0xD8, 0xFF, 0xDB},
      {0xFF, 0xD8, 0xFF, 0xE0},
      {0xFF, 0xD8, 0xFF, 0xE1},
      {0xFF, 0xD8, 0xFF, 0xE2},
      {0xFF, 0xD8, 0xFF, 0xEE},
  }};
  static const std::array<std::array<uint8_t, 9>, 1> kJxlBoxSignatures = {{
      {0x00, 0x00, 0x00, 0x0C, 'J', 'X', 'L', ' ', 0x0D},
  }};
  static const std::array<std::array<uint8_t, 2>, 1> kJxlSignatures = {{
      {0xFF, 0x0A},
  }};

  if (CheckSignatures(bytes, kExrSignatures)) return Codec::kEXR;
  if (CheckSignatures(bytes, kGifSignatures)) return Codec::kGIF;
  if (CheckSignatures(bytes, kJpgSignatures)) return Codec::kJPG;
  if (CheckSignatures(bytes, kJxlBoxSignatures)) return Codec::kJXL;
  if (CheckSignatures(bytes, kJxlSignatures)) return Codec::kJXL;
  if (CheckSignatures(bytes, kPgxSignatures)) return Codec::kPGX;
  if (CheckSignatures(bytes, kPngSignatures)) return Codec::kPNG;
  if (CheckSignatures(bytes, kPnmSignatures)) return Codec::kPNM;
  return Codec::kUnknown;
}

}  // namespace extras
}  // namespace jxl
