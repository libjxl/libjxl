// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/enc/pgx.h"

#include <stdio.h>
#include <string.h>

#include "lib/extras/packed_image_convert.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/dec_external_image.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/enc_image_bundle.h"
#include "lib/jxl/fields.h"  // AllDefault
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {
namespace extras {
namespace {

constexpr size_t kMaxHeaderSize = 200;

Status EncodeHeader(const ImageBundle& ib, const size_t bits_per_sample,
                    char* header, int* JXL_RESTRICT chars_written) {
  if (ib.HasAlpha()) return JXL_FAILURE("PGX: can't store alpha");
  if (!ib.IsGray()) return JXL_FAILURE("PGX: must be grayscale");
  // TODO(lode): verify other bit depths: for other bit depths such as 1 or 4
  // bits, have a test case to verify it works correctly. For bits > 16, we may
  // need to change the way external_image works.
  if (bits_per_sample != 8 && bits_per_sample != 16) {
    return JXL_FAILURE("PGX: bits other than 8 or 16 not yet supported");
  }

  // Use ML (Big Endian), LM may not be well supported by all decoders.
  *chars_written = snprintf(header, kMaxHeaderSize,
                            "PG ML + %" PRIuS " %" PRIuS " %" PRIuS "\n",
                            bits_per_sample, ib.xsize(), ib.ysize());
  JXL_RETURN_IF_ERROR(static_cast<unsigned int>(*chars_written) <
                      kMaxHeaderSize);
  return true;
}

Status EncodeImagePGX(const ImageBundle& ib, const ColorEncoding& c_desired,
                      size_t bits_per_sample, ThreadPool* pool,
                      std::vector<uint8_t>* bytes) {
  if (!Bundle::AllDefault(*ib.metadata())) {
    JXL_WARNING("PGX encoder ignoring metadata - use a different codec");
  }
  if (!c_desired.IsSRGB()) {
    JXL_WARNING(
        "PGX encoder cannot store custom ICC profile; decoder\n"
        "will need hint key=color_space to get the same values");
  }

  ImageMetadata metadata = *ib.metadata();
  ImageBundle store(&metadata);
  const ImageBundle* transformed;
  JXL_RETURN_IF_ERROR(TransformIfNeeded(ib, c_desired, GetJxlCms(), pool,
                                        &store, &transformed));
  std::vector<uint8_t> pixels(ib.xsize() * ib.ysize() *
                              (bits_per_sample / kBitsPerByte));
  size_t stride = ib.xsize() * (bits_per_sample / kBitsPerByte);
  JXL_RETURN_IF_ERROR(
      ConvertToExternal(*transformed, bits_per_sample,
                        /*float_out=*/false,
                        /*num_channels=*/1, JXL_BIG_ENDIAN, stride, pool,
                        pixels.data(), pixels.size(),
                        /*out_callback=*/{}, metadata.GetOrientation()));

  char header[kMaxHeaderSize];
  int header_size = 0;
  JXL_RETURN_IF_ERROR(EncodeHeader(ib, bits_per_sample, header, &header_size));

  bytes->resize(static_cast<size_t>(header_size) + pixels.size());
  memcpy(bytes->data(), header, static_cast<size_t>(header_size));
  memcpy(bytes->data() + header_size, pixels.data(), pixels.size());

  return true;
}

class PGXEncoder : public Encoder {
 public:
  std::vector<JxlPixelFormat> AcceptedFormats() const override {
    std::vector<JxlPixelFormat> formats;
    for (const JxlDataType data_type : {JXL_TYPE_UINT8, JXL_TYPE_UINT16}) {
      for (JxlEndianness endianness : {JXL_BIG_ENDIAN, JXL_LITTLE_ENDIAN}) {
        formats.push_back(JxlPixelFormat{/*num_channels=*/1,
                                         /*data_type=*/data_type,
                                         /*endianness=*/endianness,
                                         /*align=*/0});
      }
    }
    return formats;
  }
  Status Encode(const PackedPixelFile& ppf, EncodedImage* encoded_image,
                ThreadPool* pool) const override {
    CodecInOut io;
    JXL_RETURN_IF_ERROR(ConvertPackedPixelFileToCodecInOut(ppf, pool, &io));
    const PaddedBytes& icc = io.Main().c_current().ICC();
    encoded_image->icc.assign(icc.begin(), icc.end());
    encoded_image->bitstreams.clear();
    encoded_image->bitstreams.reserve(io.frames.size());
    for (const ImageBundle& ib : io.frames) {
      encoded_image->bitstreams.emplace_back();
      JXL_RETURN_IF_ERROR(EncodeImagePGX(ib, io.metadata.m.color_encoding,
                                         ppf.info.bits_per_sample, pool,
                                         &encoded_image->bitstreams.back()));
    }
    return true;
  }
};

}  // namespace

std::unique_ptr<Encoder> GetPGXEncoder() {
  return jxl::make_unique<PGXEncoder>();
}

Status EncodeImagePGX(const CodecInOut* io, const ColorEncoding& c_desired,
                      size_t bits_per_sample, ThreadPool* pool,
                      std::vector<uint8_t>* bytes) {
  return EncodeImagePGX(io->Main(), c_desired, bits_per_sample, pool, bytes);
}

}  // namespace extras
}  // namespace jxl
