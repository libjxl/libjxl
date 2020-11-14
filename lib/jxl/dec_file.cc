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

#include "lib/jxl/dec_file.h"

#include <stddef.h>

#include <utility>
#include <vector>

#include "jxl/decode.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {
namespace {

Status DecodePreview(const DecompressParams& dparams,
                     BitReader* JXL_RESTRICT reader, AuxOut* aux_out,
                     ThreadPool* pool, CodecInOut* JXL_RESTRICT io) {
  // No preview present in file.
  if (!io->metadata.m.have_preview) {
    if (dparams.preview == Override::kOn) {
      return JXL_FAILURE("preview == kOn but no preview present");
    }
    return true;
  }

  // Have preview; prepare to skip or read it.
  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  if (dparams.preview == Override::kOff) {
    JXL_RETURN_IF_ERROR(SkipFrame(io->metadata, reader, /*is_preview=*/true));
    return true;
  }

  // Else: default or kOn => decode preview.
  PassesDecoderState dec_state;
  JXL_RETURN_IF_ERROR(DecodeFrame(dparams, &dec_state, pool, reader, aux_out,
                                  &io->preview_frame, io->metadata, io,
                                  /*is_preview=*/true));
  io->dec_pixels += dec_state.shared->frame_dim.xsize_upsampled *
                    dec_state.shared->frame_dim.ysize_upsampled;
  return true;
}

Status DecodeHeaders(BitReader* reader, CodecInOut* io) {
  JXL_RETURN_IF_ERROR(ReadSizeHeader(reader, &io->metadata.size));

  JXL_RETURN_IF_ERROR(ReadImageMetadata(reader, &io->metadata.m));

  io->metadata.transform_data.nonserialized_xyb_encoded =
      io->metadata.m.xyb_encoded;
  JXL_RETURN_IF_ERROR(Bundle::Read(reader, &io->metadata.transform_data));

  return true;
}

}  // namespace

// To avoid the complexity of file I/O and buffering, we assume the bitstream
// is loaded (or for large images/sequences: mapped into) memory.
Status DecodeFile(const DecompressParams& dparams,
                  const Span<const uint8_t> file, CodecInOut* JXL_RESTRICT io,
                  AuxOut* aux_out, ThreadPool* pool) {
  PROFILER_ZONE("DecodeFile uninstrumented");

  // Marker
  JxlSignature signature = JxlSignatureCheck(file.data(), file.size());
  if (signature == JXL_SIG_NOT_ENOUGH_BYTES || signature == JXL_SIG_INVALID) {
    return JXL_FAILURE("File does not start with known JPEG XL signature");
  }

  std::unique_ptr<brunsli::JPEGData> jpeg_data = nullptr;
  if (dparams.keep_dct) {
    if (io->Main().jpeg_data == nullptr) {
      return JXL_FAILURE("Caller must set jpeg_data");
    }
    jpeg_data = std::move(io->Main().jpeg_data);
  }

  Status ret = true;
  {
    BitReader reader(file);
    BitReaderScopedCloser reader_closer(&reader, &ret);
    (void)reader.ReadFixedBits<16>();  // skip marker

    {
      JXL_RETURN_IF_ERROR(DecodeHeaders(&reader, io));
      size_t xsize = io->metadata.xsize();
      size_t ysize = io->metadata.ysize();
      JXL_RETURN_IF_ERROR(io->VerifyDimensions(xsize, ysize));
    }

    if (io->metadata.m.color_encoding.WantICC()) {
      PaddedBytes icc;
      JXL_RETURN_IF_ERROR(ReadICC(&reader, &icc));
      JXL_RETURN_IF_ERROR(io->metadata.m.color_encoding.SetICC(std::move(icc)));
    }

    JXL_RETURN_IF_ERROR(DecodePreview(dparams, &reader, aux_out, pool, io));

    // Only necessary if no ICC and no preview.
    JXL_RETURN_IF_ERROR(reader.JumpToByteBoundary());
    if (io->metadata.m.have_animation && dparams.keep_dct) {
      return JXL_FAILURE("Cannot decode to JPEG an animation");
    }

    if (io->metadata.m.bit_depth.floating_point_sample &&
        !io->metadata.m.xyb_encoded) {
      io->dec_target = DecodeTarget::kLosslessFloat;
    }

    PassesDecoderState dec_state;

    io->frames.clear();
    do {
      io->frames.emplace_back(&io->metadata.m);
      if (jpeg_data) {
        io->frames.back().jpeg_data = std::move(jpeg_data);
      }
      // Skip frames that are not displayed.
      do {
        JXL_RETURN_IF_ERROR(DecodeFrame(dparams, &dec_state, pool, &reader,
                                        aux_out, &io->frames.back(),
                                        io->metadata, io));
      } while (dec_state.shared->frame_header.frame_type !=
               FrameType::kRegularFrame);
      io->dec_pixels += io->frames.back().xsize() * io->frames.back().ysize();
    } while (!dec_state.shared->frame_header.is_last);

    if (dparams.check_decompressed_size && !dparams.allow_partial_files &&
        dparams.max_downsampling == 1) {
      if (reader.TotalBitsConsumed() != file.size() * kBitsPerByte) {
        return JXL_FAILURE("DecodeFile reader position not at EOF.");
      }
    }

    io->CheckMetadata();
    // reader is closed here.
  }
  return ret;
}

}  // namespace jxl
