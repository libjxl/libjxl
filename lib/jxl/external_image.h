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

#ifndef LIB_JXL_EXTERNAL_IMAGE_H_
#define LIB_JXL_EXTERNAL_IMAGE_H_

// Interleaved image for color transforms and Codec.

#include <stddef.h>
#include <stdint.h>

#include "jxl/types.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {

constexpr size_t RowSize(size_t xsize, size_t channels,
                         size_t bits_per_sample) {
  return bits_per_sample == 1
             ? DivCeil(xsize, kBitsPerByte)
             : xsize * channels * DivCeil(bits_per_sample, kBitsPerByte);
}

// Converts ib to interleaved void* pixel buffer with the given format.
// bits_per_sample: must be 8, 16 or 32, and must be 32 if float_out
// is true. 1 and 32 int are not yet implemented.
// num_channels: must be 1, 2, 3 or 4 for gray, gray+alpha, RGB, RGB+alpha.
// This supports the features needed for the C API and does not perform
// color space conversion.
// TODO(lode): support 1-bit output (bits_per_sample == 1)
// TODO(lode): support rectangle crop.
// apply_srgb_tf applies conversion from linear sRGB to nonlinear sRGB. This
// requires that the ImageBundle is in linear sRGB.
// stride_out is output scanline size in bytes, must be >=
// output_xsize * bytes_per_pixel.
// undo_orientation is an EXIF orientation to undo. Depending on the
// orientation, the output xsize and ysize are swapped compared to input
// xsize and ysize.
Status ConvertImage(const jxl::ImageBundle& ib, size_t bits_per_sample,
                    bool float_out, bool apply_srgb_tf, size_t num_channels,
                    JxlEndianness endianness, size_t stride_out,
                    jxl::ThreadPool* thread_pool, void* out_image,
                    size_t out_size, jxl::Orientation undo_orientation);

// Does the inverse conversion, from an interleaved pixel buffer to ib.
Status ConvertImage(Span<const uint8_t> bytes, size_t xsize, size_t ysize,
                    const ColorEncoding& c_current, bool has_alpha,
                    bool alpha_is_premultiplied, size_t bits_per_sample,
                    JxlEndianness endianness, bool flipped_y, ThreadPool* pool,
                    ImageBundle* ib);

}  // namespace jxl

#endif  // LIB_JXL_EXTERNAL_IMAGE_H_
