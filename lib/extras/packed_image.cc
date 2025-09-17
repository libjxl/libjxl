// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Helper class for storing external (int or float, interleaved) images. This is
// the common format used by other libraries and in the libjxl API.

#include "packed_image.h"

#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/encode.h>
#include <jxl/types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {

// Class representing an interleaved image with a bunch of channels.
StatusOr<PackedImage> PackedImage::Create(size_t xsize, size_t ysize,
                                          const JxlPixelFormat& format) {
  PackedImage image(xsize, ysize, format, CalcStride(format, xsize));
  if (!image.pixels()) {
    // TODO(szabadka): use specialized OOM error code
    return JXL_FAILURE("Failed to allocate memory for image");
  }
  return image;
}

PackedImage PackedImage::Copy() const {
  size_t copy_stride = CalcStride(format, xsize);
  PackedImage copy(xsize, ysize, format, copy_stride);
  const uint8_t* orig_pixels = reinterpret_cast<const uint8_t*>(pixels());
  uint8_t* copy_pixels = reinterpret_cast<uint8_t*>(copy.pixels());
  if (stride == copy_stride) {
    // Same stride -> copy in one go.
    memcpy(copy_pixels, orig_pixels, ysize * stride);
  } else {
    // Otherwise, copy row-wise.
    JXL_DASSERT(copy_stride < stride);
    for (size_t y = 0; y < ysize; ++y) {
      memcpy(copy_pixels + y * copy_stride, orig_pixels + y * stride,
             copy_stride);
    }
  }
  return copy;
}

Status PackedImage::ValidateDataType(JxlDataType data_type) {
  if ((data_type != JXL_TYPE_UINT8) && (data_type != JXL_TYPE_UINT16) &&
      (data_type != JXL_TYPE_FLOAT) && (data_type != JXL_TYPE_FLOAT16)) {
    return JXL_FAILURE("Unhandled data type: %d", static_cast<int>(data_type));
  }
  return true;
}

size_t PackedImage::BitsPerChannel(JxlDataType data_type) {
  switch (data_type) {
    case JXL_TYPE_UINT8:
      return 8;
    case JXL_TYPE_UINT16:
      return 16;
    case JXL_TYPE_FLOAT:
      return 32;
    case JXL_TYPE_FLOAT16:
      return 16;
    default:
      JXL_DEBUG_ABORT("Unreachable");
      return 0;
  }
}

// Logical resize; use Copy() for storage reallocation, if necessary.
Status PackedImage::ShrinkTo(size_t new_xsize, size_t new_ysize) {
  if (new_xsize > xsize || new_ysize > ysize) {
    return JXL_FAILURE("Cannot shrink PackedImage to a larger size");
  }
  xsize = new_xsize;
  ysize = new_ysize;
  return true;
}

PackedImage::PackedImage(size_t xsize, size_t ysize,
                         const JxlPixelFormat& format, size_t stride)
    : xsize(xsize),
      ysize(ysize),
      stride(stride),
      format(format),
      pixels_size(ysize * stride),
      pixels_(malloc(std::max<size_t>(1, pixels_size)), free) {
  bytes_per_channel_ = BitsPerChannel(format.data_type) / jxl::kBitsPerByte;
  pixel_stride_ = format.num_channels * bytes_per_channel_;
  swap_endianness_ = SwapEndianness(format.endianness);
}

size_t PackedImage::CalcStride(const JxlPixelFormat& format, size_t xsize) {
  size_t stride = xsize * (BitsPerChannel(format.data_type) *
                           format.num_channels / jxl::kBitsPerByte);
  if (format.align > 1) {
    stride = jxl::DivCeil(stride, format.align) * format.align;
  }
  return stride;
}

PackedFrame::PackedFrame(PackedImage&& image) : color(std::move(image)) {}

PackedFrame::PackedFrame(PackedFrame&& other) = default;

PackedFrame& PackedFrame::operator=(PackedFrame&& other) = default;

PackedFrame::~PackedFrame() = default;

StatusOr<PackedFrame> PackedFrame::Create(size_t xsize, size_t ysize,
                                          const JxlPixelFormat& format) {
  JXL_ASSIGN_OR_RETURN(PackedImage image,
                       PackedImage::Create(xsize, ysize, format));
  PackedFrame frame(std::move(image));
  return frame;
}

StatusOr<PackedFrame> PackedFrame::Copy() const {
  JXL_ASSIGN_OR_RETURN(
      PackedFrame copy,
      PackedFrame::Create(color.xsize, color.ysize, color.format));
  copy.frame_info = frame_info;
  copy.name = name;
  copy.color = color.Copy();
  for (const auto& ec : extra_channels) {
    copy.extra_channels.emplace_back(ec.Copy());
  }
  return copy;
}

// Logical resize; use Copy() for storage reallocation, if necessary.
Status PackedFrame::ShrinkTo(size_t new_xsize, size_t new_ysize) {
  JXL_RETURN_IF_ERROR(color.ShrinkTo(new_xsize, new_ysize));
  for (auto& ec : extra_channels) {
    JXL_RETURN_IF_ERROR(ec.ShrinkTo(new_xsize, new_ysize));
  }
  frame_info.layer_info.xsize = new_xsize;
  frame_info.layer_info.ysize = new_ysize;
  return true;
}

ChunkedPackedFrame::ChunkedPackedFrame(
    size_t xsize, size_t ysize,
    std::function<JxlChunkedFrameInputSource()> get_input_source)
    : xsize(xsize),
      ysize(ysize),
      get_input_source_(std::move(get_input_source)) {
  const auto input_source = get_input_source_();
  input_source.get_color_channels_pixel_format(input_source.opaque, &format);
}

PackedPixelFile::PackedPixelFile() { JxlEncoderInitBasicInfo(&info); };

Status PackedPixelFile::ShrinkTo(size_t new_xsize, size_t new_ysize) {
  for (auto& frame : frames) {
    JXL_RETURN_IF_ERROR(frame.ShrinkTo(new_xsize, new_ysize));
  }
  info.xsize = new_xsize;
  info.ysize = new_ysize;
  return true;
}

}  // namespace extras
}  // namespace jxl
