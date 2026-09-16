// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Helper class for storing external (int or float, interleaved) images. This is
// the common format used by other libraries and in the libjxl API.

#include "packed_image.h"

#include <jxl/codestream_header.h>
#include <jxl/encode.h>
#include <jxl/types.h>

#include <algorithm>
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
#include "lib/jxl/base/float.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {

// Class representing an interleaved image with a bunch of channels.
StatusOr<PackedImage> PackedImage::Create(size_t xsize, size_t ysize,
                                          const JxlPixelFormat& format) {
  JXL_ASSIGN_OR_RETURN(size_t stride, CalcStride(format, xsize));
  size_t pixels_size = ysize * stride;
  if ((pixels_size / stride) != ysize) {
    return JXL_FAILURE("Image too big");
  }
  PackedImage image(xsize, ysize, format, stride);
  if (!image.pixels()) {
    // TODO(szabadka): use specialized OOM error code
    return JXL_FAILURE("Failed to allocate memory for image");
  }
  return image;
}

StatusOr<PackedImage> PackedImage::Copy() const {
  // Resulting copy_stride have to be less or equal to original -> always ok.
  JXL_ASSIGN_OR_RETURN(size_t copy_stride, CalcStride(format, xsize));
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

StatusOr<size_t> PackedImage::CalcStride(const JxlPixelFormat& format,
                                         size_t xsize) {
  size_t multiplier = (BitsPerChannel(format.data_type) * format.num_channels /
                       jxl::kBitsPerByte);
  size_t stride;
  if (!SafeMul(xsize, multiplier, stride)) {
    return JXL_FAILURE("Image too big");
  }
  if (!SafeRoundUpTo(stride, format.align, stride)) {
    return JXL_FAILURE("Image too big");
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
  JXL_ASSIGN_OR_RETURN(copy.color, color.Copy());
  for (const auto& ec : extra_channels) {
    JXL_ASSIGN_OR_RETURN(PackedImage ec_copy, ec.Copy());
    copy.extra_channels.emplace_back(std::move(ec_copy));
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

bool PackedPixelFile::HasAlpha() const {
  if (info.alpha_bits > 0) return true;
  for (const auto& ec : extra_channels_info) {
    if (ec.ec_info.type == JXL_CHANNEL_ALPHA) return true;
  }
  return false;
}

bool PackedPixelFile::HasOpaqueAlpha() const {
  if (!HasAlpha()) return false;
  if (frames.empty()) return false;

  size_t num_color = info.num_color_channels;

  auto is_channel_opaque = [this](const PackedImage& image,
                                  size_t alpha_c) -> bool {
    size_t xs = image.xsize;
    size_t ys = image.ysize;
    size_t stride = image.stride;
    size_t p_stride = image.pixel_stride();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(image.pixels());
    if (!p) return false;

    if (image.format.data_type == JXL_TYPE_UINT8) {
      size_t offset = alpha_c;
      for (size_t y = 0; y < ys; ++y) {
        const uint8_t* row = p + y * stride;
        for (size_t x = 0; x < xs; ++x) {
          if (row[x * p_stride + offset] != 255) {
            return false;
          }
        }
      }
      return true;
    } else if (image.format.data_type == JXL_TYPE_UINT16) {
      size_t offset = alpha_c * 2;
      bool swap = SwapEndianness(image.format.endianness);
      uint16_t expected = 65535;
      if (info.alpha_bits > 0 && info.alpha_bits < 16) {
        expected = (1u << info.alpha_bits) - 1;
      }
      for (size_t y = 0; y < ys; ++y) {
        const uint8_t* row = p + y * stride;
        for (size_t x = 0; x < xs; ++x) {
          uint16_t val;
          memcpy(&val, row + x * p_stride + offset, 2);
          if (swap) val = JXL_BSWAP16(val);
          if (val != expected && val != 65535) {
            return false;
          }
        }
      }
      return true;
    } else if (image.format.data_type == JXL_TYPE_FLOAT) {
      size_t offset = alpha_c * 4;
      bool swap = SwapEndianness(image.format.endianness);
      for (size_t y = 0; y < ys; ++y) {
        const uint8_t* row = p + y * stride;
        for (size_t x = 0; x < xs; ++x) {
          float val;
          memcpy(&val, row + x * p_stride + offset, 4);
          if (swap) val = BSwapFloat(val);
          if (val < 0.9999f) {
            return false;
          }
        }
      }
      return true;
    } else if (image.format.data_type == JXL_TYPE_FLOAT16) {
      size_t offset = alpha_c * 2;
      bool swap = SwapEndianness(image.format.endianness);
      for (size_t y = 0; y < ys; ++y) {
        const uint8_t* row = p + y * stride;
        for (size_t x = 0; x < xs; ++x) {
          uint16_t bits;
          memcpy(&bits, row + x * p_stride + offset, 2);
          if (swap) bits = JXL_BSWAP16(bits);
          float val = jxl::detail::LoadFloat16(bits);
          if (val < 0.9999f) {
            return false;
          }
        }
      }
      return true;
    }
    return false;
  };

  auto is_frame_opaque = [&](const PackedFrame& frame) -> bool {
    // Interleaved alpha
    if (frame.color.format.num_channels == num_color + 1) {
      return is_channel_opaque(frame.color, num_color);
    }
    // Extra channel alpha
    for (size_t i = 0;
         i < extra_channels_info.size() && i < frame.extra_channels.size();
         ++i) {
      if (extra_channels_info[i].ec_info.type == JXL_CHANNEL_ALPHA) {
        return is_channel_opaque(frame.extra_channels[i], 0);
      }
    }
    return false;
  };

  for (const auto& frame : frames) {
    if (!is_frame_opaque(frame)) return false;
  }
  if (preview_frame && !is_frame_opaque(*preview_frame)) {
    return false;
  }
  return true;
}

Status PackedPixelFile::DropAlpha() {
  size_t num_color = info.num_color_channels;

  auto drop_frame_alpha = [&](PackedFrame* frame) -> Status {
    if (!frame) return true;
    if (frame->color.format.num_channels == num_color + 1) {
      JxlPixelFormat new_format = frame->color.format;
      new_format.num_channels = num_color;
      JXL_ASSIGN_OR_RETURN(
          PackedImage new_color,
          PackedImage::Create(frame->color.xsize, frame->color.ysize,
                              new_format));

      size_t bpc = PackedImage::BitsPerChannel(new_format.data_type) /
                   jxl::kBitsPerByte;
      size_t color_bytes_per_pixel = num_color * bpc;
      size_t src_stride = frame->color.stride;
      size_t dst_stride = new_color.stride;
      size_t src_pixel_stride = frame->color.pixel_stride();
      size_t dst_pixel_stride = new_color.pixel_stride();

      const uint8_t* src_base =
          reinterpret_cast<const uint8_t*>(frame->color.pixels());
      uint8_t* dst_base = reinterpret_cast<uint8_t*>(new_color.pixels());

      for (size_t y = 0; y < frame->color.ysize; ++y) {
        const uint8_t* src_row = src_base + y * src_stride;
        uint8_t* dst_row = dst_base + y * dst_stride;
        for (size_t x = 0; x < frame->color.xsize; ++x) {
          memcpy(dst_row + x * dst_pixel_stride,
                 src_row + x * src_pixel_stride, color_bytes_per_pixel);
        }
      }
      frame->color = std::move(new_color);
    }
    return true;
  };

  for (auto& frame : frames) {
    JXL_RETURN_IF_ERROR(drop_frame_alpha(&frame));
  }
  if (preview_frame) {
    JXL_RETURN_IF_ERROR(drop_frame_alpha(preview_frame.get()));
  }

  // Remove separate extra channel if present
  int alpha_ec_idx = -1;
  for (size_t i = 0; i < extra_channels_info.size(); ++i) {
    if (extra_channels_info[i].ec_info.type == JXL_CHANNEL_ALPHA) {
      alpha_ec_idx = static_cast<int>(i);
      break;
    }
  }
  if (alpha_ec_idx >= 0) {
    extra_channels_info.erase(extra_channels_info.begin() + alpha_ec_idx);
    for (auto& frame : frames) {
      if (static_cast<size_t>(alpha_ec_idx) < frame.extra_channels.size()) {
        frame.extra_channels.erase(frame.extra_channels.begin() + alpha_ec_idx);
      }
    }
    if (preview_frame && static_cast<size_t>(alpha_ec_idx) <
                             preview_frame->extra_channels.size()) {
      preview_frame->extra_channels.erase(
          preview_frame->extra_channels.begin() + alpha_ec_idx);
    }
    for (size_t i = 0; i < extra_channels_info.size(); ++i) {
      extra_channels_info[i].index = i;
    }
    if (info.num_extra_channels > 0) {
      info.num_extra_channels--;
    }
  }

  info.alpha_bits = 0;
  info.alpha_exponent_bits = 0;
  info.alpha_premultiplied = JXL_FALSE;

  return true;
}

bool HasAlpha(const PackedPixelFile& ppf) { return ppf.HasAlpha(); }
bool HasOpaqueAlpha(const PackedPixelFile& ppf) { return ppf.HasOpaqueAlpha(); }
Status DropAlpha(PackedPixelFile* ppf) {
  if (!ppf) return JXL_FAILURE("Null ppf");
  return ppf->DropAlpha();
}

}  // namespace extras
}  // namespace jxl
