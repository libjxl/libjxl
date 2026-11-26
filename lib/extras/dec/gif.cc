// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/gif.h"

#include <cstdint>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/size_constraints.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"

#if !JPEGXL_ENABLE_GIF

namespace jxl {
namespace extras {
bool CanDecodeGIF() { return false; }
Status DecodeImageGIF(Span<const uint8_t> bytes, const ColorHints& color_hints,
                      PackedPixelFile* ppf,
                      const SizeConstraints* constraints) {
  return false;
}
}  // namespace extras
}  // namespace jxl

#else  // JPEGXL_ENABLE_GIF

#include <gif_lib.h>
#include <jxl/codestream_header.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/sanitizers.h"

namespace jxl {
namespace extras {

namespace {

struct ReadState {
  Span<const uint8_t> bytes;
};

struct DGifCloser {
  void operator()(GifFileType* const ptr) const { DGifCloseFile(ptr, nullptr); }
};
using GifUniquePtr = std::unique_ptr<GifFileType, DGifCloser>;

struct PackedRgba {
  uint8_t r, g, b, a;
};

struct PackedRgb {
  uint8_t r, g, b;
};

template <typename T>
StatusOr<Span<T>> PackedImageSpan(PackedImage& img) {
  if (img.xsize > 0 &&
      img.ysize > std::numeric_limits<size_t>::max() / img.xsize) {
    return JXL_FAILURE("PackedImage pixel count overflow");
  }
  const size_t count = img.xsize * img.ysize;
  if (count > img.pixels_size / sizeof(T)) {
    return JXL_FAILURE("PackedImage buffer smaller than expected pixel count");
  }
  return Span<T>(static_cast<T*>(img.pixels()), count);
}

template <typename T>
StatusOr<Span<const T>> PackedImageConstSpan(const PackedImage& img) {
  if (img.xsize > 0 &&
      img.ysize > std::numeric_limits<size_t>::max() / img.xsize) {
    return JXL_FAILURE("PackedImage pixel count overflow");
  }
  const size_t count = img.xsize * img.ysize;
  if (count > img.pixels_size / sizeof(T)) {
    return JXL_FAILURE("PackedImage buffer smaller than expected pixel count");
  }
  return Span<const T>(static_cast<const T*>(img.pixels()), count);
}

Status ensure_have_alpha(PackedFrame* frame) {
  if (!frame->extra_channels.empty()) return true;
  const JxlPixelFormat alpha_format{
      /*num_channels=*/1u,
      /*data_type=*/JXL_TYPE_UINT8,
      /*endianness=*/JXL_NATIVE_ENDIAN,
      /*align=*/0,
  };
  JXL_ASSIGN_OR_RETURN(PackedImage image,
                       PackedImage::Create(frame->color.xsize,
                                           frame->color.ysize, alpha_format));
  frame->extra_channels.emplace_back(std::move(image));
  // We need to set opaque-by-default.
  JXL_ASSIGN_OR_RETURN(Span<uint8_t> alpha_span,
                       PackedImageSpan<uint8_t>(frame->extra_channels[0]));
  std::fill(alpha_span.begin(), alpha_span.end(), static_cast<uint8_t>(255u));
  return true;
}
}  // namespace

bool CanDecodeGIF() { return true; }

Status DecodeImageGIF(Span<const uint8_t> bytes, const ColorHints& color_hints,
                      PackedPixelFile* ppf,
                      const SizeConstraints* constraints) {
  int error = GIF_OK;
  ReadState state = {bytes};
  const auto ReadFromSpan = [](GifFileType* const gif, GifByteType* const out,
                               int n) -> int {
    ReadState* const state = reinterpret_cast<ReadState*>(gif->UserData);
    // giflib API requires the input size `n` to be signed int.
    if (n <= 0 || out == nullptr) return 0;
    const size_t to_read =
        std::min(static_cast<size_t>(n), state->bytes.size());
    if (to_read == 0) return 0;
    const Span<const uint8_t> chunk(state->bytes.data(), to_read);
    memcpy(out, chunk.data(), chunk.size());
    if (!state->bytes.remove_prefix(chunk.size())) return 0;
    return static_cast<int>(chunk.size());
  };
  GifUniquePtr gif(DGifOpen(&state, ReadFromSpan, &error));
  if (gif == nullptr) {
    if (error == D_GIF_ERR_NOT_GIF_FILE) {
      // Not an error.
      return false;
    } else {
      return JXL_FAILURE("Failed to read GIF: %s", GifErrorString(error));
    }
  }
  error = DGifSlurp(gif.get());
  if (error != GIF_OK) {
    return JXL_FAILURE("Failed to read GIF: %s", GifErrorString(gif->Error));
  }

  msan::UnpoisonMemory(gif.get(), sizeof(*gif));
  if (gif->SWidth < 0 || gif->SHeight < 0 || gif->ImageCount < 0) {
    return JXL_FAILURE("Invalid negative GIF header dimensions or image count");
  }
  if (gif->SColorMap) {
    msan::UnpoisonMemory(gif->SColorMap, sizeof(*gif->SColorMap));
    if (gif->SColorMap->ColorCount < 0 ||
        (gif->SColorMap->ColorCount > 0 && gif->SColorMap->Colors == nullptr)) {
      return JXL_FAILURE("Invalid GIF global color map");
    }
    msan::UnpoisonMemory(
        gif->SColorMap->Colors,
        sizeof(*gif->SColorMap->Colors) *
            static_cast<size_t>(gif->SColorMap->ColorCount));
  }
  const size_t image_count = static_cast<size_t>(gif->ImageCount);
  if (image_count > 0 && gif->SavedImages == nullptr) {
    return JXL_FAILURE("Missing GIF saved images");
  }
  msan::UnpoisonMemory(gif->SavedImages,
                       sizeof(*gif->SavedImages) * image_count);
  const Span<const SavedImage> saved_images(gif->SavedImages, image_count);

  JXL_RETURN_IF_ERROR(
      VerifyDimensions<uint32_t>(constraints, gif->SWidth, gif->SHeight));
  uint64_t total_pixel_count =
      static_cast<uint64_t>(gif->SWidth) * gif->SHeight;
  for (const SavedImage& image : saved_images) {
    if (image.ImageDesc.Left < 0 || image.ImageDesc.Top < 0 ||
        image.ImageDesc.Width < 0 || image.ImageDesc.Height < 0) {
      return JXL_FAILURE("Invalid negative GIF frame dimensions");
    }
    uint32_t w = static_cast<uint32_t>(image.ImageDesc.Width);
    uint32_t h = static_cast<uint32_t>(image.ImageDesc.Height);
    JXL_RETURN_IF_ERROR(VerifyDimensions<uint32_t>(constraints, w, h));
    uint64_t pixel_count = static_cast<uint64_t>(w) * h;
    if (total_pixel_count + pixel_count < total_pixel_count) {
      return JXL_FAILURE("Image too big");
    }
    total_pixel_count += pixel_count;
    if (constraints && (total_pixel_count > constraints->dec_max_pixels)) {
      return JXL_FAILURE("Image too big");
    }
  }

  if (!gif->SColorMap) {
    for (const SavedImage& image : saved_images) {
      if (!image.ImageDesc.ColorMap) {
        return JXL_FAILURE("Missing GIF color map");
      }
    }
  }

  if (gif->ImageCount > 1) {
    ppf->info.have_animation = JXL_TRUE;
    // Delays in GIF are specified in censiseconds.
    ppf->info.animation.tps_numerator = 100;
    ppf->info.animation.tps_denominator = 1;
  }

  ppf->frames.clear();
  ppf->frames.reserve(image_count);

  ppf->info.xsize = gif->SWidth;
  ppf->info.ysize = gif->SHeight;
  ppf->info.bits_per_sample = 8;
  ppf->info.exponent_bits_per_sample = 0;
  // alpha_bits is later set to 8 if we find a frame with transparent pixels.
  ppf->info.alpha_bits = 0;
  ppf->info.alpha_exponent_bits = 0;
  JXL_RETURN_IF_ERROR(ApplyColorHints(color_hints, /*color_already_set=*/false,
                                      /*is_gray=*/false, ppf));

  ppf->info.num_color_channels = 3;

  // Pixel format for the 'canvas' onto which we paint
  // the (potentially individually cropped) GIF frames
  // of an animation.
  const JxlPixelFormat canvas_format{
      /*num_channels=*/4u,
      /*data_type=*/JXL_TYPE_UINT8,
      /*endianness=*/JXL_NATIVE_ENDIAN,
      /*align=*/0,
  };

  // Pixel format for the JXL PackedFrame that goes into the
  // PackedPixelFile. Here, we use 3 color channels, and provide
  // the alpha channel as an extra_channel wherever it is used.
  const JxlPixelFormat packed_frame_format{
      /*num_channels=*/3u,
      /*data_type=*/JXL_TYPE_UINT8,
      /*endianness=*/JXL_NATIVE_ENDIAN,
      /*align=*/0,
  };

  GifColorType background_color;
  if (gif->SColorMap == nullptr || gif->SBackGroundColor < 0 ||
      gif->SBackGroundColor >= gif->SColorMap->ColorCount) {
    background_color = {0, 0, 0};
  } else {
    const Span<const GifColorType> global_colors(
        gif->SColorMap->Colors,
        static_cast<size_t>(gif->SColorMap->ColorCount));
    background_color =
        global_colors[static_cast<size_t>(gif->SBackGroundColor)];
  }
  const PackedRgba background_rgba{background_color.Red, background_color.Green,
                                   background_color.Blue, 0};
  JXL_ASSIGN_OR_RETURN(
      PackedFrame canvas,
      PackedFrame::Create(gif->SWidth, gif->SHeight, canvas_format));
  {
    JXL_ASSIGN_OR_RETURN(Span<PackedRgba> canvas_span,
                         PackedImageSpan<PackedRgba>(canvas.color));
    std::fill(canvas_span.begin(), canvas_span.end(), background_rgba);
  }
  Rect canvas_rect{0, 0, canvas.color.xsize, canvas.color.ysize};

  Rect previous_rect_if_restore_to_background;

  bool replace = true;
  bool last_base_was_none = true;
  for (size_t i = 0; i < saved_images.size(); ++i) {
    const SavedImage& image = saved_images[i];
    const size_t img_w = static_cast<size_t>(image.ImageDesc.Width);
    const size_t img_h = static_cast<size_t>(image.ImageDesc.Height);
    const size_t raster_size = img_w * img_h;
    if (raster_size > 0 && image.RasterBits == nullptr) {
      return JXL_FAILURE("Missing GIF raster data");
    }
    msan::UnpoisonMemory(image.RasterBits,
                         sizeof(*image.RasterBits) * raster_size);
    const Span<const GifByteType> raster_bits(image.RasterBits, raster_size);
    const Rect image_rect(static_cast<size_t>(image.ImageDesc.Left),
                          static_cast<size_t>(image.ImageDesc.Top), img_w,
                          img_h);

    Rect total_rect;
    if (previous_rect_if_restore_to_background.xsize() != 0 ||
        previous_rect_if_restore_to_background.ysize() != 0) {
      const size_t xbegin = std::min(
          image_rect.x0(), previous_rect_if_restore_to_background.x0());
      const size_t ybegin = std::min(
          image_rect.y0(), previous_rect_if_restore_to_background.y0());
      const size_t xend =
          std::max(image_rect.x0() + image_rect.xsize(),
                   previous_rect_if_restore_to_background.x0() +
                       previous_rect_if_restore_to_background.xsize());
      const size_t yend =
          std::max(image_rect.y0() + image_rect.ysize(),
                   previous_rect_if_restore_to_background.y0() +
                       previous_rect_if_restore_to_background.ysize());
      total_rect = Rect(xbegin, ybegin, xend - xbegin, yend - ybegin);
      previous_rect_if_restore_to_background = Rect();
      replace = true;
    } else {
      total_rect = image_rect;
      replace = false;
    }
    if (!image_rect.IsInside(canvas_rect)) {
      return JXL_FAILURE("GIF frame extends outside of the canvas");
    }

    // Allocates the frame buffer.
    {
      JXL_ASSIGN_OR_RETURN(
          PackedFrame frame,
          PackedFrame::Create(total_rect.xsize(), total_rect.ysize(),
                              packed_frame_format));
      ppf->frames.emplace_back(std::move(frame));
    }

    PackedFrame* frame = &ppf->frames.back();

    // We cannot tell right from the start whether there will be a
    // need for an alpha channel. This is discovered only as soon as
    // we see a transparent pixel. We hence initialize alpha lazily.
    auto set_pixel_alpha = [&frame](size_t x, size_t y, uint8_t a) -> Status {
      // If we do not have an alpha-channel and a==255 (fully opaque),
      // we can skip setting this pixel-value and rely on
      // "no alpha channel = no transparency".
      if (a == 255 && frame->extra_channels.empty()) return true;
      JXL_RETURN_IF_ERROR(ensure_have_alpha(frame));
      JXL_ASSIGN_OR_RETURN(Span<uint8_t> alpha_span,
                           PackedImageSpan<uint8_t>(frame->extra_channels[0]));
      JXL_ASSIGN_OR_RETURN(
          Span<uint8_t> alpha_row,
          alpha_span.subspan(y * frame->color.xsize, frame->color.xsize));
      if (x >= alpha_row.size()) {
        return JXL_FAILURE("GIF alpha pixel coordinate out of bounds");
      }
      alpha_row[x] = a;
      return true;
    };

    const ColorMapObject* const color_map =
        image.ImageDesc.ColorMap ? image.ImageDesc.ColorMap : gif->SColorMap;
    JXL_ENSURE(color_map);
    msan::UnpoisonMemory(color_map, sizeof(*color_map));
    if (color_map->ColorCount < 0 ||
        (color_map->ColorCount > 0 && color_map->Colors == nullptr)) {
      return JXL_FAILURE("Invalid GIF frame color map");
    }
    const size_t color_count = static_cast<size_t>(color_map->ColorCount);
    msan::UnpoisonMemory(color_map->Colors,
                         sizeof(*color_map->Colors) * color_count);
    const Span<const GifColorType> colors(color_map->Colors, color_count);

    GraphicsControlBlock gcb;
    DGifSavedExtensionToGCB(gif.get(), static_cast<int>(i), &gcb);
    msan::UnpoisonMemory(&gcb, sizeof(gcb));
    bool is_full_size = total_rect.x0() == 0 && total_rect.y0() == 0 &&
                        total_rect.xsize() == canvas.color.xsize &&
                        total_rect.ysize() == canvas.color.ysize;
    if (ppf->info.have_animation) {
      frame->frame_info.duration = gcb.DelayTime;
      frame->frame_info.layer_info.have_crop = static_cast<int>(!is_full_size);
      frame->frame_info.layer_info.crop_x0 = total_rect.x0();
      frame->frame_info.layer_info.crop_y0 = total_rect.y0();
      frame->frame_info.layer_info.xsize = frame->color.xsize;
      frame->frame_info.layer_info.ysize = frame->color.ysize;
      if (last_base_was_none) {
        replace = true;
      }
      frame->frame_info.layer_info.blend_info.blendmode =
          replace ? JXL_BLEND_REPLACE : JXL_BLEND_BLEND;
      // We always only reference at most the last frame
      frame->frame_info.layer_info.blend_info.source =
          last_base_was_none ? 0u : 1u;
      frame->frame_info.layer_info.blend_info.clamp = 1;
      frame->frame_info.layer_info.blend_info.alpha = 0;
      // TODO(veluca): this could in principle be implemented.
      if (last_base_was_none &&
          (total_rect.x0() != 0 || total_rect.y0() != 0 ||
           total_rect.xsize() != canvas.color.xsize ||
           total_rect.ysize() != canvas.color.ysize || !replace)) {
        if (!JXL_IS_DEBUG_BUILD) {
          fprintf(stderr,
              "GIF with dispose-to-0 is not supported for non-full or blended "
              "frames\n");
        }
        return JXL_FAILURE(
            "GIF with dispose-to-0 is not supported"
            "for non-full or blended frames");
      }
      switch (gcb.DisposalMode) {
        case DISPOSE_DO_NOT:
        case DISPOSE_BACKGROUND:
          frame->frame_info.layer_info.save_as_reference = 1u;
          last_base_was_none = false;
          break;
        case DISPOSE_PREVIOUS:
          frame->frame_info.layer_info.save_as_reference = 0u;
          break;
        default:
          frame->frame_info.layer_info.save_as_reference = 0u;
          last_base_was_none = true;
      }
    }

    // Update the canvas by creating a copy first.
    JXL_ASSIGN_OR_RETURN(
        PackedImage new_canvas_image,
        PackedImage::Create(canvas.color.xsize, canvas.color.ysize,
                            canvas.color.format));
    JXL_ASSIGN_OR_RETURN(Span<PackedRgba> new_canvas_span,
                         PackedImageSpan<PackedRgba>(new_canvas_image));
    {
      JXL_ASSIGN_OR_RETURN(Span<const PackedRgba> old_canvas_span,
                           PackedImageConstSpan<PackedRgba>(canvas.color));
      if (old_canvas_span.size() != new_canvas_span.size()) {
        return JXL_FAILURE("GIF canvas size mismatch");
      }
      std::copy(old_canvas_span.begin(), old_canvas_span.end(),
                new_canvas_span.begin());
    }
    for (size_t y = 0; y < image_rect.ysize(); ++y) {
      JXL_ASSIGN_OR_RETURN(
          Span<PackedRgba> row,
          new_canvas_span.subspan(
              (y + image_rect.y0()) * new_canvas_image.xsize + image_rect.x0(),
              image_rect.xsize()));
      JXL_ASSIGN_OR_RETURN(
          Span<const GifByteType> raster_row,
          raster_bits.subspan(y * image_rect.xsize(), image_rect.xsize()));
      for (size_t x = 0; x < image_rect.xsize(); ++x) {
        const GifByteType byte = raster_row[x];
        if (static_cast<size_t>(byte) >= colors.size()) {
          return JXL_FAILURE("GIF color is out of bounds");
        }

        if (byte == gcb.TransparentColor) continue;
        GifColorType color = colors[byte];
        row[x].r = color.Red;
        row[x].g = color.Green;
        row[x].b = color.Blue;
        row[x].a = 255;
      }
    }
    PackedImage& sub_frame_image = frame->color;
    JXL_ASSIGN_OR_RETURN(Span<PackedRgb> sub_frame_span,
                         PackedImageSpan<PackedRgb>(sub_frame_image));
    if (replace) {
      // Copy from the new canvas image to the subframe
      for (size_t y = 0; y < total_rect.ysize(); ++y) {
        JXL_ASSIGN_OR_RETURN(
            Span<PackedRgba> row_in,
            new_canvas_span.subspan(
                (y + total_rect.y0()) * new_canvas_image.xsize +
                    total_rect.x0(),
                sub_frame_image.xsize));
        JXL_ASSIGN_OR_RETURN(
            Span<PackedRgb> row_out,
            sub_frame_span.subspan(y * sub_frame_image.xsize,
                                   sub_frame_image.xsize));
        for (size_t x = 0; x < sub_frame_image.xsize; ++x) {
          row_out[x].r = row_in[x].r;
          row_out[x].g = row_in[x].g;
          row_out[x].b = row_in[x].b;
          JXL_RETURN_IF_ERROR(set_pixel_alpha(x, y, row_in[x].a));
        }
      }
    } else {
      for (size_t y = 0; y < image_rect.ysize(); ++y) {
        JXL_ASSIGN_OR_RETURN(
            Span<PackedRgb> row,
            sub_frame_span.subspan(y * sub_frame_image.xsize,
                                   image_rect.xsize()));
        JXL_ASSIGN_OR_RETURN(
            Span<const GifByteType> raster_row,
            raster_bits.subspan(y * image_rect.xsize(), image_rect.xsize()));
        for (size_t x = 0; x < image_rect.xsize(); ++x) {
          const GifByteType byte = raster_row[x];
          if (static_cast<size_t>(byte) >= colors.size()) {
            return JXL_FAILURE("GIF color is out of bounds");
          }
          if (byte == gcb.TransparentColor) {
            row[x].r = 0;
            row[x].g = 0;
            row[x].b = 0;
            JXL_RETURN_IF_ERROR(set_pixel_alpha(x, y, 0));
            continue;
          }
          GifColorType color = colors[byte];
          row[x].r = color.Red;
          row[x].g = color.Green;
          row[x].b = color.Blue;
          JXL_RETURN_IF_ERROR(set_pixel_alpha(x, y, 255));
        }
      }
    }

    if (!frame->extra_channels.empty()) {
      ppf->info.alpha_bits = 8;
    }

    switch (gcb.DisposalMode) {
      case DISPOSE_DO_NOT:
        canvas.color = std::move(new_canvas_image);
        break;

      case DISPOSE_BACKGROUND: {
        JXL_ASSIGN_OR_RETURN(Span<PackedRgba> canvas_span,
                             PackedImageSpan<PackedRgba>(canvas.color));
        std::fill(canvas_span.begin(), canvas_span.end(), background_rgba);
        previous_rect_if_restore_to_background = image_rect;
        break;
      }

      case DISPOSE_PREVIOUS:
        break;

      case DISPOSAL_UNSPECIFIED:
      default: {
        JXL_ASSIGN_OR_RETURN(Span<PackedRgba> canvas_span,
                             PackedImageSpan<PackedRgba>(canvas.color));
        std::fill(canvas_span.begin(), canvas_span.end(), background_rgba);
      }
    }
  }
  // Finally, if any frame has an alpha-channel, every frame will need
  // to have an alpha-channel.
  bool seen_alpha = false;
  for (const PackedFrame& frame : ppf->frames) {
    if (!frame.extra_channels.empty()) {
      seen_alpha = true;
      break;
    }
  }
  if (seen_alpha) {
    for (PackedFrame& frame : ppf->frames) {
      JXL_RETURN_IF_ERROR(ensure_have_alpha(&frame));
    }
  }
  return true;
}

}  // namespace extras
}  // namespace jxl

#endif  // JPEGXL_ENABLE_GIF
