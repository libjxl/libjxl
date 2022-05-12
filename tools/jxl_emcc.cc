// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/render_hdr.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/enc_color_management.h"

extern "C" {

void* jxlDecompress(const uint8_t* jxl, size_t size, bool want_sdr,
                    uint32_t display_nits) {
  jxl::ThreadPoolInternal pool(4);
  jxl::DecompressParams dparams;
  jxl::CodecInOut image;
  auto status = jxl::DecodeFile(dparams, jxl::Span<const uint8_t>(jxl, size),
                                &image, &pool);
  if (!status) {
    return reinterpret_cast<void*>(1);
  }

  if (display_nits != 0) {
    status = RenderHDR(&image, display_nits, &pool);
    if (!status) {
      return reinterpret_cast<void*>(6);
    }
  }

  JxlColorEncoding rec2100_pq;
  rec2100_pq.color_space = JxlColorSpace::JXL_COLOR_SPACE_RGB;
  rec2100_pq.primaries = JxlPrimaries::JXL_PRIMARIES_2100;
  if (want_sdr) {
    rec2100_pq.transfer_function =
        JxlTransferFunction::JXL_TRANSFER_FUNCTION_SRGB;
  } else {
    rec2100_pq.transfer_function =
        JxlTransferFunction::JXL_TRANSFER_FUNCTION_PQ;
  }
  rec2100_pq.white_point = JxlWhitePoint::JXL_WHITE_POINT_D65;
  rec2100_pq.rendering_intent =
      JxlRenderingIntent::JXL_RENDERING_INTENT_ABSOLUTE;

  jxl::ColorEncoding colorEncoding;
  status =
      jxl::ConvertExternalToInternalColorEncoding(rec2100_pq, &colorEncoding);
  if (!status) {
    return reinterpret_cast<void*>(2);
  }

  image.metadata.m.color_encoding = colorEncoding;
  status = image.TransformTo(colorEncoding, jxl::GetJxlCms(), &pool);
  if (!status) {
    return reinterpret_cast<void*>(3);
  }

  auto main = image.Main().color();
  size_t w = main->xsize();
  size_t h = main->ysize();
  size_t sample_size = want_sdr ? sizeof(uint8_t) : sizeof(float);

  void* result = malloc(2 * sizeof(uint32_t) + w * h * 4 * sample_size);
  if (!result) {
    return reinterpret_cast<void*>(4);
  }

  uint32_t* header = reinterpret_cast<uint32_t*>(result);
  header[0] = w;
  header[1] = h;
  float* float_pixels = reinterpret_cast<float*>(header + 2);
  uint8_t* byte_pixels = reinterpret_cast<uint8_t*>(header + 2);

  size_t slice_size = 1;

  const auto convert_to_float = [&](const int slice, const int thread) {
    size_t y0 = slice * slice_size;
    size_t y1 = std::min(h, y0 + slice_size);
    for (size_t y = y0; y < y1; ++y) {
      float* out = float_pixels + y * w * 4;
      const float* r = main->ConstPlaneRow(0, y);
      const float* g = main->ConstPlaneRow(1, y);
      const float* b = main->ConstPlaneRow(2, y);
      for (size_t x = 0; x < w; ++x) {
        out[4 * x + 0] = r[x];
        out[4 * x + 1] = g[x];
        out[4 * x + 2] = b[x];
        out[4 * x + 3] = 1.0f;
      }
    }
  };

  const auto convert_to_bytes = [&](const int slice, const int thread) {
    size_t y0 = slice * slice_size;
    size_t y1 = std::min(h, y0 + slice_size);
    for (size_t y = y0; y < y1; ++y) {
      uint8_t* out = byte_pixels + y * w * 4;
      const float* r = main->ConstPlaneRow(0, y);
      const float* g = main->ConstPlaneRow(1, y);
      const float* b = main->ConstPlaneRow(2, y);
      for (size_t x = 0; x < w; ++x) {
        out[4 * x + 0] = 255.0f * r[x];
        out[4 * x + 1] = 255.0f * g[x];
        out[4 * x + 2] = 255.0f * b[x];
        out[4 * x + 3] = 255;
      }
    }
  };

  if (want_sdr) {
    status =
        jxl::RunOnPool(&pool, 0, jxl::DivCeil(h, slice_size),
                       jxl::ThreadPool::NoInit, convert_to_bytes, "convert");
  } else {
    status =
        jxl::RunOnPool(&pool, 0, jxl::DivCeil(h, slice_size),
                       jxl::ThreadPool::NoInit, convert_to_float, "convert");
  }
  if (!status) {
    return reinterpret_cast<void*>(5);
  }

  return result;
}

}  // extern "C"
