// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>

#include <tuple>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tools/cjpeg_hdr.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/extras/codec.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_color_management.h"
#include "lib/jxl/enc_transforms.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/jpeg/dec_jpeg_data_writer.h"
#include "lib/jxl/quant_weights.h"

HWY_BEFORE_NAMESPACE();
namespace jpegxl {
namespace tools {
namespace HWY_NAMESPACE {
void FillJPEGData(const jxl::Image3F& ycbcr, const jxl::PaddedBytes& icc,
                  const jxl::ImageF& quant_field,
                  const jxl::FrameDimensions& frame_dim,
                  jxl::jpeg::JPEGData* out) {
  // JFIF
  out->marker_order.push_back(0xE0);
  out->app_data.emplace_back(std::vector<uint8_t>{
      0xe0,                      // Marker
      0, 16,                     // Length
      'J', 'F', 'I', 'F', '\0',  // ID
      1, 1,                      // Version (1.1)
      0,                         // No density units
      0, 1, 0, 1,                // Pixel density 1
      0, 0                       // No thumbnail
  });
  // ICC
  if (!icc.empty()) {
    out->marker_order.push_back(0xE2);
    std::vector<uint8_t> icc_marker(17 + icc.size());
    icc_marker[0] = 0xe2;
    icc_marker[1] = (icc_marker.size() - 1) >> 8;
    icc_marker[2] = (icc_marker.size() - 1) & 0xFF;
    memcpy(&icc_marker[3], "ICC_PROFILE", 12);
    icc_marker[15] = 1;
    icc_marker[16] = 1;
    memcpy(&icc_marker[17], icc.data(), icc.size());
    out->app_data.push_back(std::move(icc_marker));
  }

  // DQT
  out->marker_order.emplace_back(0xdb);
  out->quant.resize(2);
  out->quant[0].is_last = false;
  out->quant[0].index = 0;
  out->quant[1].is_last = true;
  out->quant[1].index = 1;
  jxl::DequantMatrices dequant;

  // mozjpeg q99
  int qluma[64] = {
      1, 1, 1, 1, 1, 1, 1, 2,  //
      1, 1, 1, 1, 1, 1, 1, 2,  //
      1, 1, 1, 1, 1, 1, 2, 3,  //
      1, 1, 1, 1, 1, 1, 2, 3,  //
      1, 1, 1, 1, 1, 2, 3, 4,  //
      1, 1, 1, 1, 2, 2, 3, 5,  //
      1, 1, 2, 2, 3, 3, 5, 6,  //
      2, 2, 3, 3, 4, 5, 6, 8,  //
  };
  // mozjpeg q95
  int qchroma[64] = {
      2, 2, 2,  2,  3,  4,  6,  9,   //
      2, 2, 2,  3,  3,  4,  5,  8,   //
      2, 2, 2,  3,  4,  6,  9,  14,  //
      2, 3, 3,  4,  5,  7,  11, 16,  //
      3, 3, 4,  5,  7,  9,  13, 19,  //
      4, 4, 6,  7,  9,  12, 17, 24,  //
      6, 5, 9,  11, 13, 17, 23, 31,  //
      9, 8, 14, 16, 19, 24, 31, 42,  //
  };
  // Disable quantization for now.
  std::fill(std::begin(qluma), std::end(qluma), 1);
  std::fill(std::begin(qchroma), std::end(qchroma), 1);

  memcpy(out->quant[0].values.data(), qluma, sizeof(qluma));
  memcpy(out->quant[1].values.data(), qchroma, sizeof(qchroma));

  // SOF
  out->marker_order.emplace_back(0xc2);
  out->components.resize(3);
  out->height = frame_dim.ysize;
  out->width = frame_dim.xsize_padded;
  out->components[0].id = 1;
  out->components[1].id = 2;
  out->components[2].id = 3;
  out->components[0].h_samp_factor = out->components[1].h_samp_factor =
      out->components[2].h_samp_factor = out->components[0].v_samp_factor =
          out->components[1].v_samp_factor = out->components[2].v_samp_factor =
              1;
  out->components[0].width_in_blocks = out->components[1].width_in_blocks =
      out->components[2].width_in_blocks = frame_dim.xsize_blocks;
  out->components[0].quant_idx = 0;
  out->components[1].quant_idx = 1;
  out->components[2].quant_idx = 1;
  out->components[0].coeffs.resize(frame_dim.xsize_blocks *
                                   frame_dim.ysize_blocks * 64);
  out->components[1].coeffs.resize(frame_dim.xsize_blocks *
                                   frame_dim.ysize_blocks * 64);
  out->components[2].coeffs.resize(frame_dim.xsize_blocks *
                                   frame_dim.ysize_blocks * 64);

  HWY_ALIGN float scratch_space[2 * 64];

  for (size_t c = 0; c < 3; c++) {
    int* qt = c == 0 ? qluma : qchroma;
    for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
      for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
        float deadzone = 0.5f / quant_field.Row(by)[bx];
        // Disable quantization for now.
        deadzone = 0;
        auto q = [&](float coeff, size_t x, size_t y) -> int {
          size_t pos = x * 8 + y;
          float scoeff = coeff / qt[pos];
          if (pos == 0) {
            return std::round(scoeff);
          }
          if (std::abs(scoeff) < deadzone) return 0;
          if (std::abs(scoeff) < 2 * deadzone && x + y >= 7) return 0;
          return std::round(scoeff);
        };
        HWY_ALIGN float dct[64];
        TransformFromPixels(jxl::AcStrategy::Type::DCT,
                            ycbcr.PlaneRow(c, 8 * by) + 8 * bx,
                            ycbcr.PixelsPerRow(), dct, scratch_space);
        for (size_t iy = 0; iy < 8; iy++) {
          for (size_t ix = 0; ix < 8; ix++) {
            float coeff = dct[iy * 8 + ix] * 2040;  // not a typo
            out->components[c]
                .coeffs[(frame_dim.xsize_blocks * by + bx) * 64 + ix * 8 + iy] =
                q(coeff, ix, iy);
          }
        }
      }
    }
  }

  // DHT
  // TODO: optimize
  out->marker_order.emplace_back(0xC4);
  out->huffman_code.resize(2);
  out->huffman_code[0].slot_id = 0x00;  // DC
  out->huffman_code[0].counts = {{0, 0, 0, 0, 13}};
  std::iota(out->huffman_code[0].values.begin(),
            out->huffman_code[0].values.end(), 0);
  out->huffman_code[0].is_last = false;

  out->huffman_code[1].slot_id = 0x10;  // AC
  out->huffman_code[1].counts = {{0, 0, 0, 0, 0, 0, 0, 0, 255}};
  std::iota(out->huffman_code[1].values.begin(),
            out->huffman_code[1].values.end(), 0);
  out->huffman_code[1].is_last = true;

  // SOS
  for (size_t _ = 0; _ < 7; _++) {
    out->marker_order.emplace_back(0xDA);
  }
  out->scan_info.resize(7);
  // DC
  // comp id, DC tbl, AC tbl
  out->scan_info[0].num_components = 3;
  out->scan_info[0].components = {{jxl::jpeg::JPEGComponentScanInfo{0, 0, 0},
                                   jxl::jpeg::JPEGComponentScanInfo{1, 0, 0},
                                   jxl::jpeg::JPEGComponentScanInfo{2, 0, 0}}};
  out->scan_info[0].Ss = 0;
  out->scan_info[0].Se = 0;
  out->scan_info[0].Ah = out->scan_info[0].Al = 0;
  // AC 1 - highest bits
  out->scan_info[1].num_components = 1;
  out->scan_info[1].components = {{jxl::jpeg::JPEGComponentScanInfo{0, 0, 0}}};
  out->scan_info[1].Ss = 1;
  out->scan_info[1].Se = 63;
  out->scan_info[1].Ah = 0;
  out->scan_info[1].Al = 1;

  // Copy for X / B-Y
  out->scan_info[2] = out->scan_info[1];
  out->scan_info[2].components[0].comp_idx = 1;
  out->scan_info[3] = out->scan_info[1];
  out->scan_info[3].components[0].comp_idx = 2;

  // AC 2 - lowest bit
  out->scan_info[4].num_components = 1;
  out->scan_info[4].components = {{jxl::jpeg::JPEGComponentScanInfo{0, 0, 0}}};
  out->scan_info[4].Ss = 1;
  out->scan_info[4].Se = 63;
  out->scan_info[4].Ah = 1;
  out->scan_info[4].Al = 0;

  // Copy for X / B-Y
  out->scan_info[5] = out->scan_info[4];
  out->scan_info[5].components[0].comp_idx = 1;
  out->scan_info[6] = out->scan_info[4];
  out->scan_info[6].components[0].comp_idx = 2;

  // EOI
  out->marker_order.push_back(0xd9);
}
}  // namespace HWY_NAMESPACE
}  // namespace tools
}  // namespace jpegxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jpegxl {
namespace tools {

HWY_EXPORT(FillJPEGData);

int HBDJPEGMain(int argc, const char* argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s input output.jpg\n", argv[0]);
    return 1;
  }
  fprintf(stderr, "Compressing %s to %s\n", argv[1], argv[2]);
  jxl::CodecInOut io;
  if (!jxl::SetFromFile(argv[1], jxl::extras::ColorHints{}, &io)) {
    fprintf(stderr, "Failed to read image %s.\n", argv[1]);
    return 1;
  }
  jxl::Image3F ycbcr(jxl::RoundUpToBlockDim(io.xsize()),
                     jxl::RoundUpToBlockDim(io.ysize()));
  ycbcr.ShrinkTo(io.xsize(), io.ysize());
  jxl::FrameDimensions frame_dim;
  frame_dim.Set(io.xsize(), io.ysize(), 0, 0, 0, false, 1);
  for (size_t y = 0; y < ycbcr.ysize(); y++) {
    for (size_t x = 0; x < ycbcr.xsize(); x++) {
      float r = io.Main().color()->PlaneRow(0, y)[x];
      float g = io.Main().color()->PlaneRow(1, y)[x];
      float b = io.Main().color()->PlaneRow(2, y)[x];
      ycbcr.PlaneRow(0, y)[x] =
          0.299 * r + 0.587 * g + 0.114 * b - (128. / 255.);
      ycbcr.PlaneRow(1, y)[x] = -0.168736 * r - 0.331264 * g + 0.5 * b;
      ycbcr.PlaneRow(2, y)[x] = 0.5 * r - 0.418688 * g - 0.081312 * b;
    }
  }
  jxl::Image3F rgb2(ycbcr.xsize(), ycbcr.ysize());
  jxl::Image3F ycbcr2(ycbcr.xsize(), ycbcr.ysize());
  for (size_t y = 0; y < ycbcr.ysize(); y++) {
    for (size_t x = 0; x < ycbcr.xsize(); x++) {
      ycbcr2.PlaneRow(0, y)[x] = ycbcr.PlaneRow(1, y)[x];
      ycbcr2.PlaneRow(1, y)[x] = ycbcr.PlaneRow(0, y)[x];
      ycbcr2.PlaneRow(2, y)[x] = ycbcr.PlaneRow(2, y)[x];
    }
  }
  jxl::YcbcrToRgb(ycbcr2, &rgb2, jxl::Rect(ycbcr));

  PadImageToBlockMultipleInPlace(&ycbcr);

  jxl::Image3F opsin(jxl::RoundUpToBlockDim(io.xsize()),
                     jxl::RoundUpToBlockDim(io.ysize()));
  opsin.ShrinkTo(io.xsize(), io.ysize());
  jxl::ToXYB(io.Main(), nullptr, &opsin, jxl::GetJxlCms());
  PadImageToBlockMultipleInPlace(&opsin);
  jxl::ImageF mask;
  jxl::ImageF qf =
      InitialQuantField(1.0, opsin, frame_dim, nullptr, 1.0, &mask);

  jxl::CodecInOut out;
  out.Main().jpeg_data = jxl::make_unique<jxl::jpeg::JPEGData>();
  HWY_DYNAMIC_DISPATCH(FillJPEGData)
  (ycbcr, io.metadata.m.color_encoding.ICC(), qf, frame_dim,
   out.Main().jpeg_data.get());
  jxl::PaddedBytes output;
  if (!jxl::jpeg::EncodeImageJPGCoefficients(&out, &output)) {
    return 1;
  }
  if (!jxl::WriteFile(output, argv[2])) {
    fprintf(stderr, "Failed to write to \"%s\"\n", argv[2]);
    return 1;
  }
  return 0;
}

}  // namespace tools
}  // namespace jpegxl

int main(int argc, const char** argv) {
  return jpegxl::tools::HBDJPEGMain(argc, argv);
}
#endif
