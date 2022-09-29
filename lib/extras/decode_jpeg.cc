// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/decode_jpeg.h"

#include "lib/jxl/base/status.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_group.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/jpeg/enc_jpeg_data.h"
#include "lib/jxl/jpeg/enc_jpeg_data_reader.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/passes_state.h"

namespace jxl {
namespace extras {

namespace {

Status SetChromaSubsamplingFromJpegData(const jpeg::JPEGData& jpeg_data,
                                        YCbCrChromaSubsampling* cs) {
  size_t nbcomp = jpeg_data.components.size();
  if (nbcomp == 3) {
    uint8_t hsample[3], vsample[3];
    for (size_t i = 0; i < nbcomp; i++) {
      hsample[i] = jpeg_data.components[i].h_samp_factor;
      vsample[i] = jpeg_data.components[i].v_samp_factor;
    }
    JXL_RETURN_IF_ERROR(cs->Set(hsample, vsample));
  } else if (nbcomp == 1) {
    uint8_t hsample[3], vsample[3];
    for (size_t i = 0; i < 3; i++) {
      hsample[i] = jpeg_data.components[0].h_samp_factor;
      vsample[i] = jpeg_data.components[0].v_samp_factor;
    }
    JXL_RETURN_IF_ERROR(cs->Set(hsample, vsample));
  }
  return true;
}

Status SetColorTransformFromJpegData(const jpeg::JPEGData& jpeg_data,
                                     ColorTransform* color_transform) {
  size_t nbcomp = jpeg_data.components.size();
  bool is_rgb = false;
  const auto& markers = jpeg_data.marker_order;
  // If there is a JFIF marker, this is YCbCr. Otherwise...
  if (std::find(markers.begin(), markers.end(), 0xE0) == markers.end()) {
    // Try to find an 'Adobe' marker.
    size_t app_markers = 0;
    size_t i = 0;
    for (; i < markers.size(); i++) {
      // This is an APP marker.
      if ((markers[i] & 0xF0) == 0xE0) {
        JXL_CHECK(app_markers < jpeg_data.app_data.size());
        // APP14 marker
        if (markers[i] == 0xEE) {
          const auto& data = jpeg_data.app_data[app_markers];
          if (data.size() == 15 && data[3] == 'A' && data[4] == 'd' &&
              data[5] == 'o' && data[6] == 'b' && data[7] == 'e') {
            // 'Adobe' marker.
            is_rgb = data[14] == 0;
            break;
          }
        }
        app_markers++;
      }
    }

    if (i == markers.size()) {
      // No 'Adobe' marker, guess from component IDs.
      is_rgb = nbcomp == 3 && jpeg_data.components[0].id == 'R' &&
               jpeg_data.components[1].id == 'G' &&
               jpeg_data.components[2].id == 'B';
    }
  }
  *color_transform =
      (!is_rgb || nbcomp == 1) ? ColorTransform::kYCbCr : ColorTransform::kNone;
  return true;
}

void RoundtripDC(const Image3F& dc, size_t group_index,
                 PassesSharedState* shared) {
  const Rect r = shared->DCGroupRect(group_index);

  Image quant_dc(r.xsize(), r.ysize(), 8, 3);
  if (shared->frame_header.chroma_subsampling.Is444()) {
    for (size_t c : {1, 0, 2}) {
      float inv_factor = shared->quantizer.GetInvDcStep(c);
      float y_factor = shared->quantizer.GetDcStep(1);
      float cfl_factor = shared->cmap.DCFactors()[c];
      for (size_t y = 0; y < r.ysize(); y++) {
        int32_t* quant_row = quant_dc.channel[c < 2 ? c ^ 1 : c].plane.Row(y);
        const float* row = r.ConstPlaneRow(dc, c, y);
        if (c == 1) {
          for (size_t x = 0; x < r.xsize(); x++) {
            quant_row[x] = roundf(row[x] * inv_factor);
          }
        } else {
          int32_t* quant_row_y = quant_dc.channel[0].plane.Row(y);
          for (size_t x = 0; x < r.xsize(); x++) {
            quant_row[x] =
                roundf((row[x] - quant_row_y[x] * (y_factor * cfl_factor)) *
                       inv_factor);
          }
        }
      }
    }
  } else {
    for (size_t c : {1, 0, 2}) {
      Rect rect(r.x0() >> shared->frame_header.chroma_subsampling.HShift(c),
                r.y0() >> shared->frame_header.chroma_subsampling.VShift(c),
                r.xsize() >> shared->frame_header.chroma_subsampling.HShift(c),
                r.ysize() >> shared->frame_header.chroma_subsampling.VShift(c));
      float inv_factor = shared->quantizer.GetInvDcStep(c);
      size_t ys = rect.ysize();
      size_t xs = rect.xsize();
      Channel& ch = quant_dc.channel[c < 2 ? c ^ 1 : c];
      ch.w = xs;
      ch.h = ys;
      ch.shrink();
      for (size_t y = 0; y < ys; y++) {
        int32_t* quant_row = ch.plane.Row(y);
        const float* row = rect.ConstPlaneRow(dc, c, y);
        for (size_t x = 0; x < xs; x++) {
          quant_row[x] = roundf(row[x] * inv_factor);
        }
      }
    }
  }

  DequantDC(r, &shared->dc_storage, &shared->quant_dc, quant_dc,
            shared->quantizer.MulDC(), 1.0, shared->cmap.DCFactors(),
            shared->frame_header.chroma_subsampling, shared->block_ctx_map);
}

Status ComputeJPEGTranscodingData(const jpeg::JPEGData& jpeg_data,
                                  ThreadPool* pool,
                                  std::vector<std::unique_ptr<ACImage>>* coeffs,
                                  PassesSharedState* shared) {
  shared->frame_header.x_qm_scale = 2;
  shared->frame_header.b_qm_scale = 2;

  const FrameDimensions& frame_dim = shared->frame_dim;
  const size_t xsize = frame_dim.xsize_padded;
  const size_t ysize = frame_dim.ysize_padded;
  const size_t xsize_blocks = frame_dim.xsize_blocks;
  const size_t ysize_blocks = frame_dim.ysize_blocks;

  // no-op chroma from luma
  shared->cmap = ColorCorrelationMap(xsize, ysize, false);
  shared->ac_strategy.FillDCT8();
  FillImage(uint8_t(0), &shared->epf_sharpness);

  coeffs->emplace_back(make_unique<ACImageT<int32_t>>(kGroupDim * kGroupDim,
                                                      frame_dim.num_groups));

  // convert JPEG quantization table to a Quantizer object
  float dcquantization[3];
  std::vector<QuantEncoding> qe(DequantMatrices::kNum,
                                QuantEncoding::Library(0));

  auto jpeg_c_map = JpegOrder(shared->frame_header.color_transform,
                              jpeg_data.components.size() == 1);

  std::vector<int> qt(192);
  for (size_t c = 0; c < 3; c++) {
    size_t jpeg_c = jpeg_c_map[c];
    const int32_t* quant =
        jpeg_data.quant[jpeg_data.components[jpeg_c].quant_idx].values.data();

    dcquantization[c] = 255 * 8.0f / quant[0];
    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        // JPEG XL transposes the DCT, JPEG doesn't.
        qt[c * 64 + 8 * x + y] = quant[8 * y + x];
      }
    }
  }
  DequantMatricesSetCustomDC(&shared->matrices, dcquantization);
  float dcquantization_r[3] = {1.0f / dcquantization[0],
                               1.0f / dcquantization[1],
                               1.0f / dcquantization[2]};

  qe[AcStrategy::Type::DCT] = QuantEncoding::RAW(qt);
  shared->matrices.SetEncodings(qe);
  DequantMatricesRoundtrip(&shared->matrices);
  JXL_RETURN_IF_ERROR(shared->matrices.EnsureComputed(1));

  // Ensure that InvGlobalScale() is 1.
  shared->quantizer = Quantizer(&shared->matrices, 1, kGlobalScaleDenom);
  // Recompute MulDC() and InvMulDC().
  shared->quantizer.RecomputeFromGlobalScale();

  // Per-block dequant scaling should be 1.
  FillImage(static_cast<int32_t>(shared->quantizer.InvGlobalScale()),
            &shared->raw_quant_field);

  auto jpeg_row = [&](size_t c, size_t y) {
    return jpeg_data.components[jpeg_c_map[c]].coeffs.data() +
           jpeg_data.components[jpeg_c_map[c]].width_in_blocks * kDCTBlockSize *
               y;
  };

  Image3F dc = Image3F(xsize_blocks, ysize_blocks);
  bool DCzero =
      (shared->frame_header.color_transform == ColorTransform::kYCbCr);
  if (!shared->frame_header.chroma_subsampling.Is444()) {
    ZeroFillImage(&dc);
    (*coeffs)[0]->ZeroFill();
  }
  // JPEG DC is from -1024 to 1023.
  std::vector<size_t> dc_counts[3] = {};
  dc_counts[0].resize(2048);
  dc_counts[1].resize(2048);
  dc_counts[2].resize(2048);
  size_t total_dc[3] = {};
  for (size_t c : {1, 0, 2}) {
    if (jpeg_data.components.size() == 1 && c != 1) {
      (*coeffs)[0]->ZeroFillPlane(c);
      ZeroFillImage(&dc.Plane(c));
      // Ensure no division by 0.
      dc_counts[c][1024] = 1;
      total_dc[c] = 1;
      continue;
    }
    size_t hshift = shared->frame_header.chroma_subsampling.HShift(c);
    size_t vshift = shared->frame_header.chroma_subsampling.VShift(c);
    for (size_t group_index = 0; group_index < frame_dim.num_groups;
         group_index++) {
      const size_t gx = group_index % frame_dim.xsize_groups;
      const size_t gy = group_index / frame_dim.xsize_groups;
      size_t offset = 0;
      int32_t* JXL_RESTRICT ac =
          (*coeffs)[0]->PlaneRow(c, group_index, 0).ptr32;
      for (size_t by = gy * kGroupDimInBlocks;
           by < ysize_blocks && by < (gy + 1) * kGroupDimInBlocks; ++by) {
        if ((by >> vshift) << vshift != by) continue;
        const int16_t* JXL_RESTRICT inputjpeg = jpeg_row(c, by >> vshift);
        float* JXL_RESTRICT fdc = dc.PlaneRow(c, by >> vshift);
        for (size_t bx = gx * kGroupDimInBlocks;
             bx < xsize_blocks && bx < (gx + 1) * kGroupDimInBlocks; ++bx) {
          if ((bx >> hshift) << hshift != bx) continue;
          size_t base = (bx >> hshift) * kDCTBlockSize;
          int idc;
          if (DCzero) {
            idc = inputjpeg[base];
          } else {
            idc = inputjpeg[base] + 1024 / qt[c * 64];
          }
          dc_counts[c][std::min(static_cast<uint32_t>(idc + 1024),
                                uint32_t(2047))]++;
          total_dc[c]++;
          fdc[bx >> hshift] = idc * dcquantization_r[c];
          for (size_t y = 0; y < 8; y++) {
            for (size_t x = 0; x < 8; x++) {
              ac[offset + y * 8 + x] = inputjpeg[base + x * 8 + y];
            }
          }
          offset += 64;
        }
      }
    }
  }

  // disable DC frame for now
  shared->frame_header.UpdateFlag(false, FrameHeader::kUseDcFrame);
  auto compute_dc_coeffs = [&](const uint32_t group_index,
                               size_t /* thread */) {
    RoundtripDC(dc, group_index, shared);
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_dc_groups,
                                ThreadPool::NoInit, compute_dc_coeffs,
                                "Compute DC coeffs"));

  // Must happen before WriteFrameHeader!
  shared->frame_header.UpdateFlag(true, FrameHeader::kSkipAdaptiveDCSmoothing);

  return true;
}
}  // namespace

Status DecodeJpeg(const std::vector<uint8_t>& compressed,
                  JxlDataType output_data_type, ThreadPool* pool,
                  PackedPixelFile* ppf) {
  jpeg::JPEGData jpeg_data;
  JXL_RETURN_IF_ERROR(jpeg::ReadJpeg(compressed.data(), compressed.size(),
                                     jpeg::JpegReadMode::kReadAll, &jpeg_data));
  size_t xsize = jpeg_data.width;
  size_t ysize = jpeg_data.height;
  uint32_t nbcomp = jpeg_data.components.size();

  CodecMetadata metadata;
  JXL_RETURN_IF_ERROR(
      SetColorEncodingFromJpegData(jpeg_data, &metadata.m.color_encoding));
  JXL_RETURN_IF_ERROR(metadata.size.Set(xsize, ysize));
  SetIntensityTarget(&metadata.m);
  metadata.m.SetUintSamples(8);  // BITS_IN_JSAMPLE
  metadata.m.xyb_encoded = false;

  size_t output_bits = PackedImage::BitsPerChannel(output_data_type);
  size_t bytes_per_channel = output_bits / 8;
  JxlPixelFormat format = {nbcomp, output_data_type, JXL_LITTLE_ENDIAN, 0};

  ppf->info.xsize = xsize;
  ppf->info.ysize = ysize;
  ppf->info.num_color_channels = nbcomp;
  ppf->info.bits_per_sample = output_bits;

  PaddedBytes icc = metadata.m.color_encoding.ICC();
  ppf->icc.assign(icc.data(), icc.data() + icc.size());
  ConvertInternalToExternalColorEncoding(metadata.m.color_encoding,
                                         &ppf->color_encoding);
  ppf->frames.emplace_back(xsize, ysize, format);
  auto& frame = ppf->frames.back();

  FrameHeader frame_header(&metadata);
  frame_header.nonserialized_is_preview = false;
  frame_header.is_last = true;
  frame_header.encoding = FrameEncoding::kVarDCT;
  frame_header.loop_filter.gab = 0;
  frame_header.loop_filter.epf_iters = 0;
  JXL_RETURN_IF_ERROR(SetChromaSubsamplingFromJpegData(
      jpeg_data, &frame_header.chroma_subsampling));
  JXL_RETURN_IF_ERROR(
      SetColorTransformFromJpegData(jpeg_data, &frame_header.color_transform));

  PassesSharedState shared;
  JXL_RETURN_IF_ERROR(
      InitializePassesSharedState(frame_header, &shared, /*encoder=*/true));

  std::vector<std::unique_ptr<ACImage>> coeffs;
  JXL_RETURN_IF_ERROR(
      ComputeJPEGTranscodingData(jpeg_data, pool, &coeffs, &shared));

  std::unique_ptr<PassesDecoderState> dec_state =
      jxl::make_unique<PassesDecoderState>();
  JXL_CHECK(dec_state->output_encoding_info.SetFromMetadata(*shared.metadata));
  dec_state->shared = &shared;

  const size_t xsize_groups = DivCeil(xsize, kGroupDim);
  const size_t ysize_groups = DivCeil(ysize, kGroupDim);
  const size_t num_groups = xsize_groups * ysize_groups;

  JXL_CHECK(dec_state->Init());
  JXL_CHECK(dec_state->InitForAC(pool));

  dec_state->width = xsize;
  dec_state->height = ysize;
  dec_state->main_output.format = format;
  dec_state->main_output.buffer =
      reinterpret_cast<uint8_t*>(frame.color.pixels());
  dec_state->main_output.buffer_size = frame.color.pixels_size;
  dec_state->main_output.stride =
      xsize * bytes_per_channel * format.num_channels;

  ImageBundle decoded(&shared.metadata->m);
  decoded.origin = shared.frame_header.frame_origin;
  decoded.SetFromImage(Image3F(xsize, ysize),
                       dec_state->output_encoding_info.color_encoding);

  PassesDecoderState::PipelineOptions options;
  options.use_slow_render_pipeline = false;
  options.coalescing = true;
  options.render_spotcolors = false;

  JXL_CHECK(dec_state->PreparePipeline(&decoded, options));

  hwy::AlignedUniquePtr<GroupDecCache[]> group_dec_caches;
  const auto allocate_storage = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(
        dec_state->render_pipeline->PrepareForThreads(num_threads,
                                                      /*use_group_ids=*/false));
    group_dec_caches = hwy::MakeUniqueAlignedArray<GroupDecCache>(num_threads);
    return true;
  };
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) {
    RenderPipelineInput input =
        dec_state->render_pipeline->GetInputBuffers(group_index, thread);
    JXL_CHECK(DecodeGroupForRoundtrip(coeffs, group_index, dec_state.get(),
                                      &group_dec_caches[thread], thread, input,
                                      &decoded, nullptr));
    input.Done();
  };
  JXL_CHECK(RunOnPool(pool, 0, num_groups, allocate_storage, process_group,
                      "Decode Groups"));

  return true;
}

}  // namespace extras
}  // namespace jxl
