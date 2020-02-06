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

#include "jxl/enc_file.h"

#include <stddef.h>

#include <type_traits>
#include <utility>
#include <vector>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_frame.h"
#include "jxl/frame_header.h"
#include "jxl/headers.h"
#include "jxl/icc_codec.h"
#include "jxl/image_bundle.h"
#include "jxl/multiframe.h"

namespace jxl {

namespace {

// DC + 'Very Low Frequency'
PassDefinition progressive_passes_dc_vlf[] = {
    {/*num_coefficients=*/2, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/4}};

PassDefinition progressive_passes_dc_lf[] = {
    {/*num_coefficients=*/2, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/4},
    {/*num_coefficients=*/3, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/2}};

PassDefinition progressive_passes_dc_lf_salient_ac[] = {
    {/*num_coefficients=*/2, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/4},
    {/*num_coefficients=*/3, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/2},
    {/*num_coefficients=*/8, /*shift=*/0, /*salient_only=*/true,
     /*suitable_for_downsampling_of_at_least=*/0}};

PassDefinition progressive_passes_dc_lf_salient_ac_other_ac[] = {
    {/*num_coefficients=*/2, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/4},
    {/*num_coefficients=*/3, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/2},
    {/*num_coefficients=*/8, /*shift=*/0, /*salient_only=*/true,
     /*suitable_for_downsampling_of_at_least=*/0},
    {/*num_coefficients=*/8, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/0}};

PassDefinition progressive_passes_dc_quant_ac_full_ac[] = {
    {/*num_coefficients=*/8, /*shift=*/2, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/4},
    {/*num_coefficients=*/8, /*shift=*/1, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/2},
    {/*num_coefficients=*/8, /*shift=*/0, /*salient_only=*/false,
     /*suitable_for_downsampling_of_at_least=*/0},
};

Status EncodePreview(const CompressParams& cparams, const ImageBundle& ib,
                     ThreadPool* pool, BitWriter* JXL_RESTRICT writer) {
  BitWriter preview_writer;
  // TODO(janwas): also support generating preview by downsampling
  if (ib.HasColor()) {
    AuxOut aux_out;
    Multiframe multiframe;
    const AnimationFrame* animation_frame = nullptr;
    PassesEncoderState passes_enc_state;
    JXL_RETURN_IF_ERROR(EncodeFrame(cparams, animation_frame, ib,
                                    &passes_enc_state, pool, &preview_writer,
                                    &aux_out, &multiframe));
    preview_writer.ZeroPadToByte();
  }

  if (preview_writer.BitsWritten() != 0) {
    writer->ZeroPadToByte();
    writer->AppendByteAligned(preview_writer);
  }

  return true;
}

Status MakeImageMetadata(const CompressParams& cparams, const CodecInOut* io,
                         ImageMetadata* metadata) {
  *metadata = io->metadata;

  // Keep ICC profile in lossless modes because a reconstructed profile may be
  // slightly different (quantization).
  const bool lossless_modular =
      cparams.modular_group_mode && cparams.quality_pair.first == 100.0f;
  if (!cparams.brunsli_group_mode && !lossless_modular) {
    metadata->color_encoding.DecideIfWantICC();
  }

  metadata->SetIntensityTarget(cparams.intensity_target);
  return true;
}

Status WriteHeaders(const CompressParams& cparams, const CodecInOut* io,
                    ImageMetadata* metadata, BitWriter* writer,
                    AuxOut* aux_out) {
  // Marker/signature
  BitWriter::Allotment allotment(writer, 16);
  writer->Write(8, 0xFF);
  writer->Write(8, kCodestreamMarker);
  ReclaimAndCharge(writer, &allotment, kLayerHeader, aux_out);

  SizeHeader size;
  JXL_RETURN_IF_ERROR(size.Set(io->xsize(), io->ysize()));
  JXL_RETURN_IF_ERROR(WriteSizeHeader(size, writer, kLayerHeader, aux_out));

  JXL_RETURN_IF_ERROR(MakeImageMetadata(cparams, io, metadata));
  JXL_RETURN_IF_ERROR(
      WriteImageMetadata(*metadata, writer, kLayerHeader, aux_out));

  if (metadata->m2.have_preview) {
    JXL_RETURN_IF_ERROR(
        WritePreviewHeader(io->preview, writer, kLayerHeader, aux_out));
  }

  if (metadata->m2.have_animation) {
    JXL_RETURN_IF_ERROR(
        WriteAnimationHeader(io->animation, writer, kLayerHeader, aux_out));
  }

  return true;
}

}  // namespace

Status EncodeFile(const CompressParams& cparams, const CodecInOut* io,
                  PassesEncoderState* passes_enc_state, PaddedBytes* compressed,
                  AuxOut* aux_out, ThreadPool* pool) {
  io->CheckMetadata();
  BitWriter writer;

  ImageMetadata metadata;
  JXL_RETURN_IF_ERROR(WriteHeaders(cparams, io, &metadata, &writer, aux_out));

  // Only send ICC (at least several hundred bytes) if fields aren't enough.
  if (metadata.color_encoding.WantICC()) {
    JXL_RETURN_IF_ERROR(WriteICC(metadata.color_encoding.ICC(), &writer,
                                 kLayerHeader, aux_out));
  }

  if (metadata.m2.have_preview) {
    JXL_RETURN_IF_ERROR(
        EncodePreview(cparams, io->preview_frame, pool, &writer));
  }

  // Each frame should start on byte boundaries.
  writer.ZeroPadToByte();

  Multiframe multiframe;
  if (cparams.progressive_mode || cparams.qprogressive_mode) {
    if (cparams.saliency_map != nullptr) {
      multiframe.SetSaliencyMap(cparams.saliency_map);
    }
    multiframe.SetSaliencyThreshold(cparams.saliency_threshold);
    if (cparams.qprogressive_mode) {
      multiframe.SetProgressiveMode(
          ProgressiveMode{progressive_passes_dc_quant_ac_full_ac});
    } else {
      switch (cparams.saliency_num_progressive_steps) {
        case 1:
          multiframe.SetProgressiveMode(
              ProgressiveMode{progressive_passes_dc_vlf});
          break;
        case 2:
          multiframe.SetProgressiveMode(
              ProgressiveMode{progressive_passes_dc_lf});
          break;
        case 3:
          multiframe.SetProgressiveMode(
              ProgressiveMode{progressive_passes_dc_lf_salient_ac});
          break;
        case 4:
          if (cparams.saliency_threshold == 0.0f) {
            // No need for a 4th pass if saliency-threshold regards everything
            // as salient.
            multiframe.SetProgressiveMode(
                ProgressiveMode{progressive_passes_dc_lf_salient_ac});
          } else {
            multiframe.SetProgressiveMode(
                ProgressiveMode{progressive_passes_dc_lf_salient_ac_other_ac});
          }
          break;
        default:
          return JXL_FAILURE("Invalid saliency_num_progressive_steps.");
      }
    }
  }

  if (metadata.m2.have_animation) {
    JXL_CHECK(io->animation_frames.size() == io->frames.size());
    for (size_t i = 0; i < io->frames.size(); i++) {
      AnimationFrame animation_frame = io->animation_frames[i];
      animation_frame.nonserialized_have_timecode =
          io->animation.have_timecodes;
      animation_frame.is_last = i == io->frames.size() - 1;
      JXL_RETURN_IF_ERROR(EncodeFrame(cparams, &animation_frame, io->frames[i],
                                      passes_enc_state, pool, &writer, aux_out,
                                      &multiframe));
    }
  } else {
    const AnimationFrame* animation_frame = nullptr;
    JXL_RETURN_IF_ERROR(EncodeFrame(cparams, animation_frame, io->frames[0],
                                    passes_enc_state, pool, &writer, aux_out,
                                    &multiframe));
  }

  *compressed = std::move(writer).TakeBytes();
  return true;
}

}  // namespace jxl
