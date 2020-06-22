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

#include "jxl/dec_frame.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <utility>
#include <vector>

#include "jxl/ac_context.h"
#include "jxl/ac_strategy.h"
#include "jxl/ans_params.h"
#include "jxl/aux_out.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/brunsli.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/coeff_order.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/common.h"
#include "jxl/compressed_dc.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/dec_cache.h"
#include "jxl/dec_group.h"
#include "jxl/dec_modular.h"
#include "jxl/dec_params.h"
#include "jxl/dec_reconstruct.h"
#include "jxl/dec_xyb.h"
#include "jxl/dot_dictionary.h"
#include "jxl/fields.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/loop_filter.h"
#include "jxl/luminance.h"
#include "jxl/multiframe.h"
#include "jxl/passes_state.h"
#include "jxl/patch_dictionary.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"
#include "jxl/splines.h"
#include "jxl/toc.h"

namespace jxl {

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class LossyFrameDecoder {
 public:
  Status Init(const FrameHeader& frame_header, const LoopFilter& loop_filter,
              const ImageMetadata& image_metadata,
              const FrameDimensions& frame_dim, Multiframe* multiframe,
              size_t downsampling, ThreadPool* pool, AuxOut* aux_out) {
    downsampling_ = downsampling;
    pool_ = pool;
    aux_out_ = aux_out;
    if (frame_header.IsLossy() &&
        frame_header.chroma_subsampling != YCbCrChromaSubsampling::k444) {
      return JXL_FAILURE("Chroma subsampling is not allowed in Lossy mode");
    }
    return InitializePassesSharedState(frame_header, loop_filter,
                                       image_metadata, frame_dim, multiframe,
                                       &dec_state_.shared_storage);
  }

  // Sets the number of threads that will be used. The value of the "thread"
  // parameter passed to DecodeDCGroup and DecodeACGroup must be smaller than
  // the "num_threads" passed here.
  void SetNumThreads(size_t num_threads) {
    group_dec_caches_.resize(num_threads);
    dec_state_.EnsureStorage(num_threads);
  }

  Status DecodeGlobalDCInfo(BitReader* reader, const ImageBundle* decoded) {
    PROFILER_FUNC;
    JXL_RETURN_IF_ERROR(dec_state_.shared_storage.matrices.DecodeDC(reader));
    JXL_RETURN_IF_ERROR(dec_state_.shared_storage.quantizer.Decode(reader));

    JXL_RETURN_IF_ERROR(dec_state_.shared_storage.cmap.DecodeDC(reader));

    // Pre-compute info for decoding a group.
    if (decoded->IsJPEG()) {
      dec_state_.shared_storage.quantizer.ClearDCMul();  // Don't dequant DC
    }

    for (size_t c = 0; c < 3; c++) {
      FillImage(dec_state_.shared_storage.quantizer.GetDcStep(c),
                const_cast<ImageF*>(
                    &dec_state_.shared_storage.dc_quant_field.Plane(c)));
    }
    if (downsampling_ != 16) {
      dec_state_.shared_storage.ac_strategy.FillInvalid();
    } else {
      dec_state_.shared_storage.ac_strategy.FillDCT8();
    }
    dec_state_.Init(pool_);
    return true;
  }
  Status DecodeModularGlobalInfo(BitReader* reader) {
    // currently just initializes the dec_state
    // TODO: add patches and maybe other features
    dec_state_.Init(pool_);
    return true;
  }

  Status DecodeDCGroup(size_t group_index, size_t thread, const Rect& rect,
                       BitReader* reader, const LoopFilter& loop_filter,
                       AuxOut* local_aux_out) {
    return jxl::DecodeDCGroup(reader, group_index, &dec_state_, local_aux_out);
  }

  Status DecodeGlobalACInfo(BitReader* reader) {
    uint64_t flags = dec_state_.shared_storage.frame_header.flags;
    if (!(flags & FrameHeader::kSkipAdaptiveDCSmoothing) &&
        !(flags & FrameHeader::kUseDcFrame)) {
      AdaptiveDCSmoothing(dec_state_.shared_storage.dc_quant_field,
                          &dec_state_.shared_storage.dc_storage, pool_);
    }

    if (aux_out_ && aux_out_->testing_aux.dc) {
      *aux_out_->testing_aux.dc = CopyImage(*dec_state_.shared_storage.dc);
    }

    JXL_RETURN_IF_ERROR(dec_state_.shared_storage.matrices.Decode(reader));

    size_t num_histo_bits =
        dec_state_.shared->frame_dim.num_groups == 1
            ? 0
            : CeilLog2Nonzero(dec_state_.shared->frame_dim.num_groups - 1);
    dec_state_.shared_storage.num_histograms =
        1 + reader->ReadBits(num_histo_bits);
    dec_state_.code.resize(kMaxNumPasses * dec_state_.shared->num_histograms);
    dec_state_.context_map.resize(kMaxNumPasses *
                                  dec_state_.shared->num_histograms);
    // Read coefficient orders and histograms.
    for (size_t i = 0;
         i < dec_state_.shared_storage.frame_header.passes.num_passes; i++) {
      for (size_t histo = 0; histo < dec_state_.shared->num_histograms;
           histo++) {
        uint16_t used_orders = U32Coder::Read(kOrderEnc, reader);
        size_t idx =
            histo * dec_state_.shared_storage.frame_header.passes.num_passes +
            i;
        JXL_RETURN_IF_ERROR(DecodeCoeffOrders(
            used_orders,
            &dec_state_.shared_storage.coeff_orders[idx * kCoeffOrderSize],
            reader));
      }
      for (size_t histo = 0; histo < dec_state_.shared->num_histograms;
           histo++) {
        size_t idx =
            histo * dec_state_.shared_storage.frame_header.passes.num_passes +
            i;
        JXL_RETURN_IF_ERROR(DecodeHistograms(
            reader, kNumContexts, ANS_MAX_ALPHA_SIZE, &dec_state_.code[idx],
            &dec_state_.context_map[idx]));
      }
    }
    return true;
  }

  Status DecodeACGroup(size_t group_index, size_t thread,
                       BitReader* JXL_RESTRICT* JXL_RESTRICT readers,
                       size_t num_passes, Image3F* JXL_RESTRICT opsin,
                       ImageBundle* JXL_RESTRICT decoded,
                       AuxOut* local_aux_out) {
    return DecodeGroup(readers, num_passes, group_index, &dec_state_,
                       &group_dec_caches_[thread], thread, opsin, decoded,
                       local_aux_out);
  }

  Status FinalizeJPEG(Image3F* JXL_RESTRICT opsin, ImageBundle* decoded) {
    decoded->SetFromImage(std::move(*opsin),
                          decoded->metadata()->color_encoding);
    decoded->color_transform =
        dec_state_.shared_storage.frame_header.color_transform;
    decoded->chroma_subsampling =
        dec_state_.shared_storage.frame_header.chroma_subsampling;
    const std::vector<QuantEncoding>& qe =
        dec_state_.shared_storage.matrices.encodings();
    if (qe.empty() || qe[0].mode != QuantEncoding::Mode::kQuantModeRAW ||
        qe[0].qraw.qtable_den_shift != 0) {
      return JXL_FAILURE(
          "Quantization table is not a JPEG quantization table.");
    }
    decoded->is_jpeg = true;
    decoded->jpeg_quant_table.resize(192);
    JXL_DASSERT(qe[0].qraw.qtable);
    JXL_DASSERT(qe[0].qraw.qtable->size() >= 3 * 8 * 8);
    for (size_t c = 0; c < 3; c++) {
      for (size_t y = 0; y < 8; y++) {
        for (size_t x = 0; x < 8; x++) {
          // JPEG XL transposes the DCT, JPEG doesn't.
          decoded->jpeg_quant_table[c * 64 + 8 * x + y] =
              (*qe[0].qraw.qtable)[c * 64 + 8 * y + x];
        }
      }
    }
    return true;
  }

  Status FinalizeDecoding(Image3F* JXL_RESTRICT opsin,
                          const LoopFilter& loop_filter, ImageBundle* decoded) {
    if (decoded->IsJPEG()) {
      JXL_RETURN_IF_ERROR(FinalizeJPEG(opsin, decoded));
      return true;
    }
    JXL_RETURN_IF_ERROR(
        FinalizeFrameDecoding(opsin, &dec_state_, pool_, aux_out_,
                              /*save_decompressed=*/true,
                              /*apply_color_transform=*/
                              dec_state_.shared->multiframe->IsDisplayed()));

    if (dec_state_.shared_storage.frame_header.color_transform ==
        ColorTransform::kXYB) {
      // Do not use decoded->IsGray() - c_current is not yet valid.
      const bool is_gray = decoded->metadata()->color_encoding.IsGray();
      decoded->SetFromImage(std::move(*opsin),
                            ColorEncoding::LinearSRGB(is_gray));
    } else {
      decoded->SetFromImage(std::move(*opsin),
                            decoded->metadata()->color_encoding);
      if (dec_state_.shared_storage.frame_header.color_transform ==
          ColorTransform::kYCbCr) {
        JXL_RETURN_IF_ERROR(Map255ToTargetNits(decoded, pool_));
      }
    }
    return true;
  }

  PassesDecoderState* State() { return &dec_state_; }

 private:
  PassesDecoderState dec_state_;
  size_t downsampling_;

  ThreadPool* pool_;
  AuxOut* aux_out_;

  std::vector<GroupDecCache> group_dec_caches_;
};

namespace {

Status DecodeFrameHeader(const AnimationHeader* animation_or_null,
                         BitReader* JXL_RESTRICT reader,
                         FrameHeader* JXL_RESTRICT frame_header,
                         FrameDimensions* frame_dim,
                         LoopFilter* JXL_RESTRICT loop_filter) {
  frame_header->animation_frame.nonserialized_have_timecode =
      animation_or_null ? animation_or_null->have_timecodes : false;
  frame_header->animation_frame.nonserialized_composite_still =
      animation_or_null ? animation_or_null->composite_still : false;
  JXL_RETURN_IF_ERROR(ReadFrameHeader(reader, frame_header));

  if (frame_header->animation_frame.have_crop) {
    frame_dim->Set(frame_header->animation_frame.xsize,
                   frame_header->animation_frame.ysize);
  }

  if (frame_header->dc_level != 0) {
    frame_dim->Set(
        DivCeil(frame_dim->xsize, 1 << (3 * frame_header->dc_level)),
        DivCeil(frame_dim->ysize, 1 << (3 * frame_header->dc_level)));
  }

  if (frame_header->IsLossy()) {
    JXL_RETURN_IF_ERROR(ReadLoopFilter(reader, loop_filter));
  }

  return true;
}

}  // namespace

Status SkipFrame(const Span<const uint8_t> file,
                 const AnimationHeader* animation_or_null,
                 FrameDimensions* JXL_RESTRICT frame_dim,
                 BitReader* JXL_RESTRICT reader) {
  FrameHeader frame_header;
  LoopFilter loop_filter;
  JXL_RETURN_IF_ERROR(DecodeFrameHeader(
      animation_or_null, reader, &frame_header, frame_dim, &loop_filter));

  // Read TOC.
  std::vector<uint64_t> group_offsets;
  std::vector<uint32_t> group_sizes;
  uint64_t groups_total_size;
  const bool has_ac_global = !frame_header.IsJpeg();
  const size_t toc_entries =
      NumTocEntries(frame_dim->num_groups, frame_dim->num_dc_groups,
                    frame_header.passes.num_passes, has_ac_global);
  JXL_RETURN_IF_ERROR(ReadGroupOffsets(toc_entries, reader, &group_offsets,
                                       &group_sizes, &groups_total_size));

  // Pretend all groups are read.
  reader->SkipBits(groups_total_size * kBitsPerByte);
  if (reader->TotalBitsConsumed() > file.size() * kBitsPerByte) {
    return JXL_FAILURE("Group code extends after stream end");
  }

  return true;
}

Status DecodeFrame(const DecompressParams& dparams,
                   const Span<const uint8_t> file,
                   const AnimationHeader* animation_or_null,
                   FrameDimensions* frame_dim,
                   Multiframe* JXL_RESTRICT multiframe,
                   ThreadPool* JXL_RESTRICT pool,
                   BitReader* JXL_RESTRICT reader, AuxOut* JXL_RESTRICT aux_out,
                   ImageBundle* decoded, AnimationFrame* animation) {
  PROFILER_ZONE("DecodeFrame uninstrumented");

  FrameHeader frame_header;
  LoopFilter loop_filter;
  JXL_RETURN_IF_ERROR(DecodeFrameHeader(
      animation_or_null, reader, &frame_header, frame_dim, &loop_filter));
  if (frame_dim->xsize == 0 || frame_dim->ysize == 0) {
    return JXL_FAILURE("Empty frame");
  }
  const size_t num_passes = frame_header.passes.num_passes;
  const size_t xsize = frame_dim->xsize;
  const size_t ysize = frame_dim->ysize;
  const size_t num_groups = frame_dim->num_groups;

  if (animation != nullptr) {
    *animation = frame_header.animation_frame;
  }

  // If the previous frame was not displayed, `decoded` may have different
  // dimensions; must reset to avoid error when setting alpha.
  decoded->RemoveColor();
  if (frame_header.HasAlpha()) {
    if (!decoded->metadata()->HasAlpha()) {
      return JXL_FAILURE("Frame has alpha but decoded metadata doesn't");
    }
    decoded->SetAlpha(
        ImageU(xsize, ysize),
        /*alpha_is_premultiplied=*/frame_header.AlphaIsPremultiplied());
  }

  if (decoded->metadata()->m2.HasDepth())
    decoded->SetDepth(
        ImageU(decoded->DepthSize(xsize), decoded->DepthSize(ysize)));
  if (decoded->metadata()->m2.num_extra_channels > 0 &&
      frame_header.IsDisplayed()) {
    std::vector<ImageU> ecv;
    for (size_t i = 0; i < decoded->metadata()->m2.num_extra_channels; i++)
      ecv.push_back(ImageU(xsize, ysize));
    decoded->SetExtraChannels(std::move(ecv));
  }
  if (frame_header.encoding == FrameEncoding::kModularGroup) {
    decoded->SetFromImage(Image3F(xsize, ysize),
                          decoded->metadata()->color_encoding);
    loop_filter.gab = false;
    loop_filter.epf = false;
  }

  // Handling of progressive decoding for kVarDCT mode.
  size_t downsampling;
  size_t max_passes = dparams.max_passes;
  // TODO(veluca): deal with downsamplings >= 8.
  if (dparams.max_downsampling >= 8) {
    downsampling = 8;
    max_passes = 0;
  } else {
    downsampling = 1;
    for (uint32_t i = 0; i < frame_header.passes.num_downsample; ++i) {
      if (dparams.max_downsampling >= frame_header.passes.downsample[i] &&
          max_passes > frame_header.passes.last_pass[i]) {
        downsampling = frame_header.passes.downsample[i];
        max_passes = frame_header.passes.last_pass[i] + 1;
      }
    }
  }
  // Do not use downsampling for animation frames or non-displayed frames.
  if (!multiframe->IsDisplayed() || !frame_header.animation_frame.is_last) {
    downsampling = 1;
    max_passes = frame_header.passes.num_passes;
  }
  if (aux_out != nullptr) {
    aux_out->downsampling = downsampling;
  }

  multiframe->StartFrame(frame_header);

  LossyFrameDecoder lossy_frame_decoder;
  JXL_RETURN_IF_ERROR(lossy_frame_decoder.Init(
      frame_header, loop_filter, *decoded->metadata(), *frame_dim, multiframe,
      downsampling, pool, aux_out));
  BrunsliFrameDecoder jpeg_frame_decoder(pool);
  ModularFrameDecoder modular_frame_decoder;

  if (dparams.keep_dct) {
    if (frame_header.encoding == FrameEncoding::kModularGroup) {
      return JXL_FAILURE("Cannot output JPEG from ModularGroup");
    }
    decoded->is_jpeg = true;
  }

  // Read TOC.
  std::vector<uint64_t> group_offsets;
  std::vector<uint32_t> group_sizes;
  uint64_t groups_total_size;
  const bool has_ac_global = !frame_header.IsJpeg();
  const size_t toc_entries = NumTocEntries(num_groups, frame_dim->num_dc_groups,
                                           num_passes, has_ac_global);
  JXL_RETURN_IF_ERROR(ReadGroupOffsets(toc_entries, reader, &group_offsets,
                                       &group_sizes, &groups_total_size));

  const size_t global_ac_index = frame_dim->num_dc_groups + 1;

  JXL_DASSERT((reader->TotalBitsConsumed() % kBitsPerByte) == 0);
  const size_t group_codes_begin = reader->TotalBitsConsumed() / kBitsPerByte;
  // group_offsets can be permuted, so we need to check the groups_total_size.
  JXL_DASSERT(!group_offsets.empty());
  if (group_codes_begin + groups_total_size > file.size() ||
      group_codes_begin + groups_total_size < group_codes_begin) {
    // The second check is for overflow on the
    // "group_codes_begin + group_offsets.back()" calculation.
    return JXL_FAILURE("group offset is out of bounds");
  }
  auto get_reader = [num_groups, num_passes, group_codes_begin, &group_offsets,
                     &group_sizes, &file,
                     &reader](BitReader* JXL_RESTRICT store, size_t index) {
    if (num_groups == 1 && num_passes == 1) return reader;
    const size_t group_offset = group_codes_begin + group_offsets[index];
    const size_t next_group_offset =
        group_codes_begin + group_offsets[index] + group_sizes[index];
    // The order of these variables must be:
    // group_codes_begin <= group_offset <= next_group_offset <= file.size()
    JXL_DASSERT(group_codes_begin <= group_offset);
    JXL_DASSERT(group_offset <= next_group_offset);
    JXL_DASSERT(next_group_offset <= file.size());
    const size_t group_size = next_group_offset - group_offset;
    const size_t remaining_size = file.size() - group_offset;
    const size_t size = std::min(group_size + 8, remaining_size);
    *store = BitReader(Span<const uint8_t>(file.data() + group_offset, size));
    return store;
  };

  Status res = true;
  {
    BitReader global_store;
    BitReaderScopedCloser global_store_closer(&global_store, &res);
    BitReader* global_reader = get_reader(&global_store, 0);

    {
      PassesSharedState& shared = lossy_frame_decoder.State()->shared_storage;

      if (shared.frame_header.flags & FrameHeader::kPatches) {
        JXL_RETURN_IF_ERROR(shared.image_features.patches.Decode(
            global_reader, shared.frame_dim.xsize_padded,
            shared.frame_dim.ysize_padded,
            shared.frame_header.save_as_reference));
      }
      if (shared.frame_header.flags & FrameHeader::kSplines) {
        JXL_RETURN_IF_ERROR(
            shared.image_features.splines.Decode(global_reader));
      }
      if (shared.frame_header.flags & FrameHeader::kNoise) {
        JXL_RETURN_IF_ERROR(
            DecodeNoise(global_reader, &shared.image_features.noise_params));
      }
    }

    if (frame_header.IsJpeg()) {
      JXL_RETURN_IF_ERROR(jpeg_frame_decoder.ReadHeader(
          frame_dim, global_reader, frame_header.chroma_subsampling));
    } else if (frame_header.IsLossy()) {
      JXL_RETURN_IF_ERROR(
          lossy_frame_decoder.DecodeGlobalDCInfo(global_reader, decoded));
    } else if (frame_header.encoding == FrameEncoding::kModularGroup) {
      JXL_RETURN_IF_ERROR(
          lossy_frame_decoder.DecodeModularGlobalInfo(global_reader));
    }
    JXL_RETURN_IF_ERROR(modular_frame_decoder.DecodeGlobalInfo(
        global_reader, frame_header, decoded,
        /*decode_color = */
        (frame_header.encoding == FrameEncoding::kModularGroup), xsize, ysize,
        /*group_id=*/0));
  }

  // global_store is either never used (in which case Close() is optional
  // and a no-op) or the real deal and we must call Close() and check the error
  // code here.
  JXL_RETURN_IF_ERROR(res);

  // Decode DC groups.
  std::atomic<int> num_errors{0};
  std::vector<AuxOut> aux_outs;
  const auto resize_aux_outs = [&](size_t num_threads) {
    // Updates aux_outs size Assimilating its elements if the size decreases.
    if (aux_out != nullptr) {
      size_t old_size = aux_outs.size();
      for (size_t i = num_threads; i < old_size; i++) {
        aux_out->Assimilate(aux_outs[i]);
      }
      aux_outs.resize(num_threads);
      // Each thread needs these INPUTS. Don't copy the entire AuxOut
      // because it may contain stats which would be Assimilated multiple
      // times below.
      for (size_t i = old_size; i < aux_outs.size(); i++) {
        aux_outs[i].testing_aux = aux_out->testing_aux;
        aux_outs[i].dump_image = aux_out->dump_image;
        aux_outs[i].debug_prefix = aux_out->debug_prefix;
      }
    }
  };

  JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);
  decoded->SetDecodedBytes(group_offsets[0] + group_sizes[0] +
                           group_codes_begin);

  const auto process_dc_group_init = [&](size_t num_threads) {
    lossy_frame_decoder.SetNumThreads(num_threads);
    resize_aux_outs(num_threads);
    return true;
  };
  const auto process_dc_group = [&](const int group_index, const int thread) {
    PROFILER_ZONE("DC group");
    BitReader group_store;
    BitReader* group_reader = get_reader(&group_store, group_index + 1);
    const size_t gx = group_index % frame_dim->xsize_dc_groups;
    const size_t gy = group_index / frame_dim->xsize_dc_groups;
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;
    bool ok = true;
    const Rect mrect(gx * kDcGroupDim, gy * kDcGroupDim, kDcGroupDim,
                     kDcGroupDim);
    ok = modular_frame_decoder.DecodeGroup(
        dparams, mrect, group_reader, my_aux_out, 3, 1000, 1 + group_index);
    if (frame_header.IsJpeg()) {
      ok &= jpeg_frame_decoder.DecodeDcGroup(group_index, group_reader);
    } else if (frame_header.IsLossy()) {
      const Rect rect(gx * kDcGroupDimInBlocks, gy * kDcGroupDimInBlocks,
                      kDcGroupDimInBlocks, kDcGroupDimInBlocks,
                      frame_dim->xsize_blocks, frame_dim->ysize_blocks);
      ok &= lossy_frame_decoder.DecodeDCGroup(
          group_index, thread, rect, group_reader, loop_filter, my_aux_out);
    }
    if (!group_store.Close() || !ok) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
    }
  };
  RunOnPool(pool, 0, frame_dim->num_dc_groups, process_dc_group_init,
            process_dc_group, "DecodeDCGroup");
  JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);
  if (downsampling < 16 && frame_dim->num_groups > 1) {
    decoded->SetDecodedBytes(group_offsets[frame_dim->num_dc_groups] +
                             group_sizes[frame_dim->num_dc_groups] +
                             group_codes_begin);
  }

  Image3F opsin(frame_dim->xsize_blocks * kBlockDim,
                frame_dim->ysize_blocks * kBlockDim);

  // Read global AC info.
  {
    PROFILER_ZONE("Global AC");
    res = true;
    {
      BitReader ac_info_store;
      BitReaderScopedCloser ac_info_store_closer(&ac_info_store, &res);
      BitReader* ac_info_reader = get_reader(&ac_info_store, global_ac_index);

      if (frame_header.IsLossy()) {
        JXL_RETURN_IF_ERROR(
            lossy_frame_decoder.DecodeGlobalACInfo(ac_info_reader));
      }
    }
    // ac_info_store is either never used (in which case Close() is optional
    // and a no-op) or the real deal and we must call Close() and check the
    // return error.
    JXL_RETURN_IF_ERROR(res);
  }

  max_passes = std::min<size_t>(max_passes, frame_header.passes.num_passes);

  // Decode groups.
  const auto process_group_init = [&](size_t num_threads) {
    // The number of threads here might be different from the previous run, so
    // we need to re-update them.
    lossy_frame_decoder.SetNumThreads(num_threads);
    resize_aux_outs(num_threads);
    return true;
  };

  const auto process_group = [&](const int task, const int thread) {
    PROFILER_ZONE("process_group");
    const size_t group_index = static_cast<size_t>(task);
    const size_t gx = group_index % frame_dim->xsize_groups;
    const size_t gy = group_index / frame_dim->xsize_groups;
    const size_t x = gx * kGroupDim;
    const size_t y = gy * kGroupDim;
    Rect rect(x, y, kGroupDim, kGroupDim, xsize, ysize);

    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;
    BitReader pass0_reader_store;
    BitReader* pass0_group_reader =
        get_reader(&pass0_reader_store,
                   AcGroupIndex(0, group_index, num_groups,
                                frame_dim->num_dc_groups, has_ac_global));

    // don't limit to image dimensions here (is done in DecodeGroup)
    const Rect mrect(x, y, kGroupDim, kGroupDim);

    // Read passes.
    BitReader storage[kMaxNumPasses];
    BitReader* JXL_RESTRICT readers[kMaxNumPasses];
    readers[0] = pass0_group_reader;
    for (size_t i = 1; i < max_passes; i++) {
      readers[i] =
          get_reader(&storage[i - 1],
                     AcGroupIndex(i, group_index, num_groups,
                                  frame_dim->num_dc_groups, has_ac_global));
    }
    int minShift = 0;
    int maxShift = 2;
    for (size_t i = 0; i < max_passes; i++) {
      for (uint32_t j = 0; j < frame_header.passes.num_downsample; ++j) {
        if (i <= frame_header.passes.last_pass[j]) {
          if (frame_header.passes.downsample[j] == 8) minShift = 3;
          if (frame_header.passes.downsample[j] == 4) minShift = 2;
          if (frame_header.passes.downsample[j] == 2) minShift = 1;
          if (frame_header.passes.downsample[j] == 1) minShift = 0;
        }
      }
      if (!modular_frame_decoder.DecodeGroup(
              dparams, mrect, readers[i], my_aux_out, minShift, maxShift,
              AcGroupIndex(i, group_index, num_groups, frame_dim->num_dc_groups,
                           has_ac_global))) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
      }
      maxShift = minShift - 1;
      minShift = 0;
    }
    if (frame_header.IsLossy()) {
      if (!lossy_frame_decoder.DecodeACGroup(group_index, thread, readers,
                                             max_passes, &opsin, decoded,
                                             my_aux_out)) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
      }
    }
    for (size_t i = 1; i < max_passes; i++) {
      if (!storage[i - 1].Close()) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
      }
    }

    bool ok = true;
    if (frame_header.IsJpeg()) {
      ok = jpeg_frame_decoder.DecodeAcGroup(group_index, pass0_group_reader,
                                            &opsin, rect);
    }

    if (!ok) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    if (!pass0_reader_store.Close()) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
    }
  };

  {
    PROFILER_ZONE("DecodeFrame pool");
    RunOnPool(pool, 0, num_groups, process_group_init, process_group,
              "DecodeFrame");
    JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);
  }

  if (downsampling < 8 && frame_dim->num_groups > 1) {
    uint64_t last_end_offset = 0;
    size_t end_index = AcGroupIndex(max_passes, 0, num_groups,
                                    frame_dim->num_dc_groups, has_ac_global);
    for (size_t i = 0; i < end_index; i++) {
      uint64_t end_offset = group_offsets[i] + group_sizes[i];
      last_end_offset = std::max(last_end_offset, end_offset);
    }
    decoded->SetDecodedBytes(group_codes_begin + last_end_offset);
  }

  // Resizing to 0 assimilates all the results when needed.
  resize_aux_outs(0);
  // undo global modular transforms and copy int pixel buffers to float ones
  JXL_RETURN_IF_ERROR(modular_frame_decoder.FinalizeDecoding(
      &opsin, decoded, pool, frame_header));

  if (frame_header.IsJpeg()) {
    jpeg_frame_decoder.FinalizeDecoding(frame_header, std::move(opsin),
                                        decoded);
  } else {
    JXL_RETURN_IF_ERROR(
        lossy_frame_decoder.FinalizeDecoding(&opsin, loop_filter, decoded));
  }

  if (num_groups == 1 && num_passes == 1) {
    // get_reader used reader, so AC groups have already been consumed by
    // reader - unless we did not decode all passes.
    if (max_passes < num_passes) {
      const size_t frame_end =
          group_codes_begin + groups_total_size * kBitsPerByte;
      reader->SkipBits(frame_end - reader->TotalBitsConsumed());
    }
  } else {
    // Used per-group readers, need to skip groups in main reader.
    reader->SkipBits(groups_total_size * kBitsPerByte);  // aligned
  }

  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  if ((reader->TotalBitsConsumed()) > file.size() * kBitsPerByte) {
    return JXL_FAILURE("Read past stream end");
  }

  if (decoded->IsJPEG()) {
    decoded->jpeg_xsize = xsize;
    decoded->jpeg_ysize = ysize;
  } else {
    decoded->ShrinkTo(xsize, ysize);
  }
  return true;
}

}  // namespace jxl
