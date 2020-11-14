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

#include "lib/jxl/dec_frame.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <hwy/aligned_allocator.h>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/blending.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_group.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/dec_params.h"
#include "lib/jxl/dec_reconstruct.h"
#include "lib/jxl/dec_upsample.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/dot_dictionary.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/filters.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/luminance.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/patch_dictionary.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/splines.h"
#include "lib/jxl/toc.h"

namespace jxl {

Status DecodeGlobalDCInfo(size_t downsampling, BitReader* reader,
                          ImageBundle* decoded, PassesDecoderState* state,
                          ThreadPool* pool) {
  PROFILER_FUNC;
  JXL_RETURN_IF_ERROR(state->shared_storage.quantizer.Decode(reader));

  JXL_RETURN_IF_ERROR(
      DecodeBlockCtxMap(reader, &state->shared_storage.block_ctx_map));

  JXL_RETURN_IF_ERROR(state->shared_storage.cmap.DecodeDC(reader));

  // Pre-compute info for decoding a group.
  if (decoded->IsJPEG()) {
    state->shared_storage.quantizer.ClearDCMul();  // Don't dequant DC
  }

  state->shared_storage.ac_strategy.FillInvalid();
  state->Init(pool);
  return true;
}

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class LossyFrameDecoder {
 public:
  Status Init(const FrameHeader& frame_header, PassesDecoderState* dec_state,
              size_t downsampling, ThreadPool* pool, AuxOut* aux_out) {
    downsampling_ = downsampling;
    pool_ = pool;
    aux_out_ = aux_out;
    dec_state_ = dec_state;
    const LoopFilter& lf = frame_header.loop_filter;
    if (!frame_header.chroma_subsampling.Is444() &&
        (lf.gab || lf.epf_iters > 0) &&
        frame_header.encoding == FrameEncoding::kVarDCT) {
      // TODO(veluca): actually implement this.
      return JXL_FAILURE(
          "Non-444 chroma subsampling is not supported when loop filters are "
          "enabled");
    }
    if (!frame_header.chroma_subsampling.Is444() &&
        !(frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing) &&
        frame_header.encoding == FrameEncoding::kVarDCT) {
      // TODO(veluca): actually implement this.
      return JXL_FAILURE(
          "Non-444 chroma subsampling is not supported when adaptive DC "
          "smoothing is enabled");
    }
    return InitializePassesSharedState(frame_header,
                                       &dec_state_->shared_storage);
  }

  // Sets the number of threads that will be used. The value of the "thread"
  // parameter passed to DecodeDCGroup and DecodeACGroup must be smaller than
  // the "num_threads" passed here.
  void SetNumThreads(size_t num_threads) {
    if (num_threads > group_dec_caches_size_) {
      group_dec_caches_size_ = num_threads;
      group_dec_caches_ =
          hwy::MakeUniqueAlignedArray<GroupDecCache>(num_threads);
    }
    dec_state_->EnsureStorage(num_threads);
  }

  Status DecodeGlobalDCInfo(BitReader* reader, ImageBundle* decoded) {
    return jxl::DecodeGlobalDCInfo(downsampling_, reader, decoded, dec_state_,
                                   pool_);
  }
  Status DecodeModularGlobalInfo(BitReader* reader) {
    // currently just initializes the dec_state
    // TODO: add patches and maybe other features
    dec_state_->Init(pool_);
    return true;
  }

  Status DecodeGlobalACInfo(BitReader* reader,
                            ModularFrameDecoder* modular_frame_decoder) {
    if (aux_out_ && aux_out_->testing_aux.dc) {
      *aux_out_->testing_aux.dc = CopyImage(*dec_state_->shared_storage.dc);
    }

    JXL_RETURN_IF_ERROR(dec_state_->shared_storage.matrices.Decode(
        reader, modular_frame_decoder));

    size_t num_histo_bits =
        CeilLog2Nonzero(dec_state_->shared->frame_dim.num_groups);
    dec_state_->shared_storage.num_histograms =
        1 + reader->ReadBits(num_histo_bits);

    dec_state_->code.resize(kMaxNumPasses);
    dec_state_->context_map.resize(kMaxNumPasses);
    // Read coefficient orders and histograms.
    for (size_t i = 0;
         i < dec_state_->shared_storage.frame_header.passes.num_passes; i++) {
      uint16_t used_orders = U32Coder::Read(kOrderEnc, reader);
      JXL_RETURN_IF_ERROR(DecodeCoeffOrders(
          used_orders,
          &dec_state_->shared_storage.coeff_orders[i * kCoeffOrderSize],
          reader));
      JXL_RETURN_IF_ERROR(DecodeHistograms(
          reader,
          dec_state_->shared->num_histograms *
              dec_state_->shared_storage.block_ctx_map.NumACContexts(),
          &dec_state_->code[i], &dec_state_->context_map[i]));
    }
    return true;
  }

  Status DecodeACGroup(size_t group_index, size_t thread,
                       BitReader* JXL_RESTRICT* JXL_RESTRICT readers,
                       size_t num_passes, Image3F* JXL_RESTRICT opsin,
                       ImageBundle* JXL_RESTRICT decoded,
                       AuxOut* local_aux_out) {
    return DecodeGroup(readers, num_passes, group_index, dec_state_,
                       &group_dec_caches_[thread], thread, opsin, decoded,
                       local_aux_out);
  }

  Status FinalizeJPEG(Image3F* JXL_RESTRICT opsin, ImageBundle* decoded) {
    decoded->SetFromImage(std::move(*opsin),
                          decoded->metadata()->color_encoding);
    decoded->color_transform =
        dec_state_->shared_storage.frame_header.color_transform;
    decoded->chroma_subsampling =
        dec_state_->shared_storage.frame_header.chroma_subsampling;
    const std::vector<QuantEncoding>& qe =
        dec_state_->shared_storage.matrices.encodings();
    if (qe.empty() || qe[0].mode != QuantEncoding::Mode::kQuantModeRAW ||
        qe[0].qraw.qtable_den_shift != 0) {
      return JXL_FAILURE(
          "Quantization table is not a JPEG quantization table.");
    }
    // TODO(veluca): figure out how to put the JPEG quantization table in
    // JPEGData.
    return true;
  }

  Status FinalizeDecoding(Image3F* JXL_RESTRICT opsin, ImageBundle* decoded) {
    if (decoded->IsJPEG()) {
      JXL_RETURN_IF_ERROR(FinalizeJPEG(opsin, decoded));
      return true;
    }
    JXL_RETURN_IF_ERROR(
        FinalizeFrameDecoding(opsin, dec_state_, pool_, aux_out_));

    if (dec_state_->shared_storage.frame_header.color_transform ==
        ColorTransform::kXYB) {
      // Do not use decoded->IsGray() - c_current is not yet valid.
      const bool is_gray = decoded->metadata()->color_encoding.IsGray();
      decoded->SetFromImage(std::move(*opsin),
                            ColorEncoding::LinearSRGB(is_gray));
    } else {
      decoded->SetFromImage(std::move(*opsin),
                            decoded->metadata()->color_encoding);
    }
    decoded->origin = dec_state_->shared->frame_header.frame_origin;
    // TODO(veluca): should be in dec_reconstruct.
    JXL_RETURN_IF_ERROR(DoBlending(*dec_state_->shared, decoded));
    if (dec_state_->shared->frame_header.CanBeReferenced()) {
      size_t id = dec_state_->shared->frame_header.save_as_reference;
      dec_state_->shared_storage.reference_frames[id].storage = decoded->Copy();
      dec_state_->shared_storage.reference_frames[id].frame =
          &dec_state_->shared_storage.reference_frames[id].storage;
      dec_state_->shared_storage.reference_frames[id].ib_is_in_xyb =
          dec_state_->shared->frame_header.save_before_color_transform;
    }
    if (dec_state_->shared->frame_header.dc_level != 0) {
      dec_state_->shared_storage
          .dc_frames[dec_state_->shared->frame_header.dc_level - 1] =
          std::move(*decoded->color());
      decoded->RemoveColor();
    }
    return true;
  }

  PassesDecoderState* State() { return dec_state_; }

 private:
  PassesDecoderState* dec_state_;
  size_t downsampling_;

  ThreadPool* pool_;
  AuxOut* aux_out_;

  // Number of allocated GroupDecCache entries in the group_dec_caches_ smart
  // pointer. This is only needed to tell whether we need to reallocate the
  // cache.
  size_t group_dec_caches_size_ = 0;
  hwy::AlignedUniquePtr<GroupDecCache[]> group_dec_caches_{
      nullptr, hwy::AlignedDeleter(nullptr)};
};

Status DecodeFrameHeader(BitReader* JXL_RESTRICT reader,
                         FrameHeader* JXL_RESTRICT frame_header) {
  JXL_ASSERT(frame_header->nonserialized_metadata != nullptr);
  JXL_RETURN_IF_ERROR(ReadFrameHeader(reader, frame_header));

  return true;
}

Status SkipFrame(const CodecMetadata& metadata, BitReader* JXL_RESTRICT reader,
                 bool is_preview) {
  FrameHeader header(&metadata);
  header.nonserialized_is_preview = is_preview;
  JXL_RETURN_IF_ERROR(DecodeFrameHeader(reader, &header));

  // Read TOC.
  std::vector<uint64_t> group_offsets;
  std::vector<uint32_t> group_sizes;
  uint64_t groups_total_size;
  const bool has_ac_global = true;
  const FrameDimensions frame_dim = header.ToFrameDimensions();
  const size_t toc_entries =
      NumTocEntries(frame_dim.num_groups, frame_dim.num_dc_groups,
                    header.passes.num_passes, has_ac_global);
  JXL_RETURN_IF_ERROR(ReadGroupOffsets(toc_entries, reader, &group_offsets,
                                       &group_sizes, &groups_total_size));

  // Pretend all groups are read.
  reader->SkipBits(groups_total_size * kBitsPerByte);
  if (reader->TotalBitsConsumed() > reader->TotalBytes() * kBitsPerByte) {
    return JXL_FAILURE("Group code extends after stream end");
  }

  return true;
}

static BitReader* GetReaderForSection(
    size_t num_groups, size_t num_passes, size_t group_codes_begin,
    const std::vector<uint64_t>& group_offsets,
    const std::vector<uint32_t>& group_sizes, BitReader* JXL_RESTRICT reader,
    BitReader* JXL_RESTRICT store, size_t index) {
  if (num_groups == 1 && num_passes == 1) return reader;
  const size_t group_offset = group_codes_begin + group_offsets[index];
  const size_t next_group_offset =
      group_codes_begin + group_offsets[index] + group_sizes[index];
  // The order of these variables must be:
  // group_codes_begin <= group_offset <= next_group_offset <= file.size()
  JXL_DASSERT(group_codes_begin <= group_offset);
  JXL_DASSERT(group_offset <= next_group_offset);
  JXL_DASSERT(next_group_offset <= reader->TotalBytes());
  const size_t group_size = next_group_offset - group_offset;
  const size_t remaining_size = reader->TotalBytes() - group_offset;
  const size_t size = std::min(group_size + 8, remaining_size);
  *store =
      BitReader(Span<const uint8_t>(reader->FirstByte() + group_offset, size));
  return store;
};

static void ResizeAuxOuts(std::vector<AuxOut>& aux_outs, size_t num_threads,
                          AuxOut* JXL_RESTRICT aux_out) {
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

Status DecodeDC(const FrameHeader& frame_header, PassesDecoderState* dec_state,
                ModularFrameDecoder& modular_frame_decoder,
                size_t group_codes_begin,
                const std::vector<uint64_t>& group_offsets,
                const std::vector<uint32_t>& group_sizes,
                ThreadPool* JXL_RESTRICT pool, BitReader* JXL_RESTRICT reader,
                std::vector<AuxOut>* aux_outs, AuxOut* JXL_RESTRICT aux_out) {
  std::atomic<int> num_errors{0};
  JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);
  FrameDimensions frame_dim = frame_header.ToFrameDimensions();
  size_t num_groups = frame_dim.num_groups;
  size_t num_passes = frame_header.passes.num_passes;

  auto get_reader = [num_groups, num_passes, group_codes_begin, &group_offsets,
                     &group_sizes,
                     &reader](BitReader* JXL_RESTRICT store, size_t index) {
    return GetReaderForSection(num_groups, num_passes, group_codes_begin,
                               group_offsets, group_sizes, reader, store,
                               index);
  };
  const auto process_dc_group_init = [&](size_t num_threads) {
    dec_state->EnsureStorage(num_threads);
    if (aux_out) ResizeAuxOuts(*aux_outs, num_threads, aux_out);
    return true;
  };
  const auto process_dc_group = [&](const int group_index, const int thread) {
    PROFILER_ZONE("DC group");
    BitReader group_store;
    BitReader* group_reader = get_reader(&group_store, group_index + 1);
    const size_t gx = group_index % frame_dim.xsize_dc_groups;
    const size_t gy = group_index / frame_dim.xsize_dc_groups;
    AuxOut* my_aux_out = aux_out ? &(*aux_outs)[thread] : nullptr;
    bool ok = true;
    const Rect mrect(gx * kDcGroupDim, gy * kDcGroupDim, kDcGroupDim,
                     kDcGroupDim);
    if (frame_header.encoding == FrameEncoding::kVarDCT &&
        !(frame_header.flags & FrameHeader::kUseDcFrame)) {
      ok &= modular_frame_decoder.DecodeVarDCTDC(group_index, group_reader,
                                                 dec_state, my_aux_out);
    }
    ok &= modular_frame_decoder.DecodeGroup(
        mrect, group_reader, my_aux_out, 3, 1000,
        ModularStreamId::ModularDC(group_index));
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      ok &= modular_frame_decoder.DecodeAcMetadata(group_index, group_reader,
                                                   dec_state, my_aux_out);
    }
    if (!group_store.Close() || !ok) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
    }
  };
  RunOnPool(pool, 0, frame_dim.num_dc_groups, process_dc_group_init,
            process_dc_group, "DecodeDCGroup");

  // Do Adaptive DC smoothing if enabled.
  if (frame_header.encoding == FrameEncoding::kVarDCT &&
      !(frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing) &&
      !(frame_header.flags & FrameHeader::kUseDcFrame)) {
    AdaptiveDCSmoothing(dec_state->shared->quantizer.MulDC(),
                        &dec_state->shared_storage.dc_storage, pool);
  }

  JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);
  return true;
}

Status DecodeFrame(const DecompressParams& dparams,
                   PassesDecoderState* dec_state, ThreadPool* JXL_RESTRICT pool,
                   BitReader* JXL_RESTRICT reader, AuxOut* JXL_RESTRICT aux_out,
                   ImageBundle* decoded, const CodecMetadata& metadata,
                   const CodecInOut* io, bool is_preview) {
  PROFILER_ZONE("DecodeFrame uninstrumented");

  // Reset the dequantization matrices to their default values.
  dec_state->shared_storage.matrices = DequantMatrices();

  FrameHeader frame_header(&metadata);
  frame_header.nonserialized_is_preview = is_preview;
  JXL_RETURN_IF_ERROR(DecodeFrameHeader(reader, &frame_header));
  FrameDimensions frame_dim = frame_header.ToFrameDimensions();
  if (frame_dim.xsize == 0 || frame_dim.ysize == 0) {
    return JXL_FAILURE("Empty frame");
  }
  if (io) {
    JXL_RETURN_IF_ERROR(io->VerifyDimensions(frame_dim.xsize, frame_dim.ysize));
  }
  const size_t num_passes = frame_header.passes.num_passes;
  const size_t xsize = frame_dim.xsize;
  const size_t ysize = frame_dim.ysize;
  const size_t num_groups = frame_dim.num_groups;

  // If the previous frame was not a kRegularFrame, `decoded` may have different
  // dimensions; must reset to avoid error when setting alpha.
  decoded->RemoveColor();

  if (metadata.m.Find(ExtraChannel::kDepth)) {
    decoded->SetDepth(
        ImageU(decoded->DepthSize(xsize), decoded->DepthSize(ysize)));
  }
  if (metadata.m.num_extra_channels > 0) {
    std::vector<ImageU> ecv;
    for (size_t i = 0; i < metadata.m.num_extra_channels; i++) {
      const auto eci = metadata.m.extra_channel_info[i];
      ecv.push_back(ImageU(eci.Size(xsize), eci.Size(ysize)));
    }
    decoded->SetExtraChannels(std::move(ecv));
  }
  if (frame_header.encoding == FrameEncoding::kModular) {
    decoded->SetFromImage(Image3F(xsize, ysize), metadata.m.color_encoding);
  }

  // Handling of progressive decoding.
  size_t downsampling;
  size_t max_passes = dparams.max_passes;
  size_t max_downsampling = std::max(
      dparams.max_downsampling >> (frame_header.dc_level * 3), size_t(1));
  // TODO(veluca): deal with downsamplings >= 8.
  if (max_downsampling >= 8) {
    downsampling = 8;
    max_passes = 0;
  } else {
    downsampling = 1;
    for (uint32_t i = 0; i < frame_header.passes.num_downsample; ++i) {
      if (max_downsampling >= frame_header.passes.downsample[i] &&
          max_passes > frame_header.passes.last_pass[i]) {
        downsampling = frame_header.passes.downsample[i];
        max_passes = frame_header.passes.last_pass[i] + 1;
      }
    }
  }
  // Do not use downsampling for kReferenceOnly frames.
  if (frame_header.frame_type == FrameType::kReferenceOnly) {
    downsampling = 1;
    max_passes = frame_header.passes.num_passes;
  }
  if (aux_out != nullptr) {
    aux_out->downsampling = downsampling;
  }
  max_passes = std::min<size_t>(max_passes, frame_header.passes.num_passes);

  // Read TOC.
  std::vector<uint64_t> group_offsets;
  std::vector<uint32_t> group_sizes;
  uint64_t groups_total_size;
  const bool has_ac_global = true;
  const size_t toc_entries = NumTocEntries(num_groups, frame_dim.num_dc_groups,
                                           num_passes, has_ac_global);
  JXL_RETURN_IF_ERROR(ReadGroupOffsets(toc_entries, reader, &group_offsets,
                                       &group_sizes, &groups_total_size));

  const size_t global_ac_index = frame_dim.num_dc_groups + 1;

  JXL_DASSERT((reader->TotalBitsConsumed() % kBitsPerByte) == 0);
  const size_t group_codes_begin = reader->TotalBitsConsumed() / kBitsPerByte;
  // group_offsets can be permuted, so we need to check the groups_total_size.
  JXL_DASSERT(!group_offsets.empty());

  // Overflow check.
  if (group_codes_begin + groups_total_size < group_codes_begin) {
    return JXL_FAILURE("Invalid group codes");
  }

  bool ac_global_available = true;
  std::vector<size_t> max_passes_for_group(num_groups, max_passes);
  if (!dparams.allow_partial_files ||
      frame_header.frame_type != FrameType::kRegularFrame ||
      !frame_header.is_last) {
    if (group_codes_begin + groups_total_size > reader->TotalBytes()) {
      return JXL_FAILURE("group offset is out of bounds");
    }
  } else {
    auto has_section = [&](size_t i) {
      return group_codes_begin + group_offsets[i] + group_sizes[i] <=
             reader->TotalBytes();
    };
    // check that all of DC is available.
    for (size_t i = 0; i < global_ac_index; i++) {
      if (!has_section(i)) {
        return JXL_FAILURE("file truncated before all of DC was read");
      }
    }
    if (!has_section(global_ac_index)) {
      std::fill(max_passes_for_group.begin(), max_passes_for_group.end(), 0);
      ac_global_available = false;
    }
    for (size_t g = 0; g < num_groups; g++) {
      for (size_t p = 0; p < num_passes; p++) {
        size_t sid = AcGroupIndex(p, g, num_groups, frame_dim.num_dc_groups,
                                  has_ac_global);
        if (!has_section(sid)) {
          max_passes_for_group[g] = std::min(p, max_passes_for_group[g]);
        }
      }
    }
  }
  auto get_reader = [num_groups, num_passes, group_codes_begin, &group_offsets,
                     &group_sizes,
                     &reader](BitReader* JXL_RESTRICT store, size_t index) {
    return GetReaderForSection(num_groups, num_passes, group_codes_begin,
                               group_offsets, group_sizes, reader, store,
                               index);
  };

  LossyFrameDecoder lossy_frame_decoder;
  JXL_RETURN_IF_ERROR(lossy_frame_decoder.Init(frame_header, dec_state,
                                               downsampling, pool, aux_out));
  ModularFrameDecoder modular_frame_decoder(frame_header.ToFrameDimensions());

  if (dparams.keep_dct) {
    if (frame_header.encoding == FrameEncoding::kModular) {
      return JXL_FAILURE("Cannot output JPEG from Modular");
    }
    if (!decoded->IsJPEG()) {
      return JXL_FAILURE("Caller must set jpeg_data");
    }
  }

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
            shared.frame_dim.ysize_padded));
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

    JXL_RETURN_IF_ERROR(
        lossy_frame_decoder.State()->shared_storage.matrices.DecodeDC(
            global_reader));
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      JXL_RETURN_IF_ERROR(
          lossy_frame_decoder.DecodeGlobalDCInfo(global_reader, decoded));
    } else if (frame_header.encoding == FrameEncoding::kModular) {
      JXL_RETURN_IF_ERROR(
          lossy_frame_decoder.DecodeModularGlobalInfo(global_reader));
    }
    JXL_RETURN_IF_ERROR(modular_frame_decoder.DecodeGlobalInfo(
        global_reader, frame_header, decoded,
        /*decode_color = */
        (frame_header.encoding == FrameEncoding::kModular), xsize, ysize));
  }

  // global_store is either never used (in which case Close() is optional
  // and a no-op) or the real deal and we must call Close() and check the error
  // code here.
  JXL_RETURN_IF_ERROR(res);

  std::vector<AuxOut> aux_outs;

  // Decode DC groups.

  decoded->SetDecodedBytes(group_offsets[0] + group_sizes[0] +
                           group_codes_begin);

  JXL_RETURN_IF_ERROR(DecodeDC(frame_header, dec_state, modular_frame_decoder,
                               group_codes_begin, group_offsets, group_sizes,
                               pool, reader, &aux_outs, aux_out));

  Image3F opsin;

  if (*std::min_element(max_passes_for_group.begin(),
                        max_passes_for_group.end()) == 0 &&
      frame_header.encoding == FrameEncoding::kVarDCT &&
      !decoded->HasExtraChannels()) {
    // Upsample DC.
    opsin = CopyImage(*dec_state->shared->dc);
    Upsample(&opsin, /*upsampling=*/8,
             frame_header.nonserialized_metadata->transform_data);
    dec_state->has_partial_ac_groups = true;
    dec_state->decoded = PadImageMirror(opsin, kMaxFilterPadding, 0);
    // TODO(veluca): this value just disables EPF in the DC areas.
    FillImage(-10.0f, &dec_state->filter_weights.sigma);
  } else {
    opsin = Image3F(frame_dim.xsize_blocks * kBlockDim,
                    frame_dim.ysize_blocks * kBlockDim);
  }

  if (downsampling < 16 && frame_dim.num_groups > 1) {
    decoded->SetDecodedBytes(group_offsets[frame_dim.num_dc_groups] +
                             group_sizes[frame_dim.num_dc_groups] +
                             group_codes_begin);
  }

  // Read global AC info.
  if (ac_global_available) {
    PROFILER_ZONE("Global AC");
    res = true;
    {
      BitReader ac_info_store;
      BitReaderScopedCloser ac_info_store_closer(&ac_info_store, &res);
      BitReader* ac_info_reader = get_reader(&ac_info_store, global_ac_index);

      if (frame_header.encoding == FrameEncoding::kVarDCT) {
        JXL_RETURN_IF_ERROR(lossy_frame_decoder.DecodeGlobalACInfo(
            ac_info_reader, &modular_frame_decoder));
      }
    }
    // ac_info_store is either never used (in which case Close() is optional
    // and a no-op) or the real deal and we must call Close() and check the
    // return error.
    JXL_RETURN_IF_ERROR(res);
  }

  // Decode groups.
  std::atomic<int> num_errors{0};
  JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  const auto process_group_init = [&](size_t num_threads) {
    // The number of threads here might be different from the previous run, so
    // we need to re-update them.
    lossy_frame_decoder.SetNumThreads(num_threads);
    ResizeAuxOuts(aux_outs, num_threads, aux_out);
    return true;
  };

  const auto process_group = [&](const int task, const int thread) {
    PROFILER_ZONE("process_group");
    const size_t group_index = static_cast<size_t>(task);
    // DC has been upsampled and this group has no other information, don't
    // overwrite it.
    if (max_passes_for_group[group_index] == 0 &&
        frame_header.encoding == FrameEncoding::kVarDCT &&
        !decoded->HasExtraChannels()) {
      return;
    }
    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    const size_t x = gx * frame_dim.group_dim;
    const size_t y = gy * frame_dim.group_dim;
    Rect rect(x, y, frame_dim.group_dim, frame_dim.group_dim, xsize, ysize);

    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;
    BitReader pass0_reader_store;
    BitReader* pass0_group_reader =
        get_reader(&pass0_reader_store,
                   AcGroupIndex(0, group_index, num_groups,
                                frame_dim.num_dc_groups, has_ac_global));

    // don't limit to image dimensions here (is done in DecodeGroup)
    const Rect mrect(x, y, frame_dim.group_dim, frame_dim.group_dim);

    // Read passes.
    BitReader storage[kMaxNumPasses];
    BitReader* JXL_RESTRICT readers[kMaxNumPasses] = {};
    readers[0] = pass0_group_reader;
    for (size_t i = 1; i < max_passes_for_group[group_index]; i++) {
      readers[i] =
          get_reader(&storage[i - 1],
                     AcGroupIndex(i, group_index, num_groups,
                                  frame_dim.num_dc_groups, has_ac_global));
    }
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      if (!lossy_frame_decoder.DecodeACGroup(group_index, thread, readers,
                                             max_passes_for_group[group_index],
                                             &opsin, decoded, my_aux_out)) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
    int minShift = 0;
    int maxShift = 2;
    for (size_t i = 0; i < max_passes_for_group[group_index]; i++) {
      for (uint32_t j = 0; j < frame_header.passes.num_downsample; ++j) {
        if (i <= frame_header.passes.last_pass[j]) {
          if (frame_header.passes.downsample[j] == 8) minShift = 3;
          if (frame_header.passes.downsample[j] == 4) minShift = 2;
          if (frame_header.passes.downsample[j] == 2) minShift = 1;
          if (frame_header.passes.downsample[j] == 1) minShift = 0;
        }
      }
      if (!modular_frame_decoder.DecodeGroup(
              mrect, readers[i], my_aux_out, minShift, maxShift,
              ModularStreamId::ModularAC(group_index, i))) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      maxShift = minShift - 1;
      minShift = 0;
    }
    for (size_t i = 1; i < max_passes_for_group[group_index]; i++) {
      if (!storage[i - 1].Close()) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }

    if (!pass0_reader_store.Close()) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  };

  {
    PROFILER_ZONE("DecodeFrame pool");
    RunOnPool(pool, 0, num_groups, process_group_init, process_group,
              "DecodeFrame");
    JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);
  }

  if (downsampling < 8 && frame_dim.num_groups > 1) {
    uint64_t last_end_offset = 0;
    size_t end_index = AcGroupIndex(max_passes, 0, num_groups,
                                    frame_dim.num_dc_groups, has_ac_global);
    for (size_t i = 0; i < end_index; i++) {
      uint64_t end_offset = group_offsets[i] + group_sizes[i];
      last_end_offset = std::max(last_end_offset, end_offset);
    }
    decoded->SetDecodedBytes(group_codes_begin + last_end_offset);
  }

  // Resizing to 0 assimilates all the results when needed.
  ResizeAuxOuts(aux_outs, 0, aux_out);
  // undo global modular transforms and copy int pixel buffers to float ones
  JXL_RETURN_IF_ERROR(modular_frame_decoder.FinalizeDecoding(
      &opsin, decoded, pool,
      lossy_frame_decoder.State()->shared->matrices.DCQuants(), frame_header));

  JXL_RETURN_IF_ERROR(lossy_frame_decoder.FinalizeDecoding(&opsin, decoded));

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
    // Don't go beyond the bit_reader boundaries, as either we are OK with a
    // truncated file, or we checked that we don't go beyond bounds already
    // anyway.
    reader->SkipBits(std::min<size_t>(
        reader->TotalBytes() * kBitsPerByte - reader->TotalBitsConsumed(),
        groups_total_size * kBitsPerByte));  // aligned
  }

  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  if ((reader->TotalBitsConsumed()) > reader->TotalBytes() * kBitsPerByte &&
      !dparams.allow_partial_files) {
    return JXL_FAILURE("Read past stream end");
  }

  if (!decoded->IsJPEG()) {
    // A kRegularFrame is blended with the other frames, and thus results in a
    // coalesced frame of size equal to image dimensions. Other frames are not
    // blended, thus their final size is the size that was defined in the
    // frame_header.
    if (dec_state->shared->frame_header.frame_type == kRegularFrame) {
      decoded->ShrinkTo(
          dec_state->shared->frame_header.nonserialized_metadata->xsize(),
          dec_state->shared->frame_header.nonserialized_metadata->ysize());
    } else {
      // xsize_upsampled is the actual frame size, after any upsampling has been
      // applied.
      decoded->ShrinkTo(frame_dim.xsize_upsampled, frame_dim.ysize_upsampled);
    }
  }
  return true;
}

}  // namespace jxl
