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

#include "lib/jxl/patch_dictionary.h"

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

#include <algorithm>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/detect_dots.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

constexpr int kMaxPatches = 1 << 24;

// Context numbers as specified in Section C.4.5, Listing C.2:
enum Contexts {
  kNumRefPatchContext = 0,
  kReferenceFrameContext = 1,
  kPatchSizeContext = 2,
  kPatchReferencePositionContext = 3,
  kPatchPositionContext = 4,
  kPatchBlendModeContext = 5,
  kPatchOffsetContext = 6,
  kPatchCountContext = 7,
  kPatchAlphaChannelContext = 8,
  kPatchClampContext = 9,
  kNumPatchDictionaryContexts
};

void PatchDictionary::Encode(BitWriter* writer, size_t layer,
                             AuxOut* aux_out) const {
  JXL_ASSERT(HasAny());
  std::vector<std::vector<Token>> tokens(1);

  auto add_num = [&](int context, size_t num) {
    tokens[0].emplace_back(context, num);
  };
  size_t num_ref_patch = 0;
  for (size_t i = 0; i < positions_.size();) {
    size_t i_start = i;
    while (i < positions_.size() &&
           positions_[i].ref_pos == positions_[i_start].ref_pos) {
      i++;
    }
    num_ref_patch++;
  }
  add_num(kNumRefPatchContext, num_ref_patch);
  for (size_t i = 0; i < positions_.size();) {
    size_t i_start = i;
    while (i < positions_.size() &&
           positions_[i].ref_pos == positions_[i_start].ref_pos) {
      i++;
    }
    size_t num = i - i_start;
    JXL_ASSERT(num > 0);
    add_num(kReferenceFrameContext, positions_[i_start].ref_pos.ref);
    add_num(kPatchReferencePositionContext, positions_[i_start].ref_pos.x0);
    add_num(kPatchReferencePositionContext, positions_[i_start].ref_pos.y0);
    add_num(kPatchSizeContext, positions_[i_start].ref_pos.xsize - 1);
    add_num(kPatchSizeContext, positions_[i_start].ref_pos.ysize - 1);
    add_num(kPatchCountContext, num - 1);
    for (size_t j = i_start; j < i; j++) {
      const PatchPosition& pos = positions_[j];
      if (j == i_start) {
        add_num(kPatchPositionContext, pos.x);
        add_num(kPatchPositionContext, pos.y);
      } else {
        add_num(kPatchOffsetContext, PackSigned(pos.x - positions_[j - 1].x));
        add_num(kPatchOffsetContext, PackSigned(pos.y - positions_[j - 1].y));
      }
      JXL_ASSERT(shared_->metadata->m.extra_channel_info.size() + 1 ==
                 pos.blending.size());
      for (size_t i = 0; i < shared_->metadata->m.extra_channel_info.size() + 1;
           i++) {
        const PatchBlending& info = pos.blending[i];
        add_num(kPatchBlendModeContext, static_cast<uint32_t>(info.mode));
        if (UsesAlpha(info.mode) &&
            shared_->metadata->m.extra_channel_info.size() > 1) {
          add_num(kPatchAlphaChannelContext, info.alpha_channel);
        }
        if (UsesClamp(info.mode)) {
          add_num(kPatchClampContext, info.clamp);
        }
      }
    }
  }

  EntropyEncodingData codes;
  std::vector<uint8_t> context_map;
  BuildAndEncodeHistograms(HistogramParams(), kNumPatchDictionaryContexts,
                           tokens, &codes, &context_map, writer, layer,
                           aux_out);
  WriteTokens(tokens[0], codes, context_map, writer, layer, aux_out);
}

Status PatchDictionary::Decode(BitReader* br, size_t xsize, size_t ysize) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(
      DecodeHistograms(br, kNumPatchDictionaryContexts, &code, &context_map));
  ANSSymbolReader decoder(&code, br);

  auto read_num = [&](size_t context) {
    size_t r = decoder.ReadHybridUint(context, br, context_map);
    return r;
  };

  size_t num_ref_patch = read_num(kNumRefPatchContext);
  // TODO(veluca): does this make sense?
  if (num_ref_patch > kMaxPatches) {
    return JXL_FAILURE("Too many patches in dictionary");
  }

  for (size_t id = 0; id < num_ref_patch; id++) {
    PatchReferencePosition ref_pos;
    ref_pos.ref = read_num(kReferenceFrameContext);
    if (ref_pos.ref >= kMaxNumReferenceFrames ||
        shared_->reference_frames[ref_pos.ref].frame->xsize() == 0) {
      return JXL_FAILURE("Invalid reference frame ID");
    }
    const ImageBundle& ib = *shared_->reference_frames[ref_pos.ref].frame;
    ref_pos.x0 = read_num(kPatchReferencePositionContext);
    ref_pos.y0 = read_num(kPatchReferencePositionContext);
    ref_pos.xsize = read_num(kPatchSizeContext) + 1;
    ref_pos.ysize = read_num(kPatchSizeContext) + 1;
    if (ref_pos.x0 + ref_pos.xsize > ib.xsize()) {
      return JXL_FAILURE("Invalid position specified in reference frame");
    }
    if (ref_pos.y0 + ref_pos.ysize > ib.ysize()) {
      return JXL_FAILURE("Invalid position specified in reference frame");
    }
    size_t id_count = read_num(kPatchCountContext) + 1;
    if (id_count > kMaxPatches) {
      return JXL_FAILURE("Too many patches in dictionary");
    }
    positions_.reserve(positions_.size() + id_count);
    for (size_t i = 0; i < id_count; i++) {
      PatchPosition pos;
      pos.ref_pos = ref_pos;
      if (i == 0) {
        pos.x = read_num(kPatchPositionContext);
        pos.y = read_num(kPatchPositionContext);
      } else {
        pos.x =
            positions_.back().x + UnpackSigned(read_num(kPatchOffsetContext));
        pos.y =
            positions_.back().y + UnpackSigned(read_num(kPatchOffsetContext));
      }
      if (pos.x + ref_pos.xsize > xsize) {
        return JXL_FAILURE("Invalid patch x: at %zu + %zu > %zu", pos.x,
                           ref_pos.xsize, xsize);
      }
      if (pos.y + ref_pos.ysize > ysize) {
        return JXL_FAILURE("Invalid patch y: at %zu + %zu > %zu", pos.y,
                           ref_pos.ysize, ysize);
      }
      for (size_t i = 0; i < shared_->metadata->m.extra_channel_info.size() + 1;
           i++) {
        uint32_t blend_mode = read_num(kPatchBlendModeContext);
        if (blend_mode >= uint32_t(PatchBlendMode::kNumBlendModes)) {
          return JXL_FAILURE("Invalid patch blend mode: %u", blend_mode);
        }
        PatchBlending info;
        info.mode = static_cast<PatchBlendMode>(blend_mode);
        if (i != 0 && info.mode != PatchBlendMode::kNone) {
          return JXL_FAILURE(
              "Blending of extra channels with patches is not supported yet");
        }
        if (info.mode != PatchBlendMode::kAdd &&
            info.mode != PatchBlendMode::kNone &&
            info.mode != PatchBlendMode::kReplace) {
          return JXL_FAILURE("Blending mode not supported yet: %u", blend_mode);
        }
        if (UsesAlpha(info.mode) &&
            shared_->metadata->m.extra_channel_info.size() > 1) {
          info.alpha_channel = read_num(kPatchAlphaChannelContext);
          if (info.alpha_channel >=
              shared_->metadata->m.extra_channel_info.size()) {
            return JXL_FAILURE(
                "Invalid alpha channel for blending: %u out of %u\n",
                info.alpha_channel,
                (uint32_t)shared_->metadata->m.extra_channel_info.size());
          }
        }
        if (UsesClamp(info.mode)) {
          info.clamp = read_num(kPatchClampContext);
        }
        pos.blending.push_back(info);
      }
      positions_.push_back(std::move(pos));
    }
  }

  if (!decoder.CheckANSFinalState()) {
    return JXL_FAILURE("ANS checksum failure.");
  }
  if (!HasAny()) {
    return JXL_FAILURE("Decoded patch dictionary but got none");
  }

  ComputePatchCache();
  return true;
}

void PatchDictionary::ComputePatchCache() {
  if (positions_.empty()) return;
  std::vector<std::pair<size_t, size_t>> sorted_patches_y;
  for (size_t i = 0; i < positions_.size(); i++) {
    const PatchPosition& pos = positions_[i];
    for (size_t y = pos.y; y < pos.y + pos.ref_pos.ysize; y++) {
      sorted_patches_y.emplace_back(y, i);
    }
  }
  // The relative order of patches that affect the same pixels is preserved.
  // This is important for patches that have a blend mode different from kAdd.
  std::sort(sorted_patches_y.begin(), sorted_patches_y.end());
  patch_starts_.resize(sorted_patches_y.back().first + 2,
                       sorted_patches_y.size());
  sorted_patches_.resize(sorted_patches_y.size());
  for (size_t i = 0; i < sorted_patches_y.size(); i++) {
    sorted_patches_[i] = sorted_patches_y[i].second;
    patch_starts_[sorted_patches_y[i].first] =
        std::min(patch_starts_[sorted_patches_y[i].first], i);
  }
  for (size_t i = patch_starts_.size() - 1; i > 0; i--) {
    patch_starts_[i - 1] = std::min(patch_starts_[i], patch_starts_[i - 1]);
  }
}

template <bool add>
void PatchDictionary::Apply(Image3F* opsin, const Rect& opsin_rect,
                            const Rect& image_rect) const {
  JXL_CHECK(SameSize(opsin_rect, image_rect));
  size_t num = 0;
  for (size_t y = image_rect.y0(); y < image_rect.y0() + image_rect.ysize();
       y++) {
    if (y + 1 >= patch_starts_.size()) continue;
    float* JXL_RESTRICT rows[3] = {
        opsin_rect.PlaneRow(opsin, 0, y - image_rect.y0()),
        opsin_rect.PlaneRow(opsin, 1, y - image_rect.y0()),
        opsin_rect.PlaneRow(opsin, 2, y - image_rect.y0()),
    };
    for (size_t id = patch_starts_[y]; id < patch_starts_[y + 1]; id++) {
      num++;
      const PatchPosition& pos = positions_[sorted_patches_[id]];
      size_t by = pos.y;
      size_t bx = pos.x;
      size_t xsize = pos.ref_pos.xsize;
      JXL_DASSERT(y >= by);
      JXL_DASSERT(y < by + pos.ref_pos.ysize);
      size_t iy = y - by;
      size_t ref = pos.ref_pos.ref;
      if (bx >= image_rect.x0() + image_rect.xsize()) continue;
      if (bx + xsize < image_rect.x0()) continue;
      // TODO(veluca): check that the reference frame is in XYB.
      const float* JXL_RESTRICT ref_rows[3] = {
          shared_->reference_frames[ref].frame->color()->ConstPlaneRow(
              0, pos.ref_pos.y0 + iy) +
              pos.ref_pos.x0,
          shared_->reference_frames[ref].frame->color()->ConstPlaneRow(
              1, pos.ref_pos.y0 + iy) +
              pos.ref_pos.x0,
          shared_->reference_frames[ref].frame->color()->ConstPlaneRow(
              2, pos.ref_pos.y0 + iy) +
              pos.ref_pos.x0,
      };
      // TODO(veluca): use the same code as in dec_reconstruct.cc.
      for (size_t ix = 0; ix < xsize; ix++) {
        // TODO(veluca): hoist branches and checks.
        // TODO(veluca): implement for extra channels.
        if (bx + ix < image_rect.x0()) continue;
        if (bx + ix >= image_rect.x0() + image_rect.xsize()) continue;
        for (size_t c = 0; c < 3; c++) {
          if (add) {
            if (pos.blending[0].mode == PatchBlendMode::kAdd) {
              rows[c][bx + ix - image_rect.x0()] += ref_rows[c][ix];
            } else if (pos.blending[0].mode == PatchBlendMode::kReplace) {
              rows[c][bx + ix - image_rect.x0()] = ref_rows[c][ix];
            } else if (pos.blending[0].mode == PatchBlendMode::kNone) {
              // Nothing to do.
            } else {
              // Checked in decoding code.
              JXL_ABORT("Blending mode %u not yet implemented",
                        (uint32_t)pos.blending[0].mode);
            }
          } else {
            if (pos.blending[0].mode == PatchBlendMode::kAdd) {
              rows[c][bx + ix - image_rect.x0()] -= ref_rows[c][ix];
            } else if (pos.blending[0].mode == PatchBlendMode::kReplace) {
              rows[c][bx + ix - image_rect.x0()] = 0;
            } else if (pos.blending[0].mode == PatchBlendMode::kNone) {
              // Nothing to do.
            } else {
              // Checked in decoding code.
              JXL_ABORT("Blending mode %u not yet implemented",
                        (uint32_t)pos.blending[0].mode);
            }
          }
        }
      }
    }
  }
}

void PatchDictionary::AddTo(Image3F* opsin, const Rect& opsin_rect,
                            const Rect& image_rect) const {
  Apply</*add=*/true>(opsin, opsin_rect, image_rect);
}

void PatchDictionary::SubtractFrom(Image3F* opsin) const {
  Apply</*add=*/false>(opsin, Rect(*opsin), Rect(*opsin));
}

namespace {

struct PatchColorspaceInfo {
  float kChannelDequant[3];
  float kChannelWeights[3];

  explicit PatchColorspaceInfo(bool is_xyb) {
    if (is_xyb) {
      kChannelDequant[0] = 0.01615;
      kChannelDequant[1] = 0.08875;
      kChannelDequant[2] = 0.1922;
      kChannelWeights[0] = 30.0;
      kChannelWeights[1] = 3.0;
      kChannelWeights[2] = 1.0;
    } else {
      kChannelDequant[0] = 20;
      kChannelDequant[1] = 22;
      kChannelDequant[2] = 20;
      kChannelWeights[0] = 0.017;
      kChannelWeights[1] = 0.02;
      kChannelWeights[2] = 0.017;
    }
  }

  float ScaleForQuantization(float val, size_t c) {
    return val / kChannelDequant[c];
  }

  int Quantize(float val, size_t c) {
    return std::trunc(ScaleForQuantization(val, c));
  }

  bool is_similar_v(const float v1[3], const float v2[3], float threshold) {
    float distance = 0;
    for (size_t c = 0; c < 3; c++) {
      distance += std::fabs(v1[c] - v2[c]) * kChannelWeights[c];
    }
    return distance <= threshold;
  };
};

std::vector<PatchInfo> FindTextLikePatches(
    const Image3F& opsin, const PassesEncoderState* JXL_RESTRICT state,
    ThreadPool* pool, AuxOut* aux_out, bool is_xyb) {
  if (state->cparams.patches == Override::kOff) return {};

  PatchColorspaceInfo pci(is_xyb);
  float kSimilarThreshold = 0.8f;

  auto is_similar_impl = [&pci](std::pair<uint32_t, uint32_t> p1,
                                std::pair<uint32_t, uint32_t> p2,
                                const float* JXL_RESTRICT rows[3],
                                size_t stride, float threshold) {
    float v1[3], v2[3];
    for (size_t c = 0; c < 3; c++) {
      v1[c] = rows[c][p1.second * stride + p1.first];
      v2[c] = rows[c][p2.second * stride + p2.first];
    }
    return pci.is_similar_v(v1, v2, threshold);
  };

  std::atomic<bool> has_screenshot_areas{false};
  const size_t opsin_stride = opsin.PixelsPerRow();
  const float* JXL_RESTRICT opsin_rows[3] = {opsin.ConstPlaneRow(0, 0),
                                             opsin.ConstPlaneRow(1, 0),
                                             opsin.ConstPlaneRow(2, 0)};

  auto is_same = [&opsin_rows, opsin_stride](std::pair<uint32_t, uint32_t> p1,
                                             std::pair<uint32_t, uint32_t> p2) {
    for (size_t c = 0; c < 3; c++) {
      float v1 = opsin_rows[c][p1.second * opsin_stride + p1.first];
      float v2 = opsin_rows[c][p2.second * opsin_stride + p2.first];
      if (std::fabs(v1 - v2) > 1e-4) {
        return false;
      }
    }
    return true;
  };

  auto is_similar = [&](std::pair<uint32_t, uint32_t> p1,
                        std::pair<uint32_t, uint32_t> p2) {
    return is_similar_impl(p1, p2, opsin_rows, opsin_stride, kSimilarThreshold);
  };

  constexpr int64_t kPatchSide = 4;
  constexpr int64_t kExtraSide = 4;

  // Look for kPatchSide size squares, naturally aligned, that all have the same
  // pixel values.
  ImageB is_screenshot_like(DivCeil(opsin.xsize(), kPatchSide),
                            DivCeil(opsin.ysize(), kPatchSide));
  ZeroFillImage(&is_screenshot_like);
  uint8_t* JXL_RESTRICT screenshot_row = is_screenshot_like.Row(0);
  const size_t screenshot_stride = is_screenshot_like.PixelsPerRow();
  const auto process_row = [&](uint64_t y, int _) {
    for (uint64_t x = 0; x < opsin.xsize() / kPatchSide; x++) {
      bool all_same = true;
      for (size_t iy = 0; iy < static_cast<size_t>(kPatchSide); iy++) {
        for (size_t ix = 0; ix < static_cast<size_t>(kPatchSide); ix++) {
          size_t cx = x * kPatchSide + ix;
          size_t cy = y * kPatchSide + iy;
          if (!is_same({cx, cy}, {x * kPatchSide, y * kPatchSide})) {
            all_same = false;
            break;
          }
        }
      }
      if (!all_same) continue;
      size_t num = 0;
      size_t num_same = 0;
      for (int64_t iy = -kExtraSide; iy < kExtraSide + kPatchSide; iy++) {
        for (int64_t ix = -kExtraSide; ix < kExtraSide + kPatchSide; ix++) {
          int64_t cx = x * kPatchSide + ix;
          int64_t cy = y * kPatchSide + iy;
          if (cx < 0 || static_cast<uint64_t>(cx) >= opsin.xsize() ||  //
              cy < 0 || static_cast<uint64_t>(cy) >= opsin.ysize()) {
            continue;
          }
          num++;
          if (is_same({cx, cy}, {x * kPatchSide, y * kPatchSide})) num_same++;
        }
      }
      // Too few equal pixels nearby.
      if (num_same * 8 < num * 7) continue;
      screenshot_row[y * screenshot_stride + x] = 1;
      has_screenshot_areas = true;
    }
  };
  RunOnPool(pool, 0, opsin.ysize() / kPatchSide, ThreadPool::SkipInit(),
            process_row, "IsScreenshotLike");

  // TODO(veluca): also parallelize the rest of this function.
  if (WantDebugOutput(aux_out)) {
    aux_out->DumpPlaneNormalized("screenshot_like", is_screenshot_like);
  }

  constexpr int kSearchRadius = 1;

  if (!ApplyOverride(state->cparams.patches, has_screenshot_areas)) {
    return {};
  }
  // Search for "similar enough" pixels near the screenshot-like areas.
  ImageB is_background(opsin.xsize(), opsin.ysize());
  ZeroFillImage(&is_background);
  Image3F background(opsin.xsize(), opsin.ysize());
  ZeroFillImage(&background);
  constexpr size_t kDistanceLimit = 50;
  float* JXL_RESTRICT background_rows[3] = {
      background.PlaneRow(0, 0),
      background.PlaneRow(1, 0),
      background.PlaneRow(2, 0),
  };
  const size_t background_stride = background.PixelsPerRow();
  uint8_t* JXL_RESTRICT is_background_row = is_background.Row(0);
  const size_t is_background_stride = is_background.PixelsPerRow();
  std::vector<
      std::pair<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>>>
      queue;
  size_t queue_front = 0;
  for (size_t y = 0; y < opsin.ysize(); y++) {
    for (size_t x = 0; x < opsin.xsize(); x++) {
      if (!screenshot_row[screenshot_stride * (y / kPatchSide) +
                          (x / kPatchSide)])
        continue;
      queue.push_back({{x, y}, {x, y}});
    }
  }
  while (queue.size() != queue_front) {
    std::pair<uint32_t, uint32_t> cur = queue[queue_front].first;
    std::pair<uint32_t, uint32_t> src = queue[queue_front].second;
    queue_front++;
    if (is_background_row[cur.second * is_background_stride + cur.first])
      continue;
    is_background_row[cur.second * is_background_stride + cur.first] = 1;
    for (size_t c = 0; c < 3; c++) {
      background_rows[c][cur.second * background_stride + cur.first] =
          opsin_rows[c][src.second * opsin_stride + src.first];
    }
    for (int dx = -kSearchRadius; dx <= kSearchRadius; dx++) {
      for (int dy = -kSearchRadius; dy <= kSearchRadius; dy++) {
        if (dx == 0 && dy == 0) continue;
        int next_first = cur.first + dx;
        int next_second = cur.second + dy;
        if (next_first < 0 || next_second < 0 ||
            static_cast<uint32_t>(next_first) >= opsin.xsize() ||
            static_cast<uint32_t>(next_second) >= opsin.ysize()) {
          continue;
        }
        if (static_cast<uint32_t>(
                std::abs(next_first - static_cast<int>(src.first)) +
                std::abs(next_second - static_cast<int>(src.second))) >
            kDistanceLimit) {
          continue;
        }
        std::pair<uint32_t, uint32_t> next{next_first, next_second};
        if (is_similar(src, next)) {
          if (!screenshot_row[next.second / kPatchSide * screenshot_stride +
                              next.first / kPatchSide] ||
              is_same(src, next)) {
            if (!is_background_row[next.second * is_background_stride +
                                   next.first])
              queue.push_back({next, src});
          }
        }
      }
    }
  }
  queue.clear();

  ImageF ccs;
  std::mt19937 rng;
  std::uniform_real_distribution<float> dist(0.5, 1.0);
  bool paint_ccs = false;
  if (WantDebugOutput(aux_out)) {
    aux_out->DumpPlaneNormalized("is_background", is_background);
    aux_out->DumpXybImage("background", background);
    ccs = ImageF(opsin.xsize(), opsin.ysize());
    ZeroFillImage(&ccs);
    paint_ccs = true;
  }

  constexpr float kVerySimilarThreshold = 0.03f;
  constexpr float kHasSimilarThreshold = 0.03f;

  const float* JXL_RESTRICT const_background_rows[3] = {
      background_rows[0], background_rows[1], background_rows[2]};
  auto is_similar_b = [&](std::pair<int, int> p1, std::pair<int, int> p2) {
    return is_similar_impl(p1, p2, const_background_rows, background_stride,
                           kVerySimilarThreshold);
  };

  constexpr int kMinPeak = 2;
  constexpr int kHasSimilarRadius = 2;

  std::vector<PatchInfo> info;

  // Find small CC outside the "similar enough" areas, compute bounding boxes,
  // and run heuristics to exclude some patches.
  ImageB visited(opsin.xsize(), opsin.ysize());
  ZeroFillImage(&visited);
  uint8_t* JXL_RESTRICT visited_row = visited.Row(0);
  const size_t visited_stride = visited.PixelsPerRow();
  std::vector<std::pair<uint32_t, uint32_t>> cc;
  std::vector<std::pair<uint32_t, uint32_t>> stack;
  for (size_t y = 0; y < opsin.ysize(); y++) {
    for (size_t x = 0; x < opsin.xsize(); x++) {
      if (is_background_row[y * is_background_stride + x]) continue;
      cc.clear();
      stack.clear();
      stack.push_back({x, y});
      size_t min_x = x;
      size_t max_x = x;
      size_t min_y = y;
      size_t max_y = y;
      std::pair<uint32_t, uint32_t> reference;
      bool found_border = false;
      bool all_similar = true;
      while (!stack.empty()) {
        std::pair<uint32_t, uint32_t> cur = stack.back();
        stack.pop_back();
        if (visited_row[cur.second * visited_stride + cur.first]) continue;
        visited_row[cur.second * visited_stride + cur.first] = 1;
        if (cur.first < min_x) min_x = cur.first;
        if (cur.first > max_x) max_x = cur.first;
        if (cur.second < min_y) min_y = cur.second;
        if (cur.second > max_y) max_y = cur.second;
        if (paint_ccs) {
          cc.push_back(cur);
        }
        for (int dx = -kSearchRadius; dx <= kSearchRadius; dx++) {
          for (int dy = -kSearchRadius; dy <= kSearchRadius; dy++) {
            if (dx == 0 && dy == 0) continue;
            int next_first = static_cast<int32_t>(cur.first) + dx;
            int next_second = static_cast<int32_t>(cur.second) + dy;
            if (next_first < 0 || next_second < 0 ||
                static_cast<uint32_t>(next_first) >= opsin.xsize() ||
                static_cast<uint32_t>(next_second) >= opsin.ysize()) {
              continue;
            }
            std::pair<uint32_t, uint32_t> next{next_first, next_second};
            if (!is_background_row[next.second * is_background_stride +
                                   next.first]) {
              stack.push_back(next);
            } else {
              if (!found_border) {
                reference = next;
                found_border = true;
              } else {
                if (!is_similar_b(next, reference)) all_similar = false;
              }
            }
          }
        }
      }
      if (!found_border || !all_similar || max_x - min_x >= kMaxPatchSize ||
          max_y - min_y >= kMaxPatchSize) {
        continue;
      }
      size_t bpos = background_stride * reference.second + reference.first;
      float ref[3] = {background_rows[0][bpos], background_rows[1][bpos],
                      background_rows[2][bpos]};
      bool has_similar = false;
      for (size_t iy = std::max<int>(
               static_cast<int32_t>(min_y) - kHasSimilarRadius, 0);
           iy < std::min(max_y + kHasSimilarRadius + 1, opsin.ysize()); iy++) {
        for (size_t ix = std::max<int>(
                 static_cast<int32_t>(min_x) - kHasSimilarRadius, 0);
             ix < std::min(max_x + kHasSimilarRadius + 1, opsin.xsize());
             ix++) {
          size_t opos = opsin_stride * iy + ix;
          float px[3] = {opsin_rows[0][opos], opsin_rows[1][opos],
                         opsin_rows[2][opos]};
          if (pci.is_similar_v(ref, px, kHasSimilarThreshold)) {
            has_similar = true;
          }
        }
      }
      if (!has_similar) continue;
      info.emplace_back();
      info.back().second.emplace_back(min_x, min_y);
      QuantizedPatch& patch = info.back().first;
      patch.xsize = max_x - min_x + 1;
      patch.ysize = max_y - min_y + 1;
      int max_value = 0;
      for (size_t c : {1, 0, 2}) {
        for (size_t iy = min_y; iy <= max_y; iy++) {
          for (size_t ix = min_x; ix <= max_x; ix++) {
            size_t offset = (iy - min_y) * patch.xsize + ix - min_x;
            patch.fpixels[c][offset] =
                opsin_rows[c][iy * opsin_stride + ix] - ref[c];
            int val = pci.Quantize(patch.fpixels[c][offset], c);
            patch.pixels[c][offset] = val;
            if (std::abs(val) > max_value) max_value = std::abs(val);
          }
        }
      }
      if (max_value < kMinPeak) {
        info.pop_back();
        continue;
      }
      if (paint_ccs) {
        float cc_color = dist(rng);
        for (std::pair<uint32_t, uint32_t> p : cc) {
          ccs.Row(p.second)[p.first] = cc_color;
        }
      }
    }
  }

  if (paint_ccs) {
    JXL_ASSERT(WantDebugOutput(aux_out));
    aux_out->DumpPlaneNormalized("ccs", ccs);
  }
  if (info.empty()) {
    return {};
  }

  // Remove duplicates.
  constexpr size_t kMinPatchOccurences = 2;
  std::sort(info.begin(), info.end());
  size_t unique = 0;
  for (size_t i = 1; i < info.size(); i++) {
    if (info[i].first == info[unique].first) {
      info[unique].second.insert(info[unique].second.end(),
                                 info[i].second.begin(), info[i].second.end());
    } else {
      if (info[unique].second.size() >= kMinPatchOccurences) {
        unique++;
      }
      info[unique] = info[i];
    }
  }
  if (info[unique].second.size() >= kMinPatchOccurences) {
    unique++;
  }
  info.resize(unique);

  size_t max_patch_size = 0;

  for (size_t i = 0; i < info.size(); i++) {
    size_t pixels = info[i].first.xsize * info[i].first.ysize;
    if (pixels > max_patch_size) max_patch_size = pixels;
  }

  // don't use patches if all patches are smaller than this
  constexpr size_t kMinMaxPatchSize = 20;
  if (max_patch_size < kMinMaxPatchSize) return {};

  // Ensure that the specified set of patches doesn't produce out-of-bounds
  // pixels.
  // TODO(veluca): figure out why this is still necessary even with RCTs that
  // don't depend on bit depth.
  if (state->cparams.modular_mode && state->cparams.quality_pair.first >= 100) {
    constexpr size_t kMaxPatchArea = kMaxPatchSize * kMaxPatchSize;
    std::vector<float> min_then_max_px(2 * kMaxPatchArea);
    for (size_t i = 0; i < info.size(); i++) {
      for (size_t c = 0; c < 3; c++) {
        float* JXL_RESTRICT min_px = min_then_max_px.data();
        float* JXL_RESTRICT max_px = min_px + kMaxPatchArea;
        std::fill(min_px, min_px + kMaxPatchArea, 255);
        std::fill(max_px, max_px + kMaxPatchArea, 0);
        size_t xsize = info[i].first.xsize;
        for (size_t j = 0; j < info[i].second.size(); j++) {
          size_t bx = info[i].second[j].first;
          size_t by = info[i].second[j].second;
          for (size_t iy = 0; iy < info[i].first.ysize; iy++) {
            for (size_t ix = 0; ix < xsize; ix++) {
              float v = opsin_rows[c][(by + iy) * opsin_stride + bx + ix];
              if (v < min_px[iy * xsize + ix]) min_px[iy * xsize + ix] = v;
              if (v > max_px[iy * xsize + ix]) max_px[iy * xsize + ix] = v;
            }
          }
        }
        for (size_t iy = 0; iy < info[i].first.ysize; iy++) {
          for (size_t ix = 0; ix < xsize; ix++) {
            float smallest = min_px[iy * xsize + ix];
            float biggest = max_px[iy * xsize + ix];
            JXL_ASSERT(smallest <= biggest);
            float& out = info[i].first.fpixels[c][iy * xsize + ix];
            // Clamp fpixels so that subtracting the patch never creates a
            // negative value, or a value above 255.
            JXL_ASSERT(biggest - 255 <= smallest);
            out = std::max(smallest, out);
            out = std::min(biggest - 255.f, out);
          }
        }
      }
    }
  }
  return info;
}

}  // namespace

void FindBestPatchDictionary(const Image3F& opsin,
                             PassesEncoderState* JXL_RESTRICT state,
                             ThreadPool* pool, AuxOut* aux_out, bool is_xyb) {
  state->shared.image_features.patches = PatchDictionary();
  state->shared.image_features.patches.SetPassesSharedState(&state->shared);

  std::vector<PatchInfo> info =
      FindTextLikePatches(opsin, state, pool, aux_out, is_xyb);

  // TODO(veluca): this doesn't work if both dots and patches are enabled.
  // For now, since dots and patches are not likely to occur in the same kind of
  // images, disable dots if some patches were found.
  if (ApplyOverride(
          state->cparams.dots,
          state->cparams.speed_tier <= SpeedTier::kSquirrel &&
              state->cparams.butteraugli_distance >= kMinButteraugliForDots)) {
    info = FindDotDictionary(state->cparams, opsin, state->shared.cmap, pool);
  }

  if (info.empty()) return;

  std::sort(
      info.begin(), info.end(), [&](const PatchInfo& a, const PatchInfo& b) {
        return a.first.xsize * a.first.ysize > b.first.xsize * b.first.ysize;
      });

  size_t max_x_size = 0;
  size_t max_y_size = 0;
  size_t total_pixels = 0;

  for (size_t i = 0; i < info.size(); i++) {
    size_t pixels = info[i].first.xsize * info[i].first.ysize;
    if (max_x_size < info[i].first.xsize) max_x_size = info[i].first.xsize;
    if (max_y_size < info[i].first.ysize) max_y_size = info[i].first.ysize;
    total_pixels += pixels;
  }

  // Bin-packing & conversion of patches.
  constexpr float kBinPackingSlackness = 1.05f;
  size_t ref_xsize = std::max<float>(max_x_size, std::sqrt(total_pixels));
  size_t ref_ysize = std::max<float>(max_y_size, std::sqrt(total_pixels));
  std::vector<std::pair<size_t, size_t>> ref_positions(info.size());
  // TODO(veluca): allow partial overlaps of patches that have the same pixels.
  size_t max_y = 0;
  do {
    max_y = 0;
    // Increase packed image size.
    ref_xsize = ref_xsize * kBinPackingSlackness + 1;
    ref_ysize = ref_ysize * kBinPackingSlackness + 1;

    ImageB occupied(ref_xsize, ref_ysize);
    ZeroFillImage(&occupied);
    uint8_t* JXL_RESTRICT occupied_rows = occupied.Row(0);
    size_t occupied_stride = occupied.PixelsPerRow();

    bool success = true;
    // For every patch...
    for (size_t patch = 0; patch < info.size(); patch++) {
      size_t x0 = 0;
      size_t y0 = 0;
      size_t xsize = info[patch].first.xsize;
      size_t ysize = info[patch].first.ysize;
      bool found = false;
      // For every possible start position ...
      for (; y0 + ysize <= ref_ysize; y0++) {
        x0 = 0;
        for (; x0 + xsize <= ref_xsize; x0++) {
          bool has_occupied_pixel = false;
          size_t x = x0;
          // Check if it is possible to place the patch in this position in the
          // reference frame.
          for (size_t y = y0; y < y0 + ysize; y++) {
            x = x0;
            for (; x < x0 + xsize; x++) {
              if (occupied_rows[y * occupied_stride + x]) {
                has_occupied_pixel = true;
                break;
              }
            }
          }  // end of positioning check
          if (!has_occupied_pixel) {
            found = true;
            break;
          }
          x0 = x;  // Jump to next pixel after the occupied one.
        }
        if (found) break;
      }  // end of start position checking

      // We didn't find a possible position: repeat from the beginning with a
      // larger reference frame size.
      if (!found) {
        success = false;
        break;
      }

      // We found a position: mark the corresponding positions in the reference
      // image as used.
      ref_positions[patch] = {x0, y0};
      for (size_t y = y0; y < y0 + ysize; y++) {
        for (size_t x = x0; x < x0 + xsize; x++) {
          occupied_rows[y * occupied_stride + x] = true;
        }
      }
      max_y = std::max(max_y, y0 + ysize);
    }

    if (success) break;
  } while (true);

  JXL_ASSERT(ref_ysize >= max_y);

  ref_ysize = max_y;

  Image3F reference_frame(ref_xsize, ref_ysize);
  // TODO(veluca): figure out a better way to fill the image.
  ZeroFillImage(&reference_frame);
  std::vector<PatchPosition> positions;
  float* JXL_RESTRICT ref_rows[3] = {
      reference_frame.PlaneRow(0, 0),
      reference_frame.PlaneRow(1, 0),
      reference_frame.PlaneRow(2, 0),
  };
  size_t ref_stride = reference_frame.PixelsPerRow();

  for (size_t i = 0; i < info.size(); i++) {
    PatchReferencePosition ref_pos;
    ref_pos.xsize = info[i].first.xsize;
    ref_pos.ysize = info[i].first.ysize;
    ref_pos.x0 = ref_positions[i].first;
    ref_pos.y0 = ref_positions[i].second;
    ref_pos.ref = 0;
    for (size_t y = 0; y < ref_pos.ysize; y++) {
      for (size_t x = 0; x < ref_pos.xsize; x++) {
        for (size_t c = 0; c < 3; c++) {
          ref_rows[c][(y + ref_pos.y0) * ref_stride + x + ref_pos.x0] =
              info[i].first.fpixels[c][y * ref_pos.xsize + x];
        }
      }
    }
    // Add color channels, ignore other channels.
    std::vector<PatchBlending> blending_info(
        state->shared.metadata->m.extra_channel_info.size() + 1,
        PatchBlending{PatchBlendMode::kNone, 0, false});
    blending_info[0].mode = PatchBlendMode::kAdd;
    for (const auto& pos : info[i].second) {
      positions.emplace_back(
          PatchPosition{pos.first, pos.second, blending_info, ref_pos});
    }
  }

  CompressParams cparams = state->cparams;
  cparams.resampling = 1;
  // Recursive application of patches could create very weird issues.
  cparams.patches = Override::kOff;
  cparams.dots = Override::kOff;
  cparams.noise = Override::kOff;
  cparams.modular_mode = true;
  cparams.responsive = 0;
  cparams.progressive_dc = false;
  cparams.progressive_mode = false;
  cparams.qprogressive_mode = false;
  // Use gradient predictor and not Predictor::Best.
  cparams.options.predictor = Predictor::Gradient;
  // TODO(veluca): possibly change heuristics here.
  if (!cparams.modular_mode) {
    cparams.quality_pair.first = cparams.quality_pair.second =
        80 - cparams.butteraugli_distance * 12;
  } else {
    cparams.quality_pair.first = (100 + 3 * cparams.quality_pair.first) * 0.25f;
    cparams.quality_pair.second =
        (100 + 3 * cparams.quality_pair.second) * 0.25f;
  }
  FrameInfo patch_frame_info;
  patch_frame_info.save_as_reference = 0;  // always saved.
  patch_frame_info.frame_type = FrameType::kReferenceOnly;
  patch_frame_info.save_before_color_transform = true;

  ImageBundle ib(&state->shared.metadata->m);
  // TODO(veluca): metadata.color_encoding is a lie: ib is in XYB, but there is
  // no simple way to express that yet.
  patch_frame_info.ib_needs_color_transform = false;
  patch_frame_info.save_as_reference = 0;
  ib.SetFromImage(std::move(reference_frame),
                  state->shared.metadata->m.color_encoding);
  if (!ib.metadata()->extra_channel_info.empty()) {
    // Add dummy extra channels to the patch image: patches do not yet support
    // extra channels, but the codec expects that the amount of extra channels
    // in frames matches that in the metadata of the codestream.
    std::vector<ImageU> extra_channels;
    extra_channels.reserve(ib.metadata()->extra_channel_info.size());
    for (size_t i = 0; i < ib.metadata()->extra_channel_info.size(); i++) {
      extra_channels.emplace_back(ib.xsize(), ib.ysize());
      // Must initialize the image with data to not affect blending with
      // uninitialized memory.
      // TODO(lode): patches must copy and use the real extra channels instead.
      ZeroFillImage(&extra_channels.back());
    }
    ib.SetExtraChannels(std::move(extra_channels));
  }

  PassesEncoderState roundtrip_state;
  auto special_frame = std::unique_ptr<BitWriter>(new BitWriter());
  JXL_CHECK(EncodeFrame(cparams, patch_frame_info, state->shared.metadata, ib,
                        &roundtrip_state, pool, special_frame.get(), nullptr));
  const Span<const uint8_t> encoded = special_frame->GetSpan();
  state->special_frames.emplace_back(std::move(special_frame));
  if (cparams.butteraugli_distance < kMinButteraugliToSubtractOriginalPatches) {
    BitReader br(encoded);
    ImageBundle decoded(&state->shared.metadata->m);
    PassesDecoderState dec_state;
    JXL_CHECK(DecodeFrame({}, &dec_state, pool, &br, nullptr, &decoded,
                          *state->shared.metadata));
    JXL_CHECK(br.Close());
    state->shared.reference_frames[0].storage = std::move(decoded);
  } else {
    state->shared.reference_frames[0].storage = std::move(ib);
  }
  state->shared.reference_frames[0].frame =
      &state->shared.reference_frames[0].storage;
  // TODO(veluca): this assumes that applying patches is commutative, which is
  // not true for all blending modes. This code only produces kAdd patches, so
  // this works out.
  std::sort(positions.begin(), positions.end());
  state->shared.image_features.patches.SetPositions(std::move(positions));
}
}  // namespace jxl
