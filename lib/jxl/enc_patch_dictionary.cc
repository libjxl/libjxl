// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_patch_dictionary.h"

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_dot_dictionary.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/patch_dictionary_internal.h"

namespace jxl {

// static
void PatchDictionaryEncoder::Encode(const PatchDictionary& pdic,
                                    BitWriter* writer, size_t layer,
                                    AuxOut* aux_out) {
  JXL_ASSERT(pdic.HasAny());
  std::vector<std::vector<Token>> tokens(1);

  auto add_num = [&](int context, size_t num) {
    tokens[0].emplace_back(context, num);
  };
  size_t num_ref_patch = 0;
  for (size_t i = 0; i < pdic.positions_.size();) {
    size_t i_start = i;
    while (i < pdic.positions_.size() &&
           pdic.positions_[i].ref_pos == pdic.positions_[i_start].ref_pos) {
      i++;
    }
    num_ref_patch++;
  }
  add_num(kNumRefPatchContext, num_ref_patch);
  for (size_t i = 0; i < pdic.positions_.size();) {
    size_t i_start = i;
    while (i < pdic.positions_.size() &&
           pdic.positions_[i].ref_pos == pdic.positions_[i_start].ref_pos) {
      i++;
    }
    size_t num = i - i_start;
    JXL_ASSERT(num > 0);
    add_num(kReferenceFrameContext, pdic.positions_[i_start].ref_pos.ref);
    add_num(kPatchReferencePositionContext,
            pdic.positions_[i_start].ref_pos.x0);
    add_num(kPatchReferencePositionContext,
            pdic.positions_[i_start].ref_pos.y0);
    add_num(kPatchSizeContext, pdic.positions_[i_start].ref_pos.xsize - 1);
    add_num(kPatchSizeContext, pdic.positions_[i_start].ref_pos.ysize - 1);
    add_num(kPatchCountContext, num - 1);
    for (size_t j = i_start; j < i; j++) {
      const PatchPosition& pos = pdic.positions_[j];
      if (j == i_start) {
        add_num(kPatchPositionContext, pos.x);
        add_num(kPatchPositionContext, pos.y);
      } else {
        add_num(kPatchOffsetContext,
                PackSigned(pos.x - pdic.positions_[j - 1].x));
        add_num(kPatchOffsetContext,
                PackSigned(pos.y - pdic.positions_[j - 1].y));
      }
      JXL_ASSERT(pdic.shared_->metadata->m.extra_channel_info.size() + 1 ==
                 pos.blending.size());
      for (size_t i = 0;
           i < pdic.shared_->metadata->m.extra_channel_info.size() + 1; i++) {
        const PatchBlending& info = pos.blending[i];
        add_num(kPatchBlendModeContext, static_cast<uint32_t>(info.mode));
        if (UsesAlpha(info.mode) &&
            pdic.shared_->metadata->m.extra_channel_info.size() > 1) {
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

// static
void PatchDictionaryEncoder::SubtractFrom(const PatchDictionary& pdic,
                                          Image3F* opsin) {
  // TODO(veluca): this can likely be optimized knowing it runs on full images.
  for (size_t y = 0; y < opsin->ysize(); y++) {
    if (y + 1 >= pdic.patch_starts_.size()) continue;
    float* JXL_RESTRICT rows[3] = {
        opsin->PlaneRow(0, y),
        opsin->PlaneRow(1, y),
        opsin->PlaneRow(2, y),
    };
    for (size_t id = pdic.patch_starts_[y]; id < pdic.patch_starts_[y + 1];
         id++) {
      const PatchPosition& pos = pdic.positions_[pdic.sorted_patches_[id]];
      size_t by = pos.y;
      size_t bx = pos.x;
      size_t xsize = pos.ref_pos.xsize;
      JXL_DASSERT(y >= by);
      JXL_DASSERT(y < by + pos.ref_pos.ysize);
      size_t iy = y - by;
      size_t ref = pos.ref_pos.ref;
      const float* JXL_RESTRICT ref_rows[3] = {
          pdic.shared_->reference_frames[ref].frame->color()->ConstPlaneRow(
              0, pos.ref_pos.y0 + iy) +
              pos.ref_pos.x0,
          pdic.shared_->reference_frames[ref].frame->color()->ConstPlaneRow(
              1, pos.ref_pos.y0 + iy) +
              pos.ref_pos.x0,
          pdic.shared_->reference_frames[ref].frame->color()->ConstPlaneRow(
              2, pos.ref_pos.y0 + iy) +
              pos.ref_pos.x0,
      };
      for (size_t ix = 0; ix < xsize; ix++) {
        for (size_t c = 0; c < 3; c++) {
          if (pos.blending[0].mode == PatchBlendMode::kAdd) {
            rows[c][bx + ix] -= ref_rows[c][ix];
          } else if (pos.blending[0].mode == PatchBlendMode::kReplace) {
            rows[c][bx + ix] = 0;
          } else if (pos.blending[0].mode == PatchBlendMode::kNone) {
            // Nothing to do.
          } else {
            JXL_ABORT("Blending mode %u not yet implemented",
                      (uint32_t)pos.blending[0].mode);
          }
        }
      }
    }
  }
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
      kChannelDequant[0] = 20.0f / 255;
      kChannelDequant[1] = 22.0f / 255;
      kChannelDequant[2] = 20.0f / 255;
      kChannelWeights[0] = 0.017 * 255;
      kChannelWeights[1] = 0.02 * 255;
      kChannelWeights[2] = 0.017 * 255;
    }
  }

  float ScaleForQuantization(float val, size_t c) {
    return val / kChannelDequant[c];
  }

  int Quantize(float val, size_t c) {
    return truncf(ScaleForQuantization(val, c));
  }

  bool is_similar_v(const float v1[3], const float v2[3], float threshold) {
    float distance = 0;
    for (size_t c = 0; c < 3; c++) {
      distance += std::fabs(v1[c] - v2[c]) * kChannelWeights[c];
    }
    return distance <= threshold;
  }
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
  const auto process_row = [&](const uint32_t y, size_t /* thread */) {
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
  JXL_CHECK(RunOnPool(pool, 0, opsin.ysize() / kPatchSide, ThreadPool::NoInit,
                      process_row, "IsScreenshotLike"));

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
              queue.emplace_back(next, src);
          }
        }
      }
    }
  }
  queue.clear();

  ImageF ccs;
  Rng rng(0);
  bool paint_ccs = false;
  if (WantDebugOutput(aux_out)) {
    aux_out->DumpPlaneNormalized("is_background", is_background);
    if (is_xyb) {
      aux_out->DumpXybImage("background", background);
    } else {
      aux_out->DumpImage("background", background);
    }
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
      stack.emplace_back(x, y);
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
      patch.o_x0 = min_x;
      patch.o_y0 = min_y;
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
        float cc_color = rng.UniformF(0.5, 1.0);
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

  return info;
}

// Heuristically tries to find regions that are better encoded as a modular
// patch
void AddMorePatches(std::vector<PatchInfo>& info, const Image3F& opsin,
                    PassesEncoderState* JXL_RESTRICT state, ThreadPool* pool,
                    AuxOut* aux_out, bool is_xyb) {
  // Only do this in VarDCT mode
  if (state->cparams.modular_mode) return;

  // STEP 1: Heuristically determine the kind of image content in each 8x8
  // block:
  //   0 = photo / vardct friendly
  //   1 = can choose
  //   2 = nonphoto / modular friendly
  //   3 = already covered by a patch (so stay away to avoid trouble with
  //   overlapping patches)
  const size_t bs = 8;
  ImageB content_type(opsin.xsize() / bs, opsin.ysize() / bs);
  ZeroFillImage(&content_type);
  size_t avgtype = 0;
  size_t count = 0;
  const float quant[3] = {8192.f, 256.f, 256.f};
  size_t nb_patches = info.size();

  for (size_t y = 0; y + bs <= state->shared.frame_dim.ysize; y += bs) {
    uint8_t* ct = content_type.Row(y / bs);
    for (size_t x = 0; x + bs <= state->shared.frame_dim.xsize; x += bs) {
      Rect block(x, y, bs, bs, opsin.xsize(), opsin.ysize());
      bool already_covered = false;
      for (size_t i = 0; i < nb_patches && !already_covered; i++) {
        for (const auto& pos : info[i].second) {
          Rect patch(pos.first, pos.second, info[i].first.xsize,
                     info[i].first.ysize);
          Rect intersection = block.Intersection(patch);
          if (intersection.xsize() > 0 && intersection.ysize() > 0) {
            already_covered = true;
            continue;
          }
        }
      }
      if (already_covered) {
        ct[x / bs] = 3;
        continue;
      }
      size_t perfect[3] = {}, mid[3] = {}, edge[3] = {};
      for (size_t c = 0; c < 3; c++) {
        for (size_t oy = 1; oy < bs; oy++) {
          const float* JXL_RESTRICT opp =
              opsin.PlaneRow(c, y + oy + (oy ? -1 : 0));
          const float* JXL_RESTRICT op = opsin.PlaneRow(c, y + oy);
          for (size_t ox = 1; ox < bs; ox++) {
            float px = op[x + ox] * quant[c];
            float N = opp[x + ox] * quant[c];
            float W = op[x - 1 + ox] * quant[c];
            float NW = opp[x - 1 + ox] * quant[c];
            float residual = abs(N + W - NW - px);
            if (residual < 1.8f) {
              // gradient predictor works well here, e.g. solid color area or
              // perfect (synthetic) gradient
              perfect[c]++;
              // perfect horizontal/vertical lines get perfectly predicted by
              // gradient, but they're also edges
              if (abs(px - N) > 50.f || abs(px - W) > 50.f) edge[c]++;
            } else if (residual > 50.f) {
              // strong edge, typical for non-photo (text, illustrations)
              edge[c]++;
            } else {
              // not smooth, but also no strong edge, typical for photo
              mid[c]++;
            }
          }
        }
      }
      // now we know for 7x7 pixels what category they are in.
      // start by assuming the block is modular-friendly and can/should be
      // encoded with a modular patch
      int type = 2;
      // if not enough pixels are 'perfect', it's at best "can choose"
      if (perfect[0] < 30) type = 1;
      if (perfect[1] < 20) type = 1;
      if (perfect[2] < 30) type = 1;
      // same if there are significantly more 'mid' than 'perfect' pixels in a
      // channel
      if (mid[0] > 10 + perfect[0]) type = 1;
      if (mid[1] > 15 + perfect[1]) type = 1;
      if (mid[2] > 10 + perfect[2]) type = 1;
      size_t sperfect = perfect[0] + perfect[1] + perfect[2],
             smid = mid[0] + mid[1] + mid[2],
             sedge = edge[0] + edge[1] + edge[2];
      // non-photo has mostly 'perfect' and 'edge', not much 'mid'
      if (smid + 10 > sperfect + sedge) type = 1;
      // not worth switching to modular if there are no hard edges
      if (sedge < 5) {
        if (smid < 15) {
          // it's mostly smooth, can still choose
          type = 1;
        } else {
          // it's not smooth, use vardct
          type = 0;
        }
      }
      // too much 'mid' -> certainly use VarDCT
      if (smid > 2 * sperfect + sedge + 8) type = 0;
      if (mid[1] > 2 * perfect[1] + edge[1] + 3) type = 0;

      // a lot of 'perfect' and also some edge -> certainly try to use Modular
      if (sperfect > 90 + smid && sedge > 8) type = 2;
      if (perfect[1] > 20 + mid[1] && sedge > 15 && mid[0] + mid[2] < 15)
        type = 2;
      ct[x / bs] = type;
      count++;
      avgtype += type;
    }
  }

  // STEP 2: Check for special case of "better to do everything with Modular"

  double imgtype = 1.0 * avgtype / count;
  // printf("image type: %.5f modular-friendly\n", imgtype);
  size_t total_pixels = 0;

  for (size_t i = 0; i < info.size(); i++) {
    size_t pixels = info[i].first.xsize * info[i].first.ysize;
    total_pixels += pixels;
  }
  if (imgtype > 1.1 && total_pixels < 0.01 * opsin.xsize() * opsin.ysize()) {
    info.clear();
    // best to just encode everything with Modular
    // TODO(jon): even better would be to skip patches in this case, and even
    // skip xyb if that can still be done
    QuantizedPatch p;
    p.xsize = opsin.xsize();
    p.ysize = opsin.ysize();
    p.o_x0 = p.o_y0 = 0;
    for (size_t c = 0; c < 3; c++) {
      p.fpixels[c].resize(p.xsize * p.ysize);
      for (size_t oy = 0; oy < opsin.ysize(); oy++) {
        const float* JXL_RESTRICT op = opsin.ConstPlaneRow(c, oy);
        for (size_t ox = 0; ox < opsin.xsize(); ox++) {
          p.fpixels[c][ox + oy * p.xsize] = op[ox];
        }
      }
    }
    info.emplace_back();
    info.back().first = std::move(p);
    info.back().second.emplace_back(0, 0);
    return;
  }

  // STEP 3: Play a bit of game of life on the content type heuristic map
  // Context map values:
  //    0 = bad (for modular),
  //    1 = neutral,
  //    2 = good (for modular),
  //    3 (already covered) = very good
  // Get rid of isolated good blocks, and allow some bad blocks to become
  // neutral
  size_t ctrow = content_type.PixelsPerRow();
  for (size_t iters = 0; iters < 5; iters++) {
    for (size_t y = 1; y + 1 < content_type.ysize(); y++) {
      uint8_t* ct = content_type.Row(y);
      for (size_t x = 1; x + 1 < content_type.xsize(); x++) {
        // sum of 8 neighbors, is between 0 and 16, 8 if all neutral
        int sum = ct[x - 1] + ct[x + 1];
        sum += ct[x - 1 - ctrow];
        sum += ct[x - ctrow];
        sum += ct[x + 1 - ctrow];
        sum += ct[x - 1 + ctrow];
        sum += ct[x + ctrow];
        sum += ct[x + 1 + ctrow];

        // isolated good block becomes neutral if it has at least 1 bad neighbor
        if (ct[x] == 2 && sum < 8) ct[x] = 1;

        // bad block in a very good neighborhood (e.g. 5 good and 3 neutral
        // neighbors) becomes neutral
        if (ct[x] == 0 && sum > 12) ct[x] = 1;
      }
    }
  }
  if (aux_out) {
    aux_out->DumpPlaneNormalized("content_type_processed", content_type);
  }

  // STEP 4: Fit rectangles to cover good blocks, allowing them to also cover
  // neutral blocks but no bad blacks or already-covered blocks. Trim them to
  // not contain more neutral blocks than needed.
  for (size_t y = 0; y < content_type.ysize(); y++) {
    uint8_t* ct = content_type.Row(y);
    for (size_t x = 0; x < content_type.xsize(); x++) {
      if (ct[x] != 2) continue;
      size_t x0 = x, y0 = y, x1 = x, y1 = y;  // current rect
      size_t x2 = x, y2 = y, x3 = x, y3 = y;  // trimmed rect
      bool did_enlarge = true;
      const size_t max_aspect_ratio = 5;
      const size_t max_dimension = 32;  // max 256x256 per patch
      while (did_enlarge) {
        did_enlarge = false;
        uint8_t* ct0 = content_type.Row(y0);
        // Try growing on the right
        bool grow = x1 - x0 < max_dimension &&
                    ((x1 - x0 + 1) < (y1 - y0 + 1) * max_aspect_ratio &&
                     x1 + 1 < content_type.xsize());
        bool added_good = false;
        for (size_t yi = 0; grow && yi <= y1 - y0; yi++) {
          int type = ct0[x1 + 1 + ctrow * yi];
          if (type == 0 || type == 3) grow = false;
          if (type == 2) added_good = true;
        }
        if (grow) {
          x1++;
          did_enlarge = true;
          if (added_good) x2 = x1;
        }
        // Try growing on the bottom
        grow = y1 - y0 < max_dimension &&
               ((y1 - y0 + 1) < (x1 - x0 + 1) * max_aspect_ratio &&
                y1 + 1 < content_type.ysize());
        added_good = false;
        for (size_t xi = 0; grow && xi <= x1 - x0; xi++) {
          int type = ct[x0 + xi + ctrow * (y1 - y + 1)];
          if (type == 0 || type == 3) grow = false;
          if (type == 2) added_good = true;
        }
        if (grow) {
          y1++;
          did_enlarge = true;
          if (added_good) y2 = y1;
        }

        // Try growing on the left
        grow = x1 - x0 < max_dimension &&
               (x1 - x0 + 1) < (y1 - y0 + 1) * max_aspect_ratio && x0 > 0;
        added_good = false;
        for (size_t yi = 0; grow && yi <= y1 - y0; yi++) {
          int type = ct0[x0 - 1 + ctrow * yi];
          if (type == 0 || type == 3) grow = false;
          if (type == 2) added_good = true;
        }
        if (grow) {
          x0--;
          did_enlarge = true;
          if (added_good) x3 = x0;
        }
        // Try growing on the top
        grow = y1 - y0 < max_dimension &&
               (y1 - y0 + 1) < (x1 - x0 + 1) * max_aspect_ratio && y0 > 0;
        added_good = false;
        for (size_t xi = 0; grow && xi <= x1 - x0; xi++) {
          int type = ct0[x0 + xi - ctrow];
          if (type == 0 || type == 3) grow = false;
          if (type == 2) added_good = true;
        }
        if (grow) {
          y0--;
          did_enlarge = true;
          if (added_good) y3 = y0;
        }
      }
      // trim to what added good blocks
      x1 = x2;
      y1 = y2;
      x0 = x3;
      y0 = y3;
      // avoiding small patches can perhaps help to reduce the signaling cost
      // and the amount of border entropy in the patch frame
      const size_t kMinPatchSizeBlocks = 2;
      if ((x1 - x0 + 1) * (y1 - y0 + 1) < kMinPatchSizeBlocks) continue;

      Rect block(x0 * bs, y0 * bs, (x1 - x0 + 1) * bs, (y1 - y0 + 1) * bs,
                 opsin.xsize(), opsin.ysize());

      float bg[3] = {};
      bool reject = false;
      // Require 4 identical corners, and assume the corner color is a good
      // background color to keep as a residual after subtracting the patch.
      // This is also unlikely to happen by chance in a non-synthetic image, so
      // it's a good extra filtering.
      //
      // TODO(jon): somehow relax this requirement. The problem is that leaving
      // discontinuities in the residual image (after subtracting patches) is
      // bad for VarDCT, especially considering gaborish. It can lead to
      // boundary artifacts. Discontinuities can be avoided by e.g. using a very
      // blurry version of the patch as 'background color', but then the patch
      // itself gets more entropy, leading to worse compression. What is
      // basically needed is some kind of "in-painting" in the VarDCT-encoded
      // residual image, in such a way that it doesn't lead to higher entropy
      // when subtracted from the Modular-encoded patch.
      for (size_t c = 0; c < 3; c++) {
        const float* JXL_RESTRICT op0 = block.ConstPlaneRow(opsin, c, 0);
        const float* JXL_RESTRICT op1 =
            block.ConstPlaneRow(opsin, c, block.ysize() - 1);
        float tl = op0[0];
        float tr = op0[block.xsize() - 1];
        float bl = op1[0];
        float br = op1[block.xsize() - 1];
        if (tl == tr && tl == bl && tl == br)
          bg[c] = tl;
        else {
          reject = true;
          break;
        }
      }
      if (reject) continue;

      // STEP 5: Add a patch

      // Mark the region as 'already covered'
      for (size_t yi = y0; yi <= y1; yi++) {
        uint8_t* cti = content_type.Row(yi);
        for (size_t xi = x0; xi <= x1; xi++) cti[xi] = 3;
      }

      // Make patch with (likely) background color subtracted, so the residual
      // will become solid background
      QuantizedPatch p;
      p.xsize = block.xsize();
      p.ysize = block.ysize();
      p.o_x0 = block.x0();
      p.o_y0 = block.y0();
      for (size_t c = 0; c < 3; c++) {
        p.fpixels[c].resize(p.xsize * p.ysize);
        for (size_t oy = 0; oy < block.ysize(); oy++) {
          const float* JXL_RESTRICT op = block.ConstPlaneRow(opsin, c, oy);
          for (size_t ox = 0; ox < block.xsize(); ox++) {
            p.fpixels[c][ox + oy * p.xsize] = op[ox] - bg[c];
          }
        }
      }

      info.emplace_back();
      info.back().first = std::move(p);
      info.back().second.emplace_back(block.x0(), block.y0());
    }
  }
}

}  // namespace

void FindBestPatchDictionary(const Image3F& opsin,
                             PassesEncoderState* JXL_RESTRICT state,
                             const JxlCmsInterface& cms, ThreadPool* pool,
                             AuxOut* aux_out, bool is_xyb) {
  if (state->cparams.patches == Override::kOff) return;

  std::vector<PatchInfo> info =
      FindTextLikePatches(opsin, state, pool, aux_out, is_xyb);

  // TODO(veluca): this doesn't work if both dots and patches are enabled.
  // For now, since dots and patches are not likely to occur in the same kind of
  // images, disable dots if some patches were found.
  if (info.empty() &&
      ApplyOverride(
          state->cparams.dots,
          state->cparams.speed_tier <= SpeedTier::kSquirrel &&
              state->cparams.butteraugli_distance >= kMinButteraugliForDots)) {
    info = FindDotDictionary(state->cparams, opsin, state->shared.cmap, pool);
  }
  if (ApplyOverride(state->cparams.more_patches,
                    state->cparams.speed_tier <= SpeedTier::kSquirrel)) {
    AddMorePatches(info, opsin, state, pool, aux_out, is_xyb);
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
  size_t ref_ysize = std::max<float>(max_y_size, total_pixels / ref_xsize);
  std::vector<std::pair<size_t, size_t>> ref_positions(info.size());
  // TODO(veluca): allow partial overlaps of patches that have the same pixels.
  size_t max_y = 0;
  bool try_fitting = true;
  do {
    max_y = 0;
    // Increase packed image size.
    ref_xsize =
        8 * (static_cast<size_t>(ref_xsize * kBinPackingSlackness / 8) + 1);
    ref_ysize =
        8 * (static_cast<size_t>(ref_ysize * kBinPackingSlackness / 8) + 1);
    if (ref_xsize >= opsin.xsize() && ref_ysize >= opsin.ysize()) {
      ref_xsize = opsin.xsize();
      ref_ysize = opsin.ysize();
      try_fitting = false;
    }

    ImageB occupied;
    if (try_fitting) {
      occupied = ImageB(ref_xsize, ref_ysize);
      ZeroFillImage(&occupied);
    }
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
      if (!try_fitting) {
        // When not trying to fit the patches in a smaller frame, just put them
        // in the same spot as where they came from
        x0 = info[patch].first.o_x0;
        y0 = info[patch].first.o_y0;
      } else {
        // For every possible start position ...
        for (; y0 + ysize <= ref_ysize; y0++) {
          x0 = 0;
          for (; x0 + xsize <= ref_xsize; x0++) {
            bool has_occupied_pixel = false;
            size_t x = x0;
            // Check if it is possible to place the patch in this position in
            // the reference frame.
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

        // We found a position: mark the corresponding positions in the
        // reference image as used.
        for (size_t y = y0; y < y0 + ysize; y++) {
          for (size_t x = x0; x < x0 + xsize; x++) {
            occupied_rows[y * occupied_stride + x] = true;
          }
        }
      }
      ref_positions[patch] = {x0, y0};
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
  // Recursive application of patches could create very weird issues.
  cparams.patches = Override::kOff;
  //  cparams.butteraugli_distance *= 1.5f;
  //  cparams.butteraugli_distance *= 0.8f;
  RoundtripPatchFrame(&reference_frame, state, 0, cparams, cms, pool, true);

  // TODO(veluca): this assumes that applying patches is commutative, which is
  // not true for all blending modes. This code only produces kAdd patches, so
  // this works out.
  std::sort(positions.begin(), positions.end());
  PatchDictionaryEncoder::SetPositions(&state->shared.image_features.patches,
                                       std::move(positions));
}

void RoundtripPatchFrame(Image3F* reference_frame,
                         PassesEncoderState* JXL_RESTRICT state, int idx,
                         CompressParams& cparams, const JxlCmsInterface& cms,
                         ThreadPool* pool, bool subtract) {
  FrameInfo patch_frame_info;
  cparams.resampling = 1;
  cparams.ec_resampling = 1;
  cparams.dots = Override::kOff;
  cparams.noise = Override::kOff;
  cparams.modular_mode = true;
  cparams.responsive = 0;
  cparams.progressive_dc = 0;
  cparams.progressive_mode = false;
  cparams.qprogressive_mode = false;
  // Use gradient predictor and not Predictor::Best.
  cparams.options.predictor = Predictor::Gradient;
  patch_frame_info.save_as_reference = idx;  // always saved.
  patch_frame_info.frame_type = FrameType::kReferenceOnly;
  patch_frame_info.save_before_color_transform = true;
  ImageBundle ib(&state->shared.metadata->m);
  // TODO(veluca): metadata.color_encoding is a lie: ib is in XYB, but there is
  // no simple way to express that yet.
  patch_frame_info.ib_needs_color_transform = false;
  ib.SetFromImage(std::move(*reference_frame),
                  state->shared.metadata->m.color_encoding);
  if (!ib.metadata()->extra_channel_info.empty()) {
    // Add dummy extra channels to the patch image: patch encoding does not yet
    // support extra channels, but the codec expects that the amount of extra
    // channels in frames matches that in the metadata of the codestream.
    std::vector<ImageF> extra_channels;
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
                        &roundtrip_state, cms, pool, special_frame.get(),
                        nullptr));
  const Span<const uint8_t> encoded = special_frame->GetSpan();
  state->special_frames.emplace_back(std::move(special_frame));
  if (subtract) {
    BitReader br(encoded);
    ImageBundle decoded(&state->shared.metadata->m);
    PassesDecoderState dec_state;
    JXL_CHECK(dec_state.output_encoding_info.Set(
        *state->shared.metadata,
        ColorEncoding::LinearSRGB(
            state->shared.metadata->m.color_encoding.IsGray())));
    JXL_CHECK(DecodeFrame({}, &dec_state, pool, &br, &decoded,
                          *state->shared.metadata, /*constraints=*/nullptr));
    size_t ref_xsize =
        dec_state.shared_storage.reference_frames[idx].storage.color()->xsize();
    // if the frame itself uses patches, we need to decode another frame
    if (!ref_xsize) {
      JXL_CHECK(DecodeFrame({}, &dec_state, pool, &br, &decoded,
                            *state->shared.metadata, /*constraints=*/nullptr));
    }
    JXL_CHECK(br.Close());
    state->shared.reference_frames[idx] =
        std::move(dec_state.shared_storage.reference_frames[idx]);
  } else {
    state->shared.reference_frames[idx].storage = std::move(ib);
  }
  state->shared.reference_frames[idx].frame =
      &state->shared.reference_frames[idx].storage;
}

}  // namespace jxl
