// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <istream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lib/extras/codec_in_out.h"
#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/color_encoding_cms.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_aux_out.h"

jxl::AuxOut* StatsAuxOut();  // (JXL_FROM_TREE_STATS, see main)
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_context_map.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_debug_tree.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/noise.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/splines.h"
#include "lib/jxl/test_utils.h"  // TODO(eustas): cut this dependency
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"

namespace jpegxl {
namespace tools {

using ::jxl::BitWriter;
using ::jxl::BlendMode;
using ::jxl::CodecInOut;
using ::jxl::CodecMetadata;
using ::jxl::ColorEncoding;
using ::jxl::ColorTransform;
using ::jxl::CompressParams;
using ::jxl::FrameDimensions;
using ::jxl::FrameInfo;
using ::jxl::Image3F;
using ::jxl::ImageF;
using ::jxl::PaddedBytes;
using ::jxl::PassesEncoderState;
using ::jxl::Predictor;
using ::jxl::PropertyDecisionNode;
using ::jxl::QuantizedSpline;
using ::jxl::Span;
using ::jxl::Spline;
using ::jxl::Splines;
using ::jxl::Status;
using ::jxl::StatusOr;
using ::jxl::Tree;

namespace {

// Like std::stoi, std::stoul and std::stof, but without exceptions: *num is the
// number of characters parsed, or std::string::npos if `t` does not start with
// a number in range, so that the callers' `*num != t.size()` checks report
// every invalid token. Unsigned values cannot be negative.
int32_t ParseInt(const std::string& t, size_t* num) {
  const char* begin = t.c_str();
  char* end = nullptr;
  errno = 0;
  long long v = strtoll(begin, &end, 10);
  if (end == begin || errno == ERANGE ||
      v < std::numeric_limits<int32_t>::min() ||
      v > std::numeric_limits<int32_t>::max()) {
    *num = std::string::npos;
    return 0;
  }
  *num = end - begin;
  return static_cast<int32_t>(v);
}

size_t ParseUnsigned(const std::string& t, size_t* num) {
  const char* begin = t.c_str();
  char* end = nullptr;
  errno = 0;
  unsigned long long v = strtoull(begin, &end, 10);
  if (end == begin || errno == ERANGE || t.find('-') != std::string::npos ||
      v > std::numeric_limits<size_t>::max()) {
    *num = std::string::npos;
    return 0;
  }
  *num = end - begin;
  return static_cast<size_t>(v);
}

float ParseFloat(const std::string& t, size_t* num) {
  const char* begin = t.c_str();
  char* end = nullptr;
  errno = 0;
  float v = strtof(begin, &end);
  if (end == begin || errno == ERANGE) {
    *num = std::string::npos;
    return 0;
  }
  *num = end - begin;
  return v;
}
struct SplineData {
  // The adjustment the splines are quantized with. With the old keyword
  // (SplineQuantizationAdjustment, and by default: 1) the file signals 0 all
  // the same (kept for existing art: such splines render 1 + adj / 8 times
  // stronger and wider than written); with SplineAdjustment it is signaled.
  int32_t quantization_adjustment = 1;
  bool signal_adjustment = false;
  std::vector<Spline> splines;
};

// A decision tree over named integer properties, in the tree syntax
// ("if <property> > <value>", "- Set <value>"): used for the HF context model
// of VarDCT frames.
struct NamedTree {
  struct Node {
    int property;  // -1 for a leaf
    int32_t split_or_value;
    size_t then_node, else_node;
  };
  std::vector<Node> nodes;

  int32_t Eval(const std::vector<int32_t>& props) const {
    size_t i = 0;
    while (nodes[i].property >= 0) {
      i = props[nodes[i].property] > nodes[i].split_or_value
              ? nodes[i].then_node
              : nodes[i].else_node;
    }
    return nodes[i].split_or_value;
  }
};

template <typename F>
bool ParseNamedTree(F& tok, const std::vector<std::string>& properties,
                    NamedTree* tree) {
  std::string t = tok();
  while (t == "/*") {
    while (t != "*/" && !t.empty()) t = tok();
    t = tok();
  }
  size_t pos = tree->nodes.size();
  tree->nodes.emplace_back();
  size_t num = 0;
  if (t == "if") {
    std::string name = tok();
    auto it = std::find(properties.begin(), properties.end(), name);
    if (it == properties.end()) {
      fprintf(stderr, "Unexpected property: %s (here:", name.c_str());
      for (const std::string& p : properties) fprintf(stderr, " %s", p.c_str());
      fprintf(stderr, ")\n");
      return false;
    }
    if (tok() != ">") {
      fprintf(stderr, "Expected >\n");
      return false;
    }
    t = tok();
    int32_t split = ParseInt(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid split value: %s\n", t.c_str());
      return false;
    }
    tree->nodes[pos] = {static_cast<int>(it - properties.begin()), split, 0, 0};
    tree->nodes[pos].then_node = tree->nodes.size();
    if (!ParseNamedTree(tok, properties, tree)) return false;
    tree->nodes[pos].else_node = tree->nodes.size();
    return ParseNamedTree(tok, properties, tree);
  }
  if (t == "-" && tok() == "Set") {
    t = tok();
    int32_t value = ParseInt(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid value: %s\n", t.c_str());
      return false;
    }
    tree->nodes[pos] = {-1, value, 0, 0};
    return true;
  }
  fprintf(stderr, "Expected 'if' or '- Set' in HF context tree, got %s\n",
          t.c_str());
  return false;
}

// The HF context model of VarDCT frames (persists across frames).
struct HFContextSettings {
  // LF thresholds in the order of the LF channels (Y, X, B), QF thresholds.
  std::vector<int> lf_thresholds[3];
  std::vector<uint32_t> qf_thresholds;
  bool custom_block_ctx = false;
  // Properties c (0 Y, 1 X, 2 B), ord, qf, lfy, lfx, lfb -> block context.
  NamedTree block_ctx;
  // Properties hfkind (0 = number of nonzeros, 1 = coefficient), bctx,
  // nzpred, k, nzleft, prev -> fixed value.
  NamedTree coefficients;
};

// The blocks of a VarDCT frame as the decoder sees them, from the LF image and
// the HF metadata that the tree gives (as in ComputeVarDCTDataFromTree).
struct VarDCTBlocks {
  size_t xsize_blocks, ysize_blocks;
  // Per block: the LF context (bucket of the quantized LF), the quant field
  // (1..256), and the AC strategy of the varblock starting there (-1 if the
  // block is covered by a varblock starting elsewhere).
  std::vector<uint8_t> lf_ctx;
  std::vector<int32_t> qf;
  std::vector<int> acs;
};

bool ComputeVarDCTBlocks(JxlMemoryManager* memory_manager, const Tree& tree,
                         const FrameDimensions& frame_dim,
                         const jxl::BlockCtxMap& map, VarDCTBlocks* blocks) {
  const size_t xb = frame_dim.xsize_blocks;
  const size_t yb = frame_dim.ysize_blocks;
  blocks->xsize_blocks = xb;
  blocks->ysize_blocks = yb;
  blocks->lf_ctx.assign(xb * yb, 0);
  blocks->qf.assign(xb * yb, 1);
  blocks->acs.assign(xb * yb, -1);
  std::vector<bool> covered(xb * yb);
  const size_t n = frame_dim.num_dc_groups;
  for (size_t g = 0; g < n; g++) {
    const jxl::Rect r = frame_dim.DCGroupRect(g);
    // LF: channels Y, X, B (VarDCT DC stream 1 + g).
    auto lf = jxl::Image::Create(memory_manager, r.xsize(), r.ysize(), 8, 3);
    if (!lf.ok()) return false;
    jxl::Image lf_image = std::move(lf).value_();
    if (!jxl::EvaluateTreeWithZeroResiduals(tree, 1 + g, &lf_image)) {
      fprintf(stderr, "Could not evaluate the LF tree\n");
      return false;
    }
    for (size_t y = 0; y < r.ysize(); y++) {
      const int32_t* row_y = lf_image.channel[0].Row(y);
      const int32_t* row_x = lf_image.channel[1].Row(y);
      const int32_t* row_b = lf_image.channel[2].Row(y);
      for (size_t x = 0; x < r.xsize(); x++) {
        int bucket_x = 0;
        int bucket_y = 0;
        int bucket_b = 0;
        for (int t : map.dc_thresholds[0]) bucket_x += row_x[x] > t;
        for (int t : map.dc_thresholds[1]) bucket_y += row_y[x] > t;
        for (int t : map.dc_thresholds[2]) bucket_b += row_b[x] > t;
        int bucket = bucket_x * (map.dc_thresholds[2].size() + 1) + bucket_b;
        bucket = bucket * (map.dc_thresholds[1].size() + 1) + bucket_y;
        blocks->lf_ctx[(r.y0() + y) * xb + r.x0() + x] = bucket;
      }
    }
    // HF metadata (stream 1 + 2n + g): c = 2 has the AC strategies (row 0)
    // and quant fields (row 1) of the varblocks, in placement order.
    auto meta = jxl::Image::Create(memory_manager, r.xsize(), r.ysize(), 8, 4);
    if (!meta.ok()) return false;
    jxl::Image meta_image = std::move(meta).value_();
    const size_t tiles_x = (r.xsize() + 7) >> 3;
    const size_t tiles_y = (r.ysize() + 7) >> 3;
    for (size_t c = 0; c < 3; c++) {
      auto ch = c < 2 ? jxl::Channel::Create(memory_manager, tiles_x, tiles_y)
                      : jxl::Channel::Create(memory_manager,
                                             r.xsize() * r.ysize(), 2);
      if (!ch.ok()) return false;
      meta_image.channel[c] = std::move(ch).value_();
    }
    if (!jxl::EvaluateTreeWithZeroResiduals(tree, 1 + 2 * n + g, &meta_image)) {
      fprintf(stderr, "Could not evaluate the HF metadata tree\n");
      return false;
    }
    const int32_t* acs_in = meta_image.channel[2].Row(0);
    const int32_t* qf_in = meta_image.channel[2].Row(1);
    size_t num = 0;
    for (size_t iy = 0; iy < r.ysize(); iy++) {
      for (size_t ix = 0; ix < r.xsize(); ix++) {
        const size_t x = r.x0() + ix;
        const size_t y = r.y0() + iy;
        if (covered[y * xb + x]) continue;
        const int32_t raw = acs_in[num];
        if (!jxl::AcStrategy::IsRawStrategyValid(raw)) {
          fprintf(stderr, "AC strategy %d (entry %zu) is not in 0..26\n", raw,
                  num);
          return false;
        }
        jxl::AcStrategy acs = jxl::AcStrategy::FromRawStrategy(raw);
        const size_t group = jxl::kGroupDimInBlocks;
        const size_t xend = std::min((x / group + 1) * group, xb);
        const size_t yend = std::min((y / group + 1) * group, yb);
        if (x + acs.covered_blocks_x() > xend ||
            y + acs.covered_blocks_y() > yend) {
          fprintf(stderr,
                  "AC strategy %d (entry %zu) at block (%zu, %zu) crosses a "
                  "group or image edge\n",
                  raw, num, x, y);
          return false;
        }
        for (size_t cy = 0; cy < acs.covered_blocks_y(); cy++) {
          for (size_t cx = 0; cx < acs.covered_blocks_x(); cx++) {
            if (covered[(y + cy) * xb + x + cx]) {
              fprintf(stderr,
                      "AC strategy %d (entry %zu) at block (%zu, %zu) "
                      "overlaps an earlier block\n",
                      raw, num, x, y);
              return false;
            }
            covered[(y + cy) * xb + x + cx] = true;
          }
        }
        blocks->acs[y * xb + x] = raw;
        blocks->qf[y * xb + x] =
            1 + std::max<int32_t>(0, std::min<int32_t>(255, qf_in[num]));
        num++;
      }
    }
  }
  return true;
}

// Encoded size in bits of a context map.
size_t ContextMapBits(JxlMemoryManager* memory_manager,
                      const std::vector<uint8_t>& context_map) {
  if (context_map.size() <= 1) return 0;
  size_t num = *std::max_element(context_map.begin(), context_map.end()) + 1;
  jxl::BitWriter writer{memory_manager};
  // EncodeContextMap writes directly for a single histogram, so it needs an
  // allotment (writing without one is out of bounds).
  if (!writer.WithMaxBits(
          1024 + 24 * context_map.size(), jxl::LayerType::Ac, nullptr, [&] {
            return jxl::EncodeContextMap(context_map, num, &writer,
                                         jxl::LayerType::Ac, nullptr);
          })) {
    return std::numeric_limits<size_t>::max();
  }
  return writer.BitsWritten();
}

// Encoded size in bits of a block context map.
size_t BlockCtxMapBits(JxlMemoryManager* memory_manager,
                       const jxl::BlockCtxMap& map) {
  jxl::BitWriter writer{memory_manager};
  if (!jxl::EncodeBlockCtxMap(map, &writer, nullptr)) {
    return std::numeric_limits<size_t>::max();
  }
  return writer.BitsWritten();
}

// Encoded size in bits of fixed HF tokens.
size_t FixedTokensBits(JxlMemoryManager* memory_manager,
                       const std::vector<uint32_t>& tokens) {
  jxl::BitWriter writer{memory_manager};
  jxl::EntropyEncodingData codes;
  auto cost = jxl::EncodeFixedTokenHistograms(
      memory_manager, tokens, &codes, &writer, jxl::LayerType::Ac, nullptr);
  if (!cost.ok()) return std::numeric_limits<size_t>::max();
  return writer.BitsWritten();
}

// Fills the entries of `values` that are -1 (don't care), in the way that
// costs the fewest bits (as measured by `bits`): with the previous or the
// next value that is set (longer runs), or with `fallback[i]` if that is not
// -1 (else the previous value).
template <typename T, typename Bits>
std::vector<T> FillDontCares(const std::vector<int32_t>& values,
                             const std::vector<int32_t>& fallback,
                             const Bits& bits) {
  std::vector<std::vector<T>> candidates;
  for (size_t mode = 0; mode < 3; mode++) {
    std::vector<int32_t> v = values;
    if (mode == 0) {
      for (size_t i = 0; i < v.size(); i++) {
        if (v[i] < 0) v[i] = fallback[i];
      }
    }
    if (mode == 2) std::reverse(v.begin(), v.end());
    int32_t last = -1;
    for (int32_t& x : v) {
      if (x < 0) {
        x = last;
      } else {
        last = x;
      }
    }
    // Leading don't cares: the first value that is set.
    int32_t first = 0;
    for (int32_t x : v) {
      if (x >= 0) {
        first = x;
        break;
      }
    }
    for (int32_t& x : v) {
      if (x < 0) x = first;
    }
    if (mode == 2) std::reverse(v.begin(), v.end());
    candidates.emplace_back(v.begin(), v.end());
  }
  size_t best = 0;
  size_t best_bits = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < candidates.size(); i++) {
    size_t b = bits(candidates[i]);
    if (b < best_bits) {
      best = i;
      best_bits = b;
    }
  }
  return candidates[best];
}

// Builds the block context map and the fixed HF tokens for cparams. With
// fixed tokens, the decoder's behaviour is a function of the frame: this
// replays it on the blocks of the frame (LF and HF metadata from `tree`), so
// that only the contexts that are actually used get values from the
// HFCoefficients tree (the others are chosen to make the context map cheap),
// and unused block contexts are dropped.
bool SetHFContexts(const HFContextSettings& hf,
                   JxlMemoryManager* memory_manager, const Tree& tree,
                   size_t width, size_t height, CompressParams& cparams) {
  jxl::BlockCtxMap& map = cparams.custom_block_ctx_map;
  map = jxl::BlockCtxMap();
  const size_t lf_chan[3] = {1, 0, 2};  // X, Y, B in libjxl order
  for (size_t c = 0; c < 3; c++) {
    map.dc_thresholds[lf_chan[c]] = hf.lf_thresholds[c];
  }
  map.qf_thresholds = hf.qf_thresholds;
  size_t nlf[3];  // buckets in libjxl order X, Y, B
  map.num_dc_ctxs = 1;
  for (size_t c = 0; c < 3; c++) {
    nlf[c] = map.dc_thresholds[c].size() + 1;
    map.num_dc_ctxs *= nlf[c];
  }
  const size_t nqf = map.qf_thresholds.size() + 1;
  if (map.num_dc_ctxs * nqf > 64) {
    fprintf(stderr, "Too many LF x QF buckets (at most 64)\n");
    return false;
  }
  if (hf.custom_block_ctx) {
    map.ctx_map.resize(3 * jxl::kNumOrders * nqf * map.num_dc_ctxs);
    for (size_t c = 0; c < 3; c++) {
      for (size_t ord = 0; ord < jxl::kNumOrders; ord++) {
        for (size_t qf = 0; qf < nqf; qf++) {
          for (size_t dc = 0; dc < map.num_dc_ctxs; dc++) {
            // dc = (bucket_x * nB + bucket_b) * nY + bucket_y
            int32_t lfy = dc % nlf[1];
            int32_t lfb = (dc / nlf[1]) % nlf[2];
            int32_t lfx = dc / nlf[1] / nlf[2];
            int32_t v = hf.block_ctx.Eval(
                {static_cast<int32_t>(c), static_cast<int32_t>(ord),
                 static_cast<int32_t>(qf), lfy, lfx, lfb});
            if (v < 0 || v > 15) {
              fprintf(stderr, "Block context %d is not in 0..15\n", v);
              return false;
            }
            map.ctx_map[((c * jxl::kNumOrders + ord) * nqf + qf) *
                            map.num_dc_ctxs +
                        dc] = v;
          }
        }
      }
    }
  } else if (map.num_dc_ctxs * nqf > 1) {
    fprintf(stderr, "LF/QF thresholds need an HFBlockContext tree\n");
    return false;
  }
  cparams.use_custom_block_ctx_map = true;
  // The ids of the block contexts as the tree gives them (0..15), the ids in
  // the file are 0..n-1 without gaps (decoders reject a context map with an
  // unused value).
  const std::vector<uint8_t> tree_ctx_map = map.ctx_map;

  if (hf.coefficients.nodes.empty()) {
    // No fixed HF tokens: only renumber.
    std::vector<int32_t> new_id(16, -1);
    for (uint8_t v : tree_ctx_map) new_id[v] = 0;
    map.num_ctxs = 0;
    for (int32_t& id : new_id) {
      if (id == 0) id = map.num_ctxs++;
    }
    for (uint8_t& v : map.ctx_map) v = new_id[v];
    return true;
  }

  FrameDimensions frame_dim;
  frame_dim.Set(width, height, /*group_size_shift=*/1, /*max_hshift=*/0,
                /*max_vshift=*/0, /*modular_mode=*/false, /*upsampling=*/1);
  VarDCTBlocks blocks;
  if (!ComputeVarDCTBlocks(memory_manager, tree, frame_dim, map, &blocks)) {
    return false;
  }
  // The value of each context: the tree's value for the first case that uses
  // it (libjxl merges some nzpred values and (k, nzleft) pairs into one
  // context), in this order: predicted counts in increasing order, then
  // coefficients densest first (k + nzleft = 64, the pairs of a block with
  // all coefficients nonzero, then 63, ...). Per tree block context: 37 slots
  // for the number of nonzeros (by bucket of the predicted number), then
  // kZeroDensityContextCount for the coefficients.
  constexpr size_t kSlots =
      jxl::kNonZeroBuckets + jxl::kZeroDensityContextCount;
  std::vector<int32_t> slots(16 * kSlots, -1);
  std::vector<int32_t> props(6);
  // Token of the tree's value for `props`, or -1 after an error message.
  auto tree_token = [&]() -> int32_t {
    // Fixed tokens are below 2^15 - 1 (the largest prefix code alphabet that
    // both decoders accept).
    int32_t v = hf.coefficients.Eval(props);
    if (props[0] == 0) {
      if (v < 0 || v > 32766) {
        fprintf(stderr, "Number of nonzeros %d is not in 0..32766\n", v);
        return -1;
      }
      return v;
    }
    if (v < -16383 || v > 16383) {
      fprintf(stderr, "Coefficient %d is not in -16383..16383\n", v);
      return -1;
    }
    return jxl::PackSigned(v);
  };
  auto count_slot = [&](size_t pred) {
    return (pred >= 64 ? 64 : pred < 8 ? pred : 4 + pred / 2);
  };
  for (uint32_t bctx = 0; bctx < 16; bctx++) {
    if (std::find(tree_ctx_map.begin(), tree_ctx_map.end(), bctx) ==
        tree_ctx_map.end()) {
      continue;
    }
    int32_t* s = &slots[bctx * kSlots];
    for (int32_t nz = 0; nz <= 64; nz++) {
      props = {0, static_cast<int32_t>(bctx), nz, 0, 0, 0};
      int32_t token = tree_token();
      if (token < 0) return false;
      if (s[count_slot(nz)] < 0) s[count_slot(nz)] = token;
    }
    for (uint32_t sum = 64; sum >= 2; sum--) {
      for (uint32_t k = 1; k < sum && k < 64; k++) {
        uint32_t nzleft = sum - k;
        if (nzleft >= 64) continue;
        for (uint32_t prev = 0; prev < 2; prev++) {
          props = {1,
                   static_cast<int32_t>(bctx),
                   0,
                   static_cast<int32_t>(k),
                   static_cast<int32_t>(nzleft),
                   static_cast<int32_t>(prev)};
          int32_t token = tree_token();
          if (token < 0) return false;
          int32_t& t = s[jxl::kNonZeroBuckets +
                         jxl::ZeroDensityContext(nzleft, k, 1, 0, prev)];
          if (t < 0) t = token;
        }
      }
    }
  }

  // Replay the decoder (DecodeACVarBlock), to find the contexts that are used
  // and to check the blocks. A case that is used but has another value than
  // its context is a conflict (the context's value is what decoders use).
  std::vector<bool> reached(slots.size());
  std::vector<bool> ctx_map_used(tree_ctx_map.size());
  std::set<std::vector<int32_t>> conflicts;
  std::string first_conflict;
  const size_t xb = blocks.xsize_blocks;
  const size_t yb = blocks.ysize_blocks;
  auto slot_value = [&](size_t bctx, size_t slot, size_t bx, size_t by,
                        size_t c) -> int32_t {
    const size_t i = bctx * kSlots + slot;
    reached[i] = true;
    const int32_t s = slots[i] < 0 ? 0 : slots[i];
    if (tree_token() != s && conflicts.insert(props).second &&
        conflicts.size() == 1) {
      char buf[300];
      snprintf(buf, sizeof(buf),
               "%s at block (%zu, %zu), channel %s (bctx %d, nzpred %d, "
               "k %d, nzleft %d, prev %d): %d, but its context has %d",
               props[0] == 0 ? "count" : "coefficient", bx, by,
               c == 1   ? "Y"
               : c == 0 ? "X"
                        : "B",
               props[1], props[2], props[3], props[4], props[5],
               hf.coefficients.Eval(props),
               props[0] == 0 ? s : jxl::UnpackSigned(s));
      first_conflict = buf;
    }
    return s;
  };
  std::vector<int32_t> nz(3 * xb * yb);
  for (size_t g = 0; g < frame_dim.num_groups; g++) {
    const jxl::Rect r = frame_dim.BlockGroupRect(g);
    for (size_t by = 0; by < r.ysize(); by++) {
      for (size_t bx = 0; bx < r.xsize(); bx++) {
        const size_t x = r.x0() + bx;
        const size_t y = r.y0() + by;
        const int raw = blocks.acs[y * xb + x];
        if (raw < 0) continue;
        jxl::AcStrategy acs = jxl::AcStrategy::FromRawStrategy(raw);
        const size_t covered = acs.covered_blocks_x() * acs.covered_blocks_y();
        const size_t log2_covered = jxl::CeilLog2Nonzero(covered);
        const size_t size = covered * jxl::kDCTBlockSize;
        const size_t ord = jxl::kStrategyOrder[raw];
        const int32_t qf = blocks.qf[y * xb + x];
        const size_t lf_ctx = blocks.lf_ctx[y * xb + x];
        size_t qf_idx = 0;
        for (uint32_t t : map.qf_thresholds) qf_idx += qf > t;
        for (size_t c : {1, 0, 2}) {
          int32_t* row_nz = &nz[c * xb * yb];
          int32_t pred;
          if (bx == 0) {
            pred = by == 0 ? 32 : row_nz[(y - 1) * xb + x];
          } else if (by == 0) {
            pred = row_nz[y * xb + x - 1];
          } else {
            pred = (row_nz[(y - 1) * xb + x] + row_nz[y * xb + x - 1] + 1) / 2;
          }
          const size_t map_idx =
              (((c < 2 ? c ^ 1 : 2) * jxl::kNumOrders + ord) * nqf + qf_idx) *
                  map.num_dc_ctxs +
              lf_ctx;
          ctx_map_used[map_idx] = true;
          const size_t bctx = tree_ctx_map[map_idx];
          props = {0, static_cast<int32_t>(bctx), std::min(pred, 64), 0, 0, 0};
          const int32_t count = slot_value(bctx, count_slot(pred), x, y, c);
          if (static_cast<size_t>(count) > size - covered) {
            fprintf(stderr,
                    "Block (%zu, %zu), channel %s: %d nonzeros, but the block "
                    "has only %zu HF coefficients\n",
                    x, y,
                    c == 1   ? "Y"
                    : c == 0 ? "X"
                             : "B",
                    count, size - covered);
            return false;
          }
          for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
            for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
              row_nz[(y + iy) * xb + x + ix] =
                  (count + covered - 1) >> log2_covered;
            }
          }
          size_t left = count;
          size_t prev = (left > size / 16 ? 0 : 1);
          size_t k = covered;
          for (; k < size && left != 0; ++k) {
            props = {1,
                     static_cast<int32_t>(bctx),
                     0,
                     static_cast<int32_t>(k >> log2_covered),
                     static_cast<int32_t>((left + covered - 1) >> log2_covered),
                     static_cast<int32_t>(prev)};
            const size_t slot =
                jxl::kNonZeroBuckets +
                jxl::ZeroDensityContext(left, k, covered, log2_covered, prev);
            const int32_t token = slot_value(bctx, slot, x, y, c);
            prev = token != 0;
            left -= prev;
          }
          if (left != 0) {
            fprintf(stderr,
                    "Block (%zu, %zu), channel %s (block context %zu): %d "
                    "nonzeros, but the coefficient values give only %zu (the "
                    "decoder rejects the block)\n",
                    x, y,
                    c == 1   ? "Y"
                    : c == 0 ? "X"
                             : "B",
                    bctx, count, count - left);
            return false;
          }
        }
      }
    }
  }
  if (!conflicts.empty()) {
    fprintf(stderr,
            "Note: %zu used (nzpred) or (k, nzleft, prev) cases share a "
            "context with an earlier case that has another value (libjxl "
            "merges them); the earlier value is used. First: %s\n",
            conflicts.size(), first_conflict.c_str());
  }

  // Block contexts used by some block, renumbered without gaps; the entries
  // of the block context map that no block uses may be any of them. Block
  // contexts that no used context tells apart (the same value where both are
  // used) are merged.
  std::vector<bool> used(16);
  for (size_t i = 0; i < tree_ctx_map.size(); i++) {
    if (ctx_map_used[i]) used[tree_ctx_map[i]] = true;
  }
  std::vector<int32_t> new_id(16, -1);
  // Per merged block context: the values of its slots, -1 where not used.
  std::vector<std::vector<int32_t>> merged;
  for (size_t b = 0; b < 16; b++) {
    if (!used[b]) continue;
    std::vector<int32_t> values(kSlots, -1);
    for (size_t i = 0; i < kSlots; i++) {
      if (reached[b * kSlots + i])
        values[i] = std::max(0, slots[b * kSlots + i]);
    }
    size_t m = 0;
    for (; m < merged.size(); m++) {
      bool compatible = true;
      for (size_t i = 0; i < kSlots && compatible; i++) {
        compatible =
            values[i] < 0 || merged[m][i] < 0 || values[i] == merged[m][i];
      }
      if (compatible) break;
    }
    if (m == merged.size()) merged.emplace_back(kSlots, -1);
    for (size_t i = 0; i < kSlots; i++) {
      if (values[i] >= 0) merged[m][i] = values[i];
    }
    new_id[b] = m;
  }
  if (merged.empty()) merged.emplace_back(kSlots, -1);
  map.num_ctxs = merged.size();
  std::vector<int32_t> entries(tree_ctx_map.size());
  std::vector<int32_t> fallback(tree_ctx_map.size());
  for (size_t i = 0; i < tree_ctx_map.size(); i++) {
    entries[i] = ctx_map_used[i] ? new_id[tree_ctx_map[i]] : -1;
    fallback[i] = new_id[tree_ctx_map[i]];
  }
  map.ctx_map = FillDontCares<uint8_t>(
      entries, fallback, [&](const std::vector<uint8_t>& m) {
        return ContextMapBits(memory_manager, m);
      });

  // The fixed tokens: the used contexts get their values, the others are
  // chosen to make the context map cheap.
  const size_t num_ctxs = map.num_ctxs;
  std::vector<int32_t> tokens(map.NumACContexts(), -1);
  for (size_t m = 0; m < merged.size(); m++) {
    for (size_t i = 0; i < jxl::kNonZeroBuckets; i++) {
      tokens[i * num_ctxs + m] = merged[m][i];
    }
    for (size_t i = 0; i < jxl::kZeroDensityContextCount; i++) {
      tokens[map.ZeroDensityContextsOffset(m) + i] =
          merged[m][jxl::kNonZeroBuckets + i];
    }
  }
  std::vector<int32_t> zeros(tokens.size(), 0);
  cparams.custom_hf_tokens = FillDontCares<uint32_t>(
      tokens, zeros, [&](const std::vector<uint32_t>& t) {
        return FixedTokensBits(memory_manager, t);
      });

  // If all blocks behave the same and there are no LF/QF buckets, the default
  // block context map (1 bit) works as well as a custom one (~20 bits), with
  // the same values in each of its contexts: keep whichever is smaller.
  if (merged.size() == 1 && map.num_dc_ctxs == 1 && map.qf_thresholds.empty()) {
    jxl::BlockCtxMap default_map;
    std::vector<int32_t> default_tokens(default_map.NumACContexts(), -1);
    for (size_t m = 0; m < default_map.num_ctxs; m++) {
      for (size_t i = 0; i < jxl::kNonZeroBuckets; i++) {
        default_tokens[i * default_map.num_ctxs + m] = merged[0][i];
      }
      for (size_t i = 0; i < jxl::kZeroDensityContextCount; i++) {
        default_tokens[default_map.ZeroDensityContextsOffset(m) + i] =
            merged[0][jxl::kNonZeroBuckets + i];
      }
    }
    std::vector<int32_t> default_zeros(default_tokens.size(), 0);
    std::vector<uint32_t> default_hf_tokens = FillDontCares<uint32_t>(
        default_tokens, default_zeros, [&](const std::vector<uint32_t>& t) {
          return FixedTokensBits(memory_manager, t);
        });
    size_t custom_bits =
        BlockCtxMapBits(memory_manager, map) +
        FixedTokensBits(memory_manager, cparams.custom_hf_tokens);
    size_t default_bits = BlockCtxMapBits(memory_manager, default_map) +
                          FixedTokensBits(memory_manager, default_hf_tokens);
    if (default_bits < custom_bits) {
      map = default_map;
      cparams.custom_hf_tokens = std::move(default_hf_tokens);
    }
  }
  return true;
}

// The name of tree property `p`, as written in tree files.
std::string PropertyName(int p) {
  static const char* kNames[16] = {
      "c",           "g",      "y",    "x",    "|N|",  "|W|",  "N",    "W",
      "W-WW-NW+NWW", "W+N-NW", "W-NW", "NW-N", "N-NE", "N-NN", "W-WW", "WGH"};
  if (p < 16) return kNames[p];
  static const char* kPrev[4] = {"Abs", "", "AbsErr", "Err"};
  return "Prev" + std::to_string((p - 16) / 4 + 1) + kPrev[(p - 16) % 4];
}

// Decoders reject a tree with a split that the splits above it already
// decide (DecodeTree: "Invalid tree"). Checks this, with a message.
bool CheckTreeSplits(const Tree& tree, const char* what) {
  struct Item {
    size_t node;
    std::vector<std::pair<int64_t, int64_t>> ranges;
  };
  std::vector<Item> stack;
  stack.push_back({0, {}});
  while (!stack.empty()) {
    Item item = std::move(stack.back());
    stack.pop_back();
    const PropertyDecisionNode& node = tree[item.node];
    if (node.property < 0) continue;
    size_t p = node.property;
    if (item.ranges.size() <= p) {
      item.ranges.resize(p + 1, {std::numeric_limits<int32_t>::min(),
                                 std::numeric_limits<int32_t>::max()});
    }
    int64_t lo = item.ranges[p].first;
    int64_t hi = item.ranges[p].second;
    if (node.splitval < lo || node.splitval >= hi) {
      fprintf(stderr,
              "Impossible split in the %s: 'if %s > %d' is always %s here "
              "(the splits above give %s in %lld..%lld); decoders reject "
              "such trees\n",
              what, PropertyName(p).c_str(), node.splitval,
              node.splitval < lo ? "true" : "false", PropertyName(p).c_str(),
              static_cast<long long>(lo), static_cast<long long>(hi));
      return false;
    }
    Item then_item{node.lchild, item.ranges};
    then_item.ranges[p].first = node.splitval + 1;
    Item else_item{node.rchild, item.ranges};
    else_item.ranges[p].second = node.splitval;
    stack.push_back(std::move(else_item));
    stack.push_back(std::move(then_item));
  }
  return true;
}

// Per-frame settings besides the tree. Patches go to cparams.custom_patches.
// An explicit palette for modular frames (Palette keyword; persists).
struct PaletteSettings {
  bool enabled = false;
  uint32_t num_c = 3;
  uint32_t nb_deltas = 0;
  uint32_t nb_colors = 0;
  Predictor predictor = Predictor::Zero;
  // ImplicitPalette: no listed entries; the tree gives the meta channel.
  bool implicit = false;
  // (nb_deltas + nb_colors) entries of num_c values: the deltas, then the
  // colours.
  std::vector<std::vector<int32_t>> entries;
};

// Copies the subtree of `src` at `node` into `dst`, knowing that property
// `prop` is in lo..hi: splits that this decides are dropped. Returns the
// index of the copy's root.
size_t CopyRestricted(const Tree& src, size_t node, int prop, int64_t lo,
                      int64_t hi, Tree* dst) {
  const jxl::PropertyDecisionNode& n = src[node];
  if (n.property == prop) {
    if (n.splitval >= hi)
      return CopyRestricted(src, n.rchild, prop, lo, hi, dst);
    if (n.splitval < lo)
      return CopyRestricted(src, n.lchild, prop, lo, hi, dst);
  }
  size_t pos = dst->size();
  dst->push_back(n);
  if (n.property < 0) return pos;
  bool on_prop = n.property == prop;
  size_t then_node = CopyRestricted(
      src, n.lchild, prop, on_prop ? std::max<int64_t>(lo, n.splitval + 1) : lo,
      hi, dst);
  size_t else_node =
      CopyRestricted(src, n.rchild, prop, lo,
                     on_prop ? std::min<int64_t>(hi, n.splitval) : hi, dst);
  (*dst)[pos].lchild = then_node;
  (*dst)[pos].rchild = else_node;
  return pos;
}

// Appends a tree over `prop` (from lo) choosing values[i] for prop in
// [lo + i, lo + i + 1), merging runs of equal values; returns its root.
size_t AppendRuns(const std::vector<int32_t>& values, int prop, int64_t lo,
                  Tree* dst) {
  std::vector<std::pair<size_t, int32_t>> runs;  // (first index, value)
  for (size_t i = 0; i < values.size(); i++) {
    if (runs.empty() || runs.back().second != values[i]) {
      runs.emplace_back(i, values[i]);
    }
  }
  std::function<size_t(size_t, size_t)> build = [&](size_t a, size_t b) {
    size_t pos = dst->size();
    if (a == b) {
      dst->push_back(
          jxl::PropertyDecisionNode::Leaf(Predictor::Zero, runs[a].second));
      return pos;
    }
    size_t mid = (a + b + 1) / 2;
    dst->push_back(jxl::PropertyDecisionNode::Split(
        prop, lo + static_cast<int64_t>(runs[mid].first) - 1, 0, 0));
    size_t then_node = build(mid, b);
    size_t else_node = build(a, mid - 1);
    (*dst)[pos].lchild = then_node;
    (*dst)[pos].rchild = else_node;
    return pos;
  };
  return build(0, runs.size() - 1);
}

// Estimated cost in bits of coding `rows` (the palette meta channel: row =
// component, column = entry) with predictor `p` and zero offset, using the
// modular edge rules.
double MetaChannelCost(const std::vector<std::vector<int32_t>>& rows,
                       Predictor p) {
  double bits = 0;
  for (size_t y = 0; y < rows.size(); y++) {
    for (size_t x = 0; x < rows[y].size(); x++) {
      int64_t w = x > 0 ? rows[y][x - 1] : (y > 0 ? rows[y - 1][x] : 0);
      int64_t n = y > 0 ? rows[y - 1][x] : w;
      int64_t nw = (x > 0 && y > 0) ? rows[y - 1][x - 1] : w;
      int64_t pred = 0;
      switch (p) {
        case Predictor::Left:
          pred = w;
          break;
        case Predictor::Top:
          pred = n;
          break;
        case Predictor::Average0:
          pred = (w + n) / 2;
          break;
        case Predictor::Gradient:
          pred = std::min(std::max(w + n - nw, std::min(w, n)), std::max(w, n));
          break;
        default:
          pred = 0;
      }
      int64_t r = rows[y][x] - pred;
      bits += 1 + 2 * std::log2(1.0 + static_cast<double>(std::abs(r)));
    }
  }
  return bits;
}

// Copies `src` restricted to channels c >= 1 (the channels after the palette
// meta channel) and renumbers them for a group stream, which has no meta
// channels: c - 1. Returns the copy's root.
// `src` with the splits on `prop` resolved for prop in lo..hi (a new tree).
Tree Restricted(const Tree& src, int prop, int64_t lo, int64_t hi) {
  Tree out;
  CopyRestricted(src, 0, prop, lo, hi, &out);
  return out;
}

// Copies `src` restricted to channels c >= 1 (the channels after the palette
// meta channel) and to the group streams (g >= 1), and renumbers the channels
// for a group stream, which has no meta channels: c - 1. Returns the copy's
// root.
size_t CopyForGroupStreams(const Tree& src, Tree* dst) {
  const Tree in_groups =
      Restricted(src, /*g=*/1, 1, std::numeric_limits<int32_t>::max());
  size_t first = dst->size();
  size_t root = CopyRestricted(in_groups, 0, /*c=*/0, 1,
                               std::numeric_limits<int32_t>::max(), dst);
  for (size_t i = first; i < dst->size(); i++) {
    if ((*dst)[i].property == 0) (*dst)[i].splitval -= 1;
  }
  return root;
}

// With a palette, channel 0 is the palette meta channel (x = entry, y =
// component) and channel 1 the indices. Replaces `tree` by the frame's tree:
// - explicit entries: "if c > 0 then tree else <meta>", where <meta> is either
//   (`inline_entries`) a tree of the entries (row = component, runs over x),
//   or one predictor leaf (the cheapest of a few), the entries then being the
//   meta channel's pixels, coded as residuals of that predictor;
// - ImplicitPalette: the tree as given (it defines the meta channel too).
// If the index channel is larger than a group (`multi_group`), it is coded in
// group streams, which leave out the meta channel, so that channel numbers are
// one lower there: the root then splits on the stream (g > 0: a group stream).
void AddPaletteTree(const PaletteSettings& palette, bool multi_group,
                    bool inline_entries, Tree* tree) {
  const size_t width = palette.nb_deltas + palette.nb_colors;
  Tree out;
  if (multi_group) {
    out.push_back(jxl::PropertyDecisionNode::Split(/*g=*/1, 0, 0, 0));
    out[0].lchild = CopyForGroupStreams(*tree, &out);
  }
  // The global stream (g = 0), which holds the palette's meta channel (a
  // global transform) and, with a single group, everything.
  const Tree global_tree = Restricted(*tree, /*g=*/1, 0, 0);
  size_t global = out.size();
  if (palette.implicit || width == 0) {
    // The meta channel is the tree's (or empty).
    CopyRestricted(global_tree, 0, /*g=*/1, 0, 0, &out);
  } else {
    out.push_back(jxl::PropertyDecisionNode::Split(/*c=*/0, 0, 0, 0));
    size_t user = CopyRestricted(global_tree, 0, /*c=*/0, 1,
                                 std::numeric_limits<int32_t>::max(), &out);
    std::vector<std::vector<int32_t>> rows(palette.num_c,
                                           std::vector<int32_t>(width));
    for (size_t i = 0; i < width; i++) {
      for (size_t c = 0; c < palette.num_c; c++) {
        rows[c][i] = palette.entries[i][c];
      }
    }
    size_t meta;
    if (inline_entries) {
      std::function<size_t(size_t, size_t)> build_rows = [&](size_t a,
                                                             size_t b) {
        if (a == b) return AppendRuns(rows[a], /*x=*/3, 0, &out);
        size_t pos = out.size();
        size_t mid = (a + b + 1) / 2;
        out.push_back(jxl::PropertyDecisionNode::Split(/*y=*/2, mid - 1, 0, 0));
        size_t then_node = build_rows(mid, b);
        size_t else_node = build_rows(a, mid - 1);
        out[pos].lchild = then_node;
        out[pos].rchild = else_node;
        return pos;
      };
      meta = build_rows(0, palette.num_c - 1);
    } else {
      Predictor best = Predictor::Zero;
      double best_bits = MetaChannelCost(rows, best);
      for (Predictor p : {Predictor::Left, Predictor::Top, Predictor::Gradient,
                          Predictor::Average0}) {
        double bits = MetaChannelCost(rows, p);
        if (bits < best_bits) {
          best_bits = bits;
          best = p;
        }
      }
      meta = out.size();
      out.push_back(jxl::PropertyDecisionNode::Leaf(best, 0));
    }
    out[global].lchild = user;
    out[global].rchild = meta;
  }
  if (multi_group) out[0].rchild = global;
  *tree = std::move(out);
}

struct FrameSettings {
  // Reference slot to save this frame to; -1 = default (1 if not last).
  int save_as_reference = -1;
  // kReferenceOnly frame (not displayed; saved before the color transform, so
  // usable as a patch source).
  bool reference_only = false;
  // Save a regular frame before the color transform (needed to use it as a
  // patch source; only allowed for kReplace, full-frame blending).
  bool save_before_ct = false;
  // Reference slot this frame is blended onto (persists). Without BlendSource,
  // slot 1, unless it holds a frame saved before the color transform (see
  // DefaultBlendSource).
  size_t blend_source = 1;
  bool blend_source_given = false;
  // Patch blend mode for extra channels of subsequent patches (persists).
  uint8_t patch_ec_mode = static_cast<uint8_t>(jxl::PatchBlendMode::kNone);
  // Whether subsequent patches clamp (kMul and alpha blend modes; persists).
  bool patch_clamp = false;
  // Explicit palette of modular frames (persists).
  PaletteSettings palette;
  // HF context model of VarDCT frames (persists).
  HFContextSettings hf;
  bool have_hf = false;
  // The first keyword of this frame that only applies to VarDCT frames (empty
  // if none): an error in a modular frame. The VarDCT settings persist through
  // modular frames, which do not use them.
  std::string vardct_keyword;
  // The loop filters (Gaborish, EPF) of the other mode: Modular and VarDCT
  // swap them, so that each mode keeps its own.
  struct LoopFilters {
    jxl::Override gaborish = jxl::Override::kDefault;
    int epf = -1;
  };
  LoopFilters other_filters;
  bool have_other_filters = false;
  // Trees for the LF image and the HF metadata of a VarDCT frame (LFTree,
  // HFMetaTree), combined with the frame's tree by stream ID.
  Tree lf_tree;
  Tree hf_meta_tree;
  // ExtraTree: the tree for the extra channels of a VarDCT frame.
  Tree extra_tree;
  // Trees for RAW dequantization tables of a VarDCT frame (DequantTable), by
  // quantization table index.
  std::map<size_t, Tree> dequant_trees;
  // ACSTree and QFTree: the AC strategy and quant field of a varblock, from
  // its position and LF.
  NamedTree acs_tree;
  NamedTree qf_tree;
  // Whether GroupShift was given (it does not apply to VarDCT frames).
  bool group_shift_given = false;
  // Image (canvas) size set by ImageSize; 0 = the first frame's size.
  size_t image_xsize = 0;
  size_t image_ysize = 0;
};

Status SplinesFromSplineData(const SplineData& spline_data,
                             std::vector<QuantizedSpline>& quantized_splines,
                             std::vector<Spline::Point>& starting_points) {
  quantized_splines.clear();
  starting_points.clear();
  quantized_splines.reserve(spline_data.splines.size());
  starting_points.reserve(spline_data.splines.size());
  for (const Spline& spline : spline_data.splines) {
    JXL_ASSIGN_OR_RETURN(
        QuantizedSpline qspline,
        QuantizedSpline::Create(spline, spline_data.quantization_adjustment,
                                0.0, 1.0));
    quantized_splines.emplace_back(std::move(qspline));
    starting_points.push_back(spline.control_points.front());
  }
  return true;
}

// Quantization tables (lib/jxl/quant_weights.h) by number (0..16), name, or
// "all" (if allowed). Returns an empty list for an invalid name.
std::vector<size_t> ParseQuantTables(std::string name, bool allow_all) {
  static const std::unordered_map<std::string, size_t> table_names = {
      {"DCT8", 0},        {"DCT", 0},     {"IDENTITY", 1},
      {"DCT2x2", 2},      {"DCT4x4", 3},  {"DCT16", 4},
      {"DCT32", 5},       {"DCT16x8", 6}, {"DCT8x16", 6},
      {"DCT32x8", 7},     {"DCT8x32", 7}, {"DCT32x16", 8},
      {"DCT16x32", 8},    {"DCT4x8", 9},  {"DCT8x4", 9},
      {"AFV", 10},        {"DCT64", 11},  {"DCT64x32", 12},
      {"DCT32x64", 12},   {"DCT128", 13}, {"DCT128x64", 14},
      {"DCT64x128", 14},  {"DCT256", 15}, {"DCT256x128", 16},
      {"DCT128x256", 16},
  };
  std::vector<size_t> tables;
  if (allow_all && name == "all") {
    for (size_t i = 0; i < jxl::kNumQuantTables; i++) tables.push_back(i);
    return tables;
  }
  if (name.compare(0, 3, "DCT") == 0) {
    std::replace(name.begin() + 3, name.end(), 'X', 'x');
  }
  if (table_names.count(name)) return {table_names.at(name)};
  size_t num = 0;
  size_t idx = 0;
  if (!name.empty() && isdigit(name[0])) idx = ParseUnsigned(name, &num);
  if (num == 0 || num != name.size() || idx >= jxl::kNumQuantTables) return {};
  return {idx};
}

// Appends a copy of `src` to `dst`, returns the index of its root.
size_t AppendTree(const Tree& src, Tree* dst) {
  size_t offset = dst->size();
  for (jxl::PropertyDecisionNode node : src) {
    if (node.property >= 0) {
      node.lchild += offset;
      node.rchild += offset;
    }
    dst->push_back(node);
  }
  return offset;
}

// The extra channels (alpha, hidden channels) of a VarDCT frame, for the
// stream IDs of their modular streams.
struct ExtraChannelLayout {
  bool present = false;
  // Upsampling of the frame and of the extra channels (Upsample, Upsample_EC).
  size_t upsampling = 1;
  size_t ec_upsampling = 1;
  // Squeeze: the channels are split over all kinds of streams.
  bool responsive = false;
};

// The ranges [first, last] of the stream IDs that hold extra channels in a
// VarDCT frame with n DC groups, as the decoders split them: channels of at
// most 256x256 are in the global stream (0), others in the modular LF streams
// (n+1..2n) if they are downsampled 8x or more, else in the modular group
// streams (after the quantization tables). With Squeeze, all of these.
std::vector<std::pair<size_t, size_t>> ExtraChannelStreams(
    const ExtraChannelLayout& ec, size_t width, size_t height,
    const FrameDimensions& frame_dim) {
  const size_t n = frame_dim.num_dc_groups;
  const size_t groups_first = 1 + 3 * n + jxl::kNumQuantTables;
  const size_t groups_last = std::numeric_limits<int32_t>::max();
  if (!ec.present) return {};
  if (ec.responsive) {
    return {{0, 0}, {n + 1, 2 * n}, {groups_first, groups_last}};
  }
  const size_t ups = std::max(ec.ec_upsampling, ec.upsampling);
  const size_t xsize = jxl::DivCeil(width * ec.upsampling, ups);
  const size_t ysize = jxl::DivCeil(height * ec.upsampling, ups);
  if (xsize <= jxl::kGroupDim && ysize <= jxl::kGroupDim) return {{0, 0}};
  const size_t shift =
      jxl::CeilLog2Nonzero(ups) - jxl::CeilLog2Nonzero(ec.upsampling);
  if (shift >= 3) return {{n + 1, 2 * n}};
  return {{groups_first, groups_last}};
}

// Replaces `tree` by a tree that splits on the stream ID (property 1) of a
// VarDCT frame of the given size: frame.lf_tree (if not empty) for the LF
// image (VarDCT DC streams 1..n, for n DC groups), frame.hf_meta_tree (if not
// empty) for the HF metadata (AC metadata streams 2n+1..3n), the trees of
// frame.dequant_trees for the streams of their quantization tables (3n+1 +
// the table index), frame.extra_tree (if not empty) for the streams of the
// extra channels, and `tree` for all other streams.
void CombineVarDCTTrees(const FrameSettings& frame, size_t width, size_t height,
                        const ExtraChannelLayout& ec, Tree* tree) {
  if (frame.lf_tree.empty() && frame.hf_meta_tree.empty() &&
      frame.dequant_trees.empty() && frame.extra_tree.empty()) {
    return;
  }
  FrameDimensions frame_dim;
  frame_dim.Set(width, height, /*group_size_shift=*/1, /*max_hshift=*/0,
                /*max_vshift=*/0, /*modular_mode=*/false, /*upsampling=*/1);
  const int n = frame_dim.num_dc_groups;
  const Tree rest = *tree;
  Tree& out = *tree;
  out.clear();
  if (ec.present) {
    // The streams that are used, in order, each with its tree; the streams in
    // between are empty, so any tree will do there. One split between
    // consecutive ranges with different trees (balanced).
    struct Range {
      size_t first, last;
      const Tree* tree;
    };
    std::vector<Range> ranges;
    const Tree* lf = frame.lf_tree.empty() ? &rest : &frame.lf_tree;
    const Tree* hf = frame.hf_meta_tree.empty() ? &rest : &frame.hf_meta_tree;
    const Tree* extra = frame.extra_tree.empty() ? &rest : &frame.extra_tree;
    ranges.push_back({1, static_cast<size_t>(n), lf});
    ranges.push_back(
        {2 * static_cast<size_t>(n) + 1, 3 * static_cast<size_t>(n), hf});
    for (const auto& table : frame.dequant_trees) {
      const size_t id = 1 + 3 * n + table.first;
      ranges.push_back({id, id, &table.second});
    }
    for (const auto& r : ExtraChannelStreams(ec, width, height, frame_dim)) {
      ranges.push_back({r.first, r.second, extra});
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const Range& a, const Range& b) { return a.first < b.first; });
    std::vector<Range> merged;
    for (const Range& r : ranges) {
      if (!merged.empty() && merged.back().tree == r.tree) {
        merged.back().last = r.last;
      } else {
        merged.push_back(r);
      }
    }
    std::function<size_t(size_t, size_t)> build = [&](size_t lo, size_t hi) {
      if (hi - lo == 1) return AppendTree(*merged[lo].tree, &out);
      const size_t mid = (lo + hi) / 2;
      const size_t pos = out.size();
      // if g > (the last stream of the lower half)
      out.push_back(jxl::PropertyDecisionNode::Split(
          1, static_cast<int>(merged[mid - 1].last), 0, 0));
      const size_t then_node = build(mid, hi);
      const size_t else_node = build(lo, mid);
      out[pos].lchild = then_node;
      out[pos].rchild = else_node;
      return pos;
    };
    build(0, merged.size());
    return;
  }
  // Without extra channels: only the LF, HF metadata and quantization table
  // streams.
  // "if g > value" (then: lchild, else: rchild), children set below.
  auto split = [&](int value) {
    out.push_back(jxl::PropertyDecisionNode::Split(1, value, 0, 0));
    return out.size() - 1;
  };
  auto subtree = [&](const Tree& t) {
    return AppendTree(t.empty() ? rest : t, &out);
  };
  auto set_children = [&](size_t node, size_t then_node, size_t else_node) {
    out[node].lchild = then_node;
    out[node].rchild = else_node;
  };
  // The streams up to the HF metadata (1..3n).
  auto vardct_streams = [&]() -> size_t {
    if (frame.lf_tree.empty() && frame.hf_meta_tree.empty()) {
      return subtree(rest);
    }
    // if g > n: HF metadata, else LF
    size_t root = split(n);
    size_t hf = subtree(frame.hf_meta_tree);
    size_t lf = subtree(frame.lf_tree);
    set_children(root, hf, lf);
    return root;
  };
  if (frame.dequant_trees.empty()) {
    vardct_streams();
    return;
  }
  // if g > 3n: the quantization tables (if g > 3n + 1 + table: the next
  // table ...); else the streams up to the HF metadata.
  size_t root = split(3 * n);
  std::vector<std::pair<size_t, const Tree*>> tables;
  for (const auto& table : frame.dequant_trees) {
    tables.emplace_back(table.first, &table.second);
  }
  size_t tables_root = 0;
  size_t parent = 0;
  for (size_t i = 0; i < tables.size(); i++) {
    size_t node;
    if (i + 1 == tables.size()) {
      node = subtree(*tables[i].second);
    } else {
      node = split(3 * n + 1 + tables[i].first);
      size_t table = subtree(*tables[i].second);
      out[node].rchild = table;
    }
    if (i == 0) {
      tables_root = node;
    } else {
      out[parent].lchild = node;
    }
    parent = node;
  }
  size_t low = vardct_streams();
  set_children(root, tables_root, low);
}

// Appends to `dst` the subtree of `src` at `node`, without the splits on c
// (property 0) or g (property 1) that the ranges [c_lo, c_hi] and [g_lo, g_hi]
// decide; if `c2_tree` is not null, the part for c = 2 is replaced by it.
// Returns the index of the root.
size_t SpecializeTree(const Tree& src, size_t node, int c_lo, int c_hi,
                      int g_lo, int g_hi, const Tree* c2_tree, Tree* dst) {
  if (c2_tree && c_lo == 2 && c_hi == 2) return AppendTree(*c2_tree, dst);
  const jxl::PropertyDecisionNode& n = src[node];
  auto split = [&](int property, int value, int then_lo, int then_hi,
                   int else_lo, int else_hi, size_t then_node,
                   size_t else_node) {
    size_t pos = dst->size();
    dst->push_back(jxl::PropertyDecisionNode::Split(property, value, 0, 0));
    size_t t = property == 0 ? SpecializeTree(src, then_node, then_lo, then_hi,
                                              g_lo, g_hi, c2_tree, dst)
                             : SpecializeTree(src, then_node, c_lo, c_hi,
                                              then_lo, then_hi, c2_tree, dst);
    size_t e = property == 0 ? SpecializeTree(src, else_node, else_lo, else_hi,
                                              g_lo, g_hi, c2_tree, dst)
                             : SpecializeTree(src, else_node, c_lo, c_hi,
                                              else_lo, else_hi, c2_tree, dst);
    (*dst)[pos].lchild = t;
    (*dst)[pos].rchild = e;
    return pos;
  };
  if (n.property < 0) {
    if (c2_tree && c_lo <= 2 && c_hi >= 2) {
      // A leaf for c = 2 and other channels: split off c = 2.
      if (c_lo < 2) return split(0, 1, 2, c_hi, c_lo, 1, node, node);
      return split(0, 2, 3, c_hi, 2, 2, node, node);
    }
    dst->push_back(n);
    return dst->size() - 1;
  }
  if (n.property == 0 || n.property == 1) {
    const int lo = n.property == 0 ? c_lo : g_lo;
    const int hi = n.property == 0 ? c_hi : g_hi;
    const int v = n.splitval;
    if (lo > v) {
      return SpecializeTree(src, n.lchild, c_lo, c_hi, g_lo, g_hi, c2_tree,
                            dst);
    }
    if (hi <= v) {
      return SpecializeTree(src, n.rchild, c_lo, c_hi, g_lo, g_hi, c2_tree,
                            dst);
    }
    return split(n.property, v, v + 1, hi, lo, v, n.lchild, n.rchild);
  }
  size_t pos = dst->size();
  dst->push_back(n);
  size_t t =
      SpecializeTree(src, n.lchild, c_lo, c_hi, g_lo, g_hi, c2_tree, dst);
  size_t e =
      SpecializeTree(src, n.rchild, c_lo, c_hi, g_lo, g_hi, c2_tree, dst);
  (*dst)[pos].lchild = t;
  (*dst)[pos].rchild = e;
  return pos;
}

// Appends a tree over x (property 3) that gives values[x], as one leaf per
// run of equal values (a balanced tree of splits at the run starts).
size_t AppendRunsTree(const std::vector<int32_t>& values, Tree* dst) {
  std::vector<std::pair<size_t, int32_t>> runs;  // start, value
  for (size_t x = 0; x < values.size(); x++) {
    if (runs.empty() || values[x] != runs.back().second) {
      runs.emplace_back(x, values[x]);
    }
  }
  if (runs.empty()) runs.emplace_back(0, 0);
  std::function<size_t(size_t, size_t)> build = [&](size_t lo, size_t hi) {
    if (hi - lo == 1) {
      dst->push_back(
          jxl::PropertyDecisionNode::Leaf(Predictor::Zero, runs[lo].second));
      return dst->size() - 1;
    }
    size_t mid = (lo + hi) / 2;
    size_t pos = dst->size();
    dst->push_back(jxl::PropertyDecisionNode::Split(
        3, static_cast<int>(runs[mid].first) - 1, 0, 0));
    size_t t = build(mid, hi);
    size_t e = build(lo, mid);
    (*dst)[pos].lchild = t;
    (*dst)[pos].rchild = e;
    return pos;
  };
  return build(0, runs.size());
}

// ACSTree and QFTree: evaluates them per varblock (with the LF image and the
// HF metadata of the frame's trees), and replaces channel 2 of the HF
// metadata (the AC strategy and quant field lists, in placement order) by a
// tree that gives these lists, in frame.hf_meta_tree.
bool BuildHFMetaLists(JxlMemoryManager* memory_manager, FrameSettings& frame,
                      const Tree& rest, size_t width, size_t height,
                      const ExtraChannelLayout& extra_channels) {
  if (frame.acs_tree.nodes.empty() && frame.qf_tree.nodes.empty()) {
    return true;
  }
  Tree tree = rest;
  CombineVarDCTTrees(frame, width, height, extra_channels, &tree);
  FrameDimensions frame_dim;
  frame_dim.Set(width, height, /*group_size_shift=*/1, /*max_hshift=*/0,
                /*max_vshift=*/0, /*modular_mode=*/false, /*upsampling=*/1);
  const size_t xb = frame_dim.xsize_blocks;
  const size_t yb = frame_dim.ysize_blocks;
  const int n = frame_dim.num_dc_groups;
  std::vector<bool> covered(xb * yb);
  Tree lists;  // channel 2 of the HF metadata streams
  std::vector<size_t> group_roots;
  for (int g = 0; g < n; g++) {
    const jxl::Rect r = frame_dim.DCGroupRect(g);
    auto lf = jxl::Image::Create(memory_manager, r.xsize(), r.ysize(), 8, 3);
    auto meta = jxl::Image::Create(memory_manager, r.xsize(), r.ysize(), 8, 4);
    if (!lf.ok() || !meta.ok()) return false;
    jxl::Image lf_image = std::move(lf).value_();
    jxl::Image meta_image = std::move(meta).value_();
    for (size_t c = 0; c < 3; c++) {
      auto ch =
          c < 2
              ? jxl::Channel::Create(memory_manager, (r.xsize() + 7) >> 3,
                                     (r.ysize() + 7) >> 3)
              : jxl::Channel::Create(memory_manager, r.xsize() * r.ysize(), 2);
      if (!ch.ok()) return false;
      meta_image.channel[c] = std::move(ch).value_();
    }
    if (!jxl::EvaluateTreeWithZeroResiduals(tree, 1 + g, &lf_image) ||
        !jxl::EvaluateTreeWithZeroResiduals(tree, 1 + 2 * n + g, &meta_image)) {
      fprintf(stderr, "Could not evaluate the LF or HF metadata tree\n");
      return false;
    }
    std::vector<int32_t> acs_list(r.xsize() * r.ysize());
    std::vector<int32_t> qf_list(r.xsize() * r.ysize());
    const int32_t* acs_in = meta_image.channel[2].Row(0);
    const int32_t* qf_in = meta_image.channel[2].Row(1);
    size_t num = 0;
    for (size_t iy = 0; iy < r.ysize(); iy++) {
      for (size_t ix = 0; ix < r.xsize(); ix++) {
        const size_t x = r.x0() + ix;
        const size_t y = r.y0() + iy;
        if (covered[y * xb + x]) continue;
        std::vector<int32_t> props = {
            static_cast<int32_t>(x), static_cast<int32_t>(y),
            lf_image.channel[0].Row(iy)[ix], lf_image.channel[1].Row(iy)[ix],
            lf_image.channel[2].Row(iy)[ix]};
        int32_t raw = frame.acs_tree.nodes.empty() ? acs_in[num]
                                                   : frame.acs_tree.Eval(props);
        if (!jxl::AcStrategy::IsRawStrategyValid(raw)) {
          fprintf(stderr,
                  "AC strategy %d at block (%zu, %zu) is not in 0..26\n", raw,
                  x, y);
          return false;
        }
        jxl::AcStrategy acs = jxl::AcStrategy::FromRawStrategy(raw);
        const size_t group = jxl::kGroupDimInBlocks;
        if (x + acs.covered_blocks_x() >
                std::min((x / group + 1) * group, xb) ||
            y + acs.covered_blocks_y() >
                std::min((y / group + 1) * group, yb)) {
          fprintf(stderr,
                  "AC strategy %d at block (%zu, %zu) crosses a group or "
                  "image edge\n",
                  raw, x, y);
          return false;
        }
        for (size_t cy = 0; cy < acs.covered_blocks_y(); cy++) {
          for (size_t cx = 0; cx < acs.covered_blocks_x(); cx++) {
            if (covered[(y + cy) * xb + x + cx]) {
              fprintf(stderr,
                      "AC strategy %d at block (%zu, %zu) overlaps an earlier "
                      "block\n",
                      raw, x, y);
              return false;
            }
            covered[(y + cy) * xb + x + cx] = true;
          }
        }
        props.push_back(raw);
        int32_t qf = frame.qf_tree.nodes.empty() ? qf_in[num]
                                                 : frame.qf_tree.Eval(props);
        if (!frame.qf_tree.nodes.empty() && (qf < 0 || qf > 255)) {
          fprintf(stderr,
                  "Quant field %d at block (%zu, %zu) is not in 0..255\n", qf,
                  x, y);
          return false;
        }
        acs_list[num] = raw;
        qf_list[num] = qf;
        num++;
      }
    }
    // The entries after the last varblock are not used: extend the last run.
    for (size_t i = num; i < acs_list.size() && num > 0; i++) {
      acs_list[i] = acs_list[num - 1];
      qf_list[i] = qf_list[num - 1];
    }
    // if y > 0: quant field, else AC strategy
    size_t root = lists.size();
    lists.push_back(jxl::PropertyDecisionNode::Split(2, 0, 0, 0));
    // The quant field as a function of the AC strategy (property N: the
    // entry above), if it is one and that takes fewer leaves than the runs.
    std::map<int32_t, int32_t> qf_of_acs;
    bool is_function = true;
    for (size_t i = 0; i < num && is_function; i++) {
      auto it = qf_of_acs.emplace(acs_list[i], qf_list[i]).first;
      is_function = it->second == qf_list[i];
    }
    size_t qf_runs = 0;
    for (size_t i = 0; i < qf_list.size(); i++) {
      qf_runs += i == 0 || qf_list[i] != qf_list[i - 1];
    }
    size_t qf_root;
    if (is_function && num > 0 && qf_of_acs.size() < qf_runs) {
      std::vector<int32_t> acs_values;
      std::vector<int32_t> qf_values;
      for (const auto& e : qf_of_acs) {
        acs_values.push_back(e.first);
        qf_values.push_back(e.second);
      }
      std::function<size_t(size_t, size_t)> build = [&](size_t lo, size_t hi) {
        if (hi - lo == 1) {
          lists.push_back(
              jxl::PropertyDecisionNode::Leaf(Predictor::Zero, qf_values[lo]));
          return lists.size() - 1;
        }
        size_t mid = (lo + hi) / 2;
        size_t pos = lists.size();
        // Property 6: N.
        lists.push_back(
            jxl::PropertyDecisionNode::Split(6, acs_values[mid] - 1, 0, 0));
        size_t t = build(mid, hi);
        size_t e = build(lo, mid);
        lists[pos].lchild = t;
        lists[pos].rchild = e;
        return pos;
      };
      qf_root = build(0, acs_values.size());
    } else {
      qf_root = AppendRunsTree(qf_list, &lists);
    }
    size_t acs_root = AppendRunsTree(acs_list, &lists);
    lists[root].lchild = qf_root;
    lists[root].rchild = acs_root;
    group_roots.push_back(root);
  }
  // Per DC group: if g > 2n + 1: (if g > 2n + 2: ...) else group 0.
  Tree c2_tree;
  std::function<size_t(int)> chain = [&](int g) -> size_t {
    const size_t begin = group_roots[g];
    const size_t end = g + 1 < n ? group_roots[g + 1] : lists.size();
    // The nodes of group g are lists[begin, end), with children in range.
    auto copy_group = [&]() {
      size_t offset = c2_tree.size();
      for (size_t i = begin; i < end; i++) {
        jxl::PropertyDecisionNode node = lists[i];
        if (node.property >= 0) {
          node.lchild = node.lchild - begin + offset;
          node.rchild = node.rchild - begin + offset;
        }
        c2_tree.push_back(node);
      }
      return offset;
    };
    if (g + 1 == n) return copy_group();
    size_t pos = c2_tree.size();
    c2_tree.push_back(jxl::PropertyDecisionNode::Split(1, 2 * n + 1 + g, 0, 0));
    size_t next = chain(g + 1);
    size_t this_group = copy_group();
    c2_tree[pos].lchild = next;
    c2_tree[pos].rchild = this_group;
    return pos;
  };
  chain(0);
  const Tree& hf_part = frame.hf_meta_tree.empty() ? rest : frame.hf_meta_tree;
  Tree hf_meta;
  SpecializeTree(hf_part, 0, 0, 3, 2 * n + 1, 3 * n, &c2_tree, &hf_meta);
  if (frame.lf_tree.empty()) {
    // Without the HF metadata part of the frame's tree.
    SpecializeTree(rest, 0, 0, 2, 1, n, nullptr, &frame.lf_tree);
  }
  frame.hf_meta_tree = std::move(hf_meta);
  frame.acs_tree.nodes.clear();
  frame.qf_tree.nodes.clear();
  return true;
}

// Checks the RAW dequantization tables that the tree gives (as
// ComputeVarDCTDataFromTree will), with messages for the tree author.
Status CheckDequantTables(JxlMemoryManager* memory_manager, const Tree& tree,
                          size_t width, size_t height,
                          const CompressParams& cparams) {
  FrameDimensions frame_dim;
  frame_dim.Set(width, height, /*group_size_shift=*/1, /*max_hshift=*/0,
                /*max_vshift=*/0, /*modular_mode=*/false, /*upsampling=*/1);
  for (size_t idx = 0; idx < cparams.vardct_dequant.size(); idx++) {
    const float den = cparams.vardct_dequant[idx].raw_den;
    if (den == 0) continue;
    const size_t xsize = jxl::DequantMatrices::required_size_x[idx] * 8;
    const size_t ysize = jxl::DequantMatrices::required_size_y[idx] * 8;
    JXL_ASSIGN_OR_RETURN(
        jxl::Image image,
        jxl::Image::Create(memory_manager, xsize, ysize, 8, 3));
    const size_t stream_id =
        1 + 3 * frame_dim.num_dc_groups + idx;  // ModularStreamId::QuantTable
    if (!jxl::EvaluateTreeWithZeroResiduals(tree, stream_id, &image)) {
      fprintf(stderr, "Could not evaluate the tree of DequantTable %zu\n", idx);
      return false;
    }
    for (size_t c = 0; c < 3; c++) {
      for (size_t y = 0; y < ysize; y++) {
        const int32_t* row = image.channel[c].Row(y);
        for (size_t x = 0; x < xsize; x++) {
          if (row[x] <= 0 || static_cast<double>(den) * row[x] > 1e8) {
            fprintf(stderr,
                    "DequantTable %zu: value %d at c = %zu, x = %zu, y = %zu; "
                    "values must be positive (and den * value at most 1e8)\n",
                    idx, row[x], c, x, y);
            return false;
          }
        }
      }
    }
  }
  return true;
}

template <typename F>
bool ParseNode(F& tok, Tree& tree, SplineData& spline_data,
               FrameSettings& frame, CompressParams& cparams, size_t& W,
               size_t& H, CodecInOut& io, JXL_BOOL& have_next, int& x0, int& y0,
               int& buffer_size) {
  std::unordered_map<std::string, int> property_map = {
      {"c", 0},
      {"g", 1},
      {"y", 2},
      {"x", 3},
      {"|N|", 4},
      {"|W|", 5},
      {"N", 6},
      {"W", 7},
      {"W-WW-NW+NWW", 8},
      {"W+N-NW", 9},
      {"W-NW", 10},
      {"NW-N", 11},
      {"N-NE", 12},
      {"N-NN", 13},
      {"W-WW", 14},
      {"WGH", 15},
      {"PrevAbs", 16},
      {"Prev", 17},
      {"PrevAbsErr", 18},
      {"PrevErr", 19},
      {"PPrevAbs", 20},
      {"PPrev", 21},
      {"PPrevAbsErr", 22},
      {"PPrevErr", 23},
      {"Prev1Abs", 16},
      {"Prev1", 17},
      {"Prev1AbsErr", 18},
      {"Prev1Err", 19},
  };
  for (size_t i = 0; i < 19; i++) {
    std::string name_prefix = "Prev" + std::to_string(i + 1);
    property_map[name_prefix + "Abs"] = i * 4 + 16;
    property_map[name_prefix] = i * 4 + 17;
    property_map[name_prefix + "AbsErr"] = i * 4 + 18;
    property_map[name_prefix + "Err"] = i * 4 + 19;
  }
  static const std::unordered_map<std::string, Predictor> predictor_map = {
      {"Set", Predictor::Zero},
      {"Zero", Predictor::Zero},
      {"W", Predictor::Left},
      {"N", Predictor::Top},
      {"AvgW+N", Predictor::Average0},
      {"Select", Predictor::Select},
      {"Gradient", Predictor::Gradient},
      {"Weighted", Predictor::Weighted},
      {"NE", Predictor::TopRight},
      {"NW", Predictor::TopLeft},
      {"WW", Predictor::LeftLeft},
      {"AvgW+NW", Predictor::Average1},
      {"AvgN+NW", Predictor::Average2},
      {"AvgN+NE", Predictor::Average3},
      {"AvgAll", Predictor::Average4},
  };
  auto t = tok();
  static const std::set<std::string> vardct_keywords = {
      "HFContextLF",    "HFContextQF", "HFBlockContext", "HFCoefficients",
      "GlobalScale",    "LFQuant",     "XQMScale",       "BQMScale",
      "LFChannelQuant", "DequantFlat", "DequantBands",   "DequantDefault",
      "CoeffOrder",     "LFCfL"};
  if (frame.vardct_keyword.empty() && vardct_keywords.count(t)) {
    frame.vardct_keyword = t;
  }
  if (t == "if") {
    // Decision node.
    int p;
    t = tok();
    if (!property_map.count(t)) {
      fprintf(stderr, "Unexpected property: %s\n", t.c_str());
      return false;
    }
    p = property_map.at(t);
    t = tok();
    if (t != ">") {
      fprintf(stderr, "Expected >, found %s\n", t.c_str());
      return false;
    }
    t = tok();
    size_t num = 0;
    int split = ParseInt(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid splitval: %s\n", t.c_str());
      return false;
    }
    size_t pos = tree.size();
    tree.emplace_back(PropertyDecisionNode::Split(p, split, pos + 1));
    JXL_RETURN_IF_ERROR(ParseNode(tok, tree, spline_data, frame, cparams, W, H,
                                  io, have_next, x0, y0, buffer_size));
    tree[pos].rchild = tree.size();
  } else if (t == "-") {
    // Leaf
    t = tok();
    Predictor p;
    if (!predictor_map.count(t)) {
      fprintf(stderr, "Unexpected predictor: %s\n", t.c_str());
      return false;
    }
    p = predictor_map.at(t);
    t = tok();
    bool subtract = false;
    if (t == "-") {
      subtract = true;
      t = tok();
    } else if (t == "+") {
      t = tok();
    }
    size_t num = 0;
    int offset = ParseInt(t, &num);
    // An optional multiplier for the residuals of this leaf: offset*m
    // (the sample is prediction + offset + m * residual).
    uint32_t multiplier = 1;
    if (num < t.size() && t[num] == '*') {
      const std::string m = t.substr(num + 1);
      size_t mnum = 0;
      const uint64_t mv = ParseUnsigned(m, &mnum);
      if (mnum != m.size() || mnum == 0 || mv < 1 || mv > (uint64_t{1} << 31)) {
        fprintf(stderr, "Invalid leaf multiplier (1..2^31): %s\n", t.c_str());
        return false;
      }
      multiplier = static_cast<uint32_t>(mv);
      num = t.size();
    }
    if (num != t.size()) {
      fprintf(stderr, "Invalid offset: %s\n", t.c_str());
      return false;
    }
    if (subtract) offset = -offset;
    tree.emplace_back(PropertyDecisionNode::Leaf(p, offset, multiplier));
    return true;
  } else if (t == "Width") {
    t = tok();
    size_t num = 0;
    W = ParseUnsigned(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid width: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Height") {
    t = tok();
    size_t num = 0;
    H = ParseUnsigned(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid height: %s\n", t.c_str());
      return false;
    }
  } else if (t == "/*") {
    t = tok();
    while (t != "*/" && t != "") t = tok();
  } else if (t == "WPParams" || t == "WPMode") {
    // The weighted predictor's header: WPParams p1C p2C p3Ca p3Cb p3Cc p3Cd
    // p3Ce (0..31) w0 w1 w2 w3 (0..15), or one of libjxl's presets (WPMode
    // 0..4; 0 is the default). They shape what Weighted predicts and WGH.
    const bool params = t == "WPParams";
    const size_t n = params ? 11 : 1;
    for (size_t i = 0; i < n; i++) {
      t = tok();
      size_t num = 0;
      uint64_t v = ParseUnsigned(t, &num);
      const uint64_t max = params ? (i < 7 ? 31 : 15) : 4;
      if (num != t.size() || v > max) {
        fprintf(stderr, "Invalid %s value (0..%u): %s\n",
                params ? "WPParams" : "WPMode", static_cast<unsigned>(max),
                t.c_str());
        return false;
      }
      if (params) {
        cparams.options.wp_params[i] = v;
      } else {
        cparams.options.wp_mode = v;
      }
    }
    cparams.options.has_wp_params = params;
  } else if (t == "Squeeze") {
    cparams.responsive = true;
  } else if (t == "GroupShift") {
    t = tok();
    size_t num = 0;
    cparams.modular_group_size_shift = ParseUnsigned(t, &num);
    frame.group_shift_given = true;
    if (num != t.size()) {
      fprintf(stderr, "Invalid GroupShift: %s\n", t.c_str());
      return false;
    }
  } else if (t == "XYB") {
    cparams.color_transform = ColorTransform::kXYB;
  } else if (t == "CbYCr") {
    cparams.color_transform = ColorTransform::kYCbCr;
  } else if (t == "HFContextLF" || t == "HFContextQF") {
    // HFContextLF <n> <thresholds> (for Y, X and B), HFContextQF <n>
    // <thresholds>: buckets of the quantized LF and quant field for the block
    // context of HF coefficients.
    size_t lists = t == "HFContextLF" ? 3 : 1;
    for (size_t c = 0; c < lists; c++) {
      t = tok();
      size_t num = 0;
      size_t n = ParseUnsigned(t, &num);
      if (num != t.size() || n > 15) {
        fprintf(stderr, "Invalid number of thresholds (max 15): %s\n",
                t.c_str());
        return false;
      }
      std::vector<int> v(n);
      for (int& i : v) {
        t = tok();
        i = ParseInt(t, &num);
        if (num != t.size()) {
          fprintf(stderr, "Invalid threshold: %s\n", t.c_str());
          return false;
        }
      }
      if (lists == 3) {
        frame.hf.lf_thresholds[c] = v;
      } else {
        // The quant field of a block is 1..256 (QFTree / HF metadata value
        // + 1); thresholds are coded as t - 1.
        for (int q : v) {
          if (q < 1 || q > 255) {
            fprintf(stderr,
                    "Invalid HFContextQF threshold %d: must be 1..255 (a "
                    "block's quant field is its QFTree value + 1, 1..256)\n",
                    q);
            return false;
          }
        }
        frame.hf.qf_thresholds.assign(v.begin(), v.end());
      }
    }
    frame.have_hf = true;
  } else if (t == "HFBlockContext") {
    frame.hf.block_ctx.nodes.clear();
    if (!ParseNamedTree(tok, {"c", "ord", "qf", "lfy", "lfx", "lfb"},
                        &frame.hf.block_ctx)) {
      return false;
    }
    frame.hf.custom_block_ctx = true;
    frame.have_hf = true;
  } else if (t == "HFCoefficients") {
    frame.hf.coefficients.nodes.clear();
    if (!ParseNamedTree(tok,
                        {"hfkind", "bctx", "nzpred", "k", "nzleft", "prev"},
                        &frame.hf.coefficients)) {
      return false;
    }
    frame.have_hf = true;
  } else if (t == "LFTree" || t == "HFMetaTree" || t == "ExtraTree") {
    // LFTree <tree>, HFMetaTree <tree>, ExtraTree <tree>: the trees for the
    // LF image, the HF metadata and the extra channels (alpha, hidden
    // channels) of a VarDCT frame, without splitting on the stream ID; the
    // frame's tree is used for the other streams (and for these, if not
    // given).
    Tree& subtree = t == "LFTree"       ? frame.lf_tree
                    : t == "HFMetaTree" ? frame.hf_meta_tree
                                        : frame.extra_tree;
    subtree.clear();
    JXL_RETURN_IF_ERROR(ParseNode(tok, subtree, spline_data, frame, cparams, W,
                                  H, io, have_next, x0, y0, buffer_size));
  } else if (t == "GlobalScale" || t == "LFQuant") {
    // GlobalScale <1..73728>, LFQuant <1..65536>: the quantizer of VarDCT
    // frames (defaults 1024 and 64). The LF step is 65536 / GlobalScale /
    // LFQuant / LFChannelQuant, the HF step 65536 / GlobalScale / quant field
    // times the dequantization matrix.
    bool global_scale = t == "GlobalScale";
    t = tok();
    size_t num = 0;
    size_t v = ParseUnsigned(t, &num);
    if (num != t.size() || v < 1 || v > (global_scale ? 73728 : 65536)) {
      fprintf(stderr, "Invalid %s: %s\n",
              global_scale ? "GlobalScale (1..73728)" : "LFQuant (1..65536)",
              t.c_str());
      return false;
    }
    (global_scale ? cparams.vardct_global_scale : cparams.vardct_quant_dc) = v;
  } else if (t == "XQMScale" || t == "BQMScale") {
    // XQMScale <0..7>, BQMScale <0..7>: the HF steps of X (B) are multiplied
    // by 0.8^(scale - 2) (XYB VarDCT frames only; by default 3 for X and 2
    // for B here, so X steps are 0.8 times the dequantization table).
    bool x = t == "XQMScale";
    t = tok();
    size_t num = 0;
    int v = ParseInt(t, &num);
    if (num != t.size() || v < 0 || v > 7) {
      fprintf(stderr, "Invalid %s (0..7): %s\n", x ? "XQMScale" : "BQMScale",
              t.c_str());
      return false;
    }
    (x ? cparams.vardct_x_qm_scale : cparams.vardct_b_qm_scale) = v;
  } else if (t == "LFChannelQuant") {
    // LFChannelQuant <Y> <X> <B>: inverse LF quantization steps of the XYB
    // channels (defaults 512 4096 256: at the default GlobalScale and LFQuant,
    // an LF value of 512 is 1.0 in Y), rounded to float16 (as 128 / value).
    float v[3];
    for (float& f : v) {
      t = tok();
      size_t num = 0;
      f = ParseFloat(t, &num);
      if (num != t.size() || !(f >= 1.0f / 256 && f <= (1 << 24))) {
        fprintf(stderr, "Invalid LFChannelQuant (1/256..2^24): %s\n",
                t.c_str());
        return false;
      }
    }
    // libjxl order: X, Y, B
    cparams.vardct_lf_inv_quant = {v[1], v[0], v[2]};
  } else if (t == "LFCfL") {
    // LFCfL <ytox> <ytob> (-128..127, default 0 0): chroma from luma for the
    // LF of VarDCT frames: X += (ytox / 84) * Y, B += (1 + ytob / 84) * Y
    // (after dequantization; the HF has its own factors, the YtoX and YtoB
    // maps of the HF metadata).
    int32_t* factors[2] = {&cparams.vardct_ytox_dc, &cparams.vardct_ytob_dc};
    for (int32_t* f : factors) {
      t = tok();
      size_t num = 0;
      *f = ParseInt(t, &num);
      if (num != t.size() || *f < -128 || *f > 127) {
        fprintf(stderr, "Invalid LFCfL factor (-128..127): %s\n", t.c_str());
        return false;
      }
    }
  } else if (t == "ACSTree" || t == "QFTree") {
    // ACSTree <tree>, QFTree <tree>: the AC strategy and the quant field of
    // each varblock, as trees over its (top-left) block position bx, by and
    // the quantized LF there (lfy, lfx, lfb); the quant field tree can also
    // use the AC strategy (acs). jxl_from_tree turns them into the lists of
    // the HF metadata (channel 2, in placement order).
    bool acs = t == "ACSTree";
    NamedTree& named = acs ? frame.acs_tree : frame.qf_tree;
    named.nodes.clear();
    std::vector<std::string> props = {"bx", "by", "lfy", "lfx", "lfb"};
    if (!acs) props.push_back("acs");
    if (!ParseNamedTree(tok, props, &named)) return false;
  } else if (t == "DequantFlat" || t == "DequantBands" || t == "DequantTable") {
    // DequantFlat <table> <Y> <X> <B>: every coefficient of the table has the
    // same dequantization step (per channel).
    // DequantBands <table> <n> <n Y steps> <n X steps> <n B steps>: the steps
    // at n (1..16) distance bands from the top-left coefficient to the
    // opposite corner, interpolated geometrically (a parametric table).
    // DequantTable <table> <den> <tree>: a RAW table, one integer (> 0) per
    // coefficient, given by the tree (with zero residuals; channels 0 X, 1 Y,
    // 2 B); the step is den * value.
    // <table>: a quantization table 0..16, a name (DCT8, DCT16, DCT16X8,
    // ...) or `all` (DequantFlat and DequantBands only).
    std::string kind = t;
    t = tok();
    std::vector<size_t> tables = ParseQuantTables(t, kind != "DequantTable");
    if (tables.empty()) {
      fprintf(stderr,
              "Invalid quantization table: %s (0..16, a name like DCT8, "
              "DCT16X8, IDENTITY, AFV%s)\n",
              t.c_str(), kind != "DequantTable" ? ", or all" : "");
      return false;
    }
    cparams.vardct_dequant.resize(jxl::kNumQuantTables);
    size_t num = 0;
    if (kind == "DequantTable") {
      t = tok();
      float den = ParseFloat(t, &num);
      if (num != t.size() || !(den >= 1.0f / (1 << 24) && den <= 65504)) {
        fprintf(stderr, "Invalid DequantTable den (2^-24..65504): %s\n",
                t.c_str());
        return false;
      }
      Tree& subtree = frame.dequant_trees[tables[0]];
      subtree.clear();
      JXL_RETURN_IF_ERROR(ParseNode(tok, subtree, spline_data, frame, cparams,
                                    W, H, io, have_next, x0, y0, buffer_size));
      cparams.vardct_dequant[tables[0]] = {};
      cparams.vardct_dequant[tables[0]].raw_den = den;
    } else {
      size_t n = 1;
      if (kind == "DequantBands") {
        t = tok();
        n = ParseUnsigned(t, &num);
        if (num != t.size() || n < 1 ||
            n > (1u << jxl::DctQuantWeightParams::kLog2MaxDistanceBands)) {
          fprintf(stderr, "Invalid number of distance bands (1..16): %s\n",
                  t.c_str());
          return false;
        }
      }
      CompressParams::CustomDequantTable table;
      table.num_bands = n;
      const size_t chan[3] = {1, 0, 2};  // Y, X, B -> libjxl order X, Y, B
      for (size_t c : chan) {
        for (size_t i = 0; i < n; i++) {
          t = tok();
          float step = ParseFloat(t, &num);
          // Signaled as quantization weights (1 / step): the first band as a
          // float16 of weight / 64, the others as float16 ratios.
          if (num != t.size() || !(step >= 1e-6f && step <= 256)) {
            fprintf(stderr, "Invalid dequantization step (1e-6..256): %s\n",
                    t.c_str());
            return false;
          }
          if (i > 0) {
            float ratio = step / table.band_steps[c][i - 1];
            if (!(ratio > 1.0f / 60000 && ratio < 60000)) {
              fprintf(stderr, "Band steps %g and %g are too far apart\n",
                      table.band_steps[c][i - 1], step);
              return false;
            }
          }
          table.band_steps[c][i] = step;
        }
      }
      for (size_t i : tables) {
        cparams.vardct_dequant[i] = table;
        frame.dequant_trees.erase(i);
      }
    }
  } else if (t == "DequantDefault") {
    // DequantDefault <table>: the default table again (<table> as for
    // DequantFlat, or `all`). Once all tables are default, frames signal them
    // with one bit instead of all 17 tables.
    t = tok();
    std::vector<size_t> tables = ParseQuantTables(t, /*allow_all=*/true);
    if (tables.empty()) {
      fprintf(stderr,
              "Invalid quantization table: %s (0..16, a name like DCT8, "
              "DCT16X8, IDENTITY, AFV, or all)\n",
              t.c_str());
      return false;
    }
    for (size_t i : tables) {
      if (i < cparams.vardct_dequant.size()) cparams.vardct_dequant[i] = {};
      frame.dequant_trees.erase(i);
    }
    if (std::all_of(cparams.vardct_dequant.begin(),
                    cparams.vardct_dequant.end(),
                    [](const CompressParams::CustomDequantTable& table) {
                      return table.num_bands == 0 && table.raw_den == 0;
                    })) {
      cparams.vardct_dequant.clear();
    }
  } else if (t == "CoeffOrder") {
    // CoeffOrder <order class> [Y|X|B] <n> <u1> <v1> ... <un> <vn>: in the
    // coefficient order of that class (an entry can also be "fill <k>", the
    // next k coefficients of the default order that the list does not have)
    // (0..12, or a DCT name like DCT64; for all channels unless one is given),
    // coefficients (u, v) come right after the LLF ones, so the first of them
    // is at scan index covered_blocks; the other coefficients keep the default
    // order. u and v are the horizontal and vertical frequency in the square or
    // tall transform of the class (like DCT16X8, which is 8 wide and 16 tall);
    // in the wide one (DCT8X16), the same entry is frequency (v, u).
    static const std::unordered_map<std::string, size_t> order_names = {
        {"DCT8", 0},    {"DCT16", 2},      {"DCT32", 3},   {"DCT16x8", 4},
        {"DCT32x8", 5}, {"DCT32x16", 6},   {"DCT64", 7},   {"DCT64x32", 8},
        {"DCT128", 9},  {"DCT128x64", 10}, {"DCT256", 11}, {"DCT256x128", 12},
    };
    t = tok();
    size_t num = 0;
    size_t ord;
    std::string name = t;
    std::replace(name.begin(), name.end(), 'X', 'x');
    if (order_names.count(name)) {
      ord = order_names.at(name);
    } else {
      ord = ParseUnsigned(t, &num);
      if (num != t.size() || ord >= jxl::kNumOrders) {
        fprintf(stderr, "Invalid coefficient order class (0..12): %s\n",
                t.c_str());
        return false;
      }
    }
    t = tok();
    std::vector<size_t> channels = {0, 1, 2};  // libjxl order: X, Y, B
    if (t == "Y" || t == "X" || t == "B") {
      channels = {t == "X" ? 0u : t == "Y" ? 1u : 2u};
      t = tok();
    }
    size_t n = ParseUnsigned(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid number of coefficients: %s\n", t.c_str());
      return false;
    }
    size_t raw = 0;
    while (jxl::kStrategyOrder[raw] != ord) raw++;
    jxl::AcStrategy acs = jxl::AcStrategy::FromRawStrategy(raw);
    size_t rows = acs.covered_blocks_y();
    size_t columns = acs.covered_blocks_x();
    jxl::CoefficientLayout(&rows, &columns);
    // In the coefficient layout (rows <= columns), the row is the horizontal
    // and the column the vertical frequency of the square or tall transform.
    // An entry is a coefficient (u, v) or "fill <k>": the next k coefficients
    // of the default order that the list does not have (resolved below).
    std::vector<uint32_t> positions;
    std::vector<std::pair<size_t, size_t>> fills;  // (index, k)
    for (size_t i = 0; i < n; i++) {
      t = tok();
      if (t == "fill") {
        t = tok();
        size_t k = ParseUnsigned(t, &num);
        if (num != t.size()) {
          fprintf(stderr, "Invalid number of fill coefficients: %s\n",
                  t.c_str());
          return false;
        }
        fills.emplace_back(positions.size(), k);
        continue;
      }
      size_t uv[2];
      for (size_t j = 0; j < 2; j++) {
        if (j > 0) t = tok();
        uv[j] = ParseUnsigned(t, &num);
        if (num != t.size()) {
          fprintf(stderr, "Invalid coefficient frequency: %s\n", t.c_str());
          return false;
        }
      }
      if (uv[0] >= rows * 8 || uv[1] >= columns * 8) {
        fprintf(stderr,
                "Coefficient (%zu, %zu) is not in the %zux%zu (WxH) block\n",
                uv[0], uv[1], rows * 8, columns * 8);
        return false;
      }
      if (uv[0] < rows && uv[1] < columns) {
        fprintf(stderr, "Coefficient (%zu, %zu) is an LLF coefficient\n", uv[0],
                uv[1]);
        return false;
      }
      uint32_t pos = uv[0] * columns * 8 + uv[1];
      if (std::find(positions.begin(), positions.end(), pos) !=
          positions.end()) {
        fprintf(stderr, "Coefficient (%zu, %zu) is repeated\n", uv[0], uv[1]);
        return false;
      }
      positions.push_back(pos);
    }
    if (!fills.empty()) {
      // The fills take the default order (after the LLF coefficients), in
      // order, skipping the coefficients that the list has.
      const size_t size =
          acs.covered_blocks_x() * acs.covered_blocks_y() * jxl::kDCTBlockSize;
      std::vector<jxl::coeff_order_t> natural(size);
      acs.ComputeNaturalCoeffOrder(natural.data());
      std::vector<bool> listed(size);
      for (uint32_t pos : positions) listed[pos] = true;
      size_t next = acs.covered_blocks_x() * acs.covered_blocks_y();
      std::vector<uint32_t> all;
      size_t explicit_pos = 0;
      for (const auto& fill : fills) {
        while (explicit_pos < fill.first)
          all.push_back(positions[explicit_pos++]);
        for (size_t j = 0; j < fill.second; j++) {
          while (next < size && listed[natural[next]]) next++;
          if (next == size) {
            fprintf(stderr,
                    "fill %zu: not enough coefficients left in the order of "
                    "class %zu\n",
                    fill.second, ord);
            return false;
          }
          listed[natural[next]] = true;
          all.push_back(natural[next++]);
        }
      }
      while (explicit_pos < positions.size()) {
        all.push_back(positions[explicit_pos++]);
      }
      positions = std::move(all);
    }
    cparams.custom_coeff_orders.resize(3 * jxl::kNumOrders);
    for (size_t c : channels) {
      cparams.custom_coeff_orders[3 * ord + c] = positions;
    }
  } else if (t == "VarDCT" || t == "Modular") {
    // VarDCT: a VarDCT frame (this one and the following ones, until
    // Modular): the tree defines the LF image (stream IDs of the VarDCT DC
    // groups; channels Y, X, B of quantized LF) and the HF metadata (stream
    // IDs of the AC metadata groups; channels YtoX, YtoB, AC strategy + quant
    // field, EPF sharpness); all HF coefficients are zero.
    // Modular: back to modular frames. The VarDCT settings persist (for a
    // later VarDCT frame) but do not apply to modular frames.
    const bool vardct = t == "VarDCT";
    if (vardct != cparams.vardct_from_tree) {
      // Each mode has its own loop filters; those given before the first
      // VarDCT apply to both.
      FrameSettings::LoopFilters current{cparams.gaborish, cparams.epf};
      if (frame.have_other_filters || !vardct) {
        cparams.gaborish = frame.other_filters.gaborish;
        cparams.epf = frame.other_filters.epf;
      }
      frame.other_filters = current;
      frame.have_other_filters = true;
    }
    cparams.modular_mode = !vardct;
    cparams.vardct_from_tree = vardct;
    // For the default loop filters (Gaborish, EPF) of VarDCT frames; lossless
    // modular frames.
    cparams.butteraugli_distance = vardct ? 1.0f : 0.0f;
  } else if (t == "HiddenChannel") {
    t = tok();
    size_t num = 0;
    cparams.move_to_front_from_channel = -1 - ParseUnsigned(t, &num);
    if (num != t.size() || num > 16) {
      fprintf(stderr, "Invalid HiddenChannel (max 16): %s\n", t.c_str());
      return false;
    }
  } else if (t == "RCT") {
    t = tok();
    size_t num = 0;
    cparams.colorspace = ParseInt(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid RCT: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Orientation") {
    t = tok();
    size_t num = 0;
    io.metadata.m.orientation = ParseUnsigned(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid Orientation: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Alpha") {
    io.metadata.m.SetAlphaBits(io.metadata.m.bit_depth.bits_per_sample);
    JXL_ASSIGN_OR_RETURN(
        ImageF alpha, ImageF::Create(jpegxl::tools::NoMemoryManager(), W, H));
    jxl::ZeroFillImage(&alpha);
    if (!io.frames[0].SetAlpha(std::move(alpha))) {
      fprintf(stderr, "Internal: SetAlpha failed\n");
      return false;
    }
  } else if (t == "Bitdepth") {
    t = tok();
    size_t num = 0;
    uint32_t bits_per_sample = ParseUnsigned(t, &num);
    if (num != t.size() || bits_per_sample < 1 || bits_per_sample > 32) {
      fprintf(stderr, "Invalid Bitdepth: %s\n", t.c_str());
      return false;
    }
    if (buffer_size == 0) {
    // Match the main encoder and use 32bit buffers for bitdepths over 12.
    buffer_size = bits_per_sample > 12 ? 2 : 3;
    }
    io.metadata.m.bit_depth.bits_per_sample = bits_per_sample;
  } else if (t == "FloatExpBits") {
    t = tok();
    size_t num = 0;
    io.metadata.m.bit_depth.floating_point_sample = true;
    io.metadata.m.bit_depth.exponent_bits_per_sample = ParseUnsigned(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid FloatExpBits: %s\n", t.c_str());
      return false;
    }
  } else if (t == "FramePos") {
    t = tok();
    size_t num = 0;
    x0 = ParseInt(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid FramePos x0: %s\n", t.c_str());
      return false;
    }
    t = tok();
    y0 = ParseInt(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid FramePos y0: %s\n", t.c_str());
      return false;
    }
  } else if (t == "NotLast") {
    have_next = JXL_TRUE;
  } else if (t == "Upsample") {
    t = tok();
    size_t num = 0;
    cparams.resampling = ParseUnsigned(t, &num);
    if (num != t.size() ||
        (cparams.resampling != 1 && cparams.resampling != 2 &&
         cparams.resampling != 4 && cparams.resampling != 8)) {
      fprintf(stderr, "Invalid Upsample: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Upsample_EC") {
    t = tok();
    size_t num = 0;
    cparams.ec_resampling = ParseUnsigned(t, &num);
    if (num != t.size() ||
        (cparams.ec_resampling != 1 && cparams.ec_resampling != 2 &&
         cparams.ec_resampling != 4 && cparams.ec_resampling != 8)) {
      fprintf(stderr, "Invalid Upsample_EC: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Animation") {
    io.metadata.m.have_animation = true;
    io.metadata.m.animation.tps_numerator = 1000;
    io.metadata.m.animation.tps_denominator = 1;
    io.frames[0].duration = 100;
  } else if (t == "AnimationFPS") {
    t = tok();
    size_t num = 0;
    io.metadata.m.animation.tps_numerator = ParseUnsigned(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid numerator: %s\n", t.c_str());
      return false;
    }
    t = tok();
    num = 0;
    io.metadata.m.animation.tps_denominator = ParseUnsigned(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid denominator: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Duration") {
    t = tok();
    size_t num = 0;
    io.frames[0].duration = ParseUnsigned(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid Duration: %s\n", t.c_str());
      return false;
    }
  } else if (t == "BlendMode") {
    t = tok();
    if (t == "kAdd") {
      io.frames[0].blendmode = BlendMode::kAdd;
    } else if (t == "kReplace") {
      io.frames[0].blendmode = BlendMode::kReplace;
    } else if (t == "kBlend") {
      io.frames[0].blendmode = BlendMode::kBlend;
    } else if (t == "kAlphaWeightedAdd") {
      io.frames[0].blendmode = BlendMode::kAlphaWeightedAdd;
    } else if (t == "kMul") {
      io.frames[0].blendmode = BlendMode::kMul;
    } else {
      fprintf(stderr, "Invalid BlendMode: %s\n", t.c_str());
      return false;
    }
  } else if (t == "SplineQuantizationAdjustment" || t == "SplineAdjustment") {
    // SplineAdjustment <n>: quantize with n and signal it (the fixed
    // behaviour); SplineQuantizationAdjustment <n>: the old one (see
    // SplineData).
    spline_data.signal_adjustment = t == "SplineAdjustment";
    t = tok();
    size_t num = 0;
    spline_data.quantization_adjustment = ParseInt(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid SplineQuantizationAdjustment: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Spline") {
    Spline spline;
    const auto parse_float = [&t, &tok](float& output) {
      t = tok();
      size_t num = 0;
      output = ParseFloat(t, &num);
      if (num != t.size()) {
        fprintf(stderr, "Invalid spline data: %s\n", t.c_str());
        return false;
      }
      return true;
    };
    for (auto& dct : spline.color_dct) {
      for (float& coefficient : dct) {
        JXL_RETURN_IF_ERROR(parse_float(coefficient));
      }
    }
    for (float& coefficient : spline.sigma_dct) {
      JXL_RETURN_IF_ERROR(parse_float(coefficient));
    }

    while (true) {
      t = tok();
      if (t == "EndSpline") break;
      size_t num = 0;
      Spline::Point point;
      point.x = ParseFloat(t, &num);
      bool ok_x = num == t.size();
      auto t_y = tok();
      point.y = ParseFloat(t_y, &num);
      if (!ok_x || num != t_y.size()) {
        fprintf(stderr, "Invalid spline control point: %s %s\n", t.c_str(),
                t_y.c_str());
        return false;
      }
      spline.control_points.push_back(point);
    }

    if (spline.control_points.empty()) {
      fprintf(stderr, "Spline with no control point\n");
      return false;
    }
    // The first spline's starting point is coded unsigned (the other
    // starting points as signed deltas from the previous one).
    if (spline_data.splines.empty() &&
        (std::round(spline.control_points[0].x) < 0 ||
         std::round(spline.control_points[0].y) < 0)) {
      fprintf(stderr,
              "The first spline of a frame must not start at a negative "
              "position (%g %g); later splines may\n",
              spline.control_points[0].x, spline.control_points[0].y);
      return false;
    }

    spline_data.splines.push_back(std::move(spline));
  } else if (t == "Gaborish") {
    cparams.gaborish = jxl::Override::kOn;
  } else if (t == "NoGaborish") {
    // Gaborish is on by default in VarDCT frames.
    cparams.gaborish = jxl::Override::kOff;
  } else if (t == "Palette" || t == "ImplicitPalette") {
    // Palette <num_c> <nb_deltas> <nb_colors> <predictor> followed by
    // (nb_deltas + nb_colors) x num_c values: an explicit palette on the
    // first num_c channels. Channel 0 becomes the palette (the tool writes
    // its tree), channel 1 the palette indices (then any other channels).
    // An index i < 0 is a default delta, 0 <= i < nb_deltas a listed delta
    // (added to <predictor>), then the listed colours, then the implicit
    // colour cubes.
    // ImplicitPalette <num_c> <nb_deltas> <nb_colors> <predictor>: the same
    // palette without listed entries: the tree gives the meta channel (c == 0,
    // x = entry, y = component), so large palettes can be cheap.
    PaletteSettings& pal = frame.palette;
    pal.implicit = t == "ImplicitPalette";
    const char* what[3] = {"Palette channel count (1..16)",
                           "Palette delta count", "Palette colour count"};
    uint32_t* field[3] = {&pal.num_c, &pal.nb_deltas, &pal.nb_colors};
    for (size_t i = 0; i < 3; i++) {
      t = tok();
      size_t num = 0;
      uint64_t v = ParseUnsigned(t, &num);
      if (num != t.size() || (i == 0 && (v < 1 || v > 16)) || v > 70000) {
        fprintf(stderr, "Invalid %s: %s\n", what[i], t.c_str());
        return false;
      }
      *field[i] = v;
    }
    t = tok();
    if (!predictor_map.count(t)) {
      fprintf(stderr, "Unexpected Palette predictor: %s\n", t.c_str());
      return false;
    }
    pal.predictor = predictor_map.at(t);
    pal.entries.assign(pal.implicit ? 0 : pal.nb_deltas + pal.nb_colors,
                       std::vector<int32_t>(pal.num_c));
    for (auto& entry : pal.entries) {
      for (int32_t& v : entry) {
        t = tok();
        size_t num = 0;
        v = ParseInt(t, &num);
        if (num != t.size()) {
          fprintf(stderr, "Invalid Palette value: %s\n", t.c_str());
          return false;
        }
      }
    }
    pal.enabled = true;
    cparams.lossy_palette = false;
  } else if (t == "NibbleCode") {
    // Prefix codes, and a flat 4-bit code for every histogram with more than
    // two of the tokens 0..15 (residuals -8..7): those samples are raw
    // nibbles of the file (bit-reversed canonical codes, LSB first).
    cparams.flat_nibble_code = true;
  } else if (t == "Residuals" || t == "MetaResiduals") {
    // (MetaResiduals: the same, but the pattern starts at the meta channels,
    // i.e. the palette entries are the tree's leaves plus the first values.)
    const bool include_meta = t == "MetaResiduals";
    // Residuals <stream|*> [ <prefix values> ] [ <period values> ] (a value
    // may be written <value>*<count> for a run of equal values, and
    // ( <values> )*<count> repeats a group): instead of
    // all zeros, the residuals of modular stream <stream> (the tree's
    // property g; * for every stream not listed) are the prefix, then the
    // period repeated forever (coded with LZ77, so a short period is cheap).
    t = tok();
    int stream = -1;
    if (t != "*") {
      size_t num = 0;
      stream = static_cast<int>(ParseUnsigned(t, &num));
      if (num != t.size() || t.empty()) {
        fprintf(stderr, "Invalid Residuals stream: %s\n", t.c_str());
        return false;
      }
    }
    jxl::ResidualPattern pattern;
    pattern.include_meta = include_meta;
    for (std::vector<int32_t>* list : {&pattern.prefix, &pattern.period}) {
      t = tok();
      if (t != "[") {
        fprintf(stderr, "Residuals: expected [, got %s\n", t.c_str());
        return false;
      }
      // ( <values> )*<count> repeats a group of values.
      size_t group_start = std::string::npos;
      for (t = tok(); t != "]"; t = tok()) {
        if (t == "(") {
          if (group_start != std::string::npos) {
            fprintf(stderr, "Residuals: nested ( groups\n");
            return false;
          }
          group_start = list->size();
          continue;
        }
        if (t.rfind(")", 0) == 0) {
          if (group_start == std::string::npos) {
            fprintf(stderr, "Residuals: ) without (\n");
            return false;
          }
          uint64_t repeat = 1;
          if (t.size() > 1) {
            const std::string count_text = t.substr(t[1] == '*' ? 2 : 1);
            size_t num = 0;
            repeat = ParseUnsigned(count_text, &num);
            if (t[1] != '*' || num != count_text.size() || count_text.empty() ||
                repeat == 0 || repeat > (1u << 28)) {
              fprintf(stderr, "Invalid residual group count: %s\n", t.c_str());
              return false;
            }
          }
          const std::vector<int32_t> group(list->begin() + group_start,
                                           list->end());
          for (uint64_t r = 1; r < repeat; r++) {
            list->insert(list->end(), group.begin(), group.end());
          }
          group_start = std::string::npos;
          continue;
        }
        // A value, or <value>*<count> for a run of equal values.
        std::string value_text = t;
        uint64_t count = 1;
        const size_t star = t.find('*');
        if (star != std::string::npos) {
          value_text = t.substr(0, star);
          const std::string count_text = t.substr(star + 1);
          size_t num = 0;
          count = ParseUnsigned(count_text, &num);
          if (num != count_text.size() || count_text.empty() || count == 0 ||
              count > (1u << 28)) {
            fprintf(stderr, "Invalid residual count: %s\n", t.c_str());
            return false;
          }
        }
        size_t num = 0;
        int64_t v = ParseInt(value_text, &num);
        if (num != value_text.size() || value_text.empty() || v < -(1 << 30) ||
            v > (1 << 30)) {
          fprintf(stderr, "Invalid residual: %s\n", t.c_str());
          return false;
        }
        list->insert(list->end(), count, static_cast<int32_t>(v));
      }
      if (group_start != std::string::npos) {
        fprintf(stderr, "Residuals: ( without )\n");
        return false;
      }
    }
    if (pattern.period.empty()) {
      fprintf(stderr, "Residuals: the repeating part must not be empty\n");
      return false;
    }
    cparams.custom_residuals[stream] = std::move(pattern);
  } else if (t == "NoPalette") {
    frame.palette.enabled = false;
  } else if (t == "DeltaPalette") {
    cparams.lossy_palette = true;
    cparams.palette_colors = 0;
  } else if (t == "EPF") {
    t = tok();
    size_t num = 0;
    cparams.epf = ParseInt(t, &num);
    if (num != t.size() || cparams.epf > 3) {
      fprintf(stderr, "Invalid EPF: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Noise") {
    cparams.manual_noise.resize(8);
    for (size_t i = 0; i < 8; i++) {
      t = tok();
      size_t num = 0;
      float v = ParseFloat(t, &num);
      if (num != t.size() || v < 0.0f || v > 1.0f) {
        fprintf(stderr, "Invalid noise entry: %s\n", t.c_str());
        return false;
      }
      cparams.manual_noise[i] = jxl::Clamp1(v, 0.0f, jxl::kNoiseLutMax);
    }
  } else if (t == "XYBFactors") {
    cparams.manual_xyb_factors.resize(3);
    for (size_t i = 0; i < 3; i++) {
      t = tok();
      size_t num = 0;
      cparams.manual_xyb_factors[i] = ParseFloat(t, &num);
      if (num != t.size()) {
        fprintf(stderr, "Invalid XYB factor: %s\n", t.c_str());
        return false;
      }
    }
  } else if (t == "PQ") {
    io.metadata.m.color_encoding.Tf().transfer_function =
        jxl::TransferFunction::kPQ;
    io.metadata.m.tone_mapping.intensity_target = 10000;
  } else if (t == "HLG") {
    io.metadata.m.color_encoding.Tf().transfer_function =
        jxl::TransferFunction::kHLG;
    io.metadata.m.tone_mapping.intensity_target = 1000;
  } else if (t == "Rec2100") {
    JXL_RETURN_IF_ERROR(
        io.metadata.m.color_encoding.SetPrimariesType(jxl::Primaries::k2100));
  } else if (t == "P3") {
    JXL_RETURN_IF_ERROR(
        io.metadata.m.color_encoding.SetPrimariesType(jxl::Primaries::kP3));
  } else if (t == "16BitBuffers") {
    buffer_size = 1;
  } else if (t == "32BitBuffers") {
    buffer_size = 2;
  } else if (t == "ImageSize") {
    // ImageSize <xsize> <ysize>: the image (canvas) size, if the first frame
    // is not canvas-sized (e.g. a large ReferenceOnly sprite sheet).
    for (size_t* v : {&frame.image_xsize, &frame.image_ysize}) {
      t = tok();
      size_t num = 0;
      *v = ParseUnsigned(t, &num);
      if (num != t.size() || *v == 0) {
        fprintf(stderr, "Invalid ImageSize: %s\n", t.c_str());
        return false;
      }
    }
  } else if (t == "ReferenceOnly") {
    frame.reference_only = true;
  } else if (t == "SaveBeforeCT") {
    frame.save_before_ct = true;
  } else if (t == "SaveAsReference" || t == "BlendSource") {
    bool save = t == "SaveAsReference";
    t = tok();
    size_t num = 0;
    size_t slot = ParseUnsigned(t, &num);
    if (num != t.size() || slot >= jxl::kMaxNumReferenceFrames) {
      fprintf(stderr, "Invalid reference slot: %s\n", t.c_str());
      return false;
    }
    if (save) {
      frame.save_as_reference = static_cast<int>(slot);
    } else {
      frame.blend_source = slot;
      frame.blend_source_given = true;
    }
  } else if (t == "Patch" || t == "PatchExtraBlendMode") {
    // Patch <ref> <x0> <y0> <xsize> <ysize> <x> <y> <blend mode>: blends the
    // rectangle (x0, y0, xsize, ysize) of reference slot <ref> onto this frame
    // at (x, y). PatchExtraBlendMode <blend mode>: mode used for the extra
    // channels by later patches (default kNone).
    static const std::unordered_map<std::string, jxl::PatchBlendMode> modes = {
        {"kNone", jxl::PatchBlendMode::kNone},
        {"kReplace", jxl::PatchBlendMode::kReplace},
        {"kAdd", jxl::PatchBlendMode::kAdd},
        {"kMul", jxl::PatchBlendMode::kMul},
        {"kBlendAbove", jxl::PatchBlendMode::kBlendAbove},
        {"kBlendBelow", jxl::PatchBlendMode::kBlendBelow},
        {"kAlphaWeightedAddAbove", jxl::PatchBlendMode::kAlphaWeightedAddAbove},
        {"kAlphaWeightedAddBelow", jxl::PatchBlendMode::kAlphaWeightedAddBelow},
    };
    bool is_patch = t == "Patch";
    size_t v[7] = {};
    for (size_t i = 0; is_patch && i < 7; i++) {
      t = tok();
      size_t num = 0;
      v[i] = ParseUnsigned(t, &num);
      if (num != t.size()) {
        fprintf(stderr, "Invalid patch coordinate: %s\n", t.c_str());
        return false;
      }
    }
    t = tok();
    if (!modes.count(t)) {
      fprintf(stderr, "Invalid patch blend mode: %s\n", t.c_str());
      return false;
    }
    uint8_t mode = static_cast<uint8_t>(modes.at(t));
    if (is_patch) {
      if (v[0] >= jxl::kMaxNumReferenceFrames || v[3] == 0 || v[4] == 0) {
        fprintf(stderr, "Invalid patch reference or size\n");
        return false;
      }
      cparams.custom_patches.push_back({v[0], v[1], v[2], v[3], v[4], v[5],
                                        v[6], mode, frame.patch_ec_mode,
                                        frame.patch_clamp});
    } else {
      frame.patch_ec_mode = mode;
    }
  } else if (t == "PatchClamp") {
    frame.patch_clamp = true;
  } else {
    fprintf(stderr, "Unexpected node type: %s\n", t.c_str());
    return false;
  }
  JXL_RETURN_IF_ERROR(ParseNode(tok, tree, spline_data, frame, cparams, W, H,
                                io, have_next, x0, y0, buffer_size));
  return true;
}
// The estimated area of the splines of a frame, as the decoders compute it
// (QuantizedSpline::Dequantize in lib/jxl/splines.cc, jxl-rs's
// QuantizedSpline::dequantize), with `area_limit` as the limit (which also caps
// the width of a spline in the estimate). The decoders refuse a frame whose
// estimate exceeds the limit: libjxl with min(1024 * pixels + 2^32, 2^42) for
// the upsampled frame size, jxl-rs (level 5) with min(8 * pixels + 2^25, 2^30)
// for the coded frame size. Returns the estimate, or area_limit + 1 if a spline
// is longer than the limit (in Manhattan distance). `splines` are dequantized
// without chroma from luma; the estimate uses the default factors (YtoX 0,
// YtoB 1), as jxl_from_tree does not change them.
uint64_t SplineAreaEstimate(const std::vector<Spline>& splines,
                            int32_t quantization_adjustment,
                            uint64_t area_limit) {
  static constexpr float kChannelWeight[4] = {0.0042f, 0.075f, 0.07f, 0.3333f};
  const float inv_quant = quantization_adjustment >= 0
                              ? 1.f / (1.f + .125f * quantization_adjustment)
                              : 1.f - .125f * quantization_adjustment;
  // The quantized values, from the dequantized ones.
  const auto quantized = [inv_quant](float v, int i, int c) {
    const float factor =
        (i == 0 ? 0.70710678f : 1.f) * kChannelWeight[c] * inv_quant;
    return std::round(std::abs(v) / factor);
  };
  uint64_t total = 0;
  for (const Spline& spline : splines) {
    uint64_t manhattan_distance = 0;
    for (size_t i = 1; i < spline.control_points.size(); i++) {
      manhattan_distance +=
          static_cast<uint64_t>(std::abs(spline.control_points[i].x -
                                         spline.control_points[i - 1].x) +
                                std::abs(spline.control_points[i].y -
                                         spline.control_points[i - 1].y));
    }
    if (manhattan_distance > area_limit) return area_limit + 1;
    uint64_t color[3] = {};
    for (int c = 0; c < 3; c++) {
      for (int i = 0; i < 32; i++) {
        color[c] += static_cast<uint64_t>(
            std::ceil(inv_quant * quantized(spline.color_dct[c][i], i, c)));
      }
    }
    color[2] += color[1];  // YtoB 1
    const uint64_t max_color = std::max({color[0], color[1], color[2]});
    uint64_t logcolor = 0;
    while ((uint64_t{1} << logcolor) < max_color + 1) logcolor++;
    logcolor = std::max<uint64_t>(1, logcolor);
    const float weight_limit =
        std::ceil(std::sqrt((static_cast<float>(area_limit) / logcolor) /
                            std::max<uint64_t>(1, manhattan_distance)));
    uint64_t width_estimate = 0;
    for (int i = 0; i < 32; i++) {
      const float weight_f =
          std::ceil(inv_quant * quantized(spline.sigma_dct[i], i, 3));
      const uint64_t weight = static_cast<uint64_t>(
          std::min(weight_limit, std::max(1.0f, weight_f)));
      width_estimate += weight * weight * logcolor;
    }
    total += width_estimate * manhattan_distance;
  }
  return total;
}

// Checks the splines of a frame against the limits of libjxl (error) and of
// jxl-rs, which applies the level 5 limits (a warning: libjxl decodes such
// files).
Status CheckSplineLimits(const std::vector<QuantizedSpline>& quantized_splines,
                         const std::vector<Spline::Point>& starting_points,
                         int32_t quantization_adjustment, size_t xsize,
                         size_t ysize, size_t upsampling, size_t frame_index) {
  if (quantized_splines.empty()) return true;
  const uint64_t coded_pixels = static_cast<uint64_t>(xsize) * ysize;
  const uint64_t pixels = coded_pixels * upsampling * upsampling;
  std::vector<Spline> splines(quantized_splines.size());
  size_t num_control_points = 0;
  for (size_t s = 0; s < splines.size(); s++) {
    uint64_t area = 0;
    if (!quantized_splines[s].Dequantize(starting_points[s],
                                         quantization_adjustment, 0.f, 0.f,
                                         pixels, &area, splines[s])) {
      fprintf(stderr,
              "Spline %zu of frame %zu is invalid for decoders: a control "
              "point is out of range, or the spline is too long or too wide "
              "(estimated area over 1024 x pixels + 2^32)\n",
              s, frame_index);
      return JXL_FAILURE("Invalid spline");
    }
    num_control_points += splines[s].control_points.size() - 1;
  }
  // Number of control points (libjxl: of the coded frame size).
  const size_t max_control_points =
      std::min<size_t>(size_t{1} << 20, coded_pixels / 2);
  if (splines.size() + 1 > max_control_points ||
      num_control_points > max_control_points) {
    fprintf(stderr,
            "Frame %zu has %zu splines with %zu control points (after the "
            "first ones), decoders allow at most %zu (half the coded pixels)\n",
            frame_index, splines.size(), num_control_points,
            max_control_points);
    return JXL_FAILURE("Too many spline control points");
  }
  const uint64_t libjxl_limit =
      std::min((pixels << 10) + (uint64_t{1} << 32), uint64_t{1} << 42);
  const uint64_t libjxl_area =
      SplineAreaEstimate(splines, quantization_adjustment, libjxl_limit);
  if (libjxl_area > libjxl_limit) {
    fprintf(stderr,
            "The splines of frame %zu are too large for decoders: estimated "
            "area %" PRIu64 ", limit %" PRIu64 " (1024 x pixels + 2^32)\n",
            frame_index, libjxl_area, libjxl_limit);
    return JXL_FAILURE("Too large spline area");
  }
  const uint64_t level5_limit =
      std::min(8 * coded_pixels + (uint64_t{1} << 25), uint64_t{1} << 30);
  const uint64_t level5_area =
      SplineAreaEstimate(splines, quantization_adjustment, level5_limit);
  if (level5_area > level5_limit) {
    fprintf(stderr,
            "Warning: the splines of frame %zu have an estimated area of "
            "%" PRIu64 ", over the level 5 limit of %" PRIu64
            " (8 x %zux%zu coded pixels + 2^25): jxl-rs refuses the file "
            "(libjxl decodes it). The estimate grows with sigma^2, the length "
            "and the log of the colour.\n",
            frame_index, level5_area, level5_limit, xsize, ysize);
  }
  return true;
}

// JXL_FAILURE prints its message only in debug builds.
Status FailWithMessage(const char* message) {
  fprintf(stderr, "%s\n", message);
  return JXL_FAILURE("%s", message);
}

// What a reference slot holds, as far as frame blending is concerned.
enum class SlotState : uint8_t { kEmpty, kAfterCT, kBeforeCT };

// The blend source of a frame without BlendSource: slot 1, unless it holds a
// frame saved before the color transform (a ReferenceOnly or SaveBeforeCT
// frame), which decoders do not blend from. Then the slot that was saved last
// after the color transform (usually the previous displayed frame), or else an
// empty slot (blending onto an empty canvas).
size_t DefaultBlendSource(const SlotState* slots, int last_after_ct) {
  if (slots[1] != SlotState::kBeforeCT) return 1;
  if (last_after_ct >= 0 && slots[last_after_ct] == SlotState::kAfterCT) {
    return last_after_ct;
  }
  for (size_t i = 0; i < jxl::kMaxNumReferenceFrames; i++) {
    if (slots[i] == SlotState::kEmpty) return i;
  }
  return 1;
}

// The blending info of the extra channels as EncodeFrame writes it by default
// (with blend source 1).
std::vector<jxl::BlendingInfo> DefaultEcBlending(
    const jxl::ImageBundle& ib, const CodecMetadata& metadata) {
  const auto& ec = metadata.m.extra_channel_info;
  size_t alpha = 0;
  if (ec.size() > 1) {
    for (size_t i = 0; i < ec.size(); i++) {
      if (ec[i].type == jxl::ExtraChannel::kAlpha) {
        alpha = i;
        break;
      }
    }
  }
  std::vector<jxl::BlendingInfo> info(ec.size());
  for (size_t i = 0; i < ec.size(); i++) {
    info[i].alpha_channel = alpha;
    // Hidden channels hold state that every frame computes anew: replaced.
    // (Otherwise a lone hidden channel would be blended as if it were the
    // alpha, and accumulate from frame to frame.) Other extra channels than
    // the blending alpha channel are added.
    BlendMode mode = ib.blendmode;
    if (ec[i].type == jxl::ExtraChannel::kOptional) {
      mode = BlendMode::kReplace;
    } else if (ec[i].type != jxl::ExtraChannel::kBlack && i != alpha) {
      mode = BlendMode::kAdd;
    }
    info[i].mode = ib.blend ? mode : BlendMode::kReplace;
    info[i].source = 1;
  }
  return info;
}

// Whether a frame of this (upsampled) size is not a full frame.
bool IsPartialFrame(const jxl::ImageBundle& ib, const CodecMetadata& metadata,
                    size_t xsize, size_t ysize) {
  return ib.origin.x0 != 0 || ib.origin.y0 != 0 || xsize != metadata.xsize() ||
         ysize != metadata.ysize();
}

// Whether decoders blend a (displayed) frame with its blend source, as in
// jxl-rs's FrameHeader::needs_blending: a cropped frame, or a blend mode other
// than kReplace (for the color or an extra channel).
bool FrameNeedsBlending(const jxl::ImageBundle& ib,
                        const CodecMetadata& metadata, size_t xsize,
                        size_t ysize) {
  if (IsPartialFrame(ib, metadata, xsize, ysize)) return true;
  if (ib.blend && ib.blendmode != BlendMode::kReplace) return true;
  for (const jxl::BlendingInfo& info : DefaultEcBlending(ib, metadata)) {
    if (info.mode != BlendMode::kReplace) return true;
  }
  return false;
}
}  // namespace

// One encoding pass. With `auto_16bit`, frames of at most 12 bits get 16-bit
// buffers unless the encoder finds modular values outside the 16-bit range
// (samples or intermediate values of the transforms), in which case nothing is
// written and *needs_32bit is set.
::jxl::Status JxlFromTreePass(std::istream& input, const char* out,
                              const char* tree_out, bool auto_16bit,
                              bool* needs_32bit) {
  Tree tree;
  SplineData spline_data;
  FrameSettings frame;
  CompressParams cparams = {};
  size_t width = 1024;
  size_t height = 1024;
  int x0 = 0;
  int y0 = 0;
  cparams.SetLossless();
  cparams.responsive = JXL_FALSE;
  cparams.resampling = 1;
  cparams.ec_resampling = 1;
  cparams.modular_group_size_shift = 3;
  cparams.colorspace = 0;
  cparams.speed_tier = jxl::SpeedTier::kGlacier;
  // The token streams are huge (all-zero residuals) but most of the file is
  // the tree: take the careful LZ77 first pass for all of them.
  cparams.lz77_careful_first_pass = true;
  cparams.buffering = 0;
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  io->metadata.m.modular_16_bit_buffer_sufficient = false;
  int have_next = JXL_FALSE;
  int buffer_size = 0;

  std::istream* f = &input;

  // Whitespace-separated tokens; comments (/* ... */) are dropped here, also
  // when they touch a word (/*like this*/, or 12/*c*/34, which is 12 34).
  std::string pending;
  auto tok = [&f, &pending]() -> std::string {
    for (;;) {
      std::string w;
      if (!pending.empty()) {
        w.swap(pending);
      } else if (!(*f >> w)) {
        return "";
      }
      const size_t open = w.find("/*");
      if (open == std::string::npos) return w;
      std::string before = w.substr(0, open);
      std::string rest = w.substr(open + 2);
      size_t close;
      while ((close = rest.find("*/")) == std::string::npos) {
        if (!(*f >> rest)) break;  // unterminated: to the end
      }
      pending = close == std::string::npos ? "" : rest.substr(close + 2);
      if (!before.empty()) return before;
    }
  };
  if (!ParseNode(tok, tree, spline_data, frame, cparams, width, height, *io,
                 have_next, x0, y0, buffer_size)) {
    return JXL_FAILURE("Failed to ParseNode");
  }

  // 16-bit buffers (as a naked codestream, at level 5, implies) if asked for,
  // or by default for up to 12 bits, if the encoder finds that the modular
  // values fit (else JxlFromTree encodes again with 32-bit buffers).
  // (buffer_size 0: no Bitdepth given, i.e. 8 bits: automatic as well.)
  const bool auto16 = auto_16bit && (buffer_size == 3 || buffer_size == 0);
  int64_t modular_range[2] = {std::numeric_limits<int64_t>::max(),
                              std::numeric_limits<int64_t>::min()};
  if (buffer_size == 1 || auto16) {
    io->metadata.m.modular_16_bit_buffer_sufficient = true;
  }
  if (auto16) cparams.modular_range_out = modular_range;

  // The sections of a VarDCT frame (LFTree, HFMetaTree, DequantTable,
  // ACSTree, QFTree, ExtraTree), combined into the frame's tree.
  // With explicit palette entries: the frame's tree with the entries in the
  // tree instead of coded (see combine_sections).
  std::optional<Tree> palette_inline_tree;
  const auto combine_sections = [&](bool have_extra_channels) -> Status {
    cparams.custom_palette.enabled = frame.palette.enabled;
    cparams.options.code_meta_channels = false;
    palette_inline_tree.reset();
    if (!cparams.custom_residuals.empty() && cparams.vardct_from_tree) {
      return FailWithMessage(
          "Residuals are for modular frames (the VarDCT tool computes its LF "
          "and HF metadata with zero residuals)");
    }
    if (!cparams.custom_residuals.empty()) {
      // Warn about streams that code no samples in this frame.
      jxl::FrameDimensions fd;
      fd.Set(width * cparams.resampling, height * cparams.resampling,
             cparams.modular_group_size_shift, 0, 0, /*modular_mode=*/true,
             cparams.resampling);
      const size_t first_group = jxl::ModularStreamId::ModularAC(0, 0).ID(fd);
      const size_t last_group =
          jxl::ModularStreamId::ModularAC(fd.num_groups - 1, 0).ID(fd);
      const size_t first_lf = jxl::ModularStreamId::ModularDC(0).ID(fd);
      const size_t last_lf =
          jxl::ModularStreamId::ModularDC(fd.num_dc_groups - 1).ID(fd);
      const bool multi_group = fd.num_groups > 1;
      for (const auto& kv : cparams.custom_residuals) {
        const int id = kv.first;
        if (id < 0) continue;
        const bool group = id >= static_cast<int>(first_group) &&
                           id <= static_cast<int>(last_group);
        const bool lf = cparams.responsive &&
                        id >= static_cast<int>(first_lf) &&
                        id <= static_cast<int>(last_lf);
        const bool global_has_samples =
            id == 0 &&
            (!multi_group || frame.palette.enabled || cparams.responsive);
        if (!(group && multi_group) && !lf && !global_has_samples) {
          fprintf(stderr,
                  "Warning: Residuals for stream %d, which codes no samples in "
                  "this %zux%zu frame (%s; the group streams are %zu..%zu)\n",
                  id, width, height,
                  multi_group ? "larger than one group: stream 0 only holds "
                                "global data such as a palette"
                              : "a single group: everything is in stream 0",
                  first_group, last_group);
        }
      }
    }
    if (frame.palette.enabled) {
      if (cparams.vardct_from_tree) {
        return FailWithMessage("Palette is for modular frames");
      }
      if (cparams.move_to_front_from_channel != -1) {
        return FailWithMessage("Palette with HiddenChannel is not supported");
      }
      cparams.custom_palette.num_c = frame.palette.num_c;
      cparams.custom_palette.nb_deltas = frame.palette.nb_deltas;
      cparams.custom_palette.nb_colors = frame.palette.nb_colors;
      cparams.custom_palette.predictor = frame.palette.predictor;
      cparams.custom_palette.entries.clear();
      const size_t nb_entries =
          frame.palette.nb_deltas + frame.palette.nb_colors;
      // (With Residuals, the entries stay in the tree; the meta channel keeps
      // zero residuals, the pattern starts at the indices.)
      const bool code_entries = !frame.palette.implicit && nb_entries > 0 &&
                                cparams.custom_residuals.empty();
      if (code_entries) {
        // The entries are the meta channel's pixels: coded, not in the tree.
        cparams.custom_palette.entries.assign(frame.palette.num_c,
                                              std::vector<int32_t>(nb_entries));
        for (size_t i = 0; i < nb_entries; i++) {
          for (size_t c = 0; c < frame.palette.num_c; c++) {
            cparams.custom_palette.entries[c][i] = frame.palette.entries[i][c];
          }
        }
      }
      cparams.options.code_meta_channels =
          !cparams.custom_palette.entries.empty();
      // The index channel is coded in group streams if it is larger than a
      // group (it has the frame's coded size).
      const size_t group_dim = 128u << cparams.modular_group_size_shift;
      const bool multi_group = width > group_dim || height > group_dim;
      if (!cparams.custom_palette.entries.empty()) {
        // The entries can also be in the tree: EncodeOneFrame keeps whichever
        // of the two is smaller (the tree is cheaper for a few entries).
        palette_inline_tree = tree;
        AddPaletteTree(frame.palette, multi_group, /*inline_entries=*/true,
                       &*palette_inline_tree);
      }
      AddPaletteTree(
          frame.palette, multi_group,
          /*inline_entries=*/!code_entries && !frame.palette.implicit, &tree);
    }
    if (frame.lf_tree.empty() && frame.hf_meta_tree.empty() &&
        frame.dequant_trees.empty() && frame.acs_tree.nodes.empty() &&
        frame.qf_tree.nodes.empty() && frame.extra_tree.empty()) {
      return true;
    }
    if (!cparams.vardct_from_tree) {
      return FailWithMessage(
          "LFTree, HFMetaTree, DequantTable, ACSTree, QFTree and ExtraTree "
          "need VarDCT");
    }
    if (!frame.extra_tree.empty() && !have_extra_channels) {
      return FailWithMessage(
          "ExtraTree needs extra channels (Alpha or HiddenChannel)");
    }
    ExtraChannelLayout ec;
    ec.present = have_extra_channels;
    ec.upsampling = cparams.resampling;
    ec.ec_upsampling = cparams.ec_resampling;
    ec.responsive = cparams.responsive != 0;
    if (!BuildHFMetaLists(memory_manager, frame, tree, width, height, ec)) {
      return JXL_FAILURE("Invalid ACSTree or QFTree");
    }
    CombineVarDCTTrees(frame, width, height, ec, &tree);
    return true;
  };
  JXL_RETURN_IF_ERROR(
      combine_sections(io->metadata.m.num_extra_channels > 0 ||
                       cparams.move_to_front_from_channel < -1));

  if (tree_out) {
    PrintTree(tree, tree_out);
  }
  // The encoder does not use the pixels, but it looks at them (e.g. for the
  // chroma adjustments of VarDCT frames): they must not be uninitialized.
  JXL_ASSIGN_OR_RETURN(Image3F image,
                       Image3F::Create(memory_manager, width, height));
  jxl::ZeroFillImage(&image);
  // Extra channels have the size of the frame (Alpha may come before Width
  // and Height, when their size was not known yet).
  for (ImageF& ec : io->frames[0].extra_channels()) {
    if (ec.xsize() != width || ec.ysize() != height) {
      JXL_ASSIGN_OR_RETURN(ec, ImageF::Create(memory_manager, width, height));
      jxl::ZeroFillImage(&ec);
    }
  }
  JXL_RETURN_IF_ERROR(
      io->SetFromImage(std::move(image), io->metadata.m.color_encoding));
  if (frame.image_xsize) {
    JXL_RETURN_IF_ERROR(io->SetSize(frame.image_xsize, frame.image_ysize));
  } else {
    JXL_RETURN_IF_ERROR(io->SetSize((width + x0), (height + y0)));
  }

  io->metadata.m.color_encoding.DecideIfWantICC(*JxlGetDefaultCms());
  cparams.options.zero_tokens = true;
  cparams.palette_colors = 0;
  cparams.channel_colors_pre_transform_percent = 0;
  cparams.channel_colors_percent = 0;
  cparams.patches = jxl::Override::kOff;
  cparams.already_downsampled = true;
  if (!CheckTreeSplits(tree, "tree")) {
    return JXL_FAILURE("Invalid tree");
  }
  cparams.custom_fixed_tree = tree;

  std::vector<QuantizedSpline> quantized_splines;
  std::vector<Spline::Point> starting_points;
  JXL_RETURN_IF_ERROR(
      SplinesFromSplineData(spline_data, quantized_splines, starting_points));
  cparams.custom_splines = {
      Span<const QuantizedSpline>(quantized_splines),
      Span<const Spline::Point>(starting_points),
      spline_data.signal_adjustment ? spline_data.quantization_adjustment : 0};
  PaddedBytes compressed{memory_manager};

  JXL_RETURN_IF_ERROR(io->CheckMetadata());
  BitWriter writer{memory_manager};

  std::unique_ptr<CodecMetadata> metadata = jxl::make_unique<CodecMetadata>();
  *metadata = io->metadata;
  // An explicit ImageSize is the final image size; otherwise the first frame's
  // (upsampled) size is.
  size_t image_ups = frame.image_xsize ? 1 : cparams.resampling;
  JXL_RETURN_IF_ERROR(
      metadata->size.Set(io->xsize() * image_ups, io->ysize() * image_ups));

  metadata->m.xyb_encoded = (cparams.color_transform == ColorTransform::kXYB);

  if (cparams.move_to_front_from_channel < -1) {
    size_t nch = -1 - cparams.move_to_front_from_channel;
    cparams.move_to_front_from_channel = 3 + metadata->m.num_extra_channels;
    metadata->m.num_extra_channels += nch;
    for (size_t _ = 0; _ < nch; _++) {
      metadata->m.extra_channel_info.emplace_back();
      auto& eci = metadata->m.extra_channel_info.back();
      eci.type = jxl::ExtraChannel::kOptional;
      JXL_ASSIGN_OR_RETURN(ImageF ch,
                           ImageF::Create(memory_manager, width, height));
      jxl::ZeroFillImage(&ch);
      io->frames[0].extra_channels().emplace_back(std::move(ch));
    }
  }

  JXL_RETURN_IF_ERROR(WriteCodestreamHeaders(metadata.get(), &writer, nullptr));
  writer.ZeroPadToByte();

  bool warned_group_shift = false;
  size_t frame_index = 0;
  SlotState slots[jxl::kMaxNumReferenceFrames] = {};
  // The size of the frame in each slot (the upsampled size).
  size_t slot_xsize[jxl::kMaxNumReferenceFrames] = {};
  size_t slot_ysize[jxl::kMaxNumReferenceFrames] = {};
  bool noted_upsampled_patches = false;
  int last_after_ct = -1;
  while (true) {
    FrameInfo info;
    info.is_last = !FROM_JXL_BOOL(have_next);
    if (!info.is_last) info.save_as_reference = 1;

    io->frames[0].origin.x0 = x0;
    io->frames[0].origin.y0 = y0;
    info.clamp = false;
    if (frame.save_as_reference >= 0) {
      info.save_as_reference = frame.save_as_reference;
    }
    if (frame.reference_only) {
      if (info.is_last) {
        fprintf(stderr,
                "The last frame cannot be ReferenceOnly (it is not displayed): "
                "give it NotLast and add a frame that uses it\n");
        return JXL_FAILURE("The last frame cannot be ReferenceOnly");
      }
      info.frame_type = jxl::FrameType::kReferenceOnly;
      info.save_before_color_transform = true;
    }
    if (frame.save_before_ct) info.save_before_color_transform = true;
    info.source = frame.blend_source;
    if (!frame.reference_only) {
      if (!frame.blend_source_given) {
        info.source = DefaultBlendSource(slots, last_after_ct);
      }
      if (slots[info.source] == SlotState::kBeforeCT &&
          FrameNeedsBlending(io->frames[0], *metadata,
                             width * cparams.resampling,
                             height * cparams.resampling)) {
        fprintf(stderr,
                "This frame is blended onto reference slot %zu, which holds a "
                "frame saved before the color transform (ReferenceOnly or "
                "SaveBeforeCT): decoders only take patches from such a slot. "
                "Use BlendSource with another slot.\n",
                info.source);
        return JXL_FAILURE("Invalid blend source");
      }
      // The extra channels' blending, with the frame's blend source (the
      // encoder's default would use slot 1, and blend a lone hidden channel as
      // if it were the alpha channel).
      info.extra_channel_blending_info =
          DefaultEcBlending(io->frames[0], *metadata);
      for (jxl::BlendingInfo& ec : info.extra_channel_blending_info) {
        ec.source = info.source;
      }
    }
    if (cparams.vardct_from_tree && frame.group_shift_given &&
        !warned_group_shift) {
      // The frame header has a group size only for modular frames.
      fprintf(stderr,
              "Note: GroupShift does not apply to VarDCT frames, which always "
              "have 256x256 groups (and 2048x2048 LF groups)\n");
      warned_group_shift = true;
    }
    if (!frame.vardct_keyword.empty() && !cparams.vardct_from_tree) {
      fprintf(stderr,
              "%s needs VarDCT (in a modular frame, the VarDCT settings are "
              "kept for later VarDCT frames, but cannot be given)\n",
              frame.vardct_keyword.c_str());
      return JXL_FAILURE("VarDCT keyword in a modular frame");
    }
    if (!CheckDequantTables(memory_manager, cparams.custom_fixed_tree, width,
                            height, cparams)) {
      return JXL_FAILURE("Invalid dequantization table");
    }
    if (frame.have_hf && cparams.vardct_from_tree) {
      if (!SetHFContexts(frame.hf, memory_manager, cparams.custom_fixed_tree,
                         width, height, cparams)) {
        return JXL_FAILURE("Invalid HF context model");
      }
    } else if (cparams.vardct_from_tree) {
      // Check the HF metadata (with messages).
      FrameDimensions frame_dim;
      frame_dim.Set(width, height, /*group_size_shift=*/1, /*max_hshift=*/0,
                    /*max_vshift=*/0, /*modular_mode=*/false,
                    /*upsampling=*/1);
      VarDCTBlocks blocks;
      if (!ComputeVarDCTBlocks(memory_manager, cparams.custom_fixed_tree,
                               frame_dim, jxl::BlockCtxMap(), &blocks)) {
        return JXL_FAILURE("Invalid HF metadata");
      }
    }
    // Patches are drawn before the upsampling (in libjxl and jxl-rs alike):
    // their position and size are in coded pixels of this frame, and they copy
    // their rectangle of the reference frame 1:1, at the reference frame's own
    // (upsampled) resolution.
    const size_t ups = cparams.resampling;
    uint64_t patch_area = 0;
    for (const auto& p : cparams.custom_patches) {
      if (p.x + p.xsize > width || p.y + p.ysize > height) {
        fprintf(stderr,
                "Patch at (%zu, %zu) of %zux%zu does not fit in the %zux%zu "
                "frame\n",
                p.x, p.y, p.xsize, p.ysize, width, height);
        if (ups > 1) {
          fprintf(stderr,
                  "(With Upsample %zu, patches are drawn before the "
                  "upsampling: position and size are in coded pixels, and the "
                  "rectangle of the reference frame is copied 1:1 and then "
                  "upsampled.)\n",
                  ups);
        }
        return JXL_FAILURE("Patch outside the frame");
      }
      if (slots[p.ref] != SlotState::kBeforeCT) {
        fprintf(stderr,
                "Patch from reference slot %zu, which holds %s: patches need "
                "a frame saved before the color transform (ReferenceOnly, or "
                "SaveBeforeCT on a full kReplace frame)\n",
                p.ref,
                slots[p.ref] == SlotState::kEmpty
                    ? "no frame"
                    : "a frame saved after the color transform");
        return JXL_FAILURE("Invalid patch reference");
      }
      if (p.x0 + p.xsize > slot_xsize[p.ref] ||
          p.y0 + p.ysize > slot_ysize[p.ref]) {
        fprintf(stderr,
                "Patch rectangle at (%zu, %zu) of %zux%zu does not fit in the "
                "%zux%zu frame in reference slot %zu\n",
                p.x0, p.y0, p.xsize, p.ysize, slot_xsize[p.ref],
                slot_ysize[p.ref], p.ref);
        return JXL_FAILURE("Patch outside the reference frame");
      }
      patch_area += static_cast<uint64_t>(p.xsize) * p.ysize;
    }
    if (!cparams.custom_patches.empty() && ups > 1) {
      if (metadata->m.num_extra_channels > 0 && cparams.ec_resampling != ups) {
        return FailWithMessage(
            "Patches in an upsampled frame with extra channels need "
            "Upsample_EC equal to Upsample (libjxl refuses the file)");
      }
      if (!noted_upsampled_patches) {
        fprintf(stderr,
                "Note: patches in an Upsample %zu frame are drawn before the "
                "upsampling: position and size are in coded pixels, and the "
                "rectangle of the reference frame (at its full resolution) is "
                "copied 1:1 and then upsampled.\n",
                ups);
        noted_upsampled_patches = true;
      }
    }
    {
      // jxl-rs applies the level 5 limit on the total patch area, for the
      // padded coded frame size.
      const uint64_t padded_pixels =
          cparams.vardct_from_tree
              ? static_cast<uint64_t>(jxl::DivCeil(width, 8) * 8) *
                    (jxl::DivCeil(height, 8) * 8)
              : static_cast<uint64_t>(width) * height;
      const uint64_t limit = std::max<uint64_t>(8 * padded_pixels, 1 << 20);
      if (patch_area > limit) {
        fprintf(stderr,
                "Warning: the patches of frame %zu cover %" PRIu64
                " pixels, over the level 5 limit of %" PRIu64
                " (8 x coded pixels, at least 2^20): jxl-rs refuses the file "
                "(libjxl decodes it).\n",
                frame_index, patch_area, limit);
      }
    }

    {
      // The quantization adjustment that the encoder writes.
      Splines splines(memory_manager);
      splines.SetData(cparams.custom_splines);
      JXL_RETURN_IF_ERROR(CheckSplineLimits(quantized_splines, starting_points,
                                            splines.GetQuantizationAdjustment(),
                                            width, height, cparams.resampling,
                                            frame_index));
    }
    // Decoded range of each channel (for the alpha check below).
    std::vector<std::pair<int64_t, int64_t>> channel_ranges;
    // (Displayed frames only: a reference-only sheet with out-of-range alpha
    // is fine when its patches are clamped or blended onto opaque pixels.)
    const bool check_alpha = metadata->m.HasAlpha() && !frame.reference_only;
    cparams.modular_channel_ranges_out =
        check_alpha ? &channel_ranges : nullptr;
    const auto encode_frame = [&](BitWriter* w) {
      return jxl::EncodeFrame(memory_manager, cparams, info, metadata.get(),
                              io->frames[0], *JxlGetDefaultCms(), nullptr, w,
                              StatsAuxOut());
    };
    if (!cparams.custom_residuals.empty() && !palette_inline_tree) {
      // Residual patterns with LZ77 inside the listed values or without, and
      // with its matches picked from two rough cost estimates: keep the
      // smallest. The plain encoding goes first: a second EncodeFrame of the
      // same frame can come out differently (see TODO.md), and this way it is
      // the same as without the choice.
      BitWriter best(memory_manager);
      cparams.options.residual_inner_lz77 = false;
      if (!encode_frame(&best)) {
        fprintf(stderr, "Failed to encode frame %zu\n", frame_index);
        return JXL_FAILURE("Failed to encode frame");
      }
      if (!cparams.flat_nibble_code) {
        cparams.options.residual_inner_lz77 = true;
        for (const bool context_costs : {false, true}) {
          BitWriter with(memory_manager);
          cparams.options.residual_lz77_context_costs = context_costs;
          if (!encode_frame(&with)) {
            fprintf(stderr, "Failed to encode frame %zu\n", frame_index);
            return JXL_FAILURE("Failed to encode frame");
          }
          if (with.BitsWritten() < best.BitsWritten()) best = std::move(with);
        }
        cparams.options.residual_lz77_context_costs = false;
      }
      cparams.options.residual_inner_lz77 = true;
      JXL_RETURN_IF_ERROR(writer.AppendUnaligned(best));
    } else if (palette_inline_tree) {
      // Palette entries coded as pixels, or in the tree: keep the smaller.
      BitWriter coded(memory_manager);
      BitWriter inlined(memory_manager);
      if (!encode_frame(&coded)) {
        fprintf(stderr, "Failed to encode frame %zu\n", frame_index);
        return JXL_FAILURE("Failed to encode frame");
      }
      const Tree coded_tree = cparams.custom_fixed_tree;
      const auto entries = cparams.custom_palette.entries;
      cparams.custom_fixed_tree = *palette_inline_tree;
      cparams.custom_palette.entries.clear();
      cparams.options.code_meta_channels = false;
      if (!encode_frame(&inlined)) {
        fprintf(stderr, "Failed to encode frame %zu\n", frame_index);
        return JXL_FAILURE("Failed to encode frame");
      }
      const bool use_inlined = inlined.BitsWritten() <= coded.BitsWritten();
      if (!use_inlined) {
        cparams.custom_fixed_tree = coded_tree;
        cparams.custom_palette.entries = entries;
        cparams.options.code_meta_channels = true;
      }
      JXL_RETURN_IF_ERROR(
          writer.AppendUnaligned(use_inlined ? inlined : coded));
    } else if (!encode_frame(&writer)) {
      fprintf(stderr, "Failed to encode frame %zu\n", frame_index);
      return JXL_FAILURE("Failed to encode frame");
    }
    cparams.modular_channel_ranges_out = nullptr;
    // Alpha outside 0..max looks fine in 8-bit checks (PNG output clamps it)
    // but some browsers (Chrome) premultiply without clamping.
    const size_t num_ec = metadata->m.extra_channel_info.size();
    for (size_t k = 0;
         check_alpha && k < num_ec && channel_ranges.size() >= num_ec; k++) {
      const auto& eci = metadata->m.extra_channel_info[k];
      if (eci.type != jxl::ExtraChannel::kAlpha) continue;
      const auto& r = channel_ranges[channel_ranges.size() - num_ec + k];
      const int64_t max = (int64_t{1} << eci.bit_depth.bits_per_sample) - 1;
      if (r.first < 0 || r.second > max) {
        fprintf(stderr,
                "Warning: frame %zu: alpha goes from %lld to %lld, outside "
                "0..%lld (browsers that premultiply without clamping, like "
                "Chrome, will show wrong colours)\n",
                frame_index, static_cast<long long>(r.first),
                static_cast<long long>(r.second), static_cast<long long>(max));
      }
    }
    frame_index++;
    // What the reference slots hold now (as in FrameHeader::CanBeReferenced).
    const bool has_duration =
        metadata->m.have_animation && !frame.reference_only;
    if (!info.is_last && (!has_duration || io->frames[0].duration == 0 ||
                          info.save_as_reference != 0)) {
      // A displayed frame can be saved before the color transform only as a
      // full frame with kReplace (otherwise the flag is not signaled).
      const bool partial =
          IsPartialFrame(io->frames[0], *metadata, width * cparams.resampling,
                         height * cparams.resampling);
      const bool replace = !io->frames[0].blend ||
                           io->frames[0].blendmode == BlendMode::kReplace;
      const bool before_ct =
          frame.reference_only ||
          (info.save_before_color_transform && replace && !partial);
      const size_t slot = info.save_as_reference;
      slots[slot] = before_ct ? SlotState::kBeforeCT : SlotState::kAfterCT;
      if (!before_ct) last_after_ct = slot;
      // A ReferenceOnly frame is saved at its own (upsampled) size, a
      // displayed frame at the image size (after blending).
      slot_xsize[slot] =
          frame.reference_only ? width * cparams.resampling : metadata->xsize();
      slot_ysize[slot] = frame.reference_only ? height * cparams.resampling
                                              : metadata->ysize();
    }
    if (!have_next) break;
    tree.clear();
    spline_data.splines.clear();
    cparams.custom_patches.clear();
    cparams.custom_residuals.clear();
    frame.save_as_reference = -1;
    frame.reference_only = false;
    frame.save_before_ct = false;
    frame.vardct_keyword.clear();
    have_next = JXL_FALSE;
    cparams.manual_noise.clear();
    frame.lf_tree.clear();
    frame.hf_meta_tree.clear();
    frame.extra_tree.clear();
    // RAW dequantization tables come from the frame's tree.
    frame.dequant_trees.clear();
    for (auto& table : cparams.vardct_dequant) table.raw_den = 0;
    const int hidden_before = cparams.move_to_front_from_channel;
    const size_t extra_before = io->metadata.m.num_extra_channels;
    const bool alpha_before = io->metadata.m.HasAlpha();
    if (!ParseNode(tok, tree, spline_data, frame, cparams, width, height, *io,
                   have_next, x0, y0, buffer_size)) {
      return JXL_FAILURE("Failed to ParseNode");
    }
    if (cparams.move_to_front_from_channel != hidden_before ||
        io->metadata.m.num_extra_channels != extra_before ||
        io->metadata.m.HasAlpha() != alpha_before) {
      return FailWithMessage(
          "HiddenChannel and Alpha must be given in the first frame (extra "
          "channels are the same for all frames)");
    }
    JXL_RETURN_IF_ERROR(combine_sections(metadata->m.num_extra_channels > 0));
    if (!CheckTreeSplits(tree, "tree")) {
      return JXL_FAILURE("Invalid tree");
    }
    cparams.custom_fixed_tree = tree;
    // This frame's own splines (previously the first frame's were reused).
    JXL_RETURN_IF_ERROR(
        SplinesFromSplineData(spline_data, quantized_splines, starting_points));
    cparams.custom_splines = {Span<const QuantizedSpline>(quantized_splines),
                              Span<const Spline::Point>(starting_points),
                              spline_data.signal_adjustment
                                  ? spline_data.quantization_adjustment
                                  : 0};
    // Extra channels (alpha, hidden channels) have the size of this frame.
    for (ImageF& ec : io->frames[0].extra_channels()) {
      if (ec.xsize() != width || ec.ysize() != height) {
        JXL_ASSIGN_OR_RETURN(ec, ImageF::Create(memory_manager, width, height));
        jxl::ZeroFillImage(&ec);
      }
    }
    JXL_ASSIGN_OR_RETURN(Image3F image,
                         Image3F::Create(memory_manager, width, height));
    jxl::ZeroFillImage(&image);
    JXL_RETURN_IF_ERROR(
        io->SetFromImage(std::move(image), ColorEncoding::SRGB()));
    io->frames[0].blend = true;
  }

  compressed = std::move(writer).TakeBytes();

  if (auto16 && (modular_range[0] < -32768 || modular_range[1] > 32767)) {
    *needs_32bit = true;
    return true;
  }
  if (!WriteFile(out, compressed)) {
    fprintf(stderr, "Failed to write to \"%s\"\n", out);
    return JXL_FAILURE("Failed to write output");
  }

  return true;
}
::jxl::Status JxlFromTree(const char* in, const char* out,
                          const char* tree_out) {
  // The input is read twice if 32-bit buffers turn out to be needed.
  std::stringstream text;
  if (strcmp(in, "-") > 0) {
    std::ifstream file(in, std::ifstream::in);
    text << file.rdbuf();
  } else {
    text << std::cin.rdbuf();
  }
  const std::string source = text.str();
  bool needs_32bit = false;
  {
    std::istringstream input(source);
    JXL_RETURN_IF_ERROR(JxlFromTreePass(input, out, tree_out,
                                        /*auto_16bit=*/true, &needs_32bit));
  }
  if (!needs_32bit) return true;
  std::istringstream input(source);
  return JxlFromTreePass(input, out, tree_out, /*auto_16bit=*/false,
                         &needs_32bit);
}

}  // namespace tools
}  // namespace jpegxl

// JXL_FROM_TREE_STATS=1: the encoded bytes per bitstream layer (summed over all
// frames), on stderr. Only for looking: with a choice between encodings of a
// frame, every encoding tried is counted.
jxl::AuxOut* StatsAuxOut() {
  static jxl::AuxOut stats;
  static const bool enabled = getenv("JXL_FROM_TREE_STATS") != nullptr;
  return enabled ? &stats : nullptr;
}

void PrintStats() {
  jxl::AuxOut* stats = StatsAuxOut();
  if (!stats) return;
  for (size_t i = 0; i < jxl::kNumImageLayers; i++) {
    const jxl::LayerType l = static_cast<jxl::LayerType>(i);
    if (stats->layer(l).total_bits == 0) continue;
    fprintf(stderr, "%-24s %10.1f B (histograms %8.1f B)\n", jxl::LayerName(l),
            stats->layer(l).total_bits / 8.0,
            stats->layer(l).histogram_bits / 8.0);
  }
}

int main(int argc, char** argv) {
  if ((argc != 3 && argc != 4) ||
      ((strcmp(argv[1], "-") > 0) && !strcmp(argv[1], argv[2]))) {
    fprintf(stderr, "Usage: %s tree_in.txt out.jxl [tree_drawing]\n", argv[0]);
    return 1;
  }
  jxl::Status result = jpegxl::tools::JxlFromTree(argv[1], argv[2],
                                                  argc < 4 ? nullptr : argv[3]);
  if (!result) {
    fprintf(stderr, "FAILURE\n");
    return EXIT_FAILURE;
  }
  PrintStats();
  return EXIT_SUCCESS;
}
