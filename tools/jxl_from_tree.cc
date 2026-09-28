// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
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
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_debug_tree.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/noise.h"
#include "lib/jxl/pack_signed.h"
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
using ::jxl::Tree;

namespace {
struct SplineData {
  int32_t quantization_adjustment = 1;
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
    int32_t split = std::stoi(t, &num);
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
    int32_t value = std::stoi(t, &num);
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

// Builds the block context map and the fixed HF tokens for cparams.
bool SetHFContexts(const HFContextSettings& hf, CompressParams& cparams) {
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
  map.num_ctxs = *std::max_element(map.ctx_map.begin(), map.ctx_map.end()) + 1;
  // The block context map is coded as a context map, which has to use every
  // value below its maximum.
  for (uint32_t ctx = 0; ctx < map.num_ctxs; ctx++) {
    if (std::find(map.ctx_map.begin(), map.ctx_map.end(), ctx) ==
        map.ctx_map.end()) {
      fprintf(stderr,
              "Block context %u is not used: block contexts have to be "
              "0..n-1 without gaps\n",
              ctx);
      return false;
    }
  }
  cparams.use_custom_block_ctx_map = true;

  if (hf.coefficients.nodes.empty()) return true;
  // libjxl merges some (k, nzleft) pairs (and nzpred values) into one
  // context. The first value assigned to a context wins: predicted counts in
  // increasing order, coefficients densest first (k + nzleft = 64, the pairs
  // of a block with all coefficients nonzero, then 63, ...).
  std::vector<int64_t> tokens(map.NumACContexts(), -1);
  size_t overridden = 0;
  auto assign = [&](size_t ctx, uint32_t token) {
    if (tokens[ctx] < 0) {
      tokens[ctx] = token;
    } else if (tokens[ctx] != token) {
      overridden++;
    }
  };
  for (uint32_t bctx = 0; bctx < map.num_ctxs; bctx++) {
    for (uint32_t nz = 0; nz <= 64; nz++) {
      int32_t v = hf.coefficients.Eval(
          {0, static_cast<int32_t>(bctx), static_cast<int32_t>(nz), 0, 0, 0});
      if (v < 0 || v > 255) {
        fprintf(stderr, "Number of nonzeros %d is not in 0..255\n", v);
        return false;
      }
      assign(map.NonZeroContext(nz, bctx), v);
    }
    for (uint32_t sum = 64; sum >= 2; sum--) {
      for (uint32_t k = 1; k < sum && k < 64; k++) {
        uint32_t nzleft = sum - k;
        if (nzleft >= 64) continue;
        for (uint32_t prev = 0; prev < 2; prev++) {
          int32_t v = hf.coefficients.Eval(
              {1, static_cast<int32_t>(bctx), 0, static_cast<int32_t>(k),
               static_cast<int32_t>(nzleft), static_cast<int32_t>(prev)});
          uint32_t token = jxl::PackSigned(v);
          if (token > 255) {
            fprintf(stderr, "Coefficient %d is not in -128..127\n", v);
            return false;
          }
          assign(map.ZeroDensityContextsOffset(bctx) +
                     jxl::ZeroDensityContext(nzleft, k, 1, 0, prev),
                 token);
        }
      }
    }
  }
  if (overridden) {
    fprintf(stderr,
            "Note: %zu (k, nzleft, prev) or nzpred cases share a context with "
            "an earlier case that has another value (libjxl merges them); the "
            "earlier value is used\n",
            overridden);
  }
  cparams.custom_hf_tokens.resize(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    cparams.custom_hf_tokens[i] = tokens[i] < 0 ? 0 : tokens[i];
  }
  return true;
}

// Per-frame settings besides the tree. Patches go to cparams.custom_patches.
struct FrameSettings {
  // Reference slot to save this frame to; -1 = default (1 if not last).
  int save_as_reference = -1;
  // kReferenceOnly frame (not displayed; saved before the color transform, so
  // usable as a patch source).
  bool reference_only = false;
  // Save a regular frame before the color transform (needed to use it as a
  // patch source; only allowed for kReplace, full-frame blending).
  bool save_before_ct = false;
  // Reference slot this frame is blended onto (persists).
  size_t blend_source = 1;
  // Patch blend mode for extra channels of subsequent patches (persists).
  uint8_t patch_ec_mode = static_cast<uint8_t>(jxl::PatchBlendMode::kNone);
  // Whether subsequent patches clamp (kMul and alpha blend modes; persists).
  bool patch_clamp = false;
  // HF context model of VarDCT frames (persists).
  HFContextSettings hf;
  bool have_hf = false;
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
    int split = std::stoi(t, &num);
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
    int offset = std::stoi(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid offset: %s\n", t.c_str());
      return false;
    }
    if (subtract) offset = -offset;
    tree.emplace_back(PropertyDecisionNode::Leaf(p, offset));
    return true;
  } else if (t == "Width") {
    t = tok();
    size_t num = 0;
    W = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid width: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Height") {
    t = tok();
    size_t num = 0;
    H = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid height: %s\n", t.c_str());
      return false;
    }
  } else if (t == "/*") {
    t = tok();
    while (t != "*/" && t != "") t = tok();
  } else if (t == "Squeeze") {
    cparams.responsive = true;
  } else if (t == "GroupShift") {
    t = tok();
    size_t num = 0;
    cparams.modular_group_size_shift = std::stoul(t, &num);
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
      size_t n = std::stoul(t, &num);
      if (num != t.size() || n > 15) {
        fprintf(stderr, "Invalid number of thresholds (max 15): %s\n",
                t.c_str());
        return false;
      }
      std::vector<int> v(n);
      for (int& i : v) {
        t = tok();
        i = std::stoi(t, &num);
        if (num != t.size()) {
          fprintf(stderr, "Invalid threshold: %s\n", t.c_str());
          return false;
        }
      }
      if (lists == 3) {
        frame.hf.lf_thresholds[c] = v;
      } else {
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
  } else if (t == "CoeffOrder") {
    // CoeffOrder <order class> [Y|X|B] <n> <u1> <v1> ... <un> <vn>: in the
    // coefficient order of that class (0..12, or a DCT name like DCT64; for
    // all channels unless one is given), coefficients (u, v) come right after
    // the LLF ones, so the first of them is at scan index covered_blocks; the
    // other coefficients keep the default order. u and v are the horizontal and
    // vertical frequency in the square or tall transform of the class (like
    // DCT16X8, which is 8 wide and 16 tall); in the wide one (DCT8X16), the
    // same entry is frequency (v, u).
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
      ord = std::stoul(t, &num);
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
    size_t n = std::stoul(t, &num);
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
    std::vector<uint32_t> positions;
    for (size_t i = 0; i < n; i++) {
      size_t uv[2];
      for (size_t& v : uv) {
        t = tok();
        v = std::stoul(t, &num);
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
    cparams.custom_coeff_orders.resize(3 * jxl::kNumOrders);
    for (size_t c : channels) {
      cparams.custom_coeff_orders[3 * ord + c] = positions;
    }
  } else if (t == "VarDCT") {
    // A VarDCT frame: the tree defines the LF image (stream IDs of the VarDCT
    // DC groups; channels Y, X, B of quantized LF) and the HF metadata (stream
    // IDs of the AC metadata groups; channels YtoX, YtoB, AC strategy + quant
    // field, EPF sharpness); all HF coefficients are zero.
    cparams.modular_mode = false;
    cparams.vardct_from_tree = true;
    // For the default loop filters (Gaborish, EPF) of VarDCT frames.
    cparams.butteraugli_distance = 1.0f;
  } else if (t == "HiddenChannel") {
    t = tok();
    size_t num = 0;
    cparams.move_to_front_from_channel = -1 - std::stoul(t, &num);
    if (num != t.size() || num > 16) {
      fprintf(stderr, "Invalid HiddenChannel (max 16): %s\n", t.c_str());
      return false;
    }
  } else if (t == "RCT") {
    t = tok();
    size_t num = 0;
    cparams.colorspace = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid RCT: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Orientation") {
    t = tok();
    size_t num = 0;
    io.metadata.m.orientation = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid Orientation: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Alpha") {
    io.metadata.m.SetAlphaBits(io.metadata.m.bit_depth.bits_per_sample);
    JXL_ASSIGN_OR_RETURN(
        ImageF alpha, ImageF::Create(jpegxl::tools::NoMemoryManager(), W, H));
    if (!io.frames[0].SetAlpha(std::move(alpha))) {
      fprintf(stderr, "Internal: SetAlpha failed\n");
      return false;
    }
  } else if (t == "Bitdepth") {
    t = tok();
    size_t num = 0;
    uint32_t bits_per_sample = std::stoul(t, &num);
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
    io.metadata.m.bit_depth.exponent_bits_per_sample = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid FloatExpBits: %s\n", t.c_str());
      return false;
    }
  } else if (t == "FramePos") {
    t = tok();
    size_t num = 0;
    x0 = std::stoi(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid FramePos x0: %s\n", t.c_str());
      return false;
    }
    t = tok();
    y0 = std::stoi(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid FramePos y0: %s\n", t.c_str());
      return false;
    }
  } else if (t == "NotLast") {
    have_next = JXL_TRUE;
  } else if (t == "Upsample") {
    t = tok();
    size_t num = 0;
    cparams.resampling = std::stoul(t, &num);
    if (num != t.size() ||
        (cparams.resampling != 1 && cparams.resampling != 2 &&
         cparams.resampling != 4 && cparams.resampling != 8)) {
      fprintf(stderr, "Invalid Upsample: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Upsample_EC") {
    t = tok();
    size_t num = 0;
    cparams.ec_resampling = std::stoul(t, &num);
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
    io.metadata.m.animation.tps_numerator = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid numerator: %s\n", t.c_str());
      return false;
    }
    t = tok();
    num = 0;
    io.metadata.m.animation.tps_denominator = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid denominator: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Duration") {
    t = tok();
    size_t num = 0;
    io.frames[0].duration = std::stoul(t, &num);
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
  } else if (t == "SplineQuantizationAdjustment") {
    t = tok();
    size_t num = 0;
    spline_data.quantization_adjustment = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid SplineQuantizationAdjustment: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Spline") {
    Spline spline;
    const auto ParseFloat = [&t, &tok](float& output) {
      t = tok();
      size_t num = 0;
      output = std::stof(t, &num);
      if (num != t.size()) {
        fprintf(stderr, "Invalid spline data: %s\n", t.c_str());
        return false;
      }
      return true;
    };
    for (auto& dct : spline.color_dct) {
      for (float& coefficient : dct) {
        JXL_RETURN_IF_ERROR(ParseFloat(coefficient));
      }
    }
    for (float& coefficient : spline.sigma_dct) {
      JXL_RETURN_IF_ERROR(ParseFloat(coefficient));
    }

    while (true) {
      t = tok();
      if (t == "EndSpline") break;
      size_t num = 0;
      Spline::Point point;
      point.x = std::stof(t, &num);
      bool ok_x = num == t.size();
      auto t_y = tok();
      point.y = std::stof(t_y, &num);
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

    spline_data.splines.push_back(std::move(spline));
  } else if (t == "Gaborish") {
    cparams.gaborish = jxl::Override::kOn;
  } else if (t == "NoGaborish") {
    // Gaborish is on by default in VarDCT frames.
    cparams.gaborish = jxl::Override::kOff;
  } else if (t == "DeltaPalette") {
    cparams.lossy_palette = true;
    cparams.palette_colors = 0;
  } else if (t == "EPF") {
    t = tok();
    size_t num = 0;
    cparams.epf = std::stoul(t, &num);
    if (num != t.size() || cparams.epf > 3) {
      fprintf(stderr, "Invalid EPF: %s\n", t.c_str());
      return false;
    }
  } else if (t == "Noise") {
    cparams.manual_noise.resize(8);
    for (size_t i = 0; i < 8; i++) {
      t = tok();
      size_t num = 0;
      float v = std::stof(t, &num);
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
      cparams.manual_xyb_factors[i] = std::stof(t, &num);
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
      *v = std::stoul(t, &num);
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
    size_t slot = std::stoul(t, &num);
    if (num != t.size() || slot >= jxl::kMaxNumReferenceFrames) {
      fprintf(stderr, "Invalid reference slot: %s\n", t.c_str());
      return false;
    }
    if (save) {
      frame.save_as_reference = static_cast<int>(slot);
    } else {
      frame.blend_source = slot;
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
      v[i] = std::stoul(t, &num);
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
}  // namespace

::jxl::Status JxlFromTree(const char* in, const char* out,
                          const char* tree_out) {
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

  std::istream* f = &std::cin;
  std::ifstream file;

  if (strcmp(in, "-") > 0) {
    file.open(in, std::ifstream::in);
    f = &file;
  }

  auto tok = [&f]() {
    std::string out;
    *f >> out;
    return out;
  };
  if (!ParseNode(tok, tree, spline_data, frame, cparams, width, height, *io,
                 have_next, x0, y0, buffer_size)) {
    return JXL_FAILURE("Failed to ParseNode");
  }

  // Auto 16bit for multi-frame art would mean parsing all frames first,
  // so for now just default to 32bit buffers instead.
  if (buffer_size == 1 || (buffer_size == 3 && !have_next)) {
    io->metadata.m.modular_16_bit_buffer_sufficient = true;
  }

  if (tree_out) {
    PrintTree(tree, tree_out);
  }
  JXL_ASSIGN_OR_RETURN(Image3F image,
                       Image3F::Create(memory_manager, width, height));
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
  cparams.custom_fixed_tree = tree;

  std::vector<QuantizedSpline> quantized_splines;
  std::vector<Spline::Point> starting_points;
  JXL_RETURN_IF_ERROR(
      SplinesFromSplineData(spline_data, quantized_splines, starting_points));
  cparams.custom_splines = {Span<const QuantizedSpline>(quantized_splines),
                            Span<const Spline::Point>(starting_points)};
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
      io->frames[0].extra_channels().emplace_back(std::move(ch));
    }
  }

  JXL_RETURN_IF_ERROR(WriteCodestreamHeaders(metadata.get(), &writer, nullptr));
  writer.ZeroPadToByte();

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
        return JXL_FAILURE("The last frame cannot be ReferenceOnly");
      }
      info.frame_type = jxl::FrameType::kReferenceOnly;
      info.save_before_color_transform = true;
    }
    if (frame.save_before_ct) info.save_before_color_transform = true;
    info.source = frame.blend_source;
    if (!cparams.custom_coeff_orders.empty() && !cparams.vardct_from_tree) {
      return JXL_FAILURE("CoeffOrder needs VarDCT");
    }
    if (frame.have_hf) {
      if (!cparams.vardct_from_tree) {
        return JXL_FAILURE("HF context keywords need VarDCT");
      }
      if (!SetHFContexts(frame.hf, cparams)) {
        return JXL_FAILURE("Invalid HF context model");
      }
    }
    for (const auto& p : cparams.custom_patches) {
      if (p.x + p.xsize > width || p.y + p.ysize > height) {
        fprintf(stderr,
                "Patch at (%zu, %zu) does not fit in the %zux%zu frame\n", p.x,
                p.y, width, height);
        return JXL_FAILURE("Patch outside the frame");
      }
    }

    JXL_RETURN_IF_ERROR(jxl::EncodeFrame(
        memory_manager, cparams, info, metadata.get(), io->frames[0],
        *JxlGetDefaultCms(), nullptr, &writer, nullptr));
    if (!have_next) break;
    tree.clear();
    spline_data.splines.clear();
    cparams.custom_patches.clear();
    frame.save_as_reference = -1;
    frame.reference_only = false;
    frame.save_before_ct = false;
    have_next = JXL_FALSE;
    cparams.manual_noise.clear();
    if (!ParseNode(tok, tree, spline_data, frame, cparams, width, height, *io,
                   have_next, x0, y0, buffer_size)) {
      return JXL_FAILURE("Failed to ParseNode");
    }
    cparams.custom_fixed_tree = tree;
    // This frame's own splines (previously the first frame's were reused).
    JXL_RETURN_IF_ERROR(
        SplinesFromSplineData(spline_data, quantized_splines, starting_points));
    cparams.custom_splines = {Span<const QuantizedSpline>(quantized_splines),
                              Span<const Spline::Point>(starting_points)};
    // Extra channels (alpha, hidden channels) have the size of this frame.
    for (ImageF& ec : io->frames[0].extra_channels()) {
      if (ec.xsize() != width || ec.ysize() != height) {
        JXL_ASSIGN_OR_RETURN(ec, ImageF::Create(memory_manager, width, height));
      }
    }
    JXL_ASSIGN_OR_RETURN(Image3F image,
                         Image3F::Create(memory_manager, width, height));
    JXL_RETURN_IF_ERROR(
        io->SetFromImage(std::move(image), ColorEncoding::SRGB()));
    io->frames[0].blend = true;
  }

  compressed = std::move(writer).TakeBytes();

  if (!WriteFile(out, compressed)) {
    fprintf(stderr, "Failed to write to \"%s\"\n", out);
    return JXL_FAILURE("Failed to write output");
  }

  return true;
}
}  // namespace tools
}  // namespace jpegxl

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
  return EXIT_SUCCESS;
}
