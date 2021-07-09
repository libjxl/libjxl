// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "lib/jxl/base/file_io.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_heuristics.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/splines.h"

namespace jxl {

namespace {
struct SplineData {
  int32_t quantization_adjustment;
  std::vector<Spline> splines;
};

Splines SplinesFromSplineData(const SplineData& spline_data,
                              const ColorCorrelationMap& cmap) {
  std::vector<QuantizedSpline> quantized_splines;
  std::vector<Spline::Point> starting_points;
  quantized_splines.reserve(spline_data.splines.size());
  starting_points.reserve(spline_data.splines.size());
  for (const Spline& spline : spline_data.splines) {
    JXL_CHECK(!spline.control_points.empty());
    quantized_splines.emplace_back(spline, spline_data.quantization_adjustment,
                                   cmap.YtoXRatio(0), cmap.YtoBRatio(0));
    starting_points.push_back(spline.control_points.front());
  }
  return Splines(spline_data.quantization_adjustment,
                 std::move(quantized_splines), std::move(starting_points));
}

template <typename F>
bool ParseNode(F& tok, Tree& tree, SplineData& spline_data,
               CompressParams& cparams, size_t& W, size_t& H, CodecInOut& io,
               int& have_next, int& x0, int& y0) {
  static const std::unordered_map<std::string, int> property_map = {
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
  };
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
    if ((t = tok()) != ">") {
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
    JXL_RETURN_IF_ERROR(ParseNode(tok, tree, spline_data, cparams, W, H, io,
                                  have_next, x0, y0));
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
    ImageF alpha(W, H);
    io.frames[0].SetAlpha(std::move(alpha), false);
  } else if (t == "Bitdepth") {
    t = tok();
    size_t num = 0;
    io.metadata.m.bit_depth.bits_per_sample = std::stoul(t, &num);
    if (num != t.size()) {
      fprintf(stderr, "Invalid Bitdepth: %s\n", t.c_str());
      return false;
    }
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
    have_next = 1;
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
    io.frames[0].duration = 100;
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
  } else {
    fprintf(stderr, "Unexpected node type: %s\n", t.c_str());
    return false;
  }
  JXL_RETURN_IF_ERROR(
      ParseNode(tok, tree, spline_data, cparams, W, H, io, have_next, x0, y0));
  return true;
}

void PrintTree(const Tree& tree, const std::string& path) {
  FILE* f = fopen((path + ".dot").c_str(), "w");
  fprintf(f, "digraph{\n");
  for (size_t cur = 0; cur < tree.size(); cur++) {
    if (tree[cur].property < 0) {
      fprintf(f, "n%05zu [label=\"%s%+lld\"];\n", cur,
              PredictorName(tree[cur].predictor),
              static_cast<long long>(tree[cur].predictor_offset));
    } else {
      fprintf(f, "n%05zu [label=\"%s>%d\"];\n", cur,
              PropertyName(tree[cur].property).c_str(), tree[cur].splitval);
      fprintf(f, "n%05zu -> n%05d [style=dashed];\n", cur, tree[cur].rchild);
      fprintf(f, "n%05zu -> n%05d;\n", cur, tree[cur].lchild);
    }
  }
  fprintf(f, "}\n");
  fclose(f);
  std::string command = "dot " + path + ".dot -T png -o " + path + ".png";
  if (system(command.c_str()) != 0) {
    JXL_ABORT("Command failed: %s", command.c_str());
  }
}

class Heuristics : public DefaultEncoderHeuristics {
 public:
  bool CustomFixedTreeLossless(const jxl::FrameDimensions& frame_dim,
                               Tree* tree) override {
    *tree = tree_;
    return true;
  }

  explicit Heuristics(Tree tree) : tree_(std::move(tree)) {}

 private:
  Tree tree_;
};
}  // namespace

int JxlFromTree(const char* in, const char* out, const char* tree_out) {
  Tree tree;
  SplineData spline_data;
  CompressParams cparams;
  size_t width = 1024, height = 1024;
  int x0 = 0, y0 = 0;
  cparams.color_transform = ColorTransform::kNone;
  cparams.modular_group_size_shift = 3;
  CodecInOut io;
  int have_next = 0;

  std::ifstream f(in);
  auto tok = [&f]() {
    std::string out;
    f >> out;
    return out;
  };
  if (!ParseNode(tok, tree, spline_data, cparams, width, height, io, have_next,
                 x0, y0)) {
    return 1;
  }

  if (tree_out) {
    PrintTree(tree, tree_out);
  }
  Image3F image(width, height);
  io.SetFromImage(std::move(image), ColorEncoding::SRGB());
  io.SetSize((width + x0) * cparams.resampling,
             (height + y0) * cparams.resampling);
  io.metadata.m.color_encoding.DecideIfWantICC();
  cparams.options.zero_tokens = true;
  cparams.modular_mode = true;
  cparams.palette_colors = 0;
  cparams.channel_colors_pre_transform_percent = 0;
  cparams.channel_colors_percent = 0;
  cparams.patches = jxl::Override::kOff;
  cparams.already_downsampled = true;
  PaddedBytes compressed;

  io.CheckMetadata();
  BitWriter writer;

  std::unique_ptr<CodecMetadata> metadata = jxl::make_unique<CodecMetadata>();
  *metadata = io.metadata;
  JXL_RETURN_IF_ERROR(metadata->size.Set(io.xsize(), io.ysize()));

  metadata->m.xyb_encoded =
      cparams.color_transform == ColorTransform::kXYB ? true : false;

  JXL_RETURN_IF_ERROR(WriteHeaders(metadata.get(), &writer, nullptr));
  writer.ZeroPadToByte();

  while (true) {
    PassesEncoderState enc_state;
    enc_state.heuristics = make_unique<Heuristics>(tree);
    enc_state.shared.image_features.splines =
        SplinesFromSplineData(spline_data, enc_state.shared.cmap);

    FrameInfo info;
    info.is_last = !have_next;
    if (!info.is_last) info.save_as_reference = 1;

    io.frames[0].origin.x0 = x0;
    io.frames[0].origin.y0 = y0;

    JXL_RETURN_IF_ERROR(EncodeFrame(cparams, info, metadata.get(), io.frames[0],
                                    &enc_state, nullptr, &writer, nullptr));
    if (!have_next) break;
    tree.clear();
    spline_data.splines.clear();
    have_next = 0;
    if (!ParseNode(tok, tree, spline_data, cparams, width, height, io,
                   have_next, x0, y0)) {
      return 1;
    }
    Image3F image(width, height);
    io.SetFromImage(std::move(image), ColorEncoding::SRGB());
    io.frames[0].blend = true;
  }

  compressed = std::move(writer).TakeBytes();

  if (!WriteFile(compressed, out)) {
    fprintf(stderr, "Failed to write to \"%s\"\n", out);
    return 1;
  }

  return 0;
}
}  // namespace jxl

int main(int argc, char** argv) {
  if ((argc != 3 && argc != 4) || !strcmp(argv[1], argv[2])) {
    fprintf(stderr, "Usage: %s tree_in.txt out.jxl [tree_drawing]\n", argv[0]);
    return 1;
  }
  return jxl::JxlFromTree(argv[1], argv[2], argc < 4 ? nullptr : argv[3]);
}
