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

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "lib/jxl/base/file_io.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/enc_heuristics.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/encoding/ma.h"

namespace jxl {

namespace {
template <typename F>
bool ParseNode(F& tok, Tree& tree) {
  static const std::unordered_map<std::string, int> property_map = {
      {"c", 0},           {"g", 1},      {"y", 2},     {"x", 3},
      {"|N|", 4},         {"|W|", 5},    {"N", 6},     {"W", 7},
      {"W-WW-NW+NWW", 8}, {"W+N-NW", 9}, {"W-NW", 10}, {"NW-N", 11},
      {"N-NE", 12},       {"N-NN", 13},  {"W-WW", 14}, {"WGH", 15},
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
    JXL_RETURN_IF_ERROR(ParseNode(tok, tree));
    tree[pos].rchild = tree.size();
    JXL_RETURN_IF_ERROR(ParseNode(tok, tree));
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
  } else {
    fprintf(stderr, "Unexpected node type: %s\n", t.c_str());
    return false;
  }
  return true;
}

void PrintTree(const Tree& tree, const std::string& path) {
  FILE* f = fopen((path + ".dot").c_str(), "w");
  fprintf(f, "digraph{\n");
  for (size_t cur = 0; cur < tree.size(); cur++) {
    if (tree[cur].property < 0) {
      fprintf(f, "n%05zu [label=\"%s%+" PRId64 "\"];\n", cur,
              PredictorName(tree[cur].predictor), tree[cur].predictor_offset);
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
  {
    std::ifstream f(in);
    auto tok = [&f]() {
      std::string out;
      f >> out;
      return out;
    };
    if (!ParseNode(tok, tree)) {
      return 1;
    }
  }
  if (tree_out) {
    PrintTree(tree, tree_out);
  }
  constexpr size_t kSize = 1024;
  Image3F image(kSize, kSize);
  Channel channel(kSize, kSize);
  for (size_t c = 0; c < 3; c++) {
    const intptr_t onerow = channel.plane.PixelsPerRow();
    Channel references(0, channel.w);
    std::array<pixel_type, kNumStaticProperties> static_props = {(int)c, 0};
    bool tree_has_wp_prop_or_pred = false;
    bool is_wp_only = false;
    bool is_gradient_only = false;
    size_t num_props;
    FlatTree flat_tree =
        FilterTree(tree, static_props, &num_props, &tree_has_wp_prop_or_pred,
                   &is_wp_only, &is_gradient_only);
    MATreeLookup tree_lookup(flat_tree);
    Properties properties(num_props);
    weighted::State wp_state(weighted::Header(), channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type* JXL_RESTRICT p = channel.Row(y);
      float* JXL_RESTRICT pf = image.PlaneRow(c, y);
      InitPropsRow(&properties, static_props, y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeWP(&properties, channel.w, p + x, onerow, x, y,
                          tree_lookup, references, &wp_state);
        p[x] = res.guess;
        if (p[x] < 0 || p[x] > 255) {
          fprintf(stderr, "Invalid pixel value %d in position (%zu, %zu)\n",
                  p[x], x, y);
          return 1;
        }
        pf[x] = p[x] * (1.0f / 255);
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
    }
  }
  CodecInOut io;
  io.SetFromImage(std::move(image), ColorEncoding::SRGB());
  io.metadata.m.color_encoding.DecideIfWantICC();
  PassesEncoderState enc_state;
  enc_state.heuristics = make_unique<Heuristics>(tree);
  CompressParams cparams;
  cparams.colorspace = 0;
  cparams.color_transform = ColorTransform::kNone;
  cparams.modular_mode = true;
  cparams.palette_colors = 0;
  cparams.channel_colors_pre_transform_percent = 0;
  cparams.channel_colors_percent = 0;
  cparams.modular_group_size_shift = 3;
  PaddedBytes compressed;
  JXL_CHECK(EncodeFile(cparams, &io, &enc_state, &compressed));
  if (!WriteFile(compressed, out)) {
    fprintf(stderr, "Failed to write to \"%s\"\n", out);
    return 1;
  }

  return 0;
}
}  // namespace jxl

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    fprintf(stderr, "Usage: %s tree_in.txt out.jxl [tree_drawing]\n", argv[0]);
    return 1;
  }
  return jxl::JxlFromTree(argv[1], argv[2], argc < 4 ? nullptr : argv[3]);
}
