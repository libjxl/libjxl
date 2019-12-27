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

#ifndef JXL_MODULAR_MA_COMPOUND_H_
#define JXL_MODULAR_MA_COMPOUND_H_

#include <stdint.h>
#include <stdio.h>

#include <cmath>
#include <vector>

#include "jxl/base/status.h"
#include "jxl/modular/ma/symbol.h"

namespace jxl {

typedef int32_t PropertyVal;
typedef std::vector<std::pair<PropertyVal, PropertyVal> > Ranges;
typedef std::vector<PropertyVal> Properties;

static const uint16_t CHANCE_INITZ[11] = {4,    128,  512,  1024, 1536, 2048,
                                          2560, 3072, 3584, 3968, 4088};
static const uint16_t CHANCE_INITS[7] = {512,  1024, 1536, 2048,
                                         2560, 3072, 3584};
// only signal sign initialization if zero is unlikely
#define NO_SIGN_SIGNAL 3

// inner nodes
class PropertyDecisionNode {
 public:
  PropertyVal splitval;
  int16_t property;  // <0 : leaf node, childID points to leaf node
  // 0..nb_properties-1 : childID refers to left branch  (in inner_node)
  //                      childID+1 refers to right branch
  uint16_t childID;

  explicit PropertyDecisionNode(int p = -1, int s = 0, int c = 0)
      : splitval(s), property(p), childID(c) {}
};

class Tree : public std::vector<PropertyDecisionNode> {
 public:
  Tree() : std::vector<PropertyDecisionNode>(1, PropertyDecisionNode()) {}
};

typedef std::vector<Tree> Trees;

// leaf nodes when tree is known
template <typename BitChance, int bits>
class FinalCompoundSymbolChances {
 public:
  SymbolChance<BitChance, bits> realChances;

  explicit FinalCompoundSymbolChances(uint16_t zero_chance)
      : realChances(zero_chance) {}

  const SymbolChance<BitChance, bits> &chances() const { return realChances; }
};

template <typename BitChance, typename RAC, int bits>
class FinalCompoundSymbolBitCoder {
 private:
  RAC &rac;
  FinalCompoundSymbolChances<BitChance, bits> &chances;

  void inline updateChances(const SymbolChanceBitType type, const int i,
                            bool bit) {
    BitChance &real = chances.realChances.bit(type, i);
    real.put(bit);
  }

 public:
  FinalCompoundSymbolBitCoder(
      RAC &racIn, FinalCompoundSymbolChances<BitChance, bits> &chancesIn)
      : rac(racIn), chances(chancesIn) {}

  bool inline read(const SymbolChanceBitType type, const int i = 0) {
    BitChance &ch = chances.realChances.bit(type, i);
    bool bit = rac.read_12bit_chance(ch.get_12bit());
    updateChances(type, i, bit);
    return bit;
  }

#ifdef HAS_ENCODER
  void inline write(const bool bit, const SymbolChanceBitType type,
                    const int i = 0);
  void estimate(const bool bit, const SymbolChanceBitType type, const int i,
                uint64_t &total);
#endif
};

template <typename BitChance, typename RAC, int bits>
class FinalCompoundSymbolCoder {
 private:
  RAC &rac;

 public:
  explicit FinalCompoundSymbolCoder(RAC &racIn) : rac(racIn) {}

  int read_int(FinalCompoundSymbolChances<BitChance, bits> &chancesIn, int min,
               int max) {
    FinalCompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn);
    int val = reader<bits>(bitCoder, min, max);
    return val;
  }
  int read_int(FinalCompoundSymbolChances<BitChance, bits> &chancesIn,
               int nbits) {
    FinalCompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn);
    int val = reader(bitCoder, nbits);
    return val;
  }

#ifdef HAS_ENCODER
  void write_int(FinalCompoundSymbolChances<BitChance, bits> &chancesIn,
                 int min, int max, int val);
  int estimate_int(FinalCompoundSymbolChances<BitChance, bits> &chancesIn,
                   int min, int max, int val);
  void write_int(FinalCompoundSymbolChances<BitChance, bits> &chancesIn,
                 int nbits, int val);
#endif
};

template <typename BitChance, typename RAC, int bits>
class FinalPropertySymbolCoder {
 private:
  FinalCompoundSymbolCoder<BitChance, RAC, bits> coder;
  const unsigned int nb_properties;
  std::vector<FinalCompoundSymbolChances<BitChance, bits> > leaf_node;
  const Tree &inner_node;

  FinalCompoundSymbolChances<BitChance, bits> inline &find_leaf(
      const Properties &properties) ATTRIBUTE_HOT {
    Tree::size_type pos = 0;
#if JXL_COMPILER_CLANG
#pragma clang loop unroll_count(8)
#endif
    while (true) {
      const PropertyDecisionNode &node = inner_node[pos];
      if (node.property < 0) return leaf_node[node.childID];
      if (properties[node.property] > node.splitval)
        pos = node.childID;
      else
        pos = node.childID + 1;
    }
  }

 public:
  FinalPropertySymbolCoder(RAC &racIn, Ranges &rangeIn, Tree &treeIn,
                           uint16_t zero_chance = ZERO_CHANCE,
                           int ignored_split_threshold = 0)
      : coder(racIn), nb_properties(rangeIn.size()), inner_node(treeIn) {
    for (size_t i = 0, leafID = 0; i < inner_node.size(); i++) {
      if (inner_node[i].property < 0) {
        leafID++;
        JXL_DASSERT(inner_node[i].property > -16);
        int zerochance = CHANCE_INITZ[-1 - inner_node[i].property];
        leaf_node.push_back(
            FinalCompoundSymbolChances<BitChance, bits>(zerochance));
        int signchance = CHANCE_INITS[inner_node[i].splitval];
        leaf_node.back().realChances.bitSign().set_12bit(signchance);
      }
    }
  }
  int context_id(const Properties &properties) const ATTRIBUTE_HOT {
    Tree::size_type pos = 0;
#if JXL_COMPILER_CLANG
#pragma clang loop unroll_count(4)
#endif
    while (true) {
      const PropertyDecisionNode &node = inner_node[pos];
      if (node.property < 0) return node.childID;
      if (properties[node.property] > node.splitval)
        pos = node.childID;
      else
        pos = node.childID + 1;
    }
  }
  int nb_contexts() const ATTRIBUTE_HOT { return leaf_node.size(); }
  int read_int(const Properties &properties, int min, int max) ATTRIBUTE_HOT {
    JXL_DASSERT(properties.size() == nb_properties);
    FinalCompoundSymbolChances<BitChance, bits> &chances =
        find_leaf(properties);
    return coder.read_int(chances, min, max);
  }

  int read_int(const Properties &properties, int nbits) {
    JXL_DASSERT(properties.size() == nb_properties);
    FinalCompoundSymbolChances<BitChance, bits> &chances =
        find_leaf(properties);
    return coder.read_int(chances, nbits);
  }

#ifdef HAS_ENCODER
  void write_int(const Properties &properties, int min, int max, int val);
  int estimate_int(const Properties &properties, int min, int max, int val);
  void stats_write_int(const Properties &properties, int min, int max,
                       int val) {}
  void write_int(const Properties &properties, int nbits, int val);
  static void simplify(int min_size = CONTEXT_TREE_MIN_SUBTREE_SIZE) {}
  static uint64_t compute_total_size() { return 0; }
#endif
};

template <typename BitChance, typename RAC>
class MetaPropertySymbolCoder {
 public:
  typedef SimpleSymbolCoder<BitChance, RAC, MAX_BIT_DEPTH> Coder;

 private:
  std::vector<Coder> coder;
  const Ranges range;
  unsigned int nb_properties;

 public:
  MetaPropertySymbolCoder(RAC &racIn, const Ranges &rangesIn)
      : coder(4, Coder(racIn)),
        range(rangesIn),
        nb_properties(rangesIn.size()) {
    for (unsigned int i = 0; i < nb_properties; i++) {
      JXL_DASSERT(range[i].first <= range[i].second);
    }
  }

#ifdef HAS_ENCODER
  void write_subtree(int pos, Ranges &subrange, const Tree &tree,
                     bool signal_chances);
  void write_tree(const Tree &tree, bool signal_chances);
#endif

  bool read_subtree(int pos, Ranges &subrange, Tree &tree, int &maxdepth,
                    int depth, bool signal_chances) {
    PropertyDecisionNode &n = tree[pos];
    int p = coder[0].read_int(0, nb_properties) - 1;
    if (signal_chances && p < 0) {
      p = -(1 + coder[1].read_int(-5, 5) + 5);
      if (p > -1 - NO_SIGN_SIGNAL)
        n.splitval = coder[2].read_int(-3, 3) + 3;
      else
        n.splitval = 3;
    }
    n.property = p;
    depth++;
    if (depth > maxdepth) maxdepth = depth;
    if (p >= 0) {
      int oldmin = subrange[p].first;
      int oldmax = subrange[p].second;
      if (oldmin >= oldmax) {
        return JXL_FAILURE("Invalid tree. Aborting tree decoding.");
      }
      JXL_DASSERT(oldmin < oldmax);
      int splitval = n.splitval = coder[3].read_int2(oldmin, oldmax - 1);
      int childID = n.childID = tree.size();
      // JXL_FAILURE("Pos %i: prop %i splitval %i in [%i..%i]", pos,
      //             n.property, splitval, oldmin, oldmax-1);
      tree.push_back(PropertyDecisionNode());
      tree.push_back(PropertyDecisionNode());
      // > splitval
      subrange[p].first = splitval + 1;
      if (!read_subtree(childID, subrange, tree, maxdepth, depth,
                        signal_chances))
        return false;

      // <= splitval
      subrange[p].first = oldmin;
      subrange[p].second = splitval;
      if (!read_subtree(childID + 1, subrange, tree, maxdepth, depth,
                        signal_chances))
        return false;

      subrange[p].second = oldmax;
    }
    return true;
  }
  bool read_tree(Tree &tree, bool signal_chances) {
    Ranges rootrange(range);
    tree.clear();
    tree.push_back(PropertyDecisionNode());
    int depth = 0;
    if (read_subtree(0, rootrange, tree, depth, 0, signal_chances)) {
      int neededdepth = ilog2(tree.size()) + 1;
      JXL_DEBUG_V(8,
                  "Read MA tree with %u nodes and depth %i (with better "
                  "balance, depth %i might have been enough).",
                  (unsigned int)tree.size(), depth, neededdepth);
      for (size_t i = 0, leafID = 0; i < tree.size(); i++) {
        if (tree[i].property < 0) {
          tree[i].childID = leafID;
          leafID++;
        }
      }
      return true;
    } else
      return false;
  }
};

}  // namespace jxl

#ifdef HAS_ENCODER
#include "jxl/modular/ma/compound_enc.h"
#endif

#endif  // JXL_MODULAR_MA_COMPOUND_H_
