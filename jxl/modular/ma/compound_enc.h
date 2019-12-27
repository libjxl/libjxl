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

#ifndef JXL_MODULAR_MA_COMPOUND_ENC_H_
#define JXL_MODULAR_MA_COMPOUND_ENC_H_

#include <inttypes.h>

#include <cmath>

#include "jxl/modular/ma/compound.h"

namespace jxl {

// leaf nodes during tree construction phase
template <typename BitChance, int bits>
class CompoundSymbolChances final
    : public FinalCompoundSymbolChances<BitChance, bits> {
 public:
  std::vector<
      std::pair<SymbolChance<BitChance, bits>, SymbolChance<BitChance, bits> > >
      virtChances;
  uint64_t realSize;
  std::vector<uint64_t> virtSize;
  std::vector<int64_t> virtPropSum;
  int32_t count;
  int16_t best_property;
  // used to collect stats for chance initialization
  uint64_t real_count;
  uint64_t zero_count;  // zero chance = zero_count / real_count
  uint64_t sign_count;  // pos chance = pos_count / sign_count
  uint64_t pos_count;

  void resetCounters() {
    best_property = -1;
    realSize = 0;
    count = 0;
    virtPropSum.assign(virtPropSum.size(), 0);
    virtSize.assign(virtSize.size(), 0);
  }

  CompoundSymbolChances(int nProp, uint16_t zero_chance)
      : FinalCompoundSymbolChances<BitChance, bits>(zero_chance),
        virtChances(nProp,
                    std::make_pair(SymbolChance<BitChance, bits>(zero_chance),
                                   SymbolChance<BitChance, bits>(zero_chance))),
        realSize(0),
        virtSize(nProp),
        virtPropSum(nProp),
        count(0),
        best_property(-1),
        real_count(0),
        zero_count(0),
        sign_count(0),
        pos_count(0) {}
};

template <typename BitChance, typename RAC, int bits>
void inline FinalCompoundSymbolBitCoder<BitChance, RAC, bits>::write(
    const bool bit, const SymbolChanceBitType type, const int i) {
  BitChance &ch = chances.realChances.bit(type, i);
  rac.write_12bit_chance(ch.get_12bit(), bit);
  updateChances(type, i, bit);
}

// This function is currently not used, but it could be used to estimate the
// current cost of writing a number without actually writing it. (that could be
// useful for a lossy encoder)
template <typename BitChance, typename RAC, int bits>
void inline FinalCompoundSymbolBitCoder<BitChance, RAC, bits>::estimate(
    const bool bit, const SymbolChanceBitType type, const int i,
    uint64_t &total) {
  BitChance &ch = chances.realChances.bit(type, i);
  ch.estim(bit, total);
}

template <typename BitChance, typename RAC, int bits>
class CompoundSymbolBitCoder {
 private:
  RAC &rac;
  CompoundSymbolChances<BitChance, bits> &chances;
  std::vector<bool> &select;

  void inline updateChances(SymbolChanceBitType type, int i, bool bit) {
    BitChance &real = chances.realChances.bit(type, i);
    real.estim(bit, chances.realSize);
    real.put(bit);

    int16_t best_property = -1;
    uint64_t best_size = chances.realSize;
    for (unsigned int j = 0; j < chances.virtChances.size(); j++) {
      BitChance &virt = (select)[j]
                            ? chances.virtChances[j].first.bit(type, i)
                            : chances.virtChances[j].second.bit(type, i);
      virt.estim(bit, chances.virtSize[j]);
      virt.put(bit);
      if (chances.virtSize[j] < best_size) {
        best_size = chances.virtSize[j];
        best_property = j;
      }
    }
    chances.best_property = best_property;
  }
  BitChance inline &bestChance(SymbolChanceBitType type, int i = 0) {
    signed short int p = chances.best_property;
    return (p < 0 ? chances.realChances.bit(type, i)
                  : ((select)[p] ? chances.virtChances[p].first.bit(type, i)
                                 : chances.virtChances[p].second.bit(type, i)));
  }

 public:
  CompoundSymbolBitCoder(RAC &racIn,
                         CompoundSymbolChances<BitChance, bits> &chancesIn,
                         std::vector<bool> &selectIn)
      : rac(racIn), chances(chancesIn), select(selectIn) {}

  bool read(SymbolChanceBitType type, int i = 0) {
    BitChance &ch = bestChance(type, i);
    bool bit = rac.read_12bit_chance(ch.get_12bit());
    updateChances(type, i, bit);
    return bit;
  }

  void write(bool bit, SymbolChanceBitType type, int i = 0) {
    BitChance &ch = bestChance(type, i);
    rac.write_12bit_chance(ch.get_12bit(), bit);
    updateChances(type, i, bit);
  }

  void estimate(const bool bit, const SymbolChanceBitType type, const int i,
                uint64_t &total) {
    BitChance &ch = bestChance(type, i);
    ch.estim(bit, total);
  }
};

template <typename BitChance, typename RAC, int bits>
void FinalCompoundSymbolCoder<BitChance, RAC, bits>::write_int(
    FinalCompoundSymbolChances<BitChance, bits> &chancesIn, int min, int max,
    int val) {
  FinalCompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn);
  writer<bits>(bitCoder, min, max, val);
}

template <typename BitChance, typename RAC, int bits>
int FinalCompoundSymbolCoder<BitChance, RAC, bits>::estimate_int(
    FinalCompoundSymbolChances<BitChance, bits> &chancesIn, int min, int max,
    int val) {
  FinalCompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn);
  return estimate_writer<bits>(bitCoder, min, max, val);
}

template <typename BitChance, typename RAC, int bits>
void FinalCompoundSymbolCoder<BitChance, RAC, bits>::write_int(
    FinalCompoundSymbolChances<BitChance, bits> &chancesIn, int nbits,
    int val) {
  FinalCompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn);
  writer(bitCoder, nbits, val);
}

template <typename BitChance, typename RAC, int bits>
class CompoundSymbolCoder {
 private:
  RAC &rac;

 public:
  explicit CompoundSymbolCoder(RAC &racIn) : rac(racIn) {}

  int read_int(CompoundSymbolChances<BitChance, bits> &chancesIn,
               std::vector<bool> &selectIn, int min, int max) {
    if (min == max) {
      return min;
    }
    CompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn,
                                                          selectIn);
    return reader<bits>(bitCoder, min, max);
  }

  void write_int(CompoundSymbolChances<BitChance, bits> &chancesIn,
                 std::vector<bool> &selectIn, int min, int max, int val) {
    if (min == max) {
      JXL_DASSERT(val == min);
      return;
    }
    CompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn,
                                                          selectIn);
    writer<bits>(bitCoder, min, max, val);
  }

  int estimate_int(CompoundSymbolChances<BitChance, bits> &chancesIn,
                   std::vector<bool> &selectIn, int min, int max, int val) {
    if (min == max) {
      JXL_DASSERT(val == min);
      return 0;
    }
    CompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn,
                                                          selectIn);
    return estimate_writer<bits>(bitCoder, min, max, val);
  }

  int read_int(CompoundSymbolChances<BitChance, bits> &chancesIn,
               std::vector<bool> &selectIn, int nbits) {
    CompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn,
                                                          selectIn);
    return reader(bitCoder, nbits);
  }

  void write_int(CompoundSymbolChances<BitChance, bits> &chancesIn,
                 std::vector<bool> &selectIn, int nbits, int val) {
    CompoundSymbolBitCoder<BitChance, RAC, bits> bitCoder(rac, chancesIn,
                                                          selectIn);
    writer(bitCoder, nbits, val);
  }
};

template <typename BitChance, typename RAC, int bits>
void FinalPropertySymbolCoder<BitChance, RAC, bits>::write_int(
    const Properties &properties, int min, int max, int val) {
  if (min == max) {
    JXL_DASSERT(val == min);
    return;
  }
  JXL_DASSERT(properties.size() == nb_properties);
  FinalCompoundSymbolChances<BitChance, bits> &chances = find_leaf(properties);
  coder.write_int(chances, min, max, val);
}

template <typename BitChance, typename RAC, int bits>
int FinalPropertySymbolCoder<BitChance, RAC, bits>::estimate_int(
    const Properties &properties, int min, int max, int val) {
  if (min == max) {
    JXL_DASSERT(val == min);
    return 0;
  }
  JXL_DASSERT(properties.size() == nb_properties);
  FinalCompoundSymbolChances<BitChance, bits> &chances = find_leaf(properties);
  return coder.estimate_int(chances, min, max, val);
}

template <typename BitChance, typename RAC, int bits>
void FinalPropertySymbolCoder<BitChance, RAC, bits>::write_int(
    const Properties &properties, int nbits, int val) {
  JXL_DASSERT(properties.size() == nb_properties);
  FinalCompoundSymbolChances<BitChance, bits> &chances = find_leaf(properties);
  coder.write_int(chances, nbits, val);
}

template <typename BitChance, typename RAC, int bits>
class PropertySymbolCoder {
 public:
  typedef CompoundSymbolCoder<BitChance, RAC, bits> Coder;

 private:
  RAC &rac;
  Coder coder;
  const Ranges range;
  unsigned int nb_properties;
  std::vector<CompoundSymbolChances<BitChance, bits> > leaf_node;
  Tree &inner_node;
  std::vector<bool> selection;
  int split_threshold;

  inline PropertyVal div_down(int64_t sum, int32_t count) const {
    JXL_DASSERT(count > 0);
    if (sum >= 0)
      return sum / count;
    else
      return -((-sum + count - 1) / count);
  }
  inline PropertyVal compute_splitval(
      const CompoundSymbolChances<BitChance, bits> &ch, int16_t p,
      const Ranges &crange) const {
    // make first branch on > 0 if range has negative and positive numbers
    if (crange[p].first < 0 && crange[p].second > 0) {
      return 0;
    }
    PropertyVal splitval = div_down(ch.virtPropSum[p], ch.count);
    if (splitval >= crange[p].second)
      splitval = crange[p].second -
                 1;  // == does happen because of rounding and running average
    return splitval;
  }

  CompoundSymbolChances<BitChance, bits> inline &find_leaf_readonly(
      const Properties &properties) {
    uint32_t pos = 0;
    Ranges current_ranges = range;
    while (inner_node[pos].property >= 0) {
      if (properties[inner_node[pos].property] > inner_node[pos].splitval) {
        current_ranges[inner_node[pos].property].first =
            inner_node[pos].splitval + 1;
        pos = inner_node[pos].childID;
      } else {
        current_ranges[inner_node[pos].property].second =
            inner_node[pos].splitval;
        pos = inner_node[pos].childID + 1;
      }
    }
    CompoundSymbolChances<BitChance, bits> &result =
        leaf_node[inner_node[pos].childID];
    set_selection(properties, result, current_ranges);
    return result;
  }

  CompoundSymbolChances<BitChance, bits> inline &find_leaf(
      const Properties &properties) {
    uint32_t pos = 0;
    Ranges current_ranges = range;
    while (inner_node[pos].property >= 0) {
      if (properties[inner_node[pos].property] > inner_node[pos].splitval) {
        current_ranges[inner_node[pos].property].first =
            inner_node[pos].splitval + 1;
        pos = inner_node[pos].childID;
      } else {
        current_ranges[inner_node[pos].property].second =
            inner_node[pos].splitval;
        pos = inner_node[pos].childID + 1;
      }
    }
    CompoundSymbolChances<BitChance, bits> &result =
        leaf_node[inner_node[pos].childID];
    set_selection_and_update_property_sums(properties, result, current_ranges);

    // split leaf node if some virtual context is performing (significantly)
    // better
    if (result.best_property != -1 &&
        result.realSize >
            result.virtSize[result.best_property] + split_threshold &&
        leaf_node.size() < 0xFFFF && inner_node.size() < 0xFFFF &&
        current_ranges[result.best_property].first <
            current_ranges[result.best_property].second) {
      int16_t p = result.best_property;
      PropertyVal splitval = compute_splitval(result, p, current_ranges);

      uint32_t new_inner = inner_node.size();
      inner_node.push_back(inner_node[pos]);
      inner_node.push_back(inner_node[pos]);
      inner_node[pos].splitval = splitval;
      inner_node[pos].property = p;
      uint32_t new_leaf = leaf_node.size();
      result.resetCounters();
      leaf_node.push_back(CompoundSymbolChances<BitChance, bits>(result));
      uint32_t old_leaf = inner_node[pos].childID;
      inner_node[pos].childID = new_inner;
      inner_node[new_inner].childID = old_leaf;
      inner_node[new_inner + 1].childID = new_leaf;
      if (properties[p] > inner_node[pos].splitval) {
        return leaf_node[old_leaf];
      } else {
        return leaf_node[new_leaf];
      }
    }
    return result;
  }

  void inline set_selection_and_update_property_sums(
      const Properties &properties,
      CompoundSymbolChances<BitChance, bits> &chances, const Ranges &crange) {
    chances.count++;
    for (unsigned int i = 0; i < nb_properties; i++) {
      JXL_DASSERT(properties[i] >= range[i].first);
      JXL_DASSERT(properties[i] <= range[i].second);
      chances.virtPropSum[i] += properties[i];
      PropertyVal splitval = compute_splitval(chances, i, crange);
      selection[i] = (properties[i] > splitval);
    }
  }
  void inline set_selection(
      const Properties &properties,
      const CompoundSymbolChances<BitChance, bits> &chances,
      const Ranges &crange) {
    if (chances.count == 0) return;
    for (unsigned int i = 0; i < nb_properties; i++) {
      JXL_DASSERT(properties[i] >= range[i].first);
      JXL_DASSERT(properties[i] <= range[i].second);
      PropertyVal splitval = compute_splitval(chances, i, crange);
      selection[i] = (properties[i] > splitval);
    }
  }

 public:
  PropertySymbolCoder(RAC &racIn, Ranges &rangeIn, Tree &treeIn,
                      int zero_chance = ZERO_CHANCE,
                      int st = CONTEXT_TREE_SPLIT_THRESHOLD)
      : rac(racIn),
        coder(racIn),
        range(rangeIn),
        nb_properties(range.size()),
        leaf_node(1, CompoundSymbolChances<BitChance, bits>(nb_properties,
                                                            zero_chance)),
        inner_node(treeIn),
        selection(nb_properties, false),
        split_threshold(st) {}

  int read_int(Properties &properties, int min, int max) {
    CompoundSymbolChances<BitChance, bits> &chances2 = find_leaf(properties);
    return coder.read_int(chances2, selection, min, max);
  }

  void write_int(Properties &properties, int min, int max, int val) {
    CompoundSymbolChances<BitChance, bits> &chances2 = find_leaf(properties);
    coder.write_int(chances2, selection, min, max, val);
  }

  int estimate_int(Properties &properties, int min, int max, int val) {
    CompoundSymbolChances<BitChance, bits> &chances =
        find_leaf_readonly(properties);
    return coder.estimate_int(chances, selection, min, max, val);
  }
  void stats_write_int(const Properties &properties, int min, int max,
                       int val) {
    CompoundSymbolChances<BitChance, bits> &chances =
        find_leaf_readonly(properties);
    if (chances.real_count < 500) {
      chances.real_count++;
      if (!val) chances.zero_count++;
    }
    if (chances.sign_count < 500 && val && max > 0 && min < 0) {
      chances.sign_count++;
      if (val > 0) chances.pos_count++;
    }
  }

  int read_int(Properties &properties, int nbits) {
    CompoundSymbolChances<BitChance, bits> &chances2 = find_leaf(properties);
    return coder.read_int(chances2, selection, nbits);
  }

  void write_int(Properties &properties, int nbits, int val) {
    CompoundSymbolChances<BitChance, bits> &chances2 = find_leaf(properties);
    coder.write_int(chances2, selection, nbits, val);
  }

  void kill_children(int pos) {
    PropertyDecisionNode &n = inner_node[pos];
    if (n.property < 0)
      n.property = 0;
    else
      kill_children(n.childID);
    PropertyDecisionNode &n1 = inner_node[pos + 1];
    if (n1.property < 0)
      n1.property = 0;
    else
      kill_children(n1.childID);
  }

  int find_nearest_chance_initz(uint16_t chance) {
    for (int i = 0; i < 11; i++)
      if (CHANCE_INITZ[i] >= chance) {
        if (i > 5)
          return i - 1;
        else
          return i;  // round towards fifty-fifty
      }
    return 10;
  }
  int find_nearest_chance_inits(uint16_t chance) {
    for (int i = 0; i < 7; i++)
      if (CHANCE_INITS[i] >= chance) {
        if (i > 3)
          return i - 1;
        else
          return i;  // round towards fifty-fifty
      }
    return 6;
  }

  // destructive simplification procedure, prunes subtrees with too low counts
  // also sets initialization chances, so not optional
  long long int simplify_subtree(int pos, int min_size, int indent) {
    PropertyDecisionNode &n = inner_node[pos];
    if (n.property < 0) {
      int zchance = 2048;
      if (leaf_node[n.childID].real_count)
        zchance = leaf_node[n.childID].zero_count * 4096 /
                  leaf_node[n.childID].real_count;
      int qchance = find_nearest_chance_initz(zchance);
      JXL_DASSERT(qchance >= 0);
      JXL_DASSERT(qchance <= 11);
      n.property = -1 - qchance;
      int pchance = 2048;
      if (n.property > -1 - NO_SIGN_SIGNAL) {
        if (leaf_node[n.childID].sign_count)
          pchance = leaf_node[n.childID].pos_count * 4096 /
                    leaf_node[n.childID].sign_count;
        n.splitval = find_nearest_chance_inits(pchance);
      } else
        n.splitval = 3;
      for (int i = 0; i < indent; i++) JXL_DEBUG_V(10, "  ");
      JXL_DEBUG_V(
          10,
          "* leaf: count=%lli, size=%.2f bits, bits per int: %f, "
          "zero chance: %i/%i = %.2f%% ~= %.2f%% ~= %.2f%% = %" PRIu64
          "/%" PRIu64
          ", "
          "sign chance: %i/%i = %.2f%% ~= %.2f%% ~= %.2f%% = %" PRIu64
          "/%" PRIu64 "",
          (long long int)leaf_node[n.childID].count,
          leaf_node[n.childID].realSize / 5461.0,
          (leaf_node[n.childID].count > 0
               ? leaf_node[n.childID].realSize * 1.0 /
                     leaf_node[n.childID].count / 5461
               : -1),
          CHANCE_INITZ[qchance], 4096, CHANCE_INITZ[qchance] * 100.0 / 4096,
          leaf_node[n.childID].realChances.bit_zero.get_12bit() * 100.0 / 4096,
          zchance * 100.0 / 4096, leaf_node[n.childID].zero_count,
          leaf_node[n.childID].real_count, CHANCE_INITS[n.splitval], 4096,
          CHANCE_INITS[n.splitval] * 100.0 / 4096,
          leaf_node[n.childID].realChances.bit_sign.get_12bit() * 100.0 / 4096,
          pchance * 100.0 / 4096, leaf_node[n.childID].pos_count,
          leaf_node[n.childID].sign_count);

      for (int i = 0; i < indent; i++) JXL_DEBUG_V(11, "  ");
      JXL_DEBUG_V(11, "  chances: ");
      JXL_DEBUG_V(11, " zero=%u",
                  leaf_node[n.childID].realChances.bit_zero.get_12bit());
      JXL_DEBUG_V(11, " sign=%u",
                  leaf_node[n.childID].realChances.bit_sign.get_12bit());
      for (int i = 0; i < 5; i++)
        JXL_DEBUG_V(10, " exp%i=%u", i,
                    leaf_node[n.childID].realChances.bit_exp[i].get_12bit());
      JXL_DEBUG_V(11, " mant=");
      for (int i = 0; i < 8; i++)
        JXL_DEBUG_V(11, "%u ",
                    leaf_node[n.childID].realChances.bit_mant[i].get_12bit());

      return leaf_node[n.childID].count;

    } else {
      for (int i = 0; i < indent; i++) JXL_DEBUG_V(10, "  ");
      JXL_DEBUG_V(10, "* test: property %i, value > %i ?", n.property,
                  n.splitval);
      long long int subtree_size = 0;
      subtree_size += simplify_subtree(n.childID, min_size, indent + 1);
      subtree_size += simplify_subtree(n.childID + 1, min_size, indent + 1);
      if (subtree_size < min_size) {
        for (int i = 0; i < indent; i++) JXL_DEBUG_V(11, "  ");
        JXL_DEBUG_V(11, "[PRUNING THE ABOVE SUBTREE]");
        n.property = (inner_node[n.childID].property +
                      inner_node[n.childID + 1].property) /
                     2;
        n.splitval = (inner_node[n.childID].splitval +
                      inner_node[n.childID + 1].splitval) /
                     2;
        JXL_DASSERT(n.property < 0);
        JXL_DASSERT(n.property >= -16);
        kill_children(n.childID);
      }
      return subtree_size;
    }
  }
  void simplify(int min_size = CONTEXT_TREE_MIN_SUBTREE_SIZE) {
    JXL_DEBUG_V(10, "MA TREE BEFORE SIMPLIFICATION:");
    simplify_subtree(0, min_size, 0);

    for (size_t i = 0, leafID = 0; i < inner_node.size(); i++) {
      if (inner_node[i].property < 0) {
        inner_node[i].childID = leafID;
        leafID++;
      }
    }
  }
  uint64_t compute_total_size_subtree(int pos) {
    PropertyDecisionNode &n = inner_node[pos];
    uint64_t total = 0;
    if (n.property < 0) {
      total += leaf_node[n.childID].realSize / 5461;
    } else {
      total += compute_total_size_subtree(n.childID);
      total += compute_total_size_subtree(n.childID + 1);
    }
    return total;
  }
  uint64_t compute_total_size() { return compute_total_size_subtree(0); }
  // never used
  int nb_contexts() const { return 0; }
  int context_id(const Properties &p) const { return 0; }
};

template <typename BitChance, typename RAC>
void MetaPropertySymbolCoder<BitChance, RAC>::write_subtree(
    int pos, Ranges &subrange, const Tree &tree, bool signal_chances) {
  const PropertyDecisionNode &n = tree[pos];
  int p = n.property;
  if (p < 0) p = -1;
  coder[0].write_int(0, nb_properties, p + 1);
  if (signal_chances && p < 0) {
    coder[1].write_int(-5, 5, -n.property - 1 - 5);
    if (n.property > -1 - NO_SIGN_SIGNAL)
      coder[2].write_int(-3, 3, n.splitval - 3);
  }
  if (p >= 0) {
    int oldmin = subrange[p].first;
    int oldmax = subrange[p].second;
    JXL_DASSERT(oldmin < oldmax);
    coder[3].write_int2(oldmin, oldmax - 1, n.splitval);

    // > splitval
    subrange[p].first = n.splitval + 1;
    write_subtree(n.childID, subrange, tree, signal_chances);

    // <= splitval
    subrange[p].first = oldmin;
    subrange[p].second = n.splitval;
    write_subtree(n.childID + 1, subrange, tree, signal_chances);

    subrange[p].second = oldmax;
  }
}
template <typename BitChance, typename RAC>
void MetaPropertySymbolCoder<BitChance, RAC>::write_tree(const Tree &tree,
                                                         bool signal_chances) {
  Ranges rootrange(range);
  write_subtree(0, rootrange, tree, signal_chances);
}

}  // namespace jxl

#endif  // JXL_MODULAR_MA_COMPOUND_ENC_H_
