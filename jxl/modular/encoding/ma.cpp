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

#include "jxl/modular/encoding/ma.h"

#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/modular/encoding/ma.cpp"
#include <hwy/foreach_target.h>

#include "jxl/fast_log-inl.h"

namespace jxl {

#include <hwy/begin_target-inl.h>

const HWY_FULL(float) df;
const HWY_FULL(int32_t) di;
size_t Padded(size_t x) { return (x + df.N - 1) / df.N * df.N; }

HWY_ATTR float EstimateBits(const int32_t counts[ANS_MAX_ALPHA_SIZE],
                            size_t num_symbols) {
  const auto inv_total =
      Set(df, 1.0f / std::accumulate(counts, counts + num_symbols, 0));
  const auto zero = Zero(df);
  auto bits_lanes = Zero(df);
  for (size_t i = 0; i < num_symbols; i += df.N) {
    const auto counts_v = ConvertTo(df, LoadU(di, &counts[i]));
    bits_lanes -= IfThenElse(counts_v == zero, zero,
                             counts_v * FastLog2f_18bits(counts_v * inv_total));
  }
  return GetLane(SumOfLanes(bits_lanes));
}

// Compute the entropy obtained by splitting up along each property.
HWY_ATTR void EstimateEntropy(
    std::vector<std::vector<int>> *data,
    std::vector<std::vector<int>> *compact_properties,
    std::vector<std::pair<float, size_t>> *props_with_entropy) {
  const size_t num_symbols =
      *std::max_element((*data)[0].begin(), (*data)[0].end()) + 1;

  int32_t dist[ANS_MAX_ALPHA_SIZE] = {};
  std::vector<size_t> indices((*data)[0].size());
  std::vector<int> prop_counts;
  for (size_t i = 1; i < data->size(); i++) {
    const auto &props = (*data)[i];

    // Counting sort.
    prop_counts.clear();
    prop_counts.resize((*compact_properties)[i - 1].size() + 1);
    for (size_t j = 0; j < props.size(); j++) {
      prop_counts[props[j] + 1]++;
    }
    for (size_t j = 0; j < prop_counts.size() - 1; j++) {
      prop_counts[j + 1] += prop_counts[j];
    }
    for (size_t j = 0; j < props.size(); j++) {
      indices[prop_counts[props[j]]++] = j;
    }

    size_t previous_position = 0;
    size_t current_position = 0;
    float entropy = 0;
    for (; current_position < props.size(); current_position++) {
      previous_position = current_position;
      while (current_position < props.size() &&
             props[indices[current_position]] ==
                 props[indices[previous_position]]) {
        dist[(*data)[0][indices[current_position]]]++;
        current_position++;
      }
      entropy += EstimateBits(dist, num_symbols);
      while (previous_position < current_position) {
        dist[(*data)[0][indices[previous_position]]]--;
        previous_position++;
      }
    }
    props_with_entropy->emplace_back(entropy, i);
  }
  std::sort(props_with_entropy->begin(), props_with_entropy->end());
}

void MakeSplitNode(size_t pos, int property, int splitval, Tree *tree) {
  // Note that the tree splits on *strictly greater*.
  (*tree)[pos].childID = tree->size();
  (*tree)[pos].splitval = splitval;
  (*tree)[pos].property = property;
  tree->emplace_back();
  tree->emplace_back();
}

HWY_ATTR void FindBestSplit(
    const std::vector<std::vector<int>> &data,
    const std::vector<int> &multiplicity,
    const std::vector<std::vector<int>> &compact_properties,
    std::vector<size_t> *indices, size_t pos, size_t begin, size_t end,
    const std::vector<size_t> &props_to_use, size_t num_symbols,
    float threshold, Tree *tree) {
  if (begin == end) return;
  int32_t counts[ANS_MAX_ALPHA_SIZE];
  memset(counts, 0, Padded(num_symbols) * sizeof *counts);
  JXL_ASSERT(begin <= end);
  JXL_ASSERT(end <= indices->size());
  for (size_t i = begin; i < end; i++) {
    counts[data[0][(*indices)[i]]] += multiplicity[(*indices)[i]];
  }
  float base_bits = EstimateBits(counts, num_symbols);

  size_t split_prop = 0;
  int split_val = 0;
  size_t split_pos = begin;
  float best_split = std::numeric_limits<float>::max();
  std::vector<int> prop_value_used_count;
  std::vector<int> prop_count_increase;
  for (size_t prop = 1; prop < data.size(); prop++) {
    if (prop_value_used_count.size() < compact_properties[prop - 1].size()) {
      prop_value_used_count.resize(compact_properties[prop - 1].size());
      prop_count_increase.resize(compact_properties[prop - 1].size() *
                                 num_symbols);
    }

    size_t first_used = compact_properties[prop - 1].size();
    size_t last_used = 0;

    for (size_t i = begin; i < end; i++) {
      size_t p = data[prop][(*indices)[i]];
      size_t sym = data[0][(*indices)[i]];
      prop_value_used_count[p]++;
      prop_count_increase[p * num_symbols + sym] += multiplicity[(*indices)[i]];
      last_used = std::max(last_used, p);
      first_used = std::min(first_used, p);
    }

    int32_t counts_above[ANS_MAX_ALPHA_SIZE];
    memcpy(counts_above, counts, Padded(num_symbols) * sizeof(*counts));
    int32_t counts_below[ANS_MAX_ALPHA_SIZE];
    memset(counts_below, 0, Padded(num_symbols) * sizeof *counts_below);
    // Exclude last used: this ensures neither counts_above nor counts_below is
    // empty.
    size_t split = begin;
    for (size_t i = first_used; i < last_used; i++) {
      if (!prop_value_used_count[i]) continue;
      split += prop_value_used_count[i];
      for (size_t sym = 0; sym < num_symbols; sym++) {
        counts_above[sym] -= prop_count_increase[i * num_symbols + sym];
        counts_below[sym] += prop_count_increase[i * num_symbols + sym];
      }
      float cost = EstimateBits(counts_above, num_symbols) +
                   EstimateBits(counts_below, num_symbols);
      if (cost < best_split) {
        split_prop = prop;
        split_val = i;
        split_pos = split;
        best_split = cost;
      }
    }
    for (size_t i = begin; i < end; i++) {
      size_t p = data[prop][(*indices)[i]];
      size_t sym = data[0][(*indices)[i]];
      prop_value_used_count[p] = 0;
      prop_count_increase[p * num_symbols + sym] = 0;
    }
  }

  if (best_split + threshold < base_bits) {
    // Split node and try to split children.
    MakeSplitNode(pos, props_to_use[split_prop - 1],
                  compact_properties[split_prop - 1][split_val], tree);
    // "Sort" according to winning property
    std::nth_element(indices->begin() + begin, indices->begin() + split_pos,
                     indices->begin() + end, [&](size_t a, size_t b) {
                       return data[split_prop][a] < data[split_prop][b];
                     });
    FindBestSplit(data, multiplicity, compact_properties, indices,
                  (*tree)[pos].childID + 1, begin, split_pos, props_to_use,
                  num_symbols, threshold, tree);
    FindBestSplit(data, multiplicity, compact_properties, indices,
                  (*tree)[pos].childID, split_pos, end, props_to_use,
                  num_symbols, threshold, tree);
  }
}

#include <hwy/end_target-inl.h>

#if HWY_ONCE
HWY_EXPORT(EstimateEntropy)
HWY_EXPORT(FindBestSplit)

namespace {

size_t TreeHeight(const Tree &tree, size_t root, std::vector<size_t> *heights) {
  if (tree[root].property < 0) {
    (*heights)[root] = 0;
    return 1;
  }
  size_t subh = std::max(TreeHeight(tree, tree[root].childID, heights),
                         TreeHeight(tree, tree[root].childID + 1, heights));
  (*heights)[root] = subh;
  return subh + 1;
}

static constexpr size_t kNumChunkLimit = 1024;

void BuildTable(uint8_t *JXL_RESTRICT state, const Tree &tree, size_t root,
                const CompactTree::ChunkData &chunk,
                const std::vector<size_t> &child_chunk, CompactTree *out) {
  int16_t value;
  bool has_value = false;
  if (child_chunk[root] != -1) {
    value = child_chunk[root];
    has_value = true;
  } else {
    const PropertyDecisionNode &node = tree[root];
    if (node.property < 0) {
      value = -node.childID;
      has_value = true;
    } else {
      // Recursion.
      uint8_t new_state[CompactTree::kChunkPropertyLimit];
      memcpy(new_state, state, sizeof(new_state));
      // property > splitval
      for (size_t i = 0; i < chunk.num_properties; i++) {
        if (chunk.properties[i] == node.property &&
            chunk.thresholds[i] <= node.splitval) {
          JXL_ASSERT(new_state[i] != 0);
          new_state[i] = 1;
        }
      }
      BuildTable(new_state, tree, node.childID, chunk, child_chunk, out);
      memcpy(new_state, state, sizeof(new_state));
      // property <= splitval
      for (size_t i = 0; i < chunk.num_properties; i++) {
        if (chunk.properties[i] == node.property &&
            chunk.thresholds[i] >= node.splitval) {
          JXL_ASSERT(new_state[i] != 1);
          new_state[i] = 0;
        }
      }
      BuildTable(new_state, tree, node.childID + 1, chunk, child_chunk, out);
    }
  }
  if (!has_value) return;
  static_assert(CompactTree::kChunkPropertyLimit == 4,
                "Wrong number of for loops");
#define FORi(x)                                 \
  for (int i##x = state[x] == 2 ? 0 : state[x]; \
       i##x < (state[x] == 2 ? 2 : state[x] + 1); i##x++)
  FORi(0) {
    FORi(1) {
      FORi(2) {
        FORi(3) {
          size_t idx = chunk.start;
          idx += i0;
          idx += i1 << 1;
          idx += i2 << 2;
          idx += i3 << 3;
          out->table[idx] = value;
        }
      }
    }
  }
}
#undef FORi

bool SplitTree(const Tree &tree, size_t root,
               const std::vector<size_t> &heights, size_t *height,
               CompactTree *out) {
  // No need to chunkify a tree with a single node.
  if (tree[root].property < 0) {
    return false;
  }
  std::vector<size_t> frontier;
  frontier.push_back(root);
  std::vector<std::pair<int, int>> nodes;

  auto cmp_pair = [&](size_t i) {
    return std::pair<int, int>(tree[i].property, tree[i].splitval);
  };

  while (!frontier.empty()) {
    size_t best = frontier.size();
    for (size_t i = 0; i < frontier.size(); i++) {
      if (std::find(nodes.begin(), nodes.end(), cmp_pair(frontier[i])) !=
          nodes.end()) {
        best = i;
        break;
      }
    }
    if (best == frontier.size()) {
      if (nodes.size() >= CompactTree::kChunkPropertyLimit) {
        break;
      }
      best = 0;
      size_t highest = heights[frontier[0]];
      for (size_t i = 1; i < frontier.size(); i++) {
        if (heights[frontier[i]] > highest) {
          highest = heights[frontier[i]];
          best = i;
        }
      }
      nodes.push_back(cmp_pair(frontier[best]));
    }

    // Split tree.
    size_t node = frontier[best];
    std::swap(frontier[best], frontier.back());
    frontier.pop_back();

    size_t child = tree[node].childID;
    if (tree[child].property >= 0) {
      frontier.push_back(child);
    }
    if (tree[child + 1].property >= 0) {
      frontier.push_back(child + 1);
    }
  }

  if (out->chunks.size() >= kNumChunkLimit) {
    return false;
  }

  size_t total = 1 << nodes.size();

  // Create chunk data in *out.
  size_t chk_id = out->chunks.size();
  out->chunks.emplace_back();
  out->chunks[chk_id].start = out->table.size();
  out->table.resize(out->table.size() + total);

  std::vector<size_t> child_chunk(tree.size(), -1);

  // Not really needed, but gives a canonical representation to the chunk tree.
  std::sort(frontier.begin(), frontier.end());

  *height = 0;
  for (size_t x : frontier) {
    child_chunk[x] = out->chunks.size();
    size_t h;
    if (!SplitTree(tree, x, heights, &h, out)) {
      return false;
    }
    *height = std::max(*height, h);
  }
  *height += 1;

  CompactTree::ChunkData &chunk = out->chunks[chk_id];

  std::fill(std::begin(chunk.properties), std::end(chunk.properties), 0);
  // Nothing is strictly greater than int32_t::max(), so the unused values will
  // not affect the result.
  std::fill(std::begin(chunk.thresholds), std::end(chunk.thresholds),
            std::numeric_limits<int32_t>::max());

  chunk.num_properties = nodes.size();
  for (size_t i = 0; i < chunk.num_properties; i++) {
    chunk.properties[i] = nodes[i].first;
    chunk.thresholds[i] = nodes[i].second;
  }

  // Possible values for the given bit in the result: 0/1 or 2 for undecided.
  uint8_t state[CompactTree::kChunkPropertyLimit] = {};
  std::fill(state, state + chunk.num_properties, 2);
  BuildTable(state, tree, root, chunk, child_chunk, out);

  return true;
}

}  // namespace

void ChooseAndQuantizeProperties(
    size_t max_properties, size_t max_property_values,
    std::vector<std::vector<int>> *data, std::vector<int> *multiplicity,
    std::vector<std::vector<int>> *compact_properties,
    std::vector<size_t> *props_to_use) {
  // Remap all properties so that there are no holes nor negative numbers.
  std::unordered_map<int, int> remap;
  std::unordered_set<int> is_present;

  std::vector<int> remap_v;
  std::vector<int> is_present_v;

  // Threshold to switch to using a hash table for property remapping.
  static constexpr size_t kVectorMaxRange = 4096;

  for (size_t i = 1; i < data->size(); i++) {
    PropertyVal min = std::numeric_limits<PropertyVal>::max();
    PropertyVal max = std::numeric_limits<PropertyVal>::min();
    for (PropertyVal x : (*data)[i]) {
      min = std::min(min, x);
      max = std::max(max, x);
    }
    if (max - min + 1 < kVectorMaxRange) {
      is_present_v.clear();
      is_present_v.resize(max - min + 1);
      remap_v.resize(max - min + 1);
      for (size_t j = 0; j < (*data)[i].size(); j++) {
        size_t idx = (*data)[i][j] - min;
        if (!is_present_v[idx]) {
          (*compact_properties)[i - 1].push_back((*data)[i][j]);
        }
        is_present_v[idx] = 1;
      }
      std::sort((*compact_properties)[i - 1].begin(),
                (*compact_properties)[i - 1].end());
      for (size_t j = 0; j < (*compact_properties)[i - 1].size(); j++) {
        remap_v[(*compact_properties)[i - 1][j] - min] = j;
      }
      for (size_t j = 0; j < (*data)[i].size(); j++) {
        (*data)[i][j] = remap_v[(*data)[i][j] - min];
      }
    } else {
      is_present.clear();
      for (size_t j = 0; j < (*data)[i].size(); j++) {
        is_present.insert((*data)[i][j]);
      }
      (*compact_properties)[i - 1].assign(is_present.begin(), is_present.end());
      std::sort((*compact_properties)[i - 1].begin(),
                (*compact_properties)[i - 1].end());
      for (size_t j = 0; j < (*compact_properties)[i - 1].size(); j++) {
        remap[(*compact_properties)[i - 1][j]] = j;
      }
      for (size_t j = 0; j < (*data)[i].size(); j++) {
        (*data)[i][j] = remap.at((*data)[i][j]);
      }
    }
  }

  std::vector<std::pair<float, size_t>> props_with_entropy;
  ChooseEstimateEntropy(hwy::SupportedTargets())(data, compact_properties,
                                                 &props_with_entropy);

  // Limit the search to the properties with the smallest resulting entropy.
  max_properties = std::min(max_properties, props_with_entropy.size());
  props_to_use->resize(max_properties);
  for (size_t i = 0; i < max_properties; i++) {
    (*props_to_use)[i] = props_with_entropy[i].second - 1;
  }
  // Remove other properties from the data.
  size_t num_property_values = 0;
  std::sort(props_to_use->begin(), props_to_use->end());
  for (size_t i = 0; i < props_to_use->size(); i++) {
    if (i == (*props_to_use)[i]) continue;
    (*data)[i + 1] = std::move((*data)[(*props_to_use)[i] + 1]);
    (*compact_properties)[i] =
        std::move((*compact_properties)[(*props_to_use)[i]]);
    num_property_values += (*compact_properties)[i].size();
  }
  data->resize(props_to_use->size() + 1);
  compact_properties->resize(props_to_use->size());

  if (num_property_values > max_property_values) {
    // Quantize properties.
    // Note that tree uses *strictly greater* comparison nodes, so when merging
    // together a sequence of consecutive values we should keep the largest one.
    // TODO(veluca): find a smarter way to do the quantization, taking into
    // account the actual distribution of symbols.
    size_t thres = (*data)[0].size() / 256;
    std::vector<int> counts;
    std::vector<int> remap;
    std::vector<int> new_cp;
    for (size_t i = 0; i < max_properties; i++) {
      counts.clear();
      counts.resize((*compact_properties)[i].size());
      remap.resize((*compact_properties)[i].size());
      new_cp.clear();
      for (int v : (*data)[i + 1]) {
        counts[v]++;
      }
      size_t running_count = 0;
      size_t remapped = 0;
      for (size_t j = 0; j < counts.size(); j++) {
        remap[j] = remapped;
        running_count += counts[j];
        if (running_count > thres) {
          remapped++;
          running_count = 0;
        }
      }
      if (running_count != 0) remapped++;
      new_cp.resize(remapped, std::numeric_limits<int>::min());
      for (size_t j = 0; j < counts.size(); j++) {
        new_cp[remap[j]] =
            std::max(new_cp[remap[j]], (*compact_properties)[i][j]);
      }
      (*compact_properties)[i] = new_cp;
      for (size_t j = 0; j < (*data)[0].size(); j++) {
        (*data)[i + 1][j] = remap[(*data)[i + 1][j]];
      }
    }

    // Remove duplicated tuples.
    auto eq = [&](size_t a, size_t b) {
      for (size_t i = 0; i < data->size(); i++) {
        if ((*data)[i][a] != (*data)[i][b]) return false;
      }
      return true;
    };
    /* Removed: indices are not used after this
    auto lt = [&](size_t a, size_t b) {
      for (size_t i = 0; i < data->size(); i++) {
        if ((*data)[i][a] < (*data)[i][b]) return true;
        if ((*data)[i][a] > (*data)[i][b]) return false;
      }
      return false;
    };
    // TODO(veluca): radix sort?
    std::sort(indices.begin(), indices.end(), lt);
*/

    size_t distinct = 0;
    multiplicity->resize((*data)[0].size());
    for (size_t cur = 0; cur < (*data)[0].size();) {
      size_t start = cur;
      while (cur < (*data)[0].size() && eq(cur, start)) cur++;
      for (size_t i = 0; i < data->size(); i++) {
        (*data)[i][distinct] = (*data)[i][start];
      }
      (*multiplicity)[distinct] = cur - start;
      distinct++;
    }
    for (size_t i = 0; i < data->size(); i++) {
      (*data)[i].resize(distinct);
    }
    multiplicity->resize(distinct);
  } else {
    multiplicity->resize((*data)[0].size(), 1);
  }
}

void ComputeBestTree(const std::vector<std::vector<int>> &data,
                     const std::vector<int> &multiplicity,
                     const std::vector<std::vector<int>> compact_properties,
                     const std::vector<size_t> &props_to_use, float threshold,
                     size_t max_properties, Tree *tree) {
  std::vector<size_t> indices(data[0].size());
  std::iota(indices.begin(), indices.end(), 0);
  size_t num_symbols = *std::max_element(data[0].begin(), data[0].end()) + 1;
  ChooseFindBestSplit(hwy::SupportedTargets())(
      data, multiplicity, compact_properties, &indices, 0, 0, indices.size(),
      props_to_use, num_symbols, threshold, tree);
  size_t leaves = 0;
  for (size_t i = 0; i < tree->size(); i++) {
    if ((*tree)[i].property < 0) {
      (*tree)[i].childID = leaves++;
    }
  }
}

bool CompactifyTree(const Tree &tree, CompactTree *compact_tree) {
#if HWY_TARGET == HWY_SCALAR || !JXL_ARCH_X64
  // Compact tree are slower if SIMD is disabled or on non-x86 CPUs.
  return false;
#endif
  size_t max_property = 0;
  size_t max_leaf = 0;
  std::vector<size_t> heights(tree.size());
  TreeHeight(tree, 0, &heights);
  for (size_t i = 0; i < tree.size(); i++) {
    if (tree[i].property < 0) {
      max_leaf = std::max<size_t>(tree[i].childID, max_leaf);
    } else {
      max_property = std::max<size_t>(tree[i].property, max_property);
    }
  }
  // Property IDs should fit in a uint8_t.
  if (max_property > 255) return false;
  // Leaf IDs should fit in the negative part of a int16_t.
  if (max_leaf > 1 << 15) return false;
  size_t height;
  if (!SplitTree(tree, 0, heights, &height, compact_tree)) {
    // Tree splitting failed.
    return false;
  }
  return true;
}

namespace {
constexpr size_t kSplitValContext = 0;
constexpr size_t kPropertyContext = 1;
}  // namespace

// TODO(veluca): very simple encoding scheme. This should be improved.
void TokenizeTree(const Tree &tree, const HybridUintConfig &uint_config,
                  size_t base_ctx, std::vector<Token> *tokens,
                  Tree *decoder_tree) {
  std::queue<int> q;
  q.push(0);
  size_t leaf_id = 0;
  decoder_tree->clear();
  while (!q.empty()) {
    int cur = q.front();
    q.pop();
    JXL_ASSERT(tree[cur].property >= -1);
    TokenizeWithConfig(uint_config, base_ctx + kPropertyContext,
                       tree[cur].property + 1, tokens);
    if (tree[cur].property == -1) {
      decoder_tree->push_back(PropertyDecisionNode(-1, 0, leaf_id++));
      continue;
    }
    decoder_tree->push_back(
        PropertyDecisionNode(tree[cur].property, tree[cur].splitval,
                             decoder_tree->size() + q.size() + 1));
    q.push(tree[cur].childID);
    q.push(tree[cur].childID + 1);
    TokenizeWithConfig(uint_config, base_ctx + kSplitValContext,
                       PackSigned(tree[cur].splitval), tokens);
  }
}
Status DecodeTree(BitReader *br, ANSSymbolReader *reader,
                  const std::vector<uint8_t> &context_map, size_t base_ctx,
                  Tree *tree) {
  size_t leaf_id = 0;
  size_t to_decode = 1;
  tree->clear();
  while (to_decode > 0) {
    to_decode--;
    int property =
        reader->ReadHybridUint(base_ctx + kPropertyContext, br, context_map) -
        1;
    if (property == -1) {
      tree->push_back(PropertyDecisionNode(-1, 0, leaf_id++));
      continue;
    }
    int splitval = UnpackSigned(
        reader->ReadHybridUint(base_ctx + kSplitValContext, br, context_map));
    tree->push_back(
        PropertyDecisionNode(property, splitval, tree->size() + to_decode + 1));
    to_decode += 2;
  }
  return true;
}

#endif  // HWY_ONCE

}  // namespace jxl
