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

#include "jxl/modular/ma/compound.h"

#include <algorithm>
#include <limits>
#include <set>

#include "jxl/base/arch_specific.h"

namespace jxl {
static constexpr size_t kNumChunkLimit = 1024;

size_t TreeHeight(const Tree& tree, size_t root, std::vector<size_t>* heights) {
  if (tree[root].property < 0) {
    (*heights)[root] = 0;
    return 1;
  }
  size_t subh = std::max(TreeHeight(tree, tree[root].childID, heights),
                         TreeHeight(tree, tree[root].childID + 1, heights));
  (*heights)[root] = subh;
  return subh + 1;
}

void BuildTable(uint8_t* JXL_RESTRICT state, const Tree& tree, size_t root,
                const CompactTree::ChunkData& chunk,
                const std::vector<size_t>& child_chunk, CompactTree* out) {
  int16_t value;
  bool has_value = false;
  if (child_chunk[root] != -1) {
    value = child_chunk[root];
    has_value = true;
  } else {
    const PropertyDecisionNode& node = tree[root];
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

bool SplitTree(const Tree& tree, size_t root,
               const std::vector<size_t>& heights, size_t* height,
               CompactTree* out) {
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

  CompactTree::ChunkData& chunk = out->chunks[chk_id];

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

bool CompactifyTree(const Tree& tree, CompactTree* compact_tree) {
#if HWY_BITS == 0 || !JXL_ARCH_X64
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

}  // namespace jxl
