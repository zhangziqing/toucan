// Preserve graph-level traversal order while selecting only one partition.
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

class PartitionLevelIndex {
  std::vector<std::pair<uint32_t, uint32_t>> order;
  uint32_t numLevels;
public:
  template <typename Levels>
  PartitionLevelIndex(uint32_t numVertices, const Levels &levels)
      : order(numVertices, {UINT32_MAX, UINT32_MAX}), numLevels(levels.size()) {
    uint32_t rank = 0;
    for (uint32_t level = 0; level < levels.size(); ++level)
      for (auto vertex : levels[level]) {
        assert(vertex < order.size() && order[vertex].second == UINT32_MAX);
        order[vertex] = {level, rank++};
      }
  }

  template <typename Output, typename Nodes>
  Output select(const Nodes &nodes) const {
    std::vector<uint32_t> sorted;
    sorted.reserve(nodes.size());
    for (auto vertex : nodes) {
      assert(vertex < order.size());
      if (order[vertex].second != UINT32_MAX) sorted.push_back(vertex);
    }
    std::sort(sorted.begin(), sorted.end(), [&](auto a, auto b) {
      return order[a].second < order[b].second;
    });
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    Output result(numLevels);
    for (auto vertex : sorted) result[order[vertex].first].push_back(vertex);
    return result;
  }
};
