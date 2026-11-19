#pragma once
#include "CapacityRepair.h"

namespace toucan::capacity {
// Independent MicroParts in one level can run in smaller sequential batches.
// The original level completes before any original downstream level starts.
// Lifetime collection therefore retains same-batch overlap and vector aliases.
inline bool fitExecutionLevels(RepCutPartitionCodeGenData &part,
    const mlir::DenseMap<mlir::Value, uint32_t> &consts) {
  const auto original = part.mpartLevels;
  for (size_t batch : {64, 16, 4, 1}) {
    RepCutPartitionCodeGenData candidate = part;
    candidate.mpartLevels.clear();
    for (const auto &level : original)
      for (size_t i = 0; i < level.size(); i += batch) {
        auto end = std::min(level.size(), i + batch);
        candidate.mpartLevels.emplace_back(level.begin() + i, level.begin() + end);
      }
    if (allocate(candidate, consts)) {
      llvm::outs() << "Capacity repair: batch " << batch << " independent MicroParts per level; "
                   << original.size() << " -> " << candidate.mpartLevels.size() << " levels\n";
      part = std::move(candidate);
      return true;
    }
  }
  return false;
}
} // namespace toucan::capacity
