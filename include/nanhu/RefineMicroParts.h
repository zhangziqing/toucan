#pragma once
#include "CapacityRepair.h"

namespace toucan::capacity {
// MicroParts use warp registers internally; only boundary values occupy the
// shared local pool. Refine an oversized boundary before treating it as an atom.
inline bool refine(std::vector<std::shared_ptr<MicroPart>> &units,
                   const PartitioningGraph &graph, MicroPartitioner &mp, bool force = false) {
  auto original = std::move(units);
  for (const auto &part : original) {
    mlir::DenseSet<mlir::Value> boundary;
    for (auto v : part->inputValues) if (!constant(v)) boundary.insert(storage(v));
    for (auto v : part->outputValueSet) boundary.insert(storage(v));
    size_t bytes = 17;
    for (auto v : boundary) {
      auto vec = mlir::dyn_cast<VecType>(v.getType());
      bytes += vec ? vec.getLength() : 1;
    }
    if (!force && bytes <= localBudget) { units.push_back(part); continue; }
    if (part->isNOPPart) {
      for (auto nop : part->nops) {
        auto one = std::make_shared<MicroPart>(); mlir::SmallVector<LUTOp> nops{nop};
        one->buildNOPRegularLUTPart(nops); units.push_back(one);
      }
    } else if (part->isRegularPart()) {
      for (const auto &level : part->levels) for (auto id : level) {
        auto one = std::make_shared<MicroPart>();
        one->buildRegularLUTPart({{id}});
        if (!one->checkAndCollectIOValues(graph, mp.allNodes, mp.newNodeIdToDepNodeId,
                                        mp.newNodeIdToOriginalVecDeclId, mp.outputVectorNopMap)) return false;
        units.push_back(one);
      }
    } else {
      for (auto op : part->specialOps) {
        auto one = std::make_shared<MicroPart>(); one->buildSpecialPart(part->opType, {op});
        if (!one->checkAndCollectIOValues(graph, mp.allNodes, mp.newNodeIdToDepNodeId,
                                        mp.newNodeIdToOriginalVecDeclId, mp.outputVectorNopMap)) return false;
        units.push_back(one);
      }
    }
  }
  return true;
}
} // namespace toucan::capacity
