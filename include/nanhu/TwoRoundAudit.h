#include "WidePool.h"
#pragma once

#include "StagedRepair.h"

namespace toucan::two_round {
// Audit the unmodified RepCut groups. Never split MicroParts or add stages.
inline mlir::LogicalResult prepare(MultiRegionMicroPartScheduler &scheduler,
                                   PartitioningManager &pm,
                                   const PartitioningGraph &graph) {
  if (capacity::stagesOnly || scheduler.regionPartData.size() != 2 ||
      scheduler.regionPartData[0].empty() || scheduler.regionPartData[1].empty() ||
      capacity::localBudget < 18 || capacity::localBudget > toucanGPUSim::widePoolMaxBytes) {
    llvm::errs() << "Two-round compilation requires two nonempty original regions and a valid budget\n";
    return mlir::failure();
  }
  scheduler.prepareLayouts(graph, pm.exchangeValPool);
  const auto consts = capacity::constants(graph);
  std::ofstream log(pm.outputDirectory / "capacity-two-round.jsonl");
  if (!log) return mlir::failure();
  size_t rejected = 0, maximum = 0;
  for (size_t region = 0; region < 2; ++region) {
    const auto &parts = scheduler.regionPartData[region];
    llvm::outs() << "Two-round region " << region << ": " << parts.size() << " partitions\n";
    for (size_t part = 0; part < parts.size(); ++part) {
      auto &data = scheduler.regionPartData[region][part];
      const bool fits = capacity::allocate(data, consts);
      capacity::record(log, region, part, data, fits ? "accepted-two-round" : "rejected-two-round");
      const auto bytes = data.preparedAllocation->getCapacityStats().allocatedBytes;
      maximum = std::max(maximum, bytes);
      if (!fits) {
        if (rejected < 16)
          llvm::errs() << "Two-round overflow: region " << region << " partition " << part
                       << " needs " << bytes << " bytes; budget " << capacity::localBudget << "\n";
        ++rejected;
      }
    }
  }
  llvm::outs() << "Two-round capacity: maximum " << maximum << " bytes; " << rejected
               << " over budget; repair and staged fallback disabled\n";
  return rejected ? mlir::failure() : mlir::success();
}
} // namespace toucan::two_round
