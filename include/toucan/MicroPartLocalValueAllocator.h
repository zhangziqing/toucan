
#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Value.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>

#include "mlir/Support/LogicalResult.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"

#include "toucan/MicroPartitioner.h"

#include <boost/graph/adjacency_list.hpp>

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "toucan/PartitioningManager.h"

namespace toucan {
class MicroPartLocalValueAllocator {
public:
  struct CapacityStats {
    size_t constantBytes = 0, pinnedInputBytes = 0, pinnedOutputBytes = 0;
    size_t initialPinnedBytes = 0, peakLiveBytes = 0, peakTemporaryBytes = 0;
    size_t peakLiveLevel = 0, peakTemporaryLevel = 0, allocatedBytes = 0;
    size_t outputCopyEndBytes = 0;
    size_t fragmentedVectorAppends = 0;
    size_t maxEligibleFreeBytesOnAppend = 0, maxEligibleFreeRunOnAppend = 0;

    bool fitsBudget(size_t budgetBytes) const {
      return allocatedBytes <= budgetBytes;
    }
  };

  // Call after lifetime collection, pin population, and allocation.
  CapacityStats getCapacityStats() const;

  size_t numTotalValSize = 0;
  // numConsts = compactConstValPool.size()
  size_t numConsts;
  size_t numOutputVals;
  size_t numInputVals;

  // Value pool:
  // const values, input values, output values, tempories
  mlir::DenseMap<mlir::Value, uint32_t> valToValId;
  mlir::SmallVector<mlir::Value> valuePool;
  // Use this as the real const pool
  // still, val 0 is always 0
  mlir::SmallVector<uint8_t> compactConstValPool;

  mlir::DenseSet<mlir::Value> activeValuesAtLast;

  void allocateLocalValuesWithoutReclaim();

  void allocateLocalValues();

  // uint32_t getValId(mlir::Value);

  void collectValueLifetime(RepCutPartitionCodeGenData &partData);

  // Populate const vals and RegWrite/ExchangeWrite. Those values are pinned
  void populateInitialPinnedVals(RepCutPartitionCodeGenData &partData,
                                 const mlir::DenseMap<mlir::Value, uint32_t> &constValToRawValue);

private:
  CapacityStats fragmentationStats;
  size_t outputCopyEnd() const;

  struct ValueLifeTime {
    uint32_t start;
    uint32_t end;
  };

  mlir::DenseSet<mlir::Value> pinnedInputVals, pinnedOutputVals, constVals;

  mlir::DenseMap<mlir::Value, ValueLifeTime> valToLifeTime;
  // mlir::DenseMap<mlir::Value, uint32_t> vecValToLength;
  mlir::DenseMap<mlir::Value, uint32_t> vecValToLength;

  mlir::DenseMap<mlir::Value, mlir::Value> vecSegmentsToVecArith;
  mlir::DenseMap<mlir::Value, mlir::SmallVector<mlir::Value>> vecArithResultToSegments;
  mlir::DenseSet<mlir::Value> vecArithAndSegmentValues;

  size_t totalLevels;
};
} // namespace toucan
