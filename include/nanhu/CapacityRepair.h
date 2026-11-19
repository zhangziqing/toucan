#pragma once

#include "toucan/MicroPartLocalValueAllocator.h"
#include "llvm/Support/CommandLine.h"
#include <fstream>
#include <functional>

namespace toucan::capacity {
inline llvm::cl::opt<unsigned> localBudget(
    "local-value-budget", llvm::cl::init(UINT16_MAX - 1),
    llvm::cl::desc("Maximum local value pool bytes per GPU partition"));
inline mlir::Value storage(mlir::Value v) {
  if (auto s = mlir::dyn_cast<StaticVectorSegmentReadOp>(v.getDefiningOp()))
    return s.getHandle();
  return v;
}
inline bool constant(mlir::Value v) {
  return mlir::isa<ConstantOp, DefConstVectorOp>(v.getDefiningOp());
}

// Terminal operations are indivisible. Keep all ports of a memory together.
struct Sink {
  mlir::SmallVector<RegWriteOp> regs;
  mlir::SmallVector<MemWriteOp> mems;
  mlir::SmallVector<PrintOp> prints;
  mlir::SmallVector<StopOp> stops;
  mlir::SmallVector<mlir::Value> exchange;
  mlir::SmallVector<mlir::Value> inputs() const {
    mlir::SmallVector<mlir::Value> out(exchange);
    for (auto op : regs) out.push_back(op.getData());
    for (auto op : mems) {
      out.push_back(op.getAddrVec()); out.push_back(op.getEn()); out.push_back(op.getData());
    }
    for (auto op : prints) out.push_back(op.getEn());
    for (auto op : stops) out.push_back(op.getEn());
    return out;
  }
};
inline std::vector<Sink> sinks(const RepCutPartitionCodeGenData &p) {
  std::vector<Sink> out;
  for (auto v : p.allExgWriteVals) { Sink s; s.exchange.push_back(v); out.push_back(s); }
  for (auto op : p.allRegWrites) { Sink s; s.regs.push_back(op); out.push_back(s); }
  mlir::DenseMap<mlir::Value, size_t> memories;
  for (auto op : p.allMemWrites) {
    auto [it, inserted] = memories.try_emplace(op.getMem(), out.size());
    if (inserted) out.emplace_back();
    out[it->second].mems.push_back(op);
  }
  for (auto op : p.allPrints) { Sink s; s.prints.push_back(op); out.push_back(s); }
  for (auto op : p.allStops) { Sink s; s.stops.push_back(op); out.push_back(s); }
  return out;
}
inline mlir::DenseMap<mlir::Value, uint32_t> constants(const PartitioningGraph &g) {
  mlir::DenseMap<mlir::Value, uint32_t> out;
  for (auto v : boost::make_iterator_range(boost::vertices(g)))
    if (g[v].op)
      if (auto op = mlir::dyn_cast<ConstantOp>(g[v].op))
        out[op.getResult()] = op.getValue().getZExtValue();
  return out;
}
inline bool allocate(RepCutPartitionCodeGenData &p,
                     const mlir::DenseMap<mlir::Value, uint32_t> &consts) {
  auto a = std::make_shared<MicroPartLocalValueAllocator>();
  a->collectValueLifetime(p);
  a->populateInitialPinnedVals(p, consts);
  a->allocateLocalValues();
  p.preparedAllocation = a;
  return a->getCapacityStats().fitsBudget(localBudget);
}

// Reconstruct complete fan-in, including every producer of a shared vector.
class Cone {
  const RepCutPartitionCodeGenData &original;
  mlir::DenseMap<mlir::Value, mlir::SmallVector<const MicroPart *>> producers;
public:
  explicit Cone(const RepCutPartitionCodeGenData &p) : original(p) {
    for (const auto &level : p.mpartLevels)
      for (const auto &mp : level)
        for (auto v : mp->outputValueSet) producers[storage(v)].push_back(mp.get());
  }
  bool subset(llvm::ArrayRef<Sink> terminals, RepCutPartitionCodeGenData &out) const {
    mlir::DenseSet<const MicroPart *> selected;
    mlir::DenseSet<mlir::Value> needed;
    mlir::SmallVector<mlir::Value> pending;
    for (const auto &s : terminals) {
      out.allRegWrites.append(s.regs); out.allMemWrites.append(s.mems);
      out.allPrints.append(s.prints); out.allStops.append(s.stops);
      out.allExgWriteVals.append(s.exchange);
      pending.append(s.inputs());
    }
    while (!pending.empty()) {
      auto v = storage(pending.pop_back_val());
      if (constant(v) || !needed.insert(v).second) continue;
      auto found = producers.find(v);
      if (found == producers.end()) continue;
      for (auto mp : found->second)
        if (selected.insert(mp).second)
          pending.append(mp->inputValues.begin(), mp->inputValues.end());
    }
    mlir::DenseSet<mlir::Value> available;
    for (auto op : original.allRegReads)
      if (needed.contains(op.getResult())) {
        out.allRegReads.push_back(op); out.readRegs.push_back(op.getReg());
        available.insert(op.getResult());
      }
    for (auto v : original.allExgReadVals)
      if (needed.contains(v)) { out.allExgReadVals.push_back(v); available.insert(v); }
    for (const auto &level : original.mpartLevels) {
      mlir::SmallVector<std::shared_ptr<MicroPart>> next;
      for (const auto &mp : level)
        if (selected.contains(mp.get())) next.push_back(mp);
      if (next.empty()) continue;
      // Same-level execution is concurrent: publish outputs after all inputs.
      for (const auto &mp : next)
        for (auto v : mp->inputValues)
          if (!constant(v) && !available.contains(storage(v))) return false;
      for (const auto &mp : next)
        for (auto v : mp->outputValueSet) available.insert(storage(v));
      out.mpartLevels.push_back(std::move(next));
    }
    for (auto op : out.allRegWrites) out.writeRegs.push_back(op.getReg());
    for (const auto &s : terminals)
      for (auto v : s.inputs())
        if (!constant(v) && !available.contains(storage(v))) return false;
    return true;
  }
};

inline void record(std::ostream &os, size_t region, size_t part,
                   const RepCutPartitionCodeGenData &p, const char *phase) {
  const auto s = p.preparedAllocation->getCapacityStats();
  os << "{\"phase\":\"" << phase << "\",\"region\":" << region
     << ",\"part\":" << part << ",\"budget\":" << localBudget
     << ",\"input_bytes\":" << s.pinnedInputBytes
     << ",\"output_bytes\":" << s.pinnedOutputBytes
     << ",\"pinned_pool_bytes\":" << s.initialPinnedBytes
     << ",\"peak_live_bytes\":" << s.peakLiveBytes
     << ",\"peak_temporary_bytes\":" << s.peakTemporaryBytes
     << ",\"peak_level\":" << s.peakLiveLevel
     << ",\"output_copy_end_bytes\":" << s.outputCopyEndBytes
     << ",\"allocated_bytes\":" << s.allocatedBytes
     << ",\"fragmented_vector_appends\":" << s.fragmentedVectorAppends << "}\n";
}

// Return false only when an individual sink needs an internal cone cut.
inline bool split(const RepCutPartitionCodeGenData &original, llvm::ArrayRef<Sink> terminals,
                  const mlir::DenseMap<mlir::Value, uint32_t> &consts,
                  std::vector<RepCutPartitionCodeGenData> &out) {
  RepCutPartitionCodeGenData candidate;
  if (!Cone(original).subset(terminals, candidate)) return false;
  if (allocate(candidate, consts)) { out.push_back(std::move(candidate)); return true; }
  if (terminals.size() <= 1) return false;
  const auto mid = terminals.size() / 2;
  return split(original, terminals.take_front(mid), consts, out) &&
         split(original, terminals.drop_front(mid), consts, out);
}

inline bool fitExecutionLevels(RepCutPartitionCodeGenData &,
    const mlir::DenseMap<mlir::Value, uint32_t> &);

inline mlir::LogicalResult repair(std::vector<std::vector<RepCutPartitionCodeGenData>> &regions,
                                 const PartitioningGraph &graph, std::filesystem::path output, bool &changed) {
  changed = false;
  if (localBudget < 18 || localBudget >= UINT16_MAX) {
    llvm::errs() << "local-value-budget must be in [18, 65534]\n";
    return mlir::failure();
  }
  const auto consts = constants(graph);
  std::ofstream log(output / "capacity.jsonl", std::ios::app);
  if (!log) return mlir::failure();
  auto repaired = regions;
  for (size_t r = 0; r < regions.size(); ++r) {
    repaired[r].clear();
    for (size_t p = 0; p < regions[r].size(); ++p) {
      auto &part = regions[r][p];
      const bool fits = allocate(part, consts);
      record(log, r, p, part, "original");
      if (fits) { repaired[r].push_back(part); continue; }
      if (fitExecutionLevels(part, consts)) {
        changed = true; repaired[r].push_back(part); continue;
      }
      std::vector<RepCutPartitionCodeGenData> children;
      if (!split(part, sinks(part), consts, children)) {
        llvm::errs() << "Capacity repair needs an internal cone cut in region " << r
                     << " partition " << p << "\n";
        return mlir::failure();
      }
      changed = true;
      llvm::outs() << "Capacity repair: region " << r << " partition " << p
                   << " -> " << children.size() << " children\n";
      for (auto &child : children) repaired[r].push_back(std::move(child));
    }
  }
  regions = std::move(repaired);
  for (size_t r = 0; r < regions.size(); ++r)
    for (size_t p = 0; p < regions[r].size(); ++p) record(log, r, p, regions[r][p], "accepted");
  return mlir::success();
}
} // namespace toucan::capacity
