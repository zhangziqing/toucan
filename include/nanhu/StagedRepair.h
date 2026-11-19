#pragma once
#include "CapacityRepair.h"
#include "RefineMicroParts.h"
#include "CapacityLevels.h"
#include "boost/graph/strong_components.hpp"
#include "mlir/IR/Builders.h"
#include <map>
#include <numeric>
#include <queue>

namespace toucan::capacity {
inline llvm::cl::opt<bool> stagesOnly("capacity-stages-only", llvm::cl::init(false),
    llvm::cl::desc("Build ordered capacity stages directly (development replay)"));
// Internal cuts operate on the unreplicated union of the two region graphs.
// All state effects remain in the final stage; all exchange locations are unique
// and retained for the entire cycle. No architectural state is introduced.
class StagedRepair {
  PartitioningManager &pm;
  const PartitioningGraph &graph;
  std::vector<std::shared_ptr<MicroPart>> units;
  std::vector<Sink> terminal;
  mlir::DenseMap<mlir::Value, RegReadOp> registers;
  mlir::DenseMap<mlir::Value, mlir::SmallVector<unsigned>> producers, users;
  mlir::DenseSet<mlir::Value> terminalValues;
  std::vector<unsigned> root, level;
  mlir::SmallVector<unsigned> failedAtom;
  mlir::DenseMap<mlir::Value, uint32_t> consts;
  unsigned find(unsigned i) { return root[i] == i ? i : root[i] = find(root[i]); }
  void join(unsigned a, unsigned b) { root[find(b)] = find(a); }
  template<class V> static void add(V &list, mlir::Value v) {
    if (!llvm::is_contained(list, v)) list.push_back(v);
  }
  bool load() {
    RepCutPartitionCodeGenData effects;
    for (auto g : {pm.microPartGraph_r0, pm.microPartGraph_r1}) {
      for (auto id : boost::make_iterator_range(boost::vertices(*g))) {
        const auto &node = (*g)[id];
        switch (node.toucanOpName) {
        case CGToucanOPName::MPart_Regular:
        case CGToucanOPName::MPart_Special: units.push_back(node.mp); break;
        case CGToucanOPName::LUT: {
          auto mp = std::make_shared<MicroPart>();
          mlir::SmallVector<LUTOp> nops{mlir::cast<LUTOp>(node.op)};
          mp->buildNOPRegularLUTPart(nops); units.push_back(mp); break;
        }
        case CGToucanOPName::RegRead: {
          auto op = mlir::cast<RegReadOp>(node.op);
          registers[op.getResult()] = op; break;
        }
        case CGToucanOPName::RegWrite:
          effects.allRegWrites.push_back(mlir::cast<RegWriteOp>(node.op)); break;
        case CGToucanOPName::MemWrite: {
          auto op = mlir::cast<MemWriteOp>(node.op);
          for (auto u : op.getMem().getUsers())
            if (auto mw = mlir::dyn_cast<MemWriteOp>(u)) effects.allMemWrites.push_back(mw);
          break;
        }
        case CGToucanOPName::Print: effects.allPrints.push_back(mlir::cast<PrintOp>(node.op)); break;
        case CGToucanOPName::Stop: effects.allStops.push_back(mlir::cast<StopOp>(node.op)); break;
        case CGToucanOPName::ExchangeRead:
        case CGToucanOPName::ExchangeWrite: break;
        default: return false;
        }
      }
    }
    // Materialize direct register/constant terminal operands before commit.
    // Only terminal uses are rewritten; the inserted identity retains its input.
    mlir::DenseMap<mlir::Value, mlir::Value> identities;
    auto materialize = [&](mlir::Operation *op, bool regWrite) {
      for (auto &operand : op->getOpOperands()) {
        auto v = operand.get();
        if (!registers.contains(v) && !(regWrite && mlir::isa<ConstantOp>(v.getDefiningOp()))) continue;
        if (!identities.contains(v)) {
          mlir::OpBuilder b(v.getDefiningOp()); b.setInsertionPointAfter(v.getDefiningOp());
          auto nop = b.create<LUTOp>(op->getLoc(), LUTOpName::LUT_Nop, v);
          auto mp = std::make_shared<MicroPart>(); mlir::SmallVector<LUTOp> nops{nop};
          mp->buildNOPRegularLUTPart(nops);
          if (constant(v)) mp->inputValues.erase(v);
          units.push_back(mp); identities[v] = nop.getResult();
        }
        operand.set(identities[v]);
      }
    };
    for (auto op : effects.allRegWrites) materialize(op, true);
    for (auto op : effects.allMemWrites) materialize(op, false);
    for (auto op : effects.allPrints) materialize(op, false);
    for (auto op : effects.allStops) materialize(op, false);
    if (!refine(units, graph, *pm.mp)) return false;
    terminal = sinks(effects);
    for (const auto &sink : terminal)
      for (auto v : sink.inputs()) if (!constant(v)) terminalValues.insert(storage(v));
    for (unsigned i = 0; i < units.size(); ++i) {
      for (auto v : units[i]->outputValueSet) producers[storage(v)].push_back(i);
      for (auto v : units[i]->inputValues) users[storage(v)].push_back(i);
    }
    return true;
  }

  bool group() {
    root.resize(units.size()); std::iota(root.begin(), root.end(), 0);
    // Multiple partial writes to one vector must share a partition and writer.
    for (const auto &[v, ids] : producers)
      for (auto id : ids) join(ids.front(), id);
    using DAG = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>;
    DAG raw(units.size()), grouped(units.size());
    std::vector<unsigned> degree(units.size()); level.resize(units.size());
    for (unsigned i = 0; i < units.size(); ++i)
      for (auto v : units[i]->inputValues) {
        if (constant(v) || registers.contains(storage(v))) continue;
        auto it = producers.find(storage(v));
        if (it == producers.end()) { llvm::errs() << "Missing staged producer\n"; return false; }
        for (auto src : it->second) if (src != i) {
          boost::add_edge(src, i, raw); ++degree[i];
          boost::add_edge(find(src), find(i), grouped);
        }
      }
    std::queue<unsigned> ready;
    for (unsigned i = 0; i < units.size(); ++i) if (!degree[i]) ready.push(i);
    size_t seen = 0;
    while (!ready.empty()) {
      auto i = ready.front(); ready.pop(); ++seen;
      for (auto edge : boost::make_iterator_range(boost::out_edges(i, raw))) {
        auto dst = boost::target(edge, raw); level[dst] = std::max(level[dst], level[i] + 1);
        if (!--degree[dst]) ready.push(dst);
      }
    }
    if (seen != units.size()) { llvm::errs() << "Cycle in staged MicroPart dependencies\n"; return false; }
    // Co-producer grouping may create cycles; collapse them before staging.
    std::vector<unsigned> component(units.size());
    boost::strong_components(grouped, boost::make_iterator_property_map(
        component.begin(), boost::get(boost::vertex_index, grouped)));
    std::map<unsigned, unsigned> first;
    for (unsigned i = 0; i < units.size(); ++i) {
      auto [it, inserted] = first.emplace(component[find(i)], i);
      if (!inserted) join(it->second, i);
    }
    return true;
  }

  RepCutPartitionCodeGenData build(llvm::ArrayRef<unsigned> ids) {
    RepCutPartitionCodeGenData p;
    mlir::DenseSet<unsigned> selected(ids.begin(), ids.end());
    mlir::DenseSet<mlir::Value> produced;
    for (auto id : ids) for (auto v : units[id]->outputValueSet) produced.insert(storage(v));
    std::map<unsigned, mlir::SmallVector<std::shared_ptr<MicroPart>>> levels;
    for (auto id : ids) {
      levels[level[id]].push_back(units[id]);
      for (auto input : units[id]->inputValues) {
        auto v = storage(input);
        if (constant(v) || produced.contains(v)) continue;
        if (registers.contains(v)) {
          auto op = registers.at(v);
          if (!llvm::is_contained(p.allRegReads, op)) { p.allRegReads.push_back(op); p.readRegs.push_back(op.getReg()); }
        } else add(p.allExgReadVals, v);
      }
      for (auto v : units[id]->outputValueSet) {
        v = storage(v); bool external = terminalValues.contains(v);
        for (auto user : users.lookup(v)) external |= !selected.contains(user);
        if (external) add(p.allExgWriteVals, v);
      }
    }
    for (auto &entry : levels) p.mpartLevels.push_back(std::move(entry.second));
    return p;
  }
  bool pack(llvm::ArrayRef<mlir::SmallVector<unsigned>> atoms,
            std::vector<RepCutPartitionCodeGenData> &parts) {
    mlir::SmallVector<unsigned> ids;
    for (const auto &atom : atoms) ids.append(atom);
    auto p = build(ids);
    if (allocate(p, consts)) { parts.push_back(std::move(p)); return true; }
    if (atoms.size() <= 1) {
      llvm::errs() << "Indivisible staged atom needs " << p.preparedAllocation->numTotalValSize
                   << " bytes from " << ids.size() << " MicroParts; budget " << localBudget << "\n";
      failedAtom = ids; return false;
    }
    const auto mid = atoms.size() / 2;
    return pack(atoms.take_front(mid), parts) && pack(atoms.drop_front(mid), parts);
  }
  bool packEffects(llvm::ArrayRef<Sink> effects, std::vector<RepCutPartitionCodeGenData> &parts) {
    RepCutPartitionCodeGenData p;
    for (const auto &s : effects) {
      p.allRegWrites.append(s.regs); p.allMemWrites.append(s.mems);
      p.allPrints.append(s.prints); p.allStops.append(s.stops);
      for (auto v : s.inputs()) if (!constant(v)) add(p.allExgReadVals, storage(v));
    }
    for (auto op : p.allRegWrites) p.writeRegs.push_back(op.getReg());
    if (allocate(p, consts)) { parts.push_back(std::move(p)); return true; }
    if (effects.size() <= 1) return false;
    return packEffects(effects.take_front(effects.size()/2), parts) &&
           packEffects(effects.drop_front(effects.size()/2), parts);
  }
public:
  StagedRepair(PartitioningManager &pm, const PartitioningGraph &g) : pm(pm), graph(g), consts(constants(g)) {}
  bool plan(std::vector<std::vector<RepCutPartitionCodeGenData>> &regions) {
    if (!group()) return false;
    std::map<unsigned, mlir::SmallVector<unsigned>> atoms;
    for (unsigned i = 0; i < units.size(); ++i) atoms[find(i)].push_back(i);
    std::map<unsigned, unsigned> stage;
    // The collapsed graph is a DAG. Compute stages by monotone relaxation.
    bool changed = true;
    for (size_t round = 0; changed && round <= atoms.size(); ++round) {
      changed = false;
      for (const auto &[r, ids] : atoms)
        for (auto id : ids) for (auto v : units[id]->inputValues)
          for (auto src : producers.lookup(storage(v))) if (find(src) != r && stage[r] <= stage[find(src)]) {
            stage[r] = stage[find(src)] + 1; changed = true;
          }
      if (changed && round == atoms.size()) return false;
    }
    std::map<unsigned, std::vector<mlir::SmallVector<unsigned>>> stages;
    for (auto &[r, ids] : atoms) stages[stage[r]].push_back(ids);
    regions.clear(); pm.exchangeValPool.clear(); pm.exchangeValues.clear();
    for (const auto &[s, groups] : stages) {
      regions.emplace_back(); if (!pack(groups, regions.back())) return false;
    }
    regions.emplace_back(); if (!packEffects(terminal, regions.back())) return false;
    std::ofstream log(pm.outputDirectory / "capacity-staged.jsonl");
    mlir::DenseMap<mlir::Value, size_t> writer;
    for (size_t s = 0; s < regions.size(); ++s) for (size_t p = 0; p < regions[s].size(); ++p) {
      record(log, s, p, regions[s][p], "staged");
      for (auto v : regions[s][p].allExgReadVals)
        if (!writer.contains(v) || writer.at(v) >= s) return false;
      for (auto v : regions[s][p].allExgWriteVals) {
        if (writer.contains(v)) return false;
        writer[v] = s; pm.exchangeValues[v] = pm.exchangeValPool.size(); pm.exchangeValPool.push_back(v);
      }
    }
    llvm::outs() << "Capacity repair: " << regions.size() << " ordered stages within one cycle\n";
    return true;
  }
  bool run(std::vector<std::vector<RepCutPartitionCodeGenData>> &regions) {
    if (!load()) return false;
    for (unsigned attempt = 0; attempt < 16; ++attempt) {
      failedAtom.clear(); level.clear();
      if (plan(regions)) return true;
      if (failedAtom.empty()) return false;
      mlir::DenseSet<unsigned> failed(failedAtom.begin(), failedAtom.end());
      std::vector<std::shared_ptr<MicroPart>> children, remaining;
      for (unsigned i = 0; i < units.size(); ++i)
        (failed.contains(i) ? children : remaining).push_back(units[i]);
      const auto before = children.size();
      if (!refine(children, graph, *pm.mp, true) || children.size() <= before) return false;
      llvm::outs() << "Refine capacity atom: " << before << " -> " << children.size() << " MicroParts\n";
      remaining.insert(remaining.end(), children.begin(), children.end()); units = std::move(remaining);
      producers.clear(); users.clear();
      for (unsigned i = 0; i < units.size(); ++i) {
        for (auto v : units[i]->outputValueSet) producers[storage(v)].push_back(i);
        for (auto v : units[i]->inputValues) users[storage(v)].push_back(i);
      }
    }
    llvm::errs() << "Capacity atom refinement did not stabilize\n"; return false;
  }
};

inline mlir::LogicalResult prepare(MultiRegionMicroPartScheduler &scheduler,
                                   PartitioningManager &pm, const PartitioningGraph &graph) {
  if (localBudget < 18 || localBudget >= UINT16_MAX) return mlir::failure();
  auto data = std::move(scheduler.regionPartData);
  bool staged = false;
  if (stagesOnly) {
    if (!StagedRepair(pm, graph).run(data)) return mlir::failure();
    staged = true;
  }
  for (unsigned round = 0; round < 64; ++round) {
    MultiRegionMicroPartScheduler candidate;
    candidate.regionPartData = std::move(data);
    candidate.prepareLayouts(graph, pm.exchangeValPool);
    bool changed = false;
    if (mlir::failed(repair(candidate.regionPartData, graph, pm.outputDirectory, changed))) {
      if (staged || !StagedRepair(pm, graph).run(candidate.regionPartData)) return mlir::failure();
      staged = true; changed = true;
    }
    if (changed) { data = std::move(candidate.regionPartData); continue; }
    scheduler.regionPartData = std::move(candidate.regionPartData);
    scheduler.codeGenInfo = std::move(candidate.codeGenInfo);
    scheduler.layoutPrepared = true;
    return mlir::success();
  }
  llvm::errs() << "Capacity repair did not stabilize after 64 layout rounds\n";
  return mlir::failure();
}
} // namespace toucan::capacity
