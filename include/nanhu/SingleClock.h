// Fixed-workload single-clock model. Clock channels carry root-cycle enables.
// Every state transition uses the pre-commit snapshot; no event scheduler exists.
#pragma once
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

namespace single_clock {
using namespace mlir;
using namespace circt;

struct Normalize : PassWrapper<Normalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Normalize)
  StringRef getArgument() const override { return "normalize-single-clock"; }
  StringRef getDescription() const override {
    return "Lower fixed simulation clocks to root-cycle state enables";
  }
  uint64_t registers = 0, writes = 0, reads = 0, gates = 0, casts = 0, aliases = 0;
  Value rootClock;
  llvm::StringMap<hw::HWModuleOp> modules;
  llvm::StringMap<SmallVector<hw::InstanceOp>> callers;
  llvm::DenseMap<Value, int> constants;
  llvm::DenseMap<Value, bool> clockCarriers;
  llvm::DenseSet<Operation *> aliasCasts;

  // A Bool clock alias must originate at FromClock through only module ports.
  // Check every caller: one instance carrying data invalidates the shared port.
  // Cache the proof before rewriting types or erasing FromClock operations.
  bool isClockCarrier(Value value) {
    auto found = clockCarriers.find(value);
    if (found != clockCarriers.end()) return found->second;
    clockCarriers[value] = false; // Reject alias cycles conservatively.
    bool carrier = false;
    if (value.getDefiningOp<seq::FromClockOp>()) {
      carrier = true;
    } else if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto mod = dyn_cast<hw::HWModuleOp>(arg.getOwner()->getParentOp());
      if (mod) {
        auto &instances = callers[mod.getSymName()];
        carrier = !instances.empty() && llvm::all_of(instances, [&](hw::InstanceOp inst) {
          return isClockCarrier(inst->getOperand(arg.getArgNumber()));
        });
      }
    } else if (auto inst = value.getDefiningOp<hw::InstanceOp>()) {
      if (auto mod = modules.lookup(inst.getModuleName()))
        carrier = isClockCarrier(mod.getBodyBlock()->getTerminator()->getOperand(
            cast<OpResult>(value).getResultNumber()));
    }
    clockCarriers[value] = carrier;
    return carrier;
  }

  // Prove constants across every instance of a module, conservatively. Unknown
  // values and cycles fail the proof; a module name alone is never a tie-off.
  int constant(Value v) {
    auto found = constants.find(v);
    if (found != constants.end()) return found->second;
    constants[v] = -1;
    int result = -1;
    if (auto c = v.getDefiningOp<hw::ConstantOp>()) {
      if (c.getValue().getBitWidth() == 1) result = !c.getValue().isZero();
    } else if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto mod = dyn_cast<hw::HWModuleOp>(arg.getOwner()->getParentOp());
      if (mod) {
        bool first = true;
        for (auto inst : callers[mod.getSymName()]) {
          int value = constant(inst->getOperand(arg.getArgNumber()));
          if (first) result = value;
          else if (value != result) result = -1;
          first = false;
          if (result < 0) break;
        }
      }
    } else if (auto inst = v.getDefiningOp<hw::InstanceOp>()) {
      unsigned index = cast<OpResult>(v).getResultNumber();
      if (inst.getModuleName() == "SimJTAG") {
        result = cast<StringAttr>(inst.getResultNames()[index]).getValue() == "jtag_TRSTn" ? 1 : 0;
      } else if (auto mod = modules.lookup(inst.getModuleName())) {
        result = constant(mod.getBodyBlock()->getTerminator()->getOperand(index));
      }
    } else if (auto op = v.getDefiningOp()) {
      if (isa<seq::ToClockOp, seq::FromClockOp>(op))
        result = constant(op->getOperand(0));
      else if (auto mux = dyn_cast<comb::MuxOp>(op)) {
        int cond = constant(mux.getCond());
        if (cond >= 0) result = constant(op->getOperand(cond ? 1 : 2));
      } else if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op)) {
        bool isAnd = isa<comb::AndOp>(op), isOr = isa<comb::OrOp>(op);
        result = isAnd ? 1 : 0;
        for (Value operand : op->getOperands()) {
          int value = constant(operand);
          if ((isAnd && value == 0) || (isOr && value == 1)) {
            result = value;
            break;
          }
          if (value < 0) result = -1;
          else if (!isAnd && !isOr && result >= 0) result ^= value;
        }
      }
    }
    constants[v] = result;
    return result;
  }

  LogicalResult audit(ModuleOp design) {
    design.walk([&](hw::InstanceOp inst) { callers[inst.getModuleName()].push_back(inst); });
    for (auto mod : design.getOps<hw::HWModuleOp>()) modules[mod.getSymName()] = mod;
    unsigned roots = 0, rootClocks = 0;
    for (auto mod : design.getOps<hw::HWModuleOp>()) {
      if (!callers[mod.getSymName()].empty()) continue;
      ++roots;
      for (auto port : mod.getPortList()) {
        if (!port.isInput() || !isa<seq::ClockType>(port.type)) continue;
        rootClock = mod.getBodyBlock()->getArgument(port.argNum);
        ++rootClocks;
      }
    }
    if (roots != 1 || rootClocks != 1)
      return design.emitError("Single-clock model requires one root module with one clock input");
    for (auto ext : design.getOps<hw::HWModuleExternOp>()) {
      unsigned outputs = 0;
      for (auto port : ext.getPortList()) {
        if (port.isOutput()) ++outputs;
        if (ext.getSymName() == "SimJTAG" && port.isOutput()) {
          bool clock = port.getName() == "jtag_TCK";
          bool bit = port.getName() == "jtag_TRSTn" || port.getName() == "jtag_TMS" ||
                     port.getName() == "jtag_TDI";
          bool exit = port.getName() == "exit";
          if ((!clock && !bit && !exit) || (clock && !isa<seq::ClockType>(port.type)) ||
              (!clock && port.type != IntegerType::get(&getContext(), exit ? 32 : 1)))
            return ext.emitError("Unexpected SimJTAG output interface");
        }
        if (port.isOutput() && isa<seq::ClockType>(port.type) &&
            ext.getSymName() != "SimJTAG" && ext.getSymName() != "ClockGate")
          return ext.emitError("Unsupported external clock source in single-clock model");
      }
      if (ext.getSymName() == "SimJTAG" && outputs != 5)
        return ext.emitError("Unexpected SimJTAG output count");
      if (ext.getSymName() == "ClockGate" && (ext.getPortList().size() != 4 || outputs != 1))
        return ext.emitError("Unexpected ClockGate interface");
    }
    for (auto mod : design.getOps<hw::HWModuleOp>()) {
      for (auto port : mod.getPortList()) {
        if (!port.isInput()) continue;
        bool dft = mod.getSymName().starts_with("MbistClockGateCell") &&
                   (port.getName() == "dft_ram_aux_clk" || port.getName() == "dft_ram_aux_ckbp");
        bool jtag = mod.getSymName() == "JtagTapController" && port.getName() == "clock";
        if ((dft || jtag) && constant(mod.getBodyBlock()->getArgument(port.argNum)) != 0)
          return mod.emitError("Single-clock model requires proven idle JTAG/DFT clocks");
      }
    }
    design.walk([&](seq::ToClockOp op) {
      if (isClockCarrier(op.getInput())) aliasCasts.insert(op.getOperation());
    });
    llvm::outs() << "Single-clock audit: one root clock; JTAG/DFT tie-offs proven\n";
    return success();
  }

  static Value bit(OpBuilder &b, Location loc, bool value) {
    return b.create<hw::ConstantOp>(loc, b.getI1Type(), value);
  }
  static Value invert(OpBuilder &b, Location loc, Value value) {
    return b.create<comb::XorOp>(loc, value, bit(b, loc, true));
  }

  // Match DISABLE_SIMJTAG_DPIC in the checked-in XiangShan SimJTAG.v.
  LogicalResult replaceClockDevices(ModuleOp design) {
    for (auto ext : llvm::make_early_inc_range(design.getOps<hw::HWModuleExternOp>())) {
      auto name = ext.getSymName();
      if (name != "ClockGate" && name != "SimJTAG") continue;
      OpBuilder b(ext);
      auto mod = b.create<hw::HWModuleOp>(ext.getLoc(), ext.getSymNameAttr(),
                                         ext.getPortList(), ext.getParameters());
      if (auto visibility = ext->getAttr("sym_visibility"))
        mod->setAttr("sym_visibility", visibility);
      b.setInsertionPointToStart(mod.getBodyBlock());
      auto input = [&](StringRef wanted) -> Value {
        for (auto port : mod.getPortList())
          if (port.isInput() && port.getName() == wanted)
            return mod.getBodyBlock()->getArgument(port.argNum);
        return {};
      };
      SmallVector<Value> outputs;
      if (name == "ClockGate") {
        if (!input("TE") || !input("E") || !input("CK"))
          return mod.emitError("Unexpected ClockGate interface");
        // Controls are sampled at the root cycle's stable low-phase boundary.
        auto en = b.create<comb::OrOp>(mod.getLoc(), input("TE"), input("E"));
        outputs.push_back(b.create<comb::AndOp>(mod.getLoc(), input("CK"), en));
        ++gates;
      } else {
        for (auto port : mod.getPortList()) {
          if (!port.isOutput()) continue;
          auto width = cast<IntegerType>(port.type).getWidth();
          outputs.push_back(b.create<hw::ConstantOp>(mod.getLoc(),
              b.getIntegerType(width), port.getName() == "jtag_TRSTn" ? 1 : 0));
        }
      }
      mod.getBodyBlock()->getTerminator()->setOperands(outputs);
      ext.erase();
    }
    return success();
  }

  LogicalResult runModule(hw::HWModuleOp mod) {
    OpBuilder b(mod.getContext());
    auto loc = mod.getLoc();
    b.setInsertionPointToStart(mod.getBodyBlock());
    Value one = bit(b, loc, true);
    Value zero = bit(b, loc, false);
    Value dummy = b.create<seq::ToClockOp>(loc, one);
    // The top clock is implicit: exactly one root rising edge per evaluation.
    if (rootClock.getParentBlock() == mod.getBodyBlock())
      rootClock.replaceAllUsesWith(one);

    // Memory delays must be expanded before replacing their clock operands.
    for (auto read : llvm::make_early_inc_range(mod.getOps<seq::FirMemReadOp>())) {
      auto mem = cast<seq::FirMemOp>(read.getMemory().getDefiningOp());
      if (mem.getReadLatency() > 1)
        return read.emitError("Single-clock model supports read latency 0 or 1");
      if (mem.getReadLatency() == 0) continue;
      b.setInsertionPointAfter(read);
      auto name = b.getStringAttr(mem.getName().value_or("memory").str() +
                                  "_single_clock_read_" + std::to_string(reads));
      auto reg = b.create<seq::FirRegOp>(read.getLoc(), read.getData(), read->getOperand(2), name);
      Value en = read.getEnable() ? read.getEnable() : one;
      auto mux = b.create<comb::MuxOp>(read.getLoc(), en, read.getData(), reg.getData());
      read.getData().replaceAllUsesExcept(reg.getData(), mux);
      reg.getNextMutable().assign(mux);
      read.getEnableMutable().assign(one);
      ++reads;
    }
    for (auto mem : mod.getOps<seq::FirMemOp>()) {
      if (mem.getWriteLatency() != 1)
        return mem.emitError("Single-clock model requires write latency 1");
      mem.setReadLatency(0);
    }

    SmallVector<Operation *> oldOps;
    for (auto &op : mod.getOps()) if (&op != dummy.getDefiningOp()) oldOps.push_back(&op);
    for (auto *op : oldOps) {
      b.setInsertionPoint(op);
      if (auto castOp = dyn_cast<seq::ToClockOp>(op)) {
        Value pulse;
        if (aliasCasts.contains(op)) {
          // FromClock and its forwarding ports already carry the source pulse.
          pulse = castOp.getInput();
          ++aliases;
        } else if (mod.getSymName() == "JtagTapController") {
          // Only the idle SimJTAG model is supported; no falling TCK edges.
          if (constant(castOp.getInput()) < 0)
            return castOp.emitError("JTAG clock inversion must be statically idle");
          pulse = zero;
        } else if (auto reg = castOp.getInput().getDefiningOp<seq::FirRegOp>()) {
          // Predict the register's rising transition from the same snapshot.
          Value enable = reg->getOperand(1);
          Value next = b.create<comb::MuxOp>(loc, enable, reg.getNext(), reg.getData());
          if (reg.hasReset()) {
            Value reset = reg.getReset();
            if (!reg.getIsAsync()) reset = b.create<comb::AndOp>(loc, enable, reset);
            next = b.create<comb::MuxOp>(loc, reset, reg.getResetValue(), next);
          }
          pulse = b.create<comb::AndOp>(loc, invert(b, loc, reg.getData()), next);
        } else if (mod.getSymName().starts_with("MbistClockGateCell")) {
          // DFT auxiliary clocks are disabled in this workload configuration.
          if (constant(castOp.getInput()) != 0)
            return castOp.emitError("DFT clock conversion must be tied low");
          pulse = zero;
        } else {
          return castOp.emitError("Unrecognized clock conversion in fixed single-clock model");
        }
        castOp->getResult(0).replaceAllUsesWith(pulse);
        castOp.erase();
        ++casts;
      } else if (auto mux = dyn_cast<seq::ClockMuxOp>(op)) {
        Value en = b.create<comb::MuxOp>(loc, mux.getCond(), mux->getOperand(1), mux->getOperand(2));
        mux->getResult(0).replaceAllUsesWith(en);
        mux.erase();
      } else if (auto from = dyn_cast<seq::FromClockOp>(op)) {
        from.getOutput().replaceAllUsesWith(from->getOperand(0));
        from.erase();
      } else if (auto c = dyn_cast<seq::ConstClockOp>(op)) {
        c->getResult(0).replaceAllUsesWith(zero);
        c.erase();
      } else if (isa<seq::ClockGateOp, seq::ClockDividerOp>(op)) {
        return op->emitError("Unrecognized clock operation in fixed single-clock model");
      }
    }
    for (auto reg : llvm::make_early_inc_range(mod.getOps<seq::FirRegOp>())) {
      b.setInsertionPoint(reg);
      Value en = reg->getOperand(1);
      auto next = b.create<comb::MuxOp>(loc, en, reg.getNext(), reg.getData());
      reg.getNextMutable().assign(next);
      if (reg.hasReset() && !reg.getIsAsync())
        reg.getResetMutable().assign(b.create<comb::AndOp>(loc, en, reg.getReset()));
      reg.getClkMutable().assign(dummy);
      ++registers;
    }
    for (auto write : mod.getOps<seq::FirMemWriteOp>()) {
      b.setInsertionPoint(write);
      Value en = write.getEnable() ? write.getEnable() : one;
      write.getEnableMutable().assign(b.create<comb::AndOp>(loc, write->getOperand(2), en));
      write.getClkMutable().assign(dummy);
      ++writes;
    }
    for (auto read : mod.getOps<seq::FirMemReadOp>()) read.getClkMutable().assign(dummy);
    return success();
  }

  void runOnOperation() override {
    auto design = getOperation();
    if (failed(audit(design))) return signalPassFailure();
    auto i1 = IntegerType::get(&getContext(), 1);
    // Preserve the hierarchy while turning clock ports/results into enables.
    for (auto &op : design.getOps()) {
      if (auto iface = dyn_cast<hw::HWModuleLike>(&op)) {
        auto type = iface.getHWModuleType();
        SmallVector<hw::ModulePort> ports(type.getPorts());
        for (auto &p : ports) if (isa<seq::ClockType>(p.type)) p.type = i1;
        op.setAttr("module_type", TypeAttr::get(hw::ModuleType::get(&getContext(), ports)));
      }
      if (auto mod = dyn_cast<hw::HWModuleOp>(&op)) {
        for (auto arg : mod.getBodyBlock()->getArguments())
          if (isa<seq::ClockType>(arg.getType())) arg.setType(i1);
        mod.walk([&](Operation *inner) {
          for (auto v : inner->getResults())
            if (isa<seq::ClockType>(v.getType())) v.setType(i1);
        });
      }
    }
    if (failed(replaceClockDevices(design))) return signalPassFailure();
    for (auto mod : design.getOps<hw::HWModuleOp>())
      if (failed(runModule(mod))) return signalPassFailure();
    llvm::outs() << "Single-clock model: " << registers << " registers, " << writes
                 << " memory writers, " << reads << " enabled read pipelines, "
                 << gates << " gate definitions, " << casts << " clock casts ("
                 << aliases << " proven Bool aliases)\n";
  }
};

struct RemoveDeadClocks : PassWrapper<RemoveDeadClocks, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveDeadClocks)
  void runOnOperation() override {
    getOperation().walk([&](seq::ToClockOp op) {
      if (op->use_empty()) op.erase();
    });
  }
};
} // namespace single_clock
