// Execute the real lowering passes and generated LUT tables against APInt shifts.
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "toucan/CodeGenCommon.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanPasses.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <cstdlib>
#include <random>
#include <set>
#include <string>

using namespace mlir;
using namespace circt;
using llvm::APInt;

static void require(bool condition) {
  if (!condition) std::abort();
}

// This interpreter knows only primitive IR operations and the production LUT
// table. It never reconstructs a source shift or the amount-bounding algorithm.
static SmallVector<APInt> evaluate(hw::HWModuleOp module, const APInt &x,
                                 const APInt &s, const toucan::CodeGenHelper &lut) {
  llvm::DenseMap<Value, APInt> values;
  llvm::DenseMap<Value, SmallVector<APInt>> vectors;
  auto &body = module.getBody().front();
  values[body.getArgument(0)] = x;
  values[body.getArgument(1)] = s;
  auto get = [&](Value value) -> const APInt & {
    require(values.contains(value));
    return values.at(value);
  };
  for (auto &raw : body) {
    if (auto op = dyn_cast<hw::OutputOp>(raw)) {
      SmallVector<APInt> outputs;
      for (auto value : op.getOperands()) outputs.push_back(get(value));
      return outputs;
    }
    if (auto op = dyn_cast<toucan::DefVectorOp>(raw)) {
      auto &vector = vectors[op.getHandle()];
      // Runtime index zero denotes the least significant vector element.
      for (auto value : llvm::reverse(op.getInputs())) vector.push_back(get(value));
      require(vector.size() <= UINT16_MAX);
      continue;
    }
    APInt result(1, 0);
    if (auto op = dyn_cast<hw::ConstantOp>(raw)) {
      result = op.getValue();
    } else if (auto op = dyn_cast<toucan::ConstantOp>(raw)) {
      result = op.getValue();
    } else if (auto op = dyn_cast<comb::ExtractOp>(raw)) {
      result = get(op.getInput()).extractBits(hw::getBitWidth(op.getType()), op.getLowBit());
    } else if (auto op = dyn_cast<comb::ConcatOp>(raw)) {
      bool first = true;
      for (auto value : op.getInputs()) {
        result = first ? get(value) : result.concat(get(value));
        first = false;
      }
    } else if (auto op = dyn_cast<toucan::LUTOp>(raw)) {
      unsigned index = 0;
      for (auto value : op.getInputs()) {
        require(get(value).getActiveBits() <= 4);
        index = (index << 4) | get(value).getZExtValue();
      }
      auto name = toucan::LUTOp::getMappedOpName(op.getOpName());
      auto position = lut.lutPos.at(static_cast<unsigned>(name)) + index;
      result = APInt(hw::getBitWidth(op.getResult().getType()), lut.lutContent.at(position), false, true);
    } else if (auto op = dyn_cast<toucan::VectorReadOp>(raw)) {
      require(op.getIndicies().size() <= 4);
      unsigned index = 0;
      for (auto value : op.getIndicies()) {
        require(hw::getBitWidth(value.getType()) <= 4);
        index = (index << 4) | get(value).getZExtValue();
      }
      index += op.getOffset().getZExtValue();
      const auto &vector = vectors.at(op.getHandle());
      result = index < vector.size() ? vector[index] : get(op.getOutRangeValue());
    } else if (auto op = dyn_cast<toucan::VectorLogicOp>(raw)) {
      require(op.getOpName() == toucan::VecLogicOpName::VecLogic_Eq);
      result = APInt(1, vectors.at(op.getV1()) == vectors.at(op.getV2()));
    } else {
      raw.emitError("Unexpected operation in lowered shift regression");
      std::abort();
    }
    require(raw.getNumResults() == 1);
    require(result.getBitWidth() == hw::getBitWidth(raw.getResult(0).getType()));
    values[raw.getResult(0)] = std::move(result);
  }
  std::abort();
}

static std::string design(unsigned width, unsigned amountWidth) {
  auto w = std::to_string(width), a = std::to_string(amountWidth);
  std::string text = "module { hw.module @Shift(in %x: i" + w + ", in %s: i" + a;
  text += ", out left: i" + w + ", out right: i" + w + ", out signed_right: i" + w + ") {\n";
  std::string amount = "%s";
  if (amountWidth < width) {
    auto pad = std::to_string(width - amountWidth);
    text += "%zero = hw.constant 0 : i" + pad + "\n";
    text += "%amount = comb.concat %zero, %s : i" + pad + ", i" + a + "\n";
    amount = "%amount";
  }
  for (auto op : {"shl", "shru", "shrs"})
    text += "%" + std::string(op) + " = comb." + op + " bin %x, " + amount + " : i" + w + "\n";
  return text + "hw.output %shl, %shru, %shrs : i" + w + ", i" + w + ", i" + w + "\n} }";
}

int main() {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect, toucan::ToucanDialect>();
  toucan::CodeGenHelper lut;
  lut.populateLUT();
  std::mt19937_64 random(0x7368696674);
  std::set<std::pair<unsigned, unsigned>> cases;
  for (unsigned width : {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 29, 30, 31,
                         32, 33, 59, 60, 63, 64, 65, 127, 128, 129, 130, 137,
                         255, 256, 257, 512}) cases.emplace(width, width);
  for (unsigned amount : {1, 2, 6, 18, 19, 30, 31, 32, 33}) cases.emplace(60, amount);
  for (unsigned amount : {31, 32, 33}) cases.emplace(64, amount);
  cases.emplace(65, 64);
  cases.emplace(256, 129);
  uint64_t comparisons = 0, exhaustivePairs = 0;
  for (auto [width, amountWidth] : cases) {
    auto mod = parseSourceString<ModuleOp>(design(width, amountWidth), &context);
    require(bool(mod));
    PassManager pm(&context);
    pm.addPass(toucan::createLowerCombTo4B_1Pass());
    pm.addPass(toucan::createLowerCombTo4B_2Pass());
    pm.addPass(toucan::createLowerCombTo4B_3Pass());
    pm.addPass(toucan::createFactorConcatExtractPass());
    require(succeeded(pm.run(*mod)));
    auto module = *mod->getOps<hw::HWModuleOp>().begin();
    auto check = [&](const APInt &x, const APInt &s) {
      auto actual = evaluate(module, x, s, lut);
      // Independent full-width unsigned amount and signed data semantics.
      unsigned shift = s.getLimitedValue(width);
      SmallVector<APInt> expected = {
          shift == width ? APInt(width, 0) : x.shl(shift),
          shift == width ? APInt(width, 0) : x.lshr(shift),
          x.ashr(std::min(shift, width - 1))};
      require(actual.size() == expected.size());
      for (unsigned op = 0; op < actual.size(); ++op) {
        if (actual[op] != expected[op]) {
          llvm::errs() << "FAIL width=" << width << " amount_width=" << amountWidth
                       << " operation=" << op << " x=" << x << " s=" << s
                       << " actual=" << actual[op] << " expected=" << expected[op] << "\n";
          std::abort();
        }
        ++comparisons;
      }
    };
    if (width <= 8) {
      for (unsigned x = 0; x < (1u << width); ++x)
        for (unsigned s = 0; s < (1u << amountWidth); ++s) {
          check(APInt(width, x), APInt(amountWidth, s));
          ++exhaustivePairs;
        }
    } else {
      SmallVector<APInt> data = {APInt(width, 0), APInt(width, 1),
          APInt::getAllOnes(width), APInt::getSignedMinValue(width),
          APInt::getSignedMaxValue(width), APInt::getSignedMinValue(width) | 1};
      for (unsigned n = 0; n < 12; ++n) {
        SmallVector<uint64_t> words((width + 63) / 64);
        for (auto &word : words) word = random();
        data.emplace_back(width, words);
      }
      SmallVector<APInt> amounts;
      for (unsigned s = 0; s <= width + 1; ++s) amounts.emplace_back(amountWidth, s, false, true);
      amounts.push_back(APInt::getAllOnes(amountWidth));
      for (unsigned bit = 0; bit < amountWidth; ++bit) {
        auto power = APInt::getOneBitSet(amountWidth, bit);
        amounts.push_back(power);
        amounts.push_back(power | APInt(amountWidth, 1));
      }
      for (const auto &x : data)
        for (const auto &s : amounts) check(x, s);
    }
  }
  llvm::outs() << "{\"cases\":" << cases.size() << ",\"comparisons\":" << comparisons
               << ",\"exhaustive_input_pairs\":" << exhaustivePairs
               << ",\"maximum_data_and_amount_width\":512,\"pass\":true}\n";
}
