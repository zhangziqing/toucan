// Execute real array lowering and production LUTs against a numeric-index oracle.
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
// table. It never reconstructs a source array or its index selection.
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
      raw.emitError("Unexpected operation in lowered array regression");
      std::abort();
    }
    require(raw.getNumResults() == 1);
    require(result.getBitWidth() == hw::getBitWidth(raw.getResult(0).getType()));
    values[raw.getResult(0)] = std::move(result);
  }
  std::abort();
}


static std::string arrayDesign(unsigned length, unsigned width,
                               unsigned amountWidth, bool constant) {
  unsigned indexWidth = llvm::Log2_64_Ceil(length);
  auto w = std::to_string(width), n = std::to_string(length);
  std::string text = "module { hw.module @Array(in %x: i";
  text += std::to_string(length * width) + ", in %s: i";
  text += std::to_string(amountWidth) + ", out value: i" + w + ") {\n";
  std::string index = "%s";
  if (amountWidth < indexWidth) {
    auto pad = std::to_string(indexWidth - amountWidth);
    text += "%zero = hw.constant 0 : i" + pad + "\n";
    text += "%index = comb.concat %zero, %s : i" + pad + ", i";
    text += std::to_string(amountWidth) + "\n";
    index = "%index";
  }
  std::string elements;
  for (int j = length - 1; j >= 0; --j) {
    if (!elements.empty()) elements += ", ";
    if (constant) {
      elements += std::to_string((j * 37 + 3) & ((1u << std::min(width, 16u)) - 1));
      elements += " : i" + w;
    } else {
      text += "%e" + std::to_string(j) + " = comb.extract %x from ";
      text += std::to_string(j * width) + " : (i";
      text += std::to_string(length * width) + ") -> i" + w + "\n";
      elements += "%e" + std::to_string(j);
    }
  }
  text += "%array = hw.";
  text += constant ? "aggregate_constant [" + elements + "] : !hw.array<" + n + "xi" + w + ">\n"
                   : "array_create " + elements + " : i" + w + "\n";
  text += "%value = hw.array_get %array[" + index + "] : !hw.array<";
  text += n + "xi" + w + ">, i" + std::to_string(indexWidth) + "\n";
  return text + "hw.output %value : i" + w + "\n} }";
}

int main(int argc, char **argv) {
  bool narrowOnly = argc == 2 && std::string(argv[1]) == "--narrow-only";
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                      toucan::ToucanDialect>();
  toucan::CodeGenHelper lut;
  lut.populateLUT();
  std::mt19937_64 random(0x6172726179);
  uint64_t cases = 0, comparisons = 0, outOfRange = 0;
  for (unsigned length : {8, 1, 2, 3, 4, 5, 6, 7}) {
    unsigned indexWidth = llvm::Log2_64_Ceil(length);
    for (unsigned amountWidth = std::min(1u, indexWidth);
         amountWidth <= indexWidth; ++amountWidth) {
      for (unsigned width : {8, 1, 4, 5, 17, 32, 65}) {
        for (bool constant : {false, true}) {
          if (narrowOnly && (length != 8 || amountWidth != 1 || width != 8 || constant))
            continue;
          auto mod = parseSourceString<ModuleOp>(
              arrayDesign(length, width, amountWidth, constant), &context);
          require(bool(mod));
          PassManager pm(&context);
          pm.addPass(toucan::createLowerHWVectorTo4BPass());
          pm.addPass(toucan::createFactorBinaryOpPass());
          pm.addPass(toucan::createLowerCombTo4B_1Pass());
          pm.addPass(toucan::createLowerCombTo4B_2Pass());
          pm.addPass(toucan::createLowerCombTo4B_3Pass());
          pm.addPass(toucan::createFactorConcatExtractPass());
          require(succeeded(pm.run(*mod)));
          auto module = *mod->getOps<hw::HWModuleOp>().begin();
          ++cases;
          for (unsigned sample = 0; sample < 24; ++sample) {
            SmallVector<uint64_t> words((length * width + 63) / 64);
            for (auto &word : words) word = random();
            APInt data(length * width, words);
            if (!sample) {
              data = APInt(length * width, 0);
              for (unsigned j = 0; j < length; ++j)
                data.insertBits(APInt(width, j + 3, false, true), j * width);
            }
            for (unsigned index = 0; index < (1u << amountWidth); ++index) {
              auto actual = evaluate(module, data, APInt(amountWidth, index), lut);
              // Independently select the packed element at the numeric index.
              APInt expected(width, 0);
              if (index < length)
                expected = constant ? APInt(width, (index * 37 + 3) &
                    ((1u << std::min(width, 16u)) - 1))
                                    : data.extractBits(width, index * width);
              else
                ++outOfRange; // Backend zero default, not an X/Z equivalence claim.
              require(actual.size() == 1);
              if (actual[0] != expected) {
                llvm::errs() << "FAIL length=" << length << " width=" << width
                             << " amount_width=" << amountWidth << " constant="
                             << constant << " index=" << index << " actual="
                             << actual[0] << " expected=" << expected << "\n";
                return 2;
              }
              ++comparisons;
            }
          }
        }
      }
    }
  }
  llvm::outs() << "{\"cases\":" << cases << ",\"comparisons\":" << comparisons
               << ",\"out_of_range_zero_checks\":" << outOfRange
               << ",\"pass\":true}\n";
}
