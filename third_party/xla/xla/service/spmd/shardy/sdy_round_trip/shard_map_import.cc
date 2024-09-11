/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_import.h"

#include <memory>

#include "absl/log/check.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using ::mlir::func::FuncOp;

namespace sdy = ::mlir::sdy;

void createManualComputation(FuncOp func,
                             SmallVector<FuncOp>& inlinedShmapBodyFuncs) {
  func.walk([&](mlir::stablehlo::CustomCallOp customCallOp) {
    if (customCallOp.getCallTargetName() !=
        kManualComputationCustomCallTargetName) {
      return;
    }
    mlir::SymbolTableCollection symbolTableCollection;
    mlir::SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(
        customCallOp->getParentOfType<ModuleOp>());

    CHECK_EQ(customCallOp.getCalledComputations().size(), 1);
    auto shmapBodyFunc = symbolTable.lookup<FuncOp>(
        mlir::cast<mlir::FlatSymbolRefAttr>(
            customCallOp.getCalledComputations().getValue().front())
            .getValue());
    CHECK(shmapBodyFunc) << "Expected a unique FuncOp per @ManualComputation "
                            "custom call. Were functions maybe somehow "
                            "shared/de-duped between two ManualComputations?";

    // Before creating the current `ManualComputationOp`, import any nested
    // `ManualComputationOp` inside the body.
    createManualComputation(shmapBodyFunc, inlinedShmapBodyFuncs);

    mlir::DictionaryAttr frontendAttrs = getFrontendAttrs(customCallOp);
    CHECK(frontendAttrs);
    auto rewriter = mlir::IRRewriter(func);
    rewriter.setInsertionPoint(customCallOp);
    auto manualComputationOp =
        rewriter.replaceOpWithNewOp<sdy::ManualComputationOp>(
            customCallOp, customCallOp->getResultTypes(),
            customCallOp->getOperands(),
            parseStringAttr<sdy::TensorShardingPerValueAttr>(
                frontendAttrs.get(kInShardings)),
            parseStringAttr<sdy::TensorShardingPerValueAttr>(
                frontendAttrs.get(kOutShardings)),
            parseStringAttr<sdy::ManualAxesAttr>(
                frontendAttrs.get(kManualAxes)));
    sdy::inlineRegionAndConvertTerminatorOp<sdy::ReturnOp>(
        shmapBodyFunc.getBody(), manualComputationOp.getRegion());

    inlinedShmapBodyFuncs.push_back(shmapBodyFunc);
  });
}

class SdyRoundTripShardMapImportPass
    : public mlir::PassWrapper<SdyRoundTripShardMapImportPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripShardMapImportPass)

 private:
  void runOnOperation() final {
    // Store the inlined functions until the walk is complete, as the walk will
    // attempt to enter them when iterating all ops.
    SmallVector<FuncOp> inlinedShmapBodyFuncs;
    ModuleOp moduleOp = getOperation();
    moduleOp->walk([&](FuncOp funcOp) {
      createManualComputation(funcOp, inlinedShmapBodyFuncs);
    });
    mlir::SymbolTableCollection symbolTableCollection;
    mlir::SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(moduleOp);
    for (FuncOp inlinedFunc : inlinedShmapBodyFuncs) {
      symbolTable.erase(inlinedFunc);
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-shard-map-import";
  }

  StringRef getDescription() const override {
    return "converts CustomCalls called kManualComputationCustomCallTargetName "
           "with in/out shardings and manual axes as frontend attrs to a "
           "`ManualComputationOp`";
  }
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<sdy::SdyDialect>();
  }
};

}  // namespace

void registerSdyRoundTripShardMapImportPass() {
  mlir::registerPass(createSdyRoundTripShardMapImportPass);
}

std::unique_ptr<mlir::Pass> createSdyRoundTripShardMapImportPass() {
  return std::make_unique<SdyRoundTripShardMapImportPass>();
}

}  // namespace sdy
}  // namespace xla
