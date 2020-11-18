//===- SPIRVLowerConstExpr.cpp - Regularize LLVM for SPIR-V ------- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements regularization of LLVM module for SPIR-V.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spv-lower-const-expr"

#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "SPIRVMDBuilder.h"
#include "SPIRVMDWalker.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#include <list>
#include <set>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

cl::opt<bool> SPIRVLowerConst(
    "spirv-lower-const-expr", cl::init(true),
    cl::desc("LLVM/SPIR-V translation enable lowering constant expression"));

class SPIRVLowerConstExpr : public ModulePass {
public:
  SPIRVLowerConstExpr() : ModulePass(ID), M(nullptr), Ctx(nullptr) {
    initializeSPIRVLowerConstExprPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;
  void visit(Module *M);

  static char ID;

private:
  Module *M;
  LLVMContext *Ctx;
};

char SPIRVLowerConstExpr::ID = 0;

bool SPIRVLowerConstExpr::runOnModule(Module &Module) {
  if (!SPIRVLowerConst)
    return false;

  M = &Module;
  Ctx = &M->getContext();

  LLVM_DEBUG(dbgs() << "Enter SPIRVLowerConstExpr:\n");
  visit(M);

  verifyRegularizationPass(*M, "SPIRVLowerConstExpr");

  return true;
}

/// Since SPIR-V cannot represent constant expression, constant expressions
/// in LLVM needs to be lowered to instructions.
///
/// For each function, the constant expressions used by instructions of the
/// function are replaced by instructions placed somewhere in the beginning of
/// the function so it dominates all other instructions.
/// According to SPIR-V 1.5 unified specification, section 2.4. Logical
/// Layout of a Module:
/// > Within a Function Definition:
/// > All OpVariable instructions in a function must be in the first block in
/// > the function. These instructions, together with any intermixed OpLine
/// > and OpNoLine instructions, must be the first instructions in that
/// > block. (Note the validation rules prevent OpPhi instructions in the
/// > first block of a function.)
/// Therefore, lowered constant expressions must not be inserted at the
/// beginning of the funcition, i.e. into the first basic block.
/// At the same time, we can't insert a new instruction right before the
/// original one, because it might be used twice and we might have one of its
/// users handled previously.
/// So, all new replacements for constant expressions are being inserted to
/// the beginning of a newly created basic block, which was split from the
/// first basic block of the function:
/// define void @function() {
/// entry:
///   ; allocas
///   ; some other instructions
///   ret void
/// }
///
/// LLVM IR above is being converted into
/// define void @function() {
/// entry:
///   ; allocas
///   br label %lowered.constexprs
/// lowered.constexprs: ; preds = %entry
///   ; replacements for constexpr instructions
///   br %the_rest
/// the_rest: ; preds: %lowered.constexprs
///   ; some other instructions
///   ret void
/// }
///
/// Each constant expression only needs to be lowered
/// once in each function and all uses of it by instructions in that function
/// is replaced by one instruction.
/// TODO: remove redundant instructions for common subexpression

void SPIRVLowerConstExpr::visit(Module *M) {
  for (auto &I : M->functions()) {
    if (I.isDeclaration())
      continue;

    std::list<Instruction *> WorkList;
    for (auto &BI : I) {
      for (auto &II : BI) {
        WorkList.push_back(&II);
      }
    }

    // We need to find a split point to insert basic block for lowered constant
    // expressions: it would be a first non-alloca instruction
    auto FirstBB = I.begin();
    Instruction *FirstNonAlloca = nullptr;
    for (Instruction &Inst : *FirstBB) {
      if (!isa<AllocaInst>(Inst)) {
        FirstNonAlloca = &Inst;
        break;
      }
    }
    assert(FirstNonAlloca &&
           "Expected to find at least one non-alloca instruction in a BB");
    // This BB is not needed in this pass and it is created to preserve existing
    // non-alloca instructions separated from the first basic block
    auto FirstBBWithoutAllocas = FirstBB->splitBasicBlock(FirstNonAlloca);
    (void)FirstBBWithoutAllocas; // Suppress unused variable warning
    // Newly created replacements for constant expressions must dominate the
    // rest of the function body
    auto InsertBB = FirstBB->splitBasicBlock(FirstBB->getTerminator(),
                                             "lowered.constexprs");
    while (!WorkList.empty()) {
      auto II = WorkList.front();

      auto LowerOp = [&II, &InsertBB, &I](Value *V) -> Value * {
        if (isa<Function>(V))
          return V;
        auto *CE = cast<ConstantExpr>(V);
        SPIRVDBG(dbgs() << "[lowerConstantExpressions] " << *CE;)
        auto ReplInst = CE->getAsInstruction();
        ReplInst->insertBefore(&*InsertBB->getFirstInsertionPt());
        SPIRVDBG(dbgs() << " -> " << *ReplInst << '\n';)
        std::vector<Instruction *> Users;
        // Do not replace use during iteration of use. Do it in another loop
        for (auto U : CE->users()) {
          SPIRVDBG(dbgs() << "[lowerConstantExpressions] Use: " << *U << '\n';)
          if (auto InstUser = dyn_cast<Instruction>(U)) {
            // Only replace users in scope of current function
            if (InstUser->getParent()->getParent() == &I)
              Users.push_back(InstUser);
          }
        }
        for (auto &User : Users)
          User->replaceUsesOfWith(CE, ReplInst);
        return ReplInst;
      };

      WorkList.pop_front();
      for (unsigned OI = 0, OE = II->getNumOperands(); OI != OE; ++OI) {
        auto Op = II->getOperand(OI);
        auto *Vec = dyn_cast<ConstantVector>(Op);
        if (Vec && std::all_of(Vec->op_begin(), Vec->op_end(), [](Value *V) {
              return isa<ConstantExpr>(V) || isa<Function>(V);
            })) {
          // Expand a vector of constexprs and construct it back with series of
          // insertelement instructions
          std::list<Value *> OpList;
          std::transform(Vec->op_begin(), Vec->op_end(),
                         std::back_inserter(OpList),
                         [LowerOp](Value *V) { return LowerOp(V); });
          Value *Repl = nullptr;
          unsigned Idx = 0;
          auto *PhiII = dyn_cast<PHINode>(II);
          auto *InsPoint = PhiII ? &PhiII->getIncomingBlock(OI)->back() : II;
          std::list<Instruction *> ReplList;
          for (auto V : OpList) {
            if (auto *Inst = dyn_cast<Instruction>(V))
              ReplList.push_back(Inst);
            Repl = InsertElementInst::Create(
                (Repl ? Repl : UndefValue::get(Vec->getType())), V,
                ConstantInt::get(Type::getInt32Ty(M->getContext()), Idx++), "",
                InsPoint);
          }
          II->replaceUsesOfWith(Op, Repl);
          WorkList.splice(WorkList.begin(), ReplList);
        } else if (auto CE = dyn_cast<ConstantExpr>(Op)) {
          WorkList.push_front(cast<Instruction>(LowerOp(CE)));
        } else if (auto MDAsVal = dyn_cast<MetadataAsValue>(Op)) {
          Metadata *MD = MDAsVal->getMetadata();
          if (auto ConstMD = dyn_cast<ConstantAsMetadata>(MD)) {
            Constant *C = ConstMD->getValue();
            if (auto CE = dyn_cast<ConstantExpr>(C)) {
              Value *RepInst = LowerOp(CE);
              Metadata *RepMD = ValueAsMetadata::get(RepInst);
              Value *RepMDVal = MetadataAsValue::get(M->getContext(), RepMD);
              II->setOperand(OI, RepMDVal);
              WorkList.push_front(cast<Instruction>(RepInst));
            }
          }
        }
      }
    }
  }
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerConstExpr, "spv-lower-const-expr",
                "Regularize LLVM for SPIR-V", false, false)

ModulePass *llvm::createSPIRVLowerConstExpr() {
  return new SPIRVLowerConstExpr();
}
