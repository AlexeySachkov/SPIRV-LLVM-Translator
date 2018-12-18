// RUN: clang -cc1 -x cl -emit-llvm-bc -triple spir64 -cl-std=CL2.0 %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck  %s

// CHECK: Capability Addresses
// CHECK: "foo"

__kernel void foo(__global int *a) {
}
