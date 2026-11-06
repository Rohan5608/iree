// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A tensor workload compiled with RVV enabled; validates all 16 output values.
func.func @abs_plus_one(%input: tensor<16xf32>) -> tensor<16xf32> {
  %one = arith.constant dense<1.0> : tensor<16xf32>
  %abs = math.absf %input : tensor<16xf32>
  %result = arith.addf %abs, %one : tensor<16xf32>
  return %result : tensor<16xf32>
}
