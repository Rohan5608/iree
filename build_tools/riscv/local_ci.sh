#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs the same steps as .github/workflows/riscv64_ci.yml on your machine.
#
# Usage: build_tools/riscv/local_ci.sh <step>...
#   lint   pre-commit on the files changed relative to origin/main
#          (all files if origin/main is unknown).
#   build  Cross-build the riscv64 iree-run-module.
#   onnx   Run every model in tests/riscv64_onnx/models on QEMU vs ONNX Runtime
#          (implies build if the binary is missing).
#   e2e    Upstream's riscv64 e2e suite: full cross-build, then CTest on QEMU
#          (build_tools/cmake/build_riscv.sh + test_riscv.sh). Slow.
#   all    lint, onnx, e2e.
#
# Environment (same as CI):
#   IREE_HOST_BIN_DIR     Directory with iree-compile & friends (default
#                         ./.venv/bin if present).
#   IREE_HOST_COMPILER_DIR  Compiler built from this checkout by
#                         build_host_compiler.sh (<build>/host-bin); takes
#                         precedence over the pip compiler.
#   RISCV_TOOLCHAIN_ROOT  RISC-V clang toolchain with sysroot.
#   QEMU_BIN              qemu-riscv64.
# Extra pytest args for `onnx` can be given via ONNX_PYTEST_ARGS, e.g.
#   ONNX_PYTEST_ARGS="-k mlp" build_tools/riscv/local_ci.sh onnx

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if [[ -d .venv/bin ]]; then
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
  export IREE_HOST_BIN_DIR="${IREE_HOST_BIN_DIR:-${REPO_ROOT}/.venv/bin}"
fi
# Use a compiler built from this checkout (build_host_compiler.sh) instead of
# the pip one, e.g. IREE_HOST_COMPILER_DIR=$PWD/build-host/host-bin.
if [[ -n "${IREE_HOST_COMPILER_DIR:-}" ]]; then
  export PATH="${IREE_HOST_COMPILER_DIR}:${PATH}"
  export IREE_HOST_BIN_DIR="${IREE_HOST_COMPILER_DIR}"
fi
BUILD_DIR="${IREE_TARGET_BUILD_DIR:-build-riscv}"

step_lint() {
  echo "=== lint ==="
  if git rev-parse --verify -q origin/main >/dev/null; then
    base="$(git merge-base origin/main HEAD)"
    pre-commit run --from-ref "${base}" --to-ref HEAD
  else
    pre-commit run --all-files
  fi
}

step_build() {
  echo "=== build riscv64 iree-run-module ==="
  ./build_tools/riscv/build_run_module.sh "${BUILD_DIR}"
}

step_onnx() {
  echo "=== onnx (riscv64 + RVV on QEMU vs ONNX Runtime) ==="
  [[ -x "${BUILD_DIR}/tools/iree-run-module" ]] || step_build
  # shellcheck disable=SC2086
  ./build_tools/riscv/run_onnx_tests.sh "${BUILD_DIR}" ${ONNX_PYTEST_ARGS:-}
}

step_e2e() {
  echo "=== e2e (CTest on QEMU) ==="
  : "${QEMU_BIN:?set QEMU_BIN to qemu-riscv64}"
  : "${RISCV_TOOLCHAIN_ROOT:?set RISCV_TOOLCHAIN_ROOT}"
  ./build_tools/cmake/build_riscv.sh "${BUILD_DIR}"
  ./build_tools/cmake/test_riscv.sh "${BUILD_DIR}"
}

if (( $# == 0 )); then
  sed -n '/^# Usage/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

for step in "$@"; do
  case "${step}" in
    lint)  step_lint ;;
    build) step_build ;;
    onnx)  step_onnx ;;
    e2e)   step_e2e ;;
    all)   step_lint; step_onnx; step_e2e ;;
    *) echo "Unknown step '${step}'" >&2; exit 2 ;;
  esac
done
