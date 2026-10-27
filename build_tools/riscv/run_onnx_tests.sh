#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs every ONNX model in tests/riscv64_onnx/models on emulated RISC-V 64
# (RVV) and compares the results against ONNX Runtime on the host.
# See tests/riscv64_onnx/README.md.
#
# Usage: run_onnx_tests.sh [build-dir] [extra pytest args...]
#
# Prerequisites (all handled by .github/workflows/pkgci_test_riscv64_onnx.yml):
#   * `iree-compile` and `iree-import-onnx` on PATH (e.g. an activated venv from
#     build_tools/pkgci/setup_venv.py).
#   * `pip install -r tests/riscv64_onnx/requirements.txt`.
#   * A riscv64 iree-run-module, built by build_tools/riscv/build_run_module.sh
#     into <build-dir>/tools/iree-run-module.
#   * qemu-riscv64 (QEMU_BIN, defaults to PATH).
#
# Environment:
#   RISCV_TOOLCHAIN_ROOT  If set, "<root>/sysroot" is used as the QEMU sysroot.
#   QEMU_LD_PREFIX        Explicit QEMU sysroot (takes precedence).
#   QEMU_CPU_FLAGS        QEMU -cpu value; default matches linux_riscv64.cmake
#                         (RVV, VLEN=512).
#   ONNX_MODELS_DIR       Models directory; default tests/riscv64_onnx/models.
#   ONNX_WORK_DIR         Keep intermediates (MLIR, vmfb, .npy) here.
#   ONNX_JUNIT_XML        JUnit report path; default <build-dir>/onnx_junit.xml.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-riscv}}"
shift || true

export IREE_RUN_MODULE="${IREE_RUN_MODULE:-${BUILD_DIR}/tools/iree-run-module}"
export QEMU_BIN="${QEMU_BIN:-qemu-riscv64}"

if [[ -z "${QEMU_LD_PREFIX:-}" && -n "${RISCV_TOOLCHAIN_ROOT:-}" \
      && -d "${RISCV_TOOLCHAIN_ROOT}/sysroot" ]]; then
  export QEMU_LD_PREFIX="${RISCV_TOOLCHAIN_ROOT}/sysroot"
fi

JUNIT_XML="${ONNX_JUNIT_XML:-${BUILD_DIR}/onnx_junit.xml}"

exec python3 -m pytest tests/riscv64_onnx \
  -rA \
  -p no:cacheprovider \
  --junitxml="${JUNIT_XML}" \
  "$@"
