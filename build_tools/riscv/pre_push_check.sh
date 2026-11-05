#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pre-push hook (see .pre-commit-config.yaml): runs the RISC-V 64 + RVV ONNX
# tests on QEMU before code leaves the machine.
#
# The hook is skipped (exit 0) with a message when the local setup is
# incomplete, so teammates without a RISC-V toolchain are not blocked.
# Set RISCV_PREPUSH_STRICT=1 to make a missing tool a failure instead, and
# RISCV_PREPUSH_SKIP=1 (or `git push --no-verify`) to skip the check entirely.
#
# Tools are looked up on PATH, plus "<repo>/.venv/bin" (see setup_dev.sh).
# The cross-compiled iree-run-module is expected at
# "${IREE_TARGET_BUILD_DIR:-build-riscv}/tools/iree-run-module".

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if [[ "${RISCV_PREPUSH_SKIP:-0}" == "1" ]]; then
  echo "riscv64 pre-push: skipped (RISCV_PREPUSH_SKIP=1)"
  exit 0
fi

if [[ -d "${REPO_ROOT}/.venv/bin" ]]; then
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

BUILD_DIR="${IREE_TARGET_BUILD_DIR:-build-riscv}"
QEMU_BIN="${QEMU_BIN:-qemu-riscv64}"
RUN_MODULE="${IREE_RUN_MODULE:-${BUILD_DIR}/tools/iree-run-module}"

missing=()
command -v iree-compile >/dev/null || missing+=("iree-compile (pip install iree-base-compiler)")
command -v iree-import-onnx >/dev/null || missing+=("iree-import-onnx (pip install iree-base-compiler)")
command -v "${QEMU_BIN}" >/dev/null || missing+=("${QEMU_BIN} (set QEMU_BIN)")
[[ -x "${RUN_MODULE}" ]] || missing+=("${RUN_MODULE} (run build_tools/riscv/build_run_module.sh)")
python3 -c "import onnxruntime, pytest" 2>/dev/null \
  || missing+=("python packages (pip install -r tests/riscv64_onnx/requirements.txt)")

if (( ${#missing[@]} > 0 )); then
  echo "riscv64 pre-push: missing tools, so the ONNX/QEMU check was NOT run:"
  printf '  - %s\n' "${missing[@]}"
  echo "See build_tools/riscv/README.md. CI will still run it."
  if [[ "${RISCV_PREPUSH_STRICT:-0}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

# Fail fast; -q keeps the output short on success.
exec ./build_tools/riscv/run_onnx_tests.sh "${BUILD_DIR}" -x -q
