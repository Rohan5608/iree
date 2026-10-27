#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Prints which tests the riscv64 CI job (pkgci_test_riscv64.yml, via
# build_tools/cmake/test_riscv.sh) runs, without building or running anything.
#
# It configures a riscv64 cross build with tests enabled (about 20s, only
# CMake runs) and asks CTest to list the tests that survive the same label and
# name exclusions as test_riscv.sh.
#
# Usage: list_e2e_tests.sh [options] [build-dir]
#   --all       Print every test name (default: summary only).
#   --rvv       Print only the RVV-specific tests (names with rvv/riscv).
#   --dir NAME  Print the tests of one tests/e2e subdirectory, e.g. matmul.
#   --runtime   Also count the runtime unit tests.
#   build-dir   Defaults to "build-riscv-list". Reused if already configured
#               with tests (a build from build_run_module.sh has none).
#
# Environment (as for build_riscv.sh):
#   RISCV_TOOLCHAIN_ROOT  RISC-V clang toolchain; if unset, CC/CXX must already
#                         target riscv64.
#   IREE_HOST_BIN_DIR     Directory with iree-flatcc-cli, iree-c-embed-data
#                         (default build/install/bin, e.g. use .venv/bin).
#
# The exclusions below must be kept in sync with test_riscv.sh. A test listed
# as "(Disabled)" is registered but never runs.

set -euo pipefail

MODE=summary
DIR_FILTER=""
WITH_RUNTIME=0
BUILD_DIR=""
while (( $# > 0 )); do
  case "$1" in
    --all) MODE=all ;;
    --rvv) MODE=rvv ;;
    --dir) MODE=dir; DIR_FILTER="${2:?--dir needs a name}"; shift ;;
    --runtime) WITH_RUNTIME=1 ;;
    -h|--help) sed -n '/^# Usage/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "Unknown option $1" >&2; exit 2 ;;
    *) BUILD_DIR="$1" ;;
  esac
  shift
done
BUILD_DIR="${BUILD_DIR:-build-riscv-list}"

cd "$(git rev-parse --show-toplevel)"

# Keep in sync with build_tools/cmake/test_riscv.sh.
LABEL_EXCLUDE='^(nodocker|driver=vulkan|driver=metal|driver=cuda|driver=hip|vulkan_uses_vk_khr_shader_float16_int8|requires-filesystem|requires-dtz|noriscv)$'
NAME_EXCLUDE='regression_llvm-cpu_lowering_config'

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  if grep -q "^IREE_BUILD_TESTS:BOOL=OFF" "${BUILD_DIR}/CMakeCache.txt"; then
    echo "${BUILD_DIR} was configured with IREE_BUILD_TESTS=OFF; pick another build dir." >&2
    exit 1
  fi
else
  HOST_BIN="$(realpath "${IREE_HOST_BIN_DIR:-build/install/bin}")"
  cmake -G Ninja -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_LLD=ON \
    -DCMAKE_TOOLCHAIN_FILE="$(realpath build_tools/cmake/linux_riscv64.cmake)" \
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT:-}" \
    -DIREE_HOST_BIN_DIR="${HOST_BIN}" \
    -DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_BENCHMARKS=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON >&2
fi

# One test name per line, e.g. "matmul/e2e_matmul_..._rvv" or "...(Disabled)".
list_tests() {
  ctest --test-dir "$1" -N --label-exclude "${LABEL_EXCLUDE}" "${@:2}" \
    | sed -nE 's/^ +Test +#[0-9]+: (iree\/)?(tests\/e2e\/)?//p'
}

E2E="$(list_tests "${BUILD_DIR}/tests/e2e" --exclude-regex "${NAME_EXCLUDE}")"

case "${MODE}" in
  all)  echo "${E2E}"; exit 0 ;;
  rvv)  echo "${E2E}" | grep -i -E 'rvv|riscv' || true; exit 0 ;;
  dir)  echo "${E2E}" | grep -E "^${DIR_FILTER}/" || true; exit 0 ;;
esac

total="$(echo "${E2E}" | grep -c .)"
disabled="$(echo "${E2E}" | grep -c '(Disabled)' || true)"
enabled="$(echo "${E2E}" | grep -v '(Disabled)' | grep -c . || true)"
echo "riscv64 e2e tests (tests/e2e), after test_riscv.sh exclusions:"
echo "  listed: ${total}   enabled: ${enabled}   disabled: ${disabled}"
echo
echo "Enabled, by backend (only llvm-cpu exercises RVV code generation):"
echo "${E2E}" | grep -v '(Disabled)' | awk '
  { b = ($0 ~ /vmvx/) ? "vmvx" : ($0 ~ /llvm-cpu|llvm_cpu/) ? "llvm-cpu" : "other"; c[b]++ }
  END { for (k in c) printf "  %-10s %d\n", k, c[k] }' | sort
echo
echo "Listed, by directory:"
echo "${E2E}" | sed -E 's#/.*##' | sort | uniq -c | sort -rn | awk '{ printf "  %-18s %d\n", $2, $1 }'
echo
echo "RVV-specific tests (names with rvv/riscv):"
echo "${E2E}" | grep -i -E 'rvv|riscv' | sed 's/^/  /' || true
if (( WITH_RUNTIME )); then
  echo
  echo "Runtime unit tests: $(list_tests "${BUILD_DIR}/runtime" | grep -c .)"
fi
echo
echo "More: --all | --rvv | --dir <name> ; disabled ones are marked (Disabled)."
