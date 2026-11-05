#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs one IREE test suite by name, on its own.
#
# Usage: build_tools/riscv/run_suite.sh <suite> [options] [-- extra args]
#
# Suites:
#   lit      Compiler lit tests (compiler/**/test/*.mlir). Host build.
#            Default build dir: build-host (configure it with
#            IREE_HOST_BUILD_TESTS=ON, see build_host_compiler.sh).
#   runtime  Runtime unit tests and HAL conformance tests (runtime/).
#   e2e      Core end-to-end `check` tests (tests/e2e/**).
#            runtime and e2e: default build dir $IREE_TARGET_BUILD_DIR or
#            build-riscv. If the build dir is a riscv64 cross-build (detected
#            from its toolchain file), tests run under QEMU and the same
#            exclusions as upstream's test_riscv.sh are applied.
#   onnx     Our ONNX-vs-ONNX-Runtime suite on riscv64 (tests/riscv64_onnx);
#            see run_onnx_tests.sh. Extra args go to pytest.
#
# Options:
#   -b, --build-dir DIR  Build directory (default depends on the suite).
#   -R, --regex RE       Only tests whose name matches RE (ctest -R). For onnx
#                        this is a pytest -k expression, e.g. -R 'mlp and vlen256'.
#   -L, --label RE       Only tests with a label matching RE (ctest -L).
#   -l, --list           List the tests that would run; run nothing.
#       --build-deps     Build first: e2e builds `iree-test-deps`; lit and
#                        runtime build everything (cmake --build DIR).
#       --no-exclude     Do not apply the riscv64 label/name exclusions.
#   -h, --help           This text.
#   -- ARGS...           Passed through to ctest (lit, runtime, e2e) or pytest
#                        (onnx).
#
# Examples:
#   run_suite.sh e2e -l                           # what runs on riscv64
#   run_suite.sh e2e -R 'linalg/.*softmax'        # one test
#   run_suite.sh runtime -R iree/base/bitfield_test
#   run_suite.sh lit -b build-host -R Codegen/LLVMCPU
#   run_suite.sh onnx -- -k mlp --vlens 128,256
#
# Environment: CTEST_PARALLEL_LEVEL (default: nproc), QEMU_BIN (required for a
# riscv64 build dir; qemu-riscv64), and for onnx the variables documented in
# run_onnx_tests.sh.

set -euo pipefail

usage() { sed -n '/^# Usage/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "run_suite.sh: $*" >&2; exit 2; }

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/build_tools/riscv/riscv_test_filters.sh"

if (( $# == 0 )); then usage; exit 2; fi
case "$1" in -h|--help) usage; exit 0 ;; esac
SUITE="$1"; shift

BUILD_DIR=""; REGEX=""; LABEL=""; LIST=0; BUILD_DEPS=0; EXCLUDE=1
EXTRA=()
while (( $# > 0 )); do
  case "$1" in
    -b|--build-dir) BUILD_DIR="${2:?$1 needs a value}"; shift ;;
    -R|--regex)     REGEX="${2:?$1 needs a value}"; shift ;;
    -L|--label)     LABEL="${2:?$1 needs a value}"; shift ;;
    -l|--list)      LIST=1 ;;
    --build-deps)   BUILD_DEPS=1 ;;
    --no-exclude)   EXCLUDE=0 ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; EXTRA=("$@"); break ;;
    *)              die "unknown option '$1' (see --help)" ;;
  esac
  shift
done

case "${SUITE}" in
  lit)           : "${BUILD_DIR:=build-host}" ;;
  runtime|e2e|onnx) : "${BUILD_DIR:=${IREE_TARGET_BUILD_DIR:-build-riscv}}" ;;
  *) die "unknown suite '${SUITE}' (lit, runtime, e2e, onnx)" ;;
esac

# --- onnx: delegate to the pytest runner --------------------------------------
if [[ "${SUITE}" == "onnx" ]]; then
  (( ! BUILD_DEPS )) || die "--build-deps does not apply to onnx; run build_run_module.sh"
  args=()
  [[ -z "${REGEX}" ]] || args+=(-k "${REGEX}")
  (( ! LIST )) || args+=(--collect-only -q)
  exec ./build_tools/riscv/run_onnx_tests.sh "${BUILD_DIR}" \
    ${args[@]+"${args[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
fi

# --- ctest-based suites ---------------------------------------------------------
[[ -f "${BUILD_DIR}/CMakeCache.txt" ]] \
  || die "'${BUILD_DIR}' is not a configured CMake build directory (use --build-dir)"

if grep -q "^IREE_BUILD_TESTS:BOOL=OFF" "${BUILD_DIR}/CMakeCache.txt"; then
  die "'${BUILD_DIR}' was configured with IREE_BUILD_TESTS=OFF, so it has no ${SUITE} tests."
fi

IS_RISCV=0
if grep -q "^CMAKE_TOOLCHAIN_FILE:.*riscv" "${BUILD_DIR}/CMakeCache.txt"; then
  IS_RISCV=1
fi

case "${SUITE}" in
  lit)     TEST_DIR="${BUILD_DIR}"; LABEL="${LABEL:-test-type=lit-test}" ;;
  runtime) TEST_DIR="${BUILD_DIR}/runtime" ;;
  e2e)     TEST_DIR="${BUILD_DIR}/tests/e2e" ;;
esac
[[ -d "${TEST_DIR}" ]] || die "no ${SUITE} tests in '${BUILD_DIR}' (expected ${TEST_DIR})"
if [[ "${SUITE}" == "lit" ]] && (( IS_RISCV )); then
  die "lit tests run on the host; '${BUILD_DIR}' is a riscv64 build (use a host build such as build-host)"
fi

if (( BUILD_DEPS )); then
  if [[ "${SUITE}" == "e2e" ]]; then
    cmake --build "${BUILD_DIR}" --target iree-test-deps
  else
    cmake --build "${BUILD_DIR}"
  fi
fi

args=(--test-dir "${TEST_DIR}" --output-on-failure --no-tests=error)
[[ -z "${REGEX}" ]] || args+=(-R "${REGEX}")
[[ -z "${LABEL}" ]] || args+=(-L "${LABEL}")
if (( IS_RISCV && EXCLUDE )); then
  args+=(--label-exclude "${RISCV_LABEL_EXCLUDE}")
  # The name exclusion only applies to the e2e suite upstream.
  [[ "${SUITE}" != "e2e" ]] || args+=(--exclude-regex "${RISCV_NAME_EXCLUDE}")
fi

if (( LIST )); then
  exec ctest "${args[@]}" -N ${EXTRA[@]+"${EXTRA[@]}"}
fi

if (( IS_RISCV )); then
  [[ -n "${QEMU_BIN:-}" ]] \
    || die "'${BUILD_DIR}' is a riscv64 build: set QEMU_BIN to qemu-riscv64 (or use --list)"
  export QEMU_BIN
fi
export CTEST_PARALLEL_LEVEL="${CTEST_PARALLEL_LEVEL:-$(nproc)}"

echo "run_suite.sh: ${SUITE} in ${TEST_DIR}$( ((IS_RISCV)) && echo ' (riscv64, QEMU)')" >&2
log="$(mktemp)"
trap 'rm -f "${log}"' EXIT
rc=0
ctest "${args[@]}" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "${log}" || rc=$?
if (( rc != 0 )) && grep -q "(Not Run)" "${log}"; then
  echo "run_suite.sh: some tests are '(Not Run)': their binaries or modules are not built." >&2
  echo "  Build them first: run_suite.sh ${SUITE} -b ${BUILD_DIR} --build-deps" >&2
fi
exit "${rc}"
