#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compiles only the riscv64 `iree-run-module` runtime tool, which is all
# that tests/riscv64_onnx needs (models are compiled with the host compiler).
# This is a much smaller build than build_tools/cmake/build_riscv.sh, which
# builds and installs the whole runtime and its test suite.
#
# Environment:
#   IREE_ENABLE_LLD         ON (default) or OFF; OFF to link with the default linker.
#   RISCV_TOOLCHAIN_ROOT    Clang+sysroot toolchain (see riscv_bootstrap.sh).
#                           If empty, CC/CXX must already target riscv64.
#   RISCV_TOOLCHAIN_PREFIX  Optional tool name prefix inside the toolchain.
#   CMAKE_BUILD_TYPE        Defaults to RelWithDebInfo.
#   IREE_HOST_BIN_DIR       Directory with host iree-flatcc-cli and
#                           iree-c-embed-data (e.g. a pip venv's bin/), used
#                           instead of building those tools for the host.
#                           Defaults to build/install/bin.
#
# The build directory can be passed as the first argument. Otherwise it uses
# IREE_TARGET_BUILD_DIR, defaulting to "build-riscv". Expects to be run from the
# root of the IREE repository. The resulting binary is
# "<build dir>/tools/iree-run-module".

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_TARGET_BUILD_DIR:-build-riscv}}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
CMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-$(realpath build_tools/cmake/linux_riscv64.cmake)}"
RISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT:-}"
RISCV_TOOLCHAIN_PREFIX="${RISCV_TOOLCHAIN_PREFIX:-}"
IREE_HOST_BIN_DIR="$(realpath "${IREE_HOST_BIN_DIR:-build/install/bin}")"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"

  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DIREE_ENABLE_LLD=${IREE_ENABLE_LLD:-ON}"

  # Cross compiling the RISC-V runtime only.
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_BUILD_TESTS=OFF"
  "-DIREE_BUILD_SAMPLES=OFF"
  "-DIREE_BUILD_BENCHMARKS=OFF"
  "-DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}"
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
  "-DRISCV_TOOLCHAIN_PREFIX=${RISCV_TOOLCHAIN_PREFIX}"
  "-DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT}"

  # The CPU drivers are enough to load the llvm-cpu embedded ELF output.
  "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
  "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
  "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"
  "-DIREE_TARGET_BACKEND_DEFAULTS=OFF"
  "-DIREE_TARGET_BACKEND_LLVM_CPU=ON"
)

"${CMAKE_BIN}" "${args[@]}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target iree-run-module

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi

file "${BUILD_DIR}/tools/iree-run-module" || true
