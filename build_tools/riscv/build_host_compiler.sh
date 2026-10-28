#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds the host compiler from this checkout, so compiler changes (RVV
# codegen, passes, flags) can be tested without building the Python wheels the
# way CI's `compiler_source=build` does.
#
# Builds only: iree-compile, iree-opt, and the two helper tools the riscv64
# runtime cross-build needs (iree-flatcc-cli, iree-c-embed-data). The compiler
# has only the llvm-cpu and vmvx backends (all the riscv64 tests use) and no
# tests, samples, benchmarks or Python bindings, which makes it much smaller
# than the full build. LLVM is still built from source: expect roughly an hour
# or more on a laptop for the first build, seconds to minutes afterwards.
#
# Usage: build_tools/riscv/build_host_compiler.sh [build-dir]
#   build-dir defaults to IREE_HOST_BUILD_DIR or "build-host".
#
# Output: <build-dir>/host-bin/ with symlinks to the built tools. Use it as
#   export IREE_HOST_COMPILER_DIR=$PWD/build-host/host-bin
# and local_ci.sh will put it first on PATH and use it as IREE_HOST_BIN_DIR.
# By hand:
#   export PATH=$PWD/build-host/host-bin:$PATH   # iree-compile from your branch
#   export IREE_HOST_BIN_DIR=$PWD/build-host/host-bin
#
# `iree-import-onnx` is a Python tool and is still taken from the pip wheels
# (.venv). It only writes MLIR text, so a locally built iree-compile can consume
# it as long as the wheel and the checkout are not far apart.
#
# Environment:
#   CMAKE_BUILD_TYPE        Default Release.
#   IREE_ENABLE_ASSERTIONS  ON (default) or OFF. Keep ON while developing.
#   IREE_ENABLE_LLD         ON (default) or OFF.
#   IREE_INIT_SUBMODULES    1 (default): fetch llvm-project, stablehlo,
#                           torch-mlir and the runtime submodules if missing
#                           (several GB). Set 0 to skip.
#   IREE_HOST_BUILD_TESTS   ON to also configure the compiler tests (lit/FileCheck
#                           and unit tests) so you can run them with ctest; then
#                           build them with: cmake --build <build-dir> --target
#                           iree-test-deps. Default OFF.
#   IREE_HOST_TOOLS         Tools to build and collect (default: iree-compile
#                           iree-opt iree-flatcc-cli iree-c-embed-data).
#   CC / CXX                Host compilers; clang is recommended.

set -xeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

BUILD_DIR="${1:-${IREE_HOST_BUILD_DIR:-build-host}}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
IREE_ENABLE_ASSERTIONS="${IREE_ENABLE_ASSERTIONS:-ON}"
IREE_ENABLE_LLD="${IREE_ENABLE_LLD:-ON}"
# Space-separated override, e.g. IREE_HOST_TOOLS="iree-compile" to build less.
read -r -a TOOLS <<< "${IREE_HOST_TOOLS:-iree-compile iree-opt iree-flatcc-cli iree-c-embed-data}"

# The compiler needs these in addition to the runtime submodules.
if [[ "${IREE_INIT_SUBMODULES:-1}" == "1" ]]; then
  if [[ ! -e third_party/llvm-project/llvm/CMakeLists.txt \
        || ! -e third_party/stablehlo/CMakeLists.txt \
        || ! -e third_party/torch-mlir/CMakeLists.txt ]]; then
    git submodule sync
    git submodule update --init --jobs 8 --depth 1 \
      third_party/llvm-project third_party/stablehlo third_party/torch-mlir
  fi
  ./build_tools/scripts/git/update_runtime_submodules.sh
fi

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

declare -a args
args=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"

  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DPython3_EXECUTABLE=${IREE_PYTHON3_EXECUTABLE}"
  "-DIREE_ENABLE_ASSERTIONS=${IREE_ENABLE_ASSERTIONS}"
  "-DIREE_ENABLE_LLD=${IREE_ENABLE_LLD}"
  "-DIREE_ENABLE_THIN_ARCHIVES=ON"

  # Host compiler only.
  "-DIREE_BUILD_COMPILER=ON"
  "-DIREE_BUILD_TESTS=${IREE_HOST_BUILD_TESTS:-OFF}"
  "-DIREE_BUILD_SAMPLES=OFF"
  "-DIREE_BUILD_BENCHMARKS=OFF"
  "-DIREE_BUILD_PYTHON_BINDINGS=OFF"
  "-DIREE_BUILD_DOCS=OFF"

  # llvm-cpu (riscv64 codegen) and vmvx; no GPU backends or drivers.
  "-DIREE_TARGET_BACKEND_DEFAULTS=OFF"
  "-DIREE_TARGET_BACKEND_LLVM_CPU=ON"
  "-DIREE_TARGET_BACKEND_VMVX=ON"
  "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
  "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
  "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"
)

"${CMAKE_BIN}" "${args[@]}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target "${TOOLS[@]}"

# Collect the tools in one directory usable as IREE_HOST_BIN_DIR.
HOST_BIN="${BUILD_DIR}/host-bin"
mkdir -p "${HOST_BIN}"
for tool in "${TOOLS[@]}"; do
  src="$(find "${BUILD_DIR}" -path "${HOST_BIN}" -prune -o -type f -name "${tool}" -perm -u+x -print -quit)"
  if [[ -z "${src}" ]]; then
    echo "Could not find built tool '${tool}' under ${BUILD_DIR}" >&2
    exit 1
  fi
  ln -sf "$(realpath "${src}")" "${HOST_BIN}/${tool}"
done

if (( IREE_USE_CCACHE == 1 )); then
  ccache --show-stats
fi

set +x
echo
echo "Built host compiler tools in: $(realpath "${HOST_BIN}")"
echo "  export IREE_HOST_COMPILER_DIR=$(realpath "${HOST_BIN}")   # used by local_ci.sh"
if [[ -x "${HOST_BIN}/iree-compile" ]]; then
  "${HOST_BIN}/iree-compile" --version | head -3
fi
