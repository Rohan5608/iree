#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Standalone Linux host -> riscv64 Linux/RVV CI. Run from any directory.
#
# Examples (after installing the prerequisites documented in riscv64-ci.md):
#   bash build_tools/riscv/ci_rvv.sh all
#   IREE_RISCV_TEST_PROFILE=standard bash build_tools/riscv/ci_rvv.sh all
#   IREE_ONNX_MODELS_DIR=/path/to/case bash build_tools/riscv/ci_rvv.sh onnx
#   IREE_ONNX_SUITE=nightly IREE_ONNX_VLEN=512 bash build_tools/riscv/ci_rvv.sh onnx
#
# Stage order: bootstrap -> host -> [lit for standard] -> runtime -> test ->
# e2e -> onnx -> package. Individual stages reuse existing prerequisites/outputs;
# invoking `onnx` does not build a missing compiler or cross-runtime automatically.
# Generated outputs stay in IREE_RISCV_CI_DIR (default: build-riscv-ci).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
CI_DIR="${IREE_RISCV_CI_DIR:-${ROOT}/build-riscv-ci}"
mkdir -p "${CI_DIR}"
CI_DIR="$(realpath "${CI_DIR}")"
HOST="${CI_DIR}/host"
TARGET="${CI_DIR}/target"
TOOLS="${CI_DIR}/tools"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
PROFILE="${IREE_RISCV_TEST_PROFILE:-smoke}"
HOST_BIN="${IREE_HOST_BIN_DIR:-${HOST}/tools}"
# Source CI uses build-tree host tools. Package CI supplies IREE_HOST_BIN_DIR
# pointing at the venv where this commit's host wheels were installed.
case "${PROFILE}" in
  smoke|standard) ;;
  *) echo "IREE_RISCV_TEST_PROFILE must be smoke or standard" >&2; exit 2 ;;
esac
# The published archive has historically used both flat and bin/ layouts.
export QEMU_BIN="${TOOLS}/qemu/qemu-riscv64"
if [[ -x "${TOOLS}/qemu/bin/qemu-riscv64" ]]; then
  export QEMU_BIN="${TOOLS}/qemu/bin/qemu-riscv64"
fi
export RISCV_TOOLCHAIN_ROOT="${TOOLS}/toolchain"
FILTER='^iree/(base/(bitmap|status|string_builder|string_view)|task/(executor|topology|process)|vm/(buffer|list|ref|stack))_test$'
TEST_TARGETS=(
  iree_base_bitmap_test iree_base_status_test iree_base_string_builder_test
  iree_base_string_view_test iree_task_executor_test iree_task_topology_test
  iree_task_process_test iree_vm_buffer_test iree_vm_list_test
  iree_vm_ref_test iree_vm_stack_test iree_runtime_task_expanded_sysfs_snapshots
)

# Download and verify pinned archives before extraction; no compiler build here.
bootstrap() {
  local base=https://sharkpublic.blob.core.windows.net/sharkpublic/GCP-Migration-Files
  mkdir -p "${TOOLS}/downloads"
  # Recheck the checksum on every run, including restored download caches.
  local name sha dest
  while read -r name sha dest; do
    local archive="${TOOLS}/downloads/${name}"
    if [[ ! -f "${archive}" ]]; then
      curl --fail --location --retry 3 "${base}/${name}" -o "${archive}.partial"
      mv "${archive}.partial" "${archive}"
    fi
    printf '%s  %s\n' "${sha}" "${archive}" | sha256sum --check --status
    mkdir -p "${TOOLS}/${dest}"
    tar -xzf "${archive}" -C "${TOOLS}/${dest}" --strip-components=1 --no-same-owner
  done <<'ARTIFACTS'
toolchain_iree_manylinux_2_28_20231012.tar.gz 3af56a58551ed5ae7441214822461a5368fee9403d7c883762fa902489bfbff0 toolchain
qemu-riscv_8.1.2_manylinux_2.28_20231026.tar.gz dd77b39820d45b80bafab9155581578b4c625cb92fd6db9e9adbb9798fde3597 qemu
ARTIFACTS
  if [[ -x "${TOOLS}/qemu/bin/qemu-riscv64" ]]; then
    export QEMU_BIN="${TOOLS}/qemu/bin/qemu-riscv64"
  fi
  "${QEMU_BIN}" --version
  "${RISCV_TOOLCHAIN_ROOT}/bin/clang" --version
}

# Build native tools and Python importer bindings. ONNX still needs Torch input
# support in smoke mode; standard additionally builds host test prerequisites.
host() {
  local tests=OFF
  if [[ "${PROFILE}" == standard ]]; then tests=ON; fi
  cmake -G Ninja -S . -B "${HOST}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DIREE_BUILD_TESTS="${tests}" -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    '-DIREE_DEFAULT_CPU_LLVM_TARGETS=X86;ARM;AArch64;RISCV' \
    -DIREE_INPUT_STABLEHLO="${tests}" -DIREE_INPUT_TORCH=ON -DIREE_INPUT_TOSA="${tests}" \
    -DLLVM_PARALLEL_LINK_JOBS=1
  cmake --build "${HOST}" --target iree-compile iree-flatcc-cli IREECompilerPythonModules -j "${JOBS}"
  if [[ "${PROFILE}" == standard ]]; then
    cmake --build "${HOST}" -j "${JOBS}"
    cmake --build "${HOST}" --target iree-test-deps -j "${JOBS}"
  fi
}

# Execute registered compiler lit tests on the host; requires a standard host
# build. These inspect compiler behavior rather than run RISC-V executables.
lit() {
  ctest --test-dir "${HOST}/compiler" -L '^test-type=lit-test$' \
    --output-on-failure --no-tests=error --timeout 900 -j "${JOBS}" \
    --output-junit "${CI_DIR}/compiler-lit.xml"
}

# Cross-build target tools/tests, importing native compile/FlatCC tools from
# HOST_BIN. Keep host and target build directories separate to avoid ABI mixing.
runtime() {
  cmake -G Ninja -S . -B "${TARGET}" \
    -DCMAKE_BUILD_TYPE=Release -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_TOOLCHAIN_FILE="${ROOT}/build_tools/cmake/linux_riscv64.cmake" \
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT}" \
    -DIREE_ENABLE_LLD=ON -DIREE_BUILD_COMPILER=OFF \
    -DIREE_HOST_BIN_DIR="${HOST_BIN}" \
    -DIREE_BUILD_TESTS=ON -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON
  if [[ "${PROFILE}" == standard ]]; then
    cmake --build "${TARGET}" -j "${JOBS}"
    cmake --build "${TARGET}" --target iree-test-deps -j "${JOBS}"
  else
    cmake --build "${TARGET}" --target iree-run-module "${TEST_TARGETS[@]}" -j "${JOBS}"
  fi
}

# Standard runs supported runtime/CTS and core e2e suites with upstream RISC-V
# exclusions. Smoke runs the fixed 11-test selection and verifies its JUnit count.
test_runtime() {
  if [[ "${PROFILE}" == standard ]]; then
    local exclude='^(nodocker|driver=vulkan|driver=metal|driver=cuda|driver=hip|vulkan_uses_vk_khr_shader_float16_int8|requires-filesystem|requires-dtz|noriscv)$'
    ctest --test-dir "${TARGET}/runtime" --label-exclude "${exclude}" \
      --output-on-failure --no-tests=error --timeout 900 -j "${JOBS}" \
      --output-junit "${CI_DIR}/runtime-tests.xml"
    ctest --test-dir "${TARGET}/tests/e2e" --label-exclude "${exclude}" \
      --exclude-regex regression_llvm-cpu_lowering_config \
      --output-on-failure --no-tests=error --timeout 900 -j "${JOBS}" \
      --output-junit "${CI_DIR}/core-e2e.xml"
    return
  fi
  ctest --test-dir "${TARGET}" --output-on-failure --no-tests=error \
    --timeout 180 -j "${JOBS}" -R "${FILTER}" \
    --output-junit "${CI_DIR}/runtime-tests.xml"
  python3 - "${CI_DIR}/runtime-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
cases = list(root.iter("testcase"))
if len(cases) != 11 or any(
    case.find(tag) is not None
    for case in cases
    for tag in ("failure", "error", "skipped")
):
    raise SystemExit("Expected 11 executed, passing runtime tests")
PY
}

# Existing handwritten MLIR smoke test uses its own 512-bit profile. This is
# separate from automatically discovered ONNX cases (default VLEN=1024).
e2e() {
  "${HOST_BIN}/iree-compile" build_tools/riscv/rvv_smoke.mlir \
    --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d \
    --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl512b \
    -o "${CI_DIR}/rvv_smoke.vmfb"
  for device in local-sync local-task; do
    "${QEMU_BIN}" -cpu rv64,v=true,vlen=512,elen=64,vext_spec=v1.0 \
      -L "${RISCV_TOOLCHAIN_ROOT}/sysroot" "${TARGET}/tools/iree-run-module" \
      --device="${device}" --module="${CI_DIR}/rvv_smoke.vmfb" \
      --function=abs_plus_one --input=16xf32=-5 --expected_output=16xf32=6
  done
}

# Dispatch common ONNX discovery/preparation/comparison. The Python runner owns
# per-model selection, VLEN, inputs and reports; this wrapper supplies tool paths.
onnx_tests() {
  local -a args=(
    --suite "${IREE_ONNX_SUITE:-pr}"
    --models-dir "${IREE_ONNX_MODELS_DIR:-${ROOT}/tests/models/onnx}"
    --output-dir "${CI_DIR}/onnx"
    --compiler "${HOST_BIN}/iree-compile"
    --runner "${TARGET}/tools/iree-run-module"
    --qemu "${QEMU_BIN}"
    --sysroot "${RISCV_TOOLCHAIN_ROOT}/sysroot"
  )
  if [[ -n "${IREE_ONNX_VLEN:-}" ]]; then args+=(--vlen "${IREE_ONNX_VLEN}"); fi
  if [[ "${HOST_BIN}" == "${HOST}/tools" ]]; then
    # Use importer bindings from the source build, not an unrelated installed
    # compiler package. Package CI's activated venv already provides its bindings.
    PYTHONPATH="${HOST}/compiler/bindings/python${PYTHONPATH:+:${PYTHONPATH}}" \
      python3 build_tools/riscv/onnx_compare.py "${args[@]}"
  else
    python3 build_tools/riscv/onnx_compare.py "${args[@]}"
  fi
}

# Assemble outputs from prior successful stages; this does not build Python
# wheels, publish releases, or deploy to hardware. Workflow uploads happen later.
package() {
  local out="${CI_DIR}/artifacts"
  mkdir -p "${out}"
  cp "${TARGET}/tools/iree-run-module" "${CI_DIR}/rvv_smoke.vmfb" \
    "${CI_DIR}/runtime-tests.xml" "${out}/"
  local report
  for report in compiler-lit.xml core-e2e.xml; do
    if [[ -f "${CI_DIR}/${report}" ]]; then cp "${CI_DIR}/${report}" "${out}/"; fi
  done
  if [[ -f "${CI_DIR}/onnx/onnx-results.xml" ]]; then
    cp "${CI_DIR}/onnx/onnx-results.xml" "${out}/"
  fi
  git rev-parse HEAD > "${out}/commit.txt"
  printf '%s\n' 'riscv64 Linux, lp64d, RVV 1.0; QEMU validation only.' \
    'Core/custom smoke profile: VLEN=512. ONNX default: VLEN=1024; actual per-case values are recorded in ONNX results.json.' \
    "Test profile: ${PROFILE}" > "${out}/target.txt"
  (cd "${out}" && sha256sum iree-run-module rvv_smoke.vmfb ./*.xml commit.txt target.txt > SHA256SUMS)
}

case "${1:-all}" in
  bootstrap) bootstrap ;;
  host) host ;;
  lit) lit ;;
  runtime) runtime ;;
  test) test_runtime ;;
  e2e) e2e ;;
  onnx) onnx_tests ;;
  package) package ;;
  all) bootstrap; host; if [[ "${PROFILE}" == standard ]]; then lit; fi; runtime; test_runtime; e2e; onnx_tests; package ;;
  *) echo "Usage: $0 [bootstrap|host|lit|runtime|test|e2e|onnx|package|all]" >&2; exit 2 ;;
esac
