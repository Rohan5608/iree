#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# One-time developer setup for the RISC-V 64 + RVV workflow:
#   * creates ./.venv with the host compiler wheels, pre-commit and the ONNX
#     test requirements,
#   * installs the git hooks (pre-commit: lint; pre-push: ONNX tests on QEMU),
#   * submodules needed to cross-build the runtime.
#
# Usage: build_tools/riscv/setup_dev.sh [nightly|stable]
#   nightly (default) uses the same wheels as the `nightly` CI option.
#
# Run from anywhere inside the repository. It does not download the RISC-V
# toolchain or QEMU; see build_tools/riscv/README.md for those.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

SOURCE="${1:-nightly}"
PYTHON="${PYTHON:-python3}"

if [[ ! -d .venv ]]; then
  "${PYTHON}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

pip_args=()
if [[ "${SOURCE}" == "nightly" ]]; then
  pip_args=(--pre --find-links https://iree.dev/pip-release-links.html)
fi
python -m pip install "${pip_args[@]}" iree-base-compiler iree-base-runtime
python -m pip install pre-commit -r tests/riscv64_onnx/requirements.txt

# Installs both hook types listed in default_install_hook_types.
pre-commit install

./build_tools/scripts/git/update_runtime_submodules.sh

cat <<EOF

Setup done. Next steps:
  source .venv/bin/activate
  # Toolchain + QEMU (interactive): ./build_tools/riscv/riscv_bootstrap.sh
  export RISCV_TOOLCHAIN_ROOT=<toolchain dir>   QEMU_BIN=<qemu-riscv64>
  export IREE_HOST_BIN_DIR=\$PWD/.venv/bin
  ./build_tools/riscv/local_ci.sh all
EOF
