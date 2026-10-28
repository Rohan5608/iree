#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Disables the upstream IREE workflows that cannot work on a fork (they need
# the iree-org runner pools, secrets or publishing rights) so that pushes and
# pull requests on the fork only run `RISC-V CI` (riscv64_ci.yml).
#
# Usage: build_tools/riscv/disable_upstream_workflows.sh [owner/repo]
#   Requires the GitHub CLI (`gh auth login`) with permission to manage
#   Actions on the repository. Defaults to the repo of the current checkout.
#   Re-enable any workflow with `gh workflow enable <file>`.
#   Alternatively do the same in the web UI: Actions -> workflow -> ... ->
#   Disable workflow.
#
# Only top-level (event-triggered) workflows are listed. The `pkgci_test_*.yml`
# and `ci_*.yml` files that are `workflow_call`ed are only used by those.

set -euo pipefail

REPO_ARGS=()
if [[ $# -ge 1 ]]; then
  REPO_ARGS=(--repo "$1")
fi

WORKFLOWS=(
  # Run on every PR/push but need iree-org runners.
  ci.yml
  pkgci.yml
  build_clang_tidy.yml
  clang_tidy.yml
  # Replaced by the `lint` job in riscv64_ci.yml.
  lint.yml
  # Scheduled / publishing / bot workflows that make no sense on a fork.
  ci_linux_arm64_clang.yml
  ci_linux_x64_clang_byollvm.yml
  ci_linux_x64_clang_debug.yml
  ci_linux_x64_clang_tsan.yml
  ci_linux_x64_gcc.yml
  ci_macos_arm64_clang.yml
  samples.yml
  schedule_candidate_release.yml
  publish_website.yml
  issue_greeter.yml
  pull_request_greeter.yml
)

for wf in "${WORKFLOWS[@]}"; do
  echo "Disabling ${wf}"
  gh workflow disable "${wf}" "${REPO_ARGS[@]}" \
    || echo "  (could not disable ${wf}; already disabled or not present)"
done
