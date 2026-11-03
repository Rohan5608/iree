#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Decides the test tier and the host compiler source for .github/workflows/
# riscv64_ci.yml. Kept as a script (not inline YAML) so it can be tested; see
# tests/riscv64_onnx/test_ci_config.py. Documentation: build_tools/riscv/STRATEGY.md.
#
# Inputs (environment; empty means unset):
#   EVENT_NAME      GitHub event (pull_request, push, schedule, workflow_dispatch).
#   REF_TYPE        branch or tag.   REF_NAME   branch or tag name.
#   INPUT_TIER      pr | nightly | stable | auto   (workflow_dispatch input)
#   INPUT_SOURCE    nightly | stable | build | auto
#   VAR_SOURCE      repository variable RISCV_COMPILER_SOURCE (pr tier only)
#   INPUT_RUNNER    runner label for compiler_source=build
#   VAR_RUNNER      repository variable RISCV_BUILD_RUNNER
#   INPUT_VERSION   pin for the nightly/stable wheels, e.g. 3.13.0
#
# Tier rules:
#   explicit INPUT_TIER wins; else a scheduled run is "nightly"; else a tag
#   named riscv-stable-* is "stable"; everything else is "pr".
# Default compiler source per tier:
#   pr -> VAR_SOURCE or nightly wheels;  nightly -> build from the branch;
#   stable -> stable wheels. INPUT_SOURCE overrides all of these.
#
# Output: key=value lines appended to $GITHUB_OUTPUT (stdout if unset):
#   tier, compiler-source, compiler-version, build-runner, run-lint

set -euo pipefail

EVENT_NAME="${EVENT_NAME:-}"
REF_TYPE="${REF_TYPE:-}"
REF_NAME="${REF_NAME:-}"
INPUT_TIER="${INPUT_TIER:-}"
INPUT_SOURCE="${INPUT_SOURCE:-}"
VAR_SOURCE="${VAR_SOURCE:-}"
INPUT_RUNNER="${INPUT_RUNNER:-}"
VAR_RUNNER="${VAR_RUNNER:-}"
INPUT_VERSION="${INPUT_VERSION:-}"

fail() { echo "::error::$*" >&2; exit 1; }

case "${INPUT_TIER}" in
  pr|nightly|stable) tier="${INPUT_TIER}" ;;
  ""|auto)
    if [[ "${EVENT_NAME}" == "schedule" ]]; then
      tier=nightly
    elif [[ "${REF_TYPE}" == "tag" && "${REF_NAME}" == riscv-stable-* ]]; then
      tier=stable
    else
      tier=pr
    fi ;;
  *) fail "Invalid tier '${INPUT_TIER}' (use pr, nightly, stable or auto)" ;;
esac

case "${tier}" in
  pr)      default_source="${VAR_SOURCE:-nightly}" ;;
  nightly) default_source=build ;;
  stable)  default_source=stable ;;
esac
case "${INPUT_SOURCE}" in
  ""|auto) source="${default_source}" ;;
  *)       source="${INPUT_SOURCE}" ;;
esac
case "${source}" in
  nightly|stable|build) ;;
  *) fail "Invalid compiler source '${source}' (use nightly, stable or build)" ;;
esac

if [[ -n "${INPUT_VERSION}" ]]; then
  [[ "${INPUT_VERSION}" =~ ^[0-9][0-9A-Za-z.+!-]*$ ]] \
    || fail "Invalid compiler version '${INPUT_VERSION}'"
  [[ "${source}" != "build" ]] \
    || fail "compiler_version has no effect with compiler_source=build"
fi

runner="${INPUT_RUNNER:-${VAR_RUNNER:-ubuntu-24.04}}"
# The nightly tier only checks the tested state; lint runs on the other tiers.
run_lint=true
[[ "${tier}" != "nightly" ]] || run_lint=false

out="${GITHUB_OUTPUT:-/dev/stdout}"
{
  echo "tier=${tier}"
  echo "compiler-source=${source}"
  echo "compiler-version=${INPUT_VERSION}"
  echo "build-runner=${runner}"
  echo "run-lint=${run_lint}"
} >> "${out}"
