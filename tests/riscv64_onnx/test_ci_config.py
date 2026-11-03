# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests the tier / compiler-source decision used by riscv64_ci.yml."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "build_tools/riscv/resolve_ci_config.sh"


def resolve(**env):
    clean = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME")}
    clean.update(env)
    proc = subprocess.run(
        ["bash", str(SCRIPT)], env=clean, capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines()), ""


@pytest.mark.parametrize(
    "env, tier, source",
    [
        # Today's behavior must not change: PRs and pushes use nightly wheels.
        (dict(EVENT_NAME="pull_request"), "pr", "nightly"),
        (dict(EVENT_NAME="push", REF_TYPE="branch", REF_NAME="main"), "pr", "nightly"),
        # The repo variable only affects the pr tier.
        (dict(EVENT_NAME="pull_request", VAR_SOURCE="stable"), "pr", "stable"),
        (dict(EVENT_NAME="schedule", VAR_SOURCE="stable"), "nightly", "build"),
        # A schedule means nightly; a stable tag means stable.
        (dict(EVENT_NAME="schedule"), "nightly", "build"),
        (
            dict(EVENT_NAME="push", REF_TYPE="tag", REF_NAME="riscv-stable-2026.11"),
            "stable",
            "stable",
        ),
        (dict(EVENT_NAME="push", REF_TYPE="tag", REF_NAME="v1.0"), "pr", "nightly"),
        # Manual runs: explicit tier and source win.
        (dict(EVENT_NAME="workflow_dispatch", INPUT_TIER="stable"), "stable", "stable"),
        (dict(EVENT_NAME="workflow_dispatch", INPUT_TIER="auto"), "pr", "nightly"),
        (
            dict(
                EVENT_NAME="workflow_dispatch",
                INPUT_TIER="nightly",
                INPUT_SOURCE="stable",
            ),
            "nightly",
            "stable",
        ),
        (
            dict(
                EVENT_NAME="workflow_dispatch",
                INPUT_TIER="stable",
                INPUT_SOURCE="build",
            ),
            "stable",
            "build",
        ),
    ],
)
def test_tier_and_source(env, tier, source):
    out, err = resolve(**env)
    assert out is not None, err
    assert (out["tier"], out["compiler-source"]) == (tier, source)


def test_lint_only_skipped_for_nightly():
    assert resolve(EVENT_NAME="schedule")[0]["run-lint"] == "false"
    assert resolve(EVENT_NAME="pull_request")[0]["run-lint"] == "true"
    assert resolve(INPUT_TIER="stable")[0]["run-lint"] == "true"


def test_runner_and_version():
    out, _ = resolve(INPUT_TIER="stable", INPUT_VERSION="3.13.0")
    assert out["compiler-version"] == "3.13.0"
    assert out["build-runner"] == "ubuntu-24.04"
    out, _ = resolve(VAR_RUNNER="big-runner")
    assert out["build-runner"] == "big-runner"
    out, _ = resolve(VAR_RUNNER="big-runner", INPUT_RUNNER="other")
    assert out["build-runner"] == "other"


@pytest.mark.parametrize(
    "env",
    [
        dict(INPUT_TIER="weekly"),
        dict(INPUT_SOURCE="docker"),
        dict(INPUT_VERSION="3.13.0; rm -rf /"),
        dict(INPUT_TIER="nightly", INPUT_VERSION="3.13.0"),  # nightly => build
    ],
)
def test_invalid_inputs_are_rejected(env):
    out, err = resolve(**env)
    assert out is None and err
