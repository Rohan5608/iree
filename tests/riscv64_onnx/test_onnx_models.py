# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One test per .onnx file in the models directory (see conftest.py).

Each test compares IREE running on emulated riscv64 + RVV against ONNX Runtime
running on the host.
"""

import pytest

import riscv_onnx_harness as harness


def test_model_matches_onnxruntime(model_path, vlen, tools, work_dir, pytestconfig):
    config = harness.ModelConfig.load(model_path)
    if config.skip:
        pytest.skip(config.skip)

    # In the stable tier a known failure (xfail) is still run, so an entry that
    # has started to pass is reported and must be removed. Other tiers do not
    # spend time on it.
    stable = pytestconfig.getoption("tier") == "stable"
    if config.xfail and not stable:
        pytest.xfail(config.xfail)

    failure = _run_model(model_path, vlen, tools, work_dir, pytestconfig, config)

    if config.xfail:  # stable tier only
        if failure is None:
            pytest.fail(
                f"xfail is stale: {model_path.name} now passes at VLEN={vlen}. "
                f'Remove "xfail" from its sidecar (was: {config.xfail})',
                pytrace=False,
            )
        pytest.xfail(f"{config.xfail} [{failure.splitlines()[0]}]")
    if failure is not None:
        pytest.fail(failure, pytrace=False)


def _run_model(model_path, vlen, tools, work_dir, pytestconfig, config):
    """Runs one model. Returns None on success or a failure message."""
    import onnxruntime as ort

    tools = harness.with_vlen(tools, vlen)
    rtol = config.rtol if config.rtol is not None else pytestconfig.getoption("rtol")
    atol = config.atol if config.atol is not None else pytestconfig.getoption("atol")

    # Reference: ONNX Runtime on the host CPU.
    try:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as e:  # onnxruntime raises various types on a bad model.
        return f"ONNX Runtime could not load {model_path.name}: {e}"
    try:
        inputs = harness.generate_inputs(
            session, config, pytestconfig.getoption("seed")
        )
        expected = harness.run_onnxruntime(inputs, session)
    except harness.HarnessError as e:
        return str(e)

    # Candidate: IREE, compiled for riscv64+RVV, run on QEMU.
    try:
        vmfb, entry = harness.import_and_compile(model_path, work_dir, tools, config)
        actual = harness.run_on_qemu(
            vmfb, entry, inputs, len(expected), work_dir, tools
        )
    except harness.HarnessError as e:
        return str(e)

    problems = harness.compare_outputs(expected, actual, rtol, atol)
    if problems:
        return (
            f"IREE (riscv64+RVV, VLEN={vlen}, on QEMU) differs from ONNX Runtime:\n  "
            + "\n  ".join(problems)
        )
    return None
