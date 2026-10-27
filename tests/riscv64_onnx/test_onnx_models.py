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


def test_model_matches_onnxruntime(model_path, tools, work_dir, pytestconfig):
    import onnxruntime as ort

    config = harness.ModelConfig.load(model_path)
    if config.skip:
        pytest.skip(config.skip)
    if config.xfail:
        pytest.xfail(config.xfail)

    rtol = config.rtol if config.rtol is not None else pytestconfig.getoption("rtol")
    atol = config.atol if config.atol is not None else pytestconfig.getoption("atol")

    # Reference: ONNX Runtime on the host CPU.
    try:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as e:  # onnxruntime raises various types on a bad model.
        pytest.fail(
            f"ONNX Runtime could not load {model_path.name}: {e}", pytrace=False
        )
    inputs = harness.generate_inputs(session, config, pytestconfig.getoption("seed"))
    expected = harness.run_onnxruntime(inputs, session)

    # Candidate: IREE, compiled for riscv64+RVV, run on QEMU.
    try:
        vmfb, entry = harness.import_and_compile(model_path, work_dir, tools, config)
        actual = harness.run_on_qemu(
            vmfb, entry, inputs, len(expected), work_dir, tools
        )
    except harness.HarnessError as e:
        pytest.fail(str(e), pytrace=False)

    problems = harness.compare_outputs(expected, actual, rtol, atol)
    if problems:
        pytest.fail(
            "IREE (riscv64+RVV on QEMU) differs from ONNX Runtime:\n  "
            + "\n  ".join(problems),
            pytrace=False,
        )
