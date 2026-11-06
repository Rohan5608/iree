# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Validation of strict comparison and ONNX input configuration."""

import numpy as np
import onnx
import pytest

from onnx_compare import case_config, case_inputs, compare, inputs_for, main, vector_length


def model(shape):
    return onnx.helper.make_model(onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])], "identity",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)]))


@pytest.mark.parametrize("actual", [np.array([2.], dtype="float32"),
                                  np.array([[1.]], dtype="float32"),
                                  np.array([1.], dtype="float64"),
                                  np.array([np.nan], dtype="float32")])
def test_reject_wrong_outputs(actual):
    with pytest.raises(AssertionError):
        compare(np.array([1.], dtype="float32"), actual, 1e-4, 1e-5)


def test_integers_are_exact():
    with pytest.raises(AssertionError):
        compare(np.array([1], dtype="int64"), np.array([2], dtype="int64"), 1, 100)


def test_dynamic_requires_shape(tmp_path):
    with pytest.raises(ValueError, match="concrete"):
        inputs_for(model(["batch", 4]), {}, tmp_path)


def test_dynamic_npz_specialization(tmp_path):
    expected = np.arange(12, dtype="float32").reshape(3, 4)
    np.savez(tmp_path / "data.npz", x=expected)
    m = model(["batch", 4])
    feeds = inputs_for(m, {"inputs": "data.npz"}, tmp_path)
    np.testing.assert_array_equal(feeds["x"], expected)
    assert [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim] == [3, 4]


def test_reproducible_inputs(tmp_path):
    a = inputs_for(model([2, 4]), {}, tmp_path)
    b = inputs_for(model([2, 4]), {}, tmp_path)
    np.testing.assert_array_equal(a["x"], b["x"])


def test_empty_discovery_fails(tmp_path):
    assert main(["--models-dir", str(tmp_path / "empty"),
                 "--output-dir", str(tmp_path / "out"), "--native",
                 "--compiler", "unused", "--runner", "unused"]) == 1
    assert 'failures="1"' in (tmp_path / "out/onnx-results.xml").read_text()


def test_static_case_rejects_symbolic_shape(tmp_path):
    with pytest.raises(ValueError, match="fixed ONNX"):
        case_inputs(model(["batch", 4]), {}, tmp_path / "model.onnx", tmp_path, 5)


def test_script_failure_is_not_silently_replaced(tmp_path):
    import subprocess

    (tmp_path / "input.py").write_text("raise RuntimeError('broken input generator')\n")
    path = tmp_path / "model.onnx"
    config = case_config(path)
    assert config["input_script"] == "input.py"
    with pytest.raises(subprocess.CalledProcessError):
        case_inputs(model([2, 4]), config, path, tmp_path, 5)
    assert "broken input generator" in (tmp_path / "input.log").read_text()


def test_script_must_produce_npz(tmp_path):
    (tmp_path / "input.py").write_text("pass\n")
    with pytest.raises(ValueError, match="did not write"):
        case_inputs(model([2, 4]), {"input_script": "input.py"},
                    tmp_path / "model.onnx", tmp_path, 5)


def test_invalid_schedule_fails(tmp_path):
    (tmp_path / "model.json").write_text('{"run_on": ["typo"]}')
    with pytest.raises(ValueError, match="run_on"):
        case_config(tmp_path / "model.onnx")


def test_expensive_case_is_reported_skipped_in_pr(tmp_path):
    import json

    models = tmp_path / "models"
    models.mkdir()
    (models / "model.onnx").write_bytes(b"not loaded when filtered")
    (models / "model.json").write_text('{"run_on": ["nightly", "release"]}')
    # No selected tests is a failure, even though excluded tests are recorded.
    assert main(["--models-dir", str(models), "--output-dir", str(tmp_path / "out"),
                 "--native", "--suite", "pr", "--compiler", "unused", "--runner", "unused"]) == 1
    results = json.loads((tmp_path / "out/results.json").read_text())
    assert results[0]["status"] == "skipped"
    assert 'skipped="1"' in (tmp_path / "out/onnx-results.xml").read_text()


def test_vector_length_precedence():
    assert vector_length({}) == 1024
    assert vector_length({"vlen": 256}) == 256
    assert vector_length({"vlen": 256}, 512) == 512


@pytest.mark.parametrize("invalid", [0, 129, 2048, True, "1024", 1024.0])
def test_invalid_vector_lengths(invalid):
    with pytest.raises(ValueError, match="vlen"):
        vector_length({"vlen": invalid})
