#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Standalone model/input preparation; usable locally or as a runner hook."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx

from onnx_compare import case_config, case_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    model_path = args.model.resolve()
    out = args.output_dir.resolve()
    if out.is_relative_to(model_path.parent):
        parser.error("Write prepared outputs outside the model case directory")
    out.mkdir(parents=True, exist_ok=True)
    config = case_config(model_path)
    model = onnx.load(model_path)  # Resolves .data/.bin weights relative to model.
    onnx.checker.check_model(model)
    feeds = case_inputs(model, config, model_path, out, args.timeout)
    np.savez(out / "inputs.npz", **feeds)
    model.graph.name = "main"
    del model.graph.value_info[:]
    for value in model.graph.output:
        if not value.type.HasField("tensor_type"):
            raise ValueError("Only tensor outputs are supported")
        value.type.tensor_type.ClearField("shape")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save_model(model, out / "model.onnx")
    prepared_config = {key: config[key] for key in ("rtol", "atol", "run_on", "vlen") if key in config}
    prepared_config.update(inputs="inputs.npz", static=True)
    (out / "model.json").write_text(json.dumps(prepared_config, indent=2) + "\n")
    print(f"Prepared model and named input arrays in {out}")


if __name__ == "__main__":
    main()
