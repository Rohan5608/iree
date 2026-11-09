#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prepare one ONNX case without compiling or running inference.

Example from the repository root:
    python build_tools/riscv/prepare_onnx.py --model-dir tests/models/onnx/relu \
        --output-dir build-prepared/relu

Reads: one ONNX file, its optional JSON manifest, external weights, and inputs.
Writes: model.onnx, inputs.npz, model.json, and input.log when using a generator.
The output directory must be outside the source case. The comparison runner
calls this same script, so local preparation and CI share one implementation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx

from onnx_case import case_config, case_inputs


def main():
    """Resolve a single case and write a self-contained prepared test bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--model-dir", type=Path, help="Case directory containing exactly one ONNX file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.model_dir is not None:
        models = sorted(args.model_dir.glob("*.onnx"))
        if len(models) != 1:
            parser.error("--model-dir must contain exactly one .onnx file")
        model_path = models[0].resolve()
    else:
        model_path = args.model.resolve()
    out = args.output_dir.resolve()
    # Keep generated ONNX files out of the source case to prevent accidental
    # overwrites and duplicate discovery on a later recursive comparison run.
    if out.is_relative_to(model_path.parent):
        parser.error("Write prepared outputs outside the model case directory")
    out.mkdir(parents=True, exist_ok=True)
    config = case_config(model_path)
    model = onnx.load(model_path)  # Resolves .data/.bin weights relative to model.
    onnx.checker.check_model(model)
    feeds = case_inputs(model, config, model_path, out, args.timeout)
    np.savez(out / "inputs.npz", **feeds)
    # Normalize the exported function name expected by iree-run-module. Reinfer
    # stale intermediate/output shapes after any concrete input specialization.
    model.graph.name = "main"
    del model.graph.value_info[:]
    for value in model.graph.output:
        if not value.type.HasField("tensor_type"):
            raise ValueError("Only tensor outputs are supported")
        value.type.tensor_type.ClearField("shape")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save_model(model, out / "model.onnx")
    # A prepared directory can itself be reused as a case. Point at saved arrays,
    # not source generators/hooks, so it does not regenerate different inputs.
    prepared_config = {key: config[key] for key in ("rtol", "atol", "run_on", "vlen") if key in config}
    prepared_config.update(inputs="inputs.npz", static=True)
    (out / "model.json").write_text(json.dumps(prepared_config, indent=2) + "\n")
    print(f"Prepared model and named input arrays in {out}")


if __name__ == "__main__":
    main()
