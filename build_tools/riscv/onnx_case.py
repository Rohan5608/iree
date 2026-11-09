# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared case contracts used by preparation and comparison.

A case consists of an ONNX file, an optional same-stem JSON manifest, and input
data or a generator. This module resolves that configuration and returns named
NumPy arrays. It does not compile models, run ONNX Runtime, or invoke QEMU.
See prepare_onnx.py for the standalone CLI and onnx_compare.py for execution.
"""
import json
import subprocess
import sys

import numpy as np
import onnx


def inputs_for(model, config, directory):
    """Load/generate validated feeds in ONNX graph-input order.

    Args:
        model: In-memory ModelProto; input shapes are specialized IN PLACE.
        config: Resolved manifest with inputs, input_files, or generation settings.
        directory: Base Path for relative input filenames.

    Returns:
        An insertion-ordered dict of graph input name to contiguous NumPy array.
        Initializers are excluded because their values are model weights.

    Raises:
        ValueError for missing/extra names or incompatible types/shapes. Loading
        errors propagate; supplied arrays are never silently reshaped or cast.
    """
    initializers = {v.name for v in model.graph.initializer}
    specs = [v for v in model.graph.input if v.name not in initializers]
    if set(config.get("input_shapes", {})) - {v.name for v in specs}:
        raise ValueError("input_shapes contains unknown input names")
    rng = np.random.default_rng(config.get("seed", 0))
    feeds = {}
    provided = None
    if "inputs" in config:
        filename = directory / config["inputs"]
        if filename.suffix == ".npy":
            # NPY stores one unnamed array. Only a single-input graph provides
            # an unambiguous name; multi-input cases must supply a mapping/NPZ.
            if len(specs) != 1:
                raise ValueError("A single NPY requires exactly one model input; use input_files for multiple inputs")
            provided = {specs[0].name: np.load(filename, allow_pickle=False)}
        else:
            with np.load(filename, allow_pickle=False) as archive:
                provided = {name: archive[name] for name in archive.files}
    elif "input_files" in config:
        provided = {name: np.load(directory / filename, allow_pickle=False)
                    for name, filename in config["input_files"].items()}
    if provided is not None:
        if set(provided) != {v.name for v in specs}:
            raise ValueError("Input keys must exactly match non-initializer input names")
    for spec in specs:
        # Preserve graph order: IREE's CLI receives positional inputs, whereas
        # ONNX Runtime accepts a name-to-array dictionary.
        tensor = spec.type.tensor_type
        if not spec.type.HasField("tensor_type") or not tensor.HasField("shape"):
            raise ValueError(f"{spec.name}: only tensor inputs with known rank are supported")
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type))
        if dtype.kind not in "fiub":
            raise ValueError(f"{spec.name}: unsupported input dtype {dtype}")
        declared = [d.dim_value if d.HasField("dim_value") else None for d in tensor.shape.dim]
        if provided is not None:
            value = provided[spec.name]
        else:
            shape = config.get("input_shapes", {}).get(spec.name, declared)
            if any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape):
                raise ValueError(f"{spec.name}: supply concrete input_shapes or an inputs NPZ")
            if np.prod(shape, dtype=object) > 10_000_000:
                raise ValueError(f"{spec.name}: generated input exceeds 10 million elements; supply NPZ")
            if dtype.kind == "f":
                value = rng.uniform(-1, 1, size=shape).astype(dtype)
            else:
                value = rng.integers(0, 2, size=shape).astype(dtype)
        if value.dtype != dtype or value.ndim != len(declared):
            raise ValueError(f"{spec.name}: input dtype/rank does not match model")
        if any(d is not None and d != actual for d, actual in zip(declared, value.shape)):
            raise ValueError(f"{spec.name}: input shape does not match fixed model dimensions")
        feeds[spec.name] = np.array(value, copy=True, order="C")
        # Specialize only the in-memory copy. Preparation saves that copy so
        # both backends see the same concrete shapes; source files are untouched.
        del tensor.shape.dim[:]
        for size in value.shape:
            tensor.shape.dim.add().dim_value = size
    return feeds


def command(argv, log, timeout, cwd=None):
    """Run an argument vector without a shell and retain its combined output.

    Nonzero exits and timeouts propagate to the caller as case failures. Flush
    the command first so its invocation survives even if the subprocess hangs.
    """
    log.write("$ " + repr([str(v) for v in argv]) + "\n")
    log.flush()
    subprocess.run([str(v) for v in argv], stdout=log, stderr=log,
                   check=True, timeout=timeout, cwd=cwd)


def case_config(path):
    """Read <model-stem>.json and resolve one explicit or detected input source.

    Defaults include all run_on suites. With no explicit input setting, detect
    generate_input.py, input.py, inputs.npz, or input.npy beside the model.
    Ambiguity is an error rather than an implicit precedence rule. This function
    reads configuration only: excluded cases need not load weights/run scripts.
    """
    sidecar = path.with_suffix(".json")
    config = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    unknown = set(config) - {"seed", "inputs", "input_files", "input_script", "input_shapes",
                             "rtol", "atol", "run_on", "static", "prepare_script", "vlen"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    tiers = config.get("run_on", ["pr", "nightly", "release"])
    if not isinstance(tiers, list) or not tiers or any(
        t not in ("pr", "nightly", "release") for t in tiers
    ):
        raise ValueError("run_on must be a nonempty list of pr, nightly, release")
    if sum(key in config for key in ("inputs", "input_files", "input_script")) > 1:
        raise ValueError("Choose only one of inputs, input_files or input_script")
    if not any(key in config for key in ("inputs", "input_files", "input_script")):
        # Listing order is not priority: more than one candidate must fail.
        candidates = [("input_script", "generate_input.py"), ("input_script", "input.py"),
                      ("inputs", "inputs.npz"), ("inputs", "input.npy")]
        found = [(key, filename) for key, filename in candidates if (path.parent / filename).is_file()]
        if len(found) > 1:
            raise ValueError("Multiple input sources found; select one explicitly in the manifest")
        if found:
            key, filename = found[0]
            config[key] = filename
    return config


def case_inputs(model, config, path, work, timeout):
    """Apply static-shape policy, optionally run a generator, and validate feeds.

    path is the source model Path; work is an existing per-run directory.
    Generators receive --output, --model-dir and --seed, run in the case
    directory, and must create the requested named NPZ. input.log captures their
    diagnostics. A generator failure never falls back to random test inputs.
    """
    if not isinstance(config.get("static", True), bool):
        raise ValueError("static must be a JSON boolean")
    if config.get("static", True):
        initializers = {v.name for v in model.graph.initializer}
        for spec in model.graph.input:
            if spec.name in initializers:
                continue
            tensor = spec.type.tensor_type
            if not tensor.HasField("shape") or any(
                not dim.HasField("dim_value") for dim in tensor.shape.dim
            ):
                raise ValueError(f"{spec.name}: static case requires fixed ONNX input shapes")
    config = dict(config)
    # Avoid replacing the caller's manifest with the generated artifact path.
    if "input_script" in config:
        script = (path.parent / config["input_script"]).resolve()
        if not script.is_file():
            raise ValueError(f"Input script not found: {script}")
        generated = work / "generated-inputs.npz"
        with (work / "input.log").open("w") as log:
            command([sys.executable, script, "--output", generated,
                     "--model-dir", path.parent, "--seed", str(config.get("seed", 0))],
                    log, timeout, cwd=path.parent)
        if not generated.is_file():
            raise ValueError("Input script did not write the requested NPZ file")
        config["inputs"] = str(generated)
    return inputs_for(model, config, path.parent)
