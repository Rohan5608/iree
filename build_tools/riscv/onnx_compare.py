#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare discovered ONNX models against ONNX Runtime using identical inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import onnx


def compare(expected, actual, rtol, atol):
    if expected.dtype.kind not in "fiub":
        raise AssertionError(f"Unsupported output dtype: {expected.dtype}")
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise AssertionError(
            f"Output metadata mismatch: expected {expected.shape}/{expected.dtype}, "
            f"got {actual.shape}/{actual.dtype}"
        )
    if np.issubdtype(expected.dtype, np.floating):
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise AssertionError("Non-finite output; NaN/Inf comparisons are not accepted")
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(actual, expected)


def inputs_for(model, config, directory):
    initializers = {v.name for v in model.graph.initializer}
    specs = [v for v in model.graph.input if v.name not in initializers]
    if set(config.get("input_shapes", {})) - {v.name for v in specs}:
        raise ValueError("input_shapes contains unknown input names")
    rng = np.random.default_rng(config.get("seed", 0))
    feeds = {}
    provided = None
    if "inputs" in config:
        with np.load(directory / config["inputs"], allow_pickle=False) as archive:
            provided = {name: archive[name] for name in archive.files}
        if set(provided) != {v.name for v in specs}:
            raise ValueError("NPZ keys must exactly match non-initializer input names")
    for spec in specs:
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
        # Specialize a copy for import. ORT still runs the original model.
        del tensor.shape.dim[:]
        for size in value.shape:
            tensor.shape.dim.add().dim_value = size
    return feeds


def command(argv, log, timeout, cwd=None):
    log.write("$ " + repr([str(v) for v in argv]) + "\n")
    log.flush()
    subprocess.run([str(v) for v in argv], stdout=log, stderr=log,
                   check=True, timeout=timeout, cwd=cwd)


def case_config(path):
    sidecar = path.with_suffix(".json")
    config = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    unknown = set(config) - {"seed", "inputs", "input_script", "input_shapes",
                             "rtol", "atol", "run_on", "static", "prepare_script", "vlen"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    tiers = config.get("run_on", ["pr", "nightly", "release"])
    if not isinstance(tiers, list) or not tiers or any(
        t not in ("pr", "nightly", "release") for t in tiers
    ):
        raise ValueError("run_on must be a nonempty list of pr, nightly, release")
    if "inputs" in config and "input_script" in config:
        raise ValueError("Choose inputs or input_script, not both")
    if "inputs" not in config and "input_script" not in config and (path.parent / "input.py").is_file():
        config["input_script"] = "input.py"
    return config


def case_inputs(model, config, path, work, timeout):
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


def vector_length(config, override=None):
    value = override if override is not None else config.get("vlen", 1024)
    if isinstance(value, bool) or value not in (128, 256, 512, 1024):
        raise ValueError("vlen must be 128, 256, 512 or 1024 bits")
    if not isinstance(value, int):
        raise ValueError("vlen must be an integer number of bits")
    return value


def run_model(path, args, work, config):
    import onnxruntime as ort

    rtol, atol = float(config.get("rtol", 1e-4)), float(config.get("atol", 1e-5))
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    vlen = vector_length(config, args.vlen)
    default_prepare = Path(__file__).resolve().with_name("prepare_onnx.py")
    script = (path.parent / config["prepare_script"]).resolve() if "prepare_script" in config else default_prepare
    with (work / "prepare.log").open("w") as log:
        command([sys.executable, script, "--model", path, "--output-dir", work],
                log, args.timeout, cwd=path.parent)
    prepared = work / "model.onnx"
    if not prepared.is_file() or not (work / "inputs.npz").is_file():
        raise ValueError("Preparation script must produce model.onnx and inputs.npz")
    model = onnx.load(prepared)
    onnx.checker.check_model(model)
    feeds = case_inputs(model, {"inputs": str(work / "inputs.npz"),
                               "static": config.get("static", True)}, prepared, work, args.timeout)
    for value in model.graph.output:
        if not value.type.HasField("tensor_type"):
            raise ValueError("Only tensor outputs are supported")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(prepared), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    expected = session.run(None, feeds)
    if not expected:
        raise ValueError("Model must produce at least one output")
    np.savez(work / "inputs.npz", **feeds)
    np.savez(work / "reference.npz", **{f"output_{i}": v for i, v in enumerate(expected)})
    with (work / "commands.log").open("w") as log:
        command([sys.executable, "-m", "iree.compiler.tools.import_onnx",
                 work / "model.onnx", "-o", work / "model.mlir"], log, args.timeout)
        flags = ["--iree-hal-target-device=local",
                 "--iree-hal-local-target-device-backends=llvm-cpu", "--iree-input-type=onnx"]
        if not args.native:
            flags += ["--iree-llvmcpu-target-triple=riscv64", "--iree-llvmcpu-target-abi=lp64d",
                      f"--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl{vlen}b"]
        else:
            flags += ["--iree-llvmcpu-target-cpu=generic"]
        command([args.compiler, work / "model.mlir", *flags, "-o", work / "model.vmfb"],
                log, args.timeout)
        input_flags = []
        for i, value in enumerate(feeds.values()):
            filename = work / f"input_{i}.npy"
            np.save(filename, value, allow_pickle=False)
            input_flags.append(f"--input=@{filename}")
        for device in args.devices:
            prefix = [] if args.native else [args.qemu, "-cpu",
                f"rv64,v=true,vlen={vlen},elen=64,vext_spec=v1.0", "-L", args.sysroot]
            outputs = [work / f"{device}_output_{i}.npy" for i in range(len(expected))]
            command([*prefix, args.runner, f"--device={device}",
                     f"--module={work / 'model.vmfb'}", "--function=main", *input_flags,
                     *[f"--output=@{v}" for v in outputs]], log, args.timeout)
            for i, (reference, output) in enumerate(zip(expected, outputs)):
                try:
                    compare(reference, np.load(output, allow_pickle=False), rtol, atol)
                except AssertionError as exc:
                    raise AssertionError(f"{device}, output {i}: {exc}") from exc
    return {"rtol": rtol, "atol": atol, "outputs": len(expected),
            "vlen_bits": None if args.native else vlen, "prepare_script": str(script)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--qemu")
    parser.add_argument("--sysroot")
    parser.add_argument("--native", action="store_true", help="Host validation only; not RISC-V coverage")
    parser.add_argument("--devices", nargs="+", choices=["local-sync", "local-task"],
                        default=["local-sync", "local-task"])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--vlen", type=int, choices=[128, 256, 512, 1024],
                        help="Override every selected model's VLEN (default: manifest or 1024 bits)")
    parser.add_argument("--suite", choices=["pr", "nightly", "release", "all"], default="pr",
                        help="Select model run_on tiers; all explicitly overrides the manifest filter")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.native and (not args.qemu or not args.sysroot):
        parser.error("RISC-V execution requires --qemu and --sysroot")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.is_relative_to(args.models_dir.resolve()):
        parser.error("--output-dir must be outside --models-dir")
    models = sorted(args.models_dir.resolve().rglob("*.onnx"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite = ET.Element("testsuite", name="onnx-reference-comparison", tests=str(len(models)))
    results = []
    executed = 0
    skipped = 0
    for model in models:
        name = str(model.relative_to(args.models_dir.resolve()))
        work = args.output_dir / (model.stem + "-" + hashlib.sha256(name.encode()).hexdigest()[:12])
        # Keep every invocation separate: stale output files can never satisfy a run.
        work.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="run-", dir=work))
        case = ET.SubElement(suite, "testcase", name=name, classname="onnx")
        start = time.monotonic()
        result = {"model": name, "work": str(work), "execution": "native" if args.native else "qemu-riscv64"}
        try:
            config = case_config(model)
            result["suite"] = args.suite
            if args.suite != "all" and args.suite not in config.get("run_on", ["pr", "nightly", "release"]):
                result.update(status="skipped", reason=f"Not selected for {args.suite}")
                ET.SubElement(case, "skipped", message=result["reason"])
                skipped += 1
                results.append(result)
                print(f"SKIPPED: {name} ({result['reason']})", flush=True)
                continue
            executed += 1
            result.update(run_model(model, args, work, config))
            result["status"] = "passed"
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            ET.SubElement(case, "failure", message=str(exc)).text = str(exc)
        case.set("time", str(time.monotonic() - start))
        results.append(result)
        print(f"{result['status'].upper()}: {name}", flush=True)
    if not models or (not executed and not any(v["status"] == "failed" for v in results)):
        ET.SubElement(ET.SubElement(suite, "testcase", name="discovery"),
                      "failure", message="No ONNX models selected")
        suite.set("tests", str(len(models) + 1))
        empty = 1
    else:
        empty = 0
    failures = sum(v["status"] == "failed" for v in results) + empty
    suite.set("failures", str(failures))
    suite.set("skipped", str(skipped))
    ET.ElementTree(suite).write(args.output_dir / "onnx-results.xml", encoding="unicode")
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
