#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Discover ONNX cases, prepare once, and compare ORT with IREE execution.

Use --model-dir for one case or --models-dir for a recursive collection.
Required tools: --compiler is a HOST executable; --runner is the TARGET runtime.
RISC-V execution also requires --qemu and --sysroot. --native instead runs a
host runtime for diagnostics and does not establish RISC-V coverage.

Each invocation writes fresh per-model artifacts plus results.json and
onnx-results.xml under --output-dir. Returns 1 for any case failure or an empty
selection; suite-excluded cases are recorded as skipped. Default ONNX VLEN is
1024 bits, overridable by manifest vlen or the higher-priority --vlen option.
See docs/website/docs/developers/building/onnx-model-tests.md for full commands.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import onnx

from onnx_case import case_config, case_inputs, command


def compare(expected, actual, rtol, atol):
    """Require identical shape/dtype, then compare values against ORT outputs.

    Floats use abs(actual-reference) <= atol + rtol*abs(reference); NaN/Inf
    fail even on both sides. Integer/bool outputs require exact equality.
    Raises AssertionError so the outer runner can report the failing case.
    """
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


def vector_length(config, override=None):
    """Resolve VLEN in bits: CLI override > manifest vlen > 1024.

    Supported widths match the intended QEMU profiles. Reject booleans/floats
    as well as unsupported sizes so JSON typing mistakes cannot change targets.
    """
    value = override if override is not None else config.get("vlen", 1024)
    if isinstance(value, bool) or value not in (128, 256, 512, 1024):
        raise ValueError("vlen must be 128, 256, 512 or 1024 bits")
    if not isinstance(value, int):
        raise ValueError("vlen must be an integer number of bits")
    return value


def run_model(path, args, work, config):
    """Execute one selected case and return metadata only after all checks pass.

    path is the absolute source model Path; work is a fresh per-run directory.
    Preparation may be customized, but its model/inputs are validated and shared
    by both backends. Command errors, timeouts and comparisons propagate to main,
    which records a failure and continues with other selected cases.
    """
    # Lazy import keeps the standalone preparation helper independent of ORT.
    import onnxruntime as ort

    rtol, atol = float(config.get("rtol", 1e-4)), float(config.get("atol", 1e-5))
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    vlen = vector_length(config, args.vlen)
    default_prepare = Path(__file__).resolve().with_name("prepare_onnx.py")
    script = (path.parent / config["prepare_script"]).resolve() if "prepare_script" in config else default_prepare
    # The standard script handles every normal case. A custom hook must honor
    # the same --model/--output-dir contract and produce both required files.
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
    # Limit reference-side CPU concurrency; CI may execute additional suites.
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
        # Import on the host, then compile for the target. ONNX import requires
        # an IREE compiler built with the Torch/ONNX input plugin enabled.
        command([sys.executable, "-m", "iree.compiler.tools.import_onnx",
                 work / "model.onnx", "-o", work / "model.mlir"], log, args.timeout)
        flags = ["--iree-hal-target-device=local",
                 "--iree-hal-local-target-device-backends=llvm-cpu", "--iree-input-type=onnx"]
        if not args.native:
            # Pair the compiler's minimum vector width with QEMU's actual VLEN.
            # Changing just one side can produce an incompatible execution target.
            flags += ["--iree-llvmcpu-target-triple=riscv64", "--iree-llvmcpu-target-abi=lp64d",
                      f"--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl{vlen}b"]
        else:
            flags += ["--iree-llvmcpu-target-cpu=generic"]
        command([args.compiler, work / "model.mlir", *flags, "-o", work / "model.vmfb"],
                log, args.timeout)
        input_flags = []
        # NPY carries each tensor's shape/dtype across the command-line boundary.
        # Graph order was preserved when constructing feeds in onnx_case.py.
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
                # Compare every graph output on every selected driver, not merely
                # whether the process exited successfully or printed a tensor.
                try:
                    compare(reference, np.load(output, allow_pickle=False), rtol, atol)
                except AssertionError as exc:
                    raise AssertionError(f"{device}, output {i}: {exc}") from exc
    return {"rtol": rtol, "atol": atol, "outputs": len(expected),
            "vlen_bits": None if args.native else vlen, "prepare_script": str(script)}


def main(argv=None):
    """CLI entry point: select cases, collect failures/skips, and write reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", "--model-dir", dest="models_dir", type=Path, required=True,
                        help="One case directory or a root collection (recursively discovered)")
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
            # Filter before preparation: costly cases must not load weights or
            # execute input generators during PR runs when excluded by run_on.
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
            # Preserve useful results for the rest of the collection. Failure is
            # accumulated below and still causes a nonzero overall exit status.
            result.update(status="failed", error=str(exc))
            ET.SubElement(case, "failure", message=str(exc)).text = str(exc)
        case.set("time", str(time.monotonic() - start))
        results.append(result)
        print(f"{result['status'].upper()}: {name}", flush=True)
    if not models or (not executed and not any(v["status"] == "failed" for v in results)):
        # An all-skipped collection must not masquerade as passing coverage.
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
