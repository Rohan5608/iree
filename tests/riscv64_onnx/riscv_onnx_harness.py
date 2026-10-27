# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for running ONNX models on RISC-V 64 (RVV) under QEMU user-mode.

For a given .onnx file the harness:

  1. Runs the model with ONNX Runtime on the host to get reference outputs.
  2. Imports the model with `iree-import-onnx` and compiles it with
     `iree-compile` for riscv64 with the vector extension enabled.
  3. Runs the compiled module with a cross-compiled `iree-run-module` under
     `qemu-riscv64` using the exact same inputs.
  4. Compares the IREE outputs against the ONNX Runtime outputs.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Keep in sync with build_tools/cmake/linux_riscv64.cmake
# (RISCV64_TEST_DEFAULT_ABI_FLAGS / RISCV64_TEST_DEFAULT_FEAT_FLAGS and
# RISCV_QEMU_CPU_FLAGS). The CPU features request RVV with a 512-bit VLEN and
# the QEMU CPU model provides a matching vector unit.
DEFAULT_COMPILE_FLAGS = [
    "--iree-hal-target-device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
    "--iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu",
    "--iree-llvmcpu-target-abi=lp64d",
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl512b,+v",
]
DEFAULT_QEMU_CPU = "rv64,Zve64d=true,vlen=512,elen=64,vext_spec=v1.0"

# Maps ONNX Runtime type strings to numpy dtypes.
_ORT_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


class HarnessError(Exception):
    """A step of the pipeline failed (as opposed to a numerical mismatch)."""


@dataclass
class ModelConfig:
    """Optional per-model settings, read from `<model>.onnx.json`.

    All fields are optional in the JSON file:
      rtol, atol:      Comparison tolerances (default: command line values).
      seed:            RNG seed for input generation.
      input_dims:      {"symbolic_dim_name": size} for dynamic dimensions.
                       Dynamic dimensions without an entry default to 1.
      int_range:       [low, high) range for generated integer inputs.
      float_range:     [low, high) range for generated floating point inputs.
      extra_compile_flags: Extra iree-compile flags for this model.
      skip:            Reason string; the model is skipped.
      xfail:           Reason string; the model is expected to fail.
    """

    rtol: float | None = None
    atol: float | None = None
    seed: int | None = None
    input_dims: dict[str, int] = field(default_factory=dict)
    int_range: tuple[int, int] = (0, 4)
    float_range: tuple[float, float] = (-1.0, 1.0)
    extra_compile_flags: list[str] = field(default_factory=list)
    skip: str | None = None
    xfail: str | None = None

    @staticmethod
    def load(model_path: Path) -> "ModelConfig":
        config_path = model_path.with_name(model_path.name + ".json")
        if not config_path.exists():
            return ModelConfig()
        raw = json.loads(config_path.read_text())
        unknown = set(raw) - set(ModelConfig.__dataclass_fields__)
        if unknown:
            raise HarnessError(f"{config_path}: unknown keys {sorted(unknown)}")
        if "int_range" in raw:
            raw["int_range"] = tuple(raw["int_range"])
        if "float_range" in raw:
            raw["float_range"] = tuple(raw["float_range"])
        return ModelConfig(**raw)


@dataclass
class Tools:
    """Locations of the tools used by the pipeline."""

    iree_import_onnx: list[str]
    iree_compile: list[str]
    # Cross-compiled (riscv64) iree-run-module binary.
    iree_run_module: str
    qemu_bin: str
    qemu_cpu: str = DEFAULT_QEMU_CPU
    # Sysroot used by qemu-user to find the dynamic loader and libc. Not
    # needed for statically linked binaries.
    qemu_sysroot: str | None = None
    device: str = "local-sync"
    compile_flags: list[str] = field(
        default_factory=lambda: list(DEFAULT_COMPILE_FLAGS)
    )
    extra_compile_flags: list[str] = field(default_factory=list)
    timeout: int = 900


def _run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int) -> str:
    """Runs |cmd|, returning combined output and raising HarnessError on failure."""
    printable = " ".join(shlex.quote(c) for c in cmd)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise HarnessError(f"Timed out after {timeout}s:\n  {printable}") from e
    except OSError as e:
        raise HarnessError(f"Failed to launch:\n  {printable}\n{e}") from e
    if proc.returncode != 0:
        raise HarnessError(
            f"Command failed with exit code {proc.returncode}:\n"
            f"  {printable}\n{proc.stdout}"
        )
    return proc.stdout


def _resolve_shape(shape: list[Any], dims: dict[str, int]) -> tuple[int, ...]:
    resolved = []
    for dim in shape:
        if isinstance(dim, int) and dim >= 0:
            resolved.append(dim)
        else:
            # Dynamic dimension: use the sidecar override or default to 1.
            resolved.append(dims.get(dim, 1) if isinstance(dim, str) else 1)
    return tuple(resolved)


def generate_inputs(session, config: ModelConfig, seed: int) -> list[np.ndarray]:
    """Generates deterministic random inputs for each ONNX graph input."""
    rng = np.random.default_rng(config.seed if config.seed is not None else seed)
    inputs = []
    for node in session.get_inputs():
        dtype = _ORT_TYPE_TO_NUMPY.get(node.type)
        if dtype is None:
            raise HarnessError(
                f"Input '{node.name}' has unsupported type '{node.type}'"
            )
        shape = _resolve_shape(node.shape, config.input_dims)
        if dtype == np.bool_:
            value = rng.integers(0, 2, size=shape).astype(np.bool_)
        elif np.issubdtype(dtype, np.integer):
            low, high = config.int_range
            value = rng.integers(low, high, size=shape).astype(dtype)
        else:
            low, high = config.float_range
            value = rng.uniform(low, high, size=shape).astype(dtype)
        inputs.append(value)
    return inputs


def run_onnxruntime(inputs: list[np.ndarray], session) -> list[np.ndarray]:
    feeds = {node.name: arr for node, arr in zip(session.get_inputs(), inputs)}
    return [np.asarray(o) for o in session.run(None, feeds)]


def import_and_compile(
    model_path: Path, work_dir: Path, tools: Tools, config: ModelConfig
) -> tuple[Path, str]:
    """Imports and compiles the model. Returns (vmfb path, entry function)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = work_dir / "model.mlir"
    vmfb_path = work_dir / "model.vmfb"

    _run(
        tools.iree_import_onnx + [str(model_path), "-o", str(mlir_path)],
        timeout=tools.timeout,
    )
    # iree-import-onnx emits one public function named after the ONNX graph.
    match = re.search(r"func\.func\s+@([\w$.\-]+)\s*\(", mlir_path.read_text())
    if not match:
        raise HarnessError(f"No entry function found in {mlir_path}")
    entry = match.group(1)

    _run(
        tools.iree_compile
        + [str(mlir_path), "-o", str(vmfb_path)]
        + tools.compile_flags
        + tools.extra_compile_flags
        + config.extra_compile_flags,
        timeout=tools.timeout,
    )
    return vmfb_path, entry


def run_on_qemu(
    vmfb_path: Path,
    entry: str,
    inputs: list[np.ndarray],
    num_outputs: int,
    work_dir: Path,
    tools: Tools,
) -> list[np.ndarray]:
    """Runs |vmfb_path| with iree-run-module under QEMU user-mode emulation."""
    input_args = []
    for i, arr in enumerate(inputs):
        path = work_dir / f"input_{i}.npy"
        np.save(path, arr)
        input_args.append(f"--input=@{path}")
    output_paths = [work_dir / f"output_{i}.npy" for i in range(num_outputs)]
    for p in output_paths:
        p.unlink(missing_ok=True)
    output_args = [f"--output=@{p}" for p in output_paths]

    cmd = [tools.qemu_bin, "-cpu", tools.qemu_cpu]
    if tools.qemu_sysroot:
        cmd += ["-L", tools.qemu_sysroot]
    cmd += [
        tools.iree_run_module,
        f"--device={tools.device}",
        f"--module={vmfb_path}",
        f"--function={entry}",
    ]
    cmd += input_args + output_args
    _run(cmd, timeout=tools.timeout)

    results = []
    for p in output_paths:
        if not p.exists():
            raise HarnessError(f"iree-run-module did not write {p}")
        results.append(np.load(p))
    return results


def compare_outputs(
    expected: list[np.ndarray],
    actual: list[np.ndarray],
    rtol: float,
    atol: float,
) -> list[str]:
    """Returns a list of mismatch descriptions (empty if outputs match)."""
    problems = []
    if len(expected) != len(actual):
        return [f"Output count mismatch: expected {len(expected)}, got {len(actual)}"]
    for i, (exp, act) in enumerate(zip(expected, actual)):
        if exp.shape != act.shape:
            problems.append(f"output {i}: shape mismatch {exp.shape} vs {act.shape}")
            continue
        if np.issubdtype(exp.dtype, np.floating) or np.issubdtype(
            act.dtype, np.floating
        ):
            exp_f = exp.astype(np.float64)
            act_f = act.astype(np.float64)
            ok = np.isclose(act_f, exp_f, rtol=rtol, atol=atol, equal_nan=True)
            if not ok.all():
                diff = np.abs(act_f - exp_f)
                diff[ok] = 0
                worst = np.unravel_index(np.argmax(diff), diff.shape)
                problems.append(
                    f"output {i}: {(~ok).sum()}/{ok.size} elements differ "
                    f"(rtol={rtol}, atol={atol}); worst at {worst}: "
                    f"onnxruntime={exp_f[worst]!r} iree={act_f[worst]!r}"
                )
        else:
            if not np.array_equal(exp, act.astype(exp.dtype)):
                n = int((exp != act.astype(exp.dtype)).sum())
                problems.append(f"output {i}: {n}/{exp.size} elements differ")
    return problems
