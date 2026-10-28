# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""pytest configuration: discovers ONNX models and resolves tool locations."""

import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import riscv_onnx_harness as harness  # noqa: E402

THIS_DIR = Path(__file__).parent
DEFAULT_MODELS_DIR = THIS_DIR / "models"


def _env(name: str, default=None):
    value = os.environ.get(name)
    return value if value else default


def pytest_addoption(parser):
    g = parser.getgroup("riscv64-onnx")
    g.addoption(
        "--models-dir",
        default=_env("ONNX_MODELS_DIR", str(DEFAULT_MODELS_DIR)),
        help="Directory searched recursively for *.onnx files "
        "(env: ONNX_MODELS_DIR).",
    )
    g.addoption(
        "--work-dir",
        default=_env("ONNX_WORK_DIR"),
        help="Where to keep intermediate files (MLIR, vmfb, inputs, outputs). "
        "Defaults to a pytest temp dir (env: ONNX_WORK_DIR).",
    )
    g.addoption(
        "--iree-run-module",
        default=_env("IREE_RUN_MODULE", "build-riscv/tools/iree-run-module"),
        help="Cross-compiled riscv64 iree-run-module (env: IREE_RUN_MODULE).",
    )
    g.addoption(
        "--iree-compile",
        default=_env("IREE_COMPILE"),
        help="iree-compile executable; defaults to PATH (env: IREE_COMPILE).",
    )
    g.addoption(
        "--iree-import-onnx",
        default=_env("IREE_IMPORT_ONNX"),
        help="iree-import-onnx executable; defaults to PATH "
        "(env: IREE_IMPORT_ONNX).",
    )
    g.addoption(
        "--qemu-bin",
        default=_env("QEMU_BIN", "qemu-riscv64"),
        help="qemu-riscv64 user-mode emulator (env: QEMU_BIN).",
    )
    g.addoption(
        "--qemu-cpu",
        default=_env("QEMU_CPU_FLAGS", harness.DEFAULT_QEMU_CPU),
        help="QEMU -cpu value; must provide RVV with the VLEN the model was "
        "compiled for (env: QEMU_CPU_FLAGS).",
    )
    g.addoption(
        "--qemu-sysroot",
        default=_env("QEMU_LD_PREFIX"),
        help="Sysroot passed to qemu as -L, for dynamically linked "
        "iree-run-module binaries (env: QEMU_LD_PREFIX).",
    )
    g.addoption(
        "--device",
        default=_env("IREE_RISCV_DEVICE", "local-sync"),
        help="IREE runtime device (env: IREE_RISCV_DEVICE).",
    )
    g.addoption(
        "--extra-compile-flags",
        default=_env("IREE_EXTRA_COMPILE_FLAGS", ""),
        help="Extra iree-compile flags, shell-quoted "
        "(env: IREE_EXTRA_COMPILE_FLAGS).",
    )
    g.addoption(
        "--override-compile-flags",
        default=_env("IREE_OVERRIDE_COMPILE_FLAGS"),
        help="Replace the default riscv64+RVV iree-compile flags entirely, "
        "shell-quoted (env: IREE_OVERRIDE_COMPILE_FLAGS).",
    )
    g.addoption(
        "--vlens",
        default=_env("ONNX_VLENS", "512"),
        help="Comma-separated RVV vector register widths in bits to test, e.g. "
        "'128,256,512,1024'. Each model is compiled for (+zvl<N>b) and run on "
        "a QEMU CPU with exactly that VLEN (env: ONNX_VLENS).",
    )
    g.addoption("--rtol", type=float, default=1e-4, help="Relative tolerance.")
    g.addoption("--atol", type=float, default=1e-5, help="Absolute tolerance.")
    g.addoption("--seed", type=int, default=0, help="Input generation seed.")
    g.addoption(
        "--run-timeout",
        type=int,
        default=900,
        help="Per-command timeout in seconds (import/compile/run).",
    )


def _parse_vlens(text):
    try:
        vlens = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        vlens = []
    # RVV requires a power of two VLEN; QEMU supports up to 1024 in older
    # versions and more in newer ones.
    if not vlens or any(v < 32 or v & (v - 1) for v in vlens):
        pytest.exit(f"--vlens must be powers of two >= 32 (got '{text}')", returncode=2)
    return list(dict.fromkeys(vlens))


def pytest_generate_tests(metafunc):
    if "model_path" not in metafunc.fixturenames:
        return
    if "vlen" in metafunc.fixturenames:
        vlens = _parse_vlens(metafunc.config.getoption("vlens"))
        metafunc.parametrize("vlen", vlens, ids=[f"vlen{v}" for v in vlens])
    models_dir = Path(metafunc.config.getoption("models_dir")).resolve()
    models = sorted(models_dir.rglob("*.onnx"))
    ids = [m.relative_to(models_dir).as_posix() for m in models]
    # An empty parametrization makes pytest skip the test, which would let a
    # misconfigured CI job pass silently. Fail loudly instead.
    if not models:
        pytest.exit(f"No .onnx files found under {models_dir}", returncode=2)
    metafunc.parametrize("model_path", models, ids=ids)


def _find_tool(configured, name):
    if configured:
        return configured
    found = shutil.which(name)
    if not found:
        pytest.exit(
            f"'{name}' not found in PATH; pass --{name} or set its env var",
            returncode=2,
        )
    return found


@pytest.fixture(scope="session")
def tools(pytestconfig):
    opt = pytestconfig.getoption

    run_module = Path(opt("iree_run_module"))
    if not run_module.exists():
        pytest.exit(
            f"riscv64 iree-run-module not found at '{run_module}'. Build it with "
            "build_tools/riscv/build_run_module.sh or pass --iree-run-module.",
            returncode=2,
        )
    qemu = shutil.which(opt("qemu_bin")) or opt("qemu_bin")
    if not Path(qemu).exists():
        pytest.exit(f"QEMU not found: '{opt('qemu_bin')}'", returncode=2)

    import_onnx = _find_tool(opt("iree_import_onnx"), "iree-import-onnx")
    compile_ = _find_tool(opt("iree_compile"), "iree-compile")

    compile_flags = list(harness.DEFAULT_COMPILE_FLAGS)
    if opt("override_compile_flags"):
        compile_flags = shlex.split(opt("override_compile_flags"))
    return harness.Tools(
        iree_import_onnx=[import_onnx],
        iree_compile=[compile_],
        iree_run_module=str(run_module.resolve()),
        qemu_bin=qemu,
        qemu_cpu=opt("qemu_cpu"),
        qemu_sysroot=opt("qemu_sysroot"),
        device=opt("device"),
        compile_flags=compile_flags,
        extra_compile_flags=shlex.split(opt("extra_compile_flags")),
        timeout=opt("run_timeout"),
    )


@pytest.fixture
def work_dir(pytestconfig, tmp_path, request):
    configured = pytestconfig.getoption("work_dir")
    if not configured:
        return tmp_path
    safe = request.node.callspec.id.replace("/", "__")
    path = Path(configured) / safe
    path.mkdir(parents=True, exist_ok=True)
    return path
