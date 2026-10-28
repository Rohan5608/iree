# RISC-V 64 + RVV ONNX tests

Drop an `.onnx` file into [`models/`](models) and it becomes a test case. For
every model the test:

1. runs it with **ONNX Runtime** on the host (reference outputs),
2. imports it with `iree-import-onnx` and compiles it with `iree-compile` for
   `riscv64` with the vector extension (`+v,+zvl512b`),
3. runs the module with a cross-compiled `iree-run-module` under
   **`qemu-riscv64`** (a QEMU CPU with RVV, VLEN=512) using the *same* inputs,
4. compares every output against ONNX Runtime (`rtol=1e-4`, `atol=1e-5` by
   default; integer and bool outputs must match exactly).

Every model is tested at each vector register width (VLEN) in `--vlens`
(env `ONNX_VLENS`, default `512`; CI uses `128,256,512,1024`). For a width N the
model is compiled with `+zvlNb` and run on a QEMU CPU with `vlen=N`, so the code
is both generated and executed for that width. Test ids look like
`vlen256-mlp_softmax.onnx`; select with e.g. `-k vlen256`. Compiling for one
width and running on a different one is not a valid test: code for a larger
VLEN than the CPU has runs without error but gives wrong results.

Models are found recursively, so subdirectories are fine. Inputs are random
but deterministic (seeded), generated from the model's input signature.

The compile flags and QEMU CPU mirror `build_tools/cmake/linux_riscv64.cmake`
(`RISCV64_TEST_DEFAULT_*_FLAGS`, `RISCV_QEMU_CPU_FLAGS`) so this suite and the
existing CTest-based RISC-V suite exercise the same configuration.

## CI

`.github/workflows/pkgci_test_riscv64_onnx.yml`, called from `pkgci.yml` as
`test_riscv64_onnx`. It uses the compiler from the PkgCI packages, the same
RISC-V clang toolchain/sysroot and QEMU as `pkgci_test_riscv64.yml`, builds only
`iree-run-module` for riscv64, then runs `build_tools/riscv/run_onnx_tests.sh`.
On failure the JUnit report and all intermediates (MLIR, vmfb, `.npy`
inputs/outputs) are uploaded as the `riscv64_onnx_failures` artifact.

Use the `ci-skip`, `ci-exactly` and `ci-extra` PR trailers described in
`docs/website/docs/developers/general/github-actions.md` with the job name `test_riscv64_onnx` to
control it on a PR.

## Running locally

```bash
# 1. Host tools: compiler (iree-compile, iree-import-onnx) + test requirements.
python3 -m venv .venv && source .venv/bin/activate
pip install iree-base-compiler iree-base-runtime   # or build_tools/pkgci/setup_venv.py
pip install -r tests/riscv64_onnx/requirements.txt

# 2. riscv64 iree-run-module. RISCV_TOOLCHAIN_ROOT/QEMU come from
#    build_tools/riscv/riscv_bootstrap.sh.
export RISCV_TOOLCHAIN_ROOT=$HOME/riscv/toolchain/clang/linux/RISCV
export IREE_HOST_BIN_DIR=$PWD/.venv/bin   # provides iree-flatcc-cli etc.
./build_tools/riscv/build_run_module.sh   # -> build-riscv/tools/iree-run-module

# 3. Run all models (or select with -k).
export QEMU_BIN=$HOME/riscv/qemu/linux/RISCV/bin/qemu-riscv64
./build_tools/riscv/run_onnx_tests.sh build-riscv -k mlp_softmax
```

Useful options (flag or environment variable; see `conftest.py`):

| Option | Env | Purpose |
| --- | --- | --- |
| `--models-dir` | `ONNX_MODELS_DIR` | Use another directory of models |
| `--work-dir` | `ONNX_WORK_DIR` | Keep MLIR/vmfb/npy intermediates |
| `--vlens` | `ONNX_VLENS` | Comma-separated VLENs in bits, e.g. `128,256,512,1024` |
| `--rtol`, `--atol`, `--seed` | | Tolerances / input seed |
| `--extra-compile-flags` | `IREE_EXTRA_COMPILE_FLAGS` | Append `iree-compile` flags |
| `--override-compile-flags` | `IREE_OVERRIDE_COMPILE_FLAGS` | Replace the riscv64+RVV flag set (e.g. another VLEN; also change `--qemu-cpu`) |
| `--qemu-cpu` | `QEMU_CPU_FLAGS` | QEMU `-cpu` value |
| `--qemu-sysroot` | `QEMU_LD_PREFIX` | `-L` sysroot for dynamically linked binaries |
| `--device` | `IREE_RISCV_DEVICE` | IREE device (default `local-sync`) |

## Per-model settings

An optional `<model>.onnx.json` next to the model (all keys optional):

```json
{
  "rtol": 1e-3,
  "atol": 1e-4,
  "seed": 7,
  "input_dims": {"batch": 2, "seq": 16},
  "int_range": [0, 100],
  "float_range": [-1.0, 1.0],
  "extra_compile_flags": ["--iree-opt-level=O3"],
  "skip": "reason, to skip the model",
  "xfail": "reason, to mark a known failure"
}
```

* `input_dims` assigns sizes to symbolic/dynamic input dimensions (default 1).
* `int_range` bounds generated integer inputs (e.g. token ids or gather
  indices); the default is `[0, 4)`.
* The `skip` and `xfail` keys are the way to quarantine a model that is known
  not to work yet, without deleting it.

## Limitations

* Only tensor inputs of numeric/bool types are supported (no sequences, maps or
  strings).
* Models are imported with `iree-import-onnx` defaults. Weights are embedded in
  the MLIR, so very large models (multi-GB) are not a good fit for QEMU anyway.
* QEMU user-mode is slow; keep models small or raise `--run-timeout`.
