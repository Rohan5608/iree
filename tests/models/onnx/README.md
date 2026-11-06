# ONNX models for RISC-V comparison

Every `.onnx` file below this directory is discovered automatically. Organize
each static model in a folder with its input generator and data:

```text
matmul/
  model.onnx       # static x[2,3] @ weights[3,4] -> y[2,4]
  model.json       # input script, run selection, tolerances
  input.py         # writes named x array to requested NPZ
  weights.bin      # external ONNX initializer, not a runtime input
  data/input.csv   # fixture used by input.py
relu/
  model.onnx       # static x[2,4] -> relu(x)[2,4]
  model.json       # supplies custom prepare_script
  prepare.py       # standalone hook that delegates to the shared helper
  input.py         # loads an existing .npy array
  data/input.npy
matmul_nightly/
  model.onnx       # static x[32,64] @ weights[64,32] -> y[32,32]
  model.json       # run_on: [nightly, release]
  input.py         # seeded input generator
  weights.bin
```

All examples use opset 18 and ONNX IR version 10. The nightly example is larger
than the basic case but is still a small scheduling demonstration, not a
performance benchmark. Input shapes are static by default. Dynamic inputs are
rejected unless explicitly opting into the legacy `"static": false` behavior.

`input.py` receives `--output`, `--model-dir`, and `--seed`; it must write a
non-pickled NPZ with keys matching the model's input names. An automatically
discovered model needs no changes to the workflow or a central test list.
Models default to all three suites; `run_on` can restrict costly cases. PR and
main-push runs select `pr`, scheduled runs select `nightly`, and stable GitHub
release publication selects `release`. Excluded cases are reported as skipped.

ONNX models default to VLEN=1024 bits. A manifest `"vlen": 512` changes one
case; `IREE_ONNX_VLEN=512` overrides the selected collection. Compiler and QEMU
settings change together. The fixed custom smoke/core tests retain their
separate 512-bit configuration.

Preparation can be run without the runner:

```sh
python tests/models/onnx/relu/prepare.py \
  --model tests/models/onnx/relu/model.onnx --output-dir build-prepared/relu
```

This writes `model.onnx`, `inputs.npz`, and a reusable `model.json`. MatMul uses
the standard `build_tools/riscv/prepare_onnx.py` helper directly. See the guide
for the custom script contract and local commands.

Run a selection after the toolchain and host/target builds are ready:

```sh
IREE_ONNX_SUITE=pr bash build_tools/riscv/ci_rvv.sh onnx
IREE_ONNX_SUITE=nightly bash build_tools/riscv/ci_rvv.sh onnx
IREE_ONNX_SUITE=release bash build_tools/riscv/ci_rvv.sh onnx
```

See the [full model testing guide](../../../docs/website/docs/developers/building/onnx-model-tests.md)
for commands, sidecar examples, output reports and supported types.
