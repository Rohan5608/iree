# Automatically discovered ONNX cases

Every `.onnx` below this directory becomes a case. Each uses the same common
preparation and comparison pipeline; no case-specific `prepare.py` is required.

```text
relu/
  model.onnx
  model.json
  input.npy
matmul/
  model.onnx
  model.json
  generate_input.py
  weights.bin
  data/input.csv
matmul_nightly/
  model.onnx
  model.json          # run_on: [nightly, release]
  generate_input.py
  weights.bin
```

ReLU uses static float32 `[2,4]` input. MatMul multiplies `[2,3]` input by
external `[3,4]` weights. The extended case uses `[32,64]` input and `[64,32]`
weights. These are opset 18 / IR 10 infrastructure tests, not model benchmarks.

Prepare a case without running inference:

```sh
python build_tools/riscv/prepare_onnx.py \
  --model-dir tests/models/onnx/relu --output-dir build-prepared/relu
```

Compare a case after the standard host/target tools have been built:

```sh
IREE_ONNX_MODELS_DIR="$PWD/tests/models/onnx/relu" \
  bash build_tools/riscv/ci_rvv.sh onnx
```

Input sources are automatically detected: `generate_input.py`, legacy `input.py`,
`inputs.npz`, or single-input `input.npy`. Multiple sources require an explicit
manifest choice. Use `input_files` to map multiple `.npy` files to input names.
Static input dimensions, shapes and dtypes are validated. External weights are
resolved through the ONNX file's references. Default ONNX VLEN is 1024 bits.

See the [full guide](../../../docs/website/docs/developers/building/onnx-model-tests.md)
for script contracts, named input mappings, tolerances, vector-width overrides,
PR/nightly/release filtering, local comparison commands and reports.
