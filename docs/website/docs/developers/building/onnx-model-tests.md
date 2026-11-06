# ONNX reference comparison on RISC-V

Both RISC-V workflows discover models recursively under `tests/models/onnx/`
and compare ONNX Runtime CPU outputs against IREE outputs under RISC-V QEMU.
The stage runs in both `smoke` and `standard` profiles. Add a static model case
folder and commit it; no workflow edit or per-model registration is needed.

```mermaid
flowchart TD
    A["Discover .onnx files and optional JSON configuration"] --> B["Generate seeded inputs or load NPZ inputs"]
    B --> C["ONNX Runtime CPU: prepared model and inputs"]
    B --> D["Specialize input shapes in a model copy"]
    D --> E["Python ONNX importer: model.mlir"]
    E --> F["Host iree-compile: RISC-V RVV model.vmfb"]
    F --> G["QEMU: target iree-run-module on both CPU drivers"]
    B --> G
    C --> H["Compare every output: shape, dtype and values"]
    G --> H
    H --> I["JUnit XML, JSON results, retained inputs and command logs"]
```

## Add a fixed-shape model

The recommended organization is one model per directory:

```text
tests/models/onnx/my_case/
  model.onnx
  input.py
  model.json          # optional
  data/               # optional input fixtures
  weights.bin         # optional external ONNX initializers
```

The runner recursively discovers `model.onnx` (or another `.onnx` filename).
Its sidecar always has the same stem: `model.json` or `my_name.json`. Fixed
ONNX input dimensions are required by default. The model is not rebuilt by
`input.py`; that script only prepares input arrays for the already exported model.

An adjacent `input.py` is automatically selected when neither `inputs` nor
`input_script` is specified. For a custom filename use
`"input_script": "make_inputs.py"`. The script is executed once per selected
model on the host, with the model directory as working directory:

```sh
python input.py --output /absolute/run/generated-inputs.npz \
  --model-dir /absolute/path/to/my_case --seed 0
```

A minimal script is:

```python
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
x = np.loadtxt(args.model_dir / "data/input.csv", delimiter=",", dtype=np.float32)
np.savez(args.output, x=x)  # x must match the ONNX graph input name
```

Both runtimes consume exactly these arrays; the script is not rerun separately
for each backend. A failed/timed-out script, missing NPZ, incorrect keys, dtype
or shape fails the case. Scripts should write generated files to the supplied
output location rather than modify the checked-in case folder. They run as CI
code with the job's permissions and should use only dependencies installed in
the CI environment (NumPy is available).

ONNX initializer weights are not input arrays. Store external weight files at
the relative locations encoded in `model.onnx`; `onnx.load` reads them from the
case directory. Preparation loads weights into a separate output model, which
both backends consume. Missing weights fail the case.
Additional data is used only when the input script or model explicitly loads it.

The supplied MatMul case uses `x[2,3]` from CSV and external `weights[3,4]`.
The ReLU case uses a fixed `[2,4]` CSV with negative, zero and positive values.

Place `my_model.onnx` in `tests/models/onnx/` or a subdirectory. Models with
fixed-shape numeric/bool tensor inputs need no sidecar file. Defaults are:

* Seed 0; reproducible floating inputs uniformly sampled from [-1, 1), and
  integer/bool inputs sampled from 0 and 1.
* CPU reference execution using ONNX Runtime 1.22.1.
* Float comparison with `rtol=1e-4` and `atol=1e-5`:
  `abs(actual - reference) <= atol + rtol * abs(reference)`.
* Exact integer/bool comparison; identical output shapes and dtypes required.
* Both `local-sync` and `local-task` under RVV 1.0 QEMU with VLEN=1024 by default.

These synthetic inputs are smoke coverage, not model accuracy evaluation.
For token IDs, masks, indices, positive-only inputs, or realistic inference,
supply meaningful test data. A fixed shape alone does not guarantee generated
inputs are valid for an operator's semantic constraints.

Flat model files remain supported. When no input generator or NPZ is provided,
the deterministic synthetic-input fallback described above is used.

## Standalone preparation and custom scripts

The runner delegates model preparation to `build_tools/riscv/prepare_onnx.py`.
Use the same script locally without discovery, compilation or QEMU:

```sh
python build_tools/riscv/prepare_onnx.py \
  --model tests/models/onnx/matmul/model.onnx --output-dir build-prepared/matmul
```

It resolves external `.data`/`.bin` weights, invokes the case's input script or
loads configured NPZ data, validates arrays, and writes `model.onnx`,
`inputs.npz`, and a reusable `model.json` pointing to those saved inputs. Keep
outputs outside the source case directory; source files remain unchanged.

For case-specific export or preprocessing, supply your own script:

```json
{
  "prepare_script": "prepare.py",
  "vlen": 1024
}
```

Its command-line contract is:

```sh
python prepare.py --model /absolute/case/model.onnx --output-dir /absolute/run
```

It must write `model.onnx` and `inputs.npz` into that output directory. The
prepared graph must export function `main`; the standard helper handles this
by setting the ONNX graph name. The runner validates both outputs and compares
the same prepared model and inputs across ONNX Runtime and IREE. A missing
output, script failure or timeout fails the case. A custom script is responsible
for preserving intended model semantics; backend comparison cannot establish
that arbitrary transformations inside the script are correct.

The checked-in ReLU hook delegates to the common helper and works standalone:

```sh
python tests/models/onnx/relu/prepare.py \
  --model tests/models/onnx/relu/model.onnx --output-dir build-prepared/relu
```

ReLU's `input.py` loads an existing `.npy` fixture with `np.load(...,
allow_pickle=False)` and writes the named NPZ. MatMul's script reads CSV. Thus
model-specific data loading stays in scripts, not the discovery runner. You can
also run either `input.py` alone using its `--output`, `--model-dir` and `--seed`
arguments to inspect the generated arrays.

## ONNX vector width: default 1024 bits

ONNX tests now default to VLEN=1024 bits (128 bytes), replacing their earlier
512-bit setting. The runner changes these settings together:

```text
Compiler: --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl1024b
QEMU:    -cpu rv64,v=true,vlen=1024,elen=64,vext_spec=v1.0
```

VLEN is the number of bits in one vector register. For LMUL=1, a 1024-bit
register can hold 32 float32 or 16 float64 elements. The active element count
also depends on VL, SEW and LMUL; this does not force every operation to use
32 lanes. `+zvl1024b` describes a minimum VLEN available to compiled code;
QEMU's `vlen=1024` supplies that exact simulated register width. ELEN remains
64 bits and describes the maximum element width, not the register width.

Choose 128, 256, 512 or 1024 bits. The precedence is:

1. Explicit runner `--vlen` / helper `IREE_ONNX_VLEN` override.
2. Per-model `"vlen"` in its manifest.
3. Default 1024.

For one model add `"vlen": 512` to its `model.json`. For a collection:

```sh
IREE_ONNX_VLEN=512 IREE_ONNX_SUITE=all bash build_tools/riscv/ci_rvv.sh onnx
```

The Python runner also accepts `--vlen 512`. Manual GitHub runs expose
`onnx_vlen`: `model` respects manifests/defaults, while a numeric choice
overrides every selected model. Results record the effective `vlen_bits` for
each executed ONNX case. Native diagnostic runs record null because no RISC-V
vector width is used.

This setting applies only to ONNX comparison. The pre-existing core/custom
RVV smoke profile remains 512 bits. QEMU execution checks functional behavior,
not hardware speed or exhaustive use of vector instructions. A model compiled
for a minimum VLEN of 1024 must not be deployed to hardware that only provides
512-bit registers; select a compatible profile and rebuild it first.

## PR, nightly and stable-release selection

Add a `run_on` list to the model's sidecar to control when it executes:

```json
{
  "static": true,
  "input_script": "input.py",
  "run_on": ["nightly", "release"],
  "seed": 123,
  "rtol": 0.0001,
  "atol": 0.00001
}
```

| Trigger / command | Selected suite |
| --- | --- |
| PR to main or push to main | `pr` |
| Scheduled source workflow (07:47 India Standard Time; cron 02:17 UTC) | `nightly` |
| Scheduled package workflow (08:47 India Standard Time; cron 03:17 UTC) | `nightly` |
| GitHub release `published`, with `prerelease=false` | `release` |
| Manual workflow run | `onnx_suite` choice: `pr`, `nightly`, `release`, or `all` |

Schedules run from the default branch and may start later than the nominal time.
Publishing a prerelease does not run the jobs. Publishing a stable release runs
against its tag; this validates existing release source and does not publish or
deploy a release. A tag push alone is not a release-publication event.

With no `run_on`, a model participates in `pr`, `nightly` and `release`. Use
`["nightly"]` for nightly-only, `["release"]` for release-only, or both as in
the `matmul_nightly` example. These are explicit memberships, not inherited
tiers. Manual `all` deliberately overrides the selection to exercise every case.
`profile` still controls the other compiler/runtime suites independently.

```sh
IREE_ONNX_SUITE=pr bash build_tools/riscv/ci_rvv.sh onnx
IREE_ONNX_SUITE=nightly bash build_tools/riscv/ci_rvv.sh onnx
IREE_ONNX_SUITE=release bash build_tools/riscv/ci_rvv.sh onnx
IREE_ONNX_SUITE=all bash build_tools/riscv/ci_rvv.sh onnx
```

The Python runner also accepts `--suite` directly. Excluded models are recorded
as skipped in JSON/JUnit, without loading their model/weights or executing their
input scripts. An invalid manifest still fails. An empty collection or selection
with no executed models fails so an accidentally misconfigured suite cannot pass.

## Dynamic shapes and custom tolerances

Static models are the default. The earlier dynamic-shape feature is retained
only as an explicit opt-in with `"static": false`; it is not used by the
checked-in case folders.

Add a sidecar with the same stem: `my_model.onnx` → `my_model.json`.

```json
{
  "static": false,
  "seed": 42,
  "input_shapes": {
    "input_ids": [1, 128],
    "attention_mask": [1, 128]
  },
  "rtol": 0.001,
  "atol": 0.0001
}
```

Input names must match the model. All generated shapes must be concrete,
nonnegative integer dimensions and match any fixed dimensions in the model.
The runner specializes a copy for import and leaves the original file unchanged.
Intermediate/output shapes are re-inferred for the selected input shapes.

Choose tolerances based on expected numerical behavior. Larger tolerances can
hide compiler errors; they are not a correction for unsupported operations.

## Supply exact inputs

Create an NPZ containing one array per non-initializer graph input:

```python
import numpy as np

np.savez(
    "tests/models/onnx/my_model_inputs.npz",
    input_ids=np.array([[1, 7, 3, 0]], dtype=np.int64),
    attention_mask=np.array([[1, 1, 1, 0]], dtype=np.int64),
)
```

Reference it from `my_model.json`:

```json
{
  "inputs": "my_model_inputs.npz",
  "rtol": 0.0001,
  "atol": 0.00001
}
```

The NPZ path is relative to the model directory. Keys must exactly match the
model's non-initializer inputs; array dtypes, ranks and fixed dimensions are
validated without implicit casting. NPZ arrays provide concrete shapes for
dynamic inputs. Pickled/object arrays are not accepted. Initializer weights
remain part of the model; external ONNX weight files must also be committed
at the relative locations referenced by the model.

## Files and execution

| Component | File / command |
| --- | --- |
| Workflow stages | `.github/workflows/ci_riscv64_linux_rvv.yml` and `.github/workflows/pkgci_riscv64_linux_rvv.yml` |
| Stage entry point | `bash build_tools/riscv/ci_rvv.sh onnx` |
| Discovery, input handling, reference execution, comparison, reports | `build_tools/riscv/onnx_compare.py` |
| Standalone/default model preparation | `build_tools/riscv/prepare_onnx.py`; optional case `prepare_script` |
| Host dependencies | `build_tools/riscv/onnx_requirements.txt` |
| Input models and sidecars | `tests/models/onnx/` |
| Import | `python -m iree.compiler.tools.import_onnx` |
| Compile | Host `iree-compile --iree-input-type=onnx` with RISC-V RVV flags |
| Target execution | QEMU invokes cross-built `iree-run-module` with `.npy` input/output files |
| Unit checks | `python -m pytest -q build_tools/riscv/onnx_compare_test.py` (requires pytest) |

The source flow builds `IREECompilerPythonModules`, enables the Torch/ONNX
compiler input, and selects its build-tree Python bindings through `PYTHONPATH`.
The package flow uses the importer and compiler from the installed wheel.
ONNX Runtime executes on the Linux host in both flows.

After the host/target build and tool bootstrap have completed:

```sh
bash build_tools/riscv/ci_rvv.sh onnx
```

To select another collection:

```sh
IREE_ONNX_MODELS_DIR=/absolute/path/to/models \
  bash build_tools/riscv/ci_rvv.sh onnx
```

For native development only, an explicit diagnostic mode is available:

```sh
python build_tools/riscv/onnx_compare.py \
  --models-dir tests/models/onnx \
  --output-dir build-onnx-native \
  --compiler /path/to/native/iree-compile \
  --runner /path/to/native/iree-run-module \
  --native
```

This mode is recorded as `native` in results and does not establish RISC-V
coverage. CI does not use it. Without `--native`, both `--qemu` and `--sysroot`
are required. Each import/compile/runtime subprocess has a 300-second timeout
by default (`--timeout` changes it); host ONNX Runtime execution is bounded by
the workflow job timeout, not this subprocess timeout.

## Failure behavior and reports

Each discovered model is a test case in `build-riscv-ci/onnx/onnx-results.xml`.
The JSON companion records execution mode and model work directories. Each
invocation uses fresh output paths, so old results cannot make a new test pass.
All selected models are attempted, and any failed model fails the stage.
Excluded models are marked skipped. An empty collection or no executed models
also fails.

Per-model directories retain the original input arrays, ONNX Runtime reference
arrays, specialized model, imported MLIR, compiled VMFB, target outputs and
`commands.log` when their stages complete. The workflows upload the ONNX
directory as diagnostics, including after failures when files exist, and include
the JUnit report with successful target artifacts.

Import failures, unsupported operators, missing dynamic dimensions, invalid
inputs, subprocess failures/timeouts, missing output files, shape/dtype
mismatches and numerical differences all fail explicitly. NaN/Inf outputs are
rejected. Tensor inputs/outputs are supported; strings, sequences, maps and
optional values are outside the initial scope. Stochastic models require a
deterministic inference form. Large models may require a future externalized
parameter path; the initial importer embeds parameters in its model copy.

Automatic discovery does not promise arbitrary ONNX compatibility: the model
must be supported by both the pinned ONNX Runtime and the chosen IREE commit.
For example, a newer ONNX opset may require updating dependencies or importer
support. These are model/compiler compatibility failures, not reasons to skip
the model silently.

## Implementation validation

The script-based preparation path and all three current static cases passed
actual RISC-V QEMU execution at default VLEN=1024 and explicit `--vlen 512`
on both CPU drivers. The custom ReLU hook ran standalone. Twenty-one validation
tests passed, including VLEN defaults, precedence and invalid values. Earlier
512-bit runs below describe the original profile.

The current static MatMul and ReLU case folders passed the `pr` selection under
actual RISC-V QEMU on both CPU drivers. The nightly selection additionally ran
and passed `matmul_nightly`, including its seeded input script and external
weights. The PR selection recorded that model as skipped. The release selection
also passed all three cases on both drivers. Fourteen input,
comparison and discovery/filtering tests passed, including script failures,
missing script output, invalid suite names and rejection of dynamic inputs in
static cases.

Execution used released IREE compiler 3.13.0 and ONNX Runtime 1.22.1, a source-built
RISC-V runtime, Debian GCC 14 cross-tools and QEMU 10.0.13, with RVV enabled
and VLEN=512. This alternative toolchain was obtained through signed Debian
package metadata because the workflow's original tool-download host was blocked.

Earlier checks of the original synthetic examples also covered native execution,
multiple inputs/outputs and dynamic specialization, and an integration check
that deliberately corrupted actual output values failed with numerical mismatch
reports. Full GitHub source/package workflows, scheduled/release event dispatch,
their same-commit compiler builds,
and their originally pinned QEMU/toolchain combination remain unverified.
