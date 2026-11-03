# RISC-V 64 + RVV CI/CD strategy

This document describes how the fork tests the `riscv64` + RVV target: what
runs, which files make it work, which upstream files were touched, and how
"nightly" and "stable" are meant to be used. For day-to-day commands see
[README.md](README.md); for the ONNX suite see
[../../tests/riscv64_onnx/README.md](../../tests/riscv64_onnx/README.md).

**Status legend.** Everything marked *implemented* exists in the repo and was
exercised. Everything marked *proposed* is a plan that is **not** implemented
and needs a decision (see [Open decisions](#open-decisions)).

## 1. Goals

1. Every change is checked on `riscv64` with the vector extension (RVV), not
   just on x86.
2. Any `.onnx` file added to the repo becomes a test automatically and is
   compared against ONNX Runtime.
3. Reuse upstream IREE's RISC-V infrastructure instead of building a parallel
   one, so merging upstream stays cheap.
4. Work on GitHub-hosted runners (the fork cannot use iree-org's runner pools).
5. The same checks can be run locally, and a subset runs before `git push`.

## 2. Architecture

```text
 .onnx model ──► ONNX Runtime (x86 host) ───────────────► reference outputs ─┐
      │                                                                       ├─► compare
      └► iree-import-onnx ► iree-compile (x86 host, riscv64+RVV flags)        │   (numpy)
              ► model.vmfb ► qemu-riscv64 + riscv64 iree-run-module ► outputs ┘

 tests/e2e/**/*.mlir ► iree-compile (riscv64+RVV) ► .vmfb ► qemu-riscv64 +
      iree-check-module (check.* asserts evaluated inside the runtime)
```

Nothing runs on real RISC-V hardware. Compilation and the reference run
happen on x86; only execution is emulated (QEMU user mode, CPU with
`Zve64d`, `vlen=N`). This validates **correctness**, not performance.

Two independent suites:

| | Upstream e2e suite | ONNX suite (ours) |
|---|---|---|
| Source of tests | `tests/e2e/**` (`.mlir` with `check.*` ops) | `tests/riscv64_onnx/models/*.onnx` |
| Expected result | Written in the test | Computed by ONNX Runtime |
| Compared in | `iree-check-module` | Python (numpy), `rtol=1e-4 atol=1e-5` |
| Driver | CTest via `build_riscv.sh` / `test_riscv.sh` | pytest via `run_onnx_tests.sh` |
| Vector width | 512 (fixed in `linux_riscv64.cmake`) | `--vlens`, default 512; CI 128,256,512,1024 |

## 3. Files

### 3.1 Scripts and tests we added

| File | Purpose |
|---|---|
| `tests/riscv64_onnx/riscv_onnx_harness.py` | The engine: inputs, ONNX Runtime run, import, compile, QEMU run, comparison, `with_vlen`. |
| `tests/riscv64_onnx/test_onnx_models.py` | One pytest test per model and VLEN. |
| `tests/riscv64_onnx/conftest.py` | Model discovery, options (`--vlens`, tolerances, tool paths). |
| `tests/riscv64_onnx/models/` | The test cases. `<model>.onnx.json` optionally holds per-model settings (`rtol`, `atol`, `input_dims`, `int_range`, `extra_compile_flags`, `skip`, `xfail`, `tier`). `models/slow/` holds the sample `slow`-tier model (external weights file). |
| `tests/riscv64_onnx/test_ci_config.py` | Unit tests for the tier / compiler-source decision. |
| `tests/riscv64_onnx/requirements.txt`, `README.md` | Python dependencies and usage. |
| `build_tools/riscv/build_run_module.sh` | Cross-builds only the riscv64 `iree-run-module`. |
| `build_tools/riscv/run_onnx_tests.sh` | Runs the ONNX suite (used by CI and locally). |
| `build_tools/riscv/resolve_ci_config.sh` | Decides the tier (pr / nightly / stable) and the host-compiler source for `riscv64_ci.yml`. Tested by `tests/riscv64_onnx/test_ci_config.py`. |
| `build_tools/riscv/build_host_compiler.sh` | Builds `iree-compile` & helpers from the checkout (for compiler changes). |
| `build_tools/riscv/list_e2e_tests.sh` | Lists the upstream e2e/runtime tests that run on riscv64. |
| `build_tools/riscv/local_ci.sh` | Local wrapper: `lint`, `build`, `onnx`, `e2e`, `all`. **Not used by GitHub CI.** |
| `build_tools/riscv/setup_dev.sh` | One-time developer setup (venv, hooks, submodules). |
| `build_tools/riscv/pre_push_check.sh` | The pre-push hook body (skips if tools are missing). |
| `build_tools/riscv/disable_upstream_workflows.sh` | Disables upstream workflows that cannot run on a fork. |
| `.github/workflows/riscv64_ci.yml` | The fork's CI entry point ("RISC-V CI"). |
| `.github/workflows/pkgci_test_riscv64_onnx.yml` | CI job that runs the ONNX suite. |
| `build_tools/riscv/README.md`, `STRATEGY.md` | Documentation. |

### 3.2 Existing files we changed

| File | Change | Merge risk |
|---|---|---|
| `.github/workflows/pkgci_build_packages.yml` | New optional `runs-on` input (default unchanged) and a disk-cleanup step for GitHub-hosted runners. | Low |
| `.github/workflows/pkgci.yml` | Registers the `test_riscv64_onnx` job and adds it to the summary. | Low |
| `.pre-commit-config.yaml` | `default_install_hook_types` and the `riscv64-onnx-qemu` pre-push hook. | Low |

Nothing in `build_tools/cmake/` or `build_tools/pkgci/` was modified.

### 3.3 Upstream files we depend on (unchanged)

`build_tools/cmake/build_riscv.sh`, `test_riscv.sh`, `run_riscv_test.sh`,
`linux_riscv64.cmake` (RVV flags and QEMU CPU), `iree_check_test.cmake` (the
`noriscv` label), `.github/workflows/pkgci_test_riscv64.yml`,
`build_tools/pkgci/setup_venv.py`, `workflow_summary.yml`.

If upstream changes `linux_riscv64.cmake`'s feature flags or QEMU CPU, update
`DEFAULT_COMPILE_FLAGS` / `DEFAULT_QEMU_CPU` in `riscv_onnx_harness.py` (they
are copied, with a comment).

### 3.4 Where to change things

| To... | Change |
|---|---|
| Add a model | Add `tests/riscv64_onnx/models/<name>.onnx` (+ weights file beside it, + optional `.onnx.json`). |
| Quarantine a failing model | `"xfail"` or `"skip"` in its `.onnx.json`. |
| Make a model nightly/stable-only (big or slow) | `"tier": "slow"` in its `.onnx.json`. |
| Pin the compiler for a stable run | Run input `compiler_version` (e.g. `3.13.0`). |
| Change the nightly time | The `cron` line in `riscv64_ci.yml`. |
| Change the vector widths in CI | Repo variable `RISCV_ONNX_VLENS` (default `128,256,512,1024`). |
| Change the compiler source | Run input `compiler_source`, or repo variable `RISCV_COMPILER_SOURCE`. |
| Change RVV compile flags | `DEFAULT_COMPILE_FLAGS` in the harness **and** `linux_riscv64.cmake` (keep equal). |
| Change the RISC-V toolchain/QEMU version | The `Bootstrap` step in `pkgci_test_riscv64.yml` and `pkgci_test_riscv64_onnx.yml`. |
| Change which upstream tests are excluded | `test_riscv.sh` **and** `LABEL_EXCLUDE` in `list_e2e_tests.sh`. |

## 4. What "nightly" and "stable" mean here

The words are used for different things. Keep them apart.

| Term | What it is | Who runs it |
|---|---|---|
| **Upstream nightly release** | iree-org's scheduled job (05:00 UTC) tags the latest green `main` commit with an `rc` version, creates a draft prerelease and builds wheels. It *produces release artifacts*. | iree-org. Does **not** run on the fork. |
| **Upstream stable release** | A release cut manually and published; what PyPI serves. | iree-org. |
| **`compiler_source=nightly` / `stable`** | Our CI downloads those upstream wheels to use as the host compiler. A way to obtain a compiler, not a test schedule. | Implemented. |
| **Fork nightly run** (tier `nightly`) | A *scheduled test run* of our own matrix on our branch. It tests; it **publishes nothing** and creates no release or tag. | Implemented. |
| **Fork stable run** (tier `stable`) | A strict, reproducible run of the full matrix against a pinned compiler, started by a tag or by hand. It also publishes nothing; its result is the record that a state was good. | Implemented. |

Facts about upstream's pipeline (read from the workflows): releases are cut
from a commit where upstream's `ci.yml` passed (x86/arm/etc.), and release
validation only smoke-tests the packages (`--help`, `iree-cpuinfo`). It runs
**no** e2e, runtime or RISC-V tests. An upstream nightly or stable wheel is
therefore **not** known to be good for RISC-V; our tiers are what establish
that.

## 5. The tiers (implemented)

`riscv64_ci.yml` picks a tier with `resolve_ci_config.sh`:

- an explicit `tier` input (manual run) wins;
- otherwise a **scheduled** run is `nightly`;
- otherwise a pushed tag named `riscv-stable-*` is `stable`;
- everything else (pull requests, pushes to `main` / `riscv/**`) is `pr`.

| | `pr` | `nightly` | `stable` |
|---|---|---|---|
| Trigger | PR, push to `main`/`riscv/**`, manual | cron 02:17 UTC daily, manual | tag `riscv-stable-*`, manual |
| Purpose | Fast feedback on every change | Test the branch's own compiler across everything | A reproducible, strict pass for a chosen state |
| Host compiler (default) | `nightly` wheels (or repo variable `RISCV_COMPILER_SOURCE`) | **built from the branch** | **`stable` wheels**, pin with `compiler_version` |
| Lint | yes | no | yes |
| Upstream e2e + runtime tests | yes, VLEN 512 | yes, VLEN 512 | yes, VLEN 512 |
| ONNX models | models of tier `pr` (not `slow`) | **all** models, including `slow` | **all** models, including `slow` |
| ONNX vector widths | `RISCV_ONNX_VLENS` (default 128,256,512,1024) | same | same |
| A model marked `xfail` | skipped (not run) | skipped (not run) | **run**; if it now passes the run **fails** ("stale xfail") |
| Test report artifact | only on failure | always | always |
| Records | | compiler wheel names (if wheels) | compiler wheel names and pinned version |

Any default can be overridden on a manual run (`tier`, `compiler_source`,
`compiler_version`, `build_runner`).

### What a fork nightly run actually is

A cron-triggered run of the same workflow with `tier=nightly`. It is **not** a
release: nothing is tagged, uploaded to an index or published. Compared with a
PR run it differs in three ways:

1. **It tests our compiler.** PR runs use upstream's nightly wheel, so compiler
   changes on the branch are not exercised. The nightly run builds the compiler
   from the branch (hours on a cold cache; cached afterwards), so it is the one
   run that tests our compiler changes against the full suite.
2. **It runs more models.** The `slow` models (large or slow to emulate) are
   skipped on PRs and included here.
3. **It does not re-lint**; lint is a PR concern.

The first nightly run builds LLVM from scratch and may need a larger runner
(`RISCV_BUILD_RUNNER`). GitHub only runs schedules from the **default branch**
and may disable them in inactive repositories, so merge `riscv64_ci.yml`
there and check the Actions tab after the first night.

### The stable tier

Goal: a result that can be reproduced. It differs from nightly in:

- the compiler is a **pinned upstream stable release** (`compiler_version`; the
  exact wheel names are written to the run summary). Without a pin it takes the
  latest PyPI release and records which one;
- a `xfail` model is **run**, and a model that now passes fails the run, so
  the known-failure list cannot rot;
- lint runs.

To test a tag of your own compiler instead, start the run with
`compiler_source=build`.

### e2e suite vs ONNX suite across tiers

These are two different test sets (section 2). Across tiers:

- **The upstream e2e + runtime tests are identical in every tier**: the same
  384 enabled e2e and 264 runtime tests at VLEN 512. What changes is *which
  compiler compiles them*: upstream's wheel in `pr`, our branch build in
  `nightly`, a pinned release in `stable`.
- **The ONNX suite changes in the set of models** (`slow` ones only in
  nightly/stable), and in how `xfail` is treated (stable only). Its vector
  widths are the same in all tiers.

## 6. Not implemented

| Item | What it needs |
|---|---|
| Publishing or releasing anything from the fork | A decision on what to publish and where. |
| A canary: also run the nightly-wheel compiler during the nightly tier, and report the difference | A second package job (the artifact name must differ), plus a report. |
| Extra vector widths for the upstream e2e suite | Make the width a CMake variable in `linux_riscv64.cmake` (an upstream file), exclude the nine RVV matmul tests that hard-code `+zvl512b` at other widths, one cross-build per width. |
| Regression report against the previous night | Store JUnit results and diff them. |
| Per-model time budgets | A time limit per tier in the sidecar. |

## 7. Verification status

Verified locally (x86 Linux container, Ubuntu clang cross toolchain, QEMU
8.2.2): the ONNX suite at widths 128, 256, 512, 1024 on both sample models;
models with external weights files; failure handling (missing weights,
corrupt model, strict tolerances, width mismatch); the pre-commit and pre-push
hooks; configure of the host-compiler build and the helper tools; the
listing of riscv64 e2e tests.

Also verified locally: model tiers (`pr` excludes `slow`; `nightly`/`stable`
include it), the stable-tier stale-xfail rule (a passing model marked xfail
fails; a broken one is tolerated), and the tier/compiler-source decision
(17 unit tests in `test_ci_config.py`, including that PR and push behaviour is
unchanged).

Verified on GitHub: one run of `RISC-V CI` (lint, nightly wheels, e2e suite,
ONNX suite at the then-default width, bare-metal smoke test, summary) passed.
That run predates the multi-width matrix and the tiers.

**Not verified:**

- The multi-width matrix and the tier logic in GitHub CI (QEMU there is
  8.1.2; widths above 1024 may not be supported by it). The workflow passes
  `actionlint`, but the schedule, the tag trigger and the `workflow_dispatch`
  inputs have not run on GitHub.
- The nightly tier end to end: it needs `compiler_source=build`, which was not
  run, and the schedule only fires from the default branch.
- `compiler_source=build` end to end (hours; not run). `iree-compile` itself
  was not built by `build_host_compiler.sh`; only the configure and the
  helper tools were.
- `IREE_HOST_COMPILER_DIR` in `local_ci.sh` (syntax-checked only).
- The `disable_upstream_workflows.sh` script (needs the GitHub CLI).
- Large models: only models up to a few hundred KB were tested.

## 8. Limitations

- Correctness only; QEMU speed says nothing about real hardware.
- A compiler built for a larger VLEN than the CPU provides runs silently and
  gives wrong results. The harness always sets compile and run widths together;
  do not set them separately.
- `nightly`/`stable` wheels do not contain branch compiler changes.
- `noriscv`-labelled upstream tests are skipped on RISC-V. Some mark real
  codegen gaps (for example f16/bf16 asserts); grep `noriscv` in `BUILD.bazel`.
- Models import with weights embedded in the MLIR; multi-GB models are not
  practical until the importer's external-parameter option is used.

## Defaults chosen (change if wrong)

These were decided without input; each is one line to change.

1. **No publishing.** "Stable" and "nightly" are test runs only.
2. **Stable = pinned upstream stable wheels** (or `build` at a tag when asked).
3. **Nightly time 02:17 UTC**, compiler built on `ubuntu-24.04` unless
   `RISCV_BUILD_RUNNER` is set.
4. **"Slow" is chosen per model** with `"tier": "slow"`; nothing is marked slow
   except the sample external-weights model.
5. **The e2e suite stays at VLEN 512** in every tier (section 6).
