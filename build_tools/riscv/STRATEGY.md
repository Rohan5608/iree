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
| `tests/riscv64_onnx/models/` | The test cases. `<model>.onnx.json` optionally holds per-model settings (`rtol`, `atol`, `input_dims`, `int_range`, `extra_compile_flags`, `skip`, `xfail`). |
| `tests/riscv64_onnx/requirements.txt`, `README.md` | Python dependencies and usage. |
| `build_tools/riscv/build_run_module.sh` | Cross-builds only the riscv64 `iree-run-module`. |
| `build_tools/riscv/run_onnx_tests.sh` | Runs the ONNX suite (used by CI and locally). |
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
| Change the vector widths in CI | Repo variable `RISCV_ONNX_VLENS` (default `128,256,512,1024`). |
| Change the compiler source | Run input `compiler_source`, or repo variable `RISCV_COMPILER_SOURCE`. |
| Change RVV compile flags | `DEFAULT_COMPILE_FLAGS` in the harness **and** `linux_riscv64.cmake` (keep equal). |
| Change the RISC-V toolchain/QEMU version | The `Bootstrap` step in `pkgci_test_riscv64.yml` and `pkgci_test_riscv64_onnx.yml`. |
| Change which upstream tests are excluded | `test_riscv.sh` **and** `LABEL_EXCLUDE` in `list_e2e_tests.sh`. |

## 4. Three meanings of "nightly" and two of "stable"

These words are used for different things. Keep them apart.

| Term | What it is | Who runs it |
|---|---|---|
| **Upstream nightly release** | iree-org's scheduled job (05:00 UTC) tags the latest green `main` commit with an `rc` version, creates a draft prerelease and builds wheels (`schedule_candidate_release.yml`, then `build_package.yml`). Published wheels are listed at `iree.dev/pip-release-links.html`. | iree-org. Does **not** run on the fork. |
| **`compiler_source=nightly`** | *Our* CI downloads those upstream nightly wheels as the host compiler. A way to get a compiler, not a test schedule. | Implemented, the default. |
| **Fork nightly run** | A scheduled run of *our* test matrix. | **Proposed, not implemented.** |
| **Upstream stable release** | A release cut manually (`oneshot_candidate_release.yml` with `stable`), validated and published by `validate_and_publish_release.yml`. Latest is what PyPI serves. | iree-org. |
| **`compiler_source=stable`** | Our CI installs the latest PyPI stable wheels as the host compiler. | Implemented. |
| **Fork stable gate** | A release-quality run of our full matrix against a pinned compiler. | **Proposed, not implemented.** |

Important facts about upstream's pipeline (read from the workflows):

- Nightly/stable releases are cut from a commit where upstream's `ci.yml` passed
  (`get_latest_green.sh`). That is x86/arm/etc. CI, not RISC-V.
- Upstream release validation only smoke-tests the packages (`--help` on the
  tools, `iree-cpuinfo`, and so on). It runs **no** e2e, runtime or RISC-V
  tests. So an upstream nightly or stable wheel is **not** known to be good for
  RISC-V.

## 5. What runs today (implemented)

`riscv64_ci.yml` runs on **pull requests** (any base branch), on **pushes to
`main` and `riscv/**`**, and on **manual dispatch**. It has no schedule. All
triggers run the same jobs:

| Job | Tests involved |
|---|---|
| `lint` | pre-commit on all files (black, clang-format, typos, YAML, whitespace, bazel_to_cmake, ...). |
| `packages_pip` / `packages_build` | Provide the host compiler: `nightly` (default) or `stable` wheels, or built from the branch (`build`). |
| `riscv64_e2e` | Upstream suite via `pkgci_test_riscv64.yml`: runtime unit tests (264 listed locally) and e2e tests (384 enabled of 447, VLEN 512), plus the bare-metal smoke test. Counts come from `list_e2e_tests.sh` and may differ in CI. |
| `riscv64_onnx` | Every model in `tests/riscv64_onnx/models`, at each width in `ONNX_VLENS` (CI default 128,256,512,1024). |
| `summary` | Aggregates results; make this the required check. |

Consequences of the defaults:

- The host compiler is **upstream's nightly wheel**, so a PR's compiler changes
  are **not** exercised unless `compiler_source=build`.
- The runtime (`iree-run-module`, runtime unit tests) **is** built from the PR
  branch in every mode.
- Models that fail import/compile/compare fail the run. There is no tiering:
  every model runs on every PR.

## 6. Proposed tiers (not implemented)

The idea: cheap, always-on checks on PRs; heavier checks on a schedule; and a
strict pinned run before declaring a state "stable". The same test cases are
used throughout; the tiers differ in **compiler**, **vector widths**, **model
set** and **schedule**.

| | PR / push (today) | Post-merge on `main` (proposed) | Nightly (proposed) | Stable gate (proposed) |
|---|---|---|---|---|
| Trigger | PR, push to `main`/`riscv/**` | push to `main` | cron + manual | manual or tag |
| Host compiler | `nightly` wheels | `build` from `main` | `build` from `main`, plus a `nightly`-wheel canary | **pinned** `stable` wheels, and/or `build` at the tag |
| Lint | yes | yes | no | yes |
| Upstream e2e + runtime tests | yes, VLEN 512 | yes | yes; add one extra width (e.g. 256) once the CMake change in 7 is done | yes, same as nightly |
| ONNX models | all, VLEN 128–1024 | all, VLEN 128–1024 | all **plus** slow/large models (tier `slow`), VLEN 128–1024 | all incl. `slow`, VLEN 128–1024 |
| Allowed failures | none | none | `xfail` models only | none; stale `xfail` entries must be reviewed |
| Output | status check | warms the compiler cache | status + report of regressions vs previous night | recorded compiler/runtime versions and a pass table |

Rationale:

- **Why `build` post-merge/nightly, not on every PR:** a cold compiler build
  takes hours on a 4-core runner. After one run on `main` the cache makes
  later runs cheap, and PRs can reuse the `main` cache.
- **Why a `nightly`-wheel canary:** it tells you whether *upstream's* latest
  compiler still works for RISC-V, independent of your branch.
- **Why pin `stable`:** a stable result must be reproducible. Floating at
  "latest" would let the result change without any change in the repo.
- **Do tiers change the test cases?** The test *sources* are the same. Only the
  `slow` model set and the extra vector width are additions that need a small
  amount of new code (see 7).

## 7. Work needed to implement section 6

| Item | What to change | Effort |
|---|---|---|
| Schedule + `schedule`/`push` triggers | `riscv64_ci.yml`: add `on: schedule` and a `tier` input; pass it to jobs. | Small |
| Model tiers | Add a `"tier": "slow"` key to the sidecar schema; `--tier` option in `conftest.py`. | Small |
| Pinned stable compiler | `compiler_source=stable` takes a version input (`iree-base-compiler==X`) instead of "latest". | Small |
| Compiler canary | A second `packages_pip` run with nightly wheels, differential report. | Medium |
| Extra vector width for the e2e suite | Make the width a CMake variable in `linux_riscv64.cmake`; exclude the nine RVV matmul tests that hard-code `+zvl512b` at other widths; one cross-build per width. | Medium; touches an upstream file |
| Regression report | Store per-test results (JUnit is already produced) and diff against the previous run. | Medium |

## 8. Verification status

Verified locally (x86 Linux container, Ubuntu clang cross toolchain, QEMU
8.2.2): the ONNX suite at widths 128, 256, 512, 1024 on both sample models;
models with external weights files; failure handling (missing weights,
corrupt model, strict tolerances, width mismatch); the pre-commit and pre-push
hooks; configure of the host-compiler build and the helper tools; the
listing of riscv64 e2e tests.

Verified on GitHub: one run of `RISC-V CI` (lint, nightly wheels, e2e suite,
ONNX suite at the then-default width, bare-metal smoke test, summary) passed.
That run predates the multi-width matrix.

**Not verified:**

- The multi-width matrix in GitHub CI (QEMU there is 8.1.2; widths above 1024
  may not be supported by it).
- `compiler_source=build` end to end (hours; not run). `iree-compile` itself
  was not built by `build_host_compiler.sh`; only the configure and the
  helper tools were.
- `IREE_HOST_COMPILER_DIR` in `local_ci.sh` (syntax-checked only).
- The `disable_upstream_workflows.sh` script (needs the GitHub CLI).
- Large models: only models up to a few hundred KB were tested.

## 9. Limitations

- Correctness only; QEMU speed says nothing about real hardware.
- A compiler built for a larger VLEN than the CPU provides runs silently and
  gives wrong results. The harness always sets compile and run widths together;
  do not set them separately.
- `nightly`/`stable` wheels do not contain branch compiler changes.
- `noriscv`-labelled upstream tests are skipped on RISC-V. Some mark real
  codegen gaps (for example f16/bf16 asserts); grep `noriscv` in `BUILD.bazel`.
- Models import with weights embedded in the MLIR; multi-GB models are not
  practical until the importer's external-parameter option is used.

## Open decisions

1. **Do you publish anything** (wheels, releases) from the fork, or is
   "stable" only a gate on a tested state? Section 6 assumes the latter.
2. **Which compiler does "stable" mean:** a pinned upstream stable release, or
   your own branch at a tag? Or both?
3. **Nightly schedule and runner size:** time, and whether a larger runner is
   available for `build`.
4. **Which models are `slow`**, and what is the largest model you need.
5. **Should the e2e suite also run at more than one vector width?** (needs
   the CMake change in section 7).
