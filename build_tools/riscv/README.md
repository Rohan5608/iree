# RISC-V 64 + RVV development workflow

CI and local tooling for a fork that develops the `riscv64` + RVV target.

| You want to... | Run |
| --- | --- |
| Set up a new machine | `build_tools/riscv/setup_dev.sh` |
| Run what CI runs, locally | `build_tools/riscv/local_ci.sh lint onnx` (or `all`) |
| See which e2e tests CI runs for riscv64 | `build_tools/riscv/list_e2e_tests.sh` (`--rvv`, `--dir matmul`, `--all`) |
| Test compiler changes from your checkout | `build_tools/riscv/build_host_compiler.sh` (see below) |
| Add a test | drop a `.onnx` into `tests/riscv64_onnx/models/` |
| Push safely | `git push` (the pre-push hook runs the ONNX/QEMU tests) |
| Stop upstream workflows from running on the fork | `build_tools/riscv/disable_upstream_workflows.sh` |

## How upstream IREE runs CI on a PR

Everything starts in `.github/workflows/ci.yml` and `pkgci.yml` (plus
`lint.yml`, `clang_tidy.yml`). Each begins with `setup.yml`, which runs
`build_tools/github_actions/configure_ci.py` to decide **which jobs run**:

* All jobs of the workflow, minus the schedule-only and postsubmit-only ones.
* Plus jobs whose files changed (`PRESUBMIT_TOUCH_ONLY_JOBS`, e.g. only touch
  `tests/external/iree-test-suites/torch*` and `test_torch` runs).
* Everything is skipped if only docs/markdown/etc. changed
  (`SKIP_PATH_PATTERNS`).
* PR-description trailers override this: `skip-ci:`, `ci-skip: job1,job2`,
  `ci-extra: job`, `ci-exactly: job1,job2`.

Then the workflows fan out:

* **`ci.yml`**: from-source builds (Linux x64/arm64, macOS, Windows, with
  clang/gcc/MSVC and ASan/TSan/UBSan variants, Bazel) that run the unit tests
  and the whole **CTest suite** (`build_tools/cmake/ctest_all.sh`). That suite
  includes the **e2e tests** in `tests/e2e/**`: `.mlir` files with
  `check.expect_*` ops (`iree_check_single_backend_test_suite` in the
  `CMakeLists.txt` files) that are compiled with `iree-compile` and executed
  with `iree-check-module`, per backend/driver (llvm-cpu local-task, vmvx,
  Vulkan, CUDA, HIP, ...). Labels such as `noriscv`, `driver=vulkan` or
  `requires-gpu` exclude tests where they can't run.
* **`pkgci.yml`**: first builds the compiler and runtime Python wheels
  (`pkgci_build_packages.yml`, Docker manylinux), then every `pkgci_test_*.yml`
  installs them with `build_tools/pkgci/setup_venv.py` and tests the *packages*:
  GPU machines (AMD, NVIDIA), Android, ONNX/Torch/TensorFlow/PJRT frontends,
  and **RISC-V 64** (`pkgci_test_riscv64.yml`).
* `pkgci_test_riscv64.yml` cross-compiles the runtime with the RISC-V clang
  toolchain (`build_tools/cmake/build_riscv.sh`), compiles the e2e tests for
  `riscv64` with RVV flags (`+v,+zvl512b`, from `linux_riscv64.cmake`) using the
  host compiler from the wheels, and runs the CTest suites under `qemu-riscv64`
  (`build_tools/cmake/test_riscv.sh`, via `run_riscv_test.sh`).
* `workflow_summary.yml` aggregates the results into one status.

## What this fork runs

`.github/workflows/riscv64_ci.yml` ("RISC-V CI") runs on pull requests, on
pushes to `main` and `riscv/**` branches, and on demand:

```text
config -> lint
       -> packages_pip | packages_build -> riscv64_e2e   (upstream suite on QEMU)
                                        -> riscv64_onnx  (your ONNX models vs ONNX Runtime)
all of the above -> summary   (make "RISC-V CI summary" the required check)
```

`riscv64_e2e` and `riscv64_onnx` reuse `pkgci_test_riscv64.yml` and
`pkgci_test_riscv64_onnx.yml` unchanged, so the fork tests the same thing as
upstream. Only the source of the host compiler differs:

| `compiler_source` | Host compiler | Use when |
| --- | --- | --- |
| `nightly` (default) | latest nightly wheels | you change tests/runtime/harness |
| `stable` | latest PyPI release | you want a fixed, known compiler |
| `build` | built from your branch | **you change the compiler** (RVV codegen, passes) |

Pick it per run (Actions -> RISC-V CI -> Run workflow) or for every run with the
repository variable `RISCV_COMPILER_SOURCE`. `build` is slow on a cold cache:
use a larger GitHub-hosted runner by setting the `RISCV_BUILD_RUNNER` variable
(or the `build_runner` input).

**Caveat:** `nightly`/`stable` test your *runtime and tests* against someone
else's compiler. A compiler change on your branch is only exercised with
`compiler_source=build`.

### One-time setup of the fork

1. Settings -> Actions: allow workflows. Settings -> Branches: require the
   status check **RISC-V CI summary** on your main branch.
2. Disable the upstream workflows that can't run on a fork:
   `build_tools/riscv/disable_upstream_workflows.sh` (needs `gh auth login`),
   or disable them in the Actions UI. Without this, `ci.yml` and `pkgci.yml`
   will queue jobs for runners (`azure-linux-scale`, GPUs) that don't exist.
3. Optionally set the repository variables above.

## Local flow

```bash
build_tools/riscv/setup_dev.sh          # .venv, hooks, runtime submodules
source .venv/bin/activate
./build_tools/riscv/riscv_bootstrap.sh  # RISC-V clang toolchain + QEMU (interactive)
export RISCV_TOOLCHAIN_ROOT=... QEMU_BIN=.../qemu-riscv64

./build_tools/riscv/local_ci.sh lint onnx       # fast loop
./build_tools/riscv/local_ci.sh e2e             # full upstream riscv64 suite (slow)
```

### Testing compiler changes locally

The default flow uses pip wheels as the host compiler, so edits under
`compiler/` have no effect on it. To build the compiler from your checkout
(only `iree-compile`, `iree-opt` and the two helper tools, with just the
llvm-cpu and vmvx backends):

```bash
./build_tools/riscv/build_host_compiler.sh          # first build: an hour or more
export IREE_HOST_COMPILER_DIR=$PWD/build-host/host-bin
./build_tools/riscv/local_ci.sh onnx e2e            # now uses your iree-compile
```

Rebuilds after a source change take seconds to minutes. `iree-import-onnx` still
comes from the pip wheels. In CI the equivalent is `compiler_source=build`.

### Git hooks (pre-commit)

`pre-commit install` (done by `setup_dev.sh`) installs two hooks from
`.pre-commit-config.yaml`:

* **pre-commit**: the same lint upstream CI runs (black, clang-format, typos,
  YAML, whitespace, bazel_to_cmake, ...). Fixes files in place; re-stage and
  commit again.
* **pre-push**: re-runs the lint on the pushed files and
  `riscv64-onnx-qemu`, which runs `tests/riscv64_onnx` on QEMU. It **skips with
  a message** when a tool is missing so teammates without the toolchain are not
  blocked. Knobs: `RISCV_PREPUSH_STRICT=1` fails instead of skipping,
  `RISCV_PREPUSH_SKIP=1` or `git push --no-verify` skips.

The ONNX test suite itself is documented in `tests/riscv64_onnx/README.md`.

## Limits and notes

* The `buildifier` hook downloads a Go toolchain on first use.
* `lint` in CI runs `pre-commit` on **all** files (like upstream), so it can fail
  on files you did not touch if upstream was not lint-clean when you branched.
* GitHub-hosted runners have 4 cores / 16GB in public repos (2 / 8GB in private
  ones, where `build` becomes impractical); the e2e job cross-builds the whole
  runtime and test suite and is the slowest of the non-`build` jobs. ccache
  (via `actions/cache`) makes subsequent runs much faster. Caches are shared
  from `main` to branches, so let `main` run the workflow once.
