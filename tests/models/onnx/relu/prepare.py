# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Example custom preparation hook, also executable directly for local testing."""
from pathlib import Path
import sys

# Reuse the standard preparation implementation; custom preprocessing can be
# added here without changing the test-discovery runner.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "build_tools/riscv"))
from prepare_onnx import main

if __name__ == "__main__":
    main()
