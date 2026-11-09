# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generate named x[32,64] float32 input for the nightly/release example.

Accepts the shared --output /path/to/inputs.npz --model-dir /path/to/case
--seed N contract. The manifest selects which suites run this script; the
generator itself does not decide when to run or execute either inference backend.
"""
import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
x = np.random.default_rng(args.seed).uniform(-1, 1, (32, 64)).astype(np.float32)
# Seeded generation makes the fixture reproducible. Preparation runs this once
# and gives identical saved values to ONNX Runtime and IREE/QEMU.
np.savez(args.output, x=x)
