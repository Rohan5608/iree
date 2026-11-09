# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generate a deterministic fixed [32, 64] input for the extended suite."""
import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
x = np.random.default_rng(args.seed).uniform(-1, 1, (32, 64)).astype(np.float32)
np.savez(args.output, x=x)
