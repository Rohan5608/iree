# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generate the fixed [2, 3] input; weights are external ONNX initializers."""
import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
x = np.loadtxt(args.model_dir / "data/input.csv", delimiter=",", dtype=np.float32)
np.savez(args.output, x=x)
