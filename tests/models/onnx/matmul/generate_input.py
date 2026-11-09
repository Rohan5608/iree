# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generate named x[2,3] float32 input from the checked-in CSV fixture.

Local use from the repository root:
    python tests/models/onnx/matmul/generate_input.py --output /tmp/matmul.npz \
        --model-dir tests/models/onnx/matmul --seed 0

The shared preparation helper invokes the same CLI. Weights are ONNX external
initializers, so they must NOT be added as another runtime input in this NPZ.
"""
import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
x = np.loadtxt(args.model_dir / "data/input.csv", delimiter=",", dtype=np.float32)
# The key must match the ONNX graph input name exactly; dtype/shape are checked
# by common preparation. This fixture is fixed, so the accepted seed is unused.
np.savez(args.output, x=x)
