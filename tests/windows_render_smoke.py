"""Exercise the public normal renderer on a native Windows OpenGL context."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np


if sys.platform != "win32":
    raise SystemExit("Run this smoke test on Windows")
if "PYOPENGL_PLATFORM" in os.environ:
    raise SystemExit("Windows test requires pyrender's default OpenGL backend")

source = Path(__file__).resolve().parents[1] / "sam3d_normal.py"
spec = importlib.util.spec_from_file_location("anygles_sam3d_normal", source)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "PYOPENGL_PLATFORM" not in os.environ

vertices = np.array(
    [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
    dtype=np.float32,
)
faces = np.array([[0, 1, 2]], dtype=np.int32)
image = module._render(
    vertices,
    faces,
    np.array([0.0, 0.0, 2.0], dtype=np.float32),
    100.0,
    128,
    128,
    1.0,
    np.zeros(2, dtype=np.float32),
)
nonblack = int(np.count_nonzero(np.asarray(image).any(axis=2)))
if nonblack < 100:
    raise SystemExit(f"Normal render is empty: {nonblack} nonblack pixels")
print(f"Windows normal render passed: {nonblack} nonblack pixels")
