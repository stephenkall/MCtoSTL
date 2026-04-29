"""
Streaming binary STL writer.

Triangles are written to disk one at a time — the full mesh never lives
in RAM simultaneously, making arbitrarily large models possible.

Usage
-----
with StreamingSTL(filepath, n_triangles) as stl:
    for v0, v1, v2 in my_generator():
        stl.write_triangle(v0, v1, v2)
"""

import struct
from typing import Generator, Iterable, Tuple

import numpy as np


# ── Normal ────────────────────────────────────────────────────────────────────

def _normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    length = np.linalg.norm(n)
    return (n / length) if length > 1e-12 else n


# ── Streaming writer ──────────────────────────────────────────────────────────

class StreamingSTL:
    """Context manager that streams triangles directly to a binary STL file."""

    TRIANGLE_BYTES = 50   # 12 (normal) + 3*12 (vertices) + 2 (attr)

    def __init__(self, filepath: str, n_triangles: int) -> None:
        self.filepath = filepath
        self.n_triangles = n_triangles
        self._fh = None
        self._written = 0

    def __enter__(self):
        self._fh = open(self.filepath, "wb")
        self._fh.write(b"MCtoSTL terrain relief model".ljust(80, b"\x00"))
        self._fh.write(struct.pack("<I", self.n_triangles))
        return self

    def __exit__(self, *_):
        if self._fh:
            self._fh.close()
            self._fh = None

    def write_triangle(
        self,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
    ) -> None:
        nrm = _normal(
            np.asarray(v0, dtype=np.float32),
            np.asarray(v1, dtype=np.float32),
            np.asarray(v2, dtype=np.float32),
        )
        self._fh.write(struct.pack("<fff", *nrm))
        self._fh.write(struct.pack("<fff", *v0))
        self._fh.write(struct.pack("<fff", *v1))
        self._fh.write(struct.pack("<fff", *v2))
        self._fh.write(b"\x00\x00")
        self._written += 1


# ── Triangle count formula ────────────────────────────────────────────────────

def count_solid_triangles(rows: int, cols: int) -> int:
    """Number of triangles in a watertight solid for an R×C heightmap."""
    surface_tris = (rows - 1) * (cols - 1) * 4   # top + bottom (2 tris/quad each)
    wall_tris    = (4 * (rows - 1) + 4 * (cols - 1))  # 2 tris/edge × 2 walls each axis
    return surface_tris + wall_tris
