"""
Low-level binary STL writer.
Triangles are stored as lists of (v0, v1, v2) vertex triples (numpy float32).
"""

import struct
import numpy as np


def _face_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    length = np.linalg.norm(n)
    return (n / length) if length > 1e-12 else n


def write_binary_stl(filepath: str, triangles: list) -> None:
    """
    Write a binary STL file.

    Parameters
    ----------
    filepath   : output path
    triangles  : list of (v0, v1, v2) where each v is a 3-element array-like
    """
    n = len(triangles)
    with open(filepath, "wb") as fh:
        fh.write(b"MCtoSTL terrain relief model".ljust(80, b"\x00"))
        fh.write(struct.pack("<I", n))
        for tri in triangles:
            v0 = np.asarray(tri[0], dtype=np.float32)
            v1 = np.asarray(tri[1], dtype=np.float32)
            v2 = np.asarray(tri[2], dtype=np.float32)
            nrm = _face_normal(v0, v1, v2)
            fh.write(struct.pack("<fff", *nrm))
            fh.write(struct.pack("<fff", *v0))
            fh.write(struct.pack("<fff", *v1))
            fh.write(struct.pack("<fff", *v2))
            fh.write(b"\x00\x00")
