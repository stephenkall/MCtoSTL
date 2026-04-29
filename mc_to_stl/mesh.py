"""
Heightmap → watertight solid STL mesh.

Memory model
------------
Triangles are streamed directly to disk via StreamingSTL — the full mesh
never resides in RAM simultaneously.  Peak memory = O(one heightmap row).

Downsampling
------------
For very large maps (>2000 blocks on a side) generating full-resolution
STLs produces files too large to process on most machines.  The caller
can pass max_vertices to auto-downsample the heightmap before meshing.

Coordinate conventions
----------------------
  X  →  block column  (East)
  Y  →  block row     (North/South = Minecraft Z)
  Z  →  surface height (Up = Minecraft Y)
"""

import math
import os
from typing import Generator, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from tqdm import tqdm

from .stl_writer import StreamingSTL, count_solid_triangles


# ── Helpers ───────────────────────────────────────────────────────────────────

def _v(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float32)


def downsample(hm: np.ndarray, max_verts: int) -> np.ndarray:
    """
    Downsample heightmap so the longest side ≤ max_verts.
    Uses cubic spline interpolation for smooth results.
    """
    rows, cols = hm.shape
    longest = max(rows, cols)
    if longest <= max_verts:
        return hm
    factor = max_verts / longest
    new_rows = max(2, int(round(rows * factor)))
    new_cols = max(2, int(round(cols * factor)))
    return zoom(hm.astype(np.float64), (new_rows / rows, new_cols / cols),
                order=3).astype(np.float32)


# ── Streaming triangle generator ──────────────────────────────────────────────

def _iter_solid(
    heights: np.ndarray,
    x_scale: float,
    y_scale: float,
    z_scale: float,
    base_mm: float,
    origin_x: float,
    origin_y: float,
    h_min_global: float,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Yield (v0, v1, v2) triangles for a watertight solid.
    Iterates row-by-row so RAM usage is proportional to one row, not the whole mesh.
    """
    R, C = heights.shape
    h_min = h_min_global

    # Pre-compute Z for the top surface in float32
    Z = (heights - h_min) * z_scale + base_mm

    def tv(r: int, c: int) -> np.ndarray:
        return _v(origin_x + c * x_scale, origin_y + r * y_scale, float(Z[r, c]))

    def bv(r: int, c: int) -> np.ndarray:
        return _v(origin_x + c * x_scale, origin_y + r * y_scale, 0.0)

    # ── Top surface ──────────────────────────────────────────────────────
    for r in range(R - 1):
        for c in range(C - 1):
            a, b = tv(r, c),   tv(r,   c+1)
            cd,  d = tv(r+1, c), tv(r+1, c+1)
            yield a, b, d
            yield a, d, cd

    # ── Bottom surface (reversed winding) ────────────────────────────────
    for r in range(R - 1):
        for c in range(C - 1):
            a, b = bv(r, c),   bv(r,   c+1)
            cd,  d = bv(r+1, c), bv(r+1, c+1)
            yield a, d, b
            yield a, cd, d

    # ── Left wall (c=0, outward = -X) ────────────────────────────────────
    for r in range(R - 1):
        t0, t1 = tv(r, 0), tv(r+1, 0)
        b0, b1 = bv(r, 0), bv(r+1, 0)
        yield t0, b1, b0
        yield t0, t1, b1

    # ── Right wall (c=C-1, outward = +X) ─────────────────────────────────
    for r in range(R - 1):
        t0, t1 = tv(r, C-1), tv(r+1, C-1)
        b0, b1 = bv(r, C-1), bv(r+1, C-1)
        yield t0, b0, b1
        yield t0, b1, t1

    # ── Front wall (r=0, outward = -Y) ───────────────────────────────────
    for c in range(C - 1):
        t0, t1 = tv(0, c), tv(0, c+1)
        b0, b1 = bv(0, c), bv(0, c+1)
        yield t0, b0, b1
        yield t0, b1, t1

    # ── Back wall (r=R-1, outward = +Y) ──────────────────────────────────
    for c in range(C - 1):
        t0, t1 = tv(R-1, c), tv(R-1, c+1)
        b0, b1 = bv(R-1, c), bv(R-1, c+1)
        yield t0, b1, b0
        yield t0, t1, b1


# ── Single STL ────────────────────────────────────────────────────────────────

def generate_single_stl(
    heightmap: np.ndarray,
    max_x_mm: float,
    max_y_mm: float,
    max_z_mm: float,
    base_mm: float,
    smooth_sigma: float,
    output_path: str,
    max_vertices: int = 2000,
) -> None:
    """Generate one watertight STL for the entire terrain."""
    print(f"\n[Single STL]")

    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)
    sm = downsample(sm, max_vertices)
    rows, cols = sm.shape

    h_range = float(sm.max() - sm.min())
    if h_range < 1e-6:
        h_range = 1.0

    x_scale = max_x_mm / (cols - 1)
    y_scale = max_y_mm / (rows - 1)
    z_scale = max_z_mm / h_range

    n_tris = count_solid_triangles(rows, cols)
    size_mb = n_tris * 50 / 1_048_576

    print(f"  Mesh        : {cols} × {rows} vertices  (original downsampled)")
    print(f"  X/Y scale   : {x_scale:.4f} / {y_scale:.4f} mm/vertex")
    print(f"  Z scale     : {z_scale:.4f} mm/unit  (range {h_range:.0f})")
    print(f"  Dimensions  : {(cols-1)*x_scale:.1f} × {(rows-1)*y_scale:.1f} × "
          f"{h_range*z_scale + base_mm:.1f} mm")
    print(f"  Triangles   : {n_tris:,}  (~{size_mb:.0f} MB)")

    gen = _iter_solid(
        sm, x_scale, y_scale, z_scale, base_mm,
        origin_x=0.0, origin_y=0.0,
        h_min_global=float(sm.min()),
    )

    with StreamingSTL(output_path, n_tris) as stl:
        for v0, v1, v2 in tqdm(gen, total=n_tris, desc="  Writing STL",
                                unit="tri", ncols=80):
            stl.write_triangle(v0, v1, v2)

    print(f"  Saved → {output_path}")


# ── Mosaic STLs ───────────────────────────────────────────────────────────────

def generate_mosaic_stl(
    heightmap: np.ndarray,
    max_x_mm: float,
    max_y_mm: float,
    max_z_mm: float,
    tile_x_mm: float,
    tile_y_mm: float,
    base_mm: float,
    smooth_sigma: float,
    output_dir: str,
    max_vertices: int = 2000,
    existing_tiles: Optional[set] = None,
) -> None:
    """
    Generate a mosaic of tiled STL files.

    existing_tiles : set of (tz, tx) already written — those are skipped.
    All tiles share the same h_min_global so Z heights are consistent.
    """
    print(f"\n[Mosaic STLs]")

    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)
    sm = downsample(sm, max_vertices)
    rows, cols = sm.shape

    h_min_global = float(sm.min())
    h_range = float(sm.max() - sm.min())
    if h_range < 1e-6:
        h_range = 1.0

    x_scale = max_x_mm / (cols - 1)
    y_scale = max_y_mm / (rows - 1)
    z_scale = max_z_mm / h_range

    # Tile vertex counts (include shared boundary vertex)
    tile_cols = max(2, int(round(tile_x_mm / x_scale)) + 1)
    tile_rows = max(2, int(round(tile_y_mm / y_scale)) + 1)
    stride_c  = tile_cols - 1
    stride_r  = tile_rows - 1

    n_tiles_x = math.ceil((cols - 1) / stride_c)
    n_tiles_z = math.ceil((rows - 1) / stride_r)

    if existing_tiles is None:
        existing_tiles = set()

    to_do = [
        (tz, tx)
        for tz in range(n_tiles_z)
        for tx in range(n_tiles_x)
        if (tz, tx) not in existing_tiles
    ]

    print(f"  Global scale : X={x_scale:.4f}  Y={y_scale:.4f}  Z={z_scale:.4f} mm/unit")
    print(f"  Tile size    : {stride_c * x_scale:.1f} × {stride_r * y_scale:.1f} mm  "
          f"({stride_c} × {stride_r} vertices)")
    print(f"  Tile grid    : {n_tiles_x} × {n_tiles_z}  "
          f"({n_tiles_x * n_tiles_z} total,  {len(existing_tiles)} already done,  "
          f"{len(to_do)} to generate)")

    os.makedirs(output_dir, exist_ok=True)

    with tqdm(to_do, desc="  Tiles", unit="tile", ncols=80) as pbar:
        for tz, tx in pbar:
            c0 = tx * stride_c
            r0 = tz * stride_r
            c1 = min(c0 + tile_cols, cols)
            r1 = min(r0 + tile_rows, rows)

            tile_hm = sm[r0:r1, c0:c1]
            TR, TC = tile_hm.shape

            n_tris = count_solid_triangles(TR, TC)
            tile_path = os.path.join(output_dir, f"tile_{tz:03d}_{tx:03d}.stl")

            gen = _iter_solid(
                tile_hm, x_scale, y_scale, z_scale, base_mm,
                origin_x=c0 * x_scale,
                origin_y=r0 * y_scale,
                h_min_global=h_min_global,
            )

            with StreamingSTL(tile_path, n_tris) as stl:
                for v0, v1, v2 in gen:
                    stl.write_triangle(v0, v1, v2)

            pbar.set_postfix_str(f"tile_{tz:03d}_{tx:03d}.stl")

    print(f"  All tiles saved in '{output_dir}/'")
