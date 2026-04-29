"""
Heightmap → watertight solid STL mesh.

Coordinate conventions (matching standard 3-D printing orientation):
  X  =  East  (block column index)
  Y  =  North  (block row / Z-in-Minecraft index)
  Z  =  Up     (Minecraft Y / surface height)

The solid comprises:
  • Top surface  – triangulated terrain surface
  • Bottom face  – flat plate at Z = 0
  • Four side walls connecting terrain edges to the base

Winding order follows the right-hand rule so outward normals point away
from the solid interior.
"""

import math
import numpy as np
from scipy.ndimage import gaussian_filter

from .stl_writer import write_binary_stl


# ── Triangle helpers ─────────────────────────────────────────────────────────

def _v(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float32)


def _quad_top(a, b, c, d) -> list:
    """Two CCW triangles for a quad facing +Z. a=TL b=TR c=BL d=BR."""
    return [(a, b, d), (a, d, c)]


def _quad_bottom(a, b, c, d) -> list:
    """Two CW triangles for a quad facing -Z. a=TL b=TR c=BL d=BR."""
    return [(a, d, b), (a, c, d)]


# ── Core mesh builder ────────────────────────────────────────────────────────

def build_solid(
    heights: np.ndarray,
    x_scale: float,
    y_scale: float,
    z_scale: float,
    base_mm: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    h_min_global: float = None,
) -> list:
    """
    Build a watertight mesh from a height sub-array.

    Parameters
    ----------
    heights       : 2-D float array, shape (R, C)
    x_scale       : mm per block column
    y_scale       : mm per block row
    z_scale       : mm per height unit
    base_mm       : bottom plate thickness in mm (min terrain → base_mm above Z=0)
    origin_x/y    : STL world-space offset for this tile (mm)
    h_min_global  : global minimum height for consistent z across tiles;
                    if None, use the local minimum

    Returns
    -------
    list of (v0, v1, v2) triangles
    """
    R, C = heights.shape
    h_min = h_min_global if h_min_global is not None else float(heights.min())

    def tz(r: int, c: int) -> float:
        return (float(heights[r, c]) - h_min) * z_scale + base_mm

    def tv(r: int, c: int) -> np.ndarray:
        return _v(origin_x + c * x_scale, origin_y + r * y_scale, tz(r, c))

    def bv(r: int, c: int) -> np.ndarray:
        return _v(origin_x + c * x_scale, origin_y + r * y_scale, 0.0)

    tris = []

    # ── Top surface ──────────────────────────────────────────────────────
    for r in range(R - 1):
        for c in range(C - 1):
            tris.extend(_quad_top(tv(r, c), tv(r, c+1), tv(r+1, c), tv(r+1, c+1)))

    # ── Bottom face ──────────────────────────────────────────────────────
    for r in range(R - 1):
        for c in range(C - 1):
            tris.extend(_quad_bottom(bv(r, c), bv(r, c+1), bv(r+1, c), bv(r+1, c+1)))

    # ── Side walls ───────────────────────────────────────────────────────
    # Each wall is a vertical quad strip.  Winding chosen so the normal
    # points away from the solid (outward).

    # Left wall  (c=0, outward = -X)
    for r in range(R - 1):
        t0, t1 = tv(r, 0), tv(r+1, 0)
        b0, b1 = bv(r, 0), bv(r+1, 0)
        tris.extend([(t0, b1, b0), (t0, t1, b1)])

    # Right wall (c=C-1, outward = +X)
    for r in range(R - 1):
        t0, t1 = tv(r, C-1), tv(r+1, C-1)
        b0, b1 = bv(r, C-1), bv(r+1, C-1)
        tris.extend([(t0, b0, b1), (t0, b1, t1)])

    # Front wall (r=0, outward = -Y)
    for c in range(C - 1):
        t0, t1 = tv(0, c), tv(0, c+1)
        b0, b1 = bv(0, c), bv(0, c+1)
        tris.extend([(t0, b0, b1), (t0, b1, t1)])

    # Back wall  (r=R-1, outward = +Y)
    for c in range(C - 1):
        t0, t1 = tv(R-1, c), tv(R-1, c+1)
        b0, b1 = bv(R-1, c), bv(R-1, c+1)
        tris.extend([(t0, b1, b0), (t0, t1, b1)])

    return tris


# ── Public API ───────────────────────────────────────────────────────────────

def generate_single_stl(
    heightmap: np.ndarray,
    max_x_mm: float,
    max_y_mm: float,
    max_z_mm: float,
    base_mm: float,
    smooth_sigma: float,
    output_path: str,
) -> None:
    """Generate a single watertight STL for the entire terrain."""
    print(f"\n[Single STL]")
    rows, cols = heightmap.shape
    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)

    h_range = float(sm.max() - sm.min())
    if h_range < 1e-6:
        h_range = 1.0

    x_scale = max_x_mm / (cols - 1)
    y_scale = max_y_mm / (rows - 1)
    z_scale = max_z_mm / h_range

    print(f"  Map       : {cols} × {rows} blocks")
    print(f"  X scale   : {x_scale:.4f} mm/block")
    print(f"  Y scale   : {y_scale:.4f} mm/block")
    print(f"  Z scale   : {z_scale:.4f} mm/unit  (range {h_range:.0f} units)")
    print(f"  Dimensions: {(cols-1)*x_scale:.1f} × {(rows-1)*y_scale:.1f} × "
          f"{h_range*z_scale + base_mm:.1f} mm  (base {base_mm} mm)")

    tris = build_solid(sm, x_scale, y_scale, z_scale, base_mm)
    write_binary_stl(output_path, tris)
    print(f"  Triangles : {len(tris)}")
    print(f"  Saved → {output_path}")


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
) -> None:
    """
    Generate a mosaic of tiled STL files.

    All tiles share the same global Z scale so they interlock at equal heights.
    Tile filenames follow the pattern: tile_ZZ_XX.stl (row, col of the tile grid).
    """
    import os

    print(f"\n[Mosaic STLs]")
    rows, cols = heightmap.shape
    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)

    h_min_global = float(sm.min())
    h_range = float(sm.max() - sm.min())
    if h_range < 1e-6:
        h_range = 1.0

    # Global scales (same as single STL for consistency)
    x_scale = max_x_mm / (cols - 1)
    y_scale = max_y_mm / (rows - 1)
    z_scale = max_z_mm / h_range

    # Tile size in blocks (vertices per tile edge, including shared boundary)
    tile_cols = max(2, int(round(tile_x_mm / x_scale)) + 1)
    tile_rows = max(2, int(round(tile_y_mm / y_scale)) + 1)

    # Stride = blocks per tile (one less than vertices, so tiles share edges)
    stride_c = tile_cols - 1
    stride_r = tile_rows - 1

    n_tiles_x = math.ceil((cols - 1) / stride_c)
    n_tiles_z = math.ceil((rows - 1) / stride_r)

    actual_tile_x_mm = stride_c * x_scale
    actual_tile_y_mm = stride_r * y_scale

    print(f"  Global scale : X={x_scale:.4f}  Y={y_scale:.4f}  Z={z_scale:.4f} mm/unit")
    print(f"  Tile size    : {actual_tile_x_mm:.1f} × {actual_tile_y_mm:.1f} mm "
          f"({stride_c} × {stride_r} blocks)")
    print(f"  Tile grid    : {n_tiles_x} × {n_tiles_z}  ({n_tiles_x * n_tiles_z} tiles)")

    os.makedirs(output_dir, exist_ok=True)

    total = n_tiles_x * n_tiles_z
    done = 0
    for tz in range(n_tiles_z):
        for tx in range(n_tiles_x):
            c0 = tx * stride_c
            r0 = tz * stride_r
            c1 = min(c0 + tile_cols, cols)
            r1 = min(r0 + tile_rows, rows)

            tile_hm = sm[r0:r1, c0:c1]

            # Origin in mm for this tile's lower-left corner
            ox = c0 * x_scale
            oy = r0 * y_scale

            tris = build_solid(
                tile_hm, x_scale, y_scale, z_scale, base_mm,
                origin_x=ox, origin_y=oy,
                h_min_global=h_min_global,
            )

            tile_path = os.path.join(output_dir, f"tile_{tz:03d}_{tx:03d}.stl")
            write_binary_stl(tile_path, tris)

            done += 1
            print(f"\r  Progress: {done}/{total} tiles written...", end="", flush=True)

    print(f"\n  All tiles saved in '{output_dir}/'")
