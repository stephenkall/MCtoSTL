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
A proper Gaussian pre-filter is applied before resampling to prevent
aliasing oscillations in the output mesh.

Coordinate conventions
----------------------
  X  →  block column  (East)
  Y  →  block row     (North/South = Minecraft Z)
  Z  →  surface height (Up = Minecraft Y)
"""

import math
import os
from typing import Generator, Optional, Set, Tuple

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

    Applies a Gaussian anti-aliasing filter before resampling so that
    high-frequency noise doesn't fold into low-frequency oscillations
    in the output mesh.
    """
    rows, cols = hm.shape
    longest = max(rows, cols)
    if longest <= max_verts:
        return hm
    factor = max_verts / longest
    new_rows = max(2, int(round(rows * factor)))
    new_cols = max(2, int(round(cols * factor)))

    # Anti-aliasing: blur by ~half the downsample stride before sampling
    sigma = 0.5 / factor
    blurred = gaussian_filter(hm.astype(np.float64), sigma=sigma)
    return zoom(blurred, (new_rows / rows, new_cols / cols),
                order=1).astype(np.float32)


def _uniform_scale(max_x_mm: float, max_y_mm: float,
                   cols: int, rows: int) -> float:
    """
    Return the largest uniform mm/vertex scale that fits within both bounds.
    Maintains aspect ratio — the smaller dimension may be less than its max.
    """
    return min(max_x_mm / (cols - 1), max_y_mm / (rows - 1))


# ── Streaming triangle generator ──────────────────────────────────────────────

def _iter_solid(
    heights: np.ndarray,
    xy_scale: float,
    z_scale: float,
    base_mm: float,
    origin_x: float,
    origin_y: float,
    h_min_global: float,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Yield (v0, v1, v2) triangles for a watertight solid.
    Iterates row-by-row so RAM usage is proportional to one row.
    """
    R, C = heights.shape
    h_min = h_min_global
    Z = (heights - h_min) * z_scale + base_mm

    def tv(r: int, c: int) -> np.ndarray:
        return _v(origin_x + c * xy_scale, origin_y + r * xy_scale, float(Z[r, c]))

    def bv(r: int, c: int) -> np.ndarray:
        return _v(origin_x + c * xy_scale, origin_y + r * xy_scale, 0.0)

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
    base_mm: float,
    smooth_sigma: float,
    output_path: str,
    z_exaggeration: float = 1.0,
    max_vertices: int = 2000,
    ocean_mask: Optional[np.ndarray] = None,
) -> None:
    """Generate one watertight STL for the entire terrain."""
    print(f"\n[Single STL]")

    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)
    sm = downsample(sm, max_vertices)
    rows, cols = sm.shape

    # Flatten ocean cells to global minimum so they print as base plate only,
    # not as a raised flat sea-level surface.
    om_small = None
    if ocean_mask is not None:
        om_small = zoom(ocean_mask.astype(np.float32),
                        (rows / heightmap.shape[0], cols / heightmap.shape[1]),
                        order=0) > 0.5
        sm[om_small] = float(sm.min())

    # Flip north-south so north (row 0) maps to Y=max (back of print bed),
    # matching the map image orientation (north = top = far side).
    sm = sm[::-1, :]
    if om_small is not None:
        om_small = om_small[::-1, :]

    h_range = float(sm.max() - sm.min())
    if h_range < 1e-6:
        h_range = 1.0

    xy_scale = _uniform_scale(max_x_mm, max_y_mm, cols, rows)
    z_scale  = xy_scale * z_exaggeration
    actual_x = (cols - 1) * xy_scale
    actual_y = (rows - 1) * xy_scale
    actual_z = h_range * z_scale

    n_tris   = count_solid_triangles(rows, cols)
    size_mb  = n_tris * 50 / 1_048_576

    print(f"  Mesh        : {cols} × {rows} vertices")
    print(f"  XY scale    : {xy_scale:.4f} mm/vertex  (aspect preserved)")
    print(f"  Z scale     : {z_scale:.4f} mm/unit  ({z_exaggeration:.2f}× XY,  range {h_range:.0f} blk)")
    print(f"  Dimensions  : {actual_x:.1f} × {actual_y:.1f} × "
          f"{actual_z + base_mm:.1f} mm")
    print(f"  Triangles   : {n_tris:,}  (~{size_mb:.0f} MB)")

    gen = _iter_solid(
        sm, xy_scale, z_scale, base_mm,
        origin_x=0.0, origin_y=0.0,
        h_min_global=float(sm.min()),
    )

    with StreamingSTL(output_path, n_tris) as stl:
        for v0, v1, v2 in tqdm(gen, total=n_tris, desc="  Writing STL",
                                unit="tri", ncols=80):
            stl.write_triangle(v0, v1, v2)

    print(f"  Saved -> {output_path}")


# ── Mosaic STLs ───────────────────────────────────────────────────────────────

def generate_mosaic_stl(
    heightmap: np.ndarray,
    max_x_mm: float,
    max_y_mm: float,
    tile_x_mm: float,
    tile_y_mm: float,
    base_mm: float,
    smooth_sigma: float,
    output_dir: str,
    z_exaggeration: float = 1.0,
    max_vertices: int = 2000,
    existing_tiles: Optional[Set[Tuple[int, int]]] = None,
    ocean_mask: Optional[np.ndarray] = None,
    skip_ocean: bool = False,
) -> None:
    """
    Generate a mosaic of tiled STL files.

    existing_tiles : set of (tz, tx) already written — those are skipped.
    ocean_mask     : boolean array matching heightmap; used to skip ocean tiles
                     when skip_ocean=True.
    All tiles share the same h_min_global so Z heights are consistent.
    """
    print(f"\n[Mosaic STLs]")

    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)
    sm = downsample(sm, max_vertices)
    rows, cols = sm.shape

    # Downsample ocean mask to match sm whenever provided.
    # Flatten ocean cells to global minimum so they print as base plate only —
    # the coastline ends at the shore rather than extending as a flat sea plate.
    om_small: Optional[np.ndarray] = None
    if ocean_mask is not None:
        om_small = zoom(ocean_mask.astype(np.float32),
                        (rows / heightmap.shape[0], cols / heightmap.shape[1]),
                        order=0) > 0.5
        sm[om_small] = float(sm.min())

    # Flip north-south so north (row 0) maps to Y=max (back of print bed).
    sm = sm[::-1, :]
    if om_small is not None:
        om_small = om_small[::-1, :]

    h_min_global = float(sm.min())
    h_range = float(sm.max() - sm.min())
    if h_range < 1e-6:
        h_range = 1.0

    xy_scale = _uniform_scale(max_x_mm, max_y_mm, cols, rows)
    z_scale  = xy_scale * z_exaggeration

    # Tile vertex counts (include shared boundary vertex)
    tile_cols = max(2, int(round(tile_x_mm / xy_scale)) + 1)
    tile_rows = max(2, int(round(tile_y_mm / xy_scale)) + 1)
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

    # Pre-count ocean-only tiles so we can report them
    ocean_skipped = 0
    if om_small is not None and skip_ocean:
        filtered = []
        for tz, tx in to_do:
            c0 = tx * stride_c
            r0 = tz * stride_r
            c1 = min(c0 + tile_cols, cols)
            r1 = min(r0 + tile_rows, rows)
            if om_small[r0:r1, c0:c1].all():
                ocean_skipped += 1
            else:
                filtered.append((tz, tx))
        to_do = filtered

    actual_x = (cols - 1) * xy_scale
    actual_y = (rows - 1) * xy_scale
    print(f"  Global scale : XY={xy_scale:.4f}  Z={z_scale:.4f} mm/unit  "
          f"(Z={z_exaggeration:.2f}× XY,  aspect preserved)")
    print(f"  Full model   : {actual_x:.1f} × {actual_y:.1f} mm")
    print(f"  Tile size    : {stride_c * xy_scale:.1f} × {stride_r * xy_scale:.1f} mm  "
          f"({stride_c} × {stride_r} vertices)")
    print(f"  Tile grid    : {n_tiles_x} × {n_tiles_z}  "
          f"({n_tiles_x * n_tiles_z} total,  {len(existing_tiles)} already done,  "
          f"{ocean_skipped} ocean-only skipped,  {len(to_do)} to generate)")

    os.makedirs(output_dir, exist_ok=True)

    with tqdm(to_do, desc="  Tiles", unit="tile", ncols=80) as pbar:
        for tz, tx in pbar:
            c0 = tx * stride_c
            r0 = tz * stride_r
            c1 = min(c0 + tile_cols, cols)
            r1 = min(r0 + tile_rows, rows)

            tile_hm = sm[r0:r1, c0:c1]
            TR, TC = tile_hm.shape

            n_tris   = count_solid_triangles(TR, TC)
            tile_path = os.path.join(output_dir, f"tile_{tz:03d}_{tx:03d}.stl")

            gen = _iter_solid(
                tile_hm, xy_scale, z_scale, base_mm,
                origin_x=c0 * xy_scale,
                origin_y=r0 * xy_scale,
                h_min_global=h_min_global,
            )

            with StreamingSTL(tile_path, n_tris) as stl:
                for v0, v1, v2 in gen:
                    stl.write_triangle(v0, v1, v2)

            pbar.set_postfix_str(f"tile_{tz:03d}_{tx:03d}.stl")

    print(f"  All tiles saved in '{output_dir}/'")
