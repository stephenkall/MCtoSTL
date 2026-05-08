"""
Heightmap -> watertight solid STL mesh.

Triangles are streamed directly to disk via StreamingSTL, so the full mesh
never resides in RAM simultaneously.
"""

import math
import os
from typing import Generator, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from tqdm import tqdm

from .stl_writer import StreamingSTL, count_solid_triangles


def _v(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float32)


def downsample(hm: np.ndarray, max_verts: int) -> np.ndarray:
    """Downsample heightmap so the longest side is <= max_verts."""
    rows, cols = hm.shape
    longest = max(rows, cols)
    if longest <= max_verts:
        return hm
    factor = max_verts / longest
    new_rows = max(2, int(round(rows * factor)))
    new_cols = max(2, int(round(cols * factor)))

    sigma = 0.5 / factor
    blurred = gaussian_filter(hm.astype(np.float64), sigma=sigma)
    return zoom(blurred, (new_rows / rows, new_cols / cols), order=1).astype(np.float32)


def _uniform_scale(max_x_mm: float, max_y_mm: float, cols: int, rows: int) -> float:
    """Return the largest uniform mm/vertex scale that fits both XY bounds."""
    return min(max_x_mm / (cols - 1), max_y_mm / (rows - 1))


def _bordered_scale(
    max_x_mm: float,
    max_y_mm: float,
    cols: int,
    rows: int,
    border_width: int,
) -> float:
    """Return mm/vertex scale for terrain plus an added border ring."""
    bw = max(0, int(border_width))
    return min(max_x_mm / (cols - 1 + 2 * bw),
               max_y_mm / (rows - 1 + 2 * bw))


def _prepare_mesh_heights(
    heights: np.ndarray,
    xy_scale: float,
    z_exaggeration: float,
    ocean_mask: Optional[np.ndarray] = None,
    sea_level: Optional[float] = None,
    sea_level_offset: float = 0.0,
    max_z_mm: Optional[float] = None,
) -> Tuple[np.ndarray, float, float, str]:
    """
    Convert Minecraft Y values into mesh-height units and return z_scale.

    New mode: z_exaggeration scales terrain relief first, then sea_level_offset
    is applied unscaled. This keeps the physical basin gap visible even with
    small relief exaggeration values such as 0.05.

    Legacy mode: when max_z_mm is provided, fit the full vertical range to that
    height, preserving the older public API used by tests and callers.
    """
    sm = heights.astype(np.float32, copy=True)

    if max_z_mm is not None:
        if ocean_mask is not None:
            sm[ocean_mask] = float(sm.min())
        h_range = float(sm.max() - sm.min())
        if h_range < 1e-6:
            h_range = 1.0
        return sm, float(max_z_mm) / h_range, h_range, f"fit {float(max_z_mm):.1f} mm"

    if sea_level is not None:
        sea = float(sea_level)
        mesh_h = sea + (sm - sea) * float(z_exaggeration)
        if ocean_mask is not None:
            mesh_h[ocean_mask] = sea - float(sea_level_offset or 0.0)
        mode = (
            f"relief {float(z_exaggeration):.3g}x; "
            f"sea offset {float(sea_level_offset or 0.0):.0f} block(s) unscaled"
        )
    else:
        mesh_h = sm * float(z_exaggeration)
        if ocean_mask is not None:
            mesh_h[ocean_mask] = float(mesh_h.min())
        mode = f"relief {float(z_exaggeration):.3g}x"

    h_range = float(mesh_h.max() - mesh_h.min())
    if h_range < 1e-6:
        h_range = 1.0
    return mesh_h.astype(np.float32, copy=False), xy_scale, h_range, mode


def _iter_box(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Yield triangles for an axis-aligned rectangular solid."""
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return

    p000 = _v(x0, y0, z0)
    p100 = _v(x1, y0, z0)
    p110 = _v(x1, y1, z0)
    p010 = _v(x0, y1, z0)
    p001 = _v(x0, y0, z1)
    p101 = _v(x1, y0, z1)
    p111 = _v(x1, y1, z1)
    p011 = _v(x0, y1, z1)

    # bottom, top, front, right, back, left
    yield p000, p110, p100
    yield p000, p010, p110
    yield p001, p101, p111
    yield p001, p111, p011
    yield p000, p100, p101
    yield p000, p101, p001
    yield p100, p110, p111
    yield p100, p111, p101
    yield p110, p010, p011
    yield p110, p011, p111
    yield p010, p000, p001
    yield p010, p001, p011


def _border_boxes(
    terrain_x: float,
    terrain_y: float,
    wall_mm: float,
    border_top_mm: float,
) -> list[Tuple[float, float, float, float, float, float]]:
    """Return four wall boxes around the whole model."""
    if wall_mm <= 0:
        return []

    total_x = terrain_x + 2 * wall_mm
    total_y = terrain_y + 2 * wall_mm
    x0 = wall_mm
    x1 = wall_mm + terrain_x
    y0 = wall_mm
    y1 = wall_mm + terrain_y

    return [
        (0.0, x0, 0.0, total_y, 0.0, border_top_mm),
        (x1, total_x, 0.0, total_y, 0.0, border_top_mm),
        (x0, x1, 0.0, y0, 0.0, border_top_mm),
        (x0, x1, y1, total_y, 0.0, border_top_mm),
    ]


def _intersect_box_xy(
    box: Tuple[float, float, float, float, float, float],
    clip_x0: float,
    clip_x1: float,
    clip_y0: float,
    clip_y1: float,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    x0, x1, y0, y1, z0, z1 = box
    ix0 = max(x0, clip_x0)
    ix1 = min(x1, clip_x1)
    iy0 = max(y0, clip_y0)
    iy1 = min(y1, clip_y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, ix1, iy0, iy1, z0, z1


def _chain_generators(*gens):
    for gen in gens:
        yield from gen


def _iter_solid(
    heights: np.ndarray,
    xy_scale: float,
    z_scale: float,
    base_mm: float,
    origin_x: float,
    origin_y: float,
    h_min_global: float,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Yield triangles for a watertight solid."""
    rows, cols = heights.shape
    z = (heights - h_min_global) * z_scale + base_mm

    def tv(r: int, c: int) -> np.ndarray:
        return _v(origin_x + c * xy_scale, origin_y + r * xy_scale, float(z[r, c]))

    def bv(r: int, c: int) -> np.ndarray:
        return _v(origin_x + c * xy_scale, origin_y + r * xy_scale, 0.0)

    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b = tv(r, c), tv(r, c + 1)
            cd, d = tv(r + 1, c), tv(r + 1, c + 1)
            yield a, b, d
            yield a, d, cd

    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b = bv(r, c), bv(r, c + 1)
            cd, d = bv(r + 1, c), bv(r + 1, c + 1)
            yield a, d, b
            yield a, cd, d

    for r in range(rows - 1):
        t0, t1 = tv(r, 0), tv(r + 1, 0)
        b0, b1 = bv(r, 0), bv(r + 1, 0)
        yield t0, b1, b0
        yield t0, t1, b1

    for r in range(rows - 1):
        t0, t1 = tv(r, cols - 1), tv(r + 1, cols - 1)
        b0, b1 = bv(r, cols - 1), bv(r + 1, cols - 1)
        yield t0, b0, b1
        yield t0, b1, t1

    for c in range(cols - 1):
        t0, t1 = tv(0, c), tv(0, c + 1)
        b0, b1 = bv(0, c), bv(0, c + 1)
        yield t0, b0, b1
        yield t0, b1, t1

    for c in range(cols - 1):
        t0, t1 = tv(rows - 1, c), tv(rows - 1, c + 1)
        b0, b1 = bv(rows - 1, c), bv(rows - 1, c + 1)
        yield t0, b1, b0
        yield t0, t1, b1


def _prepare_resampled(
    heightmap: np.ndarray,
    smooth_sigma: float,
    max_vertices: int,
    ocean_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)
    sm = downsample(sm, max_vertices)
    rows, cols = sm.shape

    om_small = None
    if ocean_mask is not None:
        om_small = zoom(
            ocean_mask.astype(np.float32),
            (rows / heightmap.shape[0], cols / heightmap.shape[1]),
            order=0,
        ) > 0.5

    sm = sm[::-1, :]
    if om_small is not None:
        om_small = om_small[::-1, :]
    return sm, om_small


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
    sea_level: Optional[float] = None,
    sea_level_offset: float = 0.0,
    max_z_mm: Optional[float] = None,
    border_width: int = 0,
) -> None:
    """Generate one watertight STL for the entire terrain."""
    print("\n[Single STL]")

    sm, om_small = _prepare_resampled(heightmap, smooth_sigma, max_vertices, ocean_mask)
    rows, cols = sm.shape
    bw = max(0, int(border_width))
    xy_scale = _bordered_scale(max_x_mm, max_y_mm, cols, rows, bw)
    sm, z_scale, h_range, z_mode = _prepare_mesh_heights(
        sm, xy_scale, z_exaggeration, om_small, sea_level,
        sea_level_offset=sea_level_offset, max_z_mm=max_z_mm,
    )

    terrain_x = (cols - 1) * xy_scale
    terrain_y = (rows - 1) * xy_scale
    wall_mm = bw * xy_scale
    actual_x = terrain_x + 2 * wall_mm
    actual_y = terrain_y + 2 * wall_mm
    actual_z = h_range * z_scale
    border_top = actual_z + base_mm + 2 * z_scale
    boxes = _border_boxes(terrain_x, terrain_y, wall_mm, border_top)
    n_tris = count_solid_triangles(rows, cols) + 12 * len(boxes)
    size_mb = n_tris * 50 / 1_048_576

    print(f"  Mesh        : {cols} x {rows} vertices")
    print(f"  XY scale    : {xy_scale:.4f} mm/vertex  (aspect preserved)")
    print(f"  Z scale     : {z_scale:.4f} mm/unit  ({z_mode}, range {h_range:.1f})")
    if bw:
        print(f"  Border      : {bw} layer(s), {wall_mm:.2f} mm thick, "
              f"{border_top:.2f} mm high")
    print(f"  Dimensions  : {actual_x:.1f} x {actual_y:.1f} x "
          f"{max(actual_z + base_mm, border_top if bw else 0.0):.1f} mm")
    print(f"  Triangles   : {n_tris:,}  (~{size_mb:.0f} MB)")

    with StreamingSTL(output_path, n_tris) as stl:
        terrain_gen = _iter_solid(
            sm, xy_scale, z_scale, base_mm,
            origin_x=wall_mm, origin_y=wall_mm,
            h_min_global=float(sm.min()),
        )
        border_gen = (tri for box in boxes for tri in _iter_box(*box))
        gen = _chain_generators(terrain_gen, border_gen)
        for v0, v1, v2 in tqdm(gen, total=n_tris, desc="  Writing STL", unit="tri", ncols=80):
            stl.write_triangle(v0, v1, v2)

    print(f"  Saved -> {output_path}")


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
    sea_level: Optional[float] = None,
    sea_level_offset: float = 0.0,
    max_z_mm: Optional[float] = None,
    border_width: int = 0,
) -> None:
    """Generate a mosaic of tiled STL files."""
    print("\n[Mosaic STLs]")

    sm, om_small = _prepare_resampled(heightmap, smooth_sigma, max_vertices, ocean_mask)
    rows, cols = sm.shape
    bw = max(0, int(border_width))
    xy_scale = _bordered_scale(max_x_mm, max_y_mm, cols, rows, bw)
    sm, z_scale, h_range, z_mode = _prepare_mesh_heights(
        sm, xy_scale, z_exaggeration, om_small, sea_level,
        sea_level_offset=sea_level_offset, max_z_mm=max_z_mm,
    )
    h_min_global = float(sm.min())

    terrain_x = (cols - 1) * xy_scale
    terrain_y = (rows - 1) * xy_scale
    wall_mm = bw * xy_scale
    total_cols = cols + 2 * bw
    total_rows = rows + 2 * bw
    total_x = (total_cols - 1) * xy_scale
    total_y = (total_rows - 1) * xy_scale
    border_top = h_range * z_scale + base_mm + 2 * z_scale
    global_boxes = _border_boxes(terrain_x, terrain_y, wall_mm, border_top)

    tile_cols = max(2, int(round(tile_x_mm / xy_scale)) + 1)
    tile_rows = max(2, int(round(tile_y_mm / xy_scale)) + 1)
    stride_c = tile_cols - 1
    stride_r = tile_rows - 1

    n_tiles_x = math.ceil((total_cols - 1) / stride_c)
    n_tiles_z = math.ceil((total_rows - 1) / stride_r)

    if existing_tiles is None:
        existing_tiles = set()

    def tile_parts(tz: int, tx: int):
        c0 = tx * stride_c
        r0 = tz * stride_r
        c1 = min(c0 + tile_cols - 1, total_cols - 1)
        r1 = min(r0 + tile_rows - 1, total_rows - 1)

        overlap_c0 = max(c0, bw)
        overlap_c1 = min(c1, bw + cols - 1)
        overlap_r0 = max(r0, bw)
        overlap_r1 = min(r1, bw + rows - 1)
        terrain = None
        terrain_origin = None
        terrain_om = None
        if overlap_c1 > overlap_c0 and overlap_r1 > overlap_r0:
            tc0 = overlap_c0 - bw
            tc1 = overlap_c1 - bw
            tr0 = overlap_r0 - bw
            tr1 = overlap_r1 - bw
            terrain = sm[tr0:tr1 + 1, tc0:tc1 + 1]
            terrain_origin = (overlap_c0 * xy_scale, overlap_r0 * xy_scale)
            if om_small is not None:
                terrain_om = om_small[tr0:tr1 + 1, tc0:tc1 + 1]

        clip_x0 = c0 * xy_scale
        clip_x1 = c1 * xy_scale
        clip_y0 = r0 * xy_scale
        clip_y1 = r1 * xy_scale
        boxes = [
            clipped
            for box in global_boxes
            if (clipped := _intersect_box_xy(box, clip_x0, clip_x1, clip_y0, clip_y1))
            is not None
        ]
        return terrain, terrain_origin, terrain_om, boxes

    to_do = []
    ocean_skipped = 0
    empty_skipped = 0
    for tz in range(n_tiles_z):
        for tx in range(n_tiles_x):
            if (tz, tx) in existing_tiles:
                continue
            terrain, _, terrain_om, boxes = tile_parts(tz, tx)
            if terrain is None and not boxes:
                empty_skipped += 1
                continue
            if terrain is not None and not boxes and terrain_om is not None and skip_ocean:
                if terrain_om.all():
                    ocean_skipped += 1
                    continue
            to_do.append((tz, tx))

    print(f"  Global scale : XY={xy_scale:.4f}  Z={z_scale:.4f} mm/unit  ({z_mode}, aspect preserved)")
    if bw:
        print(f"  Border       : {bw} layer(s), {wall_mm:.2f} mm thick, "
              f"{border_top:.2f} mm high")
    print(f"  Full model   : {total_x:.1f} x {total_y:.1f} mm")
    print(f"  Tile size    : {stride_c * xy_scale:.1f} x {stride_r * xy_scale:.1f} mm  "
          f"({stride_c} x {stride_r} vertices)")
    print(f"  Tile grid    : {n_tiles_x} x {n_tiles_z}  "
          f"({n_tiles_x * n_tiles_z} total,  {len(existing_tiles)} already done,  "
          f"{ocean_skipped} ocean-only skipped,  {empty_skipped} empty skipped,  "
          f"{len(to_do)} to generate)")

    os.makedirs(output_dir, exist_ok=True)

    with tqdm(to_do, desc="  Tiles", unit="tile", ncols=80) as pbar:
        for tz, tx in pbar:
            terrain, terrain_origin, _, boxes = tile_parts(tz, tx)
            n_tris = 12 * len(boxes)
            if terrain is not None:
                tile_rows_actual, tile_cols_actual = terrain.shape
                n_tris += count_solid_triangles(tile_rows_actual, tile_cols_actual)
            tile_path = os.path.join(output_dir, f"tile_{tz:03d}_{tx:03d}.stl")

            with StreamingSTL(tile_path, n_tris) as stl:
                gens = []
                if terrain is not None and terrain_origin is not None:
                    gens.append(_iter_solid(
                        terrain, xy_scale, z_scale, base_mm,
                        origin_x=terrain_origin[0],
                        origin_y=terrain_origin[1],
                        h_min_global=h_min_global,
                    ))
                gens.append((tri for box in boxes for tri in _iter_box(*box)))
                for v0, v1, v2 in _chain_generators(*gens):
                    stl.write_triangle(v0, v1, v2)

            pbar.set_postfix_str(f"tile_{tz:03d}_{tx:03d}.stl")

    print(f"  All tiles saved in '{output_dir}/'")
