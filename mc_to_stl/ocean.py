"""
Ocean detection, masking, and micro-island removal.

Strategy
--------
1. Auto-detect sea level: prominent height peak in the lower histogram.

2. Ocean = water at-or-below sea level that is border-connected (reaches the
   map edge without crossing higher terrain).  Landlocked lakes/rivers are
   never marked ocean.

3. Optionally also mark very large landlocked water bodies as ocean.

4. Micro-island removal: connected land components smaller than
   min_land_area blocks are absorbed into the ocean mask so they don't
   show up as noise in the image or STL.

Performance notes
-----------------
scipy.ndimage.label() on a 16k×16k map takes several seconds but is
unavoidable (it's a BFS flood-fill in C).  What we avoid is calling it
more than once per function, and we replace per-component Python loops
with vectorized LUT indexing:

    sizes = np.bincount(labeled.ravel())        # O(N), all sizes at once
    keep  = np.zeros(n+1, dtype=bool)
    keep[wanted_ids] = True
    mask  = keep[labeled]                        # O(N), one array pass
"""

import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label


# ── Sea-level auto-detection ─────────────────────────────────────────────────

def detect_sea_level(heightmap: np.ndarray) -> int:
    """
    Return the most likely sea-level Y.

    Looks for the dominant (most frequent) height value in the lower 60th
    percentile of the distribution — where the flat water surface creates a
    histogram plateau.
    """
    flat = heightmap.flatten().astype(np.int32)
    lo = int(flat.min())
    hi = int(np.percentile(flat, 60))
    if lo >= hi:
        return lo
    sub = flat[(flat >= lo) & (flat <= hi)]
    counts = np.bincount(sub - lo)
    return lo + int(np.argmax(counts))


# ── Public API ────────────────────────────────────────────────────────────────

def build_ocean_mask(
    heightmap: np.ndarray,
    sea_level: int,
    min_ocean_blocks: int = 500_000,
    margin_blocks: int = 0,
) -> np.ndarray:
    """
    Return boolean mask: True = open sea.

    Rivers, lakes, and small coastal bays are NOT marked ocean.

    Parameters
    ----------
    min_ocean_blocks : Connected-component area threshold.  Landlocked water
                       bodies larger than this are also treated as ocean.
                       Set to 0 to use border-connectivity only.
    margin_blocks    : Erode the ocean mask inward by this many blocks so
                       shallow coastal pixels are kept as land.
    """
    rows, cols = heightmap.shape
    candidate = heightmap <= sea_level
    water_px = int(candidate.sum())
    print(f"    Water pixels (≤ Y={sea_level}): {water_px:,} of {rows*cols:,}  "
          f"({100.0*water_px/(rows*cols):.1f}%)")

    t0 = time.perf_counter()
    print(f"    Labeling water components …", end=" ", flush=True)
    labeled, n = label(candidate)
    print(f"{n:,} components  ({time.perf_counter()-t0:.1f}s)")

    # Component sizes in one O(N) pass
    sizes = np.bincount(labeled.ravel())   # sizes[0] = background (non-water)

    # Collect border-touching component IDs
    border_ids: set = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_ids.update(int(x) for x in np.unique(edge) if x != 0)

    # Large landlocked bodies
    if min_ocean_blocks > 0:
        large_ids = set(int(i) for i in np.where(sizes >= min_ocean_blocks)[0] if i != 0)
    else:
        large_ids = set()

    ocean_ids = border_ids | large_ids
    print(f"    Ocean components: {len(border_ids)} border-connected"
          + (f" + {len(large_ids - border_ids)} large landlocked" if large_ids else ""))

    # Build mask via LUT — O(N), no Python loop over components
    lut = np.zeros(n + 1, dtype=bool)
    for cid in ocean_ids:
        lut[cid] = True
    ocean = lut[labeled]

    if margin_blocks > 0:
        t1 = time.perf_counter()
        print(f"    Eroding ocean mask by {margin_blocks} block(s) …", end=" ", flush=True)
        struct = np.ones((margin_blocks * 2 + 1, margin_blocks * 2 + 1), dtype=bool)
        ocean = binary_erosion(ocean, structure=struct)
        print(f"{time.perf_counter()-t1:.1f}s")

    return ocean


def remove_micro_islands(
    heightmap: np.ndarray,
    ocean_mask: np.ndarray,
    sea_level: int,
    min_land_area: int,
) -> np.ndarray:
    """
    Absorb land components smaller than min_land_area blocks into the ocean.

    Returns an updated ocean_mask with small islands marked as sea.

    Parameters
    ----------
    min_land_area : Land components with fewer blocks than this are removed.
                    Typical values: 500–5000 for noise dots; 50000 for tiny
                    uninhabited islands you want to keep.
    """
    if min_land_area <= 0:
        return ocean_mask

    land = ~ocean_mask
    t0 = time.perf_counter()
    print(f"    Labeling land components …", end=" ", flush=True)
    labeled, n = label(land)
    print(f"{n:,} components  ({time.perf_counter()-t0:.1f}s)")

    if n == 0:
        return ocean_mask

    # All component sizes in one pass
    sizes = np.bincount(labeled.ravel())   # sizes[0] = ocean/background

    # IDs of components below threshold (exclude background 0)
    small_ids = np.where((sizes > 0) & (sizes < min_land_area))[0]
    small_ids = small_ids[small_ids > 0]

    if len(small_ids) == 0:
        print(f"    No micro-islands found (threshold {min_land_area:,} blocks).")
        return ocean_mask

    print(f"    Removing {len(small_ids):,} micro-island component(s) "
          f"(< {min_land_area:,} blocks each) …", end=" ", flush=True)

    # Build mask via LUT — one array pass, no Python loop
    lut = np.zeros(n + 1, dtype=bool)
    lut[small_ids] = True
    micro = lut[labeled]

    print(f"{int(micro.sum()):,} blocks absorbed.")
    return ocean_mask | micro


def remove_floating_blocks(
    heightmap: np.ndarray,
    drop_threshold: int = 100,
) -> np.ndarray:
    """
    Remove isolated high-altitude "floating block" clusters.

    These are blocks placed by map designers that hover in mid-air — they
    appear as small islands near the global altitude maximum that are
    completely surrounded by a sheer cliff (all 4-connected neighbors are
    drop_threshold+ blocks lower than the island's minimum height).

    Overhangs that are part of real terrain are NOT removed: they connect to
    at least one adjacent cell at similar altitude, so their island minimum is
    close to at least one border neighbor.

    Replacement value: the maximum height of the immediately adjacent cells
    (the "implied ground level" were the floating cluster absent).

    Parameters
    ----------
    drop_threshold : A cluster is considered floating when its minimum height
                     exceeds the maximum height of all adjacent cells by at
                     least this many blocks.  Default 100 is conservative; lower
                     values catch more artifacts but risk flagging steep peaks.
    """
    h_max = int(heightmap.max())
    # Only inspect cells that are meaningfully high — near the global maximum.
    # Use drop_threshold as the margin: anything within drop_threshold blocks
    # of h_max is a candidate.
    candidate = heightmap >= (h_max - drop_threshold)

    t0 = time.perf_counter()
    print(f"    Labeling high-altitude components (≥ Y={h_max - drop_threshold}) …",
          end=" ", flush=True)
    labeled, n = label(candidate)
    print(f"{n:,} components  ({time.perf_counter()-t0:.1f}s)")

    if n == 0:
        return heightmap

    result = heightmap.copy()
    n_removed = 0

    for cid in range(1, n + 1):
        comp = labeled == cid
        comp_min_h = int(heightmap[comp].min())

        # Border = cells immediately adjacent but outside the component
        border = binary_dilation(comp) & ~comp

        if not border.any():
            # Component touches map edge — treat as grounded
            continue

        max_border_h = int(heightmap[border].max())

        # Floating if component hangs more than drop_threshold above all neighbors
        if comp_min_h - max_border_h > drop_threshold:
            result[comp] = max_border_h
            n_removed += 1

    print(f"    Removed {n_removed} floating cluster(s) of {n} high-altitude "
          f"component(s)  (drop > {drop_threshold} blocks above all neighbors)")
    return result


def apply_ocean_mask(
    heightmap: np.ndarray,
    ocean_mask: np.ndarray,
    sea_level: int,
) -> np.ndarray:
    """Replace ocean cells with sea_level; non-ocean water keeps its height."""
    result = heightmap.copy()
    result[ocean_mask] = sea_level
    return result


def apply_polygon_masks(
    heightmap: np.ndarray,
    ocean_mask: Optional[np.ndarray],
    polygons: List,
    sea_level: int,
    world_origin: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each polygon, set all heightmap cells inside its convex hull to
    sea_level and mark them as ocean.

    Parameters
    ----------
    polygons     : list of polygons, each polygon in one of two formats:
                     new: [[x, z], [x, z], ...]
                     old: {"coordinates": [[x, z], ...]}
                   where x, z are Minecraft block coordinates.
    sea_level    : Y value assigned to masked cells.
    world_origin : (min_x_block, min_z_block) — Minecraft block coords that
                   map to heightmap pixel (row=0, col=0).

    Returns
    -------
    (heightmap_copy, ocean_mask_copy) with polygon areas modified.
    """
    from scipy.spatial import ConvexHull, Delaunay  # lazy import — not always needed

    origin_x, origin_z = world_origin
    rows, cols = heightmap.shape
    result = heightmap.copy()
    new_ocean = ocean_mask.copy() if ocean_mask is not None else np.zeros((rows, cols), dtype=bool)

    total_masked = 0
    for poly in polygons:
        # Accept both new [[x,z],...] and old {"coordinates": [[x,z],...]} format
        if isinstance(poly, dict):
            coords = poly.get("coordinates", [])
        else:
            coords = poly
        if len(coords) < 3:
            continue

        # Minecraft (x, z) → heightmap (row, col):  row = z − origin_z, col = x − origin_x
        pts = np.array([[c[1] - origin_z, c[0] - origin_x] for c in coords], dtype=float)
        print(f"      Coords sample (first 3): {coords[:3]}")
        print(f"      Pixel bbox raw: row {pts[:,0].min():.0f}..{pts[:,0].max():.0f}, "
              f"col {pts[:,1].min():.0f}..{pts[:,1].max():.0f}  "
              f"(map is {rows}×{cols})")

        try:
            hull = ConvexHull(pts)
            tri = Delaunay(pts[hull.vertices])
        except Exception:
            continue

        r_min = max(0, int(pts[:, 0].min()))
        r_max = min(rows - 1, int(pts[:, 0].max()))
        c_min = max(0, int(pts[:, 1].min()))
        c_max = min(cols - 1, int(pts[:, 1].max()))
        print(f"      Pixel bbox clamped: rows {r_min}..{r_max}, cols {c_min}..{c_max}")

        if r_max < r_min or c_max < c_min:
            print(f"    Warning: polygon entirely outside heightmap bounds — skipped.")
            continue

        rr, cc = np.mgrid[r_min:r_max + 1, c_min:c_max + 1]
        grid_pts = np.column_stack([rr.ravel(), cc.ravel()])
        inside = tri.find_simplex(grid_pts) >= 0

        r_in = rr.ravel()[inside]
        c_in = cc.ravel()[inside]
        result[r_in, c_in] = sea_level
        new_ocean[r_in, c_in] = True
        total_masked += int(inside.sum())

    print(f"    {len(polygons)} polygon(s), {total_masked:,} block(s) forced to sea level.")
    return result, new_ocean
