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
from typing import Optional

import numpy as np
from scipy.ndimage import binary_erosion, label


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


def apply_ocean_mask(
    heightmap: np.ndarray,
    ocean_mask: np.ndarray,
    sea_level: int,
) -> np.ndarray:
    """Replace ocean cells with sea_level; non-ocean water keeps its height."""
    result = heightmap.copy()
    result[ocean_mask] = sea_level
    return result
