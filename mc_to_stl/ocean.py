"""
Ocean detection and masking.

Strategy
--------
1. Auto-detect sea level: the most common height value in the lower half of
   the height distribution.  For Minecraft worlds, the water surface is a
   flat plateau that shows up as a prominent peak in the histogram.

2. Flood-fill from the map border: ocean cells are at-or-below sea level AND
   reachable from the map edge without crossing higher terrain.  This robustly
   identifies the sea while ignoring landlocked lakes and rivers.

3. Erode the ocean mask slightly so narrow coastal shallows and river mouths
   are not accidentally treated as ocean (optional, controlled by margin_blocks).
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, label


# ── Sea-level auto-detection ─────────────────────────────────────────────────

def detect_sea_level(heightmap: np.ndarray) -> int:
    """
    Return the most likely sea-level Y by finding the dominant height value
    in the lower half of the height distribution.
    """
    flat = heightmap.flatten().astype(np.int32)
    # Work within [min, median] to focus on flat water areas
    lo = int(flat.min())
    hi = int(np.percentile(flat, 60))   # 60th percentile keeps us in low zone

    if lo >= hi:
        return lo

    sub = flat[(flat >= lo) & (flat <= hi)]
    counts = np.bincount(sub - lo)
    return lo + int(np.argmax(counts))


# ── Ocean mask ───────────────────────────────────────────────────────────────

def _border_connected_components(candidate: np.ndarray) -> np.ndarray:
    """
    Return a mask of all True cells in `candidate` that belong to a connected
    component touching the map border (4-connectivity).
    """
    from scipy.ndimage import label as _label

    labeled, n = _label(candidate)
    if n == 0:
        return np.zeros_like(candidate, dtype=bool)

    # Which component IDs touch the border?
    border_ids = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        for cid in np.unique(edge):
            if cid != 0:
                border_ids.add(int(cid))

    ocean = np.zeros_like(candidate, dtype=bool)
    for cid in border_ids:
        ocean |= labeled == cid
    return ocean


def build_ocean_mask(
    heightmap: np.ndarray,
    sea_level: int,
    min_ocean_blocks: int = 500_000,
    margin_blocks: int = 0,
) -> np.ndarray:
    """
    Return a boolean mask where True = ocean (large open-sea water body).

    Rivers, lakes, and small coastal bays are NOT marked as ocean.

    Parameters
    ----------
    heightmap       : 2-D float/int heightmap array
    sea_level       : Y value of the water surface
    min_ocean_blocks: connected-component area threshold; components smaller
                      than this are kept as rivers/lakes (not masked out).
                      For WesterosEssos-scale maps a value of 500_000–2_000_000
                      works well.  Set to 0 to disable (border-flood only).
    margin_blocks   : erode the ocean mask inward by this many blocks so that
                      shallow coastal pixels are retained as "land".
    """
    # Candidate water: at or below sea level
    candidate = heightmap <= sea_level

    # ── Method A: border-connected components ─────────────────────────────
    # Ocean is water reachable from the map edge.  This naturally excludes
    # landlocked lakes no matter how large.
    ocean = _border_connected_components(candidate)

    # ── Method B: area threshold (belt-and-suspenders) ────────────────────
    # Additionally mark very large landlocked water bodies as ocean.
    if min_ocean_blocks > 0:
        labeled, n = label(candidate)
        for comp_id in range(1, n + 1):
            comp = labeled == comp_id
            if comp.sum() >= min_ocean_blocks:
                ocean |= comp

    # ── Coastal margin erosion ────────────────────────────────────────────
    if margin_blocks > 0:
        struct = np.ones((margin_blocks * 2 + 1, margin_blocks * 2 + 1), dtype=bool)
        ocean = binary_erosion(ocean, structure=struct)

    return ocean


# ── Apply mask to heightmap ───────────────────────────────────────────────────

def apply_ocean_mask(
    heightmap: np.ndarray,
    ocean_mask: np.ndarray,
    sea_level: int,
) -> np.ndarray:
    """
    Replace ocean cells with sea_level so they form a flat base in the output.
    Non-ocean water (rivers, lakes) keeps its original height.
    """
    result = heightmap.copy()
    result[ocean_mask] = sea_level
    return result
