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
"""

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


# ── Connected-component helpers ───────────────────────────────────────────────

def _components_touching_border(mask: np.ndarray) -> np.ndarray:
    """Return boolean mask of all True cells whose component touches the edge."""
    labeled, n = label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    border_ids: set = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        for cid in np.unique(edge):
            if cid != 0:
                border_ids.add(int(cid))
    result = np.zeros_like(mask, dtype=bool)
    for cid in border_ids:
        result |= labeled == cid
    return result


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
    candidate = heightmap <= sea_level

    # Border-connected water → open sea
    ocean = _components_touching_border(candidate)

    # Very large landlocked water bodies → also ocean
    if min_ocean_blocks > 0:
        labeled, n = label(candidate)
        for cid in range(1, n + 1):
            comp = labeled == cid
            if comp.sum() >= min_ocean_blocks:
                ocean |= comp

    if margin_blocks > 0:
        struct = np.ones(
            (margin_blocks * 2 + 1, margin_blocks * 2 + 1), dtype=bool
        )
        ocean = binary_erosion(ocean, structure=struct)

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
    labeled, n = label(land)

    new_mask = ocean_mask.copy()
    for cid in range(1, n + 1):
        comp = labeled == cid
        if int(comp.sum()) < min_land_area:
            new_mask |= comp   # absorb into ocean

    return new_mask


def apply_ocean_mask(
    heightmap: np.ndarray,
    ocean_mask: np.ndarray,
    sea_level: int,
) -> np.ndarray:
    """Replace ocean cells with sea_level; non-ocean water keeps its height."""
    result = heightmap.copy()
    result[ocean_mask] = sea_level
    return result
