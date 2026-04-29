"""
Load a Minecraft Java Edition save directory into a full heightmap array.
Supports resume: previously processed region files are cached as .npy arrays.
"""

import glob
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from .anvil import parse_region, diagnose_region


# ── Region directory lookup ───────────────────────────────────────────────────

def find_region_dir(save_path: str) -> str:
    candidates = [
        os.path.join(save_path, "region"),
        os.path.join(save_path, "DIM0", "region"),
        os.path.join(save_path, "world", "region"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"No 'region' folder found in '{save_path}'.\n"
        "Expected: <save>/region/r.X.Z.mca"
    )


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose_save(save_path: str, n_regions: int = 2, chunks_per: int = 3) -> None:
    region_path = find_region_dir(save_path)
    files = sorted(glob.glob(os.path.join(region_path, "r.*.*.mca")))[:n_regions]
    if not files:
        print("  No .mca files found.")
        return
    for fp in files:
        diagnose_region(fp, max_chunks=chunks_per)


# ── Chunk-level caching ───────────────────────────────────────────────────────

def _cache_dir(out_dir: str) -> str:
    return os.path.join(out_dir, ".chunk_cache")


def _cache_path(out_dir: str, region_filename: str) -> str:
    stem = os.path.splitext(region_filename)[0]
    return os.path.join(_cache_dir(out_dir), stem + ".npy")


def _load_cached_chunks(
    cache_file: str,
) -> Optional[Dict[Tuple[int, int], np.ndarray]]:
    """Load cached dict from a .npy file, or None if missing/corrupt."""
    try:
        data = np.load(cache_file, allow_pickle=True).item()
        return data
    except Exception:
        return None


def _save_cached_chunks(
    cache_file: str, chunks: Dict[Tuple[int, int], np.ndarray]
) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, chunks)


# ── Main loader ───────────────────────────────────────────────────────────────

def load_save(
    save_path: str,
    out_dir: Optional[str] = None,
    debug: bool = False,
    ground_only: bool = False,
    use_cache: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Load all region files from a Minecraft save.

    Parameters
    ----------
    out_dir     : If given, chunk data is cached here so interrupted runs
                  can resume without re-parsing already-processed .mca files.
    ground_only : Use terrain-only heightmap (ignores trees/structures).
    use_cache   : Whether to read/write the chunk cache (default True).

    Returns
    -------
    heightmap : np.ndarray  shape (Z_blocks, X_blocks), dtype float32
    meta      : dict with width_blocks, height_blocks, chunk extents, region_path
    """
    region_path = find_region_dir(save_path)
    mca_files = sorted(glob.glob(os.path.join(region_path, "r.*.*.mca")))

    if not mca_files:
        raise FileNotFoundError(f"No .mca files found in '{region_path}'")

    print(f"  Found {len(mca_files)} region file(s) in {region_path}")

    all_chunks: Dict[Tuple[int, int], np.ndarray] = {}
    cached_count = 0

    with tqdm(mca_files, desc="  Loading chunks", unit="region", ncols=80) as pbar:
        for fp in pbar:
            fname = os.path.basename(fp)
            pbar.set_postfix_str(fname, refresh=False)

            # Try cache first
            if use_cache and out_dir:
                cp = _cache_path(out_dir, fname)
                cached = _load_cached_chunks(cp)
                if cached is not None:
                    all_chunks.update(cached)
                    cached_count += len(cached)
                    continue

            chunks = parse_region(fp, debug=debug, ground_only=ground_only)
            all_chunks.update(chunks)

            # Save to cache
            if use_cache and out_dir and chunks:
                _save_cached_chunks(_cache_path(out_dir, fname), chunks)

    if cached_count:
        print(f"  ({cached_count} chunks from cache, remainder freshly parsed)")
    print(f"  Total: {len(all_chunks)} chunks with heightmap data.")

    if not all_chunks:
        raise ValueError(
            "No valid heightmap data found. "
            "Run with --diagnose to inspect the save format."
        )

    positions = list(all_chunks.keys())
    min_cx = min(p[0] for p in positions)
    max_cx = max(p[0] for p in positions)
    min_cz = min(p[1] for p in positions)
    max_cz = max(p[1] for p in positions)

    width_blocks  = (max_cx - min_cx + 1) * 16
    height_blocks = (max_cz - min_cz + 1) * 16

    print(f"  Map extent : {width_blocks} × {height_blocks} blocks")

    heightmap = np.zeros((height_blocks, width_blocks), dtype=np.float32)
    filled    = np.zeros((height_blocks, width_blocks), dtype=bool)

    for (cx, cz), chunk_hm in all_chunks.items():
        row = (cz - min_cz) * 16
        col = (cx - min_cx) * 16
        heightmap[row : row + 16, col : col + 16] = chunk_hm
        filled[row : row + 16, col : col + 16] = True

    if not filled.all():
        missing = int((~filled).sum())
        print(f"  Filling {missing:,} missing block columns (nearest-neighbour).")
        _, nearest = distance_transform_edt(~filled, return_indices=True)
        heightmap[~filled] = heightmap[nearest[0][~filled], nearest[1][~filled]]

    meta = {
        "width_blocks":  width_blocks,
        "height_blocks": height_blocks,
        "min_cx": min_cx, "max_cx": max_cx,
        "min_cz": min_cz, "max_cz": max_cz,
        "region_path": region_path,
    }
    return heightmap, meta
