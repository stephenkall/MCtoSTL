"""
Load a Minecraft Java Edition save directory into a full heightmap array.
"""

import glob
import os
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from .anvil import parse_region


def find_region_dir(save_path: str) -> str:
    """Locate the 'region' folder inside a save directory."""
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
        "Expected layout: <save>/region/r.X.Z.mca"
    )


def load_save(save_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load all region files from a Minecraft save.

    Returns
    -------
    heightmap : np.ndarray  shape (Z_blocks, X_blocks), dtype float32
        Surface Y-coordinate for every block column.
    meta : dict
        width_blocks  – number of blocks on the X axis
        height_blocks – number of blocks on the Z axis
        min_cx / max_cx / min_cz / max_cz – chunk coordinate extremes
        region_path   – path to the region directory used
    """
    region_path = find_region_dir(save_path)
    mca_files = sorted(glob.glob(os.path.join(region_path, "r.*.*.mca")))

    if not mca_files:
        raise FileNotFoundError(f"No .mca files found in '{region_path}'")

    print(f"  Found {len(mca_files)} region file(s) in {region_path}")

    all_chunks: Dict[Tuple[int, int], np.ndarray] = {}
    for idx, fp in enumerate(mca_files):
        print(
            f"\r  Parsing {idx + 1}/{len(mca_files)}: {os.path.basename(fp)}...",
            end="",
            flush=True,
        )
        all_chunks.update(parse_region(fp))

    print(f"\n  Loaded {len(all_chunks)} chunk(s) with heightmap data.")

    if not all_chunks:
        raise ValueError(
            "No valid heightmap data found. "
            "The save may be corrupted or use an unsupported format."
        )

    positions = list(all_chunks.keys())
    min_cx = min(p[0] for p in positions)
    max_cx = max(p[0] for p in positions)
    min_cz = min(p[1] for p in positions)
    max_cz = max(p[1] for p in positions)

    width_blocks = (max_cx - min_cx + 1) * 16
    height_blocks = (max_cz - min_cz + 1) * 16

    print(f"  Map extent : {width_blocks} × {height_blocks} blocks")
    print(
        f"  Chunk range: X [{min_cx}, {max_cx}]  Z [{min_cz}, {max_cz}]"
    )

    heightmap = np.zeros((height_blocks, width_blocks), dtype=np.float32)
    filled = np.zeros((height_blocks, width_blocks), dtype=bool)

    for (cx, cz), chunk_hm in all_chunks.items():
        row = (cz - min_cz) * 16
        col = (cx - min_cx) * 16
        heightmap[row : row + 16, col : col + 16] = chunk_hm
        filled[row : row + 16, col : col + 16] = True

    # Fill any unloaded chunks by nearest-neighbour propagation
    if not filled.all():
        missing = (~filled).sum()
        print(f"  Filling {missing} missing block columns via nearest-neighbour.")
        _, nearest = distance_transform_edt(~filled, return_indices=True)
        heightmap[~filled] = heightmap[nearest[0][~filled], nearest[1][~filled]]

    meta = {
        "width_blocks": width_blocks,
        "height_blocks": height_blocks,
        "min_cx": min_cx,
        "max_cx": max_cx,
        "min_cz": min_cz,
        "max_cz": max_cz,
        "region_path": region_path,
    }
    return heightmap, meta
