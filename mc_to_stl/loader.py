"""
Load a Minecraft Java Edition save directory into a full heightmap array.

Parallelism
-----------
Parsing .mca files is CPU-bound (zlib/gzip + NBT parsing).  We use
ProcessPoolExecutor so each worker gets its own Python interpreter and
bypasses the GIL.  Cache hits are resolved in the main process (fast I/O),
so workers only do real work on uncached files.

Cache layout (inside out_dir):
  .chunk_cache/r.X.Z.npy  –  dict {(cx,cz): 16×16 int32 array}
"""

import glob
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

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


def _load_cached_chunks(cache_file: str) -> Optional[Dict]:
    try:
        return np.load(cache_file, allow_pickle=True).item()
    except Exception:
        return None


def _save_cached_chunks(cache_file: str, chunks: Dict) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, chunks)


# ── Worker function (must be module-level for pickling) ───────────────────────

def _parse_worker(args: Tuple) -> Tuple[str, Dict]:
    """
    Worker entry point.  Runs in a subprocess.
    Returns (filepath, chunks_dict) so the caller can identify the result.
    """
    filepath, debug, ground_only = args
    chunks = parse_region(filepath, debug=debug, ground_only=ground_only)
    return filepath, chunks


# ── Main loader ───────────────────────────────────────────────────────────────

def load_save(
    save_path: str,
    out_dir: Optional[str] = None,
    debug: bool = False,
    ground_only: bool = False,
    use_cache: bool = True,
    n_workers: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """
    Load all region files from a Minecraft save.

    Parameters
    ----------
    out_dir     : Cache directory; resumed runs skip already-parsed files.
    ground_only : Use terrain-only heightmap (skips trees/structures).
    use_cache   : Read/write chunk cache (default True).
    n_workers   : Number of parallel processes (0 = all available CPUs).

    Returns
    -------
    heightmap : np.ndarray  shape (Z_blocks, X_blocks), float32
    meta      : dict  width_blocks, height_blocks, chunk extents, region_path
    """
    region_path = find_region_dir(save_path)
    mca_files = sorted(glob.glob(os.path.join(region_path, "r.*.*.mca")))

    if not mca_files:
        raise FileNotFoundError(f"No .mca files found in '{region_path}'")

    if n_workers <= 0:
        n_workers = multiprocessing.cpu_count()
    n_workers = min(n_workers, len(mca_files))

    print(f"  Found {len(mca_files)} region file(s) in {region_path}")
    print(f"  Workers: {n_workers}")

    all_chunks: Dict[Tuple[int, int], np.ndarray] = {}
    cached_count = 0

    # ── Separate cached vs. uncached files ────────────────────────────────
    to_parse: List[str] = []
    for fp in mca_files:
        fname = os.path.basename(fp)
        if use_cache and out_dir:
            cached = _load_cached_chunks(_cache_path(out_dir, fname))
            if cached is not None:
                all_chunks.update(cached)
                cached_count += len(cached)
                continue
        to_parse.append(fp)

    if cached_count:
        print(f"  {cached_count:,} chunks loaded from cache  "
              f"({len(mca_files) - len(to_parse)} region files skipped)")

    # ── Parse uncached files in parallel ──────────────────────────────────
    if to_parse:
        print(f"  Parsing {len(to_parse)} region file(s) with {n_workers} worker(s) …")
        args_list = [(fp, debug, ground_only) for fp in to_parse]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_parse_worker, a): a[0] for a in args_list}

            with tqdm(
                total=len(futures),
                desc="  Parsing regions",
                unit="file",
                ncols=80,
            ) as pbar:
                for future in as_completed(futures):
                    fp = futures[future]
                    fname = os.path.basename(fp)
                    try:
                        _, chunks = future.result()
                    except Exception as exc:
                        chunks = {}
                        if debug:
                            tqdm.write(f"  Warning [{fname}]: {exc}")

                    all_chunks.update(chunks)

                    # Write cache from main process (safe, single-writer)
                    if use_cache and out_dir and chunks:
                        _save_cached_chunks(_cache_path(out_dir, fname), chunks)

                    pbar.update(1)
                    pbar.set_postfix_str(fname, refresh=False)

    print(f"  Total: {len(all_chunks):,} chunks with heightmap data.")

    if not all_chunks:
        raise ValueError(
            "No valid heightmap data found. "
            "Run with --diagnose to inspect the save format."
        )

    # ── Assemble full heightmap array ─────────────────────────────────────
    positions = list(all_chunks.keys())
    min_cx = min(p[0] for p in positions)
    max_cx = max(p[0] for p in positions)
    min_cz = min(p[1] for p in positions)
    max_cz = max(p[1] for p in positions)

    width_blocks  = (max_cx - min_cx + 1) * 16
    height_blocks = (max_cz - min_cz + 1) * 16

    print(f"  Map extent : {width_blocks:,} × {height_blocks:,} blocks")

    heightmap = np.zeros((height_blocks, width_blocks), dtype=np.float32)
    filled    = np.zeros((height_blocks, width_blocks), dtype=bool)

    for (cx, cz), chunk_hm in all_chunks.items():
        row = (cz - min_cz) * 16
        col = (cx - min_cx) * 16
        heightmap[row : row + 16, col : col + 16] = chunk_hm
        filled[row : row + 16, col : col + 16] = True

    if not filled.all():
        missing = int((~filled).sum())
        print(f"  Filling {missing:,} missing block columns via nearest-neighbour.")
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
