"""
Load a Minecraft save directory into a full heightmap array.

Supports both editions:
  Java Edition   — parses .mca region files (Anvil format)
  Bedrock Edition — reads the LevelDB database in <world>/db/

Java parallelism
----------------
Parsing .mca files is CPU-bound (zlib/gzip + NBT parsing).  We use
ProcessPoolExecutor so each worker gets its own Python interpreter and
bypasses the GIL.  Cache hits are resolved in the main process (fast I/O),
so workers only do real work on uncached files.

Cache layout (inside out_dir):
  .chunk_cache/r.X.Z.npy   – Java: one file per region
  .chunk_cache/bedrock.npy  – Bedrock: full chunk dict
"""

import glob
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from .anvil import parse_region, diagnose_region
from .bedrock import parse_bedrock_world, diagnose_bedrock_world


# ── Format detection ──────────────────────────────────────────────────────────

def detect_save_format(save_path: str) -> str:
    """Return 'java' or 'bedrock'; raises FileNotFoundError if neither matches."""
    # Java: has region/ subfolder with at least one .mca file
    for candidate in [
        os.path.join(save_path, "region"),
        os.path.join(save_path, "dimensions", "minecraft", "overworld", "region"),
        os.path.join(save_path, "DIM0", "region"),
        os.path.join(save_path, "world", "region"),
    ]:
        if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, "r.*.*.mca")):
            return "java"

    # Bedrock: has db/ subfolder with LevelDB files
    db_path = os.path.join(save_path, "db")
    if os.path.isdir(db_path) and (
        os.path.exists(os.path.join(db_path, "MANIFEST")) or
        glob.glob(os.path.join(db_path, "*.ldb"))
    ):
        return "bedrock"

    raise FileNotFoundError(
        f"Cannot determine Minecraft save format for '{save_path}'.\n"
        "  Java Edition   : expects <save>/region/r.X.Z.mca files\n"
        "  Bedrock Edition: expects <save>/db/MANIFEST (LevelDB)"
    )


# ── Region directory lookup (Java only) ──────────────────────────────────────

def find_region_dir(save_path: str) -> str:
    candidates = [
        os.path.join(save_path, "region"),
        os.path.join(save_path, "dimensions", "minecraft", "overworld", "region"),
        os.path.join(save_path, "DIM0", "region"),
        os.path.join(save_path, "world", "region"),
    ]
    for path in candidates:
        if os.path.isdir(path) and glob.glob(os.path.join(path, "r.*.*.mca")):
            return path
    raise FileNotFoundError(
        f"No region folder with .mca files found in '{save_path}'.\n"
        "Expected one of:\n"
        "  <save>/region/r.X.Z.mca  (classic format)\n"
        "  <save>/dimensions/minecraft/overworld/region/r.X.Z.mca  (post-1.21 format)"
    )


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose_save(save_path: str, n_regions: int = 2, chunks_per: int = 3) -> None:
    fmt = detect_save_format(save_path)
    if fmt == "bedrock":
        print(f"  Format: Bedrock Edition (LevelDB)")
        diagnose_bedrock_world(os.path.join(save_path, "db"), max_chunks=chunks_per)
    else:
        print(f"  Format: Java Edition (Anvil)")
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

def _points_in_polygon(xs: np.ndarray, zs: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon. poly: (N,2) float array of (x,z)."""
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, zi = float(poly[i, 0]), float(poly[i, 1])
        xj, zj = float(poly[j, 0]), float(poly[j, 1])
        dz = zj - zi
        cross = (zi > zs) != (zj > zs)
        x_int = (xj - xi) * (zs - zi) / (dz if abs(dz) > 1e-12 else 1e-12) + xi
        inside ^= cross & (xs < x_int)
        j = i
    return inside


def _crop_to_polygon(
    heightmap: np.ndarray,
    meta: Dict,
    crop_poly: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """Crop heightmap to bounding box of crop_poly; pixels outside polygon set to global min."""
    min_cx = meta["min_cx"]
    min_cz = meta["min_cz"]

    poly_x, poly_z = crop_poly[:, 0], crop_poly[:, 1]
    bb_x0 = int(np.floor(poly_x.min()))
    bb_x1 = int(np.ceil(poly_x.max()))
    bb_z0 = int(np.floor(poly_z.min()))
    bb_z1 = int(np.ceil(poly_z.max()))

    rows_total, cols_total = heightmap.shape
    col0 = max(0, bb_x0 - min_cx * 16)
    col1 = min(cols_total, bb_x1 - min_cx * 16 + 1)
    row0 = max(0, bb_z0 - min_cz * 16)
    row1 = min(rows_total, bb_z1 - min_cz * 16 + 1)

    if col0 >= col1 or row0 >= row1:
        raise ValueError("Crop polygon is entirely outside the loaded map extent.")

    cropped = heightmap[row0:row1, col0:col1].copy()
    rows_c, cols_c = cropped.shape

    block_x0 = min_cx * 16 + col0
    block_z0 = min_cz * 16 + row0
    xs_grid = np.tile(np.arange(cols_c, dtype=np.float32) + block_x0, (rows_c, 1))
    zs_grid = np.tile((np.arange(rows_c, dtype=np.float32) + block_z0)[:, None], (1, cols_c))

    outside = ~_points_in_polygon(xs_grid, zs_grid, crop_poly.astype(np.float32))
    cropped[outside] = float(heightmap.min())

    new_min_cx = min_cx + col0 // 16
    new_min_cz = min_cz + row0 // 16
    new_meta = {
        **meta,
        "width_blocks": cols_c,
        "height_blocks": rows_c,
        "min_cx": new_min_cx,
        "max_cx": new_min_cx + (cols_c + 15) // 16 - 1,
        "min_cz": new_min_cz,
        "max_cz": new_min_cz + (rows_c + 15) // 16 - 1,
    }
    return cropped, new_meta


def _parse_worker(args: Tuple) -> Tuple[str, Dict]:
    """
    Worker entry point.  Runs in a subprocess.
    Returns (filepath, chunks_dict) so the caller can identify the result.
    """
    filepath, debug, ground_only, detect_floating, force_scan = args
    chunks = parse_region(filepath, debug=debug, ground_only=ground_only,
                          detect_floating=detect_floating, force_scan=force_scan)
    return filepath, chunks


# ── Heightmap assembly (shared by both editions) ──────────────────────────────

def _assemble_heightmap(
    all_chunks: Dict[Tuple[int, int], np.ndarray],
) -> Tuple[np.ndarray, Dict]:
    """Stitch chunk heightmaps into one 2-D float32 array; fill gaps via EDT."""
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
        heightmap[row:row+16, col:col+16] = chunk_hm
        filled   [row:row+16, col:col+16] = True

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
    }
    return heightmap, meta


# ── Bedrock loader ────────────────────────────────────────────────────────────

def _load_bedrock(
    save_path: str,
    out_dir: Optional[str],
    debug: bool,
    use_cache: bool,
) -> Tuple[np.ndarray, Dict]:
    db_path = os.path.join(save_path, "db")
    print(f"  Format  : Bedrock Edition")
    print(f"  DB path : {db_path}")

    cache_file = os.path.join(_cache_dir(out_dir), "bedrock.npy") if out_dir else None

    if use_cache and cache_file:
        cached = _load_cached_chunks(cache_file)
        if cached is not None:
            print(f"  {len(cached):,} chunks loaded from cache.")
            hm, meta = _assemble_heightmap(cached)
            meta["format"] = "bedrock"
            return hm, meta

    all_chunks = parse_bedrock_world(db_path, debug=debug)
    print(f"  Total: {len(all_chunks):,} chunks with heightmap data.")

    if not all_chunks:
        raise ValueError(
            "No Data2D heightmap records found in the Bedrock database.\n"
            "Run with --diagnose to inspect the world."
        )

    if use_cache and cache_file:
        _save_cached_chunks(cache_file, all_chunks)

    hm, meta = _assemble_heightmap(all_chunks)
    meta["format"] = "bedrock"
    return hm, meta


# ── Main loader ───────────────────────────────────────────────────────────────

def load_save(
    save_path: str,
    out_dir: Optional[str] = None,
    debug: bool = False,
    ground_only: bool = False,
    use_cache: bool = True,
    n_workers: int = 0,
    detect_floating: bool = False,
    force_scan: bool = False,
    crop_poly: Optional[List] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Load a Minecraft save (Java or Bedrock Edition) into a heightmap array.

    Parameters
    ----------
    out_dir        : Cache directory; resumed runs skip re-parsing.
    ground_only    : Java only — use terrain-only heightmap (skips trees/structures).
    use_cache      : Read/write chunk cache (default True).
    n_workers      : Java only — parallel worker processes (0 = all CPUs).
    detect_floating: Java only — remove floating block artefacts.
    force_scan     : Java only — bypass stored Heightmaps NBT, always scan sections.
                     Fixes stale heightmap values (holes in mountains) at the cost
                     of slower parsing.  Automatically disables chunk cache.
    crop_poly      : Optional list of 4 [x, z] corners (block coords) defining a
                     quadrilateral region of interest.  Only blocks inside this
                     polygon appear in the output; blocks outside are set to the
                     global minimum height so ocean masking absorbs them.

    Returns
    -------
    heightmap : np.ndarray  shape (Z_blocks, X_blocks), float32
    meta      : dict  width_blocks, height_blocks, chunk extents, format
    """
    if force_scan:
        use_cache = False   # stale stored heightmaps → cached data would be wrong
    fmt = detect_save_format(save_path)
    if fmt == "bedrock":
        return _load_bedrock(save_path, out_dir, debug, use_cache)

    # ── Java Edition ──────────────────────────────────────────────────────
    region_path = find_region_dir(save_path)
    mca_files = sorted(glob.glob(os.path.join(region_path, "r.*.*.mca")))

    if not mca_files:
        raise FileNotFoundError(f"No .mca files found in '{region_path}'")

    # ── Filter region files by crop polygon bounding box ─────────────────
    if crop_poly is not None:
        crop_arr = np.array(crop_poly, dtype=np.float64)
        bb_x0 = int(np.floor(crop_arr[:, 0].min()))
        bb_x1 = int(np.ceil(crop_arr[:, 0].max()))
        bb_z0 = int(np.floor(crop_arr[:, 1].min()))
        bb_z1 = int(np.ceil(crop_arr[:, 1].max()))
        filtered = []
        for fp in mca_files:
            m = re.search(r"r\.(-?\d+)\.(-?\d+)\.mca$", fp)
            if m is None:
                continue
            rx, rz = int(m.group(1)), int(m.group(2))
            if (rx * 512 + 511 >= bb_x0 and rx * 512 <= bb_x1 and
                    rz * 512 + 511 >= bb_z0 and rz * 512 <= bb_z1):
                filtered.append(fp)
        print(f"  Crop filter  : {len(filtered)}/{len(mca_files)} region files overlap crop polygon")
        mca_files = filtered
        if not mca_files:
            raise ValueError("Crop polygon does not overlap any region files in this save.")

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
        args_list = [(fp, debug, ground_only, detect_floating, force_scan) for fp in to_parse]

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

    hm, meta = _assemble_heightmap(all_chunks)
    meta["region_path"] = region_path
    meta["format"] = "java"

    if crop_poly is not None:
        crop_arr = np.array(crop_poly, dtype=np.float64)
        print(f"  Applying crop polygon …", end=" ", flush=True)
        hm, meta = _crop_to_polygon(hm, meta, crop_arr)
        print(f"done  ({meta['width_blocks']} × {meta['height_blocks']} blocks)")

    return hm, meta
