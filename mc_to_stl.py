#!/usr/bin/env python3
"""
MCtoSTL  –  Minecraft Save → Heightmap Image + STL Converter
=============================================================

All configuration is collected upfront before any processing begins, so
you can answer the prompts and walk away while the conversion runs.

Outputs (inside the chosen output directory):
  heightmap.png   – color-coded relief image
  terrain.stl     – single watertight solid
  tiles/          – mosaic of printable tiles

Flags:
  --diagnose      print chunk structure of first 2 region files and exit
  --fresh         ignore any saved checkpoint and start over
"""

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

# Ensure UTF-8 output on Windows (arrows, em-dashes, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mc_to_stl.checkpoint import Checkpoint
from mc_to_stl.loader import load_save, diagnose_save
from mc_to_stl.image import generate_image
from mc_to_stl.mesh import generate_single_stl, generate_mosaic_stl
from mc_to_stl.ocean import (
    detect_sea_level, build_ocean_mask, remove_micro_islands, apply_ocean_mask,
)


# ─── Prompt helpers ───────────────────────────────────────────────────────────

def _ask(prompt: str, default: Any, cast, minimum=None) -> Any:
    label = f"[{default}]" if default != "" else ""
    while True:
        raw = input(f"  {prompt} {label}: ").strip()
        if raw == "" and default != "":
            return default
        try:
            val = cast(raw) if raw else default
            if minimum is not None and val < minimum:
                print(f"    ✗  Must be ≥ {minimum}.")
                continue
            return val
        except (ValueError, TypeError):
            print("    ✗  Invalid input.")


def _ask_float(p, d, mn=1e-9):  return _ask(p, d, float, mn)
def _ask_int(p, d, mn=1):       return _ask(p, d, int,   mn)
def _ask_str(p, d=""):          return _ask(p, d, str)


def _ask_bool(prompt: str, default: bool = True) -> bool:
    sfx = "Y/n" if default else "y/N"
    while True:
        raw = input(f"  {prompt} [{sfx}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    ✗  Please type y or n.")


def _ask_path(prompt: str, default: str = "") -> str:
    sfx = f" [{default}]" if default else ""
    while True:
        raw = input(f"  {prompt}{sfx}: ").strip()
        if raw == "" and default:
            return default
        if raw:
            return os.path.expanduser(raw)
        print("    ✗  Please enter a path.")


def _sec(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def _banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         MCtoSTL  –  Minecraft  →  STL  Converter            ║")
    print("╚══════════════════════════════════════════════════════════════╝")


# ─── Collect ALL parameters before touching the save ─────────────────────────

def collect_params(saved: Dict = None) -> Dict:
    """
    Ask all configuration questions upfront.
    If `saved` is provided (resume), pre-fill answers with saved values.
    """
    p = saved or {}

    def d(key, fallback):
        return p.get(key, fallback)

    _sec("Minecraft Save Folder")
    save_path = _ask_path("Path to Minecraft save folder",
                          default=d("save_path", ""))
    if not os.path.isdir(save_path):
        print(f"\n  ✗  '{save_path}' is not a directory.")
        sys.exit(1)

    _sec("Output")
    out_dir = _ask_path("Output directory", default=d("out_dir", "mc_output"))

    _sec("Performance")
    import multiprocessing as _mp
    _cpu = _mp.cpu_count()
    print(f"  System has {_cpu} CPU core(s).")
    print(f"  Each worker parses one .mca region file in parallel.")
    print(f"  After the first run, results are cached — subsequent runs are instant.")
    n_workers = _ask_int(
        f"Parallel workers for chunk loading",
        d("n_workers", _cpu),
        mn=1,
    )

    _sec("Block Filtering")
    print("  ground_only = YES  →  ignore trees, plants, man-made structures")
    print("                        (uses MOTION_BLOCKING_NO_LEAVES or scans sections)")
    print("  ground_only = NO   →  use WORLD_SURFACE (faster, includes everything)")
    ground_only = _ask_bool("Ground-only heightmap?", default=d("ground_only", True))

    _sec("Anti-aliasing  (Gaussian blur — applied to all outputs)")
    print("  Recommended: 1.0–2.0  |  0 = none  |  higher = smoother")
    smooth_sigma = _ask_float("Gaussian sigma", d("smooth_sigma", 1.5), mn=0.0)

    _sec("Ocean / Sea Masking")
    print("  Large open-sea areas will be flattened so the continent fills")
    print("  the output dimensions.  Rivers and lakes are always kept.")
    mask_ocean = _ask_bool("Mask out ocean?", default=d("mask_ocean", True))

    sea_level = d("sea_level", 63)
    min_ocean_blocks = d("min_ocean_blocks", 500_000)
    min_land_area = d("min_land_area", 2_000)

    if mask_ocean:
        print(f"  (Sea level will be auto-detected from the heightmap.)")
        print(f"  Override sea level Y — leave blank to auto-detect:")
        sea_level_input = input(f"  Sea level Y [auto]: ").strip()
        sea_level = int(sea_level_input) if sea_level_input else None   # None = auto
        min_ocean_blocks = _ask_int(
            "Min ocean area (blocks²) — larger = keep more small seas as land",
            d("min_ocean_blocks", 500_000), mn=1,
        )
        min_land_area = _ask_int(
            "Min island area (blocks²) — smaller islands removed as noise",
            d("min_land_area", 2_000), mn=0,
        )

    _sec("Floating Block Removal")
    print("  When enabled, chunk parsing uses full 3D connectivity analysis:")
    print("  blocks are only included if they are connected (6-directional)")
    print("  to the bottom section of their chunk.  Floating platforms,")
    print("  isolated artefacts, and hanging blocks are discarded at any")
    print("  altitude — no height threshold.  Overhangs are safe: they")
    print("  connect to their cliff face through horizontal neighbours.")
    print("  Note: bypasses pre-computed Heightmaps; ~2-3x slower per chunk.")
    detect_floating = _ask_bool("Remove floating block artefacts (3D)?",
                                default=d("detect_floating", False))

    _sec("Heightmap Image")
    print("  Color: sea level = green, max altitude = red, below sea = blue")
    print("  Gamma < 1.0 amplifies low-relief areas (0.5–0.7 recommended for flat maps)")
    max_px_w  = _ask_int("Max image width   (px)", d("max_px_w", 4096))
    max_px_h  = _ask_int("Max image height  (px)", d("max_px_h", 4096))
    gamma     = _ask_float("Relief gamma", d("gamma", 0.6), mn=0.05)

    _sec("STL Physical Dimensions")
    print("  These are MAXIMUM bounds — aspect ratio is preserved.")
    print("  A 2000×1400 mm limit on a square map produces ~1400×1400 mm.")
    print("  Z scale is independent of XY.")
    max_x_mm  = _ask_float("Max X  (mm)", d("max_x_mm", 200.0))
    max_y_mm  = _ask_float("Max Y  (mm)", d("max_y_mm", 200.0))
    max_z_mm  = _ask_float("Max Z  (altitude, mm)", d("max_z_mm", 30.0))
    base_mm   = _ask_float("Base plate thickness (mm)", d("base_mm", 2.0))

    _sec("STL Mesh Resolution")
    print("  Max vertices on the longest side.  Higher = more detail but larger file.")
    print("  Recommended: 1000–2000 for most printers; 500 for quick preview.")
    max_verts = _ask_int("Max vertices (longest side)", d("max_verts", 1500))

    _sec("Mosaic Tile Dimensions")
    print(f"  Full model fits within {max_x_mm} × {max_y_mm} mm (aspect preserved).")
    tile_x_mm  = _ask_float("Tile width  (mm)", d("tile_x_mm", min(100.0, max_x_mm)))
    tile_y_mm  = _ask_float("Tile height (mm)", d("tile_y_mm", min(100.0, max_y_mm)))
    print("  When ocean masking is on, ocean areas in every tile (including")
    print("  coastal ones) print as base-plate only — no raised sea surface.")
    skip_ocean_stl = _ask_bool(
        "Also skip tiles that are 100% ocean?",
        default=d("skip_ocean_stl", True),
    )

    return dict(
        save_path=save_path, out_dir=out_dir,
        n_workers=n_workers,
        ground_only=ground_only,
        smooth_sigma=smooth_sigma,
        mask_ocean=mask_ocean,
        sea_level=sea_level,
        min_ocean_blocks=min_ocean_blocks,
        min_land_area=min_land_area,
        detect_floating=detect_floating,
        max_px_w=max_px_w, max_px_h=max_px_h,
        gamma=gamma,
        max_x_mm=max_x_mm, max_y_mm=max_y_mm, max_z_mm=max_z_mm,
        base_mm=base_mm,
        max_verts=max_verts,
        tile_x_mm=tile_x_mm, tile_y_mm=tile_y_mm,
        skip_ocean_stl=skip_ocean_stl,
    )


# ─── Processing stages ────────────────────────────────────────────────────────

def stage_load(cp: Checkpoint, params: Dict):
    """Load or resume heightmap from save."""
    if cp.is_done("loaded"):
        print("\n[Resume] Loading cached heightmap …")
        hm = cp.load_raw_heightmap()
        if hm is not None:
            print(f"  Heightmap: {hm.shape[1]} × {hm.shape[0]} blocks  "
                  f"Y={hm.min():.0f}..{hm.max():.0f}")
            return hm
        print("  Cache missing — re-loading from save.")

    _sec("Loading Save")
    try:
        hm, meta = load_save(
            params["save_path"],
            out_dir=params["out_dir"],
            ground_only=params["ground_only"],
            use_cache=True,
            n_workers=params.get("n_workers", 0),
            detect_floating=params.get("detect_floating", False),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ✗  {exc}")
        print("  Tip: run with --diagnose to inspect the save format.")
        sys.exit(1)

    print(f"  Height range: Y={hm.min():.0f} .. Y={hm.max():.0f}")
    cp.save_raw_heightmap(hm)
    cp.cleanup_after_load()   # chunk cache no longer needed
    return hm


def stage_process(cp: Checkpoint, params: Dict, hm_raw):
    """Apply ocean masking and island filtering."""
    if cp.is_done("processed"):
        print("\n[Resume] Loading cached processed heightmap …")
        hm_work, ocean_mask = cp.load_work_heightmap()
        if hm_work is not None:
            return hm_work, ocean_mask
        print("  Cache missing — re-processing.")

    _sec("Processing Heightmap")

    if not params["mask_ocean"]:
        cp.save_work_heightmap(hm_raw, None)
        return hm_raw, None

    # Auto-detect sea level if user left it as None
    sea_level = params["sea_level"]
    if sea_level is None:
        sea_level = detect_sea_level(hm_raw)
        print(f"  Auto-detected sea level: Y={sea_level}")
        # Persist for resume
        params["sea_level"] = sea_level
        cp.set_params(params)

    print(f"  Sea level: Y={sea_level}")
    print(f"  Map size : {hm_raw.shape[1]:,} × {hm_raw.shape[0]:,} blocks "
          f"({hm_raw.size:,} total)")

    print(f"\n  [Ocean mask]")
    t0 = time.perf_counter()
    ocean_mask = build_ocean_mask(
        hm_raw,
        sea_level=sea_level,
        min_ocean_blocks=params["min_ocean_blocks"],
    )
    pct = 100.0 * ocean_mask.sum() / ocean_mask.size
    print(f"    → {pct:.1f}% ocean  ({time.perf_counter()-t0:.1f}s total)")

    if params["min_land_area"] > 0:
        print(f"\n  [Micro-island filter]")
        t1 = time.perf_counter()
        before = int(ocean_mask.sum())
        ocean_mask = remove_micro_islands(
            hm_raw, ocean_mask,
            sea_level=sea_level,
            min_land_area=params["min_land_area"],
        )
        removed = int(ocean_mask.sum()) - before
        print(f"    → {removed:,} block(s) absorbed  ({time.perf_counter()-t1:.1f}s)")

    hm_work = apply_ocean_mask(hm_raw, ocean_mask, sea_level=sea_level)

    cp.save_work_heightmap(hm_work, ocean_mask)
    cp.cleanup_after_process()   # raw heightmap no longer needed
    return hm_work, ocean_mask


def stage_image(cp: Checkpoint, params: Dict, hm_work, ocean_mask):
    if cp.is_done("image"):
        print("\n[Resume] Heightmap image already done — skipping.")
        return
    _sec("Heightmap Image")
    sea_level = params.get("sea_level") or 0.0
    img_path = os.path.join(params["out_dir"], "heightmap.png")
    generate_image(
        hm_work,
        params["max_px_w"], params["max_px_h"],
        params["smooth_sigma"], img_path,
        sea_level=float(sea_level),
        ocean_mask=ocean_mask,
        gamma=params["gamma"],
    )
    cp.mark_done("image")


def stage_stl(cp: Checkpoint, params: Dict, hm_work, ocean_mask):
    if cp.is_done("stl"):
        print("\n[Resume] terrain.stl already done — skipping.")
        return
    stl_path = os.path.join(params["out_dir"], "terrain.stl")
    generate_single_stl(
        hm_work,
        params["max_x_mm"], params["max_y_mm"], params["max_z_mm"],
        params["base_mm"], params["smooth_sigma"], stl_path,
        max_vertices=params["max_verts"],
        ocean_mask=ocean_mask,
    )
    cp.mark_done("stl")


def stage_mosaic(cp: Checkpoint, params: Dict, hm_work, ocean_mask):
    tiles_dir = os.path.join(params["out_dir"], "tiles")
    existing = cp.existing_tiles(tiles_dir)
    if cp.is_done("mosaic") and not existing:
        print("\n[Resume] Mosaic already complete — skipping.")
        return
    generate_mosaic_stl(
        hm_work,
        params["max_x_mm"], params["max_y_mm"], params["max_z_mm"],
        params["tile_x_mm"], params["tile_y_mm"],
        params["base_mm"], params["smooth_sigma"], tiles_dir,
        max_vertices=params["max_verts"],
        existing_tiles=existing,
        ocean_mask=ocean_mask,
        skip_ocean=params.get("skip_ocean_stl", False),
    )
    cp.mark_done("mosaic")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    diagnose_mode = "--diagnose" in args
    force_fresh   = "--fresh"    in args
    args = [a for a in args if not a.startswith("--")]

    _banner()

    # ── Diagnose mode ─────────────────────────────────────────────────────
    if diagnose_mode:
        save = os.path.expanduser(args[0]) if args else _ask_path("Save folder")
        _sec("Diagnosing chunk format")
        diagnose_save(save)
        return

    # ── Check for existing checkpoint ─────────────────────────────────────
    # We need the output directory before we can load a checkpoint, so peek
    # at the first positional arg or ask briefly.
    resume_params = None
    cp_out_dir = None

    # Quick peek: if a save path was given, try the default output dir
    candidate_save = os.path.expanduser(args[0]) if args else None
    if candidate_save:
        candidate_out = os.path.join(os.path.dirname(candidate_save) or ".",
                                     "mc_output")
        test_cp = Checkpoint(candidate_out)
        if not force_fresh and test_cp.load():
            sp = test_cp.save_path
            if sp == candidate_save:
                done = test_cp.params.get("done", test_cp._state.get("done", []))
                ts = test_cp._state.get("timestamp", "unknown time")
                print(f"\n  Found checkpoint from {ts}")
                print(f"  Completed stages: {', '.join(done) if done else 'none'}")
                if _ask_bool("Resume previous session?", default=True):
                    resume_params = test_cp.params
                    cp_out_dir = candidate_out

    # ── Collect all parameters (possibly pre-filled from checkpoint) ──────
    params = collect_params(saved=resume_params)
    out_dir = params["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Initialise / load checkpoint ──────────────────────────────────────
    cp = Checkpoint(out_dir)
    if force_fresh:
        cp.clear()
    elif cp_out_dir == out_dir and resume_params:
        cp.load()   # already loaded above, reload into this instance
    else:
        # Check again with the final out_dir
        if not force_fresh and cp.load() and cp.save_path == params["save_path"]:
            done = cp._state.get("done", [])
            if done:
                print(f"\n  Existing checkpoint found ({', '.join(done)} done).")
                if not _ask_bool("Resume?", default=True):
                    cp.clear()

    params["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    cp.set_params(params)

    # ── Run stages ────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print(f"  Starting conversion — outputs → {out_dir}/")
    print(f"{'═' * 62}")

    hm_raw   = stage_load(cp, params)
    hm_work, ocean_mask = stage_process(cp, params, hm_raw)
    stage_image(cp, params, hm_work, ocean_mask)
    stage_stl(cp, params, hm_work, ocean_mask)
    stage_mosaic(cp, params, hm_work, ocean_mask)
    cp.cleanup_after_outputs()   # work heightmap + ocean mask no longer needed

    # ── Summary ───────────────────────────────────────────────────────────
    img_path  = os.path.join(out_dir, "heightmap.png")
    stl_path  = os.path.join(out_dir, "terrain.stl")
    tiles_dir = os.path.join(out_dir, "tiles")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                          Done!                               ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Heightmap image  →  {os.path.relpath(img_path):<39}║")
    print(f"║  Full terrain STL →  {os.path.relpath(stl_path):<39}║")
    print(f"║  Mosaic tiles     →  {os.path.relpath(tiles_dir) + '/':<39}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
