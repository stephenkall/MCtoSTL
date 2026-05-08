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

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows (arrows, em-dashes, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mc_to_stl.checkpoint import Checkpoint
from mc_to_stl.loader import load_save, diagnose_save, detect_save_format
from mc_to_stl.image import generate_image
from mc_to_stl.mesh import generate_single_stl, generate_mosaic_stl
from mc_to_stl.ocean import (
    detect_sea_level, build_ocean_mask, remove_micro_islands,
    apply_ocean_mask, apply_polygon_masks,
)


# ─── Config format helpers ────────────────────────────────────────────────────

def _parse_config(data: Dict) -> Dict:
    """
    Accept both the old flat config format and the new sectioned format.

    New format:
      { "configuration": {...}, "crop_area": [[x,z],...], "sea_masking": [[[x,z],...]] }

    Old flat format (backward compat):
      { "save_path": ..., "crop_x1": ..., "polygon_json": ..., ... }
    """
    if "configuration" in data:
        params: Dict = dict(data["configuration"])
        params["crop_area"] = data.get("crop_area")        # list of [x,z] or None
        params["sea_masking"] = data.get("sea_masking") or []
    else:
        params = dict(data)
        # Convert old individual crop corners → crop_area list
        if params.get("crop_x1") is not None:
            params["crop_area"] = [
                [params.pop("crop_x1"), params.pop("crop_z1")],
                [params.pop("crop_x2"), params.pop("crop_z2")],
                [params.pop("crop_x3"), params.pop("crop_z3")],
                [params.pop("crop_x4"), params.pop("crop_z4")],
            ]
        # Old polygon_json handled at runtime in stage_process for backward compat
        params.setdefault("sea_masking", [])
    return params


def _serialize_config(params: Dict) -> Dict:
    """Serialize params to the new unified sectioned config format."""
    skip = {"crop_area", "sea_masking", "timestamp", "polygon_json",
            "crop_x1", "crop_z1", "crop_x2", "crop_z2",
            "crop_x3", "crop_z3", "crop_x4", "crop_z4"}
    configuration = {k: v for k, v in params.items()
                     if not k.startswith("_") and k not in skip}
    return {
        "configuration": configuration,
        "crop_area": params.get("crop_area"),
        "sea_masking": params.get("sea_masking") or [],
    }


# ─── Prompt helpers ───────────────────────────────────────────────────────────

# Mutable context — set to True when running from a config file (no prompts).
_ctx: Dict[str, Any] = {"unattended": False}


def _ask(prompt: str, default: Any, cast, minimum=None) -> Any:
    if _ctx["unattended"]:
        if default == "" or default is None:
            print(f"\n  ERROR: No config value for '{prompt}' — required in unattended mode.")
            sys.exit(1)
        val = cast(default)
        if minimum is not None and val < minimum:
            print(f"\n  ERROR: Config value for '{prompt}' = {val} is below minimum {minimum}.")
            sys.exit(1)
        return val
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


def _ask_bool(prompt: str, default: bool = True) -> bool:
    if _ctx["unattended"]:
        return bool(default)
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
    if _ctx["unattended"]:
        if not default:
            print(f"\n  ERROR: No config path for '{prompt}' — required in unattended mode.")
            sys.exit(1)
        return os.path.expanduser(default)
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

def collect_params(saved: Dict = None, unattended: bool = False) -> Dict:
    """
    Ask all configuration questions upfront.
    If `saved` is provided, pre-fill answers with saved values.
    If `unattended=True`, use saved values without prompting (requires a
    complete config — missing required fields cause an immediate error).
    """
    _ctx["unattended"] = unattended
    p = saved or {}

    def d(key, fallback):
        return p.get(key, fallback)

    _sec("Minecraft Save Folder")
    save_path = _ask_path("Path to Minecraft save folder",
                          default=d("save_path", ""))
    if not os.path.isdir(save_path):
        print(f"\n  ✗  '{save_path}' is not a directory.")
        sys.exit(1)

    # Detect edition so we can skip inapplicable prompts
    try:
        save_fmt = detect_save_format(save_path)
    except FileNotFoundError as e:
        print(f"\n  ✗  {e}")
        sys.exit(1)
    is_bedrock = save_fmt == "bedrock"
    print(f"  Detected: {'Bedrock Edition' if is_bedrock else 'Java Edition'}")

    _sec("Output")
    out_dir = _ask_path("Output directory", default=d("out_dir", "mc_output"))

    # Only offer checkpoint resume in interactive mode and when no config was
    # loaded; if the user provided a config, they want those exact values.
    if saved is None:
        _cp_path = os.path.join(out_dir, ".checkpoint.json")
        if os.path.exists(_cp_path):
            try:
                with open(_cp_path, encoding="utf-8") as _f:
                    _cp_data = json.load(_f)
                _cp_params = _cp_data.get("params") or _cp_data
                if _cp_params.get("save_path") == save_path:
                    if _ask_bool("Checkpoint found - resume from where you left off?", default=True):
                        p = _cp_params
            except Exception:
                pass

    # ── Java-only: parallelism ────────────────────────────────────────────
    if not is_bedrock:
        _sec("Performance  (Java Edition)")
        import multiprocessing as _mp
        _cpu = _mp.cpu_count()
        print(f"  System has {_cpu} CPU core(s).")
        print(f"  Each worker parses one .mca region file in parallel.")
        print(f"  After the first run, results are cached — subsequent runs are instant.")
        n_workers = _ask_int(
            "Parallel workers for chunk loading",
            d("n_workers", _cpu),
            mn=1,
        )
    else:
        n_workers = 1  # Bedrock is a single sequential DB read

    # ── Java-only: block filtering ────────────────────────────────────────
    if not is_bedrock:
        _sec("Block Filtering  (Java Edition)")
        print("  ground_only = YES  →  ignore trees, plants, man-made structures")
        print("                        (uses MOTION_BLOCKING_NO_LEAVES or scans sections)")
        print("  ground_only = NO   →  use WORLD_SURFACE (faster, includes everything)")
        ground_only = _ask_bool("Ground-only heightmap?", default=d("ground_only", True))

        print("\n  Water block detection:")
        print("  detect_water_blocks = YES  →  identify actual water blocks, prevents land")
        print("                               depressions below sea level from being marked as water")
        print("  detect_water_blocks = NO   →  faster, uses only altitude (default)")
        detect_water_blocks = _ask_bool("Detect water blocks?", default=d("detect_water_blocks", False))
    else:
        ground_only = False  # Bedrock Data2D is always surface (no filtering option)
        detect_water_blocks = False

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
        if _ctx["unattended"]:
            sea_level = d("sea_level", None)
        else:
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

    # In unattended mode, always use inline sea_masking from the config.
    # In interactive mode, optionally load / replace from a file.
    sea_masking: List = d("sea_masking", []) or []
    if not _ctx["unattended"]:
        print("  Sea masking polygons: JSON file with areas to force to sea level.")
        print("  Format: [[[x,z],...], ...] OR legacy [{\"coordinates\":[[x,z],...]}]")
        print("  Leave blank to keep existing inline polygons (if any) or skip.")
        _sea_masking_file = input("  Sea masking JSON file path (blank to skip/keep): ").strip()
        if _sea_masking_file and os.path.isfile(_sea_masking_file):
            with open(_sea_masking_file, encoding="utf-8") as _f:
                _raw_polys = json.load(_f)
            sea_masking = []
            for _poly in _raw_polys:
                if isinstance(_poly, dict):
                    sea_masking.append(_poly.get("coordinates", []))
                else:
                    sea_masking.append(_poly)
            print(f"  Loaded {len(sea_masking)} polygon(s) from {_sea_masking_file}")
        elif _sea_masking_file:
            print(f"  File not found: {_sea_masking_file} — keeping existing polygons.")
        elif sea_masking:
            print(f"  Keeping {len(sea_masking)} existing inline polygon(s).")

    # ── Java-only: floating block removal ────────────────────────────────
    if not is_bedrock:
        _sec("Floating Block Removal  (Java Edition)")
        print("  When enabled, chunk parsing uses full 3D connectivity analysis:")
        print("  blocks are only included if they are connected (6-directional)")
        print("  to the bottom section of their chunk.  Floating platforms,")
        print("  isolated artefacts, and hanging blocks are discarded at any")
        print("  altitude — no height threshold.  Overhangs are safe: they")
        print("  connect to their cliff face through horizontal neighbours.")
        print("  Note: bypasses pre-computed Heightmaps; ~2-3x slower per chunk.")
        detect_floating = _ask_bool("Remove floating block artefacts (3D)?",
                                    default=d("detect_floating", False))
    else:
        detect_floating = False

    # ── Java-only: force section scan ────────────────────────────────────
    if not is_bedrock:
        _sec("Stale Heightmap Fix  (Java Edition)")
        print("  force_scan = YES  →  always scan block sections, ignore stored Heightmaps.")
        print("  Use when you see holes in mountains that don't appear in Unmined.")
        print("  Matches what Unmined does; ~2-3x slower per chunk. Disables chunk cache.")
        force_scan = _ask_bool("Force section scan (fix stale heightmaps)?",
                               default=d("force_scan", False))
    else:
        force_scan = False

    # ── Crop region ───────────────────────────────────────────────────────
    _sec("Crop Region  (optional)")
    print("  Define a quadrilateral in Minecraft block coords (X, Z).")
    print("  Only blocks inside this polygon appear in the output.")
    _existing_crop = d("crop_area", None)
    # Backward compat: old configs stored crop as individual corner keys
    if _existing_crop is None and d("crop_x1", None) is not None:
        _existing_crop = [
            [d("crop_x1", 0), d("crop_z1", 0)],
            [d("crop_x2", 0), d("crop_z2", 0)],
            [d("crop_x3", 0), d("crop_z3", 0)],
            [d("crop_x4", 0), d("crop_z4", 0)],
        ]
    enable_crop = _ask_bool("Crop to a quadrilateral region?",
                            default=_existing_crop is not None)
    if enable_crop:
        print("  Enter block coordinates for the 4 corners (can be negative).")
        _def = _existing_crop or [[0, 0], [0, 0], [0, 0], [0, 0]]
        crop_area: Optional[List] = [
            [_ask(f"Corner {i+1} X", _def[i][0], int),
             _ask(f"Corner {i+1} Z", _def[i][1], int)]
            for i in range(4)
        ]
    else:
        crop_area = None

    _sec("Output Stages")
    generate_heightmap = _ask_bool("Generate heightmap images?", default=d("generate_heightmap", True))
    generate_stl = _ask_bool("Generate single STL?", default=d("generate_stl", True))
    generate_mosaic = _ask_bool("Generate mosaic STLs?", default=d("generate_mosaic", True))

    _sec("Heightmap Image")
    print("  Use 0 for full block resolution (one pixel per Minecraft block).")
    max_px_w  = _ask_int("Max image width   (px, 0 = full)", d("max_px_w", 4096), mn=0)
    max_px_h  = _ask_int("Max image height  (px, 0 = full)", d("max_px_h", 4096), mn=0)
    print("  Rectangular crop: YES = fill outside polygon with ocean (rectangular image).")
    print("  NO = transparent background (RGBA PNG, outside polygon has alpha=0).")
    rectangular_crop = _ask_bool("Rectangular crop output?",
                                 default=d("rectangular_crop", True))

    _sec("STL Physical Dimensions")
    print("  Max X/Y are maximum bounds — aspect ratio is always preserved.")
    max_x_mm  = _ask_float("Max X  (mm)", d("max_x_mm", 200.0))
    max_y_mm  = _ask_float("Max Y  (mm)", d("max_y_mm", 200.0))
    print("  Z exaggeration: 1.0 = same scale as XY (physically accurate but very flat),")
    print("  10.0 = 10× taller. A value of 5–15 is typical for printable terrain models.")
    z_exaggeration = _ask_float("Z exaggeration factor", d("z_exaggeration", 1.0), mn=0.01)
    base_mm   = _ask_float("Base plate thickness (mm)", d("base_mm", 2.0))
    print("  Sea level drain: pretend the sea is N blocks lower than it really is.")
    print("  Shallow seafloor exposed this way prints as terrain; deep ocean stays flat.")
    print("  Fill the printed ocean basin with resin — its surface = the real sea level.")
    sea_level_offset = _ask_int(
        "Sea level drain (blocks, 0 = use actual sea level)",
        d("sea_level_offset", 0), mn=0,
    )
    print("  Resin border: 0 = none; >0 adds that many wall layers around the whole model.")
    print("  The total model size, including the border, stays within Max X/Y.")
    border_width = _ask_int("Border width (mesh layers)", d("border_width", 0), mn=0)

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
        detect_water_blocks=detect_water_blocks,
        smooth_sigma=smooth_sigma,
        mask_ocean=mask_ocean,
        sea_level=sea_level,
        min_ocean_blocks=min_ocean_blocks,
        min_land_area=min_land_area,
        detect_floating=detect_floating,
        force_scan=force_scan,
        crop_area=crop_area,
        sea_masking=sea_masking,
        generate_heightmap=generate_heightmap,
        generate_stl=generate_stl,
        generate_mosaic=generate_mosaic,
        max_px_w=max_px_w, max_px_h=max_px_h,
        rectangular_crop=rectangular_crop,
        max_x_mm=max_x_mm, max_y_mm=max_y_mm,
        z_exaggeration=z_exaggeration,
        base_mm=base_mm,
        sea_level_offset=sea_level_offset,
        border_width=border_width,
        max_verts=max_verts,
        tile_x_mm=tile_x_mm, tile_y_mm=tile_y_mm,
        skip_ocean_stl=skip_ocean_stl,
    )


# ─── Checkpoint invalidation ─────────────────────────────────────────────────

def _val_eq(a: Any, b: Any) -> bool:
    """Equality that treats 1 and 1.0 as the same (JSON int vs Python float)."""
    if a == b:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def _invalidate_stale_stages(cp: Checkpoint, params: Dict) -> None:
    """
    Compare current params against what is stored in the checkpoint.
    Unmark any downstream stages whose inputs have changed so they are
    regenerated on this run.

    Stage dependency groups:
      loaded    → save_path, ground_only, detect_floating
      processed → mask_ocean, sea_level, sea_level_offset, ocean thresholds,
                  polygon masks
      image/stl/mosaic → sigma, gamma, physical dimensions, tile sizes
    """
    old = cp.params
    if not old:
        return

    def changed(keys):
        return any(not _val_eq(old.get(k), params.get(k)) for k in keys)

    LOAD_KEYS   = {"save_path", "ground_only", "detect_floating", "force_scan",
                   "crop_area"}
    PROC_KEYS   = {"mask_ocean", "sea_level", "sea_level_offset",
                   "min_ocean_blocks", "min_land_area", "sea_masking"}
    IMG_KEYS    = {"smooth_sigma", "max_px_w", "max_px_h", "rectangular_crop"}
    STL_KEYS    = {"smooth_sigma", "max_x_mm", "max_y_mm", "z_exaggeration",
                   "base_mm", "border_width", "max_verts"}
    MOSAIC_KEYS = {"tile_x_mm", "tile_y_mm", "skip_ocean_stl"}

    msgs = []

    if changed(LOAD_KEYS):
        cp.unmark_from("loaded")
        msgs.append("save/load params changed — full re-parse needed")
    elif changed(PROC_KEYS):
        cp.unmark_from("processed")
        msgs.append("ocean/masking params changed — re-processing terrain")
    else:
        if changed(IMG_KEYS):
            cp.unmark("image")
            msgs.append("image params changed — regenerating heightmap image")
        if changed(STL_KEYS):
            cp.unmark("stl")
            msgs.append("STL params changed — regenerating terrain.stl")
        if changed(STL_KEYS | MOSAIC_KEYS):
            cp.unmark("mosaic")
            # Tile files must be deleted so generate_mosaic_stl re-creates them
            import shutil as _sh
            tiles_dir = os.path.join(cp.out_dir, "tiles")
            if os.path.isdir(tiles_dir):
                _sh.rmtree(tiles_dir, ignore_errors=True)
            msgs.append("tile params changed — regenerating mosaic tiles")

    if msgs:
        print(f"\n  [Checkpoint] {'; '.join(msgs)}.")
    elif any(cp.is_done(s) for s in cp._STAGE_ORDER):
        print(f"\n  [Checkpoint] Params unchanged — resuming from checkpoint.")


# ─── Processing stages ────────────────────────────────────────────────────────

def stage_load(cp: Checkpoint, params: Dict):
    """Load or resume heightmap from save. Returns (hm, crop_mask, water_map)."""
    _crop_mask_path = os.path.join(params.get("out_dir", ""), ".crop_mask.npy")
    _water_map_path = os.path.join(params.get("out_dir", ""), ".water_map.npy")

    if cp.is_done("loaded"):
        print("\n[Resume] Loading cached heightmap …")
        hm = cp.load_raw_heightmap()
        if hm is not None:
            print(f"  Heightmap: {hm.shape[1]} × {hm.shape[0]} blocks  "
                  f"Y={hm.min():.0f}..{hm.max():.0f}")
            for key in ("_min_cx", "_min_cz"):
                if params.get(key) is None and cp.params.get(key) is not None:
                    params[key] = cp.params[key]
            crop_mask = None
            if os.path.exists(_crop_mask_path):
                import numpy as _np
                crop_mask = _np.load(_crop_mask_path)
            water_map = None
            if os.path.exists(_water_map_path):
                import numpy as _np
                wm = _np.load(_water_map_path)
                # Only use if shape matches (old cache may have pre-crop shape)
                if wm.shape == hm.shape:
                    water_map = wm
                else:
                    print(f"  Note: cached water_map shape {wm.shape} != heightmap {hm.shape}; ignoring.")
            return hm, crop_mask, water_map
        print("  Cache missing — re-loading from save.")

    _sec("Loading Save")
    crop_poly = params.get("crop_area")  # list of [x,z] pairs or None
    detect_water = params.get("detect_water_blocks", False)
    try:
        result = load_save(
            params["save_path"],
            out_dir=params["out_dir"],
            ground_only=params["ground_only"],
            use_cache=True,
            n_workers=params.get("n_workers", 0),
            detect_floating=params.get("detect_floating", False),
            force_scan=params.get("force_scan", False),
            crop_poly=crop_poly,
            return_crop_mask=True,
            detect_water_blocks=detect_water,
        )
        # Unpack based on return type
        if detect_water:
            hm, meta, crop_mask, water_map = result
        else:
            hm, meta, crop_mask = result
            water_map = None
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ✗  {exc}")
        print("  Tip: run with --diagnose to inspect the save format.")
        sys.exit(1)

    print(f"  Height range: Y={hm.min():.0f} .. Y={hm.max():.0f}")
    params["_min_cx"] = int(meta.get("min_cx", 0))
    params["_min_cz"] = int(meta.get("min_cz", 0))
    cp.set_params(params)
    cp.save_raw_heightmap(hm)
    cp.cleanup_after_load()
    if crop_mask is not None:
        import numpy as _np
        _np.save(_crop_mask_path, crop_mask)
    if water_map is not None:
        import numpy as _np
        _np.save(_water_map_path, water_map)
    return hm, crop_mask, water_map


def stage_process(cp: Checkpoint, params: Dict, hm_raw, water_map=None):
    """Apply ocean masking, island filtering, and polygon masks."""
    if cp.is_done("processed"):
        print("\n[Resume] Loading cached processed heightmap …")
        hm_work, ocean_mask = cp.load_work_heightmap()
        if hm_work is not None:
            # Restore values resolved during a previous run.
            for key in ("sea_level", "_effective_sea_level", "_min_cx", "_min_cz"):
                if params.get(key) is None and cp.params.get(key) is not None:
                    params[key] = cp.params[key]
            return hm_work, ocean_mask
        print("  Cache missing — re-processing.")

    _sec("Processing Heightmap")

    sea_masking: List = params.get("sea_masking") or []
    # Backward compat: if old-format polygon_json is still in params, load it now
    if not sea_masking and params.get("polygon_json"):
        _pj = params["polygon_json"]
        if os.path.isfile(_pj):
            with open(_pj, encoding="utf-8") as _f:
                _raw = json.load(_f)
            sea_masking = [p.get("coordinates", p) if isinstance(p, dict) else p
                           for p in _raw]
    has_polygons = bool(sea_masking)

    # Early exit only when there is truly nothing to do
    if not params["mask_ocean"] and not has_polygons:
        cp.save_work_heightmap(hm_raw, None)
        return hm_raw, None

    # Resolve sea level (needed for ocean mask and/or polygon masks)
    sea_level = params["sea_level"]
    if sea_level is None:
        sea_level = detect_sea_level(hm_raw)
        print(f"  Auto-detected sea level: Y={sea_level}")
        params["sea_level"] = sea_level
        cp.set_params(params)

    # sea_level_offset controls the STL basin depth only:
    #   - Ocean DETECTION uses sea_level (original) so all surface water is
    #     correctly masked regardless of the drain offset.
    #   - Ocean FLATTENING uses effective_sea_level so the printed basin is
    #     offset blocks lower than sea level — fill with resin to restore.
    sea_level_offset = int(params.get("sea_level_offset", 0))
    effective_sea_level = sea_level - sea_level_offset

    if sea_level_offset > 0:
        print(f"  Sea level: Y={sea_level}  "
              f"(STL basin floor Y={effective_sea_level}, fill {sea_level_offset}-block gap with resin)")
    else:
        print(f"  Sea level: Y={sea_level}")
    print(f"  Map size : {hm_raw.shape[1]:,} × {hm_raw.shape[0]:,} blocks "
          f"({hm_raw.size:,} total)")

    ocean_mask = None

    if params["mask_ocean"]:
        print(f"\n  [Ocean mask]")
        t0 = time.perf_counter()
        ocean_mask = build_ocean_mask(
            hm_raw,
            sea_level=sea_level,          # detect at real sea level
            min_ocean_blocks=params["min_ocean_blocks"],
            water_map=water_map,           # filter by actual water blocks if available
        )
        pct = 100.0 * ocean_mask.sum() / ocean_mask.size
        print(f"    → {pct:.1f}% ocean  ({time.perf_counter()-t0:.1f}s total)")

        if params["min_land_area"] > 0:
            print(f"\n  [Micro-island filter]")
            t1 = time.perf_counter()
            before = int(ocean_mask.sum())
            ocean_mask = remove_micro_islands(
                hm_raw, ocean_mask,
                sea_level=sea_level,      # detect at real sea level
                min_land_area=params["min_land_area"],
            )
            removed = int(ocean_mask.sum()) - before
            print(f"    → {removed:,} block(s) absorbed  ({time.perf_counter()-t1:.1f}s)")

        # Flatten to effective_sea_level so the STL basin is offset blocks deeper.
        hm_work = apply_ocean_mask(hm_raw, ocean_mask, sea_level=effective_sea_level)
    else:
        hm_work = hm_raw.copy()

    params["_effective_sea_level"] = effective_sea_level
    cp.set_params(params)

    if has_polygons:
        print(f"\n  [Sea masking polygons]  ({len(sea_masking)} polygon(s))")
        world_origin = (
            params.get("_min_cx", 0) * 16,
            params.get("_min_cz", 0) * 16,
        )
        print(f"    World origin  : block X={world_origin[0]}, Z={world_origin[1]}")
        if ocean_mask is not None:
            print(f"    Ocean before  : {100.0*ocean_mask.sum()/ocean_mask.size:.1f}%")
        else:
            print(f"    Ocean before  : (no ocean mask — mask_ocean is off)")
        hm_work, ocean_mask = apply_polygon_masks(
            hm_work, ocean_mask, sea_masking, effective_sea_level, world_origin,
        )
        if ocean_mask is not None:
            print(f"    Ocean after   : {100.0*ocean_mask.sum()/ocean_mask.size:.1f}%")

    cp.save_work_heightmap(hm_work, ocean_mask)
    # Downstream outputs (image, STL, tiles) are now stale — force regeneration.
    cp.unmark_from("image")
    # Raw heightmap is intentionally kept (heightmap_raw.npy) so the user can
    # re-run with different ocean/image params without re-parsing region files.
    return hm_work, ocean_mask


def stage_image(cp: Checkpoint, params: Dict, hm_work, ocean_mask, crop_mask=None):
    if cp.is_done("image"):
        gray_path = os.path.join(params["out_dir"], "heightmap_gray.png")
        if os.path.isfile(gray_path):
            print("\n[Resume] Heightmap image already done — skipping.")
            return
        print("\n[Resume] Grayscale image missing — regenerating.")
        cp.unmark("image")
    _sec("Heightmap Image")
    _sl = params.get("sea_level") or 0
    img_path = os.path.join(params["out_dir"], "heightmap.png")
    generate_image(
        hm_work,
        params["max_px_w"], params["max_px_h"],
        params["smooth_sigma"], img_path,
        sea_level=float(_sl),
        ocean_mask=ocean_mask,
        crop_mask=crop_mask,
        rectangular_crop=params.get("rectangular_crop", True),
    )
    cp.mark_done("image")


def stage_stl(cp: Checkpoint, params: Dict, hm_work, ocean_mask):
    if cp.is_done("stl"):
        print("\n[Resume] terrain.stl already done — skipping.")
        return
    stl_path = os.path.join(params["out_dir"], "terrain.stl")
    generate_single_stl(
        hm_work,
        params["max_x_mm"], params["max_y_mm"],
        params["base_mm"], params["smooth_sigma"], stl_path,
        z_exaggeration=params.get("z_exaggeration", 1.0),
        max_vertices=params["max_verts"],
        ocean_mask=ocean_mask,
        sea_level=params.get("sea_level"),
        sea_level_offset=params.get("sea_level_offset", 0),
        border_width=params.get("border_width", 0),
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
        params["max_x_mm"], params["max_y_mm"],
        params["tile_x_mm"], params["tile_y_mm"],
        params["base_mm"], params["smooth_sigma"], tiles_dir,
        z_exaggeration=params.get("z_exaggeration", 1.0),
        max_vertices=params["max_verts"],
        existing_tiles=existing,
        ocean_mask=ocean_mask,
        skip_ocean=params.get("skip_ocean_stl", False),
        sea_level=params.get("sea_level"),
        sea_level_offset=params.get("sea_level_offset", 0),
        border_width=params.get("border_width", 0),
    )
    cp.mark_done("mosaic")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    raw_args = sys.argv[1:]
    diagnose_mode = "--diagnose" in raw_args
    force_fresh   = "--fresh"    in raw_args

    # Collect --config=path or plain path positional arg (for diagnose)
    config_arg = None
    positional_args = []
    for a in raw_args:
        if a.startswith("--config="):
            config_arg = os.path.expanduser(a[9:])
        elif not a.startswith("--"):
            positional_args.append(a)

    _banner()

    # ── Diagnose mode ─────────────────────────────────────────────────────
    if diagnose_mode:
        save = os.path.expanduser(positional_args[0]) if positional_args else _ask_path("Save folder")
        _sec("Diagnosing chunk format")
        diagnose_save(save)
        return

    # ── Load config file (from CLI arg or interactive prompt) ─────────────
    saved_config = None
    unattended = False

    if config_arg:
        if os.path.isfile(config_arg):
            with open(config_arg, encoding="utf-8") as _f:
                saved_config = _parse_config(json.load(_f))
            print(f"\n  Config: {config_arg}  (unattended mode)")
            unattended = True
        else:
            print(f"\n  ERROR: Config file not found: {config_arg}")
            sys.exit(1)
    else:
        _sec("Configuration File")
        _cfg_raw = input("  Config file path (or blank to answer prompts): ").strip()
        if _cfg_raw:
            _cfg_path = os.path.expanduser(_cfg_raw)
            if os.path.isfile(_cfg_path):
                with open(_cfg_path, encoding="utf-8") as _f:
                    saved_config = _parse_config(json.load(_f))
                print(f"  Loaded: {_cfg_path}  (unattended mode)")
                unattended = True
            else:
                print(f"  File not found: {_cfg_path} — answering prompts.")

    # ── Collect all parameters ────────────────────────────────────────────
    params = collect_params(saved=saved_config, unattended=unattended)
    out_dir = params["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Save config for future runs (new sectioned format) ───────────────
    _cfg_out = os.path.join(out_dir, "config.json")
    with open(_cfg_out, "w", encoding="utf-8") as _f:
        json.dump(_serialize_config(params), _f, indent=2, ensure_ascii=False)
    if not unattended:
        print(f"\n  Config saved → {_cfg_out}")

    # ── Initialise / load checkpoint ──────────────────────────────────────
    cp = Checkpoint(out_dir)
    if force_fresh:
        cp.clear()
    else:
        cp.load()
        # Invalidate downstream stages whose inputs have changed so they
        # are regenerated automatically (e.g. sigma, gamma, z dimensions).
        _invalidate_stale_stages(cp, params)

    params["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Preserve internal keys (_min_cx, _min_cz, _effective_sea_level, etc.)
    # that were written by a previous stage run and are not in the user-facing params.
    for k, v in cp.params.items():
        if k.startswith("_") and params.get(k) is None:
            params[k] = v
    cp.set_params(params)

    # ── Run stages ────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print(f"  Starting conversion — outputs → {out_dir}/")
    print(f"{'═' * 62}")

    hm_raw, crop_mask, water_map = stage_load(cp, params)
    hm_work, ocean_mask = stage_process(cp, params, hm_raw, water_map=water_map)
    if params.get("generate_heightmap", True):
        stage_image(cp, params, hm_work, ocean_mask, crop_mask=crop_mask)
    else:
        print("\n[Skip] Heightmap image generation disabled by config.")
    if params.get("generate_stl", True):
        stage_stl(cp, params, hm_work, ocean_mask)
    else:
        print("\n[Skip] Single STL generation disabled by config.")
    if params.get("generate_mosaic", True):
        stage_mosaic(cp, params, hm_work, ocean_mask)
    else:
        print("\n[Skip] Mosaic STL generation disabled by config.")
    # Intermediate .npy files (heightmap_raw, heightmap_work, ocean_mask) are
    # intentionally kept so that re-running with changed sigma / gamma / z-scale
    # regenerates only the affected outputs without re-parsing region files.

    # ── Summary ───────────────────────────────────────────────────────────
    img_path   = os.path.join(out_dir, "heightmap.png")
    gray_path  = os.path.join(out_dir, "heightmap_gray.png")
    stl_path   = os.path.join(out_dir, "terrain.stl")
    tiles_dir  = os.path.join(out_dir, "tiles")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                          Done!                               ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Heightmap (color)  →  {os.path.relpath(img_path):<37}║")
    print(f"║  Heightmap (gray)   →  {os.path.relpath(gray_path):<37}║")
    print(f"║  Full terrain STL   →  {os.path.relpath(stl_path):<37}║")
    print(f"║  Mosaic tiles       →  {os.path.relpath(tiles_dir) + '/':<37}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
