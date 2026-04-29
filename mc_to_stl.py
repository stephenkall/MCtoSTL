#!/usr/bin/env python3
"""
MCtoSTL  –  Minecraft Save → Heightmap Image + STL Converter
=============================================================

Outputs:
  1. heightmap.png  – color-coded surface-height image
                      (max altitude = red, sea level = green, min = blue;
                       ocean areas shown in steel-blue)
  2. terrain.stl    – single watertight 3-D printable terrain model
  3. tiles/         – mosaic of smaller STL tiles for large-format printing

Usage:
  python mc_to_stl.py [save_folder]
  python mc_to_stl.py --diagnose [save_folder]   # print chunk format info
"""

import os
import sys

from mc_to_stl.loader import load_save, diagnose_save
from mc_to_stl.image import generate_image
from mc_to_stl.mesh import generate_single_stl, generate_mosaic_stl
from mc_to_stl.ocean import detect_sea_level, build_ocean_mask, apply_ocean_mask


# ── Prompts ──────────────────────────────────────────────────────────────────

def _ask(prompt: str, default, cast, minimum=None):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = cast(raw)
            if minimum is not None and val < minimum:
                print(f"    ✗  Must be ≥ {minimum}.")
                continue
            return val
        except ValueError:
            print("    ✗  Invalid input.")


def _ask_float(prompt, default, minimum=1e-9):
    return _ask(prompt, default, float, minimum)


def _ask_int(prompt, default, minimum=1):
    return _ask(prompt, default, int, minimum)


def _ask_bool(prompt, default=True):
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"  {prompt} [{suffix}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    ✗  Please enter y or n.")


def _ask_path(prompt, default=""):
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if raw == "" and default:
            return default
        if raw:
            return os.path.expanduser(raw)
        print("    ✗  Please enter a path.")


def _section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Parse --diagnose flag
    args = sys.argv[1:]
    diagnose_mode = "--diagnose" in args
    args = [a for a in args if a != "--diagnose"]

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          MCtoSTL  –  Minecraft  →  STL  Converter        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Save folder ──────────────────────────────────────────────────────
    _section("Minecraft Save Folder")
    if args:
        save_path = os.path.expanduser(args[0])
        print(f"  Using: {save_path}")
    else:
        save_path = _ask_path("Path to Minecraft save folder")

    if not os.path.isdir(save_path):
        print(f"\n  ✗  '{save_path}' is not a directory.")
        sys.exit(1)

    # ── Diagnose mode ─────────────────────────────────────────────────────
    if diagnose_mode:
        _section("Diagnosing chunk format (first 2 region files)")
        diagnose_save(save_path)
        print()
        if not _ask_bool("Continue to full conversion?", default=True):
            return

    # ── Load save ─────────────────────────────────────────────────────────
    _section("Loading Save")
    try:
        heightmap, meta = load_save(save_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ✗  {exc}")
        print()
        print("  Tip: run with --diagnose to inspect chunk structure:")
        print("       python mc_to_stl.py --diagnose /path/to/save")
        sys.exit(1)

    W = meta["width_blocks"]
    H = meta["height_blocks"]
    h_min = float(heightmap.min())
    h_max = float(heightmap.max())
    print(f"  Height range : Y = {h_min:.0f}  ..  Y = {h_max:.0f}")

    # ── Output directory ──────────────────────────────────────────────────
    _section("Output Settings")
    out_dir = _ask_path("Output directory", default="mc_output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Outputs will be written to: {out_dir}/")

    # ── Anti-aliasing ─────────────────────────────────────────────────────
    _section("Anti-aliasing  (Gaussian blur – applied to all outputs)")
    print("  Recommended: 1.0–2.0  |  0 = none  |  higher = smoother")
    smooth_sigma = _ask_float("Gaussian sigma", default=1.5, minimum=0.0)

    # ── Ocean masking ─────────────────────────────────────────────────────
    _section("Ocean / Sea Masking")
    print("  Large open-sea areas can be excluded from the relief so the")
    print("  continent fills the output dimensions.")
    print("  Rivers and lakes (small water bodies) are always kept.")
    mask_ocean = _ask_bool("Mask out ocean areas?", default=True)

    ocean_mask = None
    sea_level = 0.0

    if mask_ocean:
        auto_sea = detect_sea_level(heightmap)
        print(f"  Auto-detected sea level: Y = {auto_sea}")
        sea_level = float(_ask_int("  Sea level Y", default=auto_sea))

        print(f"  Ocean = water body at/below Y={sea_level:.0f} that touches the map edge,")
        print(f"  OR any water body larger than the threshold below.")
        default_threshold = max(100_000, int(W * H * 0.005))
        min_ocean_blocks = _ask_int(
            "  Min ocean area (blocks²)",
            default=default_threshold,
            minimum=1,
        )

        print(f"  Building ocean mask …", end=" ", flush=True)
        ocean_mask = build_ocean_mask(
            heightmap,
            sea_level=int(sea_level),
            min_ocean_blocks=min_ocean_blocks,
            margin_blocks=0,
        )
        pct = 100.0 * ocean_mask.sum() / ocean_mask.size
        print(f"done  ({pct:.1f}% of map identified as ocean)")

        # Clamp ocean cells to sea_level in the working heightmap
        hm_work = apply_ocean_mask(heightmap, ocean_mask, sea_level=int(sea_level))
    else:
        sea_level = (h_min + h_max) / 2.0   # neutral; green = midpoint
        hm_work = heightmap

    # ── Heightmap image ───────────────────────────────────────────────────
    _section("Heightmap Image")
    print(f"  Map is {W} × {H} blocks.")
    print("  Image fits within the given pixel bounds, preserving aspect ratio.")
    max_px_w = _ask_int("Max image width  (px)", default=2048)
    max_px_h = _ask_int("Max image height (px)", default=2048)

    img_path = os.path.join(out_dir, "heightmap.png")
    generate_image(
        hm_work, max_px_w, max_px_h, smooth_sigma, img_path,
        sea_level=sea_level,
        ocean_mask=ocean_mask,
    )

    # ── STL dimensions ────────────────────────────────────────────────────
    _section("STL Physical Dimensions")
    print(f"  Map is {W} × {H} blocks.")
    print("  X/Y scales are derived from max_x/max_y so the model exactly")
    print("  fills the bounding box.  Z (altitude) scale is independent.")

    default_y = round(200.0 * H / W, 1)
    max_x_mm  = _ask_float("Max X dimension  (mm)", default=200.0)
    max_y_mm  = _ask_float("Max Y dimension  (mm)", default=default_y)
    max_z_mm  = _ask_float("Max Z (altitude) (mm)", default=20.0)
    base_mm   = _ask_float("Base plate thickness (mm)", default=2.0)

    # ── Single STL ────────────────────────────────────────────────────────
    stl_path = os.path.join(out_dir, "terrain.stl")
    generate_single_stl(
        hm_work, max_x_mm, max_y_mm, max_z_mm, base_mm, smooth_sigma, stl_path
    )

    # ── Mosaic ────────────────────────────────────────────────────────────
    _section("Mosaic Tile Dimensions")
    print("  Each tile is a self-contained printable STL.")
    print("  Tiles share exact edge coordinates so they fit together perfectly.")
    print(f"  Full model footprint: {max_x_mm} × {max_y_mm} mm")

    tile_x_mm = _ask_float("Tile width  (mm)", default=min(100.0, max_x_mm))
    tile_y_mm = _ask_float("Tile height (mm)", default=min(100.0, max_y_mm))

    tiles_dir = os.path.join(out_dir, "tiles")
    generate_mosaic_stl(
        hm_work, max_x_mm, max_y_mm, max_z_mm,
        tile_x_mm, tile_y_mm,
        base_mm, smooth_sigma, tiles_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                        Done!                             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Heightmap image  →  {os.path.relpath(img_path):<35}║")
    print(f"║  Full terrain STL →  {os.path.relpath(stl_path):<35}║")
    print(f"║  Mosaic tiles     →  {os.path.relpath(tiles_dir) + '/':<35}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
