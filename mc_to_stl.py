#!/usr/bin/env python3
"""
MCtoSTL  –  Minecraft Save → Heightmap Image + STL Converter
=============================================================

Reads a Minecraft Java Edition save folder (the one containing the
'region/' directory with *.mca files) and produces:

  1. heightmap.png  – color-coded surface-height image
  2. terrain.stl    – single watertight 3-D printable terrain model
  3. tiles/         – mosaic of smaller STL tiles for large-format printing

Run:
    python mc_to_stl.py [save_folder]
"""

import os
import sys

from mc_to_stl.loader import load_save
from mc_to_stl.image import generate_image
from mc_to_stl.mesh import generate_single_stl, generate_mosaic_stl


# ── Interactive prompts ──────────────────────────────────────────────────────

def _ask_float(prompt: str, default: float, minimum: float = 1e-9) -> float:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if val >= minimum:
                return val
            print(f"    ✗  Must be ≥ {minimum}.")
        except ValueError:
            print("    ✗  Please enter a number.")


def _ask_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val >= minimum:
                return val
            print(f"    ✗  Must be ≥ {minimum}.")
        except ValueError:
            print("    ✗  Please enter an integer.")


def _ask_path(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if raw == "" and default:
            return default
        if raw:
            return os.path.expanduser(raw)
        print("    ✗  Please enter a path.")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          MCtoSTL  –  Minecraft  →  STL  Converter        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Save folder ──────────────────────────────────────────────────────
    _section("Minecraft Save Folder")
    if len(sys.argv) > 1:
        save_path = os.path.expanduser(sys.argv[1])
        print(f"  Using: {save_path}")
    else:
        save_path = _ask_path("Path to Minecraft save folder")

    if not os.path.isdir(save_path):
        print(f"\n  ✗  '{save_path}' is not a directory.")
        sys.exit(1)

    # ── Load save ─────────────────────────────────────────────────────────
    _section("Loading Save")
    try:
        heightmap, meta = load_save(save_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ✗  {exc}")
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
    _section("Anti-aliasing  (Gaussian blur applied before all outputs)")
    print("  Higher sigma = smoother result, less block-y appearance.")
    print("  Recommended: 1.0–2.0  |  0 = no smoothing")
    smooth_sigma = _ask_float("Gaussian sigma", default=1.5, minimum=0.0)

    # ── Heightmap image ───────────────────────────────────────────────────
    _section("Heightmap Image")
    print(f"  Map is {W} × {H} blocks.")
    print("  The image will fit within the given pixel dimensions")
    print("  while preserving the map's aspect ratio.")
    max_px_w = _ask_int("Max image width  (px)", default=2048)
    max_px_h = _ask_int("Max image height (px)", default=2048)

    img_path = os.path.join(out_dir, "heightmap.png")
    generate_image(heightmap, max_px_w, max_px_h, smooth_sigma, img_path)

    # ── STL dimensions ────────────────────────────────────────────────────
    _section("STL Physical Dimensions")
    print(f"  Map is {W} × {H} blocks.")
    print("  X and Y scales are derived from max_x / max_y so the model")
    print("  fits exactly within the specified bounding box.")
    print("  Z (altitude) scale is independent – you can exaggerate it.")

    default_y = round(200.0 * H / W, 1)
    max_x_mm = _ask_float("Max X dimension  (mm)", default=200.0)
    max_y_mm = _ask_float("Max Y dimension  (mm)", default=default_y)
    max_z_mm = _ask_float("Max Z (altitude) (mm)", default=20.0)
    base_mm  = _ask_float("Base plate thickness (mm)", default=2.0)

    # ── Single STL ────────────────────────────────────────────────────────
    stl_path = os.path.join(out_dir, "terrain.stl")
    generate_single_stl(
        heightmap, max_x_mm, max_y_mm, max_z_mm, base_mm, smooth_sigma, stl_path
    )

    # ── Mosaic ────────────────────────────────────────────────────────────
    _section("Mosaic Tile Dimensions")
    print("  Each tile will be a self-contained printable STL.")
    print("  Tiles share exact edge coordinates so they fit together.")
    print(f"  Full model footprint: {max_x_mm} × {max_y_mm} mm")

    tile_x_mm = _ask_float("Tile width  (mm)", default=min(100.0, max_x_mm))
    tile_y_mm = _ask_float("Tile height (mm)", default=min(100.0, max_y_mm))

    tiles_dir = os.path.join(out_dir, "tiles")
    generate_mosaic_stl(
        heightmap, max_x_mm, max_y_mm, max_z_mm,
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
