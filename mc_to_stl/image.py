"""
Heightmap → PNG image matching Unmined's heightmap rendering style.

Color scheme:
  land  → grayscale by elevation (darker = lower, lighter = higher)
  ocean → deep blue  (#1E50A0)
  + directional hillshading applied to land (and land-edge ocean)

Hillshading
-----------
Replicates Unmined's CalcShading algorithm: a weighted 3×3 kernel of
height differences with sun_angle=120° (ShadingMatrix / HighlightMatrix),
scaled so steep slopes get ±33% lightness adjustment.

Grayscale normalisation
-----------------------
Land heights are percentile-stretched to [2nd, 98th] so that isolated
peaks or deep valleys don't collapse the gradient for the majority of
the terrain.

Memory model
------------
The heightmap is downsampled to the output pixel dimensions BEFORE any
processing so that all intermediate arrays are output-sized, not
source-sized.  A 16k-block map rendered at 4096 px saves ~15× RAM.
"""

import os
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

_OCEAN_RGB = np.array([30, 80, 160], dtype=np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unmined_hillshade(sm: np.ndarray) -> np.ndarray:
    """
    Replicate Unmined's CalcShading for sun_angle=120°.

    For that angle, flag1=True, flag2=False, so the active neighbours are:
      bottom-left (z+1, x-1)  shadow=8   highlight=8
      bottom      (z+1, x)    shadow=12  highlight=12
      bottom-right(z+1, x+1)  shadow=1   highlight=0
      left        (z,   x-1)  shadow=4   highlight=6
      top-left    (z-1, x-1)  shadow=1   highlight=0

    The raw shading value is clamped via DiffMap and scaled 0.09×, then
    the lightness multiplier = 1.0 + shading × 2.5  (range ≈ 0.66 … 1.34).
    """
    h = sm.astype(np.float64)
    hp = np.pad(h, 1, mode="edge")

    n_bl = hp[2:,  :-2]   # bottom-left
    n_b  = hp[2:,  1:-1]  # bottom
    n_br = hp[2:,  2:]    # bottom-right
    n_l  = hp[1:-1, :-2]  # left
    n_tl = hp[:-2, :-2]   # top-left

    def _contrib(diff: np.ndarray, w_shadow: float, w_highlight: float) -> np.ndarray:
        return np.where(diff < 0, w_shadow * diff, w_highlight * diff)

    num2 = (
        _contrib(h - n_bl, 8.0,  8.0) +
        _contrib(h - n_b,  12.0, 12.0) +
        _contrib(h - n_br, 1.0,  0.0) +
        _contrib(h - n_l,  4.0,  6.0) +
        _contrib(h - n_tl, 1.0,  0.0)
    )
    num3 = 26.0  # sum of ShadingMatrix

    # DiffMap: clamp index to [0, 255], then (index-128)/128 * 1.5
    idx = np.clip(num2 / num3 * 128.0 + 128.0, 0.0, 255.0)
    diffmap = (idx - 128.0) / 128.0 * 1.5
    shading = diffmap * 0.09
    factor = 1.0 + shading * 2.5
    return factor.astype(np.float32)


def _build_rgb(
    sm: np.ndarray,
    ocean_mask: Optional[np.ndarray],
) -> np.ndarray:
    """
    Build uint8 RGB matching Unmined's heightmap mode:
      land  → grayscale [2nd..98th percentile], hillshaded
      ocean → _OCEAN_RGB (uniform steel-blue)
    """
    rows, cols = sm.shape
    out = np.zeros((rows, cols, 3), dtype=np.uint8)

    # ── Grayscale land ────────────────────────────────────────────────────
    land_mask = (~ocean_mask) if ocean_mask is not None else np.ones((rows, cols), dtype=bool)
    land_heights = sm[land_mask]
    if land_heights.size > 0:
        lo = float(np.percentile(land_heights, 2))
        hi = float(np.percentile(land_heights, 98))
    else:
        lo, hi = float(sm.min()), float(sm.max())
    if hi <= lo:
        hi = lo + 1.0

    gray_f = np.clip((sm - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)  # [0, 1]

    # ── Hillshading ───────────────────────────────────────────────────────
    hs = _unmined_hillshade(sm)
    gray_shaded = np.clip(gray_f * hs, 0.0, 1.0)

    gray_u8 = (gray_shaded * 255).astype(np.uint8)
    out[..., 0] = gray_u8
    out[..., 1] = gray_u8
    out[..., 2] = gray_u8

    # ── Ocean override ────────────────────────────────────────────────────
    if ocean_mask is not None and ocean_mask.any():
        out[ocean_mask] = _OCEAN_RGB

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def generate_image(
    heightmap: np.ndarray,
    max_px_w: int,
    max_px_h: int,
    smooth_sigma: float,
    output_path: str,
    sea_level: float = 0.0,
    ocean_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    Generate a heightmap PNG (grayscale + hillshading + blue ocean) and a
    plain grayscale PNG (elevation only, ocean=black).

    The grayscale image is saved to <output_path_stem>_gray.png and is
    suitable as an STL displacement reference or further processing.

    Downsamples to output dimensions before all processing so that memory
    usage scales with the output size, not the source map size.
    """
    rows, cols = heightmap.shape
    print(f"\n[Heightmap Image]")
    print(f"  Map size     : {cols} × {rows} blocks")
    print(f"  Sea level    : Y={sea_level:.0f}")

    # ── Compute output size ───────────────────────────────────────────────
    if max_px_w <= 0 or max_px_h <= 0:
        out_w, out_h = cols, rows
        scale = 1.0
    else:
        scale = min(max_px_w / cols, max_px_h / rows)
        out_w = max(1, int(round(cols * scale)))
        out_h = max(1, int(round(rows * scale)))
    print(f"  Output size  : {out_w} x {out_h} px  (scale {scale:.3f} px/block)")

    # ── Downsample to output size FIRST (saves memory on large maps) ──────
    if scale < 0.999:
        print(f"  Downsampling …", end=" ", flush=True)
        sm = zoom(heightmap.astype(np.float32),
                  (out_h / rows, out_w / cols), order=1)
        if ocean_mask is not None:
            om_small = zoom(ocean_mask.astype(np.float32),
                            (out_h / rows, out_w / cols), order=0) > 0.5
        else:
            om_small = None
        print(f"done  ({sm.shape[1]} × {sm.shape[0]} px)")
        if om_small is not None:
            print(f"  [debug] om_small coverage after downsample: {100.0*om_small.sum()/om_small.size:.1f}%")
        else:
            print(f"  [debug] om_small is None after downsample")
    else:
        sm = heightmap.astype(np.float32)
        om_small = ocean_mask

    # ── Smooth at output resolution ───────────────────────────────────────
    sigma_px = smooth_sigma * scale   # scale sigma to output pixels
    if sigma_px > 0.1:
        sm = gaussian_filter(sm, sigma=sigma_px)

    # ── Stats (on original full-res data to be accurate) ─────────────────
    land_mask = (~ocean_mask) if ocean_mask is not None else np.ones(heightmap.shape, dtype=bool)
    land = heightmap[land_mask]
    land_rel = land - sea_level
    rel_max = float(land_rel.max()) if land_rel.size else 1.0
    rel_min = float(land_rel.min()) if land_rel.size else 0.0
    print(f"  Land range   : {rel_min:+.0f} .. {rel_max:+.0f}  (relative to sea)")
    if ocean_mask is not None:
        pct = 100.0 * ocean_mask.sum() / ocean_mask.size
        print(f"  Ocean cover  : {pct:.1f}%")

    # ── Build hillshaded RGB and save ─────────────────────────────────────
    print(f"  Building RGB …", end=" ", flush=True)
    rgb = _build_rgb(sm, om_small)
    print("done")

    img = Image.fromarray(rgb, "RGB")
    img.save(output_path)
    print(f"  Saved -> {output_path}")

    # ── Build plain grayscale and save ────────────────────────────────────
    gray_path = os.path.splitext(output_path)[0] + "_gray.png"
    print(f"  Building grayscale …", end=" ", flush=True)
    lo = float(land.min()) if land.size else float(np.nanmin(heightmap))
    hi = float(land.max()) if land.size else float(np.nanmax(heightmap))
    if hi <= lo:
        hi = lo + 1.0
    print(f"range {lo:.0f}..{hi:.0f} ", end="")
    gray = np.clip((sm - lo) / (hi - lo), 0.0, 1.0)
    gray_u8 = np.rint(gray * 255).astype(np.uint8)
    if om_small is not None:
        gray_u8[om_small] = 0  # ocean = black
    print("done")
    gray_img = Image.fromarray(gray_u8, "L")
    gray_img.save(gray_path)
    print(f"  Saved -> {gray_path}")

    return img
