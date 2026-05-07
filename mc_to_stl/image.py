"""
Heightmap → PNG image matching Unmined's heightmap rendering style.

Color scheme:
  land  → grayscale by elevation (darker = lower, lighter = higher)
  ocean → steel-blue  (#1E50A0)

No hillshading is applied — each gray value maps directly to one elevation,
making the image useful as a working reference.

Grayscale normalisation
-----------------------
Land heights are percentile-stretched to [2nd, 98th] so isolated peaks or
deep valleys don't collapse the gradient for the majority of the terrain.
Both the color PNG and the gray PNG use the same stretch parameters.

Output dimensions
-----------------
The image is always produced at exactly (out_w × out_h) pixels.  If the map
is smaller than max_px, it is upscaled so that every block maps to >1 pixel.
Set max_px_w = max_px_h = 0 to keep the native 1 block = 1 pixel resolution.

Outlier suppression
-------------------
Before rendering a 3×3 median filter is applied to the FULL-resolution
heightmap to eliminate single-pixel terrain anomalies (cave mouths, ravines,
isolated missing-chunk holes) that would otherwise appear as dark specks.

Memory model
------------
After the median filter the heightmap is immediately downsampled to the
output pixel dimensions so that all subsequent arrays are output-sized.
"""

import os
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, zoom

_OCEAN_RGB = np.array([30, 80, 160], dtype=np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_rgb(
    sm: np.ndarray,
    ocean_mask: Optional[np.ndarray],
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Build uint8 RGB: grayscale land (percentile-normalised) + steel-blue ocean.
    lo/hi are the land-height percentile bounds computed from the full-res data.
    """
    rows, cols = sm.shape
    out = np.zeros((rows, cols, 3), dtype=np.uint8)

    gray_f = np.clip((sm - lo) / (hi - lo), 0.0, 1.0)
    gray_u8 = (gray_f * 255).astype(np.uint8)

    out[..., 0] = gray_u8
    out[..., 1] = gray_u8
    out[..., 2] = gray_u8

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
    Generate a color-coded heightmap PNG and a grayscale PNG.

    heightmap.png   — grayscale by elevation + steel-blue ocean
    heightmap_gray.png — pure grayscale (ocean = black)

    Both images use the same 2nd–98th percentile normalisation over land
    heights so colors are directly comparable between the two files.
    """
    rows, cols = heightmap.shape
    print(f"\n[Heightmap Image]")
    print(f"  Map size     : {cols} × {rows} blocks")
    print(f"  Sea level    : Y={sea_level:.0f}")

    # ── Compute output size ───────────────────────────────────────────────
    if max_px_w <= 0 or max_px_h <= 0:
        out_w, out_h = cols, rows
    else:
        scale = min(max_px_w / cols, max_px_h / rows)
        out_w = max(1, int(round(cols * scale)))
        out_h = max(1, int(round(rows * scale)))
    print(f"  Output size  : {out_w} x {out_h} px")

    # ── Stats (on original full-res data to be accurate) ─────────────────
    land_mask = (~ocean_mask) if ocean_mask is not None else np.ones((rows, cols), dtype=bool)
    land = heightmap[land_mask]
    if land.size > 0:
        lo = float(np.percentile(land, 2))
        hi = float(np.percentile(land, 98))
        land_rel = land - sea_level
        rel_min = float(land_rel.min())
        rel_max = float(land_rel.max())
    else:
        lo = float(heightmap.min())
        hi = float(heightmap.max())
        rel_min = rel_max = 0.0
    if hi <= lo:
        hi = lo + 1.0
    print(f"  Land range   : {rel_min:+.0f} .. {rel_max:+.0f}  (relative to sea)")
    print(f"  Gray stretch : Y={lo:.0f} → black,  Y={hi:.0f} → white  (2nd–98th pct)")
    if ocean_mask is not None:
        pct = 100.0 * ocean_mask.sum() / ocean_mask.size
        print(f"  Ocean cover  : {pct:.1f}%")

    # ── Median filter: remove isolated deep-pixel anomalies ──────────────
    print(f"  Median filter …", end=" ", flush=True)
    hm_filt = median_filter(heightmap.astype(np.float32), size=3)
    print("done")

    # ── Downsample / upsample to exact output size ────────────────────────
    if out_w != cols or out_h != rows:
        zoom_factors = (out_h / rows, out_w / cols)
        print(f"  Resampling …", end=" ", flush=True)
        sm = zoom(hm_filt, zoom_factors, order=1)
        if ocean_mask is not None:
            om_small = zoom(ocean_mask.astype(np.float32),
                            zoom_factors, order=0) > 0.5
        else:
            om_small = None
        print(f"done  ({sm.shape[1]} × {sm.shape[0]} px)")
        if om_small is not None:
            print(f"  [debug] om_small coverage after resample: "
                  f"{100.0*om_small.sum()/om_small.size:.1f}%")
        else:
            print(f"  [debug] om_small is None after resample")
    else:
        sm = hm_filt
        om_small = ocean_mask

    # ── Gaussian smooth at output resolution ─────────────────────────────
    scale_for_sigma = out_w / cols if cols > 0 else 1.0
    sigma_px = smooth_sigma * scale_for_sigma
    if sigma_px > 0.1:
        sm = gaussian_filter(sm, sigma=sigma_px)

    # ── Build hillshaded RGB and save ─────────────────────────────────────
    print(f"  Building RGB …", end=" ", flush=True)
    rgb = _build_rgb(sm, om_small, lo, hi)
    print("done")

    img = Image.fromarray(rgb, "RGB")
    img.save(output_path)
    print(f"  Saved -> {output_path}")

    # ── Build plain grayscale and save ────────────────────────────────────
    gray_path = os.path.splitext(output_path)[0] + "_gray.png"
    print(f"  Building grayscale …", end=" ", flush=True)
    gray_f = np.clip((sm - lo) / (hi - lo), 0.0, 1.0)
    gray_u8 = (gray_f * 255).astype(np.uint8)
    if om_small is not None:
        gray_u8[om_small] = 0  # ocean = black
    print("done")
    gray_img = Image.fromarray(gray_u8, "L")
    gray_img.save(gray_path)
    print(f"  Saved -> {gray_path}")

    return img
