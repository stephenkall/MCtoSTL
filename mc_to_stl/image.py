"""
Heightmap → PNG image.

Color scheme (heightmap.png)
-----------------------------
  below sea level  → dark blue → sea-green gradient
  at sea level     → sea-green  (#14A037)
  above sea level  → sea-green → red gradient
  ocean cells      → steel-blue (#1E50A0) override

Grayscale (heightmap_gray.png)
-------------------------------
  Pure grayscale: global min → black, global max → white.
  Ocean cells set to black (Y = 0).

Transparent crop (rectangular_crop=False)
------------------------------------------
  When a crop_mask is provided and rectangular_crop is False, both PNG
  files are saved as RGBA with alpha=0 for pixels outside the crop polygon
  and alpha=255 inside.  The bounding-box dimensions are unchanged.

Normalisation
--------------
Strict global min→black, global max→white (no percentile clipping).

Output dimensions
------------------
Always produced at exactly (out_w × out_h) pixels.  Set max_px_w =
max_px_h = 0 to keep native 1 block = 1 pixel resolution.

Outlier suppression
--------------------
Before rendering a 3×3 median filter is applied at full resolution to
eliminate single-pixel terrain anomalies.
"""

import os
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, zoom

_OCEAN_RGB  = np.array([30, 80, 160],   dtype=np.float64)
_C_DEEP     = np.array([0,  30,  80],   dtype=np.float64)  # deep below sea level
_C_SEA      = np.array([20, 160, 60],   dtype=np.float64)  # sea level (green)
_C_HIGH     = np.array([200, 30,  30],  dtype=np.float64)  # max altitude (red)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lerp_colors(c0: np.ndarray, c1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Interpolate between two RGB float64 colors. t: 1-D float array → (N,3) uint8."""
    return np.clip(c0 * (1 - t[:, None]) + c1 * t[:, None], 0, 255).astype(np.uint8)


def _build_rgb(
    sm: np.ndarray,
    ocean_mask: Optional[np.ndarray],
    lo: float,
    hi: float,
    sea_level: float,
) -> np.ndarray:
    """
    Build uint8 RGB array using elevation-based color scheme.

    Below sea level → dark-blue-to-green gradient.
    Above sea level → green-to-red gradient.
    Ocean cells     → steel-blue override.
    """
    rows, cols = sm.shape
    out = np.zeros((rows, cols, 3), dtype=np.uint8)

    sea = float(np.clip(sea_level, lo, hi))

    # Below sea level
    below = sm < sea
    if below.any():
        denom = max(sea - lo, 1e-6)
        t = np.clip((sm[below] - lo) / denom, 0.0, 1.0)
        out[below] = _lerp_colors(_C_DEEP, _C_SEA, t)

    # At / above sea level
    above = ~below
    if above.any():
        denom = max(hi - sea, 1e-6)
        t = np.clip((sm[above] - sea) / denom, 0.0, 1.0)
        out[above] = _lerp_colors(_C_SEA, _C_HIGH, t)

    # Ocean override
    if ocean_mask is not None and ocean_mask.any():
        out[ocean_mask] = _OCEAN_RGB.astype(np.uint8)

    return out


def _apply_alpha(
    arr: np.ndarray,
    crop_mask_small: Optional[np.ndarray],
) -> Image.Image:
    """
    Convert RGB or grayscale array to PIL Image, adding alpha channel from
    crop_mask_small when provided (inside=255, outside=0).
    """
    if crop_mask_small is None:
        if arr.ndim == 2:
            return Image.fromarray(arr, "L")
        return Image.fromarray(arr, "RGB")

    alpha = np.where(crop_mask_small, np.uint8(255), np.uint8(0))
    if arr.ndim == 2:
        rgba = np.dstack([arr, arr, arr, alpha])
    else:
        rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


# ── Public API ────────────────────────────────────────────────────────────────

def generate_image(
    heightmap: np.ndarray,
    max_px_w: int,
    max_px_h: int,
    smooth_sigma: float,
    output_path: str,
    sea_level: float = 0.0,
    ocean_mask: Optional[np.ndarray] = None,
    crop_mask: Optional[np.ndarray] = None,
    rectangular_crop: bool = True,
) -> Image.Image:
    """
    Generate a color-coded heightmap PNG and a grayscale PNG.

    heightmap.png      — elevation colour + steel-blue ocean
    heightmap_gray.png — pure grayscale (ocean = black)

    Parameters
    ----------
    crop_mask       : Optional bool array (same shape as heightmap); True = inside
                      crop polygon.  When rectangular_crop=False, pixels outside the
                      mask become transparent (alpha=0) in both output files.
    rectangular_crop: True  → fill outside-crop area with ocean (rectangular image).
                      False → transparent background (RGBA PNG).
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

    # ── Stats ─────────────────────────────────────────────────────────────
    lo = float(heightmap.min())
    hi = float(heightmap.max())
    if hi <= lo:
        hi = lo + 1.0
    land_mask = (~ocean_mask) if ocean_mask is not None else np.ones((rows, cols), dtype=bool)
    land = heightmap[land_mask]
    if land.size > 0:
        land_rel = land - sea_level
        rel_min = float(land_rel.min())
        rel_max = float(land_rel.max())
    else:
        rel_min = rel_max = 0.0
    print(f"  Land range   : {rel_min:+.0f} .. {rel_max:+.0f}  (relative to sea)")
    print(f"  Gray stretch : Y={lo:.0f} → black,  Y={hi:.0f} → white  (global min–max)")
    if ocean_mask is not None:
        pct = 100.0 * ocean_mask.sum() / ocean_mask.size
        print(f"  Ocean cover  : {pct:.1f}%")
    if crop_mask is not None and not rectangular_crop:
        pct_crop = 100.0 * crop_mask.sum() / crop_mask.size
        print(f"  Crop (transparent outside): {pct_crop:.1f}% of bbox is inside polygon")

    # ── Median filter ─────────────────────────────────────────────────────
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
        if crop_mask is not None and not rectangular_crop:
            cm_small = zoom(crop_mask.astype(np.float32),
                            zoom_factors, order=0) > 0.5
        else:
            cm_small = None
        print(f"done  ({sm.shape[1]} × {sm.shape[0]} px)")
    else:
        sm = hm_filt
        om_small = ocean_mask
        cm_small = (crop_mask if (crop_mask is not None and not rectangular_crop)
                    else None)

    # ── Gaussian smooth ───────────────────────────────────────────────────
    scale_for_sigma = out_w / cols if cols > 0 else 1.0
    sigma_px = smooth_sigma * scale_for_sigma
    if sigma_px > 0.1:
        sm = gaussian_filter(sm, sigma=sigma_px)

    # ── Color PNG ─────────────────────────────────────────────────────────
    print(f"  Building RGB …", end=" ", flush=True)
    rgb = _build_rgb(sm, om_small, lo, hi, sea_level)
    print("done")

    img = _apply_alpha(rgb, cm_small)
    img.save(output_path)
    suffix = "RGBA" if cm_small is not None else "RGB"
    print(f"  Saved ({suffix}) -> {output_path}")

    # ── Grayscale PNG ─────────────────────────────────────────────────────
    gray_path = os.path.splitext(output_path)[0] + "_gray.png"
    print(f"  Building grayscale …", end=" ", flush=True)
    gray_f = np.clip((sm - lo) / (hi - lo), 0.0, 1.0)
    gray_u8 = (gray_f * 255).astype(np.uint8)
    if om_small is not None:
        gray_u8[om_small] = 0  # ocean = black
    print("done")
    gray_img = _apply_alpha(gray_u8, cm_small)
    gray_img.save(gray_path)
    suffix_g = "RGBA" if cm_small is not None else "L"
    print(f"  Saved ({suffix_g}) -> {gray_path}")

    return img
