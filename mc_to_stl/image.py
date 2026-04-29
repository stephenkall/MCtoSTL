"""
Heightmap → color-coded PNG image.

Color scheme (user-specified, applied relative to sea level):
  altitude = max_positive  →  (255,   0,   0)  red
  altitude = 0  (sea level)  →  (  0, 255,   0)  green
  altitude = max_negative  →  (  0,   0, 255)  blue

Ocean pixels (large sea areas) are rendered in a distinct steel-blue so
they don't "waste" the color range that should express terrain relief.
"""

from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter


# ── Color mapping ─────────────────────────────────────────────────────────────

# Color used for masked-out ocean cells
_OCEAN_RGB = np.array([30, 80, 160], dtype=np.uint8)


def _build_rgb(
    sm: np.ndarray,
    sea_level: float,
    ocean_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Map (heightmap - sea_level) → RGB.

    Positive altitudes: green → red
    Negative altitudes: blue  → green
    Ocean pixels: steel blue (_OCEAN_RGB)
    """
    rel = (sm - sea_level).astype(np.float32)  # altitude relative to sea

    h_max = float(rel.max())
    h_min = float(rel.min())

    rgb = np.zeros((*rel.shape, 3), dtype=np.float32)

    # Positive range [0 .. h_max] → green → red
    pos = rel >= 0
    if h_max > 0:
        t = np.where(pos, np.clip(rel / h_max, 0.0, 1.0), 0.0)
    else:
        t = np.zeros_like(rel)
    rgb[..., 0] += np.where(pos, t * 255.0, 0.0)
    rgb[..., 1] += np.where(pos, (1.0 - t) * 255.0, 0.0)

    # Negative range [h_min .. 0] → blue → green
    neg = rel < 0
    if h_min < 0:
        t2 = np.where(neg, np.clip(rel / h_min, 0.0, 1.0), 0.0)
    else:
        t2 = np.zeros_like(rel)
    rgb[..., 1] += np.where(neg, (1.0 - t2) * 255.0, 0.0)
    rgb[..., 2] += np.where(neg, t2 * 255.0, 0.0)

    result = np.clip(rgb, 0, 255).astype(np.uint8)

    # Paint ocean mask in steel-blue
    if ocean_mask is not None and ocean_mask.any():
        result[ocean_mask] = _OCEAN_RGB

    return result


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
    Generate a color-coded heightmap PNG.

    Fits within max_px_w × max_px_h while preserving the block-level
    aspect ratio.  Colors are computed relative to sea_level.
    """
    rows, cols = heightmap.shape
    print(f"\n[Heightmap Image]")

    # Gaussian smoothing (reduces pixelation)
    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)

    rel_min = float((sm - sea_level).min())
    rel_max = float((sm - sea_level).max())
    print(f"  Map size     : {cols} × {rows} blocks")
    print(f"  Sea level    : Y={sea_level:.0f}")
    print(f"  Alt. range   : {rel_min:+.0f} .. {rel_max:+.0f}  (relative to sea)")
    if ocean_mask is not None:
        pct = 100.0 * ocean_mask.sum() / ocean_mask.size
        print(f"  Ocean cover  : {pct:.1f}%")

    rgb = _build_rgb(sm, sea_level, ocean_mask)
    img = Image.fromarray(rgb, "RGB")

    # Scale to fit (preserve aspect ratio)
    scale = min(max_px_w / cols, max_px_h / rows)
    out_w = max(1, int(round(cols * scale)))
    out_h = max(1, int(round(rows * scale)))
    print(f"  Output size  : {out_w} × {out_h} px  (scale {scale:.3f} px/block)")

    img = img.resize((out_w, out_h), Image.LANCZOS)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH)

    img.save(output_path)
    print(f"  Saved → {output_path}")
    return img
