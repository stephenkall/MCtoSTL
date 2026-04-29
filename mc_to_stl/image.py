"""
Heightmap → color-coded PNG image.

Color scheme (relative to sea level):
  land maximum  →  (255,   0,   0)  red
  sea level     →  (  0, 255,   0)  green
  below sea     →  (  0,   0, 255)  blue
  ocean areas   →  steel-blue  (#1E50A0)

Relief exaggeration
-------------------
A gamma < 1.0 stretches low-relief land (more color variation on plains)
while compressing the very highest peaks.  1.0 = linear (no change).
Recommended: 0.5–0.7 for flat maps like Westeros/Essos.

Contrast stretch
----------------
By default the colour range is normalised to the [2nd, 98th] percentile
of land heights so isolated peaks or submerged valleys don't collapse
the gradient for the majority of the terrain.
"""

from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


_OCEAN_RGB = np.array([30, 80, 160], dtype=np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_gamma(rel: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma to relative altitude, preserving sign.
    gamma < 1  → stretch low relief  (more dramatic-looking flat terrain)
    gamma > 1  → compress peaks      (rarely needed)
    gamma = 1  → no change
    """
    if abs(gamma - 1.0) < 1e-6:
        return rel
    sign = np.sign(rel)
    return sign * (np.abs(rel) ** gamma)


def _percentile_norm(
    values: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0
) -> np.ndarray:
    """Clip+scale values to [lo_pct, hi_pct] percentile range → [0, 1]."""
    lo = float(np.percentile(values, lo_pct))
    hi = float(np.percentile(values, hi_pct))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _build_rgb(
    sm: np.ndarray,
    sea_level: float,
    gamma: float,
    ocean_mask: Optional[np.ndarray],
) -> np.ndarray:
    rel = (sm - sea_level).astype(np.float32)

    # Apply gamma to amplify relief variation
    rel_g = _apply_gamma(rel, gamma)

    rgb = np.zeros((*rel.shape, 3), dtype=np.float32)

    # ── Positive (land above sea): green → red ────────────────────────────
    pos = rel_g >= 0
    pos_vals = rel_g[pos]
    if pos_vals.size > 0:
        t = _percentile_norm(pos_vals, lo_pct=0.0, hi_pct=98.0)
        tmp = np.zeros(rel.shape, dtype=np.float32)
        tmp[pos] = t
        rgb[..., 0] += np.where(pos, tmp * 255.0, 0.0)
        rgb[..., 1] += np.where(pos, (1.0 - tmp) * 255.0, 0.0)

    # ── Negative (below sea level): blue → green ──────────────────────────
    neg = rel_g < 0
    neg_vals = rel_g[neg]
    if neg_vals.size > 0:
        # Most-negative → 1.0 (pure blue), just-below-zero → 0.0 (green)
        t = _percentile_norm(-neg_vals, lo_pct=0.0, hi_pct=98.0)
        tmp = np.zeros(rel.shape, dtype=np.float32)
        tmp[neg] = t
        rgb[..., 1] += np.where(neg, (1.0 - tmp) * 255.0, 0.0)
        rgb[..., 2] += np.where(neg, tmp * 255.0, 0.0)

    result = np.clip(rgb, 0, 255).astype(np.uint8)

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
    gamma: float = 0.6,
) -> Image.Image:
    """
    Generate a color-coded heightmap PNG.

    Parameters
    ----------
    gamma       : Relief exaggeration exponent (< 1 = more drama on flat terrain).
                  0.5–0.7 works well for Westeros/Essos-style maps.
    """
    rows, cols = heightmap.shape
    print(f"\n[Heightmap Image]")

    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)

    land = sm[~ocean_mask] if ocean_mask is not None else sm.flatten()
    land_rel = land - sea_level
    rel_max = float(land_rel.max()) if land_rel.size else 1.0
    rel_min = float(land_rel.min()) if land_rel.size else 0.0

    print(f"  Map size     : {cols} × {rows} blocks")
    print(f"  Sea level    : Y={sea_level:.0f}")
    print(f"  Land range   : {rel_min:+.0f} .. {rel_max:+.0f}  (relative to sea)")
    print(f"  Gamma        : {gamma:.2f}  ({'linear' if gamma == 1 else 'amplified' if gamma < 1 else 'compressed'})")
    if ocean_mask is not None:
        pct = 100.0 * ocean_mask.sum() / ocean_mask.size
        print(f"  Ocean cover  : {pct:.1f}%")

    rgb = _build_rgb(sm, sea_level, gamma, ocean_mask)
    img = Image.fromarray(rgb, "RGB")

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
