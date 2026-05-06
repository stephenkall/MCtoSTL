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

Memory model
------------
The heightmap is downsampled to the output pixel dimensions BEFORE any
processing so that all intermediate arrays are output-sized, not
source-sized.  A 16k-block map rendered at 4096 px saves ~15× RAM.
"""

from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, zoom

_OCEAN_RGB = np.array([30, 80, 160], dtype=np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_gamma(rel: np.ndarray, gamma: float) -> np.ndarray:
    """Sign-preserving power curve; gamma < 1 stretches low relief."""
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
    """
    Build uint8 RGB array from a (small, output-sized) smoothed heightmap.

    All intermediates are kept small: t_grid is float32 (one channel) and is
    reused; the output `out` is uint8 (3 channels).
    """
    rows, cols = sm.shape
    rel_g = _apply_gamma((sm - sea_level).astype(np.float32), gamma)
    out = np.zeros((rows, cols, 3), dtype=np.uint8)

    # ── Positive (land above sea): green → red ────────────────────────────
    pos = rel_g >= 0
    # Exclude ocean pixels from land statistics — they sit at rel_g=0 and
    # would inflate the low end of the distribution, biasing the percentile.
    land_pos = pos & (~ocean_mask if ocean_mask is not None else pos)
    if land_pos.any():
        t = _percentile_norm(rel_g[land_pos], lo_pct=0.0, hi_pct=100.0)
        t_grid = np.zeros((rows, cols), dtype=np.float32)
        t_grid[land_pos] = t
        out[..., 0] = (t_grid * 255).astype(np.uint8)
        out[..., 1] = ((1.0 - t_grid) * 255).astype(np.uint8)

    # ── Negative (below sea level): blue → green ──────────────────────────
    neg = ~pos
    if neg.any():
        t = _percentile_norm(-rel_g[neg], lo_pct=0.0, hi_pct=100.0)
        t_grid = np.zeros((rows, cols), dtype=np.float32)
        t_grid[neg] = t
        out[neg, 1] = ((1.0 - t_grid[neg]) * 255).astype(np.uint8)
        out[..., 2] = (t_grid * 255).astype(np.uint8)

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
    gamma: float = 0.6,
) -> Image.Image:
    """
    Generate a color-coded heightmap PNG.

    Downsamples to output dimensions before all processing so that memory
    usage scales with the output size, not the source map size.
    """
    rows, cols = heightmap.shape
    print(f"\n[Heightmap Image]")
    print(f"  Map size     : {cols} × {rows} blocks")
    print(f"  Sea level    : Y={sea_level:.0f}")

    # ── Compute output size ───────────────────────────────────────────────
    scale = min(max_px_w / cols, max_px_h / rows)
    out_w = max(1, int(round(cols * scale)))
    out_h = max(1, int(round(rows * scale)))
    print(f"  Output size  : {out_w} × {out_h} px  (scale {scale:.3f} px/block)")

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
    print(f"  Gamma        : {gamma:.2f}  ({'linear' if gamma == 1 else 'amplified' if gamma < 1 else 'compressed'})")
    if ocean_mask is not None:
        pct = 100.0 * ocean_mask.sum() / ocean_mask.size
        print(f"  Ocean cover  : {pct:.1f}%")

    # ── Build RGB and save ────────────────────────────────────────────────
    print(f"  Building RGB …", end=" ", flush=True)
    rgb = _build_rgb(sm, sea_level, gamma, om_small)
    print("done")

    img = Image.fromarray(rgb, "RGB")
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH)
    img.save(output_path)
    print(f"  Saved -> {output_path}")
    return img
