"""
Heightmap → color-coded PNG image.

Color scheme (user-specified):
  Y = max_positive  →  (255,   0,   0)  red
  Y = 0             →  (  0, 255,   0)  green
  Y = max_negative  →  (  0,   0, 255)  blue

The mapping uses two linear segments:
  [h_min, 0]  blue  → green
  [0,  h_max] green → red
"""

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter


def _build_rgb(sm: np.ndarray) -> np.ndarray:
    """Vectorised height → RGB mapping (float32 intermediate)."""
    h_max = float(sm.max())
    h_min = float(sm.min())

    rgb = np.zeros((*sm.shape, 3), dtype=np.float32)

    # Positive range  [0 .. h_max]  →  green → red
    pos = sm >= 0
    if h_max > 0:
        t = np.where(pos, np.clip(sm / h_max, 0.0, 1.0), 0.0)
    else:
        t = np.zeros_like(sm)
    rgb[..., 0] += np.where(pos, t * 255.0, 0.0)
    rgb[..., 1] += np.where(pos, (1.0 - t) * 255.0, 0.0)

    # Negative range  [h_min .. 0]  →  blue → green
    neg = sm < 0
    if h_min < 0:
        t2 = np.where(neg, np.clip(sm / h_min, 0.0, 1.0), 0.0)
    else:
        t2 = np.zeros_like(sm)
    rgb[..., 1] += np.where(neg, (1.0 - t2) * 255.0, 0.0)
    rgb[..., 2] += np.where(neg, t2 * 255.0, 0.0)

    return np.clip(rgb, 0, 255).astype(np.uint8)


def generate_image(
    heightmap: np.ndarray,
    max_px_w: int,
    max_px_h: int,
    smooth_sigma: float,
    output_path: str,
) -> Image.Image:
    """
    Generate a color-coded heightmap PNG.

    The image fits within max_px_w × max_px_h while preserving the
    block-level aspect ratio of the original map.
    """
    rows, cols = heightmap.shape
    print(f"\n[Heightmap Image]")

    # ── Gaussian smoothing to reduce pixelation ──────────────────────────
    sm = gaussian_filter(heightmap.astype(np.float32), sigma=smooth_sigma)

    # ── Color mapping ────────────────────────────────────────────────────
    rgb = _build_rgb(sm)
    img = Image.fromarray(rgb, "RGB")

    # ── Scale to fit max dimensions (preserve aspect ratio) ──────────────
    scale = min(max_px_w / cols, max_px_h / rows)
    out_w = max(1, int(round(cols * scale)))
    out_h = max(1, int(round(rows * scale)))

    print(
        f"  Map size    : {cols} × {rows} blocks"
    )
    print(
        f"  Output size : {out_w} × {out_h} px  (scale {scale:.3f} px/block)"
    )
    print(
        f"  Height range: Y={heightmap.min():.0f} .. Y={heightmap.max():.0f}"
    )

    # High-quality Lanczos resize + post-process smoothing
    img = img.resize((out_w, out_h), Image.LANCZOS)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH)

    img.save(output_path)
    print(f"  Saved → {output_path}")
    return img
