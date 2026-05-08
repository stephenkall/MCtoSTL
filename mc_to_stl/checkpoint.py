"""
Checkpoint system: save and restore conversion progress between runs.

Layout inside the output directory:
  .checkpoint.json   – parameters + completion flags
  heightmap_raw.npy  – raw heightmap as loaded from the save (float32)
  heightmap_work.npy – processed heightmap (ocean/island masking applied)
  ocean_mask.npy     – boolean ocean mask (present only if masking was used)

Mosaic resume: handled by checking which tile_ZZZ_XXX.stl files already
exist inside tiles/ at startup.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


_CHECKPOINT_FILE = ".checkpoint.json"
_RAW_HM_FILE = "heightmap_raw.npy"
_WORK_HM_FILE = "heightmap_work.npy"
_OCEAN_MASK_FILE = "ocean_mask.npy"


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# ── Public API ────────────────────────────────────────────────────────────────

class Checkpoint:
    """
    Manages conversion state for one output directory.

    Stages (each implies all previous are done):
      "loaded"    – heightmap_raw.npy is ready
      "processed" – heightmap_work.npy + ocean_mask.npy are ready
      "image"     – heightmap.png is written
      "stl"       – terrain.stl is written
      "mosaic"    – all tiles in tiles/ are written
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self._json_path = os.path.join(out_dir, _CHECKPOINT_FILE)
        self._state: Dict[str, Any] = {}

    # ── Persistence ───────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load existing checkpoint. Returns True if one was found."""
        data = _load_json(self._json_path)
        if data is None:
            return False
        self._state = data
        return True

    def save(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        _save_json(self._json_path, self._state)

    def clear(self) -> None:
        """Delete all checkpoint artefacts (start fresh)."""
        for name in (_CHECKPOINT_FILE, _RAW_HM_FILE, _WORK_HM_FILE, _OCEAN_MASK_FILE):
            p = os.path.join(self.out_dir, name)
            if os.path.exists(p):
                os.remove(p)
        self._delete_chunk_cache()
        self._state = {}

    # ── Incremental cleanup ───────────────────────────────────────────────

    def _delete_chunk_cache(self) -> None:
        """Remove the .chunk_cache directory (chunk NPY files)."""
        import shutil
        cache_dir = os.path.join(self.out_dir, ".chunk_cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

    def cleanup_after_load(self) -> None:
        """Delete chunk cache once the raw heightmap NPY is saved."""
        self._delete_chunk_cache()

    def cleanup_after_process(self) -> None:
        """Delete raw heightmap NPY once the work heightmap is saved."""
        p = os.path.join(self.out_dir, _RAW_HM_FILE)
        if os.path.exists(p):
            os.remove(p)

    def cleanup_after_outputs(self) -> None:
        """Delete work heightmap + ocean mask once all outputs are finished."""
        for name in (_WORK_HM_FILE, _OCEAN_MASK_FILE):
            p = os.path.join(self.out_dir, name)
            if os.path.exists(p):
                os.remove(p)

    # ── Parameters ────────────────────────────────────────────────────────

    @property
    def params(self) -> Dict[str, Any]:
        return self._state.get("params", {})

    def set_params(self, params: Dict[str, Any]) -> None:
        self._state["params"] = params
        self._state["save_path"] = params.get("save_path", "")
        self.save()

    @property
    def save_path(self) -> str:
        return self._state.get("save_path", "")

    # ── Stage flags ───────────────────────────────────────────────────────

    _STAGE_ORDER = ["loaded", "processed", "image", "stl", "mosaic"]

    def is_done(self, stage: str) -> bool:
        return stage in self._state.get("done", [])

    def mark_done(self, stage: str) -> None:
        done = self._state.setdefault("done", [])
        if stage not in done:
            done.append(stage)
        self.save()

    def unmark(self, stage: str) -> None:
        """Remove a single stage from the done list."""
        done = self._state.get("done", [])
        if stage in done:
            done.remove(stage)
        self.save()

    def unmark_from(self, stage: str) -> None:
        """Remove stage and all downstream stages from the done list."""
        if stage not in self._STAGE_ORDER:
            return
        idx = self._STAGE_ORDER.index(stage)
        to_remove = set(self._STAGE_ORDER[idx:])
        self._state["done"] = [s for s in self._state.get("done", [])
                                if s not in to_remove]
        self.save()

    # ── Heightmap I/O ─────────────────────────────────────────────────────

    def save_raw_heightmap(self, hm: np.ndarray) -> None:
        np.save(os.path.join(self.out_dir, _RAW_HM_FILE), hm)
        self.mark_done("loaded")

    def load_raw_heightmap(self) -> Optional[np.ndarray]:
        p = os.path.join(self.out_dir, _RAW_HM_FILE)
        return np.load(p) if os.path.exists(p) else None

    def save_work_heightmap(
        self, hm: np.ndarray, ocean_mask: Optional[np.ndarray]
    ) -> None:
        np.save(os.path.join(self.out_dir, _WORK_HM_FILE), hm)
        if ocean_mask is not None:
            np.save(os.path.join(self.out_dir, _OCEAN_MASK_FILE), ocean_mask)
        self.mark_done("processed")

    def load_work_heightmap(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        wp = os.path.join(self.out_dir, _WORK_HM_FILE)
        op = os.path.join(self.out_dir, _OCEAN_MASK_FILE)
        hm = np.load(wp) if os.path.exists(wp) else None
        mask = np.load(op) if os.path.exists(op) else None
        return hm, mask

    # ── Mosaic helpers ────────────────────────────────────────────────────

    def existing_tiles(self, tiles_dir: str) -> set:
        """Return set of (tz, tx) tuples for tile files that already exist."""
        import re
        done = set()
        if not os.path.isdir(tiles_dir):
            return done
        for name in os.listdir(tiles_dir):
            m = re.match(r"tile_(\d+)_(\d+)\.stl$", name)
            if m:
                done.add((int(m.group(1)), int(m.group(2))))
        return done
