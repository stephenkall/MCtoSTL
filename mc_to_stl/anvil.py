"""
Minecraft Anvil (.mca) region file parser.
Supports Java Edition formats from 1.12 through 1.21+.
"""

import gzip
import io
import math
import re
import struct
import zlib
from typing import Dict, Optional, Tuple

import numpy as np
import nbtlib


def _parse_nbt_bytes(data: bytes):
    """
    Parse raw NBT bytes (big-endian, no gzip wrapper) into a Compound.

    nbtlib 2.x does not expose read_nbt(); we handle the outer tag header
    (tag-id byte + name-length short + name) manually, then delegate to
    Compound.parse for the payload.
    """
    buf = io.BytesIO(data)
    tag_id = struct.unpack("B", buf.read(1))[0]
    if tag_id != 10:  # 10 = TAG_Compound
        raise ValueError(f"Expected TAG_Compound (10), got {tag_id}")
    name_len = struct.unpack(">H", buf.read(2))[0]
    buf.read(name_len)  # skip root name (usually empty)
    return nbtlib.Compound.parse(buf, byteorder="big")


# ── Heightmap bit-packing ────────────────────────────────────────────────────

def _unpack_longs(longs: list, bits: int, count: int) -> list:
    """
    Unpack a Minecraft packed-long array into individual integer values.

    Minecraft uses two packing layouts:
    - Pre-1.16:  values MAY span long boundaries (compact packing).
    - 1.16+:     values never span long boundaries (aligned packing).

    We detect which to use by checking the array length against expectations.
    """
    mask = (1 << bits) - 1
    values_per_long_aligned = 64 // bits
    needed_aligned = math.ceil(count / values_per_long_aligned)

    if len(longs) >= needed_aligned:
        # Aligned packing (1.16+)
        out = []
        for i, raw in enumerate(longs):
            v = raw if raw >= 0 else raw + (1 << 64)
            for j in range(values_per_long_aligned):
                idx = i * values_per_long_aligned + j
                if idx >= count:
                    break
                out.append((v >> (j * bits)) & mask)
        return out

    # Compact packing (pre-1.16)
    out = []
    bit_buf = 0
    bits_in_buf = 0
    long_idx = 0
    for _ in range(count):
        while bits_in_buf < bits and long_idx < len(longs):
            v = longs[long_idx]
            if v < 0:
                v += (1 << 64)
            bit_buf |= v << bits_in_buf
            bits_in_buf += 64
            long_idx += 1
        out.append(bit_buf & mask)
        bit_buf >>= bits
        bits_in_buf -= bits
    return out


# ── NBT extraction ───────────────────────────────────────────────────────────

_HM_KEYS = (
    "WORLD_SURFACE",
    "MOTION_BLOCKING",
    "WORLD_SURFACE_WG",
    "MOTION_BLOCKING_NO_LEAVES",
)


def _extract_chunk_heightmap(root) -> Optional[np.ndarray]:
    """
    Return a 16×16 int32 array of surface Y-values for a chunk.
    Returns None if no heightmap data is found.
    """
    try:
        # Pre-1.18 saves wrap everything under 'Level'
        chunk = root.get("Level", root)

        hm_compound = chunk.get("Heightmaps")
        if hm_compound is None:
            return None

        for key in _HM_KEYS:
            if key not in hm_compound:
                continue
            longs = list(hm_compound[key])
            values = _unpack_longs(longs, bits=9, count=256)
            if len(values) == 256:
                return np.array(values, dtype=np.int32).reshape(16, 16)

    except Exception:
        pass
    return None


# ── Region file parsing ──────────────────────────────────────────────────────

def parse_region(filepath: str) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Parse a single .mca file.

    Returns a dict mapping (chunk_x, chunk_z) in global chunk coordinates
    to 16×16 int32 height arrays.
    """
    chunks: Dict[Tuple[int, int], np.ndarray] = {}

    m = re.search(r"r\.(-?\d+)\.(-?\d+)\.mca$", filepath)
    if m is None:
        return chunks
    rx, rz = int(m.group(1)), int(m.group(2))

    try:
        with open(filepath, "rb") as fh:
            raw = fh.read()
    except OSError:
        return chunks

    if len(raw) < 8192:
        return chunks

    for i in range(1024):
        loc = struct.unpack_from(">I", raw, i * 4)[0]
        sector_offset = loc >> 8
        sector_count = loc & 0xFF

        if sector_offset == 0 or sector_count == 0:
            continue

        byte_off = sector_offset * 4096
        if byte_off + 5 > len(raw):
            continue

        data_len = struct.unpack_from(">I", raw, byte_off)[0]
        compression = raw[byte_off + 4]
        payload = raw[byte_off + 5 : byte_off + 4 + data_len]

        if len(payload) < data_len - 1:
            continue

        try:
            if compression == 1:
                nbt_bytes = gzip.decompress(payload)
            elif compression == 2:
                nbt_bytes = zlib.decompress(payload)
            elif compression == 3:
                nbt_bytes = payload
            else:
                continue

            root = _parse_nbt_bytes(nbt_bytes)
            hm = _extract_chunk_heightmap(root)
            if hm is not None:
                cx_local = i % 32
                cz_local = i // 32
                chunks[(rx * 32 + cx_local, rz * 32 + cz_local)] = hm
        except Exception:
            continue

    return chunks
