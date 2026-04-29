"""
Minecraft Anvil (.mca) region file parser.
Supports Java Edition 1.8 through 1.21+.

Heightmap extraction priority per chunk:
  1. Heightmaps compound (1.13+) – packed long arrays
  2. Level.HeightMap (pre-1.13) – flat IntArray
  3. Compute top non-air block by scanning chunk sections
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

# ── NBT helpers ──────────────────────────────────────────────────────────────

def _parse_nbt(data: bytes) -> Optional[nbtlib.Compound]:
    """
    Parse raw (uncompressed) NBT bytes into a Compound, handling the outer
    tag-id + name-length header that nbtlib.Compound.parse does not consume.
    Returns None on any error.
    """
    try:
        buf = io.BytesIO(data)
        tag_id = struct.unpack("B", buf.read(1))[0]
        if tag_id != 10:
            return None
        name_len = struct.unpack(">H", buf.read(2))[0]
        buf.read(name_len)
        return nbtlib.Compound.parse(buf, byteorder="big")
    except Exception:
        return None


# ── Heightmap bit-packing ────────────────────────────────────────────────────

def _unpack_longs(longs, bits: int, count: int) -> list:
    """
    Unpack a Minecraft packed-long array.

    Supports both layouts:
    - Aligned   (1.16+): each long holds floor(64/bits) values; no value spans
                         two longs.  Expected length = ceil(count/vpL).
    - Compact (pre-1.16): values may span long boundaries (bit stream).
                         Expected length = ceil(count*bits/64).
    """
    mask = (1 << bits) - 1
    vpL = 64 // bits                             # values per long, aligned
    need_aligned = math.ceil(count / vpL)

    raw = [int(v) for v in longs]               # ensure plain Python ints

    if len(raw) >= need_aligned:
        # Aligned packing
        out = []
        for i, v in enumerate(raw):
            if v < 0:
                v += (1 << 64)
            for j in range(vpL):
                idx = i * vpL + j
                if idx >= count:
                    break
                out.append((v >> (j * bits)) & mask)
        return out

    # Compact packing (pre-1.16)
    out = []
    bit_buf = 0
    bits_in_buf = 0
    li = 0
    for _ in range(count):
        while bits_in_buf < bits and li < len(raw):
            v = raw[li]
            if v < 0:
                v += (1 << 64)
            bit_buf |= v << bits_in_buf
            bits_in_buf += 64
            li += 1
        out.append(bit_buf & mask)
        bit_buf >>= bits
        bits_in_buf -= bits
    return out


# ── Block-section fallback ───────────────────────────────────────────────────

_AIR = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})


def _is_air(palette_entry) -> bool:
    name = str(palette_entry.get("Name", "")) if hasattr(palette_entry, "get") else str(palette_entry)
    return name in _AIR


def _decode_section_to_air_mask(section) -> Optional[np.ndarray]:
    """
    Decode one 16×16×16 chunk section into a boolean array (True = air).
    Shape: (16, 16, 16) in (local_y, z, x) order.
    Returns None if the section cannot be decoded.
    """
    # ── 1.18+ ────────────────────────────────────────────────────────────
    if "block_states" in section:
        bs = section["block_states"]
        palette = list(bs.get("palette", []))
        if not palette:
            return None
        air_flags = [_is_air(p) for p in palette]
        if "data" not in bs:
            # Single-entry palette (all same block)
            val = air_flags[0]
            return np.full((16, 16, 16), val, dtype=bool)
        longs = list(bs["data"])
        bits = max(4, math.ceil(math.log2(max(len(palette), 2))))
        vals = _unpack_longs(longs, bits, 4096)
        mask = np.array([air_flags[v] for v in vals], dtype=bool)
        return mask.reshape(16, 16, 16)

    # ── 1.13–1.17 ────────────────────────────────────────────────────────
    if "BlockStates" in section and "Palette" in section:
        palette = list(section["Palette"])
        air_flags = [_is_air(p) for p in palette]
        longs = list(section["BlockStates"])
        bits = max(4, math.ceil(math.log2(max(len(palette), 2))))
        vals = _unpack_longs(longs, bits, 4096)
        mask = np.array([air_flags[v] for v in vals], dtype=bool)
        return mask.reshape(16, 16, 16)

    # ── Pre-1.13 ─────────────────────────────────────────────────────────
    if "Blocks" in section:
        raw = bytes(section["Blocks"])
        if len(raw) == 4096:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(16, 16, 16)
            return arr == 0  # block id 0 = air

    return None


def _heightmap_from_sections(chunk) -> Optional[np.ndarray]:
    """
    Compute a 16×16 surface heightmap by scanning block sections from top to
    bottom and finding the highest non-air block in each column.
    """
    sections_key = next((k for k in ("sections", "Sections") if k in chunk), None)
    if sections_key is None:
        return None
    sections = chunk[sections_key]
    if not sections:
        return None

    section_map: Dict[int, object] = {}
    for sec in sections:
        y = int(sec.get("Y", sec.get("y", 0)))
        section_map[y] = sec

    if not section_map:
        return None

    max_sy = max(section_map)
    min_sy = min(section_map)

    heightmap = np.zeros((16, 16), dtype=np.int32)
    found = np.zeros((16, 16), dtype=bool)

    for sy in range(max_sy, min_sy - 1, -1):
        if found.all():
            break
        sec = section_map.get(sy)
        if sec is None:
            continue
        try:
            air = _decode_section_to_air_mask(sec)
        except Exception:
            continue
        if air is None:
            continue
        for ly in range(15, -1, -1):
            solid = ~air[ly]             # shape (16, 16) = (z, x)
            mask = solid & ~found
            if mask.any():
                heightmap[mask] = sy * 16 + ly + 1
                found |= mask

    return heightmap


# ── Per-chunk heightmap extraction ───────────────────────────────────────────

_HM_KEYS = (
    "WORLD_SURFACE",
    "MOTION_BLOCKING",
    "WORLD_SURFACE_WG",
    "MOTION_BLOCKING_NO_LEAVES",
)


def _extract_heightmap(root: nbtlib.Compound) -> Optional[np.ndarray]:
    """
    Extract a 16×16 int32 heightmap array from a parsed chunk Compound.
    Tries three approaches in order of preference.
    """
    # Pre-1.18 wraps everything under 'Level'
    chunk = root.get("Level", root)

    # 1 ── New format: Heightmaps compound (1.13+) ────────────────────────
    if "Heightmaps" in chunk:
        hm_c = chunk["Heightmaps"]
        for key in _HM_KEYS:
            if key not in hm_c:
                continue
            longs = list(hm_c[key])
            if not longs:
                continue
            vals = _unpack_longs(longs, bits=9, count=256)
            if len(vals) == 256:
                return np.array(vals, dtype=np.int32).reshape(16, 16)

    # 2 ── Old format: HeightMap flat array (pre-1.13) ────────────────────
    if "HeightMap" in chunk:
        hm = list(chunk["HeightMap"])
        if len(hm) >= 256:
            return np.array([int(v) for v in hm[:256]], dtype=np.int32).reshape(16, 16)

    # 3 ── Fallback: compute from block sections ───────────────────────────
    return _heightmap_from_sections(chunk)


# ── Region file parsing ──────────────────────────────────────────────────────

def parse_region(
    filepath: str,
    debug: bool = False,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Parse a single .mca file.

    Returns {(chunk_x, chunk_z): 16×16 int32 height array} in global
    chunk coordinates.  Pass debug=True to print per-chunk diagnostics
    on failure.
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
                if debug:
                    print(f"    [chunk {i}] unknown compression {compression}")
                continue

            root = _parse_nbt(nbt_bytes)
            if root is None:
                if debug:
                    print(f"    [chunk {i}] NBT parse returned None")
                continue

            hm = _extract_heightmap(root)
            if hm is None:
                if debug:
                    chunk = root.get("Level", root)
                    print(f"    [chunk {i}] no heightmap; keys={list(chunk.keys())[:10]}")
                continue

            cx_local = i % 32
            cz_local = i // 32
            chunks[(rx * 32 + cx_local, rz * 32 + cz_local)] = hm

        except Exception as exc:
            if debug:
                print(f"    [chunk {i}] exception: {exc}")

    return chunks


def diagnose_region(filepath: str, max_chunks: int = 3) -> None:
    """Print diagnostic information for the first few chunks in a region file."""
    m = re.search(r"r\.(-?\d+)\.(-?\d+)\.mca$", filepath)
    if m is None:
        print(f"Not a valid region file: {filepath}")
        return

    print(f"\nDiagnosing: {filepath}")
    try:
        with open(filepath, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        print(f"  Cannot read file: {e}")
        return

    found = 0
    for i in range(1024):
        if found >= max_chunks:
            break

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

        try:
            if compression == 1:
                nbt_bytes = gzip.decompress(payload)
            elif compression == 2:
                nbt_bytes = zlib.decompress(payload)
            elif compression == 3:
                nbt_bytes = payload
            else:
                print(f"  Chunk {i}: unknown compression={compression}")
                found += 1
                continue

            root = _parse_nbt(nbt_bytes)
            if root is None:
                print(f"  Chunk {i}: NBT parse failed (tag_id={nbt_bytes[0]})")
                found += 1
                continue

            chunk = root.get("Level", root)
            keys = list(chunk.keys())
            dv = int(chunk.get("DataVersion", -1))
            status = str(chunk.get("Status", chunk.get("status", "?")))

            has_hm_new = "Heightmaps" in chunk
            has_hm_old = "HeightMap" in chunk
            has_sections = any(k in chunk for k in ("sections", "Sections"))

            hm_keys = list(chunk["Heightmaps"].keys()) if has_hm_new else []

            hm = _extract_heightmap(root)

            print(f"  Chunk {i}: DataVersion={dv}  Status={status!r}")
            print(f"    Keys           : {keys[:12]}")
            print(f"    Heightmaps(new): {has_hm_new}  keys={hm_keys}")
            print(f"    HeightMap(old) : {has_hm_old}")
            print(f"    Has sections   : {has_sections}")
            print(f"    Extracted HM   : {'OK  min=%d max=%d' % (hm.min(), hm.max()) if hm is not None else 'FAILED'}")
            found += 1

        except Exception as exc:
            print(f"  Chunk {i}: exception – {exc}")
            found += 1
