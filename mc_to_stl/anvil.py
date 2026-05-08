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

# Blocks to skip in ground-only mode (plants, wood, decorative, man-made).
# Matched as substrings of the bare block name (after stripping "namespace:").
# Unknown / modded blocks are treated as solid ground (conservative fallback).
_NON_TERRAIN_SUBSTRINGS = (
    # Leaves / foliage
    "_leaves", "azalea_leaves",
    # Logs / wood (above-ground trunks)
    "_log", "_wood", "stripped_",
    # Man-made wood
    "_planks", "_slab", "_stairs", "_fence", "_gate", "_wall",
    "_door", "_trapdoor", "_button", "_pressure_plate",
    # Glass
    "glass",
    # Plants standing on ground
    "grass",        # tall_grass, grass (short) — NOT grass_block
    "fern", "large_fern",
    "dead_bush", "bush",
    "bamboo", "sugarcane", "cactus",
    "kelp", "seagrass", "coral",
    "vine", "hanging_roots", "spore_blossom",
    "azalea",       # azalea itself (not azalea_leaves, matched above)
    "dripleaf", "glow_lichen", "moss_carpet",
    # Flowers
    "dandelion", "poppy", "orchid", "allium", "bluet",
    "tulip", "daisy", "cornflower", "lily_of", "sunflower",
    "lilac", "rose_bush", "peony", "wither_rose",
    "pitcher", "torchflower",
    # Mushrooms on ground
    "brown_mushroom", "red_mushroom",
    "crimson_fungus", "warped_fungus",
    "crimson_roots", "warped_roots", "nether_sprouts",
    # Decorative / structural
    "torch", "lantern", "chain", "bars",
    "carpet", "banner", "sign", "bed",
    "flower_pot", "skull", "head",
    "lever", "rail", "ladder",
    "cobweb", "string",
    "chest", "barrel", "barrel",
    "bookshelf", "scaffolding",
    "sapling",
)

# Exceptions: names containing a substring from above but that ARE terrain
_NON_TERRAIN_EXCEPTIONS = frozenset({
    "minecraft:grass_block",
    "minecraft:mycelium",       # contains no substring but keep safe
    "minecraft:podzol",
    "minecraft:dirt_path",
    "minecraft:farmland",
})


def _is_non_terrain(name: str) -> bool:
    """Return True if this block should be skipped in ground-only scanning."""
    if name in _AIR:
        return True
    if name in _NON_TERRAIN_EXCEPTIONS:
        return False
    bare = name.split(":")[-1]
    return any(sub in bare for sub in _NON_TERRAIN_SUBSTRINGS)


def _is_air(palette_entry) -> bool:
    name = str(palette_entry.get("Name", "")) if hasattr(palette_entry, "get") else str(palette_entry)
    return name in _AIR


def _get_block_name(palette_entry) -> str:
    return str(palette_entry.get("Name", "")) if hasattr(palette_entry, "get") else str(palette_entry)


def _is_water(name: str) -> bool:
    """Return True if this block is water (flowing or static)."""
    return name in {"minecraft:water", "minecraft:flowing_water"}


def _decode_section(
    section, ground_only: bool = False
) -> Optional[np.ndarray]:
    """
    Decode one 16×16×16 chunk section into a boolean "skip" mask
    (True = skip this block when scanning downward).

    ground_only=False: skip only air
    ground_only=True:  skip air + plants + wood + structures
    Shape: (16, 16, 16) in (local_y, z, x) order.
    """
    skip_fn = _is_non_terrain if ground_only else _is_air

    # ── 1.18+ ────────────────────────────────────────────────────────────
    if "block_states" in section:
        bs = section["block_states"]
        palette = list(bs.get("palette", []))
        if not palette:
            return None
        skip_flags = [skip_fn(_get_block_name(p)) for p in palette]
        if "data" not in bs:
            val = skip_flags[0]
            return np.full((16, 16, 16), val, dtype=bool)
        longs = list(bs["data"])
        bits = max(4, math.ceil(math.log2(max(len(palette), 2))))
        vals = _unpack_longs(longs, bits, 4096)
        mask = np.array([skip_flags[v] for v in vals], dtype=bool)
        return mask.reshape(16, 16, 16)

    # ── 1.13–1.17 ────────────────────────────────────────────────────────
    if "BlockStates" in section and "Palette" in section:
        palette = list(section["Palette"])
        skip_flags = [skip_fn(_get_block_name(p)) for p in palette]
        longs = list(section["BlockStates"])
        bits = max(4, math.ceil(math.log2(max(len(palette), 2))))
        vals = _unpack_longs(longs, bits, 4096)
        mask = np.array([skip_flags[v] for v in vals], dtype=bool)
        return mask.reshape(16, 16, 16)

    # ── Pre-1.13 ─────────────────────────────────────────────────────────
    if "Blocks" in section:
        raw = bytes(section["Blocks"])
        if len(raw) == 4096:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(16, 16, 16)
            return arr == 0  # pre-1.13: only block id 0 is air


    return None


def _section_map(chunk) -> Optional[Dict[int, object]]:
    sections_key = next((k for k in ("sections", "Sections") if k in chunk), None)
    if sections_key is None:
        return None
    sections = chunk[sections_key]
    if not sections:
        return None
    sm: Dict[int, object] = {}
    for sec in sections:
        y = int(sec.get("Y", sec.get("y", 0)))
        sm[y] = sec
    return sm or None


def _heightmap_from_sections(
    chunk,
    ground_only: bool = False,
    detect_floating: bool = False,
) -> Optional[np.ndarray]:
    """
    Compute a 16×16 surface heightmap from block sections.

    detect_floating=False (fast): top-down scan, returns highest non-air block.
    detect_floating=True  (3D):   builds full 3D solid array per chunk, labels
        connected components (scipy 6-connectivity), seeds "grounded" from the
        bottom section (bedrock level), and returns the highest GROUNDED block
        per column.  Truly isolated floating blocks are discarded with no height
        threshold.  Overhangs are kept correctly because horizontal neighbours
        are captured by the 6-connectivity labelling.
    """
    sm = _section_map(chunk)
    if sm is None:
        return None

    max_sy = max(sm)
    min_sy = min(sm)

    if not detect_floating:
        # ── Fast top-down scan ─────────────────────────────────────────────
        heightmap = np.zeros((16, 16), dtype=np.int32)
        found = np.zeros((16, 16), dtype=bool)
        for sy in range(max_sy, min_sy - 1, -1):
            if found.all():
                break
            sec = sm.get(sy)
            if sec is None:
                continue
            try:
                skip = _decode_section(sec, ground_only=ground_only)
            except Exception:
                continue
            if skip is None:
                continue
            for ly in range(15, -1, -1):
                solid = ~skip[ly]
                mask = solid & ~found
                if mask.any():
                    heightmap[mask] = sy * 16 + ly + 1
                    found |= mask
        return heightmap

    # ── 3D connected-component floating-block detection ────────────────────
    from scipy.ndimage import label as _label

    y_min   = min_sy * 16
    y_max   = (max_sy + 1) * 16
    n_y     = y_max - y_min          # number of Y levels in this chunk

    # solid[iy, iz, ix] = True if the block is solid (not to be skipped)
    solid = np.zeros((n_y, 16, 16), dtype=bool)
    for sy in range(min_sy, max_sy + 1):
        sec = sm.get(sy)
        if sec is None:
            continue
        try:
            skip = _decode_section(sec, ground_only=ground_only)
        except Exception:
            continue
        if skip is None:
            continue
        y_off = (sy - min_sy) * 16
        solid[y_off : y_off + 16] = ~skip     # (local_y, z, x)

    if not solid.any():
        return np.full((16, 16), y_min, dtype=np.int32)

    labeled, _ = _label(solid)                # 6-connectivity (default)

    # Grounded = components that include at least one block in the BOTTOM section.
    # The bottom section is bedrock level — anything there is definitionally terrain.
    bottom_slab = labeled[:16]                 # first 16 Y-levels
    ground_ids_arr = np.unique(bottom_slab)
    ground_ids_arr = ground_ids_arr[ground_ids_arr != 0]

    if ground_ids_arr.size == 0:
        # No blocks in the bottom section — seed from lowest available blocks.
        for iy in range(n_y):
            ids = np.unique(labeled[iy])
            ids = ids[ids != 0]
            if ids.size:
                ground_ids_arr = ids
                break

    if ground_ids_arr.size == 0:
        return np.full((16, 16), y_min, dtype=np.int32)

    # Build a LUT: grounded[label_id] = True
    lut = np.zeros(labeled.max() + 1, dtype=bool)
    lut[ground_ids_arr] = True
    grounded = lut[labeled]                    # (n_y, 16, 16) bool

    # Highest grounded block per column, vectorised.
    # Flip Y axis so argmax finds the TOPMOST grounded block.
    has_any  = grounded.any(axis=0)            # (16, 16)
    idx_top  = np.argmax(grounded[::-1], axis=0)   # index from top (0 = topmost)
    # Actual Y of that block, then +1 for the "surface above" convention.
    surface_y = (y_max - idx_top).astype(np.int32)
    return np.where(has_any, surface_y, np.int32(y_min))


def _water_map_from_sections(
    chunk,
    heightmap: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Build a 16×16 bool map indicating which columns have water as the top block.

    For each (x, z) column, checks the block type at the Y given by heightmap[z, x].
    Returns True if that block is water, False otherwise.
    """
    sm = _section_map(chunk)
    if sm is None:
        return None

    rows, cols = heightmap.shape
    is_water = np.zeros((rows, cols), dtype=bool)

    for sy in sorted(sm.keys()):
        sec = sm[sy]
        if sec is None:
            continue
        try:
            # Extract palette
            palette = None
            if "block_states" in sec:
                bs = sec.get("block_states", {})
                palette = list(bs.get("palette", []))
            elif "Palette" in sec:
                palette = list(sec["Palette"])
            if not palette:
                continue
            block_names = [_get_block_name(p) for p in palette]
            water_flags = [_is_water(name) for name in block_names]

            # Decode block states
            if "block_states" in sec:
                bs = sec["block_states"]
                if "data" not in bs:
                    # All blocks are palette[0]
                    continue
                longs = list(bs["data"])
                bits = max(4, math.ceil(math.log2(max(len(palette), 2))))
                block_ids = _unpack_longs(longs, bits, 4096)
                blocks_3d = np.array(block_ids, dtype=np.uint16).reshape(16, 16, 16)
            elif "BlockStates" in sec and "Palette" in sec:
                longs = list(sec["BlockStates"])
                bits = max(4, math.ceil(math.log2(max(len(palette), 2))))
                block_ids = _unpack_longs(longs, bits, 4096)
                blocks_3d = np.array(block_ids, dtype=np.uint16).reshape(16, 16, 16)
            else:
                continue

            # Check each column.
            # heightmap[z,x] = Y+1 of surface block (Minecraft convention: first
            # air above).  Subtract 1 to get the actual surface block Y.
            y_min = sy * 16
            y_max = y_min + 16
            for z in range(16):
                for x in range(16):
                    block_y = int(heightmap[z, x]) - 1  # actual surface block
                    if block_y < 0:
                        continue
                    if not (y_min <= block_y < y_max):
                        continue
                    y_local = block_y - y_min
                    block_id = int(blocks_3d[y_local, z, x])
                    if block_id < len(water_flags) and water_flags[block_id]:
                        is_water[z, x] = True
        except Exception:
            continue

    return is_water


# ── Per-chunk heightmap extraction ───────────────────────────────────────────

# Heightmap key preference by mode
_HM_KEYS_SURFACE = (
    "WORLD_SURFACE",
    "MOTION_BLOCKING",
    "WORLD_SURFACE_WG",
    "MOTION_BLOCKING_NO_LEAVES",
)
_HM_KEYS_GROUND = (
    # MOTION_BLOCKING_NO_LEAVES ignores leaves, keeps water → shows rivers
    "MOTION_BLOCKING_NO_LEAVES",
    "MOTION_BLOCKING",
    "WORLD_SURFACE",
)


def _extract_heightmap(
    root: nbtlib.Compound,
    ground_only: bool = False,
    detect_floating: bool = False,
    force_scan: bool = False,
) -> Optional[np.ndarray]:
    """
    Extract a 16×16 int32 heightmap from a parsed chunk Compound.

    ground_only=False:    standard WORLD_SURFACE (includes trees, structures)
    ground_only=True:     prefers MOTION_BLOCKING_NO_LEAVES; section fallback
                          skips plants, wood, and decorative blocks
    detect_floating=True: always use section-based 3D connectivity analysis
                          (bypasses Heightmaps compound) to exclude floating blocks
    force_scan=True:      always scan block sections, ignore stored Heightmaps.
                          Use when the stored heightmap is stale (e.g. terrain was
                          raised after the chunk was first generated in an older
                          version).  Slower than reading stored values but matches
                          what Unmined does.
    """
    chunk = root.get("Level", root)

    # 3D floating-block detection and force_scan both require section data.
    if not detect_floating and not force_scan:
        hm_keys = _HM_KEYS_GROUND if ground_only else _HM_KEYS_SURFACE

        # 1 ── New format: Heightmaps compound (1.13+) ────────────────────
        if "Heightmaps" in chunk:
            hm_c = chunk["Heightmaps"]
            for key in hm_keys:
                if key not in hm_c:
                    continue
                longs = list(hm_c[key])
                if not longs:
                    continue
                vals = _unpack_longs(longs, bits=9, count=256)
                if len(vals) == 256:
                    return np.array(vals, dtype=np.int32).reshape(16, 16)

        # 2 ── Old format: HeightMap flat array (pre-1.13) ────────────────
        if "HeightMap" in chunk:
            hm = list(chunk["HeightMap"])
            if len(hm) >= 256:
                return np.array([int(v) for v in hm[:256]], dtype=np.int32).reshape(16, 16)

    # 3 ── Section scan (always used when detect_floating=True) ───────────
    return _heightmap_from_sections(
        chunk, ground_only=ground_only, detect_floating=detect_floating
    )


# ── Region file parsing ──────────────────────────────────────────────────────

def parse_region(
    filepath: str,
    debug: bool = False,
    ground_only: bool = False,
    detect_floating: bool = False,
    force_scan: bool = False,
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

            hm = _extract_heightmap(root, ground_only=ground_only,
                                     detect_floating=detect_floating,
                                     force_scan=force_scan)
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


def parse_region_with_water(
    filepath: str,
    debug: bool = False,
    ground_only: bool = False,
    detect_floating: bool = False,
    force_scan: bool = False,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    Parse a single .mca file, returning both heightmaps and water maps.

    Returns:
      (heightmaps, water_maps) where each is {(chunk_x, chunk_z): 16×16 array}
      heightmaps: int32 height values
      water_maps: bool arrays (True where top block is water)
    """
    heightmaps: Dict[Tuple[int, int], np.ndarray] = {}
    water_maps: Dict[Tuple[int, int], np.ndarray] = {}

    m = re.search(r"r\.(-?\d+)\.(-?\d+)\.mca$", filepath)
    if m is None:
        return heightmaps, water_maps
    rx, rz = int(m.group(1)), int(m.group(2))

    try:
        with open(filepath, "rb") as fh:
            raw = fh.read()
    except OSError:
        return heightmaps, water_maps

    if len(raw) < 8192:
        return heightmaps, water_maps

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

            hm = _extract_heightmap(root, ground_only=ground_only,
                                     detect_floating=detect_floating,
                                     force_scan=force_scan)
            if hm is None:
                if debug:
                    chunk = root.get("Level", root)
                    print(f"    [chunk {i}] no heightmap; keys={list(chunk.keys())[:10]}")
                continue

            wm = _water_map_from_sections(root, hm)
            if wm is None:
                wm = np.zeros((16, 16), dtype=bool)

            cx_local = i % 32
            cz_local = i // 32
            chunk_key = (rx * 32 + cx_local, rz * 32 + cz_local)
            heightmaps[chunk_key] = hm
            water_maps[chunk_key] = wm

        except Exception as exc:
            if debug:
                print(f"    [chunk {i}] exception: {exc}")

    return heightmaps, water_maps


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

            hm = _extract_heightmap(root, ground_only=False)
            hm_g = _extract_heightmap(root, ground_only=True)

            print(f"  Chunk {i}: DataVersion={dv}  Status={status!r}")
            print(f"    Keys           : {keys[:12]}")
            print(f"    Heightmaps(new): {has_hm_new}  keys={hm_keys}")
            print(f"    HeightMap(old) : {has_hm_old}")
            print(f"    Has sections   : {has_sections}")
            print(f"    Extracted HM   : {'OK  min=%d max=%d' % (hm.min(), hm.max()) if hm is not None else 'FAILED'}")
            print(f"    Ground-only HM : {'OK  min=%d max=%d' % (hm_g.min(), hm_g.max()) if hm_g is not None else 'FAILED (will use WORLD_SURFACE)'}")
            found += 1

        except Exception as exc:
            print(f"  Chunk {i}: exception – {exc}")
            found += 1
