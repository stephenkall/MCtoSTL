"""
Create a minimal synthetic Minecraft Java Edition save for testing.

Generates 4 region files (r.0.0, r.1.0, r.0.1, r.1.1), each with 4×4
chunks, so the final map is 128×128 blocks.  Heights follow a sine-wave
landscape including negative values to exercise the full color range.
"""

import gzip
import io
import math
import os
import struct
import sys

import nbtlib
import numpy as np


def _pack_longs_aligned(values: list, bits: int) -> list:
    """Pack integers into longs using the 1.16+ aligned scheme."""
    vpL = 64 // bits
    mask = (1 << bits) - 1
    longs = []
    for i in range(0, len(values), vpL):
        chunk = values[i : i + vpL]
        v = 0
        for j, val in enumerate(chunk):
            v |= (int(val) & mask) << (j * bits)
        # Convert to signed int64
        if v >= (1 << 63):
            v -= 1 << 64
        longs.append(v)
    return longs


def make_chunk_nbt(cx: int, cz: int, heights_16x16: np.ndarray) -> bytes:
    """Create minimal chunk NBT bytes (1.18+ format)."""
    flat = heights_16x16.flatten().tolist()
    longs = _pack_longs_aligned(flat, bits=9)

    # nbtlib.File is the correct root-level writer in nbtlib 2.x
    nbt_file = nbtlib.File(
        {
            "DataVersion": nbtlib.Int(2860),
            "xPos": nbtlib.Int(cx),
            "zPos": nbtlib.Int(cz),
            "yPos": nbtlib.Int(-4),
            "Status": nbtlib.String("minecraft:full"),
            "Heightmaps": nbtlib.Compound(
                {
                    "WORLD_SURFACE": nbtlib.LongArray(longs),
                }
            ),
        }
    )
    buf = io.BytesIO()
    nbt_file.write(buf, byteorder="big")
    return buf.getvalue()


def write_region(filepath: str, chunks: dict) -> None:
    """Write a .mca file containing the given {(cx_local, cz_local): heights}."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Build chunk payload bytes
    payloads = {}
    for (lx, lz), heights in chunks.items():
        raw_nbt = make_chunk_nbt(lx, lz, heights)
        compressed = zlib_compress(raw_nbt)
        # 4-byte length (compression byte + data), 1-byte compression type
        chunk_data = struct.pack(">I", len(compressed) + 1) + b"\x02" + compressed
        # Pad to 4 KiB sectors
        pad = (4096 - len(chunk_data) % 4096) % 4096
        payloads[(lx, lz)] = chunk_data + b"\x00" * pad

    location_table = bytearray(4096)
    timestamp_table = bytearray(4096)
    chunk_bytes = bytearray()

    sector = 2  # sectors 0 and 1 are the header
    for lz in range(32):
        for lx in range(32):
            idx = lz * 32 + lx
            if (lx, lz) not in payloads:
                continue
            data = payloads[(lx, lz)]
            n_sectors = len(data) // 4096
            loc = (sector << 8) | n_sectors
            struct.pack_into(">I", location_table, idx * 4, loc)
            chunk_bytes.extend(data)
            sector += n_sectors

    with open(filepath, "wb") as fh:
        fh.write(bytes(location_table))
        fh.write(bytes(timestamp_table))
        fh.write(bytes(chunk_bytes))


def zlib_compress(data: bytes) -> bytes:
    import zlib
    return zlib.compress(data)


def make_test_save(save_dir: str) -> None:
    region_dir = os.path.join(save_dir, "region")
    os.makedirs(region_dir, exist_ok=True)

    with open(os.path.join(save_dir, "level.dat"), "wb") as fh:
        fh.write(b"\x00")

    # 8×8 chunks = 128×128 blocks in a single region file r.0.0
    # Heights include negative values to exercise the full color range
    TOTAL = 128
    x_idx, z_idx = np.meshgrid(np.arange(TOTAL), np.arange(TOTAL))
    heights_full = (
        40.0 * np.sin(x_idx / 20.0) * np.cos(z_idx / 15.0)
        + 20.0 * np.sin(x_idx / 8.0 + 1.0)
        - 10.0  # forces some values below 0 → blue channel in image
    ).astype(np.float32)

    chunks = {}
    for cz_local in range(8):
        for cx_local in range(8):
            bx = cx_local * 16
            bz = cz_local * 16
            tile = heights_full[bz : bz + 16, bx : bx + 16]
            # Heightmap stores absolute Y ≥ 0; shift so minimum maps to Y=64
            shift = int(tile.min()) - 64
            hm = (tile - shift).astype(np.int32)
            chunks[(cx_local, cz_local)] = hm

    write_region(os.path.join(region_dir, "r.0.0.mca"), chunks)

    print(f"Test save written to: {save_dir}")
    print(f"  128 × 128 blocks in 1 region file (8×8 chunks)")
    print(f"  Height range: {heights_full.min():.1f} .. {heights_full.max():.1f}")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mc_test_save"
    make_test_save(dest)
