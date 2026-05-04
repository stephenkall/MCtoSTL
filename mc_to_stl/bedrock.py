"""
Minecraft Bedrock Edition world parser.

Reads chunk surface heightmaps from the LevelDB database stored in
<world>/db/.

Dependency
----------
Bedrock uses a modified LevelDB that replaces Snappy compression with
Zlib.  Standard `plyvel` or `leveldb` packages use Snappy and will
silently corrupt or fail to read the blocks.  Use `amulet-leveldb`:

    pip install amulet-leveldb

Key layout (overworld)
----------------------
Data2D (tag 0x2B)
  key   : [cx:int32 LE][cz:int32 LE][0x2B]  — 9 bytes
  value : 512 bytes surface heights (256 × int16 LE, index = z*16 + x)
           + 256 bytes biome data (unused here)

For Nether / The End the key has an extra 4-byte dimension field
(making it 13 bytes); those are skipped — only the overworld is parsed.

ground_only
-----------
Bedrock does not store pre-filtered (no-leaves) heightmaps in Data2D.
Emulating ground_only would require scanning all sub-chunk block
palettes, which is significantly slower.  The flag is accepted but
silently ignored; the surface heightmap is always used.
"""

import struct
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


_TAG_DATA2D = 0x2B  # surface heightmap + biome data (9-byte overworld key)


def _open_db(db_path: str):
    """Open a Bedrock LevelDB; raises ImportError with install hint if missing."""
    try:
        from leveldb import LevelDB  # provided by amulet-leveldb
    except ImportError:
        raise ImportError(
            "\n  Bedrock worlds require the 'amulet-leveldb' package.\n"
            "  Install with:  pip install amulet-leveldb\n"
            "  (The standard 'leveldb' or 'plyvel' packages use Snappy\n"
            "   compression and cannot read Bedrock databases.)"
        )
    return LevelDB(str(db_path))


def parse_bedrock_world(
    db_path: str,
    debug: bool = False,
    ground_only: bool = False,  # accepted but unused — see module docstring
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Load overworld surface heightmaps from a Bedrock world's LevelDB database.

    Returns
    -------
    dict  {(chunk_x, chunk_z): 16×16 int32 ndarray}
    """
    db = _open_db(db_path)
    chunks: Dict[Tuple[int, int], np.ndarray] = {}
    seen = 0

    try:
        with tqdm(desc="  Reading Bedrock DB", unit=" chunks", ncols=80) as pbar:
            for key, value in db.iterate():
                # Only overworld Data2D keys are exactly 9 bytes with tag 0x2B
                if len(key) != 9 or key[8] != _TAG_DATA2D:
                    continue
                if len(value) < 512:
                    if debug:
                        cx, cz = struct.unpack_from("<ii", key, 0)
                        print(f"  Short Data2D for chunk ({cx},{cz}): "
                              f"{len(value)} bytes — skipped.")
                    continue

                cx, cz = struct.unpack_from("<ii", key, 0)
                # heights stored as int16 LE, z*16 + x indexing → shape (16, 16)
                heights = (
                    np.frombuffer(value[:512], dtype="<i2")
                    .reshape(16, 16)
                    .astype(np.int32)
                )
                chunks[(cx, cz)] = heights
                seen += 1
                pbar.update(1)
    finally:
        db.close()

    return chunks


def diagnose_bedrock_world(db_path: str, max_chunks: int = 5) -> None:
    """Print a sample of chunk keys and height ranges for debugging."""
    db = _open_db(db_path)
    shown = 0
    try:
        for key, value in db.iterate():
            if len(key) != 9 or key[8] != _TAG_DATA2D:
                continue
            cx, cz = struct.unpack_from("<ii", key, 0)
            if len(value) >= 512:
                h = np.frombuffer(value[:512], dtype="<i2")
                print(f"  Chunk ({cx:4d},{cz:4d})  "
                      f"Y={h.min():.0f}..{h.max():.0f}  "
                      f"(mean {h.mean():.1f})")
            else:
                print(f"  Chunk ({cx:4d},{cz:4d})  — short record ({len(value)} bytes)")
            shown += 1
            if shown >= max_chunks:
                break
    finally:
        db.close()
    if shown == 0:
        print("  No Data2D records found — may not be a valid Bedrock world.")
