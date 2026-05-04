# MCtoSTL

Convert a Minecraft world save into a color-coded heightmap image and 3-D printable STL terrain models.

Supports **Java Edition** (`.mca` Anvil files) and **Bedrock Edition** (LevelDB `db/` folder).

---

## Outputs

| File | Description |
|------|-------------|
| `heightmap.png` | Color-coded relief image: red = highest, green = sea level, blue = below sea, steel-blue = open ocean |
| `terrain.stl` | Single watertight solid of the entire terrain |
| `tiles/tile_ZZZ_XXX.stl` | Mosaic pack of printable tiles that fit your print bed |

---

## Quick start

```bash
pip install -r requirements.txt
# Bedrock worlds also need:
pip install amulet-leveldb

python mc_to_stl.py
```

The script asks all configuration questions **upfront** before any processing begins, so you can answer the prompts and walk away.

---

## Requirements

```
numpy>=1.24
Pillow>=9.0
scipy>=1.10
nbtlib>=2.0      # Java Edition only
tqdm>=4.60
```

**Bedrock Edition** additionally requires:

```
amulet-leveldb   # pip install amulet-leveldb
```

Standard `leveldb` and `plyvel` packages use **Snappy** compression and cannot read Bedrock databases. `amulet-leveldb` uses Zlib (the compression Bedrock actually uses) and provides pre-built Windows wheels.

---

## Configuration options

### Minecraft Save Folder
Path to the world save directory. The format (Java or Bedrock) is auto-detected.

### Output directory
Where to write the outputs. Defaults to `mc_output/` next to the save folder.

### Performance *(Java only)*
Number of parallel worker processes for parsing `.mca` region files. Defaults to all available CPU cores. Region files are parsed in separate processes (bypassing the GIL) and cached after the first run — subsequent runs are nearly instant.

### Block filtering *(Java only)*
- **Ground-only (recommended):** Uses `MOTION_BLOCKING_NO_LEAVES` — ignores trees, leaves, plants, and man-made structures. Shows the actual terrain surface.
- **World surface:** Uses `WORLD_SURFACE` — faster, but tree canopies and building roofs contribute to the heightmap.

### Gaussian blur (anti-aliasing)
Applied to all outputs before rendering or meshing. Recommended: 1.0–2.0. Higher = smoother, lower = more block detail.

### Ocean / sea masking
Detects the flat water surface in the heightmap histogram and masks out open-sea areas. Ocean cells are **flattened to the base plate** in STL output — coastlines end at the shore with no raised sea surface.

- **Sea level** — auto-detected by default (histogram peak in the lower 60th percentile). Override if auto-detection picks the wrong value.
- **Min ocean area** — water bodies smaller than this (blocks²) are kept as lakes/rivers rather than marked as ocean.
- **Min island area** — land components smaller than this are absorbed into the ocean mask as noise.

### Floating block removal *(Java only)*
Full 3-D connectivity analysis per chunk. Blocks that are not connected (6-directional) to the bottom section of their chunk are discarded — removes floating platforms, isolated mid-air artefacts, and single hanging blocks. Overhangs are safe: they connect to the cliff face through horizontal neighbours. About 2–3× slower per chunk than the default heightmap method.

### Heightmap image
- **Max width / height (px):** Output image dimensions. The larger side is capped at this value; aspect ratio is preserved. The heightmap is downsampled to the output resolution before processing — memory usage scales with output size, not source map size.
- **Relief gamma:** Exponent applied to relative altitude before colour mapping. Values < 1.0 stretch low-relief areas for more visible colour variation on flat maps (0.5–0.7 recommended for maps like Westeros). 1.0 = linear.

### STL physical dimensions
Maximum bounding box for the printed model in mm. Aspect ratio is always preserved — the smaller dimension will be less than its maximum.

- **Max X / Y (mm):** Horizontal bounds.
- **Max Z (mm):** Altitude relief scale. Independent of XY — use a higher value to exaggerate terrain height.
- **Base plate thickness (mm):** Solid base below the lowest terrain point.

### STL mesh resolution
Maximum number of vertices on the longest side. The heightmap is downsampled to this resolution before meshing. Higher = more detail and larger files.

| Value | Typical file size | Use case |
|-------|-------------------|----------|
| 500 | ~3 MB | Quick preview |
| 1000 | ~10 MB | Most desktop printers |
| 1500 | ~23 MB | High-detail prints |
| 2000 | ~40 MB | Large-format printers |

### Mosaic tile dimensions
Tile width and height in mm — set to your print bed size. Tiles share a consistent Z scale so they align when assembled.

**Skip 100%-ocean tiles:** Tiles that are entirely ocean after masking would print as featureless flat base plates. Enable this to skip them and save filament.

---

## Resume / checkpoints

The script saves a checkpoint after each completed stage:

| Stage | Checkpoint |
|-------|------------|
| `loaded` | Raw heightmap array (`.npy`) |
| `processed` | Ocean-masked heightmap + ocean mask (`.npy`) |
| `image` | Stage flag |
| `stl` | Stage flag |
| `mosaic` | Stage flag + individual tile files |

If the process is interrupted, rerun the same command. The script detects the existing checkpoint and asks whether to resume. Pass `--fresh` to discard the checkpoint and start over.

---

## Command-line flags

```
python mc_to_stl.py [save_path] [--fresh] [--diagnose]
```

| Flag | Effect |
|------|--------|
| *(none)* | Interactive prompts |
| `save_path` | Pre-fill the save folder path |
| `--fresh` | Ignore checkpoint; start over |
| `--diagnose` | Print chunk format info and exit (useful when "Loaded 0 chunks") |

---

## Memory usage

For large maps (16k+ blocks) the tool is designed to stay within available RAM:

- **Chunk loading:** Region files are parsed and cached independently; the chunk dict is assembled after all files are read.
- **Image generation:** The heightmap is downsampled to the output pixel resolution *before* any processing. A 16k-block map at 4096 px uses ~15× less memory than processing at native resolution.
- **STL generation:** Triangles are streamed directly to disk one at a time via `StreamingSTL`. Peak memory is O(one heightmap row), not O(full mesh).
- **Ocean masking:** Uses vectorized LUT indexing (`np.bincount` + LUT `lut[labeled]`) instead of per-component Python loops, which would be catastrophically slow on large maps.

---

## Project structure

```
mc_to_stl.py           Main script (prompts + pipeline orchestration)
mc_to_stl/
  anvil.py             Minecraft Java Edition Anvil (.mca) parser
  bedrock.py           Minecraft Bedrock Edition LevelDB parser
  loader.py            Format detection + heightmap assembly
  ocean.py             Ocean mask detection + micro-island removal
  image.py             Heightmap → color PNG
  mesh.py              Heightmap → watertight STL (streaming)
  stl_writer.py        Binary STL writer (context manager)
  checkpoint.py        Resume / checkpoint system
tests/
  make_test_save.py    Generates a synthetic Java Edition save for testing
  test_pipeline.py     Integration tests (load → image → STL → mosaic)
requirements.txt
```

---

## Tested worlds

- **WesterosEssos** (Java Edition, ~18k × 18k blocks) — the primary target map
- Synthetic 128 × 128 test saves (automated tests)
- Bedrock Edition support added; tested against Bedrock key format specification

---

## Known limitations

- **Bedrock ground-only filtering** is not supported. The `Data2D` record stores only the surface heightmap (equivalent to `WORLD_SURFACE`); ground-only would require scanning all sub-chunk block palettes.
- **Nether and The End** are not parsed — only the overworld is extracted.
- Very tall structures (build-limit towers, etc.) will appear as peaks in the heightmap when ground-only filtering is disabled.
