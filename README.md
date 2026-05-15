# MCtoSTL

Convert a Minecraft world save into a color-coded heightmap image and 3-D printable STL terrain models.

Supports **Java Edition** (`.mca` Anvil files) and **Bedrock Edition** (LevelDB `db/` folder).

---

## Outputs

| File | Description |
|------|-------------|
| `heightmap.png` | Color-coded relief image: red = highest, green = sea level, blue = below sea, steel-blue = open ocean |
| `heightmap_gray.png` | Monochromatic relief image: white = highest, black = sea level, dark-gray = below sea |
| `terrain.stl` | Single watertight solid of the entire terrain |
| `tiles/tile_ZZZ_XXX.stl` | Mosaic pack of printable tiles that fit your print bed |

Examples:
heightmap.png
<img width="3328" height="3379" alt="heightmap" src="https://github.com/user-attachments/assets/a045f16b-902d-4122-891f-0fa3027be473" />

heightmap_gray.png
<img width="16640" height="16896" alt="heightmap_gray" src="https://github.com/user-attachments/assets/350ca55e-ec1b-4559-a843-ffb0426656d9" />

terrain.stl
<img width="1306" height="819" alt="terrain" src="https://github.com/user-attachments/assets/27dfc18a-2778-4052-93b4-a6f3c2d473c5" />

3D-Printed STL
<img width="1600" height="1108" alt="image" src="https://github.com/user-attachments/assets/d126e61c-3437-4b70-878a-3a04e7b417b3" />
<img width="1600" height="1200" alt="image" src="https://github.com/user-attachments/assets/82928f2b-3d96-43a3-9364-6cfdd95d2031" />
<img width="1600" height="1200" alt="image" src="https://github.com/user-attachments/assets/2c87a1c3-477d-44e2-acbb-510faa611fb7" />

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

### Water block detection *(Java only, optional)*
When enabled, identifies actual water blocks during region parsing. Ocean detection then requires **both**:
- Altitude ≤ sea level **AND**
- Block type is water

This prevents land depressions (canyons, trenches) below sea level from being incorrectly marked as ocean. Disabled by default because it adds parsing overhead; only enable if your map has terrain below sea level that should not be marked as ocean.

### Gaussian blur (anti-aliasing)
Applied to all outputs before rendering or meshing. Recommended: 1.0–2.0. Higher = smoother, lower = more block detail.

### Ocean / sea masking
Detects the flat water surface in the heightmap histogram and masks out open-sea areas. Ocean cells are **flattened to the base plate** in STL output — coastlines end at the shore with no raised sea surface.

- **Sea level** — auto-detected by default (histogram peak in the lower 60th percentile). Override if auto-detection picks the wrong value.
- **Min ocean area** — water bodies smaller than this (blocks²) are kept as lakes/rivers rather than marked as ocean.
- **Min island area** — land components smaller than this are absorbed into the ocean mask as noise.

### Floating block removal *(Java only)*
Full 3-D connectivity analysis per chunk. Blocks that are not connected (6-directionally) to the bottom section of their chunk are discarded — removes floating platforms, isolated mid-air artefacts, and single hanging blocks. Overhangs are safe: they connect to the cliff face through horizontal neighbours. About 2–3× slower per chunk than the default heightmap method.

**Note:** On custom maps where terrain is not always connected to bedrock (e.g. builder-placed terrain), disable this feature.

### Force scan *(Java only)*
By default, the parser uses stored heightmap data from the NBT (fast but potentially stale). `force_scan` bypasses stored heightmaps and always scans block sections directly. Slower, but fixes holes caused by outdated chunk data. Automatically disables chunk caching when enabled.

### Heightmap image
- **Max width / height (px):** Output image dimensions. The larger side is capped at this value; aspect ratio is preserved. The heightmap is downsampled to the output resolution before processing — memory usage scales with output size, not source map size. Set to 0 for full native resolution (one pixel per block).
- **Rectangular crop:** When using a crop polygon, choose whether the output PNG is:
  - **YES** — rectangular (outside polygon filled with ocean color)
  - **NO** — transparent RGBA (outside polygon has alpha=0, invisible)

### STL physical dimensions
Maximum bounding box for the printed model in mm. Aspect ratio is always preserved — the smaller dimension will be less than its maximum.

- **Max X / Y (mm):** Horizontal bounds in mm. The map is scaled to fit within these dimensions.
- **Z exaggeration (factor):** Multiplier for vertical relief independent of XY scale. Examples:
  - `1.0` = physically accurate (same scale as XY) — usually too flat for terrain
  - `5.0–15.0` = typical for printable terrain models
  - `0.1` = extreme flattening (great for contour maps)
- **Base plate thickness (mm):** Solid base below the lowest terrain point. Prevents sharp edges on the bottom of the print.
- **Border width (layers):** Optional wall around the entire model. Helps prevent resin leakage when using a mold. Model size including border stays within Max X/Y.

### Crop polygon and sea masking
Both are defined in the unified `config.json` file under `crop_area` and `sea_masking` sections (list of polygon vertices).

- **Crop area:** Quadrilateral region (4 `[x, z]` block coordinates). Only terrain inside this polygon appears in outputs; outside is set to the minimum height.
- **Sea masking polygons:** One or more polygons that force specific regions to sea level. Useful for defining exact coastlines or masking off unwanted water bodies.

Format: `[[[x1, z1], [x2, z2], ...], ...]` — list of polygons, each polygon is a list of `[x, z]` block coordinates (Minecraft world coordinates).

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

- **WesterosEssos** (Java Edition, ~16k × 16k blocks) — the primary target map
- Synthetic 128 × 128 test saves (automated tests)
- Bedrock Edition support added; tested against Bedrock key format specification

---

## Known limitations

- **Bedrock ground-only filtering** is not supported. The `Data2D` record stores only the surface heightmap (equivalent to `WORLD_SURFACE`); ground-only would require scanning all sub-chunk block palettes.
- **Nether and The End** are not parsed — only the overworld is extracted.
- Very tall structures (build-limit towers, etc.) will appear as peaks in the heightmap when ground-only filtering is disabled.

---

## Example config.json

```json
{
  "configuration": {
    "_comment_paths": "Minecraft save location and output directory",
    "save_path": "C:\\Minecraft\\WesterosEssos16k",
    "out_dir": "C:\\Minecraft\\WesterosEssos16k\\output",

    "_comment_parsing": "Java Edition parsing options",
    "n_workers": 16,
    "ground_only": true,
    "detect_water_blocks": false,
    "detect_floating": false,
    "force_scan": true,

    "_comment_ocean": "Ocean detection and water body filtering",
    "mask_ocean": true,
    "sea_level": null,
    "min_ocean_blocks": 500000,
    "min_land_area": 2000,

    "_comment_image": "Heightmap image generation settings",
    "generate_heightmap": true,
    "smooth_sigma": 1.5,
    "max_px_w": 0,
    "max_px_h": 0,
    "rectangular_crop": true,

    "_comment_stl_single": "Single STL physical dimensions and options",
    "generate_stl": true,
    "max_x_mm": 260.0,
    "max_y_mm": 260.0,
    "z_exaggeration": 0.1,
    "base_mm": 3.0,
    "sea_level_offset": 5,
    "border_width": 4,
    "max_verts": 1500,

    "_comment_mosaic": "Mosaic tile STL generation",
    "generate_mosaic": true,
    "tile_x_mm": 260.0,
    "tile_y_mm": 260.0,
    "skip_ocean_stl": false
  },

  "_comment_crop": "Crop area: 4-point polygon in Minecraft block coordinates",
  "crop_area": [
    [-8320, -8448],
    [8319, -8448],
    [8319, 8447],
    [-8320, 8447]
  ],

  "_comment_masking": "Sea masking polygons: force regions to sea level",
  "sea_masking": [
    [
      [-4566.5, -4400.5],
      [-3920.5, -5594.5],
      [-3225.5, -6647.5],
      [-2389.5, -6743.5],
      [-1328.5, -7361.5],
      [97.5, -8447.5],
      [8318.5, -8445.5],
      [8319.5, -3366.5],
      [-3006.5, -3131.5],
      [-3182.5, -4355.5]
    ]
  ]
}
```

### Key parameter explanations

**Parsing & Performance:**
- `n_workers` — Parallel processes. Higher = faster but uses more CPU/memory.
- `ground_only` — Ignore trees and structures; show terrain only.
- `detect_water_blocks` — Prevent land depressions below sea level from being marked ocean (opt-in, slower).
- `detect_floating` — Remove floating block artefacts via 3D connectivity (slower; disable for builder-placed terrain).
- `force_scan` — Bypass potentially stale NBT heightmaps; always scan sections (slower but fixes holes).

**Ocean Detection:**
- `mask_ocean` — Flatten open sea for better coastline detail.
- `sea_level` — Y coordinate; auto-detected if `null`.
- `min_ocean_blocks` — Water bodies < this size become lakes/rivers.
- `min_land_area` — Land areas < this size become noise/ocean.

**Image Output:**
- `smooth_sigma` — Gaussian blur (0 = sharp, 1.5-2.0 = recommended, higher = blurrier).
- `max_px_w/h` — Output image size; 0 = native (1 pixel per block).
- `rectangular_crop` — TRUE = rectangular PNG; FALSE = transparent outside crop polygon.

**STL Physical Scale:**
- `max_x_mm` / `max_y_mm` — Horizontal bounds (aspect ratio preserved).
- `z_exaggeration` — Vertical relief multiplier (1.0 = accurate, 5-15 = typical prints).
- `sea_level_offset` — Expose seafloor by pretending sea is lower (0 = actual sea level).
- `base_mm` — Solid base thickness.
- `border_width` — Wall perimeter for molds (0 = none).
- `max_verts` — Mesh detail (500-2000 typical).

**Mosaic Tiles:**
- `tile_x_mm` / `tile_y_mm` — Tile dimensions (typically print bed size).
- `skip_ocean_stl` — Don't generate 100%-ocean tiles (saves filament).
