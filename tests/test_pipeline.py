"""
Integration tests for the MCtoSTL pipeline.

Run with:  python -m pytest tests/  or  python tests/test_pipeline.py
"""

import os
import struct
import sys
import tempfile
import unittest

import numpy as np

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.make_test_save import make_test_save
from mc_to_stl.loader import load_save
from mc_to_stl.image import generate_image
from mc_to_stl.mesh import generate_single_stl, generate_mosaic_stl
from mc_to_stl.anvil import _unpack_longs


class TestUnpackLongs(unittest.TestCase):

    def test_aligned_packing(self):
        """9-bit aligned packing: 7 values per long."""
        # Pack values 0-255 manually, then unpack
        values = list(range(256))
        bits = 9
        vpL = 64 // bits
        mask = (1 << bits) - 1
        longs = []
        for i in range(0, len(values), vpL):
            chunk = values[i : i + vpL]
            v = 0
            for j, val in enumerate(chunk):
                v |= (val & mask) << (j * bits)
            if v >= (1 << 63):
                v -= 1 << 64
            longs.append(v)
        result = _unpack_longs(longs, bits=9, count=256)
        self.assertEqual(result, values)

    def test_compact_packing(self):
        """Compact packing: values may span long boundaries."""
        bits = 9
        values = [i % (1 << bits) for i in range(256)]
        # Pack compactly (stream of bits)
        bit_stream = 0
        total_bits = 0
        for v in values:
            bit_stream |= (v << total_bits)
            total_bits += bits
        longs = []
        for i in range(0, total_bits, 64):
            chunk = (bit_stream >> i) & ((1 << 64) - 1)
            if chunk >= (1 << 63):
                chunk -= 1 << 64
            longs.append(chunk)
        result = _unpack_longs(longs, bits=9, count=256)
        self.assertEqual(result, values)


class TestFullPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.save_dir = os.path.join(cls.tmpdir, "save")
        cls.out_dir = os.path.join(cls.tmpdir, "output")
        os.makedirs(cls.out_dir, exist_ok=True)
        make_test_save(cls.save_dir)

    def test_load_save(self):
        hm, meta = load_save(self.save_dir)
        self.assertEqual(hm.shape, (128, 128))
        self.assertTrue(hm.max() > hm.min(), "heightmap has no variation")
        self.assertEqual(meta["width_blocks"], 128)
        self.assertEqual(meta["height_blocks"], 128)

    def test_image_generation(self):
        hm, _ = load_save(self.save_dir)
        img_path = os.path.join(self.out_dir, "heightmap.png")
        img = generate_image(hm, max_px_w=256, max_px_h=256,
                             smooth_sigma=1.0, output_path=img_path)
        self.assertTrue(os.path.exists(img_path))
        self.assertGreater(os.path.getsize(img_path), 1000)
        arr = np.array(img)
        # At least one non-zero channel
        self.assertTrue(arr.max() > 0)
        # Image fits within requested bounds
        self.assertLessEqual(img.width, 256)
        self.assertLessEqual(img.height, 256)

    def _read_stl(self, path):
        """Return list of (v0,v1,v2) triangles from binary STL."""
        triangles = []
        with open(path, "rb") as f:
            f.read(80)
            n = struct.unpack("<I", f.read(4))[0]
            for _ in range(n):
                f.read(12)  # normal
                verts = [
                    np.array(struct.unpack("<fff", f.read(12)))
                    for _ in range(3)
                ]
                f.read(2)
                triangles.append(tuple(verts))
        return triangles

    def test_single_stl(self):
        hm, _ = load_save(self.save_dir)
        stl_path = os.path.join(self.out_dir, "terrain.stl")
        generate_single_stl(hm, max_x_mm=100, max_y_mm=100, max_z_mm=10,
                            base_mm=2, smooth_sigma=1.0, output_path=stl_path)

        tris = self._read_stl(stl_path)
        self.assertGreater(len(tris), 0)

        all_verts = np.array([v for t in tris for v in t])
        # X/Y extents match requested dimensions
        self.assertAlmostEqual(all_verts[:, 0].max(), 100.0, delta=0.5)
        self.assertAlmostEqual(all_verts[:, 1].max(), 100.0, delta=0.5)
        # Z minimum is 0 (base plate)
        self.assertAlmostEqual(all_verts[:, 2].min(), 0.0, delta=0.01)
        # Z maximum includes base + terrain
        self.assertGreater(all_verts[:, 2].max(), 2.0)

        # No degenerate triangles
        for v0, v1, v2 in tris:
            n = np.cross(v1 - v0, v2 - v0)
            self.assertGreater(np.linalg.norm(n), 1e-10)

    def test_mosaic_stl(self):
        hm, _ = load_save(self.save_dir)
        tiles_dir = os.path.join(self.out_dir, "tiles")
        generate_mosaic_stl(hm, max_x_mm=100, max_y_mm=100, max_z_mm=10,
                            tile_x_mm=50, tile_y_mm=50,
                            base_mm=2, smooth_sigma=1.0,
                            output_dir=tiles_dir)

        tile_files = sorted(f for f in os.listdir(tiles_dir)
                            if f.endswith(".stl"))
        self.assertGreater(len(tile_files), 1)

        # All tiles must have Z floor at 0 (consistent base)
        for fname in tile_files:
            tris = self._read_stl(os.path.join(tiles_dir, fname))
            if not tris:
                continue
            all_verts = np.array([v for t in tris for v in t])
            self.assertAlmostEqual(all_verts[:, 2].min(), 0.0, delta=0.01,
                                   msg=f"{fname}: Z floor is not 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
