"""Small checks against surviving measured clips, plus missing-detection behavior.

Run: DAY12_GUIDE_ROOT=/tmp/day12-guides DAY12_VIDEO_VAE_ROOT=/tmp/day12-video-vae
     python3 -m unittest discover -s experiments/day12 -p test_recovered_metrics.py
"""
import json
import os
from pathlib import Path
import unittest

import numpy as np

from assess import load_guides, score_clip
from guide_report import compare_tree


class RecoveredMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guide_root = Path(os.environ.get("DAY12_GUIDE_ROOT", "/tmp/day12-guides"))
        cls.vae_root = Path(os.environ.get("DAY12_VIDEO_VAE_ROOT", "/tmp/day12-video-vae"))
        if not (cls.guide_root / "manifest.json").is_file() or not (cls.vae_root / "manifest.json").is_file():
            raise unittest.SkipTest("Recovered fixture archives are required for this real-data recount")
        _, cls.bundles = load_guides(cls.guide_root)

    def test_all_four_saved_metric_trees(self):
        manifest = json.loads((self.vae_root / "manifest.json").read_text())
        leaves = 0
        for row in manifest["scenes"]:
            bundle = self.bundles[row["seed"]]
            with np.load(self.vae_root / row["path"], allow_pickle=False) as archive:
                roundtrip = archive["roundtrip_rgb"]
            for stage, rgb in (("guide", bundle["guide_jepa"]), ("roundtrip", roundtrip)):
                with self.subTest(scene=row["seed"], stage=stage):
                    comparison = compare_tree(row[stage+"_metrics_vs_oracle"], score_clip(rgb, bundle))
                    self.assertTrue(comparison["numerically_equal_at_1e-15"])
                    self.assertEqual(comparison["structure_mismatches"], [])
                    leaves += comparison["numeric_leaves"]
        self.assertEqual(leaves, 1004)

    def test_oracle_and_direction_controls(self):
        for seed, bundle in self.bundles.items():
            with self.subTest(scene=seed):
                oracle = score_clip(bundle["guide_oracle"], bundle)["summary"]
                static = score_clip(bundle["guide_source_static"], bundle)["summary"]
                wrong = score_clip(bundle["guide_wrong_direction"], bundle)["summary"]
                self.assertEqual(oracle["mean_centroid_error_px"], 0)
                self.assertEqual(oracle["mean_motion_compensated_texture_mse"], 0)
                self.assertEqual(oracle["mean_global_rgb_mse"], 0)
                self.assertEqual(oracle["correct_direction_steps"], 8)
                self.assertEqual(static["observed_x_amplitude_px"], 0)
                self.assertEqual(static["stationary_steps"], 8)
                self.assertEqual(wrong["reverse_direction_steps"], 8)
                self.assertGreater(wrong["mean_centroid_error_px"], static["mean_centroid_error_px"])

    def test_missing_detection_never_becomes_zero_error(self):
        bundle = self.bundles[14000]
        blank = np.zeros_like(bundle["guide_oracle"])
        summary = score_clip(blank, bundle)["summary"]
        self.assertEqual(summary["detected_frames"], 0)
        self.assertIsNone(summary["mean_centroid_error_px"])
        self.assertIsNone(summary["max_centroid_error_px"])
        self.assertIsNone(summary["observed_x_amplitude_px"])
        self.assertIsNone(summary["x_amplitude_error_px"])
        self.assertEqual(summary["unscorable_steps"], 8)


if __name__ == "__main__":
    unittest.main()
