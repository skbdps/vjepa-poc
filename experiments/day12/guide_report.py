"""Recount surviving pregeneration guides; never labels them generated videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from assess import ARMS, array_sha256, load_guides, score_clip, sha256


def compare_tree(expected, actual):
    """Compare every leaf, retaining roundoff rather than claiming identity."""
    result = {"numeric_leaves": 0, "exact_numeric_leaves": 0,
              "maximum_absolute_difference": 0.0, "differences": [], "structure_mismatches": []}
    def visit(left, right, path):
        if isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                result["structure_mismatches"].append(path)
                return
            for key in left:
                visit(left[key], right[key], path + "/" + key)
        elif isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                result["structure_mismatches"].append(path)
                return
            for index, value in enumerate(left):
                visit(value, right[index], path + "/" + str(index))
        elif isinstance(left, (int, float)) and not isinstance(left, bool):
            result["numeric_leaves"] += 1
            if not isinstance(right, (int, float)) or isinstance(right, bool):
                result["structure_mismatches"].append(path)
                return
            error = abs(left-right)
            result["maximum_absolute_difference"] = max(error, result["maximum_absolute_difference"])
            if error == 0:
                result["exact_numeric_leaves"] += 1
            else:
                result["differences"].append({"path": path, "saved": left, "recounted": right, "absolute_difference": error})
        elif left != right:
            result["structure_mismatches"].append(path)
    visit(expected, actual, "")
    result["numerically_equal_at_1e-15"] = not result["structure_mismatches"] and result["maximum_absolute_difference"] <= 1e-15
    return result


def contact_sheet(bundle, seed, destination):
    """Five complete arm rows and a fixed temporal subsample, with clear labels."""
    columns = (0, 2, 4, 6, 8)
    rows = (("oracle", "Oracle target / scoring only"), ("jepa", "JEPA translator guide"),
            ("no_f", "Copy + local fill (no F)"),
            ("wrong_direction", "Wrong direction guide"), ("source_static", "Static source guide"))
    tile, left, top, gap = 240, 245, 85, 8
    image = Image.new("RGB", (left + 5 * (tile+gap), top+5*(tile+gap)), "#f5f5f5")
    draw = ImageDraw.Draw(image)
    try:
        regular = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        regular = title = ImageFont.load_default()
    draw.text((12, 8), f"Development scene {seed}: PREGENERATION guides", font=title, fill="#161616")
    draw.text((12, 36), "No Wan denoiser output. Independent image decodes form these guide frames.", font=regular, fill="#444444")
    for column, frame in enumerate(columns):
        draw.text((left+column*(tile+gap), 61), f"Frame {frame}: shift {int(bundle['shifts_px'][frame]):+d} px", font=regular, fill="#161616")
    for row, (arm, label) in enumerate(rows):
        y = top+row*(tile+gap)
        draw.text((12, y+12), label, font=regular, fill="#161616")
        for column, frame in enumerate(columns):
            thumbnail = Image.fromarray(bundle["guide_"+arm][frame]).resize((tile, tile), Image.Resampling.LANCZOS)
            image.paste(thumbnail, (left+column*(tile+gap), y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def build_report(guide_root, video_vae_root, out):
    guide_root, video_vae_root, out = map(Path, (guide_root, video_vae_root, out))
    manifest, bundles = load_guides(guide_root)
    video_manifest_path = video_vae_root / "manifest.json"
    video_manifest = json.loads(video_manifest_path.read_text())
    if not video_manifest.get("complete") or video_manifest["guide_manifest_sha256"] != sha256(guide_root / "manifest.json"):
        raise ValueError("Video VAE manifest does not bind the surviving complete guides")
    report = {
        "version": "day12_recovered_pregeneration_report_v1", "complete": True,
        "stage": "PREGENERATION ONLY: no Wan denoising or generated-video result",
        "scope": "Two previously seen procedural development scenes; five guide arms per scene; JEPA-only video VAE roundtrips",
        "guide_manifest_sha256": sha256(guide_root / "manifest.json"),
        "video_vae_manifest_sha256": sha256(video_manifest_path),
        "historical_assess_source_sha256": video_manifest["source_hashes"]["experiments/day12/assess.py"],
        "reconstructed_assess_source_sha256": sha256(Path(__file__).with_name("assess.py")),
        "reconstructed_guide_report_source_sha256": sha256(__file__),
        "source_identity_claimed": False,
        "scenes": [], "recounts": [],
    }
    video_rows = {row["seed"]: row for row in video_manifest["scenes"]}
    if set(video_rows) != set(bundles):
        raise ValueError("Video VAE scene set differs from guides")
    for guide_row in manifest["scenes"]:
        seed = guide_row["seed"]
        bundle = bundles[seed]
        scores = {arm: score_clip(bundle["guide_"+arm], bundle) for arm in ARMS}
        vae_row = video_rows[seed]
        vae_path = (video_vae_root / vae_row["path"]).resolve()
        if not vae_path.is_relative_to(video_vae_root.resolve()) or sha256(vae_path) != vae_row["sha256"]:
            raise ValueError("Video VAE archive hash mismatch")
        with np.load(vae_path, allow_pickle=False) as archive:
            for key, digest in vae_row["array_sha256"].items():
                if array_sha256(archive[key]) != digest:
                    raise ValueError("Video VAE array hash mismatch")
            roundtrip = archive["roundtrip_rgb"]
        roundtrip_score = score_clip(roundtrip, bundle)
        for name, actual in (("guide", scores["jepa"]), ("roundtrip", roundtrip_score)):
            comparison = compare_tree(vae_row[name+"_metrics_vs_oracle"], actual)
            report["recounts"].append({"scene_seed": seed, "stage": name, **comparison})
        relative_figure = f"figures/development_video_{seed}_pregeneration.png"
        contact_sheet(bundle, seed, out / relative_figure)
        report["scenes"].append({"scene_seed": seed, "guide_archive_sha256": guide_row["sha256"],
                                  "guide_scores": scores, "jepa_video_vae_roundtrip_score": roundtrip_score,
                                  "figure": {"path": relative_figure, "sha256": sha256(out / relative_figure)}})
    report["recount_validation"] = {
        "numeric_leaves": sum(row["numeric_leaves"] for row in report["recounts"]),
        "exact_numeric_leaves": sum(row["exact_numeric_leaves"] for row in report["recounts"]),
        "maximum_absolute_difference": max(row["maximum_absolute_difference"] for row in report["recounts"]),
        "all_numeric_leaves_equal_at_1e-15": all(row["numerically_equal_at_1e-15"] for row in report["recounts"]),
        "explanation": "The source is reconstructed. Tiny floating-point reduction differences, if any, are recorded without rounding them to zero.",
    }
    path = out / "validation/guide_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-root", type=Path, required=True)
    parser.add_argument("--video-vae-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.guide_root, args.video_vae_root, args.output)
    print(json.dumps(report["recount_validation"], indent=2))


if __name__ == "__main__":
    main()
