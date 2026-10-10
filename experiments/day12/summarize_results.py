"""Validate and summarize all 20 Day12 generated clips without rescoring them.

Usage: python experiments/day12/summarize_results.py --run-root PATH
       [--guide-root PATH] [--out PATH]

GIFs retain all nine frames and native 384x384 panels, with labels outside the
images. GIF palette conversion is for display only; source RGB arrays are never
modified. This is a development comparison, not a pass/fail research gate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from archive_evidence import child, sha256
from common import array_sha256, file_record, load_guides
from config import ARMS, NOISE_SEEDS, SCENES, SHIFTS


CSV_METRICS = (
    "detected_frames", "mean_centroid_error_px", "mean_selected_iou",
    "x_amplitude_error_px", "mean_motion_compensated_texture_mse",
    "mean_source_hole_rgb_mse", "mean_distractor_rgb_mse",
    "mean_static_background_rgb_mse", "static_background_temporal_variance",
    "loop_closure_global_mse",
)
SCOPE = (
    "Two previously seen procedural development scenes and two noise seeds. "
    "Metrics describe colored-sprite motion and RGB preservation, not face or "
    "part identity, unseen-scene generalization, or statistical significance. "
    "The no_f arm retains provenance and source appearance transport."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    result = json.loads(Path(path).read_text())
    json.dumps(result, allow_nan=False)  # Reject nonfinite input before output.
    return result


def validate_inputs(run_root, guide_root=None):
    """Read and validate everything before the caller creates output files."""
    run_root = Path(run_root).resolve()
    render_root = run_root / "render"
    assessment_path, render_path = run_root / "assessment.json", render_root / "metrics.json"
    assessment, rendering = read_json(assessment_path), read_json(render_path)
    expected = {f"dev_{s}/noise_{n}/{a}": (s, n, a)
                for s in SCENES for n in NOISE_SEEDS for a in ARMS}
    require(assessment.get("complete") is True and assessment.get("problems") == [],
            "Assessment must be complete with an empty problem list")
    require(rendering.get("complete") is True, "Renderer must be complete")
    require(set(assessment.get("scores", {})) == set(expected), "Expected exactly 20 assessment scores")
    require(set(rendering.get("renders", {})) == set(expected), "Expected exactly 20 render rows")
    require(set(assessment.get("aggregate") or {}) == set(ARMS), "Missing fixed-arm aggregates")
    expected_pairs = {(s, n, a) for s in SCENES for n in NOISE_SEEDS
                      for a in ARMS if a != "jepa"}
    pairs = assessment.get("paired", [])
    require(len(pairs) == 16 and {(p["scene_seed"], p["noise_seed"], p["comparison"])
                                 for p in pairs} == expected_pairs, "Expected exactly 16 paired comparisons")
    bindings = rendering["bindings"]
    for key, values in (("scenes", SCENES), ("noise_seeds", NOISE_SEEDS), ("arms", ARMS)):
        require(bindings.get(key) == list(values), "Unexpected render binding: " + key)
    require(assessment["render_manifest_sha256"] == sha256(render_path), "Render manifest hash mismatch")
    require(assessment["guide_manifest_sha256"] == bindings["guide_manifest_sha256"],
            "Assessment and renderer used different guide manifests")
    bundles = None
    if guide_root is not None:
        _, bundles = load_guides(guide_root)
        require(sha256(Path(guide_root) / "manifest.json") == bindings["guide_manifest_sha256"],
                "Optional guide manifest differs from rendered guides")

    clips, records, paths = {}, [], set()
    for key, identity in expected.items():
        scene, noise, arm = identity
        row, score = rendering["renders"][key], assessment["scores"][key]
        require((row["scene_seed"], row["noise_seed"], row["arm"]) == identity,
                "Render identity mismatch: " + key)
        require(score.get("method_version") == "day12_procedural_rgb_metrics_v1",
                "Unexpected metric version: " + key)
        require(len(score.get("frames", [])) == 9 and
                [f["frame"] for f in score["frames"]] == list(range(9)),
                "Expected nine ordered scored frames: " + key)
        require(all(name in score["summary"] for name in CSV_METRICS), "Missing summary metric: " + key)
        require(set(row["files"]) == {"rgb", "latent", "gif"}, "Incomplete payload catalogue: " + key)
        for kind, record in row["files"].items():
            path = child(render_root, record["path"])
            require(path not in paths, "Duplicate payload path: " + record["path"])
            require(path.is_file() and path.stat().st_size == record["bytes"] and
                    sha256(path) == record["sha256"], "Payload hash/size mismatch: " + record["path"])
            paths.add(path)
            records.append(dict(key=key, kind=kind, **record))
        rgb = np.load(child(render_root, row["files"]["rgb"]["path"]), allow_pickle=False)
        require(rgb.dtype == np.uint8 and rgb.shape == (9, 384, 384, 3), "Malformed RGB clip: " + key)
        clips[key] = rgb
        if bundles is not None:
            require(row["guide_array_sha256"] == array_sha256(bundles[scene]["guide_" + arm]),
                    "Rendered guide array hash mismatch: " + key)
    require(len(records) == 60, "Expected exactly 60 verified render payloads")
    return assessment, rendering, clips, bundles, records


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def comparison_frames(scene, noise, clips, bundle=None):
    """Use all native frames, with no cropping, resizing, or interpolation."""
    tile, gap, label, top = 384, 12, 34, 50
    canvas_size = (3 * tile + 4 * gap, top + 2 * (tile + label + gap) + gap)
    regular, title = font(18), font(22)
    frames = []
    for index in range(9):
        canvas = Image.new("RGB", canvas_size, "#f3f4f6")
        draw = ImageDraw.Draw(canvas)
        draw.text((gap, 12), f"Day12 development | scene {scene} | noise {noise} | "
                  f"frame {index}/8 | command {SHIFTS[scene][index]:+d} px", font=title, fill="#15171a")
        for panel, arm in enumerate(ARMS):
            x = gap + (panel % 3) * (tile + gap)
            y = top + (panel // 3) * (tile + label + gap)
            draw.text((x, y + 5), arm + " | generated", font=regular, fill="#15171a")
            canvas.paste(Image.fromarray(clips[f"dev_{scene}/noise_{noise}/{arm}"][index]), (x, y + label))
        x, y = gap + 2 * (tile + gap), top + tile + label + gap
        if bundle is not None:
            draw.text((x, y + 5), "Target | procedural RGB", font=regular, fill="#15171a")
            canvas.paste(Image.fromarray(bundle["guide_oracle"][index]), (x, y + label))
        else:
            draw.multiline_text((x + 10, y + label + 22),
                "All five generated arms\nAll nine source frames\nNative 384 x 384 panels\n\n"
                "Oracle panel is generated\nfrom the oracle RGB guide.\n\n"
                "GIF colors are quantized.\nOriginal RGB arrays are kept.",
                font=regular, fill="#41464d", spacing=9)
        frames.append(canvas)
    return frames


def summarize(run_root, out, guide_root=None):
    run_root, out = Path(run_root).resolve(), Path(out).resolve()
    require(not out.exists(), "Output already exists; choose a new path to preserve it")
    assessment, rendering, clips, bundles, input_records = validate_inputs(run_root, guide_root)
    rows = [dict(scene_seed=s, noise_seed=n, arm=a,
                 **assessment["scores"][f"dev_{s}/noise_{n}/{a}"]["summary"])
            for s in SCENES for n in NOISE_SEEDS for a in ARMS]
    report = dict(
        version="day12_generated_video_summary_v1", complete=True, scope=SCOPE,
        metrics="Copied from the completed assessment; no new metrics or success threshold",
        nulls="JSON null / blank CSV means unavailable, never zero error",
        assessment_sha256=sha256(run_root / "assessment.json"),
        render_manifest_sha256=sha256(run_root / "render/metrics.json"),
        summarize_source_sha256=sha256(__file__),
        guide_manifest_sha256=assessment["guide_manifest_sha256"],
        guide_content_validated=bundles is not None,
        verified_payloads=input_records, bindings=rendering["bindings"],
        scores=rows, aggregate=assessment["aggregate"], paired=assessment["paired"],
        display=dict(panel_order=list(ARMS), native_panel_size=[384, 384], frame_indices=list(range(9)),
                     interpolation=False, target_panel=bundles is not None,
                     note="GIF palette conversion changes display colors; original RGB arrays are untouched",
                     gif_duration_ms=120, frames=9),
        artifacts=[])
    json.dumps(report, allow_nan=False)  # Validate serialization before any write.
    out.mkdir(parents=True)
    csv_path = out / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_seed", "noise_seed", "arm", *CSV_METRICS],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report["artifacts"].append(file_record(csv_path, out))
    for scene in SCENES:
        for noise in NOISE_SEEDS:
            frames = comparison_frames(scene, noise, clips, bundles[scene] if bundles is not None else None)
            path = out / f"scene_{scene}_noise_{noise}.gif"
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=120,
                           loop=0, disposal=2, optimize=False)
            with Image.open(path) as gif:
                require(gif.n_frames == 9 and gif.size == frames[0].size, "Incomplete comparison GIF")
            report["artifacts"].append(dict(scene_seed=scene, noise_seed=noise, **file_record(path, out)))
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--guide-root", type=Path)
    parser.add_argument("--out", type=Path, help="New output directory; defaults to RUN_ROOT/summary")
    args = parser.parse_args()
    out = args.out or args.run_root / "summary"
    report = summarize(args.run_root, out, args.guide_root)
    print(json.dumps(dict(complete=report["complete"], scores=len(report["scores"]),
                         payloads_verified=len(report["verified_payloads"]),
                         comparison_gifs=4, out=str(out.resolve()))))


if __name__ == "__main__":
    main()
