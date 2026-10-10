"""Recovered Day12 procedural video metrics, not a general identity metric.

The original source was removed by workspace maintenance before publication.
This reconstruction is independently checked against the numeric metric trees
in the surviving, hash-bound video VAE manifest. Source identity is not claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCENES = (14000, 14001)
NOISE_SEEDS = (1111, 2222)
ARMS = ("source_static", "jepa", "no_f", "wrong_direction", "oracle")
VERSION = "day12_procedural_rgb_metrics_v1"
DEFINITIONS = {
    "source_hole": "Source selected mask minus this frame's target mask; empty regions report null",
    "destination": "Entire desired selected-object support, including any overlap with its source",
    "background": "Outside source selection, current target and distractor",
    "static_background": "Outside the union of the full commanded path, source selection and distractor; the same pixels in every frame",
    "motion_compensated_texture_mse": "Generated pixels at source-mask coordinates plus the known integer command, compared to source selected pixels",
    "temporal_variance": "Population variance across all nine frames at each constant-region pixel, then mean over region and RGB",
    "successive_frame_error_mse": "Mean square difference between generated and oracle successive-frame RGB differences on the constant region",
    "path_lag_counts": "Frames 1..8: desired minus observed x projected onto the incoming command direction; >8 px means lagging, <-8 leading, otherwise within half a step. Not an estimated causal delay.",
    "direction_counts": "Steps 0->1 through 7->8; observed x movement projected onto commanded direction; >2 px correct, <-2 reversed, otherwise stationary",
    "mean_missing_rule": "Primary centroid and amplitude errors require all nine detections, otherwise null. The separately named mean_detected_centroid_error_px describes only detected frames. Missing detections never count as zero error.",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def centroid(mask):
    yy, xx = np.nonzero(mask)
    return [float(xx.mean()), float(yy.mean())] if len(xx) else None


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _region_mse(error, mask):
    return float(np.square(error)[mask].mean()) if mask.any() else None


def _packed(rgb):
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def _detect(rgb, source, selected, distractor):
    source_colors = _packed(source)
    selected_colors = np.unique(source_colors[selected])
    other_colors = np.unique(source_colors[~selected])
    unambiguous = np.setdiff1d(selected_colors, other_colors)
    colors = _packed(rgb)
    known = np.isin(colors, np.unique(source_colors))
    exact_selected = np.isin(colors, unambiguous)
    source_f = source.astype(np.float64) / 255
    means = [source_f[mask].mean(0) for mask in
             (selected, distractor, ~(selected | distractor))]
    predicted = rgb.astype(np.float64) / 255
    distances = [np.linalg.norm(predicted - mean, axis=-1) for mean in means]
    novel_selected = ((distances[0] < distances[1]) & (distances[0] < distances[2])
                      & (distances[0] <= .35) & (np.ptp(predicted, axis=-1) >= .15))
    return exact_selected | (~known & novel_selected), {
        "method": "Source-exact unambiguous color labels; novel colors use nearest source object mean versus distractor/background means, distance<=0.35 and RGB range>=0.15",
        "general_detector": False, "target_location_crop": False,
        "source_selected_mean_rgb_01": means[0].tolist(),
        "ambiguous_source_selected_pixels": int(np.isin(source_colors[selected], other_colors).sum()),
    }


def score_clip(rgb, bundle):
    """Score a saved nine-frame uint8 clip against offline fixture arrays."""
    source = bundle["source_rgb"]
    oracle = bundle["guide_oracle"]
    if rgb.dtype != np.uint8 or rgb.shape != oracle.shape or rgb.shape != (9, 384, 384, 3):
        raise ValueError("Expected saved uint8 RGB [9,384,384,3]")
    selected = bundle["source_mask"].astype(bool)
    distractor = bundle["distractor_mask"].astype(bool)
    target_masks = bundle["target_masks"].astype(bool)
    shifts = bundle["shifts_px"]
    static = ~(selected | distractor | target_masks.any(0))
    all_pixels = np.ones_like(selected)
    predicted, desired, source_f = (a.astype(np.float64) / 255 for a in (rgb, oracle, source))
    detection, detector = _detect(rgb, source, selected, distractor)
    yy, xx = np.nonzero(selected)
    frames = []
    for index in range(9):
        destination = target_masks[index]
        masks = {"source_hole": selected & ~destination, "destination": destination,
                 "distractor": distractor, "background": ~(selected | destination | distractor),
                 "static_background": static, "global": all_pixels}
        center = centroid(detection[index])
        target_center = centroid(destination)
        union = int((detection[index] | destination).sum())
        frames.append({
            "frame": index, "command_shift_px": int(shifts[index]),
            "desired_centroid_xy": target_center, "detected_centroid_xy": center,
            "detected_selected_pixels": int(detection[index].sum()),
            "centroid_error_px": float(np.linalg.norm(np.asarray(center)-target_center)) if center is not None else None,
            "selected_iou": float((detection[index] & destination).sum() / union) if union else None,
            "region_pixels": {name: int(mask.sum()) for name, mask in masks.items()},
            "rgb_mse": {name: _region_mse(predicted[index]-desired[index], mask) for name, mask in masks.items()},
            "motion_compensated_texture_mse": float(np.square(predicted[index, yy, xx + int(shifts[index])] - source_f[yy, xx]).mean()),
        })
    temporal = {}
    for name, mask in (("static_background", static), ("distractor", distractor)):
        temporal[name] = {
            "pixels": int(mask.sum()),
            "temporal_variance": float(predicted[:, mask].var(axis=0).mean()),
            "successive_frame_error_mse": float(np.square(np.diff(predicted, axis=0)-np.diff(desired, axis=0))[:, mask].mean()),
            "rgb_mse_to_constant_source": float(np.square(predicted-source_f)[:, mask].mean()),
        }
    closure = {name: _region_mse(predicted[-1]-predicted[0], mask) for name, mask in
               (("global", all_pixels), ("selected_source", selected), ("distractor", distractor), ("static_background", static))}
    centers = [frame["detected_centroid_xy"] for frame in frames]
    present = all(center is not None for center in centers)
    amplitude = float(np.ptp([center[0] for center in centers])) if present else None
    requested_amplitude = float(np.ptp(shifts))
    direction = dict.fromkeys(("correct_direction_steps", "reverse_direction_steps", "stationary_steps", "unscorable_steps"), 0)
    lag = dict.fromkeys(("lagging_frames", "leading_frames", "within_half_step_frames", "unscorable_frames"), 0)
    for index in range(1, 9):
        sign = int(np.sign(shifts[index]-shifts[index-1]))
        if centers[index] is None or centers[index-1] is None:
            direction["unscorable_steps"] += 1
        else:
            progress = (centers[index][0]-centers[index-1][0])*sign
            direction["correct_direction_steps" if progress > 2 else "reverse_direction_steps" if progress < -2 else "stationary_steps"] += 1
        if centers[index] is None:
            lag["unscorable_frames"] += 1
        else:
            lag_px = (frames[index]["desired_centroid_xy"][0]-centers[index][0])*sign
            lag["lagging_frames" if lag_px > 8 else "leading_frames" if lag_px < -8 else "within_half_step_frames"] += 1
    errors = [frame["centroid_error_px"] for frame in frames]
    summary = {
        "detected_frames": sum(center is not None for center in centers),
        "mean_centroid_error_px": _mean(errors) if present else None,
        "mean_detected_centroid_error_px": _mean(errors),
        "max_centroid_error_px": max(errors) if present else None,
        "mean_selected_iou": _mean([frame["selected_iou"] for frame in frames]),
        "mean_motion_compensated_texture_mse": _mean([frame["motion_compensated_texture_mse"] for frame in frames]),
        "observed_x_amplitude_px": amplitude,
        "x_amplitude_error_px": abs(amplitude-requested_amplitude) if present else None,
        "loop_closure_global_mse": closure["global"],
    }
    for name in frames[0]["rgb_mse"]:
        summary["mean_" + name + "_rgb_mse"] = _mean([frame["rgb_mse"][name] for frame in frames])
    for name in temporal:
        for metric in ("temporal_variance", "successive_frame_error_mse"):
            summary[name + "_" + metric] = temporal[name][metric]
    summary.update(direction)
    summary.update(lag)
    return {
        "method_version": VERSION,
        "scope": "Procedural colored-sprite motion and RGB preservation only; not face identity, physics or general perceptual quality",
        "rgb_error_scale": "MSE of saved uint8 RGB divided by 255, averaged over channels and the declared region",
        "ground_truth_use": "Oracle RGB, masks and path are offline scoring inputs only. Motion compensation uses the commanded translation, not a fitted output motion.",
        "detector": detector, "metric_definitions": DEFINITIONS, "frames": frames,
        "temporal": temporal, "loop_closure_mse": closure,
        "path": {"source_centroid_xy": centroid(selected), "requested_x_amplitude_px": requested_amplitude,
                 "observed_x_amplitude_px": amplitude, "all_frames_detected": present,
                 "missing_detection_frames": [i for i, center in enumerate(centers) if center is None],
                 "direction_counts": direction, "lag_counts": lag},
        "summary": summary,
    }


def load_guides(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete") or [r["seed"] for r in manifest["scenes"]] != list(SCENES):
        raise ValueError("Both complete development guide scenes are required")
    bundles = {}
    for row in manifest["scenes"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != row["sha256"]:
            raise ValueError("Guide path/hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            bundle = {key: archive[key] for key in archive.files}
        for key, digest in row.get("array_sha256", {}).items():
            if array_sha256(bundle[key]) != digest:
                raise ValueError("Guide array hash mismatch: " + key)
        bundles[row["seed"]] = bundle
    return manifest, bundles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-root", required=True, type=Path)
    parser.add_argument("--render-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _, bundles = load_guides(args.guide_root)
    render_path = args.render_root / "metrics.json"
    rendering = json.loads(render_path.read_text())
    expected = {f"dev_{scene}/noise_{noise}/{arm}": (scene, noise, arm)
                for scene in SCENES for noise in NOISE_SEEDS for arm in ARMS}
    renders = rendering.get("renders", {})
    problems, scored = [], {}
    bindings = rendering.get("bindings", {})
    if any(bindings.get(k) != list(v) for k, v in (("scenes", SCENES), ("noise_seeds", NOISE_SEEDS), ("arms", ARMS))):
        problems.append("Renderer scene/noise/arm bindings differ from fixed protocol")
    if not rendering.get("complete"):
        problems.append("Renderer does not report a complete run")
    if set(renders) != set(expected):
        problems.append("Expected exactly 20 output rows")
    for key, (scene, noise, arm) in expected.items():
        row = renders.get(key)
        try:
            if row is None or (row["scene_seed"], row["noise_seed"], row["arm"]) != (scene, noise, arm):
                raise ValueError("Missing or misidentified output row")
            reference = row["files"]["rgb"]
            path = (args.render_root / reference["path"]).resolve()
            if not path.is_relative_to(args.render_root.resolve()) or sha256(path) != reference["sha256"]:
                raise ValueError("Output RGB path/hash mismatch")
            rgb = np.load(path, allow_pickle=False)
            scored[key] = score_clip(rgb, bundles[scene])
        except (ValueError, KeyError, FileNotFoundError, TypeError) as error:
            problems.append(key + ": " + str(error))
    complete = not problems and len(scored) == 20
    aggregate = None
    if complete:
        aggregate = {arm: {metric: _mean([scored[f"dev_{scene}/noise_{noise}/{arm}"]["summary"][metric]
                                         for scene in SCENES for noise in NOISE_SEEDS])
                           if all(scored[f"dev_{scene}/noise_{noise}/{arm}"]["summary"][metric] is not None
                                  for scene in SCENES for noise in NOISE_SEEDS) else None
                           for metric in next(iter(scored.values()))["summary"]} for arm in ARMS}
    pairs = []
    for scene in SCENES:
        for noise in NOISE_SEEDS:
            primary = scored.get(f"dev_{scene}/noise_{noise}/jepa")
            for arm in ("source_static", "no_f", "wrong_direction", "oracle"):
                comparison = scored.get(f"dev_{scene}/noise_{noise}/{arm}")
                difference = None if primary is None or comparison is None else {
                    metric: primary["summary"][metric]-comparison["summary"][metric]
                    if primary["summary"][metric] is not None and comparison["summary"][metric] is not None else None
                    for metric in primary["summary"]}
                pairs.append({"scene_seed": scene, "noise_seed": noise, "comparison": arm, "jepa_minus_comparison": difference})
    result = {"version": "day12_recovered_video_assessment_v1", "complete": complete,
              "scope": "Development pilot; all 20 outputs required for any aggregate; no output selection",
              "guide_manifest_sha256": sha256(args.guide_root / "manifest.json"),
              "render_manifest_sha256": sha256(render_path), "assess_source_sha256": sha256(__file__),
              "problems": problems, "scores": scored, "aggregate": aggregate, "paired": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": complete, "scored": len(scored), "problems": problems}))


if __name__ == "__main__":
    main()
