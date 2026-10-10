"""Sealed-test assessment of direct JEPA latent interventions.

All source/target regions are fixed before inspecting method predictions. The
frozen semantic readout sees tokens only; masks are used by the scorer, never
passed to the readout. This file performs no optimization or test selection.
Saved NPZs contain pointwise squared errors and every coarse readout output so
the CSV metrics can be recounted without rerunning models. They do not contain
full edited embeddings, which can be reproduced from frozen inputs and heads.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

import data
import operators

HERE = Path(__file__).resolve().parent
EVALUATOR_VERSION = "day8_direct_latent_test_v1"
ARMS = ("learned_residual", "geometry_only")
SEEDS = (1801, 1802, 1803)
BASE_ARMS = ("noop", "naive", "residual", "wrong_direction", "wrong_object")
REGIONS = ("source_hole", "destination", "halo", "distractor", "background", "common")
RATIO_FLOOR = 1e-12
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 18008
IDENTITY_COLOR_SEPARATION = .15
IDENTITY_READOUT_GATE = .90
IDENTITY_ACCURACY_TOLERANCE = .05


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing an empty metric table")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_binding(features, run):
    """Check freeze, sources, complete checkpoint set, and feature provenance."""
    root, run = Path(features), Path(run)
    if not (root / "manifest.json").is_file() and root.name == "cache":
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text())
    frozen_path = run / "checkpoint_freeze.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("test_accessed") is not False:
        raise ValueError("Checkpoint record is not a pretest freeze")
    if manifest.get("test_freeze_sha256") != sha256(frozen_path):
        raise ValueError("Test extraction was not bound to this checkpoint freeze")
    for name, digest in frozen["source_hashes"].items():
        if sha256(HERE / name) != digest:
            raise ValueError("Frozen training source changed: " + name)
    for name, digest in manifest["source_hashes"].items():
        if sha256(HERE / name) != digest:
            raise ValueError("Extraction source changed: " + name)
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS} | {("frozen_readout", 1800)}
    actual = [(record["arm"], record["seed"]) for record in frozen["models"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Freeze does not contain exactly the predeclared checkpoints")
    for record in frozen["models"]:
        if sha256(run / record["path"]) != record["sha256"]:
            raise ValueError("Frozen checkpoint changed: " + record["path"])
    binding_path = run / "feature_binding.json"
    if sha256(binding_path) != frozen["feature_binding_sha256"]:
        raise ValueError("Training feature-binding file changed")
    binding = json.loads(binding_path.read_text())
    for field in ("model", "upstream", "encoder_context", "oracle_budget",
                  "extraction_device", "inference_precision", "cache_precision", "weights"):
        if binding[field] != manifest[field]:
            raise ValueError("Train/test extraction differs: " + field)
    if binding["extractor_source_hashes"] != manifest["source_hashes"]:
        raise ValueError("Train/test extraction sources differ")
    indexed = {row["spec"]["name"]: row for row in manifest["clips"]}
    if len(indexed) != len(manifest["clips"]):
        raise ValueError("Feature manifest has duplicate scene names")
    for row in binding["clips"]:
        if indexed.get(row["spec"]["name"]) != row:
            raise ValueError("Training/dev feature manifest changed after freeze")
    if {row["spec"]["name"] for row in binding["clips"]} != {
            spec["name"] for split in ("train", "dev") for spec in data.scene_specs(split)}:
        raise ValueError("Training feature binding is incomplete")
    for spec in data.scene_specs("test"):
        record = indexed.get(spec["name"])
        if record is None or record["spec"] != spec:
            raise ValueError("Missing or mismatched test cache: " + spec["name"])
    return root, manifest, frozen, indexed


def load_cache(root, record):
    path = root / record["path"]
    if sha256(path) != record["sha256"]:
        raise ValueError("Test feature checksum changed: " + str(path))
    with np.load(path, allow_pickle=False) as archive:
        cached = {key: archive[key].copy() for key in
                  ("source", "target", "source_frac", "target_frac", "distractor_frac", "rgb_source", "rgb_target")}
    for key, array in cached.items():
        shape = (16, 24, 24, 1024) if key in ("source", "target") else (
            (16, 24, 24, 3) if key.startswith("rgb_") else (16, 24, 24))
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Invalid cached test array: " + key)
        if key not in ("source", "target") and np.any((array < 0) | (array > 1)):
            raise ValueError("Coverage/RGB target outside [0,1]")
    return cached


def fixed_regions(source_frac, target_frac, distractor_frac, dx):
    """The same scoring regions for correct, wrong, and learned interventions."""
    shifted = operators.shift_horizontal(source_frac, dx)
    if not np.allclose(shifted, target_frac, atol=1e-6, rtol=0):
        raise ValueError("Ground truth is not the requested source-mask translation")
    source, target, distractor = source_frac > 0, target_frac > 0, distractor_frac > 0
    union = source | target
    support = operators.dilate_spatial(union, 1)
    regions = {"source_hole": source & ~target, "destination": target,
               "halo": support & ~union, "distractor": distractor,
               "background": ~(support | distractor),
               "common": source | target | operators.shift_horizontal(source, -dx)
                         | distractor | operators.shift_horizontal(distractor, dx)}
    return regions, support


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def region_metrics(error, noop_error, regions):
    """Return raw MSE, true no-op MSE, token count, ratio, and degeneracy."""
    result = {}
    for name in REGIONS:
        mask = regions[name]
        count = int(mask.sum())
        numerator = float(np.sum(error[mask], dtype=np.float64)) if count else 0.0
        denominator = float(np.sum(noop_error[mask], dtype=np.float64)) if count else 0.0
        mse, noop_mse = (numerator / count, denominator / count) if count else (None, None)
        result.update({name + "_tokens": count, name + "_squared_error_sum": numerator,
                       name + "_noop_squared_error_sum": denominator,
                       name + "_mse": mse, name + "_noop_mse": noop_mse,
                       name + "_ratio": mse / max(noop_mse, RATIO_FLOOR) if count else None,
                       name + "_degenerate": int(count == 0 or noop_mse < RATIO_FLOOR)})
    result["primary_ratio"] = _mean(result[name + "_ratio"] for name in ("source_hole", "destination"))
    return result


def _lane_mask(fraction):
    """Scoring-only upper/lower lane chosen from ground-truth soft coverage."""
    _, height, width = fraction.shape
    yy = (np.arange(height) + .5) * data.PATCH
    mean_y = float(np.sum(fraction * yy[None, :, None]) / np.sum(fraction))
    upper = mean_y < data.SIZE / 2
    return np.broadcast_to((yy < data.SIZE / 2 if upper else yy >= data.SIZE / 2)[:, None], (height, width))


def centroid(weights, lane=None, threshold=None):
    weights = np.asarray(weights, dtype=np.float64).copy()
    if lane is not None:
        weights[~lane] = 0
    if threshold is not None:
        weights[weights < threshold] = 0
    mass = float(weights.sum())
    if mass <= 0:
        return None, None, mass
    yy, xx = np.mgrid[:weights.shape[0], :weights.shape[1]]
    return (float(np.sum(weights * (xx + .5) * data.PATCH) / mass),
            float(np.sum(weights * (yy + .5) * data.PATCH) / mass), mass)


def _centroid_error(predicted, actual):
    if predicted[0] is None or actual[0] is None:
        return float(data.SIZE)
    return float(np.hypot(predicted[0] - actual[0], predicted[1] - actual[1]))


def _iou(prediction, actual, lane):
    pred_binary, gt_binary = (prediction >= .5) & lane, (actual >= .5) & lane
    intersection, union = int(np.sum(pred_binary & gt_binary)), int(np.sum(pred_binary | gt_binary))
    return float(intersection / union) if union else 1.0, intersection, union


def coarse_identity_metrics(predicted_rgb, true_source_rgb, source_frac, target_frac, distractor_frac):
    """Two-object color identity, using labels only in this scoring function.

    Average predicted RGB on the intended target's >=70%-covered patches and
    compare it to selected/distractor reference colors from true SOURCE RGB.
    These are coarse object colors, not a claim about detailed identity. True
    colors must be >=.15 apart in Euclidean [0,1] RGB; ineligible cases remain
    explicit in counts and have no invented accuracy value. Ties are failures.
    """
    source_core, target_core, distractor_core = source_frac >= .7, target_frac >= .7, distractor_frac >= .7
    sufficient_cores = source_core.any() and target_core.any() and distractor_core.any()
    separation, selected_distance, distractor_distance = None, None, None
    if sufficient_cores:
        selected_color = np.asarray(true_source_rgb[source_core], np.float64).mean(axis=0)
        distractor_color = np.asarray(true_source_rgb[distractor_core], np.float64).mean(axis=0)
        prediction_color = np.asarray(predicted_rgb[target_core], np.float64).mean(axis=0)
        separation = float(np.linalg.norm(selected_color - distractor_color))
        selected_distance = float(np.linalg.norm(prediction_color - selected_color))
        distractor_distance = float(np.linalg.norm(prediction_color - distractor_color))
    eligible = bool(sufficient_cores and separation >= IDENTITY_COLOR_SEPARATION)
    correct = int(selected_distance < distractor_distance) if eligible else None
    return {"appearance_identity_eligible": int(eligible), "appearance_identity_skipped": int(not eligible),
            "appearance_identity_correct": correct, "appearance_identity_accuracy": float(correct) if eligible else None,
            "appearance_true_color_separation": separation,
            "appearance_distance_to_selected_color": selected_distance,
            "appearance_distance_to_distractor_color": distractor_distance}


def semantic_metrics(prediction, source_prediction, cached, regions):
    """Readout assessment; returns one plain dictionary per tubelet."""
    selected_lane = _lane_mask(cached["target_frac"])
    distractor_lane = _lane_mask(cached["distractor_frac"])
    rows = []
    for t in range(len(cached["target_frac"])):
        occupancy, rgb = prediction["occupancy"][t], prediction["rgb"][t]
        row = {}
        for label, fraction, lane in (("selected", cached["target_frac"][t], selected_lane),
                                      ("distractor", cached["distractor_frac"][t], distractor_lane)):
            pred_centroid = centroid(occupancy, lane, .25)
            true_centroid = centroid(fraction)
            iou, intersection, union = _iou(occupancy, fraction, lane)
            core = fraction >= .7
            rgb_error = np.mean((rgb.astype(np.float64) - cached["rgb_target"][t]) ** 2, axis=-1)
            row.update({label + "_centroid_error_px": _centroid_error(pred_centroid, true_centroid),
                        label + "_centroid_present": int(pred_centroid[0] is not None),
                        label + "_centroid_x": pred_centroid[0], label + "_centroid_y": pred_centroid[1],
                        label + "_centroid_mass": pred_centroid[2],
                        label + "_gt_centroid_x": true_centroid[0], label + "_gt_centroid_y": true_centroid[1],
                        label + "_iou": iou, label + "_iou_intersection": intersection, label + "_iou_union": union,
                        label + "_rgb_core_mse": float(rgb_error[core].mean()) if core.any() else None,
                        label + "_rgb_core_error_sum": float(rgb_error[core].sum()), label + "_rgb_core_tokens": int(core.sum())})
        hole, distractor = regions["source_hole"][t], regions["distractor"][t]
        source_occ, source_rgb = source_prediction["occupancy"][t], source_prediction["rgb"][t]
        source_distractor_centroid = centroid(source_occ, distractor_lane, .25)
        edited_distractor_centroid = centroid(occupancy, distractor_lane, .25)
        # Identically missing centroids are an unchanged readout, while the
        # separate error-to-ground-truth remains the full missing penalty.
        distractor_change = (0.0 if source_distractor_centroid[0] is None and edited_distractor_centroid[0] is None
                             else _centroid_error(edited_distractor_centroid, source_distractor_centroid))
        row.update({"source_hole_ghost_mean_occupancy": float(occupancy[hole].mean()) if hole.any() else None,
                    "source_hole_ghost_fraction_above_half": float((occupancy[hole] >= .5).mean()) if hole.any() else None,
                    "distractor_occupancy_change_mse": float(np.mean((occupancy[distractor] - source_occ[distractor]) ** 2)) if distractor.any() else None,
                    "distractor_rgb_change_mse": float(np.mean((rgb[distractor] - source_rgb[distractor]) ** 2)) if distractor.any() else None,
                    "distractor_centroid_change_px": distractor_change})
        row.update(coarse_identity_metrics(rgb, cached["rgb_source"][t], cached["source_frac"][t],
                                           cached["target_frac"][t], cached["distractor_frac"][t]))
        rows.append(row)
    return rows


def preservation_metrics(error_to_source, max_delta, outside):
    count = int(outside.sum())
    return {"outside_support_tokens": count,
            "outside_support_source_mse": float(np.mean(error_to_source[outside], dtype=np.float64)) if count else None,
            "outside_support_source_squared_error_sum": float(np.sum(error_to_source[outside], dtype=np.float64)),
            "outside_support_max_abs_delta": float(max_delta[outside].max()) if count else None,
            "outside_support_changed_tokens": int(np.sum(max_delta[outside] != 0))}


def aggregate_semantic(rows):
    # Equal tubelet means are explicit; retain pixel/token sums separately so
    # a verifier can also compute pooled alternatives without hidden weights.
    result = {key: _mean(row[key] for row in rows) for key in rows[0]}
    for key in ("appearance_identity_eligible", "appearance_identity_skipped", "appearance_identity_correct"):
        result[key] = sum(row[key] or 0 for row in rows)
    result["appearance_identity_accuracy"] = (result["appearance_identity_correct"] / result["appearance_identity_eligible"]
                                               if result["appearance_identity_eligible"] else None)
    for label in ("selected", "distractor"):
        for suffix in ("rgb_core_error_sum", "rgb_core_tokens", "iou_intersection", "iou_union"):
            key = label + "_" + suffix
            result[key] = sum(row[key] for row in rows)
        result[label + "_missing_tubelets"] = sum(not row[label + "_centroid_present"] for row in rows)
        velocity_errors = []
        for previous, current in zip(rows[:-1], rows[1:]):
            if not previous[label + "_centroid_present"] or not current[label + "_centroid_present"]:
                velocity_errors.append(float(data.SIZE))
                continue
            pred_delta = np.array([current[label + "_centroid_" + axis] - previous[label + "_centroid_" + axis] for axis in ("x", "y")])
            gt_delta = np.array([current[label + "_gt_centroid_" + axis] - previous[label + "_gt_centroid_" + axis] for axis in ("x", "y")])
            velocity_errors.append(float(np.linalg.norm(pred_delta - gt_delta)))
        result[label + "_temporal_velocity_error_px"] = _mean(velocity_errors)
    return result


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n_scenes": 0}
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    sampled = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(sampled, [.025, .975]).tolist(), "n_scenes": int(len(values))}


SUMMARY_METRICS = ("primary_ratio", *(name + "_ratio" for name in REGIONS),
                   "selected_centroid_error_px", "distractor_centroid_error_px",
                   "selected_iou", "distractor_iou", "selected_rgb_core_mse", "distractor_rgb_core_mse",
                   "source_hole_ghost_mean_occupancy", "distractor_occupancy_change_mse",
                   "distractor_rgb_change_mse", "distractor_centroid_change_px",
                   "outside_support_source_mse", "outside_support_changed_tokens",
                   "selected_temporal_velocity_error_px", "distractor_temporal_velocity_error_px",
                   "appearance_identity_accuracy", "appearance_identity_eligible", "appearance_identity_skipped",
                   "appearance_true_color_separation", "appearance_distance_to_selected_color",
                   "appearance_distance_to_distractor_color")


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["arm"])].append(row)
    averaged = []
    for (scene, arm), repeated in sorted(groups.items()):
        expected_count = 3 if arm in ARMS else 1
        if len(repeated) != expected_count:
            raise ValueError("Incomplete learned seed set")
        averaged.append({"scene": scene, "arm": arm, "shift_regime": repeated[0]["shift_regime"],
                         "n_seeds": len(repeated), **{metric: _mean(row[metric] for row in repeated) for metric in SUMMARY_METRICS}})
    by_arm_scene = {(row["arm"], row["scene"]): row for row in averaged}
    summary = {"evaluator_version": EVALUATOR_VERSION, "n_independent_scenes": len({row["scene"] for row in rows}),
               "seed_policy": "Average three seed results per scene and arm BEFORE bootstrapping independent scenes",
               "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "interval": "percentile 95%; exploratory, no multiplicity adjustment"},
               "ratio_floor": RATIO_FLOOR, "methods": {}, "paired_differences": {}, "decision_checks": {}}
    for subgroup in ("all", "seen_magnitude", "heldout_magnitude"):
        included = [row for row in averaged if subgroup == "all" or row["shift_regime"] == subgroup]
        summary["methods"][subgroup] = {}
        summary["paired_differences"][subgroup] = {}
        for arm in (*BASE_ARMS, *ARMS, "genuine_target"):
            arm_rows = [row for row in included if row["arm"] == arm]
            summary["methods"][subgroup][arm] = {
                metric: bootstrap([row[metric] for row in arm_rows if row[metric] is not None]) for metric in SUMMARY_METRICS}
        comparisons = [(arm, baseline) for arm in (*ARMS, "naive", "residual") for baseline in ("noop", "naive", "residual") if arm != baseline]
        comparisons.append(("learned_residual", "geometry_only"))
        for arm, baseline in comparisons:
            arm_rows = [row for row in included if row["arm"] == arm]
            summary["paired_differences"][subgroup][arm + "_minus_" + baseline] = {
                metric: bootstrap([row[metric] - by_arm_scene[(baseline, row["scene"])][metric]
                                   for row in arm_rows if row[metric] is not None and by_arm_scene[(baseline, row["scene"])][metric] is not None])
                for metric in SUMMARY_METRICS}
    genuine = summary["methods"]["all"]["genuine_target"]["selected_centroid_error_px"]
    gate = genuine["mean"] is not None and genuine["mean"] < data.PATCH
    summary["genuine_target_readout_gate"] = {"pass": gate, "threshold_px": data.PATCH,
                                              "rule": "Mean selected-ball centroid error over genuine target test clips < one patch",
                                              **genuine}
    genuine_identity = summary["methods"]["all"]["genuine_target"]["appearance_identity_accuracy"]
    identity_gate = genuine_identity["mean"] is not None and genuine_identity["mean"] >= IDENTITY_READOUT_GATE
    genuine_rows = [row for row in rows if row["arm"] == "genuine_target"]
    summary["genuine_target_identity_gate"] = {
        "pass": identity_gate, "threshold_accuracy": IDENTITY_READOUT_GATE,
        "minimum_true_color_separation": IDENTITY_COLOR_SEPARATION,
        "eligible_tubelets": int(sum(row["appearance_identity_eligible"] for row in genuine_rows)),
        "skipped_tubelets": int(sum(row["appearance_identity_skipped"] for row in genuine_rows)),
        "rule": "Mean per-scene eligible-tubelet two-color identity accuracy on genuine targets >=.90",
        **genuine_identity}
    for arm in (*ARMS, "naive", "residual"):
        metrics = summary["methods"]["all"][arm]
        difference = summary["paired_differences"]["all"][arm + "_minus_noop"]
        latent_pass = (metrics["primary_ratio"]["ci95"][1] < 1
                       and metrics["source_hole_ratio"]["mean"] < 1 and metrics["destination_ratio"]["mean"] < 1)
        semantic_pass = gate and difference["selected_centroid_error_px"]["ci95"][1] < 0
        identity_accuracy = metrics["appearance_identity_accuracy"]["mean"]
        identity_preserved = (identity_gate and identity_accuracy is not None
                              and identity_accuracy >= genuine_identity["mean"] - IDENTITY_ACCURACY_TOLERANCE)
        rgb_difference_upper = difference["selected_rgb_core_mse"]["ci95"][1]
        rgb_improved = rgb_difference_upper is not None and rgb_difference_upper < 0
        appearance_pass = identity_preserved and rgb_improved
        distractor_preserved = metrics["distractor_occupancy_change_mse"]["mean"] <= RATIO_FLOOR and metrics["distractor_rgb_change_mse"]["mean"] <= RATIO_FLOOR
        summary["decision_checks"][arm] = {"latent_rule_pass": latent_pass, "semantic_rule_pass": semantic_pass,
                                              "appearance_rule_pass": appearance_pass,
                                              "coarse_identity_within_genuine_accuracy_tolerance": identity_preserved,
                                              "identity_accuracy_tolerance": IDENTITY_ACCURACY_TOLERANCE,
                                              "selected_rgb_mse_improves_on_noop_with_ci": rgb_improved,
                                              "distractor_probe_output_preserved": distractor_preserved,
                                              "combined_exploratory_pass": latent_pass and semantic_pass and appearance_pass and distractor_preserved}
    summary["degenerate_region_counts_across_method_seed_scene_rows"] = {
        name: int(sum(row[name + "_degenerate"] for row in rows)) for name in REGIONS}
    summary["interpretation_limits"] = ["Readout is coarse occupancy/RGB, not a pixel-video decoder",
                                         "Two-color identity is coarse appearance discrimination, not detailed texture, face, or object identity",
                                         "Readout failure on genuine targets makes semantic intervention conclusions inconclusive",
                                         "Pointwise distractor preservation is partly enforced by spatial edit support",
                                         "Contextual target embeddings may change outside changed pixels",
                                         "Oracle full-sequence selections and simple separated-lane synthetic scenes",
                                         "Sixteen independent scenes, not forty-eight independent learned-seed observations"]
    return summary, averaged


def _load_heads(run, frozen, device):
    import torch
    heads = []
    for record in frozen["models"]:
        if record["arm"] not in ARMS:
            continue
        payload = torch.load(Path(run) / record["path"], map_location="cpu", weights_only=True)
        if payload["arm"] != record["arm"] or payload["seed"] != record["seed"] or payload["epoch"] != record["epoch"]:
            raise ValueError("Checkpoint contents disagree with frozen selection")
        head = operators.build_model(dim=payload["dim"], content_blind=payload["content_blind"])
        head.load_state_dict(payload["state_dict"], strict=True)
        heads.append((record, head.to(device).eval().requires_grad_(False)))
    return heads


def _learned_prediction(model, prepared, device):
    import torch
    selected = prepared["edit_mask"]
    result = prepared["original"].copy()
    inputs = {name: torch.from_numpy(np.ascontiguousarray(prepared[name][selected])).to(device) for name in
              ("original", "base", "anchor", "geometry", "edit_mask")}
    with torch.inference_mode():
        result[selected] = model(**inputs).cpu().numpy()
    return result


def run(features, training_run, out, device="cuda"):
    import torch
    import probe
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    root, manifest, frozen, indexed = validate_binding(features, training_run)
    heads = _load_heads(training_run, frozen, device)
    readout_record = next(record for record in frozen["models"] if record["arm"] == "frozen_readout")
    readout, _ = probe.load_probe(Path(training_run) / readout_record["path"], device)
    clip_rows, step_rows, artifacts = [], [], []
    for spec in data.scene_specs("test"):
        cached = load_cache(root, indexed[spec["name"]])
        source, target = cached["source"].astype(np.float32), cached["target"].astype(np.float32)
        dx = int(spec["dx"]) // data.PATCH
        regions, support = fixed_regions(cached["source_frac"], cached["target_frac"], cached["distractor_frac"], dx)
        baseline = operators.base_edits(source, cached["source_frac"], None, cached["distractor_frac"], dx)
        prepared = operators.prepare_inputs(source, cached["source_frac"], None, cached["distractor_frac"], dx, baseline["residual"])
        if not np.array_equal(prepared["edit_mask"], support):
            raise AssertionError("Operator support and fixed scoring support disagree")
        noop_error = np.mean((source - target) ** 2, axis=-1, dtype=np.float32)
        source_prediction = probe.predict_probe(readout, source, device)
        npz = {"source_frac": cached["source_frac"], "target_frac": cached["target_frac"],
               "distractor_frac": cached["distractor_frac"], "rgb_source": cached["rgb_source"],
               "rgb_target": cached["rgb_target"], "noop_latent_error": noop_error,
               "region_names": np.array(REGIONS), "region_masks": np.stack([regions[name] for name in REGIONS]),
               "edit_support": support, "dx_pixels": np.int32(spec["dx"])}
        collected = defaultdict(list)
        method_specs = [(name, -1, name, None) for name in BASE_ARMS]
        method_specs += [(record["arm"], record["seed"], f"{record['arm']}_{record['seed']}", head) for record, head in heads]
        method_specs += [("genuine_target", -1, "genuine_target", None)]
        for arm, seed, method, head in method_specs:
            edited = target if arm == "genuine_target" else (_learned_prediction(head, prepared, device) if head is not None else baseline[arm])
            if not np.isfinite(edited).all():
                raise ValueError("Nonfinite edited representation: " + method)
            prediction = source_prediction if arm == "noop" else probe.predict_probe(readout, edited, device)
            error = np.mean((edited - target) ** 2, axis=-1, dtype=np.float32)
            delta = edited - source
            source_error, max_delta = np.mean(delta ** 2, axis=-1, dtype=np.float32), np.max(np.abs(delta), axis=-1)
            semantic = semantic_metrics(prediction, source_prediction, cached, regions)
            identity = {"scene": spec["name"], "scene_seed": spec["seed"], "dx_pixels": spec["dx"],
                        "shift_regime": spec["shift_regime"], "arm": arm, "seed": seed, "method": method}
            clip_rows.append({**identity, **region_metrics(error, noop_error, regions),
                              **aggregate_semantic(semantic), **preservation_metrics(source_error, max_delta, ~support)})
            for t in range(data.N_STEPS):
                step_rows.append({**identity, "tubelet": t, "first_frame": 2 * t,
                                  **region_metrics(error[t], noop_error[t], {key: value[t] for key, value in regions.items()}),
                                  **semantic[t], **preservation_metrics(source_error[t], max_delta[t], ~support[t])})
            for key, value in (("occupancy", prediction["occupancy"]), ("rgb", prediction["rgb"]),
                               ("latent_error", error), ("source_preservation_error", source_error), ("source_max_abs_delta", max_delta)):
                collected[key].append(value)
            collected["method_names"].append(method)
            collected["method_arms"].append(arm)
            collected["method_seeds"].append(seed)
            collected["edited_latents_float32_sha256"].append(hashlib.sha256(np.ascontiguousarray(edited).tobytes()).hexdigest())
            print("TEST_METHOD", spec["name"], method, "ratio", round(clip_rows[-1]["primary_ratio"], 6), flush=True)
            del edited, delta, error, source_error, max_delta, prediction
        for key, values in collected.items():
            npz[key] = np.stack(values) if isinstance(values[0], np.ndarray) else np.asarray(values)
        path = out / "predictions" / (spec["name"] + ".npz")
        np.savez_compressed(path, **npz)
        artifacts.append({"scene": spec["name"], "path": str(path.relative_to(out)), "sha256": sha256(path),
                          "feature_cache_sha256": indexed[spec["name"]]["sha256"], "bytes": path.stat().st_size})
        del cached, source, target, baseline, prepared, collected, npz, source_prediction
        gc.collect()
    write_csv(out / "per_clip.csv", clip_rows)
    write_csv(out / "per_tubelet.csv", step_rows)
    summary, averaged = summarize(clip_rows)
    write_csv(out / "seed_averaged_per_clip.csv", averaged)
    summary["binding"] = {"checkpoint_freeze_sha256": sha256(Path(training_run) / "checkpoint_freeze.json"),
                          "feature_manifest_sha256": sha256(root / "manifest.json"),
                          "evaluator_sha256": sha256(Path(__file__)), "device": device,
                          "torch": str(torch.__version__), "numpy": str(np.__version__)}
    write_json(out / "summary.json", summary)
    write_json(out / "evaluation_manifest.json", {"version": EVALUATOR_VERSION, "binding": summary["binding"],
               "input_feature_manifest": manifest, "checkpoint_freeze": frozen, "artifacts": artifacts,
               "metrics": {"per_clip_rows": len(clip_rows), "per_tubelet_rows": len(step_rows),
                           "ratio": "Mean per-token 1024-channel MSE / max(true no-op mean MSE,1e-12); primary equal mean of hole and destination",
                           "centroid": "Weights predicted occupancy>=.25 within GT-assigned upper/lower lane; GT soft-coverage patch-center centroid; missing=384px",
                           "iou": "Predicted occupancy>=.5 against true coverage>=.5 in the same scoring lane",
                           "rgb": "Mean 3-channel error on GT object coverage>=.7; per-tubelet mean then per-clip mean",
                           "appearance_identity": "Predicted RGB mean on target core>=.7 must be closer to true selected source-core RGB mean than true distractor source-core mean; ties fail; true separation>=.15 and all cores nonempty; report eligible/skipped; genuine-target mean accuracy>=.90 gate; edited accuracy within .05 of genuine plus RGB-MSE paired CI improvement required",
                           "outside": "Fixed requested edit support (source/destination union plus one-patch halo), compared to original source",
                           "recount": "NPZ stores every method readout, per-token latent error, source-preservation error and maximum delta, ground truth, fixed region masks",
                           "seed": "-1 denotes deterministic operator/reference, not a fitted seed"}})
    print("TEST_COMPLETE", len(data.scene_specs("test")), "scenes", len(clip_rows), "method/seed rows", flush=True)
    return summary


def self_check():
    """Small analytic fixture; never access rendered scenes or cached features."""
    source = np.zeros((2, 24, 24), dtype=np.float32)
    source[:, 4:7, 5:8] = 1
    target = operators.shift_horizontal(source, 3)
    distractor = np.zeros_like(source)
    distractor[:, 17:20, 15:18] = 1
    regions, support = fixed_regions(source, target, distractor, 3)
    error = np.ones_like(source)
    raw = region_metrics(error, error, regions)
    perfect = region_metrics(np.zeros_like(source), error, regions)
    assert raw["primary_ratio"] == 1 and perfect["primary_ratio"] == 0
    assert all(raw[name + "_ratio"] == 1 for name in REGIONS)
    assert not any(raw[name + "_degenerate"] for name in REGIONS)
    degenerate = region_metrics(np.zeros_like(source), np.zeros_like(source), regions)
    assert all(degenerate[name + "_degenerate"] for name in REGIONS)
    rgb = np.full((*source.shape, 3), .5, np.float32)
    cached = {"source_frac": source, "target_frac": target, "distractor_frac": distractor, "rgb_source": rgb, "rgb_target": rgb}
    source_prediction = {"occupancy": source + distractor, "rgb": rgb}
    target_prediction = {"occupancy": target + distractor, "rgb": rgb}
    truth = semantic_metrics(target_prediction, source_prediction, cached, regions)
    noop = semantic_metrics(source_prediction, source_prediction, cached, regions)
    missing = semantic_metrics({"occupancy": np.zeros_like(source), "rgb": rgb}, source_prediction, cached, regions)
    assert all(row["selected_centroid_error_px"] == 0 and row["selected_iou"] == 1 and row["selected_rgb_core_mse"] == 0 for row in truth)
    assert all(row["selected_centroid_error_px"] == 3 * data.PATCH for row in noop)
    assert all(row["selected_centroid_error_px"] == data.SIZE and row["selected_centroid_present"] == 0 for row in missing)
    assert all(row["source_hole_ghost_mean_occupancy"] == 0 and row["distractor_occupancy_change_mse"] == 0 for row in truth)
    # A moved ball with the distractor's appearance must fail identity even if
    # its occupancy and centroid are perfect. This exercises actual scoring,
    # not just a label copied from which source object was requested.
    reference_colors = rgb[0].copy()
    reference_colors[source[0] >= .7] = [1.0, .1, .1]
    reference_colors[distractor[0] >= .7] = [.1, .1, 1.0]
    correct_colors = rgb[0].copy()
    correct_colors[target[0] >= .7] = [1.0, .1, .1]
    swapped_colors = correct_colors.copy()
    swapped_colors[target[0] >= .7] = [.1, .1, 1.0]
    correct_identity = coarse_identity_metrics(correct_colors, reference_colors, source[0], target[0], distractor[0])
    swapped_identity = coarse_identity_metrics(swapped_colors, reference_colors, source[0], target[0], distractor[0])
    assert correct_identity["appearance_identity_accuracy"] == 1 and correct_identity["appearance_identity_eligible"] == 1
    assert swapped_identity["appearance_identity_accuracy"] == 0 and swapped_identity["appearance_identity_eligible"] == 1
    tied_colors = correct_colors.copy()
    tied_colors[target[0] >= .7] = [.55, .1, .55]
    assert coarse_identity_metrics(tied_colors, reference_colors, source[0], target[0], distractor[0])["appearance_identity_accuracy"] == 0
    no_color_difference = coarse_identity_metrics(correct_colors, rgb[0], source[0], target[0], distractor[0])
    assert no_color_difference["appearance_identity_skipped"] == 1 and no_color_difference["appearance_identity_accuracy"] is None
    missing_core = coarse_identity_metrics(correct_colors, reference_colors, np.zeros_like(source[0]), target[0], distractor[0])
    assert missing_core["appearance_identity_skipped"] == 1
    preservation = preservation_metrics(np.zeros_like(source), np.zeros_like(source), ~support)
    assert preservation["outside_support_changed_tokens"] == 0
    assert bootstrap([1, 1, 1])["ci95"] == [1.0, 1.0]
    # Actual grouping verifies seeds are not counted as independent scenes.
    rows = []
    for scene_index in range(2):
        for arm in (*BASE_ARMS, *ARMS, "genuine_target"):
            for seed in (SEEDS if arm in ARMS else (-1,)):
                row = {"scene": f"fixture_{scene_index}", "arm": arm, "seed": seed,
                       "shift_regime": "seen_magnitude" if scene_index == 0 else "heldout_magnitude",
                       **{metric: 1.0 for metric in SUMMARY_METRICS},
                       **{name + "_degenerate": 0 for name in REGIONS}}
                rows.append(row)
    summary, averaged = summarize(rows)
    assert summary["methods"]["all"]["learned_residual"]["primary_ratio"]["n_scenes"] == 2
    assert all(row["n_seeds"] == 3 for row in averaged if row["arm"] in ARMS)
    return {"status": "passed", "fixture": "analytic masks and errors only; no experimental scene access",
            "checks": ["no-op ratio exactly one", "perfect intervention ratio zero", "degenerate denominator recorded",
                       "known 48px centroid displacement", "missing prediction penalty", "fixed-region IoU/RGB/ghost scoring",
                       "correct color identity passes and swapped identity fails", "identity ties fail", "ineligible color/core cases explicitly skipped",
                       "exact outside-support preservation", "seed averaging before scene bootstrap"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        result = self_check()
        if args.out:
            write_json(args.out / "evaluation_self_check.json", result)
        print(json.dumps(result, indent=2))
        return
    if not all((args.features, args.run, args.out)):
        parser.error("--features, --run and --out are required unless --self-check")
    run(args.features, args.run, args.out, args.device)


if __name__ == "__main__":
    main()
