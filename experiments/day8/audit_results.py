"""Independent, read-only numerical audit of saved Day 8 results.

Does not import experiment/evaluator helpers, render scenes, load an encoder,
or rerun a learned model. Only --out is written. Run after evaluation finishes.
The saved per-token errors for edited representations are inputs to this audit:
edited 1024-D representations are not saved, so their raw errors cannot be
independently reconstructed here. True no-op errors ARE recomputed from the
original 1024-channel source/target caches. Every aggregate and semantic score
is independently recounted from the saved NPZ arrays.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import traceback

# Limit both BLAS pools and the process's eligible CPUs before importing NumPy.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "4"
if hasattr(os, "sched_getaffinity"):
    try:
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    except OSError:
        pass
import numpy as np


HERE = Path(__file__).resolve().parent
PINNED_COMMIT = "598f5bc1770c72b4af437a44f834cef41f0aeb08"
PINNED_SOURCE_COMMIT = "9ebd665f1be56535199af13d2f6deb8c27af92b1"
PINNED_FREEZE = "9b6b16e4122394b7bad17ce46f0bf7ae6ab8fe95c137b6dbf81b9b452b16dba5"
PINNED_EVALUATOR = "fec3a0489c846c96c0b58e8725ab22a20923eed57ac0cce65abfa073b8b0929c"
PINNED_WEIGHTS = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
REGIONS = ("source_hole", "destination", "halo", "distractor", "background", "common")
LEARNED = ("learned_residual", "geometry_only")
DETERMINISTIC = ("noop", "naive", "residual", "wrong_direction", "wrong_object", "genuine_target")
SEEDS = (1801, 1802, 1803)
SUMMARY_METRICS = (
    "primary_ratio", *(name + "_ratio" for name in REGIONS),
    "selected_centroid_error_px", "distractor_centroid_error_px", "selected_iou", "distractor_iou",
    "selected_rgb_core_mse", "distractor_rgb_core_mse", "source_hole_ghost_mean_occupancy",
    "distractor_occupancy_change_mse", "distractor_rgb_change_mse", "distractor_centroid_change_px",
    "outside_support_source_mse", "outside_support_changed_tokens", "selected_temporal_velocity_error_px",
    "distractor_temporal_velocity_error_px", "appearance_identity_accuracy", "appearance_identity_eligible",
    "appearance_identity_skipped", "appearance_true_color_separation", "appearance_distance_to_selected_color",
    "appearance_distance_to_distractor_color")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values, dtype=np.float64)) if values else None


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = []
        self.warnings = []
        self.max_numeric_abs_difference = 0.0

    def require(self, condition, label):
        self.count += 1
        if not condition:
            raise AssertionError(label)

    def equal(self, actual, expected, label, rtol=2e-6, atol=2e-7):
        self.count += 1
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                self.errors.append(label + ": dictionary keys differ")
                return
            for key in expected:
                self.equal(actual[key], expected[key], label + "/" + key, rtol, atol)
            return
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.errors.append(label + ": sequence length/type differs")
                return
            for index, value in enumerate(expected):
                self.equal(actual[index], value, f"{label}/{index}", rtol, atol)
            return
        if expected is None:
            okay = actual is None or actual == ""
        elif isinstance(expected, (bool, str)):
            okay = actual == expected
        elif isinstance(expected, (int, float, np.integer, np.floating)):
            try:
                value = float(actual)
                difference = abs(value - float(expected))
                self.max_numeric_abs_difference = max(self.max_numeric_abs_difference, difference)
                okay = np.isfinite(value) and (value == expected if isinstance(expected, (int, np.integer))
                                               else np.isclose(value, expected, rtol=rtol, atol=atol))
            except (TypeError, ValueError):
                okay = False
        else:
            okay = actual == expected
        if not okay and len(self.errors) < 100:
            self.errors.append(f"{label}: actual={actual!r}, expected={expected!r}")

    def array(self, actual, expected, label, exact=False):
        self.require(actual.shape == expected.shape, label + ": shape")
        self.require(np.array_equal(actual, expected) if exact else
                     np.allclose(actual, expected, rtol=2e-6, atol=2e-7), label + ": values")


def specs(split):
    first, count = {"train": (11000, 24), "dev": (11100, 8), "test": (11200, 16)}[split]
    result = []
    for index in range(count):
        heldout = split == "test" and index >= 8
        dx = ((48, -48, 80, -80) if heldout else (32, -32, 64, -64))[index % 4]
        result.append(dict(name=f"{split}_{first + index}_dx{dx:+d}", seed=first + index,
                           dx=dx, split=split,
                           shift_regime="heldout_magnitude" if heldout else "seen_magnitude"))
    return result


def provenance(root, checks):
    training, feature_root, test = root / "training", root / "features", root / "test"
    frozen_path = training / "checkpoint_freeze.json"
    frozen, binding = read_json(frozen_path), read_json(training / "feature_binding.json")
    manifest, evaluation = read_json(feature_root / "manifest.json"), read_json(test / "evaluation_manifest.json")
    published = read_json(root / "pretest_publication.json")
    checks.equal(sha(frozen_path), PINNED_FREEZE, "raw freeze SHA")
    checks.equal(sha(HERE / "run_2026-10-09/checkpoint_freeze.json"), PINNED_FREEZE, "published freeze copy SHA")
    checks.equal(published["published_commit"], PINNED_COMMIT, "publication commit")
    checks.equal(published["source_commit"], PINNED_SOURCE_COMMIT, "publication source commit")
    checks.equal(published["sha256"], PINNED_FREEZE, "publication freeze SHA")
    checks.equal(published["file"], "experiments/day8/run_2026-10-09/checkpoint_freeze.json", "publication path")
    checks.require(published["bytes_equal_to_GitHub"] is True and
                   published["test_encoded_before_verification"] is False, "pretest attestation")
    checks.require(frozen["test_accessed"] is False, "freeze precedes test access")
    checks.equal(frozen["train_scenes"], 24, "frozen train count")
    checks.equal(frozen["dev_scenes"], 8, "frozen dev count")
    checks.equal(sha(HERE / "evaluate.py"), PINNED_EVALUATOR, "pinned evaluator source")
    checks.equal(set(frozen["source_hashes"]), {"train.py", "operators.py", "probe.py", "data.py"}, "frozen source roster")
    checks.equal(set(manifest["source_hashes"]), {"extract.py", "data.py"}, "extractor source roster")
    for name, digest in {**frozen["source_hashes"], **manifest["source_hashes"]}.items():
        checks.equal(sha(HERE / name), digest, "source hash " + name)
    checks.equal(sha(training / "feature_binding.json"), frozen["feature_binding_sha256"], "raw feature binding SHA")
    canonical_binding = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    checks.equal(binding["extractor_source_hashes"], manifest["source_hashes"], "train/test extractor source")
    for field in ("model", "upstream", "encoder_context", "oracle_budget", "extraction_device",
                  "inference_precision", "cache_precision", "weights"):
        checks.equal(manifest[field], binding[field], "train/test feature provenance " + field)
    checks.equal(manifest["model"], "vjepa2_1_vit_large_384", "encoder model")
    checks.equal(manifest["upstream"], "204698b45b3712590f06245fbfba32d3be539812", "encoder revision")
    checks.equal([manifest[k] for k in ("extraction_device", "inference_precision", "cache_precision")],
                 ["cpu", "bfloat16", "float16"], "production precision")
    checks.equal(manifest["weights"], [{"name": "vjepa2_1_vitl_dist_vitG_384.pt", "bytes": 5151198524,
                                      "sha256": PINNED_WEIGHTS}], "recorded encoder weight identity")
    checks.equal(manifest["test_freeze_sha256"], PINNED_FREEZE, "test extraction binding")
    expected = {(arm, seed) for arm in LEARNED for seed in SEEDS} | {("frozen_readout", 1800)}
    records = {(r["arm"], r["seed"]): r for r in frozen["models"]}
    checks.require(set(records) == expected and len(frozen["models"]) == 7, "exact seven-checkpoint roster")
    checks.equal(read_json(training / "training_policy.json"), frozen["policy"], "training policy")
    training_log = (root / "training.log").read_text().splitlines()
    incomplete_histories = []
    for key, record in records.items():
        expected_path = "probe.pt" if key[0] == "frozen_readout" else f"{key[0]}/{key[1]}/best.pt"
        checks.equal(record["path"], expected_path, "checkpoint path")
        checks.equal(sha(training / expected_path), record["sha256"], "trained weight hash " + expected_path)
        if key[0] in LEARNED:
            folder = (training / expected_path).parent
            completed = read_json(folder / "completed.json")
            checks.equal(completed["record"], record, "completed selection " + expected_path)
            checks.equal(completed["policy"], frozen["policy"], "completed policy")
            checks.equal(completed["source_hashes"], frozen["source_hashes"], "completed source hashes")
            checks.equal(completed["feature_binding_sha256"], canonical_binding, "canonical completed binding SHA")
            curves = read_csv(folder / "curves.csv")
            logged = [line.split() for line in training_log
                      if line.startswith(f"EDIT_EPOCH {key[0]} {key[1]} ")]
            checks.require([int(parts[3]) for parts in logged] == list(range(1, 31)),
                           "all 30 training epochs logged: " + expected_path)
            checks.equal(int(logged[-1][-1]), record["epoch"], "final logged best epoch matches freeze")
            epochs = [int(r["epoch"]) for r in curves]
            checks.require(epochs == list(range(1, len(curves) + 1)) and 0 < len(curves) <= 30,
                           "contiguous nonempty raw curve history")
            if epochs != list(range(1, 31)):
                finding = {"arm": key[0], "seed": key[1], "saved_epochs": epochs,
                           "missing_csv_epochs": sorted(set(range(1, 31)) - set(epochs)),
                           "all_30_epochs_present_in_training_log": True,
                           "final_logged_best_epoch": int(logged[-1][-1]),
                           "raw_fields_reconstructed": False}
                incomplete_histories.append(finding)
                self_note = (f"Incomplete raw curves.csv for {key[0]}/{key[1]}: missing epochs "
                             f"{finding['missing_csv_epochs']}. All 30 epochs and the frozen final best epoch "
                             "are logged. Original CSV retained; missing raw fields are not fabricated.")
                checks.warnings.append(self_note)
            for row, parts in zip(curves, logged):
                checks.equal(round(float(row["dev_score"]), 6), float(parts[7]), "curve/log rounded dev score")
            best = min(curves, key=lambda r: (float(r["dev_score"]), int(r["epoch"])))
            checks.equal(record["epoch"], int(best["epoch"]), "earliest minimum dev epoch")
            checks.equal(record["dev_score"], float(best["dev_score"]), "best dev score")
            checks.equal(read_json(folder / "best_dev.json")["summary"]["dev_score"], record["dev_score"], "selected dev summary")
    for seed in SEEDS:
        checks.equal(records[(LEARNED[0], seed)]["initialization_sha256"],
                     records[(LEARNED[1], seed)]["initialization_sha256"], "paired initialization")
    history = read_json(training / "probe_history.json")
    best_probe = min(history["epochs"], key=lambda r: (r["dev"]["loss"], r["epoch"]))
    checks.equal(records[("frozen_readout", 1800)]["epoch"], best_probe["epoch"], "readout selected epoch")
    checks.equal(records[("frozen_readout", 1800)]["dev_score"], best_probe["dev"]["loss"], "readout selected loss")
    normal = read_json(training / "normalization.json")
    checks.equal(normal["fit_split"], "train", "normalization fitted on training")
    checks.equal(normal["floors"], frozen["normalization_floors"], "frozen normalization floors")
    for role, item in normal["roles"].items():
        checks.equal(len(item["train"]["no_op_mse"]), 24, "normalization training observations")
        floor = max(1e-8, .01 * float(np.median(item["train"]["no_op_mse"])))
        checks.equal(normal["floors"][role], floor, "train median floor " + role)
    precision = read_json(root / "precision_validation.json")
    checks.equal(precision["scene"], specs("train")[0], "BF16 pretest fixture")
    checks.equal(precision["checkpoint_sha256"], PINNED_WEIGHTS, "BF16 weight identity")
    checks.equal(precision["source_hashes"], manifest["source_hashes"], "BF16 extractor identity")
    checks.equal(precision["chosen_precision"], "bfloat16", "BF16 decision")
    gate_rows = precision["precisions"]["bfloat16"]["metrics"]
    checks.require(set(gate_rows) == {"global", "source_hole", "destination"}, "six BF16 fidelity comparisons")
    for role, item in gate_rows.items():
        checks.require(item["true_edit_mse"] > 0, "positive precision denominator")
        for side in ("source", "target"):
            ratio = item[side + "_precision_mse"] / item["true_edit_mse"]
            checks.equal(item[side + "_ratio"], ratio, "BF16 raw ratio " + role + side)
            checks.require(ratio < .01, "BF16 fidelity gate " + role + side)
    all_specs = {s["name"]: s for split in ("train", "dev", "test") for s in specs(split)}
    indexed = {r["spec"]["name"]: r for r in manifest["clips"]}
    checks.require(len(indexed) == len(manifest["clips"]) == 48 and set(indexed) == set(all_specs), "all 48 unique feature scenes")
    binding_names = {r["spec"]["name"] for r in binding["clips"]}
    checks.require(binding_names == {s["name"] for split in ("train", "dev") for s in specs(split)}
                   and len(binding["clips"]) == 32, "training binding membership")
    for r in binding["clips"]:
        checks.equal(indexed[r["spec"]["name"]], r, "unchanged train/dev feature record")
    for name, record in indexed.items():
        checks.equal(record["spec"], all_specs[name], "feature scene spec")
        path = feature_root / record["path"]
        checks.require(path.resolve().is_relative_to(feature_root.resolve()), "cache stays in feature directory")
        checks.equal(path.stat().st_size, record["bytes"], "feature cache bytes " + name)
        checks.equal(sha(path), record["sha256"], "feature cache SHA " + name)
    checks.equal(evaluation["input_feature_manifest"], manifest, "evaluation feature manifest snapshot")
    checks.equal(evaluation["checkpoint_freeze"], frozen, "evaluation checkpoint freeze snapshot")
    for field, value in (("checkpoint_freeze_sha256", PINNED_FREEZE),
                         ("feature_manifest_sha256", sha(feature_root / "manifest.json")),
                         ("evaluator_sha256", PINNED_EVALUATOR)):
        checks.equal(evaluation["binding"][field], value, "evaluation binding " + field)
    artifacts = {r["scene"]: r for r in evaluation["artifacts"]}
    checks.require(len(artifacts) == len(evaluation["artifacts"]) == 16 and
                   set(artifacts) == {s["name"] for s in specs("test")}, "16 prediction artifacts")
    for name, record in artifacts.items():
        path = test / record["path"]
        checks.require(path.resolve().is_relative_to(test.resolve()), "prediction path containment")
        checks.equal(sha(path), record["sha256"], "prediction NPZ hash " + name)
        checks.equal(path.stat().st_size, record["bytes"], "prediction NPZ bytes " + name)
        checks.equal(record["feature_cache_sha256"], indexed[name]["sha256"], "prediction to feature cache binding")
    return manifest, evaluation, indexed, artifacts, {
        "freeze_sha256": PINNED_FREEZE, "publication_commit": PINNED_COMMIT,
        "evaluator_sha256": PINNED_EVALUATOR, "trained_checkpoints_hashed": 7,
        "feature_caches_hashed": 48, "prediction_npzs_hashed": 16,
        "incomplete_training_curve_histories": incomplete_histories,
        "training_log_sha256": sha(root / "training.log"),
        "publication_limit": "Checks pinned bytes and local record of prior exact-GitHub verification; does not independently query GitHub or prove chronology.",
        "encoder_weight_limit": "Checks the recorded official backbone checksum across manifests; does not rehash or rerun the 5GB backbone.",
        "precision_gate_limit": "Recounts saved six pretest ratios and positive denominators, not new encoder inference."}


def translate(array, dx):
    result = np.zeros_like(array)
    if dx >= 0:
        if dx < array.shape[2]:
            result[:, :, dx:] = array[:, :, :array.shape[2] - dx]
    elif -dx < array.shape[2]:
        result[:, :, :dx] = array[:, :, -dx:]
    return result


def regions_from_coverage(source_fraction, target_fraction, distractor_fraction, dx, checks):
    checks.array(target_fraction, translate(source_fraction, dx), "target coverage translation")
    source, target, distractor = source_fraction > 0, target_fraction > 0, distractor_fraction > 0
    union = source | target
    padded = np.pad(union, ((0, 0), (1, 1), (1, 1)))
    support = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2)).any(axis=(-1, -2))
    return dict(source_hole=source & ~target, destination=target, halo=support & ~union,
                distractor=distractor, background=~(support | distractor),
                common=source | target | translate(source, -dx) | distractor | translate(distractor, dx)), support


def latent_metrics(error, noop, regions):
    output = {}
    for name, mask in regions.items():
        n = int(np.count_nonzero(mask))
        total, denominator = float(error[mask].sum(dtype=np.float64)), float(noop[mask].sum(dtype=np.float64))
        mse, reference = (total / n, denominator / n) if n else (None, None)
        output.update({name + "_tokens": n, name + "_squared_error_sum": total,
            name + "_noop_squared_error_sum": denominator, name + "_mse": mse,
            name + "_noop_mse": reference, name + "_ratio": mse / max(reference, 1e-12) if n else None,
            name + "_degenerate": int(not n or reference < 1e-12)})
    output["primary_ratio"] = mean(output[n + "_ratio"] for n in ("source_hole", "destination"))
    return output


def preservation(error, maximum, outside):
    n = int(outside.sum())
    total = float(error[outside].sum(dtype=np.float64))
    return {"outside_support_tokens": n, "outside_support_source_mse": total / n if n else None,
            "outside_support_source_squared_error_sum": total,
            "outside_support_max_abs_delta": float(maximum[outside].max()) if n else None,
            "outside_support_changed_tokens": int(np.count_nonzero(maximum[outside]))}


def lane(fraction):
    y = (np.arange(24, dtype=np.float64) + .5) * 16
    upper = float((fraction * y[None, :, None]).sum() / fraction.sum()) < 192
    return np.broadcast_to((y < 192 if upper else y >= 192)[:, None], (24, 24))


def center(weights, scoring_lane=None):
    w = np.asarray(weights, dtype=np.float64)
    if scoring_lane is not None:
        w = np.where(scoring_lane & (w >= .25), w, 0)
    mass = float(w.sum())
    if mass == 0:
        return None, None, mass
    y, x = np.indices(w.shape, dtype=np.float64)
    return float((w * (x + .5) * 16).sum() / mass), float((w * (y + .5) * 16).sum() / mass), mass


def distance(a, b):
    return 384.0 if a[0] is None or b[0] is None else float(np.hypot(a[0] - b[0], a[1] - b[1]))


def semantic_step(occ, rgb, original_occ, original_rgb, cached, t, regions, lanes):
    row = {}
    color_error = ((rgb.astype(np.float64) - cached["rgb_target"][t]) ** 2).mean(axis=-1)
    for label, fraction, scoring_lane in (
            ("selected", cached["target_frac"][t], lanes[0]),
            ("distractor", cached["distractor_frac"][t], lanes[1])):
        predicted, truth = center(occ, scoring_lane), center(fraction)
        binary, actual = (occ >= .5) & scoring_lane, (fraction >= .5) & scoring_lane
        intersection, union = int((binary & actual).sum()), int((binary | actual).sum())
        core = fraction >= .7
        row.update({label + "_centroid_error_px": distance(predicted, truth),
            label + "_centroid_present": int(predicted[0] is not None),
            label + "_centroid_x": predicted[0], label + "_centroid_y": predicted[1],
            label + "_centroid_mass": predicted[2], label + "_gt_centroid_x": truth[0],
            label + "_gt_centroid_y": truth[1], label + "_iou": intersection / union if union else 1.0,
            label + "_iou_intersection": intersection, label + "_iou_union": union,
            label + "_rgb_core_mse": float(color_error[core].mean()) if core.any() else None,
            label + "_rgb_core_error_sum": float(color_error[core].sum()), label + "_rgb_core_tokens": int(core.sum())})
    hole, other = regions["source_hole"][t], regions["distractor"][t]
    before, after = center(original_occ, lanes[1]), center(occ, lanes[1])
    row.update({"source_hole_ghost_mean_occupancy": float(occ[hole].mean()) if hole.any() else None,
        "source_hole_ghost_fraction_above_half": float((occ[hole] >= .5).mean()) if hole.any() else None,
        "distractor_occupancy_change_mse": float(((occ[other] - original_occ[other]) ** 2).mean()) if other.any() else None,
        "distractor_rgb_change_mse": float(((rgb[other] - original_rgb[other]) ** 2).mean()) if other.any() else None,
        "distractor_centroid_change_px": 0.0 if before[0] is None and after[0] is None else distance(after, before)})
    cores = [cached[k][t] >= .7 for k in ("source_frac", "target_frac", "distractor_frac")]
    separation = selected_distance = other_distance = None
    eligible, correct = False, None
    if all(mask.any() for mask in cores):
        selected_color = cached["rgb_source"][t][cores[0]].astype(np.float64).mean(axis=0)
        other_color = cached["rgb_source"][t][cores[2]].astype(np.float64).mean(axis=0)
        predicted_color = rgb[cores[1]].astype(np.float64).mean(axis=0)
        separation = float(np.linalg.norm(selected_color - other_color))
        selected_distance = float(np.linalg.norm(predicted_color - selected_color))
        other_distance = float(np.linalg.norm(predicted_color - other_color))
        eligible = separation >= .15
        correct = int(selected_distance < other_distance) if eligible else None
    row.update({"appearance_identity_eligible": int(eligible), "appearance_identity_skipped": int(not eligible),
        "appearance_identity_correct": correct, "appearance_identity_accuracy": float(correct) if eligible else None,
        "appearance_true_color_separation": separation, "appearance_distance_to_selected_color": selected_distance,
        "appearance_distance_to_distractor_color": other_distance})
    return row


def semantic_clip(steps):
    output = {key: mean(s[key] for s in steps) for key in steps[0]}
    for key in ("appearance_identity_eligible", "appearance_identity_skipped", "appearance_identity_correct"):
        output[key] = sum(s[key] or 0 for s in steps)
    output["appearance_identity_accuracy"] = (output["appearance_identity_correct"] / output["appearance_identity_eligible"]
                                              if output["appearance_identity_eligible"] else None)
    for label in ("selected", "distractor"):
        for suffix in ("rgb_core_error_sum", "rgb_core_tokens", "iou_intersection", "iou_union"):
            key = label + "_" + suffix
            output[key] = sum(s[key] for s in steps)
        output[label + "_missing_tubelets"] = sum(not s[label + "_centroid_present"] for s in steps)
        velocities = []
        for before, after in zip(steps, steps[1:]):
            if not before[label + "_centroid_present"] or not after[label + "_centroid_present"]:
                velocities.append(384.0)
            else:
                components = [after[label + "_centroid_" + axis] - before[label + "_centroid_" + axis]
                    - after[label + "_gt_centroid_" + axis] + before[label + "_gt_centroid_" + axis] for axis in ("x", "y")]
                velocities.append(float(np.linalg.norm(components)))
        output[label + "_temporal_velocity_error_px"] = mean(velocities)
    return output


def bootstrap(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n_scenes": 0}
    selection = np.random.default_rng(18008).integers(len(values), size=(5000, len(values)))
    return {"mean": float(values.mean()), "ci95": np.percentile(values[selection].mean(axis=1), [2.5, 97.5]).tolist(),
            "n_scenes": len(values)}


def recount_summaries(rows, saved, seed_csv, checks):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["arm"])].append(row)
    averaged = []
    for (scene, arm), group in sorted(grouped.items()):
        checks.require({r["seed"] for r in group} == (set(SEEDS) if arm in LEARNED else {-1}), "per-scene exact seed roster")
        averaged.append(dict(scene=scene, arm=arm, shift_regime=group[0]["shift_regime"], n_seeds=len(group),
                             **{m: mean(r[m] for r in group) for m in SUMMARY_METRICS}))
    checks.equal(len(seed_csv), 128, "seed-averaged row count")
    indexed_csv = {(r["scene"], r["arm"]): r for r in seed_csv}
    checks.require(len(indexed_csv) == len(seed_csv), "unique seed-averaged CSV keys")
    for row in averaged:
        checks.equal(indexed_csv[(row["scene"], row["arm"])], row, "seed-averaged CSV")
    indexed = {(r["arm"], r["scene"]): r for r in averaged}
    methods, paired = {}, {}
    arms = (*DETERMINISTIC[:-1], *LEARNED, "genuine_target")
    comparisons = [(a, b) for a in (*LEARNED, "naive", "residual") for b in ("noop", "naive", "residual") if a != b]
    comparisons.append(("learned_residual", "geometry_only"))
    for subgroup in ("all", "seen_magnitude", "heldout_magnitude"):
        included = [r for r in averaged if subgroup == "all" or r["shift_regime"] == subgroup]
        methods[subgroup], paired[subgroup] = {}, {}
        for arm in arms:
            subset = [r for r in included if r["arm"] == arm]
            methods[subgroup][arm] = {m: bootstrap([r[m] for r in subset]) for m in SUMMARY_METRICS}
        for arm, baseline in comparisons:
            subset = [r for r in included if r["arm"] == arm]
            paired[subgroup][arm + "_minus_" + baseline] = {m: bootstrap([
                r[m] - indexed[(baseline, r["scene"])][m] for r in subset
                if r[m] is not None and indexed[(baseline, r["scene"])][m] is not None]) for m in SUMMARY_METRICS}
    checks.equal(saved["n_independent_scenes"], 16, "independent scene count")
    checks.equal(saved["bootstrap"]["draws"], 5000, "bootstrap draw count")
    checks.equal(saved["bootstrap"]["seed"], 18008, "bootstrap seed")
    checks.equal(saved["ratio_floor"], 1e-12, "test ratio floor", atol=0)
    checks.equal(saved["methods"], methods, "all method CIs")
    checks.equal(saved["paired_differences"], paired, "all paired CIs")
    true_metrics = methods["all"]["genuine_target"]
    position_gate = true_metrics["selected_centroid_error_px"]["mean"] < 16
    identity_mean = true_metrics["appearance_identity_accuracy"]["mean"]
    identity_gate = identity_mean is not None and identity_mean >= .90
    checks.equal(saved["genuine_target_readout_gate"]["pass"], position_gate, "genuine positional gate")
    checks.equal(saved["genuine_target_readout_gate"]["threshold_px"], 16, "genuine positional threshold")
    checks.equal(saved["genuine_target_identity_gate"]["pass"], identity_gate, "genuine identity gate")
    checks.equal(saved["genuine_target_identity_gate"]["threshold_accuracy"], .90, "genuine identity threshold")
    checks.equal(saved["genuine_target_identity_gate"]["minimum_true_color_separation"], .15, "identity eligibility threshold")
    for metric, output_key in (("appearance_identity_eligible", "eligible_tubelets"),
                              ("appearance_identity_skipped", "skipped_tubelets")):
        checks.equal(saved["genuine_target_identity_gate"][output_key],
                     sum(r[metric] for r in rows if r["arm"] == "genuine_target"), "genuine identity denominator")
    for field in ("mean", "ci95", "n_scenes"):
        checks.equal(saved["genuine_target_readout_gate"][field], true_metrics["selected_centroid_error_px"][field], "genuine positional summary")
        checks.equal(saved["genuine_target_identity_gate"][field], true_metrics["appearance_identity_accuracy"][field], "genuine identity summary")
    decisions = {}
    for arm in (*LEARNED, "naive", "residual"):
        m, d = methods["all"][arm], paired["all"][arm + "_minus_noop"]
        latent = m["primary_ratio"]["ci95"][1] < 1 and all(m[r + "_ratio"]["mean"] < 1 for r in ("source_hole", "destination"))
        position = position_gate and d["selected_centroid_error_px"]["ci95"][1] < 0
        accuracy = m["appearance_identity_accuracy"]["mean"]
        identity = identity_gate and accuracy is not None and accuracy >= identity_mean - .05
        rgb = d["selected_rgb_core_mse"]["ci95"][1] is not None and d["selected_rgb_core_mse"]["ci95"][1] < 0
        distractor = all(m[x]["mean"] <= 1e-12 for x in ("distractor_occupancy_change_mse", "distractor_rgb_change_mse"))
        decisions[arm] = dict(latent_rule_pass=latent, semantic_rule_pass=position, appearance_rule_pass=identity and rgb,
            coarse_identity_within_genuine_accuracy_tolerance=identity, identity_accuracy_tolerance=.05,
            selected_rgb_mse_improves_on_noop_with_ci=rgb, distractor_probe_output_preserved=distractor,
            combined_exploratory_pass=latent and position and identity and rgb and distractor)
    checks.equal(saved["decision_checks"], decisions, "all exploratory decision flags")
    checks.equal(saved["degenerate_region_counts_across_method_seed_scene_rows"],
                 {name: sum(r[name + "_degenerate"] for r in rows) for name in REGIONS}, "degenerate region counts")
    return dict(genuine_position_gate=position_gate, genuine_identity_gate=identity_gate,
                decision_checks=decisions, selected_summary=methods["all"],
                paired_primary_ratios={key: value["primary_ratio"] for key, value in paired["all"].items()})


def run(root, checks):
    manifest, evaluation, indexed, artifacts, provenance_report = provenance(root, checks)
    test = root / "test"
    clips, steps = read_csv(test / "per_clip.csv"), read_csv(test / "per_tubelet.csv")
    checks.require(len(clips) == 192 and len(steps) == 3072, "exact 192 clip and 3072 tubelet rows")
    clip_index = {(r["scene"], r["method"]): r for r in clips}
    step_index = {(r["scene"], r["method"], int(r["tubelet"])): r for r in steps}
    checks.require(len(clip_index) == 192 and len(step_index) == 3072, "no duplicated CSV keys")
    checks.equal(evaluation["metrics"]["per_clip_rows"], 192, "manifest clip rows")
    checks.equal(evaluation["metrics"]["per_tubelet_rows"], 3072, "manifest tubelet rows")
    expected_methods = {name: (name, -1) for name in DETERMINISTIC}
    expected_methods.update({f"{arm}_{seed}": (arm, seed) for arm in LEARNED for seed in SEEDS})
    recomputed_clips, coverage_records = [], []
    for spec in specs("test"):
        name = spec["name"]
        with np.load(root / "features" / indexed[name]["path"], allow_pickle=False) as archive:
            cached = {key: archive[key].copy() for key in
                      ("source", "target", "source_frac", "target_frac", "distractor_frac", "rgb_source", "rgb_target")}
        for side in ("source", "target"):
            checks.require(cached[side].shape == (16, 24, 24, 1024) and cached[side].dtype == np.float16,
                           "original 1024-dimensional float16 feature cache")
        with np.load(test / artifacts[name]["path"], allow_pickle=False) as archive:
            z = {key: archive[key].copy() for key in archive.files}
        methods = z["method_names"].tolist()
        checks.require(len(methods) == len(set(methods)) == 12 and set(methods) == set(expected_methods), "exact prediction method roster")
        checks.equal(int(z["dx_pixels"]), spec["dx"], "requested displacement")
        for key in ("source_frac", "target_frac", "distractor_frac", "rgb_source", "rgb_target"):
            checks.array(z[key], cached[key], "prediction cached ground truth " + key, exact=True)
            checks.require(np.isfinite(z[key]).all() and ((z[key] >= 0) & (z[key] <= 1)).all(), "finite bounded coverage/RGB")
        regions, support = regions_from_coverage(z["source_frac"], z["target_frac"], z["distractor_frac"], spec["dx"] // 16, checks)
        checks.equal(z["region_names"].tolist(), list(REGIONS), "region name ordering")
        checks.array(z["region_masks"], np.stack(list(regions.values())), "all fixed region masks", exact=True)
        checks.array(z["edit_support"], support, "fixed edit support", exact=True)
        checks.require(not np.any(support & regions["distractor"]), "distractor outside selected edit support")
        for key in ("latent_error", "source_preservation_error", "source_max_abs_delta", "occupancy"):
            checks.require(z[key].shape == (12, 16, 24, 24) and np.isfinite(z[key]).all(), "saved array shape/finite " + key)
            checks.require((z[key] >= 0).all(), "nonnegative errors/readout " + key)
        checks.require(z["rgb"].shape == (12, 16, 24, 24, 3) and np.isfinite(z["rgb"]).all(), "RGB prediction shape/finite")
        checks.require((z["rgb"] >= 0).all() and (z["rgb"] <= 1).all() and (z["occupancy"] <= 1).all(), "bounded probe outputs")
        source, target = cached.pop("source").astype(np.float32), cached.pop("target").astype(np.float32)
        noop_error = np.mean((source - target) ** 2, axis=-1, dtype=np.float32)
        noop_i, true_i = methods.index("noop"), methods.index("genuine_target")
        checks.array(z["noop_latent_error"], noop_error, "raw 1024-channel no-op MSE", exact=True)
        checks.array(z["latent_error"][noop_i], noop_error, "no-op error map", exact=True)
        checks.require(not np.any(z["latent_error"][true_i]), "genuine target latent error is zero")
        checks.array(z["source_preservation_error"][true_i], noop_error, "genuine target source-change error", exact=True)
        checks.array(z["source_max_abs_delta"][true_i], np.max(np.abs(source - target), axis=-1), "genuine target maximum delta", exact=True)
        for method, array in (("noop", source), ("genuine_target", target)):
            checks.equal(str(z["edited_latents_float32_sha256"][methods.index(method)]),
                         hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(), "unedited/reference latent bytes")
        del source, target
        lanes = lane(z["target_frac"]), lane(z["distractor_frac"])
        for mi, method in enumerate(methods):
            arm, seed = expected_methods[method]
            checks.equal(str(z["method_arms"][mi]), arm, "stored method arm")
            checks.equal(int(z["method_seeds"][mi]), seed, "stored method seed")
            error, source_error, maximum = z["latent_error"][mi], z["source_preservation_error"][mi], z["source_max_abs_delta"][mi]
            checks.require(np.all(source_error <= maximum ** 2 * (1 + 2e-6) + 1e-7) and
                           np.all(source_error >= maximum ** 2 / 1024 * (1 - 2e-6) - 1e-7), "1024-channel source error/max bounds")
            checks.require(np.all(np.sqrt(error) <= np.sqrt(source_error) + np.sqrt(noop_error) + 2e-5) and
                           np.all(np.sqrt(error) + 2e-5 >= np.abs(np.sqrt(source_error) - np.sqrt(noop_error))), "latent-distance triangle inequality")
            if arm in (*LEARNED, "noop", "naive", "residual"):
                checks.require(not np.any(maximum[~support]) and not np.any(source_error[~support]), "exact outside-support latent preservation")
                checks.array(z["occupancy"][mi][~support], z["occupancy"][noop_i][~support], "outside pointwise occupancy preservation", exact=True)
                checks.array(z["rgb"][mi][~support], z["rgb"][noop_i][~support], "outside pointwise RGB preservation", exact=True)
            semantic = [semantic_step(z["occupancy"][mi, t], z["rgb"][mi, t], z["occupancy"][noop_i, t],
                                     z["rgb"][noop_i, t], z, t, regions, lanes) for t in range(16)]
            identity = dict(scene=name, scene_seed=spec["seed"], dx_pixels=spec["dx"], shift_regime=spec["shift_regime"],
                            arm=arm, seed=seed, method=method)
            clip = {**identity, **latent_metrics(error, noop_error, regions), **semantic_clip(semantic),
                    **preservation(source_error, maximum, ~support)}
            checks.equal(clip_index[(name, method)], clip, name + "/" + method + "/clip")
            recomputed_clips.append(clip)
            for t in range(16):
                row = {**identity, "tubelet": t, "first_frame": 2 * t,
                       **latent_metrics(error[t], noop_error[t], {k: v[t] for k, v in regions.items()}),
                       **semantic[t], **preservation(source_error[t], maximum[t], ~support[t])}
                checks.equal(step_index[(name, method, t)], row, name + "/" + method + "/tubelet/" + str(t))
        coverage_records.append({"scene": name, "method_rows": 12, "tubelet_rows": 192,
                                 "region_tokens": {k: int(v.sum()) for k, v in regions.items()}})
        print("AUDITED_SCENE", name, flush=True)
    summary = read_json(test / "summary.json")
    checks.equal(summary["binding"], evaluation["binding"], "summary/evaluation binding")
    recount = recount_summaries(recomputed_clips, summary, read_csv(test / "seed_averaged_per_clip.csv"), checks)
    return {"provenance": provenance_report, "counts": {"scenes": 16, "method_seed_scene_rows": 192,
             "tubelet_rows": 3072, "seed_averaged_rows": 128}, "scene_audits": coverage_records,
            "independent_recount": recount,
            "limitations": [
                "Edited latent vectors are not saved. Learned/deterministic edit error maps are input evidence; this audit independently verifies their aggregation, bounds and provenance, not their raw model outputs.",
                "Original source-to-target no-op MSE is independently recomputed in all 1024 channels; genuine-target reference and no-op byte hashes are checked.",
                "Readout occupancy/RGB arrays are input evidence. All semantic, color-identity, trajectory, seed-averaged, bootstrap and gate values are independently recomputed without evaluator helpers.",
                "No model or encoder rerun, no test tuning. Local publication attestations are not a fresh remote-GitHub verification."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Completed day8_compute directory")
    parser.add_argument("--out", type=Path, required=True, help="New audit JSON report; no other files are written")
    args = parser.parse_args()
    checks, result, fatal = Checks(), {}, None
    try:
        result = run(args.root, checks)
    except Exception as exc:
        fatal = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    passed = not fatal and not checks.errors
    report = {"audit_version": "day8_independent_saved_results_v1", "status":
              "passed_with_warnings" if passed and checks.warnings else "passed" if passed else "failed",
              "audit_source_sha256": sha(__file__), "checks": checks.count,
              "max_compared_numeric_abs_difference": checks.max_numeric_abs_difference,
              "mismatches": checks.errors, "warnings": checks.warnings, "fatal": fatal, "cpu_limit": 4, **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "checks", "max_compared_numeric_abs_difference", "mismatches", "warnings", "fatal")}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
