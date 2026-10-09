"""Independent CPU audit of a COMPLETED Day5 result and prediction caches.

The frozen renderer is reused to regenerate ground truth. Dense masks, pixel
counts, overlap protection, aggregates, success guards and bootstrap intervals
are independently derived here; the production scorer is never imported.
No model runs. Refuse to render any scene until a complete result and all twelve
test cache/manifest files (or all three smoke files) have been verified present.
This file is outside the frozen experiment's source digest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_TEST = {"long_occlusion": [6100, 6101, 6102, 6103],
                 "crossing": [6200, 6201, 6202, 6203],
                 "scale_camera": [6300, 6301, 6302, 6303]}
EXPECTED_SMOKE = {"long_occlusion": [5100], "crossing": [5200], "scale_camera": [5300]}
ARMS = ("raw_part", "parent_intersection")
REPRESENTATIONS = ("raw_mask", "effective_edit")
PARTS = {1: "car_A.front_door", 2: "car_A.window", 3: "car_B.front_door", 4: "car_B.window"}
FLOAT_FIELDS = {"iou", "pixel_recall"}
BOOLEAN_FIELDS = {"visible", "predicted_present", "pair_has_eligible_gt_patch", "wrong_car_present"}
TEXT_FIELDS = {"scene", "condition", "arm", "representation", "target"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, path="value"):
    """Recursive exact discrete comparison and tightly bounded float comparison."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: keys differ")
        for key in expected:
            equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: list differs")
        for i, value in enumerate(expected):
            equal(actual[i], value, f"{path}[{i}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"{path}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def eligible_pairs(truth):
    """Day4 target eligibility independently recomputed from visible GT pixels."""
    n, h, w = truth.shape
    result = {}
    for pid in PARTS:
        per_frame = (truth == pid).reshape(n, h // 16, 16, w // 16, 16).mean(axis=(2, 4))
        per_pair = per_frame.reshape(n // 2, 2, h // 16, w // 16)
        result[pid] = ((per_pair.mean(axis=1) >= .70) &
                       (per_pair.min(axis=1) >= .65)).any(axis=(1, 2))
    return result


def independently_score(scene, owners, masks, arm, representation, parents=False):
    """Count original NumPy pixel intersections; no production metric helpers."""
    ids = (101, 102) if parents else (1, 2, 3, 4)
    eligible = None if parents else eligible_pairs(scene.masks)
    rows = []
    for frame in range(1, len(scene.frames)):
        for column, pid in enumerate(ids):
            owner = column + 1 if parents else 1 + (pid - 1) // 2
            target = owners[frame] == owner if parents else scene.masks[frame] == pid
            predicted = masks[frame, column]
            gt_area = int(np.count_nonzero(target))
            area = int(np.count_nonzero(predicted))
            intersection = int(np.count_nonzero(predicted & target))
            union = int(np.count_nonzero(predicted | target))
            other_car = int(np.count_nonzero(predicted & (owners[frame] == 3 - owner)))
            other_own_surface = int(np.count_nonzero(predicted & (owners[frame] == owner) & ~target))
            background = int(np.count_nonzero(predicted & (owners[frame] == 0)))
            require(intersection + other_car + other_own_surface + background == area,
                    f"Pixel ownership partition failed: {scene.name}/{frame}/{pid}")
            rows.append({"scene": scene.name, "seed": scene.seed, "condition": scene.condition,
                         "arm": arm, "representation": representation, "frame": frame,
                         "part_id": pid, "parent_owner_id": owner,
                         "target": f"car_{'A' if owner == 1 else 'B'}" if parents else PARTS[pid],
                         "visible": gt_area > 0, "predicted_present": area > 0,
                         "pair_has_eligible_gt_patch": None if parents else bool(eligible[pid][frame // 2]),
                         "gt_pixels": gt_area, "predicted_pixels": area,
                         "intersection_pixels": intersection, "union_pixels": union,
                         "false_positive_pixels": area - intersection,
                         "false_negative_pixels": gt_area - intersection,
                         "wrong_car_pixels": other_car, "wrong_car_present": other_car > 0,
                         "same_car_other_pixels": other_own_surface, "background_pixels": background,
                         "image_pixels": target.size,
                         "iou": intersection / union if gt_area else None,
                         "pixel_recall": intersection / gt_area if gt_area else None})
    return rows


def summary(rows):
    visible = [r for r in rows if r["gt_pixels"] > 0]
    absent = [r for r in rows if r["gt_pixels"] == 0]
    total = lambda group, field: sum(r[field] for r in group)
    correct = total(rows, "intersection_pixels")
    area = total(rows, "predicted_pixels")
    visible_area = total(visible, "predicted_pixels")
    gt_area = total(visible, "gt_pixels")
    wrong = total(rows, "wrong_car_pixels")
    false_pixels = total(rows, "false_positive_pixels")
    return {"scored_part_frames": len(rows), "visible_part_frames": len(visible),
            "absent_part_frames": len(absent),
            "mean_iou_given_visible": ratio(float(total(visible, "iou")), len(visible)),
            "visible_micro_pixel_recall": ratio(correct, gt_area),
            "visible_micro_pixel_precision": ratio(correct, visible_area),
            "all_frame_micro_pixel_precision": ratio(correct, area),
            "wrong_car_pixels": wrong,
            "wrong_car_pixel_fraction_of_predictions": ratio(wrong, area),
            "wrong_car_pixel_fraction_of_false_positives": ratio(wrong, false_pixels),
            "wrong_car_part_frame_rate": ratio(total(rows, "wrong_car_present"), len(rows)),
            "false_presence_given_absent": ratio(total(absent, "predicted_present"), len(absent)),
            "absent_predicted_pixels": total(absent, "predicted_pixels"),
            "predicted_pixels": area, "false_positive_pixels": false_pixels,
            "same_car_other_pixels": total(rows, "same_car_other_pixels"),
            "background_pixels": total(rows, "background_pixels"),
            "predicted_part_frame_coverage": ratio(total(rows, "predicted_present"), len(rows)),
            "predicted_image_pixel_coverage": ratio(area, total(rows, "image_pixels")),
            "visible_predicted_to_truth_area_ratio": ratio(visible_area, gt_area),
            "clips": len({r["scene"] for r in rows})}


def aggregate(rows):
    return {"overall": summary(rows),
            "visible_without_eligible_gt_patch_in_pair": summary(
                [r for r in rows if r["visible"] and r["pair_has_eligible_gt_patch"] is False]),
            "by_condition": {c: summary([r for r in rows if r["condition"] == c])
                             for c in sorted({r["condition"] for r in rows})},
            "per_clip": {c: summary([r for r in rows if r["scene"] == c])
                         for c in sorted({r["scene"] for r in rows})}}


def compare_csv(path, expected):
    key = lambda r: (r["scene"], r["arm"], r["representation"], r["frame"], r["part_id"])
    with Path(path).open(newline="") as stream:
        actual = []
        for row in csv.DictReader(stream):
            parsed = {}
            for field, value in row.items():
                if field in TEXT_FIELDS:
                    parsed[field] = value
                elif value == "":
                    parsed[field] = None
                elif field in BOOLEAN_FIELDS:
                    require(value in {"True", "False"}, f"Invalid CSV boolean {field}={value}")
                    parsed[field] = value == "True"
                elif field in FLOAT_FIELDS:
                    parsed[field] = float(value)
                else:
                    parsed[field] = int(value)
            actual.append(parsed)
    index = {key(r): r for r in actual}
    require(len(index) == len(actual) == len(expected), f"{path}: duplicate/missing rows")
    require(set(index) == {key(r) for r in expected}, f"{path}: row keys differ")
    for row in expected:
        equal(index[key(row)], row, f"{path.name}:{key(row)}")
    return len(actual)


def independent_bootstrap(raw, gated, config):
    names = sorted({r["scene"] for r in raw})
    per_clip = []
    # Store numerator/denominator pairs directly, avoiding production sufficient-statistic layout.
    metrics = ("wrong_car_pixels", "mean_iou_given_visible", "visible_micro_pixel_recall",
               "wrong_car_pixel_fraction_of_predictions", "wrong_car_part_frame_rate",
               "false_presence_given_absent")
    for rows in (raw, gated):
        stats = [summary([r for r in rows if r["scene"] == name]) for name in names]
        per_clip.append({metric: np.asarray([[s[metric], 1] if metric == "wrong_car_pixels" else
                        [s[metric]["numerator"], s[metric]["denominator"]] for s in stats], dtype=float)
                         for metric in metrics})
    draws = np.random.default_rng(config["seed"]).integers(0, len(names),
                  size=(config["resamples"], len(names)))
    result = {}
    for metric in metrics:
        a, b = per_clip[0][metric], per_clip[1][metric]
        a_sum, b_sum = a.sum(axis=0), b.sum(axis=0)
        a_draw, b_draw = a[draws].sum(axis=1), b[draws].sum(axis=1)
        if metric == "wrong_car_pixels":
            difference = float(b_sum[0] - a_sum[0])
            samples = b_draw[:, 0] - a_draw[:, 0]
        elif a_sum[1] and b_sum[1]:
            difference = float(b_sum[0] / b_sum[1] - a_sum[0] / a_sum[1])
            valid = (a_draw[:, 1] > 0) & (b_draw[:, 1] > 0)
            samples = b_draw[valid, 0] / b_draw[valid, 1] - a_draw[valid, 0] / a_draw[valid, 1]
        else:
            difference, samples = None, np.asarray([])
        result[metric] = {"difference_parent_minus_raw": difference,
                          "paired_clip_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
                          "valid_resamples": len(samples)}
    return {"configuration": config, "clips": names, "metrics": result}


def assessment(arms):
    raw = arms["raw_part"]["effective_edit"]["overall"]
    gated = arms["parent_intersection"]["effective_edit"]["overall"]
    guards = {}
    for representation in REPRESENTATIONS:
        for metric in ("mean_iou_given_visible", "visible_micro_pixel_recall"):
            a = arms["raw_part"][representation]["overall"][metric]["rate"]
            b = arms["parent_intersection"][representation]["overall"][metric]["rate"]
            loss = None if a is None or b is None else a - b
            guards[f"{representation}.{metric}"] = {"loss_raw_minus_parent": loss,
                "max_allowed_loss": .01, "passed": loss is not None and loss <= .01 + 1e-12}
    decreased = gated["wrong_car_pixels"] < raw["wrong_car_pixels"]
    return {"wrong_car_effective_paint_strictly_lower": decreased,
            "wrong_car_pixel_difference_parent_minus_raw": gated["wrong_car_pixels"] - raw["wrong_car_pixels"],
            "relative_wrong_car_effective_pixel_reduction": ratio(
                raw["wrong_car_pixels"] - gated["wrong_car_pixels"], raw["wrong_car_pixels"]),
            "guardrails": guards, "passed": decreased and all(g["passed"] for g in guards.values()),
            "meaning": "fixed synthetic criterion only; uncertainty descriptive; no real-video/face/generative claim"}


def audit(root, stage, freeze):
    root, freeze = Path(root).resolve(), Path(freeze).resolve()
    result_path = root / stage / "results.json"
    require(result_path.exists(), "Completed results.json required before rendering any audit scene")
    report = json.loads(result_path.read_text())
    frozen = json.loads(freeze.read_text())
    equal(report["frozen_config"], frozen, "frozen_config")
    require(report["stage"] == stage, "Stage mismatch")
    equal(frozen["test_seeds"], EXPECTED_TEST, "test_seeds")
    equal(frozen["smoke_seeds"], EXPECTED_SMOKE, "smoke_seeds")
    hashes = {name: sha(HERE.parent / name) for name in frozen["source_hashes"]}
    equal(hashes, frozen["source_hashes"], "source_hashes")
    require(hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() == frozen["source_digest"],
            "Combined source digest differs")
    seeds = EXPECTED_TEST if stage == "test" else EXPECTED_SMOKE
    names = {f"{stage}_{condition}_{seed}": (condition, seed)
             for condition, values in seeds.items() for seed in values}
    clip_metadata = {entry["scene"]: entry for entry in report["clips"]}
    require(len(clip_metadata) == len(report["clips"]) == len(names) and clip_metadata.keys() == names.keys(),
            "Incomplete/duplicate fixed clip list")
    # Complete cache check happens BEFORE importing the renderer or generating ANY scene.
    manifests = {}
    for name in names:
        folder = root / "clips" / name
        manifest_path = folder / "clip_manifest.json"
        require(manifest_path.is_file(), f"Missing completed manifest: {name}")
        require(sha(manifest_path) == clip_metadata[name]["clip_manifest_sha256"], f"Manifest hash: {name}")
        manifests[name] = json.loads(manifest_path.read_text())
        for role in ("children", "parents"):
            path = folder / role / "sam2_masks.npz"
            require(path.is_file(), f"Missing prediction cache: {name}/{role}")
            require(sha(path) == manifests[name]["cache_sha256"][role], f"Cache hash: {name}/{role}")
    for required in ("dense_rows.csv", "parent_rows.csv"):
        require((root / stage / required).is_file(), f"Missing completed {required}")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import parent_benchmark as renderer
    all_rows, all_parents, invariants = [], [], []
    for name, (condition, seed) in names.items():
        scene, owners = renderer.generate_scene(seed, condition, name)
        identity = manifests[name]["identity"]
        expected_identity = {"source_digest": frozen["source_digest"],
            "rgb_sha256": array_sha(scene.frames),
            "children0_sha256": array_sha(np.stack([scene.masks[0] == pid for pid in PARTS])),
            "parents0_sha256": array_sha(np.stack([owners[0] == owner for owner in (1, 2)])),
            "checkpoint_sha256": frozen["checkpoint_sha256"], "policy": frozen["policy"],
            "model_revision": frozen["model_revision"], "config": frozen["model_config"],
            "postprocessing": False, "precision": "float32", "prompt_frames": [0]}
        equal(identity, expected_identity, f"identity:{name}")
        require(array_sha(scene.masks) == clip_metadata[name]["part_truth_sha256"], f"Part truth hash: {name}")
        require(array_sha(owners) == clip_metadata[name]["owner_truth_sha256"], f"Owner truth hash: {name}")
        require(clip_metadata[name]["condition"] == condition and clip_metadata[name]["seed"] == seed,
                f"Clip condition/seed differs: {name}")
        arrays = {}
        for role, ids in (("children", [1, 2, 3, 4]), ("parents", [101, 102])):
            with np.load(root / "clips" / name / role / "sam2_masks.npz", allow_pickle=False) as cache:
                require(cache["ids"].tolist() == ids, f"Prediction ID order: {name}/{role}")
                arrays[role] = cache["masks"]
            require(arrays[role].dtype == bool and arrays[role].shape == (64, len(ids), 384, 384),
                    f"Prediction shape/type: {name}/{role}")
        child, parent = arrays["children"], arrays["parents"]
        owning_parent = parent[:, [0, 0, 1, 1]]
        # A pixel has exactly one child owner iff it is permitted by original overlap protection.
        unambiguous = np.count_nonzero(child, axis=1) == 1
        effective = child & unambiguous[:, None]
        constrained = child & owning_parent
        constrained_effective = effective & owning_parent
        require(not np.any(constrained_effective & ~effective), f"Newly released edit pixels: {name}")
        require(not np.any(constrained & ~child), f"Constrained tracker adds pixels: {name}")
        require(int(np.count_nonzero(effective, axis=1).max()) <= 1, f"Overlapping edits: {name}")
        require(not np.any(constrained_effective & ~owning_parent), f"Edit outside predicted parent: {name}")
        variants = {"raw_part": {"raw_mask": child, "effective_edit": effective},
                    "parent_intersection": {"raw_mask": constrained, "effective_edit": constrained_effective}}
        for arm in ARMS:
            for representation in REPRESENTATIONS:
                all_rows.extend(independently_score(scene, owners, variants[arm][representation], arm, representation))
        all_parents.extend(independently_score(scene, owners, parent, "independent_parent", "raw_mask", parents=True))
        invariants.append({"scene": name, "source_and_cache_hashes": True, "ground_truth_hashes": True,
                           "original_overlap_protection": True, "gated_edit_subset": True,
                           "effective_added_pixels": 0})
        print(f"AUDIT_CLIP_OK {name}", flush=True)
    dense_count = compare_csv(root / stage / "dense_rows.csv", all_rows)
    parent_count = compare_csv(root / stage / "parent_rows.csv", all_parents)
    groups = {arm: {representation: [r for r in all_rows if r["arm"] == arm and r["representation"] == representation]
                   for representation in REPRESENTATIONS} for arm in ARMS}
    arms = {arm: {representation: aggregate(groups[arm][representation])
                  for representation in REPRESENTATIONS} for arm in ARMS}
    equal(report["arms"], arms, "all_arm_aggregates")
    equal(report["parents"], aggregate(all_parents), "all_parent_aggregates")
    expected_rows = len(names) * 63 * 4
    for arm in ARMS:
        for representation in REPRESENTATIONS:
            require(arms[arm][representation]["overall"]["scored_part_frames"] == expected_rows,
                    "Unexpected scored part-frame total")
    success = assessment(arms)
    equal(report["success"], success, "success_assessment")
    bootstrap = {representation: independent_bootstrap(groups["raw_part"][representation],
                 groups["parent_intersection"][representation], frozen["policy"]["bootstrap"])
                 for representation in REPRESENTATIONS}
    equal(report["paired_comparison"], bootstrap, "paired_clip_bootstrap")
    raw, gated = arms["raw_part"]["effective_edit"], arms["parent_intersection"]["effective_edit"]
    return {"status": "passed", "stage": stage, "audited_result_sha256": sha(result_path),
            "frozen_config_sha256": sha(freeze), "frozen_source_digest": frozen["source_digest"],
            "audit_source_sha256": sha(__file__), "clips": len(names),
            "dense_csv_rows_checked": dense_count, "parent_csv_rows_checked": parent_count,
            "scored_part_frames_per_arm_representation": expected_rows,
            "visible_part_frames": raw["overall"]["visible_part_frames"],
            "absent_part_frames": raw["overall"]["absent_part_frames"],
            "checks": {"complete_seed_set": True, "frozen_sources_and_policy": True,
                       "cache_prompt_RGB_and_truth_hashes": True, "independent_dense_rows_equal_csv": True,
                       "independent_parent_rows_equal_csv": True, "pixel_ownership_partition": True,
                       "original_overlap_protection": True, "gated_edit_subset": True,
                       "all_aggregates_conditions_clips_and_slivers": True,
                       "four_success_guards": True, "independent_paired_bootstrap": True},
            "success": success, "effective_raw": raw, "effective_parent": gated,
            "paired_comparison": bootstrap, "clip_invariants": invariants,
            "limits": ["Reuses the frozen renderer; independently verifies scoring, not renderer realism.",
                       "Cache hashes and manifests establish artifact consistency, not independent observation of model execution.",
                       "Secondary patch readout is outside this dense-edit audit.",
                       "This audit does not prove generalization outside the twelve fixed synthetic scenes."]}


def markdown(result):
    raw, gated = result["effective_raw"], result["effective_parent"]
    reduction = result["success"]["relative_wrong_car_effective_pixel_reduction"]
    relative = "undefined (zero baseline)" if reduction["rate"] is None else f"{100 * reduction['rate']:.3f}%"
    lines = ["# Independent Day5 result audit", "", f"Status: **{result['status']}**. Frozen milestone pass: **{result['success']['passed']}**.", "",
             f"Recomputed {result['dense_csv_rows_checked']:,} dense CSV rows and {result['parent_csv_rows_checked']:,} parent rows directly from cached NumPy masks and regenerated labels. All source/cache/prompt/truth hashes, aggregates, sliver counts, four guards and paired bootstrap intervals match.", "",
             f"Per arm and representation: {result['scored_part_frames_per_arm_representation']:,} part-frames = {result['visible_part_frames']:,} visible + {result['absent_part_frames']:,} absent; frame zero excluded.", "",
             f"Effective wrong-car pixels: {raw['overall']['wrong_car_pixels']:,} → {gated['overall']['wrong_car_pixels']:,}; relative reduction {relative}. No protected overlap pixels were released.", "",
             "| Condition | Wrong-car pixels, raw → parent | Visible IoU change | Visible recall change |",
             "|---|---:|---:|---:|"]
    for condition in raw["by_condition"]:
        a, b = raw["by_condition"][condition], gated["by_condition"][condition]
        iou = 100 * (b["mean_iou_given_visible"]["rate"] - a["mean_iou_given_visible"]["rate"])
        recall = 100 * (b["visible_micro_pixel_recall"]["rate"] - a["visible_micro_pixel_recall"]["rate"])
        lines.append(f"| {condition} | {a['wrong_car_pixels']:,} → {b['wrong_car_pixels']:,} | {iou:+.3f} pp | {recall:+.3f} pp |")
    lines += ["", "The subset policy guarantees non-increasing wrong-car pixels. The measured reduction must be interpreted with its visible-detail cost. A pooled milestone pass can conceal a condition-specific regression.", ""]
    lines += [f"- {limit}" for limit in result["limits"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--stage", choices=("smoke", "test"), default="test")
    parser.add_argument("--frozen-path", type=Path, default=HERE / "frozen_config.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.run_root / args.stage / "independent_audit.json"
    try:
        result = audit(args.run_root, args.stage, args.frozen_path)
    except Exception as error:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"status": "failed", "stage": args.stage,
                              "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    destination.with_suffix(".md").write_text(markdown(result))
    print(json.dumps({"status": result["status"], "clips": result["clips"],
                      "criterion_passed": result["success"]["passed"], "output": str(destination)}))


if __name__ == "__main__":
    main()
