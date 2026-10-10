"""Compare frozen Day4 test outputs without inference, tuning, or model changes.

Inputs must contain the same complete 18-clip test set and the same evaluation
labels. The V-JEPA variant is selected from the supplied development freeze,
never by ranking test results. All uncertainty resamples whole paired clips.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import benchmark as bench

VJEPA_METHODS = ("global_vjepa", "template_flow", "persistent_vjepa",
                 "no_motion", "no_context", "no_memory")
BOOL_FIELDS = ("boundary", "recovery", "present", "hit", "wrong_car", "wrong_part", "id_switch")
INT_FIELDS = ("seed", "tubelet", "first_frame", "window", "row", "col", "predicted_gt")
FLOAT_FIELDS = ("score", "threshold", "chance")
METRICS = ("localization_accuracy_given_visible", "false_presence_given_absent", "recovery_accuracy")
GROUND_TRUTH_FIELDS = ("seed", "condition", "first_frame", "state", "window", "boundary", "recovery", "chance")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text())


def _boolean(value, field, line):
    if value not in ("True", "False"):
        raise ValueError(f"Line {line}: {field} must be True or False, got {value!r}")
    return value == "True"


def read_prediction_rows(path):
    """Parse CSV types explicitly; the string 'False' must never count as true."""
    rows = []
    required = {"scene", "condition", "method", "target", "state", *BOOL_FIELDS, *INT_FIELDS, *FLOAT_FIELDS}
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Prediction CSV missing fields: {sorted(required-set(reader.fieldnames or []))}")
        for line, raw in enumerate(reader, 2):
            row = dict(raw)
            for field in BOOL_FIELDS:
                row[field] = _boolean(row[field], field, line)
            for field in INT_FIELDS:
                row[field] = int(row[field])
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
                if not math.isfinite(row[field]):
                    raise ValueError(f"Line {line}: nonfinite {field}")
            if row["state"] not in ("visible", "absent", "ambiguous"):
                raise ValueError(f"Line {line}: unexpected state {row['state']}")
            if row["target"] not in bench.PART_NAMES.values():
                raise ValueError(f"Line {line}: unknown part {row['target']}")
            if not (0 <= row["row"] < bench.GRID and 0 <= row["col"] < bench.GRID):
                raise ValueError(f"Line {line}: out-of-grid prediction")
            if row["first_frame"] != bench.TUBELET*row["tubelet"]:
                raise ValueError(f"Line {line}: mismatched frame/tubelet")
            if row["window"] != row["tubelet"]//bench.WINDOW_STEPS+1:
                raise ValueError(f"Line {line}: mismatched encoder window")
            if row["boundary"] != (row["tubelet"] % bench.WINDOW_STEPS == 0):
                raise ValueError(f"Line {line}: mismatched boundary label")
            if row["recovery"] and row["state"] != "visible":
                raise ValueError(f"Line {line}: recovery event must be visible")
            if not 0 <= row["chance"] <= 1:
                raise ValueError(f"Line {line}: invalid patch chance")
            if (row["chance"] > 0) != (row["state"] == "visible"):
                raise ValueError(f"Line {line}: visibility disagrees with eligible patch coverage")
            _validate_score_flags(row, line)
            rows.append(row)
    if not rows:
        raise ValueError("Prediction CSV is empty")
    return rows


def _validate_score_flags(row, line):
    pid = next(pid for pid, name in bench.PART_NAMES.items() if name == row["target"])
    actual = row["predicted_gt"]
    if actual not in (-2, -1, 0, *bench.PART_NAMES):
        raise ValueError(f"Line {line}: invalid predicted ground-truth label")
    if not row["present"] and actual != -2:
        raise ValueError(f"Line {line}: absent prediction must use label -2")
    if row["present"] and (actual == -2 or row["score"] < row["threshold"]):
        raise ValueError(f"Line {line}: inconsistent emitted prediction")
    is_visible = row["state"] == "visible"
    expected_hit = is_visible and row["present"] and actual == pid
    expected_car = is_visible and row["present"] and actual > 0 and (actual-1)//2 != (pid-1)//2
    expected_part = is_visible and row["present"] and actual > 0 and (actual-1) % 2 != (pid-1) % 2
    for field, expected in (("hit", expected_hit), ("wrong_car", expected_car), ("wrong_part", expected_part)):
        if row[field] != expected:
            raise ValueError(f"Line {line}: inconsistent {field} flag")


def _key(row):
    return row["scene"], row["tubelet"], row["target"]


def _expected_test_keys():
    # Read the frozen seed manifest only: no scenes or image labels are generated.
    return {(f"test_{condition}_{seed}", t, target)
            for condition, seeds in bench.SPLIT_SEEDS["test"].items()
            for seed in seeds for t in range(1, bench.N_STEPS)
            for target in bench.PART_NAMES.values()}


def _index_method(rows, method):
    indexed = {}
    for row in rows:
        if row["method"] != method:
            continue
        key = _key(row)
        if key in indexed:
            raise ValueError(f"Duplicate prediction key for {method}: {key}")
        indexed[key] = row
    expected = _expected_test_keys()
    if set(indexed) != expected:
        missing, extra = expected-set(indexed), set(indexed)-expected
        raise ValueError(f"{method} is not the complete frozen test split: missing={len(missing)},extra={len(extra)}")
    for row in indexed.values():
        if row["scene"] != f"test_{row['condition']}_{row['seed']}":
            raise ValueError("Scene name disagrees with seed/condition")
    for scene in {key[0] for key in expected}:
        for pid, target in bench.PART_NAMES.items():
            pending_recovery = False
            previous_car = (pid-1)//2
            for t in range(1, bench.N_STEPS):
                row = indexed[(scene, t, target)]
                if row["state"] == "absent":
                    pending_recovery = True
                expected_recovery = row["state"] == "visible" and pending_recovery
                if row["state"] == "visible":
                    pending_recovery = False
                if row["recovery"] != expected_recovery:
                    raise ValueError(f"Inconsistent recovery label for {method}, {scene}, {target}, {t}")
                expected_switch = False
                if row["state"] == "visible" and row["present"] and row["predicted_gt"] > 0:
                    predicted_car = (row["predicted_gt"]-1)//2
                    expected_switch = predicted_car != previous_car
                    previous_car = predicted_car
                if row["id_switch"] != expected_switch:
                    raise ValueError(f"Inconsistent identity switch for {method}, {scene}, {target}, {t}")
    return indexed


def _check_equal_labels(reference, candidate, method):
    if set(reference) != set(candidate):
        raise ValueError(f"Evaluation keys differ for {method}")
    for key, original in reference.items():
        for field in GROUND_TRUTH_FIELDS:
            a, b = original[field], candidate[key][field]
            equal = math.isclose(a, b, rel_tol=0., abs_tol=1e-12) if field == "chance" else a == b
            if not equal:
                raise ValueError(f"Evaluation label mismatch for {method}, key={key}, field={field}")


def _load_frozen(vjepa_csv, explicit):
    if explicit:
        path = Path(explicit)
        frozen = _load_json(path)
    elif (Path(vjepa_csv).parent.parent/"frozen_config.json").exists():
        path = Path(vjepa_csv).parent.parent/"frozen_config.json"
        frozen = _load_json(path)
    else:
        path = Path(vjepa_csv).with_name("results.json")
        report = _load_json(path)
        if report.get("split") != "test":
            raise ValueError("Sibling results.json is not a test report")
        frozen = report["frozen_config"]
    if set(frozen.get("thresholds", {})) != set(VJEPA_METHODS):
        raise ValueError("Frozen configuration must name the six expected V-JEPA/baseline methods")
    if frozen.get("selected_method") not in VJEPA_METHODS:
        raise ValueError("Development-selected method missing or invalid")
    if not frozen.get("selection_rule"):
        raise ValueError("Frozen configuration has no development selection rule")
    return frozen, path


def _paired(rows_a, rows_b, metric, resamples, seed):
    summary_a, summary_b = bench.summarize_rows(rows_a), bench.summarize_rows(rows_b)
    if not summary_a[metric]["denominator"] or not summary_b[metric]["denominator"]:
        return {"metric": metric, "difference_a_minus_b": None,
                "clip_bootstrap_95ci": None, "reason": "zero denominator",
                "unit": "paired clip", "resamples": resamples, "seed": seed}
    return bench.paired_bootstrap_difference(rows_a, rows_b, metric=metric,
                                             resamples=resamples, seed=seed)


def _rate(value):
    if value["rate"] is None:
        return "N/A"
    return f"{100*value['rate']:.1f}% ({value['numerator']}/{value['denominator']})"


def _delta(value):
    difference, ci = value["difference_a_minus_b"], value["clip_bootstrap_95ci"]
    if difference is None:
        return "N/A", "N/A"
    return f"{100*difference:+.1f} pp", f"[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}] pp" if ci else "N/A"


def _markdown(report):
    selected, sam = report["development_selected_method"], report["sam2_method"]
    lines = ["# Day4 frozen test comparison", "",
        f"The development freeze selected **`{selected}`**. Test scores do not change this choice.", "",
        "All methods use the same 18 synthetic clips and frame-zero part annotations. Later labels are scoring data only.", "",
        "| Method | Visible localization ↑ | Hidden false presence ↓ | Recovery ↑ | Identity switches ↓ |",
        "|---|---:|---:|---:|---:|"]
    for method in [*VJEPA_METHODS, sam]:
        summary = report["methods"][method]["overall"]
        label = f"`{method}`"+(" **(development-selected)**" if method == selected else "")
        lines.append(f"| {label} | {_rate(summary[METRICS[0]])} | {_rate(summary[METRICS[1]])} | {_rate(summary[METRICS[2]])} | {summary['identity_switches']} |")
    lines.extend(["", f"Paired differences below are `{sam}` minus the development-selected `{selected}`.", "",
                  "| Metric | Difference | Paired clip bootstrap 95% interval |",
                  "|---|---:|---:|"])
    pair = report["paired_comparisons"][f"{sam}_minus_{selected}"]
    for metric, label in zip(METRICS, ("Visible localization (positive favors SAM2)",
                                     "Hidden false presence (negative favors SAM2)",
                                     "Recovery (positive favors SAM2)")):
        delta, ci = _delta(pair[metric])
        lines.append(f"| {label} | {delta} | {ci} |")
    lines.extend(["", "Intervals resample whole paired clips, not frames. They describe variation among these generated clips; they do not establish real-video generalization. The JSON includes every SAM2-versus-baseline and selected-variant-versus-baseline comparison.", "",
        "SAM2 is a specialist mask tracker with sequential memory; V-JEPA supplies frozen patch features with offline attention inside independent 16-frame windows. SAM2 masks are reduced to a patch location for this table. Training objectives, temporal processing, internal resolution, and compute are not matched. SAM2 received quality-100 JPEG inputs; V-JEPA received original rendered RGB.", ""])
    if "sam2_dense_masks" in report:
        dense = report["sam2_dense_masks"]
        overall = dense["overall"]
        lines.extend(["## SAM2 dense masks (separate readout)", "",
            "Raw masks are scored on every frame after frame zero, without the calibrated patch-presence threshold. Any nonempty ground-truth part is visible, including thin slivers excluded from the patch table.", "",
            "| Dense metric | Value |", "|---|---:|"])
        for key, label in (("mean_iou_given_visible", "Mean visible part-frame IoU"),
                           ("visible_micro_pixel_precision", "Visible pixel precision"),
                           ("visible_micro_pixel_recall", "Visible pixel recall"),
                           ("all_frame_micro_pixel_precision", "Pixel precision including hidden targets"),
                           ("false_presence_given_absent", "False mask presence while hidden")):
            value = overall[key]
            rendered = f"{100*value['rate']:.1f}%" if value["rate"] is not None else "N/A"
            lines.append(f"| {label} | {rendered} |")
        lines.extend(["", f"Dense denominators: {overall['visible_part_frames']} visible and {overall['absent_part_frames']} absent part-frames across {overall['clips']} clips. These differ from the patch benchmark denominators.", ""])
    lines.append("No model inference, threshold fitting, or configuration selection is performed by this comparison script.")
    return "\n".join(lines)+"\n"


def compare_results(vjepa_csv, sam2_csv, out_dir, frozen_config_path=None,
                    sam2_dense_path=None, resamples=2000, seed=914):
    if resamples <= 0:
        raise ValueError("Resamples must be positive")
    vjepa_rows, sam_rows = read_prediction_rows(vjepa_csv), read_prediction_rows(sam2_csv)
    if {row["method"] for row in vjepa_rows} != set(VJEPA_METHODS):
        raise ValueError("V-JEPA CSV must include exactly the six frozen methods")
    sam_methods = {row["method"] for row in sam_rows}
    if len(sam_methods) != 1:
        raise ValueError("SAM2 CSV must contain exactly one method")
    sam_method = next(iter(sam_methods))
    if sam_method != "sam2_1_tiny":
        raise ValueError(f"Unexpected specialist method label {sam_method!r}")
    frozen, frozen_path = _load_frozen(vjepa_csv, frozen_config_path)
    selected = frozen["selected_method"]
    indexed = {method: _index_method(vjepa_rows, method) for method in VJEPA_METHODS}
    indexed[sam_method] = _index_method(sam_rows, sam_method)
    reference = indexed[selected]
    for method, index in indexed.items():
        _check_equal_labels(reference, index, method)
        thresholds = {row["threshold"] for row in index.values()}
        if len(thresholds) != 1:
            raise ValueError(f"Multiple evaluation thresholds for {method}")
        if method in VJEPA_METHODS and not math.isclose(next(iter(thresholds)), frozen["thresholds"][method], rel_tol=0., abs_tol=1e-12):
            raise ValueError(f"CSV threshold disagrees with development freeze for {method}")
    ordered = {method: [indexed[method][key] for key in sorted(reference)] for method in indexed}
    methods = {method: {"overall": bench.summarize_with_ci(rows, resamples=resamples, seed=seed),
                        "by_condition": {condition: bench.summarize_rows([r for r in rows if r["condition"] == condition])
                                         for condition in bench.CONDITIONS}}
               for method, rows in ordered.items()}
    pairs = [(sam_method, method) for method in VJEPA_METHODS]
    pairs += [(selected, method) for method in VJEPA_METHODS if method != selected]
    comparisons = {f"{a}_minus_{b}": {metric: _paired(ordered[a], ordered[b], metric, resamples, seed)
                                      for metric in METRICS} for a, b in pairs}
    report = {"protocol": bench.PROTOCOL_VERSION, "split": "test", "validated_clip_count": 18,
        "validated_rows_per_method": len(reference), "sam2_method": sam_method,
        "development_selected_method": selected, "selection_rule": frozen["selection_rule"],
        "inputs": {"vjepa_csv": {"path": str(Path(vjepa_csv).resolve()), "sha256": _sha256(vjepa_csv)},
                   "sam2_csv": {"path": str(Path(sam2_csv).resolve()), "sha256": _sha256(sam2_csv)},
                   "frozen_config": {"path": str(frozen_path.resolve()), "sha256": _sha256(frozen_path)}},
        "validation": {"complete_test_split": True, "all_six_vjepa_methods": True,
                       "identical_evaluation_keys_and_labels": True, "frozen_vjepa_thresholds": True,
                       "csv_boolean_and_outcome_consistency": True},
        "bootstrap": {"unit": "paired clip", "resamples": resamples, "seed": seed,
                      "difference_direction": "A minus B; negative false-presence difference favors A"},
        "methods": methods, "paired_comparisons": comparisons}
    if sam2_dense_path:
        dense = _load_json(sam2_dense_path)
        if "dense_masks" in dense:
            dense = dense["dense_masks"]
        expected_names = {key[0] for key in reference}
        if set(dense.get("per_clip", {})) != expected_names:
            raise ValueError("Dense SAM2 report does not contain the same test clips")
        if dense["overall"].get("clips") != 18 or dense["overall"].get("scored_part_frames") != 18*(bench.N_FRAMES-1)*len(bench.PART_NAMES):
            raise ValueError("Dense SAM2 report has unexpected frame denominators")
        report["sam2_dense_masks"] = dense
        report["inputs"]["sam2_dense"] = {"path": str(Path(sam2_dense_path).resolve()), "sha256": _sha256(sam2_dense_path)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (out/"comparison.md").write_text(_markdown(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vjepa-csv", required=True)
    parser.add_argument("--sam2-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frozen-config")
    parser.add_argument("--sam2-dense")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=914)
    args = parser.parse_args()
    result = compare_results(args.vjepa_csv, args.sam2_csv, args.out,
        frozen_config_path=args.frozen_config, sam2_dense_path=args.sam2_dense,
        resamples=args.resamples, seed=args.seed)
    print(json.dumps({"selected_method": result["development_selected_method"],
                      "clips": result["validated_clip_count"],
                      "outputs": [str(Path(args.out)/"comparison.json"), str(Path(args.out)/"comparison.md")]}, indent=2))
