"""Supplementary fixed presence gating for SAM2 masks before downstream edits.

Policy: use the already frozen SAM2 patch decision, eligible AND score >= its
existing calibration threshold, on both frames of each pair. Preserve frame 0.
There is no new threshold fitting, model inference, or alternative-policy search.
This is a postprocessing comparison, not a replacement for the primary test.
The score is mask patch coverage, not a calibrated probability; the two-frame
readout is not a causal streaming claim.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import benchmark as bench
import sam2_benchmark as sam_bench

POLICY_ID = "sam2_frozen_patch_presence_to_dense_masks_v1"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_digest():
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(sam_bench.source_digest().encode())
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _frozen_config(sam2_run):
    path = Path(sam2_run)/"sam2_frozen_config.json"
    frozen = _read(path)
    if frozen["source_digest"] != sam_bench.source_digest():
        raise ValueError("SAM2 source does not match its calibration freeze")
    if frozen["readout_version"] != sam_bench.READOUT_VERSION:
        raise ValueError("SAM2 readout differs from its calibration freeze")
    if frozen["method"] != sam_bench.METHOD or not np.isfinite(frozen["threshold"]):
        raise ValueError("Unexpected model label or threshold")
    return frozen, path


def freeze_policy(sam2_run, policy_path):
    """Record the exact derived policy before evaluating its effect.

    Reads calibration evidence and frozen configuration only. Existing policies
    cannot be silently overwritten after code or threshold changes.
    """
    frozen, frozen_path = _frozen_config(sam2_run)
    calibration_path = Path(sam2_run)/"calibration"/"results.json"
    calibration = _read(calibration_path)
    if calibration["split"] != "calibration":
        raise ValueError("Policy rationale must come from calibration results")
    raw_absence = calibration["dense_masks"]["overall"]["false_presence_given_absent"]
    patch_absence = calibration["methods"][sam_bench.METHOD]["overall"]["false_presence_given_absent"]
    exact = {"policy_id": POLICY_ID, "source_digest": _source_digest(),
        "script_sha256": _hash(__file__), "sam2_source_digest": sam_bench.source_digest(),
        "sam2_frozen_config_sha256": _hash(frozen_path),
        "sam2_checkpoint_sha256": frozen["checkpoint_sha256"],
        "readout_version": sam_bench.READOUT_VERSION, "threshold": frozen["threshold"],
        "decision": "masks_to_predictions(raw_masks).eligible AND scores >= frozen_threshold",
        "expansion": "Repeat each part decision onto both source frames of its two-frame pair.",
        "initialization": "Preserve every raw mask in frame 0 regardless of its pair decision.",
        "new_parameter_selection": "none; no new threshold or alternative gate",
        "primary_test_status": "supplementary postprocessing comparison; primary raw-mask and patch results remain unchanged",
        "probability_note": "Score is maximum predicted patch coverage, not a calibrated probability.",
        "temporal_note": "Two-frame readout; not claimed to be causal streaming.",
        "calibration_evidence": {"raw_hidden_mask_reports": raw_absence,
            "patch_hidden_presence_reports": patch_absence,
            "denominator_note": "Raw metrics count part-frames; patch metrics count unambiguous part-pairs. Their counts are not directly comparable.",
            "results_sha256": _hash(calibration_path)}}
    path = Path(policy_path)
    if path.exists():
        existing = _read(path)
        if any(existing.get(key) != value for key, value in exact.items()):
            raise ValueError("Existing derived-policy freeze differs; do not overwrite it")
        return existing
    policy = {**exact, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_provenance": "Proposed during the SAM2 test execution, before the controller viewed its held-out masks or scores; prompted by the calibration raw-mask versus patch-presence discrepancy.",
        "provenance_basis": "Controller's contemporaneous task statement plus recorded calibration result hashes."}
    _write(path, policy)
    return policy


def apply_presence_gate(raw_masks, threshold):
    """Pure prediction-only control layer; no ground-truth input is accepted."""
    pred = sam_bench.masks_to_predictions(raw_masks)
    pair_gate = pred.eligible & (pred.scores >= threshold)
    frame_gate = np.repeat(pair_gate, bench.TUBELET, axis=0)
    frame_gate[0] = True
    gated = raw_masks & frame_gate[:, :, None, None]
    if not np.array_equal(gated[0], raw_masks[0]):
        raise AssertionError("Frame-zero initialization changed")
    return gated, pred, frame_gate


def _summary(raw_rows, gated_rows, paired_rows):
    removed = [r for r in paired_rows if r["raw_present"] and not r["gated_present"]]
    visible_removed = [r for r in removed if r["visible"]]
    small = [r for r in paired_rows if r["visible"] and not r["pair_has_eligible_gt_patch"]]
    small_removed = [r for r in small if r["raw_present"] and not r["gated_present"]]
    raw_summary, gated_summary = sam_bench.summarize_dense_rows(raw_rows), sam_bench.summarize_dense_rows(gated_rows)
    def difference(metric):
        a, b = raw_summary[metric]["rate"], gated_summary[metric]["rate"]
        return b-a if a is not None and b is not None else None
    return {"raw": raw_summary, "gated": gated_summary,
        "gated_minus_raw": {metric: difference(metric) for metric in (
            "mean_iou_given_visible", "visible_micro_pixel_precision", "visible_micro_pixel_recall",
            "all_frame_micro_pixel_precision", "false_presence_given_absent")},
        "removed": {"nonempty_part_frame_masks": len(removed),
            "visible_part_frame_masks": len(visible_removed),
            "hidden_part_frame_masks": sum(not r["visible"] for r in removed),
            "pixels": sum(r["raw_pixels"]-r["gated_pixels"] for r in paired_rows),
            "true_positive_pixels": sum(r["raw_intersection"]-r["gated_intersection"] for r in paired_rows),
            "false_positive_pixels": sum((r["raw_pixels"]-r["raw_intersection"])-(r["gated_pixels"]-r["gated_intersection"]) for r in paired_rows)},
        "visible_without_eligible_gt_patch_in_pair": {
            "part_frames": len(small), "nonempty_predicted_masks_removed": len(small_removed),
            "raw_mean_iou": float(np.mean([r["raw_iou"] for r in small])) if small else None,
            "gated_mean_iou": float(np.mean([r["gated_iou"] for r in small])) if small else None,
            "interpretation": "Includes thin/sliver or moving visible parts without a qualifying 16x16 patch across the pair; not only small physical parts."}}


def evaluate_gate(sam2_run, policy_path, out_dir, split="calibration"):
    """Evaluate one already-frozen gate against raw masks; never fit a policy."""
    if split not in ("calibration", "test"):
        raise ValueError("Split must be calibration or test")
    policy = _read(policy_path)
    frozen, frozen_path = _frozen_config(sam2_run)
    expected = {"policy_id": POLICY_ID, "source_digest": _source_digest(),
        "sam2_source_digest": sam_bench.source_digest(),
        "sam2_frozen_config_sha256": _hash(frozen_path),
        "sam2_checkpoint_sha256": frozen["checkpoint_sha256"],
        "threshold": frozen["threshold"], "readout_version": sam_bench.READOUT_VERSION}
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Derived-policy freeze, source, or SAM2 configuration changed")
    out = Path(out_dir)
    (out/"decisions").mkdir(parents=True, exist_ok=True)
    raw_rows, gated_rows, paired = [], [], []
    scenes = bench.generate_suite(split=split)
    clip_sources = []
    for scene in scenes:
        folder = Path(sam2_run)/"clips"/scene.name
        manifest = _read(folder/"clip_manifest.json")
        mask_path = folder/"sam2_masks.npz"
        required = {"scene": scene.name, "source_digest": policy["sam2_source_digest"],
            "checkpoint_sha256": policy["sam2_checkpoint_sha256"],
            "original_rgb_sha256": hashlib.sha256(scene.frames.tobytes()).hexdigest(),
            "initial_mask_sha256": hashlib.sha256(scene.masks[0].tobytes()).hexdigest(),
            "predicted_masks_sha256": _hash(mask_path)}
        if any(manifest.get(key) != value for key, value in required.items()):
            raise ValueError(f"Cached SAM2 provenance does not match for {scene.name}")
        with np.load(mask_path) as saved:
            raw_masks, ids = saved["masks"], saved["ids"].tolist()
        if ids != list(bench.PART_NAMES):
            raise ValueError("Unexpected mask identity order")
        gated_masks, pred, decisions = apply_presence_gate(raw_masks, policy["threshold"])
        with np.load(folder/"predictions.npz") as saved_pred:
            for key in ("scores", "cells", "eligible"):
                if not np.array_equal(getattr(pred, key), saved_pred[key]):
                    raise ValueError(f"Recomputed patch readout differs for {scene.name}: {key}")
        np.savez_compressed(out/"decisions"/f"{scene.name}.npz", ids=ids,
            frame_gate=decisions, pair_scores=pred.scores, pair_eligible=pred.eligible)
        # Ground truth enters only after the prediction-only gate has completed.
        raw = sam_bench.dense_mask_rows(scene, raw_masks)
        gated = sam_bench.dense_mask_rows(scene, gated_masks)
        labels = bench.patch_labels(scene.masks)
        for a, b in zip(raw, gated):
            if (a["frame"], a["part_id"]) != (b["frame"], b["part_id"]):
                raise AssertionError("Dense rows are misaligned")
            pair_has_patch = bool(np.any(labels[a["frame"]//bench.TUBELET] == a["part_id"]))
            paired.append({"scene": scene.name, "seed": scene.seed, "condition": scene.condition,
                "frame": a["frame"], "target": a["target"], "part_id": a["part_id"],
                "visible": a["visible"], "gt_pixels": a["gt_pixels"],
                "pair_has_eligible_gt_patch": pair_has_patch,
                "gate": bool(decisions[a["frame"], a["part_id"]-1]),
                "raw_present": a["predicted_present"], "gated_present": b["predicted_present"],
                "raw_pixels": a["predicted_pixels"], "gated_pixels": b["predicted_pixels"],
                "raw_intersection": a["intersection_pixels"], "gated_intersection": b["intersection_pixels"],
                "raw_iou": a["iou"], "gated_iou": b["iou"]})
        raw_rows.extend(raw)
        gated_rows.extend(gated)
        clip_sources.append({"scene": scene.name, "raw_masks_sha256": _hash(mask_path),
                             "frame0_preserved": bool(np.array_equal(raw_masks[0], gated_masks[0]))})
        del raw_masks, gated_masks
    def subset_summary(field, value):
        return _summary([r for r in raw_rows if r[field] == value],
                        [r for r in gated_rows if r[field] == value],
                        [r for r in paired if r[field] == value])
    report = {"experiment": "Supplementary frozen presence gate before mask editing",
        "split": split, "policy": policy, "policy_sha256": _hash(policy_path),
        "scoring": "Dense masks on every frame after frame0; same ground truth for raw and gated outputs.",
        "evaluation_status": "Postprocessing comparison, not a replacement for primary results; no threshold tuning.",
        "overall": _summary(raw_rows, gated_rows, paired),
        "by_condition": {condition: subset_summary("condition", condition) for condition in bench.CONDITIONS},
        "per_clip": {scene.name: subset_summary("scene", scene.name) for scene in scenes},
        "by_part": {target: subset_summary("target", target) for target in bench.PART_NAMES.values()},
        "sources": clip_sources,
        "storage": "Raw masks are unchanged. Saved boolean decisions reconstruct gated masks with raw_mask AND frame_gate; frame0 gate is forced true."}
    _write(out/"presence_gate_results.json", report)
    with (out/"paired_dense_rows.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary = report["overall"]
    lines = ["# Supplementary frozen mask-presence gate", "",
        f"Split: **{split}**. Threshold: **{policy['threshold']}**, copied unchanged from the original SAM2 calibration freeze.", "",
        "A part's mask is retained in both frames of a pair only when the existing patch readout is eligible and its score reaches that threshold. Frame 0 is preserved. Scores measure mask coverage; they are not calibrated probabilities. This is a two-frame postprocessing rule, not a causal streaming claim.", "",
        "| Dense metric | Raw masks | Gated masks |", "|---|---:|---:|"]
    for metric, title in (("mean_iou_given_visible", "Mean visible IoU"),
                          ("visible_micro_pixel_precision", "Visible pixel precision"),
                          ("visible_micro_pixel_recall", "Visible pixel recall"),
                          ("all_frame_micro_pixel_precision", "Precision including hidden targets"),
                          ("false_presence_given_absent", "False mask presence while hidden")):
        def percent(value):
            return f"{100*value:.2f}%" if value is not None else "N/A"
        lines.append(f"| {title} | {percent(summary['raw'][metric]['rate'])} | {percent(summary['gated'][metric]['rate'])} |")
    removed = summary["removed"]
    lines.extend(["", f"Removed {removed['nonempty_part_frame_masks']} nonempty predicted part-frame masks: {removed['visible_part_frame_masks']} while the true part was visible, and {removed['hidden_part_frame_masks']} while hidden.", "",
        "Small/sliver diagnostic: "+json.dumps(summary["visible_without_eligible_gt_patch_in_pair"]), "",
        "This comparison does not replace the primary raw-mask or patch metrics. The exact policy was proposed and frozen during test execution before its scores were viewed. No threshold was adjusted after this comparison."])
    (out/"presence_gate_results.md").write_text("\n".join(lines)+"\n")
    return report


def self_test():
    masks = np.zeros((4, 4, bench.SIZE, bench.SIZE), bool)
    masks[:, :, 64:96, 64:96] = True
    masks[1, 0] = False
    gated, pred, decisions = apply_presence_gate(masks, .94)
    assert np.array_equal(gated[0], masks[0])
    assert decisions[0].all() and not decisions[1, 0]
    assert not pred.eligible[0, 0]
    assert np.array_equal(gated[2:], masks[2:])
    assert not np.any(gated & ~masks)
    return {"frame0_preserved": True, "exact_pair_gate": True,
            "two_frame_expansion": True, "never_adds_predicted_pixels": True,
            "real_model_results": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["freeze", "calibration", "test"])
    parser.add_argument("--sam2-run")
    parser.add_argument("--policy")
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.stage == "freeze":
        if not args.sam2_run or not args.policy:
            parser.error("Freeze requires --sam2-run and --policy")
        print(json.dumps(freeze_policy(args.sam2_run, args.policy), indent=2))
    elif args.stage in ("calibration", "test"):
        if not args.sam2_run or not args.policy or not args.out:
            parser.error("Evaluation requires --sam2-run, --policy, and --out")
        result = evaluate_gate(args.sam2_run, args.policy, args.out, args.stage)
        print(json.dumps(result["overall"], indent=2))
    else:
        parser.error("Choose --stage or --self-test")
