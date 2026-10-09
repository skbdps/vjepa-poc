"""One-shot, pretest check of the frozen Day 8 probe on genuine DEV encodings.

No encoder is loaded, no feature is edited, no model is updated, and no train or
test feature cache is opened. Run after the complete training freeze exists and
before test extraction. This is a development diagnostic, not held-out evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def run(features, training_run, out=None, threads=2, code=None):
    if not 1 <= threads <= 4:
        raise ValueError("Use between 1 and 4 CPU threads")
    here = Path(__file__).resolve().parent
    code = Path(code) if code else (here if (here / "data.py").exists() else here.parent / "vjepa-run" / "experiments" / "day8")
    sys.path.insert(0, str(code))
    import torch
    import data
    import evaluate
    import probe
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    features, training_run = Path(features), Path(training_run)
    if features.name == "cache" and not (features / "manifest.json").is_file():
        features = features.parent
    out = Path(out) if out else training_run / "dev_readout_check.json"
    if out.exists():
        raise FileExistsError("One-shot diagnostic already exists: " + str(out))
    freeze_path = training_run / "checkpoint_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Wait for complete training/checkpoint_freeze.json before running")
    frozen = json.loads(freeze_path.read_text())
    if frozen.get("test_accessed") is not False:
        raise ValueError("Not a completed pretest training freeze")
    expected = {(arm, seed) for arm in evaluate.ARMS for seed in evaluate.SEEDS} | {("frozen_readout", 1800)}
    identities = [(record["arm"], record["seed"]) for record in frozen["models"]]
    if set(identities) != expected or len(identities) != len(expected):
        raise ValueError("Freeze does not contain all six edit models plus one probe")
    for name, digest in frozen["source_hashes"].items():
        if evaluate.sha256(code / name) != digest:
            raise ValueError("Training source changed after freeze: " + name)
    for record in frozen["models"]:
        if evaluate.sha256(training_run / record["path"]) != record["sha256"]:
            raise ValueError("Frozen model checksum differs: " + record["path"])
    binding_path = training_run / "feature_binding.json"
    if evaluate.sha256(binding_path) != frozen["feature_binding_sha256"]:
        raise ValueError("Feature binding changed after freeze")
    binding = json.loads(binding_path.read_text())
    manifest_path = features / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_freeze_sha256") or any(record["spec"]["split"] == "test" for record in manifest["clips"]):
        raise ValueError("Pretest diagnostic refused after test extraction started")
    for name, digest in manifest["source_hashes"].items():
        if evaluate.sha256(code / name) != digest:
            raise ValueError("Extraction source changed: " + name)
    indexed = {record["spec"]["name"]: record for record in manifest["clips"]}
    if len(indexed) != len(manifest["clips"]):
        raise ValueError("Duplicate feature-cache scene names")
    if any(indexed.get(record["spec"]["name"]) != record for record in binding["clips"]):
        raise ValueError("Feature manifest changed after training freeze")
    for key in ("model", "upstream", "encoder_context", "oracle_budget", "extraction_device",
                "inference_precision", "cache_precision", "weights"):
        if binding[key] != manifest[key]:
            raise ValueError("Feature extraction metadata changed: " + key)
    readout_record = next(record for record in frozen["models"] if record["arm"] == "frozen_readout")
    readout, history = probe.load_probe(training_run / readout_record["path"], "cpu")
    if history["selected_epoch"] != readout_record["epoch"]:
        raise ValueError("Probe selected epoch disagrees with freeze")
    rows = []
    seen_caches = []
    started = time.perf_counter()
    for spec in data.scene_specs("dev"):
        record = indexed.get(spec["name"])
        if record is None or record["spec"] != spec:
            raise ValueError("Missing or mismatched development cache: " + spec["name"])
        cached = evaluate.load_cache(features, record)
        source_prediction = probe.predict_probe(readout, cached["source"], device="cpu")
        target_prediction = probe.predict_probe(readout, cached["target"], device="cpu")
        regions, _ = evaluate.fixed_regions(cached["source_frac"], cached["target_frac"],
                                             cached["distractor_frac"], spec["dx"] // data.PATCH)
        # Score the genuine source against its OWN ground truth. Reusing the
        # target labels here would incorrectly turn a readout check into a
        # no-op-edit error measurement.
        source_truth = {**cached, "target_frac": cached["source_frac"], "rgb_target": cached["rgb_source"]}
        source_regions, _ = evaluate.fixed_regions(cached["source_frac"], cached["source_frac"],
                                                    cached["distractor_frac"], 0)
        for side, prediction, truth, selected_regions in (
                ("genuine_source", source_prediction, source_truth, source_regions),
                ("genuine_target", target_prediction, cached, regions)):
            per_tubelet = evaluate.semantic_metrics(prediction, source_prediction, truth, selected_regions)
            aggregated = evaluate.aggregate_semantic(per_tubelet)
            rows.append({"scene": spec["name"], "side": side, "tubelets": len(per_tubelet), **aggregated})
            print("DEV_READOUT_SCENE", spec["name"], side,
                  "centroid_px", round(aggregated["selected_centroid_error_px"], 4),
                  "identity", aggregated["appearance_identity_accuracy"],
                  "iou", round(aggregated["selected_iou"], 4), flush=True)
        seen_caches.append({"scene": spec["name"], "path": record["path"], "sha256": record["sha256"]})
        del cached, source_truth, source_prediction, target_prediction
    summaries = {}
    for side in ("genuine_source", "genuine_target"):
        included = [row for row in rows if row["side"] == side]
        eligible = sum(row["appearance_identity_eligible"] for row in included)
        correct = sum(row["appearance_identity_correct"] for row in included)
        tubelets = sum(row["tubelets"] for row in included)
        summary = {"scenes": len(included), "tubelets": tubelets,
                   "identity_eligible_tubelets": eligible,
                   "identity_skipped_tubelets": sum(row["appearance_identity_skipped"] for row in included),
                   "identity_correct_tubelets": correct,
                   "identity_accuracy_equal_scene_mean": _mean(row["appearance_identity_accuracy"] for row in included),
                   "identity_accuracy_pooled_eligible": correct / eligible if eligible else None}
        for label in ("selected", "distractor"):
            for metric in ("centroid_error_px", "iou", "rgb_core_mse", "temporal_velocity_error_px"):
                key = label + "_" + metric
                summary[key] = _mean(row[key] for row in included)
            summary[label + "_missing_tubelets"] = sum(row[label + "_missing_tubelets"] for row in included)
            summary[label + "_missing_fraction"] = summary[label + "_missing_tubelets"] / tubelets
        summaries[side] = summary
    target = summaries["genuine_target"]
    target_identity = target["identity_accuracy_equal_scene_mean"]
    report = {"scope": "Pretest development-only genuine-feature readout diagnostic; not test results or edited-video evidence",
              "encoder_executed": False, "test_data_opened": False, "model_updated": False,
              "probe_input": "JEPA tokens only; masks used by scorer",
              "threads": threads, "torch": str(torch.__version__), "elapsed_seconds": time.perf_counter() - started,
              "checkpoint_freeze_sha256": evaluate.sha256(freeze_path),
              "probe_sha256": readout_record["sha256"], "probe_selected_epoch": history["selected_epoch"],
              "feature_manifest_sha256": evaluate.sha256(manifest_path),
              "diagnostic_source_sha256": evaluate.sha256(Path(__file__)),
              "evaluator_source_sha256": evaluate.sha256(code / "evaluate.py"),
              "caches_opened": seen_caches, "summaries": summaries,
              "development_gate_preview": {
                  "centroid_pass": target["selected_centroid_error_px"] < data.PATCH,
                  "centroid_threshold_px": data.PATCH,
                  "coarse_identity_pass": target_identity is not None and target_identity >= evaluate.IDENTITY_READOUT_GATE,
                  "coarse_identity_threshold": evaluate.IDENTITY_READOUT_GATE,
                  "note": "Uses the existing genuine-target gates on DEV only; test gates are still unevaluated"},
              "per_scene": rows}
    evaluate.write_json(out, report)
    print("DEV_READOUT_COMPLETE", str(out), json.dumps(report["development_gate_preview"]), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, dest="training_run")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--code", type=Path)
    args = parser.parse_args()
    run(args.features, args.training_run, args.out, args.threads, args.code)
