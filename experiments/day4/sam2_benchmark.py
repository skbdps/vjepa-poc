"""Fixed SAM2.1-tiny specialist comparator on the existing Day4 video suite.

This comparator was added after V-JEPA development results and the first real-
video demo, before V-JEPA held-out results were viewed. SAM2 parameters and the
mask-to-point readout are fixed before SAM2 test. Only the final patch-presence
threshold is chosen on the original six calibration clips, then frozen before
any SAM2 test run. Later ground-truth masks enter scoring only. Dense-mask metrics
are also reported separately, without applying the patch-presence threshold.

SAM2 predicts dense masks while V-JEPA predicts patch locations. We reduce masks
to the same grid for the common localization task; this does not make model
training objectives, temporal receptive fields, or computational budgets equal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import benchmark as bench
import real_video

METHOD = "sam2_1_tiny"
READOUT_VERSION = "mean-pair-coverage_centroid-tiebreak_v1"
JPEG_CONFIG = {"format": "JPEG", "quality": 100, "subsampling": 0}


def source_digest():
    h = hashlib.sha256()
    for name in ("sam2_benchmark.py", "benchmark.py", "real_video.py"):
        h.update(name.encode())
        h.update((HERE/name).read_bytes())
    return h.hexdigest()


def masks_to_predictions(masks):
    """Reduce predicted masks [frames,parts,384,384] to benchmark locations.

    The location maximizes mean predicted coverage of a 16x16 patch over its
    two frames. Exact ties choose the patch nearest the predicted mask centroid.
    Presence confidence is that maximum coverage in [0,1]; it is not a calibrated
    SAM2 probability. Emission additionally requires a nonempty predicted mask
    in each frame. This function never receives ground-truth masks.
    """
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[1:] != (4, bench.SIZE, bench.SIZE) or len(masks) % 2:
        raise ValueError(f"Expected [even T,4,384,384], received {masks.shape}")
    if masks.dtype != bool:
        raise ValueError("Mask readout expects thresholded boolean predictions")
    steps = len(masks)//2
    pair = masks.reshape(steps, 2, 4, bench.SIZE, bench.SIZE)
    coverage = pair.reshape(steps, 2, 4, bench.GRID, bench.PATCH,
                            bench.GRID, bench.PATCH).mean(axis=(1, 4, 6))
    scores = coverage.max(axis=(2, 3)).astype(np.float32)
    eligible = pair.any(axis=(3, 4)).all(axis=1)
    cells = np.zeros((steps, 4, 2), np.int16)
    for t in range(steps):
        for ki in range(4):
            if not pair[t, :, ki].any():
                continue
            # Union centroid does not overweight pixels visible in both frames.
            yy, xx = np.where(pair[t, :, ki].any(axis=0))
            center = np.array([yy.mean(), xx.mean()])
            candidates = np.argwhere(coverage[t, ki] == scores[t, ki])
            distance = ((candidates*bench.PATCH+bench.PATCH/2-center)**2).sum(axis=1)
            cells[t, ki] = candidates[distance.argmin()]
    pred = bench.Predictions(scores, cells, eligible)
    bench.validate_predictions(pred, steps)
    return pred


def _save_jpegs(frames, folder):
    """SAM2's public video loader accepts JPEG frame directories.

    Use actual high-quality JPEGs, never lossless files with misleading suffixes.
    Record round-trip pixel error because V-JEPA saw the original rendered RGB.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    expected = {f"{i:05d}.jpg" for i in range(len(frames))}
    if {p.name for p in folder.glob("*.jpg")} - expected:
        raise ValueError("Frame folder contains an unexpected sequence")
    absolute_error = 0
    maximum_error = 0
    nonzero = 0
    total = 0
    encoded_hash = hashlib.sha256()
    for index, frame in enumerate(frames):
        path = folder/f"{index:05d}.jpg"
        Image.fromarray(frame).save(path, **JPEG_CONFIG)
        decoded = np.asarray(Image.open(path).convert("RGB"))
        error = np.abs(decoded.astype(np.int16)-np.asarray(frame, np.int16))
        absolute_error += int(error.sum())
        maximum_error = max(maximum_error, int(error.max()))
        nonzero += int(np.count_nonzero(error))
        total += error.size
        encoded_hash.update(path.read_bytes())
    return {"jpeg": JPEG_CONFIG, "encoded_files_sha256": encoded_hash.hexdigest(),
            "rgb_mean_absolute_error_0_255": absolute_error/total,
            "rgb_max_absolute_error_0_255": maximum_error,
            "rgb_changed_channel_fraction": nonzero/total,
            "comparison_note": "SAM2 receives quality100,4:4:4 JPEGs; V-JEPA saw original lossless rendered RGB."}


def _predict_scene(scene, folder, checkpoint, predictor, frozen_source):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    initial = {pid: (scene.masks[0] == pid).copy() for pid in bench.PART_NAMES}
    if any(not m.any() for m in initial.values()):
        raise ValueError("All four parts must be visible in frame zero")
    identity = {"scene": scene.name,
        "original_rgb_sha256": hashlib.sha256(scene.frames.tobytes()).hexdigest(),
        "initial_mask_sha256": hashlib.sha256(scene.masks[0].tobytes()).hexdigest(),
        "source_digest": frozen_source, "model_revision": real_video.SAM2_REVISION,
        "checkpoint_sha256": real_video.sha256_file(checkpoint),
        "readout_version": READOUT_VERSION, "jpeg": JPEG_CONFIG}
    manifest_path = folder/"clip_manifest.json"
    mask_path = folder/"sam2_masks.npz"
    if manifest_path.exists() and mask_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"Stale cache for {scene.name}; use a new output folder")
        if manifest["predicted_masks_sha256"] != real_video.sha256_file(mask_path):
            raise RuntimeError("Cached mask hash mismatch")
        with np.load(mask_path) as cache:
            masks, ids = cache["masks"], cache["ids"].tolist()
    else:
        jpeg = _save_jpegs(scene.frames, folder/"frames")
        # Information barrier: only first-frame masks reach SAM2.
        masks, ids, timing = real_video.run_sam2(folder/"frames", initial, folder,
                                               checkpoint=checkpoint, predictor=predictor)
        manifest = {**identity, "input": jpeg, "timing": timing,
                    "predicted_masks_sha256": real_video.sha256_file(mask_path),
                    "prompt_frames": [0]}
        real_video.write_json(manifest_path, manifest)
    if ids != list(bench.PART_NAMES):
        raise ValueError("Unexpected SAM2 part ordering")
    pred = masks_to_predictions(masks)
    np.savez_compressed(folder/"predictions.npz", scores=pred.scores,
                        cells=pred.cells, eligible=pred.eligible)
    del masks
    return pred, manifest


def _summary(rows):
    return {"overall": bench.summarize_rows(rows),
            "by_condition": {condition: bench.summarize_rows([r for r in rows if r["condition"] == condition])
                             for condition in sorted({r["condition"] for r in rows})},
            "by_window": {str(window): bench.summarize_rows([r for r in rows if r["window"] == window])
                          for window in sorted({r["window"] for r in rows})},
            "window_boundaries": bench.summarize_rows([r for r in rows if r["boundary"]]),
            "uncertainty": bench.summarize_with_ci(rows)}


def dense_mask_rows(scene, predicted_masks):
    """Post-inference scoring only; raw SAM2 masks, excluding source frame 0.

    Every nonempty ground-truth part mask is visible, including thin slivers that
    the patch benchmark excludes. No calibrated patch threshold gates masks.
    Absent targets have no IoU value and are evaluated separately for hallucinated
    mask presence. Empty predictions on visible targets have IoU and recall zero.
    """
    predicted_masks = np.asarray(predicted_masks)
    expected = (len(scene.frames), len(bench.PART_NAMES), bench.SIZE, bench.SIZE)
    if predicted_masks.shape != expected or predicted_masks.dtype != bool:
        raise ValueError(f"Expected boolean predicted masks {expected}")
    if np.asarray(scene.masks).shape != (len(scene.frames), bench.SIZE, bench.SIZE):
        raise ValueError("Unexpected scoring ground-truth shape")
    rows = []
    for ki, pid in enumerate(bench.PART_NAMES):
        for frame in range(1, len(scene.frames)):
            pred = predicted_masks[frame, ki]
            truth = scene.masks[frame] == pid
            intersection = int(np.count_nonzero(pred & truth))
            pred_pixels = int(np.count_nonzero(pred))
            gt_pixels = int(np.count_nonzero(truth))
            union = pred_pixels+gt_pixels-intersection
            rows.append({"scene": scene.name, "seed": scene.seed,
                "condition": scene.condition, "method": METHOD,
                "frame": frame, "part_id": pid, "target": bench.PART_NAMES[pid],
                "visible": gt_pixels > 0, "predicted_present": pred_pixels > 0,
                "gt_pixels": gt_pixels, "predicted_pixels": pred_pixels,
                "intersection_pixels": intersection, "union_pixels": union,
                "false_positive_pixels": pred_pixels-intersection,
                "false_negative_pixels": gt_pixels-intersection,
                "iou": intersection/union if gt_pixels else None,
                "pixel_precision": intersection/pred_pixels if pred_pixels else None,
                "pixel_recall": intersection/gt_pixels if gt_pixels else None})
    return rows


def summarize_dense_rows(rows):
    visible = [r for r in rows if r["visible"]]
    absent = [r for r in rows if not r["visible"]]
    def ratio(numerator, denominator):
        return {"numerator": numerator, "denominator": denominator,
                "rate": numerator/denominator if denominator else None}
    tp_visible = sum(r["intersection_pixels"] for r in visible)
    pred_visible = sum(r["predicted_pixels"] for r in visible)
    gt_visible = sum(r["gt_pixels"] for r in visible)
    pred_all = sum(r["predicted_pixels"] for r in rows)
    iou_sum = sum(r["iou"] for r in visible)
    return {"mean_iou_given_visible": ratio(iou_sum, len(visible)),
            "visible_micro_pixel_precision": ratio(tp_visible, pred_visible),
            "visible_micro_pixel_recall": ratio(tp_visible, gt_visible),
            "all_frame_micro_pixel_precision": ratio(tp_visible, pred_all),
            "false_presence_given_absent": ratio(sum(r["predicted_present"] for r in absent), len(absent)),
            "absent_predicted_pixels": sum(r["predicted_pixels"] for r in absent),
            "visible_part_frames": len(visible), "absent_part_frames": len(absent),
            "scored_part_frames": len(rows), "clips": len({r["scene"] for r in rows})}


def _dense_report(rows):
    per_clip = {name: summarize_dense_rows([r for r in rows if r["scene"] == name])
                for name in sorted({r["scene"] for r in rows})}
    return {"readout": "raw SAM2 logits>0 masks; no calibrated patch threshold applied",
            "unit": "part-frame, every frame after frame0; any nonempty GT part is visible",
            "aggregation": "IoU averaged over visible part-frames; pixel precision/recall pooled over stated pixels",
            "comparability": "Dense denominators differ from the two-frame patch benchmark, which excludes ambiguous slivers.",
            "overall": summarize_dense_rows(rows),
            "by_condition": {condition: summarize_dense_rows([r for r in rows if r["condition"] == condition])
                             for condition in sorted({r["condition"] for r in rows})},
            "by_part": {target: summarize_dense_rows([r for r in rows if r["target"] == target])
                        for target in sorted({r["target"] for r in rows})},
            "per_clip": per_clip}


def _write_report(folder, rows, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    report = {**metadata, "methods": {METHOD: _summary(rows)}}
    real_video.write_json(folder/"results.json", report)
    with (folder/"predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def run_stage(out_dir, stage="calibration", checkpoint=None, predictor=None,
              frozen_path=None, write_videos=True):
    """Run six calibration clips OR eighteen fixed test clips, with resumable masks.

    Call calibration first, persist/commit its frozen config, then call test.
    A loaded predictor can be shared across clips; parameters never change.
    """
    if stage not in ("calibration", "test"):
        raise ValueError("Stage must be calibration or test")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = Path(frozen_path) if frozen_path else out/"sam2_frozen_config.json"
    digest = source_digest()
    frozen = None
    if stage == "test":
        frozen = json.loads(frozen_path.read_text())
        if frozen["source_digest"] != digest or frozen["readout_version"] != READOUT_VERSION:
            raise RuntimeError("Code/readout changed after calibration freeze")
    if checkpoint is None:
        checkpoint = real_video.prepare_sam2()
    checkpoint = str(checkpoint)
    checkpoint_hash = real_video.sha256_file(checkpoint)
    if frozen is not None and frozen["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("Checkpoint changed after calibration freeze")
    if predictor is None:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(real_video.SAM2_CONFIG, checkpoint,
                                                device="cuda", apply_postprocessing=False)
    scenes = bench.generate_suite(split=stage)
    predictions, manifests = {}, []
    for scene in scenes:
        start = time.perf_counter()
        pred, manifest = _predict_scene(scene, out/"clips"/scene.name,
                                        checkpoint, predictor, digest)
        predictions[scene.name] = pred
        manifests.append(manifest)
        print(f"SAM2_{stage.upper()}_DONE {scene.name} {time.perf_counter()-start:.1f}s", flush=True)
    if stage == "calibration":
        threshold, calibration = bench.calibrate_threshold(scenes, predictions)
        frozen = {"source_digest": digest, "readout_version": READOUT_VERSION,
                  "method": METHOD, "threshold": threshold, "calibration": calibration,
                  "calibration_scenes": [s.name for s in scenes],
                  "checkpoint_sha256": checkpoint_hash,
                  "model_revision": real_video.SAM2_REVISION,
                  "checkpoint_url": real_video.SAM2_CHECKPOINT,
                  "model_config": real_video.SAM2_CONFIG,
                  "postprocessing": False, "precision": "float32",
                  "jpeg": JPEG_CONFIG, "model_parameter_selection": "none",
                  "threshold_selection": "calibration clips only; same presence objective as V-JEPA",
                  "comparison_status": "added after V-JEPA development results and first real-video demo; fixed before SAM2 test; V-JEPA held-out results not viewed when comparator designed"}
        real_video.write_json(frozen_path, frozen)
    threshold = frozen["threshold"]
    rows = [row for scene in scenes for row in
            bench.score_scene(scene, predictions[scene.name], threshold, METHOD)]
    # Dense evaluation is a separate post-inference pass. Future annotations are
    # never supplied to the tracker or to the frozen mask-to-patch readout.
    dense_rows = []
    for scene in scenes:
        with np.load(out/"clips"/scene.name/"sam2_masks.npz") as archive:
            dense_rows.extend(dense_mask_rows(scene, archive["masks"]))
    dense_report = _dense_report(dense_rows)
    (out/stage).mkdir(parents=True, exist_ok=True)
    with (out/stage/"dense_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dense_rows[0]))
        writer.writeheader()
        writer.writerows(dense_rows)
    real_video.write_json(out/stage/"dense_results.json", dense_report)
    report = _write_report(out/stage, rows, {"split": stage,
        "frozen_config": frozen, "benchmark": bench.benchmark_manifest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "clips": manifests, "dense_masks": dense_report,
        "comparison_limitations": [
            "Fixed specialist tracker with dense initial masks; same frame-zero masks are available to V-JEPA.",
            "SAM2 masks reduced to maximum-coverage patch; V-JEPA directly matches patch features.",
            "SAM2 is sequential with memory; V-JEPA uses offline attention within independent16-frame windows.",
            "SAM2 receives quality100 JPEG frames; V-JEPA receives original rendered RGB; per-clip conversion errors recorded.",
            "Same fixed synthetic benchmark; specialist comparator added after V-JEPA development and a real-video demo, before V-JEPA held-out results were viewed.",
            "Dense raw-mask metrics use every frame afterframe0, and are independent of the calibrated patch readout."]})
    if write_videos:
        for condition in sorted({s.condition for s in scenes}):
            scene = next(s for s in scenes if s.condition == condition)
            bench.annotated_video(scene, {METHOD: predictions[scene.name]},
                                  {METHOD: threshold}, out/stage/f"{scene.name}_sam2.mp4")
    print(f"SAM2_{stage.upper()}_COMPLETE", json.dumps(report["methods"][METHOD]["overall"]), flush=True)
    return report


def self_test():
    """Readout plumbing checks with fabricated predictions, no model/test data."""
    masks = np.zeros((4, 4, bench.SIZE, bench.SIZE), bool)
    for ki in range(4):
        masks[:, ki, 32+48*ki:64+48*ki, 80:112] = True
    masks[2:, 3] = False
    pred = masks_to_predictions(masks)
    assert np.array_equal(pred.scores[0], np.ones(4))
    assert pred.scores[1, 3] == 0 and not pred.eligible[1, 3]
    for ki in range(4):
        assert 2+3*ki <= pred.cells[0, ki, 0] < 4+3*ki
        assert 5 <= pred.cells[0, ki, 1] < 7
    changed = masks.copy()
    changed[2:] = ~changed[2:]
    later_changed = masks_to_predictions(changed)
    assert np.array_equal(pred.cells[0], later_changed.cells[0])
    assert np.array_equal(pred.scores[0], later_changed.scores[0])
    one_frame = masks.copy()
    one_frame[1, 0] = False
    one_frame_pred = masks_to_predictions(one_frame)
    assert not one_frame_pred.eligible[0, 0] and one_frame_pred.scores[0, 0] == .5
    # Dense scorer counts absent hallucinations separately from visible IoU.
    from types import SimpleNamespace
    truth = np.zeros((4, bench.SIZE, bench.SIZE), np.uint8)
    for ki, pid in enumerate(bench.PART_NAMES):
        truth[masks[:, ki]] = pid
    scene = SimpleNamespace(name="fixture", seed=0, condition="fixture",
                            frames=np.zeros((4, 1, 1, 3), np.uint8), masks=truth)
    dense = summarize_dense_rows(dense_mask_rows(scene, masks))
    assert dense["mean_iou_given_visible"]["rate"] == 1.
    assert dense["visible_part_frames"] == 10 and dense["absent_part_frames"] == 2
    hallucinated = masks.copy()
    hallucinated[2:, 3, 0:4, 0:4] = True
    changed_dense = summarize_dense_rows(dense_mask_rows(scene, hallucinated))
    assert changed_dense["mean_iou_given_visible"]["rate"] == 1.
    assert changed_dense["false_presence_given_absent"]["rate"] == 1.
    assert changed_dense["absent_predicted_pixels"] == 32
    assert changed_dense["all_frame_micro_pixel_precision"]["rate"] < 1.
    return {"readout_shape": True, "full_patch": True, "absence": True,
            "both_frames_required": True, "future_pair_independence": True,
            "dense_oracle_iou": True, "dense_absent_hallucination_separated": True,
            "real_model_results": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["calibration", "test"])
    parser.add_argument("--out", default="/content/day4_sam2_run")
    parser.add_argument("--checkpoint")
    parser.add_argument("--frozen-path")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.stage:
        run_stage(args.out, args.stage, checkpoint=args.checkpoint,
                  frozen_path=args.frozen_path, write_videos=not args.no_videos)
    else:
        parser.error("Choose --stage or --self-test")
