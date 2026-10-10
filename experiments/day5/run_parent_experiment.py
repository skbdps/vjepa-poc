"""Frozen parent-intersection ablation on new synthetic clips, without tuning.

Inference receives RGB and frame-zero child/parent masks only. Four children
and two whole cars are tracked in separate SAM2 video states. The existing
child predictions are shared by both arms. Future truth enters scoring only.
Run --freeze before smoke/test; commit that JSON and source before held-out test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DAY4 = HERE.parent / "day4"
for folder in (DAY4, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import benchmark as bench
import parent_benchmark as renderer
import real_video
import sam2_benchmark
import part_editor

CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
PATCH_THRESHOLD = 0.9404296875
ARMS = ("raw_part", "parent_intersection")
REPRESENTATIONS = ("raw_mask", "effective_edit")
PART_IDS = tuple(bench.PART_NAMES)
PARENT_IDS = (101, 102)
POLICY = {
    "version": "day5_parent_intersection_v1",
    "raw_mask": "M_i",
    "parent_mask": "M_i AND P_own(i)",
    "effective_raw": "M_i AND NOT union(other RAW child masks)",
    "effective_parent": "effective_raw_i AND P_own(i)",
    "parent_mapping": {"1": 101, "2": 101, "3": 102, "4": 102},
    "parent_ids": [101, 102], "child_ids": [1, 2, 3, 4],
    "sam2_states": "four children jointly; two parents jointly; states independent",
    "prompt_frames": [0], "logit_threshold": 0.0,
    "dilation": 0, "parameter_search": "none", "patch_presence_gates_edits": False,
    "secondary_patch_threshold": PATCH_THRESHOLD,
    "primary_improvement": "strictly lower absolute wrong-other-car effective paint pixels",
    "guardrails": "raw AND effective visible mean IoU and pooled recall losses each <=0.01 absolute",
    "max_allowed_absolute_loss": 0.01,
    "bootstrap": {"unit": "paired clip", "resamples": 2000, "seed": 9514,
                  "interpretation": "descriptive; pooled ratios recomputed; no tuning"},
}


def source_hashes():
    paths = [HERE / "run_parent_experiment.py", HERE / "parent_benchmark.py",
             DAY4 / "benchmark.py", DAY4 / "real_video.py",
             DAY4 / "sam2_benchmark.py", DAY4 / "part_editor.py"]
    return {str(p.relative_to(HERE.parent)): real_video.sha256_file(p) for p in paths}


def source_digest():
    return hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()


def configuration():
    return {"source_digest": source_digest(), "source_hashes": source_hashes(),
            "policy": POLICY, "smoke_seeds": renderer.SMOKE_SEEDS,
            "test_seeds": renderer.TEST_SEEDS, "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_revision": real_video.SAM2_REVISION, "model_config": real_video.SAM2_CONFIG,
            "checkpoint_url": real_video.SAM2_CHECKPOINT, "precision": "float32",
            "postprocessing": False, "jpeg": sam2_benchmark.JPEG_CONFIG,
            "renderer_protocol": renderer.OWNER_PROTOCOL_VERSION,
            "freeze_status": "fixed policy; no smoke or test parameter selection"}


def freeze_config(path):
    path = Path(path)
    config = configuration()
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("Refusing to overwrite a different freeze; use a new explicit path")
    real_video.write_json(path, config)
    return config


def validate_freeze(path):
    frozen = json.loads(Path(path).read_text())
    if frozen != configuration():
        raise RuntimeError("Source, seed manifest, model, or policy differs from the freeze")
    return frozen


def _validate_predictions(children, parents):
    children, parents = np.asarray(children), np.asarray(parents)
    if children.dtype != bool or parents.dtype != bool:
        raise ValueError("Predictions must be boolean logits>0 masks")
    if children.ndim != 4 or children.shape[1] != 4:
        raise ValueError("Children must have shape [T,4,H,W]")
    if parents.shape != (len(children), 2, *children.shape[-2:]):
        raise ValueError("Parents must have shape [T,2,H,W]")
    return children, parents


def arm_masks(children, parents):
    """Prediction-only policy; a gated edit can never release a protected pixel."""
    children, parents = _validate_predictions(children, parents)
    gated = children & parents[:, [0, 0, 1, 1]]
    effective = np.stack([part_editor.edit_masks(children, PART_IDS, pid)
                          for pid in PART_IDS], axis=1)
    gated_effective = effective & parents[:, [0, 0, 1, 1]]
    if np.any(gated_effective & ~effective):
        raise AssertionError("Parent edit unexpectedly adds paint")
    return {"raw_part": {"raw_mask": children, "effective_edit": effective},
            "parent_intersection": {"raw_mask": gated, "effective_edit": gated_effective}}


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def score_masks(scene, owners, predicted, arm, representation, parent=False):
    """Post-inference scoring. Owners are FULL visible cars, not tagged parts."""
    predicted, owners = np.asarray(predicted), np.asarray(owners)
    ids = PARENT_IDS if parent else PART_IDS
    shape = (len(scene.frames), len(ids), *scene.masks.shape[-2:])
    if predicted.shape != shape or predicted.dtype != bool:
        raise ValueError(f"Expected boolean predicted masks {shape}")
    if owners.shape != scene.masks.shape or not set(np.unique(owners)).issubset({0, 1, 2}):
        raise ValueError("Owner labels must match the scene and use IDs0,1,2")
    # Scoring-only diagnostic; never used by the mask policy or inference.
    labels = (bench.patch_labels(scene.masks) if not parent and
              scene.masks.shape[1:] == (bench.SIZE, bench.SIZE) and len(scene.frames) % 2 == 0 else None)
    rows = []
    for frame in range(1, len(scene.frames)):
        for j, pid in enumerate(ids):
            own = j + 1 if parent else renderer.PART_TO_OWNER[pid]
            truth = owners[frame] == own if parent else scene.masks[frame] == pid
            mask = predicted[frame, j]
            intersection = int(np.count_nonzero(mask & truth))
            area, gt_area = int(mask.sum()), int(truth.sum())
            union = area + gt_area - intersection
            wrong = int(np.count_nonzero(mask & (owners[frame] == 3 - own)))
            own_other = int(np.count_nonzero(mask & (owners[frame] == own) & ~truth))
            background = int(np.count_nonzero(mask & (owners[frame] == 0)))
            if wrong + own_other + background != area - intersection:
                raise AssertionError("False-positive ownership categories do not partition pixels")
            rows.append({"scene": scene.name, "seed": scene.seed, "condition": scene.condition,
                         "arm": arm, "representation": representation, "frame": frame,
                         "part_id": pid, "parent_owner_id": own,
                         "target": renderer.OWNER_NAMES[own] if parent else bench.PART_NAMES[pid],
                         "visible": bool(gt_area), "predicted_present": bool(area),
                         "pair_has_eligible_gt_patch": bool(np.any(labels[frame // 2] == pid)) if labels is not None else None,
                         "gt_pixels": gt_area, "predicted_pixels": area,
                         "intersection_pixels": intersection, "union_pixels": union,
                         "false_positive_pixels": area - intersection,
                         "false_negative_pixels": gt_area - intersection,
                         "wrong_car_pixels": wrong, "wrong_car_present": bool(wrong),
                         "same_car_other_pixels": own_other, "background_pixels": background,
                         "image_pixels": int(truth.size),
                         "iou": intersection / union if gt_area else None,
                         "pixel_recall": intersection / gt_area if gt_area else None})
    return rows


def sufficient(rows):
    visible = [r for r in rows if r["visible"]]
    absent = [r for r in rows if not r["visible"]]
    return np.asarray([len(rows), len(visible), len(absent),
                       sum(r["iou"] for r in visible),
                       sum(r["intersection_pixels"] for r in visible),
                       sum(r["gt_pixels"] for r in visible),
                       sum(r["predicted_pixels"] for r in rows),
                       sum(r["predicted_pixels"] for r in visible),
                       sum(r["wrong_car_pixels"] for r in rows),
                       sum(r["wrong_car_present"] for r in rows),
                       sum(r["predicted_present"] for r in absent),
                       sum(r["predicted_pixels"] for r in absent),
                       sum(r["predicted_present"] for r in rows),
                       sum(r["image_pixels"] for r in rows),
                       sum(r["false_positive_pixels"] for r in rows),
                       sum(r["same_car_other_pixels"] for r in rows),
                       sum(r["background_pixels"] for r in rows)], dtype=np.float64)


def _from_sufficient(s):
    n, vis, absent, iou, tp, gt, area, area_vis, wrong, wrong_n, absent_n, absent_area, present, image_area, fp, own_other, bg = s
    integer = lambda x: int(round(float(x)))
    return {"scored_part_frames": integer(n), "visible_part_frames": integer(vis),
            "absent_part_frames": integer(absent),
            "mean_iou_given_visible": _ratio(float(iou), integer(vis)),
            "visible_micro_pixel_recall": _ratio(integer(tp), integer(gt)),
            "visible_micro_pixel_precision": _ratio(integer(tp), integer(area_vis)),
            "all_frame_micro_pixel_precision": _ratio(integer(tp), integer(area)),
            "wrong_car_pixels": integer(wrong),
            "wrong_car_pixel_fraction_of_predictions": _ratio(integer(wrong), integer(area)),
            "wrong_car_pixel_fraction_of_false_positives": _ratio(integer(wrong), integer(fp)),
            "wrong_car_part_frame_rate": _ratio(integer(wrong_n), integer(n)),
            "false_presence_given_absent": _ratio(integer(absent_n), integer(absent)),
            "absent_predicted_pixels": integer(absent_area),
            "predicted_pixels": integer(area), "false_positive_pixels": integer(fp),
            "same_car_other_pixels": integer(own_other), "background_pixels": integer(bg),
            "predicted_part_frame_coverage": _ratio(integer(present), integer(n)),
            "predicted_image_pixel_coverage": _ratio(integer(area), integer(image_area)),
            "visible_predicted_to_truth_area_ratio": _ratio(integer(area_vis), integer(gt))}


def summarize(rows):
    return {**_from_sufficient(sufficient(rows)), "clips": len({r["scene"] for r in rows})}


def aggregate(rows):
    return {"overall": summarize(rows),
            "visible_without_eligible_gt_patch_in_pair": summarize(
                [r for r in rows if r["visible"] and r["pair_has_eligible_gt_patch"] is False]),
            "by_condition": {c: summarize([r for r in rows if r["condition"] == c])
                             for c in sorted({r["condition"] for r in rows})},
            "per_clip": {s: summarize([r for r in rows if r["scene"] == s])
                         for s in sorted({r["scene"] for r in rows})}}


def paired_bootstrap(raw_rows, gated_rows):
    """Paired whole-clip resampling; recompute pooled ratios for every draw."""
    key = lambda r: (r["scene"], r["frame"], r["part_id"])
    raw_index, gate_index = {key(r): r for r in raw_rows}, {key(r): r for r in gated_rows}
    if len(raw_index) != len(raw_rows) or len(gate_index) != len(gated_rows) or raw_index.keys() != gate_index.keys():
        raise ValueError("Paired rows must have identical unique clip/frame/part keys")
    for k in raw_index:
        for field in ("gt_pixels", "visible", "condition", "image_pixels"):
            if raw_index[k][field] != gate_index[k][field]:
                raise ValueError("Paired ground truth differs")
    names = sorted({r["scene"] for r in raw_rows})
    raw = np.stack([sufficient([r for r in raw_rows if r["scene"] == n]) for n in names])
    gate = np.stack([sufficient([r for r in gated_rows if r["scene"] == n]) for n in names])
    specs = {"wrong_car_pixels": (8, None), "mean_iou_given_visible": (3, 1),
             "visible_micro_pixel_recall": (4, 5),
             "wrong_car_pixel_fraction_of_predictions": (8, 6),
             "wrong_car_part_frame_rate": (9, 0), "false_presence_given_absent": (10, 2)}
    rng = np.random.default_rng(POLICY["bootstrap"]["seed"])
    draws = rng.integers(0, len(names), (POLICY["bootstrap"]["resamples"], len(names)))
    raw_draw, gate_draw = raw[draws].sum(1), gate[draws].sum(1)
    result = {}
    for name, (num, den) in specs.items():
        a, b = raw.sum(0), gate.sum(0)
        if den is None:
            difference = b[num] - a[num]
            samples = gate_draw[:, num] - raw_draw[:, num]
        elif a[den] and b[den]:
            difference = b[num] / b[den] - a[num] / a[den]
            valid = (raw_draw[:, den] > 0) & (gate_draw[:, den] > 0)
            samples = gate_draw[valid, num] / gate_draw[valid, den] - raw_draw[valid, num] / raw_draw[valid, den]
        else:
            difference, samples = None, np.asarray([])
        result[name] = {"difference_parent_minus_raw": None if difference is None else float(difference),
                        "paired_clip_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
                        "valid_resamples": len(samples)}
    return {"configuration": POLICY["bootstrap"], "clips": names, "metrics": result}


def success_assessment(arms):
    raw = arms["raw_part"]["effective_edit"]["overall"]
    gated = arms["parent_intersection"]["effective_edit"]["overall"]
    improved = gated["wrong_car_pixels"] < raw["wrong_car_pixels"]
    guards = {}
    for representation in REPRESENTATIONS:
        for metric in ("mean_iou_given_visible", "visible_micro_pixel_recall"):
            a = arms["raw_part"][representation]["overall"][metric]["rate"]
            b = arms["parent_intersection"][representation]["overall"][metric]["rate"]
            loss = None if a is None or b is None else a - b
            guards[f"{representation}.{metric}"] = {
                "loss_raw_minus_parent": loss, "max_allowed_loss": .01,
                "passed": loss is not None and loss <= .01 + 1e-12}
    return {"wrong_car_effective_paint_strictly_lower": improved,
            "wrong_car_pixel_difference_parent_minus_raw": gated["wrong_car_pixels"] - raw["wrong_car_pixels"],
            "relative_wrong_car_effective_pixel_reduction": _ratio(
                raw["wrong_car_pixels"] - gated["wrong_car_pixels"], raw["wrong_car_pixels"]),
            "guardrails": guards,
            "passed": improved and all(g["passed"] for g in guards.values()),
            "meaning": "fixed synthetic criterion only; uncertainty descriptive; no real-video/face/generative claim"}


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_initial(frames, children0, parents0):
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (384, 384, 3):
        raise ValueError("Expected RGB uint8 frames [T,384,384,3]")
    for masks, ids in ((children0, PART_IDS), (parents0, PARENT_IDS)):
        if set(masks) != set(ids):
            raise ValueError("Unexpected first-frame object IDs")
        if any(np.asarray(m).shape != (384, 384) or np.asarray(m).dtype != bool or not np.any(m) for m in masks.values()):
            raise ValueError("First-frame prompts must be nonempty boolean [384,384] masks")
    for pid in PART_IDS:
        if np.any(children0[pid] & ~parents0[101 + (pid - 1) // 2]):
            raise ValueError("Initial child is not a subset of its own parent")


def infer_sequence(frames, children0, parents0, folder, checkpoint, predictor,
                   frozen_source, inference_fn=None):
    """Information barrier: the signature has NO later truth/Scene/owner labels.

    The injected inference function is solely for a CPU interface self-test.
    Production uses real_video.run_sam2, with a fresh state on EACH invocation.
    """
    _validate_initial(frames, children0, parents0)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"source_digest": frozen_source,
                "rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                "children0_sha256": hashlib.sha256(np.stack([children0[i] for i in PART_IDS]).tobytes()).hexdigest(),
                "parents0_sha256": hashlib.sha256(np.stack([parents0[i] for i in PARENT_IDS]).tobytes()).hexdigest(),
                "checkpoint_sha256": real_video.sha256_file(checkpoint),
                "policy": POLICY, "model_revision": real_video.SAM2_REVISION,
                "config": real_video.SAM2_CONFIG, "postprocessing": False,
                "precision": "float32", "prompt_frames": [0]}
    if identity["checkpoint_sha256"] != CHECKPOINT_SHA256 and inference_fn is None:
        raise RuntimeError("Official SAM2.1 tiny checkpoint checksum mismatch")
    manifest_path = folder / "clip_manifest.json"
    paths = {"children": folder / "children" / "sam2_masks.npz",
             "parents": folder / "parents" / "sam2_masks.npz"}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise RuntimeError("Stale cache; source, prompts, RGB, or model changed")
        for role, path in paths.items():
            if not path.exists() or manifest["cache_sha256"][role] != real_video.sha256_file(path):
                raise RuntimeError(f"Cache checksum mismatch: {role}")
    else:
        jpeg = sam2_benchmark._save_jpegs(frames, folder / "frames")
        run = inference_fn or real_video.run_sam2
        timings = {}
        # Two function calls mean two independent init_state/propagate/reset cycles.
        # Copies prevent a model implementation from changing prompt dictionaries.
        for role, prompts in (("children", children0), ("parents", parents0)):
            print(f"DAY5_INFERENCE_START {folder.name} {role}", flush=True)
            _, ids, timing = run(folder / "frames", {k: v.copy() for k, v in prompts.items()},
                                 folder / role, checkpoint=checkpoint, predictor=predictor)
            if ids != list(PART_IDS if role == "children" else PARENT_IDS):
                raise RuntimeError("SAM2 returned unexpected object ordering")
            timings[role] = timing
            print(f"DAY5_INFERENCE_DONE {folder.name} {role} {timing['seconds']:.1f}s", flush=True)
        manifest = {"identity": identity, "jpeg": jpeg, "timing": timings,
                    "cache_sha256": {role: real_video.sha256_file(path) for role, path in paths.items()}}
        real_video.write_json(manifest_path, manifest)
    arrays = {}
    for role, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            expected = list(PART_IDS if role == "children" else PARENT_IDS)
            if data["ids"].tolist() != expected:
                raise RuntimeError("Cache ID ordering mismatch")
            arrays[role] = data["masks"]
    _validate_predictions(arrays["children"], arrays["parents"])
    if arrays["children"].shape != (len(frames), 4, 384, 384):
        raise RuntimeError("Cache frame count or spatial shape differs")
    return arrays["children"], arrays["parents"], manifest


def _verify_model(checkpoint, predictor):
    import torch
    import sam2
    if real_video.sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Official checkpoint checksum mismatch")
    repository = Path(sam2.__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if revision != real_video.SAM2_REVISION:
        raise RuntimeError(f"SAM2 source revision changed: {revision}")
    if predictor is not None:
        raise ValueError("Production builds the model from the verified checkpoint; do not supply an unverified predictor")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(real_video.SAM2_CONFIG, checkpoint,
                                           device="cuda", apply_postprocessing=False)
    predictor.eval().float()
    if any(p.dtype != torch.float32 for p in predictor.parameters() if p.is_floating_point()):
        raise RuntimeError("Expected FP32 parameters")
    return predictor, {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                       "sam2_repository_revision": revision, "sam2_repository": str(repository)}


def render_diagnostic(scene, owners, arms, parents, path):
    """Fixed first crossing-seed panels; mask-risk colors, not a new model output."""
    palette = np.asarray([(255, 100, 40), (240, 210, 55), (65, 215, 145), (150, 135, 255)], np.uint8)
    frames = []
    for t, original in enumerate(scene.frames):
        panels = []
        for arm in ARMS:
            maskset = arms[arm]["effective_edit"][t]
            canvas = original.copy()
            wrong_union = np.zeros(original.shape[:2], bool)
            for j, pid in enumerate(PART_IDS):
                mask = maskset[j]
                canvas[mask] = (.45 * canvas[mask] + .55 * palette[j]).astype(np.uint8)
                wrong_union |= mask & (owners[t] == 3 - renderer.PART_TO_OWNER[pid])
            canvas[wrong_union] = (255, 0, 220)
            tile = Image.new("RGB", (384, 430), (19, 23, 33))
            tile.paste(Image.fromarray(canvas), (0, 46))
            draw = ImageDraw.Draw(tile)
            draw.text((8, 5), f"{arm} | frame {t:02d}", fill="white")
            draw.text((8, 23), f"Wrong-car paint {int(wrong_union.sum())} px (magenta)", fill="white")
            panels.append(np.asarray(tile))
        frames.append(np.concatenate(panels, axis=1))
    bench.write_video(path, frames, fps=12)
    Image.fromarray(frames[len(frames)//2]).save(Path(path).with_suffix(".png"))


def _markdown(report):
    lines = ["# Day5 parent-intersection experiment", "",
             f"Stage: **{report['stage']}**. Fixed masks, first-frame prompts only; no tuning.", "",
             "Parent masks cost a second SAM2 pass and two additional whole-car annotations.", "",
             "## Primary effective edit region", "",
             "Original overlapping child masks remain protected in both arms. Parent clipping cannot release protected paint.", "",
             "| Arm | Wrong-car pixels | Wrong-car / predicted pixels | Wrong-car part-frames | Visible mean IoU | Visible pooled recall | Absent false presence |", "|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda r: "n/a" if r["rate"] is None else f"{100*r['rate']:.3f}% ({r['numerator']:.4g}/{r['denominator']})"
    for arm in ARMS:
        s = report["arms"][arm]["effective_edit"]["overall"]
        lines.append(f"| {arm} | {s['wrong_car_pixels']:,} | {fmt(s['wrong_car_pixel_fraction_of_predictions'])} | {fmt(s['wrong_car_part_frame_rate'])} | {fmt(s['mean_iou_given_visible'])} | {fmt(s['visible_micro_pixel_recall'])} | {fmt(s['false_presence_given_absent'])} |")
    lines += ["", "## Raw mask accuracy (before overlap protection)", "",
              "| Arm | Visible mean IoU | Visible pooled recall | Predicted pixels | Absent predicted pixels |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        s = report["arms"][arm]["raw_mask"]["overall"]
        lines.append(f"| {arm} | {fmt(s['mean_iou_given_visible'])} | {fmt(s['visible_micro_pixel_recall'])} | {s['predicted_pixels']:,} | {s['absent_predicted_pixels']:,} |")
    reduction = report['success']['relative_wrong_car_effective_pixel_reduction']
    lines += ["", f"Fixed success criterion passed: **{report['success']['passed']}**.",
              f"Relative reduction in wrong-car effective pixels: {fmt(reduction)}.",
              "Requires strictly lower absolute wrong-car effective pixels, with no more than 1 percentage point loss in raw AND effective visible mean IoU and pooled recall. A zero-paint solution cannot pass the recall guard.", "",
              "## Paired clip bootstrap: parent minus raw", "",
              "Intervals are descriptive; paired whole clips are resampled and pooled ratios recomputed.", "",
              "| Representation | Metric | Difference | 95% interval |", "|---|---|---:|---|"]
    for representation in REPRESENTATIONS:
        for metric, value in report["paired_comparison"][representation]["metrics"].items():
            delta, ci = value["difference_parent_minus_raw"], value["paired_clip_bootstrap_95ci"]
            scale = 1 if metric == "wrong_car_pixels" else 100
            unit = "pixels" if scale == 1 else "pp"
            dv = "n/a" if delta is None else f"{delta*scale:+.4f} {unit}"
            cv = "n/a" if ci is None else f"[{ci[0]*scale:+.4f}, {ci[1]*scale:+.4f}] {unit}"
            lines.append(f"| {representation} | {metric} | {dv} | {cv} |")
    parent = report["parents"]["overall"]
    lines += ["", "## Parent accuracy and visible slivers", "",
              f"Whole-parent visible mean IoU: {fmt(parent['mean_iou_given_visible'])}; pooled recall: {fmt(parent['visible_micro_pixel_recall'])}; wrong-owner pixels: {parent['wrong_car_pixels']:,}.", "",
              "The subgroup below contains visible source part-frames whose two-frame pair has no eligible Day4 ground-truth patch. It includes moving/sliver parts, not only small physical parts.", "",
              "| Arm | Representation | Visible subgroup frames | Mean IoU | Pooled recall |", "|---|---|---:|---:|---:|"]
    for arm in ARMS:
        for representation in REPRESENTATIONS:
            sub = report["arms"][arm][representation]["visible_without_eligible_gt_patch_in_pair"]
            lines.append(f"| {arm} | {representation} | {sub['visible_part_frames']} | {fmt(sub['mean_iou_given_visible'])} | {fmt(sub['visible_micro_pixel_recall'])} |")
    n = report["arms"]["raw_part"]["raw_mask"]["overall"]
    lines += ["", f"Scored units per arm/representation: {n['scored_part_frames']} = {n['visible_part_frames']} visible + {n['absent_part_frames']} absent part-frames. Frame0 excluded.", "",
              "Wrong-car pixels intersect the full visible OTHER car, including its untagged body. False-positive fractions use predicted-pixel denominators; event rates use all scored part-frames. Dense visibility means any nonempty true part; empty visible predictions score zero. Raw and effective denominators are reported separately.", "",
              "Secondary patch localization reuses the Day4 readout and fixed 0.9404296875 threshold, without gating edits. It has a different tubelet denominator and excludes ambiguous slivers.", "",
              "This is a new seed set in the same simple 2-D renderer. It tests containment at extra annotation/inference cost, not learned parent identity, face consistency, generative edits, or real-video generalization."]
    return "\n".join(lines) + "\n"


def run_stage(out_dir, stage="smoke", checkpoint=None, predictor=None, frozen_path=None, write_videos=True):
    if stage not in ("smoke", "test"):
        raise ValueError("Stage must be smoke or test")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if frozen_path is None:
        frozen_path = out / "frozen_config.json"
    # BOTH smoke and test are frozen; smoke only checks mechanics, never selects policy.
    frozen = validate_freeze(frozen_path)
    checkpoint = str(checkpoint or real_video.prepare_sam2())
    predictor, runtime = _verify_model(checkpoint, predictor)
    import torch
    seeds = renderer.SMOKE_SEEDS if stage == "smoke" else renderer.TEST_SEEDS
    all_rows, parent_rows, patch_rows, manifests = [], [], [], []
    for condition, condition_seeds in seeds.items():
        for seed in condition_seeds:
            name = f"{stage}_{condition}_{seed}"
            started = time.perf_counter()
            scene, owners = renderer.generate_scene(seed, condition, name)
            children0 = {pid: (scene.masks[0] == pid).copy() for pid in PART_IDS}
            parents0 = {101 + own - 1: (owners[0] == own).copy() for own in (1, 2)}
            # Future labels stop at this boundary; neither inference nor policy receives them.
            with torch.autocast(device_type="cuda", enabled=False):
                children, parents, manifest = infer_sequence(scene.frames, children0, parents0,
                    out / "clips" / name, checkpoint, predictor, frozen["source_digest"])
            derived = arm_masks(children, parents)
            for arm in ARMS:
                for representation in REPRESENTATIONS:
                    all_rows.extend(score_masks(scene, owners, derived[arm][representation], arm, representation))
                pred = sam2_benchmark.masks_to_predictions(derived[arm]["raw_mask"])
                np.savez_compressed(out / "clips" / name / f"{arm}_patch_predictions.npz",
                                    scores=pred.scores, cells=pred.cells, eligible=pred.eligible)
                patch_rows.extend(bench.score_scene(scene, pred, PATCH_THRESHOLD, arm))
            parent_rows.extend(score_masks(scene, owners, parents, "independent_parent", "raw_mask", parent=True))
            scored_manifest = {"scene": name, "condition": condition, "seed": seed,
                "part_truth_sha256": hashlib.sha256(scene.masks.tobytes()).hexdigest(),
                "owner_truth_sha256": hashlib.sha256(owners.tobytes()).hexdigest(),
                "clip_manifest_sha256": real_video.sha256_file(out / "clips" / name / "clip_manifest.json"),
                "model_seconds": sum(t["seconds"] for t in manifest["timing"].values())}
            manifests.append(scored_manifest)
            if write_videos and condition == "crossing" and seed == condition_seeds[0]:
                (out / stage).mkdir(parents=True, exist_ok=True)
                render_diagnostic(scene, owners, derived, parents, out / stage / f"{name}_comparison.mp4")
            print(f"DAY5_{stage.upper()}_DONE {name} {time.perf_counter()-started:.1f}s", flush=True)
            # Recoverable scored progress after every finished clip. Final report still requires all clips.
            _write_csv(out / stage / "dense_rows.partial.csv", all_rows)
            del scene, owners, children, parents, derived
    expected = sum(map(len, seeds.values())) * (bench.N_FRAMES - 1) * len(PART_IDS)
    arms = {arm: {representation: aggregate([r for r in all_rows if r["arm"] == arm and r["representation"] == representation])
                  for representation in REPRESENTATIONS} for arm in ARMS}
    if any(arms[a][r]["overall"]["scored_part_frames"] != expected for a in ARMS for r in REPRESENTATIONS):
        raise AssertionError("Incomplete fixed seed set")
    paired = {representation: paired_bootstrap(
        [r for r in all_rows if r["arm"] == "raw_part" and r["representation"] == representation],
        [r for r in all_rows if r["arm"] == "parent_intersection" and r["representation"] == representation])
        for representation in REPRESENTATIONS}
    report = {"stage": stage, "frozen_config": frozen, "runtime": runtime,
              "python": platform.python_version(), "numpy": np.__version__, "clips": manifests,
              "unit": "part-frame; all frames after0; any nonempty true part is visible",
              "arms": arms, "paired_comparison": paired, "success": success_assessment(arms),
              "parents": aggregate(parent_rows),
              "secondary_patch": {"threshold": PATCH_THRESHOLD, "readout_version": sam2_benchmark.READOUT_VERSION,
                  "methods": {arm: bench.summarize_rows([r for r in patch_rows if r["method"] == arm]) for arm in ARMS}},
              "annotation_cost": "four child masks PLUS two whole-car masks at frame0",
              "inference_cost": "one four-child SAM2 state plus independent two-parent SAM2 state",
              "claim_scope": "synthetic2D containment ablation; no real-video, face, generation or learned hierarchy claim"}
    _write_csv(out / stage / "dense_rows.csv", all_rows)
    _write_csv(out / stage / "parent_rows.csv", parent_rows)
    _write_csv(out / stage / "patch_rows.csv", patch_rows)
    real_video.write_json(out / stage / "results.json", report)
    (out / stage / "RESULTS.md").write_text(_markdown(report))
    (out / stage / "dense_rows.partial.csv").unlink(missing_ok=True)
    print(f"DAY5_{stage.upper()}_COMPLETE", json.dumps(report["success"]), flush=True)
    return report


def self_test():
    """Small fabricated CPU fixtures; no held-out seed rendered or scored."""
    import tempfile
    from types import SimpleNamespace
    children = np.zeros((4, 4, 12, 12), bool)
    children[:, 0, 1:5, 1:5] = True
    children[:, 1, 3:7, 3:7] = True
    children[:, 2, 7:10, 1:4] = True
    children[:, 3, 7:10, 7:10] = True
    parents = np.ones((4, 2, 12, 12), bool)
    parents[:, 0, 3:7, 3:7] = False
    derived = arm_masks(children, parents)
    raw = derived["raw_part"]["effective_edit"]
    gate = derived["parent_intersection"]["effective_edit"]
    assert not np.any(gate & ~raw)
    assert np.max(raw.sum(1)) <= 1 and np.max(gate.sum(1)) <= 1
    assert not gate[:, 0, 3:5, 3:5].any(), "Lost other child must not release protected overlap"
    for j, pid in enumerate(PART_IDS):
        assert np.array_equal(raw[:, j], part_editor.edit_masks(children, PART_IDS, pid))
    truth = np.zeros((4, 12, 12), np.uint8)
    truth[:, 1:3, 1:3] = 1
    truth[:, 3:5, 5:7] = 2
    truth[:, 7:9, 1:3] = 3
    truth[:, 7:9, 7:9] = 4
    owners = np.zeros_like(truth)
    owners[:, :6] = 1
    owners[:, 6:] = 2
    truth[2:, 1:3, 1:3] = 0
    oracle = np.stack([truth == pid for pid in PART_IDS], axis=1)
    scene = SimpleNamespace(name="fixture", seed=0, condition="fixture", masks=truth,
                            frames=np.zeros((4, 12, 12, 3), np.uint8))
    rows = score_masks(scene, owners, oracle, "oracle", "raw_mask")
    s = summarize(rows)
    assert s["scored_part_frames"] == 12 and s["visible_part_frames"] == 10 and s["absent_part_frames"] == 2
    assert s["mean_iou_given_visible"]["rate"] == s["visible_micro_pixel_recall"]["rate"] == 1
    bad = oracle.copy()
    bad[1:, 0, 10, 10] = True  # Untagged other-car BODY must count as wrong-car paint.
    b = score_masks(scene, owners, bad, "bad", "raw_mask")
    bs = summarize(b)
    assert bs["wrong_car_pixels"] == 3 and bs["absent_predicted_pixels"] == 2
    assert bs["false_presence_given_absent"]["rate"] == 1
    empty = summarize(score_masks(scene, owners, np.zeros_like(oracle), "empty", "raw_mask"))
    assert empty["mean_iou_given_visible"]["rate"] == empty["visible_micro_pixel_recall"]["rate"] == 0
    paired = paired_bootstrap(b, rows)
    assert paired["metrics"]["wrong_car_pixels"]["difference_parent_minus_raw"] == -3
    assert paired["metrics"]["wrong_car_pixels"]["paired_clip_bootstrap_95ci"] == [-3, -3]
    sig = set(inspect.signature(infer_sequence).parameters)
    assert not sig & {"scene", "owners", "truth", "labels", "future_masks"}
    # Exercise the actual cache/inference boundary with a fake model receiving only 2D prompts.
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        checkpoint = folder / "fake.pt"
        checkpoint.write_bytes(b"CPU self-test checkpoint")
        frames = np.zeros((2, 384, 384, 3), np.uint8)
        c0 = {pid: np.zeros((384, 384), bool) for pid in PART_IDS}
        p0 = {pid: np.zeros((384, 384), bool) for pid in PARENT_IDS}
        for j, pid in enumerate(PART_IDS):
            c0[pid][5+20*j:15+20*j, 5:15] = True
            p0[101+j//2] |= c0[pid]
        calls = []
        def fake_run(frames_dir, prompts, outfolder, checkpoint=None, predictor=None):
            assert all(m.ndim == 2 and m.dtype == bool for m in prompts.values())
            calls.append(sorted(prompts))
            outfolder.mkdir(parents=True, exist_ok=True)
            masks = np.stack([np.stack([prompts[pid] for pid in sorted(prompts)])] * 2)
            np.savez_compressed(outfolder / "sam2_masks.npz", ids=sorted(prompts), masks=masks)
            return masks, sorted(prompts), {"seconds": 0.0, "prompt_frames": [0]}
        a, p, _ = infer_sequence(frames, c0, p0, folder / "cache", checkpoint, None, "fixture", fake_run)
        assert calls == [list(PART_IDS), list(PARENT_IDS)] and a.shape[1] == 4 and p.shape[1] == 2
        infer_sequence(frames, c0, p0, folder / "cache", checkpoint, None, "fixture", fake_run)
        assert len(calls) == 2, "Cache should resume without rerunning"
    return {"original_overlap_protection": True, "gated_edit_subset": True,
            "full_other_car_body_counted": True, "visible_absent_denominators": True,
            "empty_visible_prediction_scores_zero": True, "paired_clip_bootstrap": True,
            "initial_only_inference_signature": True, "two_independent_inference_calls": True,
            "hash_checked_cache_resume": True, "test_scenes_rendered": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "test"])
    parser.add_argument("--out", default="/content/day5_parent_run")
    parser.add_argument("--checkpoint")
    parser.add_argument("--frozen-path")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.freeze:
        print(json.dumps(freeze_config(args.frozen_path or Path(args.out) / "frozen_config.json"), indent=2))
    elif args.stage:
        run_stage(args.out, args.stage, args.checkpoint, frozen_path=args.frozen_path,
                  write_videos=not args.no_videos)
    else:
        parser.error("Choose --self-test, --freeze, or --stage")
