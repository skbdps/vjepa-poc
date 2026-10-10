"""Small, frozen-feature diagnostic of video part identity, not an editing benchmark.

Inputs are RGB frames and V-JEPA features [16,24,24,D] from two independently
encoded 16-frame windows. All matching sees only first-tubelet annotations.
Only calibration clips choose cosine absence thresholds. Later masks are scoring
data. Output metrics are conditional on unambiguous 16x16 patch visibility.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw

SIZE, PATCH, N_FRAMES, TUBELET = 384, 16, 32, 2
GRID, N_STEPS, WINDOW_STEPS = SIZE // PATCH, N_FRAMES // TUBELET, 8
PART_NAMES = {1: "car_A.front_door", 2: "car_A.window",
              3: "car_B.front_door", 4: "car_B.window"}
CALIBRATION_SEEDS = [100, 101, 102]
EVALUATION_SEEDS = [200, 201, 202, 203, 204, 205]
COLORS = [(255, 88, 88), (255, 219, 74), (72, 226, 158), (128, 164, 255)]


@dataclass
class Scene:
    name: str
    frames: np.ndarray
    masks: np.ndarray
    part_names: dict
    seed: int
    condition: str
    parameters: dict | None = None


@dataclass
class Predictions:
    scores: np.ndarray                 # [16,4], maximum cosine or status
    cells: np.ndarray                  # [16,4,2], row and column
    eligible: np.ndarray               # [16,4], can algorithm emit a point?


def _car(draw, labels, x, y, identity, variant):
    """Same-colour corresponding parts; distinct persistent body context."""
    body = (49, 112 + variant, 118) if identity == 0 else (129, 68, 85 + variant)
    draw.rounded_rectangle((x, y+28, x+166, y+103), 11, fill=body)
    draw.polygon([(x+28, y+29), (x+46, y), (x+131, y), (x+150, y+29)], fill=body)
    for wx in (x+30, x+134):
        draw.ellipse((wx-17, y+84, wx+17, y+118), fill=(26, 29, 33))
        draw.ellipse((wx-8, y+93, wx+8, y+109), fill=(165, 170, 178))
    # Unlabelled rear window and rear door remain similar distractors.
    draw.rectangle((x+31, y+13, x+74, y+38), fill=(121, 155, 175))
    draw.rectangle((x+25, y+44, x+75, y+92), fill=(174, 172, 144))
    regions = [(x+87, y+44, x+145, y+94), (x+85, y+8, x+141, y+38)]
    for kind, rect in enumerate(regions):
        pid = 1 + identity*2 + kind
        fill = (191, 177, 123) if kind == 0 else (113, 168, 199)
        draw.rectangle(rect, fill=fill)
        labels.rectangle(rect, fill=pid)
        # Shared sparse texture makes point tracking possible without encoding ID.
        for dx, dy in ((9, 8), (30, 8), (9, 22), (30, 22)):
            px, py = rect[0]+dx, rect[1]+dy
            draw.rectangle((px, py, px+4, py+4), fill=tuple(c-25 for c in fill))
    draw.line((x+81, y+40, x+81, y+98), fill=(35, 45, 52), width=2)
    draw.rectangle((x+128, y+48, x+139, y+51), fill=(231, 234, 221))
    # Different body stripe outside target masks provides persistent identity cue.
    if identity == 0:
        draw.line((x+8, y+73, x+21, y+73), fill=(233, 233, 220), width=4)
    else:
        draw.line((x+8, y+64, x+20, y+81), fill=(233, 233, 220), width=4)


def generate_scene(seed: int, name: str | None = None) -> Scene:
    rng = np.random.default_rng(seed)
    condition = ("motion", "partial_occlusion", "full_occlusion")[seed % 3]
    frames, masks = [], []
    jitter = int(rng.integers(-7, 8))
    variant = int(rng.integers(-15, 16))
    a_start, a_end = int(rng.integers(0, 9)), int(rng.integers(3, 27))
    a_peak = int(rng.integers(174, 201))
    b_start, b_end = int(rng.integers(181, 208)), int(rng.integers(2, 27))
    peak_time, a_gamma, b_gamma = rng.uniform(.41, .59), rng.uniform(.85, 1.3), rng.uniform(.8, 1.4)
    occluder_left = int(rng.integers(166, 181))
    partial_left, partial_width = int(rng.integers(184, 221)), int(rng.integers(12, 24))
    parameters = dict(a_start=a_start, a_end=a_end, a_peak=a_peak, b_start=b_start,
                      b_end=b_end, peak_time=float(peak_time), a_gamma=float(a_gamma),
                      b_gamma=float(b_gamma), occluder_left=occluder_left,
                      partial_left=partial_left, partial_width=partial_width,
                      vertical_jitter=jitter, body_color_variant=variant)
    for t in range(N_FRAMES):
        p = t/(N_FRAMES-1)
        # Full-occlusion trajectories return, permitting a genuine recovery test.
        if condition == "full_occlusion":
            if p <= peak_time:
                progress = (p/peak_time)**a_gamma
                ax = int(a_start+(a_peak-a_start)*progress)
            else:
                progress = ((p-peak_time)/(1-peak_time))**a_gamma
                ax = int(a_peak+(a_end-a_peak)*progress)
        else:
            ax = int(a_start+(a_peak-a_start)*p**a_gamma)
        bx = int(b_start+(b_end-b_start)*p**b_gamma)
        image = Image.new("RGB", (SIZE, SIZE), (223, 229, 231))
        mask = Image.new("L", (SIZE, SIZE), 0)
        draw, labels = ImageDraw.Draw(image), ImageDraw.Draw(mask)
        for road_y in (187, 360):
            draw.rectangle((0, road_y, SIZE, road_y+7), fill=(135, 143, 148))
        # Small unrelated object moves across the background.
        dx = int((seed*13+t*7) % (SIZE-18))
        draw.rectangle((dx, 215, dx+16, 231), fill=(191, 177, 123))
        _car(draw, labels, ax, 53+jitter, 0, variant)
        _car(draw, labels, bx, 234-jitter, 1, variant)
        if condition != "motion":
            rect = ((partial_left, 35, partial_left+partial_width, 188)
                    if condition == "partial_occlusion" else (occluder_left, 35, 376, 188))
            draw.rectangle(rect, fill=(91, 98, 109))
            labels.rectangle(rect, fill=0)
            for xx in range(rect[0]+4, rect[2], 12):
                draw.line((xx, rect[1], xx, rect[3]), fill=(100, 107, 119), width=2)
        frames.append(np.asarray(image))
        masks.append(np.asarray(mask))
    return Scene(name or f"seed_{seed}", np.stack(frames), np.stack(masks),
                 PART_NAMES.copy(), seed, condition, parameters)


def generate_suite(out_dir=None, split="calibration", seeds=None):
    if split not in ("calibration", "evaluation"):
        raise ValueError("split must be calibration or evaluation")
    seeds = seeds if seeds is not None else (
        CALIBRATION_SEEDS if split == "calibration" else EVALUATION_SEEDS)
    scenes = [generate_scene(int(s), f"{split}_{s}") for s in seeds]
    if out_dir is not None:
        folder = Path(out_dir); folder.mkdir(parents=True, exist_ok=True)
        for scene in scenes:
            np.savez_compressed(folder/f"{scene.name}.npz", frames=scene.frames,
                                masks=scene.masks, seed=scene.seed)
    return scenes


def patch_labels(masks):
    """>=70% target coverage over both frames and >=65% in each frame.

    -1 denotes mixed/boundary patches; 0 denotes >=90% background/occluder.
    Target absence is determined separately from pixel masks, not this grid.
    """
    masks = np.asarray(masks)
    if masks.shape != (N_FRAMES, SIZE, SIZE):
        raise ValueError(f"Unexpected masks shape: {masks.shape}")
    blocked = masks.reshape(N_STEPS, 2, GRID, PATCH, GRID, PATCH)
    result = np.full((N_STEPS, GRID, GRID), -1, dtype=np.int8)
    for pid in range(5):
        fraction = (blocked == pid).mean(axis=(3, 5))  # time, frame, row, col
        keep = ((fraction.mean(1) >= .70) & (fraction.min(1) >= .65)
                if pid else (fraction.min(1) >= .90))
        result[keep] = pid
    return result


def rgb_features(frames):
    return np.asarray(frames, np.float32).reshape(
        N_STEPS, 2, GRID, PATCH, GRID, PATCH, 3).mean(axis=(1, 3, 5))/255.


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def match_features(features, initial_labels):
    """No scene, later annotation, trajectory or object position is accessible."""
    features = np.asarray(features, np.float32)
    if features.ndim != 4 or features.shape[:3] != (N_STEPS, GRID, GRID):
        raise ValueError(f"Expected [16,24,24,D], received {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Features contain non-finite values")
    unit = _unit(features)
    prototypes = []
    for pid in PART_NAMES:
        selected = unit[0][initial_labels == pid]
        if not len(selected):
            raise ValueError(f"Part {pid} lacks an initial unambiguous patch")
        prototypes.append(_unit(selected.mean(0)))
    similarities = np.einsum("tyxd,kd->tkyx", unit, np.stack(prototypes))
    flat = similarities.reshape(N_STEPS, 4, -1)
    index = flat.argmax(-1)
    cells = np.stack((index//GRID, index % GRID), -1)
    return Predictions(flat.max(-1), cells, np.ones(index.shape, dtype=bool))


def fixed_position(initial_labels):
    cells = []
    for pid in PART_NAMES:
        candidates = np.argwhere(initial_labels == pid)
        cells.append(candidates[np.argmin(((candidates-candidates.mean(0))**2).sum(1))])
    return Predictions(np.ones((N_STEPS, 4)), np.broadcast_to(
        np.asarray(cells), (N_STEPS, 4, 2)).copy(), np.ones((N_STEPS, 4), bool))


def optical_flow(frames, initial_mask):
    """Pyramidal Lucas-Kanade points seeded in frame 0; no later labels/reseed.

    Track initial textured points, estimate centre from their initial offsets,
    and use median displacement. Both frames must have surviving points for a
    tubelet prediction. OpenCV tracking status provides absence, not ground truth.
    """
    import cv2
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    centers = np.full((N_FRAMES, 4, 2), np.nan, np.float32)
    for ki, pid in enumerate(PART_NAMES):
        region = (initial_mask == pid).astype(np.uint8)*255
        ys, xs = np.where(region)
        center = np.array([xs.mean(), ys.mean()], np.float32)
        points = cv2.goodFeaturesToTrack(grey[0], maxCorners=12, qualityLevel=.01,
                                         minDistance=4, mask=region, blockSize=3)
        if points is None:
            continue
        offsets = center-points[:, 0]
        centers[0, ki] = center
        for t in range(1, N_FRAMES):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                grey[t-1], grey[t], points, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            if nxt is None:
                break
            good = status[:, 0].astype(bool) & np.isfinite(nxt[:, 0]).all(1)
            good &= (nxt[:, 0] >= 0).all(1) & (nxt[:, 0] < SIZE).all(1)
            if not good.any():
                break
            points, offsets = nxt[good], offsets[good]
            centers[t, ki] = np.median(points[:, 0]+offsets, axis=0)
    paired = centers.reshape(N_STEPS, 2, 4, 2)
    valid = np.isfinite(paired).all(axis=(1, 3))
    middle = np.nan_to_num(paired.mean(1), nan=0)
    cells = (middle[..., ::-1]/PATCH).astype(int)
    valid &= (cells >= 0).all(-1) & (cells < GRID).all(-1)
    return Predictions(np.ones((N_STEPS, 4)), cells.clip(0, GRID-1), valid)


def target_states(scene, labels):
    states = np.full((N_STEPS, 4), -1, dtype=np.int8)
    pixel_pairs = scene.masks.reshape(N_STEPS, 2, SIZE, SIZE)
    for ki, pid in enumerate(PART_NAMES):
        states[:, ki] = np.where((labels == pid).any(axis=(1, 2)), 1,
            np.where((pixel_pairs != pid).all(axis=(1, 2, 3)), 0, -1))
    return states  # 1 visible, 0 absent in both frames, -1 ambiguous (skip)


def calibrate_threshold(scenes, predictions):
    scores, states = [], []
    for scene in scenes:
        gt = target_states(scene, patch_labels(scene.masks))[1:].ravel()
        pred = predictions[scene.name]
        score = np.where(pred.eligible, pred.scores, -2.)[1:].ravel()
        scores.extend(score[gt >= 0]); states.extend(gt[gt >= 0])
    scores, states = np.asarray(scores), np.asarray(states)
    unique = np.unique(scores)
    candidates = np.r_[unique[0]-1e-6, (unique[:-1]+unique[1:])/2, unique[-1]+1e-6]
    values = []
    for threshold in candidates:
        emit = scores >= threshold
        terms = [float((emit[states == 1]).mean())] if (states == 1).any() else []
        if (states == 0).any():
            terms.append(float((~emit[states == 0]).mean()))
        values.append(np.mean(terms))
    # Stable conservative tie break: highest threshold with the same best score.
    best = np.flatnonzero(np.isclose(values, max(values), rtol=0, atol=1e-12))[-1]
    return float(candidates[best]), {"balanced_presence_accuracy": float(values[best]),
        "visible_samples": int((states == 1).sum()), "absent_samples": int((states == 0).sum()),
        "objective": "mean(visible presence recall, absent specificity); highest-threshold tie break"}


def score_scene(scene, pred, threshold, method):
    labels = patch_labels(scene.masks)
    states = target_states(scene, labels)
    rows = []
    for ki, pid in enumerate(PART_NAMES):
        pending_recovery = False
        for t in range(1, N_STEPS):
            state = int(states[t, ki])
            if state == 0:
                pending_recovery = True
            recovery = state == 1 and pending_recovery
            if state == 1:
                pending_recovery = False
            row, col = map(int, pred.cells[t, ki])
            present = bool(pred.eligible[t, ki] and pred.scores[t, ki] >= threshold)
            actual = int(labels[t, row, col]) if present else -2
            hit = state == 1 and present and actual == pid
            wrong_car = state == 1 and present and actual > 0 and (actual-1)//2 != (pid-1)//2
            wrong_part = state == 1 and present and actual > 0 and (actual-1) % 2 != (pid-1) % 2
            rows.append(dict(scene=scene.name, seed=scene.seed, condition=scene.condition,
                method=method, tubelet=t, first_frame=2*t, target=PART_NAMES[pid],
                state=("visible" if state == 1 else "absent" if state == 0 else "ambiguous"),
                window=("within_first_window" if t < WINDOW_STEPS else "across_window"),
                boundary=t == WINDOW_STEPS, recovery=bool(recovery), present=present,
                score=float(pred.scores[t, ki]), threshold=float(threshold),
                row=row, col=col, predicted_gt=actual, hit=bool(hit),
                wrong_car=bool(wrong_car), wrong_part=bool(wrong_part)))
    return rows


def summarize_rows(rows):
    visible = [r for r in rows if r["state"] == "visible"]
    absent = [r for r in rows if r["state"] == "absent"]
    recovery = [r for r in visible if r["recovery"]]
    def count_rate(selected, field):
        n = sum(bool(r[field]) for r in selected)
        return {"numerator": n, "denominator": len(selected),
                "rate": n/len(selected) if selected else None}
    return {"localization_accuracy_given_visible": count_rate(visible, "hit"),
            "wrong_car_given_visible": count_rate(visible, "wrong_car"),
            "wrong_part_given_visible": count_rate(visible, "wrong_part"),
            "presence_recall_given_visible": count_rate(visible, "present"),
            "false_presence_given_absent": count_rate(absent, "present"),
            "recovery_accuracy": count_rate(recovery, "hit"),
            "ambiguous_samples_skipped": sum(r["state"] == "ambiguous" for r in rows),
            "uniform_patch_chance_given_visible": float(np.mean(
                [r["chance"] for r in visible])) if visible and "chance" in visible[0] else None}


def write_video(path, frames, fps=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    try:
        process = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{w}x{h}", "-framerate", str(fps),
            "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, error = process.communicate(b"".join(np.ascontiguousarray(f).tobytes() for f in frames))
        if process.returncode:
            raise RuntimeError(error.decode()[-1000:])
    except FileNotFoundError:
        import cv2
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("No MP4 encoder available")
        for frame in frames:
            writer.write(np.ascontiguousarray(frame[..., ::-1]))
        writer.release()


def annotated_video(scene, predictions, thresholds, path):
    methods = list(predictions)
    # 2x2 comparison of emitted points; no later annotations reach the matcher.
    output = []
    for frame_index, frame in enumerate(scene.frames):
        canvas = Image.new("RGB", (SIZE*2, (SIZE+44)*2), (15, 18, 23))
        for mi, method in enumerate(methods[:4]):
            panel = Image.fromarray(frame).copy()
            draw = ImageDraw.Draw(panel)
            pred, t = predictions[method], frame_index//2
            for ki, name in enumerate(PART_NAMES.values()):
                emitted = pred.eligible[t, ki] and pred.scores[t, ki] >= thresholds[method]
                color = COLORS[ki]
                if emitted:
                    row, col = pred.cells[t, ki]
                    x, y = int((col+.5)*PATCH), int((row+.5)*PATCH)
                    draw.ellipse((x-6, y-6, x+6, y+6), outline=color, width=3)
                    draw.text((max(0, min(x+7, SIZE-130)), max(0, y-12)), name, fill=color,
                              stroke_width=1, stroke_fill=(0, 0, 0))
                else:
                    draw.text((5, 5+ki*13), f"{name}: absent", fill=color,
                              stroke_width=1, stroke_fill=(0, 0, 0))
            xoff, yoff = (mi % 2)*SIZE, (mi//2)*(SIZE+44)
            canvas.paste(panel, (xoff, yoff+44))
            header = ImageDraw.Draw(canvas)
            header.text((xoff+8, yoff+5), f"{method} | frame {frame_index} | window {t//8+1}", fill="white")
            header.text((xoff+8, yoff+20), f"{scene.name}: {scene.condition}", fill=(186, 195, 211))
        output.append(np.asarray(canvas))
    write_video(path, output)


def evaluate_features(calibration_scenes, evaluation_scenes,
                      calibration_features: Mapping[str, np.ndarray],
                      evaluation_features: Mapping[str, np.ndarray], out_dir,
                      include_flow=True, export_videos=True):
    if set(s.seed for s in calibration_scenes) & set(s.seed for s in evaluation_scenes):
        raise ValueError("Calibration and evaluation seeds overlap")
    folder = Path(out_dir); folder.mkdir(parents=True, exist_ok=True)
    scenes = list(calibration_scenes)+list(evaluation_scenes)
    features = dict(calibration_features) | dict(evaluation_features)
    predictions = {m: {} for m in ("vjepa", "rgb_patch_mean", "fixed_initial_position")}
    skipped = {}
    if include_flow:
        try:
            import cv2  # noqa: F401
            predictions["lucas_kanade_flow"] = {}
        except ImportError:
            skipped["lucas_kanade_flow"] = "OpenCV unavailable"
    for scene in scenes:
        initial = patch_labels(scene.masks)[0]
        predictions["vjepa"][scene.name] = match_features(features[scene.name], initial)
        predictions["rgb_patch_mean"][scene.name] = match_features(rgb_features(scene.frames), initial)
        predictions["fixed_initial_position"][scene.name] = fixed_position(initial)
        if "lucas_kanade_flow" in predictions:
            predictions["lucas_kanade_flow"][scene.name] = optical_flow(scene.frames, scene.masks[0])
    thresholds, calibration = {}, {}
    for method, pred in predictions.items():
        if method in ("vjepa", "rgb_patch_mean"):
            thresholds[method], calibration[method] = calibrate_threshold(calibration_scenes, pred)
        else:
            thresholds[method] = .5
            calibration[method] = {"rule": "always emit fixed location" if method == "fixed_initial_position"
                                   else "emit only while LK points survive in both tubelet frames"}
    rows = []
    for method, pred in predictions.items():
        for scene in evaluation_scenes:
            result = score_scene(scene, pred[scene.name], thresholds[method], method)
            labels = patch_labels(scene.masks)
            reverse = {v: k for k, v in PART_NAMES.items()}
            for row in result:
                row["chance"] = float((labels[row["tubelet"]] == reverse[row["target"]]).mean())
            rows.extend(result)
    report = {"experiment": "frozen video part identity diagnostic", "config": {
        "resolution": SIZE, "frames": N_FRAMES, "tubelet_frames": 2, "patch_size": PATCH,
        "independently_encoded_windows": 2, "frames_per_window": 16,
        "annotation": "first tubelet only (first frame only for LK)",
        "calibration_seeds": [s.seed for s in calibration_scenes],
        "evaluation_seeds": [s.seed for s in evaluation_scenes],
        "parts": PART_NAMES, "matching": "L2-normalized mean initial prototype; global max cosine; row-major ties",
        "visibility": "patch >=70% target over pair, >=65% in each frame; absent only if zero target pixels in both",
        "exclusions": "first tubelet; partially visible targets with no unambiguous patch",
        "wrong_identity_notes": "wrong-car and wrong-part may overlap; background/boundary errors are localization misses",
        "scope": "Small offline synthetic diagnostic; cars stay in separate lanes; no learned editing, camera change, realistic faces, 3D or product claim",
        "seed_variation": "start/end positions, peak positions, nonlinear speed, reversal and occlusion timing, occluder positions/widths, vertical jitter and body colour",
        "scenes": [{"name": s.name, "seed": s.seed, "condition": s.condition, "parameters": s.parameters} for s in scenes]},
        "thresholds": thresholds, "calibration": calibration, "skipped_methods": skipped,
        "methods": {}}
    for method in predictions:
        subset = [r for r in rows if r["method"] == method]
        report["methods"][method] = {"overall": summarize_rows(subset),
            "by_condition": {condition: summarize_rows([r for r in subset if r["condition"] == condition])
                             for condition in sorted(set(s.condition for s in evaluation_scenes))},
            "by_window": {window: summarize_rows([r for r in subset if r["window"] == window])
                          for window in ("within_first_window", "across_window")},
            "window_boundary": summarize_rows([r for r in subset if r["boundary"]])}
    (folder/"results.json").write_text(json.dumps(report, indent=2))
    if rows:
        with (folder/"predictions.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if export_videos:
        chosen = []
        for condition in ("motion", "full_occlusion"):
            chosen.extend([s for s in evaluation_scenes if s.condition == condition][:1])
        for scene in chosen:
            annotated_video(scene, {m: p[scene.name] for m, p in predictions.items()},
                            thresholds, folder/f"{scene.name}_comparison.mp4")
    return report


def self_test(out_dir):
    calibration, evaluation = generate_suite(split="calibration"), generate_suite(split="evaluation")
    fixtures = {}
    for scene in calibration+evaluation:
        labels = patch_labels(scene.masks)
        assert all((labels[0] == pid).any() for pid in PART_NAMES), (scene.seed, labels[0])
        # Perfect oracle fixtures validate the evaluator only, never actual inference.
        fixtures[scene.name] = np.eye(6, dtype=np.float32)[np.where(labels >= 0, labels, 5)]
    report = evaluate_features(calibration, evaluation, fixtures, fixtures, out_dir,
                               include_flow=False, export_videos=False)
    metrics = report["methods"]["vjepa"]["overall"]
    assert metrics["localization_accuracy_given_visible"]["rate"] == 1., metrics
    assert metrics["false_presence_given_absent"]["rate"] == 0., metrics
    assert metrics["recovery_accuracy"]["rate"] == 1., metrics
    assert metrics["recovery_accuracy"]["denominator"] > 0
    assert report["methods"]["vjepa"]["by_window"]["across_window"]["localization_accuracy_given_visible"]["denominator"] > 0
    scene = evaluation[0]
    original = fixtures[scene.name]
    corrupted = original.copy()
    corrupted[8:] = corrupted[8:, ..., [0, 3, 4, 1, 2, 5]]
    rows = score_scene(scene, match_features(corrupted, patch_labels(scene.masks)[0]), .5, "corrupted")
    assert any(r["wrong_car"] for r in rows if r["tubelet"] >= 8), "Identity swaps must be detected"
    assert summarize_rows(rows)["localization_accuracy_given_visible"]["rate"] < 1.
    # Changing later masks cannot change the matcher, whose argument is first labels.
    initial = patch_labels(scene.masks)[0]
    first = match_features(original, initial)
    copy = scene.masks.copy(); copy[2:] = 0
    second = match_features(original, patch_labels(copy)[0])
    assert np.array_equal(first.cells, second.cells)
    checks = {"status": "passed", "oracle_fixture_accuracy": 1., "identity_swap_detected": True,
              "future_labels_not_matcher_input": True, "real_model_results": False,
              "visible_denominator": metrics["localization_accuracy_given_visible"]["denominator"],
              "absent_denominator": metrics["false_presence_given_absent"]["denominator"],
              "recovery_denominator": metrics["recovery_accuracy"]["denominator"]}
    (Path(out_dir)/"self_test.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="part_consistency_output")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--features-dir", help="Directory of scene_name.npy features for both splits")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.out_dir)
    elif args.generate:
        for split in ("calibration", "evaluation"):
            generate_suite(args.out_dir, split)
        print(f"Generated {len(CALIBRATION_SEEDS)+len(EVALUATION_SEEDS)} scenes in {args.out_dir}")
    elif args.features_dir:
        cal, ev = generate_suite(split="calibration"), generate_suite(split="evaluation")
        features = {s.name: np.load(Path(args.features_dir)/f"{s.name}.npy") for s in cal+ev}
        report = evaluate_features(cal, ev, features, features, args.out_dir)
        print(json.dumps({k: v["overall"] for k, v in report["methods"].items()}, indent=2))
    else:
        parser.error("Choose --self-test, --generate, or --features-dir")
