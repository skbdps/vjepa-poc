"""Day 4 synthetic part identity benchmark and annotation-isolated evaluator.

The renderer provides exact visible-pixel masks for evaluation. Tracking code is
given only RGB, frozen encoder features, and frame-zero annotation. A car's body
colour and stripe distinguish its identity; corresponding tagged parts share
appearance. These are 2-D sprites, not evidence about faces, 3-D or real video.

Frozen protocol: 6 calibration, 6 development, 18 sealed-test clips. Every clip
has 64 frames at 384 pixels, encoded in four independent 16-frame windows.
Calibration chooses absence thresholds; development chooses configurations.
Test labels must only be read after method/configuration selection is frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

SIZE, PATCH, N_FRAMES, TUBELET = 384, 16, 64, 2
GRID, N_STEPS, WINDOW_STEPS = SIZE // PATCH, N_FRAMES // TUBELET, 8
PART_NAMES = {1: "car_A.front_door", 2: "car_A.window",
              3: "car_B.front_door", 4: "car_B.window"}
CONDITIONS = ("long_occlusion", "crossing", "scale_camera")
SPLIT_SEEDS = {
    "calibration": {c: list(range(1000 + 100*i, 1002 + 100*i)) for i, c in enumerate(CONDITIONS)},
    "development": {c: list(range(2000 + 100*i, 2002 + 100*i)) for i, c in enumerate(CONDITIONS)},
    "test": {c: list(range(3000 + 100*i, 3006 + 100*i)) for i, c in enumerate(CONDITIONS)},
}
COLORS = [(255, 88, 88), (255, 219, 74), (72, 226, 158), (128, 164, 255)]
PROTOCOL_VERSION = "day4_v1"


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
    scores: np.ndarray                 # [tubelets, 4]
    cells: np.ndarray                  # [tubelets, 4, 2], row and column
    eligible: np.ndarray               # [tubelets, 4], algorithm may emit


def _car_sprite(identity, variant):
    sprite = Image.new("RGBA", (168, 132), (0, 0, 0, 0))
    mask = Image.new("L", sprite.size, 0)
    draw, labels = ImageDraw.Draw(sprite), ImageDraw.Draw(mask)
    body = (49, 112 + variant, 118, 255) if identity == 0 else (129, 68, 85 + variant, 255)
    draw.rounded_rectangle((0, 30, 166, 106), 11, fill=body)
    draw.polygon([(24, 33), (43, 0), (131, 0), (153, 33)], fill=body)
    for wx in (30, 134):
        draw.ellipse((wx-17, 91, wx+17, 125), fill=(26, 29, 33))
        draw.ellipse((wx-8, 100, wx+8, 116), fill=(165, 170, 178))
    draw.rectangle((29, 10, 73, 40), fill=(121, 155, 175))
    draw.rectangle((24, 46, 74, 99), fill=(174, 172, 144))
    # The same texture, colour and geometry on both cars: no part-ID colour shortcut.
    for kind, rect in enumerate(((87, 45, 145, 100), (85, 4, 141, 39))):
        pid = 1 + identity*2 + kind
        fill = (191, 177, 123) if kind == 0 else (113, 168, 199)
        draw.rectangle(rect, fill=fill)
        labels.rectangle(rect, fill=pid)
        for dx, dy in ((9, 8), (30, 8), (9, 22), (30, 22)):
            px, py = rect[0]+dx, rect[1]+dy
            draw.rectangle((px, py, px+4, py+4), fill=tuple(c-25 for c in fill))
    draw.line((80, 41, 80, 101), fill=(35, 45, 52), width=2)
    draw.rectangle((128, 49, 139, 52), fill=(231, 234, 221))
    if identity == 0:
        draw.line((8, 78, 21, 78), fill=(233, 233, 220), width=4)
    else:
        draw.line((8, 67, 21, 85), fill=(233, 233, 220), width=4)
    return sprite, mask


def _paste_car(image, mask, sprite_pair, x, y, scale=1., angle=0.):
    sprite, labels = sprite_pair
    if scale != 1.:
        shape = tuple(int(round(v*scale)) for v in sprite.size)
        sprite = sprite.resize(shape, Image.Resampling.BICUBIC)
        labels = labels.resize(shape, Image.Resampling.NEAREST)
    if angle:
        sprite = sprite.rotate(angle, Image.Resampling.BICUBIC, expand=True)
        labels = labels.rotate(angle, Image.Resampling.NEAREST, expand=True)
    pos = (int(round(x)), int(round(y)))
    alpha = sprite.getchannel("A")
    image.paste(sprite.convert("RGB"), pos, alpha)
    # The entire opaque sprite occludes earlier cars, including unlabelled body.
    opaque = alpha.point(lambda p: 255 if p >= 128 else 0)
    mask.paste(labels, pos, opaque)


def _smooth(p):
    return p*p*(3.-2.*p)


def generate_scene(seed: int, condition: str, name: str | None = None) -> Scene:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(-.15, .15))
    variant = int(rng.integers(-14, 15))
    x0, xb0 = int(rng.integers(2, 10)), int(rng.integers(201, 210))
    ya, yb = int(rng.integers(40, 48)), int(rng.integers(226, 236))
    peak = int(rng.integers(189, 207))
    occluder_x = int(rng.integers(174, 184))
    turn_start, turn_end = float(rng.uniform(.29, .35)), float(rng.uniform(.62, .69))
    motion_power = float(rng.uniform(.82, 1.22))
    camera_amp = float(rng.uniform(9., 17.))
    scale_amp = float(rng.uniform(.09, .15))
    rotation_amp = float(rng.uniform(4., 8.))
    parameters = dict(phase=phase, body_color_variant=variant, a_start_x=x0,
        b_start_x=xb0, a_start_y=ya, b_start_y=yb, a_peak_x=peak,
        occluder_x=occluder_x, turn_start=turn_start, turn_end=turn_end,
        motion_power=motion_power, camera_amplitude=camera_amp,
        scale_amplitude=scale_amp, rotation_degrees=rotation_amp,
        foreground_car="B", protocol=PROTOCOL_VERSION,
        source_annotation="frame_zero_only", first_four_frames_stationary=True)
    sprites = [_car_sprite(i, variant) for i in range(2)]
    frames, masks = [], []
    for t in range(N_FRAMES):
        # Stationary annotation prefix avoids a source point crossing a patch edge.
        p = max(0., (t-3)/(N_FRAMES-4))
        ease = _smooth(p**motion_power)
        camx = camera_amp*np.sin(2*np.pi*p) if condition == "scale_camera" else 0.
        camy = 5*np.sin(4*np.pi*p) if condition == "scale_camera" else 0.
        image = Image.new("RGB", (SIZE, SIZE), (222, 229, 232))
        mask = Image.new("L", (SIZE, SIZE), 0)
        draw = ImageDraw.Draw(image)
        for road_y in (181, 368):
            draw.rectangle((0, road_y+camy, SIZE, road_y+7+camy), fill=(132, 141, 147))
        for xx in range(-40, SIZE+40, 56):
            draw.rectangle((xx+camx, 375+camy, xx+28+camx, 379+camy), fill=(248, 246, 231))
        # Repeated matching-colour background panels, never labelled as targets.
        for j in range(3):
            dx = (28 + 111*j + 22*np.sin(2*np.pi*p+phase+j) + camx)
            draw.rectangle((dx, 192+camy, dx+47, 214+camy), fill=(191, 177, 123))
            draw.rectangle((dx+4, 195+camy, dx+17, 204+camy), fill=(113, 168, 199))
            draw.rectangle((dx+30, 200+camy, dx+34, 204+camy), fill=(166, 152, 98))
        scale_a = scale_b = 1.
        angle_a = angle_b = 0.
        if condition == "long_occlusion":
            if p < turn_start:
                progress = _smooth(p/turn_start)
                ax = x0+(peak-x0)*progress
            elif p < turn_end:
                ax = peak + 4*np.sin((p-turn_start)*10)
            else:
                progress = _smooth((p-turn_end)/(1-turn_end))
                ax = peak+(x0+13-peak)*progress
            ay, bx, by = ya, xb0-(xb0-12)*ease, yb
        elif condition == "crossing":
            ax, bx = x0+(xb0-x0)*ease, xb0-(xb0-x0)*ease
            ay, by = ya+(yb-ya)*ease, yb-(yb-ya)*ease
            # Distinct trajectories and persistent body cues make recovery identifiable.
            ax += 8*np.sin(np.pi*p)
            bx -= 7*np.sin(np.pi*p)
        else:
            ax = x0+125*np.sin(np.pi*p)**2
            bx = xb0-138*ease
            ay, by = ya, yb-9*np.sin(np.pi*p)
            scale_a = 1.+scale_amp*np.sin(2*np.pi*p)
            scale_b = 1.-scale_amp*np.sin(2*np.pi*p)
            angle_a = rotation_amp*np.sin(2*np.pi*p)
            angle_b = -rotation_amp*np.sin(2*np.pi*p)
        _paste_car(image, mask, sprites[0], ax+camx, ay+camy, scale_a, angle_a)
        _paste_car(image, mask, sprites[1], bx+camx, by+camy, scale_b, angle_b)
        if condition == "long_occlusion":
            draw, labels = ImageDraw.Draw(image), ImageDraw.Draw(mask)
            rect = (occluder_x, 25, 383, 180)
            draw.rectangle(rect, fill=(89, 98, 111))
            labels.rectangle(rect, fill=0)
            for xx in range(occluder_x+4, 384, 12):
                draw.line((xx, 25, xx, 180), fill=(101, 110, 121), width=2)
        frames.append(np.asarray(image).copy())
        masks.append(np.asarray(mask).copy())
    scene = Scene(name or f"{condition}_{seed}", np.stack(frames), np.stack(masks),
                  PART_NAMES.copy(), int(seed), condition, parameters)
    initial = initial_labels(scene.masks[0])
    if not all((initial == pid).any() for pid in PART_NAMES):
        raise RuntimeError(f"Source annotation lacks a usable patch: {scene.name}")
    return scene


def generate_suite(split="calibration", out_dir=None):
    if split not in SPLIT_SEEDS:
        raise ValueError(f"split must be one of {tuple(SPLIT_SEEDS)}")
    scenes = [generate_scene(seed, condition, f"{split}_{condition}_{seed}")
              for condition in CONDITIONS for seed in SPLIT_SEEDS[split][condition]]
    if out_dir is not None:
        folder = Path(out_dir); folder.mkdir(parents=True, exist_ok=True)
        for scene in scenes:
            np.savez_compressed(folder/f"{scene.name}.npz", frames=scene.frames,
                                masks=scene.masks, seed=scene.seed, condition=scene.condition)
    return scenes


def initial_labels(mask):
    """Source labels from frame ZERO only; >=70% coverage of a 16-pixel patch."""
    mask = np.asarray(mask)
    if mask.shape != (SIZE, SIZE):
        raise ValueError(f"Expected one frame-zero mask; received {mask.shape}")
    blocked = mask.reshape(GRID, PATCH, GRID, PATCH)
    result = np.full((GRID, GRID), -1, dtype=np.int8)
    for pid in range(5):
        fraction = (blocked == pid).mean(axis=(1, 3))
        result[fraction >= (.70 if pid else .90)] = pid
    return result


def patch_labels(masks):
    """>=70% target coverage over the pair, >=65% in EACH frame; -1 ambiguous."""
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[1:] != (SIZE, SIZE) or len(masks) % TUBELET:
        raise ValueError(f"Unexpected masks shape: {masks.shape}")
    blocked = masks.reshape(len(masks)//TUBELET, TUBELET, GRID, PATCH, GRID, PATCH)
    result = np.full((len(masks)//TUBELET, GRID, GRID), -1, dtype=np.int8)
    for pid in range(5):
        fraction = (blocked == pid).mean(axis=(3, 5))
        keep = ((fraction.mean(1) >= .70) & (fraction.min(1) >= .65)
                if pid else (fraction.min(1) >= .90))
        result[keep] = pid
    return result


def rgb_features(frames):
    frames = np.asarray(frames, np.float32)
    return frames.reshape(len(frames)//2, 2, GRID, PATCH, GRID, PATCH, 3).mean(axis=(1, 3, 5))/255.


def target_states(scene, labels=None):
    labels = patch_labels(scene.masks) if labels is None else labels
    nt = len(labels)
    states = np.full((nt, len(PART_NAMES)), -1, dtype=np.int8)
    pairs = scene.masks.reshape(nt, 2, SIZE, SIZE)
    for ki, pid in enumerate(PART_NAMES):
        states[:, ki] = np.where((labels == pid).any(axis=(1, 2)), 1,
            np.where((pairs != pid).all(axis=(1, 2, 3)), 0, -1))
    return states


def validate_predictions(pred, nt):
    if pred.scores.shape != (nt, 4) or pred.cells.shape != (nt, 4, 2) or pred.eligible.shape != (nt, 4):
        raise ValueError("Predictions must have [tubelets,4] scores/eligible and [tubelets,4,2] cells")
    if not np.isfinite(pred.scores).all():
        raise ValueError("Nonfinite prediction confidence")
    if not np.isfinite(pred.cells).all() or not ((pred.cells >= 0) & (pred.cells < GRID)).all():
        raise ValueError("Prediction cell outside feature grid")


def calibrate_threshold(scenes, predictions):
    """Calibration-only global absence threshold; equal presence/absence class weight."""
    if any(not scene.name.startswith("calibration_") for scene in scenes):
        raise ValueError("Absence thresholds may only be calibrated on calibration scenes")
    scores, eligible, states = [], [], []
    for scene in scenes:
        gt = target_states(scene)[1:].ravel()
        pred = predictions[scene.name]
        validate_predictions(pred, len(scene.frames)//2)
        keep = gt >= 0
        scores.extend(pred.scores[1:].ravel()[keep])
        eligible.extend(pred.eligible[1:].ravel()[keep])
        states.extend(gt[keep])
    scores, eligible, states = np.asarray(scores), np.asarray(eligible, bool), np.asarray(states)
    if not len(scores):
        raise ValueError("No calibration observations")
    unique = np.unique(scores)
    candidates = np.r_[unique[0]-1e-6, (unique[:-1]+unique[1:])/2, unique[-1]+1e-6]
    values = []
    for threshold in candidates:
        emit = eligible & (scores >= threshold)
        terms = [float(emit[states == 1].mean())] if (states == 1).any() else []
        if (states == 0).any():
            terms.append(float((~emit[states == 0]).mean()))
        values.append(float(np.mean(terms)))
    best = np.flatnonzero(np.isclose(values, max(values), rtol=0, atol=1e-12))[-1]
    return float(candidates[best]), {"balanced_presence_accuracy": float(values[best]),
        "visible_samples": int((states == 1).sum()), "absent_samples": int((states == 0).sum()),
        "objective": "mean(visible presence recall, absent specificity); highest-threshold tie break"}


def score_scene(scene, pred, threshold, method):
    labels = patch_labels(scene.masks)
    states = target_states(scene, labels)
    validate_predictions(pred, len(labels))
    rows = []
    for ki, pid in enumerate(PART_NAMES):
        pending_recovery = False
        previous_identified_car = (pid-1)//2
        for t in range(1, len(labels)):
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
            # Changes between identified cars; preserve last identity over absent/ambiguous outputs.
            id_switch = False
            if state == 1 and present and actual > 0:
                identified_car = (actual-1)//2
                id_switch = identified_car != previous_identified_car
                previous_identified_car = identified_car
            rows.append(dict(scene=scene.name, seed=scene.seed, condition=scene.condition,
                method=method, tubelet=t, first_frame=2*t, target=PART_NAMES[pid],
                state="visible" if state == 1 else "absent" if state == 0 else "ambiguous",
                window=t//WINDOW_STEPS+1, boundary=t % WINDOW_STEPS == 0,
                recovery=bool(recovery), present=present, score=float(pred.scores[t, ki]),
                threshold=float(threshold), row=row, col=col, predicted_gt=actual,
                hit=bool(hit), wrong_car=bool(wrong_car), wrong_part=bool(wrong_part),
                id_switch=bool(id_switch), chance=float((labels[t] == pid).mean())))
    return rows


def summarize_rows(rows):
    visible = [r for r in rows if r["state"] == "visible"]
    absent = [r for r in rows if r["state"] == "absent"]
    recovery = [r for r in visible if r["recovery"]]
    def count_rate(selected, field):
        n = sum(bool(r[field]) for r in selected)
        return {"numerator": n, "denominator": len(selected),
                "rate": n/len(selected) if selected else None}
    accuracy = count_rate(visible, "hit")
    false_presence = count_rate(absent, "present")
    balanced_localization = None
    if accuracy["rate"] is not None and false_presence["rate"] is not None:
        balanced_localization = .5*(accuracy["rate"]+1-false_presence["rate"])
    return {"localization_accuracy_given_visible": accuracy,
            "wrong_car_given_visible": count_rate(visible, "wrong_car"),
            "wrong_part_given_visible": count_rate(visible, "wrong_part"),
            "presence_recall_given_visible": count_rate(visible, "present"),
            "false_presence_given_absent": false_presence,
            "recovery_accuracy": count_rate(recovery, "hit"),
            "identity_switches": sum(r.get("id_switch", False) for r in rows),
            "balanced_localization_and_absence": balanced_localization,
            "clips": len({r["scene"] for r in rows}),
            "ambiguous_samples_skipped": sum(r["state"] == "ambiguous" for r in rows),
            "uniform_patch_chance_given_visible": float(np.mean([r["chance"] for r in visible])) if visible else None}


def summarize_with_ci(rows, resamples=2000, seed=914):
    """Percentile 95% CI, resampling whole clips; frames are not independent."""
    result = summarize_rows(rows)
    names = sorted({r["scene"] for r in rows})
    if not names:
        return result
    counts = [summarize_rows([r for r in rows if r["scene"] == name]) for name in names]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(resamples, len(names)))
    for metric, value in result.items():
        if not isinstance(value, dict) or "numerator" not in value:
            continue
        nums = np.array([c[metric]["numerator"] for c in counts])
        dens = np.array([c[metric]["denominator"] for c in counts])
        sampled_den = dens[draws].sum(1)
        valid = sampled_den > 0
        sampled = nums[draws].sum(1)[valid]/sampled_den[valid]
        value["clip_bootstrap_95ci"] = np.quantile(sampled, [.025, .975]).tolist() if len(sampled) else None
    result["bootstrap"] = {"unit": "clip", "resamples": resamples, "seed": seed,
        "interpretation": "descriptive interval over generated clips, not real-video generalization"}
    return result


def paired_bootstrap_difference(rows_a, rows_b, metric="localization_accuracy_given_visible", resamples=2000, seed=914):
    """Paired whole-clip difference A minus B; report alongside absolute results."""
    names = sorted({r["scene"] for r in rows_a})
    if set(names) != {r["scene"] for r in rows_b}:
        raise ValueError("Paired comparison requires identical scenes")
    ca = [summarize_rows([r for r in rows_a if r["scene"] == n])[metric] for n in names]
    cb = [summarize_rows([r for r in rows_b if r["scene"] == n])[metric] for n in names]
    na, da = np.array([c["numerator"] for c in ca]), np.array([c["denominator"] for c in ca])
    nb, db = np.array([c["numerator"] for c in cb]), np.array([c["denominator"] for c in cb])
    draws = np.random.default_rng(seed).integers(0, len(names), (resamples, len(names)))
    valid = (da[draws].sum(1) > 0) & (db[draws].sum(1) > 0)
    difference = (na[draws].sum(1)[valid]/da[draws].sum(1)[valid]
                  - nb[draws].sum(1)[valid]/db[draws].sum(1)[valid])
    return {"metric": metric, "difference_a_minus_b": float(na.sum()/da.sum()-nb.sum()/db.sum()),
            "clip_bootstrap_95ci": np.quantile(difference, [.025, .975]).tolist() if len(difference) else None,
            "resamples": resamples, "seed": seed, "unit": "paired clip"}


def benchmark_manifest():
    return {"protocol": PROTOCOL_VERSION, "resolution": SIZE, "frames": N_FRAMES,
        "tubelet_frames": TUBELET, "patch_size": PATCH, "parts": PART_NAMES,
        "split_seeds": SPLIT_SEEDS, "annotation": "frame-zero mask only",
        "test_policy": "Freeze thresholds and algorithm configuration before opening sealed-test metrics",
        "visibility": "at least one patch >=70% target over pair and >=65% each frame; absence requires zero target pixels both frames",
        "exclusions": "first tubelet and visible slivers with no eligible patch",
        "identity_switch": "transition between predicted car IDs on unambiguous visible targets, initialized to true source ID; last ID retained over missing outputs",
        "conditions": {
            "long_occlusion": "stationary foreground wall hides A for extended interval; return and reappearance; B moves separately",
            "crossing": "two distinct-bodied cars exchange diagonal positions; foreground B occludes A; visible context resolves identity",
            "scale_camera": "up to 15% sprite scale, 8-degree in-plane rotation, global camera translation, moving repeated-colour distractors"},
        "limitations": "2-D rendered sprites, fixed same-colour target appearance; no 3-D viewpoint change, real faces, generative rendering or causal encoder claim"}


def write_video(path, frames, fps=12):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    process = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{w}x{h}", "-framerate", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, error = process.communicate(b"".join(np.ascontiguousarray(f).tobytes() for f in frames))
    if process.returncode:
        raise RuntimeError(error.decode()[-1000:])


def annotated_video(scene, predictions, thresholds, path):
    methods = list(predictions)[:4]
    panels = len(methods)
    if not panels:
        raise ValueError("No predictions to visualize")
    width = SIZE*min(2, panels)
    height = (SIZE+44)*((panels+1)//2)
    output = []
    for frame_index, frame in enumerate(scene.frames):
        canvas = Image.new("RGB", (width, height), (15, 18, 23))
        for mi, method in enumerate(methods):
            panel = Image.fromarray(frame).copy(); draw = ImageDraw.Draw(panel)
            pred, t = predictions[method], frame_index//2
            for ki, name in enumerate(PART_NAMES.values()):
                emitted = pred.eligible[t, ki] and pred.scores[t, ki] >= thresholds[method]
                color = COLORS[ki]
                if emitted:
                    row, col = pred.cells[t, ki]
                    x, y = int((col+.5)*PATCH), int((row+.5)*PATCH)
                    draw.ellipse((x-6, y-6, x+6, y+6), outline=color, width=3)
                    draw.text((max(0, min(x+7, SIZE-125)), max(0, y-12)), name, fill=color,
                              stroke_width=1, stroke_fill=(0, 0, 0))
                else:
                    draw.text((5, 5+ki*13), f"{name}: absent", fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
            xoff, yoff = (mi % 2)*SIZE, (mi//2)*(SIZE+44)
            canvas.paste(panel, (xoff, yoff+44)); header = ImageDraw.Draw(canvas)
            header.text((xoff+8, yoff+5), f"{method} | frame {frame_index} | window {t//8+1}", fill="white")
            header.text((xoff+8, yoff+20), f"{scene.name}", fill=(186, 195, 211))
        output.append(np.asarray(canvas))
    write_video(path, output)


def self_test(out_dir=None):
    """Renderer/scorer plumbing only; never a real-model result or tuning signal."""
    all_seeds = [s for groups in SPLIT_SEEDS.values() for seeds in groups.values() for s in seeds]
    assert len(all_seeds) == len(set(all_seeds)) == 30
    scenes = generate_suite("calibration")
    predictions, totals = {}, {"visible": 0, "absent": 0, "recovery": 0, "ambiguous": 0}
    deterministic = generate_scene(scenes[0].seed, scenes[0].condition, scenes[0].name)
    assert np.array_equal(scenes[0].frames, deterministic.frames)
    assert np.array_equal(scenes[0].masks, deterministic.masks)
    for scene in scenes:
        labels = patch_labels(scene.masks)
        assert all((initial_labels(scene.masks[0]) == pid).any() for pid in PART_NAMES)
        cells = np.zeros((N_STEPS, 4, 2), dtype=int)
        scores = np.zeros((N_STEPS, 4), dtype=np.float32)
        for t in range(N_STEPS):
            for ki, pid in enumerate(PART_NAMES):
                positions = np.argwhere(labels[t] == pid)
                if len(positions):
                    cells[t, ki] = positions[len(positions)//2]
                    scores[t, ki] = 1.
        pred = Predictions(scores, cells, np.ones_like(scores, bool))
        predictions[scene.name] = pred
        rows = score_scene(scene, pred, .5, "oracle_fixture_only")
        metrics = summarize_rows(rows)
        assert metrics["localization_accuracy_given_visible"]["rate"] == 1.
        if metrics["false_presence_given_absent"]["denominator"]:
            assert metrics["false_presence_given_absent"]["rate"] == 0.
        if metrics["recovery_accuracy"]["denominator"]:
            assert metrics["recovery_accuracy"]["rate"] == 1.
        totals["visible"] += metrics["localization_accuracy_given_visible"]["denominator"]
        totals["absent"] += metrics["false_presence_given_absent"]["denominator"]
        totals["recovery"] += metrics["recovery_accuracy"]["denominator"]
        totals["ambiguous"] += metrics["ambiguous_samples_skipped"]
    threshold, calibration = calibrate_threshold(scenes, predictions)
    assert 0. < threshold <= 1. and calibration["balanced_presence_accuracy"] == 1.
    scene = scenes[0]; pred = predictions[scene.name]
    swapped = Predictions(pred.scores[:, [2, 3, 0, 1]], pred.cells[:, [2, 3, 0, 1]], pred.eligible.copy())
    assert summarize_rows(score_scene(scene, swapped, .5, "swap_fixture_only"))["wrong_car_given_visible"]["numerator"] > 0
    assert totals["absent"] > 0 and totals["recovery"] > 0
    initial = initial_labels(scene.masks[0])
    altered = scene.masks.copy(); altered[1:] = 0
    assert np.array_equal(initial, initial_labels(altered[0]))
    output = {"status": "passed", "real_model_results": False, "deterministic_render": True,
        "split_seeds_disjoint": True, "oracle_fixture_accuracy": 1.,
        "identity_swap_detected": True, "initial_annotation_uses_only_frame_zero": True,
        "calibration_denominators": totals,
        "representative_frame_sha256": hashlib.sha256(scenes[0].frames.tobytes()).hexdigest()}
    if out_dir is not None:
        folder = Path(out_dir); folder.mkdir(parents=True, exist_ok=True)
        (folder/"benchmark_self_test.json").write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out-dir", default="day4_benchmark_check")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.out_dir), indent=2))
    if args.preview:
        for condition in CONDITIONS:
            scene = generate_scene(SPLIT_SEEDS["calibration"][condition][0], condition)
            write_video(Path(args.out_dir)/f"{scene.name}.mp4", scene.frames)
