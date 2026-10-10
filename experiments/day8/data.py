"""Paired controlled scenes for the direct JEPA latent-intervention experiment.

Each target video differs from its source ONLY by an exact horizontal
translation of the selected ball. Appearance, vertical trajectory, distractor,
and static textured background are preserved pixel for pixel. The target and
its masks are evaluation/training supervision, not inputs to an edit operator.
Source masks and the requested displacement are an explicit oracle-selection
budget. This renderer tests a controlled 2-D hypothesis, not real-video editing.

No SAM, tracking model, learned renderer, or downloaded image is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_VERSION = "day8_paired_latent_translation_v1"
N_FRAMES, SIZE, PATCH, TUBELET = 32, 384, 16, 2
GRID, N_STEPS = SIZE // PATCH, N_FRAMES // TUBELET
TRAIN_OFFSETS = (32, -32, 64, -64)
HELDOUT_OFFSETS = (48, -48, 80, -80)
SPLIT_CONFIG = {"train": (11000, 24), "dev": (11100, 8), "test": (11200, 16)}


def scene_specs(split: str) -> list[dict]:
    """Return a frozen, JSON-serializable manifest without rendering any scene."""
    if split not in SPLIT_CONFIG:
        raise ValueError(f"Unknown split {split!r}; expected {tuple(SPLIT_CONFIG)}")
    first_seed, count = SPLIT_CONFIG[split]
    specs = []
    for index in range(count):
        # Test has eight independent scenes at familiar magnitudes and eight
        # independent scenes at held-out magnitudes. Directions are balanced.
        heldout = split == "test" and index >= count // 2
        offsets = HELDOUT_OFFSETS if heldout else TRAIN_OFFSETS
        dx = offsets[index % len(offsets)]
        seed = first_seed + index
        specs.append({"name": f"{split}_{seed}_dx{dx:+d}", "seed": seed,
                      "dx": dx, "split": split,
                      "shift_regime": "heldout_magnitude" if heldout else "seen_magnitude"})
    return specs


def _triangle_walk(low: float, high: float, speed: float, phase: float) -> np.ndarray:
    """A bounded constant-speed trajectory with exact reflected bounces."""
    span = high - low
    if span <= 0:
        raise ValueError("Trajectory range must be positive")
    distance = np.mod(phase + np.arange(N_FRAMES) * speed, 2 * span)
    return np.rint(low + span - np.abs(distance - span)).astype(np.int32)


def _background(rng: np.random.Generator) -> np.ndarray:
    """One static background with low-frequency and fine spatial information."""
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    base = rng.uniform(38, 95, size=3)
    angles = rng.uniform(0, 2 * np.pi, size=3)
    texture = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    for channel, angle in enumerate(angles):
        a = xx * np.cos(angle) + yy * np.sin(angle)
        b = -xx * np.sin(angle) + yy * np.cos(angle)
        texture[..., channel] = (base[channel]
            + 11 * np.sin(a / rng.uniform(15, 31) + rng.uniform(0, 6))
            + 7 * np.cos(b / rng.uniform(8, 20) + rng.uniform(0, 6)))
    # Random rectangles and fine texture prevent empty-background restoration
    # from being equivalent to inserting a single globally known color.
    for _ in range(18):
        x0, y0 = rng.integers(0, SIZE - 25, size=2)
        width, height = rng.integers(15, 90, size=2)
        texture[y0:min(SIZE, y0 + height), x0:min(SIZE, x0 + width)] += rng.uniform(-16, 16, size=3)
    texture += rng.normal(0, 2.0, size=(SIZE, SIZE, 1))
    return np.clip(np.rint(texture), 0, 255).astype(np.uint8)


def _sprite(radius: int, rng: np.random.Generator, palette: np.ndarray,
            style: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed ball appearance in object coordinates, shared by every frame."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    mask = xx * xx + yy * yy <= radius * radius
    angle = rng.uniform(0, np.pi)
    a = xx * np.cos(angle) + yy * np.sin(angle)
    b = -xx * np.sin(angle) + yy * np.cos(angle)
    period = rng.uniform(6, 11)
    if style == 0:
        pattern = np.sin(a / period * np.pi) > 0
    else:
        pattern = (np.floor((a + radius) / period) + np.floor((b + radius) / period)) % 2 == 0
    image = np.where(pattern[..., None], palette[0], palette[1]).astype(np.float32)
    # Asymmetric fixed spot disambiguates object content from a uniform disk.
    spot = ((xx + radius * 0.28) ** 2 + (yy - radius * 0.24) ** 2) < (radius * 0.22) ** 2
    image[spot] = palette[2]
    shade = 0.74 + 0.26 * np.sqrt(np.maximum(0, 1 - (xx * xx + yy * yy) / radius**2))
    image *= shade[..., None]
    return np.clip(np.rint(image), 0, 255).astype(np.uint8), mask


def _draw(frame: np.ndarray, mask_out: np.ndarray, sprite: np.ndarray,
          support: np.ndarray, x: int, y: int) -> None:
    radius = support.shape[0] // 2
    if min(x, y) < radius or max(x, y) >= SIZE - radius:
        raise ValueError("A ball would be cropped")
    window = frame[y - radius:y + radius + 1, x - radius:x + radius + 1]
    window[support] = sprite[support]
    mask_out[y - radius:y + radius + 1, x - radius:x + radius + 1] = support


def generate_pair(spec: dict) -> dict:
    """Render source, counterfactual target, absent-object control, and masks.

    Returned arrays are uint8 RGB [32,384,384,3] and boolean masks [32,384,384].
    ``frames_empty`` retains the unchanged moving distractor; only the selected
    ball is absent. Accessing its features is an ORACLE diagnostic, not an edit
    available from ordinary source-video inputs. The metadata is not a model
    input and records exact desired geometry for renderer validation/scoring.
    """
    seed, dx = int(spec["seed"]), int(spec["dx"])
    if dx == 0 or dx % PATCH or abs(dx) > max(map(abs, HELDOUT_OFFSETS)):
        raise ValueError("Expected nonzero patch-aligned horizontal shift of at most 80 pixels")
    rng = np.random.default_rng(seed)
    background = _background(rng)
    radius, distractor_radius = map(int, rng.integers(23, 32, size=2))
    # Source geometry has the same distribution for all requested offsets.
    # It accommodates both signs and all magnitudes without displacement-based
    # resampling or boundary-dependent trajectory shortcuts.
    x_low = float(rng.integers(radius + 88, radius + 108))
    x_high = float(rng.integers(SIZE - radius - 108, SIZE - radius - 88))
    y_low = float(rng.integers(40, 59))
    y_high = float(rng.integers(128, 152))
    speed_x, speed_y = rng.uniform(3.2, 6.8, size=2)
    x = _triangle_walk(x_low, x_high, speed_x, rng.uniform(0, 2 * (x_high - x_low)))
    y = _triangle_walk(y_low, y_high, speed_y, rng.uniform(0, 2 * (y_high - y_low)))
    # Vertical lanes rule out occlusion in this first intervention experiment.
    # Random lane assignment avoids assigning selected-object identity a fixed
    # absolute upper/lower position. This restriction is disclosed in manifest.
    xd_low = float(rng.integers(distractor_radius + 12, 90))
    xd_high = float(rng.integers(285, SIZE - distractor_radius - 12))
    yd_low = float(rng.integers(231, 251))
    yd_high = float(rng.integers(315, 337))
    xd = _triangle_walk(xd_low, xd_high, rng.uniform(4, 8), rng.uniform(0, 2 * (xd_high - xd_low)))
    yd = _triangle_walk(yd_low, yd_high, rng.uniform(2.8, 6), rng.uniform(0, 2 * (yd_high - yd_low)))
    vertical_flip = bool(rng.integers(0, 2))
    if vertical_flip:
        y, yd = SIZE - 1 - y, SIZE - 1 - yd
    # Hue roles are randomly exchanged; selected-vs-distractor is not a color
    # label shared across scenes. The pair remains visibly distinguishable.
    warm = np.array([[238, 99, 57], [153, 45, 119], [252, 231, 156]], dtype=np.float32)
    cool = np.array([[55, 194, 226], [54, 79, 181], [154, 245, 207]], dtype=np.float32)
    warm += rng.uniform(-20, 20, size=(3, 3))
    cool += rng.uniform(-20, 20, size=(3, 3))
    color_swap = bool(rng.integers(0, 2))
    palettes = (cool, warm) if color_swap else (warm, cool)
    style = int(rng.integers(0, 2))
    selected_rgb, selected_mask = _sprite(radius, rng, palettes[0], style)
    distractor_rgb, distractor_mask = _sprite(distractor_radius, rng, palettes[1], 1 - style)
    frames_empty = np.broadcast_to(background, (N_FRAMES, SIZE, SIZE, 3)).copy()
    masks_distractor = np.zeros((N_FRAMES, SIZE, SIZE), dtype=bool)
    for t in range(N_FRAMES):
        _draw(frames_empty[t], masks_distractor[t], distractor_rgb, distractor_mask, int(xd[t]), int(yd[t]))
    frames_source, frames_target = frames_empty.copy(), frames_empty.copy()
    masks_source = np.zeros_like(masks_distractor)
    masks_target = np.zeros_like(masks_distractor)
    for t in range(N_FRAMES):
        _draw(frames_source[t], masks_source[t], selected_rgb, selected_mask, int(x[t]), int(y[t]))
        _draw(frames_target[t], masks_target[t], selected_rgb, selected_mask, int(x[t] + dx), int(y[t]))
    if np.any((masks_source | masks_target) & masks_distractor):
        raise AssertionError("Renderer created selected/distractor overlap")
    metadata = {**spec, "data_version": DATA_VERSION, "selected_radius": radius,
                "distractor_radius": distractor_radius,
                "selected_source_xy": np.stack([x, y], axis=1).tolist(),
                "selected_target_xy": np.stack([x + dx, y], axis=1).tolist(),
                "distractor_xy": np.stack([xd, yd], axis=1).tolist(),
                "vertical_lane_flip": vertical_flip, "color_role_swap": color_swap,
                "selected_texture_style": style, "horizontal_speed": float(speed_x),
                "vertical_speed": float(speed_y),
                "background_sha256": hashlib.sha256(background.tobytes()).hexdigest(),
                "selected_sprite_sha256": hashlib.sha256(selected_rgb.tobytes()).hexdigest(),
                "oracle_selection": "Full source-mask sequence and requested dx; target masks are scoring only",
                "frames_empty_role": "Optional oracle erasure diagnostic; never an ordinary source-only input"}
    return {"frames_source": frames_source, "frames_target": frames_target,
            "frames_empty": frames_empty, "masks_source": masks_source,
            "masks_target": masks_target, "masks_distractor": masks_distractor,
            "metadata": metadata}


def patch_fractions(masks: np.ndarray) -> np.ndarray:
    """Mean pixel coverage per temporal tubelet and 16x16 patch, [16,24,24]."""
    masks = np.asarray(masks)
    if masks.shape != (N_FRAMES, SIZE, SIZE) or masks.dtype != np.bool_:
        raise ValueError(f"Expected boolean [{N_FRAMES},{SIZE},{SIZE}] masks")
    return masks.reshape(N_STEPS, TUBELET, GRID, PATCH, GRID, PATCH).mean(axis=(1, 3, 5)).astype(np.float32)


def manifest() -> dict:
    return {"data_version": DATA_VERSION, "frames": N_FRAMES, "resolution": SIZE,
            "patch": PATCH, "tubelet": TUBELET, "encoding_block_frames": 16,
            "specs": {split: scene_specs(split) for split in SPLIT_CONFIG},
            "train_shift_pixels": list(TRAIN_OFFSETS),
            "heldout_shift_pixels": list(HELDOUT_OFFSETS),
            "geometry": "Moving reflected trajectories; randomized separate vertical lanes; no occlusion or crop",
            "identity": "Per-scene textured circles, constant radii and object-coordinate texture across frames",
            "source_inputs": "Source RGB-derived features, full oracle source masks, requested signed dx",
            "supervision_only": "Target video/features, target masks, exact trajectories, seed and renderer metadata",
            "optional_oracle": "frames_empty retains distractor and background, removes selected ball",
            "scope": "Unseen scene seeds in the same procedural 2-D family; test half held-out shift magnitudes; no real-video or pixel-generation claim"}


def self_check(output: str | Path | None = None) -> dict:
    """Validate renderer invariants using TRAIN fixtures only, never test pixels."""
    specs = [s for split in SPLIT_CONFIG for s in scene_specs(split)]
    assert len({s["seed"] for s in specs}) == len(specs) == 48
    assert {s["dx"] for s in scene_specs("train")} == set(TRAIN_OFFSETS)
    assert {s["dx"] for s in scene_specs("test") if s["shift_regime"] == "heldout_magnitude"} == set(HELDOUT_OFFSETS)
    checked = []
    for spec in scene_specs("train")[:4]:
        pair = generate_pair(spec)
        fs, ft, empty = (pair[k] for k in ("frames_source", "frames_target", "frames_empty"))
        ms, mt, md = (pair[k] for k in ("masks_source", "masks_target", "masks_distractor"))
        assert fs.shape == ft.shape == empty.shape == (N_FRAMES, SIZE, SIZE, 3)
        assert fs.dtype == ft.dtype == empty.dtype == np.uint8
        assert ms.dtype == mt.dtype == md.dtype == np.bool_
        union = ms | mt
        assert np.array_equal(fs[~union], ft[~union]), "An unrelated source pixel changed"
        assert np.array_equal(fs[~ms], empty[~ms])
        assert np.array_equal(ft[~mt], empty[~mt])
        assert np.array_equal(np.roll(ms, spec["dx"], axis=2), mt)
        assert np.array_equal(np.roll(fs, spec["dx"], axis=2)[mt], ft[mt]), "Object texture did not translate exactly"
        assert not np.any(union & md)
        source_xy = np.array(pair["metadata"]["selected_source_xy"])
        target_xy = np.array(pair["metadata"]["selected_target_xy"])
        assert np.array_equal(target_xy - source_xy, np.tile([spec["dx"], 0], (N_FRAMES, 1)))
        assert np.ptp(source_xy[:, 0]) > 20 and np.ptp(source_xy[:, 1]) > 20
        # Confirm identical scene/appearance for a changed edit request. The
        # source has no accidental dependence on requested displacement.
        other = generate_pair({**spec, "dx": -spec["dx"]})
        assert np.array_equal(fs, other["frames_source"])
        assert np.array_equal(empty, other["frames_empty"])
        assert np.array_equal(patch_fractions(ms), patch_fractions(other["masks_source"]))
        assert np.allclose(patch_fractions(ms).sum(axis=(1, 2)), ms.reshape(N_STEPS, 2, SIZE, SIZE).sum(axis=(1, 2, 3)) / (2 * PATCH**2))
        checked.append({"name": spec["name"], "source_sha256": hashlib.sha256(fs.tobytes()).hexdigest(),
                        "target_sha256": hashlib.sha256(ft.tobytes()).hexdigest(),
                        "selected_pixels_per_frame": int(ms[0].sum()),
                        "source_xy_range": np.ptp(source_xy, axis=0).tolist()})
    result = {"status": "passed", "data_version": DATA_VERSION, "fixtures": checked,
              "checks": ["disjoint split seeds", "balanced displacement manifest", "integer translation geometry",
                         "exact transported object texture", "unchanged outside intervention union",
                         "exact optional empty-object control", "no object/distractor overlap",
                         "moving source trajectory", "source independent of requested displacement",
                         "deterministic regeneration", "tubelet coverage accounting"],
              "test_pixels_rendered": False}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest(), indent=2) + "\n")
    if args.self_check:
        print(json.dumps(self_check(args.output), indent=2))
    elif not args.manifest:
        print(json.dumps(manifest(), indent=2))


if __name__ == "__main__":
    main()
