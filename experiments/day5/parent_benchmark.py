"""Day5 whole-parent annotation extension of the frozen Day4 2-D renderer.

This deliberately copies/adapts the render path instead of changing or
monkeypatching Day4. RGB, target masks and Scene metadata remain byte-identical.
Whole-car owner masks are additional ground truth, not a tracker input after
frame-zero initialization. They do not establish 3-D or real-video behavior.

The smoke self-test does not render, inspect, or score any held-out test seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

# Support both module import and direct execution from any working directory.
DAY4 = Path(__file__).resolve().parents[1] / "day4"
if str(DAY4) not in sys.path:
    sys.path.insert(0, str(DAY4))
import benchmark as day4

Scene = day4.Scene
SIZE, N_FRAMES = day4.SIZE, day4.N_FRAMES
CONDITIONS = day4.CONDITIONS
PART_NAMES = day4.PART_NAMES
PROTOCOL_VERSION = day4.PROTOCOL_VERSION
initial_labels = day4.initial_labels

OWNER_NAMES = {1: "car_A", 2: "car_B"}
PART_TO_OWNER = {1: 1, 2: 1, 3: 2, 4: 2}
SMOKE_SEEDS = {"long_occlusion": [5100], "crossing": [5200], "scale_camera": [5300]}
TEST_SEEDS = {"long_occlusion": list(range(6100, 6104)),
              "crossing": list(range(6200, 6204)),
              "scale_camera": list(range(6300, 6304))}
OWNER_PROTOCOL_VERSION = "day5_visible_parent_v1"


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


def _paste_car(image, mask, owner_mask, sprite_pair, identity, x, y,
               scale=1., angle=0.):
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
    # Match the exact same opaque-pixel convention used by the part labels.
    # Earlier parents are overwritten even by an unlabelled foreground body.
    owner_mask.paste(identity + 1, (pos[0], pos[1],
                     pos[0] + sprite.width, pos[1] + sprite.height), opaque)


def _smooth(p):
    return p*p*(3.-2.*p)


def generate_scene(seed: int, condition: str, name: str | None = None
                   ) -> tuple[Scene, np.ndarray]:
    """Return the unchanged Day4 scene and its visible whole-car owner labels.

    Owners are uint8 [64,384,384]: 0 background/occluder, 1 car A, 2 car B.
    The owner raster includes the full alpha>=128 sprite silhouette, including
    wheels, body and tagged parts. It follows the same transform and draw order
    as Day4. Later-frame owners and part labels are evaluation-only; a tracker
    may receive owners[0] and scene.masks[0] for initialization, then only RGB.
    """
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
    frames, masks, owners = [], [], []
    for t in range(N_FRAMES):
        # Stationary annotation prefix avoids a source point crossing a patch edge.
        p = max(0., (t-3)/(N_FRAMES-4))
        ease = _smooth(p**motion_power)
        camx = camera_amp*np.sin(2*np.pi*p) if condition == "scale_camera" else 0.
        camy = 5*np.sin(4*np.pi*p) if condition == "scale_camera" else 0.
        image = Image.new("RGB", (SIZE, SIZE), (222, 229, 232))
        mask = Image.new("L", (SIZE, SIZE), 0)
        owner_mask = Image.new("L", (SIZE, SIZE), 0)
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
        _paste_car(image, mask, owner_mask, sprites[0], 0, ax+camx, ay+camy, scale_a, angle_a)
        _paste_car(image, mask, owner_mask, sprites[1], 1, bx+camx, by+camy, scale_b, angle_b)
        if condition == "long_occlusion":
            draw, labels = ImageDraw.Draw(image), ImageDraw.Draw(mask)
            rect = (occluder_x, 25, 383, 180)
            draw.rectangle(rect, fill=(89, 98, 111))
            labels.rectangle(rect, fill=0)
            ImageDraw.Draw(owner_mask).rectangle(rect, fill=0)
            for xx in range(occluder_x+4, 384, 12):
                draw.line((xx, 25, xx, 180), fill=(101, 110, 121), width=2)
        frames.append(np.asarray(image).copy())
        masks.append(np.asarray(mask).copy())
        owners.append(np.asarray(owner_mask).copy())
    scene = Scene(name or f"{condition}_{seed}", np.stack(frames), np.stack(masks),
                  PART_NAMES.copy(), int(seed), condition, parameters)
    initial = initial_labels(scene.masks[0])
    if not all((initial == pid).any() for pid in PART_NAMES):
        raise RuntimeError(f"Source annotation lacks a usable patch: {scene.name}")
    return scene, np.stack(owners)



# Explicit alternate name for callers combining multiple benchmark modules.
generate_parent_scene = generate_scene


def _array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def self_test(output: str | Path | None = None):
    """Verify only smoke scenes against the immutable Day4 implementation."""
    results = []
    for condition in CONDITIONS:
        for seed in SMOKE_SEEDS[condition]:
            name = f"smoke_{condition}_{seed}"
            reference = day4.generate_scene(seed, condition, name)
            scene, owners = generate_scene(seed, condition, name)
            assert np.array_equal(scene.frames, reference.frames), name + ": RGB changed"
            assert np.array_equal(scene.masks, reference.masks), name + ": parts changed"
            assert scene.name == reference.name and scene.seed == reference.seed
            assert scene.condition == reference.condition
            assert scene.parameters == reference.parameters
            assert scene.part_names == reference.part_names
            assert owners.shape == (N_FRAMES, SIZE, SIZE)
            assert owners.dtype == np.uint8 and set(np.unique(owners)) <= {0, 1, 2}
            for part, owner in PART_TO_OWNER.items():
                assert np.all(owners[scene.masks == part] == owner), (name, part)
                assert np.any(scene.masks[0] == part), (name, part)
            for owner in OWNER_NAMES:
                assert np.any(owners[0] == owner), (name, owner)
                assert np.any((owners[0] == owner) & (scene.masks[0] == 0))
            # All test-independent smoke scenes start with separated cars.
            # Check a labelled-background wheel center belongs to its parent.
            for owner, xkey, ykey in ((1, "a_start_x", "a_start_y"),
                                      (2, "b_start_x", "b_start_y")):
                x, y = scene.parameters[xkey] + 30, scene.parameters[ykey] + 109
                assert owners[0, y, x] == owner
                assert scene.masks[0, y, x] == 0
            absent = {OWNER_NAMES[owner]: np.flatnonzero(
                ~(owners == owner).any(axis=(1, 2))).tolist() for owner in OWNER_NAMES}
            if condition == "long_occlusion":
                assert absent["car_A"], "Occluded parent never fully disappeared"
                ox = scene.parameters["occluder_x"]
                assert not owners[:, 25:181, ox:384].any(), "Occluder retains ownership"
                for frame in absent["car_A"]:
                    assert not np.isin(scene.masks[frame], [1, 2]).any()
            results.append({"name": name, "seed": seed, "condition": condition,
                "rgb_equal": True, "part_masks_equal": True, "metadata_equal": True,
                "part_subset_parent": True, "first_frame_parents_available": True,
                "unlabelled_body_and_wheels_included": True,
                "owner_ids": [int(x) for x in np.unique(owners)],
                "fully_absent_frames": absent,
                "rgb_sha256": _array_digest(scene.frames),
                "parts_sha256": _array_digest(scene.masks),
                "owners_sha256": _array_digest(owners)})
    report = {"status": "passed", "owner_protocol": OWNER_PROTOCOL_VERSION,
        "day4_renderer_sha256": hashlib.sha256((DAY4 / "benchmark.py").read_bytes()).hexdigest(),
        "day5_renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_smoke_clips": len(results), "held_out_test_seeds_rendered": False,
        "owner_definition": "Visible transformed sprite alpha >=128; foreground car B and solid occluders overwrite earlier owners.",
        "tracking_annotation_boundary": "Only frame-zero parent/part masks allowed for initialization; all later labels are scoring-only.",
        "scope": "2-D synthetic sprites; does not establish performance on real video, faces, or 3-D objects.",
        "clips": results}
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Render and compare smoke seeds only")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("renderer_self_test.json"))
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test; test-seed generation is available through the Python API only.")
    result = self_test(args.output)
    print(json.dumps({"status": result["status"], "n_smoke_clips": result["n_smoke_clips"],
                      "held_out_test_seeds_rendered": False, "output": str(args.output)}))


if __name__ == "__main__":
    main()
