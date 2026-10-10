"""Day7 supervised, prompt-conditioned persistent-part learning data.

Fresh deterministic splits extend the frozen Day4 2-D renderer. Every clip is
transformed by seed-hashed spatial flips and a coherent car-A/car-B label swap.
Only RGB/features and frame-zero part prompts may reach test-time inference.
All later labels belong to the separate training/evaluation supervision API.
This experiment is supervised training of a head over a frozen JEPA backbone;
it is not self-supervised JEPA training or evidence about real-world identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

DAY4 = Path(__file__).resolve().parents[1] / "day4"
if str(DAY4) not in sys.path:
    sys.path.insert(0, str(DAY4))
import benchmark as bench

Scene = bench.Scene
SIZE, PATCH, GRID, N_FRAMES = bench.SIZE, bench.PATCH, bench.GRID, bench.N_FRAMES
N_STEPS, N_PARTS, N_PATCHES = N_FRAMES // 2, 4, GRID * GRID
CONDITIONS = bench.CONDITIONS
DATA_VERSION = "day7_supervised_prompt_parts_v1"
TRANSFORM_SALT = "vjepa-poc/day7/spatial-and-parent-permutation/v1"
SIBLING_GROUPS = ((0, 1), (2, 3))
SPLIT_SEEDS = {
    "train": {"long_occlusion": list(range(7100, 7112)),
              "crossing": list(range(7200, 7212)),
              "scale_camera": list(range(7300, 7312))},
    "dev": {"long_occlusion": list(range(9100, 9103)),
            "crossing": list(range(9200, 9203)),
            "scale_camera": list(range(9300, 9303))},
    "test": {"long_occlusion": list(range(10100, 10106)),
             "crossing": list(range(10200, 10206)),
             "scale_camera": list(range(10300, 10306))},
}


def scene_specs(split: str) -> list[dict]:
    """List the frozen manifest without rendering or opening any clip labels."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split {split!r}; expected {tuple(SPLIT_SEEDS)}")
    return [{"name": f"{split}_{condition}_{seed}", "seed": seed,
             "condition": condition}
            for condition in CONDITIONS for seed in SPLIT_SEEDS[split][condition]]


def transform_spec(seed: int, condition: str) -> dict:
    """Independent deterministic transform bits; no RNG state shared with renderer."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    payload = f"{TRANSFORM_SALT}|{condition}|{int(seed)}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return {"horizontal_flip": bool(digest[0] & 1),
            "vertical_flip": bool(digest[1] & 1),
            "swap_parent_labels": bool(digest[2] & 1),
            "sha256": digest.hex()}


def _transform(frames: np.ndarray, masks: np.ndarray, spec: dict
               ) -> tuple[np.ndarray, np.ndarray]:
    # Whole-clip transforms preserve temporal correspondence and frame order.
    if spec["horizontal_flip"]:
        frames, masks = frames[:, :, ::-1], masks[:, :, ::-1]
    if spec["vertical_flip"]:
        frames, masks = frames[:, ::-1], masks[:, ::-1]
    if spec["swap_parent_labels"]:
        # Use one simultaneous lookup, never a sequence of in-place replacements.
        masks = np.asarray([0, 3, 4, 1, 2], dtype=np.uint8)[masks]
    return np.ascontiguousarray(frames), np.ascontiguousarray(masks)


def generate_scene(seed: int, condition: str, name: str | None = None) -> Scene:
    """Render RGB and scoring labels; never pass this entire object to a model.

    A/B are arbitrary prompt-group identities after the coherent label swap.
    The optional transform metadata must not be supplied to the learned head.
    No whole-parent annotation is supplied at initialization or inference.
    """
    source = bench.generate_scene(seed, condition, name)
    spec = transform_spec(seed, condition)
    frames, masks = _transform(source.frames, source.masks, spec)
    scene = Scene(source.name, frames, masks, source.part_names.copy(),
                  source.seed, source.condition,
                  {**(source.parameters or {}), "day7_data_version": DATA_VERSION,
                   "day7_transform": spec,
                   "identity_semantics": "A/B identify initial prompt groups after label permutation",
                   "tracking_annotation": "frame-zero tagged parts only; no parent masks"})
    # Flips are on a patch-aligned 384-pixel canvas, preserving usable prompts.
    initial_weights(scene.masks[0])
    return scene


def generate_suite(split: str) -> Iterator[Scene]:
    """Yield one scene at a time to bound CPU memory during feature extraction."""
    for spec in scene_specs(split):
        yield generate_scene(**spec)


def _validate_masks(masks: np.ndarray, initial_only: bool = False) -> np.ndarray:
    masks = np.asarray(masks)
    expected = (SIZE, SIZE) if initial_only else (N_FRAMES, SIZE, SIZE)
    if masks.shape != expected:
        raise ValueError(f"Expected mask shape {expected}, got {masks.shape}")
    if not np.issubdtype(masks.dtype, np.integer):
        raise ValueError("Part labels must be integer IDs")
    if not np.isin(masks, [0, 1, 2, 3, 4]).all():
        raise ValueError("Part labels must be in 0..4")
    return masks


def initial_weights(initial_mask: np.ndarray) -> np.ndarray:
    """[4,576] normalized weights over >=70%-covered frame-zero part patches.

    Together with frozen RGB-derived features, this is the full annotation
    budget at inference. The parent relation is just the two sibling groups.
    No target trajectory, future mask, condition, seed or transform is an input.
    """
    labels = bench.initial_labels(_validate_masks(initial_mask, initial_only=True))
    weights = np.stack([(labels == pid).ravel() for pid in range(1, 5)]).astype(np.float32)
    counts = weights.sum(axis=1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("Every tagged part needs a usable frame-zero patch")
    return weights / counts


def supervision(masks: np.ndarray) -> dict[str, np.ndarray]:
    """Build training/evaluation targets, forbidden as an inference input.

    positive[t,k,n] uses exactly Day4's >=70% average tubelet coverage and
    >=65% coverage in each of its two frames. Fully absent means zero pixels
    in both frames. Visible slivers without a qualifying patch are ambiguous
    and must be ignored by localization/presence loss. Tubelet zero initializes
    prompts and is excluded from optimization/scoring by score_steps below.
    Fractions are auxiliary training targets; they never reach inference.
    """
    masks = _validate_masks(masks)
    labels = bench.patch_labels(masks)
    positive = np.stack([(labels == pid).reshape(N_STEPS, N_PATCHES)
                         for pid in range(1, 5)], axis=1)
    visible = positive.any(axis=2)
    pairs = masks.reshape(N_STEPS, 2, SIZE, SIZE)
    absent = np.stack([~(pairs == pid).any(axis=(1, 2, 3))
                       for pid in range(1, 5)], axis=1)
    blocked = masks.reshape(N_STEPS, 2, GRID, PATCH, GRID, PATCH)
    fractions = np.stack([(blocked == pid).mean(axis=(1, 3, 5)).reshape(N_STEPS, N_PATCHES)
                          for pid in range(1, 5)], axis=1).astype(np.float32)
    score_steps = np.arange(N_STEPS) > 0
    state = np.where(visible, 1, np.where(absent, 0, -1)).astype(np.int8)
    return {"positive": positive, "visible": visible, "absent": absent,
            "ambiguous": ~(visible | absent), "fractions": fractions,
            "state": state, "pixel_present": ~absent,
            "patch_labels": labels.reshape(N_STEPS, N_PATCHES),
            "initial_weights": initial_weights(masks[0]), "score_steps": score_steps}


def manifest() -> dict:
    return {"data_version": DATA_VERSION, "split_seeds": SPLIT_SEEDS,
            "n_clips": {split: len(scene_specs(split)) for split in SPLIT_SEEDS},
            "frames": N_FRAMES, "resolution": SIZE, "tubelet": 2, "grid": GRID,
            "transform_salt": TRANSFORM_SALT,
            "transforms": "SHA256 bits independently select horizontal flip, vertical flip, and coherent A/B part-group permutation",
            "sibling_groups_zero_based": SIBLING_GROUPS,
            "inference_inputs": "Frozen RGB-derived features; normalized part prompts from frame zero; sibling-group relation",
            "forbidden_inference_inputs": "Later masks; owner masks; seed; condition; motion parameters; transform metadata",
            "learning": "Supervised learned head over frozen JEPA features; not self-supervised training",
            "visibility": "Day4 >=70% pair coverage and >=65% each frame; absent iff zero target pixels in both frames",
            "exclusions": "Tubelet zero and ambiguous visible slivers",
            "scope": "Held-out seeds in the same three 2-D procedural trajectory families; no real-video/OOD claim"}


def self_test(output: str | Path | None = None) -> dict:
    """Validate only the first TRAIN fixture of each condition; never render test."""
    all_seeds = [seed for split in SPLIT_SEEDS.values()
                 for seeds in split.values() for seed in seeds]
    assert len(all_seeds) == len(set(all_seeds)), "Split seed overlap"
    clips = []
    for condition in CONDITIONS:
        seed = SPLIT_SEEDS["train"][condition][0]
        name = f"train_{condition}_{seed}"
        raw = bench.generate_scene(seed, condition, name)
        scene = generate_scene(seed, condition, name)
        repeat = generate_scene(seed, condition, name)
        assert np.array_equal(scene.frames, repeat.frames)
        assert np.array_equal(scene.masks, repeat.masks)
        # Every transform is an involution and its geometric operations commute.
        restored_rgb, restored_labels = _transform(scene.frames, scene.masks,
                                                   transform_spec(seed, condition))
        assert np.array_equal(restored_rgb, raw.frames)
        assert np.array_equal(restored_labels, raw.masks)
        target = supervision(scene.masks)
        states = bench.target_states(scene)
        assert np.array_equal(target["state"], states)
        assert np.array_equal(target["visible"], states == 1)
        assert np.array_equal(target["absent"], states == 0)
        assert np.array_equal(target["ambiguous"], states == -1)
        assert target["positive"].shape == (32, 4, 576)
        assert np.allclose(target["initial_weights"].sum(1), 1)
        assert not (target["visible"] & target["absent"]).any()
        assert np.array_equal(target["patch_labels"], bench.patch_labels(scene.masks).reshape(32, 576))
        assert np.all(target["fractions"][target["positive"]] >= .70 - 1e-7)
        assert target["score_steps"].sum() == 31 and not target["score_steps"][0]
        # Future labels may alter supervision, but cannot alter inference prompts.
        corrupt = scene.masks.copy()
        corrupt[1:] = 0
        assert np.array_equal(initial_weights(corrupt[0]), target["initial_weights"])
        clips.append({"name": name, "transform": transform_spec(seed, condition),
                      "rgb_sha256": hashlib.sha256(scene.frames.tobytes()).hexdigest(),
                      "mask_sha256": hashlib.sha256(scene.masks.tobytes()).hexdigest(),
                      "visible": int(target["visible"][1:].sum()),
                      "absent": int(target["absent"][1:].sum()),
                      "ambiguous": int(target["ambiguous"][1:].sum()),
                      "checks": "determinism; transform invertibility; exact Day4 scoring equivalence; valid frame-zero prompts; label separation"})
    report = {"status": "passed", "data_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "manifest": manifest(), "rendered_split": "train only",
              "held_out_test_rendered": False, "clips": clips}
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("data_self_test.json"))
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test. Test generation is available only through the Python API.")
    result = self_test(args.output)
    print(json.dumps({"status": result["status"], "train_fixtures": len(result["clips"]),
                      "held_out_test_rendered": False, "output": str(args.output)}))
