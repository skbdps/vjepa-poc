"""Replay four stable part controls on completed Day5 caches, without a GPU.

Recipes contain settings only. Native compositing is deterministic on original
rendered RGB. The protection guarantees concern predicted masks before lossy
encoding, not anatomical correctness. Held-out RGB is regenerated only after
the complete frozen test results and all twelve prediction caches are present.
This file is a post-experiment artifact helper; it never changes the frozen model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE.parent / "day4", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import benchmark
import parent_benchmark
import real_video
import run_parent_experiment as experiment

# One controls schema and compositing implementation shared with the browser builder.
from build_hierarchy_demo import (SCHEMA, PARTS, default_recipe, validate_recipe,
                                  effective_masks, composite_frame)
PART_IDS = tuple(part["mask_id"] for part in PARTS)
PARENT_IDS = (101, 102)


def apply_recipe_frame(rgb, children, parents, recipe):
    """Use the editor's exact native compositor and audit protected regions.

    Each color channel is floor(source*(1-strength)+color*strength), matching
    browser arithmetic on identical RGB. Browser video compression can change
    source pixels, so browser/native images are not claimed bit-identical.
    """
    sequence = recipe.get("sequence") if isinstance(recipe, dict) else None
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Recipe sequence must be a nonempty string")
    recipe = validate_recipe(recipe, sequence)
    rgb, children, parents = np.asarray(rgb), np.asarray(children), np.asarray(parents)
    masks = effective_masks(children, parents, recipe["containment"])
    output, permitted = composite_frame(rgb, children, parents, recipe)
    disabled = np.zeros(rgb.shape[:2], bool)
    active_ids, per_part = [], []
    for j, part in enumerate(recipe["parts"]):
        active = part["enabled"] and part["strength"] > 0
        if active:
            active_ids.append(part["id"])
        else:
            disabled |= children[j]
        per_part.append({"part_id": part["id"], "active": active,
                         "raw_mask_pixels": int(children[j].sum()),
                         "effective_mask_pixels": int(masks[j].sum()),
                         "permitted_pixels": int(masks[j].sum()) if active else 0})
    changed = np.any(output != rgb, axis=-1)
    overlap = children.sum(axis=0) > 1
    raw_permitted = (children & (children.sum(axis=0) == 1)[None]).any(axis=0)
    checks = {"active_part_ids": active_ids, "containment": recipe["containment"],
              "changed_pixels": int(changed.sum()), "permitted_pixels": int(permitted.sum()),
              "ambiguous_overlap_pixels": int(overlap.sum()),
              "changes_outside_enabled_permitted_masks": int((changed & ~permitted).sum()),
              "changes_in_ambiguous_overlaps": int((changed & overlap).sum()),
              "changes_in_disabled_predicted_parts": int((changed & disabled).sum()),
              "changes_outside_original_unique_child_regions": int((changed & ~raw_permitted).sum()),
              "parts": per_part}
    if any(checks[key] for key in checks if key.startswith("changes_")):
        raise AssertionError("Edit changed a protected pixel")
    return output, checks


def _expected_scenes():
    return {f"test_{condition}_{seed}": (condition, seed)
            for condition, seeds in parent_benchmark.TEST_SEEDS.items() for seed in seeds}


def verify_complete_run(run_root):
    """No frame generation occurs until the entire held-out run is complete."""
    run_root = Path(run_root)
    path = run_root / "test" / "results.json"
    if not path.exists():
        raise RuntimeError("Final twelve-clip test/results.json is required before creating a replay")
    report = json.loads(path.read_text())
    if report.get("stage") != "test" or report.get("frozen_config") != experiment.configuration():
        raise ValueError("Completed report does not match the fixed test source/policy")
    names = [c["scene"] for c in report["clips"]]
    expected = _expected_scenes()
    if len(names) != 12 or set(names) != set(expected):
        raise ValueError("Complete frozen twelve-clip test results required")
    for arm in experiment.ARMS:
        for representation in experiment.REPRESENTATIONS:
            if report["arms"][arm][representation]["overall"]["scored_part_frames"] != 3024:
                raise ValueError("Incomplete scored part-frame count")
    for name in names:
        folder = run_root / "clips" / name
        manifest_path = folder / "clip_manifest.json"
        recorded = next(c for c in report["clips"] if c["scene"] == name)
        if not manifest_path.exists() or real_video.sha256_file(manifest_path) != recorded["clip_manifest_sha256"]:
            raise ValueError(f"Missing or changed completed clip manifest: {name}")
        manifest = json.loads(manifest_path.read_text())
        identity = manifest["identity"]
        if identity["source_digest"] != report["frozen_config"]["source_digest"] or identity["checkpoint_sha256"] != experiment.CHECKPOINT_SHA256:
            raise ValueError("Prediction cache source/checkpoint mismatch")
        for role in ("children", "parents"):
            cache = folder / role / "sam2_masks.npz"
            if not cache.exists() or real_video.sha256_file(cache) != manifest["cache_sha256"][role]:
                raise ValueError(f"Missing or corrupt completed prediction cache: {name}/{role}")
    return report


def load_completed_scene(run_root, scene_name):
    """Load checked caches, then regenerate RGB for a completed fixed test scene."""
    report = verify_complete_run(run_root)
    if scene_name not in _expected_scenes():
        raise ValueError("Scene must belong to the complete frozen test set")
    folder = Path(run_root) / "clips" / scene_name
    manifest = json.loads((folder / "clip_manifest.json").read_text())
    predictions = {}
    for role, ids in (("children", PART_IDS), ("parents", PARENT_IDS)):
        path = folder / role / "sam2_masks.npz"
        with np.load(path, allow_pickle=False) as saved:
            if saved["ids"].tolist() != list(ids):
                raise ValueError("Prediction cache IDs do not match the frozen order")
            masks = saved["masks"]
        if masks.dtype != bool or masks.shape != (64, len(ids), 384, 384):
            raise ValueError("Prediction cache has an invalid shape or dtype")
        predictions[role] = masks
    condition, seed = _expected_scenes()[scene_name]
    scene, owners = parent_benchmark.generate_scene(seed, condition, scene_name)
    rgb_hash = hashlib.sha256(scene.frames.tobytes()).hexdigest()
    if rgb_hash != manifest["identity"]["rgb_sha256"]:
        raise ValueError("Regenerated source RGB differs from the completed inference input")
    scored = next(c for c in report["clips"] if c["scene"] == scene_name)
    if hashlib.sha256(scene.masks.tobytes()).hexdigest() != scored["part_truth_sha256"] or hashlib.sha256(owners.tobytes()).hexdigest() != scored["owner_truth_sha256"]:
        raise ValueError("Regenerated scoring labels differ from the saved report")
    provenance = {"scene": scene_name, "source_digest": manifest["identity"]["source_digest"],
                  "rgb_sha256": rgb_hash, "cache_sha256": manifest["cache_sha256"],
                  "complete_test_results_sha256": real_video.sha256_file(Path(run_root) / "test" / "results.json"),
                  "clip_manifest_sha256": real_video.sha256_file(folder / "clip_manifest.json"),
                  "renderer_input": "original rendered lossless RGB; tracker input was quality100 JPEG"}
    return scene, predictions["children"], predictions["parents"], provenance


def replay(run_root, scene_name, recipe, output, fps=12):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    recipe = validate_recipe(recipe, scene_name)
    scene, children, parents, provenance = load_completed_scene(run_root, scene_name)
    rendered, checks = [], []
    for frame, rgb in enumerate(scene.frames):
        edited, audit = apply_recipe_frame(rgb, children[frame], parents[frame], recipe)
        rendered.append(edited)
        checks.append({"frame": frame, **audit})
    output = Path(output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output filename must end in.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_video(output, rendered, fps=fps)
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report = {"recipe": recipe, "canonical_recipe_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
              "provenance": provenance, "fps": fps, "frame_count": len(rendered),
              "operation": "deterministic independent alpha recolor; no model inference",
              "guarantee_scope": "RGB arrays before lossy video encoding; predicted mask protection, not anatomical truth",
              "containment_policy": experiment.POLICY["effective_parent"],
              "all_frames_passed": all(not row[key] for row in checks for key in row if key.startswith("changes_")),
              "frames": checks, "replay_source_sha256": real_video.sha256_file(__file__),
              "compositor_source_sha256": real_video.sha256_file(HERE / "build_hierarchy_demo.py"),
              "output_video_sha256": real_video.sha256_file(output)}
    real_video.write_json(output.with_suffix(".recipe.json"), recipe)
    real_video.write_json(output.with_suffix(".audit.json"), report)
    return report


def self_test():
    """Fabricated masks only; no test scene generation or model execution."""
    recipe = default_recipe("fixture")
    recipe["containment"] = True
    recipe["parts"][1].update(enabled=True, strength=.45)
    recipe["parts"][2].update(enabled=True, strength=.65)
    recipe["parts"][3]["enabled"] = False
    rgb = np.full((16, 16, 3), 137, np.uint8)
    children = np.zeros((4, 16, 16), bool)
    children[0, 1:9, 1:9] = True
    children[1, 6:12, 6:12] = True
    children[2, 10:15, 1:5] = True
    children[3, 11:15, 11:15] = True
    parents = np.ones((2, 16, 16), bool)
    parents[0, 7:, :] = False
    raw = effective_masks(children, parents, False)
    gated = effective_masks(children, parents, True)
    assert not np.any(gated & ~raw)
    assert not gated[:, 6:9, 6:9].any(), "Dropped child must not release protected overlap"
    expected = experiment.arm_masks(children[None], parents[None])
    assert np.array_equal(raw, expected["raw_part"]["effective_edit"][0])
    assert np.array_equal(gated, expected["parent_intersection"]["effective_edit"][0])
    output, audit = apply_recipe_frame(rgb, children, parents, recipe)
    assert np.any(output != rgb) and not any(audit[k] for k in audit if k.startswith("changes_"))
    reordered = copy.deepcopy(recipe)
    reordered["parts"].reverse()
    assert np.array_equal(output, apply_recipe_frame(rgb, children, parents, reordered)[0])
    assert validate_recipe(json.loads(json.dumps(recipe)), "fixture") == validate_recipe(recipe, "fixture")
    off = copy.deepcopy(recipe)
    for part in off["parts"]:
        part["enabled"] = False
    assert np.array_equal(rgb, apply_recipe_frame(rgb, children, parents, off)[0])
    zero = copy.deepcopy(recipe)
    for part in zero["parts"]:
        part["strength"] = 0
    assert np.array_equal(rgb, apply_recipe_frame(rgb, children, parents, zero)[0])
    malformed = []
    for field, value in (("schema", "wrong"), ("version", True), ("sequence", "other"),
                         ("containment", 1), ("parts", recipe["parts"][:3])):
        bad = copy.deepcopy(recipe); bad[field] = value; malformed.append(bad)
    for field, value in (("id", True), ("id", "car_B.front_door"), ("id", "unknown"), ("enabled", 1),
                         ("color", [256, 0, 0]), ("color", [True, 0, 0]),
                         ("strength", float("nan")), ("strength", True), ("strength", 1.1)):
        bad = copy.deepcopy(recipe); bad["parts"][0][field] = value; malformed.append(bad)
    for bad in malformed:
        try:
            validate_recipe(bad, "fixture")
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed recipe accepted")
    # An incomplete run must fail BEFORE renderer access; use no test labels.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        original = parent_benchmark.generate_scene
        def forbidden(*args, **kwargs):
            raise AssertionError("Renderer must not run before completed results exist")
        parent_benchmark.generate_scene = forbidden
        try:
            try:
                load_completed_scene(temp, "test_crossing_6200")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Incomplete run accepted")
        finally:
            parent_benchmark.generate_scene = original
    return {"four_stable_ids": True, "recipe_roundtrip": True,
            "order_independent_controls": True, "original_overlap_protection": True,
            "frozen_effective_mask_policy_matches": True, "parent_edit_is_subset": True,
            "disabled_and_outside_pixels_unchanged": True, "zero_strength_unchanged": True,
            "malformed_recipes_rejected": len(malformed), "incomplete_run_blocks_renderer": True,
            "held_out_scenes_generated": False, "gpu_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root")
    parser.add_argument("--scene", default="test_crossing_6200")
    parser.add_argument("--recipe")
    parser.add_argument("--out")
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--default-recipe", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.default_recipe:
        print(json.dumps(default_recipe(args.scene), indent=2))
    else:
        if not args.run_root or not args.recipe or not args.out:
            parser.error("Replay needs --run-root, --recipe and --out")
        report = replay(args.run_root, args.scene, json.loads(Path(args.recipe).read_text()), args.out, args.fps)
        print(json.dumps({"output": args.out, "frame_count": report["frame_count"],
                          "all_frames_passed": report["all_frames_passed"]}, indent=2))
