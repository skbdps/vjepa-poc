"""Validate and replay portable, simultaneous edits on stable predicted part IDs.

Recipes contain controls only, never pixels, masks, model state or executable
code. Every enabled part is composited through part_editor.apply_recolor, which
excludes all other predicted masks. Ambiguous overlaps therefore stay original,
including when both overlapping parts are enabled. No model is run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from part_editor import apply_recolor

SCHEMA = "vjepa.part-edit-recipe"
VERSION = 1
ROOT_KEYS = {"schema", "version", "sequence", "parts"}
PART_KEYS = {"id", "name", "enabled", "color", "strength"}


def default_recipe(registry):
    """Door enabled initially; other parts retain independent disabled controls."""
    parts = registry["parts"]
    door_id = next((p["id"] for p in parts if p["name"] == "front_door_panel"), None)
    return {"schema": SCHEMA, "version": VERSION, "sequence": registry["sequence"],
            "parts": [{"id": int(p["id"]), "name": p["name"], "enabled": p["id"] == door_id,
                       "color": [240, 180, 35] if p["name"] == "front_window" else [35, 140, 240],
                       "strength": .75} for p in parts]}


def validate_recipe(recipe, registry):
    """Return a normalized copy in registry order, rejecting any wrong binding.

    Unknown fields are rejected to keep this file a small controls-only format.
    The complete registry is required: absent entries must not be confused with
    deliberately disabled edits. Part identity never depends on list ordering.
    """
    if not isinstance(recipe, dict) or set(recipe) != ROOT_KEYS:
        raise ValueError("Recipe needs exactly schema, version, sequence and parts")
    if recipe["schema"] != SCHEMA or type(recipe["version"]) is not int or recipe["version"] != VERSION:
        raise ValueError("Unsupported recipe schema/version")
    if recipe["sequence"] != registry["sequence"]:
        raise ValueError("Recipe sequence does not match this video")
    expected = {int(p["id"]): p["name"] for p in registry["parts"]}
    if len(expected) != len(registry["parts"]):
        raise ValueError("Registry contains duplicate part IDs")
    if not isinstance(recipe["parts"], list) or len(recipe["parts"]) != len(expected):
        raise ValueError("Recipe must contain every registered part exactly once")
    normalized = {}
    for part in recipe["parts"]:
        if not isinstance(part, dict) or set(part) != PART_KEYS:
            raise ValueError("Part controls need exactly id, name, enabled, color and strength")
        part_id = part["id"]
        if type(part_id) is not int or part_id not in expected or part_id in normalized:
            raise ValueError("Unknown or duplicate recipe part ID")
        if part["name"] != expected[part_id]:
            raise ValueError("Recipe part name does not match its registered ID")
        if type(part["enabled"]) is not bool:
            raise ValueError("enabled must be a JSON boolean")
        color = part["color"]
        if not isinstance(color, list) or len(color) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in color):
            raise ValueError("color must contain three integer RGB channels from 0 to 255")
        strength = part["strength"]
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be a finite number from 0 to 1")
        normalized[part_id] = {"id": part_id, "name": part["name"], "enabled": part["enabled"],
                               "color": color.copy(), "strength": float(strength)}
    return {"schema": SCHEMA, "version": VERSION, "sequence": registry["sequence"],
            "parts": [normalized[int(p["id"])] for p in registry["parts"]]}


def apply_recipe(rgb, part_masks, ids, recipe, registry):
    """Recolor all enabled IDs; return edited RGB and a pixel-preservation audit."""
    recipe = validate_recipe(recipe, registry)
    ids = list(map(int, ids))
    if len(set(ids)) != len(ids) or set(ids) != {p["id"] for p in recipe["parts"]}:
        raise ValueError("Saved mask IDs must match recipe/registry IDs exactly")
    rgb = np.asarray(rgb)
    masks = np.asarray(part_masks, dtype=bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image")
    if masks.shape != (len(ids), *rgb.shape[:2]):
        raise ValueError("Mask and image sizes differ")
    output = rgb.copy()
    permitted = np.zeros(rgb.shape[:2], bool)
    overlapping = masks.sum(axis=0) > 1
    enabled_ids = []
    for part in recipe["parts"]:
        if not part["enabled"] or part["strength"] == 0:
            continue
        output, effective, _ = apply_recolor(output, masks, ids, part["id"],
                                             tuple(part["color"]), part["strength"])
        permitted |= effective
        enabled_ids.append(part["id"])
    changed = np.any(output != rgb, axis=-1)
    outside = int((changed & ~permitted).sum())
    overlap_changes = int((changed & overlapping).sum())
    if outside or overlap_changes:
        raise AssertionError("Recipe modified a protected or ambiguous pixel")
    return output, {"active_part_ids": enabled_ids, "changed_pixels": int(changed.sum()),
                    "permitted_pixels": int(permitted.sum()),
                    "ambiguous_overlap_pixels": int(overlapping.sum()),
                    "changes_outside_enabled_permitted_masks": outside,
                    "changes_in_ambiguous_overlaps": overlap_changes}


def replay_recipe(frames_dir, masks_file, registry_file, recipe_file, output, fps=12):
    from benchmark import write_video
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    registry = json.loads(Path(registry_file).read_text())
    recipe = validate_recipe(json.loads(Path(recipe_file).read_text()), registry)
    with np.load(masks_file, allow_pickle=False) as saved:
        masks, ids = saved["masks"], saved["ids"].tolist()
    if not frames or len(frames) != len(masks) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame/mask count or frame rate")
    rendered, checks = [], []
    for t, path in enumerate(frames):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Cannot decode frame {path.name}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        edited, check = apply_recipe(rgb, masks[t], ids, recipe, registry)
        rendered.append(edited)
        checks.append({"frame": t, **check})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_video(output, rendered, fps=fps)
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report = {"recipe": recipe, "canonical_recipe_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
              "mask_source_sha256": hashlib.sha256(Path(masks_file).read_bytes()).hexdigest(),
              "fps": fps, "frame_count": len(frames),
              "operation": "independent shading-preserving recolors on saved stable part IDs",
              "guarantee_scope": "RGB arrays before lossy encoding; predicted masks, not anatomical truth",
              "preview_note": "Browser uses compressed embedded MP4 RGB; native replay uses original input JPEG RGB.",
              "frames": checks}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".recipe.json").write_text(json.dumps(recipe, indent=2, allow_nan=False) + "\n")
    return report


def self_test():
    """Behavior tests for recipe identity, round trips and overlapping masks."""
    registry = {"sequence": "fixture", "parts": [
        {"id": 11, "name": "front_door_panel", "parent": "car_1"},
        {"id": 29, "name": "front_window", "parent": "car_1"}]}
    recipe = default_recipe(registry)
    rgb = np.full((20, 22, 3), 100, np.uint8)
    masks = np.zeros((2, 20, 22), bool)
    masks[0, 2:14, 2:14] = True
    masks[1, 9:18, 9:19] = True
    overlap = masks.all(axis=0)
    door_only, audit = apply_recipe(rgb, masks, [11, 29], recipe, registry)
    assert np.array_equal(door_only[masks[1]], rgb[masks[1]])
    recipe["parts"][1]["enabled"] = True
    recipe["parts"][1]["strength"] = .42
    both, audit = apply_recipe(rgb, masks, [11, 29], recipe, registry)
    assert np.any(both[masks[0] & ~masks[1]] != rgb[masks[0] & ~masks[1]])
    assert np.any(both[masks[1] & ~masks[0]] != rgb[masks[1] & ~masks[0]])
    assert np.array_equal(both[overlap], rgb[overlap])
    assert np.array_equal(both[~masks.any(axis=0)], rgb[~masks.any(axis=0)])
    decoded = validate_recipe(json.loads(json.dumps(recipe)), registry)
    assert np.array_equal(apply_recipe(rgb, masks, [11, 29], decoded, registry)[0], both)
    reordered = json.loads(json.dumps(recipe))
    reordered["parts"].reverse()
    assert np.array_equal(apply_recipe(rgb, masks[[1, 0]], [29, 11], reordered, registry)[0], both)
    for part in decoded["parts"]:
        part["enabled"] = False
    assert np.array_equal(apply_recipe(rgb, masks, [11, 29], decoded, registry)[0], rgb)
    rejected = 0
    for mutate in (
        lambda r: r.update(sequence="other-video"),
        lambda r: r.update(version=2),
        lambda r: r.update(model_state={}),
        lambda r: r["parts"][0].update(id=29),
        lambda r: r["parts"][0].update(name="wrong-part"),
        lambda r: r["parts"][0].update(enabled="true"),
        lambda r: r["parts"][0].update(strength=float("nan")),
        lambda r: r["parts"][0].update(strength=1.1),
        lambda r: r["parts"][0].update(color=[256, 0, 0]),
        lambda r: r["parts"].pop(),
    ):
        invalid = json.loads(json.dumps(recipe))
        mutate(invalid)
        try:
            validate_recipe(invalid, registry)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Invalid recipe was accepted")
    return {"recipe_json_roundtrip": True, "simultaneous_parts_changed": True,
            "ambiguous_overlaps_unchanged": True, "disabled_part_unchanged": True,
            "outside_pixels_unchanged": True, "stable_ids_ignore_order": True,
            "all_disabled_restores_original": True, "invalid_recipes_rejected": rejected,
            "real_model_results": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir")
    parser.add_argument("--masks")
    parser.add_argument("--registry")
    parser.add_argument("--recipe")
    parser.add_argument("--out")
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        if not all((args.frames_dir, args.masks, args.registry, args.recipe, args.out)):
            parser.error("Supply --frames-dir, --masks, --registry, --recipe and --out")
        report = replay_recipe(args.frames_dir, args.masks, args.registry, args.recipe, args.out, args.fps)
        print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
