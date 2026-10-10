"""Validation and provenance for reconstructed Day12 continuation drivers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from config import ARMS, EXPECTED_GUIDE_HASHES, MODEL_ID, REVISION, SCENES, SHIFTS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_record(path, root):
    path = Path(path)
    return dict(path=str(path.relative_to(root)), sha256=sha256(path), bytes=path.stat().st_size)


def load_guides(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("version") != "day12_image_jepa_rgb_video_guides_v1"
            or manifest.get("complete") is not True or manifest.get("frames") != 9
            or manifest.get("frame_index") != 15 or manifest.get("arms") != list(ARMS)):
        raise ValueError("Guide manifest does not describe the completed fixed pilot")
    if manifest["vae"]["model_id"] != MODEL_ID or manifest["vae"]["revision"] != REVISION:
        raise ValueError("Guide VAE is not the pinned renderer VAE")
    expected_specs = [dict(seed=seed, sign=1 if seed == 14000 else -1,
                           shifts_px=list(SHIFTS[seed])) for seed in SCENES]
    if manifest["scene_specs"] != expected_specs:
        raise ValueError("Guide scene specifications changed")
    rows = {int(row["seed"]): row for row in manifest["scenes"]}
    if len(rows) != len(manifest["scenes"]) or set(rows) != set(SCENES):
        raise ValueError("Missing or duplicate fixed scene")
    bundles = {}
    for seed in SCENES:
        row = rows[seed]
        expected_path = f"scenes/development_video_{seed}.npz"
        if row["path"] != expected_path:
            raise ValueError("Unexpected guide path")
        path = root / expected_path
        if sha256(path) != row["sha256"] or row["sha256"] != EXPECTED_GUIDE_HASHES[seed]:
            raise ValueError("Prepared guide bytes differ from recovered evidence")
        with np.load(path, allow_pickle=False) as archive:
            bundle = {key: archive[key].copy() for key in archive.files}
        metadata = json.loads(bytes(bundle.pop("metadata_json")).decode("utf-8"))
        if metadata["seed"] != seed or metadata["sign"] != expected_specs[SCENES.index(seed)]["sign"]:
            raise ValueError("Guide metadata scene mismatch")
        if metadata["array_sha256"] != row["array_sha256"] or set(bundle) != set(row["array_sha256"]):
            raise ValueError("Guide metadata and manifest array inventories disagree")
        for key, value in bundle.items():
            if array_sha256(value) != row["array_sha256"][key]:
                raise ValueError("Guide array hash mismatch: " + key)
        if not np.array_equal(bundle["shifts_px"], np.asarray(SHIFTS[seed], dtype=np.int32)):
            raise ValueError("Guide path is not the predeclared path")
        for arm in ARMS:
            value = bundle["guide_" + arm]
            if value.shape != (9, 384, 384, 3) or value.dtype != np.uint8:
                raise ValueError("Malformed RGB guide " + arm)
        for key in ("source_mask", "distractor_mask"):
            if bundle[key].shape != (384, 384) or bundle[key].dtype != np.bool_:
                raise ValueError("Malformed source scoring mask")
        if bundle["target_masks"].shape != (9, 384, 384) or bundle["target_masks"].dtype != np.bool_:
            raise ValueError("Malformed target scoring masks")
        bundles[seed] = bundle
    return manifest, bundles
