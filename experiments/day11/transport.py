"""Day11 development follow-up: transport source appearance using JEPA provenance.

PREDECLARED METHODS (no optimizer; the original final CNN remains frozen).
Let j_s/j_e be source/edited JEPA, z_s the normalized source Wan latent,
and f_s=F(j_s), f_e=F(j_e). First identify unchanged cells by exact equality
at the same coordinate. For every other cell q, require a UNIQUE exact
1024-vector match j_e[q]=j_s[p], or classify q as unmatched. Multiple source
matches are an error, never an arbitrary nearest-neighbor choice. Hash keys
only accelerate lookup; full-vector equality verifies every match.

Each JEPA cell corresponds to a 2x2 tile in the 48x48 Wan latent:
  provenance_copy:     unchanged=z_s[q]; moved=z_s[p]; unmatched=f_e[q].
  provenance_local:    unchanged=z_s[q]; moved=z_s[p]; unmatched=local(z_s).
  provenance_residual: unchanged=z_s[q]; moved=f_e[q]+(z_s[p]-f_s[p]);
                       unmatched=f_e[q].
The local comparator averages immutable source tiles in a radius-2 square,
excluding all changed destination cells AND inferred moved source cells.
No eligible local donor -> eligible global mean. No eligible global donor
-> entire immutable source mean, explicitly counted (not claimed background).
No-F comparator values are independent of both F predictions.

WHY: the previous z_s+F(j_e)-F(j_s) retains reconstruction residual/detail at
the old coordinates. These methods test moving that residual or the entire
source latent tile. Exact source values at unchanged cells prevent the
projector's spatial GroupNorm from changing unrelated conditioning latents.

SCOPE: exact provenance is an editor-specific input channel, not learned
semantic correspondence or evidence of a uniquely JEPA advantage. Ordinary
genuine-target encodings can have no exact matches and then the learned arms
fall back to F everywhere; this is advertised behavior, not an upper bound.
Contextual VAE tiles may carry background/seams. Exact latent preservation
does not guarantee preserved decoded pixels, and no video rollout is tested.

OPERATOR INPUTS ONLY: source JEPA, edited JEPA, source Wan latent, and the two
frozen F predictions. No displacement, selection, target mask/RGB, source RGB,
or renderer metadata. The development CLI separately creates feature edits
using the existing Day10 API, then scores against privileged references.

CLI evaluates all eight EXISTING development scenes and all five feature
conditions. Initial CNN results are retained as comparators. Full follow-up
latent evidence is saved for every scene; combined renderer bundles are saved
for the fixed first two scenes. Optional decoding is explicitly scoped.
Original training, evaluation, and rendering files are never modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = "day11_development_provenance_transport_v1"
ROUTES = ("provenance_copy", "provenance_local", "provenance_residual")
FEATURE_ARMS = ("source", "genuine_target", "copy_repair", "wrong_direction", "shuffled")
NEW_ARMS = tuple(route + "_" + condition for route in ROUTES for condition in FEATURE_ARMS)
GRID, CHANNELS, LATENT_CHANNELS, LATENT_GRID = 24, 1024, 16, 48
LOCAL_RADIUS = 2


def _array(value, shape, name):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32:
            raise ValueError(name + " must be float32")
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite float32 {shape}, got {value.shape}/{value.dtype}")
    return np.ascontiguousarray(value)


def _vector_key(vector):
    # +0 and -0 compare equal; canonicalize only the lookup key accordingly.
    canonical = vector.copy()
    canonical[canonical == 0] = 0
    return hashlib.sha256(canonical.tobytes()).digest()


def infer_provenance(source_jepa, edited_jepa):
    """Exact value matching, with identity precedence and ambiguous-move rejection."""
    source = _array(source_jepa, (1, GRID, GRID, CHANNELS), "source_jepa").reshape(-1, CHANNELS)
    edited = _array(edited_jepa, (1, GRID, GRID, CHANNELS), "edited_jepa").reshape(-1, CHANNELS)
    index = defaultdict(list)
    for p, vector in enumerate(source):
        index[_vector_key(vector)].append(p)
    identity = np.all(source == edited, axis=1)
    mapping = np.full(GRID * GRID, -1, np.int32)
    mapping[identity] = np.flatnonzero(identity)
    ambiguous = []
    for q in np.flatnonzero(~identity):
        matches = [p for p in index.get(_vector_key(edited[q]), ())
                   if np.array_equal(source[p], edited[q])]
        if len(matches) > 1:
            ambiguous.append({"destination_index": int(q), "source_indices": matches})
        elif matches:
            mapping[q] = matches[0]
    if ambiguous:
        raise ValueError("Ambiguous exact JEPA provenance: " + json.dumps(ambiguous[:8]))
    matched = mapping >= 0
    moved = matched & ~identity
    unmatched = ~matched
    # Duplicates are harmless at a directly unchanged coordinate; report them.
    duplicate_groups = sum(len(group) > 1 for group in index.values())
    statistics = {
        "total_cells": GRID * GRID,
        "identity_cells": int(identity.sum()), "matched_cells": int(matched.sum()),
        "moved_cells": int(moved.sum()), "unmatched_cells": int(unmatched.sum()),
        "ambiguous_moved_cells": 0, "source_duplicate_hash_groups": int(duplicate_groups),
        "matched_fraction": float(matched.mean()),
        "moved_match_fraction_of_changed": float(moved.sum() / (~identity).sum())
        if (~identity).any() else None,
        "zero_edit": bool(identity.all()),
        "mapping_sha256": hashlib.sha256(mapping.tobytes()).hexdigest(),
        "matching": "exact finite float32 vector equality; hash prefilter; identity precedence",
    }
    return mapping.reshape(GRID, GRID), identity.reshape(GRID, GRID), statistics


def _tiles(latent):
    # [C,24,2,24,2] -> one [C,2,2] tile at each coarse location.
    return latent.reshape(LATENT_CHANNELS, GRID, 2, GRID, 2).transpose(1, 3, 0, 2, 4).copy()


def _untile(tiles):
    return tiles.transpose(2, 0, 3, 1, 4).reshape(LATENT_CHANNELS, LATENT_GRID, LATENT_GRID).copy()


def _local_fill(source_tiles, excluded, unmatched):
    result = source_tiles.copy()
    eligible = ~excluded
    eligible_mean = source_tiles[eligible].mean(axis=0, dtype=np.float32) if eligible.any() else None
    entire_mean = source_tiles.mean(axis=(0, 1), dtype=np.float32)
    counts = {"local_donor_queries": 0, "eligible_global_fallback_queries": 0,
              "entire_source_mean_fallback_queries": 0,
              "eligible_source_cells": int(eligible.sum()), "radius": LOCAL_RADIUS}
    for y, x in np.argwhere(unmatched):
        y0, y1 = max(0, y - LOCAL_RADIUS), min(GRID, y + LOCAL_RADIUS + 1)
        x0, x1 = max(0, x - LOCAL_RADIUS), min(GRID, x + LOCAL_RADIUS + 1)
        local = eligible[y0:y1, x0:x1]
        if local.any():
            result[y, x] = source_tiles[y0:y1, x0:x1][local].mean(axis=0, dtype=np.float32)
            counts["local_donor_queries"] += 1
        elif eligible_mean is not None:
            result[y, x] = eligible_mean
            counts["eligible_global_fallback_queries"] += 1
        else:
            result[y, x] = entire_mean
            counts["entire_source_mean_fallback_queries"] += 1
    return result, counts


def transport_predictions(source_jepa, edited_jepa, source_wan,
                          source_prediction, edited_prediction):
    """Return three methods, provenance, and exact invariants; no geometry input."""
    latent_shape = (LATENT_CHANNELS, LATENT_GRID, LATENT_GRID)
    source_wan = _array(source_wan, latent_shape, "source_wan")
    source_prediction = _array(source_prediction, latent_shape, "source_prediction")
    edited_prediction = _array(edited_prediction, latent_shape, "edited_prediction")
    mapping, identity, statistics = infer_provenance(source_jepa, edited_jepa)
    moved, unmatched = (mapping >= 0) & ~identity, mapping < 0
    source_tiles, f_source, f_edit = map(_tiles, (source_wan, source_prediction, edited_prediction))
    copied = source_tiles.copy()
    residual = source_tiles.copy()
    copied[unmatched] = f_edit[unmatched]
    residual[unmatched] = f_edit[unmatched]
    excluded = ~identity
    excluded = excluded.copy()
    excluded.flat[mapping[moved]] = True
    local, local_statistics = _local_fill(source_tiles, excluded, unmatched)
    flat_source = source_tiles.reshape(GRID * GRID, LATENT_CHANNELS, 2, 2)
    flat_f_source = f_source.reshape(GRID * GRID, LATENT_CHANNELS, 2, 2)
    for y, x in np.argwhere(moved):
        p = int(mapping[y, x])
        copied[y, x] = flat_source[p]
        local[y, x] = flat_source[p]
        residual[y, x] = f_edit[y, x] + (flat_source[p] - flat_f_source[p])
    outputs = dict(zip(ROUTES, map(_untile, (copied, local, residual))))
    for name, result in outputs.items():
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite transport output: " + name)
        if not np.array_equal(_tiles(result)[identity], source_tiles[identity]):
            raise AssertionError("Unchanged Wan tiles changed: " + name)
        if statistics["zero_edit"] and not np.array_equal(result, source_wan):
            raise AssertionError("Zero edit is not exact source identity: " + name)
    for y, x in np.argwhere(moved):
        if not np.array_equal(copied[y, x], flat_source[int(mapping[y, x])]):
            raise AssertionError("Destination did not copy immutable source tile")
    statistics.update({"local_fill": local_statistics, "unchanged_wan_tiles_exact": True,
                       "copy_destination_tiles_exact": True,
                       "copy_and_residual_unmatched_equal_F_edit": True,
                       "copy_and_local_destinations_equal": True,
                       "input_budget": ["source_jepa", "edited_jepa", "source_wan",
                                        "source_prediction", "edited_prediction"]})
    return outputs, {"source_index": mapping, "identity": identity,
                     "moved": moved, "unmatched": unmatched}, statistics


def _evaluator():
    name = "day11_transport_evaluation"
    spec = importlib.util.spec_from_file_location(name, HERE / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _new_metadata():
    routes = {
        "provenance_copy": "exact inferred source Wan tile copy; frozen F(edit) at unmatched changed cells",
        "provenance_local": "same inferred source Wan tile copy; deterministic source-only local tile mean at unmatched cells; no F values",
        "provenance_residual": "F(edit)+transport[A(source)-F(source)] at copied cells; F(edit) unmatched; exact source unchanged",
    }
    return {route + "_" + feature: {
        "oracle": feature == "genuine_target", "route": description,
        "input": feature + " JEPA plus source JEPA and source Wan latent",
        "scope": "exact token provenance; genuine encodings may fall back almost everywhere"}
        for route, description in routes.items() for feature in FEATURE_ARMS}


def _summary(rows, arms):
    methods = {}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        values, counts = {}, {}
        for key in sorted(set().union(*(row.keys() for row in selected))):
            if key.endswith("_mse") or key.endswith("_ratio"):
                available = [row[key] for row in selected if row.get(key) is not None]
                values[key] = float(np.mean(available)) if available else None
                counts[key] = len(available)
        methods[arm] = {"scene_count": len(selected),
                        "decoded_scene_count": sum(bool(row.get("decoded")) for row in selected),
                        **values, "metric_scene_counts": counts}
    return methods


def evaluate(checkpoint, cache_root, target_root, baseline_root, out,
             decode_scenes="none", threads=4, model_cache="/tmp/day11-model", device="cpu"):
    ev = _evaluator()
    torch.set_num_threads(threads)
    torch.manual_seed(ev.SHUFFLE_SEED)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dataset = ev.data.NativeImageDataset(cache_root, "dev")
    targets = ev.data.LatentTargetStore(target_root, dataset.manifest_sha256)
    targets.validate_dataset_coverage(dataset)
    projector, saved = ev.load_projector(checkpoint, dataset, targets, device)
    if saved["arm"] != "cnn":
        raise ValueError("This separately declared follow-up uses the original frozen CNN only")
    baseline_root, out = Path(baseline_root).resolve(), Path(out).resolve()
    original = json.loads((baseline_root / "evaluation_manifest.json").read_text())
    if (not original["complete"] or original["translator_arm"] != "cnn"
            or original["checkpoint_sha256"] != ev.data.sha256(checkpoint)
            or original["jepa_manifest_sha256"] != dataset.manifest_sha256
            or original["target_manifest_sha256"] != targets.manifest_sha256
            or original["all_development_seeds"] != list(range(13500, 13508))
            or original["runtime"]["device"] != device):
        raise ValueError("Initial CNN evaluation does not match this checkpoint/data/runtime")
    for path, digest in original["source_hashes"].items():
        if ev.data.sha256(REPO / path) != digest:
            raise ValueError("Original numerical source changed: " + path)
    if ev.data.sha256(baseline_root / "metrics.json") != original["metrics_sha256"]:
        raise ValueError("Initial CNN metrics changed")
    original_rows = json.loads((baseline_root / "metrics.json").read_text())
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty new output directory; previous evidence is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    animation = ev._legacy_animation()
    vae = ev.load_frozen_vae(targets, model_cache, device) if decode_scenes != "none" else None
    arms = {**original["arms"], **_new_metadata()}
    manifest = {
        "version": VERSION, "complete": False, "split": "existing eight development scenes only",
        "fresh_test_accessed": False, "checkpoint_sha256": original["checkpoint_sha256"],
        "jepa_manifest_sha256": dataset.manifest_sha256,
        "target_manifest_sha256": targets.manifest_sha256, "vae": targets.manifest["vae"],
        "source_hashes": {**original["source_hashes"],
                          "experiments/day11/transport.py": ev.data.sha256(Path(__file__))},
        "initial_evaluation_manifest_sha256": ev.data.sha256(baseline_root / "evaluation_manifest.json"),
        "initial_metrics_sha256": original["metrics_sha256"], "arms": arms,
        "renderer_seeds": list(ev.RENDERER_SEEDS), "all_development_seeds": list(range(13500, 13508)),
        "decode_scenes": decode_scenes, "local_radius": LOCAL_RADIUS,
        "algorithm": __doc__, "training": "none; original fixed epoch150 CNN",
        "runtime": {"device": device, "threads": threads, "python": platform.python_version(),
                    "torch": str(torch.__version__), "numpy": str(np.__version__)},
        "aggregation": "Equal-scene means; per-metric valid scene counts; decoding subset explicit; no holdout claim",
        "scenes": [],
    }
    ev._write_json(out / "evaluation_manifest.json", manifest)
    rows, all_counts = [], []
    initial_scenes = {scene["scene"]: scene for scene in original["scenes"]}
    for scene_index, row in enumerate(dataset.rows):
        scene, seed = row["spec"]["name"], int(row["spec"]["seed"])
        record = initial_scenes[scene]
        baseline_path = ev.data._child_path(baseline_root, record["path"])
        if ev.data.sha256(baseline_path) != record["sha256"]:
            raise ValueError("Initial scene bundle changed: " + scene)
        with np.load(baseline_path, allow_pickle=False) as archive:
            bundle = {key: archive[key].copy() for key in archive.files}
        metadata = json.loads(bytes(bundle.pop("metadata_json")).decode())
        source_example, target_example = dataset[2 * scene_index], dataset[2 * scene_index + 1]
        source = source_example["jepa"].numpy().transpose(1, 2, 0)[None].copy()
        fraction, latent_masks, rgb_masks, _ = ev._scoring_masks(dataset, row, animation)
        states, _, _ = animation.translate_features(source, fraction, [0, row["spec"]["dx"]])
        permutation = bundle["shuffle_permutation"]
        conditions = {
            "source": source,
            "genuine_target": target_example["jepa"].numpy().transpose(1, 2, 0)[None].copy(),
            "copy_repair": states["copy_repair"][1],
            "wrong_direction": states["wrong_direction"][1],
            "shuffled": states["copy_repair"][1].reshape(576, 1024)[permutation].reshape(1, 24, 24, 1024).copy(),
        }
        evidence, counts, new_latents = {}, {}, {}
        for condition, features in conditions.items():
            with torch.inference_mode():
                tensor = torch.from_numpy(features.transpose(0, 3, 1, 2).copy()).to(device)
                predicted = projector(tensor).cpu().numpy()[0]
            expected = bundle["latent__absolute_" + condition]
            if not np.array_equal(predicted, expected):
                raise AssertionError("Frozen CNN did not reproduce original prediction exactly: " + scene + "/" + condition)
            outputs, provenance, statistics = transport_predictions(
                source, features, bundle["latent__true_source"],
                bundle["latent__absolute_source"], predicted)
            for route, value in outputs.items():
                new_latents[route + "_" + condition] = value
            evidence.update({"provenance__" + condition + "__" + key: value
                             for key, value in provenance.items()})
            counts[condition] = statistics
            all_counts.append({"scene": scene, "seed": seed, "condition": condition, **statistics})
        latents = {arm: bundle["latent__" + arm] for arm in original["arms"]}
        latents.update(new_latents)
        scene_rows = [dict(value) for value in original_rows if value["scene"] == scene]
        for arm, value in new_latents.items():
            result = {"scene": scene, "seed": seed, "shift_px": int(row["spec"]["dx"]),
                      "arm": arm, "oracle": arms[arm]["oracle"], "translator_arm": "cnn",
                      "followup": VERSION}
            result.update(ev._error_metrics(value, latents["true_target"], latents["true_source"],
                                            latent_masks, "latent", 0))
            scene_rows.append(result)
        decode = decode_scenes == "all" or (decode_scenes == "renderer" and seed in ev.RENDERER_SEEDS)
        decoded_source = None
        for result in scene_rows:
            result["decoded"] = bool(decode)
            if decode:
                arm = result["arm"]
                with torch.inference_mode():
                    normalized = torch.from_numpy(latents[arm][None, :, None]).to(device)
                    decoded = vae.decode(ev.bridge.denormalize_wan_latents(vae, normalized)).sample
                if tuple(decoded.shape) != (1, 3, 1, 384, 384) or not torch.isfinite(decoded).all():
                    raise ValueError("Unexpected/nonfinite VAE reconstruction")
                rgb = decoded[0, :, 0].float().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().numpy().copy()
                if arm == "true_source":
                    decoded_source = rgb
                evidence["decoded_rgb__" + arm] = rgb
                result.update(ev._error_metrics(
                    rgb, bundle["target_rgb"].astype(np.float32) / 255,
                    bundle["source_rgb"].astype(np.float32) / 255, rgb_masks, "rgb", 2))
                if decoded_source is None:
                    raise AssertionError("True-source decode must precede preservation scoring")
                for region in ("background", "distractor"):
                    selected = rgb_masks[region]
                    result["decoded_source_" + region + "_mse"] = float(
                        np.square(rgb.astype(np.float64) - decoded_source)[selected].mean()) if selected.any() else None
        scene_metadata = {
            **metadata, "version": VERSION, "arms": arms, "provenance": counts,
            "initial_bundle_sha256": record["sha256"],
            "transport_source_sha256": manifest["source_hashes"]["experiments/day11/transport.py"],
            "fresh_test_accessed": False, "decoded": bool(decode),
            "latent_sha256": {arm: ev.data.array_sha256(value) for arm, value in latents.items()},
        }
        evidence.update({"latent__" + arm: value for arm, value in new_latents.items()})
        evidence["metadata_json"] = np.frombuffer(json.dumps(scene_metadata, sort_keys=True).encode(), np.uint8)
        evidence_path = out / "analysis" / (scene + ".npz")
        ev._save_npz(evidence_path, evidence)
        ev._write_json(out / "analysis" / (scene + "_metrics.json"), scene_rows)
        entry = {"scene": scene, "seed": seed, "shift_px": int(row["spec"]["dx"]),
                 "path": str(evidence_path.relative_to(out)), "sha256": ev.data.sha256(evidence_path),
                 "renderer_bundle": seed in ev.RENDERER_SEEDS, "decoded": bool(decode)}
        if seed in ev.RENDERER_SEEDS:
            bundle.update(evidence)
            renderer_path = out / "scenes" / (scene + ".npz")
            ev._save_npz(renderer_path, bundle)
            entry.update({"renderer_path": str(renderer_path.relative_to(out)),
                          "renderer_sha256": ev.data.sha256(renderer_path)})
        manifest["scenes"].append(entry)
        rows.extend(scene_rows)
        ev._write_json(out / "evaluation_manifest.json", manifest)
        print("PROVENANCE_DEV_COMPLETE", scene, "decoded", decode, "counts",
              {name: {k: value[k] for k in ("identity_cells", "moved_cells", "unmatched_cells")}
               for name, value in counts.items()}, flush=True)
    if len(manifest["scenes"]) != 8 or len(rows) != 8 * len(arms):
        raise AssertionError("Every development scene and every old/new arm must be retained")
    ev._write_json(out / "metrics.json", rows)
    ev._write_csv(out / "metrics.csv", rows)
    ev._write_json(out / "provenance_counts.json", all_counts)
    summary = {"version": VERSION, "scene_count": 8, "arms": arms,
               "methods": _summary(rows, arms), "aggregation": manifest["aggregation"],
               "decode_scenes": decode_scenes, "fresh_test_accessed": False,
               "scope": "Development follow-up; provenance transport, not a uniquely JEPA semantic capability",
               "limits": ["No optimizer or new checkpoint; genuine F predictions reused and independently repeated.",
                          "Source-only local fill can include other scene content; no object oracle is used.",
                          "Genuine target JEPA is privileged and often unmatched, triggering fallback.",
                          "Conditioning-latent exactness is not a decoded/rendered preservation guarantee.",
                          "Original CNN metrics retained; optional RGB metrics use an explicitly declared subset."]}
    ev._write_json(out / "summary.json", summary)
    manifest.update({"complete": True, "metrics_sha256": ev.data.sha256(out / "metrics.json"),
                     "summary_sha256": ev.data.sha256(out / "summary.json"),
                     "provenance_counts_sha256": ev.data.sha256(out / "provenance_counts.json"),
                     "renderer_scene_paths": [entry["renderer_path"] for entry in manifest["scenes"]
                                              if entry["renderer_bundle"]]})
    ev._write_json(out / "evaluation_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "cache-root", "target-root", "baseline-root", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--decode-scenes", choices=("none", "renderer", "all"), default="none")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-cache", default="/tmp/day11-model")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    evaluate(args.checkpoint, args.cache_root, args.target_root, args.baseline_root, args.out,
             args.decode_scenes, args.threads, args.model_cache, args.device)


if __name__ == "__main__":
    main()
