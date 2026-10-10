"""Package completed Day7 evidence without altering frozen training or results.

The full ZIP preserves original run-file bytes, including nine selected heads
and all primary logits/future predictions. Reproducible feature cache/full_features
directories are excluded. A separate compact tree strips head weights and keeps
only scores/cells/eligible from primary NPZs; its limitations are explicit.
No model inference, scene generation, selection, or result recomputation occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent
ARMS = ("retrieval_only", "predictive", "coordinate_only")
SEEDS = (1701, 1702, 1703)
BASELINES = ("global_full", "selected_day4", "global_projected_centered")
CORE_PREDICTION_KEYS = ("scores", "cells", "eligible")
CLIP_BINDING_KEYS = ("name", "split", "seed", "condition", "path", "sha256", "rgb_sha256")
INTERVENTIONS = ("forecast_to_anchor", "owner_context_zero")
TEXT_MEDIA_SUFFIXES = {".json", ".csv", ".md", ".txt", ".log", ".yaml", ".yml",
                       ".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".mp4", ".webm", ".html"}
FORMAT_VERSION = "day7_artifact_export_v1"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _load(path):
    return json.loads(Path(path).read_text())


def _safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact paths must stay relative to their root")
    path = Path(root) / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected an ordinary file: {relative}")
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escaped its source root")
    return path


def _count_csv(path):
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return sum(1 for _ in reader)


def _source_files(freeze):
    """Check frozen files and copy helpers separately as current artifact source."""
    config = freeze["training_config"]
    expected = {"experiments/" + name: value for name, value in config["source_hashes"].items()}
    expected.update(config["feature_binding"]["extractor_provenance"]["extractor_sources"])
    for relative, digest in expected.items():
        if sha256(_safe_path(REPOSITORY, relative)) != digest:
            raise ValueError(f"Frozen source changed before export: {relative}")
    paths = [p for p in HERE.iterdir() if p.is_file() and p.suffix in {".py", ".md", ".ipynb"}]
    paths += [REPOSITORY / name for name in expected]
    return {"source/" + str(p.relative_to(REPOSITORY)): p for p in sorted(set(paths))}


def _validate_feature_manifest(manifest, training_binding, test_binding):
    """Validate metadata without needing the intentionally omitted feature arrays."""
    for key in ("mean", "projection_sha256", "extractor_provenance"):
        if test_binding.get(key) != training_binding[key]:
            raise ValueError(f"Test feature {key} differs from the frozen training binding")
    for key in ("mean", "projection_sha256"):
        if manifest.get(key) != training_binding[key]:
            raise ValueError(f"Feature manifest {key} differs from frozen results")
    provenance = training_binding["extractor_provenance"]
    if {key: manifest.get(key) for key in provenance} != provenance:
        raise ValueError("Feature manifest extractor provenance differs from frozen results")
    expected = training_binding["clips"] + test_binding["clips"]
    actual = manifest.get("clips", [])
    if len(expected) != 63 or len(actual) != 63:
        raise ValueError("Feature manifest must cover all63 train/dev/test scenes")
    def indexed(records):
        values = [{key: record[key] for key in CLIP_BINDING_KEYS} for record in records]
        mapping = {record["name"]: record for record in values}
        if len(mapping) != len(values):
            raise ValueError("Duplicate feature scene metadata")
        return mapping
    if indexed(actual) != indexed(expected):
        raise ValueError("Feature scene metadata differs from the frozen train/dev/test bindings")
    if {split: sum(record["split"] == split for record in actual) for split in ("train", "dev", "test")} != {"train": 36, "dev": 9, "test": 18}:
        raise ValueError("Feature manifest split counts differ from the fixed cohort")


def _feature_directory(run, frozen, result, explicit=None):
    binding = frozen["training_config"]["feature_binding"]
    candidates = ([Path(explicit)] if explicit else
                  [run / "features", *sorted(run.parent.glob("*features*"))])
    matches = []
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _load(manifest_path)
        if (manifest.get("mean") == binding["mean"] and
                manifest.get("projection_sha256") == binding["projection_sha256"]):
            _validate_feature_manifest(manifest, binding, result["test_feature_binding"])
            matches.append(candidate.resolve())
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise ValueError("Need exactly one matching features directory; supply --features explicitly")
    features = matches[0]
    manifest = _load(features / "manifest.json")
    mean_path = _safe_path(features, binding["mean"]["path"])
    projection_path = _safe_path(features, manifest.get("projection_path", "projection.npy"))
    if sha256(mean_path) != binding["mean"]["sha256"] or sha256(projection_path) != binding["projection_sha256"]:
        raise ValueError("Mean/projection bytes differ from the trained checkpoint freeze")
    return features, {"manifest.json": features / "manifest.json",
                      "mean.npy": mean_path, "projection.npy": projection_path}


def _validate_audit(run, audit_path, prediction_paths, prediction_rows):
    audit = _load(audit_path)
    if (audit.get("status") != "passed" or audit.get("independent_prediction_rows") != prediction_rows or
            audit.get("independent_clips") != 18):
        raise ValueError("Independent analysis is not a complete passed audit")
    if audit.get("audit_source_sha256") != sha256(HERE / "analyze.py"):
        raise ValueError("Independent analysis was produced by a different analyzer source")
    hashes = audit.get("source_files", {})
    required = set(prediction_paths) | {"test/results.json", "checkpoint_freeze.json",
                                      "test/prediction_rows.csv", "test/future_rows.csv", "training_config.json"}
    if not required <= set(hashes):
        raise ValueError("Independent audit does not bind all primary evidence")
    for relative, digest in hashes.items():
        if sha256(_safe_path(run, relative)) != digest:
            raise ValueError(f"Independent audit is stale for {relative}")


def _validate_diagnostic_rows(path, diagnostic, visual, scenes):
    expected = {(method, scene, intervention) for method in visual for scene in scenes for intervention in INTERVENTIONS}
    key = lambda row: (row["method"], row["scene"], row["intervention"])
    rows = diagnostic["sensitivity_rows"]
    if len(rows) != len(expected) or {key(row) for row in rows} != expected:
        raise ValueError("Diagnostic JSON has missing/duplicate/unexpected intervention records")
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames
        csv_rows = list(reader)
    if not headers or len(headers) != len(set(headers)):
        raise ValueError("Diagnostic CSV header is missing or duplicated")
    if len(csv_rows) != len(expected) or {key(row) for row in csv_rows} != expected:
        raise ValueError("Diagnostic CSV has missing/duplicate/unexpected intervention records")
    recorded = {key(row): row for row in csv_rows}
    for row in rows:
        normalized = {name: "" if value is None else str(value) for name, value in row.items()}
        if set(headers) != set(row) or recorded[key(row)] != normalized:
            raise ValueError("Diagnostic CSV values differ from JSON sensitivity records")
    return len(rows)


def validate_completed(run):
    """Integrity/count gate only; deliberately imports no scene/model code."""
    run = Path(run)
    freeze_path = _safe_path(run, "checkpoint_freeze.json")
    frozen = _load(freeze_path)
    result = _load(_safe_path(run, "test/results.json"))
    if result.get("checkpoint_freeze_sha256") != sha256(freeze_path) or result.get("checkpoint_freeze") != frozen:
        raise ValueError("Completed test report is not bound to this checkpoint freeze")
    if result.get("split") != "test" or result.get("all_seeds_reported") is not True or result.get("test_selection") != "none":
        raise ValueError("A complete unselected test report is required")
    records = frozen.get("models", [])
    if len(records) != 9 or {(r["arm"], r["seed"]) for r in records} != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Expected all nine selected checkpoints")
    methods = set(BASELINES) | {f"{a}_seed{s}" for a in ARMS for s in SEEDS}
    if set(result["methods"]) != methods:
        raise ValueError("Expected three baselines plus all nine learned methods")
    specs = frozen["test_seed_manifest"]
    scenes = [s["name"] for s in specs]
    if len(scenes) != 18 or len(set(scenes)) != 18:
        raise ValueError("Expected exactly eighteen unique held-out scenes")
    checkpoints = []
    for record in records:
        path = _safe_path(run, record["path"])
        if sha256(path) != record["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint checksum mismatch")
        selected_path = _safe_path(run, str(Path(record["path"]).with_name("selected.json")))
        if sha256(selected_path) != record["selection_sha256"]:
            raise ValueError("Development selection checksum mismatch")
        selected = _load(selected_path)
        curves = _safe_path(run, str(Path(record["path"]).with_name("curves.csv")))
        if sha256(curves) != selected["curves_sha256"] or _count_csv(curves) != 30:
            raise ValueError("Need the original complete thirty-epoch curve for every model")
        checkpoints.append(str(path.relative_to(run)))
    predictions = {f"test/predictions/{method}/{scene}.npz" for method in methods for scene in scenes}
    actual = {str(p.relative_to(run)) for p in (run / "test/predictions").rglob("*.npz")}
    if actual != predictions:
        raise ValueError(f"Primary prediction files incomplete/unexpected: expected216, found{len(actual)}")
    prediction_rows = _count_csv(_safe_path(run, "test/prediction_rows.csv"))
    if prediction_rows != 12 * 18 * 31 * 4:
        raise ValueError("Primary prediction CSV must contain26,784 rows")
    expected_future = sum(result["methods"][f"{arm}_seed{seed}"]["future"]["targets"] for arm in ARMS for seed in SEEDS)
    future_rows = _count_csv(_safe_path(run, "test/future_rows.csv"))
    if future_rows != expected_future:
        raise ValueError("Future CSV count differs from the completed report")
    passed = [p for p in run.rglob("independent_analysis.json") if _load(p).get("status") == "passed"]
    if not passed:
        raise ValueError("Run the independent analysis successfully before packaging")
    for audit_path in passed:
        _validate_audit(run, audit_path, predictions, prediction_rows)
    diagnostic_path = _safe_path(run, "diagnostics/mechanism_reliance_v1/reliance_results.json")
    diagnostic = _load(diagnostic_path)
    visual = {f"{arm}_seed{seed}" for arm in ("retrieval_only", "predictive") for seed in SEEDS}
    if (set(diagnostic["methods"]) != visual or diagnostic.get("all_six_visual_checkpoints_reported") is not True or
            diagnostic.get("training_or_selection_performed") is not False or diagnostic.get("primary_predictions_overwritten") is not False):
        raise ValueError("Need completed, nonselecting diagnostics for all six visual heads")
    if (diagnostic.get("diagnostic_source_sha256") != sha256(HERE / "diagnose.py") or
            diagnostic.get("completion_guard_source_sha256") != sha256(HERE / "analyze.py") or
            diagnostic.get("frozen_source_hashes") != frozen["training_config"]["source_hashes"]):
        raise ValueError("Diagnostic source provenance differs from the archived helper/frozen sources")
    for key, relative in (("primary_results_sha256", "test/results.json"),
                          ("primary_prediction_rows_sha256", "test/prediction_rows.csv"),
                          ("checkpoint_freeze_sha256", "checkpoint_freeze.json")):
        if diagnostic[key] != sha256(run / relative):
            raise ValueError("Diagnostic evidence is not bound to the current primary results")
    diagnostic_rows = _validate_diagnostic_rows(
        _safe_path(run, "diagnostics/mechanism_reliance_v1/per_clip_reliance.csv"), diagnostic, visual, scenes)
    sources = _source_files(frozen)
    return frozen, result, predictions, sources, {
        "selected_checkpoints": len(checkpoints), "checkpoint_paths": sorted(checkpoints),
        "methods": 12, "held_out_clips": 18, "primary_prediction_files": len(predictions),
        "primary_prediction_rows": prediction_rows, "future_diagnostic_rows": future_rows,
        "completed_epoch_curve_rows": 9 * 30, "mechanism_diagnostic_rows": diagnostic_rows,
        "independent_analysis": [str(p.relative_to(run)) for p in sorted(passed)]}


def _zip_info(name):
    info = zipfile.ZipInfo(name, ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    return info


def _zip_file(bundle, name, path):
    with Path(path).open("rb") as source, bundle.open(_zip_info(name), "w", force_zip64=True) as destination:
        shutil.copyfileobj(source, destination, 1 << 20)


def compact_prediction(source, destination, learned):
    """Write a deterministic reduced archive, preserving three array values/dtypes."""
    with np.load(source, allow_pickle=False) as archive:
        required = set(CORE_PREDICTION_KEYS) | ({"logits", "future", "visibility", "retrieved"} if learned else set())
        if not required <= set(archive.files):
            raise ValueError(f"Original primary NPZ lacks required arrays: {source}")
        arrays = {key: archive[key].copy() for key in CORE_PREDICTION_KEYS}
        if arrays["scores"].shape != (32, 4) or arrays["cells"].shape != (32, 4, 2) or arrays["eligible"].shape != (32, 4):
            raise ValueError("Primary prediction shape mismatch")
        if arrays["eligible"].dtype != bool or not np.isfinite(arrays["scores"]).all():
            raise ValueError("Invalid prediction score/eligibility values")
        if not np.issubdtype(arrays["cells"].dtype, np.integer) or not ((arrays["cells"] >= 0) & (arrays["cells"] < 24)).all():
            raise ValueError("Invalid prediction grid cells")
        if learned:
            if (archive["logits"].shape != (32, 4, 577) or archive["future"].shape != (32, 4, 256) or
                    archive["visibility"].shape != (32, 4) or archive["retrieved"].shape != (32, 4, 256)):
                raise ValueError("Original learned logit/future/visibility/retrieved shapes differ")
            for key in ("logits", "future", "visibility", "retrieved"):
                if not np.isfinite(archive[key]).all():
                    raise ValueError("Nonfinite original learned prediction arrays")
        removed = sorted(set(archive.files) - set(CORE_PREDICTION_KEYS))
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for key in sorted(arrays):
            buffer = io.BytesIO(); np.save(buffer, arrays[key], allow_pickle=False)
            bundle.writestr(_zip_info(key + ".npy"), buffer.getvalue())
    with np.load(destination, allow_pickle=False) as archive:
        if set(archive.files) != set(CORE_PREDICTION_KEYS):
            raise AssertionError("Unexpected compact prediction keys")
        for key in arrays:
            if archive[key].dtype != arrays[key].dtype or not np.array_equal(archive[key], arrays[key]):
                raise AssertionError("Compact prediction changed values or dtypes")
    return removed


def _record(name, path, **extra):
    return {"path": name, "bytes": Path(path).stat().st_size, "sha256": sha256(path), **extra}


def _excluded_cache(path, features):
    try:
        parts = path.relative_to(features).parts
    except ValueError:
        return False
    return bool({"cache", "full_features"}.intersection(parts))


def export(run, out, features=None):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.is_relative_to(run):
        raise ValueError("Export output must be outside the run to avoid recursive archives")
    frozen, result, prediction_paths, sources, counts = validate_completed(run)
    features, sidecars = _feature_directory(run, frozen, result, features)
    if out.is_relative_to(features):
        raise ValueError("Export output must be outside the feature directory")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty export directory; existing evidence is not overwritten")
    out.mkdir(parents=True, exist_ok=True)
    compact = out / "compact"; compact.mkdir()
    original_files, cache_omissions = {}, []
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not included in evidence: {path.relative_to(run)}")
        if not path.is_file():
            continue
        relative = str(path.relative_to(run))
        if _excluded_cache(path, features):
            cache_omissions.append({"path": relative, "bytes": path.stat().st_size,
                                    "reason": "reproducible backbone feature arrays excluded by policy"})
        else:
            original_files[relative] = path
    # The normal layout is run/features; support a matching external feature root too.
    for name, path in sidecars.items():
        if not path.is_relative_to(run):
            original_files["features/" + name] = path
    full_inventory, compact_inventory, compact_omissions, derivations = [], [], [], []
    archive = out / "full_run.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        for relative, path in sorted(original_files.items()):
            entry = _record("run/" + relative, path, content="original bytes")
            _zip_file(bundle, entry["path"], path); full_inventory.append(entry)
            target = compact / relative
            if relative in prediction_paths:
                method = Path(relative).parts[2]
                removed = compact_prediction(path, target, method not in BASELINES)
                derivations.append({"path": relative, "original_sha256": entry["sha256"],
                                    "removed_arrays": removed, "retained_arrays": list(CORE_PREDICTION_KEYS),
                                    "claim": "retained arrays equal originals, including dtype"})
                compact_inventory.append(_record(relative, target, content="derived scores/cells/eligible only"))
            elif path.suffix.lower() in TEXT_MEDIA_SUFFIXES or relative in {"features/mean.npy", "features/projection.npy"}:
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
                compact_inventory.append(_record(relative, target, content="original bytes"))
            else:
                reason = ("learned checkpoint weight omitted" if path.suffix.lower() in {".pt", ".pth", ".safetensors"}
                          else "non-text/media binary omitted from compact evidence")
                compact_omissions.append({"path": relative, "bytes": path.stat().st_size,
                                          "original_sha256": entry["sha256"], "reason": reason})
        for relative, path in sorted(sources.items()):
            entry = _record(relative, path, content="source snapshot; frozen sources verified separately")
            _zip_file(bundle, relative, path); full_inventory.append(entry)
            target = compact / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
            compact_inventory.append(_record(relative, target, content=entry["content"]))
        manifest = {"format": FORMAT_VERSION, "kind": "full_original_evidence", "counts": counts,
                    "checkpoint_freeze_sha256": sha256(run / "checkpoint_freeze.json"),
                    "primary_results_sha256": sha256(run / "test/results.json"),
                    "exporter_source_sha256": sha256(__file__), "files": full_inventory,
                    "excluded_reproducible_feature_files": cache_omissions,
                    "excluded_external_feature_directories": ["cache", "full_features"] if not features.is_relative_to(run) else [],
                    "manifest_self_hash": "not included to avoid recursion; all listed hashes address payload bytes",
                    "scope": "original run bytes, learned heads and full predictions retained; feature arrays and frozen backbone weights excluded"}
        bundle.writestr(_zip_info("EXPORT_MANIFEST.json"), _json_bytes(manifest))
        full_readme = ("# Full Day7 run evidence\n\n"
            "Files under run/ preserve their exact original bytes, including all nine selected head checkpoints and all216 primary prediction archives (original logits/future arrays included). source/ preserves the matching experiment and artifact-helper source.\n\n"
            "Reproducible features/cache and features/full_features arrays and frozen backbone weights are excluded. The centering mean, projection and feature manifest remain. Saved-prediction analysis can use these full predictions; rerunning head inference or mechanism diagnostics requires regenerating or separately retaining the bound feature caches.\n\n"
            "EXPORT_MANIFEST.json lists SHA-256 payload hashes, exclusions and completeness counts. ZIP timestamps/order are fixed; no model, scene, checkpoint selection or primary metric was changed by export.\n")
        bundle.writestr(_zip_info("README_EXPORT.md"), full_readme.encode())
    # Verify every archived payload against its original content hash, not just ZIP CRC.
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise AssertionError("Full evidence ZIP CRC failure")
        for entry in full_inventory:
            digest = hashlib.sha256()
            with bundle.open(entry["path"]) as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise AssertionError("Full archive changed an original file")
    compact_manifest = {"format": FORMAT_VERSION, "kind": "compact_git_evidence_NOT_full_run",
                        "counts": counts, "full_archive": "../full_run.zip", "full_archive_sha256": sha256(archive),
                        "checkpoint_freeze_sha256": manifest["checkpoint_freeze_sha256"],
                        "files": compact_inventory, "prediction_derivations": derivations,
                        "omitted_files": compact_omissions, "excluded_reproducible_feature_files": cache_omissions,
                        "not_standalone": "Cannot rerun the full analysis/diagnostic pipeline from this tree alone; use full_run.zip, and regenerate or separately retain features when required.",
                        "missing_from_learned_predictions": "logits, future, visibility, retrieved and any other non-core arrays; originals remain in full_run.zip"}
    (compact / "EXPORT_MANIFEST.json").write_bytes(_json_bytes(compact_manifest))
    (compact / "README_EXPORT.md").write_text(
        "# Compact Day7 Git evidence, not a complete runnable archive\n\n"
        "This tree retains original reports, CSVs, curves, selections, manifests, logs, figures/videos and source snapshots. Its216 primary NPZs are derived: only scores, cells and eligible remain, with original values/dtypes verified. All nine learned head weights and full logits/future/visibility/retrieved arrays are omitted.\n\n"
        "The complete original files are preserved in full_run.zip, whose SHA-256 is in EXPORT_MANIFEST.json. This compact tree cannot rerun the full analyze.py or mechanism-diagnostic pipeline standalone. Restore the full archive; head inference/mechanism replay also requires regenerating or separately retaining the bound feature caches.\n\n"
        "The primary scores can be independently counted using the saved cells, eligibility, frozen thresholds and renderer, but that is narrower than full model/future-prediction validation. No omitted array was regenerated or replaced with a synthetic value.\n")
    retained_weights = [e for e in full_inventory if e["path"] in {"run/" + p for p in counts["checkpoint_paths"]}]
    if len(retained_weights) != 9 or len(derivations) != 216:
        raise AssertionError("Export count validation failed")
    if any(p.suffix.lower() in {".pt", ".pth", ".safetensors"} for p in compact.rglob("*")):
        raise AssertionError("Compact export accidentally included learned weights")
    (out / "full_manifest.json").write_bytes(_json_bytes(manifest))
    summary = {"format": FORMAT_VERSION, "full_archive": archive.name, "full_archive_bytes": archive.stat().st_size,
               "full_archive_sha256": sha256(archive), "full_manifest_sha256": sha256(out / "full_manifest.json"),
               "compact_directory": "compact", "compact_manifest_sha256": sha256(compact / "EXPORT_MANIFEST.json"),
               "compact_payload_bytes": sum(e["bytes"] for e in compact_inventory), "counts": counts,
               "full_original_payloads_verified": len(full_inventory), "compact_predictions_array_exact": len(derivations),
               "test_scenes_generated": False, "model_inference_performed": False}
    (out / "export_summary.json").write_bytes(_json_bytes(summary))
    return summary


def self_test():
    """Fabricated NPZ byte/value checks only; no experiment data accessed."""
    def rejected(action):
        try:
            action()
        except (ValueError, KeyError, FileNotFoundError):
            return
        raise AssertionError("Invalid fabricated evidence passed the export gate")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = root / "original.npz"
        arrays = {"scores": np.full((32, 4), .5, np.float32), "cells": np.zeros((32, 4, 2), np.int16),
                  "eligible": np.ones((32, 4), bool), "logits": np.zeros((32, 4, 577), np.float32),
                  "future": np.zeros((32, 4, 256), np.float32), "visibility": np.ones((32, 4), np.float32),
                  "retrieved": np.ones((32, 4, 256), np.float32)}
        np.savez_compressed(original, **arrays)
        original_hash = sha256(original)
        removed = compact_prediction(original, root / "a.npz", True)
        compact_prediction(original, root / "b.npz", True)
        assert sha256(root / "a.npz") == sha256(root / "b.npz")
        assert sha256(original) == original_hash and set(removed) == {"logits", "future", "visibility", "retrieved"}
        broken = root / "broken.npz"
        np.savez_compressed(broken, **{k: v for k, v in arrays.items() if k != "retrieved"})
        rejected(lambda: compact_prediction(broken, root / "reject.npz", True))
        malformed = dict(arrays, retrieved=np.full((32, 4, 256), np.nan, np.float32))
        np.savez_compressed(broken, **malformed)
        rejected(lambda: compact_prediction(broken, root / "reject.npz", True))
        for name in ("a.zip", "b.zip"):
            with zipfile.ZipFile(root / name, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                _zip_file(bundle, "run/original.npz", original)
            with zipfile.ZipFile(root / name) as bundle:
                assert hashlib.sha256(bundle.read("run/original.npz")).hexdigest() == original_hash
        assert sha256(root / "a.zip") == sha256(root / "b.zip")
        features = root / "features"
        assert _excluded_cache(features / "cache" / "x.npz", features)
        assert _excluded_cache(features / "full_features" / "x.npy", features)
        assert not _excluded_cache(features / "mean.npy", features)
        clips = [{"name": f"{split}_{i}", "split": split, "seed": i, "condition": "fabricated",
                  "path": f"cache/{split}_{i}.npz", "sha256": "a" * 64, "rgb_sha256": "b" * 64}
                 for split, count in (("train", 36), ("dev", 9), ("test", 18)) for i in range(count)]
        common = {"mean": {"path": "mean.npy", "sha256": "c" * 64, "fit_split": "train", "fit_clips": 36},
                  "projection_sha256": "d" * 64, "extractor_provenance": {"model": "fabricated"}}
        train_binding = dict(common, clips=clips[:45]); test_binding = dict(common, clips=clips[45:])
        manifest = {"clips": clips, "mean": common["mean"], "projection_sha256": common["projection_sha256"], "model": "fabricated"}
        _validate_feature_manifest(manifest, train_binding, test_binding)
        rejected(lambda: _validate_feature_manifest(dict(manifest, clips=clips[:45]), train_binding, test_binding))
        rejected(lambda: _validate_feature_manifest(dict(manifest, clips=clips[:-1] + [clips[0]]), train_binding, test_binding))
        changed = [dict(record) for record in clips]; changed[-1]["rgb_sha256"] = "e" * 64
        rejected(lambda: _validate_feature_manifest(dict(manifest, clips=changed), train_binding, test_binding))
        rejected(lambda: _validate_feature_manifest(dict(manifest, model="stale"), train_binding, test_binding))
        audit_files = {"test/results.json", "checkpoint_freeze.json", "test/prediction_rows.csv",
                       "test/future_rows.csv", "training_config.json", "test/predictions/fabricated.npz"}
        for name in audit_files:
            path = root / "audit_run" / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fabricated")
        audit = {"status": "passed", "independent_prediction_rows": 26784, "independent_clips": 18,
                 "audit_source_sha256": sha256(HERE / "analyze.py"),
                 "source_files": {name: sha256(root / "audit_run" / name) for name in audit_files}}
        audit_path = root / "audit.json"; audit_path.write_bytes(_json_bytes(audit))
        check_audit = lambda: _validate_audit(root / "audit_run", audit_path, {"test/predictions/fabricated.npz"}, 26784)
        check_audit()
        (root / "audit_run/test/predictions/fabricated.npz").write_bytes(b"stale prediction")
        rejected(check_audit)
        (root / "audit_run/test/predictions/fabricated.npz").write_bytes(b"fabricated")
        changed_audit = dict(audit, source_files={k: v for k, v in audit["source_files"].items() if k != "test/future_rows.csv"})
        audit_path.write_bytes(_json_bytes(changed_audit)); rejected(check_audit)
        audit_path.write_bytes(_json_bytes(dict(audit, audit_source_sha256="0" * 64))); rejected(check_audit)
        diagnostic_rows = [{"method": "fabricated", "scene": "fabricated", "intervention": name,
                            "flag": True, "optional": None, "rate": .125} for name in INTERVENTIONS]
        diagnostic_csv = root / "diagnostic.csv"
        def write_diagnostic(rows):
            with diagnostic_csv.open("w", newline="") as source:
                writer = csv.DictWriter(source, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        check_diagnostic = lambda: _validate_diagnostic_rows(diagnostic_csv, {"sensitivity_rows": diagnostic_rows}, {"fabricated"}, ["fabricated"])
        write_diagnostic(diagnostic_rows); assert check_diagnostic() == 2
        write_diagnostic([diagnostic_rows[0], diagnostic_rows[0]]); rejected(check_diagnostic)
        write_diagnostic([dict(diagnostic_rows[0], rate=.25), diagnostic_rows[1]]); rejected(check_diagnostic)
        rejected(lambda: validate_completed(root))
    return {"retained_arrays_exact": True, "original_bytes_unchanged": True,
            "compact_npz_deterministic": True, "full_zip_payload_exact_and_deterministic": True,
            "both_feature_cache_directories_excluded": True, "incomplete_run_rejected": True,
            "stale_audit_and_missing_bindings_rejected": True, "feature_cohort_and_provenance_bound": True,
            "diagnostic_csv_matches_json": True, "missing_or_nonfinite_retrieved_rejected": True,
            "test_scenes_generated": False, "model_inference_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.run and args.out:
        result = export(args.run, args.out, args.features)
        print(json.dumps({key: result[key] for key in ("full_archive", "full_archive_bytes", "full_archive_sha256", "compact_directory", "counts")}, indent=2))
    else:
        parser.error("Need --run and --out, or --self-test")
