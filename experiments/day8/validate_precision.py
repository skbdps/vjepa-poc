"""Validate CPU extraction precision against the genuine train_11000 pair.

This checks runtime fidelity, NOT latent-editing quality. It evaluates six
errors: source and target precision MSE, each over the global grid, source hole,
and destination. Each is divided by the corresponding genuine source-to-target
edit MSE. All six must be strictly below 1% of that signal. CPU bfloat16 is tried
first; float32 is tried only if bfloat16 fails. No test scene is opened.

Example (TORCH_HOME selects the existing official checkpoint cache)::

    TORCH_HOME=/path/to/torch-cache python experiments/day8/validate_precision.py \
        --reference /path/to/features/cache/train_11000_dx+32.npz \
        --upstream /path/to/vjepa2 --outdir /path/to/precision_check --threads 8

The reference must be the original, genuine train_11000_dx+32 source/target
feature pair. If a neighboring manifest.json is available, its cache checksum,
scene specification, renderer hashes, and backbone identity are verified and
its runtime/source provenance and elapsed time are retained in the report.
Without a manifest, reference provenance is limited to the exact NPZ checksum
and matching rendered mask/RGB supervision; backend/precision/time are unknown.

The earlier scratch invocation measured 62.138017565 s for CPU bfloat16 pair
encoding, versus the Colab reference record's 502.193796251 s for rendering,
encoding and cache writing. Its maximum normalized errors were .008083523344
globally, .0003288611124 in the source hole and .0003139614686 at destination.
Those are historical measurements, not results produced by importing this file.
Its reference NPZ SHA256 was
92806392a95ecdcc98a36750ec0c533aa78dafb05351054cac95a20217ea6d14.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

import data
import extract
import operators


VALIDATOR_VERSION = "day8_cpu_precision_fidelity_v1"
REGIONS = ("global", "source_hole", "destination")
TOLERANCE = .01
RATIO_FLOOR = 1e-12
FEATURE_SHAPE = (16, 24, 24, 1024)


def precision_metrics(reference_source, reference_target, candidate_source,
                      candidate_target, regions, tolerance=TOLERANCE):
    """Pure NumPy six-check metric helper, independent of encoder execution.

    Arrays may use small synthetic grids for an analytical check. Leading
    dimensions must match the three nonempty boolean region masks, with the
    last dimension representing features. Reductions deliberately use float32
    just as the original scratch validation did; ratios use Python floats.
    """
    arrays = [np.asarray(array, dtype=np.float32) for array in
              (reference_source, reference_target, candidate_source, candidate_target)]
    shape = arrays[0].shape
    if len(shape) < 2 or any(array.shape != shape or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Reference/candidate features must have equal finite shapes")
    if set(regions) != set(REGIONS) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Expected global/source_hole/destination regions and positive tolerance")
    source, target, candidate_source, candidate_target = arrays
    metrics, passed = {}, 0
    for name in REGIONS:
        mask = np.asarray(regions[name])
        if mask.dtype != np.bool_ or mask.shape != shape[:-1] or not mask.any():
            raise ValueError("Precision region must be nonempty and boolean: " + name)
        edit = float(np.square(source[mask] - target[mask]).mean())
        source_error = float(np.square(candidate_source[mask] - source[mask]).mean())
        target_error = float(np.square(candidate_target[mask] - target[mask]).mean())
        source_ratio = source_error / max(edit, RATIO_FLOOR)
        target_ratio = target_error / max(edit, RATIO_FLOOR)
        passed += int(source_ratio < tolerance) + int(target_ratio < tolerance)
        metrics[name] = {"source_precision_mse": source_error,
                         "target_precision_mse": target_error,
                         "true_edit_mse": edit,
                         "source_ratio": source_ratio, "target_ratio": target_ratio,
                         "tokens": int(mask.sum()),
                         "denominator_floored": bool(edit < RATIO_FLOOR)}
    return {"metrics": metrics, "passed_checks": passed, "total_checks": 6,
            "passes_1pct_signal_tolerance": passed == 6,
            "tolerance": float(tolerance), "ratio_floor": RATIO_FLOOR}


def _reference(reference_path, pair, spec):
    reference_path = Path(reference_path).resolve()
    if reference_path.stem != spec["name"]:
        raise ValueError("Validation is restricted to train_11000_dx+32.npz")
    reference_hash = extract.sha256(reference_path)
    with np.load(reference_path, allow_pickle=False) as archive:
        required = ("source", "target", "source_frac", "target_frac", "distractor_frac",
                    "rgb_source", "rgb_target")
        if any(key not in archive for key in required):
            raise ValueError("Reference lacks genuine paired feature/supervision arrays")
        arrays = {key: archive[key].copy() for key in required}
    for key in ("source", "target"):
        if arrays[key].shape != FEATURE_SHAPE or not np.isfinite(arrays[key]).all():
            raise ValueError("Invalid reference feature grid: " + key)
    for side in ("source", "target", "distractor"):
        expected = data.patch_fractions(pair["masks_" + side])
        if arrays[side + "_frac"].shape != expected.shape or not np.allclose(
                arrays[side + "_frac"], expected, atol=1e-6, rtol=0):
            raise ValueError("Reference mask does not match the fixed training fixture: " + side)
    rgb_hashes = {}
    for side in ("source", "target"):
        expected = extract.patch_rgb(pair["frames_" + side])
        if arrays["rgb_" + side].shape != expected.shape or not np.allclose(
                arrays["rgb_" + side], expected, atol=1e-6, rtol=0):
            raise ValueError("Reference RGB supervision differs from the fixed fixture: " + side)
        rgb_hashes[side + "_rgb_sha256"] = hashlib.sha256(pair["frames_" + side].tobytes()).hexdigest()
    provenance = {"path": str(reference_path), "sha256": reference_hash,
                  "bytes": reference_path.stat().st_size,
                  "feature_dtypes": {side: str(arrays[side].dtype) for side in ("source", "target")},
                  "verified_training_fixture": spec, **rgb_hashes,
                  "manifest": None, "record_seconds": None,
                  "record_timing_scope": "rendering, pair encoding, compression and cache write; excludes encoder load"}
    candidates = [reference_path.parent / "manifest.json", reference_path.parent.parent / "manifest.json"]
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        matching = [row for row in manifest["clips"] if row["spec"]["name"] == spec["name"]]
        if len(matching) != 1:
            raise ValueError("Reference manifest needs exactly one matching training scene")
        record = matching[0]
        if record["spec"] != spec or record["sha256"] != reference_hash:
            raise ValueError("Reference manifest scene/checksum mismatch")
        if manifest["model"] != extract.MODEL or manifest["upstream"] != extract.UPSTREAM:
            raise ValueError("Reference backbone/source revision differs")
        for key, expected in rgb_hashes.items():
            if record.get(key) != expected:
                raise ValueError("Reference renderer provenance differs: " + key)
        if manifest.get("source_hashes", {}).get("data.py") != extract.sha256(Path(data.__file__)):
            raise ValueError("Reference renderer source differs from the current fixed fixture")
        weights = [row for row in manifest.get("weights", []) if row.get("name") == extract.WEIGHT_NAME]
        if weights and (len(weights) != 1 or weights[0].get("sha256") != extract.WEIGHT_SHA256):
            raise ValueError("Reference checkpoint provenance differs")
        provenance.update({"manifest": {"path": str(manifest_path), "sha256": extract.sha256(manifest_path),
                                        "source_hashes": manifest["source_hashes"],
                                        "runtime": manifest.get("runtime"),
                                        "inference_precision": manifest.get("inference_precision"),
                                        "cache_precision": manifest.get("cache_precision"),
                                        "encoder_context": manifest.get("encoder_context"),
                                        "weights": manifest.get("weights"), "record": record},
                           "record_seconds": record.get("seconds")})
    return arrays, provenance


def run(reference, upstream, outdir, threads=8):
    """Run only the predeclared training fixture and persist both errors and caches."""
    import torch
    if int(threads) != threads or threads < 1:
        raise ValueError("--threads must be a positive integer")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "precision_validation.json"
    if report_path.exists():
        raise FileExistsError("Use a fresh output directory to preserve prior validation evidence")
    spec = data.scene_specs("train")[0]
    if spec["seed"] != 11000 or spec["name"] != "train_11000_dx+32" or spec["dx"] != 32:
        raise ValueError("The predeclared training fixture changed")
    pair = data.generate_pair(spec)
    arrays, reference_provenance = _reference(reference, pair, spec)
    source, target = arrays["source"].astype(np.float32), arrays["target"].astype(np.float32)
    prepared = operators.prepare_inputs(source, arrays["source_frac"], arrays["target_frac"],
                                        arrays["distractor_frac"], spec["dx"] // data.PATCH)
    regions = {"global": np.ones(source.shape[:-1], dtype=bool),
               "source_hole": prepared["source_hole"] > 0,
               "destination": prepared["destination"] > 0}
    del prepared
    torch.set_num_threads(int(threads))
    report = {"version": VALIDATOR_VERSION, "scope": "runtime extraction fidelity; not an editing result",
              "scene": spec, "reference": reference_provenance,
              "local_torch": str(torch.__version__), "local_python": platform.python_version(),
              "local_numpy": str(np.__version__), "device": "cpu", "threads": int(threads),
              "torch_hub_cache": str(torch.hub.get_dir()),
              "checkpoint_sha256": extract.WEIGHT_SHA256, "upstream": extract.UPSTREAM,
              "source_hashes": {**extract.sources(), "operators.py": extract.sha256(Path(operators.__file__)),
                                "validate_precision.py": extract.sha256(Path(__file__))},
              "test_data_opened": False, "precisions": {}, "chosen_precision": None,
              "rule": "All six source/target errors, normalized by reference edit signal, strictly below .01",
              "local_timing_scope": "source and target pair encoding only; excludes load, rendering and cache write",
              "timing_comparison_limit": "Reference record and local encoding timers cover different work"}
    print("LOAD_START", torch.__version__, torch.get_num_threads(), flush=True)
    tick = time.perf_counter()
    encoder = extract.load_encoder(upstream, "cpu")
    report["encoder_load_seconds"] = time.perf_counter() - tick
    print("LOAD_DONE", report["encoder_load_seconds"], flush=True)
    for precision in ("bfloat16", "float32"):
        tick = time.perf_counter()
        candidate_source = extract.encode(encoder, pair["frames_source"], "cpu", precision)
        candidate_target = extract.encode(encoder, pair["frames_target"], "cpu", precision)
        seconds = time.perf_counter() - tick
        assessed = precision_metrics(source, target, candidate_source, candidate_target, regions)
        output_path = outdir / ("local_reference_" + precision + ".npz")
        np.savez_compressed(output_path, source=candidate_source, target=candidate_target)
        report["precisions"][precision] = {"seconds": seconds, **assessed,
                                           "cache": {"path": output_path.name, "sha256": extract.sha256(output_path),
                                                     "bytes": output_path.stat().st_size}}
        if assessed["passes_1pct_signal_tolerance"]:
            report["chosen_precision"] = precision
        extract.write_json(report_path, report)
        print("PRECISION_DONE", precision, seconds, json.dumps(assessed["metrics"]), flush=True)
        if report["chosen_precision"] is not None:
            break
    if report["chosen_precision"] is None:
        raise RuntimeError("No precision agrees with reference under the six-check 1% signal tolerance")
    print("PRECISION_VALIDATED", report["chosen_precision"], flush=True)
    return report


def self_check():
    """Analytical helper check only; never loads weights or experimental data."""
    source = np.zeros((2, 3, 4, 5), np.float32)
    target = np.ones_like(source)
    global_mask = np.ones(source.shape[:-1], bool)
    hole = np.zeros_like(global_mask)
    hole[:, :, 0] = True
    destination = np.zeros_like(global_mask)
    destination[:, :, -1] = True
    regions = {"global": global_mask, "source_hole": hole, "destination": destination}
    exact = precision_metrics(source, target, source, target, regions)
    small = precision_metrics(source, target, source + .05, target + .05, regions)
    large = precision_metrics(source, target, source + .2, target + .2, regions)
    assert exact["passed_checks"] == small["passed_checks"] == 6
    assert large["passed_checks"] == 0
    for item in small["metrics"].values():
        np.testing.assert_allclose([item["source_ratio"], item["target_ratio"]], [.0025, .0025], rtol=1e-5)
    return {"status": "passed", "checks": ["six exact-agreement checks", "known .0025 ratios pass",
                                             "known .04 ratios fail"], "encoder_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run(args.reference, args.upstream, args.outdir, args.threads)
