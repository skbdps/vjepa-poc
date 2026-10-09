"""Frozen development selection and fresh-test evaluation of source-hole repair.

Only genuine Day8 development caches select a temporal repair and blend. Test
requires a separately published freeze; this program never fits a readout.
Destination tokens remain exactly those of Day8's successful naive transport.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DAY8 = HERE.parent / "day8"
# Day8's scorer uses absolute imports. Resolve them against its own directory,
# then import the scorer by a unique name to avoid this module's same basename.
sys.path.insert(0, str(DAY8))
import data as day8_data
import operators as day8_operators
import probe as day8_probe
spec = importlib.util.spec_from_file_location("day8_frozen_scorer", DAY8 / "evaluate.py")
day8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day8)
sys.path.insert(0, str(HERE))
import temporal

VERSION = "day9_source_hole_repair_v1"
ALPHAS = (0., .25, .5, .75, 1.)
KINDS = ("temporal_mean", "aligned_temporal_mean")
ARMS = ("noop", "naive", *KINDS, "dev_selected", "genuine_target")
PROVENANCE = ("model", "upstream", "encoder_context", "extraction_device",
              "inference_precision", "cache_precision", "weights")
SOURCE_PATHS = ("experiments/day9/evaluate.py", "experiments/day9/temporal.py",
                "experiments/day8/data.py", "experiments/day8/operators.py",
                "experiments/day8/evaluate.py", "experiments/day8/probe.py")
sha256, write_json, write_csv = day8.sha256, day8.write_json, day8.write_csv


def current_hashes():
    return {path: sha256(REPO / path) for path in SOURCE_PATHS}


def check_hashes(hashes):
    for path, expected in hashes.items():
        full = (REPO / path).resolve()
        if not full.is_relative_to(REPO) or sha256(full) != expected:
            raise ValueError("Frozen source changed: " + path)


def feature_index(features):
    root = Path(features)
    if root.name == "cache" and not (root / "manifest.json").exists():
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text())
    indexed = {record["spec"]["name"]: record for record in manifest["clips"]}
    if len(indexed) != len(manifest["clips"]):
        raise ValueError("Duplicate scenes in feature manifest")
    return root, manifest, indexed


def blend(naive, repaired, hole, alpha):
    result = naive.copy()
    # Preserve exact endpoints, including the no-change alpha=0 case.
    if alpha == 1:
        result[hole] = repaired[hole]
    elif alpha != 0:
        result[hole] += np.float32(alpha) * (repaired[hole] - naive[hole])
    return result


def latent_error(edited, target):
    return np.mean((edited - target) ** 2, axis=-1, dtype=np.float32)


def mean_region(values, mask):
    return float(np.mean(values[mask], dtype=np.float64)) if mask.any() else None


def source_hole(cached, dx):
    regions, support = day8.fixed_regions(cached["source_frac"], cached["target_frac"],
                                         cached["distractor_frac"], dx)
    if not regions["source_hole"].any():
        raise ValueError("Empty primary source-hole region")
    return regions, support


def select_development(features, probe_path, out):
    """Reads exactly eight old development caches; no test NPZ is opened."""
    root, manifest, indexed = feature_index(features)
    hashes = current_hashes()
    for filename, digest in manifest["source_hashes"].items():
        if sha256(DAY8 / filename) != digest:
            raise ValueError("Day8 development extractor source changed: " + filename)
    rows, bindings = [], []
    for scene in day8_data.scene_specs("dev"):
        record = indexed.get(scene["name"])
        if record is None or record["spec"] != scene:
            raise ValueError("Incomplete or changed Day8 development split")
        cached = day8.load_cache(root, record)
        source, target = (cached[key].astype(np.float32) for key in ("source", "target"))
        dx = scene["dx"] // day8_data.PATCH
        regions, _ = source_hole(cached, dx)
        hole = regions["source_hole"]
        result = temporal.build_repairs(source, cached["source_frac"], cached["distractor_frac"], dx)
        variants = result["variants"]
        denominator = mean_region(latent_error(source, target), hole)
        for kind in KINDS:
            for alpha in ALPHAS:
                edited = blend(variants["naive"], variants[kind], hole, alpha)
                if not np.array_equal(edited[~hole], variants["naive"][~hole]):
                    raise AssertionError("Repair changed a non-hole token")
                mse = mean_region(latent_error(edited, target), hole)
                rows.append({"scene": scene["name"], "kind": kind, "alpha": alpha,
                             "source_hole_tokens": int(hole.sum()), "source_hole_mse": mse,
                             "source_hole_noop_mse": denominator,
                             "source_hole_degenerate": int(denominator < day8.RATIO_FLOOR),
                             "source_hole_ratio": mse / max(denominator, day8.RATIO_FLOOR)})
        bindings.append(record)
        print("DEV_SCENE_COMPLETE", scene["name"], flush=True)
        del cached, source, target, result, variants, edited
        gc.collect()
    candidates = [{"kind": kind, "alpha": alpha,
                   "mean_source_hole_ratio": float(np.mean([r["source_hole_ratio"] for r in rows
                                                            if r["kind"] == kind and r["alpha"] == alpha]))}
                  for kind in KINDS for alpha in ALPHAS]
    minimum = min(row["mean_source_hole_ratio"] for row in candidates)
    tied = [row for row in candidates if row["mean_source_hole_ratio"] <= minimum + 1e-12]
    chosen = min(tied, key=lambda r: (r["alpha"], KINDS.index(r["kind"])))
    selection = {"version": VERSION, "test_accessed": False, "selection_split": "Day8 dev only",
                 "chosen": chosen, "candidate_means": candidates, "per_scene_candidates": rows,
                 "selection_rule": "Minimum mean per-scene source-hole MSE/no-op MSE; global minimum plus absolute 1e-12 tie tolerance; smaller alpha, then temporal_mean",
                 "alpha_grid": list(ALPHAS), "kinds": list(KINDS),
                 "probe_sha256": sha256(probe_path), "source_hashes": hashes,
                 "dev_manifest_sha256": sha256(root / "manifest.json"), "dev_cache_bindings": bindings,
                 "encoder_provenance": {key: manifest[key] for key in PROVENANCE},
                 "operator_precision": "float32", "bootstrap": {"draws": 5000, "seed": 1909}}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dev_selection.json", selection)
    write_csv(out / "dev_candidates.csv", rows)
    print("DEV_SELECTION_COMPLETE", json.dumps(chosen), flush=True)
    return selection


def validate_test(features, probe_path, freeze_path):
    root, manifest, indexed = feature_index(features)
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("test_accessed") is not False:
        raise ValueError("Freeze must explicitly predate test access")
    if manifest.get("test_freeze_sha256") != sha256(freeze_path):
        raise ValueError("Test extraction is not bound to the provided freeze")
    check_hashes(freeze["source_hashes"])
    if any(freeze["source_hashes"].get(path) != digest for path, digest in current_hashes().items()):
        raise ValueError("Freeze omits or changes an evaluator/operator/readout dependency")
    selection_path = Path(freeze["dev_selection"]["path"])
    if not selection_path.is_absolute():
        selection_path = freeze_path.parent / selection_path
    if sha256(selection_path) != freeze["dev_selection"]["sha256"]:
        raise ValueError("Development selection record changed")
    selection = json.loads(selection_path.read_text())
    if selection.get("test_accessed") is not False or selection["source_hashes"] != current_hashes():
        raise ValueError("Development selection was not frozen against this exact implementation")
    if sha256(probe_path) != selection["probe_sha256"] or sha256(probe_path) != freeze["probe_sha256"]:
        raise ValueError("Frozen readout checkpoint changed")
    for field in PROVENANCE:
        if manifest[field] != selection["encoder_provenance"][field]:
            raise ValueError("Development/test encoder provenance differs: " + field)
    expected = []
    for old in day8_data.scene_specs("test"):
        scene = dict(old, seed=old["seed"] + 1000)
        scene["name"] = f"test_{scene['seed']}_dx{scene['dx']:+d}"
        expected.append(scene)
    if freeze["test_specs"] != expected:
        raise ValueError("Frozen test is not the predeclared fresh sixteen-scene split")
    if len(manifest["clips"]) != 16:
        raise ValueError("Fresh test manifest must contain exactly sixteen scenes")
    for scene in expected:
        if scene["name"] not in indexed or indexed[scene["name"]]["spec"] != scene:
            raise ValueError("Fresh test scene missing or changed: " + scene["name"])
    for record in manifest["clips"]:
        if sha256(root / record["path"]) != record["sha256"]:
            raise ValueError("Fresh test feature cache changed: " + record["path"])
    chosen = selection["chosen"]
    if chosen["kind"] not in KINDS or chosen["alpha"] not in ALPHAS:
        raise ValueError("Invalid frozen blend choice")
    return root, manifest, indexed, freeze, selection


def bootstrap(values):
    values = np.asarray([x for x in values if x is not None], np.float64)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n_scenes": 0}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scene metric")
    samples = np.random.default_rng(1909).integers(0, len(values), (5000, len(values)))
    return {"mean": float(values.mean()), "ci95": np.quantile(values[samples].mean(1), [.025, .975]).tolist(),
            "n_scenes": len(values)}


def extra_metrics(prediction, cached, error, naive_delta, regions):
    hole, destination = regions["source_hole"], regions["destination"]
    rgb_error = np.mean((prediction["rgb"].astype(np.float64) - cached["rgb_target"]) ** 2, axis=-1)
    return {"hole_rgb_mse": mean_region(rgb_error, hole),
            "hole_ghost_mean_occupancy": mean_region(prediction["occupancy"], hole),
            "hole_ghost_fraction_above_half": mean_region(prediction["occupancy"] >= .5, hole),
            "outside_hole_vs_naive_max_abs_delta": float(naive_delta[~hole].max()),
            "outside_hole_vs_naive_changed_tokens": int(np.sum(naive_delta[~hole] != 0)),
            "destination_vs_naive_max_abs_delta": float(naive_delta[destination].max()),
            "destination_vs_naive_changed_tokens": int(np.sum(naive_delta[destination] != 0))}


SUMMARY_METRICS = ("source_hole_ratio", "source_hole_mse", "source_hole_noop_mse", "destination_ratio",
                   "destination_mse", "balanced_region_ratio", "hole_rgb_mse", "hole_ghost_mean_occupancy", "hole_ghost_fraction_above_half",
                   "selected_centroid_error_px", "selected_iou", "selected_rgb_core_mse",
                   "selected_temporal_velocity_error_px", "selected_missing_tubelets", "appearance_identity_accuracy",
                   "appearance_identity_eligible", "appearance_identity_skipped", "appearance_identity_correct",
                   "distractor_centroid_error_px", "distractor_rgb_change_mse", "distractor_occupancy_change_mse",
                   "distractor_iou", "distractor_missing_tubelets", "distractor_temporal_velocity_error_px",
                   "outside_hole_vs_naive_max_abs_delta", "outside_hole_vs_naive_changed_tokens",
                   "destination_vs_naive_max_abs_delta", "destination_vs_naive_changed_tokens")


def summarize(rows):
    methods, differences, checks = {}, {}, {}
    by_scene_arm = {(row["scene"], row["arm"]): row for row in rows}
    for group in ("all", "seen_magnitude", "heldout_magnitude"):
        included = [row for row in rows if group == "all" or row["shift_regime"] == group]
        methods[group], differences[group] = {}, {}
        for arm in ARMS:
            arm_rows = [row for row in included if row["arm"] == arm]
            methods[group][arm] = {key: bootstrap([r[key] for r in arm_rows]) for key in SUMMARY_METRICS}
            if arm not in ("naive", "genuine_target"):
                differences[group][arm + "_minus_naive"] = {
                    key: bootstrap([r[key] - by_scene_arm[(r["scene"], "naive")][key] for r in arm_rows
                                    if r[key] is not None and by_scene_arm[(r["scene"], "naive")][key] is not None])
                    for key in SUMMARY_METRICS}
    for arm in (*KINDS, "dev_selected"):
        diff = differences["all"][arm + "_minus_naive"]
        metrics = methods["all"][arm]
        checks[arm] = {
            "hole_latent_improves_paired_ci": diff["source_hole_ratio"]["ci95"][1] < 0,
            "mean_hole_rgb_not_worse": diff["hole_rgb_mse"]["mean"] <= 0,
            "mean_hole_ghost_not_worse": diff["hole_ghost_mean_occupancy"]["mean"] <= 0,
            "outside_hole_exactly_preserved": metrics["outside_hole_vs_naive_max_abs_delta"]["mean"] == 0,
            "destination_exactly_preserved": metrics["destination_vs_naive_max_abs_delta"]["mean"] == 0}
        checks[arm]["combined_exploratory_pass"] = all(checks[arm].values())
    genuine = methods["all"]["genuine_target"]
    readout_valid = (genuine["selected_centroid_error_px"]["mean"] < 16
                     and genuine["appearance_identity_accuracy"]["mean"] is not None
                     and genuine["appearance_identity_accuracy"]["mean"] >= .9)
    return {"version": VERSION, "n_independent_scenes": len({r["scene"] for r in rows}),
            "methods": methods, "paired_differences": differences, "decision_checks": checks,
            "genuine_readout_gate": {"pass": readout_valid, "centroid_threshold_px": 16,
                                     "identity_threshold": .9, "centroid": genuine["selected_centroid_error_px"],
                                     "identity_accuracy": genuine["appearance_identity_accuracy"],
                                     "eligible_tubelets": int(sum(r["appearance_identity_eligible"] for r in rows if r["arm"] == "genuine_target")),
                                     "skipped_tubelets": int(sum(r["appearance_identity_skipped"] for r in rows if r["arm"] == "genuine_target"))},
            "degenerate_scene_arm_counts": {name: int(sum(r[name + "_degenerate"] for r in rows)) for name in day8.REGIONS},
            "primary": "Dev-selected blend minus naive, per-scene source-hole MSE/no-op MSE; all sixteen scenes including fallback",
            "bootstrap": {"draws": 5000, "seed": 1909, "unit": "independent scene", "interval": "percentile 95%; exploratory"},
            "rgb_ghost_aggregation": "Mean over all source-hole tokens within each scene, then equal scene mean",
            "interpretation_limits": ["Oracle full-trajectory masks and a static-camera synthetic scene",
                                      "Temporal donor availability is measured and fallback scenes remain in primary analysis",
                                      "Pointwise occupancy/RGB readout is not a video decoder or independent proof of identity",
                                      "Exact destination/outside preservation is enforced by construction",
                                      "A source-only temporal repair is not a learned JEPA predictor or causal single-edit propagation"]}


def coverage_metrics(result, tubelet=None):
    masks, donors = result["masks"], result["donors"]
    hole = masks["hole"] if tubelet is None else masks["hole"][tubelet]
    values = {}
    for name in ("donor_count", "aligned_donor_count", "same_block_donor_count", "cross_block_donor_count",
                 "aligned_same_block_donor_count", "aligned_cross_block_donor_count"):
        array = masks[name] if tubelet is None else masks[name][tubelet]
        counts = array[hole]
        values[name + "_mean"] = float(counts.mean()) if len(counts) else None
        if name in ("donor_count", "aligned_donor_count"):
            prefix = "temporal" if name == "donor_count" else "aligned"
            values[prefix + "_coverage"] = float((counts > 0).mean()) if len(counts) else None
            values[prefix + "_fallback_tokens"] = int(np.sum(counts == 0))
            for count in range(17):
                values[prefix + "_tokens_with_" + str(count) + "_donors"] = int(np.sum(counts == count))
    selected = slice(None) if tubelet is None else donors["hole_indices"][:, 0] == tubelet
    modes = donors["alignment_mode"][selected]
    for label, mode in (("local", 1), ("global", 2), ("rejected", 3)):
        values["aligned_" + label + "_donor_pairs"] = int(np.sum(modes == mode))
    return values


def run_test(features, probe_path, out, freeze_path, device="cpu"):
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    root, manifest, indexed, frozen, selection = validate_test(features, probe_path, freeze_path)
    readout, _ = day8_probe.load_probe(probe_path, device)
    out = Path(out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows, step_rows, strata_rows, artifacts = [], [], [], []
    for scene in frozen["test_specs"]:
        cached = day8.load_cache(root, indexed[scene["name"]])
        source, target = (cached[key].astype(np.float32) for key in ("source", "target"))
        dx = scene["dx"] // day8_data.PATCH
        regions, _ = source_hole(cached, dx)
        hole = regions["source_hole"]
        result = temporal.build_repairs(source, cached["source_frac"], cached["distractor_frac"], dx)
        variants = result["variants"]
        variants["dev_selected"] = blend(variants["naive"], variants[selection["chosen"]["kind"]], hole, selection["chosen"]["alpha"])
        variants.update(noop=source, genuine_target=target)
        if not np.array_equal(result["masks"]["hole"], hole):
            raise AssertionError("Operator and scorer source-hole regions disagree")
        counts = result["masks"]["donor_count"]
        shifted_source = day8_operators.shift_horizontal(source, dx)
        destination = regions["destination"]
        if not np.array_equal(variants["naive"][destination], shifted_source[destination]):
            raise AssertionError("Destination does not match original-snapshot token transport")
        del shifted_source
        strata = {}
        for policy, count_key in (("temporal", "donor_count"), ("aligned", "aligned_donor_count")):
            available = result["masks"][count_key]
            for label, mask in {"all_hole": hole, "zero_donors_fallback": hole & (available == 0),
                                "covered_any_donor": hole & (available > 0),
                                "one_to_three_donors": hole & (available > 0) & (available <= 3),
                                "four_or_more_donors": hole & (available >= 4)}.items():
                strata[(policy, label)] = mask
        scene_coverage = coverage_metrics(result)
        step_coverage = [coverage_metrics(result, t) for t in range(day8_data.N_STEPS)]
        noop_error = latent_error(source, target)
        source_prediction = day8_probe.predict_probe(readout, source, device)
        saved = {key: cached[key] for key in ("source_frac", "target_frac", "distractor_frac", "rgb_source", "rgb_target")}
        saved.update(noop_latent_error=noop_error, region_names=np.array(day8.REGIONS),
                     region_masks=np.stack([regions[key] for key in day8.REGIONS]),
                     donor_count=counts, dx_pixels=np.int32(scene["dx"]),
                     diagnostics_json=np.array(json.dumps(result["diagnostics"], allow_nan=False)))
        for group in ("masks", "donors"):
            for key, array in result[group].items():
                array = np.asarray(array)
                if array.dtype == object:
                    raise ValueError("Raw donor/mask evidence must be non-object arrays")
                saved[group + "_" + key] = array
        collected = defaultdict(list)
        for arm in ARMS:
            edited = variants[arm]
            if edited.dtype != np.float32 or not np.isfinite(edited).all():
                raise ValueError("Operator output must be finite float32")
            delta = np.max(np.abs(edited - variants["naive"]), axis=-1)
            if arm not in ("noop", "genuine_target") and np.any(delta[~hole] != 0):
                raise AssertionError("Source-hole-only repair changed destination or context")
            prediction = source_prediction if arm == "noop" else day8_probe.predict_probe(readout, edited, device)
            error = latent_error(edited, target)
            semantics = day8.semantic_metrics(prediction, source_prediction, cached, regions)
            identity = {"scene": scene["name"], "scene_seed": scene["seed"], "dx_pixels": scene["dx"],
                        "shift_regime": scene["shift_regime"], "arm": arm}
            metrics = day8.region_metrics(error, noop_error, regions)
            metrics["balanced_region_ratio"] = metrics.pop("primary_ratio")
            rows.append({**identity, **metrics, **day8.aggregate_semantic(semantics),
                         **extra_metrics(prediction, cached, error, delta, regions), **scene_coverage})
            for t in range(day8_data.N_STEPS):
                t_cached = {key: value[t] for key, value in cached.items()}
                t_prediction = {key: value[t] for key, value in prediction.items()}
                t_regions = {key: value[t] for key, value in regions.items()}
                t_metrics = day8.region_metrics(error[t], noop_error[t], t_regions)
                t_metrics["balanced_region_ratio"] = t_metrics.pop("primary_ratio")
                step_rows.append({**identity, "tubelet": t, **t_metrics, **semantics[t],
                                  **extra_metrics(t_prediction, t_cached, error[t], delta[t], t_regions), **step_coverage[t]})
            rgb_error = np.mean((prediction["rgb"].astype(np.float64) - cached["rgb_target"]) ** 2, axis=-1)
            for (policy, label), mask in strata.items():
                mse, denom = mean_region(error, mask), mean_region(noop_error, mask)
                strata_rows.append({**identity, "donor_policy": policy, "coverage_stratum": label, "tokens": int(mask.sum()),
                                    "latent_mse": mse, "noop_mse": denom,
                                    "latent_ratio": mse / max(denom, day8.RATIO_FLOOR) if mse is not None else None,
                                    "hole_rgb_mse": mean_region(rgb_error, mask),
                                    "hole_ghost_mean_occupancy": mean_region(prediction["occupancy"], mask)})
            for key, value in (("occupancy", prediction["occupancy"]), ("rgb", prediction["rgb"]),
                               ("latent_error", error), ("max_abs_delta_vs_naive", delta)):
                collected[key].append(value)
            collected["edited_latents_float32_sha256"].append(hashlib.sha256(np.ascontiguousarray(edited).tobytes()).hexdigest())
            print("TEST_METHOD", scene["name"], arm, "hole_ratio", rows[-1]["source_hole_ratio"], flush=True)
        saved["method_names"] = np.asarray(ARMS)
        for key, values in collected.items():
            saved[key] = np.stack(values)
        path = out / "predictions" / (scene["name"] + ".npz")
        np.savez_compressed(path, **saved)
        artifacts.append({"scene": scene["name"], "path": str(path.relative_to(out)), "sha256": sha256(path),
                          "feature_cache_sha256": indexed[scene["name"]]["sha256"], "bytes": path.stat().st_size})
        del cached, source, target, variants, result, saved, collected, edited, prediction, source_prediction
        gc.collect()
    write_csv(out / "per_clip.csv", rows)
    write_csv(out / "per_tubelet.csv", step_rows)
    write_csv(out / "coverage_strata.csv", strata_rows)
    summary = summarize(rows)
    summary["chosen"] = selection["chosen"]
    summary["qualified_adoption"] = (selection["chosen"]["alpha"] > 0 and summary["genuine_readout_gate"]["pass"]
                                      and summary["decision_checks"]["dev_selected"]["combined_exploratory_pass"])
    summary["binding"] = {"freeze_sha256": sha256(freeze_path), "dev_selection_sha256": frozen["dev_selection"]["sha256"],
                          "probe_sha256": sha256(probe_path), "feature_manifest_sha256": sha256(root / "manifest.json"),
                          "source_hashes": current_hashes(), "device": device, "operator_readout_precision": "float32",
                          "numpy": str(np.__version__), "torch": str(torch.__version__)}
    write_json(out / "summary.json", summary)
    write_json(out / "evaluation_manifest.json", {"version": VERSION, "binding": summary["binding"],
               "input_feature_manifest": manifest, "freeze": frozen, "artifacts": artifacts,
               "row_counts": {"per_clip": len(rows), "per_tubelet": len(step_rows), "coverage_strata": len(strata_rows)},
               "audit_scope": "NPZ includes all coarse readouts, full latent-error maps, exact maximum deltas, donor inputs and candidates; full edited tensors can be reconstructed from frozen source caches and saved source-only operators"})
    print("TEST_COMPLETE", len(frozen["test_specs"]), "scenes", len(rows), "method rows", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dev", "test"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.mode == "dev":
        select_development(args.features, args.probe, args.out)
    elif args.freeze:
        run_test(args.features, args.probe, args.out, args.freeze, args.device)
    else:
        parser.error("--freeze is mandatory in test mode")


if __name__ == "__main__":
    main()
