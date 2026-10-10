"""Predeclared all-eight fresh engineering assessment; never fits or selects.

The fixed CNN primary is provenance_residual_copy_repair. Every scene must
meet exactly the development engineering thresholds: latent hole/destination
ratios <=.5, calibrated true-target centroid <=2px, detected primary <=4px,
GPU hole/destination excess ratios <=.5 with denominator >1/255^2, protected
error <= max(source,target)+max(.25*max(source,target),1/255^2), correct position
better than wrong direction, and both wrong/shuffled balanced excess scores
at least .10 worse. Missing/indeterminate evidence cannot produce a pass.

Only 8/8 individually passing scenes with valid evidence bindings qualify as
'8/8 engineering pilot pass'. Means never replace any individual criterion.
No superiority over the deterministic no-F comparator is required. Fixed
paired arm and CNN/linear latent comparisons are descriptive, not selection.

assessment_config() is a pure pretest-freeze value. validate_pretest() checks
that exact configuration, the publication receipt, source bytes, development
gate and trained freeze BEFORE this runner opens any fresh metrics/manifests.
Imports load only standard-library code and the pure development qualifier.
No model, image, feature cache, latent array or fresh scene is opened here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("day11_fresh_qualification_math", HERE / "qualify_development.py")
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)

VERSION = "day11_fresh_engineering_assessment_v1"
CONFIG_VERSION = "day11_fresh_assessment_configuration_v1"
EVALUATION_VERSION = "day11_fresh_bridge_evaluation_v1"
RENDER_VERSION = "day11_fresh_vace_render_v1"
GPU_ARMS = ("true_source", "true_target", "absolute_copy_repair", "residual_copy_repair",
            "provenance_copy_copy_repair", "provenance_local_copy_repair",
            q.PRIMARY, q.WRONG, q.SHUFFLED)
REQUIRED_SOURCES = tuple("experiments/day11/" + name for name in
                        ("assess_fresh.py", "qualify_development.py", "prepare_fresh.py",
                         "evaluate_fresh.py", "render_fresh.py", "render.py", "transport.py",
                         "evaluate.py", "vace_bridge.py", "model.py", "data.py", "train.py"))


def test_specs():
    return [{"name": f"test_{14000+i}_dx{dx:+d}", "seed": 14000+i, "dx": dx,
             "split": "test", "frame_index": 15, "views": ["source", "genuine_shifted_target"]}
            for i, dx in enumerate((32, -32, 48, -48, 64, -64, 80, -80))]


def assessment_config():
    """Pure JSON-compatible object; freeze this before any fresh extraction."""
    return {"version": CONFIG_VERSION, "primary_translator": "cnn", "primary_arm": q.PRIMARY,
            "test_specs": test_specs(), "required_scene_count": 8, "required_passing_scene_count": 8,
            "latent_arms": list(q.ARMS), "latent_translators": ["cnn", "linear"],
            "GPU_arms": list(GPU_ARMS), "latent_region_ratio_max": q.LATENT_LIMIT,
            "GPU_region_excess_max": q.RGB_EXCESS_LIMIT, "GPU_centroid_max_px": q.CENTROID_LIMIT,
            "oracle_centroid_max_px": q.ORACLE_CENTROID_LIMIT,
            "GPU_negative_balanced_excess_margin_min": q.NEGATIVE_MARGIN,
            "RGB_denominator_min_exclusive": q.RGB_EPSILON,
            "protected_relative_allowance": q.PRESERVATION_FRACTION,
            "protected_absolute_allowance": q.RGB_EPSILON, "parity_tolerance": q.PARITY_TOLERANCE,
            "regions": {"edit": list(q.EDIT_REGIONS), "protected": list(q.PROTECTED_REGIONS)},
            "excess_formula": "(E_arm_to_target-E_true_target_to_target)/(E_true_source_to_target-E_true_target_to_target)",
            "protected_formula": "E_primary<=max(E_true_source,E_true_target)+max(.25*max(E_true_source,E_true_target),1/255^2)",
            "negative_position": "correct finite error strictly less than wrong-direction error; missing control detection is positional failure",
            "negative_regional": "mean(hole,destination) excess per scene; wrong and shuffled each >= correct+.10",
            "missing_evidence": "fail or indeterminate; never excluded or counted passing",
            "aggregation": "all eight individual scenes must pass every criterion; no average substitutes",
            "learned_vs_no_F_superiority_required": False, "comparator_arms": list(q.COMPARATORS),
            "adaptation_or_selection": False, "scope": "eight procedural still-image pairs; engineering pilot only"}


def _bound_file(anchor, reference):
    path = Path(reference["path"])
    if not path.is_absolute():
        path = Path(anchor).parent / path
    if q.sha256(path) != reference["sha256"]:
        raise ValueError("Bound pretest evidence changed: " + str(path))
    return path


def validate_pretest(freeze, publication, checkpoint_freeze):
    """Only pre-existing pretest/training/development files are read here."""
    frozen, receipt, trained = map(q.read_json, (freeze, publication, checkpoint_freeze))
    if (frozen.get("test_accessed") is not False or frozen.get("test_specs") != test_specs()
            or frozen.get("assessment") != assessment_config()):
        raise ValueError("Fresh assessment differs from the published pretest configuration")
    source_hashes = frozen.get("source_hashes", {})
    if not set(REQUIRED_SOURCES) <= set(source_hashes):
        raise ValueError("Pretest freeze does not bind all assessment dependencies")
    for name, digest in source_hashes.items():
        path = (REPO / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(REPO.resolve()) or q.sha256(path) != digest:
            raise ValueError("Source changed or escaped repository after pretest publication: " + name)
    freeze_hash, receipt_hash, trained_hash = map(q.sha256, (freeze, publication, checkpoint_freeze))
    commit = receipt.get("commit", "")
    if (receipt.get("freeze_sha256") != freeze_hash or receipt.get("bytes_equal_to_GitHub") is not True
            or receipt.get("test_encoded_before_verification") is not False
            or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)):
        raise ValueError("Missing byte-verified publication before fresh access")
    if (_get(frozen, "trained_checkpoint_freeze", "sha256") != trained_hash
            or trained.get("version") != "day11_genuine_jepa_to_wan_training_v1"
            or trained.get("fresh_test_accessed") is not False or trained.get("test_accessed") is not False):
        raise ValueError("Trained checkpoint freeze differs from the pretest binding")
    models = trained.get("models", [])
    if ({row.get("arm") for row in models} != {"cnn", "linear"} or len(models) != 2
            or any(row.get("selected_epoch") != 150 for row in models)):
        raise ValueError("Both fixed epoch150 translator checkpoints must be frozen")
    for name, digest in trained.get("source_hashes", {}).items():
        if source_hashes.get(name) != digest:
            raise ValueError("Training/pretest source binding differs: " + name)
    gate = q.read_json(_bound_file(freeze, frozen["development_gate"]))
    if gate.get("pass") is not True or gate.get("checkpoint_freeze_sha256") != trained_hash:
        raise ValueError("No qualifying development gate for these trained checkpoints")
    evaluation = frozen.get("evaluation", {})
    if (evaluation.get("arms") != list(q.ARMS) or evaluation.get("translator_arms") != ["cnn", "linear"]
            or evaluation.get("precision") != "float32" or evaluation.get("shuffle_seed") != 1111
            or evaluation.get("local_radius") != 2):
        raise ValueError("Pretest evaluation arm/operator settings differ")
    rendering = frozen.get("rendering", {})
    if (rendering.get("arms") != list(GPU_ARMS) or rendering.get("test_specs") != test_specs()
            or rendering.get("translator") != "cnn" or rendering.get("threads") != 4
            or rendering.get("diffusers") != "0.35.1" or rendering.get("precision_fallback") is not False
            or any(rendering.get(key) != value for key, value in q.RENDER_CONFIGURATION.items())):
        raise ValueError("Pretest renderer differs from fixed engineering assessment settings")
    return frozen, trained, {"test_freeze": freeze_hash, "test_publication": receipt_hash,
                            "trained_checkpoint_freeze": trained_hash}


_get = q._get


def _add(checks, name, outcome, observed, required, indeterminate=False):
    passed = outcome is True
    checks.append({"id": name, "pass": passed,
                   "status": "pass" if passed else "indeterminate" if indeterminate else "fail",
                   "observed": observed, "required": required})


def _status(checks):
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    return "indeterminate" if any(check["status"] == "indeterminate" for check in checks) else "pass"


def assess(latent_rows, summary, evaluation, feature_manifest, target_manifest, rendering,
           frozen, trained, hashes):
    """Pure arithmetic/binding assessment over already parsed evidence objects."""
    checks, specs = [], test_specs()
    checkpoints = {row["arm"]: row["checkpoint_sha256"] for row in trained["models"]}
    bindings = rendering.get("bindings", {})
    pretest = bindings.get("pretest", {})
    expected_pretest = {"test_freeze_sha256": hashes["test_freeze"],
                        "test_publication_sha256": hashes["test_publication"],
                        "trained_checkpoint_freeze_sha256": hashes["trained_checkpoint_freeze"]}
    _add(checks, "binding.assessment", frozen.get("assessment") == assessment_config(),
         frozen.get("assessment"), assessment_config())
    _add(checks, "binding.evaluation", evaluation.get("version") == EVALUATION_VERSION and
         evaluation.get("complete") is True and evaluation.get("fresh_test_accessed") is True and
         evaluation.get("test_specs") == specs and evaluation.get("evaluation") == frozen.get("evaluation") and
         evaluation.get("source_hashes") == frozen.get("source_hashes") and
         evaluation.get("checkpoint_sha256_by_translator") == checkpoints and
         all(evaluation.get(k) == v for k, v in expected_pretest.items()) and
         evaluation.get("metrics_sha256") == hashes["evaluation_metrics"] and
         evaluation.get("summary_sha256") == hashes["evaluation_summary"] and
         evaluation.get("feature_manifest_sha256") == hashes["feature_manifest"] and
         evaluation.get("target_manifest_sha256") == hashes["target_manifest"] and
         set(evaluation.get("arms", {})) == set(q.ARMS) and evaluation.get("translator_arms") == ["cnn", "linear"] and
         [(x.get("scene"), x.get("seed"), x.get("shift_px")) for x in evaluation.get("scenes", [])] ==
         [(x["name"], x["seed"], x["dx"]) for x in specs],
         {"complete": evaluation.get("complete"), "version": evaluation.get("version"),
          "checkpoint_sha256_by_translator": evaluation.get("checkpoint_sha256_by_translator")},
         "complete eight-scene/two-translator/27-arm evaluation bound to all supplied hashes and pretest config")
    _add(checks, "binding.summary", summary.get("version") == EVALUATION_VERSION and
         summary.get("scene_count") == 8 and summary.get("fresh_test_accessed") is True and
         summary.get("test_freeze_sha256") == hashes["test_freeze"] and
         set(summary.get("arms", {})) == set(q.ARMS) and
         all(_get(summary, "methods_by_translator", translator, arm, "scene_count") == 8
             for translator in ("cnn", "linear") for arm in q.ARMS),
         {"scene_count": summary.get("scene_count"), "version": summary.get("version")},
         "all eight scenes retained in every translator/arm summary")
    _add(checks, "binding.feature_and_target_manifests",
         feature_manifest.get("version") == "day11_fresh_jepa_pairs_v1" and
         feature_manifest.get("test_specs") == specs and
         all(feature_manifest.get(k) == v for k, v in expected_pretest.items()) and
         feature_manifest.get("source_hashes") == frozen.get("source_hashes") and
         [r.get("spec") for r in feature_manifest.get("scenes", [])] == specs and
         target_manifest.get("version") == "day11_fresh_wan_targets_v1" and
         target_manifest.get("complete") is True and
         target_manifest.get("feature_manifest_sha256") == hashes["feature_manifest"] and
         target_manifest.get("test_freeze_sha256") == hashes["test_freeze"] and
         target_manifest.get("vae") == evaluation.get("vae") and
         [r.get("sample_id") for r in target_manifest.get("items", [])] ==
         [s["name"]+"__"+view for s in specs for view in ("source", "target")],
         {"feature_scene_count": len(feature_manifest.get("scenes", [])),
          "target_item_count": len(target_manifest.get("items", []))},
         "all eight frozen JEPA pairs and16 true VAE targets share evaluation/pretest bindings")
    expected_render_keys = {s["name"]+"/"+a for s in specs for a in GPU_ARMS}
    expected_render_keys |= {s["name"]+f"/parity_{v}_{r}" for s in specs
                             for v in ("source", "target") for r in ("RGB", "direct")}
    renders = rendering.get("renders", {})
    _add(checks, "binding.rendering", rendering.get("version") == RENDER_VERSION and
         rendering.get("complete") is True and rendering.get("fresh_test_accessed") is True and
         bindings.get("arms") == list(GPU_ARMS) and bindings.get("translator") == "cnn" and
         bindings.get("evaluation_manifest_sha256") == hashes["evaluation_manifest"] and
         bindings.get("source_files") == frozen.get("source_hashes") and
         all(pretest.get(k) == v for k, v in expected_pretest.items()) and
         pretest.get("checkpoint_sha256") == checkpoints["cnn"] and
         pretest.get("rendering_config") == frozen.get("rendering") and
         set(renders) == expected_render_keys,
         {"version": rendering.get("version"), "complete": rendering.get("complete"),
          "render_count": len(renders), "missing": sorted(expected_render_keys-set(renders)),
          "unexpected": sorted(set(renders)-expected_render_keys)},
         "all104 actual runs (8 scenes x (9 arms+4 parity)) under exact frozen bindings")
    config_keys = set(q.RENDER_CONFIGURATION) - {"deduplicate_conditions"}
    _add(checks, "binding.render_configuration",
         all(bindings.get(k) == q.RENDER_CONFIGURATION[k] for k in config_keys) and
         all(bindings.get(k) == _get(frozen, "rendering", k) for k in
             ("prompt_cache_sha256", "prompt_metadata_sha256")) and
         _get(bindings, "runtime", "diffusers") == "0.35.1" and _get(bindings, "runtime", "threads") == 4,
         {k: bindings.get(k) for k in config_keys}, "original fixed GPU model/precision/prompts/settings/parity")
    index, duplicates = {}, []
    for row in latent_rows:
        key = (row.get("scene"), row.get("translator_arm"), row.get("arm"))
        if key in index:
            duplicates.append(key)
        index[key] = row
    expected_rows = {(s["name"], translator, arm) for s in specs for translator in ("cnn", "linear") for arm in q.ARMS}
    _add(checks, "binding.latent_rows", not duplicates and set(index) == expected_rows,
         {"row_count": len(latent_rows), "duplicates": duplicates,
          "missing": sorted(expected_rows-set(index)), "unexpected": sorted(set(index)-expected_rows, key=str)},
         "exactly432 latent rows; no missing, duplicate, or unexpected cases")
    globally_valid = all(c["pass"] for c in checks)
    records, comparisons = [], []
    evaluation_scenes = {r["scene"]: r for r in evaluation.get("scenes", [])}
    for spec in specs:
        scene, seed, scene_checks = spec["name"], spec["seed"], []
        def metric(arm):
            return _get(renders, scene+"/"+arm, "metrics") or {}
        bundle = _get(bindings, "bundles", str(seed)) or {}
        expected_bundle = _get(evaluation_scenes.get(scene, {}), "bundles", "cnn") or {}
        _add(scene_checks, "bundle", bool(bundle.get("sha256")) and
             bundle.get("sha256") == expected_bundle.get("sha256") and
             _get(bundle, "metadata", "checkpoint_sha256") == checkpoints["cnn"] and
             _get(bundle, "metadata", "scene") == scene and
             all(_get(bundle, "metadata", k) == v for k, v in expected_pretest.items()),
             {"render_bundle_sha256": bundle.get("sha256"), "evaluation_bundle_sha256": expected_bundle.get("sha256")},
             "exact fresh evaluation CNN bundle and checkpoint")
        for view in ("source", "target"):
            parity = _get(rendering, "parity", scene+"/"+view) or {}
            values = [q._number(parity.get(k), True) for k in ("cached_vs_live_teacher_max_abs", "denoised_latent_max_abs")]
            _add(scene_checks, "parity."+view, parity.get("passed") is True and
                 parity.get("tolerance") == q.PARITY_TOLERANCE and
                 all(v is not None and v <= q.PARITY_TOLERANCE for v in values),
                 parity, "both parity errors<=1e-4", indeterminate=not bool(parity))
        row = index.get((scene, "cnn", q.PRIMARY), {})
        for region in q.EDIT_REGIONS:
            ratio = q._number(row.get("latent_"+region+"_ratio"), True)
            count = q._number(row.get("latent_"+region+"_count"), True)
            valid = ratio is not None and count is not None and count > 0 and row.get("latent_"+region+"_ratio_degenerate") is False
            _add(scene_checks, "latent."+region, valid and ratio <= q.LATENT_LIMIT,
                 {"ratio": ratio, "count": count}, {"maximum": q.LATENT_LIMIT}, indeterminate=not valid)
        source, target = metric("true_source"), metric("true_target")
        detections = {a: q._detection(metric(a)) for a in ("true_target", q.PRIMARY, q.WRONG, q.SHUFFLED)}
        oracle, primary, wrong = (detections[a] for a in ("true_target", q.PRIMARY, q.WRONG))
        calibrated = oracle["present"] and oracle["centroid_error_to_target_px"] <= q.ORACLE_CENTROID_LIMIT
        _add(scene_checks, "oracle_detection", calibrated, oracle, {"present": True, "maximum_error_px": q.ORACLE_CENTROID_LIMIT},
             indeterminate=not bool(target))
        _add(scene_checks, "primary_detection", calibrated and primary["present"] and
             primary["centroid_error_to_target_px"] <= q.CENTROID_LIMIT, primary,
             {"present": True, "maximum_error_px": q.CENTROID_LIMIT, "oracle_calibrated": True},
             indeterminate=not calibrated or not bool(metric(q.PRIMARY)))
        excess = {a: {region: q._excess(metric(a), source, target, region) for region in q.EDIT_REGIONS}
                  for a in (q.PRIMARY, q.WRONG, q.SHUFFLED)}
        balanced = {a: q._balanced(regions) for a, regions in excess.items()}
        for region in q.EDIT_REGIONS:
            value = excess[q.PRIMARY][region]
            _add(scene_checks, "GPU_excess."+region, value["determinate"] and value["excess_ratio"] <= q.RGB_EXCESS_LIMIT,
                 value, {"maximum": q.RGB_EXCESS_LIMIT, "denominator_min_exclusive": q.RGB_EPSILON},
                 indeterminate=not value["determinate"])
        for region in q.PROTECTED_REGIONS:
            edit, src, tgt = (q._error(m, region) for m in (metric(q.PRIMARY), source, target))
            valid = all(v is not None for v in (edit, src, tgt))
            reference = max(src, tgt) if valid else None
            limit = reference + max(q.PRESERVATION_FRACTION*reference, q.RGB_EPSILON) if valid else None
            _add(scene_checks, "preservation."+region, valid and edit <= limit,
                 {"edit_mse": edit, "source_mse": src, "target_mse": tgt, "reference_mse": reference, "maximum_mse": limit},
                 assessment_config()["protected_formula"], indeterminate=not valid)
        position = primary["present"] and (not wrong["present"] or
                    primary["centroid_error_to_target_px"] < wrong["centroid_error_to_target_px"])
        _add(scene_checks, "negative_position", calibrated and bool(metric(q.WRONG)) and position,
             {"correct": primary, "wrong": wrong}, assessment_config()["negative_position"],
             indeterminate=not calibrated or not bool(metric(q.WRONG)))
        for control in (q.WRONG, q.SHUFFLED):
            valid = balanced[q.PRIMARY] is not None and balanced[control] is not None
            margin = balanced[control]-balanced[q.PRIMARY] if valid else None
            _add(scene_checks, "negative_regional."+control, valid and margin >= q.NEGATIVE_MARGIN,
                 {"correct_balanced_excess": balanced[q.PRIMARY], "control_balanced_excess": balanced[control],
                  "control_minus_correct": margin, "control_detection": detections[control]},
                 {"minimum_control_minus_correct": q.NEGATIVE_MARGIN}, indeterminate=not valid)
        metric_status = _status(scene_checks)
        records.append({"scene": scene, "seed": seed, "shift_px": spec["dx"],
                        "status": metric_status if globally_valid else "indeterminate",
                        "metric_status": metric_status, "global_bindings_valid": globally_valid,
                        "pass": globally_valid and metric_status == "pass", "checks": scene_checks,
                        "failed_check_ids": [c["id"] for c in scene_checks if not c["pass"]],
                        "detections": detections, "region_excess": excess, "balanced_excess": balanced})
        pair = {"scene": scene, "is_gate": False, "GPU": {}, "latent_by_translator": {}, "paired_primary_minus_comparator": {}}
        for arm in q.COMPARATORS:
            regions = {region: q._excess(metric(arm), source, target, region) for region in q.EDIT_REGIONS}
            pair["GPU"][arm] = {"detection": q._detection(metric(arm)), "region_excess": regions,
                                "balanced_excess": q._balanced(regions),
                                "protected_mse": {r: q._error(metric(arm), r) for r in q.PROTECTED_REGIONS}}
        for translator in ("cnn", "linear"):
            pair["latent_by_translator"][translator] = {
                arm: {region: q._number(index.get((scene, translator, arm), {}).get("latent_"+region+"_ratio"), True)
                      for region in q.EDIT_REGIONS} for arm in q.COMPARATORS}
        for arm in q.COMPARATORS:
            if arm == q.PRIMARY:
                continue
            first, second = (pair["GPU"][a]["balanced_excess"] for a in (q.PRIMARY, arm))
            pair["paired_primary_minus_comparator"][arm] = first-second if first is not None and second is not None else None
        comparisons.append(pair)
    counts = {status: sum(r["status"] == status for r in records) for status in ("pass", "fail", "indeterminate")}
    passed = globally_valid and counts["pass"] == 8 and len(records) == 8
    return {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "pass": passed,
            "result": "8/8 engineering pilot pass" if passed else "8/8 engineering pilot criterion not met",
            "scope": "Frozen eight procedural still-image cases; not production success or a video consistency result",
            "fresh_test_accessed": True, "adaptation_or_selection": False, "assessment": assessment_config(),
            "input_sha256": hashes, "checkpoint_freeze_sha256": hashes["trained_checkpoint_freeze"],
            "global_bindings_valid": globally_valid, "binding_checks": checks,
            "scene_count": len(records), "scene_status_counts": counts,
            "metric_scene_status_counts": {status: sum(r["metric_status"] == status for r in records)
                                          for status in ("pass", "fail", "indeterminate")},
            "scenes": records, "paired_comparisons": comparisons,
            "comparison_policy": "descriptive only; no learned-versus-no-F superiority gate; no discarded pairs",
            "predeclared_protocol": __doc__}


def run(evaluation_root, feature_manifest, render_metrics, freeze, publication, checkpoint_freeze, out):
    # Crucially before reading/hashing any fresh manifest, metrics, or summary.
    frozen, trained, hashes = validate_pretest(freeze, publication, checkpoint_freeze)
    root, out = Path(evaluation_root), Path(out)
    if out.exists():
        raise ValueError("Assessment output exists; use a new path to preserve evidence")
    paths = {"evaluation_metrics": root/"metrics.json", "evaluation_summary": root/"summary.json",
             "evaluation_manifest": root/"evaluation_manifest.json", "feature_manifest": Path(feature_manifest),
             "target_manifest": root/"targets"/"latent_manifest.json", "render_metrics": Path(render_metrics),
             "assessment_source": Path(__file__), "qualification_source": HERE/"qualify_development.py"}
    hashes.update({name: q.sha256(path) for name, path in paths.items()})
    result = assess(*(q.read_json(paths[name]) for name in
                      ("evaluation_metrics", "evaluation_summary", "evaluation_manifest", "feature_manifest",
                       "target_manifest", "render_metrics")), frozen, trained, hashes)
    result["pretest_validated_before_fresh_metrics_access"] = True
    result["validated_source_hashes"] = frozen["source_hashes"]
    paths.update({"test_freeze": Path(freeze), "test_publication": Path(publication),
                  "trained_checkpoint_freeze": Path(checkpoint_freeze)})
    result["input_paths"] = {key: str(path.resolve()) for key, path in paths.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"path": str(out), "sha256": q.sha256(out), "pass": result["pass"],
                      "scene_status_counts": result["scene_status_counts"]}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-root", "feature-manifest", "render-metrics", "freeze", "publication", "checkpoint-freeze", "out"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    run(args.evaluation_root, args.feature_manifest, args.render_metrics, args.freeze,
        args.publication, args.checkpoint_freeze, args.out)


if __name__ == "__main__":
    main()
