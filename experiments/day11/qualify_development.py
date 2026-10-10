"""Fixed engineering qualification for the Day11 DEVELOPMENT transport pilot.

Declared before reading the follow-up GPU results. No optimization, arm
selection, threshold fitting, fresh-scene access, or experiment execution.
The sole primary arm is provenance_residual_copy_repair. All checks must pass:

1. Each of eight development scenes: latent source-hole AND destination
   source-relative MSE ratios <= 0.5 (not an average over scenes/regions).
2. Each of two GPU scenes: primary detection exists and target-centroid error
   <= 4 pixels. True-target detection must exist with error <= 2 pixels before
   interpreting this fixed synthetic-color diagnostic.
3. GPU source-hole AND destination: (E_primary - E_true_target) /
   (E_true_source - E_true_target) <= 0.5. E is rendered RGB MSE to target.
   Denominator must exceed 1/255^2; otherwise this check is indeterminate/fail.
4. GPU background AND distractor: E_primary <= E_ref + max(.25 E_ref,1/255^2),
   where E_ref=max(E_true_source,E_true_target), using the same regional MSE.
5. Correct centroid error is strictly less than wrong-direction error. Both
   wrong-direction and shuffled mean(hole excess,destination excess) scores
   must be >= correct + 0.10. A missing negative-control detection is recorded
   as positional failure, treated as +infinity only for position ordering;
   it never waives either required regional comparison. Infinity is not
   serialized as a numeric measurement.

No learned-versus-no-F superiority criterion is imposed; fixed comparator
results are reported separately. A pass is an engineering-pilot gate only,
not production success, general object detection, or held-out generalization.
All inputs and this source are hash-bound. Missing/incomplete/mismatched
evidence makes the gate fail; no result is silently dropped from aggregation.
This module uses only JSON/math/filesystem helpers and never loads JEPA,
generator weights, RGB images, latent arrays, or fresh data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


VERSION = "day11_development_engineering_gate_v1"
TRANSPORT_VERSION = "day11_development_provenance_transport_v1"
RENDER_VERSION = "day11_vace_transport_render_v1"
PRIMARY = "provenance_residual_copy_repair"
WRONG = "provenance_residual_wrong_direction"
SHUFFLED = "provenance_residual_shuffled"
LOCAL = "provenance_local_copy_repair"
COMPARATORS = (PRIMARY, LOCAL, "provenance_copy_copy_repair",
               "residual_copy_repair", "absolute_copy_repair")
CONDITIONS = ("source", "genuine_target", "copy_repair", "wrong_direction", "shuffled")
ARMS = ("true_source", "true_target") + tuple(
    route + "_" + condition for route in
    ("absolute", "residual", "provenance_copy", "provenance_local", "provenance_residual")
    for condition in CONDITIONS)
DEV_SEEDS = tuple(range(13500, 13508))
GPU_SEEDS = (13500, 13501)
EDIT_REGIONS = ("source_hole", "destination")
PROTECTED_REGIONS = ("background", "distractor")
LATENT_LIMIT = .5
RGB_EXCESS_LIMIT = .5
CENTROID_LIMIT = 4.
ORACLE_CENTROID_LIMIT = 2.
NEGATIVE_MARGIN = .10
PRESERVATION_FRACTION = .25
RGB_EPSILON = 1. / 255 ** 2
PARITY_TOLERANCE = 1e-4
# Literal configuration avoids importing torch/diffusers or executing renderer
# modules during qualification. These are the original, fixed pilot settings.
RENDER_CONFIGURATION = {
    "model_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "model_revision": "ec4d2cb062b548996b179d493fdd05340de702a1",
    "settings": {"height": 384, "width": 384, "num_frames": 1,
                 "num_inference_steps": 20, "guidance_scale": 5.0,
                 "conditioning_scale": 1.0, "seed": 1111},
    "scheduler_class": "FlowMatchEulerDiscreteScheduler",
    "transformer_dtype": "float16", "vae_dtype": "float32",
    "attention": "Diffusers native PyTorch SDPA",
    "parity_max_abs_tolerance": PARITY_TOLERANCE,
    "prompt": "Two patterned colored balls on a textured background. Static camera, simple geometric scene.",
    "negative_prompt": "text, watermark, blurry, distorted, extra objects",
    "max_sequence_length": 128, "deduplicate_conditions": False,
}
RENDER_SOURCES = ("render_transport.py", "render.py", "vace_bridge.py", "transport.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _get(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _number(value, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    value = float(value)
    return value if math.isfinite(value) and (not nonnegative or value >= 0) else None


def _detection(metrics):
    diagnostic = _get(metrics, "selected_color_diagnostic") or {}
    pixels = _number(diagnostic.get("pixels"), nonnegative=True)
    center = diagnostic.get("centroid_xy")
    error = _number(diagnostic.get("centroid_error_to_target_px"), nonnegative=True)
    present = (pixels is not None and pixels > 0 and isinstance(center, list)
               and len(center) == 2 and all(_number(x) is not None for x in center)
               and error is not None)
    return {"present": bool(present), "pixels": pixels, "centroid_xy": center,
            "centroid_error_to_target_px": error,
            "positional_failure": not bool(present),
            "general_detector": False,
            "missing_detection_ordering": "positive infinity, not a measured centroid" if not present else None}


def _error(metrics, region):
    return _number(_get(metrics, "rgb_mse", region, "target_rgb"), nonnegative=True)


def _excess(metrics, source_metrics, target_metrics, region):
    edit, source, target = (_error(m, region) for m in (metrics, source_metrics, target_metrics))
    denominator = source - target if source is not None and target is not None else None
    valid = edit is not None and denominator is not None and denominator > RGB_EPSILON
    return {"edit_mse_to_target": edit, "true_source_render_mse_to_target": source,
            "true_target_render_mse_to_target": target, "denominator": denominator,
            "denominator_min_exclusive": RGB_EPSILON, "determinate": bool(valid),
            "excess_ratio": (edit - target) / denominator if valid else None}


def _balanced(regions):
    values = [regions[r]["excess_ratio"] for r in EDIT_REGIONS]
    return sum(values) / len(values) if all(value is not None for value in values) else None


def evaluate_gate(latent_rows, summary, manifest, render, trained, hashes):
    """Pure qualification; caller supplies parsed evidence and exact-file hashes."""
    checks = []

    def check(name, outcome, observed, required, indeterminate=False):
        passed = outcome is True
        checks.append({"id": name, "pass": passed,
                       "status": "pass" if passed else "indeterminate" if indeterminate else "fail",
                       "observed": observed, "required": required})

    primary_models = [m for m in trained.get("models", []) if m.get("arm") == "cnn"]
    checkpoint = primary_models[0].get("checkpoint_sha256") if len(primary_models) == 1 else None
    check("binding.trained_checkpoint", bool(checkpoint) and
          trained.get("version") == "day11_genuine_jepa_to_wan_training_v1" and
          trained.get("fresh_test_accessed") is False and trained.get("test_accessed") is False and
          len(primary_models) == 1 and primary_models[0].get("selected_epoch") == 150,
          {"version": trained.get("version"), "cnn_checkpoint_sha256": checkpoint,
           "fresh_test_accessed": trained.get("fresh_test_accessed"),
           "test_accessed": trained.get("test_accessed")},
          "one frozen CNN at fixed epoch150; no fresh test accessed")
    check("binding.transport", manifest.get("version") == TRANSPORT_VERSION and
          manifest.get("complete") is True and manifest.get("fresh_test_accessed") is False and
          manifest.get("checkpoint_sha256") == checkpoint and
          manifest.get("jepa_manifest_sha256") == trained.get("jepa_manifest_sha256") and
          manifest.get("target_manifest_sha256") == trained.get("target_manifest_sha256") and
          manifest.get("metrics_sha256") == hashes.get("transport_metrics") and
          manifest.get("summary_sha256") == hashes.get("transport_summary") and
          manifest.get("all_development_seeds") == list(DEV_SEEDS) and
          manifest.get("renderer_seeds") == list(GPU_SEEDS) and
          set(manifest.get("arms", {})) == set(ARMS),
          {key: manifest.get(key) for key in
           ("version", "complete", "fresh_test_accessed", "checkpoint_sha256", "metrics_sha256",
            "summary_sha256", "all_development_seeds", "renderer_seeds")},
          "completed eight-scene/27-arm development evidence bound to original checkpoint and supplied files")
    check("binding.summary", summary.get("version") == TRANSPORT_VERSION and
          summary.get("scene_count") == 8 and summary.get("fresh_test_accessed") is False and
          set(summary.get("arms", {})) == set(ARMS) and
          _get(summary, "methods", PRIMARY, "scene_count") == 8,
          {"version": summary.get("version"), "scene_count": summary.get("scene_count"),
           "primary_scene_count": _get(summary, "methods", PRIMARY, "scene_count")},
          "same completed development summary; primary has all eight scenes")
    index, duplicate = {}, []
    for row in latent_rows:
        key = (row.get("scene"), row.get("arm"))
        if key in index:
            duplicate.append(list(key))
        index[key] = row
    expected_rows = {(f"dev_{seed}", arm) for seed in DEV_SEEDS for arm in ARMS}
    check("binding.latent_rows", not duplicate and set(index) == expected_rows,
          {"row_count": len(latent_rows), "unique_count": len(index), "duplicates": duplicate,
           "missing": sorted(expected_rows - set(index)),
           "unexpected": sorted(set(index) - expected_rows, key=str)},
          "exactly eight scenes x 27 arms, without missing or repeated rows")
    bindings = render.get("bindings", {})
    renders = render.get("renders", {})
    expected_renders = {f"dev_{seed}/{arm}" for seed in GPU_SEEDS for arm in ARMS}
    expected_renders |= {f"dev_{seed}/parity_{view}_{route}" for seed in GPU_SEEDS
                         for view in ("source", "target") for route in ("RGB", "direct")}
    check("binding.gpu_render", render.get("version") == RENDER_VERSION and
          render.get("complete") is True and render.get("fresh_test_accessed") is False and
          bindings.get("followup_version") == TRANSPORT_VERSION and
          bindings.get("fresh_test_accessed") is False and
          bindings.get("evaluation_manifest_sha256") == hashes.get("transport_manifest") and
          set(bindings.get("arm_catalogue", [])) == set(ARMS) and set(renders) == expected_renders,
          {"version": render.get("version"), "complete": render.get("complete"),
           "evaluation_manifest_sha256": bindings.get("evaluation_manifest_sha256"),
           "render_count": len(renders), "missing": sorted(expected_renders - set(renders)),
           "unexpected": sorted(set(renders) - expected_renders)},
          "completed actual two-scene transport GPU render, all27 arms plus all8 parity runs")
    check("binding.gpu_configuration",
          all(bindings.get(key) == value for key, value in RENDER_CONFIGURATION.items()) and
          _get(bindings, "runtime", "diffusers") == "0.35.1" and
          _get(bindings, "runtime", "threads") == 4,
          {**{key: bindings.get(key) for key in RENDER_CONFIGURATION},
           "diffusers": _get(bindings, "runtime", "diffusers"),
           "threads": _get(bindings, "runtime", "threads")},
          {**RENDER_CONFIGURATION, "diffusers": "0.35.1", "threads": 4})
    check("binding.gpu_sources",
          all(bool(hashes.get("source:" + name)) and
              _get(bindings, "source_files", name) == hashes.get("source:" + name)
              for name in RENDER_SOURCES) and
          _get(manifest, "source_hashes", "experiments/day11/transport.py") == hashes.get("source:transport.py"),
          {"render_source_files": bindings.get("source_files"),
           "transport_manifest_source": _get(manifest, "source_hashes", "experiments/day11/transport.py")},
          "GPU and transport manifest bind the exact current, separately published source files")
    manifest_scenes = {str(row.get("seed")): row for row in manifest.get("scenes", [])}
    for seed in GPU_SEEDS:
        bound = _get(bindings, "bundles", str(seed)) or {}
        scene = manifest_scenes.get(str(seed), {})
        check(f"binding.gpu_bundle.{seed}",
              bound.get("sha256") == scene.get("renderer_sha256") and bool(bound.get("sha256")) and
              _get(bound, "metadata", "checkpoint_sha256") == checkpoint,
              {"render_bundle_sha256": bound.get("sha256"),
               "transport_bundle_sha256": scene.get("renderer_sha256"),
               "render_checkpoint_sha256": _get(bound, "metadata", "checkpoint_sha256")},
              "GPU uses exact transport bundle and same frozen CNN checkpoint")
        for view in ("source", "target"):
            parity = _get(render, "parity", f"dev_{seed}/{view}") or {}
            tolerance = _number(parity.get("tolerance"), nonnegative=True)
            errors = [_number(parity.get(key), nonnegative=True) for key in
                      ("cached_vs_live_teacher_max_abs", "denoised_latent_max_abs")]
            check(f"binding.parity.{seed}.{view}", parity.get("passed") is True and
                  tolerance == PARITY_TOLERANCE and all(e is not None and e <= PARITY_TOLERANCE for e in errors),
                  parity, "both cached/live and official/direct parity passed at fixed 1e-4 tolerance")

    latent_details = {}
    for seed in DEV_SEEDS:
        row = index.get((f"dev_{seed}", PRIMARY), {})
        details = {}
        for region in EDIT_REGIONS:
            ratio = _number(row.get("latent_" + region + "_ratio"), nonnegative=True)
            degenerate = row.get("latent_" + region + "_ratio_degenerate")
            count = _number(row.get("latent_" + region + "_count"), nonnegative=True)
            determinate = ratio is not None and degenerate is False and count is not None and count > 0
            details[region] = {"ratio": ratio, "ratio_degenerate": degenerate, "count": count}
            check(f"latent.{seed}.{region}", determinate and ratio <= LATENT_LIMIT,
                  details[region], {"maximum_ratio": LATENT_LIMIT, "nondegenerate": True},
                  indeterminate=not determinate)
        latent_details[str(seed)] = details

    gpu_details, comparison = {}, {"is_gate": False, "arm_selection": "none", "arms": {}}
    for seed in GPU_SEEDS:
        def metrics(arm):
            return _get(renders, f"dev_{seed}/{arm}", "metrics") or {}
        source, target = metrics("true_source"), metrics("true_target")
        detections = {arm: _detection(metrics(arm)) for arm in ("true_target", PRIMARY, WRONG, SHUFFLED)}
        oracle = detections["true_target"]
        calibrated = oracle["present"] and oracle["centroid_error_to_target_px"] <= ORACLE_CENTROID_LIMIT
        check(f"gpu.{seed}.oracle_detection", calibrated, oracle,
              {"present": True, "maximum_target_error_px": ORACLE_CENTROID_LIMIT})
        primary = detections[PRIMARY]
        check(f"gpu.{seed}.primary_detection", calibrated and primary["present"] and
              primary["centroid_error_to_target_px"] <= CENTROID_LIMIT, primary,
              {"present": True, "maximum_target_error_px": CENTROID_LIMIT, "oracle_calibrated": True},
              indeterminate=not calibrated)
        excess = {arm: {region: _excess(metrics(arm), source, target, region)
                        for region in EDIT_REGIONS} for arm in (PRIMARY, WRONG, SHUFFLED)}
        for region in EDIT_REGIONS:
            value = excess[PRIMARY][region]
            check(f"gpu.{seed}.{region}.excess", value["determinate"] and
                  value["excess_ratio"] <= RGB_EXCESS_LIMIT, value,
                  {"maximum_excess_ratio": RGB_EXCESS_LIMIT,
                   "denominator_min_exclusive": RGB_EPSILON}, indeterminate=not value["determinate"])
        protected = {}
        for region in PROTECTED_REGIONS:
            edit, src, tgt = (_error(m, region) for m in (metrics(PRIMARY), source, target))
            valid = all(value is not None for value in (edit, src, tgt))
            reference = max(src, tgt) if valid else None
            limit = reference + max(PRESERVATION_FRACTION * reference, RGB_EPSILON) if valid else None
            value = {"edit_mse_to_target": edit, "source_render_mse_to_target": src,
                     "target_render_mse_to_target": tgt, "reference_mse": reference, "maximum_mse": limit}
            protected[region] = value
            check(f"gpu.{seed}.{region}.preservation", valid and edit <= limit, value,
                  "edit <= max(source,target) + max(.25*max(source,target),1/255^2)",
                  indeterminate=not valid)
        wrong = detections[WRONG]
        position_order = primary["present"] and (not wrong["present"] or
                         primary["centroid_error_to_target_px"] < wrong["centroid_error_to_target_px"])
        check(f"gpu.{seed}.negative_direction_position", calibrated and position_order,
              {"correct": primary, "wrong_direction": wrong},
              "correct finite target-centroid error strictly below wrong direction; missing control is positional failure",
              indeterminate=not calibrated)
        balanced = {arm: _balanced(value) for arm, value in excess.items()}
        for control in (WRONG, SHUFFLED):
            valid = balanced[PRIMARY] is not None and balanced[control] is not None
            margin = balanced[control] - balanced[PRIMARY] if valid else None
            check(f"gpu.{seed}.negative_regional.{control}", valid and margin >= NEGATIVE_MARGIN,
                  {"correct_balanced_excess": balanced[PRIMARY], "control_balanced_excess": balanced[control],
                   "control_minus_correct": margin, "control_detection": detections[control]},
                  {"minimum_control_minus_correct": NEGATIVE_MARGIN,
                   "both_hole_and_destination_required": True}, indeterminate=not valid)
        gpu_details[str(seed)] = {"detections": detections, "region_excess": excess,
                                 "balanced_excess": balanced, "protected_regions": protected}
        for arm in COMPARATORS:
            regions = {region: _excess(metrics(arm), source, target, region) for region in EDIT_REGIONS}
            comparison["arms"].setdefault(arm, {"gpu_scenes": {}, "latent_scenes": {}})["gpu_scenes"][str(seed)] = {
                "detection": _detection(metrics(arm)), "region_excess": regions,
                "balanced_excess": _balanced(regions),
                "protected_mse_to_target": {r: _error(metrics(arm), r) for r in PROTECTED_REGIONS}}
    for arm in COMPARATORS:
        for seed in DEV_SEEDS:
            row = index.get((f"dev_{seed}", arm), {})
            comparison["arms"][arm]["latent_scenes"][str(seed)] = {
                region: _number(row.get("latent_" + region + "_ratio"), nonnegative=True)
                for region in EDIT_REGIONS}
    comparison["primary_minus_no_F_balanced_gpu_excess"] = {}
    for seed in GPU_SEEDS:
        a = comparison["arms"][PRIMARY]["gpu_scenes"][str(seed)]["balanced_excess"]
        b = comparison["arms"][LOCAL]["gpu_scenes"][str(seed)]["balanced_excess"]
        comparison["primary_minus_no_F_balanced_gpu_excess"][str(seed)] = a - b if a is not None and b is not None else None
    return {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "pass": all(item["pass"] for item in checks), "primary_arm": PRIMARY,
            "scope": "Fixed engineering development pilot; not production success or held-out generalization",
            "fresh_test_accessed": False, "threshold_selection": "predeclared; no fitted thresholds or arm selection",
            "checkpoint_freeze_sha256": hashes.get("checkpoint_freeze"), "input_sha256": hashes,
            "criteria": {"latent_region_ratio_max": LATENT_LIMIT, "GPU_region_excess_max": RGB_EXCESS_LIMIT,
                         "GPU_centroid_max_px": CENTROID_LIMIT, "oracle_centroid_max_px": ORACLE_CENTROID_LIMIT,
                         "GPU_negative_balanced_excess_margin_min": NEGATIVE_MARGIN,
                         "RGB_denominator_min_exclusive": RGB_EPSILON,
                         "protected_relative_allowance": PRESERVATION_FRACTION,
                         "protected_absolute_allowance": RGB_EPSILON},
            "check_count": len(checks), "passed_check_count": sum(item["pass"] for item in checks),
            "failed_check_ids": [item["id"] for item in checks if not item["pass"]],
            "checks": checks, "latent_scenes": latent_details, "gpu_scenes": gpu_details,
            "comparator_report": comparison, "predeclared_protocol": __doc__}


def run(transport_root, render_metrics, checkpoint_freeze, out):
    root, out = Path(transport_root), Path(out)
    paths = {"transport_metrics": root / "metrics.json", "transport_summary": root / "summary.json",
             "transport_manifest": root / "evaluation_manifest.json", "render_metrics": Path(render_metrics),
             "checkpoint_freeze": Path(checkpoint_freeze), "qualification_source": Path(__file__)}
    paths.update({"source:" + name: Path(__file__).with_name(name) for name in RENDER_SOURCES})
    if out.exists():
        raise ValueError("Qualification output already exists; use a new path to preserve evidence")
    hashes = {key: sha256(path) for key, path in paths.items()}
    result = evaluate_gate(*(read_json(paths[key]) for key in
                             ("transport_metrics", "transport_summary", "transport_manifest", "render_metrics", "checkpoint_freeze")),
                           hashes)
    result["input_paths"] = {key: str(path.resolve()) for key, path in paths.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an existing gate.
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"path": str(out), "sha256": sha256(out), "pass": result["pass"],
                      "passed_check_count": result["passed_check_count"], "check_count": result["check_count"]}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-root", required=True)
    parser.add_argument("--render-metrics", required=True)
    parser.add_argument("--checkpoint-freeze", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.transport_root, args.render_metrics, args.checkpoint_freeze, args.out)


if __name__ == "__main__":
    main()
