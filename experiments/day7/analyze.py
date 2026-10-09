"""Independent Day7 accounting and paired-seed/paired-clip report.

Run only after test/results.json and every saved prediction archive exist.
The audit does not import train.py or load a model. It regenerates scoring
labels only after validating the completed report and checkpoint freeze, then
scores saved predictions with the unchanged Day4 benchmark. All bootstrap
resamples use the SAME18 clip draw across three paired training seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for folder in (HERE.parent / "day4", HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import benchmark as bench
import data

SEEDS = (1701, 1702, 1703)
ARMS = ("retrieval_only", "predictive", "coordinate_only")
BASELINES = ("global_full", "selected_day4", "global_projected_centered")
METHODS = BASELINES + tuple(f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS)
RESAMPLES, BOOTSTRAP_SEED = 2000, 914
LOCALIZATION = "localization_accuracy_given_visible"
FALSE_PRESENCE = "false_presence_given_absent"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _bool(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise ValueError(f"Invalid boolean {value!r}")


def _equal(actual, expected, path="value"):
    """Recursive numeric comparison; raise rather than silently repair evidence."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected dictionary")
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f"{path}.{key}: missing")
            _equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length mismatch")
        for i, (a, b) in enumerate(zip(actual, expected)):
            _equal(a, b, f"{path}[{i}]")
    elif isinstance(expected, bool):
        if _bool(actual) != expected:
            raise AssertionError(f"{path}: boolean mismatch")
    elif expected is None:
        if actual is not None:
            raise AssertionError(f"{path}: expected null")
    elif isinstance(expected, (int, float, np.number)):
        if not np.isclose(float(actual), float(expected), rtol=1e-8, atol=1e-8):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _test_binding_records(binding):
    """Validate every held-out binding record before rendering any scene."""
    specs = data.scene_specs("test")
    records = binding.get("clips")
    if not isinstance(records, list) or len(records) != len(specs):
        raise RuntimeError("Test feature binding must contain exactly18 records")
    if any(not isinstance(v, dict) for v in records):
        raise RuntimeError("Invalid test feature binding record")
    indexed = {v.get("name"): v for v in records}
    if len(indexed) != len(specs) or set(indexed) != {v["name"] for v in specs}:
        raise RuntimeError("Test feature binding has duplicate, missing or extra scenes")
    for spec in specs:
        record = indexed[spec["name"]]
        if record.get("split") != "test" or any(record.get(k) != spec[k] for k in ("name", "seed", "condition")):
            raise RuntimeError("Test feature binding metadata differs from fixed split manifest")
    return indexed


def _validate_feature_provenance(training_binding, test_binding):
    for key in ("projection_sha256", "mean", "extractor_provenance"):
        if key not in training_binding or key not in test_binding or test_binding[key] != training_binding[key]:
            raise RuntimeError(f"Test feature {key} differs from frozen training feature binding")
    if training_binding["mean"].get("fit_split") != "train":
        raise RuntimeError("Feature centering was not bound to the training split")
    sources = training_binding["extractor_provenance"]["extractor_sources"]
    expected = {"experiments/day7/extract.py", "experiments/day7/data.py",
                "experiments/day4/benchmark.py", "experiments/day4/tracking.py",
                "experiments/day4/run_experiment.py"}
    if set(sources) != expected:
        raise RuntimeError("Extractor provenance source-key set differs from expected files")
    for relative, digest in sources.items():
        source = HERE.parent.parent / relative
        if not source.is_file() or sha256(source) != digest:
            raise RuntimeError(f"Frozen extractor source mismatch: {relative}")
    _test_binding_records(test_binding)


def validate_completed_run(run):
    """Guard before any test scene is rendered; source and files must be complete."""
    run = Path(run)
    mandatory = ("test/results.json", "test/prediction_rows.csv", "test/future_rows.csv",
                 "checkpoint_freeze.json", "training_config.json")
    for name in mandatory:
        if not (run / name).is_file():
            raise RuntimeError(f"Completed run required before audit: missing {name}")
    report = json.loads((run / "test/results.json").read_text())
    freeze = json.loads((run / "checkpoint_freeze.json").read_text())
    if (report.get("split") != "test" or report.get("test_selection") != "none"
            or report.get("all_seeds_reported") is not True):
        raise RuntimeError("Report is not a completed all-seed unselected test run")
    if report.get("checkpoint_freeze_sha256") != sha256(run / "checkpoint_freeze.json"):
        raise RuntimeError("Report/freeze checksum mismatch")
    if report.get("checkpoint_freeze") != freeze:
        raise RuntimeError("Report contains a different checkpoint freeze")
    if freeze.get("test_accessed") is not False:
        raise RuntimeError("Expected pre-test checkpoint freeze")
    if freeze.get("test_seed_manifest") != data.scene_specs("test"):
        raise RuntimeError("Held-out manifest changed")
    if set(report.get("methods", {})) != set(METHODS):
        raise RuntimeError("Report must include exactly three baselines and nine learned models")
    records = freeze.get("models", [])
    if len(records) != 9 or {(v["arm"], v["seed"]) for v in records} != {(a, s) for a in ARMS for s in SEEDS}:
        raise RuntimeError("Freeze does not contain every predeclared training seed/arm")
    config = json.loads((run / "training_config.json").read_text())
    if freeze.get("training_config") != config:
        raise RuntimeError("Training configuration changed after freeze")
    expected_sources = {"day7/train.py", "day7/model.py", "day7/data.py", "day4/benchmark.py"}
    if set(config["source_hashes"]) != expected_sources:
        raise RuntimeError("Training source-key set differs from expected frozen files")
    digest = hashlib.sha256(json.dumps(config["source_hashes"], sort_keys=True).encode()).hexdigest()
    if config.get("source_digest") != digest:
        raise RuntimeError("Training source digest does not match frozen source hashes")
    for relative, digest in config["source_hashes"].items():
        source = HERE.parent / relative
        if not source.is_file() or sha256(source) != digest:
            raise RuntimeError(f"Frozen source mismatch: {relative}")
    _validate_feature_provenance(config["feature_binding"], report["test_feature_binding"])
    checksums = {name: sha256(run / name) for name in mandatory}
    checkpoint_status = {}
    for record in records:
        method = f"{record['arm']}_seed{record['seed']}"
        if report["methods"][method].get("selected_epoch") != record["epoch"]:
            raise RuntimeError("Reported test epoch differs from development freeze")
        checkpoint = run / record["path"]
        if checkpoint.is_file():
            if sha256(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError(f"Frozen checkpoint changed: {checkpoint}")
            checkpoint_status[record["path"]] = "present, checksum verified"
        else:
            checkpoint_status[record["path"]] = "not included in audit bundle; prediction accounting only"
        selected = checkpoint.with_name("selected.json")
        curves = checkpoint.with_name("curves.csv")
        if not selected.is_file() or not curves.is_file():
            raise RuntimeError("Selected development records and full learning curves are required")
        if sha256(selected) != record["selection_sha256"]:
            raise RuntimeError(f"Development selection record changed: {selected}")
        selected_report = json.loads(selected.read_text())
        if sha256(curves) != selected_report["curves_sha256"]:
            raise RuntimeError("Learning curve checksum mismatch")
        checksums[str(selected.relative_to(run))] = sha256(selected)
        checksums[str(curves.relative_to(run))] = sha256(curves)
    for method in METHODS:
        for spec in data.scene_specs("test"):
            relative = f"test/predictions/{method}/{spec['name']}.npz"
            path = run / relative
            if not path.is_file():
                raise RuntimeError(f"Incomplete test predictions: missing {relative}")
            checksums[relative] = sha256(path)
    return report, freeze, checksums, checkpoint_status


def _load_prediction(path, learned=False):
    with np.load(path, allow_pickle=False) as archive:
        scores, cells, eligible = (archive[k].copy() for k in ("scores", "cells", "eligible"))
        prediction = bench.Predictions(scores, cells, eligible)
        bench.validate_predictions(prediction, 32)
        if eligible.dtype != np.bool_ or not np.issubdtype(cells.dtype, np.integer):
            raise AssertionError("Saved eligibility/cell dtypes are invalid")
        if learned:
            visibility = archive["visibility"]
            logits = archive["logits"]
            if logits.shape != (32, 4, 577) or visibility.shape != (32, 4):
                raise AssertionError("Learned logits/visibility shape mismatch")
            if not np.isfinite(logits).all() or not np.isfinite(visibility).all():
                raise AssertionError("Nonfinite learned output")
            if not np.array_equal(eligible, visibility >= .5):
                raise AssertionError("Learned emission differs from fixed visibility>=0.5")
            flat = logits[..., :-1].argmax(-1)
            if not np.array_equal(cells, np.stack((flat // 24, flat % 24), -1)):
                raise AssertionError("Saved cell differs from spatial argmax")
            shifted = logits - logits.max(-1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(-1, keepdims=True)
            if not np.allclose(visibility, 1 - probabilities[..., -1], atol=2e-6, rtol=2e-6):
                raise AssertionError("Visibility differs from factorized null probability")
            expected_score = np.take_along_axis(probabilities[..., :-1], flat[..., None], -1)[..., 0]
            if not np.allclose(scores, expected_score, atol=2e-7, rtol=2e-6):
                raise AssertionError("Saved confidence differs from logits")
    return prediction


def recount_predictions(run, report, freeze):
    """Independent canonical scorer, using no train.py implementation or features."""
    run = Path(run)
    rows, expected_future = [], {}
    bindings = _test_binding_records(report["test_feature_binding"])
    for spec in data.scene_specs("test"):
        scene = data.generate_scene(**spec)
        if hashlib.sha256(scene.frames.tobytes()).hexdigest() != bindings[scene.name]["rgb_sha256"]:
            raise AssertionError("Scored scene RGB differs from recorded feature source")
        target = data.supervision(scene.masks)
        if not np.array_equal(target["state"], bench.target_states(scene)):
            raise AssertionError("Day7 supervision differs from canonical Day4 states")
        if not np.array_equal(target["patch_labels"], bench.patch_labels(scene.masks).reshape(32, 576)):
            raise AssertionError("Day7 supervision differs from canonical Day4 patch labels")
        expected_future[scene.name] = {(t, t + 8, bench.PART_NAMES[k + 1]): int(target["visible"][t + 8].sum())
                                       for t in range(24) for k in range(4) if target["visible"][t + 8, k]}
        for method in METHODS:
            prediction = _load_prediction(run / "test/predictions" / method / f"{scene.name}.npz",
                                          learned=method not in BASELINES)
            threshold = freeze["baselines"][method]["threshold"] if method in BASELINES else 0.
            rows.extend(bench.score_scene(scene, prediction, threshold, method))
    # Compare every saved CSV observation, including identity-switch/recovery bookkeeping.
    with (run / "test/prediction_rows.csv").open(newline="") as stream:
        recorded = list(csv.DictReader(stream))
    key = lambda r: (r["method"], r["scene"], r["target"], int(r["tubelet"]))
    saved = {key(r): r for r in recorded}
    if len(saved) != len(recorded) or len(saved) != len(rows):
        raise AssertionError("Prediction CSV has duplicate/missing/extra rows")
    for row in rows:
        if key(row) not in saved:
            raise AssertionError(f"Prediction CSV missing {key(row)}")
        _equal(saved[key(row)], row, str(key(row)))
    methods = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        overall = bench.summarize_rows(subset)
        condition = {c: bench.summarize_rows([r for r in subset if r["condition"] == c]) for c in data.CONDITIONS}
        _equal(report["methods"][method]["overall"], overall, method + ".overall")
        _equal(report["methods"][method]["by_condition"], condition, method + ".by_condition")
        _equal(report["methods"][method]["uncertainty"], bench.summarize_with_ci(subset),
               method + ".uncertainty")
        methods[method] = {"overall": overall, "by_condition": condition}
    for seed in SEEDS:
        a = [r for r in rows if r["method"] == f"predictive_seed{seed}"]
        b = [r for r in rows if r["method"] == f"retrieval_only_seed{seed}"]
        expected = {metric: bench.paired_bootstrap_difference(a, b, metric=metric)
                    for metric in (LOCALIZATION, FALSE_PRESENCE, "recovery_accuracy")}
        _equal(report["predictive_minus_retrieval_paired_by_training_seed"][str(seed)],
               expected, f"seed{seed}.paired")
    for arm in ARMS:
        for metric in (LOCALIZATION, FALSE_PRESENCE, "wrong_car_given_visible", "recovery_accuracy"):
            values = [methods[f"{arm}_seed{s}"]["overall"][metric]["rate"] for s in SEEDS]
            _equal(report["seed_summary"][arm][metric], {"values": values, "mean": float(np.mean(values))},
                   arm + ".seed_summary." + metric)
    return rows, methods, expected_future


def _counts(rows, method_names, names):
    """[training seed, clip, hit/visible/false-presence/absent] sufficient counts."""
    result = np.zeros((len(method_names), len(names), 4), np.int64)
    indexed = {}
    for row in rows:
        indexed.setdefault((row["method"], row["scene"]), []).append(row)
    for s, method in enumerate(method_names):
        for c, name in enumerate(names):
            subset = indexed.get((method, name))
            if not subset:
                raise ValueError(f"Missing paired observations for {method}/{name}")
            result[s, c] = (sum(r["hit"] for r in subset if r["state"] == "visible"),
                            sum(r["state"] == "visible" for r in subset),
                            sum(r["present"] for r in subset if r["state"] == "absent"),
                            sum(r["state"] == "absent" for r in subset))
    return result


def _rate_vectors(counts):
    """Last dimension is four counts; maintain seed and draw axes."""
    if np.any(counts[..., 1] <= 0) or np.any(counts[..., 3] <= 0):
        raise ValueError("Utility needs positive visible and absent denominators")
    visible = counts[..., 0] / counts[..., 1]
    specificity = 1 - counts[..., 2] / counts[..., 3]
    return np.stack(((visible + specificity) / 2, visible, specificity), axis=-1)


def paired_mean_contrast(counts_a, counts_b, resamples=RESAMPLES, seed=BOOTSTRAP_SEED):
    """Pair clips AND training seeds; share each clip resample across all seeds.

    Utility is computed from each seed's pooled clip counts, then averaged over
    the three seeds. Seeds are not resampled and repeated models do not multiply
    the number of independent videos. The interval is conditional on these seeds.
    """
    a, b = np.asarray(counts_a), np.asarray(counts_b)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 4:
        raise ValueError("Counts must be matching [seed,clip,4] arrays")
    if not np.array_equal(a[..., [1, 3]], b[..., [1, 3]]):
        raise ValueError("Paired methods have different target denominators")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Negative counts")
    nseed, nclip, _ = a.shape
    draws = np.random.default_rng(seed).integers(0, nclip, (resamples, nclip))
    pooled_a, pooled_b = a.sum(axis=1), b.sum(axis=1)
    per_seed = _rate_vectors(pooled_a) - _rate_vectors(pooled_b)
    # a[:,draws] has [seed,bootstrap,drawn_clip,4]; exactly one draw per bootstrap.
    sampled_a, sampled_b = a[:, draws].sum(axis=2), b[:, draws].sum(axis=2)
    valid = ((sampled_a[..., 1] > 0) & (sampled_a[..., 3] > 0)).all(axis=0)
    if not valid.any():
        raise ValueError("No bootstrap draw includes both visible and absent targets")
    sampled = (_rate_vectors(sampled_a[:, valid]) - _rate_vectors(sampled_b[:, valid])).mean(axis=0)
    metrics = ("balanced_utility", "visible_localization", "absent_specificity")
    result = {}
    for j, metric in enumerate(metrics):
        effect = per_seed[:, j]
        result[metric] = {"mean_difference": float(effect.mean()),
                          "clip_bootstrap_95ci": np.quantile(sampled[:, j], [.025, .975]).tolist(),
                          "per_seed_differences": effect.tolist(),
                          "between_seed_sample_std": float(effect.std(ddof=1)) if nseed > 1 else 0.}
    result["bootstrap"] = {"resamples": resamples, "valid_resamples": int(valid.sum()),
                           "seed": seed, "independent_clips": nclip,
                           "paired_training_seeds": nseed,
                           "unit": "paired whole clip, same draw shared across training seeds",
                           "interpretation": "descriptive clip interval conditional on the three trained seeds; seeds are not independent videos"}
    return result


def primary_verdict(contrast):
    utility = contrast["balanced_utility"]
    visible_guard = contrast["visible_localization"]["mean_difference"] >= -.01 - 1e-12
    absence_guard = contrast["absent_specificity"]["mean_difference"] >= -.01 - 1e-12
    positive = utility["mean_difference"] > 0 and utility["clip_bootstrap_95ci"][0] > 0
    return {"positive_mean_and_interval_excluding_zero": bool(positive),
            "visible_loss_no_more_than_one_percentage_point": bool(visible_guard),
            "absent_specificity_loss_no_more_than_one_percentage_point": bool(absence_guard),
            "predictive_addition_supported_by_prespecified_rule": bool(positive and visible_guard and absence_guard),
            "guard_type": "mean effect across predetermined training seeds; not a confidence-bound noninferiority claim"}


def contrasts(rows):
    names = [s["name"] for s in data.scene_specs("test")]
    groups = {a: [f"{a}_seed{s}" for s in SEEDS] for a in ARMS}
    groups.update({b: [b] * len(SEEDS) for b in BASELINES})
    counts = {m: _counts(rows, group, names) for m, group in groups.items()}
    pairs = [("predictive", "retrieval_only"), ("predictive", "coordinate_only"),
             ("retrieval_only", "coordinate_only")]
    pairs += [(a, b) for a in ("retrieval_only", "predictive") for b in BASELINES]
    return {a + "_minus_" + b: paired_mean_contrast(counts[a], counts[b]) for a, b in pairs}


def summarize_future(run, report, expected):
    """Summarize saved diagnostics, never recompute frozen features or targets."""
    with (Path(run) / "test/future_rows.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    methods = [f"{a}_seed{s}" for a in ("retrieval_only", "predictive") for s in SEEDS]
    if set(r["method"] for r in rows) != set(methods):
        raise AssertionError("Future rows must contain all six visual models and no coordinate model")
    metrics = ("forecast_cosine", "anchor_copy_cosine", "last_retrieved_copy_cosine")
    identities = ("forecast", "anchor", "last_retrieved")
    summaries = {}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        seen = set()
        for row in subset:
            key = (row["scene"], int(row["forecast_step"]), int(row["target_step"]), row["target"])
            if key in seen or key[0] not in expected or key[1:] not in expected[key[0]]:
                raise AssertionError("Invalid, repeated or nonvisible future diagnostic target")
            seen.add(key)
            if int(row["visible_future_candidates"]) != expected[key[0]][key[1:]]:
                raise AssertionError("Future candidate count differs from scoring labels")
            for metric in metrics:
                value = float(row[metric])
                if not np.isfinite(value) or not -1.00001 <= value <= 1.00001:
                    raise AssertionError("Invalid future cosine")
            for identity in identities:
                _bool(row[identity + "_own_future_part_hit"])
        if len(seen) != sum(len(v) for v in expected.values()):
            raise AssertionError("Missing future diagnostic rows")
        summary = {"targets": len(subset),
                   "forecast_mean_cosine": float(np.mean([float(r[metrics[0]]) for r in subset])),
                   "anchor_copy_mean_cosine": float(np.mean([float(r[metrics[1]]) for r in subset])),
                   "last_retrieved_copy_mean_cosine": float(np.mean([float(r[metrics[2]]) for r in subset])),
                   "own_future_part_retrieval": {}}
        for identity in identities:
            hit_key = identity + "_own_future_part_hit"
            multiple = [r for r in subset if int(r["visible_future_candidates"]) >= 2]
            summary["own_future_part_retrieval"][identity] = {
                "all_visible_targets": {"hits": sum(_bool(r[hit_key]) for r in subset), "targets": len(subset)},
                "at_least_two_candidates": {"hits": sum(_bool(r[hit_key]) for r in multiple), "targets": len(multiple)}}
        _equal(report["methods"][method]["future"], summary, method + ".future")
        summaries[method] = summary
    pooled = {}
    for arm in ("retrieval_only", "predictive"):
        records = [summaries[f"{arm}_seed{s}"] for s in SEEDS]
        pooled[arm] = {key: {"per_seed": [r[key] for r in records],
                            "mean": float(np.mean([r[key] for r in records])),
                            "between_seed_sample_std": float(np.std([r[key] for r in records], ddof=1))}
                       for key in ("forecast_mean_cosine", "anchor_copy_mean_cosine", "last_retrieved_copy_mean_cosine")}
    return {"source": "saved future_rows.csv only; no feature recomputation",
            "distinct_scored_targets_per_seed": sum(len(v) for v in expected.values()),
            "note": "Repeated training seeds share the same target videos. Cosine/latent identity diagnostics alone are not tracking success.",
            "methods": summaries, "seed_summary": pooled}


def seed_summary(methods):
    result = {}
    metrics = (LOCALIZATION, FALSE_PRESENCE, "wrong_car_given_visible", "recovery_accuracy")
    for arm in ARMS:
        records = [methods[f"{arm}_seed{s}"]["overall"] for s in SEEDS]
        summary = {}
        for metric in metrics:
            values = [r[metric]["rate"] for r in records]
            summary[metric] = {"values": values, "mean": float(np.mean(values)),
                               "between_seed_sample_std": float(np.std(values, ddof=1))}
        values = [.5 * (r[LOCALIZATION]["rate"] + 1 - r[FALSE_PRESENCE]["rate"]) for r in records]
        summary["balanced_utility"] = {"values": values, "mean": float(np.mean(values)),
                                       "between_seed_sample_std": float(np.std(values, ddof=1))}
        summary["identity_switches"] = {"values": [r["identity_switches"] for r in records],
                                        "mean": float(np.mean([r["identity_switches"] for r in records]))}
        result[arm] = summary
    return result


def make_plots(run, out, summary, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"retrieval_only": "#2b74ba", "predictive": "#ba3d65", "coordinate_only": "#8d7144"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm in ARMS:
        curves = []
        for seed in SEEDS:
            with (Path(run) / "models" / arm / str(seed) / "curves.csv").open(newline="") as stream:
                curves.append(list(csv.DictReader(stream)))
        epochs = [int(r["epoch"]) for r in curves[0]]
        if any([int(r["epoch"]) for r in c] != epochs for c in curves[1:]):
            raise AssertionError("Learning curves have different epoch grids")
        for ax, key, label, multiplier in zip(axes.flat,
            ("train_retrieval_loss", "train_auxiliary_loss", "dev_utility", "dev_visible_localization"),
            ("Training retrieval loss", "Training future objective (before coefficient)", "Development balanced utility (%)", "Development visible localization (%)"),
            (1, 1, 100, 100)):
            if key == "train_auxiliary_loss" and arm != "predictive":
                # Other arms record zero placeholders because this objective is
                # not evaluated; those zeros are not measured future errors.
                continue
            values = np.array([[float(r[key]) for r in c] for c in curves]) * multiplier
            ax.plot(epochs, values.mean(0), label=arm, color=colors[arm])
            ax.fill_between(epochs, values.min(0), values.max(0), color=colors[arm], alpha=.13)
            ax.set(xlabel="Epoch", ylabel=label); ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].text(.02, .98, "Predictive arm only; other arms not measured",
                    transform=axes[0, 1].transAxes, va="top", fontsize=9)
    fig.suptitle("Day7 learning curves: mean and range over three training seeds")
    fig.savefig(Path(out) / "learning_curves.png", dpi=170)
    plt.close(fig)
    labels = ["Retrieval", "Predictive", "Coordinates", "Cosine full", "Day4 fixed", "Cosine projected"]
    groups = list(ARMS) + list(BASELINES)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ("balanced_utility", LOCALIZATION, FALSE_PRESENCE),
                                ("Balanced utility ↑", "Visible localization ↑", "False presence while absent ↓")):
        values, error = [], []
        for group in groups:
            if group in ARMS:
                values.append(summary[group][metric]["mean"] * 100)
                error.append(summary[group][metric]["between_seed_sample_std"] * 100)
            else:
                r = methods[group]["overall"]
                value = .5 * (r[LOCALIZATION]["rate"] + 1 - r[FALSE_PRESENCE]["rate"]) if metric == "balanced_utility" else r[metric]["rate"]
                values.append(value * 100); error.append(0)
        ax.bar(np.arange(len(groups)), values, yerr=error, capsize=3,
               color=[colors.get(g, "#87939b") for g in groups])
        ax.set_xticks(np.arange(len(groups)), labels, rotation=35, ha="right")
        ax.set(title=title, ylabel="Percent", ylim=(0, 100)); ax.grid(axis="y", alpha=.2)
    fig.suptitle("18 held-out clips; learned bars average three seeds, error bars show seed SD")
    fig.savefig(Path(out) / "test_summary.png", dpi=170)
    plt.close(fig)


def make_fixed_video(run, out, freeze):
    spec = next(s for s in data.scene_specs("test") if s["seed"] == 10200 and s["condition"] == "crossing")
    scene = data.generate_scene(**spec)
    methods = ("global_projected_centered", "retrieval_only_seed1701", "predictive_seed1701", "coordinate_only_seed1701")
    predictions = {m: _load_prediction(Path(run) / "test/predictions" / m / f"{scene.name}.npz",
                                      learned=m not in BASELINES) for m in methods}
    thresholds = {m: freeze["baselines"][m]["threshold"] if m in BASELINES else 0. for m in methods}
    path = Path(out) / "test_crossing_10200_fixed_comparison.mp4"
    bench.annotated_video(scene, predictions, thresholds, path)
    return {"path": path.name, "scene": spec, "methods": methods,
            "selection": "first predeclared crossing test seed and training seed1701, fixed before results; no best-case selection"}


def _pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def write_markdown(path, report):
    p = report["primary_contrast"]
    lines = ["# Day7 independent audit and predictive-supervision contrast", "",
             "All saved prediction arrays were independently scored and matched the runner's per-target CSV and aggregate results.", "",
             "The primary contrast averages predictive minus retrieval-only utility over the three predetermined training seeds. Every bootstrap draw samples the same18 whole clips across all seeds. These are18 independent clips, not54 videos.", "",
             "| Primary difference | Mean effect (pp) | Paired clip95% interval (pp) | Seed sample SD (pp) |",
             "|---|---:|---:|---:|"]
    for name in ("balanced_utility", "visible_localization", "absent_specificity"):
        metric = p[name]; lo, hi = metric["clip_bootstrap_95ci"]
        lines.append(f"| {name} | {100 * metric['mean_difference']:+.3f} | [{100 * lo:+.3f}, {100 * hi:+.3f}] | {100 * metric['between_seed_sample_std']:.3f} |")
    lines += ["", "| Training seed | Utility difference (pp) | Visible difference (pp) | Absent-specificity difference (pp) |",
              "|---|---:|---:|---:|"]
    for i, seed in enumerate(SEEDS):
        values = [p[k]["per_seed_differences"][i] for k in ("balanced_utility", "visible_localization", "absent_specificity")]
        lines.append(f"| {seed} | " + " | ".join(f"{100 * v:+.3f}" for v in values) + " |")
    verdict = report["primary_decision"]
    lines += ["", "Prespecified predictive-addition evidence rule: **" + ("passed" if verdict["predictive_addition_supported_by_prespecified_rule"] else "not passed") + "**.",
              "The rule requires positive mean utility with an interval excluding zero, plus no more than1pp mean loss in visible localization or absent specificity. These are point-estimate guards, not formal noninferiority confidence bounds.", "",
              "| Method | Visible localization | False presence | Wrong car | Immediate recovery | ID switches |",
              "|---|---:|---:|---:|---:|---:|"]
    for method, value in report["methods"].items():
        r = value["overall"]
        parts = [_pct(r[k]["rate"]) for k in (LOCALIZATION, FALSE_PRESENCE, "wrong_car_given_visible", "recovery_accuracy")]
        lines.append("| " + method + " | " + " | ".join(parts) + f" | {r['identity_switches']} |")
    lines += ["", "Future-target cosine averaged over three training seeds:", "",
              "| Arm | Learned forecast | Immutable-anchor copy | Last-retrieved copy |",
              "|---|---:|---:|---:|"]
    for arm, values in report["future_diagnostics"]["seed_summary"].items():
        cells = [f"{values[k]['mean']:.4f} ± {values[k]['between_seed_sample_std']:.4f}"
                 for k in ("forecast_mean_cosine", "anchor_copy_mean_cosine", "last_retrieved_copy_mean_cosine")]
        lines.append("| " + arm + " | " + " | ".join(cells) + " |")
    lines += ["", "Full per-seed effects, sample standard deviations, condition breakdowns, coordinates/frozen-baseline contrasts and saved future-latent diagnostics are in independent_analysis.json.", "",
              "Future-latent diagnostics compare the learned forecast with copying the immutable anchor and last retrieved representation. They summarize saved CSV values, not an independent recomputation of the feature tensors. Low latent error alone does not establish correct tracking.", "",
              "The video always shows crossing seed10200 and training seed1701 across the same four fixed methods. It is not selected by outcome.", "",
              "This is supervised prediction over frozen JEPA features in the same2-D procedural family. Encoder attention is offline within16-frame blocks. The result does not establish real-video/face identity, generative editing, 3-D reasoning, or self-supervised discovery."]
    Path(path).write_text("\n".join(lines) + "\n")


def analyze(run, output=None, video=True, plots=True):
    run = Path(run)
    report, freeze, checksums, checkpoint_status = validate_completed_run(run)
    rows, methods, expected_future = recount_predictions(run, report, freeze)
    comparisons = contrasts(rows)
    primary = comparisons["predictive_minus_retrieval_only"]
    future = summarize_future(run, report, expected_future)
    seeds = seed_summary(methods)
    out = Path(output) if output else run / "test/analysis"
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "passed", "audit_source_sha256": sha256(__file__),
              "source_files": checksums, "checkpoint_files": checkpoint_status,
              "independent_prediction_rows": len(rows), "independent_clips": 18,
              "training_seeds": SEEDS, "methods": methods, "seed_summary": seeds,
              "primary_contrast": primary, "primary_decision": primary_verdict(primary),
              "comparisons": comparisons, "future_diagnostics": future,
              "audited": "Saved predictions recomputed with frozen Day4 scorer; every CSV row and method/condition result matched; future CSV aggregate bookkeeping matched",
              "not_audited": "No model inference or future feature/cosine recomputation; missing checkpoints, if any, are explicitly listed",
              "scope": "Supervised head over frozen JEPA features on18 same-family2-D clips; three trained seeds are not54 independent videos"}
    if plots:
        make_plots(run, out, seeds, methods)
        result["plots"] = ["learning_curves.png", "test_summary.png"]
    if video:
        result["fixed_comparison_video"] = make_fixed_video(run, out, freeze)
    _json(out / "independent_analysis.json", result)
    write_markdown(out / "INDEPENDENT_ANALYSIS.md", result)
    print(json.dumps({"status": "passed", "prediction_rows": len(rows), "independent_clips": 18,
                      "primary_decision": result["primary_decision"], "output": str(out)}, indent=2))
    return result


def self_test():
    """Fabricated count fixtures only; never generate a held-out scene."""
    base = np.zeros((3, 18, 4), np.int64)
    base[:] = [50, 100, 20, 100]
    improved = base.copy(); improved[..., 0] += 20
    contrast = paired_mean_contrast(improved, base)
    assert np.isclose(contrast["balanced_utility"]["mean_difference"], .1)
    assert np.allclose(contrast["balanced_utility"]["clip_bootstrap_95ci"], [.1, .1])
    assert primary_verdict(contrast)["predictive_addition_supported_by_prespecified_rule"]
    # Equal/opposite seed-by-clip effects must cancel under every SHARED clip draw.
    paired = base.copy(); pattern = np.tile([10, -10], 9)
    paired[0, :, 0] += pattern; paired[1, :, 0] -= pattern
    contrast = paired_mean_contrast(paired, base)
    assert np.allclose(contrast["balanced_utility"]["clip_bootstrap_95ci"], [0, 0], atol=1e-14)
    assert not primary_verdict(contrast)["predictive_addition_supported_by_prespecified_rule"]
    tradeoff = base.copy(); tradeoff[..., 0] -= 2; tradeoff[..., 2] -= 8
    contrast = paired_mean_contrast(tradeoff, base)
    verdict = primary_verdict(contrast)
    assert verdict["positive_mean_and_interval_excluding_zero"]
    assert not verdict["visible_loss_no_more_than_one_percentage_point"]
    assert not verdict["predictive_addition_supported_by_prespecified_rule"]
    bad = base.copy(); bad[..., 1] += 1
    try:
        paired_mean_contrast(bad, base)
    except ValueError:
        pass
    else:
        raise AssertionError("Denominator mismatch was accepted")
    valid_binding = {"clips": [{**spec, "split": "test"} for spec in data.scene_specs("test")]}
    assert len(_test_binding_records(valid_binding)) == 18
    duplicate = {"clips": [dict(v) for v in valid_binding["clips"]]}
    duplicate["clips"][-1] = dict(duplicate["clips"][0])
    wrong_seed = {"clips": [dict(v) for v in valid_binding["clips"]]}
    wrong_seed["clips"][0]["seed"] += 1
    for invalid in (duplicate, wrong_seed):
        try:
            _test_binding_records(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid test binding was accepted")
    return {"status": "passed", "fixtures": ["constant paired effect", "shared-draw seed cancellation", "one-pp guard", "denominator mismatch rejection", "duplicate/incorrect test metadata rejection"],
            "test_scenes_rendered": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.run:
        analyze(args.run, args.out, not args.no_video, not args.no_plots)
    else:
        parser.error("Specify --self-test or --run with a completed test output directory")
