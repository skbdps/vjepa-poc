"""Post-completed-test diagnostics of frozen Day7 mechanism reliance.

No training, checkpoint selection, or primary prediction overwrite occurs.
Forecast intervention replaces queued forecasts by immutable anchors from t=8.
Owner intervention passes unique owner IDs to the original model, removing all
sibling context. These are distribution-shifting inference interventions, not
retrained architectural ablations or tests of the auxiliary objective's benefit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import train
import analyze
from model import PersistentPartJEPA, _unit

INTERVENTIONS = ("forecast_to_anchor", "owner_context_zero")
METRICS = ("localization_accuracy_given_visible", "false_presence_given_absent",
           "recovery_accuracy", "wrong_car_given_visible")


def replay_forward(model, features, initial_weights, owner_ids, replace_forecast=False):
    """Explicit frozen-forward copy; normal mode must exactly match native code.

    The only intervention is the ``prior`` assignment below. Existing layers,
    parameters, prediction heads, and recurrent updates are reused unmodified.
    Normal-parity checks guard against this copy drifting from frozen model.py.
    """
    features, anchors, owner_ids = model._prepare(features, initial_weights, owner_ids)
    b, steps, _, _ = features.shape
    parts = anchors.shape[1]
    states = torch.tanh(model.anchor_state(anchors))
    futures, state_history, logits_history = [], [], []
    visibility_history, attention_history, prior_history = [], [], []
    scale = model.logit_scale.exp().clamp(1., 50.)
    for t in range(steps):
        tokens = features[:, t]
        unit_tokens = _unit(tokens)
        keys = _unit(unit_tokens+model.key_residual(unit_tokens))
        owner = model._sibling_context(states, owner_ids)
        prior = (futures[t-model.horizon]
                 if t >= model.horizon and not replace_forecast else anchors)
        context = torch.cat((anchors, states, owner, prior), dim=-1)
        query = _unit(anchors+model.forecast_weight*prior+model.query_residual(context))
        spatial_logits = scale*torch.einsum("bkd,bnd->bkn", query, keys)
        attention = spatial_logits.softmax(-1)
        retrieved = torch.einsum("bkn,bnd->bkd", attention, unit_tokens)
        similarities = spatial_logits/scale
        statistics = torch.stack((similarities.max(-1).values, similarities.mean(-1),
                                  similarities.std(-1, unbiased=False)), dim=-1)
        absence_log_odds = model.absence(torch.cat((context, retrieved, statistics), -1)).squeeze(-1)
        null_logit = torch.logsumexp(spatial_logits, dim=-1)+absence_log_odds
        logits = torch.cat((spatial_logits, null_logit[..., None]), dim=-1)
        visibility = torch.sigmoid(-absence_log_odds)
        update = model.update_input(torch.cat((retrieved, anchors, owner), dim=-1))
        proposed = model.memory(update.reshape(b*parts, -1), states.reshape(b*parts, -1))
        proposed = proposed.reshape(b, parts, model.hidden_dim)
        states = states+visibility[..., None]*(proposed-states)
        updated_owner = model._sibling_context(states, owner_ids)
        future = _unit(model.future_predictor(torch.cat((anchors, states, updated_owner), dim=-1)))
        futures.append(future); state_history.append(states); logits_history.append(logits)
        visibility_history.append(visibility); attention_history.append(attention); prior_history.append(prior)
    return {"logits": torch.stack(logits_history, 1), "future": torch.stack(futures, 1),
            "states": torch.stack(state_history, 1), "visibility": torch.stack(visibility_history, 1),
            "attention": torch.stack(attention_history, 1), "forecast_used": torch.stack(prior_history, 1),
            "anchors": anchors}


def _exact_parity(native, replay):
    if set(native) != set(replay):
        raise AssertionError("Native and replay output fields differ")
    for name in native:
        if not torch.equal(native[name], replay[name]):
            error = float((native[name]-replay[name]).abs().max().cpu())
            raise AssertionError(f"Normal replay differs from native {name}: max error {error}")


def _prefix_parity(native, intervention, horizon):
    for name in ("logits", "future", "states", "visibility", "attention", "forecast_used"):
        if not torch.equal(native[name][:, :horizon], intervention[name][:, :horizon]):
            raise AssertionError(f"Forecast intervention changed pre-horizon {name}")
    if not torch.equal(native["anchors"], intervention["anchors"]):
        raise AssertionError("Immutable anchors changed")


def _decode(output):
    return train.predictions_from_logits(output["logits"][0].cpu().numpy(),
                                         output["visibility"][0].cpu().numpy())


def _archive_parity(native, prediction, archive_path):
    errors = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for name in ("logits", "future", "visibility"):
            current = native[name][0].cpu().numpy()
            if name not in archive or current.shape != archive[name].shape:
                raise ValueError(f"Missing or malformed saved {name}: {archive_path}")
            errors[name+"_max_abs"] = float(np.max(np.abs(current-archive[name])))
            if not np.allclose(current, archive[name], atol=2e-5, rtol=2e-5):
                raise AssertionError(f"Native output does not reproduce archived {name}")
        for name in ("cells", "eligible"):
            if not np.array_equal(getattr(prediction, name), archive[name]):
                raise AssertionError(f"Native decoded {name} differs from primary predictions")
        if not np.allclose(prediction.scores, archive["scores"], atol=2e-6, rtol=2e-6):
            raise AssertionError("Native scores differ from primary predictions")
    return errors


def _sensitivity(native, changed, base_pred, new_pred, sample):
    scored = sample["target"]["state"][1:] >= 0
    both_present = base_pred.eligible[1:] & new_pred.eligible[1:]
    cell_change = np.any(base_pred.cells[1:] != new_pred.cells[1:], axis=-1)
    presence_change = base_pred.eligible[1:] != new_pred.eligible[1:]
    emitted_change = presence_change | (both_present & cell_change)
    difference = changed["logits"][:, 1:]-native["logits"][:, 1:]
    states = (changed["states"][:, 1:]-native["states"][:, 1:]).norm(dim=-1)
    future_distance = 1-(changed["future"][:, 1:]*native["future"][:, 1:]).sum(-1).clamp(-1, 1)
    return {"scorable_part_pairs": int(scored.sum()),
        "emitted_prediction_changes": int(emitted_change[scored].sum()),
        "presence_changes": int(presence_change[scored].sum()),
        "conditional_patch_changes": int(cell_change[scored].sum()),
        "patch_changes_when_both_present": int((cell_change & both_present & scored).sum()),
        "mean_abs_logit_change": float(difference.abs().mean().cpu()),
        "max_abs_logit_change": float(difference.abs().max().cpu()),
        "mean_state_l2_change": float(states.mean().cpu()),
        "max_state_l2_change": float(states.max().cpu()),
        "mean_abs_visibility_change": float((changed["visibility"][:, 1:]-native["visibility"][:, 1:]).abs().mean().cpu()),
        "mean_future_cosine_distance": float(future_distance.mean().cpu())}


def _gate_completed_run(run, features):
    """All checks here precede the first test-label loader."""
    run = Path(run)
    # Reuse the read-only audit's exhaustive completion/provenance guard. It
    # requires all baseline/model prediction archives before any test rendering.
    analyze.validate_completed_run(run)
    for name in ("test/results.json", "test/prediction_rows.csv", "test/RESULTS.md", "checkpoint_freeze.json"):
        if not (run/name).is_file():
            raise RuntimeError(f"Completed primary test required: missing {name}")
    report = json.loads((run/"test/results.json").read_text())
    frozen = json.loads((run/"checkpoint_freeze.json").read_text())
    if report.get("split") != "test" or report.get("all_seeds_reported") is not True or report.get("test_selection") != "none":
        raise RuntimeError("Primary report is not a completed all-seed test")
    if report["checkpoint_freeze_sha256"] != train.sha256(run/"checkpoint_freeze.json"):
        raise RuntimeError("Primary test does not bind this checkpoint freeze")
    config = frozen["training_config"]
    if config["source_digest"] != train.source_digest() or config["source_hashes"] != train.source_hashes():
        raise RuntimeError("Frozen training/model source changed")
    if config["feature_binding"] != train.feature_binding(features):
        raise RuntimeError("Train/dev features differ from the checkpoint freeze")
    if report["test_feature_binding"] != train.feature_binding(features, ("test",)):
        raise RuntimeError("Test features differ from completed primary evaluation")
    expected = {(arm, seed) for arm in train.ARMS for seed in train.SEEDS}
    if len(frozen["models"]) != len(expected) or {(r["arm"], r["seed"]) for r in frozen["models"]} != expected:
        raise RuntimeError("Checkpoint freeze does not retain all predetermined arms/seeds")
    for record in frozen["models"]:
        method = f"{record['arm']}_seed{record['seed']}"
        if method not in report["methods"]:
            raise RuntimeError(f"Primary test missing {method}")
        if train.sha256(run/record["path"]) != record["checkpoint_sha256"]:
            raise RuntimeError("Frozen checkpoint checksum changed")
        if train.sha256((run/record["path"]).with_name("selected.json")) != record["selection_sha256"]:
            raise RuntimeError("Development checkpoint selection changed")
    return report, frozen


def _paired(a, b):
    result = {}
    for metric in METRICS:
        da, db = train.bench.summarize_rows(a)[metric]["denominator"], train.bench.summarize_rows(b)[metric]["denominator"]
        result[metric] = (train.bench.paired_bootstrap_difference(a, b, metric=metric) if da and db else
                          {"metric": metric, "difference_a_minus_b": None, "reason": "zero denominator"})
    return result


def _assert_primary_summary(actual, expected):
    for metric in METRICS+("wrong_part_given_visible",):
        for field in ("numerator", "denominator"):
            if actual[metric][field] != expected[metric][field]:
                raise AssertionError(f"Native replay disagrees with completed primary metric {metric}/{field}")
    if actual["identity_switches"] != expected["identity_switches"]:
        raise AssertionError("Native replay identity-switch count changed")


def _aggregate_sensitivity(entries):
    keys = ("scorable_part_pairs", "emitted_prediction_changes", "presence_changes",
            "conditional_patch_changes", "patch_changes_when_both_present")
    result = {key: sum(entry[key] for entry in entries) for key in keys}
    for key in ("mean_abs_logit_change", "mean_state_l2_change",
                "mean_abs_visibility_change", "mean_future_cosine_distance"):
        result[key] = float(np.mean([entry[key] for entry in entries]))
    for key in ("max_abs_logit_change", "max_state_l2_change"):
        result[key] = max(entry[key] for entry in entries)
    denominator = result["scorable_part_pairs"]
    result["emitted_prediction_change_rate"] = result["emitted_prediction_changes"]/denominator if denominator else None
    result["presence_change_rate"] = result["presence_changes"]/denominator if denominator else None
    return result


def run_diagnostics(run, features, device="cuda"):
    run, features = Path(run), Path(features)
    primary, frozen = _gate_completed_run(run, features)
    # The loader is allowed only after the completed-test/source/checkpoint gate.
    samples = train.load_split(features, "test")
    out = run/"diagnostics"/"mechanism_reliance_v1"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Diagnostic output already exists; preserve this analysis instead of overwriting")
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = [r for r in frozen["models"] if r["arm"] in ("retrieval_only", "predictive")]
    if len(checkpoints) != 6:
        raise RuntimeError("Exactly six visual checkpoints are required")
    result, per_clip = {}, []
    started = time.perf_counter()
    for record in checkpoints:
        method = f"{record['arm']}_seed{record['seed']}"
        model = PersistentPartJEPA(256, 128, 8).to(device).float().eval().requires_grad_(False)
        saved = torch.load(run/record["path"], map_location=device, weights_only=True)
        if (saved["arm"], saved["seed"], saved["epoch"]) != (record["arm"], record["seed"], record["epoch"]):
            raise RuntimeError("Checkpoint identity differs from its frozen record")
        model.load_state_dict(saved["state_dict"])
        rows = {"normal": [], **{name: [] for name in INTERVENTIONS}}
        for sample in samples:
            features_tensor = torch.from_numpy(sample["features"])[None].to(device)
            initial = torch.from_numpy(sample["initial_weights"])[None].to(device)
            owners = torch.tensor(train.OWNER_IDS, device=device)
            with torch.inference_mode():
                normal = model(features_tensor, initial, owners)
                replay = replay_forward(model, features_tensor, initial, owners)
                _exact_parity(normal, replay)
                base_pred = _decode(normal)
                archive = run/"test"/"predictions"/method/f"{sample['name']}.npz"
                parity = _archive_parity(normal, base_pred, archive)
                changed = {
                    "forecast_to_anchor": replay_forward(model, features_tensor, initial, owners, True),
                    "owner_context_zero": model(features_tensor, initial, torch.arange(initial.shape[1], device=device))}
                _prefix_parity(normal, changed["forecast_to_anchor"], model.horizon)
                base_rows = train.score_prediction(sample, base_pred, method)
                rows["normal"].extend(base_rows)
                base_summary = train.bench.summarize_rows(base_rows)
                for intervention, output in changed.items():
                    prediction = _decode(output)
                    current_rows = train.score_prediction(sample, prediction, method+"::"+intervention)
                    rows[intervention].extend(current_rows)
                    summary = train.bench.summarize_rows(current_rows)
                    entry = {"method": method, "arm": record["arm"], "training_seed": record["seed"],
                             "scene": sample["name"], "condition": sample["condition"], "intervention": intervention,
                             "normal_replay_exact": True, "forecast_prefix_exact": True if intervention == "forecast_to_anchor" else None,
                             **parity, **_sensitivity(normal, output, base_pred, prediction, sample)}
                    for metric in METRICS:
                        for prefix, value in (("normal", base_summary), ("intervention", summary)):
                            for field in ("numerator", "denominator", "rate"):
                                entry[f"{prefix}_{metric}_{field}"] = value[metric][field]
                    per_clip.append(entry)
        base_summary = train.bench.summarize_rows(rows["normal"])
        _assert_primary_summary(base_summary, primary["methods"][method]["overall"])
        result[method] = {"checkpoint_sha256": record["checkpoint_sha256"], "normal": base_summary,
            "interventions": {name: {"overall": train.bench.summarize_rows(rows[name]),
                "paired_change_intervention_minus_normal": _paired(rows[name], rows["normal"]),
                "sensitivity": _aggregate_sensitivity([entry for entry in per_clip
                    if entry["method"] == method and entry["intervention"] == name]),
                "by_condition": {condition: train.bench.summarize_rows([r for r in rows[name] if r["condition"] == condition])
                                 for condition in train.data.CONDITIONS}}
                              for name in INTERVENTIONS}}
        print(f"DAY7_DIAGNOSTIC_COMPLETE {method}", flush=True)
        del model
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "Post-completed-test checkpoint reliance diagnostics; no new selected method",
        "diagnostic_source_sha256": train.sha256(__file__), "frozen_source_hashes": train.source_hashes(),
        "completion_guard_source_sha256": train.sha256(HERE/"analyze.py"),
        "primary_results_sha256": train.sha256(run/"test/results.json"),
        "primary_prediction_rows_sha256": train.sha256(run/"test/prediction_rows.csv"),
        "checkpoint_freeze_sha256": train.sha256(run/"checkpoint_freeze.json"),
        "torch": torch.__version__, "device": str(device), "seconds": time.perf_counter()-started,
        "all_six_visual_checkpoints_reported": True, "training_or_selection_performed": False,
        "primary_predictions_overwritten": False, "methods": result,
        "sensitivity_rows": per_clip,
        "numeric_sensitivity_scope": "All part-pairs after t0, including ambiguous ones; prediction-change counts use only scorable pairs.",
        "difference_sign": "Intervention minus normal: negative localization/recovery means removal harmed performance; negative false presence means removal improved it.",
        "interpretation": [
            "Interventions measure reliance of fixed checkpoints and can shift their input/state distribution.",
            "Performance loss is not evidence that this architecture is superior to a retrained alternative.",
            "No performance loss may reflect redundancy or cancellation; it does not establish that auxiliary prediction never helped optimization.",
            "Forecast replacement has downstream recurrent effects from step8; owner removal changes context, memory, and later forecasts.",
            "No hyperparameter, epoch, seed, or method is selected from these diagnostics."]}
    train.write_json(out/"reliance_results.json", report)
    train._write_csv(out/"per_clip_reliance.csv", per_clip)
    return report


def self_test():
    """CPU fabricated-input parity only; no checkpoint or held-out data access."""
    torch.manual_seed(733)
    model = PersistentPartJEPA(16, 8, 3).eval()
    features, weights = torch.randn(2, 7, 11, 16), torch.rand(2, 4, 11)
    owners = torch.tensor([0, 0, 1, 1])
    with torch.inference_mode():
        normal = model(features, weights, owners)
        replay = replay_forward(model, features, weights, owners)
        _exact_parity(normal, replay)
        anchor = replay_forward(model, features, weights, owners, True)
        _prefix_parity(normal, anchor, 3)
        assert not torch.equal(normal["logits"][:, 3:], anchor["logits"][:, 3:])
        unique = torch.arange(4).expand(2, -1)
        assert torch.count_nonzero(model._sibling_context(normal["states"][:, 0], unique)) == 0
        without_owner = model(features, weights, unique)
        assert not torch.equal(normal["logits"], without_owner["logits"])
    return {"normal_replay_exact": True, "forecast_prefix_exact": True,
            "forecast_intervention_changes_later_retrieval": True, "unique_owners_zero_context": True,
            "owner_intervention_changes_retrieval": True, "held_out_data_accessed": False,
            "diagnostic_source_sha256": train.sha256(__file__), "torch": torch.__version__}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run"); parser.add_argument("--features")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.run and args.features:
        report = run_diagnostics(args.run, args.features, args.device)
        print(json.dumps({"checkpoints": list(report["methods"]), "seconds": report["seconds"]}, indent=2))
    else:
        parser.error("Provide --run and --features, or --self-test")
