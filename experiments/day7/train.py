"""Train and freeze supervised persistent-part heads over frozen JEPA tokens.

Only training labels produce gradients. Development selects an epoch/absence
threshold; all three predetermined seeds are retained. Test requires a committed
checkpoint freeze supplied by the caller, and never selects a model or policy.
The model input boundary is features, frame-zero prompt weights and sibling IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
DAY4 = HERE.parent / "day4"
for folder in (DAY4, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import benchmark as bench
import data
from model import PersistentPartJEPA

ARMS = ("retrieval_only", "predictive", "coordinate_only")
SEEDS = (1701, 1702, 1703)
BASELINES = ("global_full", "selected_day4", "global_projected_centered")
OWNER_IDS = [0, 0, 1, 1]
POLICY = {
    "version": "day7_supervised_jepa_head_v1", "epochs": 30,
    "batch_scenes": 2, "optimizer": "AdamW", "learning_rate": 3e-4,
    "weight_decay": 1e-4, "precision": "float32", "seeds": list(SEEDS),
    "input_dim": 256, "hidden_dim": 128, "horizon": 8,
    "future_lambda": {"retrieval_only": 0., "predictive": 1., "coordinate_only": 0.},
    "infonce_weight": .2, "infonce_temperature": .1,
    "retrieval_loss": "batchwise mean of available visible multi-positive CE and absent null CE class means; equal if both present; exclude t0 and ambiguous",
    "future_target": "stop-gradient unit mean of centered-unit JEPA patch tokens on positive future GT cells",
    "future_validity": "future target visible at t+8; future absent/ambiguous ignored; t0 forecast included",
    "future_negatives": "other visible parts in the same scene and future step; one target yields zero NCE",
    "selection": "best dev mean(visible localization, absent specificity); earliest epoch on ties",
    "learned_presence": "factorized visibility>=0.5 then spatial argmax; same fixed cutoff for all arms",
    "baseline_threshold": "best dev utility on201score quantiles plus finite outside endpoints; higher threshold tie",
    "coordinate_control": "fixed 2D sine/cosine256position vectors, grid-centered and normalized, repeated every step; no RGB",
    "test_scope": "all nine arm/seed checkpoints retained; no test winner selection",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    paths = [HERE / name for name in ("train.py", "model.py", "data.py")] + [DAY4 / "benchmark.py"]
    return {str(p.relative_to(HERE.parent)): sha256(p) for p in paths}


def source_digest():
    return hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()


def unit_np(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def coordinate_tokens(steps=32, grid=24, dim=256):
    """Image-independent position control, in a fixed basis with no fitted state."""
    if dim % 4:
        raise ValueError("Coordinate dimension must be divisible by4")
    yy, xx = np.meshgrid(np.linspace(-1, 1, grid), np.linspace(-1, 1, grid), indexing="ij")
    freq = np.pi * np.power(2., np.linspace(0., 5., dim // 4))
    pieces = [np.sin(xx[..., None] * freq), np.cos(xx[..., None] * freq),
              np.sin(yy[..., None] * freq), np.cos(yy[..., None] * freq)]
    position = np.concatenate(pieces, axis=-1).reshape(grid * grid, dim).astype(np.float32)
    position = unit_np(position - position.mean(axis=0, keepdims=True))
    return np.broadcast_to(position[None], (steps, grid * grid, dim)).copy()


def _manifest(features):
    features = Path(features)
    manifest = json.loads((features / "manifest.json").read_text())
    if not isinstance(manifest.get("clips"), list):
        raise ValueError("Feature manifest needs a clips list")
    indexed = {r["name"]: r for r in manifest["clips"]}
    if len(indexed) != len(manifest["clips"]):
        raise ValueError("Duplicate feature-cache scene names")
    mean_info = manifest["mean"]
    if mean_info.get("fit_split") != "train":
        raise ValueError("Centering mean must be fit on training clips only")
    mean_path = features / mean_info["path"]
    if sha256(mean_path) != mean_info["sha256"]:
        raise ValueError("Centering mean checksum mismatch")
    mean = np.load(mean_path, allow_pickle=False).astype(np.float32)
    if mean.shape != (256,) or not np.isfinite(mean).all():
        raise ValueError("Expected finite256-dimensional training mean")
    if not isinstance(manifest.get("projection_sha256"), str):
        raise ValueError("Fixed projection checksum missing")
    return manifest, indexed, mean


def feature_binding(features, splits=("train", "dev")):
    """Bind train/dev files individually so adding later test caches is allowed."""
    manifest, indexed, _ = _manifest(features)
    clips = []
    for split in splits:
        for spec in data.scene_specs(split):
            record = indexed.get(spec["name"])
            if record is None or record.get("split") != split:
                raise ValueError(f"Missing {split} cache {spec['name']}")
            if any(record[k] != spec[k] for k in ("name", "seed", "condition")):
                raise ValueError("Feature metadata does not match split manifest")
            if not record.get("rgb_sha256"):
                raise ValueError("Per-clip RGB source hash required")
            path = Path(features) / record["path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"Cache checksum mismatch: {path}")
            clips.append({k: record[k] for k in ("name", "split", "seed", "condition", "path", "sha256", "rgb_sha256")})
    return {"clips": clips, "mean": manifest["mean"],
            "projection_sha256": manifest["projection_sha256"],
            "extractor_provenance": {k: manifest[k] for k in
                ("extractor_sources", "model", "upstream_revision", "encoder_context", "projection_seed", "projection_dim", "data")}}


def load_split(features, split):
    """Supervised/evaluation loader; never supply its target fields to the model."""
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    _, indexed, mean = _manifest(features)
    feature_binding(features, (split,))
    samples = []
    for spec in data.scene_specs(split):
        row = indexed[spec["name"]]
        scene = data.generate_scene(**spec)
        if hashlib.sha256(scene.frames.tobytes()).hexdigest() != row["rgb_sha256"]:
            raise ValueError("Rendered scene RGB does not match extracted features")
        target = data.supervision(scene.masks)
        with np.load(Path(features) / row["path"], allow_pickle=False) as archive:
            tokens = archive["tokens"].astype(np.float32)
            if tokens.shape != (32, 576, 256) or not np.isfinite(tokens).all():
                raise ValueError("Expected finite [32,576,256] projected tokens")
            baselines = {}
            for method, prefix in (("global_full", "global"), ("selected_day4", "selected")):
                scores, cells = archive[prefix + "_scores"], archive[prefix + "_cells"]
                eligible = archive[prefix + "_eligible"] if prefix + "_eligible" in archive else np.ones((32, 4), bool)
                pred = bench.Predictions(scores.copy(), cells.copy(), eligible.copy())
                bench.validate_predictions(pred, 32)
                if pred.eligible.dtype != bool:
                    raise ValueError("Baseline eligibility must be boolean")
                baselines[method] = pred
        features_unit = unit_np(tokens - mean)
        baselines["global_projected_centered"] = projected_global(features_unit, target["initial_weights"])
        samples.append({"name": spec["name"], "seed": spec["seed"], "condition": spec["condition"],
                        "features": features_unit, "initial_weights": target["initial_weights"],
                        "target": target, "baselines": baselines})
        del scene, tokens
    return samples


def projected_global(features, initial_weights):
    """No learned head; immutable frame-zero prototypes in the same256-D basis."""
    anchors = unit_np(np.einsum("kn,nd->kd", initial_weights, features[0]))
    similarities = np.einsum("tnd,kd->tkn", features, anchors)
    flat = similarities.argmax(axis=-1)
    scores = np.take_along_axis(similarities, flat[..., None], axis=-1)[..., 0]
    cells = np.stack((flat // 24, flat % 24), axis=-1).astype(np.int16)
    return bench.Predictions(scores.astype(np.float32), cells, np.ones_like(scores, bool))


def balanced_retrieval_loss(logits, positive, state, score_steps):
    """Multi-positive likelihood, absent null, equal visible/absent class weight."""
    log_prob = logits.log_softmax(-1)
    selected = score_steps[:, :, None]
    visible, absent = (state == 1) & selected, (state == 0) & selected
    terms = []
    counts = {"visible": int(visible.sum().item()), "absent": int(absent.sum().item())}
    if visible.any():
        value = log_prob[..., :-1][visible].masked_fill(~positive[visible], -torch.inf)
        terms.append(-torch.logsumexp(value, dim=-1).mean())
    if absent.any():
        terms.append(-log_prob[..., -1][absent].mean())
    if not terms:
        return logits.sum() * 0., counts
    return torch.stack(terms).mean(), counts


def future_targets(features, positive, state, horizon=8):
    """Training/scoring-only fixed teacher targets; no model call sees these."""
    weights = positive[:, horizon:].to(features.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
    targets = F.normalize(torch.einsum("btkn,btnd->btkd", weights, features[:, horizon:]), dim=-1, eps=1e-8)
    return targets.detach(), (state[:, horizon:] == 1)


def auxiliary_future_loss(predicted, targets, valid):
    predicted = F.normalize(predicted[:, :targets.shape[1]], dim=-1, eps=1e-8)
    if not valid.any():
        zero = predicted.sum() * 0.
        return zero, {"cosine_loss": 0., "infonce_loss": 0., "valid_targets": 0}
    cosine = 1 - (predicted * targets).sum(-1)
    cosine_loss = cosine[valid].mean()
    similarity = torch.einsum("btkd,btjd->btkj", predicted, targets) / POLICY["infonce_temperature"]
    similarity = similarity.masked_fill(~valid[:, :, None, :], -torch.inf)
    indices = torch.arange(valid.shape[-1], device=valid.device).expand_as(valid)
    # Select valid rows before cross-entropy; all-absent rows never create NaNs.
    nce = F.cross_entropy(similarity[valid], indices[valid])
    loss = cosine_loss + POLICY["infonce_weight"] * nce
    return loss, {"cosine_loss": float(cosine_loss.detach()), "infonce_loss": float(nce.detach()),
                  "valid_targets": int(valid.sum())}


def model_inference(model, features, initial_weights, owner_ids):
    """Explicit annotation barrier: no future truth, seed, condition or transforms."""
    return model(features, initial_weights, owner_ids)


def predictions_from_logits(logits, visibility=None):
    values = np.asarray(logits)
    if values.shape != (32, 4, 577) or not np.isfinite(values).all():
        raise ValueError("Expected finite [32,4,577] logits")
    flat = values[..., :-1].argmax(-1)
    shift = values - values.max(-1, keepdims=True)
    probs = np.exp(shift); probs /= probs.sum(-1, keepdims=True)
    scores = np.take_along_axis(probs[..., :-1], flat[..., None], axis=-1)[..., 0]
    # Sum the probability of ALL spatial cells. A joint577-way argmax would
    # incorrectly penalize a visible part whose mass spans multiple valid cells.
    present_probability = 1. - probs[..., -1]
    if visibility is not None:
        visibility = np.asarray(visibility)
        if visibility.shape != (32, 4) or not np.isfinite(visibility).all():
            raise ValueError("Expected finite factorized visibility [32,4]")
        if not np.allclose(visibility, present_probability, atol=2e-6, rtol=2e-6):
            raise ValueError("Visibility disagrees with factorized null probability")
        present_probability = visibility
    return bench.Predictions(scores.astype(np.float32), np.stack((flat // 24, flat % 24), -1).astype(np.int16), present_probability >= .5)


def score_prediction(sample, prediction, method, threshold=0.):
    """Same Day4 patch scoring conventions applied to the independent Day7 split."""
    target = sample["target"]; labels, states = target["patch_labels"], target["state"]
    rows = []
    for k, pid in enumerate(bench.PART_NAMES):
        pending, previous_car = False, (pid - 1) // 2
        for t in range(1, 32):
            state = int(states[t, k])
            if state == 0:
                pending = True
            recovery = state == 1 and pending
            if state == 1:
                pending = False
            row, col = map(int, prediction.cells[t, k])
            present = bool(prediction.eligible[t, k] and prediction.scores[t, k] >= threshold)
            actual = int(labels[t, row * 24 + col]) if present else -2
            wrong_car = state == 1 and present and actual > 0 and (actual - 1) // 2 != (pid - 1) // 2
            wrong_part = state == 1 and present and actual > 0 and (actual - 1) % 2 != (pid - 1) % 2
            switch = False
            if state == 1 and present and actual > 0:
                owner = (actual - 1) // 2
                switch, previous_car = owner != previous_car, owner
            rows.append({"scene": sample["name"], "seed": sample["seed"], "condition": sample["condition"],
                         "method": method, "tubelet": t, "first_frame": t * 2, "target": bench.PART_NAMES[pid],
                         "state": "visible" if state == 1 else "absent" if state == 0 else "ambiguous",
                         "window": t // 8 + 1, "boundary": t % 8 == 0, "recovery": bool(recovery),
                         "present": present, "score": float(prediction.scores[t, k]), "threshold": float(threshold),
                         "row": row, "col": col, "predicted_gt": actual,
                         "hit": bool(state == 1 and present and actual == pid),
                         "wrong_car": bool(wrong_car), "wrong_part": bool(wrong_part), "id_switch": bool(switch),
                         "chance": float((labels[t] == pid).mean())})
    return rows


def _utility(summary):
    loc = summary["localization_accuracy_given_visible"]["rate"]
    fp = summary["false_presence_given_absent"]["rate"]
    if loc is None or fp is None:
        raise ValueError("Development utility requires both visible and absent targets")
    return (loc + 1 - fp) / 2


def select_baseline_threshold(samples, method):
    scores = np.concatenate([s["baselines"][method].scores[1:].ravel() for s in samples])
    candidates = np.unique(np.r_[float(scores.min()) - 1e-5, np.quantile(scores, np.linspace(0, 1, 201)), float(scores.max()) + 1e-5])
    best = None
    for threshold in candidates:
        rows = [r for sample in samples for r in score_prediction(sample, sample["baselines"][method], method, threshold)]
        summary = bench.summarize_rows(rows); utility = _utility(summary)
        if best is None or (utility, float(threshold)) > (best["utility"], best["threshold"]):
            best = {"threshold": float(threshold), "utility": utility, "development": summary}
    return best


def _batch(samples, arm, device):
    features = torch.from_numpy(np.stack([s["features"] for s in samples])).to(device)
    inputs = (torch.from_numpy(np.stack([coordinate_tokens() for _ in samples])).to(device)
              if arm == "coordinate_only" else features)
    initial = torch.from_numpy(np.stack([s["initial_weights"] for s in samples])).to(device)
    positive = torch.from_numpy(np.stack([s["target"]["positive"] for s in samples])).to(device)
    states = torch.from_numpy(np.stack([s["target"]["state"] for s in samples])).to(device)
    score_steps = torch.from_numpy(np.stack([s["target"]["score_steps"] for s in samples])).to(device)
    owners = torch.tensor(OWNER_IDS, dtype=torch.long, device=device)
    return inputs, features, initial, positive, states, score_steps, owners


def evaluate(model, samples, arm, device, method, output=None):
    model.eval(); rows, future_rows = [], []
    with torch.inference_mode():
        for sample in samples:
            inputs, features, initial, positive, states, _, owners = _batch([sample], arm, device)
            result = model_inference(model, inputs, initial, owners)
            logits = result["logits"][0].cpu().numpy()
            visibility = result["visibility"][0].cpu().numpy()
            prediction = predictions_from_logits(logits, visibility)
            rows.extend(score_prediction(sample, prediction, method))
            retrieved = F.normalize(torch.einsum("btkn,btnd->btkd", result["attention"], inputs), dim=-1, eps=1e-8)
            if arm != "coordinate_only":
                targets, valid = future_targets(features, positive, states)
                future = F.normalize(result["future"][:, :targets.shape[1]], dim=-1, eps=1e-8)
                anchors = F.normalize(torch.einsum("bkn,bnd->bkd", initial, features[:, 0]), dim=-1, eps=1e-8)
                cos = (future * targets).sum(-1)[0].cpu().numpy()
                copy_cos = (anchors[:, None] * targets).sum(-1)[0].cpu().numpy()
                last = retrieved[:, :targets.shape[1]]
                last_cos = (last * targets).sum(-1)[0].cpu().numpy()
                selected = valid[0].cpu().numpy()
                identity_hits = {}
                for label, values in (("forecast", future), ("anchor", anchors[:, None].expand_as(future)), ("last_retrieved", last)):
                    similarities = torch.einsum("btkd,btjd->btkj", values, targets)
                    similarities = similarities.masked_fill(~valid[:, :, None, :], -torch.inf)
                    own = torch.arange(4, device=device)[None, None]
                    identity_hits[label] = (similarities.argmax(-1) == own)[0].cpu().numpy()
                for t, k in np.argwhere(selected):
                    future_rows.append({"scene": sample["name"], "method": method, "forecast_step": int(t),
                                        "target_step": int(t + 8), "target": bench.PART_NAMES[int(k) + 1],
                                        "forecast_cosine": float(cos[t, k]), "anchor_copy_cosine": float(copy_cos[t, k]),
                                        "last_retrieved_copy_cosine": float(last_cos[t, k]),
                                        "visible_future_candidates": int(selected[t].sum()),
                                        "forecast_own_future_part_hit": bool(identity_hits["forecast"][t, k]),
                                        "anchor_own_future_part_hit": bool(identity_hits["anchor"][t, k]),
                                        "last_retrieved_own_future_part_hit": bool(identity_hits["last_retrieved"][t, k])})
            if output is not None:
                destination = Path(output) / f"{sample['name']}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, logits=logits, scores=prediction.scores, cells=prediction.cells,
                                    eligible=prediction.eligible, visibility=visibility,
                                    future=result["future"][0].cpu().numpy(), retrieved=retrieved[0].cpu().numpy())
    summary = bench.summarize_rows(rows)
    return {"overall": summary, "utility": _utility(summary),
            "by_condition": {c: bench.summarize_rows([r for r in rows if r["condition"] == c]) for c in data.CONDITIONS},
            "future": {"targets": len(future_rows),
                       "forecast_mean_cosine": float(np.mean([r["forecast_cosine"] for r in future_rows])) if future_rows else None,
                       "anchor_copy_mean_cosine": float(np.mean([r["anchor_copy_cosine"] for r in future_rows])) if future_rows else None,
                       "last_retrieved_copy_mean_cosine": float(np.mean([r["last_retrieved_copy_cosine"] for r in future_rows])) if future_rows else None,
                       "own_future_part_retrieval": {label: {
                           "all_visible_targets": {"hits": sum(r[label + "_own_future_part_hit"] for r in future_rows), "targets": len(future_rows)},
                           "at_least_two_candidates": {"hits": sum(r[label + "_own_future_part_hit"] for r in future_rows if r["visible_future_candidates"] >= 2),
                                                       "targets": sum(r["visible_future_candidates"] >= 2 for r in future_rows)}}
                           for label in ("forecast", "anchor", "last_retrieved")},
                       "note": "not reported for coordinate basis; not directly comparable to JEPA latent targets" if arm == "coordinate_only" else "scoring labels only; all visible future targets at horizon8"}}, rows, future_rows


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(train_samples, dev_samples, out, arm, seed, config, device):
    folder = Path(out) / "models" / arm / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    selected_path = folder / "selected.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        if selected["training_config"] != config or sha256(folder / "best.pt") != selected["checkpoint_sha256"]:
            raise RuntimeError("Existing selected checkpoint differs; use a fresh output folder")
        print(f"DAY7_RESUMED {arm} seed{seed}", flush=True)
        return selected
    _seed_everything(seed)
    model = PersistentPartJEPA(256, 128, 8).to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=POLICY["learning_rate"], weight_decay=POLICY["weight_decay"])
    initial_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    best, curves = None, []
    total_started = time.perf_counter()
    for epoch in range(1, POLICY["epochs"] + 1):
        model.train(); started = time.perf_counter()
        # Identical scene order for every arm at a given seed/epoch.
        order = np.random.default_rng(seed * 1000 + epoch).permutation(len(train_samples))
        losses, retrieval_losses, future_losses = [], [], []
        for offset in range(0, len(order), config["batch_scenes"]):
            batch = [train_samples[i] for i in order[offset:offset + config["batch_scenes"]]]
            inputs, features, initial, positive, states, steps, owners = _batch(batch, arm, device)
            optimizer.zero_grad(set_to_none=True)
            result = model_inference(model, inputs, initial, owners)
            retrieval, _ = balanced_retrieval_loss(result["logits"], positive, states, steps)
            if POLICY["future_lambda"][arm]:
                targets, valid = future_targets(features, positive, states)
                auxiliary, _ = auxiliary_future_loss(result["future"], targets, valid)
            else:
                auxiliary = retrieval.detach() * 0.
            loss = retrieval + POLICY["future_lambda"][arm] * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach())); retrieval_losses.append(float(retrieval.detach())); future_losses.append(float(auxiliary.detach()))
        report, _, _ = evaluate(model, dev_samples, arm, device, f"{arm}_seed{seed}")
        row = {"arm": arm, "training_seed": seed, "epoch": epoch,
               "train_loss": float(np.mean(losses)), "train_retrieval_loss": float(np.mean(retrieval_losses)),
               "train_auxiliary_loss": float(np.mean(future_losses)), "dev_utility": report["utility"],
               "dev_visible_localization": report["overall"]["localization_accuracy_given_visible"]["rate"],
               "dev_false_presence": report["overall"]["false_presence_given_absent"]["rate"],
               "seconds": time.perf_counter() - started}
        curves.append(row); _write_csv(folder / "curves.csv", curves)
        if best is None or report["utility"] > best["development"]["utility"]:
            torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed, "epoch": epoch}, folder / "best.pt")
            best = {"arm": arm, "seed": seed, "epoch": epoch, "development": report}
        print("DAY7_EPOCH", json.dumps(row), flush=True)
    selected = {**best, "last_epoch_development": report, "training_config": config, "initial_state_sha256": initial_hash,
                "checkpoint_sha256": sha256(folder / "best.pt"), "curves_sha256": sha256(folder / "curves.csv"),
                "training_seconds": time.perf_counter() - total_started}
    write_json(selected_path, selected)
    print(f"DAY7_MODEL_COMPLETE {arm} seed{seed} bestepoch{selected['epoch']}", flush=True)
    return selected


def train_all(features, out, device="cuda", batch_scenes=2, oom_fallback_note=None):
    if batch_scenes not in (1, 2) or (batch_scenes == 1 and not oom_fallback_note):
        raise ValueError("Batch1 only allowed for a documented global OOM fallback")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    config = {"policy": POLICY, "source_digest": source_digest(), "source_hashes": source_hashes(),
              "data": data.manifest(), "feature_binding": feature_binding(features),
              "batch_scenes": batch_scenes, "oom_fallback_note": oom_fallback_note,
              "torch": torch.__version__, "device": device}
    # Normalize tuple-valued data metadata to its persisted JSON form.
    config = json.loads(json.dumps(config))
    config_path = out / "training_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Training configuration changed; start a fresh output folder")
    write_json(config_path, config)
    train_samples, dev_samples = load_split(features, "train"), load_split(features, "dev")
    baselines = {method: select_baseline_threshold(dev_samples, method) for method in BASELINES}
    write_json(out / "baseline_development.json", baselines)
    selected = []
    for seed in SEEDS:
        for arm in ARMS:
            selected.append(train_one(train_samples, dev_samples, out, arm, seed, config, device))
    for seed in SEEDS:
        if len({v["initial_state_sha256"] for v in selected if v["seed"] == seed}) != 1:
            raise AssertionError("Paired arm initialization differs")
    write_json(out / "training_summary.json", {"models": selected, "baseline_development": baselines,
               "all_predetermined_seeds_retained": True, "test_evaluated": False})
    frozen = freeze_checkpoints(features, out)
    print("DAY7_TRAINING_COMPLETE commit checkpoint_freeze.json BEFORE test", flush=True)
    return frozen


def freeze_checkpoints(features, out):
    out = Path(out)
    config = json.loads((out / "training_config.json").read_text())
    if config["source_digest"] != source_digest() or config["feature_binding"] != feature_binding(features):
        raise RuntimeError("Training source/features changed before checkpoint freeze")
    records = []
    for seed in SEEDS:
        for arm in ARMS:
            folder = out / "models" / arm / str(seed)
            selected = json.loads((folder / "selected.json").read_text())
            if selected["training_config"] != config or selected["checkpoint_sha256"] != sha256(folder / "best.pt"):
                raise RuntimeError("Selected model does not match completed training")
            records.append({"arm": arm, "seed": seed, "epoch": selected["epoch"],
                            "path": str((folder / "best.pt").relative_to(out)),
                            "checkpoint_sha256": selected["checkpoint_sha256"],
                            "selection_sha256": sha256(folder / "selected.json"),
                            "dev_utility": selected["development"]["utility"]})
    frozen = {"training_config": config, "models": records,
              "baselines": json.loads((out / "baseline_development.json").read_text()),
              "selection_rule": POLICY["selection"], "test_accessed": False,
              "test_seed_manifest": data.scene_specs("test")}
    write_json(out / "checkpoint_freeze.json", frozen)
    return frozen


def _method_report(rows):
    return {"overall": bench.summarize_rows(rows),
            "by_condition": {c: bench.summarize_rows([r for r in rows if r["condition"] == c]) for c in data.CONDITIONS},
            "uncertainty": bench.summarize_with_ci(rows)}


def test_all(features, out, frozen_path=None, device="cuda"):
    out = Path(out)
    frozen_path = Path(frozen_path) if frozen_path else out / "checkpoint_freeze.json"
    frozen = json.loads(frozen_path.read_text()); config = frozen["training_config"]
    if config["source_digest"] != source_digest() or config["feature_binding"] != feature_binding(features):
        raise RuntimeError("Source or train/dev features changed after checkpoint freeze")
    if frozen["test_seed_manifest"] != data.scene_specs("test") or frozen["selection_rule"] != POLICY["selection"]:
        raise RuntimeError("Held-out manifest/selection changed")
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if {(r["arm"], r["seed"]) for r in frozen["models"]} != expected or len(frozen["models"]) != 9:
        raise ValueError("Freeze must retain all nine predetermined checkpoints")
    for record in frozen["models"]:
        if sha256(out / record["path"]) != record["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed after freeze")
        if sha256((out / record["path"]).with_name("selected.json")) != record["selection_sha256"]:
            raise RuntimeError("Development selection record changed")
    # This is the first loader allowed to render held-out labels.
    test_samples = load_split(features, "test")
    all_rows, future_rows, methods = [], [], {}
    for method in BASELINES:
        threshold = frozen["baselines"][method]["threshold"]
        rows = [r for s in test_samples for r in score_prediction(s, s["baselines"][method], method, threshold)]
        all_rows.extend(rows); methods[method] = _method_report(rows)
        for sample in test_samples:
            pred = sample["baselines"][method]
            path = out / "test" / "predictions" / method / f"{sample['name']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, scores=pred.scores, cells=pred.cells, eligible=pred.eligible)
    for record in frozen["models"]:
        arm, seed = record["arm"], record["seed"]; method = f"{arm}_seed{seed}"
        model = PersistentPartJEPA(256, 128, 8).to(device).float()
        saved = torch.load(out / record["path"], map_location=device, weights_only=True)
        if (saved["arm"], saved["seed"], saved["epoch"]) != (arm, seed, record["epoch"]):
            raise RuntimeError("Checkpoint identity differs from selection")
        model.load_state_dict(saved["state_dict"])
        summary, rows, forecasts = evaluate(model, test_samples, arm, device, method,
                                           out / "test" / "predictions" / method)
        methods[method] = {**_method_report(rows), "future": summary["future"], "selected_epoch": record["epoch"]}
        all_rows.extend(rows); future_rows.extend(forecasts)
        print("DAY7_TEST_MODEL_COMPLETE", method, json.dumps(summary["overall"]), flush=True)
        del model
    paired = {}
    for seed in SEEDS:
        a = [r for r in all_rows if r["method"] == f"predictive_seed{seed}"]
        b = [r for r in all_rows if r["method"] == f"retrieval_only_seed{seed}"]
        paired[str(seed)] = {metric: bench.paired_bootstrap_difference(a, b, metric=metric)
                            for metric in ("localization_accuracy_given_visible", "false_presence_given_absent", "recovery_accuracy")}
    report = {"split": "test", "checkpoint_freeze_sha256": sha256(frozen_path),
              "checkpoint_freeze": frozen, "test_feature_binding": feature_binding(features, ("test",)),
              "methods": methods, "predictive_minus_retrieval_paired_by_training_seed": paired,
              "seed_summary": {arm: {metric: {"values": [methods[f"{arm}_seed{s}"]["overall"][metric]["rate"] for s in SEEDS],
                  "mean": float(np.mean([methods[f"{arm}_seed{s}"]["overall"][metric]["rate"] for s in SEEDS]))}
                  for metric in ("localization_accuracy_given_visible", "false_presence_given_absent", "wrong_car_given_visible", "recovery_accuracy")} for arm in ARMS},
              "claim_scope": "supervised learned head; frozen JEPA encoder; same2D procedural family; noSAM, nofaces, nogeneration",
              "temporal_scope": "head causal in feature steps; encoder windows may use future video frames",
              "all_seeds_reported": True, "test_selection": "none"}
    write_json(out / "test" / "results.json", report)
    _write_csv(out / "test" / "prediction_rows.csv", all_rows)
    _write_csv(out / "test" / "future_rows.csv", future_rows)
    lines = ["# Day7 learned persistent-part heads", "", "All predetermined training seeds are reported. Test results select no model.", "",
             "| Method | Visible localization | False presence while absent | Wrong car given visible | Recovery |", "|---|---:|---:|---:|---:|"]
    for method, value in methods.items():
        cells = []
        for metric in ("localization_accuracy_given_visible", "false_presence_given_absent", "wrong_car_given_visible", "recovery_accuracy"):
            v = value["overall"][metric]; cells.append("n/a" if v["rate"] is None else f"{100*v['rate']:.3f}% ({v['numerator']}/{v['denominator']})")
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
    lines += ["", "`coordinate_only` receives no image/JEPA content; strong performance would expose procedural-position shortcuts.",
              "The predictive arm differs from retrieval_only only in future-target auxiliary loss. Both architectures feed delayed forecasts into later retrieval.",
              "Future latent cosine and paired per-seed clip intervals are in results.json; raw per-target forecasts are in future_rows.csv.",
              "This is supervised part-label learning over a frozen JEPA backbone, not self-supervised JEPA training. No real-video, face, or generative claim follows."]
    (out / "test" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("DAY7_TEST_COMPLETE", flush=True)
    return report


def self_test():
    """Fabricated CPU loss fixtures; no feature extraction or held-out rendering."""
    logits = torch.zeros((1, 3, 2, 5), requires_grad=True)
    positive = torch.zeros((1, 3, 2, 4), dtype=torch.bool)
    positive[0, 1, 0, :2] = True
    state = torch.tensor([[[-1, -1], [1, 0], [-1, -1]]])
    steps = torch.tensor([[False, True, True]])
    loss, counts = balanced_retrieval_loss(logits, positive, state, steps)
    expected = (-math.log(2/5) - math.log(1/5)) / 2
    assert abs(loss.item() - expected) < 1e-6 and counts == {"visible": 1, "absent": 1}
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0 and torch.count_nonzero(logits.grad[:, 2]) == 0
    feature = F.normalize(torch.randn(1, 12, 4, 6), dim=-1).requires_grad_(True)
    pos = torch.zeros((1, 12, 2, 4), dtype=torch.bool); pos[..., 0, 0] = True; pos[..., 1, 1] = True
    states = torch.ones((1, 12, 2), dtype=torch.int8)
    target, valid = future_targets(feature, pos, states, horizon=8)
    assert not target.requires_grad and torch.allclose(target[:, :, 0], feature[:, 8:, 0], atol=1e-6)
    future = target.clone().requires_grad_(True)
    auxiliary, _ = auxiliary_future_loss(future, target, valid)
    assert torch.isfinite(auxiliary); auxiliary.backward(); assert torch.isfinite(future.grad).all()
    assert feature.grad is None, "Future teacher must stay detached even for gradient-bearing features"
    none = torch.zeros_like(valid); empty, _ = auxiliary_future_loss(future, target, none)
    assert empty.item() == 0
    coord = coordinate_tokens(3, 4, 16)
    assert np.array_equal(coord[0], coord[2]) and np.isfinite(coord).all()
    spread = np.zeros((32, 4, 577), np.float32)
    spread[..., -1] = math.log(576) + math.log(.1 / .9)
    # Each individual spatial cell loses to null, yet aggregate visibility=.9.
    assert (spread.argmax(-1) == 576).all()
    assert predictions_from_logits(spread).eligible.all()
    spread[..., -1] = math.log(576) + math.log(.6 / .4)
    assert not predictions_from_logits(spread).eligible.any()
    assert set(inspect.signature(model_inference).parameters) == {"model", "features", "initial_weights", "owner_ids"}
    # Verify the scorer against the frozen Day4 evaluator on the first TRAIN scene.
    spec = data.scene_specs("train")[0]; scene = data.generate_scene(**spec); supervision = data.supervision(scene.masks)
    sample = {**spec, "target": supervision}
    pred = bench.Predictions(np.ones((32, 4), np.float32), np.zeros((32, 4, 2), np.int16), np.ones((32, 4), bool))
    assert score_prediction(sample, pred, "fixture") == bench.score_scene(scene, pred, 0., "fixture")
    return {"multi_positive_balanced_loss": True, "t0_ambiguous_gradients_zero": True,
            "fixed_teacher_target": True, "finite_auxiliary_empty_and_visible": True,
            "coordinate_control_no_image_input": True, "inference_annotation_barrier": True,
            "factorized_presence_not_joint_argmax": True,
            "scorer_matches_frozen_day4": True, "test_scene_generated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["train", "freeze", "test"])
    parser.add_argument("--features")
    parser.add_argument("--out")
    parser.add_argument("--frozen-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-scenes", type=int, default=2)
    parser.add_argument("--oom-fallback-note")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif not args.stage or not args.features or not args.out:
        parser.error("Need --stage, --features and --out")
    elif args.stage == "train":
        train_all(args.features, args.out, args.device, args.batch_scenes, args.oom_fallback_note)
    elif args.stage == "freeze":
        print(json.dumps(freeze_checkpoints(args.features, args.out), indent=2))
    else:
        test_all(args.features, args.out, args.frozen_path, args.device)
