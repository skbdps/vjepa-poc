"""Train source-conditioned latent interventions; freeze before test extraction.

The target clip is used only as supervision. At inference the edit receives
source JEPA features, oracle source selections, and requested displacement.
Train/development files are bound individually; this module never opens test.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

import data
import operators
import probe

HERE = Path(__file__).resolve().parent
ARMS = ("learned_residual", "geometry_only")
SEEDS = (1801, 1802, 1803)
POLICY = {
    "version": "day8_direct_latent_intervention_v1",
    "arms": list(ARMS), "seeds": list(SEEDS), "epochs": 30,
    "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
    "batch_scenes": 1, "precision": "float32", "gradient_clip_norm": 5.0,
    "train_loss": "mean of nonempty source-hole, destination, halo raw MSE means; equal scene weighting",
    "selection": "lowest dev mean of source-hole and destination no-op-normalized MSE; equal scenes; earliest epoch on ties",
    "normalization_floor": "per-role max(1e-8,0.01*median training no-op role MSE)",
    "halo": "trained as separate equally weighted role; reported separately at dev; not included in selection",
    "probe_seed": 1800,
    "probe_policy": "fit genuine train videos, select epoch on genuine dev videos; frozen before any test or edited tokens",
    "test_policy": "never opened in this module; all six predeclared checkpoints retained",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_hashes():
    return {name: sha256(HERE / name) for name in
            ("train.py", "operators.py", "probe.py", "data.py")}


def cache_root(features):
    features = Path(features)
    if (features / "manifest.json").is_file():
        return features
    if features.name == "cache" and (features.parent / "manifest.json").is_file():
        return features.parent
    raise ValueError("--features must contain manifest.json or be its cache subdirectory")


def bind_features(features):
    root = cache_root(features)
    manifest = json.loads((root / "manifest.json").read_text())
    required_provenance = ("extraction_device", "inference_precision", "cache_precision", "weights")
    if any(field not in manifest for field in required_provenance) or not manifest["weights"]:
        raise ValueError("Extraction must finish and record its precision/checkpoint provenance before training")
    if any(row["spec"]["split"] == "test" for row in manifest["clips"]):
        raise ValueError("Refusing a fresh pretest training run after test caches exist")
    indexed = {row["spec"]["name"]: row for row in manifest["clips"]}
    if len(indexed) != len(manifest["clips"]):
        raise ValueError("Duplicate feature-cache names")
    records = []
    for split in ("train", "dev"):
        for spec in data.scene_specs(split):
            row = indexed.get(spec["name"])
            if row is None or row["spec"] != spec:
                raise ValueError("Missing or mismatched cache: " + spec["name"])
            path = root / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError("Cache checksum mismatch: " + str(path))
            records.append(row)
    binding = {"model": manifest["model"], "upstream": manifest["upstream"],
               "extractor_source_hashes": manifest["source_hashes"],
               "encoder_context": manifest["encoder_context"],
               "oracle_budget": manifest["oracle_budget"], "clips": records,
               **{field: manifest[field] for field in required_provenance}}
    return root, indexed, binding


def load_cache(root, record):
    with np.load(root / record["path"], allow_pickle=False) as archive:
        item = {name: archive[name].copy() for name in
                ("source", "target", "source_frac", "target_frac",
                 "distractor_frac", "rgb_source", "rgb_target")}
    for name in ("source", "target"):
        if item[name].shape != (16, 24, 24, 1024) or not np.isfinite(item[name]).all():
            raise ValueError("Invalid latent feature cache: " + name)
    for name in ("source_frac", "target_frac", "distractor_frac"):
        value = item[name]
        if value.shape != (16, 24, 24) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Invalid patch occupancy: " + name)
    for name in ("rgb_source", "rgb_target"):
        value = item[name]
        if value.shape != (16, 24, 24, 3) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Invalid patch RGB: " + name)
    return item


def write_curves(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(root, indexed, split):
    """Keep only editable tokens; full target grids never enter the head."""
    if split not in ("train", "dev"):
        raise ValueError("Training loader supports train/development only")
    result = []
    for spec in data.scene_specs(split):
        cached = load_cache(root, indexed[spec["name"]])
        prepared = operators.prepare_inputs(
            cached["source"], cached["source_frac"], None,
            cached["distractor_frac"], int(spec["dx"]) // data.PATCH)
        selected = prepared["edit_mask"]
        if not selected.any():
            raise ValueError("A nonzero training edit has no editable patches")
        # Destination is derived from the source mask. Cached target masks only
        # validate the controlled renderer; they are never inputs to the model.
        if not np.allclose(operators.shift_horizontal(cached["source_frac"], int(spec["dx"]) // data.PATCH),
                           cached["target_frac"], atol=1e-6, rtol=0):
            raise ValueError("Rendered target does not equal requested translation")
        item = {"spec": spec}
        for name in ("original", "base", "anchor", "geometry", "edit_mask",
                     "source_hole", "destination", "halo"):
            item[name] = torch.from_numpy(np.ascontiguousarray(prepared[name][selected]))
        item["target"] = torch.from_numpy(np.ascontiguousarray(cached["target"][selected].astype(np.float32)))
        no_op = (item["original"] - item["target"]).square().mean(dim=-1)
        item["noop_mse"] = {name: float(no_op[item[name]].mean())
                            for name in ("source_hole", "destination", "halo") if item[name].any()}
        result.append(item)
        print("TRAIN_INPUT_READY", split, spec["name"], int(selected.sum()), flush=True)
        del cached, prepared
    gc.collect()
    return result


def fit_normalization(train):
    floors = {}
    for role in ("source_hole", "destination", "halo"):
        values = [item["noop_mse"][role] for item in train if role in item["noop_mse"]]
        if not values:
            raise ValueError("Training has no samples for role " + role)
        floors[role] = max(1e-8, 0.01 * float(np.median(values)))
    return floors


def device_inputs(item, device):
    return {name: item[name].to(device) for name in
            ("original", "base", "anchor", "geometry", "edit_mask")}


@torch.inference_mode()
def evaluate_dev(model, examples, floors, device):
    model.eval()
    rows = []
    for item in examples:
        prediction = model(**device_inputs(item, device))
        error = (prediction - item["target"].to(device)).square().mean(dim=-1)
        row = {"name": item["spec"]["name"]}
        normalized = []
        for role in ("source_hole", "destination", "halo"):
            mask = item[role].to(device)
            if not mask.any():
                continue
            mse = float(error[mask].mean())
            no_op = item["noop_mse"][role]
            row[role + "_mse"] = mse
            row[role + "_noop_mse"] = no_op
            row[role + "_ratio"] = mse / max(no_op, floors[role])
            row[role + "_denominator_clipped"] = int(no_op < floors[role])
            if role in ("source_hole", "destination"):
                normalized.append(row[role + "_ratio"])
        if len(normalized) != 2:
            raise ValueError("Each dev scene must have a source hole and destination")
        row["selection_score"] = float(np.mean(normalized))
        rows.append(row)
    summary = {"dev_score": float(np.mean([row["selection_score"] for row in rows]))}
    for role in ("source_hole", "destination", "halo"):
        summary["dev_" + role + "_mse"] = float(np.mean([row[role + "_mse"] for row in rows]))
        summary["dev_" + role + "_ratio"] = float(np.mean([row[role + "_ratio"] for row in rows]))
        summary["dev_" + role + "_denominator_clipped"] = int(sum(row[role + "_denominator_clipped"] for row in rows))
    return summary, rows


def train_one(train, dev, floors, arm, seed, out, device):
    set_seed(seed)
    model = operators.build_model(dim=1024, content_blind=(arm == "geometry_only")).to(device)
    # Initialization and random scene order are independent of the arm name.
    initialization = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                           for value in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=POLICY["learning_rate"],
                                 weight_decay=POLICY["weight_decay"])
    folder = Path(out) / arm / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    curves, best_score, best_epoch = [], float("inf"), None
    for epoch in range(1, POLICY["epochs"] + 1):
        tick = time.perf_counter()
        model.train()
        losses = []
        order = np.random.default_rng(seed * 1000 + epoch).permutation(len(train))
        for index in order:
            item = train[int(index)]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(**device_inputs(item, device))
            target = item["target"].to(device)
            regions = {name: item[name].to(device) for name in ("source_hole", "destination", "halo")}
            loss = operators.region_balanced_mse(prediction, target, regions)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite intervention loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), POLICY["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, dev_rows = evaluate_dev(model, dev, floors, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics,
               "seconds": time.perf_counter() - tick}
        curves.append(row)
        if metrics["dev_score"] < best_score:
            best_score, best_epoch = metrics["dev_score"], epoch
            payload = {"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                       "arm": arm, "seed": seed, "epoch": epoch, "dev_score": best_score,
                       "dim": 1024, "content_blind": arm == "geometry_only", "policy": POLICY,
                       "normalization_floors": floors, "initialization_sha256": initialization}
            torch.save(payload, folder / "best.pt")
            write_json(folder / "best_dev.json", {"summary": metrics, "scenes": dev_rows})
        write_curves(folder / "curves.csv", curves)
        print("EDIT_EPOCH", arm, seed, epoch, "train", round(row["train_loss"], 7),
              "dev_ratio", round(metrics["dev_score"], 6), "best", best_epoch, flush=True)
    checkpoint = folder / "best.pt"
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"arm": arm, "seed": seed, "path": str(checkpoint.relative_to(out)),
            "sha256": sha256(checkpoint), "epoch": best_epoch, "dev_score": best_score,
            "initialization_sha256": initialization}


def genuine_examples(root, indexed, split):
    """Yield only true video encodings to the independent frozen readout."""
    if split not in ("train", "dev"):
        raise ValueError("Probe fitting cannot access test")
    for spec in data.scene_specs(split):
        cached = load_cache(root, indexed[spec["name"]])
        for side in ("source", "target"):
            occupancy = cached[side + "_frac"] + cached["distractor_frac"]
            if np.any(occupancy > 1):
                raise ValueError("Unexpected overlapping occupancy labels")
            yield {"tokens": cached[side], "occupancy": occupancy,
                   "rgb": cached["rgb_" + side]}


def model_smoke_check(device):
    """Assert intervention boundaries before spending time on a real run."""
    rng = np.random.default_rng(1800)
    tokens = rng.normal(size=(2, 8, 8, 16)).astype(np.float32)
    source = np.zeros((2, 8, 8), np.float32)
    source[:, 2:4, 2:4] = 1
    distractor = np.zeros_like(source)
    distractor[:, 6, 6] = 1
    prepared = operators.prepare_inputs(tokens, source, None, distractor, 2)
    model = operators.build_model(dim=16).to(device)
    inputs = {name: torch.from_numpy(prepared[name]).to(device) for name in
              ("original", "base", "anchor", "geometry", "edit_mask")}
    prediction = model(**inputs)
    if not torch.equal(prediction, inputs["base"]):
        raise AssertionError("Zero-initialized head does not equal transport base")
    target = torch.from_numpy(rng.normal(size=tokens.shape).astype(np.float32)).to(device)
    loss = (prediction - target).square().mean()
    loss.backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise AssertionError("Nonfinite smoke-check gradients")
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-0.01)
    after = model(**inputs)
    outside = ~inputs["edit_mask"]
    if not torch.equal(after[outside], inputs["original"][outside]):
        raise AssertionError("Head changed tokens outside allowed support")
    zero = operators.prepare_inputs(tokens, source, None, distractor, 0)
    zero_inputs = {name: torch.from_numpy(zero[name]).to(device) for name in inputs}
    if not torch.equal(model(**zero_inputs), zero_inputs["original"]):
        raise AssertionError("Zero request is not exact identity after an update")
    return {"status": "passed", "train_or_test_data_accessed": False,
            "checks": ["initial output equals residual base", "finite gradients",
                       "outside support exactly preserved after update", "zero request exactly identity after update"]}


def run(features, out, device="cuda"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    freeze_path = out / "checkpoint_freeze.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text())
        if frozen["source_hashes"] != source_hashes():
            raise ValueError("Training source changed after freeze")
        for record in frozen["models"]:
            if sha256(out / record["path"]) != record["sha256"]:
                raise ValueError("Frozen checkpoint changed")
        print("TRAINING_ALREADY_FROZEN", str(freeze_path), flush=True)
        return frozen
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    write_json(out / "model_smoke_check.json", model_smoke_check(device))
    root, indexed, binding = bind_features(features)
    sources = source_hashes()
    binding_digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    write_json(out / "feature_binding.json", binding)
    write_json(out / "training_policy.json", POLICY)
    train = prepare_split(root, indexed, "train")
    dev = prepare_split(root, indexed, "dev")
    floors = fit_normalization(train)
    normal = {"floors": floors, "fit_split": "train", "roles": {}}
    for role in floors:
        normal["roles"][role] = {split: {
            "denominator_clipped": sum(item["noop_mse"][role] < floors[role] for item in examples),
            "no_op_mse": [item["noop_mse"][role] for item in examples]}
            for split, examples in (("train", train), ("dev", dev))}
    write_json(out / "normalization.json", normal)
    records = []
    for seed in SEEDS:
        for arm in ARMS:
            completed = out / arm / str(seed) / "completed.json"
            if completed.exists():
                saved = json.loads(completed.read_text())
                if saved["source_hashes"] != sources or saved["feature_binding_sha256"] != binding_digest or saved["policy"] != POLICY:
                    raise ValueError("Cannot resume a model from different sources, inputs, or policy")
                record = saved["record"]
                if sha256(out / record["path"]) != record["sha256"]:
                    raise ValueError("Completed model checkpoint changed")
                print("EDIT_REUSED", arm, seed, flush=True)
            else:
                record = train_one(train, dev, floors, arm, seed, out, device)
                write_json(completed, {"record": record, "policy": POLICY,
                                       "source_hashes": sources, "feature_binding_sha256": binding_digest})
            records.append(record)
        if records[-1]["initialization_sha256"] != records[-2]["initialization_sha256"]:
            raise AssertionError("Correction arms did not start with identical weights")
    del train, dev
    gc.collect()
    print("PROBE_FIT_STARTED", flush=True)
    fitted, history = probe.fit_probe(
        genuine_examples(root, indexed, "train"), device=str(device), seed=1800,
        dev_examples=genuine_examples(root, indexed, "dev"),
        max_epochs=40, batches_per_epoch=32, batch_size=512)
    probe.save_probe(fitted, out / "probe.pt", history)
    write_json(out / "probe_history.json", history)
    records.append({"arm": "frozen_readout", "seed": 1800, "path": "probe.pt",
                    "sha256": sha256(out / "probe.pt"), "epoch": history["selected_epoch"],
                    "dev_score": history["selected_dev"]["loss"]})
    if sources != source_hashes():
        raise RuntimeError("Source files changed while training; refusing to freeze")
    frozen = {"version": POLICY["version"], "source_hashes": sources,
              "models": records, "policy": POLICY, "normalization_floors": floors,
              "feature_binding_sha256": sha256(out / "feature_binding.json"),
              "test_accessed": False, "train_scenes": len(data.scene_specs("train")),
              "dev_scenes": len(data.scene_specs("dev")),
              "selected_checkpoints": "all two correction arms times three seeds plus one independent readout",
              "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                          "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    write_json(freeze_path, frozen)
    print("TRAINING_FROZEN", str(freeze_path), sha256(freeze_path), flush=True)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.features, args.out, args.device)

