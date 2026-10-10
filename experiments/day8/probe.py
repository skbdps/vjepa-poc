"""Frozen coarse semantic readout for the Day 8 latent-edit experiment.

This is a diagnostic readout, not a video generator. It maps ONE JEPA token to
any-ball pixel coverage and mean RGB in its 2-frame, 16x16-pixel tubelet. Its
only inference input is the token: no coordinates, masks, displacement, scene
ID, or requested edit. Fit it on genuine train source/target encodings, select
the epoch on genuine dev encodings, then freeze before opening the test set.

``fit_probe`` accepts dictionaries with ``tokens`` [..., C], ``occupancy``
[...] in [0,1], and ``rgb`` [...,3] in [0,1]. The normal Day 8 grid is
[16,24,24] with C=1024, but the implementation permits small synthetic fixtures.
The RGB labels are patch means (including background at boundaries), not the
selected object's isolated color. Occupancy covers BOTH objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn


PROBE_VERSION = "day8_frozen_pointwise_readout_v1"


class FrozenReadout(nn.Module):
    """Train-standardized, pointwise MLP with one occupancy and three RGB heads."""

    def __init__(self, feature_mean, feature_std, hidden_dim: int = 128):
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(feature_std, dtype=torch.float32).reshape(-1)
        if mean.shape != std.shape or len(mean) == 0:
            raise ValueError("Feature mean/std must have equal, nonempty shapes")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Feature statistics must be finite, with positive std")
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(nn.Linear(len(mean), self.hidden_dim), nn.GELU(),
                                     nn.Linear(self.hidden_dim, 4), nn.Sigmoid())
        self.metadata = {}

    def forward(self, tokens):
        if tokens.shape[-1] != len(self.feature_mean):
            raise ValueError("Input channel count differs from fitted probe")
        normalized = (tokens.float() - self.feature_mean) / self.feature_std
        return self.network(normalized)


def _prepare(examples: Iterable[dict], label: str, channels=None):
    prepared = []
    for index, example in enumerate(examples):
        tokens = np.asarray(example["tokens"])
        occupancy = np.asarray(example["occupancy"], dtype=np.float32)
        rgb = np.asarray(example["rgb"], dtype=np.float32)
        if tokens.ndim < 2 or occupancy.shape != tokens.shape[:-1] or rgb.shape != (*occupancy.shape, 3):
            raise ValueError(f"{label} example {index}: incompatible token/label shapes")
        if channels is None:
            channels = tokens.shape[-1]
        if channels != tokens.shape[-1]:
            raise ValueError(f"{label} example {index}: inconsistent channel count")
        if not np.isfinite(occupancy).all() or not np.isfinite(rgb).all():
            raise ValueError(f"{label} example {index}: nonfinite labels")
        if occupancy.min() < 0 or occupancy.max() > 1 or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError(f"{label} example {index}: labels must be scaled to [0,1]")
        # These views keep large feature arrays on CPU. Only sampled minibatches
        # are copied to the accelerator; all scenes never have to fit its RAM.
        prepared.append({"tokens": tokens.reshape(-1, channels),
                         "labels": np.concatenate([occupancy.reshape(-1, 1), rgb.reshape(-1, 3)], axis=1)})
    if not prepared:
        raise ValueError(f"{label} examples must not be empty")
    return prepared, int(channels)


def _train_statistics(examples, channels):
    total = np.zeros(channels, dtype=np.float64)
    squares = np.zeros(channels, dtype=np.float64)
    count = 0
    for example in examples:
        tokens = example["tokens"]
        for start in range(0, len(tokens), 4096):
            # Double precision sufficient statistics avoid catastrophic
            # cancellation for feature channels with a nonzero global mean.
            block = np.asarray(tokens[start:start + 4096], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("Training tokens contain a nonfinite value")
            total += block.sum(axis=0)
            squares += np.einsum("nc,nc->c", block, block)
            count += len(block)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0))
    floored = int((std < 1e-6).sum())
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32), count, floored


class _BalancedSampler:
    """Uniform sampling within foreground/background, with an equal class mix."""

    def __init__(self, examples, channels):
        self.examples = examples
        self.channels = channels
        pools = {False: [], True: []}
        for index, example in enumerate(examples):
            foreground = example["labels"][:, 0] > 0
            for flag in (False, True):
                row = np.flatnonzero(foreground == flag).astype(np.int32)
                pools[flag].append(np.column_stack([np.full(len(row), index, np.int32), row]))
        self.background = np.concatenate(pools[False])
        self.foreground = np.concatenate(pools[True])
        if not len(self.background) or not len(self.foreground):
            raise ValueError("Probe fitting/selection needs both any-ball>0 and background tokens")

    def sample(self, rng, count):
        n_foreground = count // 2
        n_background = count - n_foreground
        locations = np.concatenate([
            self.foreground[rng.integers(len(self.foreground), size=n_foreground)],
            self.background[rng.integers(len(self.background), size=n_background)]])
        rng.shuffle(locations)
        features = np.empty((count, self.channels), dtype=np.float32)
        labels = np.empty((count, 4), dtype=np.float32)
        for index in np.unique(locations[:, 0]):
            keep = locations[:, 0] == index
            rows = locations[keep, 1]
            features[keep] = self.examples[index]["tokens"][rows]
            labels[keep] = self.examples[index]["labels"][rows]
        if not np.isfinite(features).all():
            raise ValueError("Sampled probe features contain a nonfinite value")
        return features, labels


def _losses(prediction, labels):
    occupancy = (prediction[:, 0] - labels[:, 0]).square().mean()
    rgb = (prediction[:, 1:] - labels[:, 1:]).square().mean()
    return occupancy + rgb, occupancy, rgb


@torch.no_grad()
def _evaluate_sample(model, sample, device, batch_size=4096):
    features, labels = sample
    totals = np.zeros(3, dtype=np.float64)
    model.eval()
    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start:start + batch_size]).to(device)
        y = torch.from_numpy(labels[start:start + batch_size]).to(device)
        values = _losses(model(x), y)
        totals += np.array([float(v.item()) for v in values]) * len(x)
    return dict(zip(("loss", "occupancy_mse", "rgb_mse"), (totals / len(features)).tolist()))


def fit_probe(train_examples: Iterable[dict], device="cpu", seed: int = 1800,
              dev_examples: Iterable[dict] | None = None, max_epochs: int = 40,
              batches_per_epoch: int = 32, batch_size: int = 512,
              learning_rate: float = 1e-3, hidden_dim: int = 128,
              dev_sample_size: int = 8192):
    """Fit genuine train data; return ``(frozen_model, JSON-safe history)``.

    Each batch is 50% any-ball>0 tokens and 50% exact background, sampled
    uniformly within each group with replacement. The unweighted loss is
    occupancy MSE + RGB-channel-mean MSE. This balance prevents a trivial
    background-only occupancy readout; outputs are soft coverage, not
    calibrated class probabilities under natural background prevalence.

    Normalization uses ALL train tokens, never dev. When dev is supplied,
    choose the earliest minimum loss over one fixed balanced dev sample.
    Train all requested epochs (no adaptive early stopping). With no dev,
    return the final epoch. No edited features enter fitting or selection.
    Iterable inputs are accepted, but their CPU token views are retained for
    random sampling. They are not concatenated or widened from float16.
    """
    if min(max_epochs, batches_per_epoch, batch_size, dev_sample_size) <= 0 or batch_size < 2:
        raise ValueError("Positive epoch/batch counts and batch_size>=2 are required")
    if learning_rate <= 0 or hidden_dim <= 0:
        raise ValueError("Learning rate and hidden dimension must be positive")
    device = torch.device(device)
    train, channels = _prepare(train_examples, "train")
    mean, std, n_tokens, n_floored = _train_statistics(train, channels)
    train_sampler = _BalancedSampler(train, channels)
    dev_sample = None
    dev = []
    if dev_examples is not None:
        dev, _ = _prepare(dev_examples, "dev", channels)
        dev_sampler = _BalancedSampler(dev, channels)
        dev_sample = dev_sampler.sample(np.random.default_rng(seed + 100003), dev_sample_size)
    # Initializing on CPU inside fork_rng leaves callers' RNG state intact.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FrozenReadout(mean, std, hidden_dim)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history = {"version": PROBE_VERSION, "seed": int(seed), "input": "JEPA token only",
               "targets": "any-ball tubelet coverage and tubelet patch-mean RGB",
               "config": {"max_epochs": max_epochs, "batches_per_epoch": batches_per_epoch,
                          "batch_size": batch_size, "learning_rate": learning_rate,
                          "hidden_dim": hidden_dim, "weight_decay": 1e-4,
                          "dev_sample_size": dev_sample_size if dev_sample is not None else 0,
                          "sampling": "50/50 any-ball>0/background, replacement, uniform within group",
                          "loss": "occupancy MSE + mean RGB-channel MSE",
                          "selection": "earliest minimum fixed balanced dev loss" if dev_sample is not None else "final epoch"},
               "train": {"examples": len(train), "tokens": n_tokens,
                         "foreground_tokens": len(train_sampler.foreground),
                         "background_tokens": len(train_sampler.background),
                         "std_floor": 1e-6, "std_floored_channels": n_floored,
                         "statistics_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()},
               "dev_examples": len(dev), "epochs": []}
    best_loss, best_state, selected_epoch = float("inf"), None, 0
    started = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_totals = np.zeros(3, dtype=np.float64)
        for _ in range(batches_per_epoch):
            features, labels = train_sampler.sample(rng, batch_size)
            x = torch.from_numpy(features).to(device)
            y = torch.from_numpy(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = _losses(model(x), y)
            if not torch.isfinite(losses[0]):
                raise RuntimeError("Nonfinite probe training loss")
            losses[0].backward()
            optimizer.step()
            epoch_totals += np.array([float(v.item()) for v in losses])
        record = {"epoch": epoch, "train": dict(zip(("loss", "occupancy_mse", "rgb_mse"),
                                                   (epoch_totals / batches_per_epoch).tolist()))}
        if dev_sample is not None:
            record["dev"] = _evaluate_sample(model, dev_sample, device)
            selection_loss = record["dev"]["loss"]
        else:
            # No dev is explicitly final-epoch selection, not noisy train-loss
            # selection. Retain every epoch in history for transparency.
            selection_loss = -float(epoch)
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            selected_epoch = epoch
        history["epochs"].append(record)
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    history["selected_epoch"] = selected_epoch
    history["selected_dev"] = history["epochs"][selected_epoch - 1].get("dev")
    history["elapsed_seconds"] = time.monotonic() - started
    history["trainable_parameters_during_fit"] = sum(p.numel() for p in model.parameters())
    history["test_accessed"] = False
    model.metadata = history
    return model, history


@torch.no_grad()
def predict_probe(model: FrozenReadout, tokens, device=None, batch_size: int = 4096) -> dict:
    """Read out tokens, preserving their leading grid dimensions.

    Returns NumPy float32 ``occupancy`` [...], ``rgb`` [...,3]. No labels or
    masks can be supplied. Caller must calibrate this readout on genuine
    held-out encodings before interpreting failures on interventions.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim < 2 or tokens.shape[-1] != len(model.feature_mean) or batch_size <= 0:
        raise ValueError("Invalid token shape, channel count, or batch size")
    if device is None:
        device = model.feature_mean.device
    else:
        device = torch.device(device)
        model.to(device)
    shape = tokens.shape[:-1]
    flat = tokens.reshape(-1, tokens.shape[-1])
    predictions = np.empty((len(flat), 4), dtype=np.float32)
    model.eval()
    for start in range(0, len(flat), batch_size):
        block = np.asarray(flat[start:start + batch_size], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Probe inference tokens contain a nonfinite value")
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        predictions[start:start + len(block)] = model(x).cpu().numpy()
    predictions = predictions.reshape(*shape, 4)
    return {"occupancy": predictions[..., 0], "rgb": predictions[..., 1:]}


def save_probe(model: FrozenReadout, path, history=None) -> None:
    """Save a CPU checkpoint; all normalizing statistics are in its state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": PROBE_VERSION, "hidden_dim": model.hidden_dim,
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "history": model.metadata if history is None else history}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_probe(path, device="cpu") -> tuple[FrozenReadout, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != PROBE_VERSION:
        raise ValueError("Unsupported probe checkpoint version")
    state = payload["state_dict"]
    model = FrozenReadout(state["feature_mean"], state["feature_std"], payload["hidden_dim"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    model.metadata = payload["history"]
    return model, payload["history"]


def self_check():
    """Small synthetic feature fixture; checks learning and checkpoint parity.

    This is an implementation check, never evidence about V-JEPA editability.
    It does not render or inspect the held-out Day 8 test scenes.
    """
    import tempfile
    rng = np.random.default_rng(19)
    def fixture():
        tokens = rng.normal(size=(2, 8, 8, 8)).astype(np.float32)
        occupancy = (tokens[..., 0] > 0).astype(np.float32)
        rgb = (1 / (1 + np.exp(-tokens[..., 1:4]))).astype(np.float32)
        return {"tokens": tokens, "occupancy": occupancy, "rgb": rgb}
    train, dev = [fixture() for _ in range(4)], [fixture()]
    model, history = fit_probe(train, dev_examples=dev, seed=71, max_epochs=12,
                               batches_per_epoch=8, batch_size=128, hidden_dim=32,
                               learning_rate=0.01, dev_sample_size=256)
    before = history["epochs"][0]["dev"]["loss"]
    after = history["selected_dev"]["loss"]
    if not after < before * 0.6:
        raise AssertionError("Probe failed to learn a simple held-out feature readout")
    result = predict_probe(model, dev[0]["tokens"], batch_size=19)
    assert result["occupancy"].shape == dev[0]["occupancy"].shape
    assert result["rgb"].shape == dev[0]["rgb"].shape
    assert all(not p.requires_grad for p in model.parameters())
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "probe.pt"
        save_probe(model, checkpoint, history)
        restored, restored_history = load_probe(checkpoint)
        again = predict_probe(restored, dev[0]["tokens"], batch_size=19)
        for key in result:
            np.testing.assert_array_equal(result[key], again[key])
        assert restored_history == history
    all_train = np.concatenate([x["tokens"].reshape(-1, 8) for x in train])
    np.testing.assert_allclose(model.feature_mean.numpy(), all_train.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(model.feature_std.numpy(), all_train.std(axis=0), atol=1e-6)
    return {"status": "passed", "fixture": "synthetic independent feature/label fixture",
            "first_dev_loss": before, "selected_dev_loss": after,
            "selected_epoch": history["selected_epoch"],
            "checks": ["held-out readout learning", "train-only normalization",
                       "grid-shaped predictions", "frozen inference parameters",
                       "checkpoint roundtrip exact", "no Day 8 test scenes opened"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("Import fit_probe/predict_probe, or pass --self-check")
    print(json.dumps(self_check(), indent=2))
