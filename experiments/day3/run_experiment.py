"""Frozen V-JEPA 2.1 part-consistency diagnostic, with resumable feature caches.

Run on a CUDA runtime; no model training, image decoder, or Drive mount required.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from part_consistency import generate_suite, evaluate_features

UPSTREAM_SHA = "204698b45b3712590f06245fbfba32d3be539812"
MODEL_NAME = "vjepa2_1_vit_large_384"


def load_encoder(root: Path):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before running frozen feature extraction.")
    actual_sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual_sha != UPSTREAM_SHA:
        raise RuntimeError(f"Expected upstream {UPSTREAM_SHA}; got {actual_sha}")
    sys.path.insert(0, str(root))
    # This pinned upstream commit leaves the weight URL on localhost.
    # Set the official public source in memory without changing upstream files.
    from src.hub import backbones
    backbones.VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
    encoder, predictor = getattr(backbones, MODEL_NAME)(pretrained=True)
    del predictor
    gc.collect()
    encoder = encoder.eval().requires_grad_(False).to("cuda")
    return encoder


def encode_scenes(encoder, scenes, cache_dir: Path):
    import torch
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1)
    for scene in scenes:
        digest = hashlib.sha256(scene.frames.tobytes() + (MODEL_NAME + UPSTREAM_SHA + "fp16-v1").encode()).hexdigest()[:16]
        path = cache_dir / f"{scene.name}_{digest}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape != (16, 24, 24, 1024) or not np.isfinite(arr).all():
                raise RuntimeError(f"Invalid cached features: {path}")
            results[scene.name] = arr.astype(np.float32)
            print(f"CACHE {scene.name}: {arr.shape}", flush=True)
            continue
        start = time.perf_counter()
        windows = []
        # Two disjoint clips. The second call cannot attend to first-window pixels.
        for offset in (0, 16):
            frames = np.ascontiguousarray(scene.frames[offset:offset + 16])
            x = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().to("cuda") / 255.0
            x = (x - mean) / std
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                features = encoder(x)
            if not isinstance(features, torch.Tensor) or tuple(features.shape) != (1, 4608, 1024):
                raise RuntimeError(f"Unexpected encoder output: {getattr(features, 'shape', type(features))}")
            arr = features.float().cpu().numpy().reshape(8, 24, 24, 1024)
            if not np.isfinite(arr).all():
                raise RuntimeError("Non-finite frozen features")
            windows.append(arr)
            del x, features
        arr = np.concatenate(windows, axis=0)
        # Match using the same precision on first run and cache reuse.
        arr = arr.astype(np.float16)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(path)
        results[scene.name] = arr.astype(np.float32)
        print(f"ENCODED {scene.name}: {arr.shape}, {time.perf_counter() - start:.1f}s", flush=True)
    return results


def run(out_dir: Path, upstream_root: Path, encoder=None):
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    calibration = generate_suite(out_dir / "inputs", split="calibration", seeds=[100, 101, 102])
    evaluation = generate_suite(out_dir / "inputs", split="evaluation", seeds=[200, 201, 202, 203, 204, 205])
    print(f"Generated {len(calibration)} calibration + {len(evaluation)} held-out clips", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    if encoder is None:
        encoder = load_encoder(upstream_root)
    all_features = encode_scenes(encoder, calibration + evaluation, out_dir / "features")
    summary = evaluate_features(calibration, evaluation, all_features, all_features, out_dir / "results")
    manifest = {
        "model": MODEL_NAME, "upstream_commit": UPSTREAM_SHA,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "calibration_seeds": [100, 101, 102], "evaluation_seeds": [200, 201, 202, 203, 204, 205],
        "input": "32 RGB frames per scene, two disjoint 16-frame windows, 384x384, ImageNet normalization",
        "inference": "frozen eval mode, no gradients, CUDA fp16 autocast; cached features fp16",
        "scope": "Small synthetic diagnostic; offline within-window features can attend to future frames. Not an editing, face, or real-video benchmark.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nEXPERIMENT_COMPLETE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("day3_run"))
    parser.add_argument("--upstream", type=Path, default=Path("/content/vjepa2"))
    args = parser.parse_args()
    run(args.out, args.upstream)
