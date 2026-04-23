"""Encode bouncing ball video through V-JEPA 2.1 ViT-L encoder on TPU.

Pipeline:
  video (mp4) -> torchcodec decode -> sample 16 frames uniformly
    -> normalize with ImageNet stats -> V-JEPA encoder -> patch embeddings

Output: (1, 4608, 1024) float tensor of patch embeddings saved as .pt
Also printed: basic stats so we can sanity-check the embeddings.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch_xla
from torchcodec.decoders import VideoDecoder

# ---------------- paths ----------------
REPO_ROOT = Path(__file__).parent.parent.parent
VIDEO_PATH = REPO_ROOT / "experiments" / "day1" / "outputs" / "bounce.mp4"
OUTPUT_PATH = REPO_ROOT / "experiments" / "day1" / "outputs" / "bounce-embeddings.pt"

# V-JEPA 2.1 expects 16 frames sampled uniformly. 384x384 crop. ImageNet norm.
NUM_FRAMES = 16
IMG_SIZE = 384
# ImageNet normalization — verified in V-JEPA demo notebook
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def load_and_preprocess(path: Path) -> torch.Tensor:
    """Load video, sample 16 frames, normalize. Returns (1, 3, 16, 384, 384) float."""
    decoder = VideoDecoder(str(path))
    native_n = decoder.metadata.num_frames
    print(f"Video: {native_n} frames @ {decoder.metadata.average_fps} fps, "
          f"{decoder.metadata.width}x{decoder.metadata.height}, codec={decoder.metadata.codec}")

    # Uniform temporal subsample
    indices = np.linspace(0, native_n - 1, NUM_FRAMES, dtype=int).tolist()
    print(f"Sampled frame indices: {indices}")

    batch = decoder.get_frames_at(indices=indices)
    # batch.data shape: (T=16, C=3, H, W), dtype uint8
    frames = batch.data
    assert frames.shape == (NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), \
        f"Unexpected shape {frames.shape}"

    # uint8 [0,255] -> float [0,1]
    frames = frames.float() / 255.0

    # Reshape to V-JEPA input: (B, C, T, H, W)
    # frames is (T, C, H, W), permute to (C, T, H, W), then add batch dim
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
    assert frames.shape == (1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE)

    # Normalize with ImageNet stats
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return frames


def main() -> None:
    print("Loading and preprocessing video...")
    frames = load_and_preprocess(VIDEO_PATH)
    print(f"Preprocessed frames: shape={tuple(frames.shape)}, "
          f"mean={frames.mean():.3f}, std={frames.std():.3f}")

    print("\nLoading V-JEPA 2.1 ViT-L encoder (pretrained)...")
    vjepa2_root = Path.home() / "vjepa2"
    encoder, _predictor = torch.hub.load(
        str(vjepa2_root),
        "vjepa2_1_vit_large_384",
        source="local",
        pretrained=True,
    )
    encoder.eval()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder loaded: {n_params/1e6:.1f}M params")

    print("\nMoving to TPU...")
    dev = torch_xla.device()
    encoder = encoder.to(dev)
    frames = frames.to(dev)

    print("\nForward pass (XLA compile on first run: 1-3 min)...")
    with torch.no_grad():
        embeddings = encoder(frames)
        torch_xla.sync()
    print(f"Output embeddings: shape={tuple(embeddings.shape)}, "
          f"dtype={embeddings.dtype}, device={embeddings.device}")

    # Bring back to CPU for saving + stats
    embeddings_cpu = embeddings.cpu()
    print(f"\nEmbedding stats:")
    print(f"  mean:    {embeddings_cpu.mean().item():.4f}")
    print(f"  std:     {embeddings_cpu.std().item():.4f}")
    print(f"  min:     {embeddings_cpu.min().item():.4f}")
    print(f"  max:     {embeddings_cpu.max().item():.4f}")
    print(f"  norm:    {embeddings_cpu.norm().item():.2f}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings_cpu,
        "frame_indices": np.linspace(0, 119, NUM_FRAMES, dtype=int).tolist(),
        "video_path": str(VIDEO_PATH),
        "model": "vjepa2_1_vit_large_384",
    }, OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
