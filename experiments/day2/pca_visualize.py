"""Step A: PCA -> RGB visualization of V-JEPA 2.1 patch embeddings.

Replicates Figure 1 / Figure 15 of the V-JEPA 2.1 paper.

Loads patch embeddings from day 1, fits PCA with 3 components, maps to RGB,
upsamples and composes side-by-side with original video frames.

Runs on CPU (sklearn). No TPU needed.

Output:
  experiments/day2/outputs/pca_rgb.mp4       (side-by-side, 768x384)
  experiments/day2/outputs/pca_stats.txt     (explained variance, etc.)
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from torchcodec.decoders import VideoDecoder

# --- paths ----------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_PATH = REPO_ROOT / "experiments" / "day1" / "outputs" / "bounce-embeddings.pt"
VIDEO_PATH = REPO_ROOT / "experiments" / "day1" / "outputs" / "bounce.mp4"
OUT_DIR = REPO_ROOT / "experiments" / "day2" / "outputs"
OUT_VIDEO = OUT_DIR / "pca_rgb.mp4"
OUT_STATS = OUT_DIR / "pca_stats.txt"

# --- tensor layout constants ---------------------------------------------
# (1, 4608, 1024) = 8 temporal * 24 * 24 spatial, 1024-d features
T_TOKENS = 8
H_TOKENS = 24
W_TOKENS = 24
FEAT_DIM = 1024
NUM_FRAMES_SAMPLED = 16   # 2 * T_TOKENS (tubelet size 2)
IMG_SIZE = 384
FPS = 30  # match source video


def load_embeddings() -> np.ndarray:
    """Load and squeeze to (4608, 1024) numpy."""
    bundle = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
    embeddings = bundle["embeddings"]  # (1, 4608, 1024)
    assert embeddings.shape == (1, T_TOKENS * H_TOKENS * W_TOKENS, FEAT_DIM), \
        f"Unexpected shape: {embeddings.shape}"
    return embeddings.squeeze(0).numpy()  # (4608, 1024)


def compute_pca(features: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Fit PCA(3) on (N, D) features, return (N, 3) projection + fitted pca."""
    pca = PCA(n_components=3, random_state=0)
    projected = pca.fit_transform(features)  # (4608, 3)
    return projected, pca


def normalize_to_rgb(projected: np.ndarray) -> np.ndarray:
    """Per-component min-max -> [0, 1], then scale to uint8 [0, 255]."""
    normalized = np.zeros_like(projected)
    for c in range(3):
        lo, hi = projected[:, c].min(), projected[:, c].max()
        if hi - lo > 1e-8:
            normalized[:, c] = (projected[:, c] - lo) / (hi - lo)
        else:
            normalized[:, c] = 0.5
    return (normalized * 255).clip(0, 255).astype(np.uint8)


def reshape_to_spatial_temporal(rgb_flat: np.ndarray) -> np.ndarray:
    """(4608, 3) -> (T, H, W, 3) = (8, 24, 24, 3)."""
    return rgb_flat.reshape(T_TOKENS, H_TOKENS, W_TOKENS, 3)


def upsample_nearest(pca_grid: np.ndarray) -> np.ndarray:
    """Upsample (T, 24, 24, 3) -> (T, 384, 384, 3) via nearest-neighbor."""
    out = np.empty((T_TOKENS, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for t in range(T_TOKENS):
        out[t] = cv2.resize(
            pca_grid[t], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
        )
    return out


def expand_to_16_frames(pca_upsampled: np.ndarray) -> np.ndarray:
    """Each temporal token covers 2 input frames -> repeat each slice twice.
    (8, 384, 384, 3) -> (16, 384, 384, 3)."""
    return np.repeat(pca_upsampled, 2, axis=0)


def load_sampled_frames() -> np.ndarray:
    """Load the same 16 frames that were fed to the encoder in day 1.
    Returns (16, 384, 384, 3) uint8 RGB."""
    decoder = VideoDecoder(str(VIDEO_PATH))
    n_native = decoder.metadata.num_frames
    indices = np.linspace(0, n_native - 1, NUM_FRAMES_SAMPLED, dtype=int).tolist()
    batch = decoder.get_frames_at(indices=indices)
    frames = batch.data  # (16, 3, 384, 384) uint8
    frames = frames.permute(0, 2, 3, 1).numpy()  # (16, 384, 384, 3)
    return frames


def compose_side_by_side(original_rgb: np.ndarray, pca_rgb: np.ndarray) -> np.ndarray:
    """(16, 384, 384, 3) + (16, 384, 384, 3) -> (16, 384, 768, 3)."""
    assert original_rgb.shape == pca_rgb.shape
    return np.concatenate([original_rgb, pca_rgb], axis=2)


def write_video(frames_rgb: np.ndarray, path: Path) -> None:
    """Write (N, H, W, 3) uint8 RGB array as mp4v."""
    n, h, w, _ = frames_rgb.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer at {path}")
    for i in range(n):
        # opencv expects BGR
        bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def main() -> None:
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    features = load_embeddings()
    print(f"  features: shape={features.shape}, "
          f"mean={features.mean():.4f}, std={features.std():.4f}")

    print("\nFitting PCA (3 components)...")
    projected, pca = compute_pca(features)
    print(f"  projected shape: {projected.shape}")
    print(f"  explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  cumulative: {pca.explained_variance_ratio_.sum():.4f}")

    print("\nNormalizing to RGB...")
    rgb_flat = normalize_to_rgb(projected)
    pca_grid = reshape_to_spatial_temporal(rgb_flat)  # (8, 24, 24, 3)
    print(f"  pca_grid shape: {pca_grid.shape}, "
          f"dtype: {pca_grid.dtype}, range: [{pca_grid.min()}, {pca_grid.max()}]")

    print("\nUpsampling to 384x384...")
    pca_upsampled = upsample_nearest(pca_grid)  # (8, 384, 384, 3)
    pca_expanded = expand_to_16_frames(pca_upsampled)  # (16, 384, 384, 3)
    print(f"  final PCA frames shape: {pca_expanded.shape}")

    print(f"\nLoading original frames from {VIDEO_PATH}...")
    original = load_sampled_frames()  # (16, 384, 384, 3)
    print(f"  original frames shape: {original.shape}")

    print("\nComposing side-by-side...")
    composed = compose_side_by_side(original, pca_expanded)  # (16, 384, 768, 3)
    print(f"  composed shape: {composed.shape}")

    print(f"\nWriting video to {OUT_VIDEO}...")
    write_video(composed, OUT_VIDEO)
    size_kb = OUT_VIDEO.stat().st_size / 1024
    print(f"  wrote {OUT_VIDEO} ({size_kb:.1f} KB)")

    print(f"\nWriting stats to {OUT_STATS}...")
    OUT_STATS.write_text(
        f"PCA stats for bounce-embeddings.pt\n"
        f"===================================\n"
        f"Features: {features.shape}\n"
        f"Feature mean: {features.mean():.4f}\n"
        f"Feature std:  {features.std():.4f}\n"
        f"Explained variance ratio: {pca.explained_variance_ratio_.tolist()}\n"
        f"Cumulative (top 3 PCs):   {pca.explained_variance_ratio_.sum():.4f}\n"
        f"Singular values: {pca.singular_values_.tolist()}\n"
    )
    print(f"  wrote {OUT_STATS}")

    print("\nDone.")


if __name__ == "__main__":
    main()
