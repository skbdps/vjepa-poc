"""Latent editability test: move ball patches to a new spatial position.

Take ball-patch embeddings from slice 0 at position ~(5,5) and place them
at a background position ~(15,15) in the same slice. Fill original position
with background mean.

If PCA shows yellow blob disappearing from (5,5) and appearing at (15,15),
the content signal is portable across spatial positions.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA

# --- paths ----------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_PATH = REPO_ROOT / "experiments" / "day1" / "outputs" / "bounce-embeddings.pt"
OUT_DIR = REPO_ROOT / "experiments" / "day2" / "outputs"

# --- constants ------------------------------------------------------------
T_TOKENS = 8
H_TOKENS = 24
W_TOKENS = 24
FEAT_DIM = 1024
IMG_SIZE = 384
PATCH_SIZE = IMG_SIZE // H_TOKENS  # 16
BALL_RADIUS = 30

# Physics params
GRAVITY = 400.0
INITIAL_X = 80.0
INITIAL_Y = 80.0
INITIAL_VX = 70.0
INITIAL_VY = 0.0
RESTITUTION = 0.75
GROUND_Y = IMG_SIZE - 30


def simulate_positions() -> list[tuple[float, float]]:
    dt = 1.0 / 30.0
    x, y = INITIAL_X, INITIAL_Y
    vx, vy = INITIAL_VX, INITIAL_VY
    positions = []
    for _ in range(120):
        positions.append((x, y))
        vy += GRAVITY * dt
        x += vx * dt
        y += vy * dt
        if y >= GROUND_Y:
            y = GROUND_Y
            vy = -vy * RESTITUTION
            vx *= 0.95
        if x < BALL_RADIUS:
            x = BALL_RADIUS
            vx = -vx
        if x > IMG_SIZE - BALL_RADIUS:
            x = IMG_SIZE - BALL_RADIUS
            vx = -vx
    return positions


def get_ball_patch_mask(ball_x: float, ball_y: float) -> np.ndarray:
    mask = np.zeros((H_TOKENS, W_TOKENS), dtype=bool)
    for r in range(H_TOKENS):
        for c in range(W_TOKENS):
            px_left = c * PATCH_SIZE
            px_right = px_left + PATCH_SIZE
            py_top = r * PATCH_SIZE
            py_bottom = py_top + PATCH_SIZE
            nearest_x = max(px_left, min(ball_x, px_right))
            nearest_y = max(py_top, min(ball_y, py_bottom))
            dist = np.sqrt((nearest_x - ball_x) ** 2 + (nearest_y - ball_y) ** 2)
            if dist < BALL_RADIUS:
                mask[r, c] = True
    return mask


def load_embeddings() -> np.ndarray:
    bundle = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
    emb = bundle["embeddings"].squeeze(0).numpy()
    return emb.reshape(T_TOKENS, H_TOKENS, W_TOKENS, FEAT_DIM)


def pca_to_rgb(embeddings_4d: np.ndarray) -> np.ndarray:
    flat = embeddings_4d.reshape(-1, FEAT_DIM)
    pca = PCA(n_components=3, random_state=0)
    projected = pca.fit_transform(flat)
    rgb = np.zeros_like(projected)
    for c in range(3):
        lo, hi = projected[:, c].min(), projected[:, c].max()
        if hi - lo > 1e-8:
            rgb[:, c] = (projected[:, c] - lo) / (hi - lo)
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return rgb.reshape(T_TOKENS, H_TOKENS, W_TOKENS, 3)


def upsample(grid_24: np.ndarray) -> np.ndarray:
    return cv2.resize(grid_24, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    # --- ball position at slice 0 ---
    all_pos = simulate_positions()
    sampled = np.linspace(0, 119, 16, dtype=int).tolist()
    f1, f2 = sampled[0], sampled[1]
    ball_x = (all_pos[f1][0] + all_pos[f2][0]) / 2
    ball_y = (all_pos[f1][1] + all_pos[f2][1]) / 2
    print(f"Slice 0 ball at ({ball_x:.0f}, {ball_y:.0f})")

    src_mask = get_ball_patch_mask(ball_x, ball_y)
    src_patches = np.argwhere(src_mask)
    print(f"  Source mask: {src_mask.sum()} patches")
    print(f"  Patch coords: rows {src_patches[:,0].min()}-{src_patches[:,0].max()}, "
          f"cols {src_patches[:,1].min()}-{src_patches[:,1].max()}")

    # --- target position: move ball to (15, 15) in patch coords ---
    # That's pixel center (15*16+8, 15*16+8) = (248, 248)
    target_x, target_y = 248.0, 248.0
    dst_mask = get_ball_patch_mask(target_x, target_y)
    dst_patches = np.argwhere(dst_mask)
    print(f"\nTarget position at pixel ({target_x:.0f}, {target_y:.0f})")
    print(f"  Dest mask: {dst_mask.sum()} patches")
    print(f"  Patch coords: rows {dst_patches[:,0].min()}-{dst_patches[:,0].max()}, "
          f"cols {dst_patches[:,1].min()}-{dst_patches[:,1].max()}")

    # verify no overlap
    overlap = src_mask & dst_mask
    assert overlap.sum() == 0, f"Source and dest overlap by {overlap.sum()} patches!"
    print(f"  No overlap between source and dest masks: OK")

    # --- load and edit ---
    print("\nLoading embeddings...")
    embeddings = load_embeddings()

    print("\nPCA on original...")
    pca_before = pca_to_rgb(embeddings)

    edited = embeddings.copy()
    SLICE = 0

    # compute background mean for filling vacated positions
    bg_mean = embeddings[SLICE][~src_mask & ~dst_mask].mean(axis=0)

    # extract ball vectors from source
    ball_vectors = embeddings[SLICE][src_mask].copy()
    print(f"\nExtracted {ball_vectors.shape[0]} ball vectors from slice {SLICE}")

    # clear source position (fill with background)
    edited[SLICE][src_mask] = bg_mean
    print(f"Cleared source position with background mean")

    # place ball vectors at destination
    # match patches by relative position within the mask bounding box
    n_src = src_patches.shape[0]
    n_dst = dst_patches.shape[0]
    n_place = min(n_src, n_dst)

    # sort both by (row, col) so relative positions align
    src_order = np.lexsort((src_patches[:, 1], src_patches[:, 0]))
    dst_order = np.lexsort((dst_patches[:, 1], dst_patches[:, 0]))

    for i in range(n_place):
        si = src_order[i]
        di = dst_order[i]
        r, c = dst_patches[di]
        edited[SLICE, r, c] = ball_vectors[si]

    print(f"Placed {n_place} ball vectors at destination position")

    # if dst has more patches than src, leave those as-is (background)
    if n_dst > n_src:
        print(f"  ({n_dst - n_src} dest patches left as background)")

    print("\nPCA on edited...")
    pca_after = pca_to_rgb(edited)

    # --- save before/after PNGs for slice 0 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before_path = OUT_DIR / "move_slice0_before.png"
    after_path = OUT_DIR / "move_slice0_after.png"

    cv2.imwrite(str(before_path),
                cv2.cvtColor(upsample(pca_before[SLICE]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(after_path),
                cv2.cvtColor(upsample(pca_after[SLICE]), cv2.COLOR_RGB2BGR))

    print(f"\nSaved {before_path.name} ({before_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved {after_path.name} ({after_path.stat().st_size / 1024:.1f} KB)")

    # --- quantitative check ---
    # count yellow-ish pixels in source and dest regions
    after_slice = pca_after[SLICE]  # (24, 24, 3)
    src_colors = after_slice[src_mask]
    dst_colors = after_slice[dst_mask]
    bg_colors = after_slice[~src_mask & ~dst_mask]

    print(f"\nColor analysis (RGB means):")
    print(f"  Source region (should be bg):  R={src_colors[:,0].mean():.0f} "
          f"G={src_colors[:,1].mean():.0f} B={src_colors[:,2].mean():.0f}")
    print(f"  Dest region (should be ball):  R={dst_colors[:,0].mean():.0f} "
          f"G={dst_colors[:,1].mean():.0f} B={dst_colors[:,2].mean():.0f}")
    print(f"  Background (reference):        R={bg_colors[:,0].mean():.0f} "
          f"G={bg_colors[:,1].mean():.0f} B={bg_colors[:,2].mean():.0f}")

    print("\nExpected: source region color ≈ background color,")
    print("          dest region color ≈ yellow/orange (high R+G, low B)")
    print("\nDone.")


if __name__ == "__main__":
    main()
