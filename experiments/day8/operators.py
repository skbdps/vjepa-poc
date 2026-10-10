"""Source-only latent interventions for the controlled Day 8 translation test.

The mask budget is deliberately generous: source-object coverage throughout
all tubelets plus unchanged distractor coverage. Destination coverage is
DERIVED from source coverage and the requested integer patch displacement.
No target features, target RGB, or empty-scene features enter these operators.

``dx`` is signed horizontal displacement IN PATCHES, not pixels. Inputs are
float feature arrays [T,H,W,D] and fractional masks [T,H,W]. All transport reads
an immutable source snapshot. Fractional masks select overlapping tokens;
coverage is not multiplied into the already mixed feature vectors a second
time. Copies and residual transport are hypotheses, not semantic guarantees.
"""
from __future__ import annotations

from typing import Any

import numpy as np
try:
    import torch
    from torch import nn
except ImportError:  # NumPy-only transport checks can run before torch installation.
    torch = None
    nn = None

GEOMETRY_NAMES = (
    "dx_over_width", "time", "x", "y", "relative_source_x", "relative_source_y",
    "source_coverage", "destination_coverage", "distractor_coverage",
    "source_hole", "destination", "halo", "edit_support",
)
GEOMETRY_DIM = len(GEOMETRY_NAMES)
OPERATOR_VERSION = "day8_source_only_transport_v1"


def shift_horizontal(array: np.ndarray, dx: int) -> np.ndarray:
    """Zero-padded spatial translation; never wraps at an image boundary.

    Supported layouts are [T,H,W] and [T,H,W,D]. Clipped content is discarded.
    """
    array = np.asarray(array)
    if array.ndim not in (3, 4):
        raise ValueError("Expected [T,H,W] or [T,H,W,D]")
    if int(dx) != dx:
        raise ValueError("dx must be an integer number of patches")
    dx = int(dx)
    result = np.zeros_like(array)
    width = array.shape[2]
    if abs(dx) >= width:
        return result
    if dx > 0:
        result[:, :, dx:] = array[:, :, :width - dx]
    elif dx < 0:
        result[:, :, :width + dx] = array[:, :, -dx:]
    else:
        result[...] = array
    return result


def dilate_spatial(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Square spatial dilation independently in each tubelet, no time leakage."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or radius < 0:
        raise ValueError("Expected a [T,H,W] mask and nonnegative radius")
    _, height, width = mask.shape
    padded = np.pad(mask, ((0, 0), (radius, radius), (radius, radius)))
    result = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[:, dy:dy + height, dx:dx + width]
    return result


def _validated(tokens: np.ndarray, source_frac: np.ndarray,
               target_frac: np.ndarray | None, distractor_frac: np.ndarray,
               dx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    tokens = np.asarray(tokens, dtype=np.float32)
    source_frac = np.asarray(source_frac, dtype=np.float32)
    distractor_frac = np.asarray(distractor_frac, dtype=np.float32)
    if tokens.ndim != 4 or source_frac.shape != tokens.shape[:3] or distractor_frac.shape != source_frac.shape:
        raise ValueError("Expected tokens [T,H,W,D] and coverage [T,H,W]")
    if not np.isfinite(tokens).all():
        raise ValueError("Nonfinite feature input")
    for name, fraction in (("source", source_frac), ("distractor", distractor_frac)):
        if not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
            raise ValueError(f"{name} coverage must be finite and in [0,1]")
    if int(dx) != dx:
        raise ValueError("dx must be an integer number of patches, not pixels")
    dx = int(dx)
    destination = shift_horizontal(source_frac, dx)
    # Supplying target masks is optional, and only checks geometry. The values
    # used by the operator always come from the source mask + request.
    if target_frac is not None:
        supplied_target = np.asarray(target_frac, dtype=np.float32)
        if supplied_target.shape != source_frac.shape or not np.allclose(supplied_target, destination, atol=1e-6, rtol=0):
            raise ValueError("Target coverage disagrees with shifted source coverage")
    return tokens, source_frac, destination, distractor_frac, dx


def local_background(tokens: np.ndarray, excluded: np.ndarray,
                     query: np.ndarray, radius: int = 2) -> np.ndarray:
    """Estimate source-hole features using neighboring object-free tokens.

    The default radius gives a 5x5 ring (all masked object tokens excluded).
    A query with no valid local neighbor falls back to the same tubelet's
    object-free mean. An entirely masked tubelet is rejected. Unqueried tokens
    are copied unchanged. This is a feature-space approximation: a neighboring
    JEPA vector is not guaranteed to encode the hidden background.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    excluded, query = np.asarray(excluded, dtype=bool), np.asarray(query, dtype=bool)
    if tokens.ndim != 4 or excluded.shape != tokens.shape[:3] or query.shape != excluded.shape:
        raise ValueError("Mismatched token/mask shapes")
    if radius < 1:
        raise ValueError("Background radius must be at least one")
    result = tokens.copy()
    _, height, width = excluded.shape
    fallback: dict[int, np.ndarray] = {}
    for t, y, x in np.argwhere(query):
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        available = ~excluded[t, y0:y1, x0:x1]
        if available.any():
            result[t, y, x] = tokens[t, y0:y1, x0:x1][available].mean(axis=0, dtype=np.float32)
        else:
            if int(t) not in fallback:
                available_global = ~excluded[t]
                if not available_global.any():
                    raise ValueError("No object-free background token in a tubelet")
                fallback[int(t)] = tokens[t][available_global].mean(axis=0, dtype=np.float32)
            result[t, y, x] = fallback[int(t)]
    return result


def _transport(tokens: np.ndarray, selected_frac: np.ndarray,
               all_objects: np.ndarray, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return whole-token copy and local-background residual transport."""
    if dx == 0:
        return tokens.copy(), tokens.copy()
    source = selected_frac > 0
    destination = shift_horizontal(source, dx)
    background = local_background(tokens, all_objects, source)
    # Erase first, then write the transported object. Destination wins in any
    # source/destination overlap, and all reads still use original snapshots.
    naive = tokens.copy()
    naive[source] = background[source]
    shifted_tokens = shift_horizontal(tokens, dx)
    naive[destination] = shifted_tokens[destination]
    residual = tokens.copy()
    residual[source] = background[source]
    source_residual = np.zeros_like(tokens)
    source_residual[source] = tokens[source] - background[source]
    moved_residual = shift_horizontal(source_residual, dx)
    # At overlap, the destination background is the erased original-source
    # background; elsewhere it is the original destination vector.
    residual[destination] += moved_residual[destination]
    return naive, residual


def base_edits(tokens: np.ndarray, source_frac: np.ndarray,
               target_frac: np.ndarray | None, distractor_frac: np.ndarray,
               dx: int) -> dict[str, np.ndarray]:
    """Deterministic interventions and controls, all float32 independent arrays.

    Controls use residual transport with the wrong direction or wrong object.
    Wrong-object transport is clipped safely when a distractor crosses bounds.
    Only source/destination union tokens change; no baseline modifies a halo.
    All arms become exact identity at dx==0.
    """
    tokens, source_frac, _, distractor_frac, dx = _validated(
        tokens, source_frac, target_frac, distractor_frac, dx)
    excluded = (source_frac > 0) | (distractor_frac > 0)
    naive, residual = _transport(tokens, source_frac, excluded, dx)
    _, wrong_direction = _transport(tokens, source_frac, excluded, -dx)
    _, wrong_object = _transport(tokens, distractor_frac, excluded, dx)
    return {"noop": tokens.copy(), "naive": naive, "residual": residual,
            "wrong_direction": wrong_direction, "wrong_object": wrong_object}


def prepare_inputs(tokens: np.ndarray, source_frac: np.ndarray,
                   target_frac: np.ndarray | None, distractor_frac: np.ndarray,
                   dx: int, base: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Build model inputs and disjoint loss regions without target features.

    ``anchor`` is the coverage-weighted mean source-object representation over
    the complete video. It is shared across every edited tubelet, providing a
    common identity reference, though it also contains background/context.
    Exposed region masks allow equally weighted mean losses per source hole,
    destination, and halo (omit empty regions), avoiding background domination.
    ``edit_mask`` is the source/destination union plus one spatial patch halo.
    Flattening/selecting all returned arrays identically supports token batches.
    """
    tokens, source_frac, destination_frac, distractor_frac, dx = _validated(
        tokens, source_frac, target_frac, distractor_frac, dx)
    source, destination = source_frac > 0, destination_frac > 0
    union = source | destination
    edit_mask = dilate_spatial(union, 1)
    hole, halo = source & ~destination, edit_mask & ~union
    if dx == 0:
        # Geometrical selection is still described by coverage, but no region
        # is editable for the identity request, including its spatial halo.
        edit_mask = np.zeros_like(edit_mask)
        hole, destination, halo = (np.zeros_like(source) for _ in range(3))
    if base is None:
        _, base = _transport(tokens, source_frac, source | (distractor_frac > 0), dx)
    else:
        base = np.asarray(base, dtype=np.float32)
        if base.shape != tokens.shape:
            raise ValueError("Base edit must match token shape")
        if not np.array_equal(base[~union], tokens[~union]):
            raise ValueError("Base edit changed tokens outside the source/destination union")
    count = float(source_frac.sum())
    if count <= 0:
        raise ValueError("Selected source object has no token coverage")
    anchor_vector = np.einsum("thwd,thw->d", tokens, source_frac, optimize=True) / count
    anchor = np.broadcast_to(anchor_vector, tokens.shape).copy()
    steps, height, width = source.shape
    tt, yy, xx = np.mgrid[:steps, :height, :width].astype(np.float32)
    mass = source_frac.sum(axis=(1, 2))
    safe_mass = np.maximum(mass, np.finfo(np.float32).eps)
    center_x = (source_frac * xx).sum(axis=(1, 2)) / safe_mass
    center_y = (source_frac * yy).sum(axis=(1, 2)) / safe_mass
    geometry = np.stack([
        np.full(source.shape, dx / width, dtype=np.float32),
        2 * tt / max(steps - 1, 1) - 1,
        2 * xx / max(width - 1, 1) - 1,
        2 * yy / max(height - 1, 1) - 1,
        (xx - center_x[:, None, None]) / width,
        (yy - center_y[:, None, None]) / height,
        source_frac, destination_frac, distractor_frac,
        hole, destination, halo, edit_mask,
    ], axis=-1).astype(np.float32)
    return {"original": tokens.copy(), "base": base.copy(), "anchor": anchor,
            "geometry": geometry, "edit_mask": edit_mask,
            "source_hole": hole, "destination": destination, "halo": halo}


class ResidualEditHead(nn.Module if nn is not None else object):
    """A shared, zero-initialized correction to deterministic residual transport.

    Full feature vectors are supervised; no PCA projection defines the loss.
    With D=1024 the head has ~243k parameters. One shared content projection is
    used for original, base, and the sequence-level identity anchor. Geometry-
    only correction uses the exact same architecture with these three content
    inputs zeroed. Its deterministic residual base still carries source content;
    it is NOT a fully content-blind editor.
    """
    def __init__(self, dim: int = 1024, content_blind: bool = False,
                 projection_dim: int = 64, hidden_dim: int = 128):
        if nn is None:
            raise ImportError("PyTorch is required to build the learned edit head")
        super().__init__()
        self.dim = int(dim)
        self.content_blind = bool(content_blind)
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim, projection_dim)
        self.correction = nn.Sequential(
            nn.Linear(3 * projection_dim + GEOMETRY_DIM, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, original: torch.Tensor, base: torch.Tensor,
                anchor: torch.Tensor, geometry: torch.Tensor,
                edit_mask: torch.Tensor) -> torch.Tensor:
        """Accept token grids or flat token batches with any leading dimensions.

        For a shuffled-anchor intervention, pass an anchor from another scene
        while keeping all other inputs unchanged. This control should be called
        an evaluation intervention rather than an independently trained arm.
        """
        if original.shape != base.shape or original.shape != anchor.shape:
            raise ValueError("Original, base, and anchor shapes must agree")
        if original.shape[-1] != self.dim or geometry.shape != original.shape[:-1] + (GEOMETRY_DIM,):
            raise ValueError("Wrong feature or geometry dimensions")
        if edit_mask.shape == original.shape[:-1] + (1,):
            edit_mask = edit_mask.squeeze(-1)
        if edit_mask.shape != original.shape[:-1]:
            raise ValueError("Edit-mask shape must match token leading dimensions")
        contents = (original, base, anchor)
        if self.content_blind:
            contents = tuple(torch.zeros_like(value) for value in contents)
        projected = [self.project(self.norm(value)) for value in contents]
        correction = self.correction(torch.cat([*projected, geometry.to(original.dtype)], dim=-1))
        requested = geometry[..., 0] != 0
        enabled = edit_mask.bool() & requested
        edited = base + correction
        # Exact identity for zero request and exact preservation outside allowed
        # support, even if a caller supplies a corrupted base outside support.
        return torch.where(enabled.unsqueeze(-1), edited, original)


def build_model(dim: int = 1024, content_blind: bool = False, **kwargs: Any) -> ResidualEditHead:
    return ResidualEditHead(dim=dim, content_blind=content_blind, **kwargs)


def region_balanced_mse(prediction: torch.Tensor, target: torch.Tensor,
                        regions: dict[str, torch.Tensor]) -> torch.Tensor:
    """Equal weight for each nonempty hole/destination/halo region's mean MSE.

    Supply a complete clip, or a batch sampled separately by region. Pooling
    arbitrary full-video batches across clips changes the weighting and should
    be disclosed by the caller. Loss normalization is independent of test data.
    """
    per_token = (prediction.float() - target.float()).square().mean(dim=-1)
    losses = []
    for name in ("source_hole", "destination", "halo"):
        mask = regions[name].bool()
        if mask.shape != per_token.shape:
            raise ValueError(f"Wrong {name} mask shape")
        if mask.any():
            losses.append(per_token[mask].mean())
    return torch.stack(losses).mean() if losses else per_token.sum() * 0
