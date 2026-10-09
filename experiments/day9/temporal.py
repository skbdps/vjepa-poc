"""Source-only temporal repair of the hole left by Day 8 whole-token copying.

The destination is always the exact frozen Day 8 naive destination. Only the
source-only hole can change. All reads use the immutable original video, and
the API deliberately has no genuine target features, images, or masks.

``build_repairs`` accepts tokens [T,H,W,D], source/distractor fractional masks
[T,H,W], and signed integer horizontal displacement IN PATCHES. It returns:

* ``variants``: independent float32 full arrays named ``naive``,
  ``temporal_mean``, and ``aligned_temporal_mean``.
* ``masks``: dense [T,H,W] source, hole, destination, object-clear masks,
  and per-hole donor count/coverage arrays (zero outside the hole).
* ``donors``: ``hole_indices`` [N,3], donor-validity/alignment/count arrays
  [N,T], and the two final candidate feature means [N,D]. Entries with no
  usable donors contain the exact naive fill. This sparse representation
  includes every hole cell, including failures. No full [N,T,D] tensor is
  retained. Counts at unavailable donor entries are -1.
* ``diagnostics``: JSON-ready counts/coverage, including same/cross encoder
  block donor counts, with means divided by ALL hole cells, not just covered
  cells. One encoder block is eight tubelets / sixteen video frames.

Donors are other tubelets at the same spatial coordinate, free of both
objects plus a one-patch square halo. Aligned donors add the mean difference
between original query and donor features at positions clear in BOTH times:
use a radius-four square with at least four clear positions; otherwise use
the global intersection if nonempty. Reject the aligned donor when that
global intersection is empty. Each arm averages its accepted donors uniformly.
These are offline source-video operations, not causal temporal prediction.

Run the independent analytic NumPy fixtures with ``python temporal.py
--self-check``. They do not inspect development or held-out experiment data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.day8 import operators as day8_operators
except ModuleNotFoundError:
    # Standalone scripts can import this file without adding the repo root or
    # exposing a generic `operators` module name that could collide later.
    _spec = importlib.util.spec_from_file_location(
        "_day9_frozen_day8_operators", Path(__file__).resolve().parents[1] / "day8" / "operators.py")
    if _spec is None or _spec.loader is None:
        raise ImportError("Could not locate frozen Day 8 operators.py")
    day8_operators = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(day8_operators)


OPERATOR_VERSION = "day9_source_only_temporal_hole_v1"
HALO_RADIUS = 1
ALIGN_RADIUS = 4
MIN_LOCAL_CLEAR = 4
TUBELETS_PER_ENCODER_BLOCK = 8


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_repairs(tokens: np.ndarray, source_frac: np.ndarray,
                  distractor_frac: np.ndarray, dx_patch: int) -> dict[str, Any]:
    """Construct three fixed arms with auditable donor availability.

    No input is mutated. Input features are converted to float32, matching
    Day 8 evaluation. Both repairs fall back to the exact naive feature when
    there is no donor. Identity requests return identity in every arm.
    """
    original, source_frac, destination_frac, distractor_frac, dx = day8_operators._validated(
        tokens, source_frac, None, distractor_frac, dx_patch)
    if any(size <= 0 for size in original.shape):
        raise ValueError("All token dimensions must be positive")
    source = source_frac > 0
    destination = destination_frac > 0
    hole = source & ~destination
    objects = source | (distractor_frac > 0)
    clear = ~day8_operators.dilate_spatial(objects, HALO_RADIUS)
    naive, _unused_residual = day8_operators._transport(original, source_frac, objects, dx)
    del _unused_residual
    steps, height, width, dim = original.shape
    indices = np.argwhere(hole).astype(np.int32)
    n_holes = len(indices)

    # N x T keeps every query, including uncovered cells. A tubelet cannot
    # donate to itself; the explicit rule also documents temporal semantics.
    valid = np.zeros((n_holes, steps), dtype=bool)
    aligned_valid = np.zeros_like(valid)
    local_counts = np.full((n_holes, steps), -1, dtype=np.int32)
    global_counts = np.full((n_holes, steps), -1, dtype=np.int32)
    alignment_mode = np.zeros((n_holes, steps), dtype=np.uint8)
    # Modes: 0 unavailable, 1 local, 2 global, 3 rejected (no clear intersection).
    temporal_values = naive[hole].copy()
    aligned_values = naive[hole].copy()
    global_cache: dict[tuple[int, int], tuple[int, np.ndarray | None]] = {}
    for i, (t0, y0, x0) in enumerate(indices):
        t, y, x = int(t0), int(y0), int(x0)
        valid[i] = clear[:, y, x]
        valid[i, t] = False
        donor_times = np.flatnonzero(valid[i])
        if not len(donor_times):
            continue
        temporal_values[i] = original[donor_times, y, x].mean(axis=0, dtype=np.float32)
        y_min, y_max = max(0, y - ALIGN_RADIUS), min(height, y + ALIGN_RADIUS + 1)
        x_min, x_max = max(0, x - ALIGN_RADIUS), min(width, x + ALIGN_RADIUS + 1)
        aligned_sum = np.zeros(dim, dtype=np.float32)
        for dt0 in donor_times:
            dt = int(dt0)
            both_local = (clear[t, y_min:y_max, x_min:x_max]
                          & clear[dt, y_min:y_max, x_min:x_max])
            local_count = int(both_local.sum())
            local_counts[i, dt] = local_count
            pair_key = (t, dt)
            # Global clear counts are recorded even when local alignment wins.
            if pair_key not in global_cache:
                both_global = clear[t] & clear[dt]
                global_count = int(both_global.sum())
                global_offset = None
                if global_count:
                    global_offset = (original[t][both_global] - original[dt][both_global]).mean(
                        axis=0, dtype=np.float32)
                global_cache[pair_key] = global_count, global_offset
            global_count, global_offset = global_cache[pair_key]
            global_counts[i, dt] = global_count
            if local_count >= MIN_LOCAL_CLEAR:
                offset = (original[t, y_min:y_max, x_min:x_max][both_local]
                          - original[dt, y_min:y_max, x_min:x_max][both_local]).mean(
                              axis=0, dtype=np.float32)
                alignment_mode[i, dt] = 1
            elif global_count:
                assert global_offset is not None
                offset = global_offset
                alignment_mode[i, dt] = 2
            else:
                alignment_mode[i, dt] = 3
                continue
            aligned_valid[i, dt] = True
            aligned_sum += original[dt, y, x] + offset
        accepted = int(aligned_valid[i].sum())
        if accepted:
            aligned_values[i] = aligned_sum / np.float32(accepted)

    temporal = naive.copy()
    aligned = naive.copy()
    temporal[hole] = temporal_values
    aligned[hole] = aligned_values
    variants = {"naive": naive, "temporal_mean": temporal, "aligned_temporal_mean": aligned}
    for name, edited in variants.items():
        if not np.isfinite(edited).all():
            raise AssertionError(f"Nonfinite {name} output")
        if not np.array_equal(edited[~hole], naive[~hole]):
            raise AssertionError(f"{name} changed a token outside the source hole")
        if not np.array_equal(edited[destination], naive[destination]):
            raise AssertionError(f"{name} changed the frozen naive destination")
        if dx == 0 and not np.array_equal(edited, original):
            raise AssertionError(f"{name} violated identity")

    donor_counts = valid.sum(axis=1, dtype=np.int32)
    aligned_counts = aligned_valid.sum(axis=1, dtype=np.int32)
    same_block = (indices[:, 0, None] // TUBELETS_PER_ENCODER_BLOCK
                  == np.arange(steps)[None, :] // TUBELETS_PER_ENCODER_BLOCK)
    same_counts = (valid & same_block).sum(axis=1, dtype=np.int32)
    cross_counts = (valid & ~same_block).sum(axis=1, dtype=np.int32)
    aligned_same_counts = (aligned_valid & same_block).sum(axis=1, dtype=np.int32)
    aligned_cross_counts = (aligned_valid & ~same_block).sum(axis=1, dtype=np.int32)

    masks: dict[str, np.ndarray] = {
        "source": source, "destination": destination, "hole": hole, "clear": clear,
    }
    for name, sparse in (
        ("donor_count", donor_counts), ("aligned_donor_count", aligned_counts),
        ("same_block_donor_count", same_counts), ("cross_block_donor_count", cross_counts),
        ("aligned_same_block_donor_count", aligned_same_counts),
        ("aligned_cross_block_donor_count", aligned_cross_counts),
    ):
        dense = np.zeros(hole.shape, dtype=np.int32)
        dense[hole] = sparse
        masks[name] = dense
    masks["donor_covered"] = masks["donor_count"] > 0
    masks["aligned_donor_covered"] = masks["aligned_donor_count"] > 0

    diagnostics: dict[str, Any] = {
        "operator_version": OPERATOR_VERSION,
        "hole_count": n_holes,
        "temporal_covered_holes": int((donor_counts > 0).sum()),
        "aligned_covered_holes": int((aligned_counts > 0).sum()),
        "temporal_coverage": _fraction(int((donor_counts > 0).sum()), n_holes),
        "aligned_coverage": _fraction(int((aligned_counts > 0).sum()), n_holes),
        "temporal_fallback_holes": int((donor_counts == 0).sum()),
        "aligned_fallback_holes": int((aligned_counts == 0).sum()),
        "temporal_donor_pairs": int(valid.sum()),
        "aligned_donor_pairs": int(aligned_valid.sum()),
        "aligned_local_donor_pairs": int((alignment_mode == 1).sum()),
        "aligned_global_donor_pairs": int((alignment_mode == 2).sum()),
        "aligned_rejected_donor_pairs": int((alignment_mode == 3).sum()),
        "alignment_radius": ALIGN_RADIUS,
        "minimum_local_clear": MIN_LOCAL_CLEAR,
        "minimum_global_clear": 1,
        "object_exclusion_halo_radius": HALO_RADIUS,
        "tubelets_per_encoder_block": TUBELETS_PER_ENCODER_BLOCK,
        "all_hole_mean_temporal_donors": _fraction(int(valid.sum()), n_holes),
        "all_hole_mean_aligned_donors": _fraction(int(aligned_valid.sum()), n_holes),
        "all_hole_mean_same_block_donors": _fraction(int(same_counts.sum()), n_holes),
        "all_hole_mean_cross_block_donors": _fraction(int(cross_counts.sum()), n_holes),
        "all_hole_mean_aligned_same_block_donors": _fraction(int(aligned_same_counts.sum()), n_holes),
        "all_hole_mean_aligned_cross_block_donors": _fraction(int(aligned_cross_counts.sum()), n_holes),
        "outside_hole_bitwise_preserved": True,
        "destination_bitwise_preserved": True,
        "identity_request": dx == 0,
    }
    return {
        "variants": variants,
        "masks": masks,
        "donors": {
            "hole_indices": indices, "valid": valid, "aligned_valid": aligned_valid,
            "local_clear_count": local_counts, "global_clear_count": global_counts,
            "alignment_mode": alignment_mode, "donor_count": donor_counts,
            "aligned_donor_count": aligned_counts, "same_block_donor_count": same_counts,
            "cross_block_donor_count": cross_counts,
            "aligned_same_block_donor_count": aligned_same_counts,
            "aligned_cross_block_donor_count": aligned_cross_counts,
            "temporal_values": temporal_values, "aligned_values": aligned_values,
        },
        "diagnostics": diagnostics,
    }


def self_check() -> dict[str, Any]:
    """Analytic tests of recovery, failure fallback, immutable reads and gates."""
    rng = np.random.default_rng(9)
    steps, height, width, dim = 10, 12, 18, 4
    background = rng.standard_normal((height, width, dim)).astype(np.float32)
    biases = np.arange(steps, dtype=np.float32)[:, None] * np.array([0.2, -0.1, 0.3, 0.5], np.float32)
    original = background[None] + biases[:, None, None]
    source = np.zeros((steps, height, width), np.float32)
    distractor = np.zeros_like(source)
    for t in range(steps):
        source[t, 4, 2 + t] = 1
        distractor[t, 10, 15] = 1
    original[source > 0] = 90
    original[distractor > 0] = -90
    snapshot = original.copy()
    out = build_repairs(original, source, distractor, 2)
    h = out["masks"]["hole"]
    truth = background[None] + biases[:, None, None]
    np.testing.assert_allclose(out["variants"]["aligned_temporal_mean"][h], truth[h], atol=2e-6, rtol=2e-6)
    assert np.mean((out["variants"]["temporal_mean"][h] - truth[h]) ** 2) > 0.01
    assert np.array_equal(original, snapshot), "Operator mutated original source"
    assert out["diagnostics"]["temporal_coverage"] == 1.0
    assert out["diagnostics"]["all_hole_mean_cross_block_donors"] > 0
    for t in range(steps):
        for arm in out["variants"].values():
            np.testing.assert_array_equal(arm[t, 4, 4 + t], original[t, 4, 2 + t])
    # Identity must survive arbitrary nonzero masks without donor computation.
    identity = build_repairs(original, source, distractor, 0)
    assert identity["diagnostics"]["hole_count"] == 0
    for arm in identity["variants"].values():
        np.testing.assert_array_equal(arm, original)

    # Permanently occluded same-cell background: zero donors must preserve the
    # exact old fill, not silently exclude the cell or borrow object features.
    stationary_source = np.zeros_like(source)
    stationary_source[:, 4, 5] = 1
    stationary = build_repairs(original, stationary_source, distractor, 1)
    assert stationary["diagnostics"]["temporal_covered_holes"] == 0
    for name in ("temporal_mean", "aligned_temporal_mean"):
        np.testing.assert_array_equal(stationary["variants"][name], stationary["variants"]["naive"])

    # A globally aligned donor remains usable when a crowded local square has
    # no four-point reference set. Its exact feature is analytically known.
    s = np.zeros((2, 18, 18), np.float32)
    d = np.zeros_like(s)
    s[0, 8, 8] = 1
    s[1, 0, 0] = 1
    d[0, 3:14, 3:14] = 1
    z = np.full((2, 18, 18, 2), 3.0, np.float32)
    z[1] = 9.0
    z[s > 0] = 70
    z[d > 0] = -70
    global_result = build_repairs(z, s, d, 1)
    i = np.flatnonzero(np.all(global_result["donors"]["hole_indices"] == [0, 8, 8], axis=1))[0]
    assert global_result["donors"]["alignment_mode"][i, 1] == 2
    assert global_result["donors"]["local_clear_count"][i, 1] == 0
    np.testing.assert_array_equal(global_result["variants"]["aligned_temporal_mean"][0, 8, 8], [3, 3])

    # A same-cell donor can exist even when no common clear reference exists.
    # Plain transport uses it; aligned transport rejects it and keeps naive.
    s = np.zeros((2, 5, 5), np.float32)
    d = np.ones_like(s)
    s[0, 2, 2], s[1, 0, 4] = 1, 1
    d[0, 0, 0] = 0
    d[1, 1:4, 1:4] = 0
    z = rng.standard_normal((2, 5, 5, 2)).astype(np.float32)
    rejected = build_repairs(z, s, d, 1)
    i = np.flatnonzero(np.all(rejected["donors"]["hole_indices"] == [0, 2, 2], axis=1))[0]
    assert rejected["donors"]["valid"][i, 1]
    assert rejected["donors"]["alignment_mode"][i, 1] == 3
    assert not rejected["donors"]["aligned_valid"][i, 1]
    np.testing.assert_array_equal(rejected["variants"]["aligned_temporal_mean"][0, 2, 2],
                                  rejected["variants"]["naive"][0, 2, 2])
    return {
        "status": "passed", "analytic_recovery": True, "unaligned_bias_detected": True,
        "destination_exact": True, "source_immutable": True, "zero_shift_identity": True,
        "zero_donor_exact_fallback": True, "global_fallback": True,
        "no_global_reference_rejection": True, "cross_encoder_block_donors": True,
        "fixture_scope": "synthetic analytic arrays only; no experiment train/dev/test data",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Run analytic NumPy fixtures")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("Use --self-check, or import build_repairs from this module")
    print(json.dumps(self_check(), indent=2))
