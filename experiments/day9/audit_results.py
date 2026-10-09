"""Independent Day9 audit, run only after the fresh-test evaluator completes.

No experiment, operator, scorer, or readout helper is imported. Reconstructs
all full 1024-D edits from immutable source caches, reruns the frozen pointwise
readout using explicit Torch functional layers, and recounts every saved
metric, donor record, bootstrap and gate. The region/semantic implementations
are carried forward from the separate Day8 auditor, not its evaluator.
Only --out is written. No encoder is loaded and no hyperparameter is selected.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
import numpy as np

REPO = Path(__file__).resolve().parents[2]
REGIONS = ("source_hole", "destination", "halo", "distractor", "background", "common")
ARMS = ("noop", "naive", "temporal_mean", "aligned_temporal_mean", "dev_selected", "genuine_target")
KINDS = ("temporal_mean", "aligned_temporal_mean")
WEIGHT_SHA = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
SUMMARY_METRICS = ("source_hole_ratio", "source_hole_mse", "source_hole_noop_mse", "destination_ratio",
    "destination_mse", "balanced_region_ratio", "hole_rgb_mse", "hole_ghost_mean_occupancy", "hole_ghost_fraction_above_half",
    "selected_centroid_error_px", "selected_iou", "selected_rgb_core_mse", "selected_temporal_velocity_error_px",
    "selected_missing_tubelets", "appearance_identity_accuracy", "appearance_identity_eligible", "appearance_identity_skipped",
    "appearance_identity_correct", "distractor_centroid_error_px", "distractor_rgb_change_mse", "distractor_occupancy_change_mse",
    "distractor_iou", "distractor_missing_tubelets", "distractor_temporal_velocity_error_px",
    "outside_hole_vs_naive_max_abs_delta", "outside_hole_vs_naive_changed_tokens",
    "destination_vs_naive_max_abs_delta", "destination_vs_naive_changed_tokens")

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values, dtype=np.float64)) if values else None


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = []
        self.warnings = []
        self.max_numeric_abs_difference = 0.0

    def require(self, condition, label):
        self.count += 1
        if not condition:
            raise AssertionError(label)

    def equal(self, actual, expected, label, rtol=2e-6, atol=2e-7):
        self.count += 1
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                self.errors.append(label + ": dictionary keys differ")
                return
            for key in expected:
                self.equal(actual[key], expected[key], label + "/" + key, rtol, atol)
            return
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.errors.append(label + ": sequence length/type differs")
                return
            for index, value in enumerate(expected):
                self.equal(actual[index], value, f"{label}/{index}", rtol, atol)
            return
        if expected is None:
            okay = actual is None or actual == ""
        elif isinstance(expected, (bool, str)):
            okay = actual == expected
        elif isinstance(expected, (int, float, np.integer, np.floating)):
            try:
                value = float(actual)
                difference = abs(value - float(expected))
                self.max_numeric_abs_difference = max(self.max_numeric_abs_difference, difference)
                okay = np.isfinite(value) and (value == expected if isinstance(expected, (int, np.integer))
                                               else np.isclose(value, expected, rtol=rtol, atol=atol))
            except (TypeError, ValueError):
                okay = False
        else:
            okay = actual == expected
        if not okay and len(self.errors) < 100:
            self.errors.append(f"{label}: actual={actual!r}, expected={expected!r}")

    def array(self, actual, expected, label, exact=False):
        self.require(actual.shape == expected.shape, label + ": shape")
        self.require(np.array_equal(actual, expected) if exact else
                     np.allclose(actual, expected, rtol=2e-6, atol=2e-7), label + ": values")


def translate(array, dx):
    result = np.zeros_like(array)
    if dx >= 0:
        if dx < array.shape[2]:
            result[:, :, dx:] = array[:, :, :array.shape[2] - dx]
    elif -dx < array.shape[2]:
        result[:, :, :dx] = array[:, :, -dx:]
    return result


def regions_from_coverage(source_fraction, target_fraction, distractor_fraction, dx, checks):
    checks.array(target_fraction, translate(source_fraction, dx), "target coverage translation")
    source, target, distractor = source_fraction > 0, target_fraction > 0, distractor_fraction > 0
    union = source | target
    padded = np.pad(union, ((0, 0), (1, 1), (1, 1)))
    support = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2)).any(axis=(-1, -2))
    return dict(source_hole=source & ~target, destination=target, halo=support & ~union,
                distractor=distractor, background=~(support | distractor),
                common=source | target | translate(source, -dx) | distractor | translate(distractor, dx)), support


def latent_metrics(error, noop, regions):
    output = {}
    for name, mask in regions.items():
        n = int(np.count_nonzero(mask))
        total, denominator = float(error[mask].sum(dtype=np.float64)), float(noop[mask].sum(dtype=np.float64))
        mse, reference = (total / n, denominator / n) if n else (None, None)
        output.update({name + "_tokens": n, name + "_squared_error_sum": total,
            name + "_noop_squared_error_sum": denominator, name + "_mse": mse,
            name + "_noop_mse": reference, name + "_ratio": mse / max(reference, 1e-12) if n else None,
            name + "_degenerate": int(not n or reference < 1e-12)})
    output["primary_ratio"] = mean(output[n + "_ratio"] for n in ("source_hole", "destination"))
    return output


def preservation(error, maximum, outside):
    n = int(outside.sum())
    total = float(error[outside].sum(dtype=np.float64))
    return {"outside_support_tokens": n, "outside_support_source_mse": total / n if n else None,
            "outside_support_source_squared_error_sum": total,
            "outside_support_max_abs_delta": float(maximum[outside].max()) if n else None,
            "outside_support_changed_tokens": int(np.count_nonzero(maximum[outside]))}


def lane(fraction):
    y = (np.arange(24, dtype=np.float64) + .5) * 16
    upper = float((fraction * y[None, :, None]).sum() / fraction.sum()) < 192
    return np.broadcast_to((y < 192 if upper else y >= 192)[:, None], (24, 24))


def center(weights, scoring_lane=None):
    w = np.asarray(weights, dtype=np.float64)
    if scoring_lane is not None:
        w = np.where(scoring_lane & (w >= .25), w, 0)
    mass = float(w.sum())
    if mass == 0:
        return None, None, mass
    y, x = np.indices(w.shape, dtype=np.float64)
    return float((w * (x + .5) * 16).sum() / mass), float((w * (y + .5) * 16).sum() / mass), mass


def distance(a, b):
    return 384.0 if a[0] is None or b[0] is None else float(np.hypot(a[0] - b[0], a[1] - b[1]))


def semantic_step(occ, rgb, original_occ, original_rgb, cached, t, regions, lanes):
    row = {}
    color_error = ((rgb.astype(np.float64) - cached["rgb_target"][t]) ** 2).mean(axis=-1)
    for label, fraction, scoring_lane in (
            ("selected", cached["target_frac"][t], lanes[0]),
            ("distractor", cached["distractor_frac"][t], lanes[1])):
        predicted, truth = center(occ, scoring_lane), center(fraction)
        binary, actual = (occ >= .5) & scoring_lane, (fraction >= .5) & scoring_lane
        intersection, union = int((binary & actual).sum()), int((binary | actual).sum())
        core = fraction >= .7
        row.update({label + "_centroid_error_px": distance(predicted, truth),
            label + "_centroid_present": int(predicted[0] is not None),
            label + "_centroid_x": predicted[0], label + "_centroid_y": predicted[1],
            label + "_centroid_mass": predicted[2], label + "_gt_centroid_x": truth[0],
            label + "_gt_centroid_y": truth[1], label + "_iou": intersection / union if union else 1.0,
            label + "_iou_intersection": intersection, label + "_iou_union": union,
            label + "_rgb_core_mse": float(color_error[core].mean()) if core.any() else None,
            label + "_rgb_core_error_sum": float(color_error[core].sum()), label + "_rgb_core_tokens": int(core.sum())})
    hole, other = regions["source_hole"][t], regions["distractor"][t]
    before, after = center(original_occ, lanes[1]), center(occ, lanes[1])
    row.update({"source_hole_ghost_mean_occupancy": float(occ[hole].mean()) if hole.any() else None,
        "source_hole_ghost_fraction_above_half": float((occ[hole] >= .5).mean()) if hole.any() else None,
        "distractor_occupancy_change_mse": float(((occ[other] - original_occ[other]) ** 2).mean()) if other.any() else None,
        "distractor_rgb_change_mse": float(((rgb[other] - original_rgb[other]) ** 2).mean()) if other.any() else None,
        "distractor_centroid_change_px": 0.0 if before[0] is None and after[0] is None else distance(after, before)})
    cores = [cached[k][t] >= .7 for k in ("source_frac", "target_frac", "distractor_frac")]
    separation = selected_distance = other_distance = None
    eligible, correct = False, None
    if all(mask.any() for mask in cores):
        selected_color = cached["rgb_source"][t][cores[0]].astype(np.float64).mean(axis=0)
        other_color = cached["rgb_source"][t][cores[2]].astype(np.float64).mean(axis=0)
        predicted_color = rgb[cores[1]].astype(np.float64).mean(axis=0)
        separation = float(np.linalg.norm(selected_color - other_color))
        selected_distance = float(np.linalg.norm(predicted_color - selected_color))
        other_distance = float(np.linalg.norm(predicted_color - other_color))
        eligible = separation >= .15
        correct = int(selected_distance < other_distance) if eligible else None
    row.update({"appearance_identity_eligible": int(eligible), "appearance_identity_skipped": int(not eligible),
        "appearance_identity_correct": correct, "appearance_identity_accuracy": float(correct) if eligible else None,
        "appearance_true_color_separation": separation, "appearance_distance_to_selected_color": selected_distance,
        "appearance_distance_to_distractor_color": other_distance})
    return row


def semantic_clip(steps):
    output = {key: mean(s[key] for s in steps) for key in steps[0]}
    for key in ("appearance_identity_eligible", "appearance_identity_skipped", "appearance_identity_correct"):
        output[key] = sum(s[key] or 0 for s in steps)
    output["appearance_identity_accuracy"] = (output["appearance_identity_correct"] / output["appearance_identity_eligible"]
                                              if output["appearance_identity_eligible"] else None)
    for label in ("selected", "distractor"):
        for suffix in ("rgb_core_error_sum", "rgb_core_tokens", "iou_intersection", "iou_union"):
            key = label + "_" + suffix
            output[key] = sum(s[key] for s in steps)
        output[label + "_missing_tubelets"] = sum(not s[label + "_centroid_present"] for s in steps)
        velocities = []
        for before, after in zip(steps, steps[1:]):
            if not before[label + "_centroid_present"] or not after[label + "_centroid_present"]:
                velocities.append(384.0)
            else:
                components = [after[label + "_centroid_" + axis] - before[label + "_centroid_" + axis]
                    - after[label + "_gt_centroid_" + axis] + before[label + "_gt_centroid_" + axis] for axis in ("x", "y")]
                velocities.append(float(np.linalg.norm(components)))
        output[label + "_temporal_velocity_error_px"] = mean(velocities)
    return output



def reference_edits(z, source_frac, other_frac, dx):
    """Separate reference loops implementing the prose-defined interventions."""
    z = np.asarray(z, np.float32)
    selected = source_frac > 0
    destination = translate(selected, dx)
    hole = selected & ~destination
    occupied = selected | (other_frac > 0)
    padded = np.pad(occupied, ((0, 0), (1, 1), (1, 1)))
    clear = ~np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2)).any(axis=(-1, -2))
    ntime, height, width, channels = z.shape
    naive = z.copy()
    if dx:
        # Only the source-exclusive hole needs a fill; overlap is overwritten
        # by the destination snapshot. Enumerate spatial samples row-major.
        for time, yy, xx in np.argwhere(hole):
            y0, y1 = max(0, yy-2), min(height, yy+3)
            x0, x1 = max(0, xx-2), min(width, xx+3)
            eligible = ~occupied[time, y0:y1, x0:x1]
            if eligible.any():
                samples = z[time, y0:y1, x0:x1][eligible]
            else:
                samples = z[time][~occupied[time]]
            if not len(samples):
                raise AssertionError('No background for naive fill')
            naive[time, yy, xx] = samples.mean(axis=0, dtype=np.float32)
        for time, yy, xx in np.argwhere(destination):
            naive[time, yy, xx] = z[time, yy, xx-dx]
    indices = np.argwhere(hole).astype(np.int32)
    n = len(indices)
    valid = np.zeros((n, ntime), bool)
    aligned_valid = valid.copy()
    local_counts = np.full((n, ntime), -1, np.int32)
    global_counts = local_counts.copy()
    modes = np.zeros((n, ntime), np.uint8)
    plain_values, aligned_values = naive[hole].copy(), naive[hole].copy()
    for i, (time, yy, xx) in enumerate(indices):
        donor_times = [u for u in range(ntime) if u != time and clear[u, yy, xx]]
        valid[i, donor_times] = True
        if donor_times:
            plain_values[i] = z[donor_times, yy, xx].mean(axis=0, dtype=np.float32)
        y0, y1 = max(0, yy-4), min(height, yy+5)
        x0, x1 = max(0, xx-4), min(width, xx+5)
        accumulated = np.zeros(channels, np.float32)
        n_aligned = 0
        for u in donor_times:
            common = clear[time] & clear[u]
            nearby = common[y0:y1, x0:x1]
            lc, gcnt = int(nearby.sum()), int(common.sum())
            local_counts[i, u], global_counts[i, u] = lc, gcnt
            if lc >= 4:
                delta = z[time, y0:y1, x0:x1][nearby] - z[u, y0:y1, x0:x1][nearby]
                modes[i, u] = 1
            elif gcnt:
                delta = z[time][common] - z[u][common]
                modes[i, u] = 2
            else:
                modes[i, u] = 3
                continue
            adjustment = delta.mean(axis=0, dtype=np.float32)
            accumulated += z[u, yy, xx] + adjustment
            aligned_valid[i, u] = True
            n_aligned += 1
        if n_aligned:
            aligned_values[i] = accumulated / np.float32(n_aligned)
    plain, aligned = naive.copy(), naive.copy()
    plain[hole], aligned[hole] = plain_values, aligned_values
    same = indices[:, 0, None] // 8 == np.arange(ntime)[None, :] // 8
    vectors = {
        'donor_count': valid.sum(1, dtype=np.int32),
        'aligned_donor_count': aligned_valid.sum(1, dtype=np.int32),
        'same_block_donor_count': (valid & same).sum(1, dtype=np.int32),
        'cross_block_donor_count': (valid & ~same).sum(1, dtype=np.int32),
        'aligned_same_block_donor_count': (aligned_valid & same).sum(1, dtype=np.int32),
        'aligned_cross_block_donor_count': (aligned_valid & ~same).sum(1, dtype=np.int32)}
    masks = {'source': selected, 'destination': destination, 'hole': hole, 'clear': clear}
    for key, values in vectors.items():
        dense = np.zeros(hole.shape, np.int32)
        dense[hole] = values
        masks[key] = dense
    masks['donor_covered'] = masks['donor_count'] > 0
    masks['aligned_donor_covered'] = masks['aligned_donor_count'] > 0
    donors = {'hole_indices': indices, 'valid': valid, 'aligned_valid': aligned_valid,
        'local_clear_count': local_counts, 'global_clear_count': global_counts,
        'alignment_mode': modes, **vectors, 'temporal_values': plain_values, 'aligned_values': aligned_values}
    fraction = lambda numerator: float(numerator / n) if n else 0.0
    diagnostic = {
        'operator_version': 'day9_source_only_temporal_hole_v1', 'hole_count': n,
        'temporal_covered_holes': int((vectors['donor_count'] > 0).sum()),
        'aligned_covered_holes': int((vectors['aligned_donor_count'] > 0).sum()),
        'temporal_coverage': fraction((vectors['donor_count'] > 0).sum()),
        'aligned_coverage': fraction((vectors['aligned_donor_count'] > 0).sum()),
        'temporal_fallback_holes': int((vectors['donor_count'] == 0).sum()),
        'aligned_fallback_holes': int((vectors['aligned_donor_count'] == 0).sum()),
        'temporal_donor_pairs': int(valid.sum()), 'aligned_donor_pairs': int(aligned_valid.sum()),
        'aligned_local_donor_pairs': int((modes == 1).sum()), 'aligned_global_donor_pairs': int((modes == 2).sum()),
        'aligned_rejected_donor_pairs': int((modes == 3).sum()), 'alignment_radius': 4,
        'minimum_local_clear': 4, 'minimum_global_clear': 1, 'object_exclusion_halo_radius': 1,
        'tubelets_per_encoder_block': 8,
        'all_hole_mean_temporal_donors': fraction(valid.sum()),
        'all_hole_mean_aligned_donors': fraction(aligned_valid.sum()),
        'all_hole_mean_same_block_donors': fraction(vectors['same_block_donor_count'].sum()),
        'all_hole_mean_cross_block_donors': fraction(vectors['cross_block_donor_count'].sum()),
        'all_hole_mean_aligned_same_block_donors': fraction(vectors['aligned_same_block_donor_count'].sum()),
        'all_hole_mean_aligned_cross_block_donors': fraction(vectors['aligned_cross_block_donor_count'].sum()),
        'outside_hole_bitwise_preserved': True, 'destination_bitwise_preserved': True,
        'identity_request': dx == 0}
    return {'naive': naive, 'temporal_mean': plain, 'aligned_temporal_mean': aligned}, masks, donors, diagnostic


def apply_choice(variants, hole, choice):
    chosen = variants['naive'].copy()
    alpha = choice['alpha']
    if alpha == 1:
        chosen[hole] = variants[choice['kind']][hole]
    elif alpha:
        chosen[hole] += np.float32(alpha) * (variants[choice['kind']][hole] - variants['naive'][hole])
    return chosen


def error_map(z, target):
    return np.square(z-target).mean(axis=-1, dtype=np.float32)


def average_mask(values, mask):
    return float(values[mask].mean(dtype=np.float64)) if mask.any() else None


def coverage(masks, donors, t=None):
    hole = masks['hole'] if t is None else masks['hole'][t]
    out = {}
    for key in ('donor_count', 'aligned_donor_count', 'same_block_donor_count', 'cross_block_donor_count',
                'aligned_same_block_donor_count', 'aligned_cross_block_donor_count'):
        values = (masks[key] if t is None else masks[key][t])[hole]
        out[key+'_mean'] = float(values.mean()) if values.size else None
        if key in ('donor_count', 'aligned_donor_count'):
            prefix = 'temporal' if key == 'donor_count' else 'aligned'
            out[prefix+'_coverage'] = float((values > 0).mean()) if values.size else None
            out[prefix+'_fallback_tokens'] = int(np.count_nonzero(values == 0))
            for n in range(17):
                out[f'{prefix}_tokens_with_{n}_donors'] = int(np.count_nonzero(values == n))
    modes = donors['alignment_mode'] if t is None else donors['alignment_mode'][donors['hole_indices'][:, 0] == t]
    for label, value in (('local', 1), ('global', 2), ('rejected', 3)):
        out[f'aligned_{label}_donor_pairs'] = int(np.count_nonzero(modes == value))
    return out


def extra(occ, rgb, rgb_target, delta, regions):
    color = np.square(rgb.astype(np.float64)-rgb_target).mean(axis=-1)
    hole, destination = regions['source_hole'], regions['destination']
    return {'hole_rgb_mse': average_mask(color, hole), 'hole_ghost_mean_occupancy': average_mask(occ, hole),
        'hole_ghost_fraction_above_half': average_mask(occ >= .5, hole),
        'outside_hole_vs_naive_max_abs_delta': float(delta[~hole].max()),
        'outside_hole_vs_naive_changed_tokens': int(np.count_nonzero(delta[~hole])),
        'destination_vs_naive_max_abs_delta': float(delta[destination].max()),
        'destination_vs_naive_changed_tokens': int(np.count_nonzero(delta[destination]))}


def direct_probe(state, z):
    """Independent inference through known frozen pointwise readout weights."""
    import torch
    import torch.nn.functional as F
    flat = z.reshape(-1, z.shape[-1])
    result = np.empty((len(flat), 4), np.float32)
    with torch.inference_mode():
        for start in range(0, len(flat), 4096):
            x = torch.from_numpy(np.ascontiguousarray(flat[start:start+4096]))
            x = (x-state['feature_mean']) / state['feature_std']
            x = F.gelu(F.linear(x, state['network.0.weight'], state['network.0.bias']))
            x = torch.sigmoid(F.linear(x, state['network.2.weight'], state['network.2.bias']))
            result[start:start+len(x)] = x.numpy()
    return result.reshape(*z.shape[:-1], 4)


def resample(values):
    x = np.asarray([v for v in values if v is not None], np.float64)
    if not len(x):
        return {'mean': None, 'ci95': [None, None], 'n_scenes': 0}
    if not np.isfinite(x).all():
        raise AssertionError('Nonfinite bootstrap metric')
    samples = np.random.default_rng(1909).integers(len(x), size=(5000, len(x)))
    estimates = x[samples].mean(axis=1)
    return {'mean': float(x.mean()), 'ci95': np.percentile(estimates, [2.5, 97.5]).tolist(), 'n_scenes': len(x)}


def recount_summary(rows, saved, choice, checks):
    table = {(r['scene'], r['arm']): r for r in rows}
    methods, differences = {}, {}
    for group in ('all', 'seen_magnitude', 'heldout_magnitude'):
        members = [r for r in rows if group == 'all' or r['shift_regime'] == group]
        methods[group], differences[group] = {}, {}
        for arm in ARMS:
            selected = [r for r in members if r['arm'] == arm]
            methods[group][arm] = {key: resample([r[key] for r in selected]) for key in SUMMARY_METRICS}
            if arm not in ('naive', 'genuine_target'):
                differences[group][arm+'_minus_naive'] = {
                    key: resample([r[key]-table[(r['scene'], 'naive')][key] for r in selected
                                    if r[key] is not None and table[(r['scene'], 'naive')][key] is not None])
                    for key in SUMMARY_METRICS}
    decision = {}
    for arm in (*KINDS, 'dev_selected'):
        diff = differences['all'][arm+'_minus_naive']
        decision[arm] = {
            'hole_latent_improves_paired_ci': diff['source_hole_ratio']['ci95'][1] < 0,
            'mean_hole_rgb_not_worse': diff['hole_rgb_mse']['mean'] <= 0,
            'mean_hole_ghost_not_worse': diff['hole_ghost_mean_occupancy']['mean'] <= 0,
            'outside_hole_exactly_preserved': methods['all'][arm]['outside_hole_vs_naive_max_abs_delta']['mean'] == 0,
            'destination_exactly_preserved': methods['all'][arm]['destination_vs_naive_max_abs_delta']['mean'] == 0}
        decision[arm]['combined_exploratory_pass'] = all(decision[arm].values())
    genuine = methods['all']['genuine_target']
    gate = {'pass': genuine['selected_centroid_error_px']['mean'] < 16 and
            genuine['appearance_identity_accuracy']['mean'] is not None and genuine['appearance_identity_accuracy']['mean'] >= .9,
        'centroid_threshold_px': 16, 'identity_threshold': .9, 'centroid': genuine['selected_centroid_error_px'],
        'identity_accuracy': genuine['appearance_identity_accuracy'],
        'eligible_tubelets': int(sum(r['appearance_identity_eligible'] for r in rows if r['arm'] == 'genuine_target')),
        'skipped_tubelets': int(sum(r['appearance_identity_skipped'] for r in rows if r['arm'] == 'genuine_target'))}
    adoption = choice['alpha'] > 0 and gate['pass'] and decision['dev_selected']['combined_exploratory_pass']
    checks.equal(saved['methods'], methods, 'all method summary metrics and CIs')
    checks.equal(saved['paired_differences'], differences, 'all paired summary differences and CIs')
    checks.equal(saved['decision_checks'], decision, 'all decision checks')
    checks.equal(saved['genuine_readout_gate'], gate, 'genuine semantic validity gate')
    checks.equal(saved['qualified_adoption'], adoption, 'qualified adoption')
    checks.equal(saved['n_independent_scenes'], 16, 'independent scenes')
    checks.equal(saved['chosen'], choice, 'frozen selection')
    checks.equal(saved['bootstrap']['draws'], 5000, 'bootstrap draws')
    checks.equal(saved['bootstrap']['seed'], 1909, 'bootstrap seed')
    checks.equal(saved['degenerate_scene_arm_counts'], {n: sum(r[n+'_degenerate'] for r in rows) for n in REGIONS}, 'degenerate counts')
    return {'primary_difference': differences['all']['dev_selected_minus_naive']['source_hole_ratio'],
        'selected_hole_rgb_difference': differences['all']['dev_selected_minus_naive']['hole_rgb_mse'],
        'selected_hole_ghost_difference': differences['all']['dev_selected_minus_naive']['hole_ghost_mean_occupancy'],
        'genuine_readout_gate': gate, 'decision_checks': decision, 'qualified_adoption': adoption}

def expected_specs(first=12200, count=16):
    specs = []
    for i in range(count):
        heldout = count == 16 and i >= 8
        dx = ((48, -48, 80, -80) if heldout else (32, -32, 64, -64))[i % 4]
        split = 'test' if first == 12200 else 'dev'
        specs.append({'name': f'{split}_{first+i}_dx{dx:+d}', 'seed': first+i, 'dx': dx, 'split': split,
                      'shift_regime': 'heldout_magnitude' if heldout else 'seen_magnitude'})
    return specs


def check_provenance(args, checks):
    frozen = read_json(args.freeze)
    receipt = read_json(args.publication)
    manifest = read_json(args.features/'manifest.json')
    evaluation = read_json(args.test/'evaluation_manifest.json')
    summary = read_json(args.test/'summary.json')
    checks.equal(frozen['test_accessed'], False, 'freeze pretest marker')
    checks.equal(frozen['test_specs'], expected_specs(), 'independent fresh split')
    checks.equal(manifest['test_freeze_sha256'], sha(args.freeze), 'feature freeze binding')
    checks.equal(receipt['freeze_sha256'], sha(args.freeze), 'publication freeze binding')
    checks.equal(receipt['bytes_equal_to_GitHub'], True, 'recorded GitHub byte verification')
    checks.equal(receipt['test_encoded_before_verification'], False, 'publication precedes test marker')
    checks.require(isinstance(receipt['commit'], str) and len(receipt['commit']) == 40, 'publication commit SHA shape')
    for name, digest in frozen['source_hashes'].items():
        full = (REPO/name).resolve()
        checks.require(full.is_relative_to(REPO), 'source remains inside repository')
        checks.equal(sha(full), digest, 'frozen source '+name)
    checks.equal(manifest['source_hashes'], frozen['source_hashes'], 'feature source manifest')
    bound = {}
    for key in ('dev_selection', 'runtime_validation'):
        path = Path(frozen[key]['path'])
        if not path.is_absolute():
            path = args.freeze.parent/path
        checks.equal(sha(path), frozen[key]['sha256'], key+' file hash')
        bound[key] = read_json(path)
    selection, precision = bound['dev_selection'], bound['runtime_validation']
    checks.equal(selection['test_accessed'], False, 'selection pretest marker')
    checks.equal(selection['chosen'], frozen['selected'], 'selection freeze choice')
    checks.equal(selection['probe_sha256'], sha(args.probe), 'development readout hash')
    checks.equal(frozen['probe_sha256'], sha(args.probe), 'freeze readout hash')
    for name, digest in selection['source_hashes'].items():
        checks.equal(digest, frozen['source_hashes'][name], 'selection source '+name)
    for field in ('model', 'upstream', 'encoder_context', 'extraction_device', 'inference_precision', 'cache_precision', 'weights'):
        checks.equal(manifest[field], selection['encoder_provenance'][field], 'encoder continuity '+field)
    checks.equal(manifest['upstream'], '204698b45b3712590f06245fbfba32d3be539812', 'official upstream')
    checks.equal(manifest['weights'][0]['sha256'], WEIGHT_SHA, 'official encoder weight')
    checks.equal(manifest['extraction_device'], 'cpu', 'CPU extraction')
    checks.equal(manifest['inference_precision'], 'bfloat16', 'BF16 inference')
    checks.equal(manifest['cache_precision'], 'float16', 'FP16 cache')
    checks.equal(precision['pass'], True, 'restored runtime gate')
    checks.equal(precision['test_data_opened'], False, 'runtime validation pretest marker')
    checks.equal(precision['chosen_precision'], 'bfloat16', 'runtime chosen precision')
    checks.equal(precision['checkpoint_sha256'], WEIGHT_SHA, 'runtime gate checkpoint')
    signal = precision['comparison_to_colab_fp32']
    checks.equal(signal['passed_checks'], 6, 'six runtime fidelity checks')
    checks.equal(signal['tolerance'], .01, 'unchanged precision threshold')
    for region in ('global', 'source_hole', 'destination'):
        row = signal['metrics'][region]
        checks.require(row['true_edit_mse'] > 1e-12 and row['denominator_floored'] is False,
                       'positive precision reference signal '+region)
        for side in ('source', 'target'):
            ratio = row[side+'_precision_mse']/row['true_edit_mse']
            checks.equal(row[side+'_ratio'], ratio, 'precision ratio arithmetic '+region+'/'+side)
            checks.require(ratio < .01, 'precision signal threshold '+region+'/'+side)
    checks.equal(evaluation['freeze'], frozen, 'evaluation embedded freeze')
    checks.equal(evaluation['input_feature_manifest'], manifest, 'evaluation input manifest')
    checks.equal(evaluation['binding'], summary['binding'], 'summary/evaluation binding')
    binding = summary['binding']
    checks.equal(binding['freeze_sha256'], sha(args.freeze), 'summary freeze SHA')
    checks.equal(binding['feature_manifest_sha256'], sha(args.features/'manifest.json'), 'summary input SHA')
    checks.equal(binding['probe_sha256'], sha(args.probe), 'summary readout SHA')
    checks.equal(binding['dev_selection_sha256'], frozen['dev_selection']['sha256'], 'summary dev SHA')
    checks.equal(binding['source_hashes'], selection['source_hashes'], 'summary sources')
    checks.equal(evaluation['row_counts'], {'per_clip': 96, 'per_tubelet': 1536, 'coverage_strata': 960}, 'declared row counts')
    records = {r['spec']['name']: r for r in manifest['clips']}
    checks.require(len(manifest['clips']) == len(records) == 16, 'exactly sixteen unique fresh inputs')
    checks.equal([r['spec'] for r in manifest['clips']], expected_specs(), 'fresh cache ordered split')
    artifacts = {r['scene']: r for r in evaluation['artifacts']}
    checks.require(len(evaluation['artifacts']) == len(artifacts) == 16, 'exactly sixteen output NPZs')
    checks.require(set(artifacts) == set(records), 'output/input scene bijection')
    for name, record in records.items():
        checks.equal(sha(args.features/record['path']), record['sha256'], 'raw input SHA '+name)
        checks.equal((args.features/record['path']).stat().st_size, record['bytes'], 'raw input bytes '+name)
        checks.equal(sha(args.test/artifacts[name]['path']), artifacts[name]['sha256'], 'output evidence SHA '+name)
        checks.equal((args.test/artifacts[name]['path']).stat().st_size, artifacts[name]['bytes'], 'output evidence bytes '+name)
        checks.equal(artifacts[name]['feature_cache_sha256'], record['sha256'], 'output input binding '+name)
    return frozen, selection, manifest, summary, evaluation


def recount_development(args, selection, checks):
    """Recount the fixed old-dev table; never select using fresh test output."""
    manifest = read_json(args.dev_features/'manifest.json')
    checks.equal(sha(args.dev_features/'manifest.json'), selection['dev_manifest_sha256'], 'old development manifest SHA')
    records = {r['spec']['name']: r for r in manifest['clips']}
    bound = {r['spec']['name']: r for r in selection['dev_cache_bindings']}
    checks.require(set(bound) == {s['name'] for s in expected_specs(11100, 8)}, 'exact eight dev scenes')
    published = {(r['scene'], r['kind'], r['alpha']): r for r in selection['per_scene_candidates']}
    checks.require(len(published) == len(selection['per_scene_candidates']) == 80, 'eighty unique frozen dev candidates')
    rows = []
    for scene in expected_specs(11100, 8):
        name = scene['name']
        checks.equal(records[name], bound[name], 'unchanged development binding '+name)
        path = args.dev_features/records[name]['path']
        checks.equal(sha(path), bound[name]['sha256'], 'development raw cache SHA '+name)
        with np.load(path, allow_pickle=False) as cache:
            source, target = cache['source'].astype(np.float32), cache['target'].astype(np.float32)
            variants, masks, donors, _ = reference_edits(source, cache['source_frac'], cache['distractor_frac'], scene['dx']//16)
        hole = masks['hole']
        denominator = average_mask(error_map(source, target), hole)
        for kind in KINDS:
            for alpha in (0., .25, .5, .75, 1.):
                chosen = apply_choice(variants, hole, {'kind': kind, 'alpha': alpha})
                mse = average_mask(error_map(chosen, target), hole)
                row = {'scene': name, 'kind': kind, 'alpha': alpha, 'source_hole_tokens': int(hole.sum()),
                    'source_hole_mse': mse, 'source_hole_noop_mse': denominator,
                    'source_hole_degenerate': int(denominator < 1e-12), 'source_hole_ratio': mse/max(denominator, 1e-12)}
                checks.equal(published[(name, kind, alpha)], row, 'development candidate '+name+'/'+kind+'/'+str(alpha))
                rows.append(row)
        print('AUDIT_DEV', name, flush=True)
        del source, target, variants, masks, donors, chosen
        gc.collect()
    candidates = [{'kind': kind, 'alpha': alpha, 'mean_source_hole_ratio': mean(
        r['source_hole_ratio'] for r in rows if r['kind'] == kind and r['alpha'] == alpha)}
        for kind in KINDS for alpha in (0., .25, .5, .75, 1.)]
    checks.equal(selection['candidate_means'], candidates, 'independently recounted candidate means')
    minimum = min(r['mean_source_hole_ratio'] for r in candidates)
    winner = sorted([r for r in candidates if r['mean_source_hole_ratio'] <= minimum+1e-12],
                    key=lambda r: (r['alpha'], KINDS.index(r['kind'])))[0]
    checks.equal(selection['chosen'], winner, 'development minimum and tie order')
    return {'scenes': 8, 'candidate_rows': 80, 'chosen': winner}


def compare_row(saved, recounted, checks, label):
    checks.require(set(saved) == set(recounted), label+' column set')
    for key, expected in recounted.items():
        checks.equal(saved[key], expected, label+'/'+key)


def run(args, checks):
    frozen, selection, manifest, summary, evaluation = check_provenance(args, checks)
    development = recount_development(args, selection, checks)
    import torch
    torch.set_num_threads(4)
    payload = torch.load(args.probe, map_location='cpu', weights_only=True)
    checks.equal(payload['version'], 'day8_frozen_pointwise_readout_v1', 'readout version')
    checks.equal(payload['history']['test_accessed'], False, 'readout training never accessed test')
    state = payload['state_dict']
    checks.require(set(state) == {'feature_mean', 'feature_std', 'network.0.weight', 'network.0.bias',
                                   'network.2.weight', 'network.2.bias'}, 'exact readout parameter schema')
    checks.require(state['feature_mean'].shape == (1024,) and state['network.0.weight'].shape == (128, 1024)
                   and state['network.2.weight'].shape == (4, 128), 'frozen readout architecture')
    raw_clips = read_csv(args.test/'per_clip.csv')
    raw_steps = read_csv(args.test/'per_tubelet.csv')
    raw_strata = read_csv(args.test/'coverage_strata.csv')
    clips = {(r['scene'], r['arm']): r for r in raw_clips}
    steps = {(r['scene'], r['arm'], int(r['tubelet'])): r for r in raw_steps}
    strata = {(r['scene'], r['arm'], r['donor_policy'], r['coverage_stratum']): r for r in raw_strata}
    checks.require(len(raw_clips) == len(clips) == 96, '96 unique scene-arm rows')
    checks.require(len(raw_steps) == len(steps) == 1536, '1536 unique tubelet-arm rows')
    checks.require(len(raw_strata) == len(strata) == 960, '960 unique coverage-stratum rows')
    records = {r['spec']['name']: r for r in manifest['clips']}
    artifacts = {r['scene']: r for r in evaluation['artifacts']}
    recounted_rows, per_scene_checks = [], []
    for scene in frozen['test_specs']:
        name, dx = scene['name'], scene['dx']//16
        with np.load(args.features/records[name]['path'], allow_pickle=False) as cache:
            cached = {key: cache[key] for key in cache.files}
        with np.load(args.test/artifacts[name]['path'], allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        source, target = cached['source'].astype(np.float32), cached['target'].astype(np.float32)
        checks.require(source.shape == target.shape == (16, 24, 24, 1024), 'full 1024-D feature shape '+name)
        checks.require(np.isfinite(source).all() and np.isfinite(target).all(), 'finite feature arrays '+name)
        for key in ('source_frac', 'target_frac', 'distractor_frac', 'rgb_source', 'rgb_target'):
            checks.array(saved[key], cached[key], 'saved cache field '+name+'/'+key, exact=True)
        regions, _ = regions_from_coverage(cached['source_frac'], cached['target_frac'], cached['distractor_frac'], dx, checks)
        checks.equal(saved['region_names'].tolist(), list(REGIONS), 'region order '+name)
        checks.array(saved['region_masks'], np.stack([regions[n] for n in REGIONS]), 'fixed regions '+name, exact=True)
        checks.equal(int(saved['dx_pixels']), scene['dx'], 'requested displacement '+name)
        variants, masks, donors, diagnostics = reference_edits(source, cached['source_frac'], cached['distractor_frac'], dx)
        hole, destination = regions['source_hole'], regions['destination']
        checks.array(masks['hole'], hole, 'operator-scoring hole '+name, exact=True)
        for group, mapping in (('masks', masks), ('donors', donors)):
            for key, array in mapping.items():
                checks.array(saved[group+'_'+key], array, 'independent '+group+'/'+key+'/'+name, exact=True)
        checks.array(saved['donor_count'], masks['donor_count'], 'dense donor counts '+name, exact=True)
        checks.equal(json.loads(str(saved['diagnostics_json'])), diagnostics, 'donor diagnostic '+name)
        variants['dev_selected'] = apply_choice(variants, hole, selection['chosen'])
        variants.update(noop=source, genuine_target=target)
        snapshot = translate(source, dx)
        checks.array(variants['naive'][destination], snapshot[destination], 'snapshot destination '+name, exact=True)
        del snapshot
        checks.equal(saved['method_names'].tolist(), list(ARMS), 'arm order '+name)
        noop_error = error_map(source, target)
        checks.array(saved['noop_latent_error'], noop_error, 'full-D no-op error '+name, exact=True)
        source_prediction = direct_probe(state, source)
        lanes = (lane(cached['target_frac']), lane(cached['distractor_frac']))
        scene_counts, step_counts = coverage(masks, donors), [coverage(masks, donors, t) for t in range(16)]
        stratum_masks = {}
        for policy, key in (('temporal', 'donor_count'), ('aligned', 'aligned_donor_count')):
            count = masks[key]
            stratum_masks.update({(policy, 'all_hole'): hole, (policy, 'zero_donors_fallback'): hole & (count == 0),
                (policy, 'covered_any_donor'): hole & (count > 0),
                (policy, 'one_to_three_donors'): hole & (count > 0) & (count <= 3),
                (policy, 'four_or_more_donors'): hole & (count >= 4)})
        arm_hashes = {}
        max_probe_difference = 0.0
        for i, arm in enumerate(ARMS):
            edited = variants[arm]
            arm_hashes[arm] = hashlib.sha256(np.ascontiguousarray(edited).tobytes()).hexdigest()
            checks.equal(str(saved['edited_latents_float32_sha256'][i]), arm_hashes[arm], 'independent full tensor SHA '+name+'/'+arm)
            error = error_map(edited, target)
            delta = np.abs(edited-variants['naive']).max(axis=-1)
            checks.array(saved['latent_error'][i], error, 'full-D latent error '+name+'/'+arm, exact=True)
            checks.array(saved['max_abs_delta_vs_naive'][i], delta, 'full-D maximum change '+name+'/'+arm, exact=True)
            if arm not in ('noop', 'genuine_target'):
                checks.array(edited[~hole], variants['naive'][~hole], 'all outside-hole equality '+name+'/'+arm, exact=True)
                checks.array(edited[destination], variants['naive'][destination], 'destination equality '+name+'/'+arm, exact=True)
            prediction = source_prediction if arm == 'noop' else direct_probe(state, edited)
            # CPU GEMM output may differ by normal last-bit roundoff across
            # Torch builds; the exact edited feature SHA is separately required.
            checks.array(saved['occupancy'][i], prediction[..., 0], 'independent readout occupancy '+name+'/'+arm)
            checks.array(saved['rgb'][i], prediction[..., 1:], 'independent readout RGB '+name+'/'+arm)
            max_probe_difference = max(max_probe_difference,
                float(np.max(np.abs(saved['occupancy'][i]-prediction[..., 0]))),
                float(np.max(np.abs(saved['rgb'][i]-prediction[..., 1:]))))
            # Recount CSVs against exact archived readout arrays, so tolerated
            # inference roundoff cannot move a threshold in the recorded score.
            occ, rgb = saved['occupancy'][i], saved['rgb'][i]
            semantics = [semantic_step(occ[t], rgb[t], saved['occupancy'][0, t], saved['rgb'][0, t],
                                       cached, t, regions, lanes) for t in range(16)]
            identity = {'scene': name, 'scene_seed': scene['seed'], 'dx_pixels': scene['dx'],
                        'shift_regime': scene['shift_regime'], 'arm': arm}
            region_score = latent_metrics(error, noop_error, regions)
            region_score['balanced_region_ratio'] = region_score.pop('primary_ratio')
            row = {**identity, **region_score, **semantic_clip(semantics),
                   **extra(occ, rgb, cached['rgb_target'], delta, regions), **scene_counts}
            compare_row(clips[(name, arm)], row, checks, 'clip '+name+'/'+arm)
            recounted_rows.append(row)
            for t in range(16):
                t_regions = {k: v[t] for k, v in regions.items()}
                metric = latent_metrics(error[t], noop_error[t], t_regions)
                metric['balanced_region_ratio'] = metric.pop('primary_ratio')
                t_row = {**identity, 'tubelet': t, **metric, **semantics[t],
                         **extra(occ[t], rgb[t], cached['rgb_target'][t], delta[t], t_regions), **step_counts[t]}
                compare_row(steps[(name, arm, t)], t_row, checks, f'tubelet {name}/{arm}/{t}')
            rgb_error = np.square(rgb.astype(np.float64)-cached['rgb_target']).mean(axis=-1)
            for (policy, label), mask in stratum_masks.items():
                mse, reference = average_mask(error, mask), average_mask(noop_error, mask)
                strat = {**identity, 'donor_policy': policy, 'coverage_stratum': label, 'tokens': int(mask.sum()),
                    'latent_mse': mse, 'noop_mse': reference, 'latent_ratio': mse/max(reference, 1e-12) if mse is not None else None,
                    'hole_rgb_mse': average_mask(rgb_error, mask), 'hole_ghost_mean_occupancy': average_mask(occ, mask)}
                compare_row(strata[(name, arm, policy, label)], strat, checks, 'stratum '+name+'/'+arm+'/'+policy+'/'+label)
        per_scene_checks.append({'scene': name, 'full_edited_tensor_sha256': arm_hashes,
            'maximum_independent_probe_absolute_difference': max_probe_difference,
            'hole_tokens': int(hole.sum()), 'temporal_coverage': scene_counts['temporal_coverage'],
            'aligned_coverage': scene_counts['aligned_coverage'], 'latent_channels_recomputed': 1024})
        print('AUDIT_TEST', name, 'all full tensor hashes, readouts and rows checked', flush=True)
        del source, target, variants, cached, saved, edited, error, delta, source_prediction, prediction
        gc.collect()
    recounted = recount_summary(recounted_rows, summary, selection['chosen'], checks)
    return {'version': 'day9_independent_full_reconstruction_audit_v1', 'development': development,
        'rows_checked': {'per_clip': len(clips), 'per_tubelet': len(steps), 'coverage_strata': len(strata)},
        'scenes': per_scene_checks, 'recomputed_decision': recounted,
        'bindings': {'freeze_sha256': sha(args.freeze), 'publication_sha256': sha(args.publication),
                     'feature_manifest_sha256': sha(args.features/'manifest.json'),
                     'probe_sha256': sha(args.probe), 'summary_sha256': sha(args.test/'summary.json'),
                     'audit_source_sha256': sha(Path(__file__)), 'publication_commit': read_json(args.publication)['commit']},
        'scope': ['Reconstructed every full 1024-D deterministic edited tensor from source caches; exact SHA matched all 96 saved outputs.',
                  'Independently reran all frozen readouts through explicit functional layers; recounted scores using archived readouts.',
                  'Reconstructed all fixed regions, donor masks, candidate features, alignment modes, counts, coverage and 960 strata.',
                  'Recounted all 96 scene-arm and 1536 tubelet rows including semantic centroids, IoU, RGB, color identity, missing detections and velocity errors.',
                  'Recounted all method and paired 5000-scene-bootstrap summaries, gates, degeneracy and qualified-adoption decision.',
                  'Recomputed all 80 development candidates from the eight original dev caches and verified the predeclared selection rule.'],
        'limits': ['Does not rerun the 5 GB encoder or independently validate whether encoded videos match renderer pixels.',
                   'Verifies recorded official encoder checksum and six saved precision ratios, not new encoder inference.',
                   'Verifies local bytes against the publication receipt and commit identity; does not independently query GitHub or prove publication chronology.',
                   'Successful arithmetic auditing is not proof of realistic video editing, causal propagation, or detailed object identity.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'test', 'freeze', 'publication', 'probe', 'dev-features', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    checks = Checks()
    report = {'version': 'day9_independent_full_reconstruction_audit_v1'}
    try:
        report.update(run(args, checks))
    except Exception as error:
        checks.errors.append(type(error).__name__+': '+str(error))
        report['exception_traceback'] = traceback.format_exc()
    report.update({'status': 'passed' if not checks.errors else 'failed', 'check_count': checks.count,
                   'maximum_scalar_absolute_difference': checks.max_numeric_abs_difference,
                   'errors': checks.errors, 'warnings': checks.warnings})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(json.dumps({'status': report['status'], 'check_count': checks.count, 'errors': checks.errors,
                      'out': str(args.out)}, indent=2), flush=True)
    if checks.errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
