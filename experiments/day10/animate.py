"""Single-image, source-only JEPA translations with an honest coarse readout.

The official encoder's native image branch receives [1,3,1,384,384], not a
repeated video. One supplied mask selects the object. Requested integer-patch
translations are commands, not predicted motion. No target, distractor mask,
hidden background, or renderer metadata is an operator input.

The fixed background estimate is exactly the existing Day8 naive local-fill
method applied once to an immutable image. Reusing it is a consistency property,
not a new repair method. The Day8 frozen probe is a diagnostic, not a decoder.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DAY8 = REPO / 'experiments/day8'
sys.path.insert(0, str(DAY8))
import data
import extract as day8_extract
import evaluate as day8_evaluate
import operators
import probe as day8_probe

VERSION = 'day10_native_single_image_translation_v1'
ARMS = ('noop', 'copy_repair', 'wrong_direction', 'genuine_target')
SIZE, PATCH, GRID, CHANNELS = 384, 16, 24, 1024
FRAME_INDEX = 15
DISPLAY_INDICES = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
sha256 = day8_extract.sha256
write_json = day8_extract.write_json


def test_specs():
    return [{'name': f'image_{13200+i}', 'seed': 13200+i,
             'frame_index': FRAME_INDEX,
             'shifts_px': [((1 if i < 2 else -1) * d) for d in (0, 16, 32, 48, 64, 80)],
             'display_indices': DISPLAY_INDICES} for i in range(4)]


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_npz(path, **arrays):
    """Only expose a complete ZIP to readers of the shared output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='staging_', suffix='.npz', delete=False) as handle:
        staging = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    staging.replace(path)


def save_png(path, pixels):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    Image.fromarray(pixels).save(temporary, format='PNG')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot write an empty metric table')
    fields = sorted(set().union(*(row.keys() for row in rows)))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def image_patch_fractions(mask):
    mask = np.asarray(mask)
    if mask.shape != (SIZE, SIZE) or mask.dtype != np.bool_:
        raise ValueError('Selected mask must be boolean [384,384]')
    return mask.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3), dtype=np.float32)[None]


def image_patch_rgb(image):
    return (np.asarray(image).reshape(GRID, PATCH, GRID, PATCH, 3)
            .mean(axis=(1, 3), dtype=np.float32)[None] / 255).astype(np.float32)


def encode_image(encoder, image, device='cpu'):
    """Official native image modality. Features remain full FP32 in the cache."""
    import torch
    image = np.asarray(image)
    if image.shape != (SIZE, SIZE, 3) or image.dtype != np.uint8:
        raise ValueError('RGB input must be uint8 [384,384,3]; no implicit resizing')
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None, :, None].to(device).float() / 255
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1, 1)
    with torch.inference_mode():
        output = encoder((x - mean) / std)
    if tuple(output.shape) != (1, GRID * GRID, CHANNELS) or not torch.isfinite(output).all():
        raise RuntimeError(f'Unexpected/nonfinite native image features: {tuple(output.shape)}')
    return output.float().cpu().numpy().reshape(1, GRID, GRID, CHANNELS).copy()


def translate_features(source, selected_fraction, shifts_px):
    """Reusable source-only operator; no scoring information is accepted here.

    Each state starts from the same source, erases the original selection using
    one fixed spatial estimate, then copies original selected tokens into the
    destination. Any source/destination overlap is resolved by destination copy.
    Wrong-direction copies are a matched negative control using the same source.
    """
    source = np.asarray(source, dtype=np.float32)
    fraction = np.asarray(selected_fraction, dtype=np.float32)
    if source.shape != (1, GRID, GRID, CHANNELS) or fraction.shape != (1, GRID, GRID):
        raise ValueError('Expected source [1,24,24,1024] and selection [1,24,24]')
    if not np.isfinite(source).all() or not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError('Nonfinite features or invalid selection coverage')
    selected = fraction > 0
    if not selected.any():
        raise ValueError('Selected object mask is empty')
    original = source.copy()
    background = operators.local_background(original, selected, selected)
    fixed_base = original.copy()
    fixed_base[selected] = background[selected]
    outputs = {'noop': [], 'copy_repair': [], 'wrong_direction': []}
    invariant_rows = []
    first_by_shift = {}
    for raw_shift in shifts_px:
        if int(raw_shift) != raw_shift or int(raw_shift) % PATCH:
            raise ValueError('Every displacement must be an integer multiple of 16 pixels')
        shift = int(raw_shift)
        dx = shift // PATCH
        destinations = {}
        states = {}
        for arm, signed_dx in (('copy_repair', dx), ('wrong_direction', -dx)):
            destination = operators.shift_horizontal(selected, signed_dx)
            if int(destination.sum()) != int(selected.sum()):
                raise ValueError(f'{arm} would crop the selected patch support at dx={shift}')
            edited = original.copy() if dx == 0 else fixed_base.copy()
            if dx:
                shifted = operators.shift_horizontal(original, signed_dx)
                edited[destination] = shifted[destination]
            outside = ~(selected | destination)
            hole = selected & ~destination
            if not np.array_equal(edited[outside], original[outside]):
                raise AssertionError('Unrelated source token changed')
            if not np.array_equal(edited[destination], operators.shift_horizontal(original, signed_dx)[destination]):
                raise AssertionError('Destination is not an exact original-source copy')
            if dx and not np.array_equal(edited[hole], background[hole]):
                raise AssertionError('An uncovered background estimate changed across states')
            if not dx and not np.array_equal(edited, original):
                raise AssertionError('Zero displacement is not exact identity')
            states[arm], destinations[arm] = edited, destination
        # This is intentionally the SAME existing naive operator, not a new arm.
        naive, _ = operators._transport(original, fraction, selected, dx)
        if not np.array_equal(states['copy_repair'], naive):
            raise AssertionError('Fixed-background implementation differs from Day8 naive')
        if shift in first_by_shift and not np.array_equal(first_by_shift[shift], states['copy_repair']):
            raise AssertionError('A revisited position changed')
        first_by_shift[shift] = states['copy_repair'].copy()
        outputs['noop'].append(original.copy())
        for arm, value in states.items():
            outputs[arm].append(value)
        invariant_rows.append({'shift_px': shift, 'zero_shift_identity': bool(dx == 0),
                               'outside_union_exact': True, 'destination_copy_exact': True,
                               'uncovered_background_fixed': True, 'day8_naive_equivalent': True,
                               'copy_repair_sha256': array_sha256(states['copy_repair']),
                               'wrong_direction_sha256': array_sha256(states['wrong_direction'])})
    if not outputs['noop']:
        raise ValueError('At least one displacement is required')
    if not np.array_equal(source, original):
        raise AssertionError('Operator modified its source input')
    return {arm: np.stack(values) for arm, values in outputs.items()}, background, invariant_rows


def animate_image(encoder, source_rgb, selected_mask, shifts_px, probe=None, device='cpu'):
    """One RGB image + one object selection + explicit path → latent states.

    Optional probe output is only a coarse diagnostic and does not reconstruct
    full-resolution animation. No automatic selection or motion inference occurs.
    """
    tokens = encode_image(encoder, source_rgb, device)
    fraction = image_patch_fractions(selected_mask)
    outputs, background, checks = translate_features(tokens, fraction, shifts_px)
    predictions = {}
    if probe is not None:
        for arm, states in outputs.items():
            predictions[arm] = day8_probe.predict_probe(probe, states, device)
    return {'source': tokens, 'source_frac': fraction, 'background': background,
            'edited': outputs, 'predictions': predictions, 'invariants': checks}


def validate_freeze(freeze, publication, probe):
    freeze, publication, probe = map(Path, (freeze, publication, probe))
    frozen = json.loads(freeze.read_text())
    if frozen.get('test_accessed') is not False or frozen.get('test_specs') != test_specs():
        raise ValueError('Missing or mismatched fresh-test freeze')
    required = {'experiments/day10/animate.py'} | {'experiments/day8/' + name for name in
               ('data.py', 'extract.py', 'operators.py', 'probe.py', 'evaluate.py')}
    if not required <= set(frozen.get('source_hashes', {})):
        raise ValueError('Freeze does not bind all numerical source dependencies')
    for path, digest in frozen['source_hashes'].items():
        if sha256(REPO / path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    if frozen.get('probe_sha256') != sha256(probe):
        raise ValueError('Frozen readout checkpoint changed')
    if 'native_calibration' in frozen:
        reference = frozen['native_calibration']
        calibration = Path(reference['path'])
        if not calibration.is_absolute():
            calibration = freeze.parent / calibration
        if sha256(calibration) != reference['sha256']:
            raise ValueError('Native image calibration changed')
    receipt = json.loads(publication.read_text())
    if (receipt.get('freeze_sha256') != sha256(freeze)
            or receipt.get('bytes_equal_to_GitHub') is not True
            or receipt.get('test_encoded_before_verification') is not False
            or len(receipt.get('commit', '')) != 40):
        raise ValueError('Missing pretest GitHub publication attestation')
    return frozen


def score_states(spec, states, predictions, cached):
    rows = []
    source = cached['source']
    for index, shift in enumerate(spec['shifts_px']):
        target = cached['targets'][index]
        labels = {'source_frac': cached['source_frac'], 'target_frac': cached['target_frac'][index],
                  'distractor_frac': cached['distractor_frac'], 'rgb_source': cached['rgb_source'],
                  'rgb_target': cached['rgb_target'][index]}
        regions, _ = day8_evaluate.fixed_regions(labels['source_frac'], labels['target_frac'],
                                                labels['distractor_frac'], shift // PATCH)
        noop_error = np.mean((source.astype(np.float64) - target) ** 2, axis=-1)
        source_prediction = {key: predictions['noop'][key][index] for key in ('occupancy', 'rgb')}
        for arm in ARMS:
            latent = states[arm][index]
            error = np.mean((latent.astype(np.float64) - target) ** 2, axis=-1)
            predicted = {key: predictions[arm][key][index] for key in ('occupancy', 'rgb')}
            row = {'scene': spec['name'], 'seed': spec['seed'], 'position_index': index,
                   'shift_px': shift, 'arm': arm, 'included_in_nonzero_summary': int(shift != 0)}
            row.update(day8_evaluate.region_metrics(error, noop_error, regions))
            # Zero-shift feature differences are zero; ratios have no meaning.
            if shift == 0:
                for key in row:
                    if key.endswith('_ratio'):
                        row[key] = None
            row.update(day8_evaluate.semantic_metrics(predicted, source_prediction, labels, regions)[0])
            hole = regions['source_hole']
            rgb_error = np.mean((predicted['rgb'].astype(np.float64) - labels['rgb_target']) ** 2, axis=-1)
            row['source_hole_rgb_mse'] = float(rgb_error[hole].mean()) if hole.any() else None
            outside = ~((labels['source_frac'] > 0) | (labels['target_frac'] > 0))
            row['outside_requested_union_changed_tokens'] = int(np.sum(np.any(latent != source, axis=-1) & outside))
            row['edited_latent_sha256'] = array_sha256(latent)
            rows.append(row)
    return rows


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


SUMMARY_METRICS = ('primary_ratio', 'source_hole_ratio', 'destination_ratio',
                   'source_hole_rgb_mse', 'source_hole_ghost_mean_occupancy',
                   'selected_centroid_error_px', 'selected_iou', 'appearance_identity_accuracy',
                   'distractor_centroid_error_px', 'distractor_rgb_change_mse')


def summarize(rows):
    scenes = sorted({row['scene'] for row in rows})
    per_image, methods = [], {}
    for scene in scenes:
        for arm in ARMS:
            subset = [r for r in rows if r['scene'] == scene and r['arm'] == arm and r['shift_px'] != 0]
            per_image.append({'scene': scene, 'arm': arm, 'nonzero_positions': len(subset),
                              **{metric: _mean(r[metric] for r in subset) for metric in SUMMARY_METRICS}})
    for arm in ARMS:
        subset = [row for row in per_image if row['arm'] == arm]
        methods[arm] = {metric: _mean(r[metric] for r in subset) for metric in SUMMARY_METRICS}
    genuine = [r for r in rows if r['arm'] == 'genuine_target']
    edited = [r for r in rows if r['arm'] == 'copy_repair']
    identity = _mean(r['appearance_identity_accuracy'] for r in genuine)
    genuine_gate = (_mean(r['selected_centroid_error_px'] for r in genuine) < PATCH
                    and all(r['selected_centroid_present'] for r in genuine)
                    and identity is not None and identity >= .9)
    latent_by_image = {r['scene']: bool(r['primary_ratio'] < 1 and r['source_hole_ratio'] < 1
                                      and r['destination_ratio'] < 1)
                       for r in per_image if r['arm'] == 'copy_repair'}
    edited_gate = (_mean(r['selected_centroid_error_px'] for r in edited) < PATCH
                   and all(r['selected_centroid_present'] for r in edited))
    nonzero_edited = [r for r in edited if r['shift_px'] != 0]
    ratio_denominators_valid = all(not r['source_hole_degenerate'] and not r['destination_degenerate']
                                   for r in nonzero_edited)
    return {'version': VERSION, 'n_images': len(scenes), 'n_unique_positions_per_image': 6,
            'n_nonzero_positions_per_image': 5,
            'aggregation': 'Mean over nonzero positions within each image, then equal image means; descriptive only; no confidence interval',
            'methods': methods, 'per_image': per_image,
            'gates': {'genuine_readout_valid': bool(genuine_gate),
                      'genuine_centroid_mean_px_all_states': _mean(r['selected_centroid_error_px'] for r in genuine),
                      'genuine_coarse_color_accuracy_all_states': identity,
                      'genuine_color_eligible_states': sum(r['appearance_identity_eligible'] for r in genuine),
                      'genuine_color_skipped_states': sum(r['appearance_identity_skipped'] for r in genuine),
                      'edited_centroid_below_one_patch_and_all_present': bool(edited_gate),
                      'edited_centroid_mean_px_all_states': _mean(r['selected_centroid_error_px'] for r in edited),
                      'region_errors_below_noop_by_image': latent_by_image,
                      'no_degenerate_primary_denominators': bool(ratio_denominators_valid),
                      'copy_repair_outside_union_exact': all(r['outside_requested_union_changed_tokens'] == 0 for r in edited)},
            'qualified_controlled_result': bool(genuine_gate and edited_gate and all(latent_by_image.values())
                                                 and ratio_denominators_valid),
            'limits': ['Four procedural images only; displacement is prescribed, not inferred.',
                       'One image cannot reveal the actual occluded background; spatial fill is an estimate.',
                       'Frozen 24x24 probe is a diagnostic, not full-resolution RGB generation.',
                       'Fixed background is mathematically Day8 naive fill, not a new repair algorithm.',
                       'Source patch vectors mix object and context; exact vector copying is not proof of detailed identity.',
                       'Static independently encoded targets score individual requested states, not a learned video rollout.']}


def benchmark(out, upstream, probe_path, freeze, publication, device='cpu', threads=4):
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    frozen = validate_freeze(freeze, publication, probe_path)
    manifest_path = out / 'manifest.json'
    base = {'version': VERSION, 'encoder': day8_extract.MODEL, 'upstream': day8_extract.UPSTREAM,
            'weights_sha256': day8_extract.WEIGHT_SHA256, 'probe_sha256': sha256(probe_path),
            'freeze_sha256': sha256(freeze), 'source_hashes': frozen['source_hashes'],
            'input_modality': 'native image; [1,3,1,384,384] with official patch_embed_img path',
            'inference_precision': 'float32', 'cache_precision': 'float32',
            'operator_input_budget': 'One RGB image, one selected-object mask, explicit horizontal displacement list',
            'target_policy': 'Independently rendered target images/features and distractor masks are scoring-only',
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'device': device, 'threads': torch.get_num_threads()},
            'test_specs': test_specs(), 'scenes': []}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else base
    for key, value in base.items():
        if key != 'scenes' and manifest[key] != value:
            raise ValueError('Cannot resume with changed run metadata: ' + key)
    write_json(manifest_path, manifest)
    encoder = None
    probe, _ = day8_probe.load_probe(probe_path, device)
    all_rows, all_invariants = [], []
    for spec in test_specs():
        name = spec['name']
        record = next((item for item in manifest['scenes'] if item['spec'] == spec), None)
        if record is not None:
            for path, digest in record['file_sha256'].items():
                if sha256(out / path) != digest:
                    raise ValueError('Completed scene evidence changed: ' + path)
            all_rows.extend(json.loads((out / record['metrics_json']).read_text()))
            all_invariants.extend(json.loads((out / record['invariants_json']).read_text()))
            print('IMAGE_REUSED', name, flush=True)
            continue
        if encoder is None:
            encoder = day8_extract.load_encoder(upstream, device)
        started = time.perf_counter()
        rendered = data.generate_pair({'seed': spec['seed'], 'dx': 16, 'name': name, 'split': 'image_test'})
        source_rgb = rendered['frames_source'][FRAME_INDEX].copy()
        selected_mask = rendered['masks_source'][FRAME_INDEX].copy()
        distractor_fraction = image_patch_fractions(rendered['masks_distractor'][FRAME_INDEX])
        del rendered
        input_rgb_path = Path('inputs') / name / 'source.png'
        input_mask_path = Path('inputs') / name / 'selected_mask.png'
        save_png(out / input_rgb_path, source_rgb)
        save_png(out / input_mask_path, selected_mask.astype(np.uint8) * 255)
        result = animate_image(encoder, source_rgb, selected_mask, spec['shifts_px'], probe, device)
        source, fraction = result['source'], result['source_frac']
        states, predictions = result['edited'], result['predictions']
        target_tokens, target_fractions, target_rgbs, target_rgb_paths = [], [], [], []
        for shift in spec['shifts_px']:
            if shift == 0:
                target_rgb, target_fraction, target = source_rgb, fraction, source.copy()
            else:
                pair = data.generate_pair({'seed': spec['seed'], 'dx': shift, 'name': name, 'split': 'image_test'})
                if not np.array_equal(pair['frames_source'][FRAME_INDEX], source_rgb):
                    raise AssertionError('Renderer source depends on requested displacement')
                target_rgb = pair['frames_target'][FRAME_INDEX].copy()
                target_fraction = image_patch_fractions(pair['masks_target'][FRAME_INDEX])
                if not np.array_equal(target_fraction, operators.shift_horizontal(fraction, shift // PATCH)):
                    raise AssertionError('Rendered target mask differs from requested selection translation')
                target = encode_image(encoder, target_rgb, device)
                del pair
            target_path = Path('targets') / name / f'dx{shift:+03d}.png'
            save_png(out / target_path, target_rgb)
            target_rgb_paths.append(str(target_path))
            target_tokens.append(target)
            target_fractions.append(target_fraction)
            target_rgbs.append(image_patch_rgb(target_rgb))
            print('TARGET_ENCODED', name, shift, round(time.perf_counter() - started, 2), flush=True)
        targets = np.stack(target_tokens)
        cached = {'source': source, 'targets': targets, 'source_frac': fraction,
                  'target_frac': np.stack(target_fractions), 'distractor_frac': distractor_fraction,
                  'rgb_source': image_patch_rgb(source_rgb), 'rgb_target': np.stack(target_rgbs),
                  'shifts_px': np.asarray(spec['shifts_px'], dtype=np.int32)}
        states['genuine_target'] = targets
        predictions['genuine_target'] = day8_probe.predict_probe(probe, targets, device)
        scene_rows = score_states(spec, states, predictions, cached)
        # Revisit invariants are actually exercised, not inferred from zero shift.
        display_shifts = [spec['shifts_px'][i] for i in spec['display_indices']]
        display_states, _, display_checks = translate_features(source, fraction, display_shifts)
        if not np.array_equal(display_states['copy_repair'], states['copy_repair'][spec['display_indices']]):
            raise AssertionError('Display revisit sequence changed the edited features')
        invariants = [{'scene': name, 'display_index': i, **check} for i, check in enumerate(display_checks)]
        feature_path = Path('features/cache') / (name + '.npz')
        edited_path = Path('edited') / (name + '.npz')
        prediction_path = Path('predictions') / (name + '.npz')
        metrics_path = Path('metrics') / (name + '.json')
        invariants_path = Path('invariants') / (name + '.json')
        save_npz(out / feature_path, **cached)
        save_npz(out / edited_path, copy_repair=states['copy_repair'], wrong_direction=states['wrong_direction'],
                 fixed_background=result['background'], shifts_px=cached['shifts_px'])
        save_npz(out / prediction_path, shifts_px=cached['shifts_px'],
                 display_indices=np.asarray(spec['display_indices'], dtype=np.int32),
                 **{arm + '__' + key: value for arm, values in predictions.items() for key, value in values.items()})
        write_json(out / metrics_path, scene_rows)
        write_json(out / invariants_path, invariants)
        paths = [input_rgb_path, input_mask_path, feature_path, edited_path, prediction_path, metrics_path,
                 invariants_path, *map(Path, target_rgb_paths)]
        record = {'spec': spec, 'source_rgb_path': str(input_rgb_path), 'selected_mask_path': str(input_mask_path),
                  'feature_path': str(feature_path), 'edited_path': str(edited_path),
                  'predictions_path': str(prediction_path), 'metrics_json': str(metrics_path),
                  'invariants_json': str(invariants_path), 'target_rgb_paths': target_rgb_paths,
                  'file_sha256': {str(path): sha256(out / path) for path in paths},
                  'source_rgb_sha256': array_sha256(source_rgb), 'seconds': time.perf_counter() - started}
        manifest['scenes'].append(record)
        write_json(manifest_path, manifest)
        all_rows.extend(scene_rows)
        all_invariants.extend(invariants)
        print('IMAGE_COMPLETE', name, round(record['seconds'], 2), flush=True)
        del result, states, predictions, cached, targets, target_tokens, display_states
        gc.collect()
    summary = summarize(all_rows)
    write_csv(out / 'per_position.csv', all_rows)
    write_csv(out / 'per_image.csv', summary['per_image'])
    write_json(out / 'summary.json', summary)
    write_json(out / 'invariants.json', {'all_passed': True, 'display_state_checks': all_invariants})
    print('BENCHMARK_COMPLETE', summary['qualified_controlled_result'], flush=True)
    return summary


def image_cli(args):
    import torch
    from PIL import Image
    torch.set_num_threads(args.threads)
    source = np.asarray(Image.open(args.image).convert('RGB'))
    raw_mask = np.asarray(Image.open(args.mask).convert('L'))
    if not np.isin(raw_mask, [0, 255]).all():
        raise ValueError('Mask PNG must contain only 0 and 255; select the object in white')
    selected = raw_mask == 255
    encoder = day8_extract.load_encoder(args.upstream, args.device)
    probe = day8_probe.load_probe(args.probe, args.device)[0] if args.probe else None
    result = animate_image(encoder, source, selected, args.shifts, probe, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    save_png(args.out / 'source.png', source)
    save_png(args.out / 'selected_mask.png', raw_mask)
    save_npz(args.out / 'edited.npz', source=result['source'], source_frac=result['source_frac'],
             fixed_background=result['background'], shifts_px=np.asarray(args.shifts, dtype=np.int32),
             **result['edited'])
    if probe is not None:
        save_npz(args.out / 'predictions.npz', shifts_px=np.asarray(args.shifts, dtype=np.int32),
                 **{arm + '__' + key: value for arm, values in result['predictions'].items() for key, value in values.items()})
    write_json(args.out / 'manifest.json', {'version': VERSION, 'mode': 'user_image_unscored',
               'input_image_sha256': sha256(args.image), 'input_mask_sha256': sha256(args.mask),
               'weights_sha256': day8_extract.WEIGHT_SHA256, 'upstream': day8_extract.UPSTREAM,
               'probe_sha256': sha256(args.probe) if args.probe else None,
               'shifts_px': args.shifts, 'invariants': result['invariants'],
               'warning': 'Day8 probe is calibrated on procedural balls; arbitrary-photo readouts are unvalidated. No target quality score or generated RGB video is produced.'})
    print('IMAGE_EDIT_COMPLETE', args.out, flush=True)


def self_check():
    rng = np.random.default_rng(13100)
    source = rng.normal(size=(1, GRID, GRID, CHANNELS)).astype(np.float32)
    selected = np.zeros((1, GRID, GRID), dtype=np.float32)
    selected[0, 8:11, 10:14] = 1
    original = source.copy()
    path = [0, 16, 48, -32, 16, 0]
    states, background, checks = translate_features(source, selected, path)
    assert np.array_equal(source, original)
    assert np.array_equal(states['copy_repair'][0], states['copy_repair'][-1])
    assert np.array_equal(states['copy_repair'][1], states['copy_repair'][4])
    assert len(checks) == len(path)
    for dx in path:
        naive, _ = operators._transport(source, selected, selected > 0, dx // PATCH)
        assert np.array_equal(states['copy_repair'][path.index(dx)], naive)
    for bad in ([3], [384]):
        try:
            translate_features(source, selected, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid/cropped displacement accepted')
    assert background.dtype == np.float32
    print('SELF_CHECK_PASSED: source-only transport, exact identity/revisits/preservation, naive equivalence, invalid shifts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    benchmark_parser = commands.add_parser('benchmark')
    benchmark_parser.add_argument('--freeze', type=Path, required=True)
    benchmark_parser.add_argument('--publication', type=Path, required=True)
    image_parser = commands.add_parser('image')
    image_parser.add_argument('--image', type=Path, required=True)
    image_parser.add_argument('--mask', type=Path, required=True)
    image_parser.add_argument('--shifts', type=int, nargs='+', required=True)
    for subparser in (benchmark_parser, image_parser):
        subparser.add_argument('--upstream', type=Path, required=True)
        subparser.add_argument('--probe', type=Path, required=(subparser is benchmark_parser))
        subparser.add_argument('--out', type=Path, required=True)
        subparser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
        subparser.add_argument('--threads', type=int, default=4)
    commands.add_parser('self-check')
    args = parser.parse_args()
    if args.command == 'benchmark':
        benchmark(args.out, args.upstream, args.probe, args.freeze, args.publication, args.device, args.threads)
    elif args.command == 'image':
        image_cli(args)
    else:
        self_check()
