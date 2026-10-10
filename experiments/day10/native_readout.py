"""Predeclared native-image diagnostic readout repair after Day10 transfer failed.

This trains only the existing small token-only occupancy/RGB probe on genuine
native-image encodings. The JEPA encoder, source-only edit operator, and region
metrics remain unchanged. It does not train a full-resolution image decoder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import animate

VERSION = 'day10_native_image_readout_followup_v1'
REPO = Path(__file__).resolve().parents[2]
HYPERPARAMETERS = {
    'seed': 1010, 'max_epochs': 40, 'batches_per_epoch': 32, 'batch_size': 512,
    'learning_rate': .001, 'hidden_dim': 128,
    'optimizer': 'AdamW', 'weight_decay': .0001, 'selection': 'final epoch',
    'normalization': 'train-only channel mean/std; floor1e-6',
    'sampling': '50/50 any-ball>0/background, replacement, uniform within group',
    'loss': 'occupancy MSE + mean RGB-channel MSE',
    'inference_precision': 'float32', 'training_precision': 'float32',
    'device': 'cpu', 'threads': 4,
}
SOURCE_PATHS = ('experiments/day10/native_readout.py',
                'experiments/day10/native_readout_PROTOCOL.md',
                'experiments/day10/animate.py',
                'experiments/day8/data.py', 'experiments/day8/extract.py',
                'experiments/day8/operators.py', 'experiments/day8/probe.py',
                'experiments/day8/evaluate.py')


def specs(split):
    if split == 'train':
        start, count, offsets = 13400, 32, (16, -16, 32, -32, 48, -48, 64, -64, 80, -80)
    elif split == 'dev':
        start, count, offsets = 13500, 8, (16, -16, 32, -32, 48, -48, 80, -80)
    else:
        raise ValueError('Only predeclared train and development features may be prepared')
    return [{'name': f'{split}_{start+i}', 'seed': start+i, 'split': split,
             'frame_index': 15, 'dx': offsets[i % len(offsets)],
             'views': ['source', 'genuine_shifted_target']} for i in range(count)]


def fresh_test_specs():
    return [{'name': f'image_{13600+i}', 'seed': 13600+i, 'frame_index': 15,
             'shifts_px': [((1 if i < 2 else -1) * d) for d in (0, 16, 32, 48, 64, 80)],
             'display_indices': list(animate.DISPLAY_INDICES)} for i in range(4)]


def make_training_freeze(out):
    record = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'training_accessed': False, 'test_accessed': False,
              'source_hashes': {name: animate.sha256(REPO / name) for name in SOURCE_PATHS},
              'hyperparameters': HYPERPARAMETERS, 'train_specs': specs('train'),
              'dev_specs': specs('dev'), 'fresh_test_specs': fresh_test_specs(),
              'upstream_revision': animate.day8_extract.UPSTREAM,
              'encoder_weights_sha256': animate.day8_extract.WEIGHT_SHA256,
              'training_inputs': 'Native-image genuine source/target encodings only; no edited features or original test scenes',
              'dev_role': 'Single pass/fail gate after fixed final epoch; no tuning or checkpoint selection'}
    animate.write_json(out, record)
    print('TRAINING_FREEZE_CREATED', out, animate.sha256(out), flush=True)
    return record


def validate_training_freeze(freeze, publication):
    freeze, publication = Path(freeze), Path(publication)
    frozen = json.loads(freeze.read_text())
    expected = {'version': VERSION, 'training_accessed': False, 'test_accessed': False,
                'hyperparameters': HYPERPARAMETERS, 'train_specs': specs('train'),
                'dev_specs': specs('dev'), 'fresh_test_specs': fresh_test_specs(),
                'upstream_revision': animate.day8_extract.UPSTREAM,
                'encoder_weights_sha256': animate.day8_extract.WEIGHT_SHA256}
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError('Pretraining freeze mismatch: ' + key)
    if not set(SOURCE_PATHS) <= set(frozen.get('source_hashes', {})):
        raise ValueError('Incomplete pretraining source binding')
    for name, digest in frozen['source_hashes'].items():
        if animate.sha256(REPO / name) != digest:
            raise ValueError('Pretraining source changed: ' + name)
    receipt = json.loads(publication.read_text())
    if (receipt.get('freeze_sha256') != animate.sha256(freeze)
            or receipt.get('bytes_equal_to_GitHub') is not True
            or receipt.get('training_encoded_before_verification') is not False
            or len(receipt.get('commit', '')) != 40):
        raise ValueError('Missing byte-verified pretraining publication receipt')
    return frozen


def prepare(out, upstream, freeze, publication):
    out = Path(out)
    torch.set_num_threads(HYPERPARAMETERS['threads'])
    frozen = validate_training_freeze(freeze, publication)
    manifest_path = out / 'features_manifest.json'
    base = {'version': VERSION, 'training_freeze_sha256': animate.sha256(freeze),
            'training_publication_sha256': animate.sha256(publication),
            'source_hashes': frozen['source_hashes'], 'test_accessed': False,
            'train_specs': specs('train'), 'dev_specs': specs('dev'),
            'encoder': animate.day8_extract.MODEL, 'upstream': animate.day8_extract.UPSTREAM,
            'weights_sha256': animate.day8_extract.WEIGHT_SHA256,
            'modality': 'native single image [1,3,1,384,384]',
            'precision': 'float32', 'cache_precision': 'float32',
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': torch.__version__, 'device': 'cpu', 'threads': torch.get_num_threads()},
            'scenes': []}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else base
    for key, value in base.items():
        if key != 'scenes' and manifest[key] != value:
            raise ValueError('Cannot mix preparation runs: ' + key)
    animate.write_json(manifest_path, manifest)
    encoder = None
    for spec in specs('train') + specs('dev'):
        old = next((row for row in manifest['scenes'] if row['spec'] == spec), None)
        if old is not None:
            if animate.sha256(out / old['path']) != old['sha256']:
                raise ValueError('Prepared feature cache changed')
            print('READOUT_FEATURE_REUSED', spec['name'], flush=True)
            continue
        if encoder is None:
            encoder = animate.day8_extract.load_encoder(upstream, 'cpu')
        started = time.perf_counter()
        pair = animate.data.generate_pair(spec)
        source_image = pair['frames_source'][15].copy()
        target_image = pair['frames_target'][15].copy()
        source_fraction = animate.image_patch_fractions(pair['masks_source'][15])
        target_fraction = animate.image_patch_fractions(pair['masks_target'][15])
        distractor_fraction = animate.image_patch_fractions(pair['masks_distractor'][15])
        source_occupancy = animate.image_patch_fractions(pair['masks_source'][15] | pair['masks_distractor'][15])
        target_occupancy = animate.image_patch_fractions(pair['masks_target'][15] | pair['masks_distractor'][15])
        del pair
        source_tokens = animate.encode_image(encoder, source_image, 'cpu')
        target_tokens = animate.encode_image(encoder, target_image, 'cpu')
        path = Path('features/cache') / (spec['name'] + '.npz')
        animate.save_npz(out / path,
                         tokens=np.stack([source_tokens, target_tokens]),
                         occupancy=np.stack([source_occupancy, target_occupancy]),
                         rgb=np.stack([animate.image_patch_rgb(source_image), animate.image_patch_rgb(target_image)]),
                         source_frac=source_fraction,
                         target_frac=np.stack([source_fraction, target_fraction]),
                         distractor_frac=distractor_fraction,
                         rgb_source=animate.image_patch_rgb(source_image),
                         view_shift_px=np.array([0, spec['dx']], dtype=np.int32))
        record = {'spec': spec, 'path': str(path), 'sha256': animate.sha256(out / path),
                  'source_rgb_sha256': animate.array_sha256(source_image),
                  'target_rgb_sha256': animate.array_sha256(target_image),
                  'bytes': (out / path).stat().st_size, 'seconds': time.perf_counter() - started}
        manifest['scenes'].append(record)
        animate.write_json(manifest_path, manifest)
        print('READOUT_FEATURE_DONE', spec['name'], round(record['seconds'], 2), flush=True)
        del source_tokens, target_tokens
        gc.collect()
    print('READOUT_FEATURES_COMPLETE', len(manifest['scenes']), flush=True)


def load_prepared(out, freeze, publication):
    frozen = validate_training_freeze(freeze, publication)
    out = Path(out)
    manifest = json.loads((out / 'features_manifest.json').read_text())
    if (manifest.get('version') != VERSION or manifest.get('test_accessed') is not False
            or manifest.get('source_hashes') != frozen['source_hashes']
            or manifest.get('training_freeze_sha256') != animate.sha256(freeze)
            or manifest.get('training_publication_sha256') != animate.sha256(publication)):
        raise ValueError('Prepared manifest is not bound to this frozen run')
    if [record['spec'] for record in manifest['scenes']] != specs('train') + specs('dev'):
        raise ValueError('Incomplete, reordered, or changed prepared scenes')
    loaded = {'train': [], 'dev': []}
    for record in manifest['scenes']:
        path = out / record['path']
        if animate.sha256(path) != record['sha256']:
            raise ValueError('Prepared cache hash changed: ' + str(path))
        with np.load(path, allow_pickle=False) as archive:
            cached = {key: archive[key] for key in archive.files}
        if cached['tokens'].shape != (2, 1, 24, 24, 1024) or cached['tokens'].dtype != np.float32:
            raise ValueError('Unexpected native-image training feature shape or precision')
        loaded[record['spec']['split']].append((record['spec'], cached))
    return loaded, manifest


def dev_assessment(model, examples):
    rows, predictions = [], {}
    for spec, cached in examples:
        prediction = animate.day8_probe.predict_probe(model, cached['tokens'], 'cpu')
        predictions[spec['name'] + '__occupancy'] = prediction['occupancy']
        predictions[spec['name'] + '__rgb'] = prediction['rgb']
        source_prediction = {key: prediction[key][0] for key in ('occupancy', 'rgb')}
        for index, dx in enumerate(cached['view_shift_px']):
            labels = {'source_frac': cached['source_frac'], 'target_frac': cached['target_frac'][index],
                      'distractor_frac': cached['distractor_frac'], 'rgb_source': cached['rgb_source'],
                      'rgb_target': cached['rgb'][index]}
            regions, _ = animate.day8_evaluate.fixed_regions(labels['source_frac'], labels['target_frac'],
                                                           labels['distractor_frac'], int(dx) // 16)
            predicted = {key: prediction[key][index] for key in ('occupancy', 'rgb')}
            row = {'scene': spec['name'], 'seed': spec['seed'], 'view_index': index, 'shift_px': int(dx)}
            row.update(animate.day8_evaluate.semantic_metrics(predicted, source_prediction, labels, regions)[0])
            row['occupancy_mse'] = float(np.mean((prediction['occupancy'][index].astype(np.float64) - cached['occupancy'][index]) ** 2))
            row['rgb_mse'] = float(np.mean((prediction['rgb'][index].astype(np.float64) - cached['rgb'][index]) ** 2))
            rows.append(row)
    centroid = animate._mean(row['selected_centroid_error_px'] for row in rows)
    color = animate._mean(row['appearance_identity_accuracy'] for row in rows)
    gate = {'version': VERSION, 'n_scenes': len(examples), 'n_genuine_images': len(rows),
            'selected_centroid_mean_px': centroid,
            'all_selected_centroids_present': all(row['selected_centroid_present'] for row in rows),
            'coarse_color_accuracy': color,
            'coarse_color_eligible': sum(row['appearance_identity_eligible'] for row in rows),
            'coarse_color_skipped': sum(row['appearance_identity_skipped'] for row in rows),
            'selected_iou_mean': animate._mean(row['selected_iou'] for row in rows),
            'occupancy_mse_mean': animate._mean(row['occupancy_mse'] for row in rows),
            'rgb_mse_mean': animate._mean(row['rgb_mse'] for row in rows),
            'rule': 'Final fixed-epoch model only; genuine mean selected centroid <16px, all present, eligible coarse-color accuracy>=.90',
            'used_for_tuning': False, 'test_accessed': False}
    gate['pass'] = bool(centroid is not None and centroid < 16 and gate['all_selected_centroids_present']
                        and color is not None and color >= .9)
    return rows, predictions, gate


def train(out, freeze, publication):
    out = Path(out)
    torch.set_num_threads(HYPERPARAMETERS['threads'])
    loaded, manifest = load_prepared(out, freeze, publication)
    if (out / 'training/probe.pt').exists():
        raise ValueError('A trained probe already exists; do not silently refit or overwrite this fixed run')
    training_examples = [{'tokens': cached['tokens'][index], 'occupancy': cached['occupancy'][index],
                          'rgb': cached['rgb'][index]}
                         for _, cached in loaded['train'] for index in range(2)]
    model, history = animate.day8_probe.fit_probe(
        training_examples, device='cpu', seed=HYPERPARAMETERS['seed'], dev_examples=None,
        max_epochs=HYPERPARAMETERS['max_epochs'], batches_per_epoch=HYPERPARAMETERS['batches_per_epoch'],
        batch_size=HYPERPARAMETERS['batch_size'], learning_rate=HYPERPARAMETERS['learning_rate'],
        hidden_dim=HYPERPARAMETERS['hidden_dim'])
    if history['selected_epoch'] != 40 or history['dev_examples'] != 0:
        raise AssertionError('The model must be the fixed final epoch, without development selection')
    history['native_readout_followup'] = {'version': VERSION, 'hyperparameters': HYPERPARAMETERS,
                                         'training_freeze_sha256': animate.sha256(freeze),
                                         'features_manifest_sha256': animate.sha256(out / 'features_manifest.json'),
                                         'training_seeds': [spec['seed'] for spec, _ in loaded['train']],
                                         'development_role': 'Single post-training gate only; never checkpoint selection',
                                         'original_test_seeds_used': [], 'fresh_test_accessed': False}
    animate.day8_probe.save_probe(model, out / 'training/probe.pt', history)
    animate.write_json(out / 'training/history.json', history)
    animate.write_csv(out / 'training/losses.csv', [
        {'epoch': row['epoch'], **{'train_' + key: value for key, value in row['train'].items()}}
        for row in history['epochs']])
    rows, predictions, gate = dev_assessment(model, loaded['dev'])
    gate.update({'probe_sha256': animate.sha256(out / 'training/probe.pt'),
                 'training_history_sha256': animate.sha256(out / 'training/history.json'),
                 'features_manifest_sha256': animate.sha256(out / 'features_manifest.json')})
    animate.write_json(out / 'training/dev_metrics.json', rows)
    animate.write_csv(out / 'training/dev_metrics.csv', rows)
    animate.save_npz(out / 'training/dev_predictions.npz', **predictions)
    animate.write_json(out / 'training/dev_gate.json', gate)
    print('NATIVE_READOUT_TRAINED', json.dumps(gate), flush=True)
    return gate


def _resolve_record(freeze, reference):
    path = Path(reference['path'])
    if not path.is_absolute():
        path = Path(freeze).parent / path
    if animate.sha256(path) != reference['sha256']:
        raise ValueError('Posttraining bound evidence changed: ' + str(path))
    return path


def test(out, upstream, probe, freeze, publication):
    frozen = json.loads(Path(freeze).read_text())
    if frozen.get('readout_followup_version') != VERSION or frozen.get('test_specs') != fresh_test_specs():
        raise ValueError('Wrong fresh native-readout test freeze')
    if not set(SOURCE_PATHS) <= set(frozen.get('source_hashes', {})):
        raise ValueError('Posttraining freeze must bind the unchanged operator and readout-training sources')
    if frozen.get('hyperparameters') != HYPERPARAMETERS:
        raise ValueError('Posttraining hyperparameter binding changed')
    training_freeze = _resolve_record(freeze, frozen['training_freeze'])
    training_publication = _resolve_record(freeze, frozen['training_publication'])
    validate_training_freeze(training_freeze, training_publication)
    trained_probe = _resolve_record(freeze, frozen['trained_readout'])
    if animate.sha256(probe) != animate.sha256(trained_probe):
        raise ValueError('Requested probe differs from the frozen final-epoch checkpoint')
    gate_path = _resolve_record(freeze, frozen['dev_validation'])
    gate = json.loads(gate_path.read_text())
    if gate.get('pass') is not True or gate.get('probe_sha256') != animate.sha256(probe):
        raise ValueError('Native readout development gate did not pass; fresh test remains unopened')
    manifest = _resolve_record(freeze, frozen['features_manifest'])
    if gate.get('features_manifest_sha256') != animate.sha256(manifest):
        raise ValueError('Development gate features binding changed')
    # Explicit version/spec override. The original frozen numerical sources
    # remain unchanged, and the existing benchmark validates the new freeze.
    animate.test_specs = fresh_test_specs
    animate.VERSION = VERSION
    summary = animate.benchmark(out, upstream, probe, freeze, publication, 'cpu', 4)
    rows = []
    for spec in fresh_test_specs():
        rows.extend(json.loads((Path(out) / 'metrics' / (spec['name'] + '.json')).read_text()))
    edited = [row for row in rows if row['arm'] == 'copy_repair']
    color = animate._mean(row['appearance_identity_accuracy'] for row in edited)
    edited_color_gate = bool(color is not None and color >= .9)
    summary['original_day10_criteria_pass'] = summary['qualified_controlled_result']
    summary['gates']['edited_coarse_color_accuracy_all_states'] = color
    summary['gates']['edited_color_eligible_states'] = sum(row['appearance_identity_eligible'] for row in edited)
    summary['gates']['edited_color_skipped_states'] = sum(row['appearance_identity_skipped'] for row in edited)
    summary['gates']['edited_coarse_color_valid'] = edited_color_gate
    summary['qualified_controlled_result'] = bool(summary['qualified_controlled_result'] and edited_color_gate)
    summary['readout_followup'] = {'version': VERSION, 'training_freeze_sha256': animate.sha256(training_freeze),
                                  'probe_sha256': animate.sha256(probe), 'dev_gate_sha256': animate.sha256(gate_path),
                                  'operator_changed': False, 'new_decoder_claim': False,
                                  'additional_gate': 'Edited coarse-color accuracy>=.90 across all eligible fresh states'}
    summary['limits'].append('This readout repair follows a disclosed failed video-to-image probe transfer; all fresh test images use new seeds.')
    animate.write_json(Path(out) / 'summary.json', summary)
    print('NATIVE_READOUT_FRESH_TEST_COMPLETE', summary['qualified_controlled_result'], flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('make-training-freeze')
    make.add_argument('--out', type=Path, required=True)
    prep = sub.add_parser('prepare')
    fitting = sub.add_parser('train')
    testing = sub.add_parser('test')
    for command in (prep, fitting, testing):
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--freeze', type=Path, required=True)
        command.add_argument('--publication', type=Path, required=True)
    for command in (prep, testing):
        command.add_argument('--upstream', type=Path, required=True)
    testing.add_argument('--probe', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'make-training-freeze':
        make_training_freeze(args.out)
    elif args.command == 'prepare':
        prepare(args.out, args.upstream, args.freeze, args.publication)
    elif args.command == 'train':
        train(args.out, args.freeze, args.publication)
    else:
        test(args.out, args.upstream, args.probe, args.freeze, args.publication)
