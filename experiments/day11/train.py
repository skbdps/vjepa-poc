"""Fixed, hash-bound genuine-image training for the JEPA -> Wan VAE bridge.

Both the spatial CNN and shared affine baseline use the same genuine training
pairs, training-only feature normalization, fixed epoch budget, and batch order.
Development metrics are logged but never select checkpoints. This module never
opens held-out images, creates edited training features, or runs VACE diffusion.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _local_module(name, filename):
    # Avoid collisions with the older experiments' unrelated data.py/model.py.
    specification = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


data = _local_module('day11_training_data', 'data.py')
model = _local_module('day11_training_model', 'model.py')
VERSION = 'day11_genuine_jepa_to_wan_training_v1'
ARMS = ('cnn', 'linear')
TRAINING_CONFIG = {
    'epochs': 150, 'batch_size': 4, 'seed': 1111, 'learning_rate': .001,
    'weight_decay': .0001, 'optimizer': 'AdamW', 'cnn_width': 64,
    'loss': 'unweighted mean squared error over all normalized Wan VAE latent channels/pixels',
    'precision': 'float32', 'selection': 'fixed final epoch; no development selection',
    'gradient_clipping': None, 'learning_rate_schedule': None,
    'input_normalization': 'genuine training JEPA channel mean/std only; std floor1e-6',
    'shuffle': 'numpy PCG64(seed+epoch), same permutation for both arms',
    'training_inputs': 'genuine JEPA tensor and fixed destination coordinates only',
    'supervision': 'separately encoded genuine normalized Wan VAE latents',
    'region_metric_masks': 'scoring only; genuine 24x24 coverage nearest-expanded2x to48x48',
    'untrained_comparators': ['train-only average spatial latent map', 'genuine source latent reused for its shifted development target'],
}
REQUIRED_SOURCES = ('experiments/day11/train.py', 'experiments/day11/data.py',
                    'experiments/day11/model.py', 'experiments/day11/PROTOCOL.md',
                    'experiments/day8/data.py')


def sha256(path):
    return data.sha256(path)


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='staging_', suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _sources():
    return {name: sha256(REPO / name) for name in REQUIRED_SOURCES}


def _manifests(jepa_root, targets_root):
    condition_path = Path(jepa_root) / 'features_manifest.json'
    target_path = Path(targets_root) / 'latent_manifest.json'
    conditions = json.loads(condition_path.read_text())
    targets = json.loads(target_path.read_text())
    if (conditions.get('version') != data.CACHE_VERSION
            or conditions.get('test_accessed') is not False
            or len(conditions.get('scenes', [])) != 40):
        raise ValueError('Expected the existing 40-scene genuine train/development manifest')
    if targets.get('version') != data.TARGET_VERSION or targets.get('jepa_manifest_sha256') != sha256(condition_path):
        raise ValueError('VAE target manifest is not bound to these genuine JEPA examples')
    if not isinstance(targets.get('vae'), dict) or not targets['vae']:
        raise ValueError('VAE targets lack encoder/normalization provenance')
    expected, condition_files = [], []
    for row in conditions['scenes']:
        spec = row['spec']
        if spec['split'] not in data.SPLIT_SEEDS or spec['seed'] not in data.SPLIT_SEEDS[spec['split']]:
            raise ValueError('Unexpected scene in train/development inputs')
        condition_files.append({'path': row['path'], 'sha256': row['sha256'], 'scene': spec['name']})
        for view in ('source', 'target'):
            expected.append({'sample_id': spec['name'] + '__' + view, 'split': spec['split'],
                             'seed': spec['seed'], 'rgb_sha256': row[view + '_rgb_sha256']})
    items = {item['sample_id']: item for item in targets.get('items', [])}
    if len(items) != len(targets.get('items', [])) or set(items) != {row['sample_id'] for row in expected}:
        raise ValueError('VAE targets must contain exactly the64train+16dev genuine image identifiers')
    for row in expected:
        if items[row['sample_id']]['rgb_sha256'] != row['rgb_sha256']:
            raise ValueError('Target image hash differs from its JEPA condition image')
    return {'jepa_manifest_sha256': sha256(condition_path), 'target_manifest_sha256': sha256(target_path),
            'vae': targets['vae'], 'samples': expected, 'jepa_files': condition_files,
            'target_files': [items[row['sample_id']] for row in expected]}


def make_freeze(jepa_root, targets_root, out, device='cpu', threads=4):
    """Metadata-only pretraining snapshot; no held-out data are available here."""
    if threads < 1:
        raise ValueError('Positive CPU thread count required')
    record = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'training_started': False, 'fresh_test_accessed': False,
              'training_config': TRAINING_CONFIG, 'arms': list(ARMS),
              'execution': {'device': device, 'threads': threads},
              'source_hashes': _sources(), 'data': _manifests(jepa_root, targets_root),
              'test_policy': 'Train on genuine examples only. Publish final checkpoint/data/source hashes before accessing fresh held-out scenes.'}
    write_json(out, record)
    print('BRIDGE_PRETRAIN_FREEZE', out, sha256(out), flush=True)
    return record


def validate_freeze(jepa_root, targets_root, freeze, publication, device, threads):
    frozen = json.loads(Path(freeze).read_text())
    expected = {'version': VERSION, 'training_started': False, 'fresh_test_accessed': False,
                'training_config': TRAINING_CONFIG, 'arms': list(ARMS),
                'execution': {'device': device, 'threads': threads}, 'source_hashes': _sources(),
                'data': _manifests(jepa_root, targets_root)}
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError('Pretraining binding mismatch: ' + key)
    receipt = json.loads(Path(publication).read_text())
    if (receipt.get('freeze_sha256') != sha256(freeze)
            or receipt.get('bytes_equal_to_GitHub') is not True
            or receipt.get('training_started_before_verification') is not False
            or len(receipt.get('commit', '')) != 40):
        raise ValueError('Missing byte-verified pretraining GitHub publication receipt')
    return frozen


def _scoring_regions(dataset):
    """Read masks separately; never attach them to a projector input tensor."""
    regions = {'selected': [], 'distractor': [], 'background': []}
    for record in dataset.records:
        path = data._child_path(dataset.root, record.cache_path)
        with np.load(path, allow_pickle=False) as archive:
            if not {'target_frac', 'distractor_frac'} <= set(archive.files):
                return {}, 'unavailable: genuine feature caches do not contain both coverage labels'
            selected = archive['target_frac'][record.view_index, 0].astype(np.float32)
            distractor = archive['distractor_frac'][0].astype(np.float32)
        if selected.shape != (24, 24) or distractor.shape != (24, 24):
            raise ValueError('Unexpected genuine region coverage shape')
        for name, values in (('selected', selected), ('distractor', distractor)):
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError('Invalid scoring-only fractional coverage')
            regions[name].append(np.repeat(np.repeat(values, 2, axis=0), 2, axis=1))
        background = 1 - np.clip(selected + distractor, 0, 1)
        regions['background'].append(np.repeat(np.repeat(background, 2, axis=0), 2, axis=1))
    return {name: torch.from_numpy(np.stack(values)) for name, values in regions.items()}, TRAINING_CONFIG['region_metric_masks']


def load_training_data(jepa_root, targets_root):
    datasets = {split: data.NativeImageDataset(jepa_root, split) for split in ('train', 'dev')}
    targets = data.LatentTargetStore(targets_root, datasets['train'].manifest_sha256)
    expected_ids = {record.sample_id for dataset in datasets.values() for record in dataset.records}
    if set(targets.items) != expected_ids:
        raise ValueError('Target store contains missing or non-training/development examples')
    mean, std = data.train_channel_statistics(datasets['train'])
    prepared, provenance = {}, {}
    for split, dataset in datasets.items():
        condition_values, latent_values = [], []
        for index, record in enumerate(dataset.records):
            # No image, mask, identity, or geometry field is an input to F.
            condition_values.append(dataset[index]['jepa'])
            latent_values.append(targets.get(record.sample_id, record.rgb_sha256))
        regions, region_note = _scoring_regions(dataset)
        x, y = torch.stack(condition_values), torch.stack(latent_values)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite genuine training/development tensors')
        if tuple(x.shape[1:]) != (1024, 24, 24) or tuple(y.shape[1:]) != (16, 48, 48):
            raise ValueError('Unexpected JEPA/Wan training tensor dimensions')
        prepared[split] = {'x': x, 'y': y, 'regions': regions,
                           'sample_ids': [record.sample_id for record in dataset.records]}
        provenance[split] = {**dataset.provenance(), 'region_metrics': region_note,
                             'jepa_stack_sha256': array_sha256(x.numpy()),
                             'target_stack_sha256': array_sha256(y.numpy())}
    return prepared, mean, std, provenance


@torch.no_grad()
def evaluate(projector, split, device, batch_size):
    projector.eval()
    all_mse, region_rows = [], {name: [] for name in split['regions']}
    for start in range(0, len(split['x']), batch_size):
        x = split['x'][start:start + batch_size].to(device)
        y = split['y'][start:start + batch_size].to(device)
        prediction = projector(x)
        if prediction.shape != y.shape or not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite or incorrectly shaped predicted Wan latents')
        # Double-precision scoring after FP32 inference; each image has equal weight.
        error = (prediction.double() - y.double()).square().mean(dim=1).cpu()
        all_mse.extend(error.mean(dim=(1, 2)).tolist())
        for name, weights in split['regions'].items():
            weight = weights[start:start + batch_size].double()
            mass = weight.sum(dim=(1, 2))
            numerator = (error * weight).sum(dim=(1, 2))
            region_rows[name].extend([float(n / m) if m > 0 else None for n, m in zip(numerator, mass)])
    result = {'mse': float(np.mean(all_mse)), 'images': len(all_mse)}
    for name, values in region_rows.items():
        valid = [value for value in values if value is not None]
        result[name + '_mse'] = float(np.mean(valid)) if valid else None
        result[name + '_images'] = len(valid)
        result[name + '_skipped_images'] = len(values) - len(valid)
    # Source-noop diagnostics use target views only. Record the matching model
    # subset so readers do not compare eight targets with an all-16-image mean.
    for view in ('source', 'target'):
        indices = [index for index, sample_id in enumerate(split['sample_ids']) if sample_id.endswith('__' + view)]
        if indices:
            result[view + '_view_mse'] = float(np.mean([all_mse[index] for index in indices]))
            result[view + '_view_images'] = len(indices)
            for name, values in region_rows.items():
                valid = [values[index] for index in indices if values[index] is not None]
                result[view + '_view_' + name + '_mse'] = float(np.mean(valid)) if valid else None
    return result


def _fixed_prediction_metrics(prediction, target, regions):
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise ValueError('Invalid non-learned comparator prediction')
    error = (prediction.double() - target.double()).square().mean(dim=1)
    result = {'mse': float(error.mean(dim=(1, 2)).mean()), 'images': len(error)}
    for name, weights in regions.items():
        weight = weights.double()
        mass = weight.sum(dim=(1, 2))
        valid = mass > 0
        result[name + '_mse'] = float(((error * weight).sum(dim=(1, 2))[valid] / mass[valid]).mean()) if valid.any() else None
        result[name + '_images'] = int(valid.sum())
        result[name + '_skipped_images'] = int((~valid).sum())
    return result


def untrained_baselines(prepared):
    """No extra architecture, fit, development selection, or pixel prediction."""
    train_set, dev_set = prepared['train'], prepared['dev']
    mean_map = train_set['y'].double().mean(dim=0).float()
    mean_prediction = mean_map[None].expand_as(dev_set['y'])
    source_indices, target_indices = [], []
    index_by_id = {name: index for index, name in enumerate(dev_set['sample_ids'])}
    for index, name in enumerate(dev_set['sample_ids']):
        if name.endswith('__target'):
            source_name = name.removesuffix('__target') + '__source'
            if source_name not in index_by_id:
                raise ValueError('Missing paired source for development source-noop comparator')
            source_indices.append(index_by_id[source_name])
            target_indices.append(index)
    target_regions = {name: values[target_indices] for name, values in dev_set['regions'].items()}
    report = {
        'train_mean_latent_map': {
            'fit': 'FP64 mean of the64genuine training target tensors, then FP32; development never enters this mean',
            'mean_map_sha256': array_sha256(mean_map.numpy()),
            'dev_all_views': _fixed_prediction_metrics(mean_prediction, dev_set['y'], dev_set['regions']),
            'dev_target_views': _fixed_prediction_metrics(mean_prediction[target_indices], dev_set['y'][target_indices], target_regions)},
        'source_noop_delta': {
            'definition': 'Use actual normalized Wan source latent as prediction for its paired genuine shifted target; MSE equals source-target delta energy',
            'sample_ids': [dev_set['sample_ids'][index] for index in target_indices],
            'dev_target_views': _fixed_prediction_metrics(dev_set['y'][source_indices], dev_set['y'][target_indices], target_regions)},
        'interpretation': 'Compare all-view metrics with model dev_mse; compare source-noop target-only metrics with model dev_target_view_mse. Neither comparator establishes edited-state or rendered success.'}
    return mean_map, report


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def fit_arm(arm, out, prepared, mean, std, binding, device, resume):
    arm_dir = Path(out) / arm
    latest = arm_dir / 'latest.pt'
    final = arm_dir / 'final.pt'
    completed = arm_dir / 'completed.json'
    if completed.exists():
        record = json.loads(completed.read_text())
        if (record.get('binding') != binding or record.get('arm') != arm
                or record.get('selected_epoch') != TRAINING_CONFIG['epochs']
                or sha256(final) != record.get('checkpoint_sha256')
                or sha256(arm_dir / 'history.json') != record.get('history_sha256')
                or sha256(arm_dir / 'losses.csv') != record.get('losses_sha256')):
            raise ValueError('Completed arm differs from the frozen training run')
        if not resume:
            raise ValueError('Completed model exists; use --resume to verify/reuse it, never silently overwrite it')
        print('BRIDGE_ARM_REUSED', arm, flush=True)
        return record
    torch.manual_seed(TRAINING_CONFIG['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed_all(TRAINING_CONFIG['seed'])
    projector = model.build_model(arm, mean, std, width=TRAINING_CONFIG['cnn_width']).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=TRAINING_CONFIG['learning_rate'],
                                 weight_decay=TRAINING_CONFIG['weight_decay'])
    history, start_epoch = [], 0
    if latest.exists():
        if not resume:
            raise ValueError('Interrupted checkpoint exists; use --resume rather than changing its optimization history')
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved.get('binding') != binding or saved.get('arm') != arm or saved.get('training_config') != TRAINING_CONFIG:
            raise ValueError('Interrupted checkpoint belongs to a different frozen run')
        projector.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history, start_epoch = saved['history'], int(saved['epoch'])
        if [row['epoch'] for row in history] != list(range(1, start_epoch + 1)):
            raise ValueError('Interrupted checkpoint history has missing or duplicated epochs')
        torch.set_rng_state(saved['torch_rng_state'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        print('BRIDGE_ARM_RESUME', arm, start_epoch, flush=True)
    elif final.exists():
        raise ValueError('Unbound final checkpoint exists without a completion record')
    train_set, dev_set = prepared['train'], prepared['dev']
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, TRAINING_CONFIG['epochs'] + 1):
        projector.train()
        order = np.random.default_rng(TRAINING_CONFIG['seed'] + epoch).permutation(len(train_set['x']))
        batch_mse_sum, images_seen = 0., 0
        for start in range(0, len(order), TRAINING_CONFIG['batch_size']):
            indices = torch.as_tensor(order[start:start + TRAINING_CONFIG['batch_size']], dtype=torch.long)
            x, y = train_set['x'][indices].to(device), train_set['y'][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = projector(x)
            loss = torch.nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {arm} epoch{epoch}')
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in projector.parameters()):
                raise RuntimeError(f'Nonfinite gradient in {arm} epoch{epoch}')
            optimizer.step()
            batch_mse_sum += float(loss.item()) * len(indices)
            images_seen += len(indices)
        train_metrics = evaluate(projector, train_set, device, TRAINING_CONFIG['batch_size'])
        dev_metrics = evaluate(projector, dev_set, device, TRAINING_CONFIG['batch_size'])
        row = {'epoch': epoch, 'train_optimization_mse': batch_mse_sum / images_seen,
               **{'train_' + key: value for key, value in train_metrics.items()},
               **{'dev_' + key: value for key, value in dev_metrics.items()}}
        history.append(row)
        payload = {'version': VERSION, 'arm': arm, 'epoch': epoch, 'binding': binding,
                   'training_config': TRAINING_CONFIG, 'model_configuration': projector.configuration(),
                   'state_dict': _cpu_copy(projector.state_dict()),
                   'optimizer_state_dict': _cpu_copy(optimizer.state_dict()), 'history': history,
                   'torch_rng_state': torch.get_rng_state(),
                   'cuda_rng_states': torch.cuda.get_rng_state_all() if device == 'cuda' else []}
        # Save resume state first: if interrupted after this atomic write,
        # regenerate reports from checkpoint history on the next invocation.
        torch_save(latest, payload)
        write_csv(arm_dir / 'losses.csv', history)
        write_json(arm_dir / 'history.json', {'version': VERSION, 'arm': arm, 'binding': binding,
                                             'training_config': TRAINING_CONFIG,
                                             'model_configuration': projector.configuration(),
                                             'epochs': history, 'selection': TRAINING_CONFIG['selection']})
        print('BRIDGE_EPOCH', arm, epoch, 'train', round(train_metrics['mse'], 8),
              'dev', round(dev_metrics['mse'], 8), flush=True)
    projector.eval().requires_grad_(False)
    if len(history) != TRAINING_CONFIG['epochs']:
        raise RuntimeError('Final checkpoint requires the complete fixed epoch history')
    final_payload = {'version': VERSION, 'arm': arm, 'selected_epoch': TRAINING_CONFIG['epochs'],
                     'selection': TRAINING_CONFIG['selection'], 'binding': binding,
                     'training_config': TRAINING_CONFIG, 'model_configuration': projector.configuration(),
                     'state_dict': _cpu_copy(projector.state_dict()), 'final_metrics': history[-1],
                     'fresh_test_accessed': False}
    torch_save(final, final_payload)
    # A resume from the final completed epoch may have no loop iteration.
    write_csv(arm_dir / 'losses.csv', history)
    write_json(arm_dir / 'history.json', {'version': VERSION, 'arm': arm, 'binding': binding,
                                         'training_config': TRAINING_CONFIG,
                                         'model_configuration': projector.configuration(),
                                         'epochs': history, 'selection': TRAINING_CONFIG['selection']})
    record = {'version': VERSION, 'arm': arm, 'binding': binding,
              'selected_epoch': TRAINING_CONFIG['epochs'], 'selection': TRAINING_CONFIG['selection'],
              'model_configuration': projector.configuration(), 'checkpoint_path': arm + '/final.pt',
              'checkpoint_sha256': sha256(final), 'history_path': arm + '/history.json',
              'history_sha256': sha256(arm_dir / 'history.json'),
              'losses_sha256': sha256(arm_dir / 'losses.csv'), 'final_metrics': history[-1],
              'seconds_this_invocation': time.perf_counter() - started,
              'fresh_test_accessed': False}
    write_json(completed, record)
    print('BRIDGE_ARM_COMPLETE', arm, record['checkpoint_sha256'], flush=True)
    return record


def run(jepa_root, targets_root, out, freeze, publication, device='cpu', threads=4, resume=False):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was frozen but is not available; do not silently change backend')
    if threads < 1:
        raise ValueError('Positive thread count required')
    torch.set_num_threads(threads)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    frozen = validate_freeze(jepa_root, targets_root, freeze, publication, device, threads)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prepared, mean, std, provenance = load_training_data(jepa_root, targets_root)
    statistics_sha = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()
    binding = {'pretraining_freeze_sha256': sha256(freeze),
               'pretraining_publication_sha256': sha256(publication),
               'source_hashes': frozen['source_hashes'],
               'jepa_manifest_sha256': frozen['data']['jepa_manifest_sha256'],
               'target_manifest_sha256': frozen['data']['target_manifest_sha256'],
               'vae': frozen['data']['vae'], 'data_provenance': provenance,
               'normalization_statistics_sha256': statistics_sha,
               'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'numpy': str(np.__version__), 'device': device, 'threads': threads,
                           'precision': 'float32',
                           'gpu': torch.cuda.get_device_name() if device == 'cuda' else None}}
    binding_path = out / 'data_binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Existing output directory belongs to another data/runtime/source binding')
    write_json(binding_path, binding)
    normalization_path = out / 'normalization.npz'
    if normalization_path.exists():
        with np.load(normalization_path, allow_pickle=False) as cached:
            if not np.array_equal(cached['mean'], mean) or not np.array_equal(cached['std'], std):
                raise ValueError('Saved train-only normalization changed')
    else:
        save_npz(normalization_path, mean=mean, std=std)
    mean_map, baseline_report = untrained_baselines(prepared)
    baseline_report['data_binding_sha256'] = sha256(binding_path)
    baselines_path = out / 'untrained_baselines.json'
    if baselines_path.exists() and json.loads(baselines_path.read_text()) != baseline_report:
        raise ValueError('Saved train-only/source-noop comparator changed')
    write_json(baselines_path, baseline_report)
    mean_map_path = out / 'train_mean_latent_map.npz'
    if mean_map_path.exists():
        with np.load(mean_map_path, allow_pickle=False) as cached:
            if not np.array_equal(cached['latent'], mean_map.numpy()):
                raise ValueError('Saved train-only mean map changed')
    else:
        save_npz(mean_map_path, latent=mean_map.numpy())
    results = [fit_arm(arm, out, prepared, mean, std, binding, device, resume) for arm in ARMS]
    checkpoint_freeze = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                         'fresh_test_accessed': False, 'test_accessed': False,
                         'source_hashes': frozen['source_hashes'], 'training_config': TRAINING_CONFIG,
                         'pretraining_freeze_sha256': sha256(freeze),
                         'pretraining_publication_sha256': sha256(publication),
                         'data_binding_path': 'data_binding.json', 'data_binding_sha256': sha256(binding_path),
                         'normalization_path': 'normalization.npz', 'normalization_sha256': sha256(normalization_path),
                         'untrained_baselines_path': 'untrained_baselines.json', 'untrained_baselines_sha256': sha256(baselines_path),
                         'train_mean_latent_map_path': 'train_mean_latent_map.npz', 'train_mean_latent_map_sha256': sha256(mean_map_path),
                         'jepa_manifest_sha256': frozen['data']['jepa_manifest_sha256'],
                         'target_manifest_sha256': frozen['data']['target_manifest_sha256'],
                         'models': [{key: result[key] for key in ('arm', 'selected_epoch', 'selection',
                                      'model_configuration', 'checkpoint_path', 'checkpoint_sha256',
                                      'history_path', 'history_sha256', 'losses_sha256', 'final_metrics')}
                                    for result in results],
                         'scope': 'Genuine latent regression only; no edited-state or generator success claim',
                         'next_gate': 'Publish and byte-verify this checkpoint freeze plus fixed fresh-test protocol before opening any fresh held-out data.'}
    final_freeze_path = out / 'trained_checkpoint_freeze.json'
    if final_freeze_path.exists():
        existing = json.loads(final_freeze_path.read_text())
        if {key: value for key, value in existing.items() if key != 'created_utc'} != {
                key: value for key, value in checkpoint_freeze.items() if key != 'created_utc'}:
            raise ValueError('Existing final checkpoint freeze differs; refusing to overwrite it')
    else:
        write_json(final_freeze_path, checkpoint_freeze)
    print('BRIDGE_TRAINING_COMPLETE', final_freeze_path, sha256(final_freeze_path), flush=True)
    return checkpoint_freeze


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freezing = commands.add_parser('make-freeze')
    training = commands.add_parser('train')
    for command in (freezing, training):
        command.add_argument('--jepa-root', type=Path, required=True)
        command.add_argument('--targets-root', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
        command.add_argument('--threads', type=int, default=4)
    training.add_argument('--freeze', type=Path, required=True)
    training.add_argument('--publication', type=Path, required=True)
    training.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'make-freeze':
        make_freeze(args.jepa_root, args.targets_root, args.out, args.device, args.threads)
    else:
        run(args.jepa_root, args.targets_root, args.out, args.freeze, args.publication,
            args.device, args.threads, args.resume)
