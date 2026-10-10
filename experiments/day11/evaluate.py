"""Development-only assessment of the fixed genuine-JEPA -> Wan VAE bridge.

Eight existing development scenes are evaluated, without optimization or arm
selection. Genuine-target JEPA and true-target Wan latents are explicitly
privileged diagnostics. The deployable edited condition uses source JEPA,
source selection and a requested shift only. This measures a native-image
bridge; it is not a learned video rollout or a renderer-consistency result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = 'day11_development_bridge_evaluation_v1'
SHUFFLE_SEED = 1111
RENDERER_SEEDS = (13500, 13501)
REGIONS = ('global', 'source_hole', 'destination', 'distractor', 'background')
FEATURE_ARMS = ('source', 'genuine_target', 'copy_repair', 'wrong_direction', 'shuffled')
ARMS = ('true_source', 'true_target') + tuple(
    route + '_' + arm for route in ('absolute', 'residual') for arm in FEATURE_ARMS)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data = _module('day11_evaluation_data', HERE / 'data.py')
model = _module('day11_evaluation_model', HERE / 'model.py')
bridge = _module('day11_evaluation_vace_bridge', HERE / 'vace_bridge.py')


def _legacy_animation():
    """Bind Day10's legacy bare imports without confusing either data.py."""
    names = ('data', 'operators', 'extract', 'evaluate', 'probe')
    saved = {name: sys.modules.get(name) for name in names}
    original_path = sys.path[:]
    try:
        for name in names:
            sys.modules[name] = _module('day11_eval_legacy_' + name,
                                       REPO / 'experiments/day8' / (name + '.py'))
        return _module('day11_eval_legacy_animation', REPO / 'experiments/day10/animate.py')
    finally:
        sys.path[:] = original_path
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_projector(checkpoint, dataset, targets, device):
    """Require the final, fixed-budget checkpoint and its original data binding."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved.get('training_config', {})
    if (saved.get('version') != 'day11_genuine_jepa_to_wan_training_v1'
            or saved.get('selected_epoch') != 150 or config.get('epochs') != 150
            or saved.get('selection') != 'fixed final epoch; no development selection'
            or saved.get('fresh_test_accessed') is not False
            or saved.get('arm') not in ('cnn', 'linear')):
        raise ValueError('Expected a fixed epoch150 genuine-training final checkpoint')
    expected = {'batch_size': 4, 'seed': 1111, 'learning_rate': .001,
                'weight_decay': .0001, 'optimizer': 'AdamW', 'cnn_width': 64,
                'precision': 'float32'}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint differs from the initial training budget')
    binding = saved.get('binding', {})
    if (binding.get('jepa_manifest_sha256') != dataset.manifest_sha256
            or binding.get('target_manifest_sha256') != targets.manifest_sha256
            or binding.get('vae') != targets.manifest['vae']):
        raise ValueError('Checkpoint is not bound to these JEPA/VAE targets')
    required = {'experiments/day11/train.py', 'experiments/day11/data.py',
                'experiments/day11/model.py', 'experiments/day11/PROTOCOL.md',
                'experiments/day8/data.py'}
    if not required <= set(binding.get('source_hashes', {})):
        raise ValueError('Checkpoint does not bind every training source')
    for relative, digest in binding['source_hashes'].items():
        if data.sha256(data._child_path(REPO, relative)) != digest:
            raise ValueError('Checkpoint source changed: ' + relative)
    projector = model.build_model(saved['arm'], np.zeros(1024, np.float32),
                                  np.ones(1024, np.float32), width=64)
    projector.load_state_dict(saved['state_dict'], strict=True)
    if not bool(projector.normalization_is_fitted):
        raise ValueError('Missing training-only feature normalization')
    statistics = (projector.feature_mean.flatten().numpy().tobytes()
                  + projector.feature_std.flatten().numpy().tobytes())
    if hashlib.sha256(statistics).hexdigest() != binding.get('normalization_statistics_sha256'):
        raise ValueError('Checkpoint feature normalization hash mismatch')
    if projector.configuration() != saved['model_configuration']:
        raise ValueError('Checkpoint architecture metadata mismatch')
    return projector.eval().requires_grad_(False).to(device), saved


def load_frozen_vae(targets, model_cache, device):
    """Decode with exactly the pinned VAE files that produced supervision."""
    from diffusers import AutoencoderKLWan
    from huggingface_hub import hf_hub_download
    provenance = targets.manifest['vae']
    if (provenance.get('model_id') != bridge.MODEL_ID
            or provenance.get('class') != 'AutoencoderKLWan'
            or provenance.get('precision') != 'float32'
            or provenance.get('posterior') != 'mode'
            or len(provenance.get('revision', '')) != 40):
        raise ValueError('Unexpected frozen Wan VAE provenance')
    for filename in ('config.json', 'diffusion_pytorch_model.safetensors'):
        path = hf_hub_download(provenance['model_id'], 'vae/' + filename,
                               revision=provenance['revision'], cache_dir=model_cache)
        if data.sha256(path) != provenance['files'][filename]['sha256']:
            raise ValueError('VAE decoder file differs from the supervision encoder')
    vae = AutoencoderKLWan.from_pretrained(
        provenance['model_id'], subfolder='vae', revision=provenance['revision'],
        cache_dir=model_cache, torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    if (list(vae.config.latents_mean) != provenance['latents_mean']
            or list(vae.config.latents_std) != provenance['latents_std']):
        raise ValueError('VAE latent normalization mismatch')
    return vae


def _empty_latent(targets):
    record = targets.manifest['empty_video_latent']
    path = data._child_path(targets.root, record['path'])
    if data.sha256(path) != record['sha256']:
        raise ValueError('Normalized empty-video latent changed')
    value = np.load(path, allow_pickle=False)
    if value.shape != (1, 16, 1, 48, 48) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('Invalid normalized empty-video latent')
    return value.copy()


def _scoring_masks(dataset, row, animation):
    """Read geometry after inference inputs are fixed; never return it to F."""
    path = data._child_path(dataset.root, row['path'])
    if data.sha256(path) != row['sha256']:
        raise ValueError('Scene cache changed before scoring')
    with np.load(path, allow_pickle=False) as archive:
        source_fraction = archive['source_frac'].copy()
        target_fraction = archive['target_frac'][1].copy()
        distractor_fraction = archive['distractor_frac'].copy()
    for value in (source_fraction, target_fraction, distractor_fraction):
        if value.shape != (1, 24, 24) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError('Invalid scoring coverage')
    coarse, support = animation.day8_evaluate.fixed_regions(
        source_fraction, target_fraction, distractor_fraction, int(row['spec']['dx']) // 16)
    latent_masks = {name: np.repeat(np.repeat(coarse[name][0], 2, axis=0), 2, axis=1)
                    for name in REGIONS if name != 'global'}
    latent_masks['global'] = np.ones((48, 48), bool)
    pair = dataset._renderer.generate_pair(row['spec'])
    source, target, distractor = (pair[name][data.FRAME_INDEX].copy() for name in
                                  ('masks_source', 'masks_target', 'masks_distractor'))
    for name, pixels in (('source', pair['frames_source'][data.FRAME_INDEX]),
                         ('target', pair['frames_target'][data.FRAME_INDEX])):
        if data.array_sha256(pixels) != row[name + '_rgb_sha256']:
            raise ValueError('Scoring renderer RGB differs from cached frame15')
    for mask, fraction in ((source, source_fraction), (target, target_fraction),
                           (distractor, distractor_fraction)):
        if not np.allclose(animation.image_patch_fractions(mask), fraction, atol=1e-6, rtol=0):
            raise ValueError('Regenerated scoring masks differ from cache')
    del pair
    protected = np.repeat(np.repeat((support | (distractor_fraction > 0))[0], 16, axis=0), 16, axis=1)
    rgb_masks = {'global': np.ones((384, 384), bool), 'source_hole': source & ~target,
                 'destination': target, 'distractor': distractor, 'background': ~protected}
    return source_fraction, latent_masks, rgb_masks, {
        'selected_source_mask': source, 'selected_target_mask': target, 'distractor_mask': distractor}


def _error_metrics(predicted, target, baseline, regions, prefix, channel_axis):
    error = np.mean(np.square(predicted.astype(np.float64) - target), axis=channel_axis)
    noop = np.mean(np.square(baseline.astype(np.float64) - target), axis=channel_axis)
    result = {}
    for name in REGIONS:
        mask = regions[name]
        count = int(mask.sum())
        numerator = float(error[mask].sum(dtype=np.float64))
        denominator = float(noop[mask].sum(dtype=np.float64))
        mse = numerator / count if count else None
        baseline_mse = denominator / count if count else None
        valid = count > 0 and baseline_mse > 1e-12
        stem = prefix + '_' + name
        result.update({stem + '_count': count, stem + '_squared_error_sum': numerator,
                       stem + '_mse': mse, stem + '_source_baseline_mse': baseline_mse,
                       stem + '_ratio': mse / baseline_mse if valid else None,
                       stem + '_ratio_degenerate': not valid})
    primary = [result[prefix + '_' + name + '_ratio'] for name in ('source_hole', 'destination')]
    result[prefix + '_primary_ratio'] = float(np.mean(primary)) if all(v is not None for v in primary) else None
    return result


def _arm_metadata():
    arms = {'true_source': {'oracle': False, 'input': 'source RGB through frozen Wan VAE',
                            'route': 'true source latent; no-edit baseline'},
            'true_target': {'oracle': True, 'input': 'genuine target RGB through frozen Wan VAE',
                            'route': 'privileged renderer-interface/decoder-ceiling diagnostic'}}
    descriptions = {'source': 'unchanged source JEPA',
                    'genuine_target': 'independently encoded genuine target JEPA; privileged diagnostic',
                    'copy_repair': 'source-only JEPA copy/repair with supplied source selection and requested shift',
                    'wrong_direction': 'matched source-only JEPA edit with the opposite displacement',
                    'shuffled': 'copy/repair JEPA grid permuted spatially using one fixed seed1111 permutation'}
    for route in ('absolute', 'residual'):
        for name, description in descriptions.items():
            arms[route + '_' + name] = {'oracle': name == 'genuine_target', 'input': description,
                                        'route': 'F(z)' if route == 'absolute' else 'A(source) + (F(z) - F(source))'}
    return arms


def _summarize(rows):
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row['arm'] == arm]
        metrics = {}
        for key in sorted(set().union(*(row.keys() for row in selected))):
            if key.endswith('_mse') or key.endswith('_ratio'):
                values = [row[key] for row in selected if row.get(key) is not None]
                metrics[key] = float(np.mean(values)) if values else None
        summary[arm] = {'scene_count': len(selected), **metrics}
    return summary


def evaluate(checkpoint, cache_root, target_root, out, decode=False,
             threads=4, model_cache='/tmp/day11-model', device='cpu'):
    torch.set_num_threads(threads)
    torch.manual_seed(SHUFFLE_SEED)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dataset = data.NativeImageDataset(cache_root, 'dev')
    targets = data.LatentTargetStore(target_root, dataset.manifest_sha256)
    targets.validate_dataset_coverage(dataset)
    projector, saved = load_projector(checkpoint, dataset, targets, device)
    animation = _legacy_animation()
    vae = load_frozen_vae(targets, model_cache, device) if decode else None
    empty = _empty_latent(targets)
    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(24 * 24)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'evaluation_manifest.json'
    if manifest_path.exists():
        raise ValueError('Use a new output directory; existing evaluation evidence is not overwritten')
    source_paths = ['experiments/day11/' + name for name in ('evaluate.py', 'data.py', 'model.py', 'vace_bridge.py')]
    source_paths += ['experiments/day10/animate.py'] + ['experiments/day8/' + name for name in
                                                      ('data.py', 'operators.py', 'extract.py', 'evaluate.py', 'probe.py')]
    base = {'version': VERSION, 'split': 'existing development only', 'complete': False,
            'checkpoint_sha256': data.sha256(checkpoint), 'translator_arm': saved['arm'],
            'selected_epoch': saved['selected_epoch'], 'jepa_manifest_sha256': dataset.manifest_sha256,
            'target_manifest_sha256': targets.manifest_sha256, 'vae': targets.manifest['vae'],
            'source_hashes': {path: data.sha256(REPO / path) for path in source_paths},
            'arms': _arm_metadata(), 'all_development_seeds': list(range(13500, 13508)),
            'renderer_seeds': list(RENDERER_SEEDS), 'shuffle_seed': SHUFFLE_SEED,
            'shuffle_permutation_sha256': data.array_sha256(permutation.astype(np.int64)),
            'shuffle_policy': 'One common permutation of all576 copy/repair feature vectors; channels preserved; coordinates remain destination coordinates',
            'latent_regions': 'Day8 fixed regions from any-coverage24x24 masks, nearest expanded2x; background excludes one-patch dilation of source/destination union and distractor patches',
            'rgb_regions': 'Exact renderer source hole/destination/distractor masks; protected background expands the same conservative coarse background mask16x',
            'metric_target': 'independently rendered target RGB or its true normalized Wan latent',
            'rgb_metric_space': 'VAE decoded RGB clipped to[0,1], FP32; source baseline is unedited raw source RGB in[0,1]' if decode else None,
            'aggregation': 'Equal mean over all eight existing development scenes, one requested signed shift per scene; no confidence interval or held-out claim',
            'decoded': bool(decode), 'fresh_test_accessed': False,
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'numpy': str(np.__version__), 'device': device, 'threads': threads},
            'scenes': []}
    _write_json(manifest_path, base)
    all_rows = []
    for scene_index, row in enumerate(dataset.rows):
        spec = row['spec']
        source_index, target_index = scene_index * 2, scene_index * 2 + 1
        source_example, target_example = dataset[source_index], dataset[target_index]
        source_feature = source_example['jepa'].numpy().transpose(1, 2, 0)[None].copy()
        fraction, latent_masks, rgb_masks, object_masks = _scoring_masks(dataset, row, animation)
        edited, _, invariants = animation.translate_features(source_feature, fraction, [0, spec['dx']])
        if not np.array_equal(edited['copy_repair'][0], source_feature):
            raise AssertionError('Zero-shift JEPA operator is not exact identity')
        shuffled = edited['copy_repair'][1].reshape(576, 1024)[permutation].reshape(1, 24, 24, 1024).copy()
        feature_conditions = {'source': source_example['jepa'][None],
                              'genuine_target': target_example['jepa'][None]}
        for name, condition in (('copy_repair', edited['copy_repair'][1]),
                                ('wrong_direction', edited['wrong_direction'][1]), ('shuffled', shuffled)):
            feature_conditions[name] = torch.from_numpy(condition.transpose(0, 3, 1, 2).copy())
        source_latent = targets.get(source_example['sample_id'], source_example['rgb_sha256']).numpy()
        target_latent = targets.get(target_example['sample_id'], target_example['rgb_sha256']).numpy()
        predictions = {}
        with torch.inference_mode():
            for name, condition in feature_conditions.items():
                predictions[name] = projector(condition.to(device)).cpu().numpy()[0].copy()
            zero = model.source_preserving_prediction(
                projector, feature_conditions['source'].to(device), feature_conditions['source'].to(device),
                torch.from_numpy(source_latent[None]).to(device)).cpu().numpy()[0]
        if not np.array_equal(zero, source_latent):
            raise AssertionError('Analytic residual no-edit is not bitwise equal to the true source latent')
        latents = {'true_source': source_latent, 'true_target': target_latent}
        for name, predicted in predictions.items():
            latents['absolute_' + name] = predicted
            latents['residual_' + name] = source_latent + (predicted - predictions['source'])
        if not np.array_equal(latents['residual_source'], source_latent):
            raise AssertionError('Saved residual_source must be an exact latent identity')
        if any(value.shape != (16, 48, 48) or value.dtype != np.float32 or not np.isfinite(value).all()
               for value in latents.values()):
            raise ValueError('Invalid/nonfinite projected Wan latent')
        source_rgb, target_rgb = dataset.rgb_uint8(source_index), dataset.rgb_uint8(target_index)
        arrays = {'source_rgb': source_rgb, 'target_rgb': target_rgb,
                  'normalized_empty_video_latent': empty, 'shuffle_permutation': permutation.astype(np.int64),
                  **object_masks, **{'latent__' + name: value for name, value in latents.items()},
                  **{'latent_mask__' + name: value for name, value in latent_masks.items()},
                  **{'rgb_mask__' + name: value for name, value in rgb_masks.items()}}
        rows = []
        for arm in ARMS:
            result = {'scene': spec['name'], 'seed': int(spec['seed']), 'shift_px': int(spec['dx']),
                      'arm': arm, 'oracle': base['arms'][arm]['oracle'], 'translator_arm': saved['arm']}
            result.update(_error_metrics(latents[arm], target_latent, source_latent, latent_masks, 'latent', 0))
            if vae is not None:
                with torch.inference_mode():
                    normalized = torch.from_numpy(latents[arm][None, :, None]).to(device)
                    decoded = vae.decode(bridge.denormalize_wan_latents(vae, normalized)).sample
                if tuple(decoded.shape) != (1, 3, 1, 384, 384) or not torch.isfinite(decoded).all():
                    raise ValueError('Unexpected/nonfinite frozen VAE reconstruction')
                rgb = decoded[0, :, 0].float().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().numpy().copy()
                arrays['decoded_rgb__' + arm] = rgb
                result.update(_error_metrics(rgb, target_rgb.astype(np.float32) / 255,
                                             source_rgb.astype(np.float32) / 255, rgb_masks, 'rgb', 2))
            rows.append(result)
        metadata = {'version': VERSION, 'scene': spec['name'], 'seed': int(spec['seed']),
                    'shift_px': int(spec['dx']), 'frame_index': data.FRAME_INDEX,
                    'checkpoint_sha256': base['checkpoint_sha256'], 'translator_arm': saved['arm'],
                    'jepa_manifest_sha256': dataset.manifest_sha256,
                    'target_manifest_sha256': targets.manifest_sha256, 'vae': base['vae'],
                    'renderer_bundle': spec['seed'] in RENDERER_SEEDS, 'arms': base['arms'],
                    'operator_invariants': invariants, 'residual_zero_exact': True,
                    'source_rgb_sha256': source_example['rgb_sha256'], 'target_rgb_sha256': target_example['rgb_sha256'],
                    'latent_sha256': {name: data.array_sha256(value) for name, value in latents.items()},
                    'latent_regions': base['latent_regions'], 'rgb_regions': base['rgb_regions'],
                    'shuffle_policy': base['shuffle_policy'], 'fresh_test_accessed': False}
        arrays['metadata_json'] = np.frombuffer(json.dumps(metadata, sort_keys=True, allow_nan=False).encode('utf-8'), np.uint8)
        scene_path = out / 'scenes' / (spec['name'] + '.npz')
        _save_npz(scene_path, arrays)
        _write_json(out / 'scenes' / (spec['name'] + '_metrics.json'), rows)
        all_rows.extend(rows)
        base['scenes'].append({'scene': spec['name'], 'seed': int(spec['seed']), 'shift_px': int(spec['dx']),
                               'path': str(scene_path.relative_to(out)), 'sha256': data.sha256(scene_path),
                               'renderer_bundle': spec['seed'] in RENDERER_SEEDS, 'residual_zero_exact': True})
        _write_json(manifest_path, base)
        print('DEV_SCENE_COMPLETE', spec['name'], 'shift', spec['dx'], 'decoded', bool(decode), flush=True)
    if len(base['scenes']) != 8 or len(all_rows) != 8 * len(ARMS):
        raise AssertionError('Evaluation must include all eight development scenes and all arms')
    _write_json(out / 'metrics.json', all_rows)
    _write_csv(out / 'metrics.csv', all_rows)
    summary = {'version': VERSION, 'translator_arm': saved['arm'], 'scene_count': 8,
               'aggregation': base['aggregation'], 'arms': base['arms'], 'methods': _summarize(all_rows),
               'all_residual_zero_exact': True, 'fresh_test_accessed': False,
               'limits': ['Existing procedural development images only; no unseen test claim.',
                          'Genuine-target conditions are privileged diagnostics, not source-only editing results.',
                          'A true-target VAE reconstruction is a decoder ceiling, not a translator result.',
                          'Exact source-latent identity does not enforce exact decoded or rendered pixels.',
                          'No temporal model or video consistency is evaluated by this native-image pilot.']}
    _write_json(out / 'summary.json', summary)
    base['complete'] = True
    base['renderer_scene_paths'] = [scene['path'] for scene in base['scenes'] if scene['renderer_bundle']]
    base['metrics_sha256'] = data.sha256(out / 'metrics.json')
    base['summary_sha256'] = data.sha256(out / 'summary.json')
    _write_json(manifest_path, base)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--target-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--decode', action='store_true', help='Also decode every arm with the exact frozen Wan VAE')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--model-cache', default='/tmp/day11-model')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    evaluate(args.checkpoint, args.cache_root, args.target_root, args.out,
             args.decode, args.threads, args.model_cache, args.device)


if __name__ == '__main__':
    main()
