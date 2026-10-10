"""Frozen eight-scene test of both Day11 translators and all transport arms.

Nothing is generated here. Fresh pairs must already have been prepared by the
separate, publication-gated prepare_fresh.py runner. This runner validates that
same freeze/receipt, all numerical sources, both final checkpoints and its own
configuration BEFORE opening the fresh cache. It never extends the train/dev
dataset whitelist, fits parameters, updates normalization or selects an arm.

The only source-mask consumer is the existing JEPA editing operator. The
translator sees JEPA only; provenance transport sees source/edited JEPA,
source Wan appearance and frozen predictions only. Genuine target JEPA and
target RGB are reserved for explicitly labelled oracle arms and scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = 'day11_fresh_bridge_evaluation_v1'
TARGET_VERSION = 'day11_fresh_wan_targets_v1'
TRANSLATORS = ('cnn', 'linear')


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _module('day11_fresh_reused_evaluator', HERE / 'evaluate.py')
prepare = _module('day11_fresh_reused_preparation', HERE / 'prepare_fresh.py')
transport = _module('day11_fresh_reused_transport', HERE / 'transport.py')
ARMS = ev.ARMS + transport.NEW_ARMS
REQUIRED_SOURCES = tuple(sorted(set(prepare.REQUIRED_SOURCES) | {
    'experiments/day11/' + name for name in
    ('evaluate_fresh.py', 'evaluate.py', 'transport.py', 'data.py', 'model.py', 'vace_bridge.py',
     'train.py', 'render_fresh.py', 'render.py')
} | {'experiments/day8/' + name for name in ('operators.py', 'evaluate.py', 'probe.py')}))


def evaluation_config(device='cpu', threads=4, decode=False):
    """Use this exact object as freeze['evaluation'] before any test access."""
    return {'device': device, 'threads': threads, 'precision': 'float32',
            'decode': bool(decode), 'translator_arms': list(TRANSLATORS),
            'arms': list(ARMS), 'shuffle_seed': ev.SHUFFLE_SEED,
            'local_radius': transport.LOCAL_RADIUS}


def validate_pretest(freeze, publication, device, threads, decode):
    """Only pre-existing development/training artifacts are opened here."""
    frozen = json.loads(Path(freeze).read_text())
    extraction = frozen.get('extraction', {})
    frozen, trained = prepare.validate_freeze(
        freeze, publication, extraction.get('device'), extraction.get('threads'))
    if not set(REQUIRED_SOURCES) <= set(frozen.get('source_hashes', {})):
        raise ValueError('Published freeze does not bind every fresh evaluation dependency')
    if frozen.get('evaluation') != evaluation_config(device, threads, decode):
        raise ValueError('Evaluation runtime/arms differ from the published pretest freeze')
    rendering = frozen.get('rendering', {})
    rendering_module = _module('day11_fresh_frozen_renderer_configuration', HERE / 'render_fresh.py')
    for field in ('prompt_cache_sha256', 'prompt_metadata_sha256'):
        value = rendering.get(field, '')
        if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
            raise ValueError('Missing frozen renderer prompt-cache hash: ' + field)
    if rendering != rendering_module.rendering_config(
            rendering['prompt_cache_sha256'], rendering['prompt_metadata_sha256'],
            threads=rendering.get('threads')):
        raise ValueError('Renderer settings are not fixed by the same published pretest freeze')
    if threads < 1 or device not in ('cpu', 'cuda'):
        raise ValueError('Invalid fixed evaluation runtime')
    trained_path = prepare._bound_file(freeze, frozen['trained_checkpoint_freeze'])
    binding_path = prepare._bound_file(trained_path, {
        'path': trained['data_binding_path'], 'sha256': trained['data_binding_sha256']})
    binding = json.loads(binding_path.read_text())
    for name in ('source_hashes', 'jepa_manifest_sha256', 'target_manifest_sha256'):
        if binding.get(name) != trained.get(name):
            raise ValueError('Training freeze/data binding disagree: ' + name)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Published CUDA evaluation requested but CUDA is unavailable')
    return frozen, trained, trained_path, binding


def load_projectors(trained, trained_path, binding, device):
    """Validate the original training binding without treating test as train."""
    projectors, checkpoints = {}, {}
    for record in trained['models']:
        name = record['arm']
        path = prepare._bound_file(trained_path, {
            'path': record['checkpoint_path'], 'sha256': record['checkpoint_sha256']})
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved.get('version') != 'day11_genuine_jepa_to_wan_training_v1'
                or saved.get('arm') != name or saved.get('selected_epoch') != 150
                or saved.get('selection') != 'fixed final epoch; no development selection'
                or saved.get('fresh_test_accessed') is not False
                or saved.get('training_config') != trained['training_config']
                or saved.get('binding') != binding):
            raise ValueError('Frozen final checkpoint does not match completed training: ' + name)
        projector = ev.model.build_model(name, np.zeros(1024, np.float32),
                                         np.ones(1024, np.float32), width=64)
        projector.load_state_dict(saved['state_dict'], strict=True)
        statistics = (projector.feature_mean.flatten().numpy().tobytes()
                      + projector.feature_std.flatten().numpy().tobytes())
        if (not bool(projector.normalization_is_fitted)
                or hashlib.sha256(statistics).hexdigest() != binding['normalization_statistics_sha256']
                or projector.configuration() != record['model_configuration']
                or projector.configuration() != saved['model_configuration']):
            raise ValueError('Frozen model architecture/training statistics mismatch')
        projectors[name] = projector.eval().requires_grad_(False).to(device)
        checkpoints[name] = record['checkpoint_sha256']
    if set(projectors) != set(TRANSLATORS):
        raise ValueError('Both frozen CNN and linear translators are required')
    return projectors, checkpoints


class FreshPairs:
    """Strict standalone test reader; never renders or changes the dev loader."""
    def __init__(self, root, frozen, freeze, publication):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / 'features_manifest.json'
        self.manifest_sha256 = ev.data.sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        manifest = self.manifest
        _, extractor = prepare._legacy_modules()
        required = {'version': prepare.VERSION, 'test_specs': prepare.test_specs(),
                    'test_freeze_sha256': ev.data.sha256(freeze),
                    'test_publication_sha256': ev.data.sha256(publication),
                    'trained_checkpoint_freeze_sha256': frozen['trained_checkpoint_freeze']['sha256'],
                    'development_gate_sha256': frozen['development_gate']['sha256'],
                    'source_hashes': frozen['source_hashes'],
                    'model': extractor.MODEL, 'upstream': extractor.UPSTREAM,
                    'weights_sha256': extractor.WEIGHT_SHA256,
                    'modality': 'native single image [1,3,1,384,384]',
                    'precision': 'float32', 'cache_precision': 'float32'}
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise ValueError('Fresh JEPA cache binding mismatch: ' + key)
        if any(manifest.get('runtime', {}).get(key) != frozen['extraction'][key]
               for key in ('device', 'threads')):
            raise ValueError('Fresh JEPA cache used a different extraction runtime')
        self.rows = manifest.get('scenes', [])
        if [row.get('spec') for row in self.rows] != prepare.test_specs():
            raise ValueError('Require exactly eight complete, ordered, predeclared fresh pairs')
        self._cache = {}

    def scene(self, index):
        row = self.rows[index]
        if index in self._cache:
            return self._cache[index]
        paths = {row['path']: row['sha256'], row['pixel_path']: row['pixel_sha256']}
        if row.get('file_sha256') != paths:
            raise ValueError('Fresh scene file bindings disagree')
        for relative, digest in paths.items():
            if ev.data.sha256(ev.data._child_path(self.root, relative)) != digest:
                raise ValueError('Fresh scene file changed: ' + relative)
        with np.load(ev.data._child_path(self.root, row['path']), allow_pickle=False) as archive:
            features = {name: archive[name].copy() for name in
                        ('tokens', 'source_frac', 'target_frac', 'distractor_frac')}
        tokens = features['tokens']
        if tokens.shape != (2, 1, 24, 24, 1024) or tokens.dtype != np.float32 or not np.isfinite(tokens).all():
            raise ValueError('Invalid fresh native-image JEPA features')
        with np.load(ev.data._child_path(self.root, row['pixel_path']), allow_pickle=False) as archive:
            if set(archive.files) != {'source_rgb', 'target_rgb', 'source_mask', 'target_mask', 'distractor_mask'}:
                raise ValueError('Unexpected fresh pixel archive fields')
            pixels = {name: archive[name].copy() for name in archive.files}
        expected_ids = [row['spec']['name'] + '__' + view for view in ('source', 'target')]
        if row.get('sample_ids') != expected_ids:
            raise ValueError('Unexpected fresh sample IDs')
        for view_index, view in enumerate(('source', 'target')):
            rgb = pixels[view + '_rgb']
            if (rgb.shape != (384, 384, 3) or rgb.dtype != np.uint8
                    or ev.data.array_sha256(rgb) != row[view + '_rgb_sha256']
                    or ev.data.array_sha256(tokens[view_index]) != row[view + '_tokens_sha256']):
                raise ValueError('Fresh RGB/JEPA raw-array hash mismatch: ' + view)
        if features['target_frac'].shape != (2, 1, 24, 24):
            raise ValueError('Invalid fresh target coverage shape')
        fractions = {'source': features['source_frac'], 'target': features['target_frac'][1],
                     'distractor': features['distractor_frac']}
        for name, fraction in fractions.items():
            mask = pixels[name + '_mask']
            if (mask.shape != (384, 384) or mask.dtype != np.bool_ or fraction.shape != (1, 24, 24)
                    or fraction.dtype != np.float32 or not np.isfinite(fraction).all()
                    or not np.array_equal(prepare.patch_fraction(mask), fraction)):
                raise ValueError('Fresh coverage differs from its scoring mask: ' + name)
        if not np.array_equal(features['target_frac'][0], fractions['source']):
            raise ValueError('Fresh source-view coverage mismatch')
        self._cache[index] = features, pixels
        return features, pixels


def _save_npy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def encode_targets(dataset, vae, provenance, root, freeze_hash, device, threads):
    """Cache all16 true evaluation latents separately from frozen model inputs."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'latent_manifest.json'
    base = {'version': TARGET_VERSION, 'feature_manifest_sha256': dataset.manifest_sha256,
            'test_freeze_sha256': freeze_hash, 'vae': provenance,
            'runtime': {'device': device, 'threads': threads, 'precision': 'float32',
                        'torch': str(torch.__version__), 'numpy': str(np.__version__)}}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        **base, 'fresh_test_accessed': True, 'complete': False, 'items': []}
    if any(manifest.get(key) != value for key, value in base.items()):
        raise ValueError('Cannot resume fresh VAE targets with changed bindings/runtime')
    expected_ids = [row['spec']['name'] + '__' + view for row in dataset.rows for view in ('source', 'target')]
    if [item['sample_id'] for item in manifest['items']] != expected_ids[:len(manifest['items'])]:
        raise ValueError('Fresh target cache contains unexpected/duplicate/reordered items')
    ev._write_json(manifest_path, manifest)
    if 'empty_video_latent' not in manifest:
        with torch.inference_mode():
            raw = vae.encode(torch.zeros(1, 3, 1, 384, 384, device=device)).latent_dist.mode()
            empty = ev.bridge.normalize_wan_latents(vae, raw).cpu().numpy()
        _save_npy(root / 'normalized_empty.npy', empty)
        manifest['empty_video_latent'] = {'path': 'normalized_empty.npy', 'sha256': ev.data.sha256(root / 'normalized_empty.npy')}
        ev._write_json(manifest_path, manifest)
    empty_path = ev.data._child_path(root, manifest['empty_video_latent']['path'])
    if ev.data.sha256(empty_path) != manifest['empty_video_latent']['sha256']:
        raise ValueError('Fresh normalized empty-video cache changed')
    empty = np.load(empty_path, allow_pickle=False)
    if empty.shape != (1, 16, 1, 48, 48) or empty.dtype != np.float32 or not np.isfinite(empty).all():
        raise ValueError('Invalid normalized empty-video latent')
    values = {}
    for index, row in enumerate(dataset.rows):
        _, pixels = dataset.scene(index)
        for view in ('source', 'target'):
            sample_id = row['spec']['name'] + '__' + view
            record = next((item for item in manifest['items'] if item['sample_id'] == sample_id), None)
            if record is None:
                image = torch.from_numpy(pixels[view + '_rgb']).permute(2, 0, 1)[None, :, None]
                image = image.to(device).float().div(127.5).sub(1)
                with torch.inference_mode():
                    raw = vae.encode(image).latent_dist.mode()
                    value = ev.bridge.normalize_wan_latents(vae, raw).cpu().numpy()[0, :, 0].copy()
                path = root / (sample_id + '.npy')
                _save_npy(path, value)
                record = {'sample_id': sample_id, 'path': path.name, 'sha256': ev.data.sha256(path),
                          'rgb_sha256': row[view + '_rgb_sha256']}
                manifest['items'].append(record)
                ev._write_json(manifest_path, manifest)
                print('FRESH_WAN_TARGET', sample_id, flush=True)
            path = ev.data._child_path(root, record['path'])
            if (record['rgb_sha256'] != row[view + '_rgb_sha256']
                    or ev.data.sha256(path) != record['sha256']):
                raise ValueError('Fresh target/RGB binding changed')
            value = np.load(path, allow_pickle=False)
            if value.shape != (16, 48, 48) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid fresh true Wan latent')
            values[sample_id] = value
    manifest['complete'] = True
    ev._write_json(manifest_path, manifest)
    return values, empty, ev.data.sha256(manifest_path)


def scoring_masks(features, pixels, dx, animation):
    coarse, support = animation.day8_evaluate.fixed_regions(
        features['source_frac'], features['target_frac'][1], features['distractor_frac'], dx // 16)
    latent = {name: np.repeat(np.repeat(coarse[name][0], 2, axis=0), 2, axis=1)
              for name in ev.REGIONS if name != 'global'}
    latent['global'] = np.ones((48, 48), bool)
    protected = support | (features['distractor_frac'] > 0)
    rgb = {'global': np.ones((384, 384), bool),
           'source_hole': pixels['source_mask'] & ~pixels['target_mask'],
           'destination': pixels['target_mask'], 'distractor': pixels['distractor_mask'],
           'background': ~np.repeat(np.repeat(protected[0], 16, axis=0), 16, axis=1)}
    return latent, rgb


def edited_conditions(source, selected_fraction, dx, permutation, animation):
    """No target tensor, mask, RGB or renderer metadata enters this function."""
    states, _, invariants = animation.translate_features(source, selected_fraction, [0, dx])
    if not np.array_equal(states['copy_repair'][0], source):
        raise AssertionError('JEPA zero edit must be exact source identity')
    return {'source': source, 'copy_repair': states['copy_repair'][1],
            'wrong_direction': states['wrong_direction'][1],
            'shuffled': states['copy_repair'][1].reshape(576, 1024)[permutation].reshape(1, 24, 24, 1024).copy()}, invariants


def predict_all(projector, conditions, source_latent, target_latent, device):
    """The target latent is copied only into the true_target oracle output."""
    predictions = {}
    with torch.inference_mode():
        for name in ev.FEATURE_ARMS:
            tensor = torch.from_numpy(conditions[name].transpose(0, 3, 1, 2).copy()).to(device)
            predictions[name] = projector(tensor).cpu().numpy()[0].copy()
    latents = {'true_source': source_latent, 'true_target': target_latent}
    evidence, counts = {}, {}
    for name in ev.FEATURE_ARMS:
        latents['absolute_' + name] = predictions[name]
        latents['residual_' + name] = source_latent + (predictions[name] - predictions['source'])
        outputs, provenance, statistics = transport.transport_predictions(
            conditions['source'], conditions[name], source_latent, predictions['source'], predictions[name])
        latents.update({route + '_' + name: value for route, value in outputs.items()})
        evidence.update({'provenance__' + name + '__' + key: value for key, value in provenance.items()})
        counts[name] = statistics
    for name in ('residual_source', 'provenance_copy_source', 'provenance_local_source', 'provenance_residual_source'):
        if not np.array_equal(latents[name], source_latent):
            raise AssertionError('Zero-edit latent identity failed: ' + name)
    if set(latents) != set(ARMS) or any(value.shape != (16, 48, 48) or value.dtype != np.float32
                                       or not np.isfinite(value).all() for value in latents.values()):
        raise ValueError('Missing or invalid fresh output arm')
    return latents, evidence, counts


def run(cache_root, freeze, publication, out, device='cpu', threads=4,
        decode=False, model_cache='/tmp/day11-model'):
    frozen, trained, trained_path, binding = validate_pretest(freeze, publication, device, threads, decode)
    torch.set_num_threads(threads)
    torch.manual_seed(ev.SHUFFLE_SEED)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    projectors, checkpoints = load_projectors(trained, trained_path, binding, device)
    # This object carries only the original hash-bound VAE provenance. It is
    # not a train/dev dataset and does not classify any test example as train.
    vae = ev.load_frozen_vae(SimpleNamespace(manifest={'vae': binding['vae']}), model_cache, device)
    # The FIRST fresh-file access is below, after every pretest gate above.
    dataset = FreshPairs(cache_root, frozen, freeze, publication)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    freeze_hash, publication_hash = ev.data.sha256(freeze), ev.data.sha256(publication)
    targets, empty, target_hash = encode_targets(dataset, vae, binding['vae'], out / 'targets',
                                                freeze_hash, device, threads)
    arms = {**ev._arm_metadata(), **transport._new_metadata()}
    animation = ev._legacy_animation()
    permutation = np.random.default_rng(ev.SHUFFLE_SEED).permutation(576)
    base = {'version': VERSION, 'split': 'fresh held-out test', 'fresh_test_accessed': True,
            'test_freeze_sha256': freeze_hash, 'test_publication_sha256': publication_hash,
            'trained_checkpoint_freeze_sha256': frozen['trained_checkpoint_freeze']['sha256'],
            'feature_manifest_sha256': dataset.manifest_sha256, 'target_manifest_sha256': target_hash,
            'checkpoint_sha256_by_translator': checkpoints, 'vae': binding['vae'],
            'source_hashes': frozen['source_hashes'], 'evaluation': evaluation_config(device, threads, decode),
            'arms': arms, 'translator_arms': list(TRANSLATORS), 'test_specs': prepare.test_specs(),
            'decoded': bool(decode), 'runtime': {'device': device, 'threads': threads,
                'precision': 'float32', 'python': platform.python_version(), 'torch': str(torch.__version__),
                'numpy': str(np.__version__)},
            'aggregation': 'Equal means over all eight predeclared test scenes; every arm retained; no adaptation or selection',
            'latent_regions': 'Day8 fixed any-coverage24x24 regions expanded2x; protected background excludes one-patch union halo and distractor patches',
            'rgb_regions': 'Exact source hole/destination/distractor; conservative protected background from coarse regions expanded16x',
            'rgb_metric_space': 'Clipped[0,1] FP32 decoded RGB vs genuine raw target; source baseline raw source RGB' if decode else None}
    manifest_path = out / 'evaluation_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {**base, 'complete': False, 'scenes': []}
    if any(manifest.get(key) != value for key, value in base.items()):
        raise ValueError('Cannot resume fresh evaluation with changed sources/data/checkpoints/runtime')
    expected_names = [spec['name'] for spec in prepare.test_specs()]
    if [record['scene'] for record in manifest['scenes']] != expected_names[:len(manifest['scenes'])]:
        raise ValueError('Existing evaluation scenes are not a unique ordered prefix')
    ev._write_json(manifest_path, manifest)
    rows, count_rows = [], []
    for index, record in enumerate(dataset.rows):
        spec = record['spec']
        if index < len(manifest['scenes']):
            previous = manifest['scenes'][index]
            if set(previous['bundles']) != set(TRANSLATORS):
                raise ValueError('Incomplete previous scene translator coverage')
            for reference in previous['bundles'].values():
                for path_field, hash_field in (('path', 'sha256'), ('metrics_path', 'metrics_sha256'),
                                                ('counts_path', 'counts_sha256')):
                    path = ev.data._child_path(out, reference[path_field])
                    if ev.data.sha256(path) != reference[hash_field]:
                        raise ValueError('Completed test evidence changed: ' + str(path))
                rows.extend(json.loads(ev.data._child_path(out, reference['metrics_path']).read_text()))
                count_rows.extend(json.loads(ev.data._child_path(out, reference['counts_path']).read_text()))
            print('FRESH_SCENE_REUSED', spec['name'], flush=True)
            continue
        features, pixels = dataset.scene(index)
        source = features['tokens'][0]
        conditions, invariants = edited_conditions(source, features['source_frac'], spec['dx'], permutation, animation)
        # Counterfactual target features are introduced only under this oracle key.
        conditions['genuine_target'] = features['tokens'][1]
        source_latent = targets[spec['name'] + '__source']
        target_latent = targets[spec['name'] + '__target']
        latent_masks, rgb_masks = scoring_masks(features, pixels, spec['dx'], animation)
        scene_record = {'scene': spec['name'], 'seed': int(spec['seed']), 'shift_px': int(spec['dx']), 'bundles': {}}
        first_local = None
        for translator in TRANSLATORS:
            latents, evidence, counts = predict_all(projectors[translator], conditions, source_latent, target_latent, device)
            local = {name: latents['provenance_local_' + name] for name in ev.FEATURE_ARMS}
            if first_local is None:
                first_local = local
            elif any(not np.array_equal(local[name], first_local[name]) for name in ev.FEATURE_ARMS):
                raise AssertionError('No-F local comparator changed with translator architecture')
            arrays = {**evidence, 'source_rgb': pixels['source_rgb'], 'target_rgb': pixels['target_rgb'],
                      'selected_source_mask': pixels['source_mask'], 'selected_target_mask': pixels['target_mask'],
                      'distractor_mask': pixels['distractor_mask'], 'normalized_empty_video_latent': empty,
                      'shuffle_permutation': permutation.astype(np.int64),
                      **{'latent__' + name: value for name, value in latents.items()},
                      **{'latent_mask__' + name: value for name, value in latent_masks.items()},
                      **{'rgb_mask__' + name: value for name, value in rgb_masks.items()}}
            scene_rows, decoded_source = [], None
            for arm in ARMS:
                result = {'scene': spec['name'], 'seed': int(spec['seed']), 'shift_px': int(spec['dx']),
                          'arm': arm, 'oracle': arms[arm]['oracle'], 'translator_arm': translator, 'decoded': bool(decode)}
                result.update(ev._error_metrics(latents[arm], target_latent, source_latent, latent_masks, 'latent', 0))
                if decode:
                    with torch.inference_mode():
                        normalized = torch.from_numpy(latents[arm][None, :, None]).to(device)
                        decoded = vae.decode(ev.bridge.denormalize_wan_latents(vae, normalized)).sample
                    if tuple(decoded.shape) != (1, 3, 1, 384, 384) or not torch.isfinite(decoded).all():
                        raise ValueError('Unexpected/nonfinite frozen VAE reconstruction')
                    rgb = decoded[0, :, 0].float().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().numpy().copy()
                    arrays['decoded_rgb__' + arm] = rgb
                    if arm == 'true_source':
                        decoded_source = rgb
                    result.update(ev._error_metrics(rgb, pixels['target_rgb'].astype(np.float32) / 255,
                                                    pixels['source_rgb'].astype(np.float32) / 255, rgb_masks, 'rgb', 2))
                    for region in ('background', 'distractor'):
                        mask = rgb_masks[region]
                        result['decoded_source_' + region + '_mse'] = float(
                            np.square(rgb.astype(np.float64) - decoded_source)[mask].mean()) if mask.any() else None
                scene_rows.append(result)
            metadata = {key: base[key] for key in ('version', 'split', 'fresh_test_accessed', 'test_freeze_sha256',
                'test_publication_sha256', 'trained_checkpoint_freeze_sha256', 'feature_manifest_sha256',
                'target_manifest_sha256', 'vae', 'arms', 'latent_regions', 'rgb_regions', 'evaluation')}
            metadata.update({'scene': spec['name'], 'seed': int(spec['seed']), 'shift_px': int(spec['dx']),
                'frame_index': 15, 'translator_arm': translator, 'checkpoint_sha256': checkpoints[translator],
                'source_rgb_sha256': record['source_rgb_sha256'], 'target_rgb_sha256': record['target_rgb_sha256'],
                'source_tokens_sha256': record['source_tokens_sha256'], 'target_tokens_sha256': record['target_tokens_sha256'],
                'latent_sha256': {name: ev.data.array_sha256(value) for name, value in latents.items()},
                'operator_invariants': invariants, 'provenance': counts, 'residual_zero_exact': True,
                'transport_zero_exact': True, 'decoded': bool(decode), 'renderer_bundle': True,
                'shuffle_policy': 'Same seed1111 permutation of whole copy/repair vectors for every scene; destination coordinates fixed'})
            arrays['metadata_json'] = np.frombuffer(json.dumps(metadata, sort_keys=True, allow_nan=False).encode(), np.uint8)
            directory = out / translator / 'scenes'
            path = directory / (spec['name'] + '.npz')
            metric_path = directory / (spec['name'] + '_metrics.json')
            counts_path = directory / (spec['name'] + '_counts.json')
            per_condition = [{'scene': spec['name'], 'seed': int(spec['seed']), 'translator_arm': translator,
                              'condition': name, **statistics} for name, statistics in counts.items()]
            ev._save_npz(path, arrays)
            ev._write_json(metric_path, scene_rows)
            ev._write_json(counts_path, per_condition)
            scene_record['bundles'][translator] = {
                'path': str(path.relative_to(out)), 'sha256': ev.data.sha256(path),
                'metrics_path': str(metric_path.relative_to(out)), 'metrics_sha256': ev.data.sha256(metric_path),
                'counts_path': str(counts_path.relative_to(out)), 'counts_sha256': ev.data.sha256(counts_path)}
            rows.extend(scene_rows)
            count_rows.extend(per_condition)
        manifest['scenes'].append(scene_record)
        ev._write_json(manifest_path, manifest)
        print('FRESH_BRIDGE_SCENE_COMPLETE', spec['name'], 'both translators/all27 arms', flush=True)
    if len(rows) != 8 * 2 * len(ARMS) or len(manifest['scenes']) != 8:
        raise AssertionError('Must retain all eight test scenes, both translators and all27 arms')
    ev._write_json(out / 'metrics.json', rows)
    ev._write_csv(out / 'metrics.csv', rows)
    ev._write_json(out / 'provenance_counts.json', count_rows)
    summary = {'version': VERSION, 'split': base['split'], 'fresh_test_accessed': True,
               'scene_count': 8, 'test_freeze_sha256': freeze_hash, 'arms': arms,
               'aggregation': base['aggregation'], 'decoded': bool(decode),
               'methods_by_translator': {name: transport._summary(
                    [row for row in rows if row['translator_arm'] == name], ARMS) for name in TRANSLATORS},
               'no_f_local_identical_across_translators': True,
               'limits': ['Eight procedural still-image pairs; no video rollout or real-footage identity claim.',
                          'Genuine-target JEPA and true-target RGB/Wan conditions are privileged diagnostics.',
                          'Exact provenance exploits an editor-specific copy operation, not learned semantic correspondence.',
                          'Exact source latents do not guarantee identical decoded or generated pixels.',
                          'No optimization, normalization update, model selection or fresh-scene adaptation is performed.']}
    ev._write_json(out / 'summary.json', summary)
    manifest.update({'complete': True, 'metrics_sha256': ev.data.sha256(out / 'metrics.json'),
                     'summary_sha256': ev.data.sha256(out / 'summary.json'),
                     'provenance_counts_sha256': ev.data.sha256(out / 'provenance_counts.json')})
    ev._write_json(manifest_path, manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root', 'freeze', 'publication', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--model-cache', default='/tmp/day11-model')
    args = parser.parse_args()
    run(args.cache_root, args.freeze, args.publication, args.out,
        args.device, args.threads, args.decode, args.model_cache)


if __name__ == '__main__':
    main()
