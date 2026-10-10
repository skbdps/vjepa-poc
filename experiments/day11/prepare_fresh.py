"""Prepare eight untouched native-image JEPA pairs after a qualified dev review.

Only the pinned JEPA encoder is loaded. No VAE, bridge fitting, VACE inference,
or change to NativeImageDataset's training/development whitelist occurs here.
Fresh RGB, masks, and counterfactual target features are evaluation material.
They must never become translator/operator conditioning beyond the declared
source image, selected mask, and requested displacement.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = 'day11_fresh_jepa_pairs_v1'
REQUIRED_SOURCES = ('experiments/day11/prepare_fresh.py', 'experiments/day11/PROTOCOL.md',
                    'experiments/day10/animate.py', 'experiments/day8/data.py',
                    'experiments/day8/extract.py')


def test_specs():
    return [{'name': f'test_{14000+i}_dx{dx:+d}', 'seed': 14000+i, 'dx': dx,
             'split': 'test', 'frame_index': 15, 'views': ['source', 'genuine_shifted_target']}
            for i, dx in enumerate((32, -32, 48, -48, 64, -64, 80, -80))]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='staging_', suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _legacy_modules():
    """Bind extract.py to Day8 data explicitly, despite Day11 also having data.py."""
    renderer = _load_module('day11_fresh_day8_renderer', REPO / 'experiments/day8/data.py')
    sentinel = object()
    old_data = sys.modules.get('data', sentinel)
    try:
        sys.modules['data'] = renderer
        extract = _load_module('day11_fresh_day8_extract', REPO / 'experiments/day8/extract.py')
    finally:
        if old_data is sentinel:
            sys.modules.pop('data', None)
        else:
            sys.modules['data'] = old_data
    if extract.data is not renderer:
        raise RuntimeError('JEPA extraction imported the wrong procedural renderer')
    return renderer, extract


def _bound_file(freeze_path, reference):
    path = Path(reference['path'])
    if not path.is_absolute():
        path = Path(freeze_path).parent / path
    if sha256(path) != reference['sha256']:
        raise ValueError('Bound evidence changed: ' + str(path))
    return path


def _verify_sources(source_hashes):
    for name, digest in source_hashes.items():
        path = (REPO / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(REPO.resolve()):
            raise ValueError('Source binding escapes the repository')
        if sha256(path) != digest:
            raise ValueError('Frozen source changed: ' + name)


def validate_freeze(freeze, publication, device, threads):
    """Run every gate before importing the encoder or rendering any fresh scene."""
    freeze, publication = Path(freeze), Path(publication)
    frozen = json.loads(freeze.read_text())
    if frozen.get('test_accessed') is not False or frozen.get('test_specs') != test_specs():
        raise ValueError('Missing or mismatched eight-scene fresh-test freeze')
    if frozen.get('extraction') != {'device': device, 'precision': 'float32', 'threads': threads}:
        raise ValueError('Extraction runtime differs from the pretest freeze')
    if not set(REQUIRED_SOURCES) <= set(frozen.get('source_hashes', {})):
        raise ValueError('Fresh-test freeze does not bind extraction dependencies')
    _verify_sources(frozen['source_hashes'])
    trained_path = _bound_file(freeze, frozen['trained_checkpoint_freeze'])
    trained = json.loads(trained_path.read_text())
    if (trained.get('version') != 'day11_genuine_jepa_to_wan_training_v1'
            or trained.get('fresh_test_accessed') is not False
            or trained.get('test_accessed') is not False):
        raise ValueError('Expected a completed, pretest genuine-image bridge-training freeze')
    _verify_sources(trained['source_hashes'])
    for name, digest in trained['source_hashes'].items():
        if frozen['source_hashes'].get(name) != digest:
            raise ValueError('Pretest and trained-model source bindings disagree')
    models = trained.get('models', [])
    if {record['arm'] for record in models} != {'cnn', 'linear'} or len(models) != 2:
        raise ValueError('Expected both frozen CNN and linear final checkpoints')
    for record in models:
        if record.get('selected_epoch') != trained['training_config']['epochs']:
            raise ValueError('A frozen bridge checkpoint is not the final training epoch')
        _bound_file(trained_path, {'path': record['checkpoint_path'], 'sha256': record['checkpoint_sha256']})
    # The final training freeze already binds histories/data/statistics; those
    # large or unused payloads need not all be materialized for JEPA extraction.
    gate_path = _bound_file(freeze, frozen['development_gate'])
    gate = json.loads(gate_path.read_text())
    if gate.get('pass') is not True or gate.get('checkpoint_freeze_sha256') != sha256(trained_path):
        raise ValueError('No qualifying development review bound to these trained bridges; fresh scenes remain unopened')
    receipt = json.loads(publication.read_text())
    if (receipt.get('freeze_sha256') != sha256(freeze)
            or receipt.get('bytes_equal_to_GitHub') is not True
            or receipt.get('test_encoded_before_verification') is not False
            or len(receipt.get('commit', '')) != 40):
        raise ValueError('Missing byte-verified fresh-test publication receipt')
    return frozen, trained


def encode_image(encoder, image, device):
    """Same native-image FP32 normalization and shape as Day10.animate.encode_image."""
    import torch
    if image.shape != (384, 384, 3) or image.dtype != np.uint8:
        raise ValueError('Expected one uint8 384x384 RGB image')
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None, :, None].to(device).float() / 255
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1, 1)
    with torch.inference_mode():
        output = encoder((x - mean) / std)
    if tuple(output.shape) != (1, 576, 1024) or not torch.isfinite(output).all():
        raise RuntimeError('Unexpected/nonfinite native-image JEPA features')
    return output.float().cpu().numpy().reshape(1, 24, 24, 1024).copy()


def patch_fraction(mask):
    if mask.shape != (384, 384) or mask.dtype != np.bool_:
        raise ValueError('Expected one boolean 384x384 scoring mask')
    return mask.reshape(24, 16, 24, 16).mean(axis=(1, 3), dtype=np.float32)[None]


def patch_rgb(image):
    return image.reshape(24, 16, 24, 16, 3).mean(axis=(1, 3), dtype=np.float32)[None] / 255


def run(out, upstream, freeze, publication, device='cpu', threads=4):
    frozen, trained = validate_freeze(freeze, publication, device, threads)
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Frozen CUDA extraction requested but unavailable')
    if threads < 1:
        raise ValueError('Positive thread count required')
    torch.set_num_threads(threads)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    renderer, extract = _legacy_modules()
    out = Path(out)
    manifest_path = out / 'features_manifest.json'
    base = {'version': VERSION, 'test_specs': test_specs(), 'source_hashes': frozen['source_hashes'],
            'test_freeze_sha256': sha256(freeze), 'test_publication_sha256': sha256(publication),
            'trained_checkpoint_freeze_sha256': frozen['trained_checkpoint_freeze']['sha256'],
            'development_gate_sha256': frozen['development_gate']['sha256'],
            'model': extract.MODEL, 'upstream': extract.UPSTREAM, 'weights_sha256': extract.WEIGHT_SHA256,
            'modality': 'native single image [1,3,1,384,384]',
            'precision': 'float32', 'cache_precision': 'float32',
            'target_policy': 'Counterfactual target RGB, masks, and JEPA features are scoring/oracle diagnostics only',
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'numpy': str(np.__version__), 'device': device, 'threads': threads},
            'scenes': []}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else base
    for key, value in base.items():
        if key != 'scenes' and manifest.get(key) != value:
            raise ValueError('Cannot resume a fresh cache with changed bindings: ' + key)
    recorded_specs = [record['spec'] for record in manifest['scenes']]
    if recorded_specs != test_specs()[:len(recorded_specs)]:
        raise ValueError('Fresh cache has unexpected, duplicated, or reordered scenes')
    write_json(manifest_path, manifest)
    encoder = None
    for spec in test_specs():
        old = next((record for record in manifest['scenes'] if record['spec'] == spec), None)
        if old is not None:
            for path, digest in old['file_sha256'].items():
                if sha256(out / path) != digest:
                    raise ValueError('Completed fresh evidence changed: ' + path)
            print('FRESH_JEPA_REUSED', spec['name'], flush=True)
            continue
        if encoder is None:
            encoder = extract.load_encoder(upstream, device)
        started = time.perf_counter()
        pair = renderer.generate_pair(spec)
        pixels = {'source_rgb': pair['frames_source'][15].copy(),
                  'target_rgb': pair['frames_target'][15].copy(),
                  'source_mask': pair['masks_source'][15].copy(),
                  'target_mask': pair['masks_target'][15].copy(),
                  'distractor_mask': pair['masks_distractor'][15].copy()}
        source_xy = pair['metadata']['selected_source_xy'][15]
        target_xy = pair['metadata']['selected_target_xy'][15]
        del pair  # No extra frame or hidden-background tensor reaches encoding.
        union = pixels['source_mask'] | pixels['target_mask']
        if not np.array_equal(pixels['source_rgb'][~union], pixels['target_rgb'][~union]):
            raise AssertionError('Renderer changed pixels unrelated to the prescribed object translation')
        if not np.array_equal(np.roll(pixels['source_mask'], spec['dx'], axis=1), pixels['target_mask']):
            raise AssertionError('Counterfactual target selection is not the requested translation')
        tokens = np.stack([encode_image(encoder, pixels['source_rgb'], device),
                           encode_image(encoder, pixels['target_rgb'], device)])
        source_fraction = patch_fraction(pixels['source_mask'])
        target_fraction = patch_fraction(pixels['target_mask'])
        feature_path = Path('features/cache') / (spec['name'] + '.npz')
        pixel_path = Path('pixels') / (spec['name'] + '.npz')
        save_npz(out / feature_path, tokens=tokens,
                 source_frac=source_fraction, target_frac=np.stack([source_fraction, target_fraction]),
                 distractor_frac=patch_fraction(pixels['distractor_mask']),
                 occupancy=np.stack([patch_fraction(pixels['source_mask'] | pixels['distractor_mask']),
                                     patch_fraction(pixels['target_mask'] | pixels['distractor_mask'])]),
                 rgb=np.stack([patch_rgb(pixels['source_rgb']), patch_rgb(pixels['target_rgb'])]),
                 rgb_source=patch_rgb(pixels['source_rgb']),
                 view_shift_px=np.array([0, spec['dx']], dtype=np.int32))
        save_npz(out / pixel_path, **pixels)
        record = {'spec': spec, 'path': str(feature_path), 'sha256': sha256(out / feature_path),
                  'pixel_path': str(pixel_path), 'pixel_sha256': sha256(out / pixel_path),
                  'source_rgb_sha256': array_sha256(pixels['source_rgb']),
                  'target_rgb_sha256': array_sha256(pixels['target_rgb']),
                  'source_tokens_sha256': array_sha256(tokens[0]), 'target_tokens_sha256': array_sha256(tokens[1]),
                  'selected_source_xy': source_xy, 'selected_target_xy': target_xy,
                  'sample_ids': [spec['name'] + '__source', spec['name'] + '__target'],
                  'file_sha256': {str(feature_path): sha256(out / feature_path),
                                  str(pixel_path): sha256(out / pixel_path)},
                  'seconds': time.perf_counter() - started}
        manifest['scenes'].append(record)
        write_json(manifest_path, manifest)
        print('FRESH_JEPA_DONE', spec['name'], round(record['seconds'], 2), flush=True)
        del tokens, pixels
        gc.collect()
    print('FRESH_JEPA_COMPLETE', len(manifest['scenes']), sha256(manifest_path), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    run(args.out, args.upstream, args.freeze, args.publication, args.device, args.threads)
