"""Fresh paired videos, inaccessible until the Day9 protocol is published."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DAY8 = REPO / 'experiments/day8'
sys.path.insert(0, str(DAY8))
import data
import extract as day8_extract


def test_specs():
    specs = []
    for i in range(16):
        heldout = i >= 8
        offsets = data.HELDOUT_OFFSETS if heldout else data.TRAIN_OFFSETS
        dx, seed = offsets[i % 4], 12200 + i
        specs.append({'name': f'test_{seed}_dx{dx:+d}', 'seed': seed, 'dx': dx, 'split': 'test',
                      'shift_regime': 'heldout_magnitude' if heldout else 'seen_magnitude'})
    return specs


def validate_freeze(freeze: Path, publication: Path):
    frozen = json.loads(freeze.read_text())
    assert frozen['test_accessed'] is False
    assert frozen['test_specs'] == test_specs()
    required_sources = {'experiments/day9/' + n for n in
        ('extract.py', 'temporal.py', 'evaluate.py', 'PROTOCOL.md', 'runtime_gate.py')}
    required_sources |= {'experiments/day8/' + n for n in
        ('data.py', 'extract.py', 'operators.py', 'probe.py', 'evaluate.py', 'validate_precision.py')}
    assert required_sources <= set(frozen['source_hashes']), 'Incomplete source freeze'
    for name, digest in frozen['source_hashes'].items():
        assert day8_extract.sha256(REPO / name) == digest, f'Changed source: {name}'
    for name in ('dev_selection', 'runtime_validation'):
        ref = frozen[name]
        path = Path(ref['path'])
        if not path.is_absolute():
            path = freeze.parent / path
        assert day8_extract.sha256(path) == ref['sha256'], f'Changed {name}'
        if name == 'runtime_validation':
            validation = json.loads(path.read_text())
            assert validation['pass'] is True and validation['chosen_precision'] == 'bfloat16'
            assert validation['comparison_to_colab_fp32']['passed_checks'] == 6
            assert validation['checkpoint_sha256'] == day8_extract.WEIGHT_SHA256
    receipt = json.loads(publication.read_text())
    assert receipt['freeze_sha256'] == day8_extract.sha256(freeze)
    assert receipt['bytes_equal_to_GitHub'] is True
    assert receipt['test_encoded_before_verification'] is False
    assert len(receipt['commit']) == 40
    return frozen


def run(out: Path, upstream: Path, reference: Path, freeze: Path, publication: Path):
    import torch
    torch.set_num_threads(8)
    frozen = validate_freeze(freeze, publication)
    assert day8_extract.sha256(reference) == frozen['reference_manifest_sha256']
    ref = json.loads(reference.read_text())
    assert ref['model'] == day8_extract.MODEL and ref['upstream'] == day8_extract.UPSTREAM
    assert ref['extraction_device'] == 'cpu' and ref['inference_precision'] == 'bfloat16'
    assert ref['cache_precision'] == 'float16'
    assert ref['weights'][0]['sha256'] == day8_extract.WEIGHT_SHA256
    keys = ('model', 'upstream', 'encoder_context', 'extraction_device', 'inference_precision', 'cache_precision', 'weights')
    base = {k: ref[k] for k in keys}
    base.update({'version': 'day9_fresh_source_hole_test_v1', 'source_hashes': frozen['source_hashes'],
                 'test_freeze_sha256': day8_extract.sha256(freeze),
                 'oracle_budget': ref['oracle_budget'], 'target_policy': ref['target_policy'],
                 'data': {'renderer': data.manifest(), 'specs': test_specs()}, 'clips': []})
    (out / 'cache').mkdir(parents=True, exist_ok=True)
    path = out / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else base
    for k, value in base.items():
        if k != 'clips':
            assert manifest[k] == value, f'Cannot mix extraction runs: {k}'
    day8_extract.write_json(path, manifest)
    encoder = None
    for spec in test_specs():
        cache = out / 'cache' / (spec['name'] + '.npz')
        old = next((r for r in manifest['clips'] if r['spec']['name'] == spec['name']), None)
        if old:
            assert old['spec'] == spec and day8_extract.sha256(cache) == old['sha256']
            print('FEATURE_REUSED', spec['name'], flush=True)
            continue
        if encoder is None:
            encoder = day8_extract.load_encoder(upstream, 'cpu')
        tick = time.perf_counter()
        pair = data.generate_pair(spec)
        source = day8_extract.encode(encoder, pair['frames_source'], 'cpu', 'bfloat16')
        target = day8_extract.encode(encoder, pair['frames_target'], 'cpu', 'bfloat16')
        np.savez_compressed(cache, source=source, target=target,
            source_frac=data.patch_fractions(pair['masks_source']),
            target_frac=data.patch_fractions(pair['masks_target']),
            distractor_frac=data.patch_fractions(pair['masks_distractor']),
            rgb_source=day8_extract.patch_rgb(pair['frames_source']),
            rgb_target=day8_extract.patch_rgb(pair['frames_target']))
        record = {'spec': spec, 'path': str(cache.relative_to(out)), 'sha256': day8_extract.sha256(cache),
                  'bytes': cache.stat().st_size,
                  'source_rgb_sha256': hashlib.sha256(pair['frames_source'].tobytes()).hexdigest(),
                  'target_rgb_sha256': hashlib.sha256(pair['frames_target'].tobytes()).hexdigest(),
                  'seconds': time.perf_counter() - tick}
        manifest['clips'].append(record)
        day8_extract.write_json(out / 'metadata' / (spec['name'] + '.json'), pair['metadata'])
        day8_extract.write_json(path, manifest)
        print('FEATURE_DONE', spec['name'], round(record['seconds'], 2), flush=True)
        del pair, source, target
        gc.collect()
    manifest['runtime'] = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                          'device': 'cpu', 'precision': 'bfloat16', 'threads': torch.get_num_threads(), 'gpu': None}
    manifest['runtime_validation'] = frozen['runtime_validation']
    day8_extract.write_json(path, manifest)
    print('EXTRACTION_COMPLETE', len(manifest['clips']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--publication', type=Path, required=True)
    a = p.parse_args()
    run(a.out, a.upstream, a.reference, a.freeze, a.publication)
