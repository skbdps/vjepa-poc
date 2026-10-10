"""Reproduce a previously validated training fixture before opening new test data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

DAY8 = Path(__file__).resolve().parents[1] / 'day8'
sys.path.insert(0, str(DAY8))
import data
import extract as day8_extract


def run(features: Path, upstream: Path, out: Path):
    import torch
    torch.set_num_threads(8)
    manifest = json.loads((features / 'manifest.json').read_text())
    fixture = next(r for r in manifest['clips'] if r['spec']['name'] == 'train_11000_dx+32')
    cache = features / fixture['path']
    assert day8_extract.sha256(cache) == fixture['sha256']
    assert manifest['inference_precision'] == 'bfloat16'
    tick = time.perf_counter()
    encoder = day8_extract.load_encoder(upstream, 'cpu')
    pair = data.generate_pair(fixture['spec'])
    checks = {}
    with np.load(cache) as old:
        for role in ('source', 'target'):
            frames = pair['frames_' + role]
            assert hashlib.sha256(frames.tobytes()).hexdigest() == fixture[role + '_rgb_sha256']
            now = day8_extract.encode(encoder, frames, 'cpu', 'bfloat16')
            reference = old[role]
            checks[role] = {
                'bitwise_equal': bool(np.array_equal(now, reference)),
                'max_abs_difference': float(np.max(np.abs(now.astype(np.float32) - reference.astype(np.float32)))),
                'sha256': hashlib.sha256(now.tobytes()).hexdigest(),
                'shape': list(now.shape),
            }
            print('RUNTIME_CHECK', role, checks[role], flush=True)
    result = {
        'version': 'day9_restored_runtime_v1',
        'fixture': fixture['spec'],
        'reference_cache_sha256': fixture['sha256'],
        'checkpoint_sha256': day8_extract.WEIGHT_SHA256,
        'checks': checks,
        'pass': all(c['bitwise_equal'] for c in checks.values()),
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                    'device': 'cpu', 'precision': 'bfloat16', 'threads': torch.get_num_threads()},
        'seconds': time.perf_counter() - tick,
        'scope': 'One old training fixture, never new held-out data. Exact agreement with production BF16 features previously validated against Colab FP32.',
    }
    day8_extract.write_json(out, result)
    if not result['pass']:
        raise RuntimeError('Restored runtime does not exactly reproduce the validated fixture')
    print('RUNTIME_VALIDATED', round(result['seconds'], 2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.features, a.upstream, a.out)
