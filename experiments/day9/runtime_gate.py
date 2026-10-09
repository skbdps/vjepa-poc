"""Assess saved precision evidence without changing the predeclared tolerance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

DAY8 = Path(__file__).resolve().parents[1] / 'day8'
sys.path.insert(0, str(DAY8))
import extract as extraction
import operators
import validate_precision


def assess(precision_dir: Path, old_features: Path, out: Path):
    report_path = precision_dir / 'precision_validation.json'
    report = json.loads(report_path.read_text())
    assert report['checkpoint_sha256'] == extraction.WEIGHT_SHA256
    assert report['test_data_opened'] is False
    reference = Path(report['reference']['path'])
    assert extraction.sha256(reference) == report['reference']['sha256']
    assert report['reference']['sha256'] == '92806392a95ecdcc98a36750ec0c533aa78dafb05351054cac95a20217ea6d14'
    record = report['precisions']['bfloat16']
    candidate = precision_dir / record['cache']['path']
    assert extraction.sha256(candidate) == record['cache']['sha256']
    with np.load(reference) as ref, np.load(candidate) as current:
        mask = ref['source_frac'] > 0
        destination = operators.shift_horizontal(mask, 2)
        regions = {'global': np.ones_like(mask), 'source_hole': mask & ~destination, 'destination': destination}
        checks = validate_precision.precision_metrics(ref['source'], ref['target'], current['source'], current['target'], regions)
        manifest = json.loads((old_features / 'manifest.json').read_text())
        old_record = next(r for r in manifest['clips'] if r['spec']['name'] == 'train_11000_dx+32')
        previous = old_features / old_record['path']
        assert extraction.sha256(previous) == old_record['sha256']
        with np.load(previous) as old:
            drift = validate_precision.precision_metrics(old['source'], old['target'], current['source'], current['target'], regions)
            exact = {side: bool(np.array_equal(old[side], current[side])) for side in ('source', 'target')}
    passed = checks['passes_1pct_signal_tolerance'] and not any(r['denominator_floored'] for r in checks['metrics'].values())
    result = {'version': 'day9_protocol_precision_gate_v1',
              'pass': passed,
              'criterion': 'The unchanged Day8/Day9 protocol: all six errors relative to original Colab FP32 edit signal strictly below 1%. Bitwise equality to prior BF16 is diagnostic only.',
              'comparison_to_colab_fp32': checks,
              'comparison_to_previous_bfloat16': drift,
              'bitwise_equal_to_previous_bfloat16': exact,
              'precision_report_sha256': extraction.sha256(report_path),
              'reference_sha256': extraction.sha256(reference),
              'candidate_sha256': extraction.sha256(candidate),
              'previous_production_sha256': old_record['sha256'],
              'checkpoint_sha256': extraction.WEIGHT_SHA256,
              'chosen_precision': 'bfloat16' if passed else None,
              'test_data_opened': False}
    extraction.write_json(out, result)
    print(json.dumps(result, indent=2))
    if not result['pass']:
        raise RuntimeError('Protocol precision gate failed; held-out extraction remains blocked')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--precision-dir', type=Path, required=True)
    p.add_argument('--old-features', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assess(a.precision_dir, a.old_features, a.out)
