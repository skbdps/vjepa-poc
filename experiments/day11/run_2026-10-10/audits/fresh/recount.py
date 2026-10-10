"""Independent saved-array audit; no production evaluator imports or model runs."""
from pathlib import Path
import hashlib
import json
import statistics
import numpy as np

ROOT = Path('/workspace/scratch/8e9e8b29a938')
REPO = ROOT / 'vjepa-run'
BASE = ROOT / 'day11_compute'
OUT = BASE / 'audit_fresh'
FRESH = BASE / 'fresh'
EVAL = BASE / 'fresh_evaluation'
RUN = REPO / 'experiments/day11/run_2026-10-10'
REGIONS = ('source_hole', 'destination', 'distractor', 'background', 'global')
PRIMARY = 'provenance_residual_copy_repair'
LOCAL = 'provenance_local_copy_repair'
COMPARATORS = (PRIMARY, LOCAL, 'provenance_copy_copy_repair', 'residual_copy_repair', 'absolute_copy_repair')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def raw_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(name, value):
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(name, sha(path))

def shift(array, dx):
    result = np.zeros_like(array)
    if dx > 0:
        result[..., dx:] = array[..., :-dx]
    elif dx < 0:
        result[..., :dx] = array[..., -dx:]
    else:
        result[...] = array
    return result

def fraction(mask):
    return mask.reshape(24, 16, 24, 16).mean((1, 3), dtype=np.float32)[None]

def masks(pixels):
    src, dst, distractor = (fraction(pixels[x + '_mask'])[0] > 0 for x in ('source', 'target', 'distractor'))
    union = src | dst
    padded = np.pad(union, 1)
    support = np.zeros_like(union)
    for y in range(3):
        for x in range(3):
            support |= padded[y:y+24, x:x+24]
    coarse = {'source_hole': src & ~dst, 'destination': dst, 'distractor': distractor,
              'background': ~(support | distractor), 'global': np.ones((24, 24), bool)}
    latent = {k: np.repeat(np.repeat(v, 2, 0), 2, 1) for k, v in coarse.items()}
    rgb = {'source_hole': pixels['source_mask'] & ~pixels['target_mask'], 'destination': pixels['target_mask'],
           'distractor': pixels['distractor_mask'], 'global': np.ones((384, 384), bool),
           'background': ~np.repeat(np.repeat(support | distractor, 16, 0), 16, 1)}
    return latent, rgb

def features_audit():
    manifest = read(FRESH / 'features_manifest.json')
    freeze = read(RUN / 'PRETEST.json')
    publication = read(RUN / 'PRETEST_PUBLICATION.json')
    expected = [{'name': f'test_{14000+i}_dx{dx:+d}', 'seed': 14000+i, 'dx': dx, 'split': 'test',
                 'frame_index': 15, 'views': ['source', 'genuine_shifted_target']}
                for i, dx in enumerate((32, -32, 48, -48, 64, -64, 80, -80))]
    assert manifest['test_specs'] == freeze['test_specs'] == expected
    assert [r['spec'] for r in manifest['scenes']] == expected
    assert manifest['test_freeze_sha256'] == publication['freeze_sha256'] == sha(RUN / 'PRETEST.json')
    assert manifest['test_publication_sha256'] == sha(RUN / 'PRETEST_PUBLICATION.json')
    assert publication['commit'] == '1bd6221bdea2dcce50697603806ab18e648eff36'
    assert publication['bytes_equal_to_GitHub'] is True and publication['test_encoded_before_verification'] is False
    assert manifest['source_hashes'] == freeze['source_hashes']
    for name, digest in freeze['source_hashes'].items():
        assert sha(REPO / name) == digest, name
    for name in ('trained_checkpoint_freeze', 'development_gate'):
        assert manifest[name+'_sha256'] == freeze[name]['sha256']
        path = Path(freeze[name]['path'])
        path = path if path.is_absolute() else RUN / path
        assert sha(path) == freeze[name]['sha256']
    assert manifest['precision'] == manifest['cache_precision'] == freeze['extraction']['precision'] == 'float32'
    assert manifest['runtime']['threads'] == 4 and manifest['runtime']['device'] == 'cpu'
    cases = []
    for row in manifest['scenes']:
        p = row['spec']
        assert row['file_sha256'] == {row['path']: row['sha256'], row['pixel_path']: row['pixel_sha256']}
        for path, digest in row['file_sha256'].items():
            assert sha(FRESH / path) == digest
        with np.load(FRESH / row['path'], allow_pickle=False) as z:
            features = {k: z[k].copy() for k in z.files}
        with np.load(FRESH / row['pixel_path'], allow_pickle=False) as z:
            pixels = {k: z[k].copy() for k in z.files}
        tok = features['tokens']
        assert tok.shape == (2, 1, 24, 24, 1024) and tok.dtype == np.float32 and np.isfinite(tok).all()
        for j, view in enumerate(('source', 'target')):
            rgb = pixels[view+'_rgb']
            assert rgb.shape == (384, 384, 3) and rgb.dtype == np.uint8
            assert raw_sha(rgb) == row[view+'_rgb_sha256']
            assert raw_sha(tok[j]) == row[view+'_tokens_sha256']
        assert np.array_equal(shift(pixels['source_mask'], p['dx']), pixels['target_mask'])
        assert row['selected_target_xy'] == [row['selected_source_xy'][0]+p['dx'], row['selected_source_xy'][1]]
        assert np.array_equal(features['source_frac'], fraction(pixels['source_mask']))
        assert np.array_equal(features['target_frac'][0], features['source_frac'])
        assert np.array_equal(features['target_frac'][1], fraction(pixels['target_mask']))
        assert np.array_equal(features['distractor_frac'], fraction(pixels['distractor_mask']))
        assert row['sample_ids'] == [p['name']+'__'+v for v in ('source', 'target')]
        lm, rm = masks(pixels)
        assert all(v.any() for v in lm.values()) and all(v.any() for v in rm.values())
        cases.append({'scene': p['name'], 'seed': p['seed'], 'dx': p['dx'],
                      'latent_region_cells': {k: int(v.sum()) for k, v in lm.items()},
                      'RGB_region_pixels': {k: int(v.sum()) for k, v in rm.items()},
                      'mask_translation_exact': True, 'array_and_file_hashes_match': True})
    report = {'version': 'day11_independent_fresh_feature_audit_v1', 'pass': True,
              'feature_manifest_sha256': sha(FRESH / 'features_manifest.json'),
              'pretest_sha256': sha(RUN / 'PRETEST.json'), 'publication_sha256': sha(RUN / 'PRETEST_PUBLICATION.json'),
              'publication_receipt': publication, 'source_file_count': len(freeze['source_hashes']),
              'scenes': cases, 'artifact_hashes_checked': 16, 'raw_RGB_and_token_hashes_checked': 32,
              'scope': 'Checks saved arrays, manifest/source bindings, exact masks and frozen cases; does not independently prove wall-clock publication ordering or rerun encoders.'}
    write('feature_manifest_audit.json', report)
    return manifest, freeze

def latent_audit(features, freeze):
    manifest_path = EVAL / 'evaluation_manifest.json'
    if not manifest_path.exists() or not read(manifest_path).get('complete'):
        print('Evaluation incomplete; latent recount postponed.')
        return
    manifest = read(manifest_path)
    rows = read(EVAL / 'metrics.json')
    summary = read(EVAL / 'summary.json')
    assert sha(EVAL / 'metrics.json') == manifest['metrics_sha256']
    assert sha(EVAL / 'summary.json') == manifest['summary_sha256']
    assert manifest['feature_manifest_sha256'] == sha(FRESH / 'features_manifest.json')
    assert manifest['target_manifest_sha256'] == sha(EVAL / 'targets/latent_manifest.json')
    assert manifest['test_specs'] == freeze['test_specs'] and manifest['evaluation'] == freeze['evaluation']
    assert manifest['source_hashes'] == freeze['source_hashes']
    assert manifest['decoded'] is False
    arms = freeze['evaluation']['arms']
    names = [r['spec']['name'] for r in features['scenes']]
    assert [r['scene'] for r in manifest['scenes']] == names
    expected = {(s, t, a) for s in names for t in ('cnn', 'linear') for a in arms}
    index = {(r['scene'], r['translator_arm'], r['arm']): r for r in rows}
    assert len(rows) == len(index) == 432 and set(index) == expected
    target_manifest = read(EVAL / 'targets/latent_manifest.json')
    assert target_manifest['complete'] is True and len(target_manifest['items']) == 16
    targets = {}
    for r in target_manifest['items']:
        path = EVAL / 'targets' / r['path']
        assert sha(path) == r['sha256']
        targets[r['sample_id']] = np.load(path, allow_pickle=False)
    comparisons, cases, all_recounted = [], [], []
    maximum_difference, checked_values, bundles_checked = 0., 0, 0
    zero_exact_checks, local_identity_checks = 0, 0
    for scene in manifest['scenes']:
        name = scene['scene']
        frow = next(r for r in features['scenes'] if r['spec']['name'] == name)
        with np.load(FRESH / frow['pixel_path'], allow_pickle=False) as z:
            pixels = {k: z[k].copy() for k in z.files}
        lm, rm = masks(pixels)
        local_cnn = {}
        for translator in ('cnn', 'linear'):
            ref = scene['bundles'][translator]
            for field, digest in (('path','sha256'),('metrics_path','metrics_sha256'),('counts_path','counts_sha256')):
                assert sha(EVAL / ref[field]) == ref[digest]
            with np.load(EVAL / ref['path'], allow_pickle=False) as z:
                meta = json.loads(z['metadata_json'].tobytes())
                assert meta['scene'] == name and meta['translator_arm'] == translator
                assert meta['checkpoint_sha256'] == manifest['checkpoint_sha256_by_translator'][translator]
                for r in REGIONS:
                    assert np.array_equal(z['latent_mask__'+r], lm[r])
                    assert np.array_equal(z['rgb_mask__'+r], rm[r])
                for v in ('source', 'target'):
                    assert np.array_equal(z['latent__true_'+v], targets[name+'__'+v])
                    assert np.array_equal(z[v+'_rgb'], pixels[v+'_rgb'])
                src, tgt = targets[name+'__source'], targets[name+'__target']
                baseline_error = np.square(src.astype(np.float64)-tgt).mean(0)
                for arm in arms:
                    value = z['latent__'+arm]
                    assert value.dtype == np.float32 and value.shape == (16,48,48) and np.isfinite(value).all()
                    assert raw_sha(value) == meta['latent_sha256'][arm]
                    error = np.square(value.astype(np.float64)-tgt).mean(0)
                    recount = {'scene': name, 'translator_arm': translator, 'arm': arm}
                    for region, mask in lm.items():
                        count = int(mask.sum())
                        numerator, denominator = float(error[mask].sum()), float(baseline_error[mask].sum())
                        mse, baseline = numerator/count, denominator/count
                        valid = baseline > 1e-12
                        prefix = 'latent_'+region
                        recount.update({prefix+'_count': count, prefix+'_squared_error_sum': numerator,
                                        prefix+'_mse': mse, prefix+'_source_baseline_mse': baseline,
                                        prefix+'_ratio': mse/baseline if valid else None,
                                        prefix+'_ratio_degenerate': not valid})
                    recount['latent_primary_ratio'] = statistics.mean(recount['latent_'+r+'_ratio'] for r in ('source_hole','destination'))
                    original = index[(name,translator,arm)]
                    for key,value_ in recount.items():
                        if not key.startswith('latent_'):
                            continue
                        other = original[key]
                        if isinstance(value_,bool) or value_ is None:
                            assert value_ is other
                        else:
                            difference = abs(value_-other)
                            maximum_difference = max(maximum_difference,difference)
                            assert difference <= 1e-12*max(1.,abs(value_)), (name,translator,arm,key,difference)
                        checked_values += 1
                    all_recounted.append(recount)
                    if arm in ('residual_source','provenance_copy_source','provenance_local_source','provenance_residual_source'):
                        assert np.array_equal(z['latent__'+arm],src)
                        zero_exact_checks += 1
                    if arm.startswith('provenance_local_'):
                        if translator == 'cnn':
                            local_cnn[arm] = z['latent__'+arm].copy()
                        else:
                            assert np.array_equal(local_cnn[arm],z['latent__'+arm])
                            local_identity_checks += 1
            bundles_checked += 1
        for translator in ('cnn','linear'):
            primary, local = (index[(name,translator,a)] for a in (PRIMARY,LOCAL))
            comparisons.append({'scene':name,'translator':translator,
                'primary':{r:{'mse':primary['latent_'+r+'_mse'],'ratio':primary['latent_'+r+'_ratio']} for r in REGIONS},
                'no_F':{r:{'mse':local['latent_'+r+'_mse'],'ratio':local['latent_'+r+'_ratio']} for r in REGIONS},
                'primary_minus_no_F_mse':{r:primary['latent_'+r+'_mse']-local['latent_'+r+'_mse'] for r in REGIONS},
                'primary_minus_no_F_balanced_ratio':primary['latent_primary_ratio']-local['latent_primary_ratio'],
                'primary_both_edit_region_ratios_le_half':all(primary['latent_'+r+'_ratio'] <= .5 for r in ('source_hole','destination'))})
        p=index[(name,'cnn',PRIMARY)]
        cases.append({'scene':name,'hole_ratio':p['latent_source_hole_ratio'],'destination_ratio':p['latent_destination_ratio'],
                      'pass_both_latent_criteria':all(p['latent_'+r+'_ratio'] <= .5 and p['latent_'+r+'_ratio_degenerate'] is False for r in ('source_hole','destination'))})
    method_means = {}
    for translator in ('cnn','linear'):
        method_means[translator] = {}
        for arm in arms:
            subset=[r for r in all_recounted if r['translator_arm']==translator and r['arm']==arm]
            method_means[translator][arm]={'scene_count':8,'balanced_edit_ratio':statistics.mean(r['latent_primary_ratio'] for r in subset),
                 **{region:{'mse':statistics.mean(r['latent_'+region+'_mse'] for r in subset),
                             'ratio':statistics.mean(r['latent_'+region+'_ratio'] for r in subset) if all(r['latent_'+region+'_ratio'] is not None for r in subset) else None} for region in REGIONS}}
    write('latent_metric_audit.json',{'version':'day11_independent_fresh_latent_recount_v1','pass':True,
        'evaluation_manifest_sha256':sha(manifest_path),'metrics_sha256':sha(EVAL/'metrics.json'),
        'feature_manifest_sha256':sha(FRESH/'features_manifest.json'),'rows_recounted':len(all_recounted),
        'numeric_or_flag_values_compared':checked_values,'maximum_absolute_difference':maximum_difference,
        'bundles_checked':bundles_checked,'zero_edit_identity_checks':zero_exact_checks,'no_F_cross_architecture_exact_checks':local_identity_checks,
        'all_scene_primary_latent_criteria_pass':all(r['pass_both_latent_criteria'] for r in cases),'primary_CNN_cases':cases,
        'paired_comparisons':comparisons,'equal_scene_method_means':method_means,
        'scope':'Independent saved-array MSE recount and mask reconstruction; no encoder, translator, VAE, or diffusion execution. Latent criteria alone are not the full frozen engineering gate. All eight cases retained.'})
    print(json.dumps({'primary_CNN_cases':cases,'means':{t:{a:method_means[t][a] for a in COMPARATORS} for t in method_means}},indent=2))

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    features,freeze=features_audit()
    latent_audit(features,freeze)
