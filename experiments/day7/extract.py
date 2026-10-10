"""Frozen V-JEPA feature extraction for the learned persistent-part experiment.

Only train/dev are opened before the trained-checkpoint freeze. No SAM model or
mask is imported. Full-feature baselines share identical RGB/initial prompts.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data
sys.path.insert(0, str(HERE.parent / 'day4'))
import run_experiment as frozen_backbone
import tracking

PROJECTION_SEED = 77001
PROJECTION_DIM = 256


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def extractor_sources():
    paths = [HERE/'extract.py', HERE/'data.py',
             HERE.parent/'day4'/'benchmark.py',
             HERE.parent/'day4'/'tracking.py',
             HERE.parent/'day4'/'run_experiment.py']
    return {str(p.relative_to(HERE.parent.parent)): sha256(p) for p in paths}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def projection(out):
    path = out/'projection.npy'
    if not path.exists():
        rng = np.random.default_rng(PROJECTION_SEED)
        q, r = np.linalg.qr(rng.standard_normal((1024, PROJECTION_DIM)))
        # Fix QR sign ambiguity; persisted bytes are canonical for this run.
        q *= np.where(np.diag(r) < 0, -1., 1.)
        np.save(path, q.astype(np.float32))
    matrix = np.load(path)
    if matrix.shape != (1024, PROJECTION_DIM) or not np.isfinite(matrix).all():
        raise ValueError('Invalid projection')
    if not np.allclose(matrix.T@matrix, np.eye(PROJECTION_DIM), atol=2e-6):
        raise ValueError('Projection is not orthonormal')
    return matrix


def validate_test_freeze(out, checkpoint_freeze, manifest):
    """Validate completed training and immutable inputs BEFORE opening test RGB.

    This gate verifies files, not a remote Git commit. The operator must commit
    the returned freeze before invoking test extraction, as the protocol states.
    The feature binding intentionally excludes test clips and mutable runtime
    fields, so an interrupted test extraction can resume under the same freeze.
    """
    import torch
    out = Path(out)
    if checkpoint_freeze is None or not Path(checkpoint_freeze).is_file():
        raise ValueError('Test extraction requires the committed trained-checkpoint freeze')
    freeze_path = Path(checkpoint_freeze)
    freeze = json.loads(freeze_path.read_text())
    required = {'training_config', 'models', 'baselines', 'selection_rule',
                'test_accessed', 'test_seed_manifest'}
    if not isinstance(freeze, dict) or not required.issubset(freeze):
        raise ValueError('Incomplete trained-checkpoint freeze')
    if freeze['test_accessed'] is not False:
        raise ValueError('Training freeze must precede test access')
    # Import only for test: training code may still be finalized while train/dev
    # features are extracted, but must match its recorded hashes at this gate.
    module_spec = importlib.util.spec_from_file_location('day7_training_gate', HERE/'train.py')
    trainer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(trainer)
    config = freeze['training_config']
    if not isinstance(config, dict) or config.get('policy') != trainer.POLICY:
        raise ValueError('Training policy differs from frozen source')
    if (config.get('source_hashes') != trainer.source_hashes() or
            config.get('source_digest') != trainer.source_digest()):
        raise ValueError('Training source changed after checkpoint freeze')
    canonical_data = json.loads(json.dumps(data.manifest()))
    if (config.get('data') != canonical_data or
            json.loads(json.dumps(manifest.get('data'))) != canonical_data):
        raise ValueError('Dataset specification changed after freeze')
    if (manifest.get('extractor_sources') != extractor_sources() or
            manifest.get('model') != frozen_backbone.MODEL_NAME or
            manifest.get('upstream_revision') != frozen_backbone.UPSTREAM_SHA):
        raise ValueError('Feature-extractor provenance changed')
    if (manifest.get('projection_seed') != PROJECTION_SEED or
            manifest.get('projection_dim') != PROJECTION_DIM or
            manifest.get('projection_path') != 'projection.npy' or
            not (out/'projection.npy').is_file() or
            sha256(out/'projection.npy') != manifest.get('projection_sha256')):
        raise ValueError('Frozen feature projection changed')
    if manifest.get('mean', {}).get('fit_clips') != len(data.scene_specs('train')):
        raise ValueError('Centering mean must use the complete training cohort')
    binding = trainer.feature_binding(out)
    if config.get('feature_binding') != binding:
        raise ValueError('Train/dev feature files or centering mean changed')
    for record in binding['clips']:
        for key in ('sha256', 'rgb_sha256'):
            value = record.get(key)
            if (not isinstance(value, str) or len(value) != 64 or
                    any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Invalid cache/source checksum')
    if (freeze['test_seed_manifest'] != data.scene_specs('test') or
            freeze['selection_rule'] != trainer.POLICY['selection']):
        raise ValueError('Held-out cohort or selection rule changed')
    training_root = freeze_path.parent
    if json.loads((training_root/'training_config.json').read_text()) != config:
        raise ValueError('Freeze differs from the completed training configuration')
    if (freeze['baselines'] != json.loads((training_root/'baseline_development.json').read_text()) or
            set(freeze['baselines']) != set(trainer.BASELINES)):
        raise ValueError('Baseline development selection is incomplete or changed')
    for record in freeze['baselines'].values():
        if not np.isfinite(float(record['threshold'])):
            raise ValueError('Nonfinite baseline threshold')
    records = freeze['models']
    expected = {(arm, seed) for arm in trainer.ARMS for seed in trainer.SEEDS}
    if (not isinstance(records, list) or len(records) != len(expected) or
            {(r.get('arm'), r.get('seed')) for r in records} != expected):
        raise ValueError('Freeze must retain every predetermined arm and training seed')
    initial_by_seed = {}
    for record in records:
        arm, seed, epoch = record['arm'], record['seed'], record['epoch']
        if (type(epoch) is not int or not 1 <= epoch <= trainer.POLICY['epochs'] or
                record.get('path') != f'models/{arm}/{seed}/best.pt'):
            raise ValueError('Invalid selected checkpoint path or epoch')
        checkpoint = training_root/record['path']
        selected_path = checkpoint.with_name('selected.json')
        curves_path = checkpoint.with_name('curves.csv')
        if (sha256(checkpoint) != record.get('checkpoint_sha256') or
                sha256(selected_path) != record.get('selection_sha256')):
            raise ValueError('Trained checkpoint or selection file changed')
        selected = json.loads(selected_path.read_text())
        if (selected.get('training_config') != config or
                (selected.get('arm'), selected.get('seed'), selected.get('epoch')) != (arm, seed, epoch) or
                selected.get('checkpoint_sha256') != record['checkpoint_sha256'] or
                sha256(curves_path) != selected.get('curves_sha256')):
            raise ValueError('Checkpoint lacks matching completed-training evidence')
        with curves_path.open(newline='') as stream:
            curves = list(csv.DictReader(stream))
        if (len(curves) != trainer.POLICY['epochs'] or
                [int(r['epoch']) for r in curves] != list(range(1, trainer.POLICY['epochs']+1)) or
                any(r['arm'] != arm or int(r['training_seed']) != seed for r in curves)):
            raise ValueError('Incomplete or mismatched training curves')
        utilities = np.array([float(r['dev_utility']) for r in curves])
        if (not np.isfinite(utilities).all() or int(utilities.argmax())+1 != epoch or
                float(record['dev_utility']) != utilities[epoch-1] or
                selected['development']['utility'] != utilities[epoch-1]):
            raise ValueError('Checkpoint violates frozen development selection rule')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (saved.get('arm'), saved.get('seed'), saved.get('epoch')) != (arm, seed, epoch):
            raise ValueError('Checkpoint identity does not match its freeze record')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = trainer.PersistentPartJEPA(256, 128, 8)
        expected_initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                                   for v in model.state_dict().values())).hexdigest()
        if selected.get('initial_state_sha256') != expected_initial:
            raise ValueError('Recorded initialization differs from predetermined seed')
        model.load_state_dict(saved['state_dict'], strict=True)
        if any(not torch.isfinite(v).all() for v in saved['state_dict'].values()):
            raise ValueError('Nonfinite trained checkpoint')
        state_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                            for v in saved['state_dict'].values())).hexdigest()
        initial_hash = selected.get('initial_state_sha256')
        if not initial_hash or state_hash == initial_hash:
            raise ValueError('Checkpoint weights did not change from initialization')
        initial_by_seed.setdefault(seed, set()).add(initial_hash)
    if any(len(values) != 1 for values in initial_by_seed.values()):
        raise ValueError('Paired model initialization differs across arms')
    freeze_hash = sha256(freeze_path)
    if manifest.get('test_checkpoint_freeze_sha256', freeze_hash) != freeze_hash:
        raise ValueError('A different checkpoint freeze already opened this test cache')
    return freeze_hash


def extract(out, upstream, splits, checkpoint_freeze=None):
    import torch
    out, upstream = Path(out), Path(upstream)
    out.mkdir(parents=True, exist_ok=True)
    (out/'cache').mkdir(exist_ok=True)
    path = out/'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {
        'version': 1, 'clips': [], 'data': data.manifest(),
        'extractor_sources': extractor_sources(),
        'model': frozen_backbone.MODEL_NAME,
        'upstream_revision': frozen_backbone.UPSTREAM_SHA,
        'annotation': 'frame-zero part masks only; no SAM or parent-mask annotations',
        'encoder_context': 'Four independent 16-frame blocks; offline within each block',
        'projection_seed': PROJECTION_SEED, 'projection_dim': PROJECTION_DIM}
    if manifest['extractor_sources'] != extractor_sources():
        raise RuntimeError('Extraction source changed; existing caches cannot be silently reused')
    matrix = projection(out)
    projection_hash = sha256(out/'projection.npy')
    if manifest.get('projection_sha256', projection_hash) != projection_hash:
        raise RuntimeError('Projection bytes changed')
    manifest['projection_sha256'] = projection_hash
    manifest['projection_path'] = 'projection.npy'
    if 'test' in splits:
        manifest['test_checkpoint_freeze_sha256'] = validate_test_freeze(out, checkpoint_freeze, manifest)
        # Persist the gate before the first test scene, including if extraction
        # later fails before its first cache can be appended.
        write_json(path, manifest)
    encoder = None
    start = time.perf_counter()
    for split in splits:
        for spec in data.scene_specs(split):
            cache = out/'cache'/f"{spec['name']}.npz"
            previous = next((e for e in manifest['clips'] if e['name']==spec['name']), None)
            if previous is not None:
                if sha256(cache) != previous['sha256']:
                    raise RuntimeError('Feature cache hash mismatch: '+spec['name'])
                print('FEATURE_REUSED', spec['name'], flush=True)
                continue
            if encoder is None:
                encoder = frozen_backbone.load_encoder(upstream)
                gpu_projection = torch.from_numpy(matrix).to('cuda')
            tick = time.perf_counter()
            scene = data.generate_scene(**spec)
            features = frozen_backbone.encode_scene(encoder, scene, out/'full_features')
            initial_labels = data.bench.initial_labels(scene.masks[0])
            global_pred = tracking.global_match(features, initial_labels)
            selected_pred = tracking.track_parts(features, scene.frames, scene.masks[0],
                replace(tracking.TrackerConfig(), memory_weight=0.), initial_labels)
            with torch.inference_mode():
                projected = (torch.from_numpy(features.reshape(-1, 1024)).to('cuda') @
                             gpu_projection).cpu().numpy().reshape(32,576,PROJECTION_DIM)
            if not np.isfinite(projected).all():
                raise RuntimeError('Nonfinite projected tokens')
            # Float16 cache is centered only after fitting the TRAIN mean.
            np.savez_compressed(cache, tokens=projected.astype(np.float16),
                global_scores=global_pred.scores, global_cells=global_pred.cells,
                selected_scores=selected_pred.scores, selected_cells=selected_pred.cells)
            entry = dict(**spec, split=split, path=str(cache.relative_to(out)),
                sha256=sha256(cache), bytes=cache.stat().st_size,
                rgb_sha256=hashlib.sha256(scene.frames.tobytes()).hexdigest(),
                initial_prompt_sha256=hashlib.sha256(scene.masks[0].tobytes()).hexdigest(),
                transform=data.transform_spec(spec['seed'],spec['condition']),
                extraction_seconds=time.perf_counter()-tick)
            manifest['clips'].append(entry)
            write_json(path,manifest)
            print('FEATURE_DONE',spec['name'],round(entry['extraction_seconds'],2),flush=True)
            del scene,features,projected,global_pred,selected_pred
            gc.collect()
    train = [e for e in manifest['clips'] if e['split']=='train']
    if len(train)==len(data.scene_specs('train')):
        mean_path=out/'mean.npy'
        if not mean_path.exists():
            total=np.zeros(PROJECTION_DIM,np.float64);count=0
            for entry in train:
                with np.load(out/entry['path']) as a:
                    tokens=a['tokens'].astype(np.float64)
                    total+=tokens.sum(axis=(0,1));count+=tokens.shape[0]*tokens.shape[1]
            np.save(mean_path,(total/count).astype(np.float32))
        current_mean={'path':'mean.npy','sha256':sha256(mean_path),'fit_split':'train',
                      'fit_clips':len(train)}
        if manifest.get('mean',current_mean)!=current_mean:
            raise RuntimeError('Frozen train mean changed')
        manifest['mean']=current_mean
    manifest['runtime']={'python':platform.python_version(),'numpy':np.__version__,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3}
    manifest['last_extraction_seconds']=time.perf_counter()-start
    checkpoints=Path(torch.hub.get_dir())/'checkpoints'
    manifest['backbone_downloads']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)}
                                   for p in checkpoints.glob('*') if p.is_file()]
    write_json(path,manifest)
    print('EXTRACTION_COMPLETE',len(manifest['clips']),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('/content/vjepa2'))
    p.add_argument('--splits',nargs='+',choices=['train','dev','test'],default=['train','dev'])
    p.add_argument('--checkpoint-freeze',type=Path)
    a=p.parse_args();extract(a.out,a.upstream,a.splits,a.checkpoint_freeze)
