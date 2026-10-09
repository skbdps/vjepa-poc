"""Encode genuine source/target videos for direct latent intervention.

Targets supervise training and scoring only. The operator never receives target
RGB or target features. Oracle masks of both source objects isolate editing from
selection/tracking. Encoder attention is offline within two disjoint blocks.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import data

HERE = Path(__file__).resolve().parent
MODEL = 'vjepa2_1_vit_large_384'
UPSTREAM = '204698b45b3712590f06245fbfba32d3be539812'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''): h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def sources():
    return {name:sha256(HERE/name) for name in ('data.py','extract.py')}

def load_encoder(upstream, device="cuda"):
    import torch
    actual=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if actual!=UPSTREAM: raise RuntimeError('Unexpected official source revision: '+actual)
    sys.path.insert(0,str(upstream))
    from src.hub import backbones
    backbones.VJEPA_BASE_URL='https://dl.fbaipublicfiles.com/vjepa2'
    encoder,predictor=getattr(backbones,MODEL)(pretrained=True)
    del predictor;gc.collect()
    return encoder.eval().requires_grad_(False).to(device)

def encode(encoder, frames, device="cuda"):
    import torch
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1,1)
    result=[]
    for start in range(0,len(frames),16):
        x=torch.from_numpy(np.ascontiguousarray(frames[start:start+16])).permute(3,0,1,2)[None].to(device).float()/255
        with torch.inference_mode(),torch.autocast(device,dtype=torch.float16,enabled=(device=='cuda')):
            z=encoder((x-mean)/std)
        if tuple(z.shape)!=(1,4608,1024) or not torch.isfinite(z).all():
            raise RuntimeError('Unexpected/nonfinite JEPA features')
        result.append(z.cpu().numpy().reshape(8,24,24,1024))
    return np.concatenate(result).astype(np.float16)

def patch_rgb(frames):
    return (frames.reshape(16,2,24,16,24,16,3).astype(np.float32).mean(axis=(1,3,5))/255).astype(np.float32)

def run(out, upstream, splits, freeze=None, device="cuda", limit=None):
    import torch
    out=Path(out);(out/'cache').mkdir(parents=True,exist_ok=True)
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'model':MODEL,'upstream':UPSTREAM,'source_hashes':sources(),'data':data.manifest(),
        'clips':[], 'encoder_context':'two disjoint16frame blocks; offline within block',
        'oracle_budget':'source selected-object and distractor masks at all32frames; requested dx',
        'target_policy':'target RGB/tokens supervise training or scoring only'}
    if manifest['source_hashes']!=sources():raise RuntimeError('Extraction source changed')
    if 'test' in splits:
        if freeze is None:raise ValueError('Test requires completed training freeze')
        frozen=json.loads(Path(freeze).read_text())
        if frozen.get('test_accessed') is not False:raise ValueError('Invalid pretest freeze')
        for name,digest in frozen['source_hashes'].items():
            if sha256(HERE/name)!=digest:raise ValueError('Training source changed: '+name)
        for record in frozen['models']:
            if sha256(Path(freeze).parent/record['path'])!=record['sha256']:raise ValueError('Trained model changed')
        manifest['test_freeze_sha256']=sha256(freeze)
        write_json(manifest_path,manifest)
    encoder=None
    for split in splits:
        for spec in data.scene_specs(split)[:limit]:
            path=out/'cache'/(spec['name']+'.npz')
            old=next((r for r in manifest['clips'] if r['spec']['name']==spec['name']),None)
            if old:
                if sha256(path)!=old['sha256']:raise ValueError('Changed feature cache')
                print('FEATURE_REUSED',spec['name'],flush=True);continue
            if encoder is None:encoder=load_encoder(upstream, device)
            tick=time.perf_counter();pair=data.generate_pair(spec)
            z0=encode(encoder,pair['frames_source'],device);z1=encode(encoder,pair['frames_target'],device)
            np.savez_compressed(path,source=z0,target=z1,
                source_frac=data.patch_fractions(pair['masks_source']),
                target_frac=data.patch_fractions(pair['masks_target']),
                distractor_frac=data.patch_fractions(pair['masks_distractor']),
                rgb_source=patch_rgb(pair['frames_source']),rgb_target=patch_rgb(pair['frames_target']))
            record={'spec':spec,'path':str(path.relative_to(out)),'sha256':sha256(path),'bytes':path.stat().st_size,
                'source_rgb_sha256':hashlib.sha256(pair['frames_source'].tobytes()).hexdigest(),
                'target_rgb_sha256':hashlib.sha256(pair['frames_target'].tobytes()).hexdigest(),
                'seconds':time.perf_counter()-tick}
            manifest['clips'].append(record)
            write_json(out/'metadata'/(spec['name']+'.json'),pair['metadata'])
            write_json(manifest_path,manifest)
            print('FEATURE_DONE',spec['name'],round(record['seconds'],2),flush=True)
            del pair,z0,z1;gc.collect()
    manifest['runtime']={'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3 if device=='cuda' else None}
    manifest['weights']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)}
        for p in (Path(torch.hub.get_dir())/'checkpoints').glob('vjepa2_1_vitl_dist_vitG_384.pt')]
    write_json(manifest_path,manifest)
    print('EXTRACTION_COMPLETE',len(manifest['clips']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('/content/vjepa2'))
    p.add_argument('--splits',nargs='+',choices=['train','dev','test'],default=['train','dev'])
    p.add_argument('--freeze',type=Path);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--limit',type=int);a=p.parse_args();run(a.out,a.upstream,a.splits,a.freeze,a.device,a.limit)
