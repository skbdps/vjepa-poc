"""Calibrate on calibration clips, freeze development choices, then evaluate test clips.

The feature cache contains no labels. Algorithm entrypoints receive RGB, features,
and the first frame annotation only. Later masks are passed exclusively to scoring.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent/'day3'))
import benchmark as bench
import tracking
from benchmark import annotated_video

MODEL_NAME = 'vjepa2_1_vit_large_384'
UPSTREAM_SHA = '204698b45b3712590f06245fbfba32d3be539812'

def source_digest():
    h=hashlib.sha256()
    for name in ('benchmark.py','tracking.py','run_experiment.py'):
        h.update(name.encode()); h.update((HERE/name).read_bytes())
    return h.hexdigest()

def load_encoder(root):
    import torch
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=UPSTREAM_SHA: raise RuntimeError(f'Expected upstream {UPSTREAM_SHA}, got {actual}')
    sys.path.insert(0,str(root))
    from src.hub import backbones
    backbones.VJEPA_BASE_URL='https://dl.fbaipublicfiles.com/vjepa2'
    encoder,predictor=getattr(backbones,MODEL_NAME)(pretrained=True)
    del predictor; gc.collect()
    return encoder.eval().requires_grad_(False).to('cuda')

def encode_scene(encoder,scene,cache_dir):
    import torch
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(scene.frames.tobytes()+(MODEL_NAME+UPSTREAM_SHA+'fp16-v1').encode()).hexdigest()[:16]
    path=cache_dir/f'{scene.name}_{fingerprint}.npy'
    expected=(len(scene.frames)//2,24,24,1024)
    if path.exists():
        arr=np.load(path)
        if arr.shape!=expected or not np.isfinite(arr).all(): raise RuntimeError(f'Invalid cache: {path}')
        return arr.astype(np.float32)
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1,1)
    windows=[]
    for offset in range(0,len(scene.frames),16):
        frames=np.ascontiguousarray(scene.frames[offset:offset+16])
        if len(frames)!=16: raise ValueError('Frame count must be divisible by16')
        x=torch.from_numpy(frames).permute(3,0,1,2).unsqueeze(0).to('cuda').float()/255
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            f=encoder((x-mean)/std)
        if tuple(f.shape)!=(1,4608,1024) or not torch.isfinite(f).all(): raise RuntimeError('Invalid encoder features')
        windows.append(f.float().cpu().numpy().reshape(8,24,24,1024).astype(np.float16))
        del x,f
    arr=np.concatenate(windows)
    np.save(path,arr)
    return arr.astype(np.float32)

def configurations():
    default=tracking.TrackerConfig()
    return {'persistent_vjepa':default,
            'no_motion':replace(default,motion_weight=0.),
            'no_context':replace(default,context_weight=0.),
            'no_memory':replace(default,memory_weight=0.)}

def predict(scene,features):
    initial=bench.initial_labels(scene.masks[0])
    predictions={'global_vjepa':tracking.global_match(features,initial),
                 'template_flow':tracking.template_tracker(scene.frames,scene.masks[0])}
    for method,cfg in configurations().items():
        predictions[method]=tracking.track_parts(features,scene.frames,scene.masks[0],config=cfg,initial_labels=initial)
    return predictions

def _save_predictions(path,predictions):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    arrays={}
    for method,p in predictions.items():
        arrays[method+'__scores']=p.scores
        arrays[method+'__cells']=p.cells
        arrays[method+'__eligible']=p.eligible
    np.savez_compressed(path,**arrays)

def _report(rows):
    methods=sorted(set(r['method'] for r in rows))
    conditions=sorted(set(r['condition'] for r in rows))
    result={}
    for method in methods:
        subset=[r for r in rows if r['method']==method]
        result[method]={'overall':bench.summarize_rows(subset),
                      'by_condition':{c:bench.summarize_rows([r for r in subset if r['condition']==c]) for c in conditions},
                      'by_window':{str(w):bench.summarize_rows([r for r in subset if r['window']==w]) for w in sorted(set(r['window'] for r in subset))},
                      'window_boundaries':bench.summarize_rows([r for r in subset if r['boundary']]),
                      'uncertainty':bench.summarize_with_ci(subset)}
    return result

def _write_report(folder,rows,metadata):
    folder.mkdir(parents=True,exist_ok=True)
    report={**metadata,'methods':_report(rows)}
    (folder/'results.json').write_text(json.dumps(report,indent=2))
    if rows:
        with (folder/'predictions.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return report

def _manifest(encoder):
    import torch
    return {'model':MODEL_NAME,'upstream_commit':UPSTREAM_SHA,'source_digest':source_digest(),
            'benchmark':bench.benchmark_manifest(),
            'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
            'peak_gpu_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
            'annotation':'frame0 pixel masks only for every method',
            'features':'frozen FP16; four disjoint16frame windows; offline attention within each window',
            'configs':{k:asdict(v) for k,v in configurations().items()},
            'template_config':asdict(tracking.TemplateConfig())}

def run_development(out,upstream,encoder=None):
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if encoder is None: encoder=load_encoder(upstream)
    torch.cuda.reset_peak_memory_stats()
    calibration=bench.generate_suite(split='calibration')
    development=bench.generate_suite(split='development')
    all_predictions={}
    for scene in calibration+development:
        start=time.perf_counter()
        features=encode_scene(encoder,scene,out/'features')
        pred=predict(scene,features);del features
        all_predictions[scene.name]=pred
        _save_predictions(out/'predictions'/f'{scene.name}.npz',pred)
        print(f'DEVELOPMENT_DONE {scene.name} {time.perf_counter()-start:.1f}s',flush=True)
    methods=list(all_predictions[calibration[0].name])
    thresholds={};calibration_notes={}
    for method in methods:
        thresholds[method],calibration_notes[method]=bench.calibrate_threshold(calibration,{s.name:all_predictions[s.name][method] for s in calibration})
    rows=[]
    for scene in development:
        for method in methods: rows.extend(bench.score_scene(scene,all_predictions[scene.name][method],thresholds[method],method))
    manifest=_manifest(encoder)
    report=_write_report(out/'development',rows,{'split':'development','thresholds':thresholds,'calibration':calibration_notes,'manifest':manifest})
    # Explicit development selection; final test is never consulted.
    def utility(method):
        m=report['methods'][method]['overall']
        loc=m['localization_accuracy_given_visible']['rate'] or 0.
        fp=m['false_presence_given_absent']['rate']
        return (loc+(1-fp if fp is not None else loc))/2
    candidates=list(configurations())
    selected=max(candidates,key=lambda m:(utility(m),m=='persistent_vjepa'))
    frozen={'source_digest':source_digest(),'thresholds':thresholds,'selected_method':selected,
            'selection_rule':'maximize mean(visible localization, absent specificity) on development only; prefer full method on ties',
            'development_utilities':{m:utility(m) for m in methods},'manifest':manifest,
            'calibration_scenes':[s.name for s in calibration],'development_scenes':[s.name for s in development]}
    (out/'frozen_config.json').write_text(json.dumps(frozen,indent=2))
    for condition in sorted(set(s.condition for s in development)):
        scene=next(s for s in development if s.condition==condition)
        view={m:all_predictions[scene.name][m] for m in ('global_vjepa','template_flow',selected)}
        annotated_video(scene,view,thresholds,out/'development'/f'{scene.name}_comparison.mp4')
    print('DEVELOPMENT_COMPLETE',json.dumps(frozen,indent=2),flush=True)
    return report,frozen

def run_test(out,upstream,encoder=None):
    import torch
    out=Path(out)
    frozen=json.loads((out/'frozen_config.json').read_text())
    if frozen['source_digest']!=source_digest(): raise RuntimeError('Code changed after development freeze; rerun development and record revision')
    if encoder is None: encoder=load_encoder(upstream)
    torch.cuda.reset_peak_memory_stats()
    thresholds=frozen['thresholds']; selected=frozen['selected_method']; rows=[]; scenes=bench.generate_suite(split='test')
    for i,scene in enumerate(scenes):
        start=time.perf_counter();features=encode_scene(encoder,scene,out/'features')
        pred=predict(scene,features);del features
        _save_predictions(out/'predictions'/f'{scene.name}.npz',pred)
        for method,p in pred.items():rows.extend(bench.score_scene(scene,p,thresholds[method],method))
        if not any(s.condition==scene.condition for s in scenes[:i]):
            annotated_video(scene,{m:pred[m] for m in ('global_vjepa','template_flow',selected)},thresholds,out/'test'/f'{scene.name}_comparison.mp4')
        print(f'TEST_DONE {scene.name} {time.perf_counter()-start:.1f}s',flush=True)
    comparisons={}
    selected_rows=[r for r in rows if r['method']==selected]
    for baseline in ('global_vjepa','template_flow'):
        baseline_rows=[r for r in rows if r['method']==baseline]
        comparisons[selected+'_minus_'+baseline]=bench.paired_bootstrap_difference(selected_rows,baseline_rows)
    report=_write_report(out/'test',rows,{'split':'test','frozen_config':frozen,'manifest':_manifest(encoder),'paired_comparisons':comparisons})
    print('TEST_COMPLETE',json.dumps({k:v['overall'] for k,v in report['methods'].items()},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','test'],required=True)
    p.add_argument('--out',type=Path,default=Path('/content/day4_run'));p.add_argument('--upstream',type=Path,default=Path('/content/vjepa2'))
    a=p.parse_args(); (run_development if a.stage=='development' else run_test)(a.out,a.upstream)
