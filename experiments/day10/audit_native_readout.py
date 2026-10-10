"""Independent provenance/statistics/development audit of the native probe.

No training, encoder, renderer, experiment, or probe helper is imported.
Uses saved genuine-image caches to verify train-only normalization, the fixed
configuration and final checkpoint, and all development readouts/metrics/gates.
Does not rerun optimizer training or encoder inference, or open fresh test data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
import audit_results as independent

VERSION='day10_native_image_readout_followup_v1'
HYPERPARAMETERS={
    'seed':1010,'max_epochs':40,'batches_per_epoch':32,'batch_size':512,
    'learning_rate':.001,'hidden_dim':128,'optimizer':'AdamW','weight_decay':.0001,
    'selection':'final epoch','normalization':'train-only channel mean/std; floor1e-6',
    'sampling':'50/50 any-ball>0/background, replacement, uniform within group',
    'loss':'occupancy MSE + mean RGB-channel MSE',
    'inference_precision':'float32','training_precision':'float32','device':'cpu','threads':4}


def specifications(split):
    first,n,offsets=((13400,32,[16,-16,32,-32,48,-48,64,-64,80,-80]) if split=='train'
                     else (13500,8,[16,-16,32,-32,48,-48,80,-80]))
    return [{'name':f'{split}_{first+i}','seed':first+i,'split':split,'frame_index':15,
             'dx':offsets[i%len(offsets)],'views':['source','genuine_shifted_target']}
            for i in range(n)]


def audit(args, checks):
    import torch
    torch.set_num_threads(4)
    run=args.run
    freeze=independent.read_json(args.freeze)
    receipt=independent.read_json(args.publication)
    manifest=independent.read_json(run/'features_manifest.json')
    probe_path=run/'training/probe.pt'
    history_path=run/'training/history.json'
    checkpoint=torch.load(probe_path,map_location='cpu',weights_only=True)
    state=checkpoint['state_dict']
    history=independent.read_json(history_path)
    checks.equal(checkpoint['history'],history,'saved checkpoint history')
    checks.equal(checkpoint['version'],'day8_frozen_pointwise_readout_v1','unchanged diagnostic architecture')
    checks.equal(checkpoint['hidden_dim'],128,'fixed hidden dimension')
    checks.equal(freeze['version'],VERSION,'training freeze version')
    checks.equal(freeze['training_accessed'],False,'pretraining freeze')
    checks.equal(freeze['test_accessed'],False,'pretraining test unopened')
    checks.equal(freeze['hyperparameters'],HYPERPARAMETERS,'predeclared hyperparameters')
    checks.equal(receipt['freeze_sha256'],independent.sha(args.freeze),'publication freeze hash')
    checks.equal(receipt['bytes_equal_to_GitHub'],True,'byte verification attestation')
    checks.equal(receipt['training_encoded_before_verification'],False,'pretraining publication chronology')
    checks.require(len(receipt['commit'])==40,'publication commit length')
    for path,digest in freeze['source_hashes'].items():
        checks.equal(independent.sha(independent.REPO/path),digest,'frozen source '+path)
    checks.equal(manifest['source_hashes'],freeze['source_hashes'],'manifest source bindings')
    checks.equal(manifest['training_freeze_sha256'],independent.sha(args.freeze),'feature freeze binding')
    checks.equal(manifest['training_publication_sha256'],independent.sha(args.publication),'feature publication binding')
    checks.equal(manifest['test_accessed'],False,'features test unopened')
    checks.equal(manifest['weights_sha256'],independent.WEIGHT_SHA,'official encoder weight identity')
    checks.equal(manifest['precision'],'float32','native encoding precision')
    checks.equal(manifest['cache_precision'],'float32','cache precision')
    for split in ('train','dev'):
        checks.equal(freeze[split+'_specs'],specifications(split),'frozen '+split+' specs')
        checks.equal(manifest[split+'_specs'],specifications(split),'manifest '+split+' specs')
    expected=specifications('train')+specifications('dev')
    checks.equal([r['spec'] for r in manifest['scenes']],expected,'all independent train/dev scenes')
    checks.require(not ({r['seed'] for r in expected}&(set(range(13200,13204))|set(range(13600,13604)))),
                   'both old and new test families excluded from train/dev')
    total=np.zeros(1024,np.float64)
    squares=np.zeros(1024,np.float64)
    count=foreground=0
    development=[]
    for record in manifest['scenes']:
        path=run/record['path']
        checks.equal(independent.sha(path),record['sha256'],'cache '+record['spec']['name'])
        checks.equal(path.stat().st_size,record['bytes'],'cache bytes')
        with np.load(path,allow_pickle=False) as f:
            cached={k:f[k].copy() for k in f.files}
        shapes={'tokens':(2,1,24,24,1024),'occupancy':(2,1,24,24),
                'rgb':(2,1,24,24,3),'source_frac':(1,24,24),
                'target_frac':(2,1,24,24),'distractor_frac':(1,24,24),'rgb_source':(1,24,24,3)}
        for key,shape in shapes.items():
            checks.require(cached[key].shape==shape and cached[key].dtype==np.float32,'FP32 shape '+key)
            checks.require(np.isfinite(cached[key]).all(),'finite '+key)
            if key!='tokens':
                checks.require(((cached[key]>=0)&(cached[key]<=1)).all(),'label range '+key)
        spec=record['spec']
        checks.array(cached['view_shift_px'],np.array([0,spec['dx']]),'two declared genuine views',exact=True)
        checks.array(cached['target_frac'][0],cached['source_frac'],'source mask view',exact=True)
        checks.array(cached['target_frac'][1],independent.translate(cached['source_frac'],spec['dx']//16),'target mask translation',exact=True)
        checks.array(cached['rgb'][0],cached['rgb_source'],'genuine source RGB label',exact=True)
        for view in range(2):
            coverage=cached['target_frac'][view]+cached['distractor_frac']
            checks.array(cached['occupancy'][view],coverage,'two disjoint-object occupancy union',exact=True)
        if spec['split']=='train':
            for view in range(2):
                block=cached['tokens'][view].reshape(-1,1024).astype(np.float64)
                total+=block.sum(axis=0)
                squares+=np.einsum('nc,nc->c',block,block)
                count+=len(block)
                foreground+=int((cached['occupancy'][view]>0).sum())
        else:
            development.append((spec,cached))
    mean=total/count
    std=np.sqrt(np.maximum(squares/count-mean*mean,0))
    floored=int((std<1e-6).sum())
    mean=mean.astype(np.float32)
    std=np.maximum(std,1e-6).astype(np.float32)
    checks.array(state['feature_mean'].numpy(),mean,'independently recounted train-only mean',exact=True)
    checks.array(state['feature_std'].numpy(),std,'independently recounted train-only std',exact=True)
    import hashlib
    statistics_sha=hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()
    checks.equal(history['train'],{'examples':64,'tokens':count,'foreground_tokens':foreground,
        'background_tokens':count-foreground,'std_floor':1e-6,'std_floored_channels':floored,
        'statistics_sha256':statistics_sha},'all training normalization provenance')
    expected_config={key:HYPERPARAMETERS[key] for key in ('max_epochs','batches_per_epoch','batch_size',
        'learning_rate','hidden_dim','weight_decay','sampling','loss','selection')}
    expected_config['dev_sample_size']=0
    checks.equal(history['config'],expected_config,'fixed optimizer/training configuration')
    checks.equal(history['seed'],1010,'fixed training seed')
    checks.equal(history['dev_examples'],0,'no development in fitting or selection')
    checks.equal(history['selected_epoch'],40,'fixed final epoch')
    checks.equal(history['selected_dev'],None,'no development checkpoint selection')
    checks.equal(history['test_accessed'],False,'training no test access')
    checks.equal([r['epoch'] for r in history['epochs']],list(range(1,41)),'all40epochs preserved')
    checks.require(all('dev' not in r for r in history['epochs']),'no dev-based epoch decisions')
    loss_rows=[]
    for epoch in history['epochs']:
        checks.require(all(np.isfinite(v) for v in epoch['train'].values()),'finite epoch losses')
        loss_rows.append({'epoch':epoch['epoch'],**{'train_'+k:v for k,v in epoch['train'].items()}})
    checks.equal(independent.read_csv(run/'training/losses.csv'),loss_rows,'loss CSV')
    checks.equal(history['native_readout_followup'],{'version':VERSION,'hyperparameters':HYPERPARAMETERS,
        'training_freeze_sha256':independent.sha(args.freeze),
        'features_manifest_sha256':independent.sha(run/'features_manifest.json'),
        'training_seeds':list(range(13400,13432)),
        'development_role':'Single post-training gate only; never checkpoint selection',
        'original_test_seeds_used':[],'fresh_test_accessed':False},'followup training provenance')
    with np.load(run/'training/dev_predictions.npz',allow_pickle=False) as f:
        saved_predictions={k:f[k].copy() for k in f.files}
    rows=[]
    for spec,cached in development:
        prediction=independent.frozen_readout(cached['tokens'],state)
        for key in ('occupancy','rgb'):
            checks.array(saved_predictions[spec['name']+'__'+key],prediction[key],'independent dev '+spec['name']+'/'+key,exact=True)
        for index,dx in enumerate(cached['view_shift_px']):
            labels={'source_frac':cached['source_frac'],'target_frac':cached['target_frac'][index],
                    'distractor_frac':cached['distractor_frac'],'rgb_source':cached['rgb_source'],'rgb_target':cached['rgb'][index]}
            regions,_=independent.regions_from_coverage(labels['source_frac'],labels['target_frac'],
                labels['distractor_frac'],int(dx)//16,checks)
            lanes=(independent.lane(labels['target_frac']),independent.lane(labels['distractor_frac']))
            row={'scene':spec['name'],'seed':spec['seed'],'view_index':index,'shift_px':int(dx)}
            row.update(independent.semantic_step(prediction['occupancy'][index,0],prediction['rgb'][index,0],
                prediction['occupancy'][0,0],prediction['rgb'][0,0],labels,0,regions,lanes))
            row['occupancy_mse']=float(((prediction['occupancy'][index].astype(np.float64)-cached['occupancy'][index])**2).mean())
            row['rgb_mse']=float(((prediction['rgb'][index].astype(np.float64)-cached['rgb'][index])**2).mean())
            rows.append(row)
    checks.equal(independent.read_json(run/'training/dev_metrics.json'),rows,'all16 independent dev metrics')
    checks.equal(independent.read_csv(run/'training/dev_metrics.csv'),rows,'dev metrics CSV')
    centroid=independent.mean(r['selected_centroid_error_px'] for r in rows)
    color=independent.mean(r['appearance_identity_accuracy'] for r in rows)
    all_present=all(r['selected_centroid_present'] for r in rows)
    gate={'version':VERSION,'n_scenes':8,'n_genuine_images':16,
        'selected_centroid_mean_px':centroid,'all_selected_centroids_present':all_present,
        'coarse_color_accuracy':color,'coarse_color_eligible':sum(r['appearance_identity_eligible'] for r in rows),
        'coarse_color_skipped':sum(r['appearance_identity_skipped'] for r in rows),
        'selected_iou_mean':independent.mean(r['selected_iou'] for r in rows),
        'occupancy_mse_mean':independent.mean(r['occupancy_mse'] for r in rows),
        'rgb_mse_mean':independent.mean(r['rgb_mse'] for r in rows),
        'rule':'Final fixed-epoch model only; genuine mean selected centroid <16px, all present, eligible coarse-color accuracy>=.90',
        'used_for_tuning':False,'test_accessed':False,
        'pass':bool(centroid is not None and centroid<16 and all_present and color is not None and color>=.9),
        'probe_sha256':independent.sha(probe_path),'training_history_sha256':independent.sha(history_path),
        'features_manifest_sha256':independent.sha(run/'features_manifest.json')}
    checks.equal(independent.read_json(run/'training/dev_gate.json'),gate,'development pass/fail gate')
    return {'train_scenes':32,'genuine_training_images':64,'normalization_tokens':count,
            'train_only_normalization_bitwise_exact':True,'development_scenes':8,
            'development_images':16,'development_readouts_bitwise_exact':True,
            'development_gate':gate,'fresh_test_data_opened_by_auditor':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','freeze','publication','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    checks=independent.Checks()
    report={'status':'running','independent_implementation':True,
            'imports_experiment_or_training_helpers':False,
            'scope_limits':['Encoder inference and optimizer fitting are not independently rerun.',
                            'Remote publication chronology is attested, not independently queried.']}
    try:
        report.update(audit(args,checks))
        report['status']='passed' if not checks.errors else 'failed'
    except Exception as error:
        checks.errors.append(str(error))
        report['status']='failed'
        report['traceback']=traceback.format_exc()
    report.update({'checks':checks.count,'errors':checks.errors,
                   'max_metric_abs_difference':checks.max_numeric_abs_difference,
                   'auditor_sha256':independent.sha(__file__),
                   'independent_base_auditor_sha256':independent.sha(independent.__file__),
                   'training_freeze_sha256':independent.sha(args.freeze)})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='passed':
        raise SystemExit(1)

if __name__=='__main__':
    main()
