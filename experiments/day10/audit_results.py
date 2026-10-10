"""Independent audit of the fixed single-image Day10 experiment.

Does not import the experiment, its operator, scorer, probe class, or renderer.
Reconstructs source-only edits with independent NumPy loops, reruns the frozen
readout with explicit Torch functional layers, and recounts every saved metric
and gate. Does not re-encode images; checkpoint inference and chronology are
bound by hashes and attestation, not independently repeated here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import traceback
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='4'
import numpy as np
REPO = Path(__file__).resolve().parents[2]
REGIONS = ('source_hole','destination','halo','distractor','background','common')
ARMS = ('noop','copy_repair','wrong_direction','genuine_target')
METRICS = ('primary_ratio','source_hole_ratio','destination_ratio','source_hole_rgb_mse',
    'source_hole_ghost_mean_occupancy','selected_centroid_error_px','selected_iou',
    'appearance_identity_accuracy','distractor_centroid_error_px','distractor_rgb_change_mse')
VERSION = 'day10_native_single_image_translation_v1'
WEIGHT_SHA = '7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values, dtype=np.float64)) if values else None


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = []
        self.warnings = []
        self.max_numeric_abs_difference = 0.0

    def require(self, condition, label):
        self.count += 1
        if not condition:
            raise AssertionError(label)

    def equal(self, actual, expected, label, rtol=2e-6, atol=2e-7):
        self.count += 1
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                self.errors.append(label + ": dictionary keys differ")
                return
            for key in expected:
                self.equal(actual[key], expected[key], label + "/" + key, rtol, atol)
            return
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.errors.append(label + ": sequence length/type differs")
                return
            for index, value in enumerate(expected):
                self.equal(actual[index], value, f"{label}/{index}", rtol, atol)
            return
        if expected is None:
            okay = actual is None or actual == ""
        elif isinstance(expected, (bool, str)):
            okay = actual == expected
        elif isinstance(expected, (int, float, np.integer, np.floating)):
            try:
                value = float(actual)
                difference = abs(value - float(expected))
                self.max_numeric_abs_difference = max(self.max_numeric_abs_difference, difference)
                okay = np.isfinite(value) and (value == expected if isinstance(expected, (int, np.integer))
                                               else np.isclose(value, expected, rtol=rtol, atol=atol))
            except (TypeError, ValueError):
                okay = False
        else:
            okay = actual == expected
        if not okay and len(self.errors) < 100:
            self.errors.append(f"{label}: actual={actual!r}, expected={expected!r}")

    def array(self, actual, expected, label, exact=False):
        self.require(actual.shape == expected.shape, label + ": shape")
        self.require(np.array_equal(actual, expected) if exact else
                     np.allclose(actual, expected, rtol=2e-6, atol=2e-7), label + ": values")


def translate(array, dx):
    result = np.zeros_like(array)
    if dx >= 0:
        if dx < array.shape[2]:
            result[:, :, dx:] = array[:, :, :array.shape[2] - dx]
    elif -dx < array.shape[2]:
        result[:, :, :dx] = array[:, :, -dx:]
    return result


def regions_from_coverage(source_fraction, target_fraction, distractor_fraction, dx, checks):
    checks.array(target_fraction, translate(source_fraction, dx), "target coverage translation")
    source, target, distractor = source_fraction > 0, target_fraction > 0, distractor_fraction > 0
    union = source | target
    padded = np.pad(union, ((0, 0), (1, 1), (1, 1)))
    support = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2)).any(axis=(-1, -2))
    return dict(source_hole=source & ~target, destination=target, halo=support & ~union,
                distractor=distractor, background=~(support | distractor),
                common=source | target | translate(source, -dx) | distractor | translate(distractor, dx)), support


def latent_metrics(error, noop, regions):
    output = {}
    for name, mask in regions.items():
        n = int(np.count_nonzero(mask))
        total, denominator = float(error[mask].sum(dtype=np.float64)), float(noop[mask].sum(dtype=np.float64))
        mse, reference = (total / n, denominator / n) if n else (None, None)
        output.update({name + "_tokens": n, name + "_squared_error_sum": total,
            name + "_noop_squared_error_sum": denominator, name + "_mse": mse,
            name + "_noop_mse": reference, name + "_ratio": mse / max(reference, 1e-12) if n else None,
            name + "_degenerate": int(not n or reference < 1e-12)})
    output["primary_ratio"] = mean(output[n + "_ratio"] for n in ("source_hole", "destination"))
    return output


def preservation(error, maximum, outside):
    n = int(outside.sum())
    total = float(error[outside].sum(dtype=np.float64))
    return {"outside_support_tokens": n, "outside_support_source_mse": total / n if n else None,
            "outside_support_source_squared_error_sum": total,
            "outside_support_max_abs_delta": float(maximum[outside].max()) if n else None,
            "outside_support_changed_tokens": int(np.count_nonzero(maximum[outside]))}


def lane(fraction):
    y = (np.arange(24, dtype=np.float64) + .5) * 16
    upper = float((fraction * y[None, :, None]).sum() / fraction.sum()) < 192
    return np.broadcast_to((y < 192 if upper else y >= 192)[:, None], (24, 24))


def center(weights, scoring_lane=None):
    w = np.asarray(weights, dtype=np.float64)
    if scoring_lane is not None:
        w = np.where(scoring_lane & (w >= .25), w, 0)
    mass = float(w.sum())
    if mass == 0:
        return None, None, mass
    y, x = np.indices(w.shape, dtype=np.float64)
    return float((w * (x + .5) * 16).sum() / mass), float((w * (y + .5) * 16).sum() / mass), mass


def distance(a, b):
    return 384.0 if a[0] is None or b[0] is None else float(np.hypot(a[0] - b[0], a[1] - b[1]))


def semantic_step(occ, rgb, original_occ, original_rgb, cached, t, regions, lanes):
    row = {}
    color_error = ((rgb.astype(np.float64) - cached["rgb_target"][t]) ** 2).mean(axis=-1)
    for label, fraction, scoring_lane in (
            ("selected", cached["target_frac"][t], lanes[0]),
            ("distractor", cached["distractor_frac"][t], lanes[1])):
        predicted, truth = center(occ, scoring_lane), center(fraction)
        binary, actual = (occ >= .5) & scoring_lane, (fraction >= .5) & scoring_lane
        intersection, union = int((binary & actual).sum()), int((binary | actual).sum())
        core = fraction >= .7
        row.update({label + "_centroid_error_px": distance(predicted, truth),
            label + "_centroid_present": int(predicted[0] is not None),
            label + "_centroid_x": predicted[0], label + "_centroid_y": predicted[1],
            label + "_centroid_mass": predicted[2], label + "_gt_centroid_x": truth[0],
            label + "_gt_centroid_y": truth[1], label + "_iou": intersection / union if union else 1.0,
            label + "_iou_intersection": intersection, label + "_iou_union": union,
            label + "_rgb_core_mse": float(color_error[core].mean()) if core.any() else None,
            label + "_rgb_core_error_sum": float(color_error[core].sum()), label + "_rgb_core_tokens": int(core.sum())})
    hole, other = regions["source_hole"][t], regions["distractor"][t]
    before, after = center(original_occ, lanes[1]), center(occ, lanes[1])
    row.update({"source_hole_ghost_mean_occupancy": float(occ[hole].mean()) if hole.any() else None,
        "source_hole_ghost_fraction_above_half": float((occ[hole] >= .5).mean()) if hole.any() else None,
        "distractor_occupancy_change_mse": float(((occ[other] - original_occ[other]) ** 2).mean()) if other.any() else None,
        "distractor_rgb_change_mse": float(((rgb[other] - original_rgb[other]) ** 2).mean()) if other.any() else None,
        "distractor_centroid_change_px": 0.0 if before[0] is None and after[0] is None else distance(after, before)})
    cores = [cached[k][t] >= .7 for k in ("source_frac", "target_frac", "distractor_frac")]
    separation = selected_distance = other_distance = None
    eligible, correct = False, None
    if all(mask.any() for mask in cores):
        selected_color = cached["rgb_source"][t][cores[0]].astype(np.float64).mean(axis=0)
        other_color = cached["rgb_source"][t][cores[2]].astype(np.float64).mean(axis=0)
        predicted_color = rgb[cores[1]].astype(np.float64).mean(axis=0)
        separation = float(np.linalg.norm(selected_color - other_color))
        selected_distance = float(np.linalg.norm(predicted_color - selected_color))
        other_distance = float(np.linalg.norm(predicted_color - other_color))
        eligible = separation >= .15
        correct = int(selected_distance < other_distance) if eligible else None
    row.update({"appearance_identity_eligible": int(eligible), "appearance_identity_skipped": int(not eligible),
        "appearance_identity_correct": correct, "appearance_identity_accuracy": float(correct) if eligible else None,
        "appearance_true_color_separation": separation, "appearance_distance_to_selected_color": selected_distance,
        "appearance_distance_to_distractor_color": other_distance})
    return row


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def specifications():
    return [{'name': 'image_'+str(seed), 'seed': seed, 'frame_index': 15,
             'shifts_px': [sign*d for d in (0,16,32,48,64,80)],
             'display_indices': [0,1,2,3,4,5,4,3,2,1,0]}
            for seed,sign in ((13200,1),(13201,1),(13202,-1),(13203,-1))]


def reference_background(z, fraction):
    selected = fraction > 0
    background = z.copy()
    # Same prescribed arithmetic order, separately implemented from the prose.
    for ti in range(z.shape[0]):
        clear = ~selected[ti]
        if not clear.any():
            raise ValueError('No unselected background token')
        for yi in range(z.shape[1]):
            for xi in range(z.shape[2]):
                if not selected[ti, yi, xi]:
                    continue
                a,b = max(0,yi-2), min(z.shape[1],yi+3)
                c,d = max(0,xi-2), min(z.shape[2],xi+3)
                pool = z[ti,a:b,c:d][clear[a:b,c:d]]
                if not len(pool):
                    pool = z[ti][clear]
                background[ti,yi,xi] = np.mean(pool,axis=0,dtype=np.float32)
    return background


def reference_edit(z, fraction, background, displacement):
    result = z.copy()
    if displacement == 0:
        return result
    selected = fraction > 0
    result[selected] = background[selected]
    # Destination-index scatter from the immutable original, no shift helper.
    for ti,yi,xi in np.argwhere(selected):
        new_x = int(xi + displacement)
        if not 0 <= new_x < z.shape[2]:
            raise ValueError('Selected token would leave the frame')
        result[ti,yi,new_x] = z[ti,yi,xi]
    return result


def frozen_readout(states, state):
    import torch
    import torch.nn.functional as F
    flat = torch.from_numpy(np.ascontiguousarray(states.reshape(-1,1024)))
    with torch.inference_mode():
        x = (flat - state['feature_mean']) / state['feature_std']
        x = F.linear(x,state['network.0.weight'],state['network.0.bias'])
        x = F.gelu(x)
        x = F.linear(x,state['network.2.weight'],state['network.2.bias'])
        y = torch.sigmoid(x).numpy().reshape(*states.shape[:-1],4)
    return {'occupancy':y[...,0], 'rgb':y[...,1:]}


def reference_summary(rows):
    scenes = sorted({r['scene'] for r in rows})
    per_image=[]
    for scene in scenes:
        for arm in ARMS:
            selected=[r for r in rows if r['scene']==scene and r['arm']==arm and r['shift_px']]
            per_image.append({'scene':scene,'arm':arm,'nonzero_positions':len(selected),
                **{metric:mean(r[metric] for r in selected) for metric in METRICS}})
    methods={arm:{metric:mean(r[metric] for r in per_image if r['arm']==arm)
                  for metric in METRICS} for arm in ARMS}
    genuine=[r for r in rows if r['arm']=='genuine_target']
    edited=[r for r in rows if r['arm']=='copy_repair']
    color=mean(r['appearance_identity_accuracy'] for r in genuine)
    good_reference=(mean(r['selected_centroid_error_px'] for r in genuine)<16
                    and all(r['selected_centroid_present'] for r in genuine)
                    and color is not None and color>=.9)
    good_localization=(mean(r['selected_centroid_error_px'] for r in edited)<16
                       and all(r['selected_centroid_present'] for r in edited))
    image_pass={r['scene']:bool(all(r[key]<1 for key in
                ('primary_ratio','source_hole_ratio','destination_ratio')))
                for r in per_image if r['arm']=='copy_repair'}
    nondegenerate=all(not r['source_hole_degenerate'] and not r['destination_degenerate']
                      for r in edited if r['shift_px'])
    gates={'genuine_readout_valid':bool(good_reference),
        'genuine_centroid_mean_px_all_states':mean(r['selected_centroid_error_px'] for r in genuine),
        'genuine_coarse_color_accuracy_all_states':color,
        'genuine_color_eligible_states':sum(r['appearance_identity_eligible'] for r in genuine),
        'genuine_color_skipped_states':sum(r['appearance_identity_skipped'] for r in genuine),
        'edited_centroid_below_one_patch_and_all_present':bool(good_localization),
        'edited_centroid_mean_px_all_states':mean(r['selected_centroid_error_px'] for r in edited),
        'region_errors_below_noop_by_image':image_pass,
        'no_degenerate_primary_denominators':bool(nondegenerate),
        'copy_repair_outside_union_exact':all(r['outside_requested_union_changed_tokens']==0 for r in edited)}
    return {'methods':methods,'per_image':per_image,'gates':gates,
        'qualified_controlled_result':bool(good_reference and good_localization and all(image_pass.values()) and nondegenerate)}


def audit(args, checks):
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    run=Path(args.run)
    freeze=read_json(args.freeze)
    publication=read_json(args.publication)
    manifest=read_json(run/'manifest.json')
    expected_specs=specifications()
    checks.equal(freeze['test_accessed'],False,'pretest unopened')
    checks.equal(freeze['test_specs'],expected_specs,'frozen specifications')
    checks.equal(manifest['test_specs'],expected_specs,'manifest specifications')
    checks.equal(manifest['freeze_sha256'],sha(args.freeze),'manifest freeze hash')
    checks.equal(publication['freeze_sha256'],sha(args.freeze),'publication freeze hash')
    checks.equal(publication['bytes_equal_to_GitHub'],True,'remote byte verification recorded')
    checks.equal(publication['test_encoded_before_verification'],False,'pretest publication ordering recorded')
    checks.require(len(publication['commit'])==40,'publication commit format')
    required={'experiments/day10/animate.py','experiments/day10/PROTOCOL.md'}|{
        'experiments/day8/'+p for p in ('data.py','extract.py','operators.py','evaluate.py','probe.py')}
    checks.require(required<=set(freeze['source_hashes']),'all frozen numerical dependencies')
    for path,digest in freeze['source_hashes'].items():
        checks.equal(sha(REPO/path),digest,'source hash '+path)
    checks.equal(manifest['source_hashes'],freeze['source_hashes'],'manifest source hashes')
    checks.equal(manifest['weights_sha256'],WEIGHT_SHA,'official checkpoint identity')
    checks.equal(freeze['probe_sha256'],sha(args.probe),'frozen probe hash')
    checks.equal(manifest['probe_sha256'],sha(args.probe),'manifest probe hash')
    checks.equal(manifest['inference_precision'],'float32','inference precision')
    checks.equal(manifest['cache_precision'],'float32','cache precision')
    if 'native_calibration' in freeze:
        calibration=Path(freeze['native_calibration']['path'])
        if not calibration.is_absolute():
            calibration=Path(args.freeze).parent/calibration
        checks.equal(sha(calibration),freeze['native_calibration']['sha256'],'native calibration hash')
    state=torch.load(args.probe,map_location='cpu',weights_only=True)['state_dict']
    checks.equal([r['spec'] for r in manifest['scenes']],expected_specs,'exact completed scene sequence')
    all_rows=[]
    all_invariants=[]
    tensor_count=0
    readout_count=0
    scene_reports=[]
    max_readout_difference=0.0
    for record in manifest['scenes']:
        spec=record['spec']
        name=spec['name']
        for path,digest in record['file_sha256'].items():
            checks.equal(sha(run/path),digest,'evidence hash '+path)
        with np.load(run/record['feature_path'],allow_pickle=False) as f:
            cache={k:f[k].copy() for k in f.files}
        with np.load(run/record['edited_path'],allow_pickle=False) as f:
            saved_edits={k:f[k].copy() for k in f.files}
        with np.load(run/record['predictions_path'],allow_pickle=False) as f:
            saved_predictions={k:f[k].copy() for k in f.files}
        saved_rows=read_json(run/record['metrics_json'])
        saved_invariants=read_json(run/record['invariants_json'])
        source=cache['source']
        shapes={'source':(1,24,24,1024),'targets':(6,1,24,24,1024),
            'source_frac':(1,24,24),'target_frac':(6,1,24,24),
            'distractor_frac':(1,24,24),'rgb_source':(1,24,24,3),'rgb_target':(6,1,24,24,3)}
        for key,shape in shapes.items():
            checks.require(cache[key].shape==shape,name+'/'+key+' shape')
            checks.require(cache[key].dtype==np.float32,name+'/'+key+' FP32 cache')
            checks.require(np.isfinite(cache[key]).all(),name+'/'+key+' finite')
        checks.array(cache['shifts_px'],np.array(spec['shifts_px']),name+'/shifts',exact=True)
        rgb=np.asarray(Image.open(run/record['source_rgb_path']).convert('RGB'))
        mask=np.asarray(Image.open(run/record['selected_mask_path']).convert('L'))
        checks.require(rgb.shape==(384,384,3),name+'/one RGB image')
        checks.require(mask.shape==(384,384) and np.isin(mask,[0,255]).all(),name+'/one binary selection')
        checks.equal(array_sha(rgb),record['source_rgb_sha256'],name+'/source RGB bytes')
        coverage=(mask==255).reshape(24,16,24,16).mean(axis=(1,3),dtype=np.float32)[None]
        checks.array(coverage,cache['source_frac'],name+'/mask patch coverage',exact=True)
        patch_rgb=rgb.reshape(24,16,24,16,3).mean(axis=(1,3),dtype=np.float32)[None]/255
        checks.array(patch_rgb,cache['rgb_source'],name+'/source RGB averages',exact=True)
        checks.array(cache['targets'][0],source,name+'/zero target identity',exact=True)
        source_hash=array_sha(source)
        background=reference_background(source,coverage)
        checks.array(saved_edits['fixed_background'],background,name+'/background reference',exact=True)
        references={'noop':np.stack([source.copy() for _ in spec['shifts_px']]),
            'copy_repair':np.stack([reference_edit(source,coverage,background,d//16) for d in spec['shifts_px']]),
            'wrong_direction':np.stack([reference_edit(source,coverage,background,-d//16) for d in spec['shifts_px']]),
            'genuine_target':cache['targets']}
        for arm in ('copy_repair','wrong_direction'):
            checks.array(saved_edits[arm],references[arm],name+'/'+arm+' full tensor',exact=True)
        predictions={arm:frozen_readout(value,state) for arm,value in references.items()}
        for arm in ARMS:
            for key in ('occupancy','rgb'):
                saved=saved_predictions[arm+'__'+key]
                actual=predictions[arm][key]
                difference=float(np.max(np.abs(saved-actual)))
                max_readout_difference=max(max_readout_difference,difference)
                checks.array(saved,actual,name+'/'+arm+'/'+key+' independent readout',exact=True)
            readout_count+=len(spec['shifts_px'])
        scene_rows=[]
        for index,shift in enumerate(spec['shifts_px']):
            target_rgb=np.asarray(Image.open(run/record['target_rgb_paths'][index]).convert('RGB'))
            true_patch=target_rgb.reshape(24,16,24,16,3).mean(axis=(1,3),dtype=np.float32)[None]/255
            checks.array(true_patch,cache['rgb_target'][index],name+f'/target RGB {index}',exact=True)
            labels={'source_frac':coverage,'target_frac':cache['target_frac'][index],
                'distractor_frac':cache['distractor_frac'],'rgb_source':cache['rgb_source'],'rgb_target':cache['rgb_target'][index]}
            regions,_=regions_from_coverage(coverage,labels['target_frac'],labels['distractor_frac'],shift//16,checks)
            target=cache['targets'][index]
            noop_error=((source.astype(np.float64)-target)**2).mean(axis=-1)
            lanes=(lane(labels['target_frac']),lane(labels['distractor_frac']))
            outside=~((coverage>0)|(labels['target_frac']>0))
            for arm in ARMS:
                latent=references[arm][index]
                error=((latent.astype(np.float64)-target)**2).mean(axis=-1)
                row={'scene':name,'seed':spec['seed'],'position_index':index,'shift_px':shift,'arm':arm,
                    'included_in_nonzero_summary':int(shift!=0)}
                row.update(latent_metrics(error,noop_error,regions))
                if not shift:
                    for key in row:
                        if key.endswith('_ratio'):
                            row[key]=None
                occ=predictions[arm]['occupancy'][index,0]
                color=predictions[arm]['rgb'][index,0]
                row.update(semantic_step(occ,color,predictions['noop']['occupancy'][index,0],
                    predictions['noop']['rgb'][index,0],labels,0,regions,lanes))
                hole=regions['source_hole']
                rgb_error=((predictions[arm]['rgb'][index].astype(np.float64)-labels['rgb_target'])**2).mean(axis=-1)
                row['source_hole_rgb_mse']=float(rgb_error[hole].mean()) if hole.any() else None
                row['outside_requested_union_changed_tokens']=int((np.any(latent!=source,axis=-1)&outside).sum())
                row['edited_latent_sha256']=array_sha(latent)
                scene_rows.append(row)
                tensor_count+=1
                if arm in ('copy_repair','wrong_direction'):
                    signed=(shift//16)*(1 if arm=='copy_repair' else -1)
                    selected=coverage>0
                    destination=translate(selected,signed)
                    union=selected|destination
                    local_hole=selected&~destination
                    checks.array(latent[~union],source[~union],name+'/'+arm+f'/outside {index}',exact=True)
                    checks.array(latent[destination],translate(source,signed)[destination],name+'/'+arm+f'/destination {index}',exact=True)
                    if shift:
                        checks.array(latent[local_hole],background[local_hole],name+'/'+arm+f'/fixed fill {index}',exact=True)
                    else:
                        checks.array(latent,source,name+'/'+arm+'/zero identity',exact=True)
                    protected=(cache['distractor_frac']>0)&~union
                    checks.array(latent[protected],source[protected],name+'/'+arm+f'/distractor preservation {index}',exact=True)
        checks.equal(saved_rows,scene_rows,name+'/all numerical rows')
        checks.equal(len(saved_rows),24,name+'/position row count')
        all_rows.extend(scene_rows)
        invariants=[]
        for display_index,position in enumerate(spec['display_indices']):
            shift=spec['shifts_px'][position]
            recomputed=reference_edit(source,coverage,background,shift//16)
            checks.array(recomputed,references['copy_repair'][position],name+f'/repeated display {display_index}',exact=True)
            invariants.append({'scene':name,'display_index':display_index,'shift_px':shift,
                'zero_shift_identity':bool(shift==0),'outside_union_exact':True,'destination_copy_exact':True,
                'uncovered_background_fixed':True,'day8_naive_equivalent':True,
                'copy_repair_sha256':array_sha(recomputed),
                'wrong_direction_sha256':array_sha(reference_edit(source,coverage,background,-shift//16))})
        checks.equal(saved_invariants,invariants,name+'/invariant records')
        checks.equal(array_sha(source),source_hash,name+'/source immutability during audit')
        all_invariants.extend(invariants)
        scene_reports.append({'scene':name,'full_tensors_reconstructed':24,
            'readouts_reproduced':24,'display_positions_checked':len(invariants),
            'background_reference_exact':True,'destination_exact':True,'outside_union_exact':True})
        print('SCENE_AUDITED',name,'checks',checks.count,flush=True)
    summary=read_json(run/'summary.json')
    reference=reference_summary(all_rows)
    for key,value in reference.items():
        checks.equal(summary[key],value,'summary/'+key)
    checks.equal(summary['n_images'],4,'four image aggregation')
    checks.equal(summary['n_unique_positions_per_image'],6,'six positions')
    checks.equal(summary['n_nonzero_positions_per_image'],5,'five nonzero positions')
    checks.equal(read_json(run/'invariants.json'),{'all_passed':True,'display_state_checks':all_invariants},'complete invariant index')
    checks.equal(read_csv(run/'per_position.csv'),all_rows,'position CSV')
    checks.equal(read_csv(run/'per_image.csv'),reference['per_image'],'image CSV')
    return {'full_tensors_reconstructed':tensor_count,'all_latent_hashes_match':True,
        'readouts_reproduced':readout_count,'max_readout_abs_difference':max_readout_difference,
        'position_rows_checked':len(all_rows),'image_rows_checked':len(reference['per_image']),
        'display_state_checks':len(all_invariants),'scenes':scene_reports,
        'qualified_controlled_result':reference['qualified_controlled_result'],
        'gates':reference['gates'],'methods':reference['methods']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--publication',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    checks=Checks()
    report={'status':'running','independent_implementation':True,
        'imports_experiment_operator_or_scorer':False,
        'scope_limits':['Encoder inference is not independently repeated.',
                        'Remote publication chronology is attested, not independently queried.',
                        'Synthetic target images are scored, not generated by this auditor.',
                        'Readout preservation cannot establish detailed visual identity.']}
    try:
        report.update(audit(args,checks))
        report['status']='passed' if not checks.errors else 'failed'
    except Exception as error:
        report['status']='failed'
        checks.errors.append(str(error))
        report['traceback']=traceback.format_exc()
    report.update({'checks':checks.count,'errors':checks.errors,
        'max_metric_abs_difference':checks.max_numeric_abs_difference,
        'auditor_sha256':sha(__file__),'freeze_sha256':sha(args.freeze),
        'manifest_sha256':sha(args.run/'manifest.json')})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='passed':
        raise SystemExit(1)

if __name__=='__main__':
    main()
