"""Independent audit of the new held-out native-image-readout run.

Reuses only the separately written, frozen Day10 independent auditor. Overrides
its predeclared expected scene IDs, not the experiment or any metric/operator.
The failed original four-image result remains unchanged. Training provenance
and native-readout development are audited separately before this fresh test.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import traceback
import audit_results as independent


def native_specs():
    return [{'name':'image_'+str(seed),'seed':seed,'frame_index':15,
             'shifts_px':[sign*d for d in (0,16,32,48,64,80)],
             'display_indices':[0,1,2,3,4,5,4,3,2,1,0]}
            for seed,sign in ((13600,1),(13601,1),(13602,-1),(13603,-1))]



_original_summary = independent.reference_summary


def native_reference_summary(rows):
    result = _original_summary(rows)
    edited = [r for r in rows if r['arm'] == 'copy_repair']
    eligible = [r for r in edited if r['appearance_identity_eligible']]
    color = (sum(r['appearance_identity_correct'] for r in eligible) / len(eligible)
             if eligible else None)
    color_pass = color is not None and color >= .9
    result['original_day10_criteria_pass'] = result['qualified_controlled_result']
    result['gates'].update({'edited_coarse_color_accuracy_all_states':color,
        'edited_color_eligible_states':len(eligible),
        'edited_color_skipped_states':len(edited)-len(eligible),
        'edited_coarse_color_valid':bool(color_pass)})
    result['qualified_controlled_result'] = bool(result['qualified_controlled_result'] and color_pass)
    return result


def validate_followup(args, checks):
    freeze = independent.read_json(args.freeze)
    summary = independent.read_json(args.run/'summary.json')
    required = {'experiments/day10/native_readout.py',
                'experiments/day10/native_readout_PROTOCOL.md'}
    checks.require(required <= set(freeze['source_hashes']), 'native-readout sources frozen')
    bound = {}
    for key in ('training_freeze','training_publication','trained_readout','dev_validation','features_manifest'):
        ref = freeze[key]
        path = Path(ref['path'])
        if not path.is_absolute():
            path = args.freeze.parent/path
        checks.equal(independent.sha(path),ref['sha256'],'followup binding '+key)
        bound[key] = path
    checks.equal(independent.sha(bound['trained_readout']),independent.sha(args.probe),'trained final probe')
    gate = independent.read_json(bound['dev_validation'])
    checks.equal(gate['pass'],True,'development passed before new test')
    checks.equal(gate['probe_sha256'],independent.sha(args.probe),'development probe binding')
    checks.equal(gate['features_manifest_sha256'],independent.sha(bound['features_manifest']),'development feature binding')
    checks.equal(summary['readout_followup'],{
        'version':'day10_native_image_readout_followup_v1',
        'training_freeze_sha256':independent.sha(bound['training_freeze']),
        'probe_sha256':independent.sha(args.probe),
        'dev_gate_sha256':independent.sha(bound['dev_validation']),
        'operator_changed':False,'new_decoder_claim':False,
        'additional_gate':'Edited coarse-color accuracy>=.90 across all eligible fresh states'},
        'followup summary bindings')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--publication',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    checks=independent.Checks()
    independent.specifications=native_specs
    independent.reference_summary=native_reference_summary
    report={'status':'running','independent_implementation':True,
        'imports_experiment_operator_or_scorer':False,
        'separate_new_test':True,
        'expected_new_test_seeds':[13600,13601,13602,13603],
        'scope_limits':['Encoder inference and probe training are not repeated.',
                        'Remote publication chronology is attested, not independently queried.',
                        'Readout preservation cannot establish detailed visual identity.']}
    try:
        report.update(independent.audit(args,checks))
        validate_followup(args,checks)
        report['status']='passed' if not checks.errors else 'failed'
    except Exception as error:
        report['status']='failed'
        checks.errors.append(str(error))
        report['traceback']=traceback.format_exc()
    report.update({'checks':checks.count,'errors':checks.errors,
        'max_metric_abs_difference':checks.max_numeric_abs_difference,
        'auditor_sha256':independent.sha(__file__),
        'independent_base_auditor_sha256':independent.sha(independent.__file__),
        'freeze_sha256':independent.sha(args.freeze),
        'manifest_sha256':independent.sha(args.run/'manifest.json')})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='passed':
        raise SystemExit(1)

if __name__=='__main__':
    main()
