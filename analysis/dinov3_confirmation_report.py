"""Publish tables and plots only after the frozen DINO study verifier accepts."""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from wafervlm.core import LABELS, metrics, parse_label, read_jsonl, sha256_file, write_json
from runpod.dinov3_study import ROOT, names, plan, reference_for, verify_results
from wafervlm.dinov3 import recipe


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def validate_predictions(rows, reference):
    lookup={r['id']:r for r in reference}
    if len(lookup)!=len(reference) or len(rows)!=len(reference) or {r['id'] for r in rows}!=set(lookup):
        raise ValueError('Missing or duplicate prediction IDs')
    for row in rows:
        if any(row[k]!=lookup[row['id']][k] for k in ('label','image_sha256')) or parse_label(row['response'],row['truncated'])!=row['prediction']:
            raise ValueError('Image, label or response mismatch')


def paired(first, second, reference):
    validate_predictions(first, reference); validate_predictions(second, reference)
    left, right = ({r['id']: r for r in rows} for rows in (first, second))
    groups = dict(both_correct=[], dino_only_correct=[], qwen_only_correct=[], both_wrong=[])
    for ref in reference:
        a, b = left[ref['id']], right[ref['id']]
        key = {(True,True): 'both_correct', (False,True): 'dino_only_correct',
               (True,False): 'qwen_only_correct', (False,False): 'both_wrong'}[(a['prediction']==ref['label'], b['prediction']==ref['label'])]
        groups[key].append(ref['id'])
    assert sum(map(len, groups.values())) == len(reference)
    return groups


def matrices(path, report):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (1300, 1980), '#f3f6fa'); draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=17); small = ImageFont.load_default(size=13)
    draw.text((20,15), 'DINOv3 / ' + report['phase'] + ' / first pass for every fixed model', fill='#152b42', font=ImageFont.load_default(size=25))
    draw.text((20,52), 'X: TRUE LABEL  |  Y: PREDICTED LABEL  |  counts; invalid output = 0', fill='#152b42', font=font)
    for index, name in enumerate(names()):
        score=report['models'][name]; x=15+650*(index%2); y=95+625*(index//2)
        draw.text((x,y), f"{name}  acc {score['accuracy']:.2%}  macro-F1 {score['macro_f1']:.4f}", fill='#152b42', font=font)
        for i,label in enumerate(LABELS):
            draw.text((x+105+i*64,y+35),label,fill='#152b42',font=small)
            draw.text((x,y+80+i*64),label,fill='#152b42',font=small)
        for col,true in enumerate(LABELS):
            support=score['per_class'][true]['support']
            for row,pred in enumerate(LABELS):
                n=score['confusion_matrix'][true][pred]; f=n/max(1,support)
                rgb=(int(236-200*f),int(242-125*f),int(249-95*f))
                a,b=x+103+64*col,y+65+64*row
                draw.rectangle((a,b,a+61,b+61),fill=rgb)
                draw.text((a+23,b+20),str(n),fill='white' if f>.65 else '#152b42',font=font)
    image.save(path)


def publish(root, phase, spec, output):
    report=verify_results(root,phase,spec)
    worker=json.loads((root/'worker.json').read_text())
    hardware=json.loads((root/'hardware.json').read_text())
    if (worker['status']!='completed' or worker['phase']!=phase
            or any(worker.get(k)!=spec[k] for k in ('controller_sha','image_source_sha','image','plan_sha256','overlay_sha256'))
            or hardware.get('gpu')!=spec['gpu'] or hardware.get('bf16_matmul_passed') is not True):
        raise ValueError('Worker identity or executed CUDA evidence mismatch')
    report['execution']={k:spec[k] for k in ('run_id','controller_sha','image_source_sha','image','gpu','plan_sha256','overlay_sha256')}
    report['hardware']=hardware
    report['protocol']=plan()
    report['training_recipe']=recipe()
    output.mkdir(parents=True,exist_ok=False)
    reference=reference_for(phase); predictions={}
    for name in names():
        for repeat in range(1,(3 if phase=='validation' else 1)+1):
            src=root/phase/name/f'repeat{repeat}'; dst=output/'predictions'/name/f'repeat{repeat}'
            dst.mkdir(parents=True)
            for file in ('predictions.jsonl','metrics.json','run.json','status.json','checksums.json'):
                shutil.copyfile(src/file,dst/file)
            if repeat==1:predictions[name]=read_jsonl(src/'predictions.jsonl')
    qpath=ROOT/('docs/showcase/checkpoint-250/predictions.jsonl' if phase=='validation' else 'docs/evidence/qwen-holdout/seed17-step250/predictions.jsonl')
    qwen=read_jsonl(qpath); validate_predictions(qwen,reference)
    expected=recipe()['qwen_predictions_sha256'] if phase=='validation' else plan()['qwen_holdout_predictions_sha256']
    if sha256_file(qpath)!=expected:raise ValueError('Qwen published comparator hash changed')
    report['qwen_published_metrics']=metrics(qwen)
    report['qwen_paired']={n:paired(qwen,predictions[n],reference) for n in names()}
    if phase=='validation':
        report['qwen_validation_source_sha']='ae2cd56f7047c3015a34711324072f90f70dac2e'
        receipt=json.loads((ROOT/'docs/evidence/vlm-repeat-complete.json').read_text())
        qwen_six={}
        for pair in receipt['pairs']:
            endpoint=f"seed{pair['seed']}-step{pair['step']}"
            first,=[r for r in pair['passes'] if r['repeat']==1]
            path=ROOT/'docs/evidence/qwen-validation-first-pass'/(endpoint+'.jsonl')
            if sha256_file(path)!=first['predictions_sha256']:raise ValueError('Qwen first pass changed')
            rows=read_jsonl(path);validate_predictions(rows,reference)
            qwen_six[endpoint]={r['id']:r for r in rows}
        if set(qwen_six)!={f'seed{s}-step{t}' for s in (17,42,123) for t in (200,250)}:
            raise ValueError('Six Qwen comparators required')
        common=[r for r in reference if all(idx[r['id']]['prediction']!=r['label'] for idx in qwen_six.values())]
        if len(common)!=16 or sum(r['label']=='loc' for r in common)!=7:raise ValueError('Qwen diagnostic membership changed')
        report['qwen_common_errors']={}
        for name in names():
            idx={r['id']:r for r in predictions[name]}
            rescued=[r for r in common if idx[r['id']]['prediction']==r['label']]
            report['qwen_common_errors'][name]=dict(count=len(rescued),by_class=dict(Counter(r['label'] for r in rescued)),
                rescued_ids=[r['id'] for r in rescued],remaining=[dict(r,prediction=idx[r['id']]['prediction']) for r in common if r not in rescued])
        dino_six={n:{r['id']:r for r in predictions[n]} for n in names()}
        report['common_loc_cross_seed']=[dict(id=r['id'],label=r['label'],image_sha256=r['image_sha256'],
            qwen={n:v[r['id']]['prediction'] for n,v in qwen_six.items()},
            dino={n:v[r['id']]['prediction'] for n,v in dino_six.items()}) for r in common if r['label']=='loc']
        common_table=[dict(id=r['id'],label=r['label'],image_sha256=r['image_sha256'],
            **{'qwen-'+n:v[r['id']]['prediction'] for n,v in qwen_six.items()},
            **{n:v[r['id']]['prediction'] for n,v in dino_six.items()}) for r in common]
        with (output/'common-errors-16.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(common_table[0]),lineterminator='\n');writer.writeheader();writer.writerows(common_table)
        report['historical_seed17_changes']={}
        checksums=json.loads((ROOT/'docs/showcase/dinov3-seed17/checksums.json').read_text())
        for mode,file in [('linear','linear-predictions.jsonl'),('lora','predictions.jsonl')]:
            path=ROOT/'docs/showcase/dinov3-seed17'/file
            if sha256_file(path)!=checksums[file]:raise ValueError('Historical selected predictions changed')
            old={r['id']:r for r in read_jsonl(path)}
            report['historical_seed17_changes'][mode]=[dict(id=r['id'],label=r['label'],historical=old[r['id']]['prediction'],repeat1=r['prediction']) for r in predictions['seed17-'+mode] if r['prediction']!=old[r['id']]['prediction']]
    indices={n:{r['id']:r for r in rows} for n,rows in {'qwen-seed17-step250':qwen,**predictions}.items()}
    rows=[]
    for ref in reference:
        row={k:ref[k] for k in ('id','label','image_sha256')}
        if phase=='holdout':row['lot_group']=ref['lot_group']
        row.update({n:idx[ref['id']]['prediction'] for n,idx in indices.items()});rows.append(row)
    with (output/'paired-per-image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=[f'# DINOv3 {phase} confirmation\n',
           f"Run [{spec['run_id']}](https://github.com/jasonjiang8866/waferVLM/actions/runs/{spec['run_id']}); controller `{spec['controller_sha']}`; inference GPU: {spec['gpu']}.\n",
           f"Accepted **{report['validation_passes']}/18 validation passes** in this artifact; **{report['holdout_predictions']}/888 holdout predictions** in this artifact. Counts for the other phase are reported separately.\n",
           'The first independent process is primary for every fixed model. Seeds are training seeds; repeated processes are reproducibility checks. Seed 17 is exploratory, seeds 42/123 confirm the frozen protocol.\n',
           table(['Model','Selected epoch','Accuracy','Macro-F1','loc recall','Inference s','Peak allocated MiB'],
                 [[e['name'],e['selected_epoch'],f"{report['models'][e['name']]['accuracy']:.4%}",f"{report['models'][e['name']]['macro_f1']:.6f}",
                   f"{report['models'][e['name']]['per_class']['loc']['recall']:.2%}",
                   f"{report['passes'][e['name']][0]['run']['elapsed_seconds']:.3f}",f"{report['passes'][e['name']][0]['run']['peak_allocated_bytes']/2**20:.2f}"] for e in report['endpoints']]),
           '## Within-seed LoRA minus linear\n',
           table(['Seed','Accuracy difference (pp)','Macro-F1 difference'],[[s,f"{d['accuracy']*100:+.4f}",f"{d['macro_f1']:+.6f}"] for s,d in report['paired_lora_minus_linear'].items()]),
           '## Paired with published Qwen seed 17, step 250\n',
           table(['DINO model','Both correct','Only DINO correct','Only Qwen correct','Both wrong','Total'],[[n,*map(len,g.values()),sum(map(len,g.values()))] for n,g in report['qwen_paired'].items()]),
           '## Per-class recall\n',table(['Model',*LABELS],[[n,*[f"{report['models'][n]['per_class'][c]['recall']:.2%}" for c in LABELS]] for n in names()]),
           '![Six confusion matrices](confusion-matrices.png)\n',
           'All per-image predictions retain eight class probabilities in the canonical label order. The CSV joins IDs, image SHA-256 and ground truth. Full counts, metrics, repeat changes and timings are in [analysis.json](analysis.json).\n']
    if phase=='validation':
        lines+=[f"Six Qwen diagnostic first passes are pinned to PR #54 source `{report['qwen_validation_source_sha']}` and verified against the committed receipt hashes.\n", '## Fresh-process reproducibility\n',table(['Model','Pass 2 label changes','Pass 3 label changes'],[[n,*[p['label_change_count'] for p in report['passes'][n][1:]]] for n in names()]),
                'Changed IDs for each pass are retained in analysis.json. Historical seed-17 selected-epoch predictions remain the primary evidence for the separate original development report; its batch-1 reload is not one of these 18 passes.\n',
                '## Qwen common-error diagnostics\n',
                table(['Model','Rescued /16','loc rescued /7','Other rescued classes'],[[n,str(r['count']),str(r['by_class'].get('loc',0)),', '.join(f'{c}: {k}' for c,k in r['by_class'].items() if c!='loc') or 'none'] for n,r in report['qwen_common_errors'].items()]),
                'This subset is selected by Qwen performance. Use the complete 160 maps for overall performance claims. Every rescued ID and remaining prediction is in analysis.json; all six Qwen and all six DINO labels are in [common-errors-16.csv](common-errors-16.csv).\n',
                table(['Common-miss loc ID',*names()],[[r['id'],*[r['dino'][n] for n in names()]] for r in report['common_loc_cross_seed']]),
                table(['Seed-17 arm','Historical selected vs new pass-1 label changes'],[[m,len(changes)] for m,changes in report['historical_seed17_changes'].items()]),
                table(['Historical arm','Changed ID','True','Historical selected','New pass 1'],[[m,r['id'],r['label'],r['historical'],r['repeat1']] for m,changes in report['historical_seed17_changes'].items() for r in changes]),
                'Historical seed-17 selected predictions were produced on RTX 5090; this validation run uses RTX 4090. The observed differences do not isolate a single cause and do not select a new weight or a favorable repeat.\n']
    else:
        comparison=report['seed17_lora_minus_qwen']
        lines+=['## Lot-group paired bootstrap\n','10,000 paired resamples; RNG seed 20261110; 143 lot groups; fixed eight classes; undefined F1 = 0. Percentile 95% intervals below are within-seed LoRA minus linear.\n',
                table(['Seed','Accuracy 95% interval (pp)','Macro-F1 95% interval'],[[s,' to '.join(f'{v*100:+.4f}' for v in ci['accuracy']),' to '.join(f'{v:+.6f}' for v in ci['macro_f1'])] for s,ci in report['lot_group_percentile_95'].items()]),
                'This is an additional prespecified comparison on the same holdout whose Qwen results were already public. Six weight hashes and the inference/statistical protocol were committed before DINO holdout inference. No endpoint or seed was selected using this set.\n',
                '## Seed-17 LoRA versus published Qwen\n',
                table(['Model','Accuracy','Macro-F1'],[[n,f"{s['accuracy']:.4%}",f"{s['macro_f1']:.6f}"] for n,s in [('Qwen seed17 step250',report['qwen_published_metrics']),('DINO seed17 LoRA',report['models']['seed17-lora'])]]),
                table(['Metric','DINO minus Qwen','Paired lot-group 95% interval'],[[k,f"{comparison['delta'][k]:+.6f}",' to '.join(f'{v:+.6f}' for v in comparison['percentile_95'][k])] for k in ('accuracy','macro_f1')])]
    lines += ['Training hardware: '+ '; '.join(f"seed {seed}: {next(e['training_gpu'] for e in report['endpoints'] if e['seed']==seed)}" for seed in plan()['seeds'])+'. Each within-seed pair shares a training GPU type. Between-seed variation also includes the seed-17 versus confirmation-hardware difference.\n']
    lines+=['DINO timers include image verification, decoding, preprocessing, transfer and forward execution, exclude model loading, and use no warmup. Qwen previously timed generation only; these are not directly comparable speed measurements. DINO uses an eight-way argmax classifier and 10 epochs / 990 updates / 7,900 training presentations. Qwen uses greedy text generation (up to 16 new tokens), microbatch 1 / accumulation 8 and 250 optimizer steps under its own continuation protocol. The comparison is not matched for model size, pretraining, adaptation location or training compute.\n']
    write_json(output/'analysis.json',report);matrices(output/'confusion-matrices.png',report)
    (output/'README.md').write_text('\n'.join(lines))
    write_json(output/'checksums.json',{str(f.relative_to(output)):sha256_file(f) for f in sorted(output.rglob('*')) if f.is_file()})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('phase',choices=['validation','holdout'])
    parser.add_argument('spec',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();publish(args.root,args.phase,json.loads(args.spec.read_text()),args.output)
