"""Recompute released validation scores and build a deterministic error gallery."""
import argparse
import base64
import html
import json
from pathlib import Path
from wafervlm.core import LABELS, audit_dataset, checked_image, metrics, read_jsonl, write_json
from wafervlm.experiment import seal_output


def analyze(evidence, output, dataset=None):
    evidence,output=Path(evidence),Path(output);output.mkdir(parents=True,exist_ok=False)
    reports={};predictions={}; identity=None
    for step in (100,150,200):
        directory=evidence/f'validation-{step}'
        rows=read_jsonl(directory/'predictions.jsonl')
        keys=[(r['id'],r['label'],r['image_sha256']) for r in rows]
        if len(rows)!=160 or (identity is not None and keys!=identity):raise ValueError('Different validation samples')
        identity=keys; score=metrics(rows); saved=json.loads((directory/'metrics.json').read_text())
        for key in ('accuracy','macro_f1','invalid_rate','per_class','confusion_matrix'):
            if score[key]!=saved[key]: raise ValueError('Stored metric differs from predictions')
        reports[str(step)]={**score,'answer_token_loss':saved['answer_token_loss']};predictions[step]=rows
    write_json(output/'metrics.json',{'split':'validation','checkpoints':reports,'source_run_id':42502913001})
    lines=['# Checkpoint 100/150/200 validation diagnostics','',
           'Same 160 validation images; metrics recomputed from released predictions. No test scores used.','',
           '| Metric | 100 | 150 | 200 |','|---|---:|---:|---:|']
    for key in ('accuracy','macro_f1','answer_token_loss','invalid_rate'):
        lines.append('| '+key+' | '+' | '.join(f'{reports[str(s)][key]:.6f}' for s in (100,150,200))+' |')
    lines+=['','## Per-class validation recall','','| Label | 100 | 150 | 200 |','|---|---:|---:|---:|']
    for label in LABELS:
        lines.append('| '+label+' | '+' | '.join(f'{reports[str(s)]["per_class"][label]["recall"]:.3f}' for s in (100,150,200))+' |')
    lines+=['','## Step-200 confusion counts','','Rows are true labels; columns are predictions.','',
            '| True / predicted | '+' | '.join((*LABELS,'invalid'))+' |', '|---|'+'---:|'*9]
    for label in LABELS:
        lines.append('| '+label+' | '+' | '.join(str(reports['200']['confusion_matrix'][label][p]) for p in (*LABELS,'invalid'))+' |')
    errors=[r for r in predictions[200] if r['label']!=r['prediction']]
    write_json(output/'errors.json',errors)
    history_path=evidence.parent/'train/checkpoints/checkpoint-200/trainer_state.json'
    if history_path.is_file():
        history=json.loads(history_path.read_text())['log_history']
        loss=[(r['step'],r['loss']) for r in history if 'loss' in r]
        series={'train_loss':loss,'validation_loss':[(s,reports[str(s)]['answer_token_loss']) for s in (100,150,200)],
                'validation_macro_f1':[(s,reports[str(s)]['macro_f1']) for s in (100,150,200)]}
        write_json(output/'learning-curves.json',series)
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440" viewBox="0 0 960 440">',
             '<rect width="960" height="440" fill="white"/><g font-family="sans-serif" font-size="13" fill="#182330">']
        for index,(title,names) in enumerate([('Answer-token loss',('train_loss','validation_loss')),('Validation macro-F1',('validation_macro_f1',))]):
            left=60+index*470;top=50;width=400;height=320
            limit=max(v for n in names for _,v in series[n])*1.1
            svg.append(f'<text x="{left}" y="24" font-size="18">{title}</text>')
            for j in range(6):
                y=top+height-j*height/5
                svg.append(f'<line x1="{left}" x2="{left+width}" y1="{y}" y2="{y}" stroke="#e1e6ec"/><text x="{left-45}" y="{y+4}">{limit*j/5:.2f}</text>')
            for step in (0,50,100,150,200):
                x=left+step/200*width
                svg.append(f'<text x="{x-10}" y="394">{step}</text>')
            svg.append(f'<line x1="{left+width/2}" x2="{left+width/2}" y1="{top}" y2="{top+height}" stroke="#777" stroke-dasharray="4 4"/>')
            for name,color in zip(names,('#087f8c','#d45a20')):
                points=' '.join(f'{left+s/200*width:.2f},{top+height-v/limit*height:.2f}' for s,v in series[name])
                svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
            svg.append(f'<text x="{left}" y="422">Global optimizer step; LR 2e-4 → 5e-5 after 100</text>')
        svg.append('</g></svg>');(output/'learning-curves.svg').write_text(''.join(svg))
        lines+=['','## Learning curves','','![Actual per-step training loss and validation metrics](learning-curves.svg)',
                '', 'Left: teal is per-step training loss; orange is validation answer-token loss. Right: validation macro-F1.',
                'The dashed line marks the LR change after step 100. This is the lower-LR trajectory, not the separate constant-LR control.']
    lines+=['', 'Error examples are observations, not proven annotation errors. ',
            'Use the validation evidence to form separately specified interventions; do not relabel or alter splits silently.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    if dataset:
        lookup={r['id']:r for r in audit_dataset(dataset)['validation']}
        cards=[]
        for label in LABELS:
            for error in sorted((r for r in errors if r['label']==label),key=lambda r:r['id'])[:4]:
                row=lookup[error['id']]
                if row['image_sha256']!=error['image_sha256']:raise ValueError('Gallery image mismatch')
                data=base64.b64encode(checked_image(dataset,row).read_bytes()).decode()
                title=html.escape(f"{row['id']} | {label} → {error['prediction']}")
                cards.append(f'<figure><img src="data:image/png;base64,{data}" alt="{title}"><figcaption>{title}</figcaption></figure>')
        page='<!doctype html><meta charset="utf-8"><title>Validation error gallery</title><style>body{font:16px system-ui;margin:32px;background:#101820;color:#e8edf2}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(256px,1fr));gap:16px}figure{margin:0;padding:12px;background:#203040}img{width:100%;image-rendering:pixelated}figcaption{overflow-wrap:anywhere;margin-top:12px}</style><h1>Step-200 validation errors</h1><p>Up to four errors per true class, sorted by sample ID. Original rendered maps; hypotheses require review.</p><main>'+''.join(cards)+'</main>'
        (output/'gallery.html').write_text(page)
    seal_output(output)
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset')
    a=p.parse_args();r=analyze(a.evidence,a.output,a.dataset)
    print(json.dumps({s:{k:v for k,v in m.items() if k in ('accuracy','macro_f1','answer_token_loss')} for s,m in r.items()}))
