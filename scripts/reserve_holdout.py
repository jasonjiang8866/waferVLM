"""Freeze unseen lot groups without reading any model outputs or changing development data."""
import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from wafervlm.core import LABELS, audit_dataset, sha256_file, write_json, write_jsonl
from wafervlm.data import group_records, quarantine_conflicts, render_map, scalar


def select_holdout(records, development, image_digest, cap=20):
    retained, quarantined = quarantine_conflicts(records)
    unique, groups = group_records(retained)
    used_groups = {groups[r['lot']] for r in development}
    used_maps = {r['map_sha256'] for r in development}
    used_images = {r['image_sha256'] for r in development}
    selected = []; selected_images = set(); collisions = 0
    available = Counter()
    candidates = []
    for r in unique:
        if groups[r['lot']] not in used_groups and r['map_sha256'] not in used_maps:
            candidates.append(r); available[r['label']] += 1
    # Fixed salted identity order; no predictions, confidence or difficulty scores.
    candidates.sort(key=lambda r: hashlib.sha256(('final-holdout-v1:'+r['id']).encode()).hexdigest())
    counts = Counter()
    for r in candidates:
        if counts[r['label']] >= cap:
            continue
        image_sha = image_digest(r)
        if image_sha in used_images or image_sha in selected_images:
            collisions += 1
            continue
        selected.append({**r, 'lot_group':groups[r['lot']], 'image_sha256':image_sha})
        selected_images.add(image_sha); counts[r['label']] += 1
    return sorted(selected,key=lambda r:r['id']), {
        'class_counts':{k:counts[k] for k in LABELS},
        'eligible_unique_maps':{k:available[k] for k in LABELS},
        'excluded_development_lot_groups':len(used_groups),
        'selected_lot_groups':sorted({r['lot_group'] for r in selected}),
        'render_collisions_skipped':collisions, 'quarantined_rows':len(quarantined),
        'all_eight_classes_available':all(counts[k]>0 for k in LABELS),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--trust-pickle',action='store_true')
    a=p.parse_args()
    if not a.trust_pickle:p.error('Trusted hash-pinned pickle requires explicit opt-in')
    from wafervlm.acquisition import PICKLE_SHA256
    import numpy as np
    import pandas as pd
    source=Path(a.source); data=Path(a.dataset);out=Path(a.output)
    if sha256_file(source)!=PICKLE_SHA256:raise ValueError('Source mismatch')
    expected=json.loads((Path(__file__).resolve().parents[1]/'docs/evidence/wm811k-lot17.json').read_text())
    if sha256_file(data/'dataset.json')!=expected['dataset_sha256']:raise ValueError('Development dataset mismatch')
    development=[r for rows in audit_dataset(data).values() for r in rows]
    records=[];arrays={}
    df=pd.read_pickle(source)
    for index,row in enumerate(df.itertuples(index=False)):
        label=scalar(row.failureType);label=str(label).strip().lower() if label is not None else None
        if label not in LABELS:continue
        lot=scalar(row.lotName)
        if lot is None or pd.isna(lot) or not str(lot).strip():raise ValueError('Missing lot')
        m=np.asarray(row.waferMap)
        if m.ndim!=2 or not m.size or not np.isin(m,[0,1,2]).all():raise ValueError('Invalid map')
        m=m.astype(np.uint8);sha=hashlib.sha256(str(m.shape).encode()+m.tobytes()).hexdigest()
        records.append({'id':hashlib.sha256(f'{PICKLE_SHA256}:{index}'.encode()).hexdigest()[:24],
            'source_row':index,'lot':str(lot),'label':label,'map_sha256':sha})
        arrays.setdefault(sha,m)
    def image_digest(row):
        buf=io.BytesIO();render_map(arrays[row['map_sha256']]).save(buf,format='PNG')
        return hashlib.sha256(buf.getvalue()).hexdigest()
    selected,report=select_holdout(records,development,image_digest)
    out.mkdir(parents=True,exist_ok=False)
    write_jsonl(out/'manifest.jsonl',selected)
    report.update(schema_version=1,source_sha256=PICKLE_SHA256,
        development_dataset_sha256=expected['dataset_sha256'],
        development_manifest_sha256=expected['dataset']['manifest_sha256'],
        manifest_sha256=sha256_file(out/'manifest.jsonl'),n=len(selected),cap_per_class=20,
        rule='exclude every development lot group; quarantine contradictory maps; fixed salted identity order; reject raw/render collisions',
        model_predictions_accessed=False,images_inspected=False,evaluated=False,
        scope='all prior waferVLM runs use this same hash-pinned development dataset; unrelated external use is not established')
    write_json(out/'reservation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
