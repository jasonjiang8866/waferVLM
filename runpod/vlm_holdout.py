"""One-shot unseen-lot inference and paired analysis for the frozen Qwen endpoints."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from wafervlm.core import LABELS, PROMPT, checked_image, metrics, parse_label, read_jsonl, sha256_file, write_json, write_jsonl
from wafervlm.modeling import MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/evidence/final-holdout-v1/manifest.jsonl'
DIGEST = 'f08050ef4fcaca3b9c35ad29905a63957a375af7a024b69e0281968d06f4564f'
FROZEN = ROOT / 'docs/evidence/vlm-six-endpoints.json'
SEEDS = (17, 42, 123)
MODELS = ['original'] + [f'seed{s}-step{t}' for s in SEEDS for t in (200, 250)]


def manifest():
    if sha256_file(MANIFEST) != DIGEST:
        raise ValueError('Reserved manifest changed')
    return read_jsonl(MANIFEST)


def render(cache, dataset):
    from wafervlm.acquisition import acquire
    from wafervlm.data import render_map, scalar
    import pandas as pd
    frame = pd.read_pickle(acquire(cache))
    dataset.mkdir(parents=True, exist_ok=False)
    for row in manifest():
        raw = frame.iloc[row['source_row']]
        arr = np.asarray(raw.waferMap).astype(np.uint8)
        if (hashlib.sha256(str(arr.shape).encode()+arr.tobytes()).hexdigest() != row['map_sha256']
                or str(scalar(raw.lotName)) != row['lot'] or str(scalar(raw.failureType)).lower() != row['label']):
            raise ValueError('Reserved source row mismatch')
        image = dataset / (row['id'] + '.png')
        render_map(arr).save(image)
        if sha256_file(image) != row['image_sha256']:
            raise ValueError('Reserved image mismatch')


def evaluate_one(name, dataset, adapters, output):
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import GenerationConfig, set_seed
    from wafervlm.modeling import load_model, encode
    if name not in MODELS or not torch.cuda.is_bf16_supported():
        raise ValueError('Expected a declared endpoint and BF16 CUDA')
    output.mkdir(exist_ok=False)
    set_seed(17)
    model, processor = load_model(MODEL_REVISION, 'cuda')
    digest = None
    if name != 'original':
        endpoints = json.loads(FROZEN.read_text())
        ep = next(e for e in endpoints if name == f"seed{e['seed']}-step{e['step']}")
        adapter = adapters / f"weights-{ep['seed']}-{ep['step']}" / 'adapter'
        for file in ('adapter_model.safetensors', 'adapter_config.json'):
            if sha256_file(adapter / file) != ep['files'][file]:
                raise ValueError('Frozen adapter mismatch')
        digest = ep['files']['adapter_model.safetensors']
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    eos = processor.tokenizer.eos_token_id
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=16,
        eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id or eos, use_cache=True)
    predictions = []
    torch.cuda.reset_peak_memory_stats()
    for row in manifest():
        with Image.open(checked_image(dataset, dict(row, image=row['id']+'.png'))) as image:
            inputs = encode(processor, image.convert('RGB')).to('cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            response = model.generate(**inputs, generation_config=generation)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        tokens = response[0, inputs['input_ids'].shape[-1]:].tolist()
        truncated = len(tokens) >= 16 and eos not in tokens
        answer = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        predictions.append(dict(id=row['id'], label=row['label'], image_sha256=row['image_sha256'],
            prediction=parse_label(answer, truncated), response=answer, truncated=truncated, seconds=seconds))
        write_jsonl(output/'predictions.jsonl', predictions)
    write_json(output/'run.json', dict(endpoint=name, model=MODEL_ID, revision=MODEL_REVISION,
        adapter_sha256=digest, manifest_sha256=DIGEST, gpu=torch.cuda.get_device_name(),
        image=os.environ['WAFERVLM_IMAGE_DIGEST'], controller=os.environ['WAFER_CONTROLLER_SHA'],
        prompt=PROMPT, peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def bootstrap(reference, first, second):
    # Group multiplicities are shared by both endpoints in every resample.
    groups, group_idx = np.unique([r['lot_group'] for r in reference], return_inverse=True)
    rng = np.random.default_rng(20261110)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=10000)
    def scores(rows):
        counts = np.zeros((len(groups), len(LABELS), len(LABELS)+1), dtype=np.int64)
        for group, row in zip(group_idx, rows, strict=True):
            pred = LABELS.index(row['prediction']) if row['prediction'] is not None else len(LABELS)
            counts[group, LABELS.index(row['label']), pred] += 1
        cm = (weights @ counts.reshape(len(groups), -1)).reshape(-1,len(LABELS),len(LABELS)+1)
        tp = cm.diagonal(axis1=1,axis2=2)
        denom = cm.sum(axis=2) + cm.sum(axis=1)[:, :len(LABELS)]
        f1 = np.divide(2*tp, denom, out=np.zeros_like(tp,dtype=float), where=denom!=0)
        return np.column_stack((tp.sum(axis=1)/cm.sum(axis=(1,2)), f1.mean(axis=1)))
    delta = scores(second)-scores(first)
    intervals = np.percentile(delta, [2.5,97.5], axis=0)
    return dict(zip(('accuracy','macro_f1'), intervals.T.tolist()))


def analyze(root, gpu, image, controller):
    reference = manifest()
    frozen = {f"seed{e['seed']}-step{e['step']}": e['files']['adapter_model.safetensors']
              for e in json.loads(FROZEN.read_text())}
    if {p.name for p in root.iterdir() if p.is_dir()} != set(MODELS):
        raise ValueError('Expected exactly seven model directories')
    predictions, reports = {}, {}
    for name in MODELS:
        folder = root/name
        rows = read_jsonl(folder/'predictions.jsonl')
        run = json.loads((folder/'run.json').read_text())
        expected = dict(endpoint=name, model=MODEL_ID, revision=MODEL_REVISION, prompt=PROMPT,
            manifest_sha256=DIGEST, gpu=gpu, image=image, controller=controller, adapter_sha256=frozen.get(name))
        if any(run.get(k)!=v for k,v in expected.items()) or run['peak_allocated_bytes']<=0:
            raise ValueError('Holdout model identity mismatch')
        for row, ref in zip(rows,reference,strict=True):
            if (any(row[k]!=ref[k] for k in ('id','label','image_sha256'))
                    or row['prediction']!=parse_label(row['response'],row['truncated'])
                    or type(row['truncated']) is not bool or not isinstance(row['response'], str)
                    or not math.isfinite(row['seconds']) or row['seconds']<0):
                raise ValueError('Invalid prediction or mismatched holdout identity')
        predictions[name]=rows
        reports[name]=dict(metrics(rows), mean_generation_seconds=np.mean([r['seconds'] for r in rows]),
                           peak_allocated_bytes=run['peak_allocated_bytes'])
    pairs={}
    for seed in SEEDS:
        early,late = (f'seed{seed}-step{step}' for step in (200,250))
        pairs[str(seed)] = dict(delta={k:reports[late][k]-reports[early][k] for k in ('accuracy','macro_f1')},
                                percentile_95=bootstrap(reference,predictions[early],predictions[late]))
    from statistics import mean, stdev
    confirmation = {key: [pairs[str(seed)]['delta'][key] for seed in (42,123)]
                    for key in ('accuracy','macro_f1')}
    dispersion = {key: dict(mean=mean(values), sample_sd=stdev(values))
                  for key,values in confirmation.items()}
    return dict(confirmation_seed_dispersion=dispersion, models=reports, paired_step250_minus_200=pairs, bootstrap_seed=20261110,
                bootstrap_resamples=10000, rows=7*len(reference), manifest_sha256=DIGEST)


def study(dataset, adapters, output):
    from wafervlm.confirmation import run_process
    output.mkdir(parents=True, exist_ok=False)
    deadline=time.monotonic()+float(os.environ['WAFER_DEADLINE'])-time.time()-100
    for name in MODELS:
        run_process([sys.executable,'-m','runpod.vlm_holdout','one',name,str(dataset),str(adapters),
                     str(output/name)],deadline,output/(name+'.log'))


if __name__=='__main__':
    mode,*args=sys.argv[1:]
    if mode=='one':
        evaluate_one(args[0],*map(Path,args[1:]))
    elif mode=='study':
        study(*map(Path,args))
    elif mode=='render':
        render(*map(Path,args))
    else:
        raise SystemExit('Unknown holdout operation')
