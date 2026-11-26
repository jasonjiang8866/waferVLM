"""Small/Base size studies, with CPU acceptance on the frozen development split."""
import json
import os
import shutil
import sys
from pathlib import Path

from wafervlm.core import read_jsonl, sha256_file
from wafervlm.dinov3 import recipe
from wafervlm.dinov3_backbones import BACKBONES
from wafervlm.confirmation import verify_tree
from runpod.dinov3_study import plan, reference_for, verify_predictions

GPU = 'NVIDIA GeForce RTX 4090'
FILES = ('config.json', 'model.safetensors')
ARMS = [(size, mode) for size in ('small', 'base') for mode in ('linear', 'lora')]


def training_arms(phase):
    if phase == 'size17':
        return [(17, size, mode, f'{size}-{mode}') for size, mode in ARMS]
    if phase == 'size-confirmation':
        return [(seed, 'base', mode, f'seed{seed}-base-{mode}')
                for seed in (42, 123) for mode in ('linear', 'lora')]
    raise ValueError('Unknown size study phase')


def base_files(download=False):
    spec = BACKBONES['base']
    root = Path(os.environ['RUNNER_TEMP']) / 'wafervlm-dino-base'
    paths = {name: root / name for name in FILES}
    if download:
        from huggingface_hub import hf_hub_download
        root.mkdir(exist_ok=True)
        for name, path in paths.items():
            source = hf_hub_download(spec.model_id, name, revision=spec.revision)
            shutil.copyfile(source, path)
    for name, path in paths.items():
        spec.verify_file(name, path)
    return paths


def train_args(size, mode, dataset, output, seed=17):
    rate = plan()['learning_rates'][mode]
    return ['scripts/run_dinov3.py', 'train', '--seed', str(seed), '--mode', mode,
            '--backbone-size', size, '--learning-rate', str(rate),
            '--dataset', str(dataset), '--output', str(output)]


def verify(root, launch, small_archive=None):
    root = Path(root)
    reference = reference_for('validation')
    if len(reference) != 160 or launch['gpu'] != GPU or launch['phase'] not in ('size17', 'size-confirmation'):
        raise ValueError('Size pilot requires the frozen validation split and RTX 4090')
    hardware = json.loads((root / 'hardware.json').read_text())
    if hardware.get('gpu') != GPU or hardware.get('bf16_matmul_passed') is not True:
        raise ValueError('Missing 4090 BF16 execution')
    if sha256_file(root/'validation.jsonl') != recipe()['validation_sha256']:
        raise ValueError('Worker validation manifest changed')
    models, predictions = {}, {}
    arms = training_arms(launch['phase'])
    if {p.name for p in (root/'models').iterdir()} != {arm[3] for arm in arms} or (root/'holdout').exists():
        raise ValueError('Unexpected size study outputs')
    for seed, size, mode, name in arms:
        folder = root / 'models' / name
        hashes = verify_tree(folder)
        def read(file):
            return json.loads((folder / (file + '.json')).read_text())
        backbone, cfg = BACKBONES[size], recipe(size)
        info, summary, history = read('run'), read('training'), read('history')
        expected = {**cfg, 'seed':seed, 'mode':mode, 'device':'cuda', 'learning_rate':plan()['learning_rates'][mode]}
        if (any(info['config'].get(k) != v for k,v in expected.items())
                or info['model_id'] != backbone.model_id or info['model_revision'] != backbone.revision
                or info['base_weights_sha256'] != backbone.weights_sha256 or info['source_sha'] != launch['controller_sha']
                or info['image_digest'] != launch['image'].split('@')[1] or info['gpu'] != GPU
                or any(info[k] != cfg[k] for k in ('dataset_sha256','validation_sha256'))
                or read('status')['status'] != 'completed' or summary['status'] != 'completed'
                or summary['examples_seen'] != 7900 or summary['optimizer_steps'] != 990
                or summary['test_evaluated'] is not False or summary['holdout_accessed'] is not False
                or summary['nonzero_adapter_gradient_steps'] != (990 if mode=='lora' else 0)
                or len(history) != 10 or any((r['epoch'],r['optimizer_steps'],r['examples_seen']) !=
                    (i,99*i,790*i) for i,r in enumerate(history,1))):
            raise ValueError('Incomplete or mismatched training export: ' + name)
        rows = read_jsonl(folder/'predictions.jsonl')
        score = verify_predictions(rows,reference)
        best = max(history, key=lambda row: row['validation']['macro_f1'])
        recorded = read('metrics')
        if (best['epoch'] != summary['selected_epoch'] or best['validation']['macro_f1'] != score['macro_f1']
                or summary['selected_macro_f1'] != score['macro_f1'] or any(recorded[k]!=v for k,v in score.items())):
            raise ValueError('Validation selection or scores changed: ' + name)
        predictions[name] = [r['prediction'] for r in rows]
        models[name] = dict(**score, selected_epoch=best['epoch'], selected_sha256=hashes['selected.safetensors'],
            parameters=info['parameter_counts'], train_seconds=summary['elapsed_seconds'],
            peak_allocated_bytes=summary['peak_allocated_bytes'], images_per_second=1/recorded['mean_end_to_end_seconds'])
    if launch['phase'] == 'size-confirmation':
        from runpod.dinov3_size_confirmation import confirmation_report
        return confirmation_report(reference, models, predictions, small_archive)
    comparisons = {mode: paired(reference, models, predictions, 'small-'+mode, 'base-'+mode)
                   for mode in ('linear', 'lora')}
    return dict(status='accepted', primary='base-lora minus small-lora', models=models, comparisons=comparisons,
                validation_predictions=640, holdout_predictions=0)


def paired(reference, models, predictions, small, base):
    counts = dict(both_correct=0, base_only=0, small_only=0, both_wrong=0)
    loc = {k:[] for k in counts}
    for row, s, b in zip(reference,predictions[small],predictions[base],strict=True):
        key = {(True,True):'both_correct',(False,True):'base_only',(True,False):'small_only',
               (False,False):'both_wrong'}[s==row['label'],b==row['label']]
        counts[key] += 1
        if row['label']=='loc':
            loc[key].append(row['id'])
    return dict(counts=counts,loc_ids=loc,
        macro_f1_delta=models[base]['macro_f1']-models[small]['macro_f1'],
        accuracy_delta=models[base]['accuracy']-models[small]['accuracy'])



if __name__ == '__main__' and sys.argv[1:] == ['stage']:
    base_files(download=True)
