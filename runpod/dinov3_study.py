"""Frozen DINO training confirmation, fresh-process evaluation and CPU acceptance."""
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from wafervlm.core import LABELS, metrics, parse_label, read_jsonl, sha256_file, write_json, write_jsonl
from wafervlm.dinov3 import check_data, recipe
from wafervlm.confirmation import verify_tree
from wafervlm.vision_comparison import completed_result

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/dinov3-confirmation.json'
FROZEN = ROOT / 'docs/evidence/dinov3-six-endpoints.json'


def plan():
    p = json.loads(PLAN.read_text())
    if (p['recipe_sha256'] != sha256_file(ROOT / 'configs/dinov3.json')
            or p['label_order'] != list(LABELS) or p['seeds'] != [17, 42, 123]
            or p['modes'] != ['linear', 'lora'] or p['epochs'] != 10
            or p['learning_rates'] != dict(linear=.003, lora=.0001)
            or p['evaluation_batch_size'] != 8 or p['validation_repeats'] != 3
            or p['headline_repeat'] != 1 or p['bootstrap_resamples'] != 10000
            or p['bootstrap_seed'] != 20261110 or p['undefined_f1'] != 0):
        raise ValueError('Frozen DINO protocol changed')
    return p


def names():
    return [f'seed{s}-{m}' for s in plan()['seeds'] for m in plan()['modes']]


def input_pin(phase):
    if phase == 'validation':
        return plan()['seed17']
    if phase == 'holdout':
        frozen = json.loads(FROZEN.read_text())
        if frozen['plan_sha256'] != sha256_file(PLAN) or [e['name'] for e in frozen['endpoints']] != names():
            raise ValueError('Six weights must be committed before holdout dispatch')
        return frozen['artifact']
    raise ValueError('Unknown DINO phase')


def unpack_zip(path, destination, pin):
    from runpod.confirmation_verify import unpack
    if Path(path).stat().st_size > 32 * 1024**2 or sha256_file(path) != pin['zip_sha256']:
        raise ValueError('DINO input artifact checksum/size mismatch')
    with tempfile.TemporaryDirectory() as tmp:
        inner = Path(tmp) / 'experiment.tar.gz'
        with zipfile.ZipFile(path) as z:
            entry = z.getinfo('experiment.tar.gz')
            if entry.file_size > 24 * 1024**2:
                raise ValueError('DINO inner archive too large')
            inner.write_bytes(z.read(entry))
        if sha256_file(inner) != pin['inner_sha256']:
            raise ValueError('DINO inner archive checksum mismatch')
        return unpack(inner, destination)


def verify_training(folder, reference, name):
    p, cfg = plan(), recipe()
    info, _, hashes = completed_result(folder, reference)
    run = info['config']
    seed, mode = int(name.split('-')[0][4:]), name.split('-')[1]
    if (name not in names() or any(run.get(k) != v for k, v in cfg.items())
            or run.get('mode') != mode or run.get('seed') != seed
            or run.get('learning_rate') != p['learning_rates'][mode] or run.get('device') != 'cuda'
            or info.get('source_sha') != p['image_source_sha']
            or info.get('image_digest') != p['image'].split('@')[1]
            or info.get('base_weights_sha256') != p['base_weights_sha256']):
        raise ValueError('Training recipe/source/weight identity mismatch')
    training = json.loads((folder / 'training.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    if (training['status'] != 'completed' or len(history) != 10 or training['optimizer_steps'] != 990
            or training['examples_seen'] != 7900 or training['holdout_accessed'] is not False
            or training['test_evaluated'] is not False
            or (mode == 'lora' and training['nonzero_adapter_gradient_steps'] != 990)):
        raise ValueError('Incomplete training exposure or unintended evaluation')
    for epoch, row in enumerate(history, 1):
        if row['epoch'] != epoch or row['optimizer_steps'] != epoch * 99 or row['examples_seen'] != epoch * 790:
            raise ValueError('Training history exposure mismatch')
    best = max(history, key=lambda r: (r['validation']['macro_f1'], -r['epoch']))
    selected = json.loads((folder / 'metrics.json').read_text())
    if (training['selected_epoch'] != best['epoch'] or training['selected_macro_f1'] != best['validation']['macro_f1']
            or selected['macro_f1'] != best['validation']['macro_f1']):
        raise ValueError('Selected epoch violates frozen rule')
    if seed == 17 and hashes['selected.safetensors'] != p['seed17']['weights'][mode]:
        raise ValueError('Published seed-17 selected weights changed')
    return dict(name=name, seed=seed, mode=mode, learning_rate=run['learning_rate'],
                selected_epoch=training['selected_epoch'], weight_sha256=hashes['selected.safetensors'],
                base_weights_sha256=info['base_weights_sha256'], training_source_sha=info['source_sha'],
                training_image=info['image_digest'], training_gpu=info['gpu'],
                training_checksums=hashes)


def endpoints(root, reference):
    return [verify_training(root / 'models' / n, reference, n) for n in names()]


def verify_predictions(rows, reference):
    if len(rows) != len(reference) or len({r['id'] for r in rows}) != len(reference):
        raise ValueError('Missing or duplicate prediction rows')
    for r, ref in zip(rows, reference, strict=True):
        if any(r.get(k) != ref[k] for k in ('id', 'label', 'image_sha256')):
            raise ValueError('Prediction manifest/order mismatch')
        p = r.get('probabilities', [])
        if (len(p) != 8 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                or not math.isclose(sum(p), 1, abs_tol=1e-5)
                or LABELS[max(range(8), key=p.__getitem__)] != r['prediction']
                or parse_label(r['response'], r['truncated']) != r['prediction']):
            raise ValueError('Invalid probabilities/argmax/response')
        if any(not math.isfinite(r[k]) or r[k] < 0 for k in ('seconds', 'forward_seconds')):
            raise ValueError('Invalid timing')
    return metrics(rows)


def reference_for(phase, dataset=None):
    if phase == 'validation':
        if dataset is not None:
            return check_data(dataset)['validation']
        path = ROOT / 'docs/showcase/dinov3-seed17/validation.jsonl'
        if sha256_file(path) != recipe()['validation_sha256']:
            raise ValueError('Validation manifest changed')
        return read_jsonl(path)
    if phase != 'holdout':
        raise ValueError('Unknown phase')
    from runpod.vlm_holdout import manifest
    rows = manifest()
    if len(rows) != 148 or len({r['lot_group'] for r in rows}) != 143:
        raise ValueError('Expected the exact 148 maps / 143 lots')
    return [dict(r, image=r['id'] + '.png') for r in rows]


def evaluate_one(phase, name, dataset, trained, output):
    """Exactly one subprocess inference; no epoch or endpoint selection here."""
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import set_seed
    from wafervlm.dinov3 import load_backbone, make_classifier, restore_weights, predict
    from wafervlm.experiment import seal_output
    p, cfg = plan(), recipe()
    if name not in names() or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Declared endpoint and real CUDA/BF16 required')
    # Validate the selected tensors against their complete training export first.
    ep = verify_training(trained, reference_for('validation'), name)
    if phase == 'holdout' and ep != next(e for e in json.loads(FROZEN.read_text())['endpoints'] if e['name'] == name):
        raise ValueError('Weights differ from committed six-endpoint freeze')
    reference = reference_for(phase, dataset)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(ep['seed'])
    model = make_classifier(load_backbone('cuda', local_files_only=True), ep['mode'], cfg['rank'])
    cached = try_to_load_from_cache(cfg['model_id'], 'model.safetensors', revision=cfg['model_revision'])
    if not isinstance(cached, str) or sha256_file(cached) != p['base_weights_sha256']:
        raise ValueError('Official base weights differ')
    restore_weights(model, trained / 'selected.safetensors')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    remaining = float(os.environ['DINO_DEADLINE']) - time.time() - 90
    if remaining <= 0:
        raise TimeoutError('Inference deadline reached')
    rows, score = predict(model, dataset, reference, 'cuda', batch_size=p['evaluation_batch_size'],
                          deadline=time.monotonic() + remaining)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    verify_predictions(rows, reference)
    write_jsonl(output / 'predictions.jsonl', rows)
    write_json(output / 'metrics.json', score)
    write_json(output / 'run.json', dict(endpoint=name, phase=phase, pid=os.getpid(),
        plan_sha256=sha256_file(PLAN), weight_sha256=ep['weight_sha256'],
        base_weights_sha256=p['base_weights_sha256'], batch_size=8,
        manifest_sha256=cfg['validation_sha256'] if phase == 'validation' else p['holdout_manifest_sha256'],
        image=p['image'], image_source_sha=p['image_source_sha'],
        controller_sha=os.environ['DINO_CONTROLLER_SHA'], gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, elapsed_seconds=elapsed,
        mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
        mean_forward_seconds=sum(r['forward_seconds'] for r in rows)/len(rows),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), precision='bf16_autocast_fp32_parameters',
        timing='synchronized verification/decode/preprocess/transfer/forward; excludes model loading; no warmup'))
    write_json(output / 'status.json', dict(status='completed', rows=len(rows)))
    seal_output(output)


def verify_results(root, phase, spec):
    p = plan()
    validation = reference_for('validation')
    eps = endpoints(root, validation)
    if json.loads((root / 'endpoints.json').read_text()) != eps:
        raise ValueError('Endpoint manifest mismatch')
    if phase == 'holdout' and eps != json.loads(FROZEN.read_text())['endpoints']:
        raise ValueError('Holdout endpoint selection changed')
    reference = reference_for(phase)
    reports, predictions, repeats, pids = {}, {}, {}, set()
    count = 3 if phase == 'validation' else 1
    expected_paths = {root / phase / n / f'repeat{r}' / 'predictions.jsonl'
                      for n in names() for r in range(1, count + 1)}
    if set((root / phase).rglob('predictions.jsonl')) != expected_paths:
        raise ValueError('Missing or unexpected inference passes')
    if phase == 'validation' and (root / 'holdout').exists():
        raise ValueError('Holdout outputs forbidden during validation')
    for ep in eps:
        n = ep['name']; passes = []
        for repeat in range(1, count + 1):
            folder = root / phase / n / f'repeat{repeat}'
            hashes = verify_tree(folder)
            run = json.loads((folder / 'run.json').read_text())
            expected = dict(endpoint=n, phase=phase, plan_sha256=sha256_file(PLAN),
                            weight_sha256=ep['weight_sha256'], base_weights_sha256=p['base_weights_sha256'],
                            batch_size=8, manifest_sha256=recipe()['validation_sha256'] if phase == 'validation' else p['holdout_manifest_sha256'],
                            image=p['image'], image_source_sha=p['image_source_sha'],
                            controller_sha=spec['controller_sha'], gpu=spec['gpu'], precision='bf16_autocast_fp32_parameters')
            if any(run.get(k) != v for k, v in expected.items()) or run['pid'] in pids or run['peak_allocated_bytes'] <= 0:
                raise ValueError('Inference identity/process/GPU mismatch')
            pids.add(run['pid'])
            if json.loads((folder / 'status.json').read_text()) != dict(status='completed', rows=len(reference)):
                raise ValueError('Incomplete inference')
            rows = read_jsonl(folder / 'predictions.jsonl')
            score = verify_predictions(rows, reference)
            stored = json.loads((folder / 'metrics.json').read_text())
            if any(stored[k] != v for k, v in score.items()):
                raise ValueError('Recorded metrics differ from predictions')
            passes.append(dict(repeat=repeat, predictions_sha256=hashes['predictions.jsonl'], metrics=score, run=run))
            if repeat == 1:
                predictions[n], reports[n] = rows, score
            else:
                passes[-1]['label_changes'] = [r['id'] for r, a in zip(rows, predictions[n], strict=True) if r['prediction'] != a['prediction']]
                passes[-1]['label_change_count'] = len(passes[-1]['label_changes'])
        repeats[n] = passes
    pairs = {str(s): {k: reports[f'seed{s}-lora'][k] - reports[f'seed{s}-linear'][k]
                     for k in ('accuracy', 'macro_f1')} for s in p['seeds']}
    report = dict(status='accepted', phase=phase, endpoints=eps, models=reports, passes=repeats,
                  paired_lora_minus_linear=pairs, validation_passes=18 if phase == 'validation' else 0,
                  holdout_predictions=888 if phase == 'holdout' else 0,
                  primary_rule='repeat1 for every fixed model; no best-repeat or best-seed selection')
    if phase == 'holdout':
        from runpod.vlm_holdout import bootstrap
        report['lot_group_percentile_95'] = {str(s): bootstrap(reference, predictions[f'seed{s}-linear'], predictions[f'seed{s}-lora']) for s in p['seeds']}
        qwen_path = ROOT / 'docs/evidence/qwen-holdout/seed17-step250/predictions.jsonl'
        if sha256_file(qwen_path) != p['qwen_holdout_predictions_sha256']:
            raise ValueError('Published Qwen holdout predictions changed')
        qwen = read_jsonl(qwen_path)
        if len(qwen) != 148 or any(any(q[k] != r[k] for k in ('id','label','image_sha256')) for q,r in zip(qwen,reference,strict=True)):
            raise ValueError('Published Qwen holdout identity mismatch')
        report['qwen_published_sha256'] = sha256_file(ROOT / 'docs/evidence/qwen-holdout/seed17-step250/predictions.jsonl')
        report['seed17_lora_minus_qwen'] = dict(delta={k:reports['seed17-lora'][k]-metrics(qwen)[k] for k in ('accuracy','macro_f1')},
            percentile_95=bootstrap(reference,qwen,predictions['seed17-lora']))
    return report


if __name__ == '__main__':
    command, *args = sys.argv[1:]
    if command == 'one':
        evaluate_one(args[0], args[1], *map(Path, args[2:]))
    else:
        raise SystemExit('Unknown DINO study operation')
