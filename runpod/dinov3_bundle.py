"""CPU-only credential boundary and portable acceptance for the DINO sweep."""
import json
import math
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {"config.json": 1_000_000, "model.safetensors": 256 * 1024**2}


def recipe():
    return json.loads((ROOT / 'configs/dinov3.json').read_text())


def trials():
    cfg = recipe()
    return [dict(name=f'{mode}-{i}', mode=mode, seed=17, learning_rate=rate)
            for mode in cfg['modes'] for i, rate in enumerate(cfg[f'{mode}_learning_rates'])]


def weight_report(path):
    report = json.loads(Path(path).read_text())
    cfg = recipe()
    import re
    if (report.get('status') != 'weights-access-and-load-verified'
            or report.get('model_id') != cfg['model_id']
            or report.get('model_revision') != cfg['model_revision']
            or set(report.get('files_sha256', {})) != set(FILES)
            or any(not re.fullmatch('[a-f0-9]{64}', v) for v in report['files_sha256'].values())):
        raise ValueError('Pinned official CPU weight preflight required')
    return report


def cache_files(report):
    from huggingface_hub import try_to_load_from_cache
    from wafervlm.core import sha256_file
    cfg = recipe()
    files = {}
    for name, limit in FILES.items():
        path = try_to_load_from_cache(cfg['model_id'], name, revision=cfg['model_revision'])
        if (not isinstance(path, str) or not 0 < Path(path).stat().st_size <= limit
                or sha256_file(path) != report['files_sha256'][name]):
            raise ValueError('Cached weights differ from CPU preflight')
        files[name] = Path(path)
    return files


def select(rows):
    """All six declared development trials; ties use lower LR, never a best seed."""
    expected = trials()
    if len(rows) != len(expected):
        raise ValueError('Incomplete sweep')
    for row, trial in zip(rows, expected):
        if (any(row.get(k) != v for k, v in trial.items())
                or not isinstance(row.get('macro_f1'), (int, float))
                or not math.isfinite(row['macro_f1']) or not 0 <= row['macro_f1'] <= 1):
            raise ValueError('Sweep identity or score mismatch')
    return {mode: max((r for r in rows if r['mode'] == mode),
                     key=lambda r: (r['macro_f1'], -r['learning_rate']))
            for mode in recipe()['modes']}


def verify_archive(archive, source, image, weights):
    """Recompute scores/checksums after relocation; never load tensor objects."""
    from runpod.confirmation_verify import unpack
    from wafervlm.core import read_jsonl, sha256_file
    from wafervlm.vision_comparison import completed_result
    cfg = recipe()
    with tempfile.TemporaryDirectory() as tmp:
        root = unpack(archive, Path(tmp) / 'unpacked')
        if sha256_file(root / 'validation.jsonl') != cfg['validation_sha256']:
            raise ValueError('Validation identity mismatch')
        if sha256_file(root / 'dataset.json') != cfg['dataset_sha256']:
            raise ValueError('Dataset identity mismatch')
        if json.loads((root / 'acquisition.json').read_text()) != json.loads(
                (ROOT / 'docs/evidence/wm811k-lot17.json').read_text()):
            raise ValueError('Acquisition mismatch')
        reference = read_jsonl(root / 'validation.jsonl')
        worker = json.loads((root / 'worker.json').read_text())
        if (worker.get('status') != 'accepted' or worker.get('source_sha') != source
                or worker.get('image') != image or worker.get('weights') != weights):
            raise ValueError('Worker provenance mismatch')
        hardware = json.loads((root / 'hardware.json').read_text())
        if not hardware.get('gpu') or not hardware.get('bf16'):
            raise ValueError('GPU evidence missing')
        rows = []
        for trial in trials():
            folder = root / trial['name']
            info, _, _ = completed_result(folder, reference)
            run = info['config']
            if (any(run.get(k) != v for k, v in cfg.items())
                    or any(run.get(k) != trial[k] for k in ('mode', 'seed', 'learning_rate'))
                    or run.get('device') != 'cuda' or info.get('source_sha') != source
                    or info.get('image_digest') != image.split('@')[1]
                    or info.get('base_weights_sha256') != weights['files_sha256']['model.safetensors']
                    or info.get('gpu') != hardware['gpu']):
                raise ValueError('Training recipe/provenance mismatch')
            history = json.loads((folder / 'history.json').read_text())
            training = json.loads((folder / 'training.json').read_text())
            if (len(history) != cfg['epochs'] or training.get('status') != 'completed'
                    or training.get('test_evaluated') is not False
                    or training.get('holdout_accessed') is not False
                    or training.get('examples_seen') != 7900 or training.get('optimizer_steps') != 990
                    or (trial['mode'] == 'lora' and not training.get('nonzero_adapter_gradient_steps'))):
                raise ValueError('Incomplete training or holdout access')
            for n, epoch in enumerate(history, 1):
                if (epoch.get('epoch') != n or epoch.get('examples_seen') != n * 790
                        or epoch.get('optimizer_steps') != n * 99
                        or not math.isfinite(epoch['validation']['macro_f1'])):
                    raise ValueError('Training exposure/history mismatch')
            best = max(history, key=lambda h: (h['validation']['macro_f1'], -h['epoch']))
            score = json.loads((folder / 'metrics.json').read_text())['macro_f1']
            if (best['epoch'] != training['selected_epoch']
                    or best['validation']['macro_f1'] != score or training['selected_macro_f1'] != score):
                raise ValueError('Selected epoch mismatch')
            rows.append({**trial, 'macro_f1': score})
        selected = select(rows)
        if json.loads((root / 'selection.json').read_text()) != selected:
            raise ValueError('Learning-rate selection mismatch')
        for mode, chosen in selected.items():
            original, _, hashes = completed_result(root / chosen['name'], reference)
            reloaded, _, _ = completed_result(root / (mode + '-reload'), reference)
            if (reloaded.get('selected_weights_sha256') != hashes['selected.safetensors']
                    or reloaded.get('source_sha') != source
                    or reloaded.get('base_weights_sha256') != original['base_weights_sha256']
                    or reloaded.get('image_digest') != image.split('@')[1]
                    or reloaded['config'] != {**original['config'], 'evaluation_batch_size': 1}):
                raise ValueError('Selected-model reload mismatch')
        return {'status': 'accepted', 'quality_accepted': False, 'stage': 'seed17-sweep',
                'selection': selected, 'archive_sha256': sha256_file(archive),
                'test_evaluated': False, 'holdout_accessed': False}
