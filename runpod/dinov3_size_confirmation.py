"""CPU comparisons against the original selected-epoch Small training exports."""
import json
import os
import tempfile
from pathlib import Path

from runpod.dinov3_size17 import GPU, paired
from runpod.dinov3_study import FROZEN, ROOT, reference_for, unpack_zip, verify_predictions
from wafervlm.confirmation import verify_tree
from wafervlm.core import read_jsonl, sha256_file

PILOT = ROOT / 'docs/showcase/dinov3-size17/acceptance.json'
PILOT_SHA256 = 'cf594b6854e6ae105120b6ef1c4c067a086a1581c179f5f4251ad312c9504ea0'


def archive_path():
    return Path(os.environ['RUNNER_TEMP']) / 'dino-small-training.zip'


def pilot():
    if sha256_file(PILOT) != PILOT_SHA256:
        raise ValueError('Seed-17 pilot report changed')
    return json.loads(PILOT.read_text())


def small_models(archive, reference):
    frozen = json.loads(FROZEN.read_text())
    models, labels = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        root = unpack_zip(archive, Path(tmp)/'exports', frozen['artifact'])
        for ep in frozen['endpoints']:
            if ep['seed'] not in (42, 123):
                continue
            name = f"seed{ep['seed']}-small-{ep['mode']}"
            folder = root/'models'/ep['name']
            hashes = verify_tree(folder)
            info = json.loads((folder/'run.json').read_text())
            if (hashes != ep['training_checksums'] or hashes['selected.safetensors'] != ep['weight_sha256']
                    or info['gpu'] != GPU or info['config']['seed'] != ep['seed']
                    or info['config']['mode'] != ep['mode']):
                raise ValueError('Small training export differs from frozen endpoint: '+name)
            rows = read_jsonl(folder/'predictions.jsonl')
            score = verify_predictions(rows, reference)
            models[name] = dict(**score, selected_epoch=ep['selected_epoch'], selected_sha256=ep['weight_sha256'],
                                predictions_sha256=hashes['predictions.jsonl'])
            labels[name] = [r['prediction'] for r in rows]
    expected = {f'seed{s}-small-{m}' for s in (42, 123) for m in ('linear', 'lora')}
    if set(models) != expected:
        raise ValueError('Missing frozen Small comparator')
    return models, labels


def stage_baselines():
    from runpod.vlm_ledger import artifact_bytes
    pin = json.loads(FROZEN.read_text())['artifact']
    path = archive_path()
    if not path.exists():
        path.write_bytes(artifact_bytes(pin['artifact_id'], pin['zip_sha256'], 32 * 1024**2))
    small_models(path, reference_for('validation'))
    pilot()
    return dict(artifact=pin, endpoints_sha256=sha256_file(FROZEN), pilot_sha256=PILOT_SHA256)


def confirmation_report(reference, models, predictions, archive):
    previous, previous_labels = small_models(archive, reference)
    scores = {**previous, **models}
    labels = {**previous_labels, **predictions}
    comparisons = {'17': pilot()['comparisons']}
    for seed in (42, 123):
        comparisons[str(seed)] = {
            mode: paired(reference, scores, labels, f'seed{seed}-small-{mode}', f'seed{seed}-base-{mode}')
            for mode in ('linear', 'lora')
        }
    return dict(status='accepted', primary='base-lora minus small-lora', models=models,
                reused_small=previous, comparisons=comparisons, seed17_report_sha256=PILOT_SHA256,
                positive_macro_f1_seeds={m: [int(s) for s, pairs in comparisons.items()
                                             if pairs[m]['macro_f1_delta'] > 0] for m in ('linear', 'lora')},
                validation_predictions=640, reused_small_predictions=640, holdout_predictions=0)
