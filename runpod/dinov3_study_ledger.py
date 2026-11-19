"""Reconcile the DINO-only cumulative allowance and every previous paid attempt."""
import json
import io
import zipfile
import os
import tempfile
from pathlib import Path

from runpod.provider import REPOSITORY, request
from runpod.dinov3_scheduling import KNOWN_PRIOR_RUNS, reconcile_prior_records
from runpod.vlm_ledger import artifact_bytes, zip_json
from runpod.dinov3_study import plan, unpack_zip, verify_training, reference_for

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'runpod/dinov3-study-ledger.json'
BASE = f'https://api.github.com/repos/{REPOSITORY}'


def check_history(data, expected):
    rows = data.get('workflow_runs', [])
    if data.get('total_count') != len(expected) or len(rows) != len(expected) or {str(r['id']) for r in rows} != set(expected):
        raise ValueError('Unknown DINO dispatch; reconcile before allocating')


def reconcile(current=None):
    token = os.environ['GH_TOKEN']
    receipt = json.loads((ROOT / 'docs/evidence/dinov3-seed17-43716864164.json').read_text())
    ledger = json.loads(LEDGER.read_text())
    if ledger['approved_usd'] != 10 or ledger['approved_seconds'] != 30000:
        raise ValueError('DINO authorization cap changed')
    history = request(BASE + '/actions/workflows/dinov3-gpu.yml/runs?event=workflow_dispatch&per_page=100', token=token)
    check_history(history, {*KNOWN_PRIOR_RUNS, str(receipt['run_id'])})
    reconcile_prior_records(token)
    run = request(BASE + '/actions/runs/' + str(receipt['run_id']), token=token)
    if any(run.get(k) != v for k, v in dict(head_sha=receipt['source_sha'], status='completed', conclusion='success', run_attempt=1).items()):
        raise ValueError('Seed-17 completion identity changed')
    p = plan()['seed17']
    raw = artifact_bytes(p['artifact_id'], p['zip_sha256'], 32 * 1024**2)
    if zip_json(raw, 'cleanup.json') != receipt['cleanup'] or zip_json(raw, 'acceptance.json') != receipt['acceptance']:
        raise ValueError('Seed-17 acceptance/cleanup changed')
    independent = receipt['cleanup_artifact']
    if zip_json(artifact_bytes(independent['id'], independent['zip_sha256'], 64000), 'cleanup.json') != receipt['independent_cleanup']:
        raise ValueError('Seed-17 independent cleanup changed')
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'seed17.zip'; archive.write_bytes(raw)
        root = unpack_zip(archive, Path(tmp) / 'unpacked', p)
        for mode in ('linear', 'lora'):
            verify_training(root / (mode + '-1'), reference_for('validation'), 'seed17-' + mode)
    usd, seconds = receipt['cleanup']['cumulative_estimated_usd'], receipt['cleanup']['cumulative_runtime_bound_s']
    # Each later attempt must be committed with both immutable cleanup receipts.
    ids = [str(r['run_id']) for r in ledger['attempts']]
    if len(ids) != len(set(ids)) or current in ids:
        raise ValueError('Duplicate/reused DINO dispatch')
    history = request(BASE + '/actions/workflows/dinov3-study.yml/runs?event=workflow_dispatch&per_page=100', token=token)
    check_history(history, set(ids) | ({current} if current else set()))
    for prior in ledger['attempts']:
        run = request(BASE + '/actions/runs/' + str(prior['run_id']), token=token)
        if any(run.get(k) != v for k, v in dict(status='completed', run_attempt=1, head_sha=prior['controller_sha'], conclusion=prior['conclusion']).items()):
            raise ValueError('Prior study run changed')
        data = artifact_bytes(prior['artifact_id'], prior['zip_sha256'], 32 * 1024**2)
        cleanup = zip_json(data, 'cleanup.json')
        ind = prior['independent']
        absent = zip_json(artifact_bytes(ind['artifact_id'], ind['zip_sha256'], 64000), 'cleanup.json')
        if (cleanup != prior['cleanup'] or absent != ind['cleanup']
                or cleanup['status'] != 'verified_absent' or absent['status'] != 'verified_absent'
                or cleanup.get('within_cap', True) is not True):
            raise ValueError('Prior DINO cleanup not verified')
        if 'provision.json' in zipfile.ZipFile(io.BytesIO(data)).namelist():
            spec = zip_json(data, 'launch-spec.json')
            if spec['ledger']['usd'] != usd or spec['ledger']['seconds'] != seconds:
                raise ValueError('DINO accounting chain broken')
            usd += cleanup['estimated_usd']; seconds += cleanup['elapsed_billing_bound_s']
        elif cleanup.get('estimated_usd', 0) or cleanup.get('elapsed_billing_bound_s', 0):
            raise ValueError('Cost without allocation intent')
    if usd >= 10 or seconds >= 30000:
        raise ValueError('DINO allowance exhausted')
    return dict(usd=usd, seconds=seconds, remaining_usd=10-usd, remaining_seconds=30000-seconds,
                basis='intent-through-verified-absence estimate; not provider invoice', attempts=ids)
