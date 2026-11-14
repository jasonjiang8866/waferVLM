"""Reconcile historical artifact bytes before spending the remaining VLM budget."""
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from runpod.provider import NoRedirect, REPOSITORY, RemoteHTTPError, request

ROOT = Path(__file__).resolve().parents[1]
BASE = f'https://api.github.com/repos/{REPOSITORY}'
MARKER = 'runpod-authorization-wafer-20261107-usd20'
HISTORY = json.loads((ROOT / 'runpod/vlm-history.json').read_text())
RESULT_ID = 13706115627
RESULT_SHA = '32024461ade6dcd34398e96a78b537c5a274e9da636ebda9a655b6618e32a250'
PAID_STEP = 'Train seed 42, verify full states, retrieve evidence, delete Pod'
PRECREATE = json.loads((ROOT / 'runpod/vlm-precreate.json').read_text())


def artifact_bytes(artifact_id, expected_sha, limit):
    """Exchange the GitHub token for a signed URL; never forward that token."""
    req = urllib.request.Request(f'{BASE}/actions/artifacts/{artifact_id}/zip',
        headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                 'User-Agent': 'wafervlm-actions/0.1'})
    try:
        response = urllib.request.build_opener(NoRedirect).open(req, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code != 302:
            raise RuntimeError('Artifact download failed') from None
        url = exc.headers.get('Location', '')
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname or ''
        if (parsed.scheme != 'https' or parsed.username or parsed.password
                or not host.endswith(('.blob.core.windows.net', '.actions.githubusercontent.com',
                                      '.githubusercontent.com'))):
            raise RuntimeError('Unexpected artifact download destination') from None
        response = urllib.request.urlopen(url, timeout=60)  # anonymous signed request
    with response:
        data = response.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != expected_sha:
        raise RuntimeError('Artifact size or SHA256 mismatch')
    return data


def zip_json(data, name):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        matches = [m for m in archive.infolist() if m.filename == name]
        if len(matches) != 1 or matches[0].file_size > 1_000_000:
            raise RuntimeError('Missing or oversized ledger JSON')
        return json.loads(archive.read(matches[0]))


def verify_marker(data, run, current=False, controller_sha=None):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        expected = ['intent.json', 'run-id.txt'] if current else ['run-id.txt']
        if sorted(names) != expected:
            raise RuntimeError('Unexpected authorization marker files')
        info = archive.getinfo('run-id.txt')
        if info.file_size > 64 or archive.read(info).decode().strip() != str(run):
            raise RuntimeError('Authorization marker run mismatch')
    if current:
        intent = zip_json(data, 'intent.json')
        if intent != {'seed': 42, 'controller_sha': controller_sha or os.environ.get('GITHUB_SHA'),
                      'run_id': str(run), 'workload': 'vlm-seed42'}:
            raise RuntimeError('Current launch intent mismatch')


def reconciled_precreate():
    """Accept only the inspected guard failure, with unchanged immutable evidence."""
    p = PRECREATE
    run = request(f"{BASE}/actions/runs/{p['run_id']}", token=os.environ['GH_TOKEN'])
    if (run.get('head_sha') != p['controller_sha'] or run.get('run_attempt') != 1
            or run.get('status') != 'completed' or run.get('conclusion') != 'failure'):
        raise RuntimeError('Previously reconciled pre-create run changed')
    data = artifact_bytes(p['result_artifact_id'], p['result_sha256'], 64_000)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sorted(z.namelist()) != ['cleanup.json', 'ledger.json', 'preflight.json']:
            raise RuntimeError('Pre-create evidence contains an unexpected resource intent')
    if zip_json(data, 'cleanup.json') != p['cleanup']:
        raise RuntimeError('Pre-create cleanup evidence mismatch')
    independent = artifact_bytes(p['independent_cleanup_artifact_id'], p['independent_cleanup_sha256'], 64_000)
    if zip_json(independent, 'cleanup.json') != p['independent_cleanup']:
        raise RuntimeError('Independent absence evidence mismatch')
    return {k:p[k] for k in ('run_id','marker_artifact_id','marker_sha256','controller_sha')}


def verify_current_artifact(artifact, run):
    digest = artifact.get('digest', '')
    identity = artifact.get('workflow_run', {})
    if (artifact.get('name') != MARKER or artifact.get('expired') is not False
            or str(identity.get('id')) != str(run)
            or identity.get('head_sha') != os.environ.get('GITHUB_SHA')
            or not digest.startswith('sha256:') or len(digest) != 71):
        raise RuntimeError('Current marker identity or digest mismatch')
    verify_marker(artifact_bytes(artifact['id'], digest[7:], 64_000), run, current=True)


def verify_upload_receipt(run):
    """The upload receipt is authoritative even while list APIs lag behind."""
    artifact_id = os.environ.get('AUTHORIZATION_ARTIFACT_ID', '')
    if not artifact_id.isdigit() or int(artifact_id) <= 0:
        raise RuntimeError('Missing exact authorization artifact upload receipt')
    for attempt in range(12):
        try:
            artifact = request(f'{BASE}/actions/artifacts/{artifact_id}', token=os.environ['GH_TOKEN'])
            break
        except RemoteHTTPError as exc:
            if exc.status != 404 or attempt == 11:
                raise
            time.sleep(5)  # GET only; no provisioning retry
    if artifact.get('id') != int(artifact_id):
        raise RuntimeError('Authorization upload receipt ID mismatch')
    verify_current_artifact(artifact, run)


def reconcile(run, require_current, output):
    """Unknown/missing/expired attempts block launch, including another agent's attempt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    precreate = reconciled_precreate()
    # Run history survives deleted/expired artifacts. A previous paid step must
    # be reconciled explicitly; only visibly skipped paid steps are read-only.
    for page in range(1, 101):
        prior = request(f'{BASE}/actions/workflows/vlm-seed42.yml/runs?event=workflow_dispatch&per_page=100&page={page}',
                        token=os.environ['GH_TOKEN'])
        runs = prior.get('workflow_runs')
        if not isinstance(runs, list):
            raise RuntimeError('Cannot verify seed-42 dispatch history')
        for previous in runs:
            if str(previous.get('id')) == str(run):
                continue
            if precreate and previous.get('id') == precreate['run_id']:
                continue  # exact run, attempt, source and archive bytes checked above
            if previous.get('status') != 'completed' or previous.get('run_attempt') != 1:
                raise RuntimeError('Unreconciled seed-42 dispatch')
            jobs = request(f"{BASE}/actions/runs/{previous['id']}/jobs?per_page=100", token=os.environ['GH_TOKEN'])
            paid = [s for j in jobs.get('jobs', []) for s in j.get('steps', []) if s.get('name') == PAID_STEP]
            if len(paid) != 1 or paid[0].get('conclusion') != 'skipped':
                raise RuntimeError('A prior paid seed-42 attempt requires reconciliation')
        if len(runs) < 100:
            break
    else:
        raise RuntimeError('Incomplete seed-42 dispatch history')
    expected = {str(x['run_id']): x for x in HISTORY + ([precreate] if precreate else [])}
    observed, current, attempts = set(), False, []
    for page in range(1, 101):
        data = request(f'{BASE}/actions/artifacts?per_page=100&page={page}', token=os.environ['GH_TOKEN'])
        artifacts = data.get('artifacts')
        if not isinstance(artifacts, list):
            raise RuntimeError('Cannot enumerate authorization ledger')
        for artifact in artifacts:
            if not artifact.get('name', '').startswith('runpod-authorization-'):
                continue
            origin = str(artifact.get('workflow_run', {}).get('id'))
            if artifact['name'] != MARKER or artifact.get('expired'):
                raise RuntimeError('Unreconciled or expired authorization marker')
            if origin in observed:
                raise RuntimeError('Duplicate authorization marker')
            observed.add(origin)
            if origin == str(run):
                verify_current_artifact(artifact, run)
                current = True
                continue
            pin = expected.get(origin)
            if (pin is None or artifact.get('id') != pin['marker_artifact_id']
                    or artifact.get('digest') != 'sha256:' + pin['marker_sha256']):
                raise RuntimeError('Unreconciled attempt; do not buy another GPU')
            verify_marker(artifact_bytes(pin['marker_artifact_id'], pin['marker_sha256'], 64_000), origin,
                          current=bool(pin.get('controller_sha')), controller_sha=pin.get('controller_sha'))
            attempts.append(dict(run_id=int(origin), marker_artifact_id=pin['marker_artifact_id'],
                                 evidence_sha256=pin['marker_sha256'], cleanup='verified_absent'))
        if len(artifacts) < 100:
            break
    else:
        raise RuntimeError('Incomplete authorization ledger')
    if not set(expected) <= observed:
        raise RuntimeError('Required historical authorization marker missing')
    if require_current and not current:
        verify_upload_receipt(run)
    # The existing controller already reconciled the first six attempts. Verify
    # the seventh attempt's cumulative cleanup record against its immutable ZIP.
    cache = output / 'historical-result.zip'
    if not cache.exists():
        cache.write_bytes(artifact_bytes(RESULT_ID, RESULT_SHA, 200_000_000))
    raw = cache.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RESULT_SHA:
        raise RuntimeError('Historical result cache changed')
    cleanup = zip_json(raw, 'cleanup.json')
    known = json.loads((ROOT / 'docs/evidence/continuation-42682266526.json').read_text())['cleanup']
    if (cleanup != known or cleanup.get('status') != 'verified_absent'
            or cleanup['estimated_total_usd'] > 5.903039
            or cleanup['estimated_total_seconds'] > 491.911 * 60):
        raise RuntimeError('Historical cumulative consumption/cleanup mismatch')
    ledger = dict(usd=5.903039, seconds=491.911 * 60, attempts=sorted(attempts, key=lambda a:a['run_id']))
    (output / 'ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    return ledger


def dataset_metadata(output):
    """Recreate only the pinned metadata, without images or holdout access."""
    evidence = json.loads((ROOT / 'docs/evidence/wm811k-lot17.json').read_text())
    data = (json.dumps(evidence['dataset'], indent=2, sort_keys=True) + '\n').encode()
    if hashlib.sha256(data).hexdigest() != evidence['dataset_sha256']:
        raise RuntimeError('Reconstructed dataset metadata does not match pin')
    directory = Path(output) / 'dataset-metadata'
    directory.mkdir(exist_ok=True)
    (directory / 'dataset.json').write_bytes(data)
    return directory
