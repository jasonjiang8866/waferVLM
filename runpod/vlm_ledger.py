"""Reconcile historical artifact bytes before spending the remaining VLM budget."""
import hashlib
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from runpod.provider import NoRedirect, REPOSITORY, request

ROOT = Path(__file__).resolve().parents[1]
BASE = f'https://api.github.com/repos/{REPOSITORY}'
MARKER = 'runpod-authorization-wafer-20261107-usd20'
HISTORY = json.loads((ROOT / 'runpod/vlm-history.json').read_text())
RESULT_ID = 13706115627
RESULT_SHA = '32024461ade6dcd34398e96a78b537c5a274e9da636ebda9a655b6618e32a250'


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


def verify_marker(data, run, current=False):
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
        if intent != {'seed': 42, 'controller_sha': os.environ.get('GITHUB_SHA'),
                      'run_id': str(run), 'workload': 'vlm-seed42'}:
            raise RuntimeError('Current launch intent mismatch')


def reconcile(run, require_current, output):
    """Unknown/missing/expired attempts block launch, including another agent's attempt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = {str(x['run_id']): x for x in HISTORY}
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
                digest = artifact.get('digest', '')
                if not digest.startswith('sha256:') or len(digest) != 71:
                    raise RuntimeError('Current marker digest missing')
                verify_marker(artifact_bytes(artifact['id'], digest[7:], 64_000), run, current=True)
                current = True
                continue
            pin = expected.get(origin)
            if (pin is None or artifact.get('id') != pin['marker_artifact_id']
                    or artifact.get('digest') != 'sha256:' + pin['marker_sha256']):
                raise RuntimeError('Unreconciled attempt; do not buy another GPU')
            verify_marker(artifact_bytes(pin['marker_artifact_id'], pin['marker_sha256'], 64_000), origin)
            attempts.append(dict(run_id=int(origin), marker_artifact_id=pin['marker_artifact_id'],
                                 evidence_sha256=pin['marker_sha256'], cleanup='verified_absent'))
        if len(artifacts) < 100:
            break
    else:
        raise RuntimeError('Incomplete authorization ledger')
    if not set(expected) <= observed or (require_current and not current):
        raise RuntimeError('Required historical/current authorization marker missing')
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
