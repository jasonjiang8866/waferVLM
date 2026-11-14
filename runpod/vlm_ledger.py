"""Reconcile historical artifact bytes before spending the remaining VLM budget."""
import hashlib
import io
import json
import os
import shutil
import tempfile
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
COMPLETED = json.loads((ROOT / 'runpod/vlm-completed.json').read_text())
DINO_MARKER = 'runpod-authorization-wafervlm-dinov3-seed17-v1'


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


def verify_marker(data, run, current=False, controller_sha=None, seed=42):
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
        if intent != {'seed': seed, 'controller_sha': controller_sha or os.environ.get('GITHUB_SHA'),
                      'run_id': str(run), 'workload': f'vlm-seed{seed}'}:
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


def verify_current_artifact(artifact, run, seed=42):
    digest = artifact.get('digest', '')
    identity = artifact.get('workflow_run', {})
    if (artifact.get('name') != MARKER or artifact.get('expired') is not False
            or str(identity.get('id')) != str(run)
            or identity.get('head_sha') != os.environ.get('GITHUB_SHA')
            or not digest.startswith('sha256:') or len(digest) != 71):
        raise RuntimeError('Current marker identity or digest mismatch')
    verify_marker(artifact_bytes(artifact['id'], digest[7:], 64_000), run, current=True, seed=seed)


def verify_upload_receipt(run, seed=42):
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
    verify_current_artifact(artifact, run, seed=seed)


def completed_receipts(output):
    """Only reviewed immutable success/cleanup receipts can advance the ledger.

    Pins are added after downloading and independently verifying the real training
    archive. An empty list deliberately leaves seed 123 blocked. This function
    does not infer success or silently bless a new run from its GitHub status.
    """
    from runpod.provider import finite
    prior_usd, prior_seconds = 5.903039, 491.911 * 60
    verified = []
    for p in COMPLETED:
        seed = p['seed']
        if seed not in {42, 123} or any(x['seed'] == seed or x['run_id'] == p['run_id'] for x in verified):
            raise RuntimeError('Duplicate or unsupported completed seed receipt')
        if seed == 123 and not any(x['seed'] == 42 for x in verified):
            raise RuntimeError('Completed seed receipts are out of order')
        run = request(f"{BASE}/actions/runs/{p['run_id']}", token=os.environ['GH_TOKEN'])
        if (run.get('head_sha') != p['controller_sha'] or run.get('run_attempt') != 1
                or run.get('status') != 'completed' or run.get('conclusion') != 'success'
                or run.get('event') != 'workflow_dispatch' or run.get('head_branch') != 'main'
                or run.get('path') != f'.github/workflows/vlm-seed{seed}.yml'):
            raise RuntimeError('Completed training run identity or outcome changed')
        cache = Path(output) / f"completed-{p['run_id']}.zip"
        if not cache.exists():
            cache.write_bytes(artifact_bytes(p['result_artifact_id'], p['result_sha256'], 2_200_000_000))
        raw = cache.read_bytes()
        if hashlib.sha256(raw).hexdigest() != p['result_sha256']:
            raise RuntimeError('Completed result cache changed')
        cleanup = zip_json(raw, 'cleanup.json')
        accepted = zip_json(raw, 'acceptance.json')
        spec = zip_json(raw, 'launch-spec.json')
        if (cleanup != p['cleanup'] or accepted != p['acceptance']
                or cleanup.get('status') != 'verified_absent'
                or cleanup.get('name') != f"wafer-20261107-usd20-vlm{seed}-{p['run_id']}"
                or cleanup.get('creation_confirmed') is not True
                or cleanup.get('consecutive_absent_checks', 0) < 3
                or cleanup.get('within_budget') is not True or cleanup.get('within_runtime') is not True
                or cleanup.get('persistent_volumes_created') is not False
                or accepted.get('status') != 'accepted' or accepted.get('seed') != seed
                or accepted.get('quality_accepted') is not False or accepted.get('holdout_accessed') is not False
                or [e.get('step') for e in accepted.get('endpoints', [])] != [200, 250]
                or spec.get('seed') != seed or spec.get('controller_sha') != p['controller_sha']
                or spec.get('ledger', {}).get('usd') != prior_usd
                or spec.get('ledger', {}).get('seconds') != prior_seconds):
            raise RuntimeError('Completed receipt acceptance, lineage or cleanup mismatch')
        verify_completed_archive(raw, output, spec, accepted)
        independent = artifact_bytes(p['independent_cleanup_artifact_id'], p['independent_cleanup_sha256'], 64_000)
        independent_cleanup = zip_json(independent, 'cleanup.json')
        if (independent_cleanup != p['independent_cleanup']
                or independent_cleanup.get('status') != 'verified_absent'
                or independent_cleanup.get('name') != cleanup.get('name')
                or independent_cleanup.get('consecutive_absent_checks', 0) < 3
                or independent_cleanup.get('checked_epoch', 0) < cleanup.get('checked_epoch', 0)):
            raise RuntimeError('Independent cleanup receipt mismatch')
        usd, seconds = cleanup.get('estimated_total_usd'), cleanup.get('estimated_total_seconds')
        if (not finite(usd) or not finite(seconds) or usd <= prior_usd or seconds <= prior_seconds
                or usd > 30 or seconds > 1100 * 60):
            raise RuntimeError('Completed cumulative consumption is invalid')
        prior_usd, prior_seconds = usd, seconds
        verified.append(p)
    return verified, prior_usd, prior_seconds


def verify_completed_archive(raw, output, spec, accepted):
    """Recompute checkpoint integrity instead of trusting an acceptance label."""
    from runpod.confirmation_verify import verify_training_archive
    with zipfile.ZipFile(io.BytesIO(raw)) as z, tempfile.TemporaryDirectory() as tmp:
        matches = [m for m in z.infolist() if m.filename == 'experiment.tar.gz']
        if len(matches) != 1 or not 0 < matches[0].file_size <= 2 * 1024**3:
            raise RuntimeError('Missing or oversized completed training archive')
        archive = Path(tmp) / 'experiment.tar.gz'
        with z.open(matches[0]) as source, archive.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if verify_training_archive(archive, dataset_metadata(output), spec) != accepted:
            raise RuntimeError('Completed checkpoint verification differs from receipt')


def separate_dino_authorization(artifact):
    """Recognize only the independently approved DINO workflow's exact marker.

    Its USD10/500-minute allocation is separate, not zero-cost VLM consumption.
    Global owned-Pod absence and the shared paid-workflow mutex still apply.
    Unknown names, dispatch identities and damaged receipts remain blocking.
    """
    if artifact.get('name') != DINO_MARKER:
        return False
    identity = artifact.get('workflow_run', {})
    run_id, digest = identity.get('id'), artifact.get('digest', '')
    if (type(run_id) is not int or run_id <= 0 or artifact.get('expired') is not False
            or not digest.startswith('sha256:') or len(digest) != 71):
        raise RuntimeError('Invalid separate DINO marker')
    run = request(f'{BASE}/actions/runs/{run_id}', token=os.environ['GH_TOKEN'])
    if (run.get('path') != '.github/workflows/dinov3-gpu.yml'
            or run.get('event') != 'workflow_dispatch' or run.get('head_branch') != 'main'
            or run.get('run_attempt') != 1 or run.get('head_sha') != identity.get('head_sha')):
        raise RuntimeError('Separate DINO authorization identity mismatch')
    verify_marker(artifact_bytes(artifact['id'], digest[7:], 64_000), run_id)
    return True


def reconcile(run, require_current, output, seed=42):
    """Unknown/missing/expired attempts block launch, including another agent's attempt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if seed not in {42, 123}:
        raise RuntimeError('Unsupported ledger seed')
    precreate = reconciled_precreate()
    completed, total_usd, total_seconds = completed_receipts(output)
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
            if any(previous.get('id') == p['run_id'] for p in completed):
                continue  # pinned successful receipt and both cleanup paths checked above
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
    # Seed 123 is a separate manual entry point, with the same cumulative budget.
    for page in range(1, 101):
        try:
            data = request(f'{BASE}/actions/workflows/vlm-seed123.yml/runs?event=workflow_dispatch&per_page=100&page={page}',
                           token=os.environ['GH_TOKEN'])
        except RemoteHTTPError as exc:
            if exc.status == 404 and seed == 42 and page == 1:
                break  # Before the new workflow exists on main.
            raise
        runs = data.get('workflow_runs')
        if not isinstance(runs, list):
            raise RuntimeError('Cannot verify seed-123 dispatch history')
        for previous in runs:
            if str(previous.get('id')) == str(run) or any(previous.get('id') == p['run_id'] for p in completed):
                continue
            if previous.get('status') != 'completed' or previous.get('run_attempt') != 1:
                raise RuntimeError('Unreconciled seed-123 dispatch')
            jobs = request(f"{BASE}/actions/runs/{previous['id']}/jobs?per_page=100", token=os.environ['GH_TOKEN'])
            paid = [s for j in jobs.get('jobs', []) for s in j.get('steps', [])
                    if s.get('name') == PAID_STEP.replace('42', '123')]
            if len(paid) != 1 or paid[0].get('conclusion') != 'skipped':
                raise RuntimeError('A prior paid seed-123 attempt requires reconciliation')
        if len(runs) < 100:
            break
    else:
        raise RuntimeError('Incomplete seed-123 dispatch history')
    expected = {str(x['run_id']): x for x in HISTORY + ([precreate] if precreate else []) + completed}
    observed, current, attempts = set(), False, []
    for page in range(1, 101):
        data = request(f'{BASE}/actions/artifacts?per_page=100&page={page}', token=os.environ['GH_TOKEN'])
        artifacts = data.get('artifacts')
        if not isinstance(artifacts, list):
            raise RuntimeError('Cannot enumerate authorization ledger')
        for artifact in artifacts:
            if not artifact.get('name', '').startswith('runpod-authorization-'):
                continue
            if separate_dino_authorization(artifact):
                continue
            origin = str(artifact.get('workflow_run', {}).get('id'))
            if artifact['name'] != MARKER or artifact.get('expired'):
                raise RuntimeError('Unreconciled or expired authorization marker')
            if origin in observed:
                raise RuntimeError('Duplicate authorization marker')
            observed.add(origin)
            if origin == str(run):
                verify_current_artifact(artifact, run, seed=seed)
                current = True
                continue
            pin = expected.get(origin)
            if (pin is None or artifact.get('id') != pin['marker_artifact_id']
                    or artifact.get('digest') != 'sha256:' + pin['marker_sha256']):
                raise RuntimeError('Unreconciled attempt; do not buy another GPU')
            verify_marker(artifact_bytes(pin['marker_artifact_id'], pin['marker_sha256'], 64_000), origin,
                          current=bool(pin.get('controller_sha')), controller_sha=pin.get('controller_sha'),
                          seed=pin.get('seed', 42))
            attempt = dict(run_id=int(origin), marker_artifact_id=pin['marker_artifact_id'],
                           evidence_sha256=pin['marker_sha256'], cleanup='verified_absent')
            if pin in completed:
                attempt.update(seed=pin['seed'], accepted=True, result_sha256=pin['result_sha256'])
            attempts.append(attempt)
        if len(artifacts) < 100:
            break
    else:
        raise RuntimeError('Incomplete authorization ledger')
    if not set(expected) <= observed:
        raise RuntimeError('Required historical authorization marker missing')
    if require_current and not current:
        verify_upload_receipt(run, seed=seed)
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
    ledger = dict(usd=total_usd, seconds=total_seconds, attempts=sorted(attempts, key=lambda a:a['run_id']))
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
