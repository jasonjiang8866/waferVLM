"""One-off DINO seed-17 sweep. Explicit manual dispatch; isolated from Qwen's ledger."""
import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from runpod.dinov3_bundle import FILES, UPLOAD_CHUNK_BYTES, cache_files, recipe, verify_archive, weight_report
from runpod.dinov3_scheduling import PAID, companion, dispatch_history
from runpod.provider import NoRedirect, REPOSITORY, RemoteHTTPError, Runpod, budget, finite, request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dinov3-results'
AUTH = 'wafervlm-dinov3-seed17-v1'
GPU = 'NVIDIA GeForce RTX 4090'
RATE_CAP = .89
DISK_GB = 60
DISK_RATE = DISK_GB * .10 / (28 * 24)
WORK_SECONDS = 5400
CLEANUP_SECONDS = 600
TOTAL_USD_CAP = 3.0  # Conservative first-attempt ceiling within aggregate approval.
APPROVED_TOTAL_USD = 10.0  # User approval 2026-11-14, DINO only.
APPROVED_TOTAL_SECONDS = 500 * 60
PRIOR_SPEND_USD = PAID['accounted_usd']
PRIOR_RUNTIME_SECONDS = PAID['accounted_seconds']
MARKER = 'runpod-authorization-' + AUTH


def save(name, data):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def identity():
    run = os.environ.get('GITHUB_RUN_ID', '')
    if os.environ.get('GITHUB_REPOSITORY') != REPOSITORY or not run.isdigit():
        raise ValueError('Unexpected repository/run')
    return run


def cost(rate):
    if not finite(rate, .001) or rate > RATE_CAP:
        raise ValueError('4090 quote exceeds cap')
    maximum = (rate + DISK_RATE) * (WORK_SECONDS + CLEANUP_SECONDS) / 3600
    if (maximum + 1 > min(TOTAL_USD_CAP, APPROVED_TOTAL_USD) - PRIOR_SPEND_USD
            or PRIOR_RUNTIME_SECONDS + WORK_SECONDS + CLEANUP_SECONDS > APPROVED_TOTAL_SECONDS):
        raise ValueError('Insufficient cleanup/uncertainty reserve')
    return dict(gpu_hourly_usd=rate, disk_hourly_usd=DISK_RATE, work_seconds=WORK_SECONDS,
                cleanup_seconds=CLEANUP_SECONDS, estimated_max_usd=maximum,
                total_cap_usd=TOTAL_USD_CAP, reserve_usd=TOTAL_USD_CAP-PRIOR_SPEND_USD-maximum,
                approved_total_usd=APPROVED_TOTAL_USD, approved_total_seconds=APPROVED_TOTAL_SECONDS,
                prior_spend_usd=PRIOR_SPEND_USD, prior_runtime_seconds=PRIOR_RUNTIME_SECONDS)


def image_provenance(image, source):
    """Verify the immutable manifest/config digest and image's embedded source SHA."""
    if not re.fullmatch(r'ghcr.io/jasonjiang8866/wafervlm@sha256:[a-f0-9]{64}', image):
        raise ValueError('Pinned waferVLM image digest required')
    base = 'https://ghcr.io/v2/jasonjiang8866/wafervlm/'
    with urllib.request.urlopen('https://ghcr.io/token?service=ghcr.io&scope=repository:jasonjiang8866/wafervlm:pull', timeout=30) as r:
        token = json.load(r)['token']
    def fetch(kind, digest):
        if not re.fullmatch(r'sha256:[a-f0-9]{64}', digest):
            raise ValueError('Invalid image content digest')
        req = urllib.request.Request(base + kind + '/' + digest, headers={
            'Authorization': 'Bearer ' + token,
            'Accept': 'application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read(4_000_001)
        if len(raw) > 4_000_000 or 'sha256:' + hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Image content digest mismatch')
        return json.loads(raw)
    manifest = fetch('manifests', image.split('@')[1])
    if 'manifests' in manifest:
        candidates = [m for m in manifest['manifests'] if m.get('platform', {}).get('os') == 'linux'
                      and m.get('platform', {}).get('architecture') == 'amd64']
        if len(candidates) != 1:
            raise ValueError('Exactly one linux/amd64 manifest required')
        manifest = fetch('manifests', candidates[0]['digest'])
    config = fetch('blobs', manifest['config']['digest'])
    values = [x for x in config.get('config', {}).get('Env', []) if x.startswith('WAFERVLM_SOURCE_SHA=')]
    if values != ['WAFERVLM_SOURCE_SHA=' + source]:
        raise ValueError('Image source differs from reviewed main SHA')


def manual_guard():
    identity()
    source = git('rev-parse', 'HEAD')
    if (os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'
            or os.environ.get('GITHUB_REF') != 'refs/heads/main'
            or os.environ.get('GITHUB_RUN_ATTEMPT') != '1'
            or os.environ.get('CONFIRM_PAID') != 'true'
            or os.environ.get('EXPECTED_SHA') != source):
        raise ValueError('Explicit paid dispatch/current SHA/first attempt required')
    token = os.environ.get('GH_TOKEN', '')
    if not token:
        raise ValueError('Actions read token required')
    base = f'https://api.github.com/repos/{REPOSITORY}'
    if request(base + '/git/ref/heads/main', token=token)['object']['sha'] != source:
        raise ValueError('main moved before launch')
    return source, token


def ledger_guard():
    _, token = manual_guard()
    run = identity()
    base = f'https://api.github.com/repos/{REPOSITORY}'
    # The first manual run is a one-off. Subsequent dispatches require reviewed
    # reconciliation, even if an old marker was deleted or expired.
    data = request(base + '/actions/workflows/dinov3-gpu.yml/runs?event=workflow_dispatch&per_page=100', token=token)
    dispatch_history(data, run, token)
    receipt = os.environ.get('AUTHORIZATION_ARTIFACT_ID', '')
    if not receipt.isdigit() or int(receipt) <= 0:
        raise ValueError('Exact authorization upload receipt required')
    # Listing can lag a successful upload. Read the exact receipt, with bounded
    # GET-only retries; this does not retry or weaken the one-create gate.
    for attempt in range(12):
        try:
            artifact = request(base + '/actions/artifacts/' + receipt, token=token)
            break
        except RemoteHTTPError as exc:
            if exc.status != 404 or attempt == 11:
                raise
            time.sleep(5)
    origin = artifact.get('workflow_run', {})
    if (artifact.get('id') != int(receipt) or artifact.get('name') != MARKER
            or artifact.get('expired') is not False or str(origin.get('id')) != run
            or origin.get('head_sha') != os.environ.get('EXPECTED_SHA')
            or not re.fullmatch('sha256:[a-f0-9]{64}', artifact.get('digest', ''))):
        raise ValueError('Authorization upload identity/source/digest mismatch')


def preflight(api):
    source, github_token = manual_guard()
    image = os.environ.get('DINO_IMAGE', '')
    report = weight_report(OUT / 'weights' / 'weights.json')
    cache_files(report)
    image_provenance(image, source)
    pods = api.listing('/pods', 'pods')
    concurrent = companion(pods, github_token)
    query = urllib.parse.urlencode(dict(include='AVAILABILITY', product='POD', cloud='SECURE', count=1,
                                       minCudaVersion=recipe()['required_min_cuda']))
    catalog = api.call('/catalog/gpus/' + urllib.parse.quote(GPU, safe='') + '?' + query)
    rate = catalog.get('price', {}).get('secure')
    quoted = cost(rate)
    regions = sorted(d['id'] for d in catalog.get('dataCenters', [])
                     if d.get('availability') in {'LOW', 'MEDIUM', 'HIGH'}
                     and re.fullmatch(r'[A-Za-z0-9-]+', d.get('id', '')))
    if not regions:
        raise ValueError('No 4090 capacity meeting CUDA minimum')
    evidence = dict(status='preflight_passed', source_sha=source, image=image, weights=report,
                    gpu=GPU, min_cuda_version=recipe()['required_min_cuda'], regions=regions, cost=quoted,
                    companion=concurrent,
                    budget=budget(api, reserve=TOTAL_USD_CAP + concurrent['account_reserve_usd']),
                    checked_epoch=time.time())
    save('preflight.json', evidence)
    return evidence


def owned(pod, run):
    env = pod.get('env', {})
    return (pod.get('name') == AUTH + '-' + run and env.get('WAFER_MANAGED_BY') == REPOSITORY
            and env.get('DINO_RUN_ID') == run and env.get('DINO_AUTH') == AUTH)


def cleanup(api):
    run = identity()
    path = OUT / 'provision.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    save('cleanup.json', dict(status='unknown', run_id=run))
    absent = 0
    for attempt in range(12):
        matches = [p for p in api.listing('/pods', 'pods')
                   if p.get('name') == AUTH + '-' + run or (state.get('pod_id') and p.get('id') == state['pod_id'])]
        if not matches:
            absent += 1
            # Independent cleanup without a create response also waits the full
            # eventual-consistency window; a stopped process proves nothing.
            if absent >= (3 if state.get('pod_id') else 12):
                record = dict(status='verified_absent', run_id=run, checks=absent,
                              persistent_volumes_created=False, checked_epoch=time.time())
                if state.get('started_epoch'):
                    elapsed = time.time() - state['started_epoch']
                    estimate = elapsed / 3600 * (state['rate'] + DISK_RATE)
                    record.update(elapsed_billing_bound_s=elapsed, estimated_usd=estimate,
                                  cost_basis='intent_to_absence_estimate_not_invoice',
                                  cumulative_estimated_usd=PRIOR_SPEND_USD+estimate,
                                  cumulative_runtime_bound_s=PRIOR_RUNTIME_SECONDS+elapsed,
                                  within_cap=PRIOR_SPEND_USD+estimate <= TOTAL_USD_CAP
                                  and elapsed <= WORK_SECONDS + CLEANUP_SECONDS
                                  and PRIOR_RUNTIME_SECONDS+elapsed <= APPROVED_TOTAL_SECONDS)
                save('cleanup.json', record)
                if record.get('within_cap') is False:
                    raise RuntimeError('Estimated cost/runtime exceeded ceiling')
                return
        else:
            absent = 0
            if len(matches) != 1 or not owned(matches[0], run) or matches[0].get('mounts'):
                raise ValueError('Cleanup ownership/persistent mount mismatch')
            pod = matches[0]
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', pod.get('id', '')):
                raise ValueError('Invalid Pod ID')
            if not state:
                started, rate = float(pod['env']['DINO_STARTED']), float(pod['env']['DINO_RATE'])
                if not finite(started, 1) or started > time.time():
                    raise ValueError('Invalid recovered start time')
                cost(rate)
                state = dict(started_epoch=started, rate=rate)
            state['pod_id'] = pod['id']
            save('provision.json', state)
            try:
                api.call('/pods/' + pod['id'], 'DELETE')
            except RuntimeError:
                pass  # Only repeated inventory absence establishes deletion.
        time.sleep(10)
    raise RuntimeError('Pod deletion unverified; reconcile immediately')


def upload_weights(url, token, files, result):
    for name, path in files.items():
        if name in result.get('received', []):
            continue
        total, offset = path.stat().st_size, 0
        with path.open('rb') as data:
            while block := data.read(UPLOAD_CHUNK_BYTES):
                for attempt in range(3):
                    record = dict(file=name, offset=offset, size=len(block), total=total, attempt=attempt+1)
                    req = urllib.request.Request(url + '/weights/' + name, data=block, method='PUT', headers={
                        'Authorization': 'Bearer ' + token, 'Content-Type': 'application/octet-stream',
                        'User-Agent': 'waferVLM-actions/0.1',
                        'Content-Range': f'bytes {offset}-{offset+len(block)-1}/{total}',
                        'X-Chunk-SHA256': hashlib.sha256(block).hexdigest()})
                    try:
                        with urllib.request.build_opener(NoRedirect).open(req, timeout=45) as response:
                            if response.status != 204:
                                raise ValueError('Weight delivery not acknowledged')
                        save('transfer.json', dict(**record, status='acknowledged'))
                        break
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                        save('transfer.json', dict(**record, status='failed', http_status=status,
                                                  error_type=type(exc).__name__))
                        if attempt == 2 or (status is not None and status not in {408, 429} and status < 500):
                            raise
                        time.sleep(2 * (attempt + 1))
                offset += len(block)
        print('DINO weights delivered: ' + name + ' (' + str(total) + ' bytes)', flush=True)


def download(url, token, meta):
    size, digest = meta.get('size'), meta.get('sha256')
    if type(size) is not int or not 0 < size <= 2 * 1024**3 or not re.fullmatch('[a-f0-9]{64}', digest or ''):
        raise ValueError('Invalid archive metadata')
    req = urllib.request.Request(url + '/archive', headers={
        'Authorization': 'Bearer ' + token, 'User-Agent': 'waferVLM-actions/0.1'})
    path = OUT / 'experiment.tar.gz'
    count, hasher = 0, hashlib.sha256()
    with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response, path.open('wb') as f:
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > size:
                raise ValueError('Archive too large')
            hasher.update(block)
            f.write(block)
    if count != size or hasher.hexdigest() != digest:
        raise ValueError('Archive checksum mismatch')
    return path


class Deadline(BaseException):
    pass


def interrupted(signum, frame):
    raise Deadline('Deadline/cancellation; cleaning up')


def launch(api):
    ledger_guard()
    evidence = preflight(api)  # fresh inventory/price after marker persistence
    run, token = identity(), secrets.token_urlsafe(32)
    print('::add-mask::' + token, flush=True)
    files = cache_files(evidence['weights'])
    started = time.time()
    deadline = started + WORK_SECONDS
    save('provision.json', dict(pod_id=None, started_epoch=started, rate=evidence['cost']['gpu_hourly_usd']))
    payload = dict(name=AUTH + '-' + run, image=evidence['image'], cloud='SECURE', disk=DISK_GB,
                   gpu=dict(id=GPU, count=1, minRamPerGpu=32, minVcpuCountPerGpu=4,
                            minCudaVersion=evidence['min_cuda_version']),
                   dataCenterIds=evidence['regions'], startSsh=False, ports=['8888/http'],
                   entrypoint=['/bin/bash', '-lc'],
                   cmd=['exec /app/.venv/bin/python -u -m runpod.dinov3_worker'],
                   env=dict(WAFER_MANAGED_BY=REPOSITORY, DINO_RUN_ID=run, DINO_AUTH=AUTH,
                            DINO_RESULT_TOKEN=token, DINO_DEADLINE=str(deadline-60),
                            DINO_WEIGHTS=json.dumps(evidence['weights']), DINO_IMAGE=evidence['image'],
                            DINO_STARTED=str(started), DINO_RATE=str(evidence['cost']['gpu_hourly_usd']),
                            WAFERVLM_IMAGE_DIGEST=evidence['image'].split('@')[1]))
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.alarm(WORK_SECONDS)
    try:
        pod = api.call('/pods', 'POST', payload)  # exactly one create, never auto-retry
        pod_id = pod.get('id', '')
        if not re.fullmatch('[a-zA-Z0-9_-]{1,128}', pod_id):
            raise ValueError('Create outcome unknown; reconcile by name')
        save('provision.json', dict(pod_id=pod_id, started_epoch=started, rate=evidence['cost']['gpu_hourly_usd']))
        current = api.call('/pods/' + pod_id)
        if (not owned(current, run) or current.get('gpu', {}).get('id') != GPU
                or current.get('dataCenterId') not in evidence['regions']
                or not finite(current.get('cost'), .001)
                or current['cost'] > evidence['cost']['gpu_hourly_usd'] + DISK_RATE):
            raise ValueError('Allocated identity/GPU/region/rate mismatch')
        save('allocation.json', dict(pod_id=pod_id, gpu=GPU, data_center=current['dataCenterId'], hourly_usd=current['cost']))
        url = 'https://' + pod_id + '-8888.proxy.runpod.net'
        next_inventory, last_phase, seen = time.time() + 60, None, False
        while time.time() < deadline - 30:
            if time.time() >= next_inventory:
                detail = api.call('/pods/' + pod_id)
                if not owned(detail, run) or detail.get('status') in {'ERROR', 'EXITED', 'TERMINATED'}:
                    raise ValueError('Worker stopped or identity changed')
                next_inventory = time.time() + 60
            try:
                result = request(url + '/result', token=token)
            except RuntimeError:
                if not seen and time.time() - started > 1200:
                    raise TimeoutError('Worker startup exceeded 20 minutes')
                time.sleep(10)
                continue
            seen = True
            phase = result.get('phase')
            if phase != last_phase and isinstance(phase, str) and re.fullmatch('[a-z0-9_-]{1,40}', phase):
                print('DINO phase: ' + phase, flush=True)
                last_phase = phase
            if result.get('status') == 'running' and phase == 'awaiting_weights':
                upload_weights(url, token, files, result)
            if result.get('status') in {'accepted', 'rejected'}:
                save('result.json', result)
                archive = download(url, token, result['archive']) if result.get('archive') else None
                if result['status'] != 'accepted' or archive is None:
                    raise ValueError('Worker did not complete all six trials')
                acceptance = verify_archive(archive, evidence['source_sha'], evidence['image'], evidence['weights'])
                save('acceptance.json', acceptance)
                return
            time.sleep(10)
        raise TimeoutError('No complete artifacts before deadline')
    finally:
        signal.alarm(0)
        for s, handler in handlers.items():
            signal.signal(s, handler)
        cleanup(api)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'run', 'cleanup'])
    args = parser.parse_args()
    identity()
    try:
        {'preflight': preflight, 'run': launch, 'cleanup': cleanup}[args.mode](Runpod())
    except Exception as exc:
        # Never expose a provider response, URL credential, or account payload.
        save('error.json', dict(mode=args.mode, error_type=type(exc).__name__))
        raise SystemExit('DINO ' + args.mode + ' failed: ' + type(exc).__name__) from None


if __name__ == '__main__':
    main()
