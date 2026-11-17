"""Bounded two-Pod import, fresh-mount verification, and ownership-safe cleanup."""
import argparse
import base64
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

from runpod.provider import Runpod, RemoteHTTPError, budget, finite
from runpod.storage_preflight import observe
from scripts.collect_wafer_data import plan, dataset_digest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'python@sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177'
MOUNT = '/wafer-public-data'
DATASETS = ['carinthia-s', 'silicon-wafer-pl', 'silicon-wafer-ebsd']
MAX_RATE = 0.10
RUNTIME_SECONDS = 1500  # aggregate, including provisioning and both sequential Pods
MIN_BUDGET = 0.50  # includes conservative reserve for temporary disk/requests/cleanup


def identity(run_id, attempt):
    if not re.fullmatch(r'[0-9]+', run_id) or not re.fullmatch(r'[0-9]+', attempt):
        raise ValueError('Invalid Actions run identity')
    return f'wafervlm-data-{run_id}-{attempt}'


def select_volume(volumes, requested):
    if requested:
        matches = [v for v in volumes if v['id'] == requested]
    else:
        matches = [v for v in volumes if v.get('name') == 'wafervlm-public-datasets']
        if not matches and len(volumes) == 1:
            matches = volumes
    if len(matches) != 1:
        raise ValueError('Select one existing Global Volume ID; no implicit creation')
    value = matches[0]['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Invalid volume identity')
    return value


def owns(pod, prefix, source):
    return (pod.get('name') in {prefix + '-import', prefix + '-verify'}
            and pod.get('image') == IMAGE
            and pod.get('env', {}).get('WAFER_STORAGE_RUN') == prefix
            and pod.get('env', {}).get('WAFER_STORAGE_SOURCE') == source)


def cleanup(api, prefix, source):
    # Inventory also reconciles a create whose HTTP outcome was unknown.
    deleted = []
    for pod in api.listing('/pods', 'pods'):
        if owns(pod, prefix, source):
            api.call('/pods/' + pod['id'], method='DELETE')
            deleted.append(pod['id'])
    for _ in range(12):
        remaining = [p['id'] for p in api.listing('/pods', 'pods') if owns(p, prefix, source)]
        if not remaining:
            return {'status': 'cleanup_verified', 'deleted_ids': deleted, 'remaining_owned_pods': []}
        time.sleep(5)
    raise RuntimeError('Owned temporary Pod removal not verified')


def payload(prefix, source, mode, volume_id, cpu):
    return {'name': prefix + '-' + mode, 'image': IMAGE,
            'entrypoint': ['python'], 'cmd': ['-c', 'import time; time.sleep(2100)'],
            'cpu': {'id': cpu['id'], 'vcpuCount': 2}, 'cloud': 'SECURE', 'disk': 5,
            'mounts': {'global': [{'volumeId': volume_id, 'path': MOUNT}]},
            'ports': [], 'env': {'WAFER_STORAGE_RUN': prefix, 'WAFER_STORAGE_SOURCE': source}}


def verify_pod(pod, prefix, source, mode, volume_id):
    if not owns(pod, prefix, source) or pod['name'] != prefix + '-' + mode:
        raise ValueError('Pod ownership mismatch')
    if (pod.get('gpu') or pod.get('cpu', {}).get('vcpuCount') != 2
            or not finite(pod.get('cost'), 0.000001) or pod['cost'] > MAX_RATE):
        raise ValueError('Unexpected compute or price')
    if pod.get('mounts', {}).get('global') != [{'volumeId': volume_id, 'path': MOUNT}]:
        raise ValueError('Global Volume identity/path mismatch')


def bundle():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in ['scripts/collect_wafer_data.py', 'data_sources/catalog.json', 'runpod/storage_worker.py']:
            archive.writestr(file, (ROOT / file).read_bytes())
    encoded = base64.b64encode(buffer.getvalue()).decode()
    if len(encoded) > 16_000:
        raise ValueError('Worker bundle exceeds bounded exec environment')
    return encoded


def exec_python(api, pod_id, code, env=None):
    result = api.call('/pods/' + pod_id + '/exec', method='POST', body={
        'command': ['python', '-c', code], 'env': env or {}, 'timeoutSeconds': 10})
    if result.get('exitCode') != 0 or result.get('timedOut') or result.get('truncated'):
        raise RuntimeError('Bounded exec did not complete cleanly')
    output = result.get('stdout', '')
    if not isinstance(output, str) or len(output) > 50_000:
        raise RuntimeError('Unexpected exec output')
    return output


def validate_result(result, selected, mode, volume_id):
    expected = 'mounted_readback_verified' if mode == 'import' else 'readback_verified'
    if result.get('status') != expected:
        raise ValueError('Worker did not finish with verified data')
    rows = result.get('receipts', [])
    if len(rows) != len(selected):
        raise ValueError('Incomplete dataset receipts')
    for d, row in zip(selected, rows):
        if (row.get('id') != d['id'] or row.get('version') != d['version']
                or row.get('volume_id') != volume_id or row.get('mount_path') != MOUNT
                or row.get('dataset_sha256') != dataset_digest(d)):
            raise ValueError('Dataset receipt identity mismatch')
        if len(row.get('files', [])) != len(d['files']):
            raise ValueError('Incomplete file receipts')
        for expected_file, actual in zip(d['files'], row['files']):
            for field in ('name', 'bytes', 'sha256'):
                if actual.get(field) != expected_file.get(field):
                    raise ValueError('File receipt mismatch')
            if actual.get('publisher_md5') != expected_file['md5']:
                raise ValueError('Publisher checksum receipt mismatch')


def stage(api, mode, prefix, source, volume_id, cpu, selected, deadline):
    if time.monotonic() + 60 >= deadline:
        raise TimeoutError('Insufficient time remaining for next Pod')
    pod = api.call('/pods', method='POST', body=payload(prefix, source, mode, volume_id, cpu))
    pod_id = pod['id']
    # No automatic create retry: on ambiguous create, outer cleanup reconciles inventory.
    while time.monotonic() < deadline:
        pod = api.call('/pods/' + pod_id)
        verify_pod(pod, prefix, source, mode, volume_id)
        if pod.get('status') == 'RUNNING':
            try:
                exec_python(api, pod_id, "print('ready')")
                break
            except RemoteHTTPError as exc:
                if exc.status not in (409, 503):
                    raise
        elif pod.get('status') in ('EXITED', 'TERMINATED'):
            raise RuntimeError('CPU Pod exited before readiness')
        time.sleep(5)
    else:
        raise TimeoutError('CPU Pod readiness deadline')
    bootstrap = """import base64,io,json,os,pathlib,subprocess,sys,zipfile
root=pathlib.Path('/tmp/wafer-storage');root.mkdir(exist_ok=True)
marker=root/'started'
with marker.open('x') as f: f.write('started')
with zipfile.ZipFile(io.BytesIO(base64.b64decode(os.environ['WAFER_BUNDLE']))) as z: z.extractall(root)
subprocess.Popen([sys.executable,'-m','runpod.storage_worker',os.environ['WAFER_MODE'],os.environ['WAFER_VOLUME'],os.environ['WAFER_MOUNT']],cwd=root,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print('started')
"""
    # Source bundle is built from reviewed files only. No account/runner secret enters Pod.
    exec_python(api, pod_id, bootstrap, {'WAFER_BUNDLE': bundle(), 'WAFER_MODE': mode,
                                      'WAFER_VOLUME': volume_id, 'WAFER_MOUNT': MOUNT})
    reader = "import pathlib; p=pathlib.Path('/tmp/wafer-storage/result.json'); print(p.read_text() if p.exists() else '{}')"
    while time.monotonic() < deadline:
        result = json.loads(exec_python(api, pod_id, reader))
        if result:
            validate_result(result, selected, mode, volume_id)
            return {'pod_id': pod_id, 'compute_usd_per_hour': pod['cost'],
                    'data_center_id': pod.get('dataCenterId'), 'result': result}
        time.sleep(10)
    raise TimeoutError('Data import/readback deadline')


def execute(api, args):
    if not args.confirm_paid or not finite(args.approved_budget_usd, MIN_BUDGET):
        raise ValueError('Explicit paid approval and at least US$0.50 total import budget required')
    if args.approved_budget_usd > 1:
        raise ValueError('This small import accepts at most US$1; review larger plans separately')
    prefix = identity(args.run_id, args.attempt)
    source = os.environ.get('GITHUB_SHA', '')
    if not re.fullmatch('[0-9a-f]{40}', source):
        raise ValueError('Exact source SHA required')
    selected, sizes = plan(DATASETS)  # 10 GB approval gate BEFORE cloud operations
    inventory = observe(api)
    if any(p.get('name', '').startswith('wafervlm-data-') for p in api.listing('/pods', 'pods')):
        raise ValueError('An existing dataset Pod requires reconciliation before another import')
    volume_id = select_volume(api.listing('/global-volumes', 'globalVolumes'), args.volume_id)
    cpus = [c for c in inventory['cpu_candidates']
            if c['compute_usd_per_hour'] <= MAX_RATE and c['availability'] in {'LOW', 'MEDIUM', 'HIGH'}]
    if not cpus:
        raise ValueError('No CPU candidate under US$0.10/hour')
    budget(api, reserve=args.approved_budget_usd)  # existing account/monthly guard
    report = {'status': 'started', 'run_id': args.run_id, 'attempt': args.attempt,
              'source_sha': source, 'image': IMAGE, 'volume_id': volume_id,
              'approved_budget_usd': args.approved_budget_usd, 'sizes': sizes,
              'aggregate_work_seconds_cap': RUNTIME_SECONDS, 'stages': []}
    args.output.mkdir(parents=True, exist_ok=True)
    def save():
        (args.output / 'import.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    started = time.monotonic()
    deadline = started + RUNTIME_SECONDS
    try:
        for mode in ['import', 'verify']:
            report['stages'].append(stage(api, mode, prefix, source, volume_id, cpus[0], selected, deadline))
            report['cleanup'] = cleanup(api, prefix, source)
            save()
        report['status'] = 'accepted'
    finally:
        try:
            report['cleanup'] = cleanup(api, prefix, source)
        except Exception:
            report['status'] = 'cleanup_unverified'
            raise
        finally:
            report['controller_elapsed_seconds'] = round(time.monotonic() - started, 3)
            report['compute_cost_upper_estimate_usd'] = round(report['controller_elapsed_seconds'] / 3600 * MAX_RATE, 6)
            report['estimate_note'] = 'Conservative compute estimate; excludes persistent storage and request fees. Not a provider billing hard cap.'
            if report['status'] not in ('accepted', 'cleanup_unverified'):
                report['status'] = 'partial_or_failed'
            save()
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['import', 'cleanup'])
    p.add_argument('--confirm-paid', action='store_true')
    p.add_argument('--approved-budget-usd', type=float, default=0)
    p.add_argument('--volume-id', default='')
    p.add_argument('--run-id', default=os.environ.get('GITHUB_RUN_ID', ''))
    p.add_argument('--attempt', default=os.environ.get('GITHUB_RUN_ATTEMPT', ''))
    p.add_argument('--output', type=Path, default=Path('storage-import'))
    args = p.parse_args()
    api = Runpod()
    if args.action == 'cleanup':
        result = cleanup(api, identity(args.run_id, args.attempt), os.environ.get('GITHUB_SHA', ''))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'cleanup.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        result = execute(api, args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
