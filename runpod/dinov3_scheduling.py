"""Narrow opt-in for the two separately approved November 14 experiments.

No create/delete calls. Unknown project Pods and any unreviewed executed DINO
job still block allocation. Reconciled CPU failures are not paid retry allowances.
"""
import io
import json
import os
import re
import zipfile
from pathlib import Path

from runpod.provider import REPOSITORY, finite, request
from runpod.vlm_ledger import artifact_bytes, zip_json

BASE = f'https://api.github.com/repos/{REPOSITORY}'
QUEUED_DINO_RUN = '43563592797'
QUEUED_DINO_SHA = '964ea036a847f36c569f32f5d21e66cf40e8aad9'
VLM_RUN = '43562568201'
VLM_SHA = '537e59bd34e1415bf5091b90bb270fcc3ac08f22'
VLM_AUTH = 'wafer-20261107-usd20'
# Reserve the ENTIRE separately approved VLM allowance, not just one hour of
# currentSpendPerHr. Deliberately conservative about billing/reporting lag.
VLM_ACCOUNT_RESERVE_USD = 30.0
CPU_FAILED_RUN = '43566277450'
PAID = json.loads(Path(__file__).with_name('dinov3-paid-history.json').read_text())


def reconciled_paid_failure(token):
    """Charge the inspected transfer failure and require both absence receipts."""
    p = PAID
    run = request(f"{BASE}/actions/runs/{p['run_id']}", token=token)
    if (str(run.get('id')) != p['run_id'] or run.get('head_sha') != p['source_sha']
            or run.get('head_branch') != 'main' or run.get('event') != 'workflow_dispatch'
            or run.get('path') != '.github/workflows/dinov3-gpu.yml'
            or run.get('run_attempt') != 1 or run.get('status') != 'completed'
            or run.get('conclusion') != 'failure'):
        raise ValueError('Reconciled paid failure changed')
    jobs = request(f"{BASE}/actions/runs/{p['run_id']}/jobs?filter=all&per_page=100", token=token)
    if jobs.get('total_count') != 1 or len(jobs.get('jobs', [])) != 1:
        raise ValueError('Paid failure job history changed')
    job = jobs['jobs'][0]
    if job.get('id') != p['job_id'] or job.get('status') != 'completed' or job.get('conclusion') != 'failure':
        raise ValueError('Paid failure job identity changed')
    for name, conclusion in p['required_steps'].items():
        matches = [step for step in job.get('steps', []) if step.get('name') == name]
        if len(matches) != 1 or matches[0].get('status') != 'completed' or matches[0].get('conclusion') != conclusion:
            raise ValueError('Paid failure step changed')
    for kind in ('result', 'independent'):
        receipt = p[kind]
        meta = request(f"{BASE}/actions/artifacts/{receipt['artifact_id']}", token=token)
        origin = meta.get('workflow_run', {})
        if (meta.get('id') != receipt['artifact_id'] or meta.get('name') != receipt['name']
                or meta.get('expired') is not False or meta.get('digest') != 'sha256:'+receipt['sha256']
                or str(origin.get('id')) != receipt['run_id'] or origin.get('head_sha') != receipt['source_sha']):
            raise ValueError('Paid failure artifact receipt changed')
        data = artifact_bytes(receipt['artifact_id'], receipt['sha256'], 64_000)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sorted(archive.namelist()) != sorted(receipt['entries']):
                raise ValueError('Paid failure file set changed')
        for name, expected in receipt['entries'].items():
            if zip_json(data, name) != expected:
                raise ValueError('Paid failure evidence changed')
    cleanup = p['result']['entries']['cleanup.json']
    independent = p['independent']['entries']['cleanup.json']
    if (cleanup['status'] != 'verified_absent' or cleanup['within_cap'] is not True
            or independent['status'] != 'verified_absent' or independent['checks'] < 12
            or not finite(p['accounted_usd'], cleanup['estimated_usd'])
            or not finite(p['accounted_seconds'], cleanup['elapsed_billing_bound_s'])):
        raise ValueError('Paid failure cost/time/cleanup not reconciled')


def reconciled_cpu_failure(token):
    """Verify the exact pre-allocation test failure; never forgive a paid step."""
    p = json.loads((Path(__file__).with_name('dinov3-precreate.json')).read_text())
    run = request(f'{BASE}/actions/runs/{CPU_FAILED_RUN}', token=token)
    if (str(run.get('id')) != CPU_FAILED_RUN or run.get('head_sha') != p['source_sha']
            or run.get('head_branch') != 'main' or run.get('event') != 'workflow_dispatch'
            or run.get('path') != '.github/workflows/dinov3-gpu.yml'
            or run.get('run_attempt') != 1 or run.get('status') != 'completed'
            or run.get('conclusion') != 'failure'):
        raise ValueError('Reconciled CPU failure changed')
    jobs = request(f'{BASE}/actions/runs/{CPU_FAILED_RUN}/jobs?filter=all&per_page=100', token=token)
    if jobs.get('total_count') != 1 or len(jobs.get('jobs', [])) != 1:
        raise ValueError('CPU failure job history changed')
    job = jobs['jobs'][0]
    if job.get('id') != p['job_id'] or job.get('status') != 'completed' or job.get('conclusion') != 'failure':
        raise ValueError('CPU failure job identity changed')
    for name, conclusion in p['required_steps'].items():
        matches = [step for step in job.get('steps', []) if step.get('name') == name]
        if len(matches) != 1 or matches[0].get('status') != 'completed' or matches[0].get('conclusion') != conclusion:
            raise ValueError('CPU failure reached an unreviewed step')
    for kind in ('result', 'independent'):
        receipt = p[kind]
        meta = request(f"{BASE}/actions/artifacts/{receipt['artifact_id']}", token=token)
        origin = meta.get('workflow_run', {})
        if (meta.get('id') != receipt['artifact_id'] or meta.get('name') != receipt['name']
                or meta.get('expired') is not False or meta.get('digest') != 'sha256:'+receipt['sha256']
                or str(origin.get('id')) != receipt['run_id'] or origin.get('head_sha') != receipt['source_sha']):
            raise ValueError('CPU failure artifact receipt changed')
        data = artifact_bytes(receipt['artifact_id'], receipt['sha256'], 64_000)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sorted(archive.namelist()) != receipt['files']:
                raise ValueError('CPU failure contains unexpected resource intent')
        if zip_json(data, 'cleanup.json') != receipt['cleanup']:
            raise ValueError('CPU failure cleanup changed')


def cancelled_before_jobs(run, token):
    if (str(run.get('id')) != QUEUED_DINO_RUN or run.get('head_sha') != QUEUED_DINO_SHA
            or run.get('head_branch') != 'main' or run.get('event') != 'workflow_dispatch'
            or run.get('path') != '.github/workflows/dinov3-gpu.yml'
            or run.get('status') != 'completed' or run.get('conclusion') != 'cancelled'
            or run.get('run_attempt') != 1):
        raise ValueError('Only the reviewed never-started queued DINO run may be replaced')
    for suffix, key in [('jobs?filter=all&per_page=100', 'jobs'), ('artifacts?per_page=100', 'artifacts')]:
        result = request(f'{BASE}/actions/runs/{QUEUED_DINO_RUN}/{suffix}', token=token)
        if result.get('total_count') != 0 or result.get(key) != []:
            raise ValueError('Queued replacement has execution/artifact evidence; reconcile separately')


def dispatch_history(data, current, token):
    """Allow exactly one recovery after the inspected paid transfer failure."""
    rows = data.get('workflow_runs', [])
    replaced = os.environ.get('REPLACE_QUEUED_RUN_ID', '')
    if not replaced:
        if data.get('total_count') == 1 and [str(r['id']) for r in rows] == [current]:
            return
        raise ValueError('Prior DINO dispatch requires ledger/spend/cleanup reconciliation')
    prior = {replaced, CPU_FAILED_RUN, PAID['run_id']}
    if (replaced != QUEUED_DINO_RUN or current in prior or data.get('total_count') != 4
            or len(rows) != 4 or {str(r['id']) for r in rows} != {current, *prior}):
        raise ValueError('Paid recovery requires exact reconciled four-run history')
    previous = request(f'{BASE}/actions/runs/{replaced}', token=token)
    cancelled_before_jobs(previous, token)
    reconciled_cpu_failure(token)
    reconciled_paid_failure(token)


def companion(pods, token):
    """Permit at most one exact live VLM Pod, with full separate budget reserve."""
    selected = os.environ.get('PARALLEL_VLM_RUN_ID', '')
    project = [p for p in pods if p.get('env', {}).get('WAFER_MANAGED_BY') == REPOSITORY
               or p.get('name', '').startswith(('wafervlm-dinov3-', VLM_AUTH))]
    if not selected:
        if project:
            raise ValueError('Existing project Pod must be reconciled')
        return dict(run_id=None, active_pod=False, account_reserve_usd=0.0)
    if selected != VLM_RUN:
        raise ValueError('Unreviewed parallel VLM run')
    run = request(f'{BASE}/actions/runs/{selected}', token=token)
    if (str(run.get('id')) != selected or run.get('head_sha') != VLM_SHA
            or run.get('event') != 'workflow_dispatch' or run.get('head_branch') != 'main'
            or run.get('path') != '.github/workflows/vlm-seed42.yml' or run.get('run_attempt') != 1
            or run.get('status') not in {'in_progress', 'completed'}):
        raise ValueError('Parallel VLM Actions identity mismatch')
    if len(project) > 1:
        raise ValueError('Multiple project Pods; refuse another allocation')
    if project:
        pod, env = project[0], project[0].get('env', {})
        if (run.get('status') != 'in_progress'
                or pod.get('name') != VLM_AUTH + '-vlm42-' + selected
                or env.get('WAFER_MANAGED_BY') != REPOSITORY or env.get('WAFER_RUN_ID') != selected
                or env.get('WAFER_AUTHORIZATION') != VLM_AUTH or env.get('WAFER_EXPERIMENT') != 'vlm-seed42'
                or env.get('WAFER_CONTROLLER_SHA') != VLM_SHA
                or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', pod.get('id', ''))
                or pod.get('mounts') or pod.get('status') != 'RUNNING'
                or not finite(pod.get('cost'), .001) or pod['cost'] > 1.20):
            raise ValueError('Parallel VLM Pod identity/state/rate mismatch')
    # Retain reserve even if inventory is temporarily empty or run just ended.
    return dict(run_id=selected, active_pod=bool(project), account_reserve_usd=VLM_ACCOUNT_RESERVE_USD)
