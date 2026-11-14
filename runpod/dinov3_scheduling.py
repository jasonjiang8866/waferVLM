"""Narrow opt-in for the two separately approved November 14 experiments.

No create/delete calls. Unknown project Pods and any previous executed DINO job
still block allocation. A queued cancellation is not a paid retry allowance.
"""
import os
import re

from runpod.provider import REPOSITORY, finite, request

BASE = f'https://api.github.com/repos/{REPOSITORY}'
QUEUED_DINO_RUN = '43563592797'
QUEUED_DINO_SHA = '964ea036a847f36c569f32f5d21e66cf40e8aad9'
VLM_RUN = '43562568201'
VLM_SHA = '537e59bd34e1415bf5091b90bb270fcc3ac08f22'
VLM_AUTH = 'wafer-20261107-usd20'
# Reserve the ENTIRE separately approved VLM allowance, not just one hour of
# currentSpendPerHr. Deliberately conservative about billing/reporting lag.
VLM_ACCOUNT_RESERVE_USD = 30.0


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
    """Allow exactly one current run plus an explicitly verified zero-job cancel."""
    rows = data.get('workflow_runs', [])
    replaced = os.environ.get('REPLACE_QUEUED_RUN_ID', '')
    if not replaced:
        if data.get('total_count') == 1 and [str(r['id']) for r in rows] == [current]:
            return
        raise ValueError('Prior DINO dispatch requires ledger/spend/cleanup reconciliation')
    if (replaced != QUEUED_DINO_RUN or current == replaced or data.get('total_count') != 2
            or len(rows) != 2 or {str(r['id']) for r in rows} != {current, replaced}):
        raise ValueError('Queued replacement requires exact two-run history')
    previous = request(f'{BASE}/actions/runs/{replaced}', token=token)
    cancelled_before_jobs(previous, token)


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
