"""Reconcile one allocated startup timeout without treating it as free capacity failure."""
import io
import json
import os
import zipfile
from pathlib import Path

PIN = dict(
    run_id=44108007848, seed='repeat',
    controller_sha='02f95227f398f1b87386e551aa96fe7cd8ea017d',
    result_artifact_id=14340483600,
    result_sha256='85ddd183ec8a7b65533ba868d93700ba1f8a962b51ddd61ba2c2e3da89440ba6',
    marker_artifact_id=14339098040,
    marker_sha256='5bdac3bf842219469322f2a75d767d34bb8a3a7ec11e1a735aad4744a3cb024b',
    independent_cleanup_run_id=44112651261,
    independent_cleanup_artifact_id=14341337469,
    independent_cleanup_sha256='3e5ec1e5b36ee9cfec0e70cd413025ed3642ac97917ce86ddec6f85d359e1213',
)
GPU = 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
REGION = 'US-PA-1'
POD = '2am9uckfk8rhcm'
WORK_SECONDS = 3600


def enabled():
    return os.environ.get('WAFER_VLM_STARTUP_RETRY') == '1'


def reconcile(output, prior_usd, prior_seconds):
    from runpod import vlm_ledger as ledger
    from runpod.provider import finite
    from runpod.vlm_repeat_resume import finish_pin
    p = PIN
    output = Path(output)
    expected = dict(head_sha=p['controller_sha'], run_attempt=1, status='completed',
                    conclusion='failure', event='workflow_dispatch', head_branch='main',
                    path='.github/workflows/vlm-repeats.yml')
    run = ledger.request(f"{ledger.BASE}/actions/runs/{p['run_id']}", token=os.environ['GH_TOKEN'])
    if any(run.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Startup-failed dispatch identity changed')
    receipts = ['resume-verification.json', 'finish-verification.json',
                'capacity-recovery.json', 'finish-capacity-recovery.json']
    raw = ledger.artifact_bytes(p['result_artifact_id'], p['result_sha256'], 100000)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if sorted(z.namelist()) != sorted(['cleanup.json', 'allocation.json',
                'dataset-metadata/dataset.json', 'input-verification.json', 'launch-spec.json',
                'ledger.json', 'preflight.json', 'provision.json', *receipts]):
            raise RuntimeError('Unexpected startup-failure evidence')
    clean, allocation, spec, intent, preflight = [ledger.zip_json(raw, n + '.json') for n in
        ('cleanup', 'allocation', 'launch-spec', 'provision', 'preflight')]
    name = f"wafer-20261107-usd20-vlmrepeats-{p['run_id']}"
    cost = intent.get('cost', {})
    if (allocation != dict(id=POD, gpu=GPU, data_center=REGION, observed_hourly_usd=2.69)
            or intent.get('pod_id') != POD or intent.get('name') != name
            or intent.get('run_id') != str(p['run_id'])
            or spec.get('phase') != 'validation-repeats' or spec.get('gpu') != GPU
            or spec.get('controller_sha') != p['controller_sha']
            or spec.get('resume_source_run_id') != finish_pin()['run_id']
            or spec.get('ledger', {}).get('usd') != prior_usd
            or spec.get('ledger', {}).get('seconds') != prior_seconds
            or preflight.get('status') != 'preflight_passed' or preflight.get('gpu') != GPU
            or REGION not in preflight.get('regions', [])
            or cost.get('prior_spend_reserve_usd') != prior_usd
            or cost.get('prior_runtime_reserve_seconds') != prior_seconds
            or cost.get('gpu_hourly_usd') != 2.69 or cost.get('disk_hourly_usd') != 60*.10/(28*24)
            or cost.get('work_seconds') != 2255 or cost.get('cleanup_seconds') != 600
            or clean.get('status') != 'verified_absent' or clean.get('name') != name
            or clean.get('creation_confirmed') is not True or clean.get('consecutive_absent_checks', 0) < 3
            or clean.get('within_budget') is not True or clean.get('within_runtime') is not True
            or clean.get('persistent_volumes_created') is not False
            or any(ledger.zip_json(raw, n) != json.loads((output/n).read_text()) for n in receipts)):
        raise RuntimeError('Startup failure lacks exact allocation, budget or cleanup evidence')
    usd, seconds, elapsed = [clean.get(k) for k in
        ('estimated_total_usd', 'estimated_total_seconds', 'elapsed_billing_bound_s')]
    if (not all(finite(v, .001) for v in (prior_usd, prior_seconds, usd, seconds, elapsed,
                    clean.get('checked_epoch'), intent.get('started_epoch')))
            or usd > 30 or seconds > 66000 or elapsed > 2855
            or abs(seconds - prior_seconds - elapsed) > 1e-6
            or abs(clean['checked_epoch'] - intent['started_epoch'] - elapsed) > .01
            or abs(usd - prior_usd - elapsed / 3600 * (cost['gpu_hourly_usd'] + cost['disk_hourly_usd'])) > 1e-9):
        raise RuntimeError('Invalid startup-failure cumulative consumption')
    independent_run = ledger.request(f"{ledger.BASE}/actions/runs/{p['independent_cleanup_run_id']}",
                                     token=os.environ['GH_TOKEN'])
    independent_expected = dict(expected, conclusion='success', event='workflow_run',
                                path='.github/workflows/vlm-repeats-cleanup.yml')
    if any(independent_run.get(k) != v for k, v in independent_expected.items()):
        raise RuntimeError('Independent startup cleanup dispatch changed')
    check = ledger.zip_json(ledger.artifact_bytes(p['independent_cleanup_artifact_id'],
                            p['independent_cleanup_sha256'], 64000), 'cleanup.json')
    if (check.get('name') != name or check.get('status') != 'verified_absent'
            or check.get('consecutive_absent_checks', 0) < 3
            or not finite(check.get('checked_epoch'), .001)
            or check['checked_epoch'] < clean['checked_epoch']):
        raise RuntimeError('Independent startup-failure cleanup mismatch')
    ledger.verify_marker(ledger.artifact_bytes(p['marker_artifact_id'], p['marker_sha256'], 64000),
                         p['run_id'], True, p['controller_sha'], 'repeat')
    rejected = dict(gpu=GPU, region=REGION, retry_after_epoch=check['checked_epoch']+1800)
    (output/'startup-recovery.json').write_text(json.dumps(dict(
        status='verified_allocated_startup_failure', source_run_id=p['run_id'],
        elapsed_billing_bound_s=elapsed, cumulative_usd=usd, cumulative_seconds=seconds,
        allocation=allocation, root_cause_verified=False), indent=2)+'\n')
    return p, usd, seconds, elapsed, rejected


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, usd, seconds, elapsed, _ = reconcile(args.output, 9.528627960290688, 44477.96864977837)
    print(json.dumps(dict(status='verified_allocated_startup_failure', cumulative_usd=usd,
                         cumulative_seconds=seconds, elapsed_billing_bound_s=elapsed)))
