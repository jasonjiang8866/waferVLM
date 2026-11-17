"""Reconcile the one capacity rejection before a bounded approved-GPU fallback."""
import json
import io
import os
import zipfile
from pathlib import Path

PIN = dict(
    run_id=44043355369, seed='repeat',
    controller_sha='3572f76f6af0d3fcf7127dd6f3ea5456451da44f',
    result_artifact_id=14311680489,
    result_sha256='43c4d9871dfaa3fb14296c469907bb00af570f9380ce71df2f6518171cdc9e7f',
    marker_artifact_id=14309833563,
    marker_sha256='c83a4df3ad2b9c6214d90adfa68aa118020e5de9a9d8b48e82843733ee1e62a1',
    independent_cleanup_artifact_id=14310073770,
    independent_cleanup_sha256='31bfaaf4f9ba12d0462106f7689a40a45b42dbbf269fbf8653388b4e7d8275a5',
)


def enabled():
    return os.environ.get('WAFER_VLM_RESUME_RETRY') == '1'


def reconcile(output, prior_usd, prior_seconds):
    from runpod import vlm_ledger as ledger
    from runpod.vlm_repeat_resume import PIN as source, RECEIPT
    from runpod.provider import finite
    p = PIN
    run = ledger.request(f"{ledger.BASE}/actions/runs/{p['run_id']}", token=os.environ['GH_TOKEN'])
    expected = dict(head_sha=p['controller_sha'], run_attempt=1, status='completed',
                    conclusion='failure', event='workflow_dispatch', head_branch='main',
                    path='.github/workflows/vlm-repeats.yml')
    if any(run.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Capacity-rejected dispatch identity changed')
    raw = ledger.artifact_bytes(p['result_artifact_id'], p['result_sha256'], 100000)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if sorted(z.namelist()) != sorted(['cleanup.json', 'create-error.json',
                'dataset-metadata/dataset.json', 'input-verification.json', 'launch-spec.json',
                'ledger.json', 'preflight.json', 'provision.json', 'resume-verification.json']):
            raise RuntimeError('Unexpected capacity-rejection evidence')
    error, clean, spec, intent = [ledger.zip_json(raw, n + '.json') for n in
                                 ('create-error', 'cleanup', 'launch-spec', 'provision')]
    name = f"wafer-20261107-usd20-vlmrepeats-{p['run_id']}"
    cost = intent.get('cost', {})
    if (error.get('status') != 'create_not_confirmed' or error.get('http_status') != 400
            or error.get('categories') != ['capacity']
            or intent.get('pod_id') is not None or intent.get('name') != name
            or intent.get('run_id') != str(p['run_id'])
            or spec.get('controller_sha') != p['controller_sha']
            or spec.get('resume_source_run_id') != source['run_id']
            or spec.get('ledger', {}).get('usd') != prior_usd
            or spec.get('ledger', {}).get('seconds') != prior_seconds
            or cost.get('prior_spend_reserve_usd') != prior_usd
            or cost.get('prior_runtime_reserve_seconds') != prior_seconds
            or clean.get('status') != 'verified_absent' or clean.get('name') != name
            or clean.get('creation_confirmed') is not False or clean.get('consecutive_absent_checks', 0) < 8
            or clean.get('within_budget') is not True or clean.get('within_runtime') is not True
            or clean.get('persistent_volumes_created') is not False
            or ledger.zip_json(raw, 'resume-verification.json') != json.loads(RECEIPT.read_text())):
        raise RuntimeError('Capacity rejection lacks exact ownership, budget or absence evidence')
    usd, seconds, elapsed = [clean[k] for k in
                            ('estimated_total_usd', 'estimated_total_seconds', 'elapsed_billing_bound_s')]
    if (not all(finite(v, .001) for v in (usd, seconds, elapsed))
            or usd > 30 or seconds > 66000
            or abs(seconds - prior_seconds - elapsed) > 1e-6
            or abs(clean['checked_epoch'] - intent['started_epoch'] - elapsed) > .01
            or abs(usd - prior_usd - elapsed / 3600 * (cost['gpu_hourly_usd'] + cost['disk_hourly_usd'])) > 1e-9):
        raise RuntimeError('Invalid capacity-rejection cumulative consumption')
    check = ledger.zip_json(ledger.artifact_bytes(p['independent_cleanup_artifact_id'],
                            p['independent_cleanup_sha256'], 64000), 'cleanup.json')
    if (check.get('name') != name or check.get('status') != 'verified_absent'
            or check.get('consecutive_absent_checks', 0) < 3
            or check.get('checked_epoch', 0) < clean['checked_epoch']):
        raise RuntimeError('Independent capacity-rejection cleanup mismatch')
    ledger.verify_marker(ledger.artifact_bytes(p['marker_artifact_id'], p['marker_sha256'], 64000),
                         p['run_id'], True, p['controller_sha'], 'repeat')
    (Path(output) / 'capacity-recovery.json').write_text(json.dumps(dict(
        status='verified_absent', source_run_id=p['run_id'], elapsed_billing_bound_s=elapsed,
        cumulative_usd=usd, cumulative_seconds=seconds), indent=2) + '\n')
    return p, usd, seconds, elapsed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # The preceding partial-source CI step reverified these exact starting totals.
    _, usd, seconds, elapsed = reconcile(args.output, 8.979243733782864, 41834.144988689426)
    print(json.dumps(dict(status='verified_capacity_rejection', cumulative_usd=usd,
                          cumulative_seconds=seconds, deducted_work_seconds=elapsed)))
