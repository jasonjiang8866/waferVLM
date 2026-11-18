"""Reconcile the first holdout dispatch, rejected before worker creation."""
import io
import os
import zipfile

from runpod import vlm_ledger as ledger

RUN = 44303065030
CLEANUP_RUN = 44304259531
SHA = '2d4a0f976f92d207d2d773bfa68982c2cf63cef2'
RESULT = (14426294299, '612b907510520464820175ea3ab20caca23f0b09f43bc4bd8f0d5cc3b928d8e8')
CLEANUP = (14426614055, '2b0ae5793630d6f21e8b4c35262c9706aa03fe04d50ab9ef37076b08c1064be9')
MARKER = (14427710002, '5fc5f38251cc10f1fc0dc1d5ff34db3518492bfcec6d2cf737abb9a95f31b191')


def verify(result, independent, usd, seconds):
    name = f'wafer-20261107-usd20-vlmholdout-{RUN}'
    cleanup = ledger.zip_json(result, 'cleanup.json')
    for receipt, minimum in ((cleanup, 8), (ledger.zip_json(independent, 'cleanup.json'), 3)):
        if (receipt['name'] != name or receipt['status'] != 'verified_absent'
                or receipt['creation_confirmed'] is not False
                or receipt['consecutive_absent_checks'] < minimum):
            raise ValueError('Missing confirmed absence for the rejected holdout placement')
    provision = ledger.zip_json(result, 'provision.json')
    spec = ledger.zip_json(result, 'launch-spec.json')
    error = ledger.zip_json(result, 'create-error.json')
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        worker_evidence = {'experiment.tar.gz', 'result.json', 'acceptance.json'} & set(archive.namelist())
    if (worker_evidence or provision['pod_id'] is not None or provision['name'] != name
            or provision['run_id'] != str(RUN) or error['http_status'] != 400
            or error['categories'] != ['capacity'] or error['status'] != 'create_not_confirmed'
            or spec['phase'] != 'final-holdout' or spec['controller_sha'] != SHA
            or (spec['ledger']['usd'], spec['ledger']['seconds']) != (usd, seconds)):
        raise ValueError('Cannot establish an unexposed holdout and continuous cost history')
    preflight = ledger.zip_json(result, 'preflight.json')
    if preflight['gpu'] != spec['gpu'] or len(preflight['regions']) != 1:
        raise ValueError('Cannot identify rejected placement')
    avoid = {'gpu': spec['gpu'], 'region': preflight['regions'][0],
             'retry_after_epoch': error['checked_epoch'] + 3600}
    marker = dict(run_id=RUN, controller_sha=SHA, seed='holdout',
                  marker_artifact_id=MARKER[0], marker_sha256=MARKER[1])
    return marker, cleanup['estimated_total_usd'], cleanup['estimated_total_seconds'], avoid


def reconcile(usd, seconds):
    for run, workflow, event, conclusion in (
        (RUN, 'vlm-holdout', 'workflow_dispatch', 'failure'),
        (CLEANUP_RUN, 'vlm-holdout-cleanup', 'workflow_run', 'success'),
    ):
        obj = ledger.request(f'{ledger.BASE}/actions/runs/{run}', token=os.environ['GH_TOKEN'])
        expected = dict(head_sha=SHA, head_branch='main', status='completed', run_attempt=1,
                        conclusion=conclusion, event=event, path=f'.github/workflows/{workflow}.yml')
        if any(obj.get(k) != v for k, v in expected.items()):
            raise ValueError('Holdout dispatch or independent cleanup identity changed')
    return verify(ledger.artifact_bytes(*RESULT, 64000), ledger.artifact_bytes(*CLEANUP, 64000),
                  usd, seconds)
