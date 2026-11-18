"""Reuse the verified six adapters from the completed repeat study."""
import json
import os
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT/'docs/evidence/vlm-repeat-complete.json'


def enabled():
    return os.environ.get('WAFER_VLM_PHASE') == 'holdout'


def pin():
    result = json.loads(RECEIPT.read_text())['artifacts']['result']
    return {'artifact_id': result['id'], 'zip_sha256': result['sha256']}


def weights(archive, destination):
    from runpod.confirmation_verify import unpack
    from wafervlm.core import sha256_file
    receipt = json.loads(RECEIPT.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        inner = Path(tmp)/'experiment.tar.gz'
        with zipfile.ZipFile(archive) as outer:
            if outer.getinfo(inner.name).file_size != receipt['artifacts']['inner']['size']:
                raise ValueError('Unexpected prior result size')
            inner.write_bytes(outer.read(inner.name))
        if sha256_file(inner) != receipt['artifacts']['inner']['sha256']:
            raise ValueError('Prior result archive changed')
        results = unpack(inner, destination)
    root = results/'repeats'
    for ep in json.loads((ROOT/'docs/evidence/vlm-six-endpoints.json').read_text()):
        adapter = root/f"weights-{ep['seed']}-{ep['step']}"/'adapter'
        for file in ('adapter_model.safetensors','adapter_config.json'):
            if sha256_file(adapter/file)!=ep['files'][file]:
                raise ValueError('Frozen adapter mismatch')
    return root


def reconcile(usd, seconds):
    from runpod import vlm_ledger as ledger
    r=json.loads(RECEIPT.read_text())
    base=ledger.BASE
    token=os.environ['GH_TOKEN']
    for run_id, event, workflow in [(r['run_id'],'workflow_dispatch','vlm-repeats'),
            (r['artifacts']['independent_cleanup']['run_id'],'workflow_run','vlm-repeats-cleanup')]:
        run=ledger.request(f'{base}/actions/runs/{run_id}',token=token)
        expected=dict(head_sha=r['controller_sha'], head_branch='main', status='completed',
                      run_attempt=1, conclusion='success', event=event, path=f'.github/workflows/{workflow}.yml')
        if any(run.get(k)!=v for k,v in expected.items()):
            raise ValueError('Prior successful dispatch identity changed')
    p=pin()
    raw=ledger.artifact_bytes(p['artifact_id'],p['zip_sha256'],125_000_000)
    clean=ledger.zip_json(raw,'cleanup.json')
    spec=ledger.zip_json(raw,'launch-spec.json')
    if clean!=r['cleanup'] or spec['ledger']['usd']!=usd or spec['ledger']['seconds']!=seconds:
        raise ValueError('Prior cost chain mismatch')
    a=ledger.zip_json(raw,'acceptance.json')
    if a['passes']!=18 or a['status']!='accepted' or a['archive_sha256']!=r['artifacts']['inner']['sha256']:
        raise ValueError('Prior evaluation incomplete')
    ind=r['artifacts']['independent_cleanup']
    if ledger.zip_json(ledger.artifact_bytes(ind['id'],ind['sha256'],64000),'cleanup.json')!=r['independent_cleanup']:
        raise ValueError('Prior independent cleanup mismatch')
    marker=r['artifacts']['authorization']
    return dict(run_id=r['run_id'],controller_sha=r['controller_sha'],marker_artifact_id=marker['id'],
                marker_sha256=marker['sha256'], seed='repeat'),clean['estimated_total_usd'],clean['estimated_total_seconds']


if __name__ == '__main__':
    receipt=json.loads(RECEIPT.read_text())
    from runpod.vlm_ledger import artifact_bytes
    p=pin()
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'completed.zip'
        archive.write_bytes(artifact_bytes(p['artifact_id'],p['zip_sha256'],125_000_000))
        weights(archive,Path(tmp)/'unpacked')
        with zipfile.ZipFile(archive) as z:
            prior=json.loads(z.read('launch-spec.json'))['ledger']
        _, usd, seconds = reconcile(prior['usd'],prior['seconds'])
        from runpod import vlm_holdout_capacity
        vlm_holdout_capacity.reconcile(usd, seconds)
    print('Verified six frozen adapters, completed run and independent cleanup.')
