"""Resume the reviewed 14/18 archive using the unchanged frozen evaluator."""
import argparse
import json
import os
import shutil
import signal
import tempfile
import zipfile
from pathlib import Path

PIN = {
    'run_id': 43681461001, 'seed': 'repeat',
    'controller_sha': '8b1679be3944faaf778cd1c7002fcb37e8ba6ac1',
    'result_artifact_id': 14150800165,
    'result_sha256': '5634554e305d686b51632d4d03ea1212ee58aac6ad421f09813c34cf2029a678',
    'archive_sha256': '821e28fb4d9724b3bfec9272640837e71b90ce51747c08a081f1efbe3e0f7c3a',
    'marker_artifact_id': 14147699996,
    'marker_sha256': '4b2c467cb54f1fc936e4919cd6607bc93be7e6123362b82c66a0eb4398e192c7',
    'independent_cleanup_run_id': 43685646665,
    'independent_cleanup_artifact_id': 14150167011,
    'independent_cleanup_sha256': '2286e28196035b945229bad7c4447e4617dab9b108b58ce86ac990254641c3ac',
}
MAX_ZIP = 150_000_000  # Six endpoint adapters make the 16-pass package ~121 MB.
GPU = 'NVIDIA GeForce RTX 4090'
RECEIPT = Path(__file__).resolve().parents[1] / 'docs/evidence/vlm-repeat-resume.json'
FINISH_RECEIPT = RECEIPT.with_name('vlm-repeat-finish.json')
FINISH_PIN_FILE = RECEIPT.with_name('vlm-repeat-finish-pin.json')
RESUME_EXPIRES = 1795132800  # 2026-11-20 08:00 SGT; only this bounded continuation
RESUME_WORK_SECONDS = 1800


def enabled():
    return os.environ.get('WAFER_VLM_RESUME') == '1'


def finishing():
    return os.environ.get('WAFER_VLM_FINISH') == '1'


def finish_pin():
    return json.loads(FINISH_PIN_FILE.read_text())


def active_pin():
    return finish_pin() if finishing() else PIN


def transfer_pin():
    pin = active_pin()
    return dict(artifact_id=pin['result_artifact_id'], zip_sha256=pin['result_sha256'])


def verify_source(archive, destination, *, pin=PIN, receipt_path=RECEIPT):
    """Verify immutable transport, completed trees and raw predictions before reuse.

    The partial 15th process is deliberately not reusable. The unchanged full
    summarizer checks every inference/runtime field again after all 18 exist.
    """
    from runpod.confirmation_verify import unpack
    from runpod.vlm_repeat_inputs import FROZEN, portable
    from wafervlm import confirmation as c
    from wafervlm.core import read_jsonl, sha256_file
    SOURCE = '11d2e5f088b350dd9391864dbe2869417fff5af7'
    IMAGE = 'ghcr.io/jasonjiang8866/wafervlm@sha256:a6d31b4f12c2c3ee64c59d08b1e67c89e9bbf9b46cbdb4179417f1d1efa3fa79'
    archive, destination = Path(archive), Path(destination)
    if archive.stat().st_size > MAX_ZIP or sha256_file(archive) != pin['result_sha256']:
        raise ValueError('Partial artifact checksum mismatch')
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if m.filename == 'experiment.tar.gz']
        if len(members) != 1 or not 0 < members[0].file_size <= MAX_ZIP:
            raise ValueError('Invalid partial archive')
        inner = destination / 'experiment.tar.gz'
        with z.open(members[0]) as r, inner.open('wb') as w:
            shutil.copyfileobj(r, w, 1024 * 1024)
    if sha256_file(inner) != pin['archive_sha256']:
        raise ValueError('Partial inner checksum mismatch')
    root = unpack(inner, destination / 'unpacked')
    if any(p.name in {'test', 'holdout', 'selected-test', 'training'} for p in root.rglob('*')):
        raise ValueError('Unexpected partial workload')
    worker = json.loads((root / 'worker.json').read_text())
    cfg = c.recipe()
    for k, v in dict(status='rejected', phase='validation', source_sha=SOURCE,
                     controller_sha=pin['controller_sha'],
                     config_sha256=sha256_file(c.CONFIG),
                     input_manifest_sha256=sha256_file(FROZEN)).items():
        if worker.get(k) != v:
            raise ValueError('Partial provenance mismatch: ' + k)
    hardware = json.loads((root / 'hardware.json').read_text())
    if hardware.get('gpu') != pin.get('gpu', GPU) or hardware.get('image_digest') != IMAGE.split('@')[1]:
        raise ValueError('Partial hardware/image mismatch')
    repeats = root / 'repeats'
    c.verify_tree(repeats)
    completed = pin.get('completed_passes', 14)
    if type(completed) is not int or not 14 <= completed < 18:
        raise ValueError('Invalid pinned partial pass count')
    if (json.loads((repeats / 'status.json').read_text()) !=
            dict(status='failed', error_type='TimeoutExpired', completed_passes=completed)
            or json.loads((repeats / 'workload.json').read_text()) != cfg
            or portable(json.loads((repeats / 'endpoints.json').read_text())) != json.loads(FROZEN.read_text())
            or sha256_file(repeats / 'validation-reference.jsonl') != cfg['validation_sha256']):
        raise ValueError('Partial matrix, status or validation changed')
    reference = read_jsonl(repeats / 'validation-reference.jsonl')
    identities = [(r['id'], r['label'], r['image_sha256']) for r in reference]
    if len(identities) != 160 or len({r[0] for r in identities}) != 160:
        raise ValueError('Invalid validation sample identities')
    records = json.loads((repeats / 'passes.json').read_text())
    expected = [(s, t, r) for s in cfg['seeds'] for t in cfg['endpoints'] for r in range(1, 4)]
    if [(r['seed'], r['step'], r['repeat']) for r in records] != expected[:completed]:
        raise ValueError('Expected exactly the pinned completed prefix')
    original = {r['output']: r for r in json.loads(RECEIPT.read_text())['reused']}
    weights = {(r['seed'], r['step']): r['files']['adapter_model.safetensors']
               for r in json.loads(FROZEN.read_text())}
    reused, instances = [], set()
    for r in records:
        name = f"seed{r['seed']}-step{r['step']}-repeat{r['repeat']}"
        if r['output'] != name or r['adapter_sha256'] != weights[r['seed'], r['step']]:
            raise ValueError('Partial pass path or weights changed')
        path = repeats / name
        c.verify_tree(path)
        run = json.loads((path / 'run.json').read_text())
        pred = read_jsonl(path / 'predictions.jsonl')
        if ([(p['id'], p['label'], p['image_sha256']) for p in pred] != identities
                or any(p['prediction'] != c.parse_label(p['response'], p['truncated']) for p in pred)
                or run['source_sha'] != SOURCE or run['image_digest'] != IMAGE.split('@')[1]
                or run['gpu'] != (GPU if name in original else pin['gpu'])
                or run['adapter_sha256'] != r['adapter_sha256']
                or not run['process_instance'] or run['process_instance'] in instances):
            raise ValueError('Partial predictions or process provenance changed')
        instances.add(run['process_instance'])
        saved = json.loads((path / 'metrics.json').read_text())
        if any(saved[k] != v for k, v in c.metrics(pred).items()):
            raise ValueError('Partial metrics differ from raw predictions')
        reused.append(dict(output=name, checksums_sha256=sha256_file(path / 'checksums.json'),
                           predictions_sha256=sha256_file(path / 'predictions.jsonl')))
        if name in original and reused[-1] != original[name]:
            raise ValueError('Original completed pass changed in continuation')
    receipt = dict(source_run_id=pin['run_id'], source_zip_sha256=pin['result_sha256'],
                   source_archive_sha256=pin['archive_sha256'], reused=reused,
                   new_outputs=[f'seed{s}-step{t}-repeat{r}' for s, t, r in expected[completed:]])
    if receipt != json.loads(Path(receipt_path).read_text()):
        raise ValueError('Completed pass receipt changed')
    return repeats, receipt


def reconcile_partial(output, prior_usd, prior_seconds, *, pin=PIN, receipt_path=RECEIPT):
    """Account for the timed-out allocation and verify both cleanup paths."""
    from runpod import vlm_ledger as ledger
    from runpod.provider import finite
    p = pin
    run = ledger.request(f"{ledger.BASE}/actions/runs/{p['run_id']}", token=os.environ['GH_TOKEN'])
    required = dict(head_sha=p['controller_sha'], run_attempt=1, status='completed',
                    conclusion='failure', event='workflow_dispatch', head_branch='main',
                    path='.github/workflows/vlm-repeats.yml')
    if any(run.get(k) != v for k, v in required.items()):
        raise RuntimeError('Partial dispatch identity changed')
    path = Path(output) / ('repeat-partial.zip' if p == PIN else 'repeat-finish-partial.zip')
    if not path.exists():
        path.write_bytes(ledger.artifact_bytes(p['result_artifact_id'], p['result_sha256'], MAX_ZIP))
    with tempfile.TemporaryDirectory() as tmp:
        _, receipt = verify_source(path, Path(tmp) / 'source', pin=p, receipt_path=receipt_path)
    raw = path.read_bytes()
    clean, spec, intent = [ledger.zip_json(raw, n + '.json') for n in ('cleanup', 'launch-spec', 'provision')]
    name = f"wafer-20261107-usd20-vlmrepeats-{p['run_id']}"
    if (clean.get('name') != name or clean.get('status') != 'verified_absent'
            or clean.get('creation_confirmed') is not True or clean.get('consecutive_absent_checks', 0) < 3
            or clean.get('within_budget') is not True or clean.get('within_runtime') is not True
            or clean.get('persistent_volumes_created') is not False
            or spec.get('ledger', {}).get('usd') != prior_usd
            or spec.get('ledger', {}).get('seconds') != prior_seconds
            or intent.get('name') != name or intent.get('run_id') != str(p['run_id'])
            or not intent.get('pod_id')
            or intent.get('cost', {}).get('prior_spend_reserve_usd') != prior_usd
            or intent.get('cost', {}).get('prior_runtime_reserve_seconds') != prior_seconds):
        raise RuntimeError('Partial ownership, cost or cleanup mismatch')
    usd, seconds, elapsed = [clean[k] for k in
                            ('estimated_total_usd', 'estimated_total_seconds', 'elapsed_billing_bound_s')]
    cost = intent['cost']
    if (not all(finite(v, 0.001) for v in (usd, seconds, elapsed))
            or usd > 30 or seconds > 66000
            or abs(seconds - prior_seconds - elapsed) > 1e-6
            or abs(clean['checked_epoch'] - intent['started_epoch'] - elapsed) > .01
            or abs(usd - prior_usd - elapsed / 3600 * (cost['gpu_hourly_usd'] + cost['disk_hourly_usd'])) > 1e-9):
        raise RuntimeError('Invalid partial cumulative consumption')
    check = ledger.zip_json(ledger.artifact_bytes(p['independent_cleanup_artifact_id'],
                            p['independent_cleanup_sha256'], 64000), 'cleanup.json')
    if (check.get('name') != name or check.get('status') != 'verified_absent'
            or check.get('consecutive_absent_checks', 0) < 3
            or check.get('checked_epoch', 0) < clean['checked_epoch']):
        raise RuntimeError('Independent partial cleanup mismatch')
    ledger.verify_marker(ledger.artifact_bytes(p['marker_artifact_id'], p['marker_sha256'], 64000),
                         p['run_id'], True, p['controller_sha'], 'repeat')
    name = 'resume-verification.json' if p == PIN else 'finish-verification.json'
    (Path(output) / name).write_text(json.dumps(receipt, indent=2) + '\n')
    return p, usd, seconds


def resume_validation(dataset, endpoints, output, source, timeout_seconds):
    from wafervlm import confirmation as c
    from wafervlm.core import write_json
    endpoints = json.loads(Path(endpoints).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        kwargs = dict(pin=finish_pin(), receipt_path=FINISH_RECEIPT) if finishing() else {}
        previous, receipt = verify_source(source, Path(tmp) / 'source', **kwargs)
        reusable = {r['output'] for r in receipt['reused']}
        fresh = set(receipt['new_outputs'])
        executed, copied = set(), set()
        original = c.run_process

        def resume_process(args, deadline, log):
            dest = Path(args[args.index('--output') + 1])
            name = dest.name
            if name in reusable and name not in copied:
                shutil.copytree(previous / name, dest)
                copied.add(name)
            elif name in fresh and name not in executed:
                original(args, deadline, log)
                executed.add(name)
            else:
                raise ValueError('Unexpected or duplicate resumed process')

        # Preserve the frozen runner, weights, CLI flags and full summarizer.
        # Only its subprocess boundary reuses verified completed directories.
        c.run_process = resume_process
        try:
            report = c.repeat_validation(dataset, output, endpoints, timeout_seconds)
            if copied != reusable or executed != fresh:
                raise ValueError('Incomplete continuation')
            write_json(Path(output) / 'resume.json', receipt)
            c.seal_output(output)
            return report
        finally:
            c.run_process = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--output', type=Path, required=True)
    finish = sub.add_parser('verify-finish')
    finish.add_argument('--output', type=Path, required=True)
    resume = sub.add_parser('resume')
    for key in ('dataset', 'endpoints', 'output', 'source'):
        resume.add_argument('--' + key, required=True)
    resume.add_argument('--timeout-seconds', type=int, required=True)
    args = vars(parser.parse_args())
    mode = args.pop('mode')
    if mode == 'resume':
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Continuation interrupted; preserve completed evidence')
        signal.signal(signal.SIGTERM, interrupted)
        resume_validation(**args)
    else:
        from runpod.vlm_ledger import REPEAT_ATTEMPTS
        prior = REPEAT_ATTEMPTS[-1]['cleanup']
        args['output'].mkdir(parents=True, exist_ok=True)
        _, usd, seconds = reconcile_partial(args['output'], prior['estimated_total_usd'],
                                             prior['estimated_total_seconds'])
        completed = 14
        if mode == 'verify-finish':
            from runpod.vlm_repeat_capacity import reconcile
            _, usd, seconds, _ = reconcile(args['output'], usd, seconds)
            pin = finish_pin()
            _, usd, seconds = reconcile_partial(args['output'], usd, seconds,
                                                pin=pin, receipt_path=FINISH_RECEIPT)
            completed = pin['completed_passes']
        print(json.dumps(dict(status='verified_partial', completed_passes=completed, missing_passes=18-completed,
                              cumulative_usd=usd, cumulative_seconds=seconds)))
