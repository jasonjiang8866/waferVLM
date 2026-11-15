"""Hash-bound six-checkpoint input staging; CPU only, no provider mutations."""
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'docs/evidence/vlm-six-endpoints.json'
MAX_ZIP = 2_200_000_000


def portable(rows):
    return [{k: v for k, v in r.items() if k != 'checkpoint'} for r in rows]


def inputs():
    from scripts.prepare_confirmation_inputs import RELEASES
    result = {}
    for step, artifact in ((200, 13630761886), (250, 13706115627)):
        result[f'seed17-{step}'] = dict(seed=17, step=step, artifact_id=artifact,
                                       zip_sha256=RELEASES[step]['zip_sha'])
    receipts = json.loads((ROOT / 'runpod/vlm-completed.json').read_text())
    if [r['seed'] for r in receipts] != [42, 123]:
        raise ValueError('Both accepted fresh seeds are required')
    for r in receipts:
        result[f"seed{r['seed']}"] = dict(seed=r['seed'], artifact_id=r['result_artifact_id'],
                                         zip_sha256=r['result_sha256'], receipt=r)
    return result


def validated_url(url):
    p = urllib.parse.urlsplit(url)
    if (p.scheme != 'https' or p.username or p.password or p.port not in (None, 443)
            or not (p.hostname or '').endswith(('.blob.core.windows.net',
                      '.actions.githubusercontent.com', '.githubusercontent.com'))):
        raise ValueError('Unexpected signed artifact destination')
    return url


def signed_input_urls():
    """Only artifact-scoped download URLs enter the Pod; never a GitHub token."""
    from runpod.provider import NoRedirect, REPOSITORY
    urls = {}
    for key, pin in inputs().items():
        req = urllib.request.Request(
            f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{pin['artifact_id']}/zip",
            headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                     'User-Agent': 'wafervlm-actions/0.1'})
        try:
            with urllib.request.build_opener(NoRedirect).open(req, timeout=30):
                raise RuntimeError('Expected a scoped artifact redirect')
        except urllib.error.HTTPError as exc:
            if exc.code != 302:
                raise RuntimeError('Cannot acquire scoped artifact download') from None
            urls[key] = validated_url(exc.headers.get('Location', ''))
    return urls


def archives(output, urls=None, deadline=None):
    """Download bounded bytes and check the committed hash before any extraction."""
    from runpod.provider import NoRedirect
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pins = inputs()
    if urls is not None and set(urls) != set(pins):
        raise ValueError('Exactly four pinned input archives are required')
    paths = {}
    for key, pin in pins.items():
        path = output / (key + '.zip')
        if not path.exists():
            tmp = path.with_suffix('.partial')
            if urls is None:
                from runpod.vlm_ledger import artifact_bytes
                tmp.write_bytes(artifact_bytes(pin['artifact_id'], pin['zip_sha256'], MAX_ZIP))
            else:
                req = urllib.request.Request(validated_url(urls[key]),
                                             headers={'User-Agent': 'wafervlm-actions/0.1'})
                count = 0
                with urllib.request.build_opener(NoRedirect).open(req, timeout=60) as r, tmp.open('wb') as w:
                    while block := r.read(1024 * 1024):
                        count += len(block)
                        if count > MAX_ZIP or (deadline is not None and time.time() >= deadline):
                            raise TimeoutError('Input download size/deadline exceeded')
                        w.write(block)
            tmp.replace(path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size > MAX_ZIP or digest != pin['zip_sha256']:
            raise ValueError('Input archive checksum mismatch: ' + key)
        paths[key] = path
    return paths


def prepare(paths, output, dataset, expected=None):
    from scripts.prepare_confirmation_inputs import prepare as released
    from runpod.confirmation_verify import unpack, verify_training_archive
    from wafervlm.confirmation import inspect_checkpoint, prepare_endpoints
    from wafervlm.core import sha256_file
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    rows = []
    pins = inputs()
    if set(paths) != set(pins):
        raise ValueError('Missing input archives')
    for key, pin in pins.items():
        archive = Path(paths[key])
        if sha256_file(archive) != pin['zip_sha256']:
            raise ValueError('Input archive checksum mismatch: ' + key)
        if pin['seed'] == 17:
            stage = root / key
            released(archive, stage, pin['step'])
            rows.append(inspect_checkpoint(stage / f"checkpoints/checkpoint-{pin['step']}",
                                           dataset, 17, pin['step']))
        else:
            with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as z:
                matches = [m for m in z.infolist() if m.filename == 'experiment.tar.gz']
                if len(matches) != 1 or not 0 < matches[0].file_size <= 2 * 1024**3:
                    raise ValueError('Invalid training archive')
                inner = Path(tmp) / 'experiment.tar.gz'
                with z.open(matches[0]) as r, inner.open('wb') as w:
                    shutil.copyfileobj(r, w, 1024 * 1024)
                spec = json.loads(z.read('launch-spec.json'))
                accepted = verify_training_archive(inner, dataset, spec)
                if accepted != pin['receipt']['acceptance']:
                    raise ValueError('Training acceptance changed')
                result = unpack(inner, root / key)
            for step in (200, 250):
                rows.append(inspect_checkpoint(result / f'training/step{step}/checkpoints/checkpoint-{step}',
                                               dataset, pin['seed'], step))
    rows = prepare_endpoints(rows, dataset)
    if expected is not None and portable(rows) != expected:
        raise ValueError('Six-checkpoint freeze mismatch')
    return rows


def preflight_inputs(output):
    from runpod.vlm_ledger import dataset_metadata
    output = Path(output)
    cache = output / 'repeat-input-cache'; cache.mkdir(exist_ok=True)
    # Reuse already reverified ledger downloads instead of transferring them twice.
    for key, pin in inputs().items():
        if pin['seed'] in (42, 123):
            previous = output / f"completed-{pin['receipt']['run_id']}.zip"
        elif pin.get('step') == 250:
            previous = output / 'historical-result.zip'
        else:
            continue
        dest = cache / (key + '.zip')
        if previous.exists() and not dest.exists():
            os.link(previous, dest)
    paths = archives(cache)
    expected = json.loads(FROZEN.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        rows = prepare(paths, Path(tmp) / 'inputs', dataset_metadata(output), expected)
    record = dict(status='verified_without_gpu', endpoints=portable(rows),
                  manifest_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                  archives={k: p['zip_sha256'] for k, p in inputs().items()}, holdout_accessed=False)
    (output / 'input-verification.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('vlm-receipt-check'))
    args = parser.parse_args(); args.output.mkdir(exist_ok=True)
    print(json.dumps(preflight_inputs(args.output)))
