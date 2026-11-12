"""Independently verify a fresh-seed training archive without unpickling tensors."""
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

MAX_FILES = 10000
MAX_BYTES = 4 * 1024**3


def unpack(archive, destination):
    """Extract only canonical regular files beneath results into a fresh directory."""
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    seen, total = set(), 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            p = PurePosixPath(member.name)
            if (p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'results'
                    or str(p) != member.name.rstrip('/') or member.name.rstrip('/') in seen):
                raise ValueError('Unsafe or duplicate archive path')
            seen.add(member.name.rstrip('/'))
            if len(seen) > MAX_FILES:
                raise ValueError('Too many archive members')
            if not member.isdir() and not member.isfile():
                raise ValueError('Links/devices are not permitted')
            total += member.size
            if (member.size < 0 or total > MAX_BYTES
                    or (p.suffix in {'.json', '.jsonl'} and member.size > 8_000_000)):
                raise ValueError('Archive exceeds unpacked size cap')
            out = destination.joinpath(*p.parts)
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, out.open('xb') as target:
                    shutil.copyfileobj(source, target, length=1024*1024)
    return destination / 'results'


def verify_training_archive(archive, dataset, spec):
    """dataset needs only the pinned dataset.json, never images or final holdout.

    Accepted here means integrity/execution, not improved model quality. Scope is
    fresh training only; recovery packages and repetition results are rejected.
    """
    from runpod.confirmation_contract import launch_plan
    from wafervlm.confirmation import recipe, verify_training, verify_tree
    from wafervlm.core import sha256_file
    plan = launch_plan(spec)
    cfg = recipe()
    if sha256_file(Path(dataset)/'dataset.json') != cfg['dataset_sha256']:
        raise ValueError('Dataset identity mismatch')
    with tempfile.TemporaryDirectory() as tmp:
        root = unpack(archive, Path(tmp)/'unpacked')
        if any(p.name in {'test', 'holdout', 'selected-test', 'recovery-parent'} for p in root.rglob('*')):
            raise ValueError('Unexpected holdout/test/recovery output')
        worker = json.loads((root/'worker.json').read_text())
        for key, expected in dict(status='accepted', profiles=['seed-confirmation'],
                                  source_sha=plan['source_sha'], controller_sha=plan['controller_sha'],
                                  config_sha256=plan['config_sha256'], seed=plan['seed']).items():
            if worker.get(key) != expected:
                raise ValueError('Worker provenance mismatch: '+key)
        hardware = json.loads((root/'hardware.json').read_text())
        if not hardware.get('gpu') or hardware.get('image_digest') != spec['image'].split('@')[1]:
            raise ValueError('Hardware/image evidence missing')
        if json.loads((root/'acquisition.json').read_text()) != json.loads(
                (Path(__file__).resolve().parents[1]/'docs/evidence/wm811k-lot17.json').read_text()):
            raise ValueError('Acquisition evidence mismatch')
        training = root/'training'
        verify_tree(training)
        if json.loads((training/'workload.json').read_text()) != cfg:
            raise ValueError('Frozen recipe mismatch')
        status = json.loads((training/'status.json').read_text())
        if status != dict(status='completed', seed=plan['seed'], end_step=250,
                          test_accessed=False, holdout_accessed=False):
            raise ValueError('Training is incomplete or used holdout')
        start, parent, endpoints = 0, None, []
        for end in cfg['stages']:
            stage = training/f'step{end}'
            endpoint = verify_training(stage, dataset, plan['seed'], start, end, parent)
            run = json.loads((stage/'run.json').read_text())
            if (run.get('source_sha') != plan['source_sha']
                    or run.get('image_digest') != spec['image'].split('@')[1]
                    or not run.get('gpu')):
                raise ValueError('Stage source/image/GPU provenance mismatch')
            if end in cfg['endpoints']:
                endpoints.append(endpoint)
            start, parent = end, stage/f'checkpoints/checkpoint-{end}'
        saved = json.loads((training/'endpoints.json').read_text())
        # Worker absolute paths change on retrieval: compare hashes and endpoint identity only.
        portable = lambda rows: [{k:v for k,v in row.items() if k != 'checkpoint'} for row in rows]
        if portable(saved) != portable(endpoints):
            raise ValueError('Endpoint manifest mismatch')
        return dict(status='accepted', seed=plan['seed'], endpoints=portable(endpoints),
                    archive_sha256=sha256_file(archive), quality_accepted=False, holdout_accessed=False)
