"""Stage the immutable released step-200 checkpoint and validation-only evidence."""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL = 'https://github.com/jasonjiang8866/waferVLM/releases/download/checkpoint-200-v1/wafervlm-run-42502913001.zip'
ZIP_SHA = 'b039ca64d37b2d343a0f94da5e26e741e3b463ee48af2ab5877659c555841490'
INNER_SHA = 'eb83e251e3a27a59ae643d89f7f98a7c620b6b225fd7e9926a4190c6f7226508'
PREFIX = 'results/continuation/'
PARENT_SHA = '8cda3bcfcacdb62a4f6649d17fa415749af01aedd663986322fa7550ed328c55'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    if digest(archive) != ZIP_SHA:
        raise ValueError('Released ZIP checksum mismatch')
    output.mkdir(parents=True, exist_ok=False)
    inner = output / 'source.tar.gz'
    with zipfile.ZipFile(archive) as z, z.open('experiment.tar.gz') as src, inner.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    if digest(inner) != INNER_SHA:
        raise ValueError('Released inner checksum mismatch')
    selected = {}
    parent_hashes = {}
    with tarfile.open(inner, 'r|gz') as tar:
        manifest = None
        for member in tar:
            if not member.isfile():
                continue
            if member.name == PREFIX + 'checksums.json':
                manifest = json.load(tar.extractfile(member))
            rel = member.name.removeprefix(PREFIX)
            dest = None
            if rel == 'step200/run.json' or rel.startswith('step200/checkpoints/checkpoint-200/'):
                dest = output / 'train' / rel.removeprefix('step200/')
            elif any(rel.startswith(f'validation-{s}/') for s in (100, 150, 200)):
                dest = output / 'evidence' / rel
            elif rel in ('step150/training.json', 'step200/training.json'):
                dest = output / 'evidence' / rel
            if dest is None:
                continue
            if (not member.name.startswith(PREFIX) or '..' in PurePosixPath(rel).parts
                    or member.size > 100_000_000 or member.name in selected):
                raise ValueError('Unsafe released member')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            selected[member.name] = (rel, digest(dest))
            if dest.is_relative_to(output / 'train'):
                parent_hashes[str(dest.relative_to(output / 'train'))] = digest(dest)
    if not manifest:
        raise ValueError('Missing source checksum manifest')
    for rel, sha in selected.values():
        if not rel.endswith('/checksums.json') and manifest.get(rel) != sha:
            raise ValueError('Released member mismatch')
    cp = output / 'train/checkpoints/checkpoint-200'
    required = {'adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'training_args.bin'}
    if (not required <= {p.name for p in cp.iterdir()}
            or json.loads((cp/'trainer_state.json').read_text())['global_step'] != 200
            or digest(cp/'adapter_model.safetensors') != PARENT_SHA):
        raise ValueError('Incomplete step-200 state')
    (output/'train/checksums.json').write_text(json.dumps(parent_hashes, indent=2)+'\n')
    (output/'train/resume-origin.json').write_text(json.dumps({
        'run_id': 42502913001, 'artifact_id': 13630761886, 'release_tag': 'checkpoint-200-v1',
        'url': URL, 'zip_sha256': ZIP_SHA, 'archive_sha256': INNER_SHA,
        'global_step': 200, 'adapter_sha256': PARENT_SHA}, indent=2)+'\n')
    inner.unlink()
    return {'staged_files': len(selected), 'parent_step': 200, 'zip_sha256': ZIP_SHA}


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--archive', default='parent-release.zip'); p.add_argument('--output', default='profile-input')
    p.add_argument('--download', action='store_true'); a=p.parse_args()
    if a.download:
        with urllib.request.urlopen(URL, timeout=60) as src, open(a.archive, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    print(json.dumps(prepare(a.archive, a.output)))
